//! Private compiler-owned CPU device-model driver. This is not the L1 C ABI.
//! The selected compiler artifact must implement the versioned word signature.
//! Its library, opaque allocations and worker threads share one confined owner.
use libloading::Library;
use sha2::{Digest, Sha256};
use std::{ffi::c_void, fs, marker::PhantomData, path::Path, rc::Rc};
type Entry = unsafe extern "C" fn(u32, *mut *mut c_void, *const u64, usize, *mut u64, usize) -> u32;
#[derive(Debug)]
pub enum Error {
    Library,
    Identity,
    Protocol,
    Allocation,
    Busy,
    Unknown,
}
pub struct Driver {
    _library: Option<Library>,
    entry: Entry,
    handle: *mut c_void,
    addresses: [u64; 12],
    sizes: [usize; 12],
    _thread: PhantomData<Rc<()>>,
}
impl Driver {
    pub fn load(
        path: &Path,
        hash: [u8; 32],
        symbol: &str,
        sizes: [usize; 12],
        input: &[u64],
        config: [u64; 6],
    ) -> Result<Self, Error> {
        if symbol.is_empty()
            || symbol.len() > 128
            || !symbol
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_')
            || sizes.iter().any(|n| *n == 0 || n % 8 != 0)
            || sizes
                .iter()
                .try_fold(0usize, |a, b| a.checked_add(*b))
                .is_none_or(|n| n > 64 * 1024 * 1024)
        {
            return Err(Error::Protocol);
        }
        let metadata = fs::metadata(path).map_err(|_| Error::Library)?;
        if metadata.len() > 64 * 1024 * 1024 {
            return Err(Error::Identity);
        }
        let bytes = fs::read(path).map_err(|_| Error::Library)?;
        if <[u8; 32]>::from(Sha256::digest(bytes)) != hash {
            return Err(Error::Identity);
        }
        // SAFETY: root generated this private artifact with the exact versioned ABI.
        // Constructors and native pointers require trust in that selected compiler.
        let library = unsafe { Library::new(path) }.map_err(|_| Error::Library)?;
        // SAFETY: the generated symbol has Entry's frozen word/opaque-handle signature.
        let entry =
            *unsafe { library.get::<Entry>(symbol.as_bytes()) }.map_err(|_| Error::Protocol)?;
        let mut driver = Self {
            _library: Some(library),
            entry,
            handle: std::ptr::null_mut(),
            addresses: [0; 12],
            sizes,
            _thread: PhantomData,
        };
        let mut packet: Vec<u64> = sizes.map(|n| (n / 8) as u64).to_vec();
        packet.extend(config);
        packet.extend(input);
        let mut addresses = [0u64; 12];
        driver.call(0, &packet, &mut addresses)?;
        if driver.handle.is_null() || addresses.contains(&0) {
            return Err(Error::Protocol);
        }
        driver.addresses = addresses;
        Ok(driver)
    }
    fn call(&mut self, command: u32, input: &[u64], output: &mut [u64]) -> Result<(), Error> {
        // SAFETY: all slices contain initialized, correctly aligned word storage for
        // their full lengths. The entry owns/validates its opaque handle; function and
        // worker allocations remain loaded throughout every call. No pointers escape.
        let status = unsafe {
            (self.entry)(
                command,
                &mut self.handle,
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        match status {
            0 => Ok(()),
            1 => Err(Error::Allocation),
            2 => Err(Error::Protocol),
            3 => Err(Error::Busy),
            _ => Err(Error::Unknown),
        }
    }
    /// Replace only an idle compiler-owned image. The generated driver checks
    /// every extent and refuses writes while workers are active.
    pub fn replace(&mut self, image: &[Vec<u64>; 12]) -> Result<(), Error> {
        if image.iter().zip(self.sizes).any(|(v, n)| v.len() != n / 8) {
            return Err(Error::Protocol);
        }
        let words: Vec<u64> = image.iter().flatten().copied().collect();
        self.call(5, &words, &mut [])
    }
    pub fn addresses(&self) -> [u64; 12] {
        self.addresses
    }
    pub fn submit(&mut self) -> Result<(), Error> {
        self.call(1, &[], &mut [])
    }
    pub fn fence(&mut self) -> Result<bool, Error> {
        let mut result = [0];
        self.call(2, &[], &mut result)?;
        match result[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Error::Protocol),
        }
    }
    pub fn read(&self) -> Result<[Vec<u64>; 12], Error> {
        let count = self.sizes.iter().sum::<usize>() / 8;
        let mut words = Vec::new();
        words
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        words.resize(count, 0);
        let mut handle = self.handle;
        // SAFETY: same immutable owner/initialized output as call. Command3 only
        // copies completed storage and does not alter the confined handle.
        let status = unsafe {
            (self.entry)(
                3,
                &mut handle,
                std::ptr::null(),
                0,
                words.as_mut_ptr(),
                words.len(),
            )
        };
        if status != 0 || handle != self.handle {
            return Err(Error::Protocol);
        }
        let mut at = 0;
        Ok(std::array::from_fn(|i| {
            let n = self.sizes[i] / 8;
            let result = words[at..at + n].to_vec();
            at += n;
            result
        }))
    }
}
impl Drop for Driver {
    fn drop(&mut self) {
        if !self.handle.is_null() && self.call(4, &[], &mut []).is_err() {
            // Unknown native workers retain their code mapping and opaque allocations.
            if let Some(library) = self._library.take() {
                std::mem::forget(library)
            }
        }
    }
}
