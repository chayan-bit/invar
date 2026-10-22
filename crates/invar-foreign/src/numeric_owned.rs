//! Synchronous private ABI10 storage boundary. Only source-host-issued trusted
//! wrappers may implement this signature; this does not extend the public C ABI.
#[cfg(unix)]
pub use super::numeric::terminate_compiler_group;
pub use super::numeric::{Carrier, Error, file_digest};
use libloading::Library;
use std::{marker::PhantomData, path::Path, rc::Rc};

type Entry = unsafe extern "C" fn(
    *const i64,
    usize,
    *mut i64,
    usize,
    *mut u64,
    usize,
    *mut i64,
    usize,
    *mut u64,
    usize,
    u64,
    u64,
    u64,
) -> u32;
type Identity = unsafe extern "C" fn(*mut u8, usize) -> u32;
const MAX_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug)]
pub struct Invocation {
    pub output: [i64; 10],
    pub trace: Vec<u64>,
    pub heap: Vec<i64>,
    pub cleanup: Vec<u64>,
}

/// Retains the mapping through synchronous calls. No native pointer escapes.
/// ```compile_fail
/// fn send<T: Send>() {} send::<invar_foreign::numeric_owned::Driver>();
/// ```
/// ```compile_fail
/// fn sync<T: Sync>() {} sync::<invar_foreign::numeric_owned::Driver>();
/// ```
pub struct Driver {
    _library: Library,
    entry: Entry,
    params: Vec<Carrier>,
    storage: [usize; 2],
    _thread: PhantomData<Rc<()>>,
}
fn zeroed<T: Default + Clone>(len: usize) -> Result<Vec<T>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|_| Error::Allocation)?;
    v.resize(len, T::default());
    Ok(v)
}
impl Driver {
    /// The host owns the immutable private directory and compiler provenance.
    /// Constructors are a trusted native boundary, not sandboxed execution.
    pub fn load(
        path: &Path,
        library_hash: [u8; 32],
        identity: [u8; 32],
        symbol: &str,
        params: Vec<Carrier>,
        storage: [usize; 2],
    ) -> Result<Self, Error> {
        let [heap, cleanup] = storage;
        if !path.is_absolute()
            || !symbol.starts_with("invar_")
            || symbol.len() > 128
            || !symbol
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_')
            || params.len() > 128
            || !(4..=2_040_000).contains(&heap)
            || cleanup > 400_000
            || !cleanup.is_multiple_of(4)
        {
            return Err(Error::Limit);
        }
        if file_digest(path)? != library_hash {
            return Err(Error::Identity);
        }
        // SAFETY: the host's compiler-owned file implements the fixed ABI10 wrapper.
        let library = unsafe { Library::new(path) }.map_err(|e| Error::Library(e.to_string()))?;
        let name = format!("{symbol}_identity");
        // SAFETY: checked generated identity symbol has this fixed signature.
        let id = *unsafe { library.get::<Identity>(name.as_bytes()) }
            .map_err(|e| Error::Symbol(e.to_string()))?;
        let mut actual = [0; 32];
        // SAFETY: initialized writable local with exact capacity; mapping is retained.
        if unsafe { id(actual.as_mut_ptr(), actual.len()) } != 0 || actual != identity {
            return Err(Error::Identity);
        }
        // SAFETY: compiler-issued wrapper binds the private Entry signature.
        let entry = *unsafe { library.get::<Entry>(symbol.as_bytes()) }
            .map_err(|e| Error::Symbol(e.to_string()))?;
        if file_digest(path)? != library_hash {
            return Err(Error::Identity);
        }
        Ok(Self {
            _library: library,
            entry,
            params,
            storage,
            _thread: PhantomData,
        })
    }
    pub fn run(&mut self, input: &[i64], limits: [u64; 3]) -> Result<Invocation, Error> {
        let [fuel, transitions, count] = limits;
        if input.len() != self.params.len()
            || fuel == 0
            || fuel > 1_000_000
            || transitions > 1_000_000
            || count > 1_000_000
        {
            return Err(Error::Limit);
        }
        for (word, ty) in input.iter().zip(&self.params) {
            let valid = match ty {
                Carrier::I32 => i32::try_from(*word).is_ok(),
                Carrier::Word => u32::try_from(*word).is_ok(),
                Carrier::Bool => *word == 0 || *word == 1,
            };
            if !valid {
                return Err(Error::Limit);
            }
        }
        let trace_words = usize::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(7))
            .ok_or(Error::Limit)?;
        let [heap_words, cleanup_words] = self.storage;
        let bytes = trace_words
            .checked_add(heap_words)
            .and_then(|n| n.checked_add(cleanup_words))
            .and_then(|n| n.checked_add(input.len() + 10))
            .and_then(|n| n.checked_mul(8))
            .ok_or(Error::Limit)?;
        if bytes > MAX_BYTES {
            return Err(Error::Limit);
        }
        let mut owned_input = zeroed::<i64>(input.len())?;
        owned_input.copy_from_slice(input);
        let mut raw = Invocation {
            output: [0; 10],
            trace: zeroed(trace_words)?,
            heap: zeroed(heap_words)?,
            cleanup: zeroed(cleanup_words)?,
        };
        // SAFETY: initialized full-capacity, aligned and disjoint buffers remain
        // owned for the entire synchronous noexcept invocation. Empty buffers
        // carry non-null dangling pointers and are never written past length.
        let status = unsafe {
            (self.entry)(
                owned_input.as_ptr(),
                owned_input.len(),
                raw.output.as_mut_ptr(),
                raw.output.len(),
                raw.trace.as_mut_ptr(),
                raw.trace.len(),
                raw.heap.as_mut_ptr(),
                raw.heap.len(),
                raw.cleanup.as_mut_ptr(),
                raw.cleanup.len(),
                fuel,
                transitions,
                count,
            )
        };
        if status != 0 {
            return Err(Error::Protocol(status));
        }
        Ok(raw)
    }
}
