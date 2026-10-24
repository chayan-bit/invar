//! Compiler-owned native region kernels (ENG-18, #101). This is not the L1 C ABI.
//! The kernels are pure `u32` loops that invar's native backend emits, audits
//! and links for the running host. Callers hand them slices; every length is
//! checked here, on the safe side, before the call.
use libloading::Library;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

/// `out[j] = f(in0[j], in1[j])` for `j < n`; `in1` is unread by one-input maps.
type MapEntry = unsafe extern "C" fn(*const u32, *const u32, *mut u32, u64);
/// The law's fold of `input[0..n]` from its identity.
type ReduceEntry = unsafe extern "C" fn(*const u32, u64) -> u32;

/// Largest library accepted, and largest slice a kernel is called on (the
/// native side states its bounds facts against this limit).
pub const MAX_LIBRARY_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_ELEMENTS: usize = u32::MAX as usize;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Library,
    Identity,
    Symbol,
    /// Slice lengths or arity disagree with the kernel's signature.
    Shape,
}

#[derive(Clone, Copy)]
enum Entry {
    Map { inputs: usize, entry: MapEntry },
    Reduce(ReduceEntry),
}

/// One loaded kernel. The library stays mapped for as long as the kernel lives.
pub struct Kernel {
    _library: Library,
    entry: Entry,
}

fn open(path: &Path, hash: [u8; 32]) -> Result<Library, Error> {
    let metadata = fs::metadata(path).map_err(|_| Error::Library)?;
    if metadata.len() > MAX_LIBRARY_BYTES {
        return Err(Error::Identity);
    }
    let bytes = fs::read(path).map_err(|_| Error::Library)?;
    if <[u8; 32]>::from(Sha256::digest(bytes)) != hash {
        return Err(Error::Identity);
    }
    // SAFETY: the native backend produced this private library from audited IR
    // that defines only the kernel function: no constructors, no globals.
    unsafe { Library::new(path) }.map_err(|_| Error::Library)
}

fn valid(symbol: &str) -> bool {
    !symbol.is_empty()
        && symbol.len() <= 128
        && symbol
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

impl Kernel {
    /// Load a map kernel over `inputs` (1 or 2) arrays from the library at
    /// `path`, whose bytes must hash to `hash`.
    pub fn load_map(
        path: &Path,
        hash: [u8; 32],
        symbol: &str,
        inputs: usize,
    ) -> Result<Self, Error> {
        if !valid(symbol) || !(1..=2).contains(&inputs) {
            return Err(Error::Shape);
        }
        let library = open(path, hash)?;
        // SAFETY: the backend emitted `symbol` with exactly MapEntry's signature.
        let entry =
            *unsafe { library.get::<MapEntry>(symbol.as_bytes()) }.map_err(|_| Error::Symbol)?;
        Ok(Self {
            _library: library,
            entry: Entry::Map { inputs, entry },
        })
    }

    /// Load a reduce kernel from the library at `path` (see [`Self::load_map`]).
    pub fn load_reduce(path: &Path, hash: [u8; 32], symbol: &str) -> Result<Self, Error> {
        if !valid(symbol) {
            return Err(Error::Shape);
        }
        let library = open(path, hash)?;
        // SAFETY: the backend emitted `symbol` with exactly ReduceEntry's signature.
        let entry =
            *unsafe { library.get::<ReduceEntry>(symbol.as_bytes()) }.map_err(|_| Error::Symbol)?;
        Ok(Self {
            _library: library,
            entry: Entry::Reduce(entry),
        })
    }

    /// Write `out[j]` from element `j` of every input, for every `j`.
    /// Every input must be exactly as long as `out`.
    pub fn map(&self, inputs: &[&[u32]], out: &mut [u32]) -> Result<(), Error> {
        let Entry::Map {
            inputs: arity,
            entry,
        } = self.entry
        else {
            return Err(Error::Shape);
        };
        let n = out.len();
        if inputs.len() != arity || n > MAX_ELEMENTS || inputs.iter().any(|input| input.len() != n)
        {
            return Err(Error::Shape);
        }
        if n == 0 {
            return Ok(());
        }
        let first = inputs[0].as_ptr();
        let second = inputs.get(1).map_or(first, |input| input.as_ptr());
        // SAFETY: each pointer addresses `n` initialized, aligned u32 (checked
        // above); `out` is a unique borrow, so it overlaps no input. The kernel
        // reads `in0[0..n]` (and `in1[0..n]` when it has two inputs), writes
        // only `out[0..n]`, keeps no pointer and cannot unwind.
        unsafe { entry(first, second, out.as_mut_ptr(), n as u64) };
        Ok(())
    }

    /// The kernel's fold of `input`.
    pub fn reduce(&self, input: &[u32]) -> Result<u32, Error> {
        let Entry::Reduce(entry) = self.entry else {
            return Err(Error::Shape);
        };
        if input.len() > MAX_ELEMENTS {
            return Err(Error::Shape);
        }
        // SAFETY: `input` addresses `input.len()` initialized, aligned u32; the
        // kernel only reads them (none when the length is 0), keeps no pointer
        // and cannot unwind.
        Ok(unsafe { entry(input.as_ptr(), input.len() as u64) })
    }
}
