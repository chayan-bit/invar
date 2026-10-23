//! Reviewed synchronous private ABI11 loader for source-host-issued CPU owned
//! numeric continuations. Opaque native tasks never escape this module and no
//! caller-supplied task, heap or cleanup bytes are ever imported or restored.
#[cfg(unix)]
pub use super::numeric::terminate_compiler_group;
pub use super::numeric::{Carrier, Error, file_digest};
use libloading::Library;
use std::{ffi::c_void, marker::PhantomData, path::Path, ptr::NonNull, rc::Rc};

type Identity = unsafe extern "C" fn(*mut u8, usize) -> u32;
type Create = unsafe extern "C" fn(*const i64, usize, u64, u64, u64, *mut *mut c_void) -> u32;
type Resume = unsafe extern "C" fn(*mut c_void, u64) -> u32;
type Capture = unsafe extern "C" fn(
    *mut c_void,
    *mut i64,
    usize,
    *mut i64,
    usize,
    *mut u64,
    usize,
    *mut u8,
    usize,
    *mut u64,
    usize,
    *mut i64,
    usize,
    *mut u64,
    usize,
    *mut u64,
) -> u32;
type Destroy = unsafe extern "C" fn(*mut c_void) -> u32;

pub const OUTPUT_WORDS: usize = 10;
pub const TASK_COUNTERS: usize = 18;
/// Aggregate host-controlled buffer cap: native storage plus host snapshots.
pub const MAX_BYTES: usize = 128 * 1024 * 1024;

/// A completed terminal packet with the storage the decoder validates.
#[derive(Debug)]
pub struct Invocation {
    pub output: [i64; OUTPUT_WORDS],
    pub trace: Vec<u64>,
    pub heap: Vec<i64>,
    pub cleanup: Vec<u64>,
}

/// Copies of initialized native storage for validation, never a resume permit.
#[derive(Debug)]
pub struct Snapshot {
    pub input: Vec<i64>,
    pub output: [i64; OUTPUT_WORDS],
    pub trace: Vec<u64>,
    pub task: Vec<u8>,
    pub counters: [u64; TASK_COUNTERS],
    pub heap: Vec<i64>,
    pub cleanup: Vec<u64>,
    pub native_calls: u64,
}

/// Retains the code mapping through creation, calls, copies and destruction.
/// A single native owner is active at a time. All operations are synchronous.
/// ```compile_fail
/// fn send<T: Send>() {} send::<invar_foreign::numeric_owned_resume::Driver>();
/// ```
/// ```compile_fail
/// fn sync<T: Sync>() {} sync::<invar_foreign::numeric_owned_resume::Driver>();
/// ```
pub struct Driver {
    library: Option<Library>,
    create: Create,
    resume: Resume,
    capture: Capture,
    destroy: Destroy,
    params: Vec<Carrier>,
    task_bytes: usize,
    storage: [usize; 2],
    owner: Option<NonNull<c_void>>,
    events: usize,
    destruction_failure: Option<u32>,
    _thread: PhantomData<Rc<()>>,
}
fn zeroes<T: Default + Clone>(len: usize) -> Result<Vec<T>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|_| Error::Allocation)?;
    v.resize(len, T::default());
    Ok(v)
}
fn status(code: u32) -> Result<(), Error> {
    match code {
        0 => Ok(()),
        1 => Err(Error::Allocation),
        _ => Err(Error::Protocol(code)),
    }
}
/// Bytes of one complete copy of the controlled buffers for these budgets.
/// The native owner holds one copy; each host snapshot holds another.
pub fn copy_bytes(
    params: usize,
    trace_events: u64,
    task_bytes: usize,
    storage: [usize; 2],
) -> Option<usize> {
    usize::try_from(trace_events)
        .ok()?
        .checked_mul(7)?
        .checked_add(params)?
        .checked_add(OUTPUT_WORDS + TASK_COUNTERS + 1)?
        .checked_add(storage[0])?
        .checked_add(storage[1])?
        .checked_mul(8)?
        .checked_add(task_bytes)
}
impl Driver {
    /// Load only a trusted generated wrapper from an immutable private directory.
    /// Exact hashes bind library and prelink identity; constructors are trusted.
    pub fn load(
        path: &Path,
        library_hash: [u8; 32],
        identity: [u8; 32],
        symbol: &str,
        params: Vec<Carrier>,
        task_bytes: usize,
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
            || task_bytes <= 144
            || task_bytes > 128 * 1024 + 144
            || !(task_bytes - 144).is_multiple_of(64)
            || !(4..=2_040_000).contains(&heap)
            || cleanup > 400_000
            || !cleanup.is_multiple_of(4)
        {
            return Err(Error::Limit);
        }
        if file_digest(path)? != library_hash {
            return Err(Error::Identity);
        }
        // SAFETY: selected compiler-issued library has exactly the private ABI.
        let library = unsafe { Library::new(path) }.map_err(|e| Error::Library(e.to_string()))?;
        // SAFETY: each versioned symbol uses the signature emitted by this host.
        let id = *unsafe { library.get::<Identity>(format!("{symbol}_identity").as_bytes()) }
            .map_err(|e| Error::Symbol(e.to_string()))?;
        let mut actual = [0; 32];
        // SAFETY: retained function writes the initialized, live 32-byte buffer.
        status(unsafe { id(actual.as_mut_ptr(), actual.len()) })?;
        if actual != identity {
            return Err(Error::Identity);
        }
        // SAFETY: fixed private signatures, with no pointer escape.
        let create = *unsafe { library.get::<Create>(format!("{symbol}_create").as_bytes()) }
            .map_err(|e| Error::Symbol(e.to_string()))?;
        // SAFETY: trusted wrapper's synchronous resume signature.
        let resume = *unsafe { library.get::<Resume>(format!("{symbol}_resume").as_bytes()) }
            .map_err(|e| Error::Symbol(e.to_string()))?;
        // SAFETY: trusted wrapper's bounded copying signature.
        let capture = *unsafe { library.get::<Capture>(format!("{symbol}_capture").as_bytes()) }
            .map_err(|e| Error::Symbol(e.to_string()))?;
        // SAFETY: trusted wrapper's noexcept destruction signature.
        let destroy = *unsafe { library.get::<Destroy>(format!("{symbol}_destroy").as_bytes()) }
            .map_err(|e| Error::Symbol(e.to_string()))?;
        if file_digest(path)? != library_hash {
            return Err(Error::Identity);
        }
        Ok(Self {
            library: Some(library),
            create,
            resume,
            capture,
            destroy,
            params,
            task_bytes,
            storage,
            owner: None,
            events: 0,
            destruction_failure: None,
            _thread: PhantomData,
        })
    }
    /// Copy typed initial input and fixed budgets once into a new C++ owner.
    pub fn start(&mut self, input: &[i64], limits: [u64; 3]) -> Result<(), Error> {
        let [fuel, transitions, events] = limits;
        if self.owner.is_some()
            || self.destruction_failure.is_some()
            || input.len() != self.params.len()
            || fuel == 0
            || fuel > 1_000_000
            || transitions > 1_000_000
            || events > 1_000_000
        {
            return Err(Error::Limit);
        }
        for (word, carrier) in input.iter().zip(&self.params) {
            if !match carrier {
                Carrier::I32 => i32::try_from(*word).is_ok(),
                Carrier::Word => u32::try_from(*word).is_ok(),
                Carrier::Bool => *word == 0 || *word == 1,
            } {
                return Err(Error::Limit);
            }
        }
        let events_len = usize::try_from(events).map_err(|_| Error::Limit)?;
        // Native storage plus one host snapshot must fit the aggregate cap here;
        // the host additionally accounts for a second snapshot and decode scratch.
        let one =
            copy_bytes(input.len(), events, self.task_bytes, self.storage).ok_or(Error::Limit)?;
        if one.checked_mul(2).is_none_or(|n| n > MAX_BYTES) {
            return Err(Error::Limit);
        }
        let mut owner = std::ptr::null_mut();
        // SAFETY: input lives through return; create copies it into its own
        // initialized C++ object and publishes a handle only on success.
        status(unsafe {
            (self.create)(
                input.as_ptr(),
                input.len(),
                fuel,
                transitions,
                events,
                &mut owner,
            )
        })?;
        self.owner = Some(NonNull::new(owner).ok_or(Error::Protocol(2))?);
        self.events = events_len;
        Ok(())
    }
    /// Advance the task. Invalid quanta are refused before calling native code.
    pub fn resume(&mut self, quantum: u64) -> Result<Snapshot, Error> {
        if quantum == 0 || quantum > 1_000_000 {
            return Err(Error::Limit);
        }
        if let Some(code) = self.destruction_failure {
            return Err(Error::Protocol(code));
        }
        let owner = self.owner.ok_or(Error::Protocol(2))?;
        // Allocate the full snapshot before advancing, so allocation failure
        // cannot leave an unreported new native state.
        let mut snapshot = self.buffers()?;
        // SAFETY: exclusively owned real task, retained library, synchronous
        // noexcept function. No caller-supplied task bytes are ever accepted.
        status(unsafe { (self.resume)(owner.as_ptr(), quantum) })?;
        self.capture_into(&mut snapshot)?;
        Ok(snapshot)
    }
    fn buffers(&self) -> Result<Snapshot, Error> {
        Ok(Snapshot {
            input: zeroes(self.params.len())?,
            output: [0; OUTPUT_WORDS],
            trace: zeroes(self.events.checked_mul(7).ok_or(Error::Limit)?)?,
            task: zeroes(self.task_bytes)?,
            counters: [0; TASK_COUNTERS],
            heap: zeroes(self.storage[0])?,
            cleanup: zeroes(self.storage[1])?,
            native_calls: 0,
        })
    }
    /// Fallibly copy storage while the owner and mapping remain live.
    pub fn snapshot(&self) -> Result<Snapshot, Error> {
        let mut snapshot = self.buffers()?;
        self.capture_into(&mut snapshot)?;
        Ok(snapshot)
    }
    fn capture_into(&self, s: &mut Snapshot) -> Result<(), Error> {
        if let Some(code) = self.destruction_failure {
            return Err(Error::Protocol(code));
        }
        let owner = self.owner.ok_or(Error::Protocol(2))?;
        // SAFETY: disjoint initialized full-capacity buffers, exact checked
        // lengths, live opaque owner and retained trusted copying function.
        // Empty vectors carry non-null dangling pointers never written past length.
        status(unsafe {
            (self.capture)(
                owner.as_ptr(),
                s.input.as_mut_ptr(),
                s.input.len(),
                s.output.as_mut_ptr(),
                OUTPUT_WORDS,
                s.trace.as_mut_ptr(),
                s.trace.len(),
                s.task.as_mut_ptr(),
                s.task.len(),
                s.counters.as_mut_ptr(),
                TASK_COUNTERS,
                s.heap.as_mut_ptr(),
                s.heap.len(),
                s.cleanup.as_mut_ptr(),
                s.cleanup.len(),
                &mut s.native_calls,
            )
        })
    }
    /// Destroy the task before unloading. On an unexpected destruction failure,
    /// retain the handle and mapping instead of making its code unavailable.
    pub fn finish(&mut self) -> Result<(), Error> {
        if let Some(code) = self.destruction_failure {
            return Err(Error::Protocol(code));
        }
        if let Some(owner) = self.owner {
            // SAFETY: exclusive handle was issued by create in this mapping;
            // wrapper destroys synchronously and clears ownership on success.
            let code = unsafe { (self.destroy)(owner.as_ptr()) };
            if code != 0 {
                self.destruction_failure = Some(code);
                return status(code);
            }
            self.owner = None;
        }
        Ok(())
    }
    pub fn active(&self) -> bool {
        self.owner.is_some()
    }
}
impl Drop for Driver {
    fn drop(&mut self) {
        if self.finish().is_err()
            && let Some(library) = self.library.take()
        {
            // Failed native destruction keeps its original code mapped forever.
            std::mem::forget(library);
        }
    }
}
