//! Reviewed synchronous ABI8 loader. Only compiler-owned trusted libraries may
//! implement this private signature; no public L1-04 tags or pointers are added.
use libloading::Library;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, marker::PhantomData, path::Path, rc::Rc};

type Entry =
    unsafe extern "C" fn(*const i64, usize, *mut i64, usize, *mut u64, usize, u64, u64, u64) -> u32;
type Identity = unsafe extern "C" fn(*mut u8, usize) -> u32;
const MAX_LIBRARY: u64 = 64 * 1024 * 1024;
#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Library(String),
    Identity,
    Symbol(String),
    Protocol(u32),
    Limit,
    Allocation,
}
#[derive(Debug)]
pub struct Invocation {
    pub output: [i64; 8],
    pub trace: Vec<u64>,
}
#[derive(Clone, Copy, Debug)]
pub enum Carrier {
    I32,
    Word,
    Bool,
}
/// A retained synchronous code mapping. No native pointer or function escapes.
/// The Rc marker forbids Send/Sync even though each call has no shared state.
/// ```compile_fail
/// fn send<T: Send>() {} send::<invar_foreign::numeric::Driver>();
/// ```
/// ```compile_fail
/// fn sync<T: Sync>() {} sync::<invar_foreign::numeric::Driver>();
/// ```
pub struct Driver {
    _library: Library,
    entry: Entry,
    params: Vec<Carrier>,
    _thread: PhantomData<Rc<()>>,
}
fn valid_symbol(s: &str) -> bool {
    s.starts_with("invar_")
        && s.len() <= 128
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}
/// Stream a capped regular file; no caller digest selects a path or creates a
/// source permit. Root owns and keeps this private file unchanged during load.
pub fn file_digest(path: &Path) -> Result<[u8; 32], Error> {
    let metadata = std::fs::symlink_metadata(path).map_err(|e| Error::Library(e.to_string()))?;
    if !metadata.file_type().is_file() || metadata.len() == 0 || metadata.len() > MAX_LIBRARY {
        return Err(Error::Limit);
    }
    let mut file = File::open(path).map_err(|e| Error::Library(e.to_string()))?;
    let mut digest = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0; 16384];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|e| Error::Library(e.to_string()))?;
        if n == 0 {
            break;
        }
        bytes = bytes.checked_add(n as u64).ok_or(Error::Limit)?;
        if bytes > MAX_LIBRARY {
            return Err(Error::Limit);
        }
        digest.update(&buffer[..n]);
    }
    if bytes != metadata.len() {
        return Err(Error::Identity);
    }
    Ok(digest.finalize().into())
}
impl Driver {
    /// Load a trusted generated wrapper whose exact library bytes and embedded
    /// prelink identity match the source host's locally issued receipt. Library
    /// constructors are a trusted native boundary, not sandboxed execution.
    pub fn load(
        path: &Path,
        library_hash: [u8; 32],
        identity: [u8; 32],
        symbol: &str,
        params: Vec<Carrier>,
    ) -> Result<Self, Error> {
        if !path.is_absolute() || !valid_symbol(symbol) || params.len() > 128 {
            return Err(Error::Limit);
        }
        if file_digest(path)? != library_hash {
            return Err(Error::Identity);
        }
        // SAFETY: the selected compiler-generated library implements the fixed
        // signatures below; its bytes were checked in the owned private directory.
        let library = unsafe { Library::new(path) }.map_err(|e| Error::Library(e.to_string()))?;
        let id_symbol = format!("{symbol}_identity");
        // SAFETY: generated identity function writes at most the supplied length.
        let id = *unsafe { library.get::<Identity>(id_symbol.as_bytes()) }
            .map_err(|e| Error::Symbol(e.to_string()))?;
        let mut actual = [0; 32];
        // SAFETY: initialized, writable 32-byte local; retained library/function.
        if unsafe { id(actual.as_mut_ptr(), actual.len()) } != 0 || actual != identity {
            return Err(Error::Identity);
        }
        // SAFETY: the trusted wrapper assigns Entry's exact versioned signature.
        let entry = *unsafe { library.get::<Entry>(symbol.as_bytes()) }
            .map_err(|e| Error::Symbol(e.to_string()))?;
        // Recheck bytes after loader constructors/symbol lookup; directory/file
        // immutability remains an explicit trust assumption, as for other loaders.
        if file_digest(path)? != library_hash {
            return Err(Error::Identity);
        }
        Ok(Self {
            _library: library,
            entry,
            params,
            _thread: PhantomData,
        })
    }
    /// All buffers are initialized and retained until a confirmed synchronous
    /// C return. The noexcept wrapper catches C++ exceptions and retains no
    /// pointers/workers. No Rust unwind is used for Invar source failure.
    pub fn run(&mut self, input: &[i64], limits: [u64; 3]) -> Result<Invocation, Error> {
        let [fuel, transitions, events] = limits;
        if input.len() != self.params.len()
            || fuel == 0
            || fuel > 1_000_000
            || transitions > 1_000_000
            || events > 1_000_000
        {
            return Err(Error::Limit);
        }
        for (word, carrier) in input.iter().zip(&self.params) {
            let valid = match carrier {
                Carrier::I32 => i32::try_from(*word).is_ok(),
                Carrier::Word => u32::try_from(*word).is_ok(),
                Carrier::Bool => *word == 0 || *word == 1,
            };
            if !valid {
                return Err(Error::Limit);
            }
        }
        let len = usize::try_from(events)
            .ok()
            .and_then(|n| n.checked_mul(7))
            .ok_or(Error::Limit)?;
        let mut owned_input = Vec::new();
        owned_input
            .try_reserve_exact(input.len())
            .map_err(|_| Error::Allocation)?;
        owned_input.extend_from_slice(input);
        let mut trace = Vec::new();
        trace
            .try_reserve_exact(len)
            .map_err(|_| Error::Allocation)?;
        trace.resize(len, 0);
        let mut output = [0; 8];
        // SAFETY: fixed trusted noexcept signature; all full-capacity buffers
        // are initialized/aligned/live and disjoint, including empty slices.
        // Library remains owned through return, validation and caller decoding.
        let status = unsafe {
            (self.entry)(
                owned_input.as_ptr(),
                owned_input.len(),
                output.as_mut_ptr(),
                output.len(),
                trace.as_mut_ptr(),
                trace.len(),
                fuel,
                transitions,
                events,
            )
        };
        if status != 0 {
            return Err(Error::Protocol(status));
        }
        Ok(Invocation { output, trace })
    }
}

/// Whether a compiler child has exited, observed without reaping it: the
/// leader stays a zombie, so its process-group id cannot be reused before the
/// host calls `terminate_compiler_group` and only then reaps it.
#[cfg(unix)]
pub fn compiler_exited(child: &mut std::process::Child) -> std::io::Result<bool> {
    let pid = libc::id_t::from(child.id());
    // SAFETY: an all-zero siginfo_t is a valid value; waitid writes only it.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    // SAFETY: `pid` is our unreaped child; WNOWAIT leaves it waitable and
    // WNOHANG returns at once. `info` is a valid, writable siginfo_t.
    let rc = unsafe {
        libc::waitid(
            libc::P_PID,
            pid,
            &mut info,
            libc::WEXITED | libc::WNOWAIT | libc::WNOHANG,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: waitid filled `info`, or left it zeroed when nothing exited.
    Ok(unsafe { info.si_pid() } != 0)
}
/// Without process groups, exit is observed by reaping.
#[cfg(not(unix))]
pub fn compiler_exited(child: &mut std::process::Child) -> std::io::Result<bool> {
    child.try_wait().map(|s| s.is_some())
}

/// Terminate only a compiler process group created by the source host with
/// process_group(0). This is not a general process-control entry point.
#[cfg(unix)]
pub fn terminate_compiler_group(child_id: u32) {
    let Ok(id) = i32::try_from(child_id) else {
        return;
    };
    if id <= 1 {
        return;
    }
    unsafe extern "C" {
        fn kill(pid: i32, signal: i32) -> i32;
    }
    // SAFETY: host retains the child/group identity while compiler/pipes remain
    // pending. Descendants stay in this private group; no pointers are passed.
    let _ = unsafe { kill(-id, 9) };
}
