//! Shared helpers for the reviewed numeric loaders: loader errors, scalar
//! carriers, capped file digests and compiler process-group control.
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};

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
#[derive(Clone, Copy, Debug)]
pub enum Carrier {
    I32,
    Word,
    Bool,
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
