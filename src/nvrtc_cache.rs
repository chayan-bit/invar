//! Opt-in on-disk cache of successful `invar-nvrtc` compiles for test runs
//! (`INVAR_NVRTC_CACHE_DIR`; the remote CUDA runner sets it).
//!
//! A key binds the compiler digest (every invar-cuda, invar-device and helper
//! source file plus Cargo.lock), the NVRTC version actually loaded, the helper
//! arguments (CUDA version, sm, mode) and the exact source bytes, so a changed
//! source, option or compiler can never hit. Each entry stores a SHA-256 over
//! its key and payload (the NVRTC log and the CUBIN); an entry that fails it
//! is a miss and is overwritten. The helper stores clean compiles and GNET
//! compiles with their warning log; refusals and other compiles with a log
//! always run, so tests of refusals and log bounds still compile.
//! Production builds never set the variable.
#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

const FORMAT: &[u8] = b"invar-nvrtc-cache-v2\0";
/// Entries beyond this many bytes evict the least recently used first.
pub const MAX_BYTES: u64 = 4 << 30;

pub fn key(compiler: &str, nvrtc: (u32, u32), args: &[String], source: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(FORMAT);
    for part in [
        compiler.as_bytes(),
        format!("{}.{}", nvrtc.0, nvrtc.1).as_bytes(),
    ]
    .into_iter()
    .chain(args.iter().map(String::as_bytes))
    .chain([source])
    {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    hex(&hash.finalize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn path(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{key}.cubin"))
}

fn seal(key: &str, payload: &[u8]) -> [u8; 32] {
    Sha256::new()
        .chain_update(key.as_bytes())
        .chain_update(payload)
        .finalize()
        .into()
}

/// The verified NVRTC log and CUBIN for `key`, or `None` (absent, unreadable
/// or corrupt).
pub fn get(dir: &Path, key: &str) -> Option<(String, Vec<u8>)> {
    let file = path(dir, key);
    let bytes = fs::read(&file).ok()?;
    let (digest, payload) = bytes.split_at_checked(32)?;
    if digest != seal(key, payload).as_slice() {
        return None;
    }
    let (length, rest) = payload.split_at_checked(8)?;
    let length = usize::try_from(u64::from_le_bytes(length.try_into().ok()?)).ok()?;
    let (log, image) = rest.split_at_checked(length)?;
    let log = String::from_utf8(log.to_vec()).ok()?;
    if image.is_empty() {
        return None;
    }
    // Recency for eviction; a failed touch only ages the entry.
    let _ = fs::File::options()
        .write(true)
        .open(&file)
        .and_then(|f| f.set_modified(SystemTime::now()));
    Some((log, image.to_vec()))
}

/// Store `log` and `image` atomically (write, then rename over), then evict the least
/// recently used entries until the directory holds at most `max_bytes`.
pub fn put(
    dir: &Path,
    key: &str,
    log: &str,
    image: &[u8],
    max_bytes: u64,
) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut payload = (log.len() as u64).to_le_bytes().to_vec();
    payload.extend_from_slice(log.as_bytes());
    payload.extend_from_slice(image);
    let mut bytes = seal(key, &payload).to_vec();
    bytes.extend_from_slice(&payload);
    let partial = dir.join(format!("{key}.{}.partial", std::process::id()));
    fs::write(&partial, bytes)?;
    fs::rename(&partial, path(dir, key))?;
    evict(dir, max_bytes)
}

fn evict(dir: &Path, max_bytes: u64) -> std::io::Result<()> {
    let mut files = Vec::new();
    for item in fs::read_dir(dir)? {
        let item = item?;
        let meta = item.metadata()?;
        if meta.is_file() {
            files.push((meta.modified()?, meta.len(), item.path()));
        }
    }
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    files.sort();
    for (_, len, file) in files {
        if total <= max_bytes {
            break;
        }
        // Another run may have removed it already.
        if fs::remove_file(&file).is_ok() {
            total -= len;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("invar-nvrtc-cache-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn every_key_input_separates_entries() {
        let args = ["12".to_owned(), "9".into(), "86".into(), "--gnet-v5".into()];
        let base = key("c", (12, 9), &args, b"src");
        assert_ne!(base, key("d", (12, 9), &args, b"src"));
        assert_ne!(base, key("c", (12, 4), &args, b"src"));
        assert_ne!(base, key("c", (12, 9), &args[..3], b"src"));
        assert_ne!(base, key("c", (12, 9), &args, b"src "));
        // Length framing: moving a byte between fields changes the key.
        assert_ne!(key("ab", (1, 2), &[], b"c"), key("a", (1, 2), &[], b"bc"));
    }

    #[test]
    fn a_stored_entry_round_trips_and_a_corrupt_or_renamed_one_misses() {
        let dir = scratch("roundtrip");
        let (a, b) = (key("c", (12, 9), &[], b"a"), key("c", (12, 9), &[], b"b"));
        assert_eq!(get(&dir, &a), None);
        put(&dir, &a, "warning: w", b"\x7fELF image", MAX_BYTES).unwrap();
        assert_eq!(
            get(&dir, &a),
            Some(("warning: w".to_owned(), b"\x7fELF image".to_vec()))
        );
        // An entry copied under another key fails its seal.
        fs::copy(path(&dir, &a), path(&dir, &b)).unwrap();
        assert_eq!(get(&dir, &b), None);
        // One flipped byte anywhere is a miss, never a different image.
        let mut bytes = fs::read(path(&dir, &a)).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        fs::write(path(&dir, &a), &bytes).unwrap();
        assert_eq!(get(&dir, &a), None);
        fs::write(path(&dir, &a), &bytes[..10]).unwrap();
        assert_eq!(get(&dir, &a), None);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_directory_never_grows_past_its_cap() {
        let dir = scratch("cap");
        for i in 0..8u8 {
            let k = key("c", (12, 9), &[], &[i]);
            put(&dir, &k, "", &[i; 1000], 3000).unwrap();
        }
        let total: u64 = fs::read_dir(&dir)
            .unwrap()
            .map(|f| f.unwrap().metadata().unwrap().len())
            .sum();
        assert!((1040..=3000).contains(&total), "{total}");
        fs::remove_dir_all(&dir).unwrap();
    }
}
