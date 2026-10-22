//! Shared single-descriptor file admission and atomic publication for Metal
//! AOT containers and binary archives. Never give a checked path back to a reader.
use std::{
    fs,
    io::{Read, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Permissions,
    Size,
}
static NEXT: AtomicU64 = AtomicU64::new(0);

pub fn read_capped(path: &Path, cap: usize) -> Result<Vec<u8>, Error> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(
        &mut options,
        libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
    );
    let file = match options.open(path) {
        Ok(file) => file,
        #[cfg(unix)]
        Err(e) if e.raw_os_error() == Some(libc::ELOOP) => return Err(Error::Permissions),
        Err(e) => return Err(Error::Io(e)),
    };
    let meta = file.metadata().map_err(Error::Io)?;
    #[cfg(unix)]
    let shared = std::os::unix::fs::PermissionsExt::mode(&meta.permissions()) & 0o022 != 0;
    #[cfg(not(unix))]
    let shared = false;
    if !meta.is_file() || shared {
        return Err(Error::Permissions);
    }
    if meta.len() > cap as u64 {
        return Err(Error::Size);
    }
    let mut bytes = Vec::new();
    file.take(cap as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(Error::Io)?;
    if bytes.len() > cap {
        return Err(Error::Size);
    }
    Ok(bytes)
}

pub fn write_atomic(output: &Path, bytes: &[u8]) -> Result<(), Error> {
    let name = output.file_name().ok_or_else(|| {
        Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "output has no file name",
        ))
    })?;
    let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
    let temp = output.with_file_name(format!(
        ".{}.{}.{sequence}.tmp",
        name.to_string_lossy(),
        std::process::id()
    ));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    // Only remove a temporary file this invocation successfully created.
    let mut file = options.open(&temp).map_err(Error::Io)?;
    let result = (|| {
        file.write_all(bytes).map_err(Error::Io)?;
        file.sync_all().map_err(Error::Io)?;
        fs::rename(&temp, output).map_err(Error::Io)?;
        fs::File::open(
            output
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        )
        .and_then(|dir| dir.sync_all())
        .map_err(Error::Io)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concurrent_publication_never_returns_partial_bytes() {
        let root = std::env::temp_dir().join(format!("p5b-files-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        assert!(
            matches!(write_atomic(Path::new(""),b"x"),Err(Error::Io(error)) if error.kind()==std::io::ErrorKind::InvalidInput)
        );
        let file = root.join("cache");
        assert!(
            matches!(read_capped(&file,4096),Err(Error::Io(error)) if error.kind()==std::io::ErrorKind::NotFound)
        );
        write_atomic(&file, &vec![0; 4096]).unwrap();
        let handles: Vec<_> = (0..8)
            .map(|word| {
                let file = file.clone();
                std::thread::spawn(move || {
                    for _ in 0..16 {
                        write_atomic(&file, &vec![word; 4096]).unwrap();
                    }
                })
            })
            .collect();
        for _ in 0..128 {
            let bytes = read_capped(&file, 4096).unwrap();
            assert_eq!(bytes.len(), 4096);
            assert!(bytes.iter().all(|b| *b == bytes[0]));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        let bytes = read_capped(&file, 4096).unwrap();
        assert_eq!(bytes.len(), 4096);
        assert!(bytes.iter().all(|b| *b == bytes[0]));
        assert!(matches!(read_capped(&file, 32), Err(Error::Size)));
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&file, root.join("link")).unwrap();
            assert!(matches!(
                read_capped(&root.join("link"), 4096),
                Err(Error::Permissions)
            ));
            fs::set_permissions(&file, std::os::unix::fs::PermissionsExt::from_mode(0o666))
                .unwrap();
            assert!(matches!(read_capped(&file, 4096), Err(Error::Permissions)));
        }
        fs::remove_dir_all(root).unwrap();
    }
}
