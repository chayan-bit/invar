//! Test executables never have a writable descriptor in the test process.
use std::{
    fs, io,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};
#[cfg(unix)]
use std::{
    io::Write,
    os::unix::fs::{DirBuilderExt, PermissionsExt},
    process::{Command, Stdio},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

/// The writer is a child, so unrelated forks cannot inherit its writable fd.
/// Wait for that child, sync through a read-only fd, then atomically publish.
#[cfg(unix)]
#[allow(dead_code)]
pub fn write(path: &Path, contents: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "executable needs a parent"))?;
    let directory = parent.join(format!(
        ".executable-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::DirBuilder::new().mode(0o700).create(&directory)?;
    let result = (|| {
        let staged = directory.join("program");
        let mut child = Command::new("/bin/sh")
            .args(["-c", "umask 077; exec /bin/cat > \"$1\"", "invar-fixture"])
            .arg(&staged)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        let written = child.stdin.take().unwrap().write_all(contents);
        if written.is_err() {
            let _ = child.kill();
        }
        let status = child.wait()?;
        written?;
        if !status.success() {
            return Err(io::Error::other(format!("fixture writer exited {status}")));
        }
        let file = fs::File::open(&staged)?;
        file.set_permissions(fs::Permissions::from_mode(0o700))?;
        file.sync_all()?;
        drop(file);
        fs::rename(staged, path)
    })();
    let cleanup = fs::remove_dir_all(directory);
    result.and(cleanup)
}

/// Windows children inherit no handle the test process did not mark
/// inheritable, and std never does, so the writer needs no child here.
#[cfg(windows)]
#[allow(dead_code)]
pub fn write(path: &Path, contents: &[u8]) -> io::Result<()> {
    let staged = path.with_file_name(format!(
        ".executable-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let result = fs::File::create_new(&staged)
        .and_then(|mut file| {
            io::Write::write_all(&mut file, contents)?;
            file.sync_all()
        })
        .and_then(|()| fs::rename(&staged, path));
    if result.is_err() {
        let _ = fs::remove_file(&staged);
    }
    result
}
