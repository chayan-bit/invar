//! Test executables never have a writable descriptor in the test process.
use std::{
    fs,
    io::{self, Write},
    os::unix::fs::{DirBuilderExt, PermissionsExt},
    path::Path,
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

/// The writer is a child, so unrelated forks cannot inherit its writable fd.
/// Wait for that child, sync through a read-only fd, then atomically publish.
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
