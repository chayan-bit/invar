//! Bounded CPU regression child processes; no semantic engine substitute.
#[allow(dead_code)]
#[path = "../../test-support/child_deadline.rs"]
mod child_deadline;

use std::{
    fs,
    os::unix::fs::DirBuilderExt,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
const MAX_LOG: u64 = 64 * 1024;
struct ChildGroup {
    child: Child,
    reaped: bool,
}
impl Drop for ChildGroup {
    fn drop(&mut self) {
        if !self.reaped {
            // An unreaped root PID cannot be reused; this is only its owned group.
            let group = format!("-{}", self.child.id());
            let _ = Command::new("/bin/kill")
                .args(["-KILL", "--", &group])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            let _ = self.child.wait();
        }
    }
}

pub struct CpuTools {
    pub dir: PathBuf,
    serial: u32,
}
impl CpuTools {
    pub fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "invar-owner-return-cpu-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new().mode(0o700).create(&dir).unwrap();
        Self { dir, serial: 0 }
    }
    /// timeout and a session-owned process group bound all descendants. The parent
    /// kills/reaps that group on its own deadline too; files never become pipes.
    pub fn run(
        &mut self,
        program: &str,
        args: &[&str],
        seconds: u64,
        address: u64,
        file: u64,
    ) -> Output {
        let wall_seconds = child_deadline::seconds(seconds);
        self.serial += 1;
        let stdout = self.dir.join(format!("stdout-{}", self.serial));
        let stderr = self.dir.join(format!("stderr-{}", self.serial));
        let mut cmd = Command::new("/usr/bin/setsid");
        cmd.arg("--wait")
            .args([
                "/usr/bin/timeout",
                "--signal=KILL",
                &format!("{wall_seconds}s"),
                "/usr/bin/prlimit",
            ])
            .args([
                format!("--as={address}"),
                format!("--cpu={}", seconds + 1),
                format!("--fsize={file}"),
                "--nofile=64".into(),
                "--core=0".into(),
                "--".into(),
            ])
            .arg(program)
            .args(args)
            .current_dir(&self.dir)
            .stdin(Stdio::null())
            .stdout(fs::File::create(&stdout).unwrap())
            .stderr(fs::File::create(&stderr).unwrap());
        let mut child = ChildGroup {
            child: cmd.spawn().unwrap(),
            reaped: false,
        };
        let start = Instant::now();
        let status = loop {
            if let Some(status) = child.child.try_wait().unwrap() {
                child.reaped = true;
                break status;
            }
            let over_log = [&stdout, &stderr]
                .iter()
                .any(|p| fs::metadata(p).unwrap().len() > MAX_LOG);
            if over_log || start.elapsed() > Duration::from_secs(wall_seconds + 3) {
                drop(child);
                panic!(
                    "native child exceeded parent deadline/log cap; owned process group killed and reaped"
                );
            }
            thread::sleep(Duration::from_millis(10));
        };
        assert!(fs::metadata(&stdout).unwrap().len() <= MAX_LOG);
        assert!(fs::metadata(&stderr).unwrap().len() <= MAX_LOG);
        Output {
            status,
            stdout: fs::read(stdout).unwrap(),
            stderr: fs::read(stderr).unwrap(),
        }
    }
}
impl Drop for CpuTools {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap();
    }
}
pub fn absent(path: &Path) {
    assert!(
        !path.exists(),
        "test scratch directory leaked: {}",
        path.display()
    );
}
