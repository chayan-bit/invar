//! Test harness wall deadlines only. Never use this module in production code.
use std::{ffi::OsStr, path::Path, process::Command, time::Duration};

pub fn parse_scale(value: &str) -> Result<u32, &'static str> {
    if value.is_empty() || !value.bytes().all(|c| c.is_ascii_digit()) {
        return Err("expected an integer in 1..=64");
    }
    value
        .parse::<u32>()
        .ok()
        .filter(|n| (1..=64).contains(n))
        .ok_or("expected an integer in 1..=64")
}

pub fn wall(base: Duration) -> Duration {
    let scale = match std::env::var("INVAR_TEST_TIMEOUT_SCALE") {
        Ok(value) => {
            parse_scale(&value).expect("INVAR_TEST_TIMEOUT_SCALE: expected an integer in 1..=64")
        }
        Err(std::env::VarError::NotPresent) => 1,
        Err(error) => panic!("INVAR_TEST_TIMEOUT_SCALE: {error}"),
    };
    base.checked_mul(scale)
        .expect("test wall deadline overflow")
}

pub fn seconds(base: u64) -> u64 {
    wall(Duration::from_secs(base)).as_secs()
}

/// GNU timeout owns the process group; TERM then KILL bounds uncooperative children.
/// 124 is timeout; forced kill is SIGKILL (shell status 137). Compilation and execution get separate calls.
pub fn command(base_seconds: u64) -> Command {
    let mut command = Command::new("timeout");
    command.args(["--kill-after=2", &seconds(base_seconds).to_string()]);
    command
}

/// C++17 with UBSan aborting on the first report: the emitted-CPU harness builds.
pub const CXX17_UBSAN_O1: &[&str] = &[
    "-std=c++17",
    "-O1",
    "-fsanitize=undefined",
    "-fno-sanitize-recover=all",
];
pub const CXX17_UBSAN_O2: &[&str] = &[
    "-std=c++17",
    "-O2",
    "-fsanitize=undefined",
    "-fno-sanitize-recover=all",
];

/// Compile a C++ test harness `source` into `exe` with `$CXX` (else `c++`) and `flags`,
/// under a 120-second base wall deadline; panics with the compiler's stderr on failure.
pub fn cxx<S: AsRef<OsStr>>(flags: &[S], source: &Path, exe: &Path) {
    let compiler = std::env::var_os("CXX").unwrap_or_else(|| "c++".into());
    let output = command(120)
        .arg(&compiler)
        .args(flags)
        .arg(source)
        .arg("-o")
        .arg(exe)
        .output()
        .unwrap_or_else(|error| panic!("timeout {compiler:?}: {error}"));
    assert!(
        output.status.success(),
        "{compiler:?} {}: {}\n{}",
        source.display(),
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}
