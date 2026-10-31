//! DX-20: `invar debug PATH` builds with `--debug` in a private
//! directory and runs gdb or lldb with the Invar helpers loaded. The live
//! sessions skip, by name, where the debugger is missing or may not attach.
#![cfg(feature = "native")]

use super::scratch_support as scratch;

const TRAP: &str = "fn main() -> i32 {\n    let x: i32 = 2147483647;\n    return x + 1;\n}\n";

fn fixture(name: &str) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!("invar-dx-20-{name}-{}", scratch::unique()));
    std::fs::create_dir(&directory).unwrap();
    let source = directory.join("trap.inv");
    std::fs::write(&source, TRAP).unwrap();
    source
}

fn invar(args: &[&str]) -> std::process::Command {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_invar"));
    command.args(args);
    command
}

fn found(debugger: &str) -> bool {
    std::process::Command::new(debugger)
        .arg("--version")
        .output()
        .is_ok()
}

#[test]
#[cfg(target_os = "linux")]
fn debug_command_gdb_session() {
    if !found("gdb") {
        eprintln!("SKIP debug_command_gdb_session: gdb not found");
        return;
    }
    let source = fixture("gdb");
    let path = source.to_str().unwrap();
    let session = invar(&[
        "debug", path, "--gdb", "--", "-batch", "-ex", "run", "-ex", "bt",
    ])
    .output()
    .unwrap();
    let stdout = String::from_utf8_lossy(&session.stdout);
    let _ = std::fs::remove_dir_all(source.parent().unwrap());
    assert_eq!(session.status.code(), Some(0), "{session:?}");
    assert!(stdout.contains("Debug info: DWARF 5"), "{stdout}");
    assert!(
        stdout.contains(&format!("invar trap: INTEGER_OVERFLOW at {path}:3:14-3:15")),
        "{stdout}"
    );
    assert!(stdout.contains("main () at trap.inv:3"), "{stdout}");
}

#[test]
#[cfg(target_os = "macos")]
fn debug_command_lldb_session() {
    if !found("lldb") {
        eprintln!("SKIP debug_command_lldb_session: lldb not found");
        return;
    }
    let source = fixture("lldb");
    let path = source.to_str().unwrap();
    let session = invar(&[
        "debug", path, "--lldb", "--", "--batch", "-o", "run", "-o", "bt",
    ])
    .output()
    .unwrap();
    let stdout = String::from_utf8_lossy(&session.stdout);
    let stderr = String::from_utf8_lossy(&session.stderr);
    let _ = std::fs::remove_dir_all(source.parent().unwrap());
    assert!(stdout.contains("Debug info: DWARF 5"), "{stdout}");
    // macOS refuses to let lldb launch a process unless Developer Mode is on
    // (`DevToolsSecurity -enable`) or the session is interactive.
    if stderr.contains("cannot get permission to debug") {
        eprintln!("SKIP debug_command_lldb_session: lldb may not launch processes here");
        return;
    }
    assert!(
        stdout.contains(&format!("invar trap: INTEGER_OVERFLOW at {path}:3:14-3:15")),
        "{stdout}\n{stderr}"
    );
    assert!(stdout.contains("main at trap.inv:3"), "{stdout}");
}

#[test]
fn debug_command_refuses_a_missing_debugger() {
    let source = fixture("missing");
    let path = source.to_str().unwrap();
    let empty = source.parent().unwrap().join("bin");
    std::fs::create_dir(&empty).unwrap();
    let refused = invar(&["debug", path, "--gdb"])
        .env("PATH", &empty)
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(source.parent().unwrap());
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert_eq!(refused.status.code(), Some(2), "{refused:?}");
    assert!(stderr.contains("DEBUG_UNSUPPORTED"), "{stderr}");
    assert!(stderr.contains("gdb was not found on PATH"), "{stderr}");
    assert!(refused.stdout.is_empty(), "nothing is built: {refused:?}");
}

/// Prints each argument on its own line, then the mode of the program's
/// directory, in place of a real debugger.
const FAKE_DEBUGGER: &str = "#!/bin/sh
[ \"$1\" = --version ] && exit 0
for arg; do printf '%s\\n' \"$arg\"; last=$arg; done
ls -ld \"$(dirname \"$last\")\"
";

/// The session directory is the caller's alone, and lldb gets the helper
/// path double-quoted, here under a temporary directory with a space, quotes,
/// a backtick and a backslash (debug command review). Apple lldb-2103 imports such a
/// quoted path; an unquoted one splits at the space.
#[test]
#[cfg(unix)]
fn debug_command_private_directory_and_quoted_helper() {
    use std::os::unix::fs::PermissionsExt;
    let source = fixture("fake");
    let path = source.to_str().unwrap();
    let root = source.parent().unwrap();
    let bin = root.join("bin");
    let temp = root.join("tmp \"dir\" `x` \\y");
    std::fs::create_dir(&bin).unwrap();
    std::fs::create_dir(&temp).unwrap();
    for name in ["gdb", "lldb"] {
        std::fs::write(bin.join(name), FAKE_DEBUGGER).unwrap();
        std::fs::set_permissions(bin.join(name), std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let paths = std::env::var_os("PATH").unwrap();
    let paths =
        std::env::join_paths(std::iter::once(bin.clone()).chain(std::env::split_paths(&paths)))
            .unwrap();
    let run = |debugger: &str| {
        let output = invar(&["debug", path, debugger])
            .env("PATH", &paths)
            .env("TMPDIR", &temp)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(0), "{output:?}");
        String::from_utf8(output.stdout).unwrap()
    };
    let gdb = run("--gdb");
    let lldb = run("--lldb");
    let _ = std::fs::remove_dir_all(root);
    for stdout in [&gdb, &lldb] {
        assert!(
            stdout.lines().any(|line| line.starts_with("drwx------")),
            "{stdout}"
        );
    }
    let quoted = format!(
        "command script import \"{}",
        temp.display()
            .to_string()
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
    );
    let import = lldb
        .lines()
        .find(|line| line.starts_with("command script import"));
    let import = import.unwrap_or_else(|| panic!("{lldb}"));
    assert!(import.starts_with(&quoted), "{import}");
    assert!(import.ends_with("/invar_lldb.py\""), "{import}");
}

#[test]
fn debug_command_rejects_bad_options() {
    for args in [
        &["debug", "trap.inv", "--fuel", "0"][..],
        &["debug", "trap.inv", "--fuel"],
        &["debug", "trap.inv", "--engine", "cpu-parallel"],
        &["debug", "trap.inv", "--c-library"],
    ] {
        let output = invar(args).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert!(output.stdout.is_empty(), "{args:?}");
    }
}
