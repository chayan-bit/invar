use std::{path::PathBuf, process::Command};

use sha2::{Digest, Sha256};

const BROKEN: &str = "fn take_it(xs: take Array<i32>) -> i32 {
    return xs[0u32];
}

fn main() -> i32 {
    var xs: Array<i32> = [4i32, 5i32];
    let first: i32 = take_it(move xs);
    return first + xs[1u32];
}
";

fn fixture(name: &str, contents: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("invar-dx-22-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("program.inv");
    std::fs::write(&path, contents).unwrap();
    path
}

fn check(args: &[&str], path: &std::path::Path) -> (i32, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .arg("check")
        .arg(path)
        .args(args)
        .output()
        .unwrap();
    (
        output.status.code().unwrap(),
        String::from_utf8(output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap(),
    )
}

#[test]
fn issue_dx_22_check_fix_prints_a_diff_and_leaves_the_file_alone() {
    let path = fixture("diff", BROKEN);
    let (status, stdout, stderr) = check(&["--fix"], &path);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.starts_with("--- a/"), "{stdout}");
    assert!(
        stdout.contains("-fn take_it(xs: take Array<i32>)"),
        "{stdout}"
    );
    assert!(
        stdout.contains("+fn take_it(xs: read Array<i32>)"),
        "{stdout}"
    );
    assert!(
        stdout.contains("-    let first: i32 = take_it(move xs);"),
        "{stdout}"
    );
    assert!(
        stdout.contains("+    let first: i32 = take_it(xs);"),
        "{stdout}"
    );
    assert!(
        stderr.contains("applied Make 'take_it' borrow its argument"),
        "{stderr}"
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), BROKEN);
}

#[test]
fn issue_dx_22_check_fix_write_updates_the_file_once() {
    let path = fixture("write", BROKEN);
    let (status, stdout, _) = check(&["--fix", "--write"], &path);
    assert_eq!(status, 0);
    assert!(stdout.contains("+    let first: i32 = take_it(xs);"));
    let fixed = std::fs::read_to_string(&path).unwrap();
    assert_ne!(fixed, BROKEN);
    assert!(!path.with_extension("inv.invar-fix").exists());
    let (status, _, _) = check(&[], &path);
    assert_eq!(status, 0, "the written file checks clean");
    // Idempotent: nothing left to fix, nothing printed, nothing rewritten.
    let (status, stdout, stderr) = check(&["--fix", "--write"], &path);
    assert_eq!((status, stdout.as_str(), stderr.as_str()), (0, "", ""));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), fixed);
}

#[test]
fn issue_dx_22_check_fix_reports_an_error_it_cannot_fix() {
    let source = "fn main() -> i32 {\n    return nothing_close;\n}\n";
    let path = fixture("unfixable", source);
    let (status, stdout, stderr) = check(&["--fix", "--write"], &path);
    assert_eq!(status, 1);
    assert_eq!(stdout, "");
    assert!(stderr.contains("NAME_UNKNOWN"), "{stderr}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), source);
}

#[test]
fn issue_dx_22_check_fix_rejects_unknown_flags() {
    let path = fixture("flags", BROKEN);
    let (code, _, usage) = check(&["--fix", "--bogus"], &path);
    assert_eq!(code, 2);
    // The usage printed for a bad flag shows the valid syntax.
    assert!(
        usage.contains("invar check PATH --fix [--write]"),
        "{usage}"
    );
    assert_eq!(check(&["--fix", "--write", "--write"], &path).0, 2);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), BROKEN);
}

#[test]
fn issue_dx_22_write_refuses_a_file_changed_after_the_check() {
    let path = fixture("stale", BROKEN);
    let checked = Sha256::digest(BROKEN.as_bytes());
    let edited = format!("{BROKEN}// edited after the check\n");
    std::fs::write(&path, &edited).unwrap();
    let error = invar::fixes::write_if_unchanged(&path, &checked, b"fixed").unwrap_err();
    assert!(error.contains("changed after it was checked"), "{error}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), edited);
    let current = Sha256::digest(edited.as_bytes());
    invar::fixes::write_if_unchanged(&path, &current, b"fixed").unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "fixed");
    assert!(!path.with_extension("inv.invar-fix").exists());
}
