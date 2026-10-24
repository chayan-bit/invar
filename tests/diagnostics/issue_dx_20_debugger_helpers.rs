//! DX-20: the gdb helper (`share/invar/invar_gdb.py`) in a live
//! session of a `--debug` build: typed handle pretty-printers through the
//! host's render hook, the trap breakpoint and `invar-where`. Skips, by
//! name, where gdb is not installed.
#![cfg(all(feature = "native", target_os = "linux"))]

use super::scratch_support as scratch;

const SOURCE: &str = "\
struct Pair { left: i32, right: u32, scale: f32 }

fn pick(xs: read Array<i32>, at: u32) -> i32 {
    return xs[at];
}

enum Choice { Low, High(f64) }

fn main() -> i32 {
    let xs: Array<i32> = [1, 2, 3];
    let fs: Array<f32> = [0.5, 1.5];
    let bs: Array<Bool> = [true, false];
    let pair: Pair = Pair { left: 4, right: 5u32, scale: 1.5 };
    let choice: Choice = Choice::High(2.5f64);
    let first: i32 = pick(xs, 0u32);
    if fs[0u32] < 1.0 and bs[1u32] == false {
        return first + pick(xs, 7u32) + pair.left;
    }
    return first;
}
";

#[test]
fn issue_dx_20_debugger_helpers_gdb_session() {
    if std::process::Command::new("gdb")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("SKIP issue_dx_20_debugger_helpers_gdb_session: gdb not found");
        return;
    }
    let directory = std::env::temp_dir().join(format!("invar-dx-20-gdb-{}", scratch::unique()));
    std::fs::create_dir(&directory).unwrap();
    let source = directory.join("helpers.inv");
    std::fs::write(&source, SOURCE).unwrap();
    let source = source.to_str().unwrap();
    let program = directory.join("program");
    let program = program.to_str().unwrap();
    let build = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "build", source, "--target", "native", "--output", program, "--debug",
        ])
        .output()
        .unwrap();
    assert_eq!(build.status.code(), Some(0), "{build:?}");

    let helper = concat!(env!("CARGO_MANIFEST_DIR"), "/share/invar/invar_gdb.py");
    let mut gdb = std::process::Command::new("gdb");
    gdb.args([
        "--batch",
        "-nx",
        "-iex",
        "set debuginfod enabled off",
        "-x",
        helper,
    ]);
    // Breakpoint 1 is the helper's trap breakpoint, 2 the line breakpoint.
    for command in [
        "break helpers.inv:4",
        "run",
        "info args",
        "print xs",
        "up",
        "print fs",
        "print bs",
        "print pair",
        "print choice",
        "delete 2",
        "continue",
        "invar-where",
        "bt 5",
    ] {
        gdb.args(["-ex", command]);
    }
    let session = gdb.args(["--args", program]).output().unwrap();
    let stdout = String::from_utf8_lossy(&session.stdout);
    let _ = std::fs::remove_dir_all(&directory);
    let trap = format!("INDEX_OUT_OF_BOUNDS at {source}:4:12-4:18");
    for expected in [
        // Frame lines leave handles to `print` and `info args`: gdb 12
        // crashes on a call into the process while it prints a frame line.
        "Breakpoint 2, pick (xs=..., at=0) at helpers.inv:4".to_owned(),
        "xs = Array<i32> [1, 2, 3]\nat = 0\n".to_owned(),
        "$1 = Array<i32> [1, 2, 3]".to_owned(),
        "$2 = Array<f32> [0.5, 1.5]".to_owned(),
        "$3 = Array<Bool> [true, false]".to_owned(),
        "$4 = Pair { left: 4, right: 5, scale: 1.5 }".to_owned(),
        "$5 = Choice::High(2.5)".to_owned(),
        format!("invar trap: {trap}"),
        format!("\n{trap}\n"),
        "pick (xs=..., at=7) at helpers.inv:4".to_owned(),
    ] {
        assert!(
            stdout.contains(&expected),
            "missing {expected:?} in\n{stdout}\n{}",
            String::from_utf8_lossy(&session.stderr)
        );
    }
}
