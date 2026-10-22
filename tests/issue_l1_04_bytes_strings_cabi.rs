use std::process::Command;

use invar::{check, eval, front, mir, mir_eval};

#[cfg(feature = "native")]
fn compile_foreign_fixture() -> (std::path::PathBuf, std::path::PathBuf) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let directory = std::env::temp_dir().join(format!(
        "invar-l1-04-native-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&directory).unwrap();
    let library = directory.join(if cfg!(target_os = "macos") {
        "libforeign.dylib"
    } else {
        "libforeign.so"
    });
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let status = Command::new("cc")
        .arg(if cfg!(target_os = "macos") {
            "-dynamiclib"
        } else {
            "-shared"
        })
        .arg("-fPIC")
        .arg("-std=c11")
        .arg("-Wall")
        .arg("-Werror")
        .arg("-I")
        .arg(root.join("include"))
        .arg(root.join("tests/fixtures/l1_04/foreign.c"))
        .arg("-lm")
        .arg("-o")
        .arg(&library)
        .status()
        .unwrap();
    assert!(status.success());
    (directory, library)
}

#[test]
fn issue_l1_04_bytes_strings_cabi_utf8_and_byte_values_agree() {
    for name in [
        "basic",
        "decode-valid",
        "decode-invalid",
        "slices",
        "empty-normalization",
        "result-propagation",
    ] {
        let path = format!(
            "{}/tests/fixtures/l1_04/{name}.inv",
            env!("CARGO_MANIFEST_DIR")
        );
        let source = std::fs::read(&path).unwrap();
        let checked = check::check(front::parse(&source).unwrap(), true).unwrap();
        let reference = eval::run(&checked, 10_000).unwrap();
        assert_eq!(
            reference.value,
            eval::Value::I32(if name == "decode-invalid" { 1 } else { 7 }),
            "{name}"
        );
        let lowered = mir::lower(&checked).unwrap();
        assert_eq!(
            mir_eval::run(&lowered, 10_000).unwrap(),
            reference,
            "{name}"
        );
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["run", &path, "--engine", "reference", "--format", "json"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            document["result"]["value"],
            if name == "decode-invalid" { 1 } else { 7 }
        );
        #[cfg(feature = "native")]
        {
            let (status, native) = invar::native::run_program(&lowered, 10_000).unwrap();
            assert_eq!(status, 0, "{name}: {native}");
            assert_eq!(native["result"], document["result"], "{name}");
            assert_eq!(native["steps_used"], reference.steps_used, "{name}");
            assert_eq!(
                native["cleanup_trace"],
                serde_json::json!(
                    reference
                        .cleanup_trace
                        .iter()
                        .map(|event| serde_json::json!({"name":event.name,"length":event.length}))
                        .collect::<Vec<_>>()
                ),
                "{name}"
            );
        }
    }
}

#[test]
fn issue_l1_04_bytes_strings_cabi_public_failures() {
    for (name, command, code) in [
        ("bad-byte-escape", "check", "LEX_INVALID_LITERAL"),
        ("bad-text-escape", "check", "LEX_INVALID_LITERAL"),
        ("owner-use-after-move", "check", "OWN_USE_AFTER_MOVE"),
        ("byte-index-oob", "run", "INDEX_OUT_OF_BOUNDS"),
        ("slice-oob", "run", "INDEX_OUT_OF_BOUNDS"),
    ] {
        let path = format!(
            "{}/tests/fixtures/l1_04/{name}.inv",
            env!("CARGO_MANIFEST_DIR")
        );
        let mut arguments = vec![command, &path];
        if command == "run" {
            arguments.extend(["--engine", "reference"]);
        }
        arguments.extend(["--format", "json"]);
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(arguments)
            .output()
            .unwrap();
        assert!(!output.status.success(), "{name}");
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let actual = if command == "check" {
            &document["diagnostics"][0]["code"]
        } else {
            &document["trap"]["category"]
        };
        assert_eq!(actual, code, "{name}: {document}");
        #[cfg(feature = "native")]
        if command == "run" {
            let native = Command::new(env!("CARGO_BIN_EXE_invar"))
                .args([
                    "run",
                    &path,
                    "--engine",
                    "native-serial",
                    "--format",
                    "json",
                ])
                .output()
                .unwrap();
            assert!(!native.status.success(), "{name}");
            let native: serde_json::Value = serde_json::from_slice(&native.stdout).unwrap();
            assert_eq!(native["trap"]["category"], code, "{name}: {native}");
        }
    }
}

#[test]
fn issue_l1_04_bytes_strings_cabi_large_bounded_value() {
    let source = format!(
        "fn main() -> i32 {{ let data: Bytes = b\"{}\"; if len(data) == 131072u32 {{ return 7; }} return 0; }}",
        "x".repeat(131_072)
    );
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let reference = eval::run(&checked, 10_000).unwrap();
    assert_eq!(reference.value, eval::Value::I32(7));
    let lowered = mir::lower(&checked).unwrap();
    assert_eq!(mir_eval::run(&lowered, 10_000).unwrap(), reference);
    #[cfg(feature = "native")]
    {
        let (status, native) = invar::native::run_program(&lowered, 10_000).unwrap();
        assert_eq!(status, 0, "{native}");
        assert_eq!(native["result"]["value"], 7);
    }
}

#[test]
fn issue_l1_04_bytes_strings_cabi_foreign_oracle_declares_and_checks() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/l1_04/foreign-oracle.inv"
    );
    let source = std::fs::read(path).unwrap();
    let checked = check::check(front::parse(&source).unwrap(), true).unwrap();
    assert_eq!(
        eval::run(&checked, 10_000).unwrap().value,
        eval::Value::I32(7)
    );
    let lowered = mir::lower(&checked).unwrap();
    assert_eq!(
        mir_eval::run(&lowered, 10_000).unwrap(),
        eval::run(&checked, 10_000).unwrap()
    );
    #[cfg(feature = "native")]
    {
        let (directory, library) = compile_foreign_fixture();
        let marker = directory.join("constructor-marker");
        let checked_cli = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["check", path, "--format", "json"])
            .env("INVAR_L104_CONSTRUCTOR_MARKER", &marker)
            .output()
            .unwrap();
        assert!(checked_cli.status.success());
        assert!(!marker.exists(), "source check loaded the C library");
        let (status, native) =
            invar::native::run_program_with_library(&lowered, 10_000, Some(&library)).unwrap();
        assert_eq!(status, 0, "{native}");
        assert_eq!(native["result"]["value"], 7);
        for engine in ["reference", "native-serial"] {
            let output = Command::new(env!("CARGO_BIN_EXE_invar"))
                .args(["run", path, "--engine", engine, "--c-library"])
                .arg(&library)
                .args(["--format", "json"])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{engine}: {}",
                String::from_utf8_lossy(&output.stdout)
            );
            let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(document["result"]["value"], 7, "{engine}");
        }
        let executable = directory.join("built-program");
        let build = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["build", path, "--target", "native", "--output"])
            .arg(&executable)
            .arg("--c-library")
            .arg(&library)
            .env("INVAR_L104_CONSTRUCTOR_MARKER", &marker)
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
        assert!(!marker.exists(), "native build loaded the C library");
        let built = Command::new(&executable)
            .arg("10000")
            .env("INVAR_L104_CONSTRUCTOR_MARKER", &marker)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "{}",
            String::from_utf8_lossy(&built.stdout)
        );
        assert!(
            marker.exists(),
            "native execution did not load the selected library"
        );
        let document: serde_json::Value = serde_json::from_slice(&built.stdout).unwrap();
        assert_eq!(document["result"]["value"], 7);
        std::fs::remove_dir_all(directory).unwrap();
    }
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["check", path, "--format", "json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn issue_l1_04_bytes_strings_cabi_foreign_buffers_and_failures_agree() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/l1_04/foreign-buffers.inv"
    );
    let source = std::fs::read(path).unwrap();
    let checked = check::check(front::parse(&source).unwrap(), true).unwrap();
    let reference = eval::run(&checked, 10_000).unwrap();
    assert_eq!(reference.value, eval::Value::I32(7));
    let lowered = mir::lower(&checked).unwrap();
    assert_eq!(mir_eval::run(&lowered, 10_000).unwrap(), reference);
    #[cfg(feature = "native")]
    {
        let (directory, library) = compile_foreign_fixture();
        let (status, native) =
            invar::native::run_program_with_library(&lowered, 10_000, Some(&library)).unwrap();
        assert_eq!(status, 0, "{native}");
        assert_eq!(native["result"]["value"], 7);
        let cleanup = serde_json::json!(
            reference
                .cleanup_trace
                .iter()
                .map(|event| serde_json::json!({"name": event.name, "length": event.length}))
                .collect::<Vec<_>>()
        );
        assert_eq!(native["cleanup_trace"], cleanup);
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn issue_l1_04_bytes_strings_cabi_detectable_foreign_failures_are_typed() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/l1_04/foreign-failures.inv"
    );
    let source = std::fs::read(path).unwrap();
    let checked = check::check(front::parse(&source).unwrap(), true).unwrap();
    let reference = eval::run(&checked, 10_000).unwrap();
    assert_eq!(reference.value, eval::Value::I32(7));
    let lowered = mir::lower(&checked).unwrap();
    assert_eq!(mir_eval::run(&lowered, 10_000).unwrap(), reference);
    #[cfg(feature = "native")]
    {
        let (directory, library) = compile_foreign_fixture();
        let (status, native) =
            invar::native::run_program_with_library(&lowered, 10_000, Some(&library)).unwrap();
        assert_eq!(status, 0, "{native}");
        assert_eq!(native["result"]["value"], 7);
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn issue_l1_04_bytes_strings_cabi_unsupported_signatures_reject_publicly() {
    for (name, code) in [
        ("foreign-cstring", "TYPE_MISMATCH"),
        ("foreign-borrowed-return", "UNSUPPORTED_FEATURE"),
        ("foreign-callback", "TYPE_MISMATCH"),
        ("foreign-mut-buffer", "TYPE_MISMATCH"),
        ("foreign-use-after-take", "OWN_USE_AFTER_MOVE"),
        ("foreign-missing-oracle", "NAME_UNKNOWN"),
        ("foreign-oracle-mismatch", "TYPE_MISMATCH"),
        ("foreign-too-many-args", "RESOURCE_LIMIT"),
    ] {
        let path = format!(
            "{}/tests/fixtures/l1_04/{name}.inv",
            env!("CARGO_MANIFEST_DIR")
        );
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["check", &path, "--format", "json"])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{name}");
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            document["diagnostics"][0]["code"], code,
            "{name}: {document}"
        );
    }
}

/// v0.2 f64math: a foreign call that leaves the process in the upward rounding
/// mode does not change invar-math-cr-v1 results (CORE-MATH honours the ambient
/// mode, so every call holds round-to-nearest).
#[test]
fn v02_math_rounds_to_nearest_after_a_foreign_call_changes_the_mode() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/l1_04/foreign-round-upward.inv"
    );
    let source = std::fs::read(path).unwrap();
    let checked = check::check(front::parse(&source).unwrap(), true).unwrap();
    assert_eq!(
        eval::run(&checked, 10_000).unwrap().value,
        eval::Value::I32(7)
    );
    #[cfg(feature = "native")]
    {
        let (directory, library) = compile_foreign_fixture();
        for engine in ["reference", "native-serial"] {
            let output = Command::new(env!("CARGO_BIN_EXE_invar"))
                .args(["run", path, "--engine", engine, "--c-library"])
                .arg(&library)
                .args(["--format", "json"])
                .output()
                .unwrap();
            let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(document["result"]["value"], 7, "{engine}: {document}");
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
}

/// v0.2 f64math: an `f64` value argument crosses the C ABI as binary64 bits
/// (`INV_TAG_F64_V1`) and agrees with its Invar oracle on every engine.
#[test]
fn v02_f64_foreign_scalar_argument_matches_its_oracle() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/l1_04/foreign-f64.inv"
    );
    let source = std::fs::read(path).unwrap();
    let checked = check::check(front::parse(&source).unwrap(), true).unwrap();
    let reference = eval::run(&checked, 10_000).unwrap();
    assert_eq!(reference.value, eval::Value::I32(7));
    assert_eq!(
        mir_eval::run(&mir::lower(&checked).unwrap(), 10_000).unwrap(),
        reference
    );
    #[cfg(feature = "native")]
    {
        let (directory, library) = compile_foreign_fixture();
        for engine in ["reference", "native-serial"] {
            let output = Command::new(env!("CARGO_BIN_EXE_invar"))
                .args(["run", path, "--engine", engine, "--c-library"])
                .arg(&library)
                .args(["--format", "json"])
                .output()
                .unwrap();
            let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(document["result"]["value"], 7, "{engine}: {document}");
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
}
