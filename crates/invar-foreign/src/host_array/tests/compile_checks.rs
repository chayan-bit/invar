//! Compile checks with pinned error codes and the crate-boundary tripwire.
use super::*;

// ---------------------------------------------------------------------------
// Compile checks with pinned error codes. Stable rustdoc does not enforce the
// `compile_fail,E....` codes of the doctests, so each case is compiled here by
// rustc (JSON diagnostics) against this module's own source, which needs only
// std. Every negative case must fail with exactly its expected code(s); every
// positive control must compile with no error.

/// The exact release selected by the repository's `rust-toolchain.toml`
/// `channel` (not the crate's `rust-version`, which is only the MSRV).
fn pinned_release() -> &'static str {
    const TOOLCHAIN: &str = include_str!("../../../../../rust-toolchain.toml");
    let channel = TOOLCHAIN
        .lines()
        .find_map(|line| line.trim().strip_prefix("channel = \"")?.strip_suffix('"'))
        .expect("rust-toolchain.toml declares a quoted channel");
    assert!(
        !channel.is_empty() && channel.chars().all(|c| c.is_ascii_digit() || c == '.'),
        "rust-toolchain.toml channel {channel:?} is not an exact release"
    );
    channel
}

/// The compiler the harness invokes: `RUSTC` if set, else the `rustc` next to
/// the `CARGO` running these tests, else `rustc` on `PATH`. Error codes are
/// pinned for one release, so its `release:` must equal the
/// `rust-toolchain.toml` channel or the harness fails instead of comparing
/// codes. This is a **release pin**, not exact compiler identity: the full
/// `rustc -vV` output (commit hash, host, LLVM version) is printed to the
/// test output as a receipt but not checked.
fn harness_rustc() -> &'static std::path::Path {
    static RUSTC: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    RUSTC.get_or_init(|| {
        let rustc: std::path::PathBuf = if let Some(rustc) = std::env::var_os("RUSTC") {
            rustc.into()
        } else {
            std::env::var_os("CARGO")
                .map(|cargo| {
                    Path::new(&cargo)
                        .with_file_name(format!("rustc{}", std::env::consts::EXE_SUFFIX))
                })
                .filter(|sibling| sibling.is_file())
                .unwrap_or_else(|| "rustc".into())
        };
        let output = Command::new(&rustc)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .arg("-vV")
            .output()
            .unwrap();
        let identity = String::from_utf8(output.stdout).unwrap();
        eprintln!("host-array compile harness: {rustc:?} -vV\n{identity}");
        let release = identity
            .lines()
            .find_map(|line| line.strip_prefix("release: "))
            .unwrap_or_default();
        assert_eq!(
            release,
            pinned_release(),
            "harness compiler {rustc:?} does not match the rust-toolchain.toml release pin:\n{identity}"
        );
        rustc
    })
}

/// Error codes of the top-level `error` diagnostics in rustc JSON output.
/// Top-level fields precede children, and string contents escape quotes.
fn error_codes(name: &str, output: &std::process::Output) -> Vec<String> {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut codes = Vec::new();
    for line in stderr.lines() {
        let Some(level) = line.find("\"level\":\"") else {
            continue;
        };
        if !line[level..].starts_with("\"level\":\"error\"") {
            continue;
        }
        if let Some(at) = line.find("\"code\":{\"code\":\"") {
            let start = at + "\"code\":{\"code\":\"".len();
            codes.push(line[start..start + 5].to_owned());
        } else if !line.contains("aborting due to") {
            codes.push("UNCODED".to_owned());
        }
    }
    assert_eq!(
        output.status.success(),
        codes.is_empty(),
        "{name}: status/diagnostic mismatch\n{stderr}"
    );
    codes
}

fn compile_errors(name: &str, body: &str) -> Vec<String> {
    let directory = std::env::temp_dir().join(format!(
        "invar-host-array-compile-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&directory).unwrap();
    let module = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/host_array.rs");
    let source = format!(
        "#[path = {module:?}]\n#[allow(dead_code, unused)]\nmod host_array;\n#[allow(unused_imports)]\nuse host_array::*;\n#[allow(unused)]\nfn main() {{\n{body}\n}}\n"
    );
    let file = directory.join("probe.rs");
    std::fs::write(&file, source).unwrap();
    let output = Command::new(harness_rustc())
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "--edition=2024",
            "--crate-type=bin",
            "--crate-name=host_array_probe",
            // The probe includes the module alone, without the real loader.
            "--cfg=host_array_isolated",
            "--emit=metadata",
            "--error-format=json",
            "-o",
        ])
        .arg(directory.join("probe.rmeta"))
        .arg(&file)
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&directory);
    error_codes(name, &output)
}

/// Build this module's source, alone, as an **isolated stand-in** library
/// crate (`host_array_lib`, std only) into `directory`. This is not the real
/// `invar_foreign` crate: it checks the module's own crate-boundary
/// visibility, not what the real crate re-exports (see
/// `real_crate_exposes_host_array_only_as_its_own_module`).
fn build_isolated_library(directory: &Path) -> std::path::PathBuf {
    let module = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/host_array.rs");
    let library_source = directory.join("lib.rs");
    std::fs::write(
        &library_source,
        format!("#[path = {module:?}]\npub mod host_array;\n"),
    )
    .unwrap();
    let library = directory.join("libhost_array_lib.rlib");
    let built = Command::new(harness_rustc())
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "--edition=2024",
            "--crate-type=rlib",
            "--crate-name=host_array_lib",
            "--cfg=host_array_isolated",
            "--error-format=json",
            "-o",
        ])
        .arg(&library)
        .arg(&library_source)
        .output()
        .unwrap();
    assert_eq!(
        error_codes("isolated library", &built),
        Vec::<String>::new()
    );
    library
}

/// As `compile_errors`, but the probe is a separate crate linked with
/// `--extern` to the isolated stand-in library, so `pub(crate)` items of the
/// module are out of reach (the in-crate probe above can see them).
fn compile_errors_isolated(name: &str, body: &str, library: &Path) -> Vec<String> {
    let directory = library.parent().unwrap().join(format!("probe-{name}"));
    std::fs::create_dir(&directory).unwrap();
    let file = directory.join("probe.rs");
    std::fs::write(
        &file,
        format!(
            "#[allow(unused_imports)]\nuse host_array_lib::host_array::*;\n#[allow(unused)]\nfn main() {{\n{body}\n}}\n"
        ),
    )
    .unwrap();
    let mut extern_arg = std::ffi::OsString::from("host_array_lib=");
    extern_arg.push(library);
    let output = Command::new(harness_rustc())
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "--edition=2024",
            "--crate-type=bin",
            "--crate-name=host_array_isolated_probe",
            "--emit=metadata",
            "--error-format=json",
            "--extern",
        ])
        .arg(extern_arg)
        .arg("-o")
        .arg(directory.join("probe.rmeta"))
        .arg(&file)
        .output()
        .unwrap();
    error_codes(name, &output)
}

const PREPARE_ONE: &str = "let mut xs = vec![1i32, 2];\nlet mut call = HostArrayCall::prepare(vec![HostArray::MutI32(&mut xs)]).unwrap();";

#[test]
fn compile_checks_pin_error_codes_and_positive_controls() {
    let negative: [(&str, String, &[&str]); 11] = [
        (
            "literal_record",
            "let r = HostArrayRecordV1 { element: 1, mode: 1, data: std::ptr::null_mut(), len: 0 };".into(),
            &["E0451"],
        ),
        (
            "call_not_send",
            "fn send<T: Send>() {}\nsend::<HostArrayCall<'static>>();".into(),
            // Both the Rc<()> marker and the record's *mut u8 are !Send.
            &["E0277", "E0277"],
        ),
        (
            "records_not_send",
            "fn send<T: Send>() {}\nsend::<HostArrayRecords<'static>>();".into(),
            &["E0277"],
        ),
        (
            "to_vec_escape",
            // The exact stale-record escape demonstrated at ab2b54b.
            "let saved: Vec<HostArrayRecordV1>;\n{\nlet mut xs = vec![1i32, 2];\nlet mut call = HostArrayCall::prepare(vec![HostArray::MutI32(&mut xs)]).unwrap();\nsaved = call.records().to_vec();\ncall.commit();\n}\nassert_eq!(saved.len(), 1);".into(),
            &["E0599"],
        ),
        (
            "copy_out_by_deref",
            format!("{PREPARE_ONE}\nlet saved: HostArrayRecordV1 = *call.records().get(0).unwrap();\ncall.commit();\nlet _ = saved;"),
            &["E0507"],
        ),
        (
            "clone_record",
            // `.clone()` on `&HostArrayRecordV1` would clone the reference;
            // name the record's own Clone impl, which must not exist.
            format!("{PREPARE_ONE}\nlet saved: HostArrayRecordV1 = <HostArrayRecordV1 as Clone>::clone(call.records().get(0).unwrap());\ncall.commit();\nlet _ = saved;"),
            &["E0277"],
        ),
        (
            "slice_from_iter",
            format!("{PREPARE_ONE}\nlet records = call.records();\nlet slice = records.iter().as_slice();\nlet _ = slice;"),
            &["E0599"],
        ),
        (
            "view_outlives_guard",
            "let saved;\n{\nlet mut xs = vec![1i32, 2];\nlet mut call = HostArrayCall::prepare(vec![HostArray::MutI32(&mut xs)]).unwrap();\nsaved = call.records();\n}\nassert_eq!(saved.len(), 1);".into(),
            &["E0597", "E0597"],
        ),
        (
            "commit_while_view_live",
            format!("{PREPARE_ONE}\nlet records = call.records();\ncall.commit();\nassert_eq!(records.len(), 1);"),
            &["E0505"],
        ),
        (
            "drop_while_view_live",
            format!("{PREPARE_ONE}\nlet records = call.records();\ndrop(call);\nassert_eq!(records.len(), 1);"),
            &["E0505"],
        ),
        (
            "write_slice_while_view_live",
            "let mut xs = [1i32];\nlet mut call = HostArrayCall::prepare(vec![HostArray::MutI32(&mut xs)]).unwrap();\nlet records = call.records();\nxs[0] = 2;\nassert_eq!(records.len(), 1);".into(),
            &["E0506"],
        ),
    ];
    for (name, body, expected) in &negative {
        let mut codes = compile_errors(name, body);
        codes.sort();
        let mut expected: Vec<String> = expected.iter().map(|c| (*c).to_owned()).collect();
        expected.sort();
        assert_eq!(codes, expected, "{name}");
    }
    let positive: [(&str, String); 3] = [
        (
            "use_then_commit_then_write",
            "let mut xs = [1i32];\nlet mut call = HostArrayCall::prepare(vec![HostArray::MutI32(&mut xs)]).unwrap();\nlet n = call.records().len();\ncall.commit();\nxs[0] = 2;\nassert_eq!(n, 1);".into(),
        ),
        (
            "observe_then_finish",
            format!("{PREPARE_ONE}\nlet records = call.records();\nlet seen: Vec<(u32, u32, usize, u64)> = records.iter().map(|r| (r.element(), r.mode(), r.address(), r.len())).collect();\nlet first = records.get(0).map(|r| r.is_empty());\nlet kept: Result<usize, ()> = call.finish(Ok(seen.len()));\nassert_eq!((kept, first), (Ok(1), Some(false)));"),
        ),
        (
            "rebuild_view_twice_then_restore",
            format!("{PREPARE_ONE}\nlet a = call.records().len();\nlet b = call.records().iter().count();\ncall.restore();\nassert_eq!(a, b);"),
        ),
    ];
    for (name, body) in &positive {
        assert_eq!(compile_errors(name, body), Vec::<String>::new(), "{name}");
    }
}

#[test]
fn isolated_module_crate_boundary_hides_private_record_access() {
    let directory = std::env::temp_dir().join(format!(
        "invar-host-array-isolated-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&directory).unwrap();
    let library = build_isolated_library(&directory);
    let negative: [(&str, String, &[&str]); 4] = [
        (
            "external_as_ptr",
            format!("{PREPARE_ONE}\nlet records = call.records();\nlet pointer = records.as_ptr();\nlet _ = pointer;"),
            // `pub(crate)` method: private outside the defining crate.
            &["E0624"],
        ),
        (
            "external_view_field",
            format!("{PREPARE_ONE}\nlet records = call.records();\nlet inner = &records.records;\nlet _ = inner;"),
            &["E0616"],
        ),
        (
            "external_literal_record",
            "let r = HostArrayRecordV1 { element: 1, mode: 1, data: std::ptr::null_mut(), len: 0 };".into(),
            &["E0451"],
        ),
        (
            "external_to_vec_escape",
            "let saved: Vec<HostArrayRecordV1>;\n{\nlet mut xs = vec![1i32, 2];\nlet mut call = HostArrayCall::prepare(vec![HostArray::MutI32(&mut xs)]).unwrap();\nsaved = call.records().to_vec();\ncall.commit();\n}\nassert_eq!(saved.len(), 1);".into(),
            &["E0599"],
        ),
    ];
    for (name, body, expected) in &negative {
        let mut codes = compile_errors_isolated(name, body, &library);
        codes.sort();
        let mut expected: Vec<String> = expected.iter().map(|c| (*c).to_owned()).collect();
        expected.sort();
        assert_eq!(codes, expected, "{name}");
    }
    // Positive control across the crate boundary: the public API is usable.
    assert_eq!(
        compile_errors_isolated(
            "external_use_then_commit_then_write",
            "let mut xs = [1i32];\nlet mut call = HostArrayCall::prepare(vec![HostArray::MutI32(&mut xs)]).unwrap();\nlet n = call.records().len();\ncall.commit();\nxs[0] = 2;\nassert_eq!(n, 1);",
            &library
        ),
        Vec::<String>::new()
    );
    let _ = std::fs::remove_dir_all(&directory);
}

/// Bounded static tripwire on the **real** crate's sources, not a proof: the
/// module must be reachable only as `invar_foreign::host_array`. `lib.rs`
/// declares it exactly once, no other source file names it (a re-export or a
/// public function handing out its records would have to), and no glob
/// re-export (`pub use ...::*`) exists that could expose it under another
/// path. Macro-generated paths are out of scope; the crate defines none.
#[test]
fn real_crate_exposes_host_array_only_as_its_own_module() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let lib = std::fs::read_to_string(source.join("lib.rs")).unwrap();
    let declarations: Vec<&str> = lib
        .lines()
        .filter(|line| line.contains("host_array"))
        .collect();
    assert_eq!(declarations, vec!["pub mod host_array;"]);
    let module = source.join("host_array.rs");
    let module_dir = source.join("host_array");
    let (mut naming, mut globs) = (Vec::new(), Vec::new());
    let mut pending = vec![source.clone()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            if path.extension().is_none_or(|extension| extension != "rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let inside = path == module || path.starts_with(&module_dir);
            if !inside && path != source.join("lib.rs") && text.contains("host_array") {
                naming.push(path.clone());
            }
            if !inside
                && text.lines().any(|line| {
                    let line = line.trim_start();
                    (line.starts_with("pub use") || line.starts_with("pub(crate) use"))
                        && line.contains("::*")
                })
            {
                globs.push(path);
            }
        }
    }
    assert!(
        naming.is_empty(),
        "other sources name host_array: {naming:?}"
    );
    assert!(
        globs.is_empty(),
        "glob re-exports could expose host_array: {globs:?}"
    );
}
