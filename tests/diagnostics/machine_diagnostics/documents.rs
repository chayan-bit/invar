#[test]
fn machine_diagnostics_corpus_has_no_invariant_codes() {
    fn visit(path: &std::path::Path, files: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().is_some_and(|extension| extension == "inv") {
                files.push(path);
            }
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    visit(&root.join("examples"), &mut files);
    if root.join("tests/fixtures").is_dir() {
        visit(&root.join("tests/fixtures"), &mut files);
    }
    if let Some(exported) = std::env::var_os("INVAR_CORPUS_DIR") {
        visit(&PathBuf::from(exported), &mut files);
    }
    files.sort();
    assert!(!files.is_empty());
    let invariants = invariant_codes();
    let registered = registry_codes();
    let diagnostic_schema = schema("invar-diagnostics-v3.schema.json");
    let explain_schema = schema("explain-v5.schema.json");
    let diagnostic_validator = jsonschema::validator_for(&diagnostic_schema).unwrap();
    let explain_validator = jsonschema::validator_for(&explain_schema).unwrap();
    let mut reached = HashSet::new();
    for file in files {
        let path = file.to_str().unwrap();
        let (check_status, check_document) = cli(&["check", path, "--format", "json"]);
        let mut documents = vec![(vec!["check", path, "--format", "json"], check_document)];
        if check_status == 0 {
            documents.push((
                vec!["explain", path, "--format", "json"],
                cli(&["explain", path, "--format", "json"]).1,
            ));
            let mut run_args = vec!["run", path, "--engine", "reference", "--format", "json"];
            if invar::front::parse(&std::fs::read(path).unwrap())
                .is_ok_and(|program| !program.foreign_functions.is_empty())
            {
                // Reference executes the Invar oracle; the required explicit path is never loaded.
                run_args.extend(["--c-library", env!("CARGO_BIN_EXE_invar")]);
            }
            documents.push((run_args.clone(), cli(&run_args).1));
            #[cfg(feature = "native")]
            if file.file_name().and_then(|name| name.to_str()) == Some("native-trace-limit.inv") {
                documents.push((
                    vec!["run", path, "--engine", "native-serial", "--format", "json"],
                    cli(&["run", path, "--engine", "native-serial", "--format", "json"]).1,
                ));
            }
        }
        for (args, document) in documents {
            if args[0] == "explain" {
                let expected = match file.file_name().and_then(|name| name.to_str()) {
                    Some("record-limit.inv") => Some("ADMISSION_RECORD_LIMIT"),
                    Some("document-limit.inv") => Some("DIAGNOSTIC_DOCUMENT_LIMIT"),
                    _ => None,
                };
                if let Some(code) = expected {
                    assert_eq!(document["diagnostics"][0]["code"], code, "{args:?}");
                }
            }
            let validator = if document["schema"] == "invar-explain-v5" {
                &explain_validator
            } else {
                &diagnostic_validator
            };
            assert!(validator.is_valid(&document), "{args:?}: {document}");
            assert!(
                invariant_fixes_absent(&document, &invariants),
                "{args:?}: {document}"
            );
            let mut emitted = HashSet::new();
            document_codes(&document, &registered, &mut emitted);
            for code in &emitted {
                assert!(!invariants.contains(code), "{args:?}: {code}: {document}");
            }
            // The region-count cap is retired (#57): the source that tripped it is kept as a
            // historical negative and must never raise the retired code again.
            if file.file_name().and_then(|name| name.to_str()) == Some("region-limit.inv") {
                assert!(
                    !emitted.contains("ADMISSION_REGION_LIMIT"),
                    "{args:?}: {document}"
                );
            }
            reached.extend(emitted);
        }
    }
    for (name, source, command) in [
        ("unknown", b"fn main() -> i32 { return missing; }".as_slice(), "check"),
        ("duplicate", b"fn main() -> i32 { return 1; } fn main() -> i32 { return 2; }".as_slice(), "check"),
        ("signature", b"fn main() -> Bool { return true; }".as_slice(), "check"),
        ("literal", b"fn main() -> i32 { return 12oops; }".as_slice(), "check"),
        ("token", b"fn main() -> i32 { return $; }".as_slice(), "check"),
        ("parse", b"fn main() -> i32 {".as_slice(), "check"),
        ("return", b"fn main() -> i32 {}".as_slice(), "check"),
        ("type", b"fn main() -> i32 { return 1u32 + 2i32; }".as_slice(), "check"),
        ("moved", b"fn main() -> i32 { let xs: Array<i32> = [1]; let ys: Array<i32> = move xs; return xs[0u32]; }".as_slice(), "check"),
        ("move-required", b"fn main() -> i32 { let xs: Array<i32> = [1]; let ys: Array<i32> = xs; return 0; }".as_slice(), "check"),
        ("borrow-conflict", b"fn f(a: mut Array<i32>, b: read Array<i32>) -> Unit { return; } fn main() -> i32 { var xs: Array<i32> = [1]; f(xs, xs); return 0; }".as_slice(), "check"),
        ("return-borrow", b"fn f(a: read Array<i32>) -> Array<i32> { return a; } fn main() -> i32 { return 0; }".as_slice(), "check"),
        ("mut-required", b"fn f(a: mut Array<i32>) -> Unit { return; } fn main() -> i32 { let xs: Array<i32> = [1]; f(xs); return 0; }".as_slice(), "check"),
        ("borrow-source", b"fn f(a: read Array<i32>) -> Unit { return; } fn main() -> i32 { f([1]); return 0; }".as_slice(), "check"),
        ("unsupported", b"fn main() -> i32 { filled_i32(1u32, 0); return 0; }".as_slice(), "check"),
        ("utf8", b"\xff".as_slice(), "check"),
        ("overflow", b"fn main() -> i32 { return 2147483647i32 + 1i32; }".as_slice(), "run"),
        ("division", b"fn main() -> i32 { return 1 / 0; }".as_slice(), "run"),
        ("index", b"fn main() -> i32 { let xs: Array<i32> = [1]; return xs[1u32]; }".as_slice(), "run"),
        ("array-limit", b"fn main() -> i32 { let xs: Array<i32> = filled_i32(16777217u32, 0); return 0; }".as_slice(), "run"),
        ("call-depth", b"fn main() -> i32 { return f(); } fn f() -> i32 { return f(); }".as_slice(), "run"),
        ("entrypoint", b"fn library() -> i32 { return 7; }".as_slice(), "run"),
    ] {
        let path = fixture(name, source);
        let path_str = path.to_str().unwrap();
        let args = if command == "run" {
            vec![command, path_str, "--engine", "reference", "--format", "json"]
        } else {
            vec![command, path_str, "--format", "json"]
        };
        let (_, document) = cli(&args);
        document_codes(&document, &registered, &mut reached);
        std::fs::remove_file(path).unwrap();
    }
    let unreadable = std::env::temp_dir().join(format!(
        "invar-ai-01-unreadable-{}-{:?}.inv",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir(&unreadable).unwrap();
    let inner_program = unreadable.join("program.inv");
    std::fs::write(&inner_program, b"fn main() -> i32 { return 0; }").unwrap();
    let (_, read_failure) = cli(&["check", unreadable.to_str().unwrap(), "--format", "json"]);
    assert_eq!(read_failure["diagnostics"][0]["code"], "FILE_READ");
    document_codes(&read_failure, &registered, &mut reached);
    std::fs::remove_file(inner_program).unwrap();
    std::fs::remove_dir(unreadable).unwrap();
    let valid = fixture("valid", b"fn main() -> i32 { return 0; }");
    let path = valid.to_str().unwrap();
    for engine in ["cuda", "metal", "native-serial"] {
        document_codes(
            &cli(&["run", path, "--engine", engine, "--format", "json"]).1,
            &registered,
            &mut reached,
        );
    }
    document_codes(
        &cli(&[
            "run",
            path,
            "--engine",
            "reference",
            "--fuel",
            "1",
            "--format",
            "json",
        ])
        .1,
        &registered,
        &mut reached,
    );
    let (_, cpu_fallback) = cli(&[
        "run",
        path,
        "--engine",
        "cpu-parallel",
        "--workers",
        "1",
        "--format",
        "json",
    ]);
    assert_eq!(
        cpu_fallback["diagnostics"][0]["code"],
        "NO_ADMITTED_CANDIDATE"
    );
    assert_eq!(cpu_fallback["diagnostics"][0]["severity"], "note");
    document_codes(&cpu_fallback, &registered, &mut reached);
    // `--debug` runs only on native-serial (DX-20).
    let (status, debug_refusal) = cli(&[
        "run",
        path,
        "--debug",
        "--engine",
        "cpu-parallel",
        "--format",
        "json",
    ]);
    assert_eq!(status, 2);
    assert!(debug_refusal.to_string().contains("DEBUG_UNSUPPORTED"));
    document_codes(&debug_refusal, &registered, &mut reached);
    std::fs::remove_file(valid).unwrap();
    let loop_source = "fn main() -> i32 { let xs: Array<u32> = [1u32, 2u32, 3u32]; var ys: Array<u32> = filled_u32(len(xs), 0u32); for i in 0u32 .. len(xs) { ys[0u32] = wrapping_add(xs[i], 7u32); } return 0; }";
    let affine = "fn main() -> i32 { let aa: Array<u32> = [2u32, 3u32, 5u32]; let bb: Array<u32> = [7u32, 11u32, 13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa), 0u32); for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i], state), bb[i]); out[i] = state; } return 0; }";
    let affine_shape = affine.replace(
        "wrapping_mul(aa[i], state)",
        "wrapping_mul(wrapping_add(state, 1u32), state)",
    );
    let affine_coefficient = affine.replace("bb[i]", "bb[0u32]");
    let unreachable = "fn main() -> i32 { let xs: Array<u32> = [1u32]; var ys: Array<u32> = filled_u32(len(xs), 0u32); if false { for i in 0u32 .. len(xs) { ys[i] = wrapping_add(xs[i], 1u32); } } return 0; }";
    let nested = format!(
        "fn main() -> i32 {{ return {}1{}; }}",
        "(".repeat(130),
        ")".repeat(130)
    );
    let oversized = vec![b' '; 1_048_577];
    let too_many_functions = format!(
        "{}fn main() -> i32 {{ return 0; }}",
        (0..129)
            .map(|n| format!("fn f{n}() -> i32 {{ return 0; }}"))
            .collect::<String>()
    );
    for (name, source, args) in [
        (
            "footprint",
            loop_source.as_bytes(),
            vec!["explain", "--format", "json"],
        ),
        (
            "affine-shape",
            affine_shape.as_bytes(),
            vec!["explain", "--format", "json"],
        ),
        (
            "affine-coefficient",
            affine_coefficient.as_bytes(),
            vec!["explain", "--format", "json"],
        ),
        (
            "unreached",
            unreachable.as_bytes(),
            vec![
                "run",
                "--engine",
                "cpu-parallel",
                "--workers",
                "1",
                "--format",
                "json",
            ],
        ),
        (
            "nesting",
            nested.as_bytes(),
            vec!["check", "--format", "json"],
        ),
        (
            "size",
            oversized.as_slice(),
            vec!["check", "--format", "json"],
        ),
        (
            "resource",
            too_many_functions.as_bytes(),
            vec!["explain", "--format", "json"],
        ),
    ] {
        let path = fixture(name, source);
        let mut command = vec![args[0], path.to_str().unwrap()];
        command.extend_from_slice(&args[1..]);
        let (_, document) = cli(&command);
        if name == "unreached" {
            assert_eq!(
                document["diagnostics"][0]["code"],
                "UNREACHED_OR_GUARD_FALSE"
            );
            assert_eq!(document["diagnostics"][0]["severity"], "note");
        }
        let validator = if document["schema"] == "invar-explain-v5" {
            &explain_validator
        } else {
            &diagnostic_validator
        };
        assert!(validator.is_valid(&document), "{command:?}: {document}");
        document_codes(&document, &registered, &mut reached);
        std::fs::remove_file(path).unwrap();
    }
    let extension = std::env::temp_dir().join(format!(
        "invar-ai-01-extension-coverage-{}-{:?}.txt",
        std::process::id(),
        std::thread::current().id()
    ));
    let extension_program = fixture("extension-source", b"fn main() -> i32 { return 0; }");
    std::fs::hard_link(&extension_program, &extension).unwrap();
    document_codes(
        &cli(&["check", extension.to_str().unwrap(), "--format", "json"]).1,
        &registered,
        &mut reached,
    );
    std::fs::remove_file(extension).unwrap();
    std::fs::remove_file(extension_program).unwrap();
    for entry in std::fs::read_dir(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s1_04/negative"),
    )
    .unwrap()
    {
        let path = entry.unwrap().path();
        let output = std::env::temp_dir().join(format!(
            "invar-ai-01-ros2-negative-{}-{}",
            std::process::id(),
            path.file_stem().unwrap().to_string_lossy()
        ));
        let (status, document) = cli(&[
            "system",
            "gen",
            "ros2",
            path.to_str().unwrap(),
            "--out",
            output.to_str().unwrap(),
        ]);
        assert_eq!(status, 1, "{}", path.display());
        assert!(!output.exists(), "{}", path.display());
        document_codes(&document, &registered, &mut reached);
    }
    let mut required = source_codes();
    // `invar-qualify` lives in the `invar-tools` package; tools/tests/bench_tooling/qualify_cli.rs reaches these.
    required.remove("M5_USAGE");
    required.remove("M5_LIBRARY_PATH");
    #[cfg(not(feature = "native"))]
    required.remove("NATIVE_TRACE_LIMIT");
    #[cfg(feature = "native")]
    required.remove("NATIVE_UNAVAILABLE");
    #[cfg(feature = "cuda")]
    required.remove("CUDA_UNAVAILABLE");
    #[cfg(all(feature = "metal", target_os = "macos"))]
    required.remove("METAL_UNAVAILABLE");
    let missing_source = required.difference(&reached).cloned().collect::<Vec<_>>();
    assert!(
        missing_source.is_empty(),
        "source codes without CLI fixtures: {missing_source:?}"
    );
}

#[test]
fn machine_diagnostics_cli_documents_validate() {
    let diagnostic_schema = schema("invar-diagnostics-v3.schema.json");
    let explain_schema = schema("explain-v5.schema.json");
    let device_schema = schema("invar-explain-device-v5.schema.json");
    let codes = registry_codes();
    let valid = fixture("valid", b"fn main() -> i32 { return 42; }");
    let invalid = fixture("invalid", b"fn main() -> i32 { return unknown; }");
    let trap = fixture("trap", b"fn main() -> i32 { return 1 / 0; }");
    let valid_path = valid.to_str().unwrap();
    let invalid_path = invalid.to_str().unwrap();
    let trap_path = trap.to_str().unwrap();
    for (args, exit, is_explain) in [
        (vec!["check", valid_path, "--format", "json"], 0, false),
        (
            vec![
                "run",
                valid_path,
                "--engine",
                "reference",
                "--format",
                "json",
            ],
            0,
            false,
        ),
        (vec!["explain", valid_path, "--format", "json"], 0, true),
        (
            vec![
                "explain",
                "examples/sum.inv",
                "--engine",
                "auto",
                "--format",
                "json",
            ],
            0,
            true,
        ),
        (
            vec![
                "explain",
                "examples/fsum.inv",
                "--engine",
                "auto",
                "--format",
                "json",
            ],
            0,
            true,
        ),
        (vec!["check", invalid_path, "--format", "json"], 1, false),
        (
            vec![
                "run",
                trap_path,
                "--engine",
                "reference",
                "--format",
                "json",
            ],
            3,
            false,
        ),
        (
            vec![
                "run",
                valid_path,
                "--engine",
                "reference",
                "--fuel",
                "1",
                "--format",
                "json",
            ],
            4,
            false,
        ),
    ] {
        let (status, document) = cli(&args);
        assert_eq!(status, exit, "{args:?}: {document}");
        if exit == 4 {
            assert_eq!(document["diagnostics"][0]["code"], "EVALUATION_LIMIT");
        }
        let validator = jsonschema::validator_for(if is_explain {
            assert_eq!(document["schema"], "invar-explain-v5", "{args:?}");
            &explain_schema
        } else {
            &diagnostic_schema
        })
        .unwrap();
        assert!(
            validator.is_valid(&document),
            "{args:?}: {document}: {:?}",
            validator
                .iter_errors(&document)
                .map(|error| error.to_string())
                .collect::<Vec<_>>()
        );
        if let Some(device) = document.get("device_execution") {
            assert!(jsonschema::is_valid(&device_schema, device), "{device}");
            assert_eq!(document["device_schema"], "invar-explain-device-v5");
        }
        if args[1] == "examples/fsum.inv" {
            let reductions = document["device_execution"]["choices"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|choice| {
                    choice["kind"]
                        .as_str()
                        .is_some_and(|k| k.starts_with("fsum."))
                })
                .collect::<Vec<_>>();
            assert_eq!(reductions.len(), 2, "{document}");
            // Ten terms never repay a worker start-up, nor a device launch:
            // the measured cost models keep them on one CPU worker. A Metal
            // build without a compiled-ahead library has no device to refuse.
            let reason = if cfg!(feature = "metal") {
                "device-unavailable"
            } else {
                "measured-cpu-cost"
            };
            assert!(
                reductions
                    .iter()
                    .all(|c| c["backend"] == "cpu-serial" && c["reason"] == reason),
                "{document}"
            );
        }
        if is_explain && args[1] == "examples/sum.inv" {
            assert!(!document["regions"].as_array().unwrap().is_empty());
            let mut changed = document.clone();
            changed["regions"][0]["new_field_without_version_bump"] = json!(1);
            assert!(!validator.is_valid(&changed));
        }
        assert!(codes_registered(&document, &codes), "{document}");
    }
    for path in [valid, invalid, trap] {
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn machine_diagnostics_source_error_fixtures_validate() {
    let schema = schema("invar-diagnostics-v3.schema.json");
    let codes = registry_codes();
    for (name, source, expected) in [
        (
            "unknown",
            b"fn main() -> i32 { return missing; }".as_slice(),
            "NAME_UNKNOWN",
        ),
        (
            "duplicate",
            b"fn main() -> i32 { return 1; } fn main() -> i32 { return 2; }".as_slice(),
            "NAME_DUPLICATE",
        ),
        (
            "signature",
            b"fn main() -> Bool { return true; }".as_slice(),
            "ENTRYPOINT_SIGNATURE",
        ),
        (
            "literal",
            b"fn main() -> i32 { return 12oops; }".as_slice(),
            "LEX_INVALID_LITERAL",
        ),
        (
            "token",
            b"fn main() -> i32 { return $; }".as_slice(),
            "LEX_INVALID_TOKEN",
        ),
        (
            "parse",
            b"fn main() -> i32 {".as_slice(),
            "PARSE_EXPECTED_TOKEN",
        ),
        (
            "return",
            b"fn main() -> i32 {}".as_slice(),
            "RETURN_MISSING",
        ),
        (
            "type",
            b"fn main() -> i32 { return 1u32 + 2i32; }".as_slice(),
            "TYPE_MISMATCH",
        ),
        (
            "moved",
            b"fn main() -> i32 { let xs: Array<i32> = [1]; let ys: Array<i32> = move xs; return xs[0u32]; }".as_slice(),
            "OWN_USE_AFTER_MOVE",
        ),
        (
            "move-required",
            b"fn main() -> i32 { let xs: Array<i32> = [1]; let ys: Array<i32> = xs; return 0; }".as_slice(),
            "OWN_MOVE_REQUIRED",
        ),
        (
            "borrow-conflict",
            b"fn f(a: mut Array<i32>, b: read Array<i32>) -> Unit { return; } fn main() -> i32 { var xs: Array<i32> = [1]; f(xs, xs); return 0; }".as_slice(),
            "OWN_BORROW_CONFLICT",
        ),
        (
            "return-borrow",
            b"fn f(a: read Array<i32>) -> Array<i32> { return a; } fn main() -> i32 { return 0; }".as_slice(),
            "OWN_RETURN_BORROW",
        ),
        (
            "mut-required",
            b"fn f(a: mut Array<i32>) -> Unit { return; } fn main() -> i32 { let xs: Array<i32> = [1]; f(xs); return 0; }".as_slice(),
            "OWN_MUT_REQUIRED",
        ),
        (
            "borrow-source",
            b"fn f(a: read Array<i32>) -> Unit { return; } fn main() -> i32 { f([1]); return 0; }".as_slice(),
            "OWN_BORROW_SOURCE",
        ),
        (
            "unsupported",
            b"fn main() -> i32 { filled_i32(1u32, 0); return 0; }".as_slice(),
            "UNSUPPORTED_FEATURE",
        ),
        ("utf8", b"\xff".as_slice(), "INVALID_UTF8"),
    ] {
        let path = fixture(name, source);
        let (status, document) = cli(&[
            "run",
            path.to_str().unwrap(),
            "--engine",
            "reference",
            "--format",
            "json",
        ]);
        assert_eq!(
            document["diagnostics"][0]["code"], expected,
            "{name}: {document}"
        );
        assert!(matches!(status, 1 | 2), "{name}: {status}");
        assert!(
            jsonschema::is_valid(&schema, &document),
            "{name}: {document}"
        );
        assert!(codes_registered(&document, &codes), "{name}: {document}");
        std::fs::remove_file(path).unwrap();
    }
    let missing =
        std::env::temp_dir().join(format!("invar-ai-01-missing-{}.inv", std::process::id()));
    let _ = std::fs::remove_file(&missing);
    let (status, document) = cli(&["check", missing.to_str().unwrap(), "--format", "json"]);
    assert_eq!(status, 2);
    assert_eq!(document["diagnostics"][0]["code"], "FILE_READ");
    assert!(jsonschema::is_valid(&schema, &document));
    assert!(codes_registered(&document, &codes));

    let wrong_extension =
        std::env::temp_dir().join(format!("invar-ai-01-extension-{}.txt", std::process::id()));
    std::fs::write(&wrong_extension, b"fn main() -> i32 { return 1; }").unwrap();
    let (status, document) = cli(&[
        "check",
        wrong_extension.to_str().unwrap(),
        "--format",
        "json",
    ]);
    assert_eq!(status, 2);
    assert_eq!(document["diagnostics"][0]["code"], "FILE_EXTENSION");
    assert!(jsonschema::is_valid(&schema, &document));
    assert!(codes_registered(&document, &codes));
    std::fs::remove_file(wrong_extension).unwrap();

    let nested = format!(
        "fn main() -> i32 {{ return {}1{}; }}",
        "(".repeat(130),
        ")".repeat(130)
    );
    for (name, source, expected) in [
        ("nesting", nested.into_bytes(), "SOURCE_NESTING_LIMIT"),
        ("size", vec![b' '; 1_048_577], "SOURCE_SIZE_LIMIT"),
    ] {
        let path = fixture(name, &source);
        let (status, document) = cli(&["check", path.to_str().unwrap(), "--format", "json"]);
        assert_eq!(status, 2, "{name}: {document}");
        assert_eq!(document["diagnostics"][0]["code"], expected);
        assert!(jsonschema::is_valid(&schema, &document));
        assert!(codes_registered(&document, &codes));
        std::fs::remove_file(path).unwrap();
    }

    let path = fixture("unavailable", b"fn main() -> i32 { return 1; }");
    let unavailable: &[(&str, &str)] = &[
        #[cfg(not(feature = "cuda"))]
        ("cuda", "CUDA_UNAVAILABLE"),
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        ("metal", "METAL_UNAVAILABLE"),
        #[cfg(not(feature = "native"))]
        ("native-serial", "NATIVE_UNAVAILABLE"),
    ];
    for &(engine, expected) in unavailable {
        let (status, document) = cli(&[
            "run",
            path.to_str().unwrap(),
            "--engine",
            engine,
            "--format",
            "json",
        ]);
        assert_eq!(status, 2, "{engine}: {document}");
        assert_eq!(document["diagnostics"][0]["code"], expected);
        assert!(jsonschema::is_valid(&schema, &document));
        assert!(codes_registered(&document, &codes));
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn machine_diagnostics_runtime_fixtures_validate() {
    let schema = schema("invar-diagnostics-v3.schema.json");
    let codes = registry_codes();
    for (name, source, exit, expected) in [
        (
            "overflow",
            "fn main() -> i32 { return 2147483647i32 + 1i32; }",
            3,
            "INTEGER_OVERFLOW",
        ),
        (
            "division",
            "fn main() -> i32 { return 1 / 0; }",
            3,
            "DIVISION_BY_ZERO",
        ),
        (
            "index",
            "fn main() -> i32 { let xs: Array<i32> = [1]; return xs[1u32]; }",
            3,
            "INDEX_OUT_OF_BOUNDS",
        ),
        (
            "array-limit",
            "fn main() -> i32 { let xs: Array<i32> = filled_i32(16777217u32, 0); return 0; }",
            2,
            "ARRAY_SIZE_LIMIT",
        ),
        (
            "call-depth",
            "fn main() -> i32 { return f(); } fn f() -> i32 { return f(); }",
            2,
            "CALL_DEPTH_LIMIT",
        ),
    ] {
        let path = fixture(name, source.as_bytes());
        let (status, document) = cli(&[
            "run",
            path.to_str().unwrap(),
            "--engine",
            "reference",
            "--format",
            "json",
        ]);
        assert_eq!(status, exit, "{name}: {document}");
        assert_eq!(document["diagnostics"][0]["code"], expected);
        assert!(
            jsonschema::is_valid(&schema, &document),
            "{name}: {document}"
        );
        assert!(codes_registered(&document, &codes), "{name}: {document}");
        std::fs::remove_file(path).unwrap();
    }
}
