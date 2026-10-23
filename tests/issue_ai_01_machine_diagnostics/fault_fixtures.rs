#[test]
fn issue_ai_01_machine_diagnostics_fault_fixtures_validate() {
    let source = b"fn main() -> i32 { return 1; }";
    let checked = invar::check::check(invar::front::parse(source).unwrap(), true).unwrap();
    let invalid_fuel = invar::eval::observe(&checked, 0)
        .result
        .unwrap_err()
        .to_diagnostics_json("fixture.inv", source);
    assert_eq!(invalid_fuel["diagnostics"][0]["code"], "INVALID_FUEL");

    let valid_mir = invar::mir::lower(&checked).unwrap();
    let mut mir = valid_mir.clone();
    mir.schema_version += 1;
    let error = invar::mir::verify(&mir).unwrap_err();
    let invalid_mir = diagnostics::document(
        "tool_error",
        vec![diagnostics::diagnostic(
            error.code,
            &error.message,
            "fixture.inv",
            source,
            None,
        )],
    );
    assert_eq!(invalid_mir["diagnostics"][0]["code"], "MIR_INVALID");

    let invalid_workers = invar::schedule::run_source(source, &valid_mir, 100, 0).unwrap_err();
    assert_eq!(invalid_workers.code(), "CPU_WORKERS");
    let invalid_candidate = invar::schedule::run_source(source, &mir, 100, 1).unwrap_err();
    assert_eq!(invalid_candidate.code(), "CPU_CANDIDATE");

    let mut oversized_mir = invar::mir::lower(&checked).unwrap();
    let function = oversized_mir.functions[0].clone();
    oversized_mir
        .functions
        .resize(invar::mir::MAX_FUNCTIONS + 1, function);
    let error = invar::mir::verify(&oversized_mir).unwrap_err();
    let resource_limit = diagnostics::document(
        "tool_error",
        vec![diagnostics::diagnostic(
            error.code,
            &error.message,
            "fixture.inv",
            source,
            None,
        )],
    );
    assert_eq!(resource_limit["diagnostics"][0]["code"], "RESOURCE_LIMIT");

    let count_limit = diagnostics::document(
        "tool_error",
        (0..65)
            .map(|_| diagnostics::diagnostic("INVALID_FUEL", "fuel", "fixture.inv", source, None))
            .collect(),
    );
    assert_eq!(
        count_limit["diagnostics"][0]["code"],
        "DIAGNOSTIC_COUNT_LIMIT"
    );

    let schema = schema("invar-diagnostics-v3.schema.json");
    let codes = registry_codes();
    for document in [invalid_fuel, invalid_mir, resource_limit, count_limit] {
        assert!(jsonschema::is_valid(&schema, &document), "{document}");
        assert!(codes_registered(&document, &codes), "{document}");
        assert_eq!(document["diagnostics"][0]["span"], Value::Null);
    }
    for error in [invalid_workers, invalid_candidate] {
        let document = diagnostics::document(
            "tool_error",
            vec![diagnostics::diagnostic(
                error.code(),
                "scheduler rejected the input",
                "fixture.inv",
                source,
                None,
            )],
        );
        assert!(jsonschema::is_valid(&schema, &document), "{document}");
        assert!(codes_registered(&document, &codes), "{document}");
    }
}

#[test]
fn issue_ai_01_machine_diagnostics_admission_fault_fixtures_validate() {
    let source = b"fn main() -> i32 { return 1; }";
    let checked = invar::check::check(invar::front::parse(source).unwrap(), true).unwrap();
    let mir = invar::mir::lower(&checked).unwrap();
    let cases = [
        (
            invar::admission::analyze(&vec![b' '; 1_048_577], &mir).unwrap_err(),
            "ADMISSION_SOURCE_LIMIT",
        ),
        (
            invar::admission::analyze(&source[..1], &mir).unwrap_err(),
            "ADMISSION_SOURCE",
        ),
        (
            invar::admission::validate_serialized(&json!({"schema_version": 2}), source, &mir)
                .unwrap_err(),
            "ADMISSION_SCHEMA",
        ),
        (
            invar::admission::validate_serialized(&json!({"schema_version": 5}), source, &mir)
                .unwrap_err(),
            "ADMISSION_STALE",
        ),
        (
            invar::admission::validate_serialized(
                &json!({"schema_version": 5, "padding": "x".repeat(1_000_000)}),
                source,
                &mir,
            )
            .unwrap_err(),
            "ADMISSION_RECORD_LIMIT",
        ),
    ];
    let schema = schema("invar-diagnostics-v3.schema.json");
    let codes = registry_codes();
    for (error, expected) in cases {
        assert_eq!(error.code, expected);
        let document = diagnostics::document(
            "tool_error",
            vec![diagnostics::diagnostic(
                error.code,
                &error.message,
                "fixture.inv",
                source,
                None,
            )],
        );
        assert!(jsonschema::is_valid(&schema, &document), "{document}");
        assert!(codes_registered(&document, &codes), "{document}");
    }
}

#[test]
fn issue_ai_01_machine_diagnostics_checked_ir_fault_fixtures_validate() {
    use invar::front::{ExprKind, StmtKind};

    let simple = b"fn main() -> i32 { return 1; }";
    let checked =
        |source: &[u8]| invar::check::check(invar::front::parse(source).unwrap(), true).unwrap();
    let mut missing_function = checked(simple);
    missing_function.program.functions.clear();
    let mut missing_return = checked(simple);
    missing_return.program.functions[0].body.statements.clear();
    let mut missing_type = checked(simple);
    missing_type.expression_types.clear();
    missing_type.function_expression_types.clear();
    let mut invalid_literal = checked(simple);
    let StmtKind::Return(Some(expr)) =
        &mut invalid_literal.program.functions[0].body.statements[0].kind
    else {
        panic!("expected return expression");
    };
    let ExprKind::Integer(raw) = &mut expr.kind else {
        panic!("expected integer expression");
    };
    *raw = "invalid".into();

    let binding = b"fn main() -> i32 { let x: i32 = 1; return x; }";
    let mut missing_binding = checked(binding);
    let StmtKind::Binding { name, .. } =
        &mut missing_binding.program.functions[0].body.statements[0].kind
    else {
        panic!("expected binding");
    };
    *name = "renamed".into();

    let array =
        b"fn main() -> i32 { let xs: Array<i32> = [1]; let ys: Array<i32> = move xs; return 0; }";
    let mut array_copy = checked(array);
    let StmtKind::Binding { value, .. } =
        &mut array_copy.program.functions[0].body.statements[1].kind
    else {
        panic!("expected array binding");
    };
    let ExprKind::Move(name) = &value.kind else {
        panic!("expected move expression");
    };
    value.kind = ExprKind::Name(name.clone());

    let borrowed = b"fn f(a: read Array<i32>) -> Unit { return; } fn main() -> i32 { let xs: Array<i32> = [1]; f(xs); return 0; }";
    let mut invalid_borrow = checked(borrowed);
    let StmtKind::Expr(expr) = &mut invalid_borrow.program.functions[1].body.statements[1].kind
    else {
        panic!("expected call statement");
    };
    let ExprKind::Call { args, .. } = &mut expr.kind else {
        panic!("expected call expression");
    };
    args[0].kind = ExprKind::Bool(true);

    let cases = [
        (missing_function, simple.as_slice(), "INTERNAL_CALL"),
        (missing_return, simple.as_slice(), "INTERNAL_RETURN"),
        (missing_type, simple.as_slice(), "INTERNAL_TYPE"),
        (invalid_literal, simple.as_slice(), "INTERNAL_LITERAL"),
        (missing_binding, binding.as_slice(), "INTERNAL_BINDING"),
        (array_copy, array.as_slice(), "INTERNAL_ARRAY_COPY"),
        (invalid_borrow, borrowed.as_slice(), "INTERNAL_BORROW"),
    ];
    let schema = schema("invar-diagnostics-v3.schema.json");
    let codes = registry_codes();
    for (program, source, expected) in cases {
        let document = invar::eval::observe(&program, 100)
            .result
            .unwrap_err()
            .to_diagnostics_json("fixture.inv", source);
        assert_eq!(document["diagnostics"][0]["code"], expected);
        assert!(jsonschema::is_valid(&schema, &document), "{document}");
        assert!(codes_registered(&document, &codes), "{document}");
    }
}

#[cfg(feature = "native")]
#[test]
fn issue_ai_01_machine_diagnostics_native_documents_validate() {
    let schema = schema("invar-diagnostics-v3.schema.json");
    let codes = registry_codes();
    let path = fixture("native", b"fn main() -> i32 { return 12; }");
    for (source, exit) in [
        ("fn main() -> i32 { return 12; }", 0),
        ("fn main() -> i32 { return 1 / 0; }", 3),
    ] {
        std::fs::write(&path, source).unwrap();
        let (status, document) = cli(&[
            "run",
            path.to_str().unwrap(),
            "--engine",
            "native-serial",
            "--format",
            "json",
        ]);
        assert_eq!(status, exit, "{document}");
        assert!(jsonschema::is_valid(&schema, &document), "{document}");
        assert!(codes_registered(&document, &codes));
    }
    std::fs::remove_file(path).unwrap();
    let trace_fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ai_01/native-trace-limit.inv");
    let (status, document) = cli(&[
        "run",
        trace_fixture.to_str().unwrap(),
        "--engine",
        "native-serial",
        "--format",
        "json",
    ]);
    assert_eq!(status, 2, "{document}");
    assert_eq!(document["diagnostics"][0]["code"], "NATIVE_TRACE_LIMIT");
    assert!(jsonschema::is_valid(&schema, &document), "{document}");
}

#[test]
fn issue_ai_01_machine_diagnostics_target_unknown_fault() {
    // Inject a device no target description selects: the built-in target
    // files without `cuda-sm90.json`.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("invar-ai-01-targets-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(root.join("targets")).unwrap() {
        let path = entry.unwrap().path();
        if path.file_stem().is_some_and(|s| s != "cuda-sm90") {
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
    }
    let registry = invar_device::target::Registry::from_dir(&dir).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let hopper = invar_device::Capabilities::Cuda {
        major: 12,
        minor: 4,
        sm: 90,
    };
    let error = invar_device::contraction::select_target(&registry, hopper).unwrap_err();
    assert_eq!(error, invar_device::floating::Error::TargetUnknown);
    let document = diagnostics::document(
        "tool_error",
        vec![diagnostics::diagnostic(
            error.code().unwrap(),
            "no target description selects cuda-sm90",
            "fixture.inv",
            b"",
            None,
        )],
    );
    assert_eq!(document["diagnostics"][0]["code"], "TARGET_UNKNOWN");
    assert!(
        jsonschema::is_valid(&schema("invar-diagnostics-v3.schema.json"), &document),
        "{document}"
    );
    assert!(codes_registered(&document, &registry_codes()), "{document}");
}
