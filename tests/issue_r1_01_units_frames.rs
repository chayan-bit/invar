use std::process::Command;

#[test]
fn issue_r1_01_units_frames_quantity_erasure_is_byte_identical() {
    let typed = "fn main() -> i32 { let x: Quantity<i32,m> = quantity::<m>(1i32); let y: Quantity<i32,m> = quantity::<m>(1i32); if x == y { return 1; } return 0; }";
    let type_pad = " ".repeat("Quantity<i32,m>".len() - "i32".len());
    let call_pad = " ".repeat("quantity::<m>(".len());
    let plain = format!(
        "fn main() -> i32 {{ let x: i32{type_pad} = {call_pad}1i32 ; let y: i32{type_pad} = {call_pad}1i32 ; if x == y {{ return 1; }} return 0; }}"
    );
    assert_eq!(typed.len(), plain.len());
    let lower = |source: &str| {
        let checked =
            invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
        invar::mir::lower(&checked).unwrap()
    };
    let typed_mir = lower(typed);
    let plain_mir = lower(&plain);
    assert_eq!(typed_mir.functions[0].span, plain_mir.functions[0].span);
    assert_eq!(
        typed_mir.functions[0].declaration_order,
        plain_mir.functions[0].declaration_order
    );
    assert_eq!(
        format!("{:?}", typed_mir.functions[0].blocks),
        format!("{:?}", plain_mir.functions[0].blocks)
    );
    #[cfg(feature = "native")]
    assert_eq!(
        invar::native::emit_mir(&typed_mir).unwrap().ir,
        invar::native::emit_mir(&plain_mir).unwrap().ir
    );
}

#[test]
fn issue_r1_01_units_frames_frame_type_erasure_is_byte_identical() {
    let typed = "frame world; fn f(v: read Vector3<i32,um,world>) -> i32 { return 7; } fn main() -> i32 { return 1; }";
    let prefix_pad = " ".repeat("frame world; ".len());
    let type_pad = " ".repeat("Vector3<i32,um,world>".len() - "Array<i32>".len());
    let plain = format!(
        "{prefix_pad}fn f(v: read Array<i32>{type_pad}) -> i32 {{ return 7; }} fn main() -> i32 {{ return 1; }}"
    );
    assert_eq!(typed.len(), plain.len());
    let lower = |source: &str| {
        let checked =
            invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
        invar::mir::lower(&checked).unwrap()
    };
    let typed_mir = lower(typed);
    let plain_mir = lower(&plain);
    assert_eq!(typed_mir.functions.len(), plain_mir.functions.len());
    for (typed, plain) in typed_mir.functions.iter().zip(&plain_mir.functions) {
        assert_eq!(typed.span, plain.span);
        assert_eq!(typed.declaration_order, plain.declaration_order);
        assert_eq!(format!("{:?}", typed.blocks), format!("{:?}", plain.blocks));
    }
    #[cfg(feature = "native")]
    assert_eq!(
        invar::native::emit_mir(&typed_mir).unwrap().ir,
        invar::native::emit_mir(&plain_mir).unwrap().ir
    );
}

#[test]
fn issue_r1_01_units_frames_device_text_is_byte_identical() {
    let kernel = "let xs: Array<u32> = [1u32,2u32,3u32];
        var ys: Array<u32> = filled_u32(len(xs),0u32);
        for i in 0u32 .. len(xs) { ys[i] = wrapping_add(xs[i],3u32); }";
    let typed = format!(
        "fn main() -> i32 {{ let marker: Quantity<i32,m> = quantity::<m>(1); {kernel}
        if marker != quantity::<m>(1) {{ return 1; }} return 0; }}"
    );
    let plain = format!(
        "fn main() -> i32 {{ let marker: i32 = 1; {kernel}
        if marker != 1 {{ return 1; }} return 0; }}"
    );
    let emit = |source: &str| {
        let checked =
            invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let lowered = invar::mir::lower(&checked).unwrap();
        let report = invar::admission::analyze(source.as_bytes(), &lowered).unwrap();
        let ir = invar::device_ir::from_region(&report.regions()[0], 3).unwrap();
        let origin = invar_device::Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let cuda = invar_device::verify(
            ir.clone(),
            origin,
            invar_device::Capabilities::cuda_12_9_sm89(),
        )
        .unwrap();
        let metal = invar_device::verify(
            ir,
            origin,
            invar_device::Capabilities::Metal {
                apple_family: 9,
                msl_major: 3,
                msl_minor: 1,
            },
        )
        .unwrap();
        (
            invar_device::emit_cuda(&cuda).unwrap(),
            invar_device::emit_msl(&metal).unwrap(),
        )
    };
    assert_eq!(emit(&typed), emit(&plain));
}

#[test]
fn issue_r1_01_units_frames_quantities_erase_and_engines_agree() {
    let source = include_str!("fixtures/r1_01_quantity_basic.inv");
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let reference = invar::eval::run(&checked, 1000).unwrap();
    assert_eq!(reference.value, invar::eval::Value::I32(1));
    let lowered = invar::mir::lower(&checked).unwrap();
    assert_eq!(invar::mir_eval::run(&lowered, 1000).unwrap(), reference);
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&lowered, 1000).unwrap();
        assert_eq!(status, 0);
        assert_eq!(document["result"]["value"], 1);
    }
}

#[test]
fn issue_r1_01_units_frames_quantity_source_errors() {
    for (fixture, code) in [
        ("mismatched-units.inv", "UNIT_MISMATCH"),
        ("unknown-unit.inv", "UNIT_UNKNOWN"),
        ("unit-as-value.inv", "UNIT_AS_VALUE"),
        ("record-mismatch.inv", "UNIT_MISMATCH"),
        ("array-quantity.inv", "UNIT_MISMATCH"),
        ("intrinsic-erasure.inv", "UNIT_MISMATCH"),
        ("loop-bound-erasure.inv", "UNIT_MISMATCH"),
        ("frame-mismatch.inv", "FRAME_MISMATCH"),
        ("transform-composition-mismatch.inv", "FRAME_MISMATCH"),
        ("frame-unknown.inv", "FRAME_UNKNOWN"),
        ("rotation-invalid.inv", "ROTATION_INVALID"),
    ] {
        let path = format!(
            "{}/tests/fixtures/r1_01/{fixture}",
            env!("CARGO_MANIFEST_DIR")
        );
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["check", &path, "--format", "json"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{fixture}");
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(document["diagnostics"][0]["code"], code, "{fixture}");
    }
}

#[test]
fn issue_r1_01_units_frames_transform_chain_agrees() {
    let source = include_str!("fixtures/r1_01_frame_values.inv");
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let reference = invar::eval::run(&checked, 10000).unwrap();
    assert_eq!(reference.value, invar::eval::Value::I32(1));
    let lowered = invar::mir::lower(&checked).unwrap();
    assert_eq!(invar::mir_eval::run(&lowered, 10000).unwrap(), reference);
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&lowered, 10000).unwrap();
        assert_eq!(status, 0, "{document}");
        assert_eq!(document["result"]["value"], 1);
    }
}

#[test]
fn issue_r1_01_units_frames_twenty_kinematics_programs_agree() {
    for case in 0..20i32 {
        let x = 100 + case * 17;
        let y = 50 + case;
        let base_offset = case + 1;
        let tool_offset = 2 * case + 3;
        let duration = case + 1;
        let speed = case % 5 + 1;
        let displacement = duration * speed;
        let heading = case + 3;
        let rotation = [
            "1,0,0, 0,1,0, 0,0,1",
            "0,-1,0, 1,0,0, 0,0,1",
            "-1,0,0, 0,-1,0, 0,0,1",
            "0,1,0, -1,0,0, 0,0,1",
        ][(case % 4) as usize];
        let (rotated_x, rotated_y, heading_x, heading_y) = match case % 4 {
            0 => (x + base_offset, y, heading, 0),
            1 => (-y, x + base_offset, 0, heading),
            2 => (-(x + base_offset), -y, -heading, 0),
            _ => (y, -(x + base_offset), 0, -heading),
        };
        let expected_position = rotated_x + tool_offset + displacement;
        let source = format!(
            "frame world; frame base: world; frame tool: base;
             fn main() -> i32 {{
                 let p: Point<i32,world> = point::<world>({x}, {y}, 0);
                 let a: Transform<world,base> = transform::<world,base>({base_offset},0,0, 1,0,0, 0,1,0, 0,0,1);
                 let b: Transform<base,tool> = transform::<base,tool>({tool_offset},0,0, {rotation});
                 let composed: Transform<world,tool> = a * b;
                 let q: Point<i32,tool> = composed * p;
                 let heading_world: Vector3<i32,um,world> = vector3::<um,world>({heading},0,0);
                 let heading_tool: Vector3<i32,um,tool> = composed * heading_world;
                 let v: Vector3<i32,um,tool> = vector3::<um,tool>({displacement},0,0);
                 let moved: Point<i32,tool> = q + v;
                 let distance: Quantity<i32,um> = moved[0u32] - q[0u32];
                 let time: Quantity<i32,s> = quantity::<s>({duration});
                 let velocity: Quantity<i32,Div<um,s>> = distance / time;
                 if moved[0u32] == quantity::<um>({expected_position}) and moved[1u32] == quantity::<um>({rotated_y}) and heading_tool[0u32] == quantity::<um>({heading_x}) and heading_tool[1u32] == quantity::<um>({heading_y}) and velocity == quantity::<Div<um,s>>({speed}) {{ return 1; }}
                 return 0;
             }}"
        );
        let checked = invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true)
            .unwrap_or_else(|error| panic!("case {case}: {error:?}"));
        let reference = invar::eval::run(&checked, 10000).unwrap();
        assert_eq!(reference.value, invar::eval::Value::I32(1), "case {case}");
        let lowered = invar::mir::lower(&checked).unwrap();
        assert_eq!(
            invar::mir_eval::run(&lowered, 10000).unwrap(),
            reference,
            "case {case}"
        );
        #[cfg(feature = "native")]
        {
            let (status, document) = invar::native::run_program(&lowered, 10000).unwrap();
            assert_eq!(status, 0, "case {case}: {document}");
            assert_eq!(document["result"]["value"], 1, "case {case}");
        }
    }
}

#[test]
fn issue_r1_01_units_frames_quantity_chain_respects_two_mib_stack() {
    let expression = vec!["distance"; 180].join(" + ");
    let source = format!(
        "fn main() -> i32 {{ let distance: Quantity<i32,m> = quantity::<m>(1); let sum: Quantity<i32,m> = {expression}; return 0; }}"
    );
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(move || {
            let checked =
                invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
            assert_eq!(
                invar::eval::run(&checked, 10000).unwrap().value,
                invar::eval::Value::I32(0)
            );
            let lowered = invar::mir::lower(&checked).unwrap();
            assert_eq!(
                invar::mir_eval::run(&lowered, 10000).unwrap().value,
                invar::eval::Value::I32(0)
            );
        })
        .unwrap()
        .join()
        .expect("quantity chain must fit the documented checker stack");
}

#[test]
fn issue_r1_01_units_frames_generated_traps_have_source_spans() {
    let source = include_str!("fixtures/r1_01/frame-overflow.inv");
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let lowered = invar::mir::lower(&checked).unwrap();
    for error in [
        invar::eval::run(&checked, 10000).unwrap_err(),
        invar::mir_eval::run(&lowered, 10000).unwrap_err(),
    ] {
        let invar::eval::EvalError::Trap { category, span } = error else {
            panic!("expected integer overflow trap: {error:?}");
        };
        assert_eq!(category, invar::eval::TrapCategory::IntegerOverflow);
        assert!(span.start <= span.end && span.end <= source.len());
    }
    let path = format!(
        "{}/tests/fixtures/r1_01/frame-overflow.inv",
        env!("CARGO_MANIFEST_DIR")
    );
    let engines = [
        "reference",
        #[cfg(feature = "native")]
        "native-serial",
    ];
    for engine in engines {
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["run", &path, "--engine", engine, "--format", "json"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(3), "{engine}");
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(document["status"], "trap", "{engine}");
        assert_eq!(document["trap"]["category"], "INTEGER_OVERFLOW", "{engine}");
        assert!(document["diagnostics"][0]["span"].is_object(), "{engine}");
        assert!(
            document["trap"]["span"]["end"].as_u64().unwrap() as usize <= source.len(),
            "{engine}"
        );
    }
}

#[test]
fn issue_r1_01_units_frames_scale_fix_applies_through_cli() {
    let source = include_str!("fixtures/r1_01/mismatched-units.inv");
    let path = std::env::temp_dir().join(format!("invar-r1-unit-fix-{}.inv", std::process::id()));
    std::fs::write(&path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["check", path.to_str().unwrap(), "--format", "json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["diagnostics"][0]["code"], "UNIT_MISMATCH");
    let edits = document["diagnostics"][0]["suggested_fixes"][0]["edits"]
        .as_array()
        .unwrap();
    let mut fixed = source.as_bytes().to_vec();
    for edit in edits.iter().rev() {
        let at = edit["span"]["start"].as_u64().unwrap() as usize;
        fixed.splice(at..at, edit["replacement"].as_str().unwrap().bytes());
    }
    std::fs::write(&path, fixed).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["check", path.to_str().unwrap(), "--format", "json"])
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn issue_r1_01_units_frames_record_fields_preserve_units() {
    let source = include_str!("fixtures/r1_01_record_quantity.inv");
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    assert_eq!(
        invar::eval::run(&checked, 1000).unwrap().value,
        invar::eval::Value::I32(2)
    );
}

#[test]
fn issue_r1_01_units_frames_dimension_products_quotients_and_rates() {
    let source = include_str!("fixtures/r1_01_dimension_algebra.inv");
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let reference = invar::eval::run(&checked, 1000).unwrap();
    assert_eq!(reference.value, invar::eval::Value::I32(1));
    let lowered = invar::mir::lower(&checked).unwrap();
    assert_eq!(invar::mir_eval::run(&lowered, 1000).unwrap(), reference);
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&lowered, 1000).unwrap();
        assert_eq!(status, 0);
        assert_eq!(document["result"]["value"], 1);
    }
}

#[test]
fn issue_r1_01_units_frames_exact_scale_conversion_agrees() {
    let source = include_str!("fixtures/r1_01_conversion.inv");
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let reference = invar::eval::run(&checked, 1000).unwrap();
    assert_eq!(reference.value, invar::eval::Value::I32(1));
    let lowered = invar::mir::lower(&checked).unwrap();
    assert_eq!(invar::mir_eval::run(&lowered, 1000).unwrap(), reference);
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&lowered, 1000).unwrap();
        assert_eq!(status, 0);
        assert_eq!(document["result"]["value"], 1);
    }
}

#[test]
fn issue_r1_01_units_frames_scale_traps_agree() {
    use invar::eval::{EvalError, TrapCategory};
    for (source, category, code) in [
        (
            include_str!("fixtures/r1_01/inexact-conversion.inv"),
            TrapCategory::UnitConversionInexact,
            "UNIT_CONVERSION_INEXACT",
        ),
        (
            include_str!("fixtures/r1_01/overflow-conversion.inv"),
            TrapCategory::IntegerOverflow,
            "INTEGER_OVERFLOW",
        ),
    ] {
        let checked =
            invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
        assert!(
            matches!(invar::eval::run(&checked, 1000), Err(EvalError::Trap { category: found, .. }) if found == category)
        );
        let lowered = invar::mir::lower(&checked).unwrap();
        assert!(
            matches!(invar::mir_eval::run(&lowered, 1000), Err(EvalError::Trap { category: found, .. }) if found == category)
        );
        #[cfg(feature = "native")]
        {
            let (status, document) = invar::native::run_program(&lowered, 1000).unwrap();
            assert_eq!(status, 3);
            assert_eq!(document["trap"]["category"], code);
        }
        #[cfg(not(feature = "native"))]
        let _ = code;
    }
}

#[test]
fn issue_r1_01_units_frames_certified_angle_conversion_agrees() {
    let source = include_str!("fixtures/r1_01_angles.inv");
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let reference = invar::eval::run(&checked, 1000).unwrap();
    assert_eq!(reference.value, invar::eval::Value::I32(1));
    let lowered = invar::mir::lower(&checked).unwrap();
    assert_eq!(invar::mir_eval::run(&lowered, 1000).unwrap(), reference);
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&lowered, 1000).unwrap();
        assert_eq!(status, 0);
        assert_eq!(document["result"]["value"], 1);
    }
}

#[test]
fn issue_r1_01_units_frames_angle_overflow_agrees() {
    use invar::eval::{EvalError, TrapCategory};
    let source = include_str!("fixtures/r1_01/angle-overflow.inv");
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    assert!(matches!(
        invar::eval::run(&checked, 1000),
        Err(EvalError::Trap {
            category: TrapCategory::IntegerOverflow,
            ..
        })
    ));
    let lowered = invar::mir::lower(&checked).unwrap();
    assert!(matches!(
        invar::mir_eval::run(&lowered, 1000),
        Err(EvalError::Trap {
            category: TrapCategory::IntegerOverflow,
            ..
        })
    ));
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&lowered, 1000).unwrap();
        assert_eq!(status, 3);
        assert_eq!(document["trap"]["category"], "INTEGER_OVERFLOW");
    }
}

#[test]
fn issue_r1_01_units_frames_nominal_tree_public_check() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/r1_01_frames.inv"
    );
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["check", path, "--format", "json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["status"], "ok");

    for (fixture, code) in [
        ("duplicate-frame.inv", "FRAME_DUPLICATE"),
        ("unknown-parent.inv", "FRAME_PARENT_UNKNOWN"),
        ("cycle.inv", "FRAME_CYCLE"),
        ("two-roots.inv", "FRAME_ROOTS"),
    ] {
        let path = format!(
            "{}/tests/fixtures/r1_01/{fixture}",
            env!("CARGO_MANIFEST_DIR")
        );
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["check", &path, "--format", "json"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{fixture}");
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(document["diagnostics"][0]["code"], code, "{fixture}");
    }

    let mut source = String::new();
    source.push_str("frame f0;");
    for i in 1..129 {
        source.push_str(&format!("frame f{i}: f{};", i - 1));
    }
    source.push_str("fn main() -> i32 { return 0i32; }");
    assert_eq!(
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true)
            .unwrap_err()
            .code,
        "RESOURCE_LIMIT"
    );
}
