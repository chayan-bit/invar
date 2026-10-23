use invar::front::{BinaryOp, ElseArm, ErrorKind, ExprKind, StmtKind, parse};

#[test]
fn issue_m0_02_examples_and_grammar() {
    for source in [
        include_bytes!("../examples/sum.inv").as_slice(),
        include_bytes!("../examples/ownership.inv").as_slice(),
    ] {
        let program = parse(source).expect("committed example parses");
        assert_eq!(
            program.functions.len(),
            if source == include_bytes!("../examples/sum.inv") {
                2
            } else {
                3
            }
        );
    }
    let source = br#"
        fn all(a: read Array<i32>, b: mut Array<u32>, c: take Array<i32>, n: u32,) -> Unit {
            let flag: Bool = true and not false or 1 < 2;
            var x: i32 = -2147483648;
            x = (x + 2) * 3 / 2 % 4 - 1;
            b[0u32] = 1u32;
            if flag { x = c[0u32]; } else if false { return; } else { x = a[0u32]; }
            while false { x = x + 1; }
            for i in 0u32 .. n { x = x + 1; }
            consume(move c, [1, 2,], x,);
            return;
        }
        fn consume(v: take Array<i32>, a: take Array<i32>, x: i32,) -> Unit { return; }
    "#;
    let program = parse(source).expect("all statement and expression forms parse");
    assert_eq!(program.functions.len(), 2);
    assert_eq!(program.functions[0].params.len(), 4);
    assert_eq!(program.functions[0].body.statements.len(), 9);
    let StmtKind::If {
        else_arm: Some(ElseArm::If(_)),
        ..
    } = &program.functions[0].body.statements[4].kind
    else {
        panic!("else-if missing")
    };
}

#[test]
fn issue_m0_02_precedence_and_exclusions() {
    let program = parse(b"fn f() -> Bool { return 1 + 2 * 3 == 7 or false and true; }").unwrap();
    let StmtKind::Return(Some(expr)) = &program.functions[0].body.statements[0].kind else {
        panic!("return")
    };
    let ExprKind::Binary {
        op: BinaryOp::Or,
        left,
        right,
        ..
    } = &expr.kind
    else {
        panic!("or root")
    };
    assert!(matches!(
        left.kind,
        ExprKind::Binary {
            op: BinaryOp::Eq,
            ..
        }
    ));
    assert!(matches!(
        right.kind,
        ExprKind::Binary {
            op: BinaryOp::And,
            ..
        }
    ));
    let more = b"fn f() -> Bool { let xs: Array<u32> = []; return 1 <= 2 and 3 >= 4 or 5 != 6 and 7 > 8 and 9 == 10; }";
    assert!(parse(more).is_ok());
    assert!(parse(b"fn f() -> Array<Bool> { return []; }").is_ok());
    for bad in [
        "fn f() -> Bool { return 1 < 2 < 3; }",
        "fn f() -> i32 { return f()(); }",
        "fn f() -> i32 { return 1 = 2; }",
        "fn f() -> i32 { (x) = 1; return 0; }",
        "fn f() -> Array<Unit> { return []; }",
    ] {
        assert!(
            parse(bad.as_bytes()).is_err(),
            "accepted unsupported grammar: {bad}"
        );
    }
}

#[test]
fn issue_m0_02_lexical_spans_and_json() {
    let source = "/* λ /* inner */ */\r\nfn f() -> i32 { return 1; }";
    let program = parse(source.as_bytes()).unwrap();
    assert_eq!(program.functions[0].span.start, source.find("fn").unwrap());
    let bad = "fn f() -> i32 { return 12oops; }";
    let error = parse(bad.as_bytes()).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Source);
    assert_eq!(error.code, "LEX_INVALID_LITERAL");
    let start = bad.find("12oops").unwrap();
    assert_eq!(error.span.unwrap().start, start);
    assert_eq!(error.span.unwrap().end, start + 6);
    let document = error.to_json("a\"b\n.inv");
    let encoded = serde_json::to_string(&document).unwrap();
    let decoded: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded["schema_version"], 3);
    assert_eq!(decoded["status"], "source_error");
    assert_eq!(decoded["diagnostics"][0]["path"], "a\"b\n.inv");
    assert_eq!(decoded["diagnostics"][0]["span"]["start"], start);
    assert!(
        decoded["diagnostics"][0]["notes"]
            .as_array()
            .unwrap()
            .is_empty()
    );

    let bad = "fn f() -> i32 { return 1 }";
    let error = parse(bad.as_bytes()).unwrap_err();
    assert_eq!(error.code, "PARSE_EXPECTED_TOKEN");
    let closing = bad.find('}').unwrap();
    assert_eq!(error.span.unwrap().start, closing);
    assert_eq!(error.span.unwrap().end, closing + 1);
    assert!(parse(b"fn f() -> f32 { return 12.5; }").is_ok());
    for bad in ["0x10", "12i32oops", "1_000", "12.5e3"] {
        let source = format!("fn f() -> i32 {{ return {bad}; }}");
        assert_eq!(
            parse(source.as_bytes()).unwrap_err().code,
            "LEX_INVALID_LITERAL"
        );
    }
    assert_eq!(
        parse(b"/* outer /* inner */").unwrap_err().code,
        "LEX_INVALID_TOKEN"
    );
    assert_eq!(
        parse(b"fn enum() -> Unit {}").unwrap_err().code,
        "UNSUPPORTED_FEATURE"
    );
}

#[test]
fn issue_m0_02_tool_limits_and_encoding() {
    let error = parse(b"\xff").unwrap_err();
    assert_eq!(error.kind, ErrorKind::Tool);
    assert_eq!(error.code, "INVALID_UTF8");
    assert!(error.to_json("bad.inv")["diagnostics"][0]["span"].is_null());
    let error = parse(&vec![b' '; 1_048_577]).unwrap_err();
    assert_eq!(error.code, "SOURCE_SIZE_LIMIT");
    let source = format!(
        "fn f() -> i32 {{ return {}1{}; }}",
        "(".repeat(130),
        ")".repeat(130)
    );
    assert_eq!(
        parse(source.as_bytes()).unwrap_err().code,
        "SOURCE_NESTING_LIMIT"
    );
    let source = format!("{}{}", "/*".repeat(130), "*/".repeat(130));
    assert_eq!(
        parse(source.as_bytes()).unwrap_err().code,
        "SOURCE_NESTING_LIMIT"
    );
    // Flat operator chains are not nesting but are bounded by tree height.
    let source = format!("fn f() -> i32 {{ return {}; }}", "1+".repeat(300) + "1");
    assert_eq!(
        parse(source.as_bytes()).unwrap_err().code,
        "SOURCE_NESTING_LIMIT"
    );
}
