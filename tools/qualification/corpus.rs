use invar::front::Span;

pub const SEED: u64 = 0x4d35_3032_2026_1014;
pub const FUEL: u64 = 100_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expected {
    Value(i32),
    Trap(&'static str, Span),
    Limit,
    Tool(&'static str),
    Refusal(&'static str),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Case {
    pub id: String,
    pub class: &'static str,
    pub source: Vec<u8>,
    pub fuel: u64,
    pub expected: Expected,
}

fn case(
    id: impl Into<String>,
    class: &'static str,
    source: impl Into<Vec<u8>>,
    expected: Expected,
) -> Case {
    Case {
        id: id.into(),
        class,
        source: source.into(),
        fuel: FUEL,
        expected,
    }
}

fn trap(id: &str, source: &str, operator: &str, category: &'static str) -> Case {
    let start = source.find(operator).unwrap();
    case(
        id,
        "trap",
        source,
        Expected::Trap(
            category,
            Span {
                start,
                end: start + operator.len(),
            },
        ),
    )
}

pub fn next(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn literal(xs: &[u32]) -> String {
    xs.iter()
        .map(|x| format!("{x}u32"))
        .collect::<Vec<_>>()
        .join(",")
}

/// Complete output checks are emitted in source, including every prefix/element.
/// The independent expectations use widened modular arithmetic, not Invar helpers.
pub fn corpus(seed: u64) -> Vec<Case> {
    let mut cases = vec![
        case(
            "return",
            "scalar",
            "fn main() -> i32 { return 42; }",
            Expected::Value(42),
        ),
        case(
            "min-remainder",
            "numeric",
            "fn main() -> i32 { return -2147483648i32 % -1i32; }",
            Expected::Value(0),
        ),
        case(
            "signed-truncation",
            "numeric",
            "fn main() -> i32 { return -7i32 / 3i32; }",
            Expected::Value(-2),
        ),
        case(
            "untaken-trap",
            "demand",
            "fn main() -> i32 { if true { return 7; } else { return 1 / 0; } }",
            Expected::Value(7),
        ),
        case(
            "short-circuit",
            "demand",
            "fn main() -> i32 { if false and (1 / 0 == 0) { return 1; } if true or (1 / 0 == 0) { return 9; } return 0; }",
            Expected::Value(9),
        ),
        case(
            "untaken-loop",
            "demand",
            "fn main() -> i32 { if false { while true {} } return 11; }",
            Expected::Value(11),
        ),
        case(
            "owner-return",
            "ownership",
            "fn make() -> Array<i32> { return [9]; } fn main() -> i32 { let a = make(); let b = move a; return b[0u32]; }",
            Expected::Value(9),
        ),
        case(
            "borrow-read-read",
            "ownership",
            "fn read2(a: read Array<i32>, b: read Array<i32>) -> i32 { return a[0u32] + b[0u32]; } fn main() -> i32 { let a: Array<i32> = [7]; return read2(a,a); }",
            Expected::Value(14),
        ),
        case(
            "rhs-before-index",
            "source-order",
            "fn bump(a: mut Array<i32>) -> i32 { a[0u32] = 1; return 9; } fn index(a: read Array<i32>) -> u32 { if a[0u32] == 1 { return 1u32; } return 0u32; } fn main() -> i32 { var a: Array<i32> = [0,0]; a[index(a)] = bump(a); return a[0u32] + a[1u32]; }",
            Expected::Value(10),
        ),
        case(
            "cleanup-prefix",
            "ownership",
            "fn main() -> i32 { if true { let retired: Array<i32> = [1]; } let live: Array<i32> = [7]; return live[0u32]; }",
            Expected::Value(7),
        ),
        case(
            "pointer-chasing",
            "dependency",
            "fn main() -> i32 { let edges: Array<u32> = [1u32,2u32,3u32,3u32]; var at = 0u32; for i in 0u32..3u32 { at = edges[at]; } if at == 3u32 { return 7; } return 0; }",
            Expected::Value(7),
        ),
        case(
            "strict-reassociation",
            "floating",
            "fn main() -> i32 { let a: f32 = 33554432.0; let b: f32 = -33554432.0; let c: f32 = 1.0; if (a+b)+c == 1.0 and a+(b+c) == 0.0 { return 7; } return 0; }",
            Expected::Value(7),
        ),
        case(
            "float-edge",
            "floating",
            "fn main() -> i32 { let zero: f32 = -0.0; let nan: f32 = 0.0/0.0; let inf: f32 = 1.0/zero; if nan != nan and inf < 0.0 and 0.00000000000000000000000000000000000000000000140129846432481707092372958328991613128026194187651577175706828388979108268586060148663818836212158203125 > 0.0 { return 7; } return 0; }",
            Expected::Value(7),
        ),
        case(
            "array-limit",
            "resource",
            format!(
                "fn main() -> i32 {{ let a = filled_i32({}u32,0); return 0; }}",
                invar::array_limits::MAX_ARRAY_LEN + 1
            ),
            Expected::Tool("ARRAY_SIZE_LIMIT"),
        ),
        case(
            "recursion-limit",
            "resource",
            "fn recurse() -> i32 { return recurse(); } fn main() -> i32 { return recurse(); }",
            Expected::Tool("CALL_DEPTH_LIMIT"),
        ),
    ];
    for (id, source, op, category) in [
        (
            "add-overflow",
            "fn main() -> i32 { return 2147483647i32 + 1i32; }",
            "+",
            "INTEGER_OVERFLOW",
        ),
        (
            "subtract-overflow",
            "fn main() -> i32 { return -2147483648i32 - 1i32; }",
            " - ",
            "INTEGER_OVERFLOW",
        ),
        (
            "multiply-overflow",
            "fn main() -> i32 { return 2147483647i32 * 2i32; }",
            "*",
            "INTEGER_OVERFLOW",
        ),
        (
            "min-division",
            "fn main() -> i32 { return -2147483648i32 / -1i32; }",
            "/",
            "INTEGER_OVERFLOW",
        ),
        (
            "zero-division",
            "fn main() -> i32 { return 7 / 0; }",
            "/",
            "DIVISION_BY_ZERO",
        ),
        (
            "zero-remainder",
            "fn main() -> i32 { return 7 % 0; }",
            "%",
            "DIVISION_BY_ZERO",
        ),
        (
            "unsigned-underflow",
            "fn main() -> i32 { let x = 0u32 - 1u32; return 0; }",
            " - ",
            "INTEGER_OVERFLOW",
        ),
        (
            "unsigned-overflow",
            "fn main() -> i32 { let x = 4294967295u32 + 1u32; return 0; }",
            "+",
            "INTEGER_OVERFLOW",
        ),
        (
            "ordered-arguments",
            "fn pick(a:i32,b:i32)->i32 { return b; } fn main()->i32 { return pick(1 / 0,2147483647i32 + 1); }",
            "/",
            "DIVISION_BY_ZERO",
        ),
        (
            "rhs-trap-before-index",
            "fn main()->i32 { var a: Array<i32> = [1]; a[9u32] = 1 / 0; return 0; }",
            "/",
            "DIVISION_BY_ZERO",
        ),
    ] {
        // Operators in the source contract have token spans, without whitespace.
        let mut item = trap(id, source, op, category);
        if let Expected::Trap(_, span) = &mut item.expected {
            span.start += op.len() - op.trim_start().len();
            span.end = span.start + op.trim().len();
        }
        cases.push(item);
    }
    let source = "fn main()->i32 { let a: Array<i32> = [1]; return a[2u32]; }";
    cases.push(trap("bad-index", source, "a[2u32]", "INDEX_OUT_OF_BOUNDS"));
    cases.push(trap(
        "unit-inexact",
        include_str!("../../tests/fixtures/r1_01/inexact-conversion.inv"),
        "to::<m>(mm)",
        "UNIT_CONVERSION_INEXACT",
    ));
    cases.push(trap(
        "unit-overflow",
        include_str!("../../tests/fixtures/r1_01/overflow-conversion.inv"),
        "to::<mm>(metres)",
        "INTEGER_OVERFLOW",
    ));
    let mut loop_case = case(
        "divergent-prefix",
        "fuel",
        "fn main()->i32 { while true {} return 0; }",
        Expected::Limit,
    );
    loop_case.fuel = 16;
    cases.push(loop_case);
    for fuel in [1, 2, 3] {
        let mut item = case(
            format!("fuel-{fuel}"),
            "fuel",
            "fn main()->i32 { return 42; }",
            if fuel >= 2 {
                Expected::Value(42)
            } else {
                Expected::Limit
            },
        );
        item.fuel = fuel;
        cases.push(item);
    }
    // Source forms stay ordinary loops; no M4 admission or device code is changed.
    for (id, class, source, value) in [
        (
            "transpose",
            "transpose",
            "fn main()->i32 { let a: Array<u32> = [1u32,2u32,3u32,4u32,5u32,6u32]; var b=filled_u32(6u32,0u32); for r in 0u32..2u32 { for c in 0u32..3u32 { b[c*2u32+r]=a[r*3u32+c]; } } let want: Array<u32> = [1u32,4u32,2u32,5u32,3u32,6u32]; for i in 0u32..6u32 { if b[i] != want[i] { return 1; } } return 0; }",
            0,
        ),
        (
            "stencil",
            "stencil",
            "fn main()->i32 { let a: Array<u32> = [1u32,2u32,3u32,4u32,5u32]; var b=filled_u32(5u32,0u32); for i in 1u32..4u32 { b[i]=wrapping_add(wrapping_add(a[i-1u32],a[i]),a[i+1u32]); } if b[0u32]==0u32 and b[1u32]==6u32 and b[2u32]==9u32 and b[3u32]==12u32 and b[4u32]==0u32 { return 0; } return 1; }",
            0,
        ),
        (
            "stable-compact",
            "compaction",
            "fn main()->i32 { let a: Array<u32> = [7u32,2u32,5u32,4u32,9u32]; var b=filled_u32(5u32,0u32); var at=0u32; for i in 0u32..len(a) { if a[i]%2u32==1u32 { b[at]=a[i]; at=at+1u32; } } if at==3u32 and b[0u32]==7u32 and b[1u32]==5u32 and b[2u32]==9u32 { return 0; } return 1; }",
            0,
        ),
        (
            "segmented-empty-ragged",
            "segmented",
            "fn main()->i32 { let starts: Array<u32> = [0u32,0u32,1u32,4u32]; let a: Array<u32> = [2u32,3u32,5u32,7u32]; var b=filled_u32(3u32,0u32); for segment in 0u32..3u32 { var acc=0u32; for i in starts[segment]..starts[segment+1u32] { acc=wrapping_add(acc,a[i]); } b[segment]=acc; } if b[0u32]==0u32 and b[1u32]==2u32 and b[2u32]==15u32 { return 0; } return 1; }",
            0,
        ),
        (
            "gather-scatter",
            "gather-scatter",
            "fn main()->i32 { let indices: Array<u32> = [2u32,0u32,1u32]; let a: Array<u32> = [7u32,11u32,13u32]; var b=filled_u32(3u32,0u32); for i in 0u32..3u32 { b[indices[i]]=a[indices[i]]; } for j in 0u32..3u32 { if b[j]!=a[j] { return 1; } } return 0; }",
            0,
        ),
        (
            "histogram",
            "histogram",
            "fn main()->i32 { let bins: Array<u32> = [0u32,1u32,0u32,2u32,1u32]; var out=filled_u32(3u32,4294967295u32); for i in 0u32..len(bins) { out[bins[i]]=wrapping_add(out[bins[i]],1u32); } if out[0u32]==1u32 and out[1u32]==1u32 and out[2u32]==0u32 { return 0; } return 1; }",
            0,
        ),
        (
            "strict-matrix",
            "matrix-product",
            "fn main()->i32 { let a: Array<f32> = [1.0,2.0,3.0,4.0]; let b: Array<f32> = [5.0,6.0,7.0,8.0]; var out=filled_f32(4u32,0.0); for r in 0u32..2u32 { for c in 0u32..2u32 { var acc:f32=0.0; for k in 0u32..2u32 { acc=acc+a[r*2u32+k]*b[k*2u32+c]; } out[r*2u32+c]=acc; } } if out[0u32]==19.0 and out[1u32]==22.0 and out[2u32]==43.0 and out[3u32]==50.0 { return 0; } return 1; }",
            0,
        ),
        (
            "finite-summary",
            "summary",
            "fn main()->i32 { let coeff: Array<u32> = [1u32,0u32,0u32,1u32,1u32,1u32]; var state=0u32; var out=filled_u32(3u32,0u32); for i in 0u32..3u32 { state=coeff[i*2u32+state]; out[i]=state; } if out[0u32]==1u32 and out[1u32]==1u32 and out[2u32]==1u32 { return 0; } return 1; }",
            0,
        ),
        (
            "module",
            "language",
            "module m { pub fn answer()->i32 { return 7; } } fn main()->i32 { return m::answer(); }",
            7,
        ),
        (
            "record",
            "language",
            "struct Pair { x:i32, y:i32 } fn main()->i32 { let p=Pair { x:7,y:11 }; return p.x+p.y; }",
            18,
        ),
        (
            "variant",
            "language",
            "enum Choice { Low, High(i32) } fn main()->i32 { let c:Choice=Choice::High(11); return match c { Choice::Low => 1i32, Choice::High(x) => x }; }",
            11,
        ),
        (
            "generic",
            "language",
            "fn echo<T:Value>(x:T)->T { return x; } fn main()->i32 { return echo::<i32>(7); }",
            7,
        ),
        (
            "closure",
            "language",
            "fn main()->i32 { let c=closure []()->i32 { return 7; }; return c(); }",
            7,
        ),
        (
            "views",
            "language",
            "fn main()->i32 { var a: Array<i32> = [1,2,3,4]; if true { let v=view_mut(a,0u32,2u32); v[1u32]=7; } let sample=view(a,1u32,4u32,2u32); return sample[0u32]+sample[1u32]; }",
            11,
        ),
        (
            "nested-comment-crlf",
            "parser",
            "/* outer /* inner */ done */\r\nfn main()->i32 { return 7; }",
            7,
        ),
    ] {
        cases.push(case(id, class, source, Expected::Value(value)));
    }
    cases.push(case(
        "affine-closure",
        "language",
        include_bytes!("../../tests/fixtures/l1_03_closures.inv").to_vec(),
        Expected::Value(10),
    ));
    cases.push(case(
        "result-propagation",
        "language",
        include_bytes!("../../tests/fixtures/l1_01/result-propagation.inv").to_vec(),
        Expected::Value(28),
    ));
    cases.push(case(
        "utf8-blobs",
        "language",
        include_bytes!("../../tests/fixtures/l1_04/basic.inv").to_vec(),
        Expected::Value(7),
    ));
    let mut state = seed;
    for n in [0usize, 1, 31, 32, 33, 63, 64, 65, 255, 256, 257] {
        let xs = (0..n).map(|_| next(&mut state)).collect::<Vec<_>>();
        let inputs = literal(&xs);
        let mut source = format!(
            "fn main()->i32 {{ let xs: Array<u32> = [{inputs}]; var first = filled_u32(len(xs),0u32); for i in 0u32..len(xs) {{ first[i] = wrapping_mul(xs[i],3u32); }} let stable: Array<u32> = move first; var ys = filled_u32(len(stable),0u32); for j in 0u32..len(stable) {{ ys[j] = wrapping_add(stable[j],7u32); }}"
        );
        for (i, x) in xs.iter().enumerate() {
            let y = ((u64::from(*x) * 3 + 7) % (1u64 << 32)) as u32;
            source.push_str(&format!("if ys[{i}u32] != {y}u32 {{ return 1; }}"));
        }
        source.push_str("return 0; }");
        cases.push(case(format!("map-{n}"), "map", source, Expected::Value(0)));
        let mut source = format!(
            "fn main()->i32 {{ let xs: Array<u32> = [{inputs}]; var acc = 11u32; var ys = filled_u32(len(xs),0u32); for i in 0u32..len(xs) {{ acc = wrapping_add(wrapping_mul(acc,3u32),xs[i]); ys[i] = acc; }}"
        );
        let mut acc = 11u64;
        for (i, x) in xs.iter().enumerate() {
            acc = (acc * 3 + u64::from(*x)) % (1u64 << 32);
            source.push_str(&format!("if ys[{i}u32] != {acc}u32 {{ return 1; }}"));
        }
        source.push_str(&format!("if acc != {acc}u32 {{ return 2; }} return 0; }}"));
        cases.push(case(
            format!("scan-{n}"),
            "scan",
            source,
            Expected::Value(0),
        ));
        let sum = xs
            .iter()
            .fold(9u64, |a, x| (a + u64::from(*x)) % (1u64 << 32));
        let source = format!(
            "fn main()->i32 {{ let xs: Array<u32> = [{inputs}]; var acc = 9u32; for i in 0u32..len(xs) {{ acc = wrapping_add(acc,xs[i]); }} if acc == {sum}u32 {{ return 0; }} return 1; }}"
        );
        cases.push(case(
            format!("reduce-{n}"),
            "reduction",
            source,
            Expected::Value(0),
        ));
    }
    for i in 0..32 {
        let n = next(&mut state) % 9;
        let bias = next(&mut state) % 7;
        let source = format!(
            "fn f(n:u32)->i32 {{ if n == 0u32 {{ return {bias}; }} return f(n-1u32)+2; }} fn main()->i32 {{ var acc = 0; for i in 0u32..{n}u32 {{ if i%2u32 == 0u32 {{ acc = acc+f(i); }} else {{ acc = acc-f(i); }} }} return acc; }}"
        );
        let expected = (0..n)
            .map(|j| {
                let x = (bias + 2 * j) as i32;
                if j % 2 == 0 { x } else { -x }
            })
            .sum();
        cases.push(case(
            format!("generated-{i}"),
            "generated-control",
            source,
            Expected::Value(expected),
        ));
    }
    for (id, source, code) in [
        (
            "alias-mut-read",
            "fn f(a:mut Array<i32>,b:read Array<i32>)->i32 { return b[0u32]; } fn main()->i32 { var a: Array<i32> = [1]; return f(a,a); }",
            "OWN_BORROW_CONFLICT",
        ),
        (
            "alias-mut-mut",
            "fn f(a:mut Array<i32>,b:mut Array<i32>)->i32 { return b[0u32]; } fn main()->i32 { var a: Array<i32> = [1]; return f(a,a); }",
            "OWN_BORROW_CONFLICT",
        ),
        (
            "use-after-move",
            "fn main()->i32 { let a: Array<i32> = [1]; let b = move a; return a[0u32]; }",
            "OWN_USE_AFTER_MOVE",
        ),
        (
            "mixed-integers",
            "fn main()->i32 { return 1i32+1u32; }",
            "TYPE_MISMATCH",
        ),
        (
            "missing-delimiter",
            "fn main()->i32 { return (1; }",
            "PARSE_EXPECTED_TOKEN",
        ),
        (
            "malformed-literal",
            "fn main()->i32 { return 12oops; }",
            "LEX_INVALID_LITERAL",
        ),
        ("unterminated-comment", "/*", "LEX_INVALID_TOKEN"),
    ] {
        cases.push(case(id, "refusal", source, Expected::Refusal(code)));
    }
    cases.push(case(
        "invalid-utf8",
        "refusal",
        vec![0xff, 0xfe],
        Expected::Refusal("INVALID_UTF8"),
    ));
    cases.push(case(
        "nesting-limit",
        "resource",
        format!(
            "fn main()->i32 {{ return {}1{}; }}",
            "(".repeat(33),
            ")".repeat(33)
        ),
        Expected::Refusal("SOURCE_NESTING_LIMIT"),
    ));
    cases
}
