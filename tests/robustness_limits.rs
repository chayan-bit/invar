#[path = "support/language_limits.rs"]
mod language_limits;
use std::time::{Duration, Instant};

use invar::{check::check, front::parse, mir};

fn checked(source: &str) -> invar::check::CheckedProgram {
    check(parse(source.as_bytes()).unwrap(), true).unwrap()
}

fn sequential_ifs(count: usize) -> String {
    let mut source = String::from("fn main() -> i32 {\n    var x: i32 = 0;\n");
    for index in 0..count {
        source.push_str(&format!("    if x < {index} {{ x = x + 1; }}\n"));
    }
    source.push_str("    return x;\n}\n");
    source
}

#[test]
fn mir_verification_is_not_superlinear_in_sequential_branches() {
    let program = checked(&sequential_ifs(1000));
    let started = Instant::now();
    let lowered = mir::lower(&program).unwrap();
    mir::verify(&lowered).unwrap();
    // Backstop only: the former set-based dominator pass took minutes here.
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[test]
fn mir_size_limits_are_enforced_by_verify_for_every_caller() {
    let small = mir::lower(&checked("fn main() -> i32 { return 1; }")).unwrap();

    let mut blocks = small.clone();
    let block = blocks.functions[0].blocks[0].clone();
    blocks.functions[0]
        .blocks
        .resize(mir::MAX_BLOCKS_PER_FUNCTION + 1, block);
    let mut functions = small.clone();
    let function = functions.functions[0].clone();
    functions.functions.resize(mir::MAX_FUNCTIONS + 1, function);
    let mut instructions = small.clone();
    let tick = instructions.functions[0].blocks[0].instructions[0].clone();
    instructions.functions[0].blocks[0]
        .instructions
        .resize(mir::MAX_INSTRUCTIONS + 1, tick);

    for program in [blocks, functions, instructions] {
        assert_eq!(mir::verify(&program).unwrap_err().code, "RESOURCE_LIMIT");
        let observed = invar::mir_eval::observe(&program, 10);
        assert!(
            matches!(
                observed.result,
                Err(invar::eval::EvalError::Tool {
                    code: "RESOURCE_LIMIT",
                    ..
                })
            ),
            "{:?}",
            observed.result
        );
    }

    // A real source program over the block limit is rejected by lowering.
    let error = mir::lower(&checked(&sequential_ifs(mir::MAX_BLOCKS_PER_FUNCTION))).unwrap_err();
    assert_eq!(error.code, "RESOURCE_LIMIT");
}

fn live_arrays_result(source: &str) -> (Result<(), &'static str>, Result<(), &'static str>) {
    use invar::eval::EvalError;
    let program = checked(source);
    let classify = |result: Result<invar::eval::Value, EvalError>| match result {
        Ok(_) => Ok(()),
        Err(EvalError::Tool { code, .. }) => Err(code),
        Err(other) => panic!("unexpected outcome {other:?}"),
    };
    let fuel = 100_000;
    let reference = invar::eval::observe(&program, fuel);
    let lowered = mir::lower(&program).unwrap();
    let mir = invar::mir_eval::observe(&lowered, fuel);
    assert_eq!(reference.steps_used, mir.steps_used, "{source}");
    let reference = classify(reference.result);
    #[cfg(feature = "native")]
    {
        let (_, document) = invar::native::run_program(&lowered, fuel).unwrap();
        let native = match document["status"].as_str() {
            Some("ok") => Ok(()),
            _ => Err(document["tool_code"].as_str().unwrap_or("?").to_owned()),
        };
        assert_eq!(native, reference.map_err(str::to_owned), "{source}");
    }
    (reference, classify(mir.result))
}

#[test]
fn live_array_elements_are_capped_in_aggregate() {
    let pad = language_limits::padding(invar::array_limits::MAX_LIVE_ELEMENTS - 2);
    let three = format!(
        "fn main() -> i32 {{ {pad} let a: Array<i32> = [1]; let b: Array<u32> = [2u32]; let c: Array<i32> = [3]; return a[0u32] + c[0u32]; }}"
    );
    assert_eq!(
        live_arrays_result(&three),
        (Err("ARRAY_SIZE_LIMIT"), Err("ARRAY_SIZE_LIMIT"))
    );
    let two = format!(
        "fn main() -> i32 {{ {pad} let a: Array<i32> = [1]; let b: Array<u32> = [2u32]; return a[0u32]; }}"
    );
    assert_eq!(live_arrays_result(&two), (Ok(()), Ok(())));

    // Released arrays return their elements: scope exit, reassignment and a
    // moved owner cleaned up by its callee never accumulate past the cap.
    let recycled = "fn consume(xs: take Array<i32>) -> i32 { return xs[0u32]; }
        fn main() -> i32 {
            var sum: i32 = 0;
            var keep: Array<i32> = filled_i32(1000000u32, 1);
            for i in 0u32 .. 3u32 {
                let scratch: Array<i32> = filled_i32(1000000u32, 1);
                sum = sum + scratch[0u32];
            }
            for j in 0u32 .. 3u32 {
                keep = filled_i32(1000000u32, 1);
            }
            for k in 0u32 .. 3u32 {
                let moved: Array<i32> = filled_i32(1000000u32, 1);
                sum = sum + consume(move moved);
            }
            return sum + keep[0u32];
        }";
    let recycled = recycled
        .replace("1000000u32", "1u32")
        .replace("fn main() -> i32 {", &format!("fn main() -> i32 {{ {pad}"));
    assert_eq!(live_arrays_result(&recycled), (Ok(()), Ok(())));
}

fn plus_chain(terms: usize) -> String {
    let chain = vec!["1"; terms].join(" + ");
    format!("fn main() -> i32 {{ return {chain}; }}")
}

fn else_if_chain(links: usize) -> String {
    let mut source = String::from("fn main() -> i32 { var x: i32 = 0; if x == 1 { x = 1; }");
    for link in 2..=links {
        source.push_str(&format!(" else if x == {link} {{ x = {link}; }}"));
    }
    source.push_str(" else { x = 7; } return x; }");
    source
}

/// Syntactic nesting close to its limit around a flat chain at the height limit.
fn nested_chain(terms: usize) -> String {
    let depth = 29;
    format!(
        "fn main() -> i32 {{ {}return {}; {}return 0; }}",
        "if true { ".repeat(depth),
        vec!["1"; terms].join(" + "),
        "} ".repeat(depth)
    )
}

fn is_height_limit(source: &str) -> bool {
    match parse(source.as_bytes()) {
        Ok(_) => false,
        Err(error) => {
            assert_eq!(error.kind, invar::front::ErrorKind::Tool, "{error:?}");
            assert_eq!(error.code, "SOURCE_NESTING_LIMIT");
            true
        }
    }
}

type Shape = fn(usize) -> String;

/// Largest `n` accepted by the parser; `n + 1` must hit the height limit.
fn largest_accepted(shape: Shape) -> usize {
    let mut n = 1;
    while !is_height_limit(&shape(n + 1)) {
        n += 1;
    }
    n
}

fn run_every_walker(source: &str) {
    let program = checked(source);
    let reference = invar::eval::observe(&program, 1_000_000);
    assert!(reference.result.is_ok(), "{:?}", reference.result);
    let lowered = mir::lower(&program).unwrap();
    let mir = invar::mir_eval::observe(&lowered, 1_000_000);
    assert_eq!(mir.result, reference.result);
    assert_eq!(mir.steps_used, reference.steps_used);
}

#[test]
fn tree_height_limit_is_stack_safe_on_a_two_mib_thread() {
    let shapes: [(Shape, usize); 3] =
        [(plus_chain, 180), (else_if_chain, 180), (nested_chain, 120)];
    for (shape, minimum) in shapes {
        let n = largest_accepted(shape);
        assert!(n >= minimum, "limit too small: {n}");
        let source = shape(n);
        std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(move || run_every_walker(&source))
            .unwrap()
            .join()
            .expect("walkers must not overflow a 2 MiB stack at the height limit");
    }
}

#[test]
fn flat_chains_are_not_syntactic_nesting() {
    run_every_walker(&plus_chain(33));
    run_every_walker(&else_if_chain(20));
    let parens = format!(
        "fn main() -> i32 {{ return {}1{}; }}",
        "(".repeat(33),
        ")".repeat(33)
    );
    assert!(is_height_limit(&parens));
}
