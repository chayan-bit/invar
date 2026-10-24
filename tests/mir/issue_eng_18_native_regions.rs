#![cfg(feature = "native")]
//! ENG-18 step 4 (#101): cpu-parallel map and reduce regions run compiled,
//! vectorized native kernels, with the plan interpreter's exact results.

use invar::{
    check::check,
    front::parse,
    laws::Law,
    mir,
    native::{
        self,
        region::{self, Policy},
    },
    schedule,
    typed_net::{MapOp, Plan, ZipOp},
};

/// The policy override is process-wide, so the tests of this module run one
/// at a time and each starts from `Always`.
fn always() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let guard = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    region::set_policy(Some(Policy::Always));
    guard
}

fn live(
    pre: Vec<Vec<MapOp>>,
    zip: Option<ZipOp>,
    stages: Vec<Vec<MapOp>>,
    slots: Vec<u32>,
) -> Plan {
    Plan::LiveArray {
        inputs: slots,
        pre,
        zip,
        stages,
        source_identity: 0,
        graph_revision: 0,
    }
}

/// Deterministic words covering 0, 1, u32::MAX and everything between.
fn words(len: usize, seed: u32) -> Vec<u32> {
    let mut state = seed;
    (0..len)
        .map(|index| match index % 17 {
            0 => 0,
            1 => u32::MAX,
            2 => 1,
            _ => {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state
            }
        })
        .collect()
}

fn ranges(len: usize) -> Vec<std::ops::Range<usize>> {
    vec![
        0..len,
        0..0,
        3..3,
        0..1,
        1..len,
        5..len.min(70),
        len / 3..len - 1,
        len - 9..len,
    ]
}

#[test]
fn issue_eng_18_map_kernels_match_the_plan_interpreter() {
    let _serial = always();
    use MapOp::{WrappingAdd as Add, WrappingMul as Mul};
    let plans = [
        live(vec![vec![]], None, vec![vec![Mul(3)]], vec![0]),
        live(
            vec![vec![Add(u32::MAX)]],
            None,
            vec![vec![Mul(2654435761), Add(7)], vec![Mul(0)]],
            vec![0],
        ),
        live(
            vec![vec![], vec![]],
            Some(ZipOp::WrappingAdd),
            vec![],
            vec![0, 1],
        ),
        // The same IR as the plan above on other slots: a separate kernel.
        live(
            vec![vec![], vec![]],
            Some(ZipOp::WrappingAdd),
            vec![],
            vec![0, 0],
        ),
        live(
            vec![vec![Mul(5), Add(1)], vec![Add(9)]],
            Some(ZipOp::WrappingMul),
            vec![vec![Add(4294967291)], vec![Mul(31)]],
            vec![1, 0],
        ),
        // Both kernel inputs bound to the same slot.
        live(
            vec![vec![], vec![Mul(7)]],
            Some(ZipOp::WrappingAdd),
            vec![],
            vec![0, 0],
        ),
    ];
    let len = 1000 + 7;
    let arrays = [words(len, 0x9e37_79b9), words(len, 0x85eb_ca6b)];
    let slices = [arrays[0].as_slice(), arrays[1].as_slice()];
    for plan in &plans {
        let names = match plan {
            Plan::LiveArray { inputs, .. } => inputs
                .iter()
                .map(|slot| format!("in{slot}"))
                .collect::<Vec<_>>(),
            _ => unreachable!(),
        };
        let kernel = region::map_kernel(plan, &names, "out", len)
            .unwrap_or_else(|| panic!("no kernel for {plan:?}: {:?}", region::last_error()));
        for range in ranges(len) {
            let expected = plan.live_values(&slices, range.clone(), None).unwrap().1;
            assert_eq!(
                kernel.map(&slices, range.clone()),
                Some(expected),
                "{plan:?} {range:?}"
            );
        }
        // A slot shorter than the range refuses, so the caller interprets.
        assert_eq!(
            kernel.map(&[&slices[0][..10], &slices[1][..10]], 0..11),
            None
        );
    }
}

/// Compile cost of one kernel (emit, audit, O3, link, load), printed to
/// size `region::AUTO_MIN_ELEMENTS`; not a timing claim.
#[test]
fn issue_eng_18_kernel_compile_cost_is_reported() {
    let _serial = always();
    let plan = live(
        vec![vec![]],
        None,
        vec![vec![MapOp::WrappingMul(0x1234_5677)]],
        vec![0],
    );
    let started = std::time::Instant::now();
    let kernel = region::map_kernel(&plan, &["xs".to_owned()], "out", 1);
    eprintln!(
        "kernel compile and load: {:?} (ok: {})",
        started.elapsed(),
        kernel.is_some()
    );
    assert!(kernel.is_some(), "{:?}", region::last_error());
}

#[test]
fn issue_eng_18_reduce_kernels_match_the_left_fold() {
    let _serial = always();
    for law in [Law::WrappingSum, Law::WrappingProduct] {
        let kernel = region::reduce_kernel(law, 1).unwrap();
        for len in [0, 1, 2, 7, 8, 9, 63, 64, 65, 1000, 4099] {
            let input = words(len, 0x2545_f491 ^ len as u32);
            let expected = input.iter().fold(law.identity(), |a, b| law.combine(a, *b));
            assert_eq!(kernel.reduce(&input), Some(expected), "{law:?} {len}");
        }
    }
}

/// The audited kernels carry facts for every promise and vectorize for the
/// live CPU: their optimized IR holds vector operations.
#[test]
fn issue_eng_18_region_kernels_are_audited_and_vectorized() {
    let _serial = always();
    let plan = live(
        vec![vec![], vec![]],
        Some(ZipOp::WrappingMul),
        vec![vec![MapOp::WrappingAdd(1)]],
        vec![0, 1],
    );
    let names = ["xs".to_owned(), "ys".to_owned()];
    let kernels = [
        region::map(&plan, &names, "out").unwrap(),
        region::reduce(Law::WrappingSum),
    ];
    for kernel in kernels {
        let promised = native::audit(&kernel.ir, &kernel.facts).unwrap();
        assert!(!promised.is_empty());
        let (object, assembly) =
            native::compile_ir(&kernel.ir, kernel.facts.clone(), &native::Build::HOST).unwrap();
        if object.machine.cpu == "generic" {
            eprintln!("unknown host CPU: vectorization evidence not available here");
            continue;
        }
        let optimized = object.optimized_ir.unwrap();
        assert!(optimized.contains("x i32>"), "{optimized}");
        let vector = match std::env::consts::ARCH {
            "x86_64" => assembly.contains("%ymm") || assembly.contains("%zmm"),
            _ => assembly.contains(".4s"),
        };
        assert!(vector, "{assembly}");
    }
    // The same owner on both sides of the output refuses disjointness.
    let aliased = region::map(&plan, &names, "xs").unwrap();
    assert!(native::audit(&aliased.ir, &aliased.facts).is_err());
}

fn lower(source: &str) -> mir::Program {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    mir::lower(&checked).unwrap()
}

/// A seeded fill, a wrapping map over a `read` input and a wrapping sum over
/// `n` elements; `main` returns 1 exactly when the sum equals the one
/// computed here in Rust.
fn program(n: u32) -> String {
    let mut seed = 2463534242u32;
    let mut acc = 0u32;
    for _ in 0..n {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        acc = acc.wrapping_add(seed.wrapping_mul(2654435761));
    }
    format!(
        "fn fill(n: u32) -> Array<u32> {{
    var xs: Array<u32> = filled_u32(n, 0u32);
    var seed: u32 = 2463534242u32;
    for i in 0u32 .. len(xs) {{ seed = wrapping_add(wrapping_mul(seed, 1664525u32), 1013904223u32); xs[i] = seed; }}
    return xs;
}}
fn mapped(xs: read Array<u32>) -> Array<u32> {{
    var ys: Array<u32> = filled_u32(len(xs), 0u32);
    for i in 0u32 .. len(xs) {{
        ys[i] = wrapping_mul(xs[i], 2654435761u32);
    }}
    return ys;
}}
fn sum_of(ys: read Array<u32>) -> u32 {{
    var acc: u32 = 0u32;
    for i in 0u32 .. len(ys) {{
        acc = wrapping_add(acc, ys[i]);
    }}
    return acc;
}}
fn main() -> i32 {{
    let xs: Array<u32> = fill({n}u32);
    let ys: Array<u32> = mapped(xs);
    if sum_of(ys) == {acc}u32 {{
        return 1;
    }}
    return 0;
}}"
    )
}

fn steps(observation: &str) -> u64 {
    let tail = observation.rsplit("steps_used: ").next().unwrap();
    tail[..tail.find(',').unwrap()].parse().unwrap()
}

/// cpu-parallel observations (value, trap, fuel, cleanup) are identical with
/// kernels on and off, at every worker count, and on fuel prefixes that end
/// before, inside and after the compiled regions.
#[test]
fn issue_eng_18_cpu_parallel_regions_observe_identically_with_kernels() {
    let _serial = always();
    let source = program(5000);
    let mir = lower(&source);
    let full = schedule::run_source(source.as_bytes(), &mir, 10_000_000, 2).unwrap();
    let total = steps(&format!("{:?}", full.observation));
    let kinds = full
        .executed
        .iter()
        .map(|step| step.kind)
        .collect::<Vec<_>>();
    assert!(kinds.contains(&schedule::PlanKind::Map), "{kinds:?}");
    assert!(kinds.contains(&schedule::PlanKind::Reduction), "{kinds:?}");
    assert!(
        format!("{:?}", full.observation).contains("I32(1)"),
        "{:?}",
        full.observation
    );
    for fuel in [
        10_000_000,
        total,
        total - 1,
        total / 2,
        total * 3 / 4,
        total / 4,
    ] {
        for workers in [1, 2, 7] {
            region::set_policy(Some(Policy::Always));
            let before = region::calls();
            let compiled = schedule::run_source(source.as_bytes(), &mir, fuel, workers).unwrap();
            let used = region::calls() - before;
            region::set_policy(Some(Policy::Never));
            let interpreted = schedule::run_source(source.as_bytes(), &mir, fuel, workers).unwrap();
            assert_eq!(region::calls() - before, used, "kernels ran under Never");
            region::set_policy(Some(Policy::Always));
            if fuel >= total {
                // The map and the reduction each ran at least one kernel chunk.
                assert!(
                    used >= 2,
                    "fuel {fuel} workers {workers}: {used} kernel calls, {:?}",
                    region::last_error()
                );
            }
            assert_eq!(
                format!("{:?}", compiled.observation),
                format!("{:?}", interpreted.observation),
                "fuel {fuel} workers {workers}"
            );
            assert_eq!(compiled.chunks, interpreted.chunks);
            assert_eq!(
                compiled
                    .executed
                    .iter()
                    .map(|step| step.kind)
                    .collect::<Vec<_>>(),
                interpreted
                    .executed
                    .iter()
                    .map(|step| step.kind)
                    .collect::<Vec<_>>()
            );
        }
    }
}

/// Checked arithmetic is never a kernel: an overflowing checked map traps in
/// source order with kernels on, exactly as with them off.
#[test]
fn issue_eng_18_checked_overflow_traps_identically_with_kernels() {
    let _serial = always();
    let source = "fn main() -> i32 {
    var xs: Array<u32> = filled_u32(4096u32, 1u32);
    xs[4000u32] = 4294967295u32;
    var ys: Array<u32> = filled_u32(len(xs), 0u32);
    for i in 0u32 .. len(xs) { ys[i] = xs[i] + 1u32; }
    return 0;
}";
    let mir = lower(source);
    let compiled = schedule::run_source(source.as_bytes(), &mir, 1_000_000, 4).unwrap();
    region::set_policy(Some(Policy::Never));
    let interpreted = schedule::run_source(source.as_bytes(), &mir, 1_000_000, 4).unwrap();
    region::set_policy(Some(Policy::Always));
    let text = format!("{:?}", compiled.observation);
    assert_eq!(text, format!("{:?}", interpreted.observation));
    assert!(text.contains("IntegerOverflow"), "{text}");
}
