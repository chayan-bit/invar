//! #57 part B: admission is bounded by byte budgets (region storage, scheduler peak), not by region counts.

use invar::{
    admission,
    budget::{ByteBudget, Resource},
    check::check,
    front::parse,
    laws, mir,
};

const MAP: &str = r#"
fn main() -> i32 {
    let xs: Array<u32> = [1u32, 2u32, 3u32];
    var ys: Array<u32> = filled_u32(len(xs), 0u32);
    for i in 0u32 .. len(xs) {
        ys[i] = wrapping_add(xs[i], 7u32);
    }
    return 0;
}
"#;

fn lowered(source: &str) -> mir::Program {
    mir::lower(&check(parse(source.as_bytes()).unwrap(), true).unwrap()).unwrap()
}

fn guard(source: &str) -> admission::Guard {
    let report = admission::analyze(source.as_bytes(), &lowered(source)).unwrap();
    report.regions()[0].guard().unwrap().clone()
}

/// `functions` functions of `per_function` empty loops each, so no single function nears MIR's
/// per-function block limit while the total passes the retired 4096-region cap.
fn empty_loops(functions: usize, per_function: usize) -> String {
    let mut source = String::new();
    for f in 0..functions {
        let body: String = (0..per_function)
            .map(|i| format!("for idx_{i} in 0u32 .. 0u32 {{}} "))
            .collect();
        source.push_str(&format!("fn f{f}() -> Unit {{ {body} return; }}\n"));
    }
    source.push_str("fn main() -> i32 { return 0; }");
    source
}

#[test]
fn issue_eng_08_admission_footprint_counts_the_scheduler_peak() {
    let guard = guard(MAP);
    assert_eq!(guard.inputs().len(), 1);
    // The input, the destination, the chunk results and the joined values: four u32 arrays.
    assert_eq!(guard.footprint(1000).bytes(Resource::Host), Some(16_000));
    assert_eq!(guard.footprint(usize::MAX).bytes(Resource::Host), None);
    assert_eq!(guard.footprint(0).bytes(Resource::Host), Some(0));
}

#[test]
fn issue_eng_08_admission_guard_follows_the_host_budget() {
    let guard = guard(MAP);
    let lengths = [("xs", 1000), ("ys", 1000)];
    let exact = ByteBudget::standard().with_host_bytes(16_000);
    assert!(guard.evaluate_within(&lengths, &exact));
    assert_eq!(
        guard.select_plan_within(&lengths, &exact),
        "wrapping-map-candidate"
    );
    assert!(guard.covers_whole_arrays_within(&lengths, &exact));
    let tight = ByteBudget::standard().with_host_bytes(15_999);
    assert!(!guard.evaluate_within(&lengths, &tight));
    assert_eq!(guard.select_plan_within(&lengths, &tight), "mir-serial");
    assert!(!guard.covers_whole_arrays_within(&lengths, &tight));
    // The guard admits what the scheduler executes: up to `laws::MAX_ELEMENTS`, never past it,
    // however large the budget (the scheduler and the device IR refuse a longer map).
    let max = laws::MAX_ELEMENTS;
    let roomy = ByteBudget::standard().with_host_bytes(usize::MAX / 2);
    assert!(guard.evaluate_within(&[("xs", max), ("ys", max)], &roomy));
    assert!(!guard.evaluate_within(&[("xs", max + 1), ("ys", max + 1)], &roomy));
    assert!(!guard.evaluate_within(&[("xs", max), ("ys", max)], &ByteBudget::ZERO));
}

/// The least analysis budget under which `source` analyzes, found by bisection (admission is
/// monotone in the budget).
fn least_analysis_bytes(source: &str) -> usize {
    let program = lowered(source);
    let admits = |bytes: usize| {
        let budget = ByteBudget::standard().with_analysis_bytes(bytes);
        admission::analyze_within(source.as_bytes(), &program, &budget).is_ok()
    };
    let (mut low, mut high) = (0usize, 1usize << 30);
    assert!(admits(high) && !admits(low));
    while high - low > 1 {
        let mid = low + (high - low) / 2;
        if admits(mid) {
            high = mid;
        } else {
            low = mid;
        }
    }
    high
}

#[test]
fn issue_eng_08_admission_region_count_is_bounded_by_analysis_bytes() {
    let one = empty_loops(1, 1);
    let three = empty_loops(1, 3);
    let (small, large) = (least_analysis_bytes(&one), least_analysis_bytes(&three));
    assert!(small > 0 && large > small, "{small} {large}");
    // The budget that holds one region refuses three, with the typed budget error.
    let budget = ByteBudget::standard().with_analysis_bytes(small);
    let error = admission::analyze_within(three.as_bytes(), &lowered(&three), &budget)
        .expect_err("one region of budget cannot hold three");
    assert_eq!(error.code, "RESOURCE_LIMIT");
    assert!(
        error.message.contains("BUDGET_EXCEEDED"),
        "{}",
        error.message
    );
    assert!(error.message.contains("analysis"), "{}", error.message);
    // Zero bytes admit no region at all, but a source with no loop region needs no storage.
    assert!(admission::analyze_within(one.as_bytes(), &lowered(&one), &ByteBudget::ZERO).is_err());
    let flat = "fn main() -> i32 { return 0; }";
    admission::analyze_within(flat.as_bytes(), &lowered(flat), &ByteBudget::ZERO).unwrap();
}

#[test]
fn issue_eng_08_admission_has_no_region_count_cap() {
    // 4800 regions used to trip ADMISSION_REGION_LIMIT at 4096. The byte budget admits them; the
    // serialized report (MAX_RECORD_BYTES, an interchange-format bound) is what now ends it.
    let source = empty_loops(120, 40);
    let program = lowered(&source);
    let roomy = ByteBudget::standard().with_analysis_bytes(256 * 1024 * 1024);
    match admission::analyze_within(source.as_bytes(), &program, &roomy) {
        Ok(report) => assert_eq!(report.regions().len(), 4800),
        Err(error) => assert_eq!(error.code, "ADMISSION_RECORD_LIMIT"),
    }
}
