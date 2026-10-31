//! Issue #5 (M4-05).
#![forbid(unsafe_code)]
use invar::{
    check, eval, front, general_net,
    histogram_source::{Budget, host},
    mir, mir_eval,
};
const HIST: &str = "fn h(a: read Array<u32>, x: read Array<u32>, n: u32) -> Array<u32> { var o = filled_u32(n,0u32); for i in 0u32..len(x) { o[x[i]] = wrapping_add(o[x[i]],a[i]); } return o; }";
fn compare(a: &eval::Observation, b: &eval::Observation) {
    assert_eq!(a.result, b.result);
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace);
}
fn source(hist: &str, a: &str, x: Option<&str>, n: u32) -> String {
    let ix = x.map_or(String::new(), |x| format!("let x: Array<u32> = {x};"));
    format!(
        "{hist} fn main() -> i32 {{let a: Array<u32> = {a}; {ix} let o = h(a,{}, {n}u32); return 7;}}",
        if x.is_some() { "x" } else { "a" }
    )
}
fn prefixes(s: &str, budget: Budget, admitted: bool) -> eval::Observation {
    let c = check::check(front::parse(s.as_bytes()).unwrap(), true).unwrap();
    let m = mir::lower(&c).unwrap();
    let net = general_net::Net::lower(&c, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let done = eval::observe(&c, 10000);
    let mut published = 0;
    for fuel in 0..=done.steps_used + 1 {
        let reference = eval::observe(&c, fuel);
        let run = host::run_source(s.as_bytes(), fuel, budget).unwrap();
        compare(&run.observation, &reference);
        compare(&mir_eval::observe(&m, fuel), &reference);
        let oracle = general_net::oracle::observe(
            &net,
            &proof,
            general_net::oracle::Limits {
                fuel,
                ..Default::default()
            },
        )
        .unwrap();
        compare(&oracle.semantic, &reference);
        assert_eq!(run.begun, run.completed);
        assert!(run.pending.is_none());
        assert!(run.visited <= run.reserved_visits);
        assert!(run.reserved_visits <= budget.visits.min(65536));
        published += run.completed;
    }
    let run = host::run_source(s.as_bytes(), 10000, budget).unwrap();
    println!(
        "prefixes={} publications={published} full begun={} completed={} reserved={} visited={} refusals={:?}",
        done.steps_used + 2,
        run.begun,
        run.completed,
        run.reserved_visits,
        run.visited,
        run.refusals
    );
    assert_eq!(run.completed > 0, admitted);
    done
}
#[test]
fn gather_scatter_frontier_histogram_runtime_source_prefixes() {
    for (a, x, n, admitted) in [
        ("[]", None, 0, false),
        ("[0u32]", None, 1, true),
        ("[0u32,1u32]", None, 3, true),
        ("[4294967295u32,2u32]", Some("[0u32,0u32]"), 3, false),
    ] {
        assert_eq!(
            prefixes(&source(HIST, a, x, n), Budget::default(), admitted).result,
            Ok(eval::Value::I32(7))
        );
    }
}
#[test]
fn gather_scatter_frontier_histogram_runtime_source_trap_allocation_unknown_law_prefixes() {
    for s in [
        source(HIST, "[]", Some("[0u32]"), 1),
        source(HIST, "[7u32]", Some("[1u32]"), 1),
        source(HIST, "[7u32]", Some("[0u32]"), 0),
        source(HIST, "[7u32]", Some("[0u32,0u32]"), 1),
        source(
            HIST,
            "[0u32]",
            None,
            invar::array_limits::MAX_ARRAY_LEN as u32 + 1,
        ),
    ] {
        let done = prefixes(&s, Budget::default(), false);
        assert!(done.result.is_err());
    }
    for changed in [
        HIST.replace("wrapping_add(o[x[i]],a[i])", "o[x[i]]+a[i]"),
        HIST.replace("wrapping_add", "wrapping_mul"),
        HIST.replace("filled_u32(n,0u32)", "filled_u32(n,1u32)"),
    ] {
        assert_eq!(
            prefixes(
                &source(&changed, "[0u32]", None, 1),
                Budget::default(),
                false
            )
            .result,
            Ok(eval::Value::I32(7))
        );
    }
    let twice = format!(
        "{HIST} fn main()->i32 {{let a: Array<u32> = [0u32]; let o=h(a,a,1u32); let q=h(a,a,1u32); return 7;}}"
    );
    assert_eq!(
        prefixes(&twice, Budget::default(), false).result,
        Ok(eval::Value::I32(7))
    );
}

/// The canonical 252-byte ordinary-main histogram program. History: under the
/// reviewed owner-return fuel rule its proof grew by 624 visits and it was
/// refused pre-entry (65,565 > 65,536; e760804 pinned that refusal). With the
/// entry-sized recognition reserve (variant B) the same cap admits it again.
/// This test requires at least the pre-fuel 624-visit headroom, so the next
/// equivalent growth fails loudly here instead of silently refusing.
#[test]
fn gather_scatter_frontier_histogram_runtime_canonical_main_admitted_with_headroom() {
    let s = source(HIST, "[0u32]", None, 1);
    assert_eq!(s.len(), 252);
    prefixes(&s, Budget::default(), true);
    let run = host::run_source(s.as_bytes(), 10000, Budget::default()).unwrap();
    assert_eq!((run.begun, run.completed), (1, 1));
    assert!(run.refusals.is_empty(), "{:?}", run.refusals);
    println!(
        "canonical reserved={} headroom={}",
        run.reserved_visits,
        65536 - run.reserved_visits
    );
    assert!(
        65536 - run.reserved_visits >= 624,
        "reserved {}",
        run.reserved_visits
    );
}

/// Explicit pre-entry refusal regression for a context that stays over the
/// frozen budget: two contributions (len(x) = 2) into three bins. It is refused before
/// entry, only the precheck stays reserved, nothing is begun, and every fuel
/// prefix still matches the reference interpreter exactly.
#[test]
fn gather_scatter_frontier_histogram_runtime_over_cap_context_refused_before_entry() {
    let s = source(HIST, "[4294967295u32,2u32]", Some("[0u32,0u32]"), 3);
    let c = check::check(front::parse(s.as_bytes()).unwrap(), true).unwrap();
    let done = eval::observe(&c, 10000);
    assert_eq!(done.result, Ok(eval::Value::I32(7)));
    for fuel in 0..=done.steps_used + 1 {
        let run = host::run_source(s.as_bytes(), fuel, Budget::default()).unwrap();
        compare(&run.observation, &eval::observe(&c, fuel));
        assert_eq!((run.begun, run.completed), (0, 0), "fuel {fuel}");
        assert!(run.pending.is_none());
    }
    let run = host::run_source(s.as_bytes(), 10000, Budget::default()).unwrap();
    assert_eq!(run.refusals, ["histogram_complete_budget"]);
    assert_eq!((run.begun, run.completed), (0, 0));
    assert_eq!((run.reserved_visits, run.visited), (79, 79));
}
