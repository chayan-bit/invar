use super::*;
use crate::mir_eval::{self, LoopOverride};
const CAPS: Capabilities = Capabilities::Metal {
    apple_family: 9,
    msl_major: 3,
    msl_minor: 1,
};
const GATHER: &str = "fn work(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out = filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i] = a[ix[i]]; } return out; }";
const SCATTER: &str = "fn work(a: read Array<u32>, ix: read Array<u32>, n: u32) -> Array<u32> { var out = filled_u32(n,0u32); for i in 0u32..len(ix) { out[ix[i]] = a[i]; } return out; }";
const HIST: &str = "fn work(a: read Array<u32>, ix: read Array<u32>, n: u32) -> Array<u32> { var out = filled_u32(n,0u32); for i in 0u32..len(ix) { out[ix[i]] = wrapping_add(out[ix[i]],a[i]); } return out; }";
const AFFINE: &str = "fn work(a: read Array<u32>) -> Array<u32> { var out = filled_u32(5u32,0u32); for i in 0u32..3u32 { out[2u32*i] = a[i]; } return out; }";
const MAIN: &str = "fn main() -> i32 { let a: Array<u32> = [4294967295u32,1u32,7u32]; let ix: Array<u32> = [2u32,0u32,1u32]; let out = work(a,ix); if out[0u32] == 7u32 and out[1u32] == 4294967295u32 and out[2u32] == 1u32 { return 7; } return 0; }";
fn lower(s: &[u8]) -> mir::Program {
    mir::lower(&check::check(front::parse(s).unwrap(), true).unwrap()).unwrap()
}
fn plan(s: &[u8], p: &mir::Program) -> SourcePlan {
    let f = p
        .functions
        .iter()
        .find(|f| f.name == "work" || f.name == "reach")
        .unwrap();
    f.blocks
        .iter()
        .find_map(|b| match b.end {
            Some(mir::Terminator::Branch { span, .. }) => {
                SourcePlan::recognize(s, p, &f.name, span).unwrap()
            }
            _ => None,
        })
        .unwrap()
}
fn receipt(p: &Prepared) -> device::Receipt {
    device::Receipt {
        target: "metal",
        reason: "verified-u32-completed",
        output: device::Output {
            values: p.expected.values.clone(),
            frontier_masks: p.expected.frontier_masks.clone(),
            compaction_maps: p.expected.compaction_maps.clone(),
        },
        source_hash: invar_device::source_hash(&device::emit_msl(p.bound.plan()).unwrap()),
        transfer_bytes: p.bound.transfer_bytes(),
        transfer_mode: "shared-memory-host-copy",
        completed_phases: device::schedule(p.bound.plan()).len() as u32,
    }
}
fn compare(source: &str, mutation: usize) -> usize {
    let s = source.as_bytes();
    let p = lower(s);
    let plan = plan(s, &p);
    let serial = mir_eval::observe(&p, 1_000_000);
    let mut submissions = 0;
    let mut provider = |live: &LiveLoop<'_>| {
        if live.function != plan.function || live.span != plan.region.span {
            return Ok(None);
        }
        let prepared = plan.prepare(live, CAPS).unwrap();
        submissions += 1;
        let mut completed = receipt(&prepared);
        let hash = completed.source_hash;
        if mutation == 1 {
            completed.output.values[0] ^= 1;
        }
        if mutation == 3 {
            completed.completed_phases = 0;
        }
        if mutation == 4 {
            completed.output.frontier_masks.swap(0, 1);
        }
        let Ok(mut region) = prepared.complete(completed, "metal", hash) else {
            return Ok(None);
        };
        if mutation == 2 {
            region.branches[0].taken -= 1;
        }
        Ok(Some(LoopOverride::completed(region)))
    };
    let (observed, applied) = mir_eval::observe_with_scheduler(&p, 1_000_000, Some(&mut provider));
    assert_eq!(submissions, 1);
    assert_eq!(observed.result, serial.result);
    assert_eq!(observed.steps_used, serial.steps_used);
    assert_eq!(observed.cleanup_trace, serial.cleanup_trace);
    assert_eq!(applied, usize::from(mutation == 0));
    applied
}
#[test]
fn a1_indexed_all_forms_completed_regions_match_serial_and_mutants_refuse() {
    let cases = [
        format!("{GATHER}{MAIN}"),
        format!("{SCATTER}{}", MAIN.replace("work(a,ix)", "work(a,ix,3u32)")),
        format!("{HIST}{}", MAIN.replace("work(a,ix)", "work(a,ix,3u32)")),
        format!(
            "{HIST}fn main() -> i32 {{ let a: Array<u32> = [4294967295u32,1u32]; let ix: Array<u32> = [0u32,0u32]; let out = work(a,ix,1u32); if out[0u32] == 0u32 {{ return 7; }} return 0; }}"
        ),
        format!(
            "{AFFINE}fn main() -> i32 {{ let a: Array<u32> = [7u32,11u32,19u32]; let out=work(a); if out[4u32] == 19u32 {{ return 7; }} return 0; }}"
        ),
        include_str!("../../bench/general-net-v01/current/frontier_graph_a.inv").into(),
        include_str!("../../bench/general-net-v01/current/frontier_graph_b.inv").into(),
    ];
    for s in cases {
        for mutation in 0..4 {
            compare(&s, mutation);
        }
    }
    compare(
        include_str!("../../bench/general-net-v01/current/frontier_graph_a.inv"),
        4,
    );
}
#[test]
#[rustfmt::skip]
fn a1_indexed_guards_fail_before_submission_and_source_limits_remain() {
    for (family, main) in [(GATHER, MAIN.replace("[2u32,0u32,1u32]", "[3u32,4u32,1u32]")),
        (SCATTER, MAIN.replace("work(a,ix)", "work(a,ix,3u32)").replace("[2u32,0u32,1u32]", "[0u32,0u32,1u32]")),
        (HIST, MAIN.replace("work(a,ix)", "work(a,ix,1u32)"))] {
        let s = format!("{family}{main}"); let p=lower(s.as_bytes()); let plan=plan(s.as_bytes(), &p); let mut reached=0;
        let mut provider = |live: &LiveLoop<'_>| { if live.function == plan.function && live.span == plan.region.span {reached+=1; assert!(plan.prepare(live, CAPS).is_err());} Ok(None) };
        let (observed, applied)=mir_eval::observe_with_scheduler(&p,1_000_000,Some(&mut provider)); let serial=mir_eval::observe(&p,1_000_000);
        assert_eq!(reached,1); assert_eq!(applied,0); assert_eq!(observed.result,serial.result); assert_eq!(observed.steps_used,serial.steps_used);
    }
    let p = Plan::from_source(GATHER.as_bytes(), "work").unwrap();
    assert!(matches!(
        p.bind(&[1], &[0, 2, 3], None, CAPS),
        Err(Error::Device(device::Error::Index { iteration: 1, .. }))
    ));
    let p = Plan::from_source(SCATTER.as_bytes(), "work").unwrap();
    assert!(matches!(
        p.bind(&[1, 2], &[0, 0], Some(2), CAPS),
        Err(Error::Device(device::Error::Duplicate { iteration: 1, .. }))
    ));
}
#[test]
fn a1_indexed_cost_uses_bytes_work_and_phases() {
    let s = format!("{GATHER}{MAIN}");
    let p = lower(s.as_bytes());
    let plan = plan(s.as_bytes(), &p);
    let mut tested = false;
    let mut provider = |live: &LiveLoop<'_>| {
        if live.function == plan.function && live.span == plan.region.span {
            let ready = plan.prepare(live, CAPS).unwrap();
            let model = crate::plan::DeviceCostModel {
                cpu_bytes_per_ms: 1.,
                device_bytes_per_ms: 100.,
                device_fixed_ms: 0.,
            };
            assert_eq!(
                ready.cost(model),
                Some(model.estimate(ready.bound.transfer_bytes(), 3, 1, 1))
            );
            assert!(ready.cost(model).unwrap().choose_device);
            assert!(
                ready
                    .cost(crate::plan::DeviceCostModel {
                        cpu_bytes_per_ms: f64::NAN,
                        ..model
                    })
                    .is_none()
            );
            assert!(
                !ready
                    .cost(crate::plan::DeviceCostModel {
                        cpu_bytes_per_ms: 1e12,
                        device_bytes_per_ms: 1.,
                        device_fixed_ms: 10.
                    })
                    .unwrap()
                    .choose_device
            );
            tested = true;
        }
        Ok(None)
    };
    mir_eval::observe_with_scheduler(&p, 1_000_000, Some(&mut provider));
    assert!(tested);
}
