//! P4b on Metal (ignored by default): the emitted task region (scheduler and
//! run kernels) on the actual device against the CPU net oracle, over the C1
//! fork-join family and generated trap-heavy fork-joins. Run with
//! `cargo test --features metal --test gpu_metal task_region:: -- --ignored`.
#![forbid(unsafe_code)]

#[path = "../support/task_device.rs"]
mod device;
#[path = "../support/task_model.rs"]
mod model;

use device::{Tally, device_differential, device_plans, print};
use invar::eval::Value;
use invar::general_net::task::{
    self, Mutation, Profile,
    host::metal::{self as host, MetalTask},
};
use invar_metal::MetalDevice;
use model::{Rng, admit, c1, c1_cases, generated, generated_cases};

#[test]
#[ignore = "requires an Apple Metal device"]
fn c1_and_generated_fork_joins_agree_with_the_net_oracle_on_metal() {
    model::with_stack(|| {
        let mut metal = MetalDevice::open_runtime_source().expect("Metal device");
        let mut total = Tally::default();
        let mut rng = Rng(0x0c1_3e7a);
        let mut sources: Vec<(String, model::Admitted, Vec<Vec<i32>>)> = (0..12)
            .map(|i| (format!("c1-{i:02}"), c1(i), c1_cases(i, &mut rng, 60)))
            .collect();
        for g in 0..16 {
            let source = generated(&mut rng);
            let admitted = admit(&source).expect("admit generated");
            sources.push((
                format!("gen-{g:02}"),
                admitted,
                generated_cases(&mut rng, 80),
            ));
        }
        // The probed path (MSL 3.2 cascade on Apple M4) and, always, the
        // MSL 3.1 staged fallback.
        let probed = host::profile(&metal);
        eprintln!("Metal probe: {:?}", probed.schedule);
        let mut profiles = vec![("probed", probed)];
        if probed != Profile::METAL_STAGED {
            profiles.push(("fallback", Profile::METAL_STAGED));
        }
        for (name, admitted, cases) in &sources {
            let plans = profiles.iter().flat_map(|(path, profile)| {
                device_plans(task::plan(&admitted.program, profile))
                    .map(move |(label, plan)| (format!("{path}-{label}"), plan))
            });
            for (label, plan) in plans {
                let mut run =
                    MetalTask::prepare(&mut metal, &admitted.program, &plan, Mutation::None)
                        .unwrap_or_else(|e| panic!("{name}-{label}: prepare {e}"));
                device_differential(
                    &format!("{name}-{label}"),
                    admitted,
                    cases,
                    &mut |args| run.call(args),
                    &mut total,
                );
            }
        }
        print("Metal", &total);
        assert!(total.mismatches.is_empty(), "{:#?}", total.mismatches);
        assert!(
            total.cases >= 3_000
                && total.traps > 0
                && total.depth_limits > 0
                && total.multi_submission > 0
        );
        if probed.schedule == task::Schedule::Cascade {
            assert!(total.serial > 0 && total.paths.len() == 2);
        }
    });
}

/// Review findings (P4b cascade liveness): idle cascade workers wait while any
/// node is pending (1-poll limit, two long leaves), and the
/// measured device class (this Apple M4) still selects the cascade.
#[test]
#[ignore = "requires an Apple Metal device"]
fn cascade_idle_workers_wait_for_running_lanes_on_metal() {
    model::with_stack(|| {
        let mut metal = MetalDevice::open_runtime_source().expect("Metal device");
        if !metal.supports_msl_3_2() {
            eprintln!("no MSL 3.2: the cascade is not offered on this device");
            return;
        }
        assert_eq!(
            host::profile(&metal) == Profile::METAL,
            task::cost::metal_cascade_measured(&metal.name()),
            "{}",
            metal.name()
        );
        if !task::cost::metal_cascade_measured(&metal.name()) {
            // Review finding: outside the measured class the cascade is not
            // offered, so it must not be forced here.
            eprintln!(
                "{}: not the measured cascade class, staged schedule only",
                metal.name()
            );
            return;
        }
        let admitted = admit(model::LONG_LEAVES).expect("admit long leaves");
        let mut plan = task::plan(&admitted.program, &Profile::METAL);
        // One poll: idle workers may count polls only when nothing is pending,
        // which before the root finishes never happens, so the gated kernel
        // must pass at the smallest limit, and a kernel without the pending
        // gate stalls on its first unpublished claim.
        plan.poll_limit = 1;
        let mut run = MetalTask::prepare(&mut metal, &admitted.program, &plan, Mutation::None)
            .expect("prepare");
        for (n, s) in [(1, 5), (1, 77)] {
            let args = [Value::I32(n), Value::I32(s), Value::I32(model::LONG_STEPS)];
            let outcome = run
                .call(&args)
                .unwrap_or_else(|e| panic!("long leaves ({n}, {s}): {e}"));
            let want = model::long_leaves_expected(n, s, model::LONG_STEPS);
            assert!(
                matches!(outcome.result, Ok(Value::I32(v)) if v == want),
                "({n}, {s}): {:?}, want {want}",
                outcome.result
            );
        }
    });
}
