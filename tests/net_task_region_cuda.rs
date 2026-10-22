//! P4b on CUDA (ignored by default): the emitted task region (scheduler and
//! run kernels) on the actual device against the CPU net oracle, over the C1
//! fork-join family and generated trap-heavy fork-joins, and the D5 bench
//! against handwritten B-form kernels (`pilot`, `session`, `report`; see
//! `tests/net_task_region_bench.rs`). Run with
//! `cargo test --features cuda --test net_task_region_cuda -- --ignored`.
#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

#[path = "support/task_bench.rs"]
mod bench;
#[path = "support/task_device.rs"]
mod device;
#[path = "support/task_model.rs"]
mod model;

use std::{cell::RefCell, path::Path, rc::Rc};

use bench::{Backend, HwCall, TCall, cuda_baseline};
use device::{Tally, device_differential, device_plans, print};
use invar::eval::Value;
use invar::general_net::task::{self, Mutation, Profile, host::cuda::CudaTask};
use invar_cuda::cuda::CudaDevice;
use model::{Rng, admit, c1, c1_cases, generated, generated_cases};

fn compiler() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}

#[test]
#[ignore = "requires a CUDA device"]
fn c1_and_generated_fork_joins_agree_with_the_net_oracle_on_cuda() {
    model::with_stack(|| {
        let mut cuda = CudaDevice::open().expect("CUDA device");
        let mut total = Tally::default();
        let mut rng = Rng(0x0c1_c0da);
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
        for (name, admitted, cases) in &sources {
            let wide = task::plan(&admitted.program, &Profile::CUDA);
            for (label, plan) in device_plans(wide) {
                let mut run = CudaTask::prepare(
                    &mut cuda,
                    &admitted.program,
                    &plan,
                    Mutation::None,
                    compiler(),
                )
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
        print("CUDA", &total);
        assert!(total.mismatches.is_empty(), "{:#?}", total.mismatches);
        assert!(
            total.cases >= 3_000 && total.traps > 0 && total.depth_limits > 0 && total.serial > 0
        );
    });
}

/// Review finding (#207): idle cascade workers counted their own polls and
/// marked the call stalled while other lanes were still running long serial
/// nodes. With a 1-poll limit and two long leaves, the call must still return
/// the reference value.
#[test]
#[ignore = "requires a CUDA device"]
fn cascade_idle_workers_wait_for_running_lanes_on_cuda() {
    model::with_stack(|| {
        let mut cuda = CudaDevice::open().expect("CUDA device");
        let admitted = admit(model::LONG_LEAVES).expect("admit long leaves");
        let mut plan = task::plan(&admitted.program, &Profile::CUDA);
        // One poll: idle workers may count polls only when nothing is pending,
        // which before the root finishes never happens, so the gated kernel
        // must pass at the smallest limit, and a kernel without the pending
        // gate stalls on its first unpublished claim.
        plan.poll_limit = 1;
        let mut run = CudaTask::prepare(
            &mut cuda,
            &admitted.program,
            &plan,
            Mutation::None,
            compiler(),
        )
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

fn with_backend(run: impl FnOnce(Backend)) {
    let cuda = Rc::new(RefCell::new(CudaDevice::open().expect("CUDA device")));
    let (c1, c2) = (cuda.clone(), cuda);
    let mut t = move |program: &_, plan: &_| -> TCall {
        let mut task = CudaTask::prepare(
            &mut c1.borrow_mut(),
            program,
            plan,
            Mutation::None,
            compiler(),
        )
        .expect("prepare task");
        Box::new(move |args| task.call(args))
    };
    let mut hw = move |index: usize| -> HwCall {
        let mut work = c2
            .borrow_mut()
            .prepare_baseline(
                &cuda_baseline(index),
                "hw_b",
                "hw_top",
                [4, 256 * 2, 256 * 2, 256 * 80, 2],
                compiler(),
            )
            .unwrap_or_else(|e| panic!("baseline {index}: {e:?}"));
        Box::new(move |params| work.call(params).expect("baseline call"))
    };
    run(Backend {
        name: "cuda",
        profile: Profile::CUDA,
        t: &mut t,
        hw: &mut hw,
    });
}

#[test]
#[ignore = "CUDA bench; set P4B_OUT"]
fn pilot() {
    model::with_stack(|| with_backend(bench::pilot));
}

#[test]
#[ignore = "CUDA bench; set P4B_OUT and P4B_SESSION"]
fn session() {
    let session = std::env::var("P4B_SESSION")
        .expect("P4B_SESSION")
        .parse()
        .unwrap();
    model::with_stack(move || with_backend(|backend| bench::session(backend, session)));
}

#[test]
#[ignore = "CUDA bench report; set P4B_OUT"]
fn report() {
    bench::report("cuda");
}
