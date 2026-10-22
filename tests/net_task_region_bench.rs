//! P4b D5 residual-tier bench on Metal (ignored): the emitted task region T
//! against handwritten B-form kernels of the C1 fork-join family, both
//! checked against the CPU net oracle on every warmup and timed call.
//!
//! 1. `pilot`: freeze the handwritten replay depth per program and input
//!    (fastest median resident of five) and calibrate T's expansion width
//!    on the C1 tuning split; writes `pilot-metal.json`.
//! 2. `session`: one scored session (`P4B_SESSION` = 0, 1, 2) in a fresh
//!    process: 10 warmup and 30 interleaved pairs per program and input.
//! 3. `report`: median of the 90 paired ratios and the hierarchical
//!    bootstrap U95, per cell and scope, against the D5 residual limits.
//!
//! `P4B_OUT` names the output directory. The scored backend `metal` is the
//! probed path (the MSL 3.2 cascade on Apple M4); `P4B_METAL_PATH=staged`
//! runs the MSL 3.1 staged fallback as backend `metal-staged` (informational).
#![cfg(all(feature = "metal", target_os = "macos"))]
#![forbid(unsafe_code)]

#[path = "support/task_bench.rs"]
mod bench;
#[path = "support/task_model.rs"]
mod model;

use std::{cell::RefCell, rc::Rc};

use bench::{Backend, HwCall, TCall, metal_baseline};
use invar::general_net::task::{
    Mutation, Profile,
    host::metal::{self as host, MetalTask},
};
use invar_metal::MetalDevice;

fn with_backend(run: impl FnOnce(Backend)) {
    let device = MetalDevice::open_runtime_source().expect("Metal device");
    let staged = std::env::var("P4B_METAL_PATH").is_ok_and(|p| p == "staged");
    let (name, profile) = if staged {
        ("metal-staged", Profile::METAL_STAGED)
    } else {
        ("metal", host::profile(&device))
    };
    eprintln!("bench backend {name}: {:?}", profile.schedule);
    let metal = Rc::new(RefCell::new(device));
    let (m1, m2) = (metal.clone(), metal);
    let mut t = move |program: &_, plan: &_| -> TCall {
        let mut task = MetalTask::prepare(&mut m1.borrow_mut(), program, plan, Mutation::None)
            .expect("prepare task");
        Box::new(move |args| task.call(args))
    };
    let mut hw = move |index: usize| -> HwCall {
        let mut work = m2
            .borrow_mut()
            .prepare_baseline(
                &metal_baseline(index),
                "hw_b",
                "hw_top",
                [16, 256 * 8, 256 * 8, 256 * 80 * 4, 8],
            )
            .unwrap_or_else(|e| panic!("baseline {index}: {e:?}"));
        Box::new(move |params| work.call(params).expect("baseline call"))
    };
    run(Backend {
        name,
        profile,
        t: &mut t,
        hw: &mut hw,
    });
}

#[test]
#[ignore = "Metal bench; set P4B_OUT"]
fn pilot() {
    model::with_stack(|| with_backend(bench::pilot));
}

#[test]
#[ignore = "Metal bench; set P4B_OUT and P4B_SESSION"]
fn session() {
    let session = std::env::var("P4B_SESSION")
        .expect("P4B_SESSION")
        .parse()
        .unwrap();
    model::with_stack(move || with_backend(|backend| bench::session(backend, session)));
}

#[test]
#[ignore = "Metal bench report; set P4B_OUT"]
fn report() {
    let staged = std::env::var("P4B_METAL_PATH").is_ok_and(|p| p == "staged");
    bench::report(if staged { "metal-staged" } else { "metal" });
}
