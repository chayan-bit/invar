//! P5b exactness checks: reachability, numeric helpers and returned owners against the net oracle.

use crate::entry;

use invar::general_net::{
    compiled::Target,
    device_host::{Limits, Plan},
};

const REACHABLE: &str = "
fn unused(x:i32)->i32{return x / 912341;}
fn even(x:i32)->i32{if x==0{return 7;} return odd(x-1);}
fn odd(x:i32)->i32{if x==0{return 9;} return even(x-1);}
fn run(x:i32,d:i32)->i32{
    if x<0 {let a:i32=1/d; return a+(2147483647-x);}
    if x>20 {return (2147483647+x)+(1/d);}
    return even(x);
}
fn main()->i32{return unused(912342);}";

#[test]
fn call_closure_keeps_recursion_both_branches_and_first_trap() {
    use invar::{eval::Value, general_net::device_host::cpu::CpuModel};
    let mut cases = 0;
    for target in [Target::Metal, Target::Cuda] {
        let plan = Plan::from_source(REACHABLE.as_bytes(), "run", target, entry::limits()).unwrap();
        assert!(plan.artifact().emitted_states() < plan.artifact().states());
        assert!(!plan.artifact().source().contains("gnet_alloc("));
        assert!(
            !plan.artifact().source().contains("912341"),
            "unreachable function"
        );
        assert!(
            !plan.artifact().source().contains("912342"),
            "uncalled main"
        );
        let mut backend = CpuModel::default();
        for (x, d) in [(-1, 0), (21, 0), (3, 1), (20, 1)] {
            entry::agree(
                &mut backend,
                target,
                REACHABLE,
                "run",
                vec![Value::I32(x), Value::I32(d)],
                100000,
            );
            cases += 1;
        }
    }
    assert_eq!(cases, 8);
}

#[test]
fn numeric_helpers_keep_only_the_transitive_dependencies() {
    let source = b"fn run(x:f32)->f32{return x+1.0;} fn unused(x:f32)->f32{return x/2.0;} fn main()->i32{return 0;}";
    for target in [Target::Metal, Target::Cuda] {
        let plan = Plan::from_source(source, "run", target, entry::limits()).unwrap();
        let emitted = plan.artifact().source();
        assert!(emitted.contains("gnet_f32_add("));
        assert!(emitted.contains("gnet_f32_round("));
        assert!(!emitted.contains("gnet_f32_div("));
        assert!(!emitted.contains("gnet_f32_fma("));
        assert!(!emitted.contains("gnet_f32_from_i32("));
    }
}

#[test]
fn zero_child_slots_omit_unusable_projections() {
    let source =
        b"fn f(x:i32)->i32{return x+1;} fn run(x:i32)->i32{let a=f(x);let b=f(x+2);return a+b;}";
    for target in [Target::Metal, Target::Cuda] {
        let enabled = Plan::from_source(source, "run", target, Limits::default()).unwrap();
        assert!(enabled.artifact().source().contains("gnet_project_"));
        let disabled = Plan::from_source(source, "run", target, entry::limits()).unwrap();
        assert!(!disabled.artifact().source().contains("gnet_project_"));
        assert_eq!(disabled.artifact().states(), enabled.artifact().states());
        assert!(disabled.artifact().source().len() < enabled.artifact().source().len());
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "physical Metal qualification"]
fn metal_reachability_and_numeric_helpers_match_oracle() {
    use invar::{eval::Value, general_net::device_host::metal::Metal};
    let mut backend = Metal::open_runtime_source().unwrap();
    entry::corpus(&mut backend, Target::Metal);
    returned_owner_cases(&mut backend, Target::Metal);
    for x in [-2147483648, -21, -1, 0, 1, 2, 3, 7, 20, 21, 2147483647] {
        for d in [-1, 0, 1] {
            entry::agree(
                &mut backend,
                Target::Metal,
                REACHABLE,
                "run",
                vec![Value::I32(x), Value::I32(d)],
                100000,
            );
        }
    }
    println!("P5B Metal: 33 call-closure cases plus 15 ABI13 entry/numeric cases");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "physical CUDA qualification"]
fn cuda_reachability_and_numeric_helpers_match_oracle() {
    use invar::{eval::Value, general_net::device_host::cuda_nvrtc::CudaNvrtc};
    let mut backend =
        CudaNvrtc::open(std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))).unwrap();
    println!("P5B CUDA actual capabilities: {:?}", backend.capabilities());
    entry::corpus(&mut backend, Target::Cuda);
    returned_owner_cases(&mut backend, Target::Cuda);
    for x in [-2147483648, -21, -1, 0, 1, 2, 3, 7, 20, 21, 2147483647] {
        for d in [-1, 0, 1] {
            entry::agree(
                &mut backend,
                Target::Cuda,
                REACHABLE,
                "run",
                vec![Value::I32(x), Value::I32(d)],
                100000,
            );
        }
    }
    println!("P5B CUDA: 33 call-closure cases plus 15 ABI13 entry/numeric cases");
}

#[test]
fn call_closure_matches_3000_trap_heavy_inputs() {
    use invar::{
        eval::Value,
        general_net::{
            compiled::OwnedLimits,
            numeric_owned_resume_host::{Plan, Progress},
            oracle,
        },
    };
    let mut plan = Plan::from_source(
        REACHABLE.as_bytes(),
        "run",
        Default::default(),
        OwnedLimits {
            elements: 128,
            allocations: 128,
            cleanup_events: 128,
        },
        std::path::Path::new("c++"),
    )
    .unwrap();
    let limits = oracle::Limits {
        fuel: 10000,
        transitions: 10000,
        trace_events: 5000,
        array_elements: 128,
    };
    let mut state = 0x503542u32;
    let mut traps = 0;
    for case in 0..3000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let x = if case % 3 == 0 {
            (state % 21) as i32
        } else {
            state as i32
        };
        let d = (case % 3) - 1;
        let args = [Value::I32(x), Value::I32(d)];
        let (net, proof) = plan.artifact().checked_net();
        let expected = oracle::observe_call(net, proof, "run", &args, limits).unwrap();
        let Progress::Complete(got) = plan
            .start(&args, limits)
            .unwrap()
            .resume(1_000_000)
            .unwrap()
        else {
            panic!("case {case} must finish in one maximal quantum")
        };
        traps += usize::from(expected.semantic.result.is_err());
        assert_eq!(got.semantic.result, expected.semantic.result, "case {case}");
        assert_eq!(
            got.failure, expected.failure,
            "first source-order trap at {case}"
        );
        assert_eq!(got.semantic.steps_used, expected.semantic.steps_used);
        assert_eq!(got.semantic.cleanup_trace, expected.semantic.cleanup_trace);
        assert_eq!(got.events, expected.events);
        assert_eq!(got.transitions, expected.transitions);
    }
    assert!(traps > 1500, "generator must stay trap-heavy: {traps}");
    println!("P5B CPU: 3000 differential inputs, {traps} traps, seed=0x503542");
}

#[test]
fn whole_candidate_admission_still_rejects_unreachable_unsupported_types() {
    use invar::{
        check, front,
        general_net::{self, compiled},
    };
    let source = b"fn run()->i32{return 7;} fn unused()->Array<i32>{return [1,2];}";
    let checked = check::check(front::parse(source).unwrap(), false).unwrap();
    let net = general_net::Net::lower(&checked, Default::default()).unwrap();
    let proof = net.verify().unwrap();
    assert_eq!(
        compiled::emit(&net, &proof, "run", Target::Metal, Default::default()).unwrap_err(),
        compiled::Error::UnsupportedType
    );
}

#[test]
fn scalar_heap_omission_keeps_owned_operations_and_cleanup() {
    use invar::{eval::Value, general_net::device_host::cpu::CpuModel};
    let source =
        "fn run(x:i32)->i32{let xs:Array<i32> = [x,7];return xs[1u32]/x;}fn main()->i32{return 0;}";
    for target in [Target::Metal, Target::Cuda] {
        let mut backend = CpuModel::default();
        for x in [0, 1] {
            entry::agree(
                &mut backend,
                target,
                source,
                "run",
                vec![Value::I32(x)],
                100000,
            );
        }
    }
}

#[test]
fn returned_owners_match_the_oracle_including_first_traps() {
    use invar::general_net::device_host::cpu::CpuModel;
    for target in [Target::Metal, Target::Cuda] {
        returned_owner_cases(&mut CpuModel::default(), target);
    }
}
fn returned_owner_cases<B: invar::general_net::device_host::Backend>(
    backend: &mut B,
    target: Target,
) {
    use invar::eval::Value;
    for source in [
        "fn run(x:i32)->Array<i32>{let gone:Array<i32> = [9];return [x,7/x];}",
        "fn run(x:i32)->Array<u32>{let gone:Array<u32> = [9u32];return [1u32,2u32];}",
        "fn run(x:i32)->Array<f32>{let gone:Array<f32> = [9.0];return [-0.0,0.5];}",
        "fn run(x:i32)->Array<f32>{return filled_f32(0u32,0.0);}",
    ] {
        for x in [0, 1, -1] {
            entry::agree(backend, target, source, "run", vec![Value::I32(x)], 100000);
        }
    }
    println!("P5B returned owners: 12 oracle cases on {target:?}");
}

/// A net whose emitted source lies between the default 1 MiB budget and
/// `MAX_GNET_SOURCE_BYTES`, with limits that admit everything but its size.
fn large_net() -> (String, Limits) {
    let mut source = String::from("fn run()->u32{var x:u32=0u32;");
    for _ in 0..1500 {
        source.push_str("x=wrapping_add(x,1u32);");
    }
    source.push_str("return x;} fn main()->i32{return 0;}");
    let mut limits = entry::limits();
    limits.compiler.values = 8192;
    limits.compiler.states = 16384;
    limits.compiler.frame_bytes = 8 * 1024 * 1024;
    (source, limits)
}

/// Review findings (GNET kernel shrinking): the GNET source ceiling is
/// `MAX_GNET_SOURCE_BYTES` wherever CUDA still compiles a source at run time.
/// Since the ahead-compiled CUDA route, that is the P2 region route behind `run_gnet_region`. Sources just
/// past the old 1 MiB gate and exactly at the ceiling must reach the compiler
/// (here `/bin/false`, so the failure is the compile, not the size), and one
/// byte past the ceiling must still be refused by size, so any lower cap fails.
/// Deterministic: no source is really compiled, unlike an end-to-end run, whose
/// NVRTC time depends on host load.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "physical CUDA qualification"]
fn cuda_region_compile_admits_sources_up_to_the_raised_ceiling() {
    use invar_cuda::{Error, cuda::CudaDevice};
    let mut device = CudaDevice::open().expect("CUDA device");
    let mut image: [Vec<u64>; 12] = std::array::from_fn(|_| vec![0u64]);
    image[3] = vec![0; 6];
    let failing = std::path::Path::new("/bin/false");
    let mut run = |len: usize| {
        device
            .run_gnet_region(&" ".repeat(len), &image, 1, failing)
            .expect_err("the /bin/false compiler cannot succeed")
    };
    for len in [1024 * 1024 + 1, invar_device::MAX_GNET_SOURCE_BYTES] {
        let inside = run(len);
        assert!(
            !matches!(inside, Error::Limit),
            "{len} bytes refused by size: {inside:?}"
        );
    }
    let past = run(invar_device::MAX_GNET_SOURCE_BYTES + 1);
    assert!(matches!(past, Error::Limit), "{past:?}");
}

#[test]
fn explicit_source_budget_admits_large_nets_without_changing_defaults() {
    use invar::general_net::compiled::{self, Error};
    let (source, limits) = large_net();
    assert_eq!(compiled::Limits::default().source_bytes, 1024 * 1024);
    for target in [Target::Metal, Target::Cuda] {
        assert!(matches!(
            Plan::from_source(source.as_bytes(), "run", target, limits),
            Err(invar::general_net::device_host::Error::Provenance(
                invar::general_net::provenance::Error::Compiler(Error::Limit)
            ))
        ));
        let mut raised = limits;
        raised.compiler.source_bytes = invar_device::MAX_GNET_SOURCE_BYTES;
        let plan = Plan::from_source(source.as_bytes(), "run", target, raised).unwrap();
        assert!(plan.artifact().source().len() > limits.compiler.source_bytes);
        assert!(plan.artifact().source().len() <= invar_device::MAX_GNET_SOURCE_BYTES);
    }
}
