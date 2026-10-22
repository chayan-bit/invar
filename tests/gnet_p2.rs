#![forbid(unsafe_code)]
#[path = "support/normal_form.rs"]
mod nf;
#[path = "support/gnet_device.rs"]
mod support;
use invar::{
    eval::Value,
    general_net::{
        compiled::{self, DeviceConfig, Target, body_map},
        device_host::{
            self,
            host_frame::{CpuLanes, Plan, RegionBackend},
        },
        normal_form, oracle,
    },
};
use std::{cell::RefCell, rc::Rc};
fn array(v: Vec<i32>) -> Value {
    Value::ArrayI32(Rc::new(RefCell::new(v)))
}
fn copy(v: &Value) -> Value {
    match v {
        Value::ArrayI32(a) => array(a.borrow().clone()),
        Value::ArrayU32(a) => Value::ArrayU32(Rc::new(RefCell::new(a.borrow().clone()))),
        Value::ArrayF32(a) => Value::ArrayF32(Rc::new(RefCell::new(a.borrow().clone()))),
        Value::I32(x) => Value::I32(*x),
        Value::U32(x) => Value::U32(*x),
        Value::F32(x) => Value::F32(*x),
        Value::Bool(x) => Value::Bool(*x),
        Value::Unit => Value::Unit,
        _ => panic!("unsupported test argument"),
    }
}
fn limits() -> device_host::Limits {
    device_host::Limits {
        owned: compiled::OwnedLimits {
            elements: 1024,
            allocations: 64,
            cleanup_events: 256,
        },
        ..Default::default()
    }
}
fn config() -> DeviceConfig {
    DeviceConfig {
        fuel: 100_000,
        transitions: 1_000_000,
        trace_events: 20_000,
        quantum: 1_000_000,
        rounds: 1,
        capacity: 0,
    }
}
fn plan(source: &str, target: Target) -> Plan {
    Plan::from_source(source.as_bytes(), "run", target, limits()).expect("P2 plan")
}
fn agree<B: RegionBackend>(
    plan: &mut Plan,
    backend: &mut B,
    args: Vec<Value>,
    config: DeviceConfig,
) -> device_host::host_frame::Report {
    let reference_args: Vec<_> = args.iter().map(copy).collect();
    let p = plan.entry_plan();
    let reference = oracle::observe_call(
        p.net(),
        p.proof(),
        "run",
        &reference_args,
        oracle::Limits {
            fuel: config.fuel,
            transitions: config.transitions,
            trace_events: config.trace_events,
            array_elements: limits().owned.elements,
        },
    )
    .unwrap();
    let got = plan.run(backend, &args, config).expect("P2 run");
    support::agree("P2", plan.entry_plan(), &got.observation, &reference);
    assert_eq!(args, reference_args, "oracle host publication boundary");
    got
}
const MAP: &str = "fn f(x:i32)->i32{if x<0{return -x;}return x*3;} fn run(a:mut Array<i32>,n:u32)->i32{for i in 0u32..n{var s=0;for j in 0u32..3u32{s=s+f(a[i]);}a[i]=s;}return 17;}fn main()->i32{return 0;}";
const FIRST: &str = "fn run(a:read Array<i32>,n:u32)->i32{for i in 0u32..n{if 100/a[i]==7{return a[i];}}return -1;}fn main()->i32{return 0;}";
const FRONTIER: &str = "fn run(seen:mut Array<u32>,front:read Array<u32>,next:mut Array<u32>,n:u32)->i32{var count=0;for u in 0u32..n{if front[u]==1u32 and seen[u]==0u32{seen[u]=1u32;count=count+1;for k in 0u32..n{if seen[k]==0u32{next[k]=1u32;}}}}return count;}fn main()->i32{return 0;}";
fn uarray(v: Vec<u32>) -> Value {
    Value::ArrayU32(Rc::new(RefCell::new(v)))
}
fn host_publication<B: RegionBackend>(b: &mut B, target: Target) {
    // Lane zero completes; lane one writes zero before its division traps.
    // Neither that partial iteration nor the completed prefix escapes the entry.
    let source = "fn run(a:mut Array<i32>,n:u32,z:i32)->i32{for i in 0u32..n{a[i]=a[i]+1;a[i]=100/a[i];}return 1/z;}fn main()->i32{return 0;}";
    let mut p = plan(source, target);
    for (xs, z, expected_status, expected) in [
        (
            vec![1, -1, 3],
            1,
            compiled::Status::DivisionByZero,
            vec![1, -1, 3],
        ),
        // All regions finish before a trap in the serial host frame.
        (
            vec![1, 3, 4],
            0,
            compiled::Status::DivisionByZero,
            vec![1, 3, 4],
        ),
        (
            vec![1, 3, 4],
            1,
            compiled::Status::Success,
            vec![50, 25, 20],
        ),
    ] {
        let a = Rc::new(RefCell::new(xs));
        let r = agree(
            &mut p,
            b,
            vec![Value::ArrayI32(Rc::clone(&a)), Value::U32(3), Value::I32(z)],
            config(),
        );
        assert!(r.launches > 0);
        assert_eq!(r.observation.status, expected_status);
        assert_eq!(*a.borrow(), expected, "external host commit/rollback");
    }
}
#[test]
fn p2_host_arrays_commit_only_on_normal_return() {
    host_publication(&mut CpuLanes::default(), Target::Cuda);
}
#[test]
fn p2_map_nested_calls_and_trap_heavy_oracle() {
    let mut p = plan(MAP, Target::Cuda);
    assert!(!p.compiled().sites().is_empty());
    let mut b = CpuLanes::default();
    let mut cases = 0;
    for n in [0, 1, 2, 7, 31, 32, 33] {
        for seed in 0..8 {
            let xs = (0..n)
                .map(|i| match (i + seed) % 13 {
                    0 => i32::MIN,
                    1 => i32::MAX,
                    2 => -400_000_000,
                    _ => i as i32 - 5,
                })
                .collect();
            let r = agree(&mut p, &mut b, vec![array(xs), Value::U32(n)], config());
            if n > 0 {
                assert!(r.launches > 0);
            }
            cases += 1;
        }
    }
    eprintln!("P2_MAP_ORACLE_CASES={cases}");
}
#[test]
fn p2_first_exit_and_prior_trap_order() {
    let mut p = plan(FIRST, Target::Cuda);
    assert!(
        p.compiled()
            .sites()
            .iter()
            .any(|s| s.shape == body_map::Shape::First)
    );
    let mut b = CpuLanes::default();
    let mut cases = 0;
    for exit in 0..8 {
        for trap in 0..9 {
            let mut xs = vec![9; 8];
            xs[exit] = 14;
            if trap < 8 {
                xs[trap] = 0;
            }
            let r = agree(&mut p, &mut b, vec![array(xs), Value::U32(8)], config());
            assert!(r.launches > 0);
            cases += 1;
        }
    }
    eprintln!("P2_FIRST_ORACLE_CASES={cases}");
}
#[test]
fn p2_frontier_reconstructs_earlier_writers_and_steps() {
    let mut p = plan(FRONTIER, Target::Cuda);
    assert!(
        p.compiled()
            .sites()
            .iter()
            .any(|s| s.shape == body_map::Shape::Frontier)
    );
    let mut b = CpuLanes::default();
    for mask in 0..64u32 {
        let front = (0..6).map(|i| (mask >> i) & 1).collect();
        let r = agree(
            &mut p,
            &mut b,
            vec![
                uarray(vec![0; 6]),
                uarray(front),
                uarray(vec![0; 6]),
                Value::U32(6),
            ],
            config(),
        );
        assert_eq!(r.launches, 2);
        assert_eq!(r.replayed, 0);
    }
    eprintln!("P2_FRONTIER_ORACLE_CASES=64");
}
#[test]
fn p2_fuel_transition_trace_cuts_and_partial_iteration_writes() {
    let source = "fn run(a:mut Array<i32>,n:u32)->i32{for i in 0u32..n{a[i]=5;a[i]=100/(a[i]-5);}return 0;}fn main()->i32{return 0;}";
    let mut p = plan(source, Target::Cuda);
    let mut b = CpuLanes::default();
    for fuel in [1, 7, 14, 25, 40, 100_000] {
        let mut c = config();
        c.fuel = fuel;
        agree(&mut p, &mut b, vec![array(vec![2; 3]), Value::U32(3)], c);
    }
    for transitions in [0, 1, 11, 35, 100] {
        let mut c = config();
        c.transitions = transitions;
        agree(&mut p, &mut b, vec![array(vec![2; 3]), Value::U32(3)], c);
    }
    for trace in [0, 1, 5, 20, 100] {
        let mut c = config();
        c.trace_events = trace;
        agree(&mut p, &mut b, vec![array(vec![2; 3]), Value::U32(3)], c);
    }
}
#[test]
fn p2_private_allocations_and_cleanup_prefix() {
    let source = "fn f(x:i32)->i32{var a=filled_i32(2u32,x);a[1u32]=x+1;return a[1u32];}fn run(a:mut Array<i32>,n:u32)->i32{for i in 0u32..n{a[i]=f(a[i]);}return 4;}fn main()->i32{return 0;}";
    let mut p = plan(source, Target::Cuda);
    let mut b = CpuLanes::default();
    for xs in [vec![1, 2, 3], vec![1, i32::MAX, 3], vec![]] {
        let n = xs.len() as u32;
        agree(&mut p, &mut b, vec![array(xs), Value::U32(n)], config());
    }
}
#[test]
fn p2_checked_partition_and_named_certificate_mutations() {
    let (net, proof) = nf::compile(MAP);
    let original = normal_form::extract(&net, &proof).unwrap();
    for mutation in [
        "drop_map_certificate",
        "change_first_to_map",
        "stale_fingerprint",
    ] {
        let mut form = original.clone();
        match mutation {
            "drop_map_certificate" => {
                form.functions
                    .iter_mut()
                    .find(|f| !f.loops.is_empty())
                    .unwrap()
                    .loops[0]
                    .certificates
                    .clear();
            }
            "change_first_to_map" => {
                if let normal_form::Node::Parallel(n) = &mut form.functions[1].loops[0].node {
                    n.first_exit = true;
                }
            }
            _ => form.fingerprint[0] ^= 1,
        }
        assert!(
            body_map::emit(
                &net,
                &proof,
                "run",
                Target::Cuda,
                &form,
                Default::default(),
                limits().owned
            )
            .is_err(),
            "{mutation}"
        );
    }
    assert_eq!(body_map::Compiled::groups(257).unwrap(), 2);
    assert_eq!(
        body_map::Compiled::groups(usize::MAX).unwrap(),
        body_map::MAX_GROUPS
    );
    assert!(body_map::Compiled::groups(0).is_err());
}
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn hardware<B: RegionBackend>(b: &mut B, target: Target) {
    host_publication(b, target);
    for source in [MAP, FIRST] {
        let mut p = plan(source, target);
        for xs in [
            vec![1, 2, 14, 3],
            vec![1, 0, 14, 3],
            vec![i32::MAX, 14, 1, 2],
            vec![1, i32::MAX, 14, 3],
        ] {
            agree(&mut p, b, vec![array(xs), Value::U32(4)], config());
        }
    }
    let mut p = plan(FRONTIER, target);
    agree(
        &mut p,
        b,
        vec![
            uarray(vec![0; 6]),
            uarray(vec![1, 0, 1, 1, 0, 1]),
            uarray(vec![0; 6]),
            Value::U32(6),
        ],
        config(),
    );
    // A tight global trace budget still launches 257 independent lanes in two
    // groups, then precisely replays the native source trace cut.
    let source = "fn run(a:mut Array<i32>,n:u32)->i32{for i in 0u32..n{a[i]=a[i]+1;}return 4;}fn main()->i32{return 0;}";
    let mut small = limits();
    small.owned = compiled::OwnedLimits {
        elements: 512,
        allocations: 8,
        cleanup_events: 0,
    };
    let mut p = Plan::from_source(source.as_bytes(), "run", target, small).unwrap();
    let mut c = config();
    c.trace_events = 64;
    let r = agree(&mut p, b, vec![array(vec![1; 257]), Value::U32(257)], c);
    assert_eq!(r.lanes, 257);
    assert_eq!(body_map::Compiled::groups(r.lanes as usize).unwrap(), 2);
    let mut p = plan(FLOAT_MAP, target);
    agree(&mut p, b, float_args(), config());
    eprintln!("P2_PHYSICAL_ORACLE_CASES=14;MULTIGROUP=257_LANES_2_GROUPS");
}
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Metal hardware"]
fn p2_metal_oracle() {
    hardware(
        &mut device_host::metal::Metal::open_runtime_source().unwrap(),
        Target::Metal,
    );
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA hardware"]
fn p2_cuda_oracle() {
    // INVAR_NVRTC when a host needs a newer NVRTC than the linked one (an sm_120 host).
    let compiler = std::env::var_os("INVAR_NVRTC")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_invar-nvrtc").into());
    hardware(
        &mut device_host::cuda_nvrtc::CudaNvrtc::open(&compiler).unwrap(),
        Target::Cuda,
    );
}

#[test]
fn p2_nonzero_and_signed_while_bounds() {
    let source = "fn run(lo:i32,hi:i32)->i32{var i=lo;while i<hi{let x=100/(i+3);i=i+1;}return 9;}fn main()->i32{return 0;}";
    let mut p = plan(source, Target::Cuda);
    assert!(!p.compiled().sites().is_empty());
    let mut b = CpuLanes::default();
    for (lo, hi) in [
        (-5, 5),
        (-2, 5),
        (8, 5),
        (5, 5),
        (1, 10),
        (i32::MAX - 2, i32::MAX),
    ] {
        agree(
            &mut p,
            &mut b,
            vec![Value::I32(lo), Value::I32(hi)],
            config(),
        );
    }
}

#[test]
fn p2_large_map_chunks_rejoin_native_frame() {
    let source = "fn run(a:mut Array<i32>,n:u32)->i32{for i in 0u32..n{a[i]=a[i]+1;}return 4;}fn main()->i32{return 0;}";
    let mut p = plan(source, Target::Cuda);
    let mut b = CpuLanes::default();
    let r = agree(
        &mut p,
        &mut b,
        vec![array(vec![0; 300]), Value::U32(300)],
        config(),
    );
    assert!(r.launches > 1);
    assert_eq!(r.lanes, 300);
    assert_eq!(r.replayed, 0);
}

#[derive(Clone, Copy)]
enum Mutation {
    ReuseLaneZeroHeap,
    LastBoundary,
    DropFrontierPrefixes,
}
struct Mutated {
    body: CpuLanes,
    mutation: Mutation,
}
impl RegionBackend for Mutated {
    fn execute(
        &mut self,
        p: &body_map::Compiled,
        image: &device_host::host_frame::Image,
        groups: usize,
    ) -> Result<device_host::host_frame::Image, device_host::Error> {
        let mut input = image.clone();
        if matches!(self.mutation, Mutation::DropFrontierPrefixes) && input[9][0] != 0 {
            input[9][0] = 0;
        }
        let mut out = self.body.execute(p, &input, groups)?;
        match self.mutation {
            Mutation::ReuseLaneZeroHeap => {
                let n = out[3][0] as usize;
                let heap = out[4].len() / n;
                let zero = out[4][..heap].to_vec();
                for k in 1..n {
                    out[4][k * heap..(k + 1) * heap].copy_from_slice(&zero);
                }
            }
            Mutation::LastBoundary => {
                out[11][9] = out[11]
                    .chunks_exact(14)
                    .rposition(|c| c[6] != 1)
                    .map_or(out[3][0], |i| i as u64);
            }
            Mutation::DropFrontierPrefixes => {}
        }
        Ok(out)
    }
}
#[test]
fn p2_l1_reuse_lane_zero_heap_mutation_must_fail() {
    let mut p = plan(MAP, Target::Cuda);
    let mut baseline = CpuLanes::default();
    agree(
        &mut p,
        &mut baseline,
        vec![array(vec![1, 2, 3]), Value::U32(3)],
        config(),
    );
    let mut b = Mutated {
        body: baseline,
        mutation: Mutation::ReuseLaneZeroHeap,
    };
    let killed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        agree(
            &mut p,
            &mut b,
            vec![array(vec![1, 2, 3]), Value::U32(3)],
            config(),
        );
    }));
    assert!(killed.is_err(), "named runtime mutation survived");
}
#[test]
fn p2_l15_last_boundary_mutation_must_fail() {
    let mut p = plan(FIRST, Target::Cuda);
    let mut baseline = CpuLanes::default();
    agree(
        &mut p,
        &mut baseline,
        vec![array(vec![9, 0, 9, 14]), Value::U32(4)],
        config(),
    );
    let mut b = Mutated {
        body: baseline,
        mutation: Mutation::LastBoundary,
    };
    let killed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        agree(
            &mut p,
            &mut b,
            vec![array(vec![9, 0, 9, 14]), Value::U32(4)],
            config(),
        );
    }));
    assert!(killed.is_err(), "named runtime mutation survived");
}
#[test]
fn p2_l14_drop_frontier_prefixes_mutation_must_fail() {
    let mut p = plan(FRONTIER, Target::Cuda);
    let mut baseline = CpuLanes::default();
    let args = || {
        vec![
            uarray(vec![0; 4]),
            uarray(vec![1; 4]),
            uarray(vec![0; 4]),
            Value::U32(4),
        ]
    };
    agree(&mut p, &mut baseline, args(), config());
    let mut b = Mutated {
        body: baseline,
        mutation: Mutation::DropFrontierPrefixes,
    };
    let killed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        agree(&mut p, &mut b, args(), config());
    }));
    assert!(killed.is_err(), "named runtime mutation survived");
}
#[test]
fn p2_unknown_fence_never_replays_or_publishes() {
    struct Unknown(usize);
    impl RegionBackend for Unknown {
        fn execute(
            &mut self,
            _: &body_map::Compiled,
            _: &device_host::host_frame::Image,
            _: usize,
        ) -> Result<device_host::host_frame::Image, device_host::Error> {
            self.0 += 1;
            Err(device_host::Error::Fence(compiled::FenceError::Unknown))
        }
    }
    let mut p = plan(MAP, Target::Cuda);
    let mut backend = Unknown(0);
    let args = vec![array(vec![1, 2, 3]), Value::U32(3)];
    assert!(matches!(
        p.run(&mut backend, &args, config()),
        Err(device_host::Error::Fence(compiled::FenceError::Unknown))
    ));
    assert_eq!(backend.0, 1);
    assert_eq!(args[0], array(vec![1, 2, 3]));
}
#[test]
fn p2_malformed_capsules_are_rejected_before_publication() {
    struct Short(usize, CpuLanes);
    impl RegionBackend for Short {
        fn execute(
            &mut self,
            p: &body_map::Compiled,
            image: &device_host::host_frame::Image,
            g: usize,
        ) -> Result<device_host::host_frame::Image, device_host::Error> {
            let mut result = self.1.execute(p, image, g)?;
            result[self.0].clear();
            Ok(result)
        }
    }
    let mut p = plan(MAP, Target::Cuda);
    for slot in [0, 1, 2, 3, 4, 5, 6, 7, 8, 11] {
        let mut backend = Short(slot, CpuLanes::default());
        let args = vec![array(vec![1, 2, 3]), Value::U32(3)];
        assert!(
            matches!(
                p.run(&mut backend, &args, config()),
                Err(device_host::Error::Malformed)
            ),
            "slot {slot}"
        );
        assert_eq!(args[0], array(vec![1, 2, 3]));
    }
}
const FLOAT_MAP: &str = "fn run(a:read Array<f32>,out:mut Array<f32>,n:u32)->Unit{for i in 0u32..n{out[i]=a[i]*0.5+0.0;}}fn main()->i32{return 0;}";
fn float_args() -> Vec<Value> {
    use invar::numeric_contract::F32Bits;
    let values = [
        1, 0x007fffff, 0x00800000, 0x80000000, 0x7f800000, 0x7fc01234, 0x3f800001,
    ];
    let make = |v: &[u32]| {
        Value::ArrayF32(Rc::new(RefCell::new(
            v.iter().map(|x| F32Bits::from_bits(*x)).collect(),
        )))
    };
    vec![make(&values), make(&[0xdeadbeef; 7]), Value::U32(7)]
}
#[test]
fn p2_soft_f32_preserves_subnormals_and_signed_zero() {
    let mut p = plan(FLOAT_MAP, Target::Cuda);
    let mut b = CpuLanes::default();
    agree(&mut p, &mut b, float_args(), config());
}

#[test]
fn p2_first_in_callee_stops_before_native_caller_continues() {
    let source = "fn search(a:read Array<i32>,n:u32)->i32{for i in 0u32..n{if 100/a[i]==7{return a[i];}}return -1;}fn run(a:read Array<i32>,n:u32)->i32{let x=search(a,n);let y=search(a,n);return x+y;}fn main()->i32{return 0;}";
    let mut p = plan(source, Target::Cuda);
    let mut b = CpuLanes::default();
    for xs in [vec![9, 9, 14, 0], vec![9, 0, 14, 9], vec![9; 4]] {
        agree(&mut p, &mut b, vec![array(xs), Value::U32(4)], config());
    }
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[path = "support/gnet_p2_perf.rs"]
mod performance;
