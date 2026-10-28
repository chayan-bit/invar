#![allow(dead_code)]
#[path = "gnet_device.rs"]
pub mod device;
use invar::{
    eval::Value,
    general_net::{
        compiled::{self, Target},
        device_host::{Backend, Limits, Plan},
        oracle,
    },
    numeric_contract::F32Bits,
};
use std::{cell::RefCell, rc::Rc};
pub const FK: &str = include_str!("../fixtures/gnet_entry/fk.inv");
pub const INTEGERS: &str = include_str!("../fixtures/gnet_entry/integers.inv");
pub const NUMERIC: &str = include_str!("../fixtures/gnet_entry/numeric.inv");
pub fn f(v: &[u32]) -> Value {
    Value::ArrayF32(Rc::new(RefCell::new(
        v.iter().map(|x| F32Bits::from_bits(*x)).collect(),
    )))
}
pub fn config() -> compiled::DeviceConfig {
    compiled::DeviceConfig {
        fuel: 100000,
        transitions: 100000,
        trace_events: 50000,
        quantum: 17,
        rounds: 64,
        capacity: 0,
    }
}
pub fn limits() -> Limits {
    Limits {
        owned: compiled::OwnedLimits {
            elements: 2048,
            allocations: 32,
            cleanup_events: 128,
        },
        private: compiled::SpeculationLimits {
            slots: 0,
            ..Default::default()
        },
        ..Default::default()
    }
}
pub fn copy(v: &Value) -> Value {
    match v {
        Value::ArrayI32(v) => Value::ArrayI32(Rc::new(RefCell::new(v.borrow().clone()))),
        Value::ArrayU32(v) => Value::ArrayU32(Rc::new(RefCell::new(v.borrow().clone()))),
        Value::ArrayF32(v) => Value::ArrayF32(Rc::new(RefCell::new(v.borrow().clone()))),
        Value::Record(v) => {
            Value::Record(Rc::new(RefCell::new(v.borrow().iter().map(copy).collect())))
        }
        Value::I32(v) => Value::I32(*v),
        Value::U32(v) => Value::U32(*v),
        Value::F32(v) => Value::F32(*v),
        Value::Bool(v) => Value::Bool(*v),
        _ => panic!("fixture type"),
    }
}
pub fn fk_args(n: usize, fail: bool) -> Vec<Value> {
    vec![
        Value::ArrayF32(Rc::new(RefCell::new(
            (1..=n * 6).map(|x| F32Bits::from_u32(x as u32)).collect(),
        ))),
        f(&[0x3f000000; 6]),
        f(&vec![0xdeadbeef; n]),
        Value::Record(Rc::new(RefCell::new(vec![
            Value::F32(F32Bits::from_bits(0x3fc00000)),
            Value::F32(F32Bits::from_bits(0x3e800000)),
        ]))),
        Value::Bool(fail),
    ]
}
pub fn agree<B: Backend>(
    backend: &mut B,
    target: Target,
    source: &str,
    entry: &str,
    args: Vec<Value>,
    fuel: u64,
) -> compiled::Status {
    let plan = Plan::from_source(source.as_bytes(), entry, target, limits()).unwrap();
    let reference_args = args.iter().map(copy).collect::<Vec<_>>();
    let config = compiled::DeviceConfig { fuel, ..config() };
    let reference = oracle::observe_call(
        plan.net(),
        plan.proof(),
        entry,
        &reference_args,
        oracle::Limits {
            fuel,
            transitions: config.transitions,
            trace_events: config.trace_events,
            array_elements: limits().owned.elements,
        },
    )
    .unwrap();
    let (got, _) = device::drive(&plan, backend, &args, config).unwrap();
    device::agree("P0b", &plan, &got, &reference);
    numeric_preflight(backend);

    assert_eq!(
        args, reference_args,
        "completion-gated host array publication"
    );
    got.status
}
pub fn corpus<B: Backend>(backend: &mut B, target: Target) {
    use compiled::Status as S;
    assert_eq!(
        agree(
            backend,
            target,
            "fn run()->Bool{return 1.0*2.0==2.0;} fn main()->i32{return 0;}",
            "run",
            vec![],
            100000
        ),
        S::Success
    );
    for n in [0, 1, 6, 31, 32, 33] {
        assert_eq!(
            agree(backend, target, FK, "fk", fk_args(n, false), 100000),
            S::Success
        );
    }
    assert_eq!(
        agree(backend, target, FK, "fk", fk_args(2, true), 100000),
        S::IndexOutOfBounds
    );
    assert_eq!(
        agree(backend, target, FK, "fk", fk_args(2, false), 1),
        S::FuelLimit
    );
    // Normal boundaries, subnormals, signed zero, Inf, NaN and contraction.
    let a = [
        1, 0x007fffff, 0x00800000, 0x80000000, 0x7f800000, 0x7fc01234, 0x3f800001,
    ];
    let b = [
        0x3f800000, 0x3f000000, 0x3f000000, 0x3f800000, 0, 0x3f800000, 0x3f7ffffe,
    ];
    for c in [0, 0xbf800000, 0x80000000] {
        assert_eq!(
            agree(
                backend,
                target,
                NUMERIC,
                "run",
                vec![f(&a), f(&b), f(&[0; 7]), Value::F32(F32Bits::from_bits(c))],
                100000
            ),
            S::Success
        );
    }
    let ints = || {
        vec![
            Value::ArrayI32(Rc::new(RefCell::new(vec![-7, 0, 2147483646]))),
            Value::ArrayU32(Rc::new(RefCell::new(vec![0, u32::MAX, 7]))),
            Value::ArrayI32(Rc::new(RefCell::new(vec![44; 3]))),
            Value::ArrayU32(Rc::new(RefCell::new(vec![55; 3]))),
        ]
    };
    assert_eq!(
        agree(backend, target, INTEGERS, "run", ints(), 100000),
        S::Success
    );
    let bad = ints();
    if let Value::ArrayI32(v) = &bad[0] {
        v.borrow_mut()[2] = i32::MAX;
    }
    assert_eq!(
        agree(backend, target, INTEGERS, "run", bad, 100000),
        S::IntegerOverflow
    );
    assert_eq!(
        agree(
            backend,
            target,
            "struct P {x:f32,n:u32} fn run(a:read P)->P{return P{x:a.x*2.0,n:a.n+1u32};} fn main()->i32{return 0;}",
            "run",
            vec![Value::Record(Rc::new(RefCell::new(vec![
                Value::F32(F32Bits::from_bits(1)),
                Value::U32(7)
            ])))],
            100000
        ),
        S::Success
    );
    cr_v1(backend, target);
}

/// f32 `sqrt` is the correctly rounded `invar-math-cr-v1` word on every
/// target (signed zeros, subnormals, both NaN classes, infinities and a
/// stratified sweep of normals), and `sin` and `cos` are refused: no device
/// lowers their cr-v1 kernels yet (#163).
fn cr_v1<B: Backend>(backend: &mut B, target: Target) {
    use invar::math::{MathFn, eval_f32};
    let xs: Vec<u32> = (0u32..256)
        .map(|i| (i << 24) | (i.wrapping_mul(0x9e37_79b1) >> 8))
        .chain([
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            2,
            3,
            0x0040_0000,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0000,
            0x4000_0000,
            0x7f7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0x7f80_0001,
            0xffc0_0001,
        ])
        .collect();
    let source = "fn run(a: read Array<f32>, b: mut Array<f32>) -> u32 { for i in 0u32..len(a) { b[i] = sqrt(a[i]); } return len(a); } fn main() -> i32 { return 0; }";
    let args = vec![f(&xs), f(&vec![0; xs.len()])];
    let Value::ArrayF32(out) = &args[1] else {
        panic!("sqrt output array")
    };
    let out = Rc::clone(out);
    assert_eq!(
        agree(backend, target, source, "run", args, 100_000),
        compiled::Status::Success
    );
    for (&x, y) in xs.iter().zip(out.borrow().iter()) {
        let want = eval_f32(MathFn::Sqrt, &[F32Bits::from_bits(x)]);
        assert_eq!(y.to_bits(), want.to_bits(), "{target:?} sqrt {x:#010x}");
    }
    for name in ["sin", "cos"] {
        let source = source.replace("sqrt(", &format!("{name}("));
        assert!(
            Plan::from_source(source.as_bytes(), "run", target, limits()).is_err(),
            "{target:?} admitted cr-v1 {name}"
        );
    }
}

/// Same full-size source and input on the net oracle and each physical backend.
pub fn large<B: Backend>(backend: &mut B, target: Target) {
    let n = invar::array_limits::MAX_ARRAY_LEN;
    let mut limits = limits();
    limits.owned.elements = n;
    limits.buffer_bytes = 512 * 1024 * 1024;
    let plan = Plan::from_source(
        b"fn run(a:read Array<f32>)->f32{return a[len(a)-1u32];} fn main()->i32{return 0;}",
        "run",
        target,
        limits,
    )
    .unwrap();
    let mut array = vec![F32Bits::POS_ZERO; n];
    array[n - 1] = F32Bits::from_bits(1);
    let args = [Value::ArrayF32(Rc::new(RefCell::new(array)))];
    let reference = oracle::observe_call(
        plan.net(),
        plan.proof(),
        "run",
        &args,
        oracle::Limits {
            fuel: config().fuel,
            transitions: config().transitions,
            trace_events: config().trace_events,
            array_elements: n,
        },
    )
    .unwrap();
    let (got, _) = device::drive(&plan, backend, &args, config()).unwrap();
    device::agree("2^24 f32 entry", &plan, &got, &reference);
    numeric_preflight(backend);
    assert_eq!(got.result, Some(Value::F32(F32Bits::from_bits(1))));
}

fn numeric_preflight<B: Backend>(backend: &B) {
    use invar::general_net::provenance::FenceBackend;
    if matches!(
        B::label(),
        FenceBackend::MetalCommand | FenceBackend::CudaDriver
    ) {
        let report = backend
            .numeric_receipt()
            .expect("completed D1 probe before GNET submission");
        assert_eq!(report["policy"], "strict-f32-v1");
        assert_eq!(report["native_float_admission"], false);
        assert_eq!(report["subnormal_fallback"], "integer-software-binary32");
        assert!(
            report["probe"]
                .as_str()
                .unwrap()
                .starts_with("invar-f32-probe-v1:")
        );
    }
}
