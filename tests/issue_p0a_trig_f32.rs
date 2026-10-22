//! P0a `invar-trig-f32-v1` (`docs/trig-f32-v1.md`): the frozen binary32 `sin`,
//! `cos` and `sqrt` sequence agrees bit for bit on the exact oracle, the host
//! evaluator, the source engines, native LLVM, Metal and CUDA.
//!
//! The default tests pin golden hard cases and a sampled digest, so any edit
//! of the shared sequence fails them. The `--ignored` sweeps cover all 2^32
//! inputs: `p0a_trig_exhaustive_cpu` (oracle, ULP bound, subnormal audit),
//! `p0a_trig_exhaustive_native` (`--features native`),
//! `p0a_trig_exhaustive_metal` (`--features metal`, macOS) and
//! `p0a_trig_exhaustive_cuda` (`--features cuda`).
#![forbid(unsafe_code)]

#[path = "issue_m4_01_device/ops.rs"]
mod ops;
#[path = "issue_p0a_trig_f32/sweep.rs"]
mod sweep;

use invar::numeric_contract::F32Bits;
use invar_device::{
    f32_ops::{self, Kind, Node, Program, Scalar},
    trig_f32::{self, CANONICAL_NAN},
};

/// (x, sin, cos, sqrt), recorded from the exact `F32Bits` oracle. Rows cover
/// zeros, subnormals, the tiny-path and fast-path boundaries, the nearest
/// binary32 values to multiples of pi/2 (including the hardest found by the
/// exhaustive sweep), huge arguments and every special class.
const GOLDEN: &[(u32, u32, u32, u32)] = &[
    (0x0000_0000, 0x0000_0000, 0x3f80_0000, 0x0000_0000),
    (0x8000_0000, 0x8000_0000, 0x3f80_0000, 0x8000_0000),
    (0x0000_0001, 0x0000_0001, 0x3f80_0000, 0x1a35_04f3),
    (0x8000_0001, 0x8000_0001, 0x3f80_0000, 0x7fc0_0000),
    (0x007f_ffff, 0x007f_ffff, 0x3f80_0000, 0x1fff_ffff),
    (0x807f_ffff, 0x807f_ffff, 0x3f80_0000, 0x7fc0_0000),
    (0x0080_0000, 0x0080_0000, 0x3f80_0000, 0x2000_0000),
    (0x8080_0000, 0x8080_0000, 0x3f80_0000, 0x7fc0_0000),
    (0x397f_ffff, 0x397f_ffff, 0x3f80_0000, 0x3c7f_ffff),
    (0x3980_0000, 0x3980_0000, 0x3f80_0000, 0x3c80_0000),
    (0xb980_0000, 0xb980_0000, 0x3f80_0000, 0x7fc0_0000),
    (0x3980_0001, 0x3980_0001, 0x3f7f_ffff, 0x3c80_0000),
    (0x3a00_0000, 0x39ff_ffff, 0x3f7f_fffe, 0x3cb5_04f3),
    (0x3f00_0000, 0x3ef5_7744, 0x3f60_a940, 0x3f35_04f3),
    (0x3f80_0000, 0x3f57_6aa5, 0x3f0a_5140, 0x3f80_0000),
    (0xbf80_0000, 0xbf57_6aa5, 0x3f0a_5140, 0x7fc0_0000),
    (0x3f49_0fdb, 0x3f35_04f3, 0x3f35_04f3, 0x3f62_dfc5),
    (0x3fc9_0fda, 0x3f80_0000, 0x33a2_2169, 0x3fa0_6c99),
    (0x3fc9_0fdb, 0x3f80_0000, 0xb33b_bd2e, 0x3fa0_6c99),
    (0x3fc9_0fdc, 0x3f80_0000, 0xb42e_ef4c, 0x3fa0_6c9a),
    (0xbfc9_0fdb, 0xbf80_0000, 0xb33b_bd2e, 0x7fc0_0000),
    (0x4049_0fda, 0x3422_2169, 0xbf80_0000, 0x3fe2_dfc4),
    (0x4049_0fdb, 0xb3bb_bd2e, 0xbf80_0000, 0x3fe2_dfc5),
    (0x4049_0fdc, 0xb4ae_ef4c, 0xbf80_0000, 0x3fe2_dfc5),
    (0x4096_cbe4, 0xbf80_0000, 0x324c_de2e, 0x400a_ee73),
    (0x40c9_0fdb, 0x343b_bd2e, 0x3f80_0000, 0x4020_6c99),
    (0xc0c9_0fdb, 0xb43b_bd2e, 0x3f80_0000, 0x7fc0_0000),
    (0x497f_ffff, 0x3e8a_b32c, 0x3f76_6d7a, 0x447f_ffff),
    (0x4980_0000, 0x3ea9_3666, 0x3f71_9d6d, 0x4480_0000),
    (0x4980_0001, 0x3ee4_2388, 0x3f65_2e78, 0x4480_0000),
    (0xc980_0001, 0xbee4_2388, 0x3f65_2e78, 0x7fc0_0000),
    (0x4b00_0000, 0x3edd_4fa3, 0xbf66_d964, 0x4535_04f3),
    (0x4b80_0000, 0xbf47_917c, 0x3f20_56b4, 0x4580_0000),
    (0x4f80_0000, 0xbeec_8981, 0xbf63_0b05, 0x4780_0000),
    (0x5f80_0000, 0x3cc1_51aa, 0xbf7f_edc0, 0x4f80_0000),
    (0x7149_f2ca, 0xbf4a_89b0, 0xbf1c_9222, 0x5863_5fa9),
    (0x7f7f_ffff, 0xbf05_99b3, 0x3f5a_5f96, 0x5f7f_ffff),
    (0xff7f_ffff, 0x3f05_99b3, 0x3f5a_5f96, 0x7fc0_0000),
    (0x7f80_0000, 0x7fc0_0000, 0x7fc0_0000, 0x7f80_0000),
    (0xff80_0000, 0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000),
    (0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000),
    (0xffc0_0001, 0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000),
    (0x7f80_0001, 0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000),
    (0xff80_0001, 0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000),
    (0x4080_0000, 0xbf41_bdcf, 0xbf27_5530, 0x4000_0000),
    (0x4000_0000, 0x3f68_c7b7, 0xbed5_1132, 0x3fb5_04f3),
    (0x3f80_0001, 0x3f57_6aa6, 0x3f0a_513f, 0x3f80_0000),
    (0xc080_0000, 0x3f41_bdcf, 0xbf27_5530, 0x7fc0_0000),
    // Nearest to k pi (sin) and odd k pi/2 (cos), largest ULP distance and
    // real-valued error per function (exhaustive sweep), and a huge argument.
    (0x6ff9_be45, 0xb15d_eea9, 0xbf80_0000, 0x57b2_cb26),
    (0x6f79_be45, 0x3f80_0000, 0xb0dd_eea9, 0x577c_da2e),
    (0x442b_2762, 0xbe7f_2b3a, 0x3f77_ecac, 0x41d1_523d),
    (0x4416_6f5f, 0xbf7d_ff31, 0x3dff_b333, 0x41c4_3e3f),
    (0x51ed_f31e, 0xbeeb_b5a0, 0x3f63_4215, 0x48ae_855c),
    (0x75d1_3c2d, 0xbf62_a5f4, 0xbeee_0bf8, 0x5aa3_a700),
    (0x5b17_a8b0, 0xbf7f_b9e1, 0xbd3d_6d0f, 0x4d45_0a31),
];

/// FNV digests of (sin, cos, sqrt) over the `sample()` inputs.
const SAMPLE_DIGEST: [u64; 3] = [
    0x9caa_85f4_900e_d22a,
    0x0c1a_0a79_eca8_9fe6,
    0xe213_b56e_4f37_5c7c,
];

/// FNV block digests of (sin, cos, sqrt) over all 2^32 inputs, recorded by
/// `p0a_trig_exhaustive_cpu` on the exact-checked host backend.
const FULL_DIGEST: [u64; 3] = [
    0x6507_8771_be70_cedb,
    0x3ad6_c172_a44f_4ce9,
    0x1df9_25c7_3304_7db5,
];

fn sample() -> impl Iterator<Item = u32> {
    (0u32..1 << 18).map(|i| i.wrapping_mul(0x9e37_79b1) ^ (i >> 7))
}

fn sample_digests() -> [u64; 3] {
    let mut digests = [sweep::FNV_SEED; 3];
    for x in sample() {
        for (f, digest) in digests.iter_mut().enumerate() {
            *digest = sweep::fnv(*digest, u64::from(sweep::host(f, x)), 4);
        }
    }
    digests
}

fn oracle(f: usize, x: u32) -> u32 {
    match f {
        0 => F32Bits::from_bits(x).sin().to_bits(),
        1 => F32Bits::from_bits(x).cos().to_bits(),
        _ => F32Bits::from_bits(x).sqrt_sequence().to_bits(),
    }
}

#[test]
fn p0a_trig_golden_hard_cases_hold_on_the_oracle_and_the_host() {
    assert!(GOLDEN.len() >= 40);
    for &(x, sin, cos, root) in GOLDEN {
        for (f, (want, name)) in [sin, cos, root]
            .into_iter()
            .zip(sweep::FUNCTIONS)
            .enumerate()
        {
            assert_eq!(oracle(f, x), want, "oracle {name} {x:#010x}");
            assert_eq!(sweep::host(f, x), want, "host {name} {x:#010x}");
        }
        assert_eq!(
            F32Bits::from_bits(x).sqrt().to_bits(),
            F32Bits::from_bits(x).sqrt_sequence().to_bits(),
            "{x:#010x}"
        );
    }
}

#[test]
fn p0a_trig_special_values_follow_the_contract() {
    for x in [
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0xffc0_0001,
        0x7f80_0001,
        0xff80_0001,
    ] {
        assert_eq!(oracle(0, x), CANONICAL_NAN);
        assert_eq!(oracle(1, x), CANONICAL_NAN);
    }
    for x in [
        0x7fc0_0000,
        0xffc0_0001,
        0x7f80_0001,
        0xff80_0000,
        0xbf80_0000,
        0x8000_0001,
    ] {
        assert_eq!(oracle(2, x), CANONICAL_NAN, "{x:#010x}");
    }
    assert_eq!(oracle(2, 0x7f80_0000), 0x7f80_0000);
    for zero in [0, 0x8000_0000] {
        assert_eq!(
            (oracle(0, zero), oracle(1, zero), oracle(2, zero)),
            (zero, 0x3f80_0000, zero)
        );
    }
    // Subnormal arguments stay exact: sin(x) = x and cos(x) = 1 correctly rounded.
    for x in [1, 0x8000_0001, 0x007f_ffff, 0x807f_ffff] {
        assert_eq!((oracle(0, x), oracle(1, x)), (x, 0x3f80_0000));
    }
}

#[test]
fn p0a_trig_sampled_digest_is_frozen_and_the_oracle_agrees_on_a_stride() {
    assert_eq!(sample_digests(), SAMPLE_DIGEST);
    for x in sample().step_by(64) {
        for f in 0..3 {
            assert_eq!(
                oracle(f, x),
                sweep::host(f, x),
                "{} {x:#010x}",
                sweep::FUNCTIONS[f]
            );
        }
    }
}

/// Exact decimal spelling of a binary32 value for source literals.
fn literal(bits: u32) -> String {
    let text = format!("{}", f32::from_bits(bits).abs());
    let text = if text.contains('.') {
        text
    } else {
        format!("{text}.0")
    };
    if bits & 0x8000_0000 != 0 {
        format!("-{text}")
    } else {
        text
    }
}

/// A source predicate that holds exactly when `name` has binary32 bits
/// `bits`, up to the NaN payload the language cannot observe.
fn exact(name: &str, bits: u32) -> String {
    let value = f32::from_bits(bits);
    if value.is_nan() {
        format!("{name} != {name}")
    } else if value.is_infinite() {
        format!("{name} == {}1.0 / 0.0", if value < 0.0 { "-" } else { "" })
    } else if value == 0.0 {
        let sign = if bits >> 31 == 1 { "<" } else { ">" };
        format!("{name} == 0.0 and 1.0 / {name} {sign} 0.0")
    } else {
        format!("{name} == {}", literal(bits))
    }
}

fn argument(bits: u32) -> String {
    let value = f32::from_bits(bits);
    if value.is_nan() {
        "0.0 / 0.0".into()
    } else if value.is_infinite() {
        format!("{}1.0 / 0.0", if value < 0.0 { "-" } else { "" })
    } else {
        literal(bits)
    }
}

fn qualify(source: &str) {
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = invar::mir::lower(&checked).unwrap();
    let complete = invar::eval::observe(&checked, 100_000);
    assert_eq!(complete.result, Ok(invar::eval::Value::I32(7)), "{source}");
    let mirrored = invar::mir_eval::observe(&program, 100_000);
    assert_eq!(mirrored.result, complete.result, "{source}");
    assert_eq!(mirrored.steps_used, complete.steps_used, "{source}");
    #[cfg(feature = "native")]
    {
        let (status, doc) = invar::native::run_program(&program, 100_000).unwrap();
        assert_eq!(
            (status, &doc["result"]["value"]),
            (0, &serde_json::json!(7)),
            "{doc}: {source}"
        );
        assert_eq!(doc["steps_used"], complete.steps_used, "{source}");
    }
}

#[test]
fn p0a_trig_source_intrinsics_agree_on_eval_mir_and_native() {
    // One program per golden row keeps a failure attributable to one input;
    // the NaN payload is pinned by the bit-level tests instead.
    for &(x, sin, cos, root) in GOLDEN {
        let source = format!(
            "fn main() -> i32 {{ let x: f32 = {}; let s = sin(x); let c = cos(x); let r = sqrt(x); if {} and {} and {} {{ return 7; }} return 0; }}",
            argument(x),
            exact("s", sin),
            exact("c", cos),
            exact("r", root),
        );
        qualify(&source);
    }
}

#[test]
fn p0a_trig_source_rejects_wrong_arity_and_non_f32_arguments() {
    for source in [
        "fn main() -> i32 { let x = sin(1.0, 2.0); return 0; }",
        "fn main() -> i32 { let x = cos(1u32); return 0; }",
        "fn main() -> i32 { let x = sqrt(1); return 0; }",
        "fn main() -> i32 { let x: i32 = sin(1.0); return 0; }",
    ] {
        let parsed = invar::front::parse(source.as_bytes()).unwrap();
        assert!(invar::check::check(parsed, true).is_err(), "{source}");
    }
}

fn unary(node: fn(usize) -> Node, length: u32) -> Program {
    Program {
        length,
        inputs: vec![Scalar::F32],
        nodes: vec![Node::Input(0), node(0)],
        kind: Kind::Map,
    }
}

const NODES: [fn(usize) -> Node; 3] = [Node::Sin, Node::Cos, Node::Sqrt];

#[test]
fn p0a_trig_source_regions_and_device_plans_match_the_host() {
    let inputs: Vec<u32> = GOLDEN.iter().map(|row| row.0).collect();
    for (f, node) in NODES.iter().enumerate() {
        let plan = ops::plan(
            unary(*node, inputs.len() as u32),
            invar_device::Capabilities::cuda_12_9_sm89(),
        );
        let expected: Vec<u32> = inputs.iter().map(|&x| sweep::host(f, x)).collect();
        for workers in [1, 3] {
            assert_eq!(
                invar::f32_source::cpu(&plan, &inputs, workers).unwrap(),
                expected
            );
        }
    }
    let source = "fn kernel(a: read Array<f32>, n: u32) -> Array<f32> { var out = filled_f32(n, 0.0); for i in 0u32..n { out[i] = fma(sin(a[i]), cos(a[i]), sqrt(a[i])); } return out; } fn main() -> i32 { let a: Array<f32> = [0.5, 3.0, 100000.0, 1000000000000000000000000000000.0, 0.0]; let out = kernel(a, 5u32); for i in 0u32..5u32 { if out[i] != fma(sin(a[i]), cos(a[i]), sqrt(a[i])) { return 0; } } return 7; }";
    let expected = ops::reference(source);
    assert_eq!(expected.result, Ok(invar::eval::Value::I32(7)));
    for workers in [1, 4] {
        ops::parity(
            &invar::f32_source::run_cpu_source(source.as_bytes(), 1_000_000, workers).unwrap(),
            &expected,
        );
    }
}

#[test]
fn p0a_trig_device_sources_embed_the_shared_helpers_within_the_size_limit() {
    let all = Program {
        length: 1,
        inputs: vec![Scalar::F32],
        nodes: vec![Node::Input(0), Node::Sin(0), Node::Cos(1), Node::Sqrt(2)],
        kind: Kind::Map,
    };
    for caps in [
        invar_device::Capabilities::cuda_12_9_sm89(),
        invar_device::Capabilities::Metal {
            apple_family: 7,
            msl_major: 3,
            msl_minor: 1,
        },
    ] {
        let source = f32_ops::emit(&ops::plan(all.clone(), caps)).unwrap();
        assert!(
            source.len() <= invar_device::MAX_SOURCE_BYTES,
            "{}",
            source.len()
        );
        for needle in [
            trig_f32::CONTRACT,
            "invar_trig_f32_v1(",
            "invar_sqrt_f32_v1(",
            "invar_trig_f32_v1_four_over_pi",
        ] {
            assert!(source.contains(needle), "{needle}");
        }
        let plain = f32_ops::emit(&ops::plan(unary(|_| Node::Neg(0), 1), caps)).unwrap();
        assert!(!plain.contains(trig_f32::CONTRACT));
    }
}

#[test]
fn p0a_trig_admission_tolerates_only_subnormal_probe_mismatches() {
    use invar_device::f32_probe::{CASES, Op, Report};
    let words = |broken: &dyn Fn(usize) -> bool| -> Vec<u32> {
        CASES
            .iter()
            .enumerate()
            .flat_map(|(i, case)| {
                [
                    if broken(i) {
                        case.expected ^ 1
                    } else {
                        case.expected
                    },
                    0,
                ]
            })
            .collect()
    };
    let trig = unary(Node::Sin, 1);
    let root = unary(Node::Sqrt, 1);
    let plain = unary(|_| Node::Neg(0), 1);
    let clean = Report::evaluate(&words(&|_| false)).unwrap();
    for p in [&trig, &root, &plain] {
        assert!(f32_ops::admits_trig(p, &clean));
    }
    let subnormal = Report::evaluate(&words(&|i| CASES[i].subnormal)).unwrap();
    assert!(f32_ops::admits_trig(&trig, &subnormal) && f32_ops::admits_trig(&root, &subnormal));
    for (op, p) in [
        (Op::Fma, &trig),
        (Op::Add, &trig),
        (Op::Mul, &trig),
        (Op::Sqrt, &root),
    ] {
        let broken =
            Report::evaluate(&words(&|i| CASES[i].op == op && !CASES[i].subnormal)).unwrap();
        assert!(!f32_ops::admits_trig(p, &broken), "{op:?}");
        assert!(f32_ops::admits_trig(&plain, &broken), "{op:?}");
    }
}

#[cfg(feature = "native")]
#[test]
fn p0a_trig_native_ir_embeds_the_helpers_without_fast_math() {
    let lower = |source: &str| {
        let checked =
            invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
        invar::native::emit_mir(&invar::mir::lower(&checked).unwrap())
            .unwrap()
            .ir
    };
    let ir = lower(
        "fn main() -> i32 { let x = 3.0; if sin(x) + cos(x) + sqrt(x) > 0.0 { return 7; } return 0; }",
    );
    for needle in [
        "@invar_trig_f32_v1(",
        "@invar_sqrt_f32_v1(",
        "@invar_trig_f32_v1_four_over_pi",
        "call float @llvm.fma.f32",
        "call float @llvm.sqrt.f32",
        "\"denormal-fp-math-f32\"=\"ieee,ieee\"",
    ] {
        assert!(ir.contains(needle), "{needle}");
    }
    for flag in [
        " fast ",
        " contract ",
        " reassoc ",
        " afn ",
        " nnan ",
        " ninf ",
        " nsz ",
        " arcp ",
    ] {
        assert!(!ir.contains(flag), "{flag}");
    }
    let plain = lower("fn main() -> i32 { let x = 3.0; if x * x > 0.0 { return 7; } return 0; }");
    // The float header declares every intrinsic (#173); a plain program calls none.
    assert!(!plain.contains("invar_trig_f32_v1") && !plain.contains("call float @llvm.sqrt.f32"));
}

/// Every input once on the host backend (instrumented), checked against the
/// exact oracle on a stride, against libm for the ULP bound, and audited for
/// subnormal operands. `INVAR_TRIG_ORACLE_STRIDE` (default 257) and
/// `INVAR_TRIG_THREADS` bound the cost.
#[test]
#[ignore = "exhaustive 2^32 sweep; run on staging"]
fn p0a_trig_exhaustive_cpu() {
    let stride: u64 = std::env::var("INVAR_TRIG_ORACLE_STRIDE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(257);
    let report = exhaustive::run(stride);
    println!("{}", report.json());
    let digests: Vec<u64> = (0..3).map(|f| sweep::combine(&report.blocks[f])).collect();
    assert_eq!(report.oracle_mismatches, 0);
    assert_eq!(report.sqrt_not_correctly_rounded, 0);
    assert_eq!(report.nan_policy_violations, 0);
    assert_eq!(report.audit.underflows, 0);
    assert!(report.audit.smallest >= f64::from(f32::MIN_POSITIVE));
    assert_eq!(digests, FULL_DIGEST);
}

mod exhaustive {
    use super::sweep::{self, Audit};
    use invar_device::trig_f32;

    #[derive(Clone, Default)]
    pub struct Error {
        pub max_ulp_int: [u64; 2],
        pub max_ulp_int_input: [u32; 2],
        pub max_error: [f64; 2],
        pub max_error_input: [u32; 2],
        /// Per function: [|x| < 2^20 max error, |x| >= 2^20 max error].
        pub split: [[f64; 2]; 2],
        pub histogram: [[u64; 4]; 2],
        pub ambiguous: u64,
        /// Smallest |sin| (near k pi) and |cos| (near odd k pi/2) seen for |x| >= 1.
        pub nearest: [(f64, u32); 2],
    }

    pub struct Report {
        pub blocks: [Vec<u64>; 3],
        pub error: Error,
        pub audit: Audit,
        pub oracle_checked: u64,
        pub oracle_mismatches: u64,
        pub sqrt_not_correctly_rounded: u64,
        pub nan_policy_violations: u64,
        pub seconds: f64,
    }

    struct Block {
        index: u32,
        digests: [u64; 3],
        error: Error,
        audit: Audit,
        oracle_checked: u64,
        oracle_mismatches: u64,
        sqrt_bad: u64,
        nan_bad: u64,
    }

    fn block(index: u32, stride: u64) -> Block {
        let mut out = Block {
            index,
            digests: [sweep::FNV_SEED; 3],
            error: Error {
                nearest: [(f64::INFINITY, 0); 2],
                ..Error::default()
            },
            audit: Audit::default(),
            oracle_checked: 0,
            oracle_mismatches: 0,
            sqrt_bad: 0,
            nan_bad: 0,
        };
        let first = index << sweep::BLOCK_BITS;
        for offset in 0..1u32 << sweep::BLOCK_BITS {
            let x = first | offset;
            out.audit.input = x;
            let results = [
                trig_f32::sincos(&mut out.audit, x, 0),
                trig_f32::sincos(&mut out.audit, x, 1),
                trig_f32::sqrt(&mut out.audit, x),
            ];
            for (digest, &r) in out.digests.iter_mut().zip(&results) {
                *digest = sweep::fnv(*digest, u64::from(r), 4);
            }
            if u64::from(x) % stride == 0 {
                out.oracle_checked += 1;
                for (f, &r) in results.iter().enumerate() {
                    if super::oracle(f, x) != r {
                        out.oracle_mismatches += 1;
                    }
                }
            }
            check(&mut out, x, results);
        }
        out
    }

    fn check(out: &mut Block, x: u32, [sin, cos, root]: [u32; 3]) {
        let value = f32::from_bits(x);
        let wide = f64::from(value);
        // sqrt: f64 sqrt rounded to binary32 is correctly rounded (53 >= 2 * 24 + 2).
        let expected_root = if value.is_nan() || (value < 0.0) {
            trig_f32::CANONICAL_NAN
        } else {
            (wide.sqrt() as f32).to_bits()
        };
        if root != expected_root {
            out.sqrt_bad += 1;
        }
        if !value.is_finite() {
            if sin != trig_f32::CANONICAL_NAN || cos != trig_f32::CANONICAL_NAN {
                out.nan_bad += 1;
            }
            return;
        }
        let e = &mut out.error;
        for (f, (result, reference)) in [(sin, wide.sin()), (cos, wide.cos())]
            .into_iter()
            .enumerate()
        {
            if value.abs() >= 1.0 && reference.abs() < e.nearest[f].0 {
                e.nearest[f] = (reference.abs(), x);
            }
            let error =
                (f64::from(f32::from_bits(result)) - reference).abs() / sweep::ulp32(reference);
            if error > e.max_error[f] {
                e.max_error[f] = error;
                e.max_error_input[f] = x;
            }
            let side = usize::from(value.abs() > 1_048_576.0);
            e.split[f][side] = e.split[f][side].max(error);
            match sweep::correctly_rounded(reference) {
                None => e.ambiguous += 1,
                Some(cr) => {
                    let d = sweep::key(result).abs_diff(sweep::key(cr));
                    e.histogram[f][(d as usize).min(3)] += 1;
                    if d > e.max_ulp_int[f] {
                        e.max_ulp_int[f] = d;
                        e.max_ulp_int_input[f] = x;
                    }
                }
            }
        }
    }

    fn merge(into: &mut Error, from: &Error) {
        for f in 0..2 {
            if from.max_ulp_int[f] > into.max_ulp_int[f] {
                into.max_ulp_int[f] = from.max_ulp_int[f];
                into.max_ulp_int_input[f] = from.max_ulp_int_input[f];
            }
            if from.max_error[f] > into.max_error[f] {
                into.max_error[f] = from.max_error[f];
                into.max_error_input[f] = from.max_error_input[f];
            }
            for side in 0..2 {
                into.split[f][side] = into.split[f][side].max(from.split[f][side]);
            }
            for d in 0..4 {
                into.histogram[f][d] += from.histogram[f][d];
            }
            if from.nearest[f].0 < into.nearest[f].0 {
                into.nearest[f] = from.nearest[f];
            }
        }
        into.ambiguous += from.ambiguous;
    }

    pub fn run(stride: u64) -> Report {
        let started = std::time::Instant::now();
        let next = std::sync::atomic::AtomicU32::new(0);
        let mut blocks: Vec<Block> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..sweep::threads())
                .map(|_| {
                    scope.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if index >= sweep::BLOCKS {
                                return done;
                            }
                            done.push(block(index, stride));
                        }
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|w| w.join().unwrap())
                .collect()
        });
        blocks.sort_by_key(|b| b.index);
        let mut report = Report {
            blocks: [Vec::new(), Vec::new(), Vec::new()],
            error: Error {
                nearest: [(f64::INFINITY, 0); 2],
                ..Error::default()
            },
            audit: Audit::default(),
            oracle_checked: 0,
            oracle_mismatches: 0,
            sqrt_not_correctly_rounded: 0,
            nan_policy_violations: 0,
            seconds: 0.0,
        };
        for b in &blocks {
            for f in 0..3 {
                report.blocks[f].push(b.digests[f]);
            }
            merge(&mut report.error, &b.error);
            report.audit.merge(&b.audit);
            report.oracle_checked += b.oracle_checked;
            report.oracle_mismatches += b.oracle_mismatches;
            report.sqrt_not_correctly_rounded += b.sqrt_bad;
            report.nan_policy_violations += b.nan_bad;
        }
        report.seconds = started.elapsed().as_secs_f64();
        report
    }

    impl Report {
        pub fn json(&self) -> String {
            let e = &self.error;
            let digests: Vec<String> = (0..3)
                .map(|f| format!("\"{:#018x}\"", sweep::combine(&self.blocks[f])))
                .collect();
            format!(
                "{{\"contract\":\"{}\",\"inputs\":4294967296,\"seconds\":{:.1},\"digests\":[{}],\
\"oracle_checked\":{},\"oracle_mismatches\":{},\"sqrt_not_correctly_rounded\":{},\"nan_policy_violations\":{},\
\"sin\":{{\"max_ulp_vs_correctly_rounded\":{},\"at\":\"{:#010x}\",\"max_error_ulp\":{:.4},\"at_error\":\"{:#010x}\",\"max_error_below_2p20\":{:.4},\"max_error_from_2p20\":{:.4},\"ulp_histogram_0_1_2_3plus\":{:?},\"nearest_zero\":[{:e},\"{:#010x}\"]}},\
\"cos\":{{\"max_ulp_vs_correctly_rounded\":{},\"at\":\"{:#010x}\",\"max_error_ulp\":{:.4},\"at_error\":\"{:#010x}\",\"max_error_below_2p20\":{:.4},\"max_error_from_2p20\":{:.4},\"ulp_histogram_0_1_2_3plus\":{:?},\"nearest_zero\":[{:e},\"{:#010x}\"]}},\
\"ambiguous_reference\":{},\"audit\":{{\"smallest_nonzero_magnitude\":{:e},\"at\":\"{:#010x}\",\"underflows_to_zero\":{},\"min_normal\":{:e}}}}}",
                trig_f32::CONTRACT,
                self.seconds,
                digests.join(","),
                self.oracle_checked,
                self.oracle_mismatches,
                self.sqrt_not_correctly_rounded,
                self.nan_policy_violations,
                e.max_ulp_int[0],
                e.max_ulp_int_input[0],
                e.max_error[0],
                e.max_error_input[0],
                e.split[0][0],
                e.split[0][1],
                e.histogram[0],
                e.nearest[0].0,
                e.nearest[0].1,
                e.max_ulp_int[1],
                e.max_ulp_int_input[1],
                e.max_error[1],
                e.max_error_input[1],
                e.split[1][0],
                e.split[1][1],
                e.histogram[1],
                e.nearest[1].0,
                e.nearest[1].1,
                e.ambiguous,
                self.audit.smallest,
                self.audit.smallest_input,
                self.audit.underflows,
                f64::from(f32::MIN_POSITIVE),
            )
        }
    }
}

/// Builds the native helper object, links it into a host harness that also
/// compiles `trig_f32.rs` (the single source of truth) and compares every
/// input. `INVAR_TRIG_BLOCKS_PER_RUN` bounds each harness process.
#[cfg(feature = "native")]
#[test]
#[ignore = "exhaustive 2^32 native sweep; run on staging"]
fn p0a_trig_exhaustive_native() {
    let object = invar::native::trig_sweep_object().unwrap();
    let dir = std::env::temp_dir().join(format!("invar-p0a-native-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let trig = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/crates/invar-device/src/trig_f32.rs"
    );
    let host = format!(
        "{}\n#[path = {trig:?}]\n#[allow(dead_code)]\nmod trig;\n",
        include_str!("issue_p0a_trig_f32/native_harness.rs")
    );
    std::fs::write(dir.join("helpers.o"), object).unwrap();
    std::fs::write(dir.join("harness.rs"), host).unwrap();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let built = std::process::Command::new(rustc)
        .args(["--edition=2024", "-C", "opt-level=2", "-C", "panic=abort"])
        .arg(dir.join("harness.rs"))
        .arg("-C")
        .arg(format!("link-arg={}", dir.join("helpers.o").display()))
        .arg("-o")
        .arg(dir.join("harness"))
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let per_run: u32 = std::env::var("INVAR_TRIG_BLOCKS_PER_RUN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let mut blocks = [Vec::new(), Vec::new(), Vec::new()];
    let started = std::time::Instant::now();
    for first in (0..sweep::BLOCKS).step_by(per_run as usize) {
        let output = std::process::Command::new(dir.join("harness"))
            .args([
                first.to_string(),
                per_run.to_string(),
                sweep::threads().to_string(),
            ])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{text}{}",
            String::from_utf8_lossy(&output.stderr)
        );
        for line in text.lines() {
            // "block <index> <sin> <cos> <sqrt> mismatches <n> first <x>"
            let words: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(words[6], "0", "native differs from the host: {line}");
            for f in 0..3 {
                blocks[f].push(u64::from_str_radix(words[2 + f], 16).unwrap());
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    let digests: Vec<u64> = (0..3).map(|f| sweep::combine(&blocks[f])).collect();
    println!(
        "{{\"target\":\"native-llvm\",\"inputs\":4294967296,\"seconds\":{:.1},\"mismatches\":0,\"digests\":[\"{:#018x}\",\"{:#018x}\",\"{:#018x}\"]}}",
        started.elapsed().as_secs_f64(),
        digests[0],
        digests[1],
        digests[2]
    );
    assert_eq!(digests, FULL_DIGEST);
}

/// The production MSL emitter over every input in 1M-element map dispatches.
/// `INVAR_TRIG_FIRST_CHUNK` and `INVAR_TRIG_CHUNKS` select a bounded slice of
/// the 4295 chunks; the digest is asserted only for a complete sweep.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal GPU; exhaustive"]
fn p0a_trig_exhaustive_metal() {
    let mut device = invar_metal::MetalDevice::open_runtime_source().unwrap();
    let caps = device.capabilities();
    // The admitted production path first, on the golden rows.
    let inputs: Vec<u32> = GOLDEN.iter().map(|row| row.0).collect();
    for (f, node) in NODES.iter().enumerate() {
        let plan = ops::plan(unary(*node, inputs.len() as u32), caps);
        let expected: Vec<u32> = inputs.iter().map(|&x| sweep::host(f, x)).collect();
        assert_eq!(
            device.run_f32_ops(&plan, &inputs).unwrap(),
            expected,
            "{}",
            sweep::FUNCTIONS[f]
        );
    }
    device_sweep(
        "metal",
        |plan, input, source| {
            let run = device.diagnose_f32_ops_source(plan, input, source).unwrap();
            assert!(!run.flagged);
            run.output
        },
        caps,
    );
}

/// Pending CUDA hardware: the production CUDA emitter over every input.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA GPU; exhaustive"]
fn p0a_trig_exhaustive_cuda() {
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let mut device =
        invar_cuda::cuda::CudaDevice::open().expect("a real CUDA GPU with a matching NVRTC");
    let caps = device.capabilities();
    device_sweep(
        "cuda",
        |plan, input, _source| device.run_f32_ops(plan, input, compiler).unwrap(),
        caps,
    );
}

#[allow(dead_code)]
fn device_sweep(
    target: &str,
    mut run: impl FnMut(&invar_device::v2::Verified, &[u32], &str) -> Vec<u32>,
    caps: invar_device::Capabilities,
) {
    let chunk = f32_ops::MAX_LENGTH as u64;
    let total = (1u64 << 32).div_ceil(chunk);
    let env = |name: &str, default: u64| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let first = env("INVAR_TRIG_FIRST_CHUNK", 0);
    let count = env("INVAR_TRIG_CHUNKS", total).min(total - first);
    let started = std::time::Instant::now();
    let mut mismatches = [0u64; 3];
    let mut checked = 0u64;
    let mut digests = [sweep::FNV_SEED; 3];
    for (f, node) in NODES.iter().enumerate() {
        let mut planned: Option<(u32, invar_device::v2::Verified, String)> = None;
        for c in first..first + count {
            let start = c * chunk;
            let length = chunk.min((1u64 << 32) - start) as u32;
            if planned.as_ref().is_none_or(|p| p.0 != length) {
                let plan = ops::plan(unary(*node, length), caps);
                let source = f32_ops::emit(&plan).unwrap();
                planned = Some((length, plan, source));
            }
            let (_, plan, source) = planned.as_ref().unwrap();
            let input: Vec<u32> = (0..length)
                .map(|i| (start as u32).wrapping_add(i))
                .collect();
            let output = run(plan, &input, source);
            for (&x, &got) in input.iter().zip(&output) {
                digests[f] = sweep::fnv(digests[f], u64::from(got), 4);
                let want = sweep::host(f, x);
                if got != want {
                    if mismatches[f] < 8 {
                        eprintln!(
                            "{target} {} {x:#010x}: device {got:#010x} host {want:#010x}",
                            sweep::FUNCTIONS[f]
                        );
                    }
                    mismatches[f] += 1;
                }
            }
            checked += u64::from(length);
        }
    }
    println!(
        "{{\"target\":\"{target}\",\"first_chunk\":{first},\"chunks\":{count},\"checked\":{checked},\"seconds\":{:.1},\"mismatches\":{mismatches:?},\"stream_digests\":[\"{:#018x}\",\"{:#018x}\",\"{:#018x}\"]}}",
        started.elapsed().as_secs_f64(),
        digests[0],
        digests[1],
        digests[2]
    );
    assert_eq!(mismatches, [0; 3]);
}
