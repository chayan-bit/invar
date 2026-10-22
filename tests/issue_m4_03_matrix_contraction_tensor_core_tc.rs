//! M4-03 `tc-f16-f32-v1` without a device: the explicit `tc_mul_add` source
//! contract (admitted only as the step of a recognized matrix contraction),
//! the Tensor Core / `simdgroup_matrix` emission from the verified plan, the
//! exact-sum reference and the amended D5 bound
//! `|r - exact| <= 2 * K * 2^-23 * sum |a_ik * b_kj|` (decided 2026-10-14).
#![forbid(unsafe_code)]

#[allow(dead_code)]
#[path = "../src/contraction_source.rs"]
mod contraction_source;
#[path = "issue_m4_03_device/corpus.rs"]
mod corpus;

use contraction_source::{Contract, DeviceEmission, Error};
use corpus::{DIRECT, SHAPES, TC_DIRECT};
use invar::{check, front, general_net, hex, layout_contract, mir, numeric_contract, tensor_cache};
use invar_device::{Capabilities, Provenance, floating, gemm, tc};
use num_bigint::BigInt;
use num_rational::BigRational;
use numeric_contract::{F32Bits, NumericPolicy};

const METAL: Capabilities = Capabilities::Metal {
    apple_family: 9,
    msl_major: 3,
    msl_minor: 1,
};
const BLACKWELL: Capabilities = Capabilities::Cuda {
    major: 12,
    minor: 9,
    sm: 120,
};

fn tc_contract() -> Contract {
    Contract::from_source(TC_DIRECT.as_bytes(), "mat").unwrap()
}
fn lengths([r, k, n]: [u32; 3]) -> [u64; 2] {
    [u64::from(r) * u64::from(k), u64::from(k) * u64::from(n)]
}
fn emit(dims: [u32; 3], caps: Capabilities) -> DeviceEmission {
    tc_contract()
        .emit_device(TC_DIRECT.as_bytes(), dims, lengths(dims), caps)
        .unwrap()
}
fn source_error(source: &str) -> &'static str {
    match Contract::from_source(source.as_bytes(), "mat") {
        Err(Error::Source(e)) => e.code,
        other => panic!("expected a source error, got {other:?}"),
    }
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_tc_mul_add_only_in_a_recognized_contraction() {
    // The strict policy keeps its pre-TC `Debug` spelling, which the frozen
    // general-net receipt snapshot embeds; only the TC policy names `tc`.
    assert_eq!(
        format!("{:?}", NumericPolicy::STRICT_F32),
        "NumericPolicy { version: 1 }"
    );
    assert_eq!(
        format!("{:?}", NumericPolicy::TC_F16_F32),
        "NumericPolicy { version: 1, tc: true }"
    );
    assert_eq!(tc_contract().policy(), NumericPolicy::TC_F16_F32);
    assert_eq!(tc_contract().policy().id(), "tc-f16-f32-v1");
    let strict = Contract::from_source(DIRECT.as_bytes(), "mat").unwrap();
    assert_eq!(strict.policy(), NumericPolicy::STRICT_F32);
    // Ordinary checking admits exactly the recognized contraction step.
    let ordinary = check::check(front::parse(TC_DIRECT.as_bytes()).unwrap(), false);
    ordinary.unwrap();
    let step = "c[i * cols + j] = tc_mul_add(a[i * inner + k], b[k * cols + j], c[i * cols + j]);";
    for refused in [
        // Scalar accumulator: not this recognized family.
        TC_DIRECT.replace(
            step,
            "var s = 0.0; s = tc_mul_add(a[i * inner + k], b[k * cols + j], s); c[i * cols + j] = s;",
        ),
        // Accumulator first (D2 spells `tc_mul_add(a, b, acc)`).
        TC_DIRECT.replace(
            step,
            "c[i * cols + j] = tc_mul_add(c[i * cols + j], a[i * inner + k], b[k * cols + j]);",
        ),
        // Transposed operand, nonzero start, and a second use elsewhere.
        TC_DIRECT.replace("b[k * cols + j], c", "b[j * cols + k], c"),
        TC_DIRECT.replace("rows * cols,0.0", "rows * cols,1.0"),
        format!(
            "{TC_DIRECT} fn other(x: f32, y: f32) -> f32 {{ return tc_mul_add(x, y, 0.0); }}"
        ),
        // A strict step mixed with the Tensor Core step.
        TC_DIRECT.replace(
            "c[i * cols + j]);",
            "c[i * cols + j]) + a[i * inner + k] * b[k * cols + j];",
        ),
    ] {
        assert_eq!(source_error(&refused), "UNSUPPORTED_FEATURE", "{refused}");
    }
    // Valid ordinary source with a missing entry is an extraction miss.
    assert!(matches!(
        Contract::from_source(TC_DIRECT.as_bytes(), "absent"),
        Err(Error::Unrecognized)
    ));
    // The name is reserved like every other intrinsic.
    assert_eq!(
        source_error("fn tc_mul_add(x: f32) -> f32 { return x; } fn mat() -> f32 { return 0.0; }"),
        "NAME_DUPLICATE"
    );
    // Distinct policy, distinct identity: the strict and TC kernels never share a key.
    let dims = [65, 17, 63];
    let strict_key = strict
        .emit_device(DIRECT.as_bytes(), dims, lengths(dims), BLACKWELL)
        .unwrap();
    assert_ne!(
        strict_key.key().to_hex(),
        emit(dims, BLACKWELL).key().to_hex()
    );
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_tc_plans_emit_only_tensor_core_kernels() {
    for (dims, cuda_tiles, cuda_groups, metal_tiles) in [
        ([65u32, 17, 63], 40u64, 10u32, 72u64),
        ([1, 1, 1], 1, 1, 1),
        ([16, 16, 8], 1, 1, 2),
        ([129, 31, 130], 153, 39, 289),
    ] {
        let cuda = emit(dims, BLACKWELL);
        let metal = emit(dims, METAL);
        assert_eq!(cuda.plan().ir(), metal.plan().ir());
        assert!(cuda.plan().is_tc() && metal.plan().is_tc());
        assert_eq!(cuda.plan().ir().numeric_policy, "tc-f16-f32-v1");
        assert_eq!(cuda.target(), "cuda-nvrtc-c++17-tc-mma-m16n8k16-v1");
        assert_eq!(metal.target(), "metal-msl-3.1-tc-simdgroup8x8-v1");
        assert_eq!(
            (
                cuda.geometry().tiles,
                cuda.geometry().groups,
                cuda.geometry().threads
            ),
            (cuda_tiles, cuda_groups, 128)
        );
        assert_eq!(
            (metal.geometry().tiles, metal.geometry().threads),
            (metal_tiles, 32)
        );
        let body = cuda.source();
        for needed in [
            "numeric=tc-f16-f32-v1",
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
            "cvt.rn.f16x2.f32",
            "__launch_bounds__(128)",
            "--fmad=false",
            tc::CUDA_SCHEDULE,
            // Second D5 amendment: subnormal operands are zeroed for the mma
            // and their exact products added on CUDA cores, per voted k-tile.
            "m != 0u && m < 0x38800000u",
            "invar_sub(lo) ? 0.0f : lo",
            "__any_sync(0xffffffffu, sub)",
            "__fadd_rn(s, __fmul_rn(x, y))",
            "invar_c(C, r0, j, __fadd_rn(c0, s0));",
        ] {
            assert!(body.contains(needed), "{needed}");
        }
        for banned in ["fmaf", "__fmaf", "wmma", "tf32", "__half"] {
            assert!(!body.contains(banned), "{banned}");
        }
        let msl = metal.source();
        for needed in [
            "numeric=tc-f16-f32-v1",
            "simdgroup_float8x8",
            "simdgroup_multiply_accumulate(acc, a, b, acc)",
            tc::METAL_SCHEDULE,
        ] {
            assert!(msl.contains(needed), "{needed}");
        }
        assert!(!msl.contains("invar_mul(") && !msl.contains("half"));
        for e in [&cuda, &metal] {
            let r = e.receipt();
            assert_eq!(r["numeric_policy"], "tc-f16-f32-v1");
            assert_eq!(r["reference"], "exact-sum-of-f16-rounded-products-i128");
            assert_eq!(
                r["bound"],
                "|r - exact| <= 2 * K * 2^-23 * sum |a_ik * b_kj|"
            );
            assert_eq!(r["track"], "generated");
            assert!(e.source().len() <= invar_device::MAX_SOURCE_BYTES);
        }
    }
    // No CPU placement for a Tensor Core plan: there is no CPU lowering and
    // the strict source loop would run unrounded f32 inputs. The same small,
    // empty and K=0 shapes keep their CPU placement under strict-f32-v1.
    for (dims, strict_reason) in [
        ([1u32, 1, 1], "below-device-work-threshold"),
        ([16, 16, 8], "below-device-work-threshold"),
        ([5, 0, 7], "zero-inner-extent"),
        ([0, 3, 2], "empty-output"),
    ] {
        for caps in [BLACKWELL, METAL] {
            let e = emit(dims, caps);
            assert_eq!(e.fallback(), None, "{dims:?}");
            assert_eq!(e.receipt()["placement"], "device", "{dims:?}");
            let [r, k, n] = dims.map(u64::from);
            let strict = Contract::from_source(DIRECT.as_bytes(), "mat")
                .unwrap()
                .emit_device(DIRECT.as_bytes(), dims, [r * k, k * n], caps)
                .unwrap();
            assert_eq!(strict.fallback(), Some(strict_reason));
            assert_eq!(
                strict.receipt()["placement"],
                format!("cpu:{strict_reason}")
            );
        }
    }
    // A zero inner extent emits no matrix instruction: every cell is +0.
    let k0 = [emit([5, 0, 7], BLACKWELL), emit([5, 0, 7], METAL)];
    assert!(!k0[0].source().contains("mma.sync") && !k0[0].source().contains("invar_pack"));
    assert!(!k0[1].source().contains("simdgroup_multiply_accumulate"));
    // Strict and Tensor Core emitters each refuse the other's plan.
    let tc_plan = emit([4, 4, 4], BLACKWELL);
    let strict = Contract::from_source(DIRECT.as_bytes(), "mat")
        .unwrap()
        .emit_device(DIRECT.as_bytes(), [4, 4, 4], [16, 16], BLACKWELL)
        .unwrap();
    assert_eq!(
        gemm::emit_cuda(tc_plan.plan()),
        Err(floating::Error::Numeric)
    );
    assert_eq!(
        floating::emit_cpu(tc_plan.plan()),
        Err(floating::Error::Numeric)
    );
    assert_eq!(tc::emit_cuda(strict.plan()), Err(floating::Error::Numeric));
    assert!(tc::emit_msl(emit([4, 4, 4], METAL).plan()).is_ok());
    assert_eq!(
        tc::emit_msl(tc_plan.plan()),
        Err(floating::Error::Capabilities)
    );
    assert!(matches!(
        tc_contract().emit_cpu(TC_DIRECT.as_bytes(), [4, 4, 4], [16, 16], BLACKWELL),
        Err(Error::Device(floating::Error::Numeric))
    ));
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_tc_refuses_excluded_devices_and_extents() {
    let c = tc_contract();
    let refuse = |dims: [u32; 3], caps| {
        c.emit_device(TC_DIRECT.as_bytes(), dims, lengths(dims), caps)
            .unwrap_err()
    };
    // mma.sync m16n8k16 f16 needs sm_80; simdgroup_matrix needs Apple7+.
    for caps in [
        Capabilities::Cuda {
            major: 12,
            minor: 9,
            sm: 75,
        },
        Capabilities::Cuda {
            major: 12,
            minor: 9,
            sm: 70,
        },
        Capabilities::Metal {
            apple_family: 6,
            msl_major: 3,
            msl_minor: 1,
        },
    ] {
        assert!(
            matches!(
                refuse([2, 2, 2], caps),
                Error::Device(floating::Error::Capabilities)
            ),
            "{caps:?}"
        );
    }
    let ampere = Capabilities::Cuda {
        major: 12,
        minor: 9,
        sm: 80,
    };
    assert!(
        c.emit_device(TC_DIRECT.as_bytes(), [2, 2, 2], [4, 4], ampere)
            .is_ok()
    );
    // The proven TC inner limit remains independent of the larger matrix cap.
    let max_k = [1, invar_device::tc::MAX_INNER, 1];
    assert_eq!(emit(max_k, BLACKWELL).plan().ir().dimensions, max_k);
    assert!(matches!(
        refuse([1, invar_device::tc::MAX_INNER + 1, 1], BLACKWELL),
        Error::Device(floating::Error::InnerLimit)
    ));
    // The plan's own TC inner limit (where the amended bound is proven).
    let buffer = |id, shape: [u32; 2]| floating::Buffer {
        id,
        elements: u64::from(shape[0]) * u64::from(shape[1]),
        shape,
        strides: [u64::from(shape[1]), 1],
        offset: 0,
        alignment: 4,
    };
    let ir = |k: u32| floating::Ir {
        version: floating::VERSION,
        numeric_version: floating::NUMERIC_VERSION,
        numeric_policy: floating::TC_POLICY.into(),
        layout_version: floating::LAYOUT_VERSION,
        span: invar_device::Span { start: 1, end: 2 },
        source_identity: 7,
        graph_revision: 1,
        function_contract: [1; 32],
        dimensions: [1, k, 1],
        inputs: [buffer(0, [1, k]), buffer(1, [k, 1])],
        output: buffer(2, [1, 1]),
    };
    let origin = Provenance {
        source_identity: 7,
        graph_revision: 1,
    };
    assert!(floating::verify(ir(tc::MAX_INNER), BLACKWELL, origin).is_ok());
    assert_eq!(
        floating::verify(ir(tc::MAX_INNER + 1), BLACKWELL, origin).unwrap_err(),
        floating::Error::InnerLimit
    );
    // The same extent is still a legal strict plan: the limit is TC-only.
    let mut strict = ir(tc::MAX_INNER + 1);
    strict.numeric_policy = floating::NUMERIC_POLICY.into();
    assert!(floating::verify(strict, BLACKWELL, origin).is_ok());
    assert_eq!(
        tc::verify([1, tc::MAX_INNER + 1, 1], &[], &[], &[0]).unwrap_err(),
        tc::Failure::Shape
    );
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_tc_inputs_round_to_binary16_or_refuse() {
    for (word, rounded) in [
        (0x3f80_0000, Some(0x3f80_0000)), // 1
        (0x3f80_0001, Some(0x3f80_0000)), // 1 + 2^-23 -> 1
        (0x3f80_1000, Some(0x3f80_0000)), // 1 + 2^-11: tie, even -> 1
        (0x3f80_3000, Some(0x3f80_4000)), // 1 + 3*2^-11: tie, even -> 1 + 2^-9
        (0x477f_e000, Some(0x477f_e000)), // 65504
        (0x477f_efff, Some(0x477f_e000)), // just below 65520
        (0x477f_f000, None),              // 65520 overflows
        (0x3380_0000, Some(0x3380_0000)), // 2^-24, smallest subnormal
        (0x3300_0000, Some(0x0000_0000)), // 2^-25: tie with 0, even -> +0
        (0xb300_0001, Some(0xb380_0000)), // just above: -2^-24
        (0x8000_0001, Some(0x8000_0000)), // -2^-149 -> -0
        (0x7f80_0000, None),
        (0x7fc0_0000, None),
    ] {
        assert_eq!(tc::f16_round(word), rounded, "{word:#010x}");
    }
    let a = [0x3f80_0000, 0x477f_f000];
    let refused = |input, index| tc::InputRange { input, index };
    assert_eq!(tc::round_inputs(0, &a), Err(refused(0, 1)));
    assert_eq!(
        tc::reference([1, 2, 1], &a, &[0x3f80_0000; 2]),
        Err(tc::Failure::Input(refused(0, 1)))
    );
    assert_eq!(
        tc::verify([1, 2, 1], &[0x3f80_0000; 2], &[0x7f80_0000, 0], &[0]),
        Err(tc::Failure::Input(refused(1, 0)))
    );
    assert_eq!(
        refused(0, 2).to_string(),
        "tc-f16-f32-v1 input 0 element 2 outside the binary16 range"
    );
    // Binary16 subnormals are admitted (second D5 amendment): they round,
    // and the reference sums their exact products.
    let sub = [0x3380_0000, 0x387f_dfff, 0xb300_0001, 0x3a80_0000];
    assert_eq!(
        tc::round_inputs(1, &sub),
        Ok(vec![0x3380_0000, 0x387f_c000, 0xb380_0000, 0x3a80_0000])
    );
    // 2^-24 + 1023 x 2^-24 - 2^-24 + 2^-10 = 1023 x 2^-24 + 2^-10, exactly.
    assert_eq!(
        tc::reference([1, 4, 1], &[0x3f80_0000; 4], &sub),
        Ok(vec![
            (1023.0f32 * 2f32.powi(-24) + 2f32.powi(-10)).to_bits()
        ])
    );
}

/// Exact binary32 value as a rational.
fn rational(word: u32) -> BigRational {
    let magnitude = word & 0x7fff_ffff;
    let (m, e) = match magnitude >> 23 {
        0 => (i64::from(magnitude), -149),
        biased => (
            i64::from((magnitude & 0x007f_ffff) | 0x0080_0000),
            biased as i32 - 150,
        ),
    };
    let m = if word >> 31 == 1 { -m } else { m };
    let two = BigRational::from_integer(BigInt::from(2));
    BigRational::from_integer(BigInt::from(m)) * two.pow(e)
}

/// Independent exact cell: rational sum of the rounded products and of their
/// magnitudes (rounding from `f16_round`, checked against a table elsewhere).
fn exact_cell(
    dims: [u32; 3],
    a: &[u32],
    b: &[u32],
    i: usize,
    j: usize,
) -> (BigRational, BigRational) {
    let [_, inner, cols] = dims.map(|n| n as usize);
    let r = |w| rational(tc::f16_round(w).unwrap());
    (0..inner).fold(
        (
            BigRational::from_integer(0.into()),
            BigRational::from_integer(0.into()),
        ),
        |(s, m), k| {
            let p = r(a[i * inner + k]) * r(b[k * cols + j]);
            let magnitude = if p < BigRational::from_integer(0.into()) {
                -p.clone()
            } else {
                p.clone()
            };
            (s + &p, m + magnitude)
        },
    )
}

/// Nearest-even binary32 of an exact rational, through the strict oracle.
fn nearest(value: &BigRational) -> u32 {
    let scale = BigInt::from(10).pow(60);
    let negative = *value < BigRational::from_integer(0.into());
    let magnitude = if negative {
        -value.clone()
    } else {
        value.clone()
    };
    let scaled = (magnitude * BigRational::from_integer(scale)).to_integer();
    let digits = format!("{:0>61}", scaled.to_string());
    let (whole, fraction) = digits.split_at(digits.len() - 60);
    // Exact for sums (multiples of 2^-48 have at most 48 decimals); a start
    // point only for the finer bound edges, which callers then step exactly.
    let bits = F32Bits::from_decimal(&format!("{whole}.{fraction}"))
        .unwrap()
        .to_bits();
    if negative { bits | 0x8000_0000 } else { bits }
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_tc_reference_is_the_rounded_exact_sum() {
    let mut cells = 0;
    for (n, dims) in SHAPES.iter().enumerate() {
        let [r, k, c] = dims.map(|v| v as usize);
        let a = corpus::tc_words(r * k, 11 + n as u64);
        let b = corpus::tc_words(k * c, 101 + n as u64);
        let reference = tc::reference(*dims, &a, &b).unwrap();
        for (q, &word) in reference.iter().enumerate() {
            let (exact, _) = exact_cell(*dims, &a, &b, q / c, q % c);
            assert_eq!(word, nearest(&exact), "{dims:?} cell {q}");
        }
        // The reference itself is admissible, with a tiny ratio.
        assert!(tc::verify(*dims, &a, &b, &reference).unwrap().max_ratio < 0.5);
        cells += reference.len();
    }
    assert!(cells > 0);
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_tc_bound_holds_for_device_orders_at_k_16_32_4096_and_max()
 {
    for k in [16, 32, 4096, 1_000_000] {
        let (dims, a, b) = corpus::tc_counterexample(k);
        let exact = 1.0 + f64::from(k - 1) * 2f64.powi(-24);
        // Source order rounds every 1 + 2^-24 back to 1.
        assert_eq!(
            corpus::tc_ordered(dims, &a, &b, 1, false, false),
            [0x3f80_0000]
        );
        for (tile, reverse, truncate) in [
            (1, false, false),
            (16, true, false),
            (16, true, true),
            (8, true, false),
            (k as usize, true, false),
            (k as usize, true, true),
        ] {
            let r = corpus::tc_ordered(dims, &a, &b, tile, reverse, truncate);
            let report = tc::verify(dims, &a, &b, &r)
                .unwrap_or_else(|f| panic!("K={k} tile={tile} {reverse} {truncate}: {f:?}"));
            assert!(report.max_ratio <= 1.0, "{report:?}");
        }
        let reference = tc::reference(dims, &a, &b).unwrap();
        assert!((f64::from(f32::from_bits(reference[0])) - exact).abs() <= exact * 2f64.powi(-24));
    }
    // Random rectangular data at K = 4096 in several device-like orders.
    let dims = [3, 4096, 5];
    let a = corpus::tc_words(3 * 4096, 7);
    let b = corpus::tc_words(4096 * 5, 9);
    for (tile, reverse, truncate) in [(1, false, false), (16, true, true), (4096, true, false)] {
        let r = corpus::tc_ordered(dims, &a, &b, tile, reverse, truncate);
        tc::verify(dims, &a, &b, &r).unwrap();
    }
}

fn next_up(w: u32) -> u32 {
    match w {
        0x8000_0000 => 1,
        _ if w >> 31 == 0 => w + 1,
        _ => w - 1,
    }
}
fn next_down(w: u32) -> u32 {
    match w {
        0 => 0x8000_0001,
        _ if w >> 31 == 0 => w - 1,
        _ => w + 1,
    }
}

/// The smallest and largest binary32 inside `[centre - slack, centre + slack]`.
fn edges(centre: &BigRational, slack: &BigRational) -> (u32, u32) {
    let (lo, hi) = (centre - slack, centre + slack);
    let mut top = nearest(&hi);
    while rational(top) > hi {
        top = next_down(top);
    }
    while rational(next_up(top)) <= hi {
        top = next_up(top);
    }
    let mut bottom = nearest(&lo);
    while rational(bottom) < lo {
        bottom = next_up(bottom);
    }
    while rational(next_down(bottom)) >= lo {
        bottom = next_down(bottom);
    }
    (bottom, top)
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_tc_verifier_rejects_one_ulp_beyond_the_bound() {
    let mut cases = vec![
        corpus::tc_counterexample(32),
        corpus::tc_counterexample(4096),
    ];
    let dims = [2, 300, 3];
    cases.push((dims, corpus::tc_words(600, 21), corpus::tc_words(900, 23)));
    for (dims, a, b) in cases {
        let k = BigRational::from_integer(BigInt::from(dims[1]));
        let reference = tc::reference(dims, &a, &b).unwrap();
        let (exact, magnitude) = exact_cell(dims, &a, &b, 0, 0);
        let two = BigRational::from_integer(BigInt::from(2));
        let slack = &two * k * magnitude * two.pow(-23);
        let (bottom, top) = edges(&exact, &slack);
        let at = |word: u32| {
            let mut out = reference.clone();
            out[0] = word;
            tc::verify(dims, &a, &b, &out)
        };
        // Exactly at the edges: admissible. One binary32 step beyond: refused.
        // A loosened bound would accept `beyond`; a source-order reference
        // (1 for the counterexample) would refuse `top`.
        assert!(at(top).is_ok(), "{dims:?} top {top:#010x}");
        assert!(at(bottom).is_ok(), "{dims:?} bottom {bottom:#010x}");
        for word in [next_up(top), next_down(bottom)] {
            assert!(
                matches!(at(word), Err(tc::Failure::Cell { cell: 0, .. })),
                "{dims:?} beyond {word:#010x}"
            );
        }
    }
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_tc_verifier_rejects_nonfinite_far_and_misshapen_results()
 {
    let (dims, a, b) = corpus::tc_counterexample(16);
    for word in [
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x7f7f_ffff,
        0x4000_0000,
        0x0000_0000,
    ] {
        assert!(
            matches!(
                tc::verify(dims, &a, &b, &[word]),
                Err(tc::Failure::Cell { .. })
            ),
            "{word:#010x}"
        );
    }
    assert_eq!(
        tc::verify(dims, &a, &b, &[]).unwrap_err(),
        tc::Failure::Shape
    );
    assert_eq!(
        tc::verify(dims, &a[1..], &b, &[0]).unwrap_err(),
        tc::Failure::Shape
    );
    // All-zero products (sum |p| = 0) admit only zeros, of either sign.
    let (a, b) = ([0x0000_0000, 0x8000_0000], [0x3f80_0000, 0x3f80_0000]);
    for word in [0x0000_0000, 0x8000_0000] {
        tc::verify([1, 2, 1], &a, &b, &[word]).unwrap();
    }
    assert!(tc::verify([1, 2, 1], &a, &b, &[1]).is_err());
}
