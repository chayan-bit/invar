//! M4-01 strict binary32 contraction on Metal, from the same verified device
//! plan as CUDA. CPU tests check the emitted MSL identity and cache key.
//! `issue_m4_01_metal_gpu_*` tests need a real Apple7+ GPU and are ignored by
//! default: run them with `--features metal -- --ignored` on the Mac.
#![forbid(unsafe_code)]

#[path = "issue_m4_01_device/corpus.rs"]
mod corpus;
#[path = "issue_m4_01_device/ops.rs"]
mod ops;

/// These tests run emitted and deliberately mutated MSL, which no
/// compiled-ahead library holds, so they open the developer source device
/// (#204). `issue_metal_aot_gpu` covers the compiled-ahead route.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn source_device() -> invar_metal::MetalDevice {
    invar_metal::MetalDevice::open_runtime_source().expect("a real Apple7+ Metal GPU is required")
}

/// The rank-1 status word uses the shared guard encoding: a kernel that never
/// claims it, or sets a bit outside {GUARD, OVERFLOW}, is a typed refusal.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal GPU"]
fn issue_m4_01_metal_gpu_rejects_unwritten_and_illegal_f32_ops_status() {
    use invar_device::f32_ops::{Kind, METAL_STATUS, Node, Program, Scalar};
    use invar_device::floating::guard::{InvalidWord, UNWRITTEN, msl_claim};
    let mut device = source_device();
    let map = ops::plan(
        Program {
            length: 1,
            inputs: vec![Scalar::F32, Scalar::F32],
            nodes: vec![Node::Input(0), Node::Input(1), Node::Mul(0, 1)],
            kind: Kind::Map,
        },
        device.capabilities(),
    );
    let input = [0x3f80_0000, 0x3f80_0000];
    let source = invar_device::f32_ops::emit(&map).unwrap();
    let claim = msl_claim(METAL_STATUS);
    assert!(source.contains(&claim));
    let real = device
        .diagnose_f32_ops_source(&map, &input, &source)
        .unwrap();
    assert_eq!((real.output, real.flagged), (vec![0x3f80_0000], false));
    let unwritten = source.replace(&claim, "");
    let illegal = source.replace(
        &claim,
        "  atomic_store_explicit(F, 4u, memory_order_relaxed);\n",
    );
    for (mutant, word) in [(unwritten, UNWRITTEN), (illegal, 4)] {
        assert!(matches!(
            device.diagnose_f32_ops_source(&map, &input, &mutant),
            Err(invar_metal::Error::InvalidGuard(InvalidWord(w))) if w == word
        ));
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal GPU"]
fn issue_m4_01_metal_gpu_v2_mutations_fail() {
    use invar_device::f32_ops::{Kind, Node, Program, Scalar};
    let mut device = source_device();
    let caps = device.capabilities();
    let reduction = ops::plan(
        Program {
            length: 3,
            inputs: vec![Scalar::F32],
            nodes: vec![Node::Input(0)],
            kind: Kind::Reduce { initial: 0 },
        },
        caps,
    );
    let input = [0x4b800000, 0x3f800000, 0xcb800000];
    let source = invar_device::f32_ops::emit(&reduction).unwrap();
    let mutant = source.replace("A[0u + i]", "A[0u + (i == 1u ? 2u : (i == 2u ? 1u : 0u))]");
    assert_ne!(mutant, source);
    let observed = device
        .diagnose_f32_ops_source(&reduction, &input, &mutant)
        .unwrap();
    assert!(!observed.flagged);
    assert_ne!(
        observed.output,
        [0],
        "reordered reduction mutation survived"
    );
    let map = ops::plan(
        Program {
            length: 1,
            inputs: vec![Scalar::F32, Scalar::F32],
            nodes: vec![Node::Input(0), Node::Input(1), Node::Mul(0, 1)],
            kind: Kind::Map,
        },
        caps,
    );
    let input = [0x00800000, 0x3f000000];
    let source = invar_device::f32_ops::emit(&map).unwrap();
    let mutant = source.replace(
        "if (flag != 0u) atomic_fetch_or_explicit(F, 1u, memory_order_relaxed);",
        "",
    );
    assert_ne!(mutant, source);
    let observed = device
        .diagnose_f32_ops_source(&map, &input, &mutant)
        .unwrap();
    assert!(!observed.flagged);
    assert_ne!(
        observed.output,
        [0x00400000],
        "removed fallback flag mutation survived"
    );
    let map = ops::plan(
        Program {
            length: 1,
            inputs: vec![Scalar::F32, Scalar::F32, Scalar::F32],
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Input(2),
                Node::Mul(0, 1),
                Node::Add(3, 2),
            ],
            kind: Kind::Map,
        },
        caps,
    );
    let input = [0x3f800800, 0x3f800800, 0xbf801000];
    let source = invar_device::f32_ops::emit(&map).unwrap();
    // Re-enable contraction at the exact multiply/add site as well as the pragma.
    let mutant = source
        .replace(
            "#pragma METAL fp contract(off)",
            "#pragma METAL fp contract(on)",
        )
        .replace(
            "as_type<uint>(as_type<float>(v3) + as_type<float>(v2))",
            "as_type<uint>(fma(as_type<float>(v0),as_type<float>(v1),as_type<float>(v2)))",
        );
    assert_ne!(mutant, source);
    let observed = device
        .diagnose_f32_ops_source(&map, &input, &mutant)
        .unwrap();
    assert_ne!(observed.output, [0], "re-enabled FMA mutation survived");
    eprintln!(
        "M4-01 v2 mutations caught: reordered reduction, re-enabled FMA, removed fallback flag"
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal GPU"]
fn issue_m4_01_metal_gpu_v2_maps_folds_comparisons_conversions() {
    let mut device = source_device();
    let caps = device.capabilities();
    let mut published = 0;
    let mut fallback = 0;
    for p in ops::programs() {
        let input = ops::packed(&p);
        let plan = ops::plan(p.clone(), caps);
        let expected = invar::f32_source::cpu(&plan, &input, 1).unwrap();
        match device.run_f32_ops(&plan, &input) {
            Ok(actual) => {
                ops::same(
                    &actual,
                    &expected,
                    p.output_type().unwrap() == invar_device::f32_ops::Scalar::F32,
                );
                published += 1;
            }
            Err(invar_metal::Error::SubnormalFallback) => {
                fallback += 1;
            }
            Err(error) => panic!("{error:?}: {p:?}"),
        }
    }
    assert!(published > 0 && fallback > 0);
    for source in ops::scalar_sources() {
        let plan = invar::f32_source::Region::from_checked(&ops::checked(&source), "kernel")
            .unwrap()
            .bind(1, caps)
            .unwrap();
        for input in [
            [1, 2],
            [0x80000000, 0],
            [0x7fc00001, 0xffc00002],
            [0x3f800000, 0x40000000],
        ] {
            let expected = invar::f32_source::cpu(&plan, &input, 1).unwrap();
            match device.run_f32_ops(&plan, &input) {
                Ok(actual) => ops::same(
                    &actual,
                    &expected,
                    invar_device::f32_ops::program(&plan)
                        .unwrap()
                        .output_type()
                        .unwrap()
                        == invar_device::f32_ops::Scalar::F32,
                ),
                Err(invar_metal::Error::SubnormalFallback) => assert!(source.contains("Quantity")),
                Err(error) => panic!("{error:?}: {source}"),
            }
        }
    }
    for source in ops::sources() {
        let expected = ops::reference(&source);
        let run =
            invar::f32_source::run_metal_source(source.as_bytes(), 1_000_000, &mut device).unwrap();
        ops::parity(&run, &expected);
        if !source.contains("fma(a[i]") && !source.contains("filled_f32(0u32") {
            assert!(run.completed > 0, "{source}: {:?}", run.refusals);
        } else if !source.contains("filled_f32(0u32") {
            assert!(run.refusals.contains(&"metal_subnormal_fallback"));
        }
    }
    eprintln!(
        "M4-01 v2 Metal: {published} published dispatches, {fallback} exact CPU fallback dispatches"
    );
    assert_eq!(device.quarantine_len(), 0);
}

use corpus::MAT;
use invar::tensor;
use invar_device::Capabilities;

const METAL: Capabilities = Capabilities::Metal {
    apple_family: 9,
    msl_major: 3,
    msl_minor: 1,
};

#[test]
fn issue_m4_01_metal_emission_shares_the_verified_plan_and_binds_the_key() {
    let cuda = tensor::emit_cuda(
        MAT.as_bytes(),
        "mat",
        [3, 5, 3],
        [15, 15],
        Capabilities::cuda_12_9_sm89(),
    )
    .unwrap();
    let metal = tensor::emit_msl(MAT.as_bytes(), "mat", [3, 5, 3], [15, 15], METAL).unwrap();
    // Same verified IR; only the target capabilities differ.
    assert_eq!(cuda.plan().ir(), metal.plan().ir());
    assert_ne!(cuda.plan().capabilities(), metal.plan().capabilities());
    assert_eq!(metal.layouts().dimensions(), [3, 5, 3]);
    assert!(metal.source().contains("#pragma METAL fp contract(off)"));
    assert!(metal.source().contains("invar_f32_contract_v1"));
    for option in metal.plan().msl_options() {
        assert!(metal.source().contains(option));
    }
    assert!(!metal.matches_hint(cuda.key()));
    let again = tensor::emit(MAT.as_bytes(), "mat", [3, 5, 3], [15, 15], METAL).unwrap();
    assert!(metal.matches_hint(again.key()));
    assert_eq!(again.source(), metal.source());
    let other_family = Capabilities::Metal {
        apple_family: 7,
        msl_major: 3,
        msl_minor: 1,
    };
    let other = tensor::emit_msl(MAT.as_bytes(), "mat", [3, 5, 3], [15, 15], other_family);
    assert!(!metal.matches_hint(other.unwrap().key()));
    assert!(
        !metal.matches_hint(
            tensor::emit_msl(MAT.as_bytes(), "mat", [3, 5, 4], [15, 20], METAL)
                .unwrap()
                .key()
        )
    );
    // Every rejection keeps the typed error; nothing falls through to CUDA.
    assert!(matches!(
        tensor::emit_msl(
            MAT.as_bytes(),
            "mat",
            [3, 5, 3],
            [15, 15],
            Capabilities::cuda_12_9_sm89()
        ),
        Err(tensor::Error::Device(
            invar_device::floating::Error::Capabilities
        ))
    ));
    for caps in [
        Capabilities::Metal {
            apple_family: 6,
            msl_major: 3,
            msl_minor: 1,
        },
        Capabilities::Metal {
            apple_family: 9,
            msl_major: 3,
            msl_minor: 2,
        },
    ] {
        assert!(matches!(
            tensor::emit_msl(MAT.as_bytes(), "mat", [3, 5, 3], [15, 15], caps),
            Err(tensor::Error::Device(
                invar_device::floating::Error::Capabilities
            ))
        ));
    }
    assert!(matches!(
        tensor::emit_msl(MAT.as_bytes(), "mat", [3, 5, 3], [14, 15], METAL),
        Err(tensor::Error::Layout(_))
    ));
}

#[test]
fn issue_m4_01_metal_probe_kernel_covers_every_d1_operation() {
    use invar::numeric_contract::{Comparison, F32Bits};
    use invar_device::f32_probe::{CASES, Op};
    // Second, independent derivation of the probe table with the root's
    // software binary32 oracle (the device crate checks host arithmetic).
    for (index, case) in CASES.iter().enumerate() {
        let [a, b, c] = [case.a, case.b, case.c].map(F32Bits::from_bits);
        let actual = match case.op {
            Op::Mul => a.mul(b).to_bits(),
            Op::Add => a.add(b).to_bits(),
            Op::Sub => a.sub(b).to_bits(),
            Op::Fma => a.fma(b, c).to_bits(),
            Op::Accumulate => c.add(a.mul(b)).to_bits(),
            Op::Lt => u32::from(a.compare(Comparison::Lt, b)),
            Op::Eq => u32::from(a.compare(Comparison::Eq, b)),
            Op::FromU32 => F32Bits::from_u32(case.a).to_bits(),
            Op::FromI32 => F32Bits::from_i32(case.a as i32).to_bits(),
            Op::TruncI32 => a.trunc_i32().unwrap() as u32,
            Op::Div => a.div(b).to_bits(),
            Op::Sqrt => a.sqrt().to_bits(),
        };
        assert_eq!(actual, case.expected, "case {index} {}", case.what);
    }
    let msl = invar_device::f32_probe::emit_msl();
    assert!(msl.contains("#pragma METAL fp contract(off)"));
    assert!(msl.contains("s = s + a * b;"));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod hardware {
    use super::corpus::{self, FUEL, SHAPES};
    use super::*;
    use invar::tensor::host::{self, metal::Metal};
    use invar_cuda::Runtime;
    use invar_device::f32_probe::{CASES, CONTRACTION_OPS, Op};
    use invar_device::floating::guard;
    use invar_metal::{Diagnostic, Error, MetalDevice};

    static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        GPU_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }
    /// A real Apple7+ GPU that the D1 probe admits for strict contractions,
    /// exactly or under the underflow guard; anything else fails the gate.
    fn open() -> MetalDevice {
        let mut device =
            MetalDevice::open_runtime_source().expect("a real Apple7+ Metal GPU is required");
        assert!(
            device.admits_strict_contraction().unwrap(),
            "{}",
            device.strict_f32_probe().unwrap().summary()
        );
        device
    }
    fn plan(caps: Capabilities, dims: [u32; 3]) -> tensor::Emission {
        let [r, k, c] = dims.map(u64::from);
        tensor::emit_msl(MAT.as_bytes(), "mat", dims, [r * k, k * c], caps).unwrap()
    }
    /// What the publishing path may observe: an unflagged dispatch must be
    /// bit-exact, and a flagged one is discarded for the exact CPU result.
    fn published(diagnostic: &Diagnostic, oracle: &[u32]) -> Result<Vec<u32>, String> {
        if diagnostic.flagged {
            return Ok(oracle.to_vec());
        }
        if diagnostic.output != oracle {
            return Err("unflagged dispatch differs from the strict oracle".into());
        }
        Ok(diagnostic.output.clone())
    }

    #[test]
    #[ignore = "requires a real Apple7+ Metal GPU"]
    fn issue_m4_01_metal_gpu_d1_probe_admits_only_guarded_differences() {
        let _gpu = lock();
        let mut device = open();
        let report = device.strict_f32_probe().unwrap().clone();
        eprintln!(
            "M4-01 Metal device={} caps={:?}",
            device.name(),
            device.capabilities()
        );
        eprintln!("{}", report.summary());
        for line in report.details() {
            eprintln!("  {line}");
        }
        // The pinned `fp contract(off)` must hold on every device.
        assert!(!report.contracts_implicitly(), "{}", report.summary());
        // The device guard computes exactly the Rust mirror's predicate on the
        // words it observed, for every `*` and `+` probe case.
        for (i, case) in CASES.iter().enumerate() {
            let r = report.observed()[i];
            let operands = guard::operand(case.a) || guard::operand(case.b);
            let mirror = match case.op {
                Op::Mul => operands || guard::mul(case.a, case.b, r),
                Op::Add => operands || guard::add(case.a, case.b, r),
                _ => continue,
            };
            assert_eq!(report.flags()[i], mirror, "case {i} {}", case.what);
        }
        // Every difference in the contraction's operations is in the guarded
        // class; differences elsewhere keep those operations refused.
        for &i in report.mismatches() {
            if CONTRACTION_OPS.contains(&CASES[i].op) {
                assert!(report.flags()[i], "case {i} {}", CASES[i].what);
            }
        }
        assert!(report.admits_strict() || report.admits_strict_with_fallback(CONTRACTION_OPS));
        assert!(report.admits_strict_with_fallback(&[Op::Mul, Op::Fma]));
        let again = open().strict_f32_probe().unwrap().clone();
        assert_eq!(again.observed(), report.observed());
        assert_eq!(again.flags(), report.flags());
    }

    #[test]
    #[ignore = "requires a real Apple7+ Metal GPU"]
    fn issue_m4_01_metal_gpu_normal_cells_publish_bit_exact_on_device() {
        let _gpu = lock();
        let mut device = open();
        let caps = device.capabilities();
        let mut cells = 0;
        for dims in SHAPES {
            let [r, k, c] = dims.map(|n| n as usize);
            let a = corpus::words(&corpus::NORMAL, r * k, 3);
            let b = corpus::words(&corpus::NORMAL, k * c, 5);
            let result = device
                .run_contraction(plan(caps, dims).plan(), [&a, &b])
                .unwrap();
            assert_eq!(result.output, corpus::oracle(dims, &a, &b), "{dims:?}");
            cells += r * c;
        }
        assert_eq!(cells, 703);
        assert_eq!(device.quarantine_len(), 0);
        eprintln!("M4-01 Metal: {cells} normal-range cells published from the GPU, bit-exact");
    }

    #[test]
    #[ignore = "requires a real Apple7+ Metal GPU"]
    fn issue_m4_01_metal_gpu_subnormal_dispatches_fall_back_and_stay_exact() {
        let _gpu = lock();
        let mut device = open();
        let caps = device.capabilities();
        let flushes = device.strict_f32_probe().unwrap().flushes_subnormals();
        let (mut diverging, mut fallbacks, mut published_exact) = (0, 0, 0);
        for dims in SHAPES {
            let [r, k, c] = dims.map(|n| n as usize);
            let emission = plan(caps, dims);
            let a = corpus::words(&corpus::SUBNORMAL, r * k, 3);
            let b = corpus::words(&corpus::SUBNORMAL, k * c, 5);
            let oracle = corpus::oracle(dims, &a, &b);
            let raw = device
                .diagnose_unadmitted_contraction(emission.plan(), [&a, &b])
                .unwrap();
            let wrong = raw
                .output
                .iter()
                .zip(&oracle)
                .filter(|(x, y)| x != y)
                .count();
            // Soundness on hardware: a dispatch with any wrong cell is flagged.
            assert!(
                wrong == 0 || raw.flagged,
                "{dims:?}: {wrong} unflagged cells"
            );
            diverging += wrong;
            assert_eq!(published(&raw, &oracle).unwrap(), oracle);
            match device.run_contraction(emission.plan(), [&a, &b]) {
                Err(Error::SubnormalFallback) => {
                    assert!(raw.flagged);
                    fallbacks += 1;
                }
                Ok(result) => {
                    assert!(!raw.flagged);
                    assert_eq!(result.output, oracle, "{dims:?}");
                    published_exact += 1;
                }
                Err(error) => panic!("{dims:?}: {error:?}"),
            }
        }
        eprintln!(
            "M4-01 Metal subnormal corpus: {diverging} diverging cells, all in {fallbacks} discarded dispatches; {published_exact} dispatches published exactly"
        );
        assert_eq!(diverging > 0, flushes);
        if flushes {
            assert_eq!(diverging, 332);
            assert!(fallbacks > 0);
        }
        assert_eq!(device.quarantine_len(), 0);
    }

    #[test]
    #[ignore = "requires a real Apple7+ Metal GPU"]
    fn issue_m4_01_metal_gpu_intermediate_underflow_with_normal_output_falls_back() {
        let _gpu = lock();
        let mut device = open();
        let caps = device.capabilities();
        let emission = plan(caps, [1, 8, 1]);
        let (a, b) = ([corpus::UNDERFLOW_A; 8], [corpus::UNDERFLOW_B; 8]);
        let oracle = corpus::oracle([1, 8, 1], &a, &b);
        assert_eq!(oracle, [corpus::UNDERFLOW_SUM]);
        assert!(!corpus::is_subnormal(oracle[0]) && oracle[0] != 0);
        let raw = device
            .diagnose_unadmitted_contraction(emission.plan(), [&a, &b])
            .unwrap();
        assert!(raw.flagged);
        if device.strict_f32_probe().unwrap().flushes_subnormals() {
            assert_eq!(raw.output, [0], "a flushing device loses the whole sum");
        }
        assert!(matches!(
            device.run_contraction(emission.plan(), [&a, &b]),
            Err(Error::SubnormalFallback)
        ));
        // A rounding-hidden underflow: 1 + 2^-127 rounds to 1 either way, but
        // the guard cannot know that and still discards the dispatch.
        let hidden = plan(caps, [1, 2, 1]);
        let (a, b) = (
            [0x3f80_0000, corpus::UNDERFLOW_A],
            [0x3f80_0000, corpus::UNDERFLOW_B],
        );
        let raw = device
            .diagnose_unadmitted_contraction(hidden.plan(), [&a, &b])
            .unwrap();
        assert!(raw.flagged);
        assert_eq!(
            published(&raw, &corpus::oracle([1, 2, 1], &a, &b)).unwrap(),
            [0x3f80_0000]
        );
        // The same case as ordinary source: the receipt records the fallback
        // and the published value is the exact CPU result.
        let source = corpus::underflow_program();
        let expected = corpus::reference(&source);
        assert_eq!(expected.result, Ok(invar::eval::Value::I32(7)));
        let device = Metal::open(
            source.as_bytes(),
            "mat",
            invar::schedule::metal::Kernels::RuntimeSource,
        )
        .unwrap();
        let caps = device.capabilities();
        let mut runtime = Runtime::new(device, 2);
        let run = host::run_source(source.as_bytes(), FUEL, &mut runtime, caps).unwrap();
        corpus::assert_parity(&run.observation, &expected);
        assert_eq!(run.completed, 0);
        assert_eq!(run.refusals, ["metal_subnormal_fallback"]);
    }

    /// The sum rule's zero branch alone: normal products whose partial sum
    /// cancels to the subnormal 2^-149. A flushing device returns +0 there and
    /// then publishes a normal but wrong word; no other predicate fires.
    #[test]
    #[ignore = "requires a real Apple7+ Metal GPU"]
    fn issue_m4_01_metal_gpu_cancelling_partial_sum_falls_back() {
        let _gpu = lock();
        let mut device = open();
        let emission = plan(device.capabilities(), [1, 3, 1]);
        let (a, b) = ([0x0080_0001, 0x8080_0000, 0x0100_0001], [0x3f80_0000; 3]);
        let oracle = corpus::oracle([1, 3, 1], &a, &b);
        // 2^-149 + (2^-125 + 2^-148) is a tie that rounds to even.
        assert_eq!(oracle, [0x0100_0002]);
        let raw = device
            .diagnose_unadmitted_contraction(emission.plan(), [&a, &b])
            .unwrap();
        assert!(raw.flagged);
        if device.strict_f32_probe().unwrap().flushes_subnormals() {
            assert_eq!(raw.output, [0x0100_0001], "the flushed partial sum is lost");
        }
        assert!(matches!(
            device.run_contraction(emission.plan(), [&a, &b]),
            Err(Error::SubnormalFallback)
        ));
    }

    #[test]
    #[ignore = "requires a real Apple7+ Metal GPU"]
    fn issue_m4_01_metal_gpu_rejects_unwritten_and_nonbinary_dispatch_flags() {
        let _gpu = lock();
        let mut device = open();
        let emission = plan(device.capabilities(), [1, 1, 1]);
        let clear = "  atomic_fetch_and_explicit(F, 1u, memory_order_relaxed);\n";
        let store = "  if (flag != 0u) atomic_fetch_or_explicit(F, 1u, memory_order_relaxed);\n";
        assert!(emission.source().contains(clear));
        assert!(emission.source().contains(store));
        let unwritten = emission.source().replace(clear, "").replace(store, "");
        let nonbinary = emission
            .source()
            .replace(
                clear,
                "  atomic_store_explicit(F, 2u, memory_order_relaxed);\n",
            )
            .replace(store, "");
        use invar_device::floating::guard::{InvalidWord, UNWRITTEN};
        for (source, word) in [(unwritten, UNWRITTEN), (nonbinary, 2)] {
            assert!(matches!(
                device.diagnose_source(emission.plan(), &source, [&[0x3f80_0000], &[0x3f80_0000]]),
                Err(Error::InvalidGuard(InvalidWord(w))) if w == word
            ));
        }
    }

    /// Disabling the guard's flag store must be caught: some corpus dispatch
    /// then publishes a wrong word unflagged.
    #[test]
    #[ignore = "requires a real Apple7+ Metal GPU"]
    fn issue_m4_01_metal_gpu_guard_mutation_is_caught() {
        let _gpu = lock();
        let mut device = open();
        let caps = device.capabilities();
        let store = "  if (flag != 0u) atomic_fetch_or_explicit(F, 1u, memory_order_relaxed);\n";
        let mut cases: Vec<([u32; 3], Vec<u32>, Vec<u32>)> = SHAPES
            .iter()
            .map(|&dims| {
                let [r, k, c] = dims.map(|n| n as usize);
                (
                    dims,
                    corpus::words(&corpus::SUBNORMAL, r * k, 3),
                    corpus::words(&corpus::SUBNORMAL, k * c, 5),
                )
            })
            .collect();
        cases.push((
            [1, 8, 1],
            vec![corpus::UNDERFLOW_A; 8],
            vec![corpus::UNDERFLOW_B; 8],
        ));
        let mut caught = 0;
        for (dims, a, b) in &cases {
            let emission = plan(caps, *dims);
            let oracle = corpus::oracle(*dims, a, b);
            let real = device
                .diagnose_source(emission.plan(), emission.source(), [a, b])
                .unwrap();
            assert_eq!(published(&real, &oracle).unwrap(), oracle, "{dims:?}");
            if !emission.source().contains(store) {
                continue; // empty output domains have no guard to disable
            }
            let mutated = emission.source().replace(store, "");
            let raw = device
                .diagnose_source(emission.plan(), &mutated, [a, b])
                .unwrap();
            assert!(!raw.flagged);
            if published(&raw, &oracle).is_err() {
                caught += 1;
            }
        }
        if device.strict_f32_probe().unwrap().flushes_subnormals() {
            assert!(caught > 0, "a guard-disabling mutation went unnoticed");
        }
        eprintln!(
            "M4-01 Metal guard mutation: {caught} of {} dispatches caught",
            cases.len()
        );
    }

    #[test]
    #[ignore = "requires a real Apple7+ Metal GPU"]
    fn issue_m4_01_metal_gpu_ordinary_source_publishes_or_records_fallback() {
        let _gpu = lock();
        let (mut completed, mut fallbacks) = (0, 0);
        for dims in SHAPES {
            let source = corpus::program(dims);
            let expected = corpus::reference(&source);
            assert_eq!(expected.result, Ok(invar::eval::Value::I32(7)), "{dims:?}");
            let device = Metal::open(
                source.as_bytes(),
                "mat",
                invar::schedule::metal::Kernels::RuntimeSource,
            )
            .unwrap();
            let caps = device.capabilities();
            let mut runtime = Runtime::new(device, 2);
            let run = host::run_source(source.as_bytes(), FUEL, &mut runtime, caps).unwrap();
            // Device or fallback, the published values pass main's in-source
            // value and zero-sign recomputation and match the reference.
            corpus::assert_parity(&run.observation, &expected);
            if !corpus::reached(dims) {
                assert_eq!((run.completed, run.refusals.len()), (0, 0), "{dims:?}");
            } else if run.completed == 1 {
                assert!(run.refusals.is_empty(), "{dims:?}");
                completed += 1;
            } else {
                assert_eq!(run.refusals, ["metal_subnormal_fallback"], "{dims:?}");
                fallbacks += 1;
            }
            assert_eq!(runtime.quarantine_len(), 0);
        }
        eprintln!(
            "M4-01 Metal source corpus: {completed} published from the GPU, {fallbacks} recorded fallbacks"
        );
        assert!(completed > 0);
    }

    #[test]
    #[ignore = "requires a real Apple7+ Metal GPU"]
    fn issue_m4_01_metal_gpu_refusals_and_padding() {
        let _gpu = lock();
        let mut device = open();
        let caps = device.capabilities();
        // Nonfinite input stays a typed refusal.
        let one = plan(caps, [1, 1, 1]);
        assert!(matches!(
            device.run_contraction(one.plan(), [&[0x7f80_0000], &[0x3f80_0000]]),
            Err(Error::Device(_))
        ));
        // Mismatched physical capacities refuse before any compile or launch.
        assert!(matches!(
            device.run_contraction(one.plan(), [&[0x3f80_0000, 0], &[0x3f80_0000]]),
            Err(Error::Limit)
        ));
        // A padded, offset C allocation keeps its padding and logical region.
        let emission = plan(caps, [2, 3, 2]);
        let mut ir = emission.plan().ir().clone();
        ir.output.offset = 3;
        ir.output.elements = 11;
        let origin = invar_device::Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let padded = invar_device::floating::verify(ir, caps, origin).unwrap();
        let a = corpus::words(&corpus::NORMAL, 6, 3);
        let b = corpus::words(&corpus::NORMAL, 6, 5);
        let physical = device.run_contraction(&padded, [&a, &b]).unwrap().output;
        assert_eq!(physical.len(), 11);
        assert!(physical[..3].iter().chain(&physical[7..]).all(|&w| w == 0));
        assert_eq!(physical[3..7], corpus::oracle([2, 3, 2], &a, &b)[..]);
        assert_eq!(device.quarantine_len(), 0);
    }
}

/// One encoding for every device-written status word: each word's legal bits
/// decode, and an untouched `UNWRITTEN` sentinel or any stray bit is refused.
/// Both rank-1 kernels claim their status word before any other statement.
#[test]
fn issue_m4_01_status_words_share_the_guard_encoding() {
    use invar_device::f32_ops::{CUDA_STATUS, Kind, METAL_STATUS, Node, Program, Scalar};
    use invar_device::f32_probe::{self, Report};
    use invar_device::floating::guard::{
        self, GUARD, InvalidWord, OVERFLOW, UNWRITTEN, decode_bits, msl_claim,
    };
    for (legal, ok, bad) in [
        (GUARD, &[0, 1][..], &[2, 3, UNWRITTEN, u32::MAX][..]),
        (CUDA_STATUS, &[0, OVERFLOW], &[GUARD, 3, 4, UNWRITTEN]),
        (
            METAL_STATUS,
            &[0, 1, 2, 3],
            &[4, 5, UNWRITTEN, UNWRITTEN | 1],
        ),
    ] {
        for &w in ok {
            assert_eq!(decode_bits(w, legal), Ok(w), "{legal} {w}");
        }
        for &w in bad {
            assert_eq!(decode_bits(w, legal), Err(InvalidWord(w)), "{legal} {w}");
        }
    }
    assert_eq!(guard::decode(0), Ok(false));
    assert_eq!(guard::decode(GUARD), Ok(true));
    assert_eq!(guard::decode(UNWRITTEN), Err(InvalidWord(UNWRITTEN)));
    assert_eq!(
        Report::evaluate(&f32_probe::unwritten()),
        Err(f32_probe::Error::Guard(InvalidWord(UNWRITTEN)))
    );
    let program = || Program {
        length: 2,
        inputs: vec![Scalar::F32, Scalar::F32],
        nodes: vec![Node::Input(0), Node::Input(1), Node::Mul(0, 1)],
        kind: Kind::Map,
    };
    let cuda =
        invar_device::f32_ops::emit(&ops::plan(program(), Capabilities::cuda_12_9_sm89())).unwrap();
    let head = cuda.find("unsigned flag = 0u;\n").unwrap() + "unsigned flag = 0u;\n".len();
    assert!(cuda[head..].starts_with(&format!("  atomicAnd(C + 2u, {CUDA_STATUS}u);\n")));
    let metal = invar_device::f32_ops::emit(&ops::plan(program(), METAL)).unwrap();
    let head = metal.find("uint flag = 0u;\n").unwrap() + "uint flag = 0u;\n".len();
    assert!(metal[head..].starts_with(&msl_claim(METAL_STATUS)));
}
