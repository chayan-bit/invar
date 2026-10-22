//! M4-03 tiled strict-f32 matrix contraction on Metal, from the same verified
//! plan as CUDA. Normal-range cells publish bit-exact; a dispatch whose
//! underflow guard fires is discarded (`SubnormalFallback`) so the source loop
//! keeps the exact CPU result. Needs a real Apple7+ GPU; ignored by default:
//! `--features metal issue_m4_03_metal_gpu -- --ignored --nocapture`.
#![cfg(all(feature = "metal", target_os = "macos"))]
#![forbid(unsafe_code)]

#[allow(dead_code)]
#[path = "../src/contraction_source.rs"]
mod contraction_source;
#[path = "issue_m4_03_device/corpus.rs"]
mod corpus;

use contraction_source::{Contract, DeviceEmission};
use corpus::{DIRECT, LARGEST, NONFINITE, NORMAL, SHAPES, SUBNORMAL, TC_DIRECT};
use invar::{check, front, general_net, hex, layout_contract, mir, numeric_contract, tensor_cache};
use invar_metal::{Error, MetalDevice};
use std::time::Instant;

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}
/// The tiled and Tensor Core kernels here are emitted per shape and some are
/// deliberately mutated, so no compiled-ahead library holds them: open the
/// developer source device (#204). `issue_metal_aot_gpu` covers the
/// compiled-ahead tensor route.
fn open() -> MetalDevice {
    let mut device =
        MetalDevice::open_runtime_source().expect("a real Apple7+ Metal GPU is required");
    assert!(device.admits_strict_contraction().unwrap());
    device
}
fn emit(dims: [u32; 3], device: &MetalDevice) -> DeviceEmission {
    emit_from(DIRECT, dims, device)
}
fn emit_from(source: &str, dims: [u32; 3], device: &MetalDevice) -> DeviceEmission {
    let [r, k, n] = dims.map(u64::from);
    Contract::from_source(source.as_bytes(), "mat")
        .unwrap()
        .emit_device(
            source.as_bytes(),
            dims,
            [r * k, k * n],
            device.capabilities(),
        )
        .unwrap()
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn issue_m4_03_metal_gpu_tiled_normal_cells_publish_bit_exact() {
    let _gpu = lock();
    let mut device = open();
    eprintln!("M4-03 Metal caps={:?}", device.capabilities());
    let mut cells = 0;
    for dims in SHAPES {
        let [r, k, n] = dims.map(|v| v as usize);
        let e = emit(dims, &device);
        let a = corpus::words(&NORMAL, r * k, 3);
        let b = corpus::words(&NORMAL, k * n, 5);
        let out = device.run_tiled_contraction(e.plan(), [&a, &b]).unwrap();
        assert_eq!(out.output, corpus::oracle(dims, &a, &b), "{dims:?}");
        cells += r * n;
    }
    eprintln!("M4-03 Metal tiled: {cells} normal-range cells bit-exact");
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn issue_m4_03_metal_gpu_subnormal_dispatches_fall_back_or_stay_exact() {
    let _gpu = lock();
    let mut device = open();
    let (mut exact, mut fallbacks) = (0, 0);
    for dims in SHAPES {
        let [r, k, n] = dims.map(|v| v as usize);
        let e = emit(dims, &device);
        let a = corpus::words(&SUBNORMAL, r * k, 5);
        let b = corpus::words(&SUBNORMAL, k * n, 7);
        let expected = corpus::oracle(dims, &a, &b);
        // Raw, unpublished words: the guard must fire whenever they differ.
        let raw = device
            .diagnose_tiled_source(e.plan(), e.source(), [&a, &b])
            .unwrap();
        if raw.output != expected {
            assert!(raw.flagged, "unflagged mismatch {dims:?}");
        }
        match device.run_tiled_contraction(e.plan(), [&a, &b]) {
            Ok(out) => {
                assert_eq!(out.output, expected, "{dims:?}");
                exact += 1;
            }
            Err(Error::SubnormalFallback) => {
                assert!(raw.flagged);
                fallbacks += 1;
            }
            Err(error) => panic!("{dims:?}: {error:?}"),
        }
    }
    eprintln!("M4-03 Metal subnormal: {exact} published exact, {fallbacks} CPU fallbacks");
    assert_eq!(exact + fallbacks, SHAPES.len());
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn issue_m4_03_metal_gpu_no_fma_contraction_or_reassociation_on_device() {
    let _gpu = lock();
    let mut device = open();
    let (dims, a, b) = corpus::fma_discriminator(70, 67);
    let e = emit(dims, &device);
    let out = device.run_tiled_contraction(e.plan(), [&a, &b]).unwrap();
    // A fused multiply-add would leave -2^-46 (0xa8800000) in every cell.
    assert!(
        out.output.iter().all(|&w| w == 0),
        "{:08x?}",
        &out.output[..4]
    );
    let dims = [64, 3, 64];
    let a = [0x4c00_0000, 0xcc00_0000, 0x3f80_0000].repeat(64);
    let b = vec![0x3f80_0000; 3 * 64];
    let e = emit(dims, &device);
    let out = device.run_tiled_contraction(e.plan(), [&a, &b]).unwrap();
    assert!(out.output.iter().all(|&w| w == 0x3f80_0000));
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn issue_m4_03_metal_gpu_largest_admitted_shapes_match_host_oracle() {
    let _gpu = lock();
    let mut device = open();
    for dims in LARGEST {
        let [r, k, n] = dims.map(|v| v as usize);
        let a = corpus::random_words(r * k, 0x9e37_79b9);
        let b = corpus::random_words(k * n, 0x85eb_ca6b);
        let expected = corpus::host_oracle(dims, &a, &b);
        let e = emit(dims, &device);
        let started = Instant::now();
        let out = device.run_tiled_contraction(e.plan(), [&a, &b]).unwrap();
        let wall = started.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(out.output, expected, "{dims:?}");
        for i in [0, r / 2, r - 1] {
            let row = corpus::oracle([1, dims[1], dims[2]], &a[i * k..(i + 1) * k], &b);
            assert_eq!(row, expected[i * n..(i + 1) * n], "row {i}");
        }
        eprintln!(
            "M4-03 Metal largest {dims:?}: bit-exact, groups={} compile_ms={:.1} kernel_ms={:.2} host_to_host_ms={wall:.1} receipt={}",
            e.geometry().groups,
            out.compile_ms,
            out.kernel_ms,
            e.receipt()
        );
    }
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn issue_m4_03_metal_gpu_refusals_publish_nothing() {
    let _gpu = lock();
    let mut device = open();
    let e = emit([3, 4, 5], &device);
    let a = corpus::words(&NONFINITE, 12, 1);
    let b = corpus::words(&NORMAL, 20, 3);
    assert!(matches!(
        device.run_tiled_contraction(e.plan(), [&a, &b]),
        Err(Error::Device(_))
    ));
    assert!(matches!(
        device.run_tiled_contraction(e.plan(), [&a[..11], &b]),
        Err(Error::Limit)
    ));
    // A wrong lane count raises the kernel's flag: discarded, never published.
    let a = corpus::words(&NORMAL, 12, 3);
    let raw = device
        .diagnose_source(e.plan(), e.source(), [&a, &b])
        .unwrap();
    assert!(raw.flagged);
    assert!(raw.output.iter().all(|&w| w == 0));
}

/// The tiled kernel's flag uses the shared guard encoding: an unclaimed
/// UNWRITTEN word or a value outside {0, GUARD} is a typed refusal.
#[test]
#[ignore = "requires a real Metal GPU"]
fn issue_m4_03_metal_gpu_rejects_unwritten_and_illegal_tiled_flags() {
    use invar_device::floating::guard::{GUARD, InvalidWord, UNWRITTEN, msl_claim};
    let _gpu = lock();
    let mut device = open();
    let e = emit([17, 9, 13], &device);
    let a = corpus::words(&NORMAL, 17 * 9, 3);
    let b = corpus::words(&NORMAL, 9 * 13, 5);
    let claim = msl_claim(GUARD);
    assert!(e.source().contains(&claim));
    let raw = device
        .diagnose_tiled_source(e.plan(), e.source(), [&a, &b])
        .unwrap();
    assert!(!raw.flagged);
    let unwritten = e.source().replace(&claim, "");
    let illegal = e.source().replace(
        &claim,
        "  atomic_store_explicit(F, 2u, memory_order_relaxed);\n",
    );
    for (source, word) in [(unwritten, UNWRITTEN), (illegal, 2)] {
        assert!(matches!(
            device.diagnose_tiled_source(e.plan(), &source, [&a, &b]),
            Err(Error::InvalidGuard(InvalidWord(w))) if w == word
        ));
    }
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn issue_m4_03_metal_gpu_fma_mutations_are_observable() {
    // An explicit fused step must fail the strict check on the device. The
    // pinned pragma is also dropped to record whether this compiler would
    // fuse the tiled kernel on its own (the guard keeps each product live).
    let _gpu = lock();
    let mut device = open();
    let (dims, a, b) = corpus::fma_discriminator(70, 67);
    let e = emit(dims, &device);
    let fused = e.source().replace(
        "const float next = acc[r][c] + product;",
        "const float next = fma(as_type<float>(aw[r]), as_type<float>(bw[c]), acc[r][c]);",
    );
    assert_ne!(fused, e.source());
    let raw = device
        .diagnose_tiled_source(e.plan(), &fused, [&a, &b])
        .unwrap();
    assert!(!raw.flagged);
    assert!(raw.output.iter().all(|&w| w == 0xa880_0000));
    let unpinned = e.source().replace("#pragma METAL fp contract(off)\n", "");
    assert_ne!(unpinned, e.source());
    let raw = device
        .diagnose_tiled_source(e.plan(), &unpinned, [&a, &b])
        .unwrap();
    let count = raw.output.iter().filter(|&&w| w == 0xa880_0000).count();
    eprintln!(
        "M4-03 Metal explicit fma: all {} cells fused and rejected; without the pragma: {count} fused",
        raw.output.len()
    );
    assert!(raw.output.iter().all(|&w| w == 0 || w == 0xa880_0000));
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn issue_m4_03_metal_gpu_tc_simdgroup_results_hold_the_amended_d5_bound() {
    let _gpu = lock();
    let mut device = open();
    let mut worst = 0.0f64;
    let (mut cells, mut differ) = (0usize, 0usize);
    for (label, dims, a, b) in corpus::tc_cases() {
        let e = emit_from(TC_DIRECT, dims, &device);
        let started = Instant::now();
        let out = device
            .run_tiled_contraction(e.plan(), [&a, &b])
            .unwrap_or_else(|error| panic!("{label} {dims:?}: {error:?}"));
        let wall = started.elapsed().as_secs_f64() * 1000.0;
        let report = invar_device::tc::verify(dims, &a, &b, &out.output)
            .unwrap_or_else(|f| panic!("{label} {dims:?}: {f:?}"));
        let reference = invar_device::tc::reference(dims, &a, &b).unwrap();
        let off = out
            .output
            .iter()
            .zip(&reference)
            .filter(|(x, y)| x != y)
            .count();
        eprintln!(
            "M4-03 Metal tc {label} {dims:?}: within bound, max_ratio={:.3e} cells_differing_from_reference={off} kernel_ms={:.2} host_to_host_ms={wall:.1}",
            report.max_ratio, out.kernel_ms
        );
        if label == "counterexample" {
            eprintln!(
                "M4-03 Metal tc counterexample K={}: result={:#010x} reference={:#010x} receipt={}",
                dims[1],
                out.output[0],
                reference[0],
                e.receipt()
            );
        }
        worst = worst.max(report.max_ratio);
        cells += report.cells;
        differ += off;
    }
    eprintln!(
        "M4-03 Metal tc: {cells} cells within the amended D5 bound, worst ratio {worst:.3e}, {differ} differ from the rounded exact sum"
    );
}

#[test]
#[ignore = "requires a real Metal GPU"]
fn issue_m4_03_metal_gpu_tc_refuses_inputs_outside_binary16() {
    let _gpu = lock();
    let mut device = open();
    let e = emit_from(TC_DIRECT, [2, 2, 2], &device);
    for bad in [0x477f_f000, 0x7f80_0000, 0x7fc0_0000, 0x7f7f_ffff] {
        let a = [0x3f80_0000, bad, 0x3f80_0000, 0x3f80_0000];
        assert!(matches!(
            device.run_tiled_contraction(e.plan(), [&a, &[0x3f80_0000; 4]]),
            Err(Error::Device(message)) if message.contains("binary16")
        ));
    }
}

/// Full output boundary with K=1 fits Metal's unchanged 2 GiB budget.
/// The M5-01 K=16384 square plan still needs the shared budget raised.
#[test]
#[ignore = "requires a real GPU and 1 GiB output storage; run alone"]
fn issue_m4_03_metal_gpu_size_2_28_is_bit_exact() {
    let _gpu = lock();
    let mut device = open();
    let dims = [16384, 1, 16384];
    let a: Vec<u32> = (0..16384)
        .map(|i| [1.0f32, 2.0, 4.0, 8.0][i % 4].to_bits())
        .collect();
    let b: Vec<u32> = (0..16384)
        .map(|i| [0.5f32, 1.0, 1.5, 2.0][i % 4].to_bits())
        .collect();
    for source in [DIRECT, TC_DIRECT] {
        let e = emit_from(source, dims, &device);
        let out = device.run_tiled_contraction(e.plan(), [&a, &b]).unwrap();
        assert_eq!(out.output.len(), 1 << 28);
        for (cell, &word) in out.output.iter().enumerate() {
            let expected = f32::from_bits(a[cell / 16384]) * f32::from_bits(b[cell % 16384]);
            assert_eq!(
                word,
                expected.to_bits(),
                "cell={cell} policy={}",
                e.plan().ir().numeric_policy
            );
        }
        eprintln!(
            "M4-03 metal 2^28 output cells bit-exact: dims={dims:?}, policy={}; square K=16384 remains shared-budget blocked",
            e.plan().ir().numeric_policy
        );
    }
}
