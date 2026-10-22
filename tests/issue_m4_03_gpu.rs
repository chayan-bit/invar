//! M4-03 tiled strict-f32 matrix contraction on a real CUDA GPU, generated
//! from the direct-cell source family and the same verified plan Metal uses.
//! Every test is ignored by default and needs a CUDA driver, NVRTC 12.4-12.9
//! matching the device (12.8+ for Blackwell) and the `invar-nvrtc` helper:
//! `--features cuda issue_m4_03_gpu -- --ignored --nocapture`.
#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

#[allow(dead_code)]
#[path = "../src/contraction_source.rs"]
mod contraction_source;
#[path = "issue_m4_03_device/corpus.rs"]
mod corpus;

use contraction_source::{Contract, DeviceEmission};
use corpus::{DIRECT, LARGEST, NONFINITE, NORMAL, SHAPES, SUBNORMAL, TC_DIRECT};
use invar::{check, front, general_net, hex, layout_contract, mir, numeric_contract, tensor_cache};
use invar_cuda::cuda::CudaDevice;
use std::{path::Path, time::Instant};

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}
fn compiler() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}
fn open() -> CudaDevice {
    CudaDevice::open().expect("a real CUDA GPU with a matching NVRTC is required")
}
fn emit(dims: [u32; 3], device: &CudaDevice) -> DeviceEmission {
    emit_from(DIRECT, dims, device)
}
fn emit_from(source: &str, dims: [u32; 3], device: &CudaDevice) -> DeviceEmission {
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
#[ignore = "requires a real CUDA GPU"]
fn issue_m4_03_gpu_tiled_contraction_is_bit_exact_on_adversarial_shapes() {
    let _gpu = lock();
    let mut device = open();
    eprintln!(
        "M4-03 CUDA device={} caps={:?}",
        device.name(),
        device.capabilities()
    );
    assert!(device.strict_f32_probe(compiler()).unwrap().admits_strict());
    let mut cells = 0;
    for dims in SHAPES {
        let [r, k, n] = dims.map(|v| v as usize);
        let e = emit(dims, &device);
        for (pattern, seed) in [(&NORMAL[..], 3), (&SUBNORMAL[..], 5)] {
            let a = corpus::words(pattern, r * k, seed);
            let b = corpus::words(pattern, k * n, seed + 2);
            let out = device
                .run_tiled_contraction(e.plan(), [&a, &b], compiler())
                .unwrap_or_else(|error| panic!("{dims:?}: {error:?}"));
            assert_eq!(out.output, corpus::oracle(dims, &a, &b), "{dims:?}");
            assert!(!out.nvrtc_log.contains("warning"), "{}", out.nvrtc_log);
            cells += r * n;
        }
    }
    // Gradual underflow survives tiling: 2^-149 * 1 + 2^-126 * 0.5.
    let e = emit([1, 2, 1], &device);
    let out = device
        .run_tiled_contraction(
            e.plan(),
            [&[1, 0x0080_0000], &[0x3f80_0000, 0x3f00_0000]],
            compiler(),
        )
        .unwrap();
    assert_eq!(out.output, [0x0040_0001]);
    eprintln!(
        "M4-03 CUDA tiled: {cells} cells bit-exact over {} shapes",
        SHAPES.len()
    );
}

#[test]
#[ignore = "requires a real CUDA GPU"]
fn issue_m4_03_gpu_no_fma_contraction_or_reassociation_on_device() {
    let _gpu = lock();
    let mut device = open();
    let (dims, a, b) = corpus::fma_discriminator(70, 67);
    let e = emit(dims, &device);
    let out = device
        .run_tiled_contraction(e.plan(), [&a, &b], compiler())
        .unwrap();
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
    let out = device
        .run_tiled_contraction(e.plan(), [&a, &b], compiler())
        .unwrap();
    assert!(out.output.iter().all(|&w| w == 0x3f80_0000));
}

#[test]
#[ignore = "requires a real CUDA GPU"]
fn issue_m4_03_gpu_largest_admitted_shapes_match_host_oracle() {
    let _gpu = lock();
    let mut device = open();
    for dims in LARGEST {
        let [r, k, n] = dims.map(|v| v as usize);
        let a = corpus::random_words(r * k, 0x9e37_79b9);
        let b = corpus::random_words(k * n, 0x85eb_ca6b);
        let expected = corpus::host_oracle(dims, &a, &b);
        let e = emit(dims, &device);
        let started = Instant::now();
        let out = device
            .run_tiled_contraction(e.plan(), [&a, &b], compiler())
            .unwrap();
        let wall = started.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(out.output, expected, "{dims:?}");
        // Spot-check the host oracle itself against the software oracle.
        let rows = [0, r / 2, r - 1];
        for &i in &rows {
            let row = corpus::oracle([1, dims[1], dims[2]], &a[i * k..(i + 1) * k], &b);
            assert_eq!(row, expected[i * n..(i + 1) * n], "row {i}");
        }
        eprintln!(
            "M4-03 CUDA largest {dims:?}: bit-exact, groups={} compile_ms={:.1} host_to_host_ms={wall:.1} receipt={}",
            e.geometry().groups,
            out.compile_ms,
            e.receipt()
        );
    }
}

#[test]
#[ignore = "requires a real CUDA GPU"]
fn issue_m4_03_gpu_refusals_publish_nothing() {
    let _gpu = lock();
    let mut device = open();
    let e = emit([3, 4, 5], &device);
    let a = corpus::words(&NONFINITE, 12, 1);
    let b = corpus::words(&NORMAL, 20, 3);
    // Infinities/NaN are refused before submission (the source loop keeps
    // the CPU result); wrong physical lengths are a typed limit refusal.
    assert!(
        device
            .run_tiled_contraction(e.plan(), [&a, &b], compiler())
            .is_err()
    );
    assert_eq!(
        device
            .run_tiled_contraction(e.plan(), [&a[..11], &b], compiler())
            .unwrap_err(),
        invar_cuda::Error::Limit
    );
    // A plan for another device is refused before any compile or launch.
    let other = Contract::from_source(DIRECT.as_bytes(), "mat")
        .unwrap()
        .emit_device(
            DIRECT.as_bytes(),
            [3, 4, 5],
            [12, 20],
            invar_device::Capabilities::cuda_12_9_sm89(),
        )
        .unwrap();
    if other.plan().capabilities() != device.capabilities() {
        let b = corpus::words(&NORMAL, 20, 3);
        let a = corpus::words(&NORMAL, 12, 3);
        assert_eq!(
            device
                .run_tiled_contraction(other.plan(), [&a, &b], compiler())
                .unwrap_err(),
            invar_cuda::Error::Limit
        );
    }
    assert_eq!(device.quarantine_len(), 0);
}

#[test]
#[ignore = "requires a real CUDA GPU"]
fn issue_m4_03_gpu_tc_tensor_core_results_hold_the_amended_d5_bound() {
    let _gpu = lock();
    let mut device = open();
    let mut worst = 0.0f64;
    let (mut cells, mut differ) = (0usize, 0usize);
    for (label, dims, a, b) in corpus::tc_cases() {
        let e = emit_from(TC_DIRECT, dims, &device);
        let started = Instant::now();
        let out = device
            .run_tiled_contraction(e.plan(), [&a, &b], compiler())
            .unwrap_or_else(|error| panic!("{label} {dims:?}: {error:?}"));
        let wall = started.elapsed().as_secs_f64() * 1000.0;
        assert!(!out.nvrtc_log.contains("warning"), "{}", out.nvrtc_log);
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
            "M4-03 CUDA tc {label} {dims:?}: within bound, max_ratio={:.3e} cells_differing_from_reference={off} host_to_host_ms={wall:.1}",
            report.max_ratio
        );
        if label == "counterexample" {
            eprintln!(
                "M4-03 CUDA tc counterexample K={}: result={:#010x} reference={:#010x} receipt={}",
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
        "M4-03 CUDA tc: {cells} cells within the amended D5 bound, worst ratio {worst:.3e}, {differ} differ from the rounded exact sum"
    );
}

#[test]
#[ignore = "requires a real CUDA GPU"]
fn issue_m4_03_gpu_tc_refuses_inputs_outside_binary16() {
    let _gpu = lock();
    let mut device = open();
    let e = emit_from(TC_DIRECT, [2, 2, 2], &device);
    for bad in [0x477f_f000, 0x7f80_0000, 0x7fc0_0000, 0x7f7f_ffff] {
        let a = [0x3f80_0000, bad, 0x3f80_0000, 0x3f80_0000];
        let error = device
            .run_tiled_contraction(e.plan(), [&a, &[0x3f80_0000; 4]], compiler())
            .unwrap_err();
        assert!(
            matches!(&error, invar_cuda::Error::Driver(m) if m.contains("binary16")),
            "{error:?}"
        );
    }
    assert_eq!(device.quarantine_len(), 0);
}

/// Second D5 amendment cost, recorded and never gated: the largest admitted
/// shape (2^26 multiply-adds) with normal binary16 words only (the warp vote
/// never fires) and with one subnormal in every 16x16 A tile (the correction
/// pass runs in every k-tile). Time is host to host minus NVRTC compile,
/// median of 7, with the same shape and transfer bytes for both inputs, so
/// the difference is the kernel's (at this size, below the run-to-run noise).
#[test]
#[ignore = "requires a real CUDA GPU"]
fn issue_m4_03_gpu_tc_subnormal_correction_overhead_is_recorded() {
    let _gpu = lock();
    let mut device = open();
    let dims = LARGEST[0];
    let [r, k, c] = dims.map(|v| v as usize);
    let e = emit_from(TC_DIRECT, dims, &device);
    let b = corpus::tc_normal_words(k * c, 63);
    let normal = corpus::tc_normal_words(r * k, 61);
    let mut sub = normal.clone();
    let tiny = corpus::tc_subnormal_words(r * k, 65);
    for (n, w) in sub.iter_mut().enumerate() {
        if (n / k).is_multiple_of(16) && (n % k).is_multiple_of(16) {
            *w = tiny[n];
        }
    }
    let mut medians = Vec::new();
    for (label, a) in [("normal-only", &normal), ("subnormal-every-k-tile", &sub)] {
        let mut ms = Vec::new();
        for _ in 0..7 {
            let started = Instant::now();
            let out = device
                .run_tiled_contraction(e.plan(), [a, &b], compiler())
                .unwrap();
            ms.push(started.elapsed().as_secs_f64() * 1000.0 - out.compile_ms);
            invar_device::tc::verify(dims, a, &b, &out.output).unwrap();
        }
        ms.sort_by(f64::total_cmp);
        medians.push(ms[3]);
        eprintln!(
            "M4-03 CUDA tc overhead {dims:?} {label}: median_ms={:.3} runs={ms:.3?}",
            ms[3]
        );
    }
    eprintln!(
        "M4-03 CUDA tc correction overhead {dims:?}: {:.3} ms ({:.1}%) with a subnormal in every k-tile; not a gate",
        medians[1] - medians[0],
        (medians[1] / medians[0] - 1.0) * 100.0
    );
}

/// Full output boundary with K=1 under the shared admission budget.
/// The M5-01 K=16384 square workload is not executed by this correctness test.
#[test]
#[ignore = "requires a real GPU and 1 GiB output storage; run alone"]
fn issue_m4_03_cuda_gpu_size_2_28_is_bit_exact() {
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
        let out = device
            .run_tiled_contraction(e.plan(), [&a, &b], compiler())
            .unwrap();
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
            "M4-03 cuda 2^28 output cells bit-exact: dims={dims:?}, policy={}; square K=16384 not executed",
            e.plan().ir().numeric_policy
        );
    }
}
