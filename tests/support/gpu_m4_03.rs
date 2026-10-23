//! M4-03 tiled strict-f32 matrix contraction on a real GPU, generated from the
//! direct-cell source family and the same verified plan on CUDA and Metal.
//! Every test is ignored by default and needs a real device. Shared by the
//! CUDA and Metal binaries; the including module supplies `Device`, `open`,
//! `lock`, `run_tiled` and `emit`/`emit_from` (which need the device's capabilities).
use super::corpus::{self, DIRECT, LARGEST, TC_DIRECT};
use super::{Device, emit, emit_from, lock, open, run_tiled};
use std::time::Instant;

/// What each backend's tiled result reports, device-independently.
pub struct Out {
    pub output: Vec<u32>,
    pub compile_ms: f64,
    /// Device time (Metal reports it; the CUDA runtime does not).
    pub kernel_ms: Option<f64>,
    /// NVRTC diagnostics were produced (CUDA only; Metal reports none).
    pub compiler_warned: bool,
}

#[test]
#[ignore = "requires a real GPU"]
fn issue_m4_03_gpu_no_fma_contraction_or_reassociation_on_device() {
    let _gpu = lock();
    let mut device: Device = open();
    let (dims, a, b) = corpus::fma_discriminator(70, 67);
    let e = emit(dims, &device);
    let out = run_tiled(&mut device, e.plan(), [&a, &b]).unwrap();
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
    let out = run_tiled(&mut device, e.plan(), [&a, &b]).unwrap();
    assert!(out.output.iter().all(|&w| w == 0x3f80_0000));
}

#[test]
#[ignore = "requires a real GPU"]
fn issue_m4_03_gpu_largest_admitted_shapes_match_host_oracle() {
    let _gpu = lock();
    let mut device: Device = open();
    for dims in LARGEST {
        let [r, k, n] = dims.map(|v| v as usize);
        let a = corpus::random_words(r * k, 0x9e37_79b9);
        let b = corpus::random_words(k * n, 0x85eb_ca6b);
        let expected = corpus::host_oracle(dims, &a, &b);
        let e = emit(dims, &device);
        let started = Instant::now();
        let out = run_tiled(&mut device, e.plan(), [&a, &b]).unwrap();
        let wall = started.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(out.output, expected, "{dims:?}");
        // Spot-check the host oracle itself against the software oracle.
        for i in [0, r / 2, r - 1] {
            let row = corpus::oracle([1, dims[1], dims[2]], &a[i * k..(i + 1) * k], &b);
            assert_eq!(row, expected[i * n..(i + 1) * n], "row {i}");
        }
        eprintln!(
            "M4-03 largest {dims:?}: bit-exact, groups={} compile_ms={:.1} kernel_ms={:?} host_to_host_ms={wall:.1} receipt={}",
            e.geometry().groups,
            out.compile_ms,
            out.kernel_ms,
            e.receipt()
        );
    }
}

#[test]
#[ignore = "requires a real GPU"]
fn issue_m4_03_gpu_tc_tensor_core_results_hold_the_amended_d5_bound() {
    let _gpu = lock();
    let mut device: Device = open();
    let mut worst = 0.0f64;
    let (mut cells, mut differ) = (0usize, 0usize);
    for (label, dims, a, b) in corpus::tc_cases() {
        let e = emit_from(TC_DIRECT, dims, &device);
        let started = Instant::now();
        let out = run_tiled(&mut device, e.plan(), [&a, &b])
            .unwrap_or_else(|error| panic!("{label} {dims:?}: {error}"));
        let wall = started.elapsed().as_secs_f64() * 1000.0;
        assert!(!out.compiler_warned, "{label} {dims:?}");
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
            "M4-03 tc {label} {dims:?}: within bound, max_ratio={:.3e} cells_differing_from_reference={off} kernel_ms={:?} host_to_host_ms={wall:.1}",
            report.max_ratio, out.kernel_ms
        );
        if label == "counterexample" {
            eprintln!(
                "M4-03 tc counterexample K={}: result={:#010x} reference={:#010x} receipt={}",
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
        "M4-03 tc: {cells} cells within the amended D5 bound, worst ratio {worst:.3e}, {differ} differ from the rounded exact sum"
    );
}

/// Full output boundary with K=1 under the shared admission budget.
/// The M5-01 K=16384 square workload is not executed by this correctness test.
#[test]
#[ignore = "requires a real GPU and 1 GiB output storage; run alone"]
fn issue_m4_03_gpu_size_2_28_is_bit_exact() {
    let _gpu = lock();
    let mut device: Device = open();
    let dims = [16384, 1, 16384];
    let a: Vec<u32> = (0..16384)
        .map(|i| [1.0f32, 2.0, 4.0, 8.0][i % 4].to_bits())
        .collect();
    let b: Vec<u32> = (0..16384)
        .map(|i| [0.5f32, 1.0, 1.5, 2.0][i % 4].to_bits())
        .collect();
    for source in [DIRECT, TC_DIRECT] {
        let e = emit_from(source, dims, &device);
        let out = run_tiled(&mut device, e.plan(), [&a, &b]).unwrap();
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
            "M4-03 2^28 output cells bit-exact: dims={dims:?}, policy={}; square K=16384 not executed",
            e.plan().ir().numeric_policy
        );
    }
}
