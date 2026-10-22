//! v0.2 reductions on a live CUDA device: kernel partials finished on the host
//! equal the CPU engines bit for bit at every size class, every leaf chunk
//! exponent and on adversarial inputs, including subnormals (no FTZ).
#![cfg(feature = "cuda")]

#[path = "support/fsum_cases.rs"]
mod cases;

use cases::{Rng, adversarial_vectors};
use invar::fsum;
use invar_cuda::cuda::CudaDevice;
use invar_device::fsum::Op;
use std::path::Path;

const OPS: [Op; 4] = [Op::SumTree, Op::DotTree, Op::SumExact, Op::DotExact];

fn compiler() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}

fn expected(op: Op, a: &[u32], b: &[u32]) -> u32 {
    match op {
        Op::SumTree => fsum::sum_tree(a),
        Op::DotTree => fsum::dot_tree(a, b),
        Op::SumExact => fsum::sum_exact(a),
        Op::DotExact => fsum::dot_exact(a, b),
    }
}

fn device_bits(device: &mut CudaDevice, op: Op, a: &[u32], b: &[u32], shift: u32) -> u32 {
    let second = op.is_dot().then_some(b);
    let result = device.run_fsum(op, a, second, shift, compiler()).unwrap();
    if op.is_exact() {
        fsum::finish_rows(&result.rows, a.len() as u64, op.is_dot())
    } else {
        let nodes = &result.nodes;
        fsum::canonical(fsum::fold_aligned(nodes.len(), |g| {
            f32::from_bits(nodes[g])
        }))
    }
}

#[test]
fn v02_fsum_cuda_adversarial_vectors_match_cpu_bits() {
    let mut device = CudaDevice::open().unwrap();
    for (label, a, b) in adversarial_vectors()
        .into_iter()
        .filter(|(_, a, _)| !a.is_empty())
    {
        for op in OPS {
            assert_eq!(
                device_bits(&mut device, op, &a, &b, 2),
                expected(op, &a, &b),
                "{label} {op:?}"
            );
        }
    }
}

#[test]
fn v02_fsum_cuda_group_boundaries_and_leaf_shifts_are_bit_identical() {
    let mut device = CudaDevice::open().unwrap();
    let mut rng = Rng(0x3e7a1);
    for n in [
        1usize,
        255,
        2047,
        2048,
        2049,
        8191,
        8193,
        65_543,
        1 << 20,
        3_000_017,
    ] {
        let a = (0..n).map(|_| rng.word()).collect::<Vec<_>>();
        let b = (0..n).map(|_| rng.finite()).collect::<Vec<_>>();
        // Finite inputs keep every rounding visible; `a` mostly ends in NaN.
        let finite = (0..n).map(|_| rng.finite()).collect::<Vec<_>>();
        for op in OPS {
            for input in [&a, &finite] {
                let want = expected(op, input, &b);
                for shift in 0..=3 {
                    let got = device_bits(&mut device, op, input, &b, shift);
                    assert_eq!(got, want, "n={n} {op:?} {shift}");
                }
            }
        }
    }
}

#[test]
fn v02_fsum_cuda_subnormal_and_contraction_cases() {
    let mut device = CudaDevice::open().unwrap();
    let subnormal = (1..=4096u32).map(|i| i % 0x7f_ffff).collect::<Vec<_>>();
    let tiny = vec![0x1f80_0000u32; 64];
    let cancel = vec![0x0080_0001u32, 0x8080_0000];
    let near = vec![0x3f80_0001u32; 4096];
    let minus = vec![0xbf80_0002u32; 4096];
    let mixed = near
        .iter()
        .zip(&minus)
        .flat_map(|(&x, &y)| [x, y])
        .collect::<Vec<_>>();
    for a in [&subnormal, &tiny, &cancel, &near, &mixed] {
        for op in OPS {
            assert_eq!(
                device_bits(&mut device, op, a, a, 2),
                expected(op, a, a),
                "{op:?} {:x}",
                a[0]
            );
        }
    }
}

/// Resident GPU medians; compare with `bench/baselines/cuda-cub-f32sum.cu`.
/// `cargo test --release --features cuda --test v02_fsum_cuda -- --ignored`
#[test]
#[ignore = "benchmark; run alone under INVAR_GPU_EXCLUSIVE=1"]
fn v02_fsum_cuda_bench() {
    let mut device = CudaDevice::open().unwrap();
    let mut rng = Rng(0xbe9c);
    for log2 in [20u32, 24, 26] {
        let a = (0..1usize << log2)
            .map(|_| rng.finite())
            .collect::<Vec<_>>();
        for (op, shifts) in [(Op::SumTree, 0..=3), (Op::SumExact, 0..=0)] {
            for shift in shifts {
                let bench = device.benchmark_fsum(op, &a, shift, compiler()).unwrap();
                println!(
                    "cuda {op:?} n=2^{log2} shift={shift} median_ms={:.4}",
                    bench.median_ms
                );
            }
        }
    }
}
