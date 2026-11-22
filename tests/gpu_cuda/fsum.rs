//! v0.2 reductions on a live CUDA device: kernel partials finished on the host
//! equal the CPU engines bit for bit at every size class, every leaf chunk
//! exponent and on adversarial inputs, including subnormals (no FTZ).
use invar_cuda::cuda::CudaDevice;
use invar_device::fsum::{Launch, Op, ROW};
use std::path::Path;

#[path = "../support/fsum_cases.rs"]
mod cases;

type Device = CudaDevice;

fn compiler() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}

fn open() -> Device {
    CudaDevice::open().unwrap()
}

/// CUDA never flushes subnormals, so every dispatch publishes.
fn device_bits(device: &mut Device, op: Op, a: &[u32], b: &[u32], shift: u32) -> Option<u32> {
    let second = op.is_dot().then_some(b);
    let result = device.run_fsum(op, a, second, shift, compiler()).unwrap();
    Some(body::finish(op, a.len(), &result.rows, &result.nodes))
}

/// The exact partials of `groups` blocks, finished on the host.
fn exact_bits(device: &mut Device, op: Op, a: &[u32], b: &[u32], groups: u32) -> u32 {
    let launch = Launch::new(op, a.len(), 0)
        .unwrap()
        .with_exact_groups(groups)
        .unwrap();
    let second = op.is_dot().then_some(b);
    let result = device
        .run_fsum_launch(launch, a, second, compiler())
        .unwrap();
    assert_eq!(result.rows.len(), groups as usize * ROW);
    body::finish(op, a.len(), &result.rows, &result.nodes)
}

fn guard_fires(_op: Op, _label: &str) -> bool {
    false
}

#[path = "../support/gpu_fsum.rs"]
mod body;

/// Resident GPU medians; compare with `tools/bench/baselines/cuda-cub-f32sum.cu`.
/// `cargo test --release --features cuda --test gpu_cuda fsum::bench -- --ignored`
#[test]
#[ignore = "benchmark; run alone under INVAR_GPU_EXCLUSIVE=1"]
fn bench() {
    let mut device = open();
    let mut rng = cases::Rng(0xbe9c);
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
