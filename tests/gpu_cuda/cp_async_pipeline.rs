//! ENG-06 (#61) part D on a real CUDA GPU (sm_80 or newer): the `cp.async`
//! pipelined contraction schedules give results bit-identical to the strict
//! oracle and to the default tiled schedule on adversarial shapes, including
//! gradual underflow. Ignored by default; needs a CUDA driver, NVRTC
//! 12.4-12.9 matching the device and the `invar-nvrtc` helper:
//! `--features cuda --test gpu_cuda -- --ignored cp_async_pipeline::`.
//! Correctness only; no timing is measured or claimed.
#[path = "../issue_m4_03_device/corpus.rs"]
mod corpus;

use std::path::Path;

use corpus::{DIRECT, LARGEST, NORMAL, SHAPES, SUBNORMAL};
use invar::contraction_source::{Contract, DeviceEmission};
use invar_cuda::cuda::CudaDevice;
use invar_device::contraction::{gemm_pipelined_v1, gemm_tiled_v1};
use invar_device::moves::Move;

fn compiler() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}

fn emit(dims: [u32; 3], device: &CudaDevice) -> DeviceEmission {
    let [r, k, n] = dims.map(u64::from);
    Contract::from_source(DIRECT.as_bytes(), "mat")
        .unwrap()
        .emit_device(
            DIRECT.as_bytes(),
            dims,
            [r * k, k * n],
            device.capabilities(),
        )
        .unwrap()
}

#[test]
#[ignore = "requires a real CUDA GPU (sm_80 or newer)"]
fn pipeline_gpu_cp_async_schedules_are_bit_exact() {
    let mut device = CudaDevice::open().expect("a real CUDA GPU with a matching NVRTC is required");
    eprintln!(
        "ENG-06 pipeline CUDA device={} caps={:?}",
        device.name(),
        device.capabilities()
    );
    assert!(device.strict_f32_probe(compiler()).unwrap().admits_strict());
    let schedules: Vec<_> = [2, 3, 4]
        .map(|stages| {
            let mut s = gemm_tiled_v1();
            s.name = format!("invar-f32-gemm-tiled-cp-async-{stages}-v1");
            s.moves.push(Move::Pipeline { stages });
            s
        })
        .into_iter()
        .chain([gemm_pipelined_v1()])
        .collect();
    let mut cells = 0;
    for dims in SHAPES.iter().chain(LARGEST.iter()).copied() {
        let [r, k, n] = dims.map(|v| v as usize);
        let e = emit(dims, &device);
        for (pattern, seed) in [(&NORMAL[..], 3), (&SUBNORMAL[..], 5)] {
            let a = corpus::words(pattern, r * k, seed);
            let b = corpus::words(pattern, k * n, seed + 2);
            let expected = corpus::oracle(dims, &a, &b);
            let tiled = device
                .run_tiled_contraction(e.plan(), [&a, &b], compiler())
                .unwrap_or_else(|error| panic!("tiled {dims:?}: {error:?}"));
            assert_eq!(tiled.output, expected, "tiled {dims:?}");
            for schedule in &schedules {
                let out = device
                    .run_scheduled_contraction(e.plan(), schedule, [&a, &b], compiler())
                    .unwrap_or_else(|error| panic!("{} {dims:?}: {error:?}", schedule.name));
                assert_eq!(out.output, expected, "{} {dims:?}", schedule.name);
                assert!(!out.nvrtc_log.contains("warning"), "{}", out.nvrtc_log);
                cells += r * n;
            }
        }
    }
    // Gradual underflow survives the pipeline: 2^-149 * 1 + 2^-126 * 0.5.
    let e = emit([1, 2, 1], &device);
    let out = device
        .run_scheduled_contraction(
            e.plan(),
            &gemm_pipelined_v1(),
            [&[1, 0x0080_0000], &[0x3f80_0000, 0x3f00_0000]],
            compiler(),
        )
        .unwrap();
    assert_eq!(out.output, [0x0040_0001]);
    assert!(cells > 0);
    eprintln!(
        "ENG-06 pipeline CUDA: {cells} cells bit-exact over {} schedules",
        schedules.len()
    );
}
