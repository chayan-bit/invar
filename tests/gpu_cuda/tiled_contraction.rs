//! M4-03 on CUDA: the shared tiled-contraction tests plus the CUDA-only ones.
use invar::contraction_source::{Contract, DeviceEmission};
use invar_cuda::cuda::CudaDevice;
use std::{path::Path, time::Instant};

#[path = "../issue_m4_03_device/corpus.rs"]
mod corpus;
use corpus::{DIRECT, LARGEST, NONFINITE, NORMAL, SHAPES, SUBNORMAL, TC_DIRECT};

type Device = CudaDevice;

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}
fn compiler() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}
fn open() -> Device {
    CudaDevice::open().expect("a real CUDA GPU with a matching NVRTC is required")
}
fn emit(dims: [u32; 3], device: &Device) -> DeviceEmission {
    emit_from(DIRECT, dims, device)
}
fn emit_from(source: &str, dims: [u32; 3], device: &Device) -> DeviceEmission {
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
fn run_tiled(
    device: &mut Device,
    plan: &invar_device::floating::Verified,
    inputs: [&[u32]; 2],
) -> Result<body::Out, String> {
    let r = device
        .run_tiled_contraction(plan, inputs, compiler())
        .map_err(|e| format!("{e:?}"))?;
    Ok(body::Out {
        output: r.output,
        compile_ms: r.compile_ms,
        kernel_ms: None,
        compiler_warned: r.nvrtc_log.contains("warning"),
    })
}

#[path = "../support/gpu_m4_03.rs"]
mod body;

#[test]
#[ignore = "requires a real CUDA GPU"]
fn gpu_tiled_contraction_is_bit_exact_on_adversarial_shapes() {
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
fn gpu_refusals_publish_nothing() {
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
fn gpu_tc_refuses_inputs_outside_binary16() {
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
fn gpu_tc_subnormal_correction_overhead_is_recorded() {
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
