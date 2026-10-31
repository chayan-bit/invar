//! M4-03 on Metal: the shared tiled-contraction tests plus the Metal-only ones
//! (subnormal fallback, guard words, mutations). Normal-range cells publish
//! bit-exact; a dispatch whose underflow guard fires is discarded
//! (`SubnormalFallback`) so the source loop keeps the exact CPU result.
use invar::contraction_source::{Contract, DeviceEmission};
use invar_metal::{Error, MetalDevice};

#[path = "../issue_m4_03_device/corpus.rs"]
mod corpus;
use corpus::{DIRECT, NONFINITE, NORMAL, SHAPES, SUBNORMAL, TC_DIRECT};

type Device = MetalDevice;

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}
/// The tiled and Tensor Core kernels here are emitted per shape and some are
/// deliberately mutated, so no compiled-ahead library holds them: open the
/// developer source device. `compiled_ahead` covers the compiled-ahead tensor route.
fn open() -> Device {
    let mut device =
        MetalDevice::open_runtime_source().expect("a real Apple7+ Metal GPU is required");
    assert!(device.admits_strict_contraction().unwrap());
    device
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
        .run_tiled_contraction(plan, inputs)
        .map_err(|e| format!("{e:?}"))?;
    Ok(body::Out {
        output: r.output,
        compile_ms: r.compile_ms,
        kernel_ms: Some(r.kernel_ms),
        compiler_warned: false,
    })
}

#[path = "../support/gpu_m4_03.rs"]
mod body;

#[test]
#[ignore = "requires a real Metal GPU"]
fn metal_gpu_tiled_normal_cells_publish_bit_exact() {
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
fn metal_gpu_subnormal_dispatches_fall_back_or_stay_exact() {
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
fn metal_gpu_refusals_publish_nothing() {
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
fn metal_gpu_rejects_unwritten_and_illegal_tiled_flags() {
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
fn metal_gpu_fma_mutations_are_observable() {
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
fn metal_gpu_tc_refuses_inputs_outside_binary16() {
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
