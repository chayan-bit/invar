//! M4-01 strict binary32 contraction on a real CUDA GPU from the same verified
//! device plan Metal uses. Every test is ignored by default and needs a CUDA
//! driver, NVRTC 12.4-12.9 matching the device (12.8+ for Blackwell), and the
//! `invar-nvrtc` helper: `--features cuda f32_contraction:: -- --ignored`.
#![forbid(unsafe_code)]

#[path = "../issue_m4_01_device/corpus.rs"]
mod corpus;
#[path = "../issue_m4_01_device/ops.rs"]
mod ops;

#[test]
#[ignore = "requires a real CUDA GPU"]
fn gpu_v2_maps_folds_comparisons_conversions() {
    let _gpu = lock();
    let mut device = open();
    let caps = device.capabilities();
    let mut words = 0;
    for p in ops::programs() {
        let input = ops::packed(&p);
        let plan = ops::plan(p.clone(), caps);
        let expected = invar::f32_source::cpu(&plan, &input, 1).unwrap();
        let actual = device
            .run_f32_ops(&plan, &input, compiler())
            .unwrap_or_else(|error| panic!("{error:?}: {:?}", plan.ir().kernel));
        ops::same(
            &actual,
            &expected,
            p.output_type().unwrap() == invar_device::f32_ops::Scalar::F32,
        );
        words += actual.len();
    }
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
            let actual = device
                .run_f32_ops(&plan, &input, compiler())
                .unwrap_or_else(|error| panic!("{error:?}: {:?}", plan.ir().kernel));
            ops::same(
                &actual,
                &expected,
                invar_device::f32_ops::program(&plan)
                    .unwrap()
                    .output_type()
                    .unwrap()
                    == invar_device::f32_ops::Scalar::F32,
            );
        }
    }
    use invar_device::f32_ops::{Kind, Node, Program, Scalar};
    for kind in [Kind::Reduce { initial: 0 }, Kind::Scan { initial: 0 }] {
        let p = Program {
            length: 3,
            inputs: vec![Scalar::F32],
            nodes: vec![Node::Input(0)],
            kind,
        };
        let plan = ops::plan(p, caps);
        let input = [0x4b800000, 0x3f800000, 0xcb800000];
        assert_eq!(
            device.run_f32_ops(&plan, &input, compiler()).unwrap(),
            invar::f32_source::cpu(&plan, &input, 1).unwrap()
        );
    }
    let trunc = ops::plan(
        Program {
            length: 1,
            inputs: vec![Scalar::F32],
            nodes: vec![Node::Input(0), Node::TruncI32(0)],
            kind: Kind::Map,
        },
        caps,
    );
    for input in [0x7fc00001, 0x7f800000, 0x4f000000, 0xcf000001] {
        assert_eq!(
            device.run_f32_ops(&trunc, &[input], compiler()),
            Err(invar_cuda::Error::IntegerOverflow)
        );
    }
    assert_eq!(
        device
            .run_f32_ops(&trunc, &[0xcf000000], compiler())
            .unwrap(),
        [0x80000000]
    );
    assert_eq!(device.run_f32_ops(&trunc, &[1], compiler()).unwrap(), [0]);
    assert_eq!(device.quarantine_len(), 0);
    eprintln!("M4-01 v2 CUDA: {words} corpus words, ordered fold witnesses, conversion traps");
}

#[test]
#[ignore = "requires a real CUDA GPU"]
fn gpu_v2_ordinary_source_parity() {
    let _gpu = lock();
    let mut device = open();
    for source in ops::sources() {
        let expected = ops::reference(&source);
        let run = invar::f32_source::run_cuda_source(
            source.as_bytes(),
            1_000_000,
            &mut device,
            compiler(),
        )
        .unwrap();
        ops::parity(&run, &expected);
        assert_eq!(
            run.completed > 0,
            !source.contains("filled_f32(0u32"),
            "{source}"
        );
        assert!(run.refusals.is_empty(), "{:?}", run.refusals);
    }
}

use corpus::{FUEL, MAT, SHAPES};
use invar::tensor::{self, host};
use invar_cuda::{Runtime, cuda::CudaDevice};
use std::path::Path;

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

#[test]
#[ignore = "requires a real CUDA GPU"]
fn gpu_d1_probe_admits_strict_binary32() {
    let _gpu = lock();
    let mut device = open();
    eprintln!(
        "M4-01 CUDA device={} caps={:?}",
        device.name(),
        device.capabilities()
    );
    let report = device.strict_f32_probe(compiler()).unwrap().clone();
    eprintln!("{}", report.summary());
    for line in report.details() {
        eprintln!("  {line}");
    }
    // CUDA with --ftz=false --fmad=false must be exact; a failure here is a
    // real device or toolchain finding, never a pass.
    assert!(report.admits_strict(), "{}", report.summary());
    assert!(!report.flushes_subnormals() && !report.contracts_implicitly());
    // Cached per opened device; a fresh open reproduces the same words.
    assert_eq!(
        device.strict_f32_probe(compiler()).unwrap().observed(),
        report.observed()
    );
    let again = open().strict_f32_probe(compiler()).unwrap().clone();
    assert_eq!(again.observed(), report.observed());
}

#[test]
#[ignore = "requires a real CUDA GPU"]
fn gpu_ordinary_source_publishes_exact_device_results() {
    let _gpu = lock();
    for dims in SHAPES {
        let source = corpus::program(dims);
        let expected = corpus::reference(&source);
        assert_eq!(expected.result, Ok(invar::eval::Value::I32(7)), "{dims:?}");
        let mut device = host::cuda::Cuda::open(source.as_bytes(), "mat", compiler()).unwrap();
        assert!(device.probe().unwrap().admits_strict());
        let caps = device.capabilities();
        let mut runtime = Runtime::new(device, 2);
        let run = host::run_source(source.as_bytes(), FUEL, &mut runtime, caps).unwrap();
        // The device result replaced the source loop and main's in-source
        // recomputation (value and zero sign per cell) still returned 7.
        corpus::assert_parity(&run.observation, &expected);
        let reached = usize::from(corpus::reached(dims));
        assert_eq!(run.completed, reached, "{dims:?} {:?}", run.refusals);
        assert!(run.refusals.is_empty(), "{dims:?} {:?}", run.refusals);
        assert_eq!(runtime.quarantine_len(), 0);
    }
}

#[test]
#[ignore = "requires a real CUDA GPU"]
fn gpu_direct_contraction_matches_oracle_bit_for_bit() {
    let _gpu = lock();
    let mut device = open();
    let caps = device.capabilities();
    let mut cells = 0;
    for dims in SHAPES {
        let [r, k, c] = dims.map(|n| n as usize);
        let plan = tensor::emit_cuda(
            MAT.as_bytes(),
            "mat",
            dims,
            [(r * k) as u64, (k * c) as u64],
            caps,
        )
        .unwrap();
        for pattern in [&corpus::NORMAL[..], &corpus::SUBNORMAL[..]] {
            let a = corpus::words(pattern, r * k, 3);
            let b = corpus::words(pattern, k * c, 5);
            let result = device
                .run_contraction(plan.plan(), [&a, &b], compiler())
                .unwrap();
            assert_eq!(result.output, corpus::oracle(dims, &a, &b), "{dims:?}");
            cells += r * c;
        }
    }
    // Gradual underflow is observable in the published words.
    let plan = tensor::emit_cuda(MAT.as_bytes(), "mat", [1, 2, 1], [2, 2], caps).unwrap();
    let a = [0x0000_0001, 0x0080_0000];
    let b = [0x3f80_0000, 0x3f00_0000];
    let out = device
        .run_contraction(plan.plan(), [&a, &b], compiler())
        .unwrap();
    assert_eq!(out.output, [0x0040_0001]);
    assert!(corpus::is_subnormal(out.output[0]));
    // A padded, offset C allocation keeps its padding and logical region.
    let emission = tensor::emit_cuda(MAT.as_bytes(), "mat", [2, 3, 2], [6, 6], caps).unwrap();
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
    let physical = device
        .run_contraction(&padded, [&a, &b], compiler())
        .unwrap()
        .output;
    assert_eq!(physical.len(), 11);
    assert!(physical[..3].iter().chain(&physical[7..]).all(|&w| w == 0));
    assert_eq!(physical[3..7], corpus::oracle([2, 3, 2], &a, &b)[..]);
    // Typed refusals publish nothing.
    let one = tensor::emit_cuda(MAT.as_bytes(), "mat", [1, 1, 1], [1, 1], caps).unwrap();
    assert!(
        device
            .run_contraction(one.plan(), [&[0x7f80_0000], &[0x3f80_0000]], compiler())
            .is_err()
    );
    assert_eq!(
        device
            .run_contraction(one.plan(), [&[0, 0], &[0]], compiler())
            .unwrap_err(),
        invar_cuda::Error::Limit
    );
    assert_eq!(device.quarantine_len(), 0);
    eprintln!(
        "M4-01 CUDA direct corpus: {cells} cells bit-exact over {} shapes",
        SHAPES.len()
    );
}
