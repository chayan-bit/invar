#![forbid(unsafe_code)]
use invar_device::sorting::{Kind, Plan};

#[test]
fn sort_device_admission_and_ordered_launch_contract() {
    assert!(Plan::new((1 << 28) + 1, Kind::U32).is_err());
    assert!(Plan::new(1 << 28, Kind::F32).is_ok());
    for length in [0, 1, 255, 256, 257, 65537, 1 << 26] {
        let plan = Plan::new(length, Kind::I32).unwrap();
        let launches = plan.launches();
        assert_eq!(launches.len(), if length == 0 { 0 } else { 32 });
        for (pass, round) in launches.chunks(4).enumerate() {
            assert_eq!(
                round.iter().map(|l| l.kernel).collect::<Vec<_>>(),
                [0, 1, 2, 3]
            );
            assert_eq!(plan.parameters(pass)[4], (pass * 4) as u32);
            assert!(
                round
                    .iter()
                    .all(|l| l.blocks > 0 && l.buffers.iter().all(|&b| b < 16))
            );
        }
        let words = plan.buffer_words();
        assert_eq!(words[0], length as usize);
        assert!(words.iter().sum::<usize>() * 4 <= 8 * 1024 * 1024 * 1024);
    }
    let plan = Plan::new(3, Kind::U32).unwrap();
    assert!(plan.validate_inputs(&[1, 2, 3], &[0, 1, 2]).is_ok());
    assert!(plan.validate_inputs(&[1, 2], &[0, 1, 2]).is_err());
    assert!(plan.validate_inputs(&[1, 2, 3], &[0, 1]).is_err());
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn differential(mut run: impl FnMut(&Plan, &[u32], &[u32]) -> (Vec<u32>, Vec<u32>)) {
    let ids: Vec<u32> = (0..137).collect();
    let (_, actual) = run(&Plan::new(137, Kind::U32).unwrap(), &vec![7; 137], &ids);
    assert_eq!(actual, ids, "all-equal keys must preserve payload order");
    for kind in [Kind::U32, Kind::I32, Kind::F32] {
        for n in [0, 1, 31, 32, 33, 255, 256, 257, 4097, 65537, 1048589] {
            let mut state = 0x5eeda11u32;
            let edges = [
                0,
                0x80000000,
                0x7f800000,
                0xff800000,
                0x7fc00001,
                0xff800001,
                1,
                0x80000001,
                u32::MAX,
            ];
            let keys: Vec<u32> = (0..n)
                .map(|i| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    if i % 3 == 0 {
                        edges[i % edges.len()]
                    } else {
                        state
                    }
                })
                .collect();
            let ids: Vec<u32> = (0..n as u32).collect();
            let canonical = |v: u32| {
                if kind == Kind::F32 && f32::from_bits(v).is_nan() {
                    0x7fc00000
                } else {
                    v
                }
            };
            let mut expected = ids.clone();
            expected.sort_by(|&a, &b| {
                let (a, b) = (keys[a as usize], keys[b as usize]);
                match kind {
                    Kind::U32 => a.cmp(&b),
                    Kind::I32 => (a as i32).cmp(&(b as i32)),
                    Kind::F32 => {
                        f32::from_bits(canonical(a)).total_cmp(&f32::from_bits(canonical(b)))
                    }
                }
            });
            let (sorted, payload) = run(&Plan::new(n as u32, kind).unwrap(), &keys, &ids);
            assert_eq!(payload, expected, "{kind:?}, n={n}");
            assert_eq!(
                sorted,
                expected
                    .iter()
                    .map(|&i| canonical(keys[i as usize]))
                    .collect::<Vec<_>>(),
                "{kind:?}, n={n}"
            );
        }
    }
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires physical CUDA, exclusive GPU lock and pinned NVRTC"]
fn sort_cuda_differential_and_faults() {
    use invar_cuda::{
        Error,
        cuda::{CollectiveFault, CudaDevice},
    };
    let compiler = std::env::var_os("INVAR_NVRTC_COMPILER")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_invar-nvrtc").into());
    let mut device = CudaDevice::open().unwrap();
    differential(|plan, keys, payload| {
        let result = device.run_sort(plan, keys, payload, &compiler).unwrap();
        assert!(result.kernel_ms.is_finite() && result.kernel_ms >= 0.0);
        (result.keys, result.payload)
    });
    let plan = Plan::new(3, Kind::U32).unwrap();
    assert!(matches!(
        device.run_sort(&plan, &[3, 2], &[0, 1, 2], &compiler),
        Err(Error::Limit)
    ));
    for fault in [CollectiveFault::OutOfMemory, CollectiveFault::LaunchFailure] {
        assert!(
            device
                .run_sort_with_fault(&plan, &[3, 2, 1], &[0, 1, 2], &compiler, fault)
                .is_err()
        );
        assert_eq!(device.quarantine_len(), 0);
    }
    assert!(matches!(
        device.run_sort_with_fault(
            &plan,
            &[3, 2, 1],
            &[0, 1, 2],
            &compiler,
            CollectiveFault::PartialCompletion
        ),
        Err(Error::Timeout)
    ));
    assert!(device.quarantine_len() > 0);
}
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires physical Metal on a 32-wide SIMD GPU"]
fn sort_metal_differential_and_faults() {
    use invar_metal::{Error, Fault, MetalDevice};
    let mut device = MetalDevice::open_runtime_source().unwrap();
    differential(|plan, keys, payload| {
        let result = device.run_sort(plan, keys, payload).unwrap();
        assert!(result.kernel_ms.is_finite() && result.kernel_ms >= 0.0);
        (result.keys, result.payload)
    });
    let plan = Plan::new(3, Kind::U32).unwrap();
    assert!(matches!(
        device.run_sort(&plan, &[3, 2], &[0, 1, 2]),
        Err(Error::Limit)
    ));
    for fault in [Fault::OutOfMemory, Fault::LaunchFailure] {
        assert!(
            device
                .run_sort_with_fault(&plan, &[3, 2, 1], &[0, 1, 2], fault)
                .is_err()
        );
    }
    assert!(matches!(
        device.run_sort_with_fault(&plan, &[3, 2, 1], &[0, 1, 2], Fault::PartialCompletion),
        Err(Error::Pending)
    ));
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn source_cases(
    mut run: impl FnMut(
        &[u8],
        &invar::mir::Program,
    ) -> (invar::eval::Observation, Vec<invar::schedule::Executed>),
) {
    for (ty, values, first, last) in [
        ("u32", "3u32,1u32,3u32,0u32", "0u32", "3u32"),
        ("i32", "3i32,-1i32,3i32,-2i32", "-2i32", "3i32"),
        ("f32", "3.0,-1.0,3.0,-2.0", "-2.0", "3.0"),
    ] {
        let source = format!(
            "fn main() -> i32 {{ var xs: Array<{ty}> = [{values}];
            var payload: Array<Bool> = [true,false,false,true]; sort_by_key(xs,payload);
            let ids = argsort(xs); let offsets: Array<u32> = [0u32,2u32,2u32,4u32];
            segmented_sort(xs,offsets); sort(xs);
            if xs[0u32] == {first} and xs[3u32] == {last}
                and payload[0u32] and not payload[1u32] and payload[2u32] and not payload[3u32]
                and ids[3u32] == 3u32 {{ return 7; }} return 0; }}"
        );
        let checked =
            invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let reference = invar::eval::observe(&checked, 100_000);
        let program = invar::mir::lower(&checked).unwrap();
        let (actual, executed) = run(source.as_bytes(), &program);
        assert_eq!(actual.result, reference.result);
        assert_eq!(actual.result, Ok(invar::eval::Value::I32(7)));
        assert_eq!(actual.steps_used, reference.steps_used);
        assert_eq!(actual.cleanup_trace, reference.cleanup_trace);
        assert_eq!(executed.len(), 4);
        assert!(executed.iter().all(|step| step.implementation.is_some()));
    }
    for body in [
        "var xs: Array<i32> = [1i32]; var vs: Array<u32> = []; sort_by_key(xs,vs);",
        "var xs: Array<i32> = [1i32]; let off: Array<u32> = [0u32,2u32]; segmented_sort(xs,off);",
        "var xs: Array<i32> = [1i32]; let off: Array<u32> = [1u32,0u32]; segmented_sort(xs,off);",
        "var xs: Array<i32> = [1i32]; let off: Array<u32> = []; segmented_sort(xs,off);",
        "var xs: Array<i32> = [1i32]; let off: Array<u32> = [0u32]; segmented_sort(xs,off);",
        "var xs: Array<i32> = [1i32]; let off: Array<u32> = [1u32,1u32]; segmented_sort(xs,off);",
        "var xs: Array<i32> = [1i32]; let off: Array<u32> = [0u32,0u32]; segmented_sort(xs,off);",
    ] {
        let source = format!("fn main() -> i32 {{ {body} return 0; }}");
        let checked =
            invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let reference = invar::eval::observe(&checked, 100_000);
        let program = invar::mir::lower(&checked).unwrap();
        let (actual, executed) = run(source.as_bytes(), &program);
        assert_eq!(actual.result, reference.result);
        assert_eq!(actual.steps_used, reference.steps_used);
        assert_eq!(actual.cleanup_trace, reference.cleanup_trace);
        assert!(executed.is_empty());
    }
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires physical CUDA and exclusive GPU lock"]
fn sort_cuda_source_surface_and_prevalidation() {
    use invar::schedule::cuda::{self, Fault};
    use invar_cuda::cuda::CudaDevice;
    let compiler = std::env::var_os("INVAR_NVRTC_COMPILER")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_invar-nvrtc").into());
    let mut device = CudaDevice::open().unwrap();
    source_cases(|source, program| {
        let run =
            cuda::run_source_with_device(source, program, 100_000, &mut device, &compiler).unwrap();
        assert!(
            run.executed
                .iter()
                .all(|step| step.implementation == Some("cuda-lsd-radix-8x4"))
        );
        (run.observation, run.executed)
    });
    let source=b"fn main() -> i32 { var xs: Array<u32> = [3u32,2u32,1u32]; let offsets: Array<u32> = [0u32,2u32,3u32,2u32]; segmented_sort(xs,offsets); return 0; }";
    let checked = invar::check::check(invar::front::parse(source).unwrap(), true).unwrap();
    let program = invar::mir::lower(&checked).unwrap();
    let run = cuda::run_source_with_device_fault(
        source,
        &program,
        100_000,
        &mut device,
        &compiler,
        Fault::OutOfMemory,
    )
    .unwrap();
    assert!(matches!(
        run.observation.result,
        Err(invar::eval::EvalError::Trap {
            category: invar::eval::TrapCategory::IndexOutOfBounds,
            ..
        })
    ));
    assert!(run.executed.is_empty());
}
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires physical Metal"]
fn sort_metal_source_surface() {
    let mut device = invar_metal::MetalDevice::open_runtime_source().unwrap();
    source_cases(|source, program| {
        let run =
            invar::schedule::metal::run_source_with_device(source, program, 100_000, &mut device)
                .unwrap();
        assert!(
            run.executed
                .iter()
                .all(|step| step.implementation == Some("metal-lsd-radix-8x4"))
        );
        (run.observation, run.executed)
    });
}
#[cfg(feature = "cuda")]
#[test]
fn sort_auto_unknown_cost_stays_on_cpu_without_opening_cuda() {
    let source = b"fn main() -> i32 { var xs: Array<u32> = [2u32,1u32]; sort(xs); return 7; }";
    let checked = invar::check::check(invar::front::parse(source).unwrap(), true).unwrap();
    let program = invar::mir::lower(&checked).unwrap();
    let run = invar::schedule::cuda::run_auto_source_with_fault(
        source,
        &program,
        100_000,
        std::path::Path::new("/absent-sort-compiler"),
        invar::plan::DeviceCostModel {
            cpu_bytes_per_ms: 1.0,
            device_bytes_per_ms: 1e30,
            device_fixed_ms: 0.0,
        },
        invar::schedule::cuda::Fault::OutOfMemory,
    )
    .unwrap();
    assert_eq!(run.observation.result, Ok(invar::eval::Value::I32(7)));
    assert_eq!(run.choices[0].reason, "sort-cost-unmeasured");
    assert!(run.executed[0].implementation.unwrap().starts_with("cpu-"));
    let path = std::env::temp_dir().join(format!("invar-sort-auto-{}.inv", std::process::id()));
    std::fs::write(&path, source).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .arg("explain")
        .arg(&path)
        .args(["--engine", "auto", "--format", "json"])
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(output.status.success(), "{output:?}");
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        json["device_execution"]["choices"][0]["implementation"],
        "cpu-lsd-radix-4x8"
    );
    assert_eq!(
        json["device_execution"]["choices"][0]["reason"],
        "sort-cost-unmeasured"
    );
    // #50: an auto explain is the unified v5.
    assert_eq!(json["schema"], "invar-explain-v5");
    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../schema/explain-v5.schema.json")).unwrap();
    let device_schema: serde_json::Value = serde_json::from_str(include_str!(
        "../schema/invar-explain-device-v5.schema.json"
    ))
    .unwrap();
    assert!(jsonschema::is_valid(&schema, &json), "{json}");
    assert!(jsonschema::is_valid(
        &device_schema,
        &json["device_execution"]
    ));
}
