use super::*;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn metal_held_out_smoke(corpus: &Value) -> Result<Value, String> {
    use invar::{check::check, eval, front::parse, mir, schedule::metal};
    // Benchmarks time generated kernels from emitted MSL: developer source path.
    let mut device =
        invar_metal::MetalDevice::open_runtime_source().map_err(|e| format!("{e:?}"))?;
    let mut cases = Vec::new();
    for item in corpus["programs"]
        .as_array()
        .ok_or("missing corpus programs")?
        .iter()
        .filter(|item| item["split"] == "held_out")
    {
        let source = item["source"].as_str().ok_or("missing corpus source")?;
        let checked = check(
            parse(source.as_bytes()).map_err(|e| format!("{e:?}"))?,
            true,
        )
        .map_err(|e| format!("{e:?}"))?;
        let program = mir::lower(&checked).map_err(|e| format!("{e:?}"))?;
        let reference = eval::observe(&checked, 100_000);
        let (matched, executed_regions, error_code) = match metal::run_source_with_device(
            source.as_bytes(),
            &program,
            100_000,
            &mut device,
        ) {
            Ok(run) => (
                !run.executed.is_empty()
                    && run.observation.result == reference.result
                    && run.observation.steps_used == reference.steps_used
                    && run.observation.cleanup_trace == reference.cleanup_trace,
                run.executed.len(),
                Value::Null,
            ),
            Err(error) => (false, 0, json!(error.code)),
        };
        cases.push(json!({"id":item["id"],"family":item["family"],
            "source_sha256":item["source_sha256"],"matched":matched,
            "executed_regions":executed_regions,"error_code":error_code}));
    }
    let matched_count = cases.iter().filter(|case| case["matched"] == true).count();
    Ok(
        json!({"schema_version":1,"kind":"held_out_generated_correctness_smoke",
        "status":if cases.len()==60 && matched_count==60 {"passed"} else {"failed"},
        "engine":"forced_metal","device":device.name(),"case_count":cases.len(),
        "matched_count":matched_count,"timing":"not_recorded","performance_gate":"not_run",
        "cases":cases}),
    )
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub(crate) fn metal_held_out_smoke(_corpus: &Value) -> Result<Value, String> {
    Err("Metal held-out smoke requires the metal feature and macOS".into())
}

#[cfg(any(test, all(feature = "metal", target_os = "macos"), feature = "cuda"))]
pub(crate) enum GeneratedReference {
    Map(Vec<Vec<u32>>),
    Collective(Vec<u32>),
}

#[cfg(any(test, all(feature = "metal", target_os = "macos"), feature = "cuda"))]
pub(crate) struct GeneratedRaw {
    pub(crate) source_hash: u64,
    pub(crate) compile_ms: f64,
    pub(crate) device_resident_ms: Vec<f64>,
    pub(crate) host_to_host_ms: Vec<f64>,
    pub(crate) transfer_bytes: usize,
    pub(crate) workspace_bytes: usize,
    pub(crate) launches: usize,
}

#[cfg(any(test, all(feature = "metal", target_os = "macos"), feature = "cuda"))]
pub(crate) fn u32_digest(buffers: &[&[u32]]) -> String {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    let mut bytes = [0u8; 16 * 1024];
    for buffer in buffers {
        for words in buffer.chunks(bytes.len() / 4) {
            for (word, out) in words.iter().zip(bytes.chunks_exact_mut(4)) {
                out.copy_from_slice(&word.to_le_bytes());
            }
            digest.update(&bytes[..words.len() * 4]);
        }
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(any(test, all(feature = "metal", target_os = "macos"), feature = "cuda"))]
pub(crate) fn measured_words(length: usize) -> Result<Vec<u32>, invar_device::Error> {
    let mut words = Vec::new();
    words.try_reserve_exact(length).map_err(|_| {
        invar_device::Error::ResourceLimit("benchmark host staging allocation failed")
    })?;
    Ok(words)
}

#[cfg(any(test, all(feature = "metal", target_os = "macos"), feature = "cuda"))]
pub(crate) fn measured_inputs(
    length: usize,
    input_count: usize,
) -> Result<Vec<Vec<u32>>, invar_device::Error> {
    (0..input_count)
        .map(|slot| {
            let mut words = measured_words(length)?;
            words.extend((0..length).map(|i| {
                (i as u32)
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223)
                    .wrapping_add((slot as u32).wrapping_mul(17))
            }));
            Ok(words)
        })
        .collect()
}

#[cfg(any(test, all(feature = "metal", target_os = "macos"), feature = "cuda"))]
pub(crate) fn measured_invocation(
    program_id: &str,
    source_sha256: &str,
    length: u32,
    initial_state: Option<u32>,
    expected_input_buffers: usize,
    inputs: &[&[u32]],
) -> Result<(Value, String), String> {
    if length == 0
        || inputs.is_empty()
        || inputs.len() != expected_input_buffers
        || inputs.iter().any(|input| input.len() != length as usize)
    {
        return Err("measured invocation input shape mismatch".into());
    }
    let invocation = json!({
        "scope":"parameterized_extracted_plan",
        "program_id":program_id,
        "source_template_sha256":source_sha256,
        "elements":length,
        "input_buffers":inputs.len(),
        "input_recipe":MEASURED_INPUT_RECIPE,
        "input_sha256":u32_digest(inputs),
        "initial_state_u32":initial_state,
    });
    let digest = hash::sha256(&serde_json::to_vec(&invocation).map_err(|e| e.to_string())?);
    Ok((invocation, digest))
}

// Source extraction retains intermediate stage arrays. The benchmark contract
// publishes only the trailing result arrays; keep every node and reverify.
#[cfg(any(test, all(feature = "metal", target_os = "macos"), feature = "cuda"))]
pub(crate) fn generated_map_plan(
    plan: invar_device::Verified,
    output_count: usize,
    source: &str,
) -> Result<invar_device::Verified, invar_device::Error> {
    let mut ir = plan.ir().clone();
    let dag = ir.dag.as_mut().ok_or(invar_device::Error::Operations)?;
    if output_count == 0 || output_count > dag.outputs.len() {
        return Err(invar_device::Error::Operations);
    }
    let first = dag.outputs.len() - output_count;
    // The trailing outputs must be the source's published `result{k}` loops,
    // not intermediate stages of a region the fusion cap cut short.
    for k in 0..output_count {
        let store = format!("result{k}[");
        let at = source
            .match_indices(&store)
            .map(|(at, _)| at)
            .find(|&at| {
                source[at..]
                    .split(']')
                    .nth(1)
                    .is_some_and(|t| t.trim_start().starts_with('='))
            })
            .ok_or(invar_device::Error::Operations)?;
        // The `for` loop that stores `result{k}`: the node must lie inside it.
        let start = source[..at]
            .rfind("for ")
            .ok_or(invar_device::Error::Operations)?;
        let end = source[at..].find('}').map_or(source.len(), |e| at + e + 1);
        let node = &dag.nodes[dag.stage_outputs[first + k]];
        if !(start <= node.span.start && node.span.end <= end) {
            return Err(invar_device::Error::Operations);
        }
    }
    dag.outputs = dag.outputs[first..first + output_count].to_vec();
    dag.stage_outputs = dag.stage_outputs[first..first + output_count].to_vec();
    ir.output = *dag.outputs.last().ok_or(invar_device::Error::Operations)?;
    let origin = invar_device::Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    invar_device::verify(ir, origin, plan.capabilities())
}

#[cfg(any(test, all(feature = "metal", target_os = "macos"), feature = "cuda"))]
pub(crate) fn generated_case_with(
    case: &Value,
    corpus: &Value,
    caps: invar_device::Capabilities,
    device_name: &str,
    toolchain: &str,
    sample_plan: (usize, usize),
    mut measure: impl FnMut(
        &invar_device::Verified,
        &[&[u32]],
        &GeneratedReference,
        u32,
    ) -> Result<GeneratedRaw, String>,
) -> Result<Value, String> {
    use invar::{admission, check::check, device_ir, eval, front::parse, mir};
    use invar_device::{CollectiveKind, DagOp, Provenance};
    let (warmups, samples) = sample_plan;
    let family = case["family"].as_str().ok_or("case family missing")?;
    let length = u32::try_from(case["elements"].as_u64().ok_or("case length missing")?)
        .map_err(|_| "case length exceeds u32")?;
    if length == 0 {
        return Err("zero-length case has no timed GPU kernel".into());
    }
    let programs = corpus["programs"]
        .as_array()
        .ok_or("corpus programs missing")?;
    let mut selected = None;
    for item in programs
        .iter()
        .filter(|item| item["family"] == family && item["split"] == "tuning")
    {
        if family == "map"
            && (item["stages"] != case["stages"]
                || item["inputs"] != case["input_buffers"]
                || item["outputs"] != case["output_buffers"])
        {
            continue;
        }
        let source = item["source"].as_str().ok_or("corpus source missing")?;
        if family == "reduction" && !source.contains("wrapping_add(acc,xs[i])") {
            continue;
        }
        if family == "scan"
            && !source
                .contains("state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); out[i] = state;")
        {
            continue;
        }
        let checked = check(
            parse(source.as_bytes()).map_err(|e| format!("parse: {e:?}"))?,
            true,
        )
        .map_err(|e| format!("check: {e:?}"))?;
        if eval::observe(&checked, 100_000).result != Ok(eval::Value::I32(0)) {
            return Err("ordinary source oracle failed".into());
        }
        let program = mir::lower(&checked).map_err(|e| format!("MIR: {e:?}"))?;
        let admission = admission::analyze(source.as_bytes(), &program)
            .map_err(|e| format!("admission: {e:?}"))?;
        for region in admission.regions() {
            let Ok(mut ir) = device_ir::from_region(region, length) else {
                continue;
            };
            if matches!(caps, invar_device::Capabilities::Metal { .. })
                && device_ir::plan_metal(&mut ir).is_err()
            {
                continue;
            }
            let shape = match family {
                "map" => ir.dag.as_ref().is_some_and(|dag| {
                    dag.inputs.len() as u64 == case["input_buffers"].as_u64().unwrap_or(0)
                        && dag.outputs.len() as u64 >= case["output_buffers"].as_u64().unwrap_or(0)
                }),
                "reduction" => matches!(
                    ir.collective.as_ref().map(|c| c.kind),
                    Some(CollectiveKind::Reduction {
                        law: invar_device::ReduceLaw::WrappingSum,
                        ..
                    })
                ),
                "scan" => matches!(
                    ir.collective.as_ref().map(|c| c.kind),
                    Some(CollectiveKind::AffineScan {
                        a: invar_device::Coefficient::Input(0),
                        b: invar_device::Coefficient::Input(1),
                        prefix: invar_device::PrefixMode::Inclusive,
                        ..
                    })
                ),
                _ => false,
            };
            if !shape {
                continue;
            }
            let origin = Provenance {
                source_identity: ir.source_identity,
                graph_revision: ir.graph_revision,
            };
            let verified = invar_device::verify(ir, origin, caps)
                .map_err(|e| format!("device verify: {e:?}"))?;
            let verified = if family == "map" {
                let outputs = usize::try_from(
                    case["output_buffers"]
                        .as_u64()
                        .ok_or("output count missing")?,
                )
                .map_err(|_| "output count overflow")?;
                // A region that does not publish the result loops is not the
                // benchmark computation; try the next region instead.
                match generated_map_plan(verified, outputs, source) {
                    Ok(plan) => plan,
                    Err(_) => continue,
                }
            } else {
                verified
            };
            selected = Some((item, verified));
            break;
        }
        if selected.is_some() {
            break;
        }
    }
    let (item, plan) = selected.ok_or("no checked source region matches this manifest case")?;
    let n = length as usize;
    let input_count = plan.ir().dag.as_ref().map_or_else(
        || plan.ir().collective.as_ref().map_or(0, |c| c.inputs.len()),
        |dag| dag.inputs.len(),
    );
    let inputs = measured_inputs(n, input_count).map_err(|e| format!("{e:?}"))?;
    let refs = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let (invocation, invocation_sha256) = measured_invocation(
        item["id"].as_str().ok_or("corpus program id missing")?,
        item["source_sha256"]
            .as_str()
            .ok_or("corpus source hash missing")?,
        length,
        (family != "map").then_some(MEASURED_SEED),
        usize::try_from(
            case["input_buffers"]
                .as_u64()
                .ok_or("case input buffer count missing")?,
        )
        .map_err(|_| "case input buffer count overflow")?,
        &refs,
    )?;
    let input_sha256 = invocation["input_sha256"].clone();
    let (reference, reference_kind) = if family == "map" {
        let dag = plan.ir().dag.as_ref().ok_or("missing DAG")?;
        let mut expected = (0..dag.outputs.len())
            .map(|_| measured_words(n))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("{e:?}"))?;
        let mut values = measured_words(dag.nodes.len()).map_err(|e| format!("{e:?}"))?;
        for (index, _) in inputs[0].iter().enumerate() {
            values.clear();
            for node in &dag.nodes {
                let value = match node.op {
                    DagOp::Input(slot) => inputs[slot as usize][index],
                    DagOp::WrappingAdd { input, constant } => values[input].wrapping_add(constant),
                    DagOp::WrappingMul { input, constant } => values[input].wrapping_mul(constant),
                    DagOp::ZipAdd { left, right } => values[left].wrapping_add(values[right]),
                    DagOp::ZipMul { left, right } => values[left].wrapping_mul(values[right]),
                };
                values.push(value);
            }
            for (output, node) in expected.iter_mut().zip(&dag.stage_outputs) {
                output.push(values[*node]);
            }
        }
        (
            GeneratedReference::Map(expected),
            "checked_source_graph_cpu_u32",
        )
    } else {
        let mut state = MEASURED_SEED;
        let mut expected = measured_words(if family == "scan" { n + 1 } else { 1 })
            .map_err(|e| format!("{e:?}"))?;
        if family == "scan" {
            for (&a, &b) in inputs[0].iter().zip(&inputs[1]) {
                state = a.wrapping_mul(state).wrapping_add(b);
                expected.push(state);
            }
            expected.push(state);
        } else {
            for value in &inputs[0] {
                state = state.wrapping_add(*value);
            }
            expected.push(state);
        }
        (
            GeneratedReference::Collective(expected),
            "independent_ordered_u32_fold",
        )
    };
    let result = measure(&plan, &refs, &reference, MEASURED_SEED)?;
    let measured_output_sha256 = match &reference {
        GeneratedReference::Collective(values) if family == "scan" => {
            u32_digest(&[&values[..values.len() - 1]])
        }
        GeneratedReference::Collective(values) => u32_digest(&[values.as_slice()]),
        GeneratedReference::Map(values) => {
            u32_digest(&values.iter().map(Vec::as_slice).collect::<Vec<_>>())
        }
    };
    let ir = plan.ir();
    let required_bytes = case["required_bytes"]
        .as_u64()
        .ok_or("required bytes missing")?;
    let timing_valid = result.device_resident_ms.len() == samples
        && result.host_to_host_ms.len() == samples
        && result
            .device_resident_ms
            .iter()
            .chain(&result.host_to_host_ms)
            .all(|ms| ms.is_finite() && *ms > 0.0);
    if !timing_valid {
        return Err("invalid raw GPU samples".into());
    }
    let runner = env::current_exe().map_err(|e| e.to_string())?;
    let runner_sha256 = hash::sha256(&read_bounded(&runner, general_net::MAX_BINARY)?);
    let (resident_boundary, host_boundary) = match caps {
        invar_device::Capabilities::Metal { .. } => (
            "metal_command_buffer_gpu_start_end_v1",
            "metal_runner_after_compile_through_host_output_copy_v1",
        ),
        invar_device::Capabilities::Cuda { .. } => (
            "cuda_events_around_verified_kernel_launches_v1",
            "cuda_runner_after_compile_through_host_output_copy_v1",
        ),
    };
    Ok(
        json!({"case_id":case["id"],"program_id":item["id"],"source_sha256":item["source_sha256"],
        "source_sha256_role":"three_element_corpus_template_only",
        "source_template_length":3,"measured_length":length,"reference":reference_kind,
        "measurement_scope":"parameterized_extracted_plan_not_source_program",
        "measured_invocation":invocation,"measured_invocation_sha256":invocation_sha256,
        "generated_code_hash":format!("{:016x}",result.source_hash),
        "runner_executable_sha256":runner_sha256,
        "input_recipe":MEASURED_INPUT_RECIPE,
        "input_sha256":input_sha256,
        "measured_output_sha256":if measured_output_sha256.is_empty() {Value::Null} else {json!(measured_output_sha256)},
        "device_resident_boundary":resident_boundary,"host_to_host_boundary":host_boundary,
        "device":device_name,"toolchain":toolchain,"geometry":{"blocks":ir.geometry.blocks,"lanes":ir.geometry.lanes},
        "launches":result.launches,"required_bytes":required_bytes,"transfer_bytes":result.transfer_bytes,
        "workspace_bytes":result.workspace_bytes,"full_output_correct":true,"warmups":warmups,
        "samples":samples,
        "device_resident_ms":result.device_resident_ms,"host_to_host_ms":result.host_to_host_ms,
        "compile_ms":result.compile_ms,"cold_process":"pending",
        "ratio_gate":"parameterized_plan_source_program_and_cold_contract_pending"}),
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn metal_generated_case(case: &Value, corpus: &Value) -> Result<Value, String> {
    // Benchmarks time generated kernels from emitted MSL: developer source path.
    let mut device =
        invar_metal::MetalDevice::open_runtime_source().map_err(|e| format!("{e:?}"))?;
    let name = device.name();
    generated_case_with(
        case,
        corpus,
        device.capabilities(),
        &name,
        "Metal MSL 3.1",
        (10, 30),
        |plan, inputs, reference, seed| {
            let raw = match reference {
                GeneratedReference::Map(outputs) => {
                    device.benchmark_generated_map(plan, inputs, outputs)
                }
                GeneratedReference::Collective(output) => {
                    device.benchmark_generated_collective(plan, inputs, seed, output)
                }
            }
            .map_err(|e| format!("Metal generated: {e:?}"))?;
            Ok(GeneratedRaw {
                source_hash: raw.source_hash,
                compile_ms: raw.compile_ms,
                device_resident_ms: raw.device_resident_ms,
                host_to_host_ms: raw.host_to_host_ms,
                transfer_bytes: raw.transfer_bytes,
                workspace_bytes: raw.workspace_bytes,
                launches: raw.launches,
            })
        },
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn metal_generated_case_once(case: &Value, corpus: &Value) -> Result<Value, String> {
    // Benchmarks time generated kernels from emitted MSL: developer source path.
    let mut device =
        invar_metal::MetalDevice::open_runtime_source().map_err(|e| format!("{e:?}"))?;
    let name = device.name();
    generated_case_with(
        case,
        corpus,
        device.capabilities(),
        &name,
        "Metal MSL 3.1",
        (0, 1),
        |plan, inputs, reference, seed| match reference {
            GeneratedReference::Map(expected) => {
                let run = device
                    .run_fused_map(plan, inputs)
                    .map_err(|e| format!("Metal map: {e:?}"))?;
                if run.outputs != *expected {
                    return Err("cold child map output mismatch".into());
                }
                Ok(GeneratedRaw {
                    source_hash: run.source_hash,
                    compile_ms: run.compile_ms,
                    device_resident_ms: vec![run.kernel_ms],
                    host_to_host_ms: vec![run.host_to_host_ms],
                    transfer_bytes: run.transfer_bytes,
                    workspace_bytes: 0,
                    launches: 1,
                })
            }
            GeneratedReference::Collective(expected) => {
                let run = device
                    .run_collective(plan, inputs, seed)
                    .map_err(|e| format!("Metal collective: {e:?}"))?;
                if run.prefixes != expected[..expected.len() - 1]
                    || run.final_state != expected[expected.len() - 1]
                {
                    return Err("cold child collective output mismatch".into());
                }
                let blocks = plan.ir().geometry.blocks;
                let scan = expected.len() > 1;
                let summary = blocks as usize * if scan { 8 } else { 4 };
                let chunks = blocks
                    .div_ceil(invar_device::METAL_SCAN_CHUNK_BLOCKS)
                    .max(1) as usize;
                let workspace = 2 * summary + 12 + chunks * 4 + (chunks + 1) * 8;
                Ok(GeneratedRaw {
                    source_hash: run.source_hash,
                    compile_ms: run.compile_ms,
                    device_resident_ms: vec![run.kernel_ms],
                    host_to_host_ms: vec![run.host_to_host_ms],
                    transfer_bytes: run.transfer_bytes,
                    workspace_bytes: workspace,
                    launches: 3,
                })
            }
        },
    )
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub(crate) fn metal_generated_case(_case: &Value, _corpus: &Value) -> Result<Value, String> {
    Err("generated Metal timing requires the metal feature and macOS".into())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub(crate) fn metal_generated_case_once(_case: &Value, _corpus: &Value) -> Result<Value, String> {
    Err("generated Metal timing requires the metal feature and macOS".into())
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_generated_sample(
    device: &mut invar_cuda::cuda::CudaDevice,
    plan: &invar_device::Verified,
    inputs: &[&[u32]],
    reference: &GeneratedReference,
    seed: u32,
    compiler: &Path,
) -> Result<GeneratedRaw, String> {
    match reference {
        GeneratedReference::Map(expected) => {
            let run = device
                .run_fused_map(plan, inputs, compiler)
                .map_err(|e| format!("CUDA map: {e:?}"))?;
            if run.outputs != *expected {
                return Err("CUDA generated map output mismatch".into());
            }
            Ok(GeneratedRaw {
                source_hash: run.source_hash,
                compile_ms: run.compile_ms,
                device_resident_ms: vec![f64::from(run.kernel_ms)],
                host_to_host_ms: vec![run.host_to_host_ms],
                transfer_bytes: run.transfer_bytes,
                workspace_bytes: 0,
                launches: 1,
            })
        }
        GeneratedReference::Collective(expected) => {
            let run = device
                .run_collective(plan, inputs, seed, compiler)
                .map_err(|e| format!("CUDA collective: {e:?}"))?;
            if run.prefixes != expected[..expected.len() - 1]
                || run.final_state != expected[expected.len() - 1]
            {
                return Err("CUDA generated collective output mismatch".into());
            }
            Ok(GeneratedRaw {
                source_hash: run.source_hash,
                compile_ms: run.compile_ms,
                device_resident_ms: vec![f64::from(
                    run.block_ms + run.propagate_ms + run.writeback_ms,
                )],
                host_to_host_ms: vec![run.host_to_host_ms],
                transfer_bytes: run.transfer_bytes,
                workspace_bytes: plan.ir().geometry.blocks as usize
                    * if expected.len() > 1 { 16 } else { 8 },
                launches: 3,
            })
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_generated_case(case: &Value, corpus: &Value) -> Result<Value, String> {
    let mut device = invar_cuda::cuda::CudaDevice::open().map_err(|e| format!("{e:?}"))?;
    let name = device.name().to_owned();
    let invar_device::Capabilities::Cuda { major, minor, sm } = device.capabilities() else {
        return Err("CUDA device reported non-CUDA capabilities".into());
    };
    let toolchain = format!("NVRTC {major}.{minor}; sm_{sm}");
    let compiler = env::current_exe()
        .map_err(|e| e.to_string())?
        .with_file_name("invar-nvrtc");
    if !compiler.is_file() {
        return Err(format!(
            "bounded NVRTC helper unavailable: {}",
            compiler.display()
        ));
    }
    let mut report = generated_case_with(
        case,
        corpus,
        device.capabilities(),
        &name,
        &toolchain,
        (10, 30),
        |plan, inputs, reference, seed| {
            let mut raw = GeneratedRaw {
                source_hash: 0,
                compile_ms: 0.0,
                device_resident_ms: Vec::with_capacity(30),
                host_to_host_ms: Vec::with_capacity(30),
                transfer_bytes: 0,
                workspace_bytes: 0,
                launches: 0,
            };
            for invocation in 0..40 {
                let run =
                    cuda_generated_sample(&mut device, plan, inputs, reference, seed, &compiler)?;
                raw.source_hash = run.source_hash;
                raw.compile_ms += run.compile_ms;
                raw.transfer_bytes = run.transfer_bytes;
                raw.workspace_bytes = run.workspace_bytes;
                raw.launches = run.launches;
                if invocation >= 10 {
                    raw.device_resident_ms.extend(run.device_resident_ms);
                    raw.host_to_host_ms.extend(run.host_to_host_ms);
                }
            }
            Ok(raw)
        },
    )?;
    report["compiler_executable_sha256"] = json!(hash::sha256(&read_bounded(
        &compiler,
        general_net::MAX_BINARY
    )?));
    report["buffer_reuse"] =
        json!("fresh owned allocations per invocation; compiled module cache reused");
    Ok(report)
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_generated_case_once(case: &Value, corpus: &Value) -> Result<Value, String> {
    let mut device = invar_cuda::cuda::CudaDevice::open().map_err(|e| format!("{e:?}"))?;
    let name = device.name().to_owned();
    let invar_device::Capabilities::Cuda { major, minor, sm } = device.capabilities() else {
        return Err("CUDA device reported non-CUDA capabilities".into());
    };
    let toolchain = format!("NVRTC {major}.{minor}; sm_{sm}");
    let compiler = env::current_exe()
        .map_err(|e| e.to_string())?
        .with_file_name("invar-nvrtc");
    if !compiler.is_file() {
        return Err("bounded NVRTC helper unavailable".into());
    }
    let mut result = generated_case_with(
        case,
        corpus,
        device.capabilities(),
        &name,
        &toolchain,
        (0, 1),
        |plan, inputs, reference, seed| {
            cuda_generated_sample(&mut device, plan, inputs, reference, seed, &compiler)
        },
    )?;
    result["compiler_executable_sha256"] = json!(hash::sha256(&read_bounded(
        &compiler,
        general_net::MAX_BINARY
    )?));
    Ok(result)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn cuda_generated_case(_case: &Value, _corpus: &Value) -> Result<Value, String> {
    Err("generated CUDA timing requires the cuda feature".into())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn cuda_generated_case_once(_case: &Value, _corpus: &Value) -> Result<Value, String> {
    Err("generated CUDA timing requires the cuda feature".into())
}
