//! Compile modes: net layouts, task field descriptions and the shared compile path.

use super::*;

pub(super) enum Mode {
    #[cfg(feature = "gnet-fault-injection")]
    FaultedDevice(
        compiled::OwnedLimits,
        compiled::SpeculationLimits,
        compiled::Fault,
    ),
    Scalar,
    Owned(compiled::OwnedLimits),
    ResumableOwnedNumeric(compiled::OwnedLimits),
    Resumable(compiled::OwnedLimits),
    Speculative(compiled::OwnedLimits, compiled::SpeculationLimits, bool),
}

pub(super) fn layouts(artifact: &compiled::Artifact) -> Result<Value, Error> {
    let owned = artifact.owned_layout().map(|layout| {
        json!({"elements": layout.elements(), "allocations": layout.allocations(),
            "cleanup_events": layout.cleanup_events(), "heap_words": layout.heap_words(),
            "cleanup_words": layout.cleanup_words(), "storage_bytes": layout.storage_bytes(),
            "output_words": layout.output_words(), "kernel_name": layout.kernel_name()})
    });
    let continuation = artifact.continuation_layout().map(|layout| {
        json!({"task_bytes": layout.task_bytes(), "kernel_name": layout.kernel_name(),
            "config_words": layout.config_words(), "max_roots": layout.max_roots(),
            "max_slices": layout.max_slices()})
    });
    let speculation = artifact.speculation_layout().map(|layout| {
        let l = layout.limits();
        json!({"slots": layout.slots(), "slot_bytes": layout.slot_bytes(),
            "heap_words": layout.heap_words(), "trace_words": layout.trace_words(),
            "cleanup_words": layout.cleanup_words(), "scratch_bytes": layout.scratch_bytes(),
            "entry_point": layout.entry_point(),
            "limits": {"slots": l.slots, "fuel": l.fuel, "transitions": l.transitions,
                "trace_events": l.trace_events, "quantum": l.quantum,
                "lookahead": l.lookahead, "scratch_bytes": l.scratch_bytes}})
    });
    let device = artifact
        .device_layout()
        .map(|layout| {
            let minimum = compiled::DeviceConfig {
                fuel: 1,
                transitions: 0,
                trace_events: 0,
                quantum: 1,
                rounds: 1,
                capacity: 0,
            };
            layout.buffer_bytes(minimum).map(|bytes| {
                json!({
            "kernel_name": layout.kernel_name(), "group_threads": layout.group_threads(),
            "groups": layout.groups(), "max_rounds": layout.max_rounds(),
            "state_words": layout.state_words(), "minimum_config_words": minimum.words(),
            "minimum_buffer_bytes": bytes, "buffers": bytes.len()})
            })
        })
        .transpose()
        .map_err(Error::Compiler)?;
    let site = |s: compiled::CallSite| json!({"function":s.function,"block":s.block,"position":s.position});
    let fork = artifact.fork_plan().map(|plan| {
        json!({
            "candidates": plan.candidates.iter().map(|c| json!({"source":site(c.source),
                "target":site(c.target), "callee":c.callee, "projection":c.projection,
                "scalar_values":c.scalar_values,"scalar_places":c.scalar_places,
                "replay_owners":c.replay_owners})).collect::<Vec<_>>(),
            "serialized": plan.serialized.iter().map(|s| json!({"source":site(s.source),
                "target":site(s.target),"reason":format!("{:?}",s.reason)})).collect::<Vec<_>>()
        })
    });
    Ok(json!({"owned": owned, "continuation": continuation,
        "speculation":speculation, "device":device, "fork_plan":fork}))
}

/// The 18 continuation task counters after the frames, shared by the ABI11
/// and ABI12 receipt blocks: name, offset and representation.
pub(super) fn task_fields(frame_bytes: usize) -> Vec<Value> {
    [
        "started",
        "done",
        "depth",
        "max_depth",
        "steps",
        "transitions",
        "used",
        "next_frame",
        "allocated",
        "live",
        "cleanup_used",
        "fuel",
        "transition_cap",
        "trace_cap",
        "status",
        "result",
        "start",
        "end",
    ]
    .iter()
    .enumerate()
    .map(|(index, name)| {
        json!({"name": name, "offset": frame_bytes + index * 8,
            "representation": if index < 14 { "u64" } else { "i64" }})
    })
    .collect()
}

pub(super) fn compile_mode(
    source: &[u8],
    entry: &str,
    target: compiled::Target,
    limits: Limits,
    mode: Mode,
) -> Result<ProvenancedArtifact, Error> {
    if source.len() > MAX_SOURCE_BYTES || entry.is_empty() || entry.len() > MAX_ENTRY_BYTES {
        return Err(Error::Limit);
    }
    let parsed = front::parse(source).map_err(Error::Source)?;
    let checked = check::check(parsed, false).map_err(Error::Source)?;
    let net = Net::lower(&checked, limits.net).map_err(Error::Net)?;
    let proof = net.verify().map_err(Error::Net)?;
    let snapshot = net_bytes(&net, &proof)?;
    // ABI12 stage 2b: size/entry limits, source parse/check and net errors keep
    // their pre-ABI12 priority. Metal is not an owned-numeric continuation
    // target. On CUDA only a successful ABI12 emission is admitted; every CUDA
    // emission refusal keeps the pre-ABI12 outcome `NumericTarget`, so only
    // valid admitted programs change outcome.
    #[cfg(feature = "gnet-fault-injection")]
    let fault = match &mode {
        Mode::FaultedDevice(_, _, fault) => Some(*fault),
        _ => None,
    };
    let device_continuation =
        matches!(mode, Mode::ResumableOwnedNumeric(_)) && target != compiled::Target::Cpu;
    if device_continuation && target != compiled::Target::Cuda {
        return Err(Error::Compiler(compiled::Error::NumericTarget));
    }
    let artifact = match mode {
        Mode::Scalar => compiled::emit(&net, &proof, entry, target, limits.compiler),
        Mode::Owned(storage) => {
            compiled::emit_owned(&net, &proof, entry, target, limits.compiler, storage)
        }
        Mode::ResumableOwnedNumeric(storage) => compiled::emit_resumable_owned_numeric(
            &net,
            &proof,
            entry,
            target,
            limits.compiler,
            storage,
        ),
        Mode::Resumable(storage) => {
            compiled::emit_resumable(&net, &proof, entry, target, limits.compiler, storage)
        }
        Mode::Speculative(storage, speculation, false) => compiled::emit_speculative(
            &net,
            &proof,
            entry,
            target,
            limits.compiler,
            storage,
            speculation,
        ),
        Mode::Speculative(storage, speculation, true) => compiled::emit_device_speculative(
            &net,
            &proof,
            entry,
            target,
            limits.compiler,
            storage,
            speculation,
        ),
        #[cfg(feature = "gnet-fault-injection")]
        Mode::FaultedDevice(storage, speculation, fault) => {
            compiled::emit_device_speculative_with_fault(
                &net,
                &proof,
                entry,
                target,
                limits.compiler,
                storage,
                speculation,
                fault,
            )
        }
    }
    .map_err(|error| {
        Error::Compiler(if device_continuation {
            compiled::Error::NumericTarget
        } else {
            error
        })
    })?;
    let output_words = artifact
        .owned_layout()
        .map_or(compiled::OUTPUT_WORDS, |l| l.output_words());
    let mut manifest = json!({
        "version": VERSION,
        "scope": "checked_source_net_emitted_source",
        "compiler_build": env!("INVAR_BUILD_DIGEST"),
        "net_contract": super::super::CONTRACT_VERSION,
        "source": {"sha256": digest(source), "bytes": source.len()},
        "net": {"sha256": digest(&snapshot), "bytes": snapshot.len()},
        "entry": entry,
        "target": format!("{:?}", artifact.target()),
        "abi_version": artifact.abi_version(),
        "layouts": layouts(&artifact)?,
        "params": artifact.params().iter().map(|t| format!("{t:?}")).collect::<Vec<_>>(),
        "result": format!("{:?}", artifact.result()),
        "artifact": {"sha256": digest(artifact.source().as_bytes()), "bytes": artifact.source().len()},
        "storage": {"states": artifact.states(), "frame_bytes": artifact.frame_bytes(),
            "max_frames": compiled::MAX_FRAMES, "output_words": output_words,
            "event_words": compiled::EVENT_WORDS},
        "limits": {
            "net": {"functions": limits.net.functions, "blocks": limits.net.blocks,
                "instructions": limits.net.instructions, "wires": limits.net.wires,
                "literal_bytes": limits.net.literal_bytes},
            "compiler": {"values": limits.compiler.values, "places": limits.compiler.places,
                "states": limits.compiler.states, "frame_bytes": limits.compiler.frame_bytes,
                "source_bytes": limits.compiler.source_bytes}
        }
    });
    // Additive numeric fields: existing ABI1-7 receipt shapes/version remain.
    if let Some(policy) = artifact.numeric_policy() {
        let mir = net.compiled_mir(&proof).map_err(Error::Net)?;
        manifest["numeric"] = json!({
            "policy": policy.to_json(), "semantic_id": policy.id(),
            "layout_version": mir.layout_version,
            "f32": {"encoding":"ieee754-binary32", "bits":32,
                "carrier":"nonnegative-i64-u32-bits", "maximum":u32::MAX,
                "identity_equality":"exact-bits", "engine_nan_equality":"unobservable-class"},
            "word_bytes":8, "output_words":output_words,
            "event_words":compiled::EVENT_WORDS
        });
        if matches!(artifact.abi_version(), 11 | 12) {
            manifest["numeric"]["array_f32"] = json!({
                "kind":258, "element_word_bytes":8, "element_bits":32,
                "encoding":"nonnegative-i64-u32-bits", "external_owner_inputs":false
            });
        }
    }
    if let Some(entry) = artifact.entry_layout() {
        manifest["entry_packet"] = json!({"version":1,"input_words":entry.input_words(),"modes":entry.modes().iter().map(|m|format!("{m:?}")).collect::<Vec<_>>(),"element_stride_bytes":8,"host_element_bytes":4,"numeric":"integer-software-binary32","native_float_admission":false,"subnormal_path":"exact-integer-software","publication":"successful-source-return-after-confirmed-completion"});
    }
    if artifact.abi_version() == 11 {
        let frame_bytes = artifact.frame_bytes();
        manifest["numeric_owned_continuation"] = json!({
            "profile": "cpu-owned-numeric-continuation-v1",
            "contract_version": 1,
            "word_bytes": 8,
            "frame_count": compiled::MAX_FRAMES,
            "frame_stride_bytes": frame_bytes / compiled::MAX_FRAMES,
            "task_fields": task_fields(frame_bytes),
            "output_words": 10,
            "quantum_unit": "mir-transitions",
            "snapshot_scope": "evidence-only-no-task-restore",
            "terminal_packet": "abi10-compatible"
        });
    }
    // Exclusive ABI12 CUDA compile-receipt block; it grants no authority.
    if artifact.abi_version() == 12 {
        manifest[owned_numeric_cuda::KEY] = owned_numeric_cuda::block(&artifact)?;
    }
    #[cfg(feature = "gnet-fault-injection")]
    if let Some(fault) = fault {
        manifest["test_only_fault_injection"] = json!(format!("{fault:?}"));
    }
    let id = document_digest(&manifest);
    Ok(ProvenancedArtifact {
        source: source.to_vec(),
        entry: entry.to_owned(),
        net,
        proof,
        net_bytes: snapshot,
        artifact,
        manifest,
        id,
    })
}
