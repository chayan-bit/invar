//! ABI12 stage 2b compile-receipt block `cuda_owned_numeric_continuation`
//! (`invar.gnet.abi12.compile-receipt` v1). It is built only from a live ABI12
//! CUDA artifact after emission and is never parsed from caller JSON. It
//! grants no execution, module-load, resume or device-qualification
//! authority. ABI12 is refused explicitly by `fenced_execution`,
//! `continuation_identity` and `device_host::cpu::CpuModel::prepare`; the
//! other identity, host and ABI5 device routes keep their own ABI gates.
//! Qualification is a separate evidence document, never a receipt flag.
use super::{Error, compiled, task_fields};
use serde_json::{Value, json};

pub(super) const KEY: &str = "cuda_owned_numeric_continuation";
const SYMBOL: &str = "invar_gnet_owned_numeric_resume_v1";
const PARAMS: [&str; 7] = [
    "const gnet_w* input",
    "gnet_w* output",
    "gnet_u* trace",
    "const gnet_u* config",
    "gnet_w* heap",
    "gnet_u* cleanup",
    "GnetTask* tasks",
];
const LANE_GUARD: &str = "if (lane>=config[0]) return;";

/// Exactly one kernel, with the frozen symbol, parameter list and lane guard.
fn kernel_text_matches(text: &str, symbol: &str) -> bool {
    let declaration = format!(
        "extern \"C\" __global__ void {symbol}({})",
        PARAMS.join(",")
    );
    symbol == SYMBOL
        && text.matches("__global__").count() == 1
        && text.matches(&declaration).count() == 1
        && text.matches(LANE_GUARD).count() == 1
}

/// Fails closed with `Compiler(State)` if the emitted text, kernel symbol,
/// workset layout or task size disagree with the frozen profile.
pub(super) fn block(artifact: &compiled::Artifact) -> Result<Value, Error> {
    if artifact.abi_version() != 12 || artifact.target() != compiled::Target::Cuda {
        return Err(Error::Compiler(compiled::Error::NumericTarget));
    }
    let state = || Error::Compiler(compiled::Error::State);
    let facts = artifact
        .owned_numeric_cuda_facts()
        .map_err(Error::Compiler)?;
    let continuation = artifact.continuation_layout().ok_or_else(state)?;
    let symbol = continuation.kernel_name();
    let frame_bytes = artifact.frame_bytes();
    if !kernel_text_matches(artifact.source(), symbol)
        || facts.lane_block != 32
        || facts.task_words * 8 != continuation.task_bytes()
        || continuation.task_bytes() != frame_bytes + 144
        || !frame_bytes.is_multiple_of(compiled::MAX_FRAMES)
        || artifact.numeric_policy().map(|p| p.id()) != Some("strict-f32-v1")
    {
        return Err(state());
    }
    Ok(json!({
        "schema": "invar.gnet.abi12.compile-receipt",
        "schema_version": 1,
        "profile": "cuda-owned-numeric-continuation-v1",
        "abi_version": 12,
        "target": "Cuda",
        "authority": {"execution": false, "module_load": false, "resume": false,
            "device_qualification": false},
        "kernel": {"symbol": symbol, "declarations": 1,
            "signature": format!("extern \"C\" __global__ void({})", PARAMS.join(", ")),
            "lane": "blockIdx.x*blockDim.x+threadIdx.x",
            "lane_guard": "lane >= config[0] returns"},
        "launch": {"block_threads": 32, "lane_block": facts.lane_block,
            "max_roots": facts.max_roots, "lanes": "roots rounded up to a multiple of 32"},
        "workset": {
            "buffers": ["input", "output", "trace", "config", "heap", "cleanup", "tasks"],
            "word_bytes": 8, "red_zone_words": facts.red_zone_words,
            "per_lane_words": {"input": facts.input_words, "output": facts.output_words,
                "trace_per_event": facts.event_words, "heap": facts.heap_words,
                "cleanup": facts.cleanup_words, "tasks": facts.task_words},
            "config_words": ["roots", "fuel", "transitions", "trace_events", "quantum"],
            "config_bounds": {"roots": [1, facts.max_roots], "fuel": [1, facts.max_fuel],
                "transitions": [0, facts.max_transitions],
                "trace_events": [0, facts.max_trace_events], "quantum": [1, facts.max_quantum]},
            "max_device_bytes": facts.max_device_bytes,
            "max_host_snapshot_bytes": facts.max_host_snapshot_bytes,
            "min_retained_snapshots": facts.min_retained_snapshots
        },
        "task": {"frame_count": compiled::MAX_FRAMES,
            "frame_stride_bytes": frame_bytes / compiled::MAX_FRAMES,
            "task_bytes": continuation.task_bytes(), "task_fields": task_fields(frame_bytes)},
        "output": {"words": facts.output_words, "terminal_packet": "abi10-compatible",
            "pending_sentinel": {"value": i64::MIN, "words": "all 10",
                "required": "every live root before the first submission and while pending"}},
        "advance_rule": "device-epoch-v1",
        "numeric": {"policy": "strict-f32-v1", "implementation": "integer-software-binary32",
            "canonical_nan_bits": 0x7fc0_0000u32},
        "semantic_reference": "cpu-owned-numeric-continuation-v1",
        "toolchain": {"nvrtc": "not-invoked", "cubin": null}
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn abi12_text() -> String {
        let source = "fn run(a: f32, n: u32) -> f32 { var xs: Array<f32> = filled_f32(n + 1u32, a); xs[0u32] = xs[0u32] * a + a; return xs[0u32]; } fn main() -> i32 { return 0; }";
        super::super::compile_resumable_owned_numeric(
            source.as_bytes(),
            "run",
            compiled::Target::Cuda,
            super::super::Limits::default(),
            compiled::OwnedLimits::default(),
        )
        .unwrap()
        .artifact()
        .source()
        .to_owned()
    }

    #[test]
    fn abi12_receipt_refuses_kernel_text_that_differs_from_the_profile() {
        let text = abi12_text();
        assert!(kernel_text_matches(&text, SYMBOL));
        let second = format!("{text}\nextern \"C\" __global__ void other() {{}}\n");
        let swapped = text.replace(
            "gnet_u* cleanup,GnetTask* tasks",
            "GnetTask* tasks,gnet_u* cleanup",
        );
        let unguarded = text.replace(LANE_GUARD, "");
        let doubled_guard = format!("{text}\n// {LANE_GUARD}\n");
        let renamed = text.replace(&format!("void {SYMBOL}("), "void invar_gnet_resume_v3(");
        for (name, bad, symbol) in [
            ("second kernel", second.as_str(), SYMBOL),
            ("parameter order", swapped.as_str(), SYMBOL),
            ("missing lane guard", unguarded.as_str(), SYMBOL),
            ("repeated lane guard", doubled_guard.as_str(), SYMBOL),
            ("renamed kernel", renamed.as_str(), SYMBOL),
            ("other symbol", text.as_str(), "invar_gnet_resume_v3"),
        ] {
            assert!(
                name == "other symbol" || bad != text,
                "{name} did not change the text"
            );
            assert!(!kernel_text_matches(bad, symbol), "{name}");
        }
    }
}
