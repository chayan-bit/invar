//! Stable calibration identities for explicit spatial operations. No generic
//! bandwidth model qualifies an unmeasured irregular spatial implementation.
use crate::mir::{Operation, Program};
use serde_json::{Value, json};

pub const UNMEASURED: &str = "spatial-cost-unmeasured";
pub const HOST_WORK: &str = "spatial-host-work";

pub fn family_id(operation: &str) -> Option<&'static str> {
    Some(match operation {
        "voxel_grid_build" => "spatial.voxel_grid_build.f32",
        "radius_neighbors" => "spatial.radius_neighbors.f32",
        "knn" => "spatial.knn.f32",
        "grid_excluded" => "spatial.grid_excluded.f32",
        _ => return None,
    })
}

/// Checked sites, including unreached branches. Execution receipts separately
/// identify reached calls; this list never fabricates calibration samples.
pub fn explain(program: &Program) -> Value {
    let mut sites = Vec::new();
    for function in &program.functions {
        for block in &function.blocks {
            for instruction in &block.instructions {
                let Operation::Call { name, .. } = &instruction.kind else {
                    continue;
                };
                let Some(family) = family_id(name) else {
                    continue;
                };
                sites.push(json!({
                    "function": function.name,
                    "operation": name,
                    "family_id": family,
                    "span": {"start": instruction.span.start, "end": instruction.span.end},
                    "default_implementation": "cpu-sorted-grid",
                    "reason": UNMEASURED,
                    "rule": "explicit-spatial-v1",
                }));
            }
        }
    }
    json!(sites)
}

/// Attach calibration identities to reached sites. A forced device program
/// still performs metadata reads and explicit host sorts on the CPU; preserve
/// that distinction instead of labeling those receipts as device launches.
pub fn annotate_device(program: &Program, device: &mut Value) {
    if !super::contains(program) {
        return;
    }
    let Some(choices) = device.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };
    for choice in choices {
        for function in &program.functions {
            if choice["function"].as_str() != Some(function.name.as_str()) {
                continue;
            }
            for instruction in function.blocks.iter().flat_map(|b| &b.instructions) {
                if choice["span"]["start"] != instruction.span.start
                    || choice["span"]["end"] != instruction.span.end
                {
                    continue;
                }
                if let Operation::Call { name, .. } = &instruction.kind
                    && let Some(family) = family_id(name)
                {
                    choice["family_id"] = family.into();
                }
            }
        }
        // Only a device receipt can be mislabelled; CPU routes keep their backend.
        let device = matches!(choice["backend"].as_str(), Some("cuda" | "metal"));
        if device
            && choice["implementation"]
                .as_str()
                .is_some_and(|id| id.starts_with("cpu-") || id.starts_with("host-"))
        {
            choice["backend"] = "cpu-serial".into();
            choice["reason"] = HOST_WORK.into();
        }
    }
}
