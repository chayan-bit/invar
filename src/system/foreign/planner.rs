use serde_json::Value;

pub(super) fn header(manifest: &Value) -> Result<String, &'static str> {
    let prefix = manifest["abi"]["prefix"]
        .as_str()
        .ok_or("missing planner ABI prefix")?;
    if !super::c_identifier(prefix) || manifest["isolation"] != "separate-process" {
        return Err("invalid planner prefix or isolation");
    }
    let ports = manifest["ports"]
        .as_array()
        .ok_or("missing planner ports")?;
    let n = crate::robotics::planner::manifest_joints(ports)?;
    let input = ports
        .iter()
        .find(|port| port["direction"] == "in")
        .ok_or("missing planner input")?;
    let output = ports
        .iter()
        .find(|port| port["direction"] == "out")
        .ok_or("missing planner output")?;
    let input = input["name"].as_str().ok_or("missing planner input name")?;
    let output = output["name"]
        .as_str()
        .ok_or("missing planner output name")?;
    if !super::c_identifier(input) || !super::c_identifier(output) || input == output {
        return Err("invalid planner port names");
    }
    let upper = prefix.to_ascii_uppercase();
    let wire = crate::robotics::planner::header(n)?;
    let header = format!(
        r#"{wire}
#ifndef {upper}_BINDING_V1_H
#define {upper}_BINDING_V1_H
#ifdef __cplusplus
extern "C" {{
#endif
typedef struct {prefix}_context {prefix}_context;
typedef inv_planner_config_wire_v1 {prefix}_config;
typedef struct {{ inv_planner_request{n}_f64_v2 port_{input}; }} {prefix}_inputs;
typedef struct {{ inv_planner_trajectory{n}_f64_v1 port_{output}; }} {prefix}_outputs;
enum {{ {upper}_JOINT_COUNT = {n}, {upper}_OK = 0, {upper}_INVALID_INPUT = 1,
    {upper}_LIFECYCLE = 2, {upper}_FAILURE = 3, {upper}_CHILD_EXIT = 4,
    {upper}_TIMEOUT = 5, {upper}_MALFORMED = 6, {upper}_CANCELLED = 7 }};
/* version must start with invar-planner-component-v2/{n}/.
 * configure borrows all pointed-to bytes only during that call.
 * step publishes output only on status 0. shutdown clears its owned context.
 * This header is not evidence that the foreign implementation passed tests.
 */
const char *{prefix}_version(void);
int32_t {prefix}_init({prefix}_context **context, const uint8_t *child_path, uint32_t child_path_len);
int32_t {prefix}_configure({prefix}_context *context, const {prefix}_config *config);
int32_t {prefix}_step({prefix}_context *context, const {prefix}_inputs *input, {prefix}_outputs *output);
int32_t {prefix}_shutdown({prefix}_context **context);
#ifdef __cplusplus
}}
#endif
#endif
"#
    );
    if header.len() > super::MAX_HEADER_BYTES {
        return Err("planner header byte limit");
    }
    Ok(header)
}
/// Structural admission of the separately versioned endpoint record. This is
/// not a new execution, signature/attestation, collision proof or S1 acceptance.
pub(super) fn verify_evidence(
    record: &serde_json::Value,
    expected_hash: &str,
) -> Result<(), &'static str> {
    use serde_json::{Value, json};
    let sha = |v: &Value| {
        v.as_str()
            .is_some_and(|h| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()))
    };
    let n = match record["library_version"].as_str() {
        Some("invar-planner-component-v2/6/INVP-v3") => 6,
        Some("invar-planner-component-v2/7/INVP-v3") => 7,
        _ => return Err("invalid production planner version"),
    };
    let metres = 0.002;
    let radians = std::f64::consts::PI / 180.0;
    if record["status"] != "pass"
        || record["manifest_hash"] != expected_hash
        || !record["component"]
            .as_str()
            .is_some_and(super::c_identifier)
        || !sha(&record["runtime_sha256"])
        || !sha(&record["library_sha256"])
        || !sha(&record["fk_manifest_hash"])
        || record["fk_library_version"]
            != "invar-component-v1/invar-pinocchio-reference-abi-1/pinocchio-4.1.0"
        || record["host"]["os"] != "linux"
        || record["host"]["arch"].as_str().is_none_or(str::is_empty)
        || record["seed"] != 123
        || record["metrics"]
            != json!({"translation":"euclidean-m","orientation":"sign-invariant-geodesic-rad","translation_max_m":metres,"orientation_max_rad":radians,"start_max_rad":1e-6})
        || record["scope"]
            != "endpoint_and_start_only; collision/live_S1_acceptance_remains_separate"
        || record["results"]["layout"] != "pass"
        || record["results"]["lifecycle"] != "pass"
        || record["results"]["invalid_inputs"] != "checked-boundary"
    {
        return Err("planner evidence is stale or incomplete");
    }
    let driver = record["host"]["nvidia_smi"]
        .as_str()
        .ok_or("missing observed NVIDIA host")?;
    if driver.is_empty()
        || driver.len() > 65536
        || driver.lines().any(|line| {
            let Some((v, name)) = line.split_once(',') else {
                return true;
            };
            let Ok(parts) = v
                .trim()
                .split('.')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()
            else {
                return true;
            };
            name.trim().is_empty() || parts.len() != 3 || parts.as_slice() < [580, 65, 6].as_slice()
        })
    {
        return Err("planner host is not qualified");
    }
    let artifacts = record["artifacts"]
        .as_object()
        .ok_or("missing planner bindings")?;
    let roles = [
        "library",
        "child",
        "model",
        "robot_config",
        "scene",
        "fk_library",
        "fk_model",
        "fk_manifest",
        "fk_evidence",
    ];
    if artifacts.len() != roles.len()
        || roles.iter().any(|role| {
            artifacts
                .get(*role)
                .is_none_or(|a| !sha(&a["sha256"]) || a["path"].as_str().is_none_or(str::is_empty))
        })
        || artifacts["library"]["sha256"] != record["library_sha256"]
        || artifacts["library"]["sha256"] == artifacts["fk_library"]["sha256"]
        || artifacts["library"]["path"] == artifacts["fk_library"]["path"]
    {
        return Err("invalid independent runtime bindings");
    }
    let inputs = record["inputs"]
        .as_array()
        .ok_or("missing planner inputs")?;
    if record["schema"] != "invar-planner-contract-evidence-v3" {
        return Err("unsupported planner evidence version");
    }
    crate::system::tool_projection::verify_value(&record["tool_projection"])?;
    if !sha(&record["cell_source"]["sha256"])
        || record["cell_source"]["path"]
            .as_str()
            .is_none_or(str::is_empty)
        || !inputs
            .iter()
            .any(|i| i["role"] == "input" && i["sha256"] == record["cell_source"]["sha256"])
        || !inputs.iter().any(|i| {
            i["role"] == "input" && i["sha256"] == record["tool_projection"]["source_scene_sha256"]
        })
        || !crate::system::tool_projection::same_kinematics(
            record["tool_projection"]["reference_model_sha256"]
                .as_str()
                .unwrap_or_default(),
            record["artifacts"]["fk_model"]["sha256"]
                .as_str()
                .unwrap_or_default(),
        )
    {
        return Err("unbound cell tool projection");
    }
    if inputs.is_empty()
        || inputs.len() > 66
        || !inputs
            .iter()
            .any(|i| i["role"] == "input" && i["sha256"] == record["runtime_sha256"])
        || inputs.iter().any(|i| {
            !sha(&i["sha256"])
                || i["path"].as_str().is_none_or(str::is_empty)
                || !matches!(
                    i["role"].as_str(),
                    Some("manifest" | "generated-header" | "shim" | "golden" | "input")
                )
        })
        || roles.iter().any(|role| {
            !inputs
                .iter()
                .any(|i| i["sha256"] == artifacts[*role]["sha256"])
        })
    {
        return Err("invalid planner input snapshot");
    }
    let vectors = record["results"]["vectors"]
        .as_array()
        .ok_or("missing endpoint observations")?;
    let bounded = |v: &Value, maximum: f64| {
        v.as_f64()
            .is_some_and(|v| v.is_finite() && v >= 0.0 && v <= maximum)
    };
    let mut ids = std::collections::BTreeSet::new();
    // Every planner world phase (free, approach, held, held-contact, home) is
    // exercised; home vectors also bind the joint goal within the start bound.
    if !(5..=16).contains(&vectors.len())
        || !(0..=4).all(|phase| vectors.iter().any(|v| v["phase"] == phase))
        || vectors.iter().any(|v| {
            let Some(id) = v["id"].as_str() else {
                return true;
            };
            if id.is_empty() || id.len() > 64 || !ids.insert(id) {
                return true;
            }
            !matches!(v["phase"].as_u64(), Some(0..=4))
                || (v["phase"] == 4) != v.get("goal_error_rad").is_some()
                || (v["phase"] == 4 && !bounded(&v["goal_error_rad"], 1e-6))
                || v["waypoints"]
                    .as_u64()
                    .is_none_or(|count| !(2..=256).contains(&count))
                || v["dt_s"]
                    .as_f64()
                    .is_none_or(|x| !x.is_finite() || x <= 0.0)
                || !bounded(&v["start_error_rad"], 1e-6)
                || !bounded(&v["translation_error_m"], metres)
                || !bounded(&v["orientation_error_rad"], radians)
                || v["final_q_rad"].as_array().is_none_or(|a| {
                    a.len() != n || a.iter().any(|x| x.as_f64().is_none_or(|x| !x.is_finite()))
                })
                || v["final_fk_translation_m"].as_array().is_none_or(|a| {
                    a.len() != 3 || a.iter().any(|x| x.as_f64().is_none_or(|x| !x.is_finite()))
                })
                || v["final_fk_quaternion_xyzw"].as_array().is_none_or(|a| {
                    a.len() != 4
                        || a.iter().any(|x| x.as_f64().is_none_or(|x| !x.is_finite()))
                        || (a
                            .iter()
                            .filter_map(Value::as_f64)
                            .map(|x| x * x)
                            .sum::<f64>()
                            .sqrt()
                            - 1.0)
                            .abs()
                            > 1e-12
                })
        })
    {
        return Err("invalid planner endpoint observations");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::verify_evidence;
    use serde_json::{Value, json};

    #[test]
    fn endpoint_record_decoder_rejects_stale_limits_host_and_incomplete_bindings() {
        // In-memory decoder specimen only. Invented unit paths/device label
        // are explicit; this value is never saved, emitted by the CLI, or
        // treated as observed hardware/production evidence.
        let roles = [
            "library",
            "child",
            "model",
            "robot_config",
            "scene",
            "fk_library",
            "fk_model",
            "fk_manifest",
            "fk_evidence",
        ];
        let artifacts: serde_json::Map<String, Value> = roles
            .iter()
            .enumerate()
            .map(|(i, role)| {
                (
                    (*role).to_owned(),
                    json!({"path":format!("unit-fixture/{role}"),"sha256":format!("{:064x}",i+1)}),
                )
            })
            .collect();
        let runtime_hash = "a".repeat(64);
        let mut inputs =
            vec![json!({"role":"input","path":"unit-fixture/runtime","sha256":runtime_hash})];
        inputs.extend(
            artifacts
                .values()
                .map(|a| json!({"role":"input","path":a["path"],"sha256":a["sha256"]})),
        );
        let vector = json!({"id":"unit-phase-0","phase":0,"waypoints":2,"dt_s":0.1,"start_error_rad":1e-6,"translation_error_m":0.002,"orientation_error_rad":std::f64::consts::PI/180.0,"final_q_rad":vec![0.0;6],"final_fk_translation_m":vec![0.0;3],"final_fk_quaternion_xyzw":[0.0,0.0,0.0,1.0]});
        let mut vectors = vec![vector.clone()];
        for phase in 1..=4 {
            let mut v = vector.clone();
            v["id"] = json!(format!("unit-phase-{phase}"));
            v["phase"] = json!(phase);
            if phase == 4 {
                v["goal_error_rad"] = json!(1e-6);
            }
            vectors.push(v);
        }
        let expected_hash = "b".repeat(64);
        let mut record = json!({"schema":"invar-planner-contract-evidence-v3","status":"pass","component":"unit_fixture","manifest_hash":expected_hash,"runtime_sha256":runtime_hash,"library_sha256":artifacts["library"]["sha256"],"fk_manifest_hash":"c".repeat(64),"library_version":"invar-planner-component-v2/6/INVP-v3","fk_library_version":"invar-component-v1/invar-pinocchio-reference-abi-1/pinocchio-4.1.0","host":{"os":"linux","arch":"unit-fixture","nvidia_smi":"580.65.06, SYNTHETIC UNIT FIXTURE"},"seed":123,"metrics":{"translation":"euclidean-m","orientation":"sign-invariant-geodesic-rad","translation_max_m":0.002,"orientation_max_rad":std::f64::consts::PI/180.0,"start_max_rad":1e-6},"scope":"endpoint_and_start_only; collision/live_S1_acceptance_remains_separate","artifacts":artifacts,"inputs":inputs,"results":{"layout":"pass","lifecycle":"pass","invalid_inputs":"checked-boundary","vectors":vectors}});
        // Structural specimen from the shipped source-base runtime, never emitted as hardware evidence.
        let runtime: Value = serde_json::from_str(include_str!(
            "../../../examples/cells/reach_place/components/curobo_planner/runtime-source-ur5e-v3.json"
        ))
        .unwrap();
        record["tool_projection"] = runtime["tool_projection"].clone();
        record["cell_source"] = runtime["cell_source"].clone();
        let old = record["artifacts"]["fk_model"]["sha256"].clone();
        record["artifacts"]["fk_model"]["sha256"] =
            runtime["tool_projection"]["reference_model_sha256"].clone();
        for i in record["inputs"].as_array_mut().unwrap() {
            if i["sha256"] == old {
                i["sha256"] = runtime["tool_projection"]["reference_model_sha256"].clone();
            }
        }
        record["inputs"].as_array_mut().unwrap().extend([json!({"role":"input","path":"unit-fixture/cell","sha256":runtime["cell_source"]["sha256"]}),json!({"role":"input","path":"unit-fixture/scene","sha256":runtime["tool_projection"]["source_scene_sha256"]})]);
        assert!(verify_evidence(&record, &expected_hash).is_ok());
        for case in 0..5 {
            let mut changed = record.clone();
            match case {
                0 => changed["schema"] = json!("invar-planner-contract-evidence-v2"),
                1 => changed["cell_source"]["sha256"] = json!("0".repeat(64)),
                2 => {
                    changed["tool_projection"]["reference_from_cell"]["quaternion_xyzw"] =
                        json!([0.0, 0.0, 0.0, 1.0])
                }
                3 => {
                    changed["tool_projection"]["source_base_from_reference_base"]["translation_m"] =
                        json!([0.0, 0.0, 0.1])
                }
                _ => changed["inputs"]
                    .as_array_mut()
                    .unwrap()
                    .retain(|i| i["sha256"] != runtime["tool_projection"]["source_scene_sha256"]),
            }
            assert!(verify_evidence(&changed, &expected_hash).is_err());
        }

        assert!(verify_evidence(&record, &"0".repeat(64)).is_err());
        for case in 0..10 {
            let mut changed = record.clone();
            match case {
                0 => {
                    changed["results"]["vectors"][0]["start_error_rad"] =
                        json!(f64::from_bits(1e-6_f64.to_bits() + 1))
                }
                1 => {
                    changed["results"]["vectors"][0]["translation_error_m"] =
                        json!(f64::from_bits(0.002_f64.to_bits() + 1))
                }
                2 => {
                    changed["results"]["vectors"][0]["orientation_error_rad"] =
                        json!(f64::from_bits((std::f64::consts::PI / 180.0).to_bits() + 1))
                }
                3 => changed["host"]["nvidia_smi"] = json!("551.61, SYNTHETIC UNIT FIXTURE"),
                4 => {
                    changed["inputs"]
                        .as_array_mut()
                        .unwrap()
                        .retain(|i| i["sha256"] != record["artifacts"]["fk_evidence"]["sha256"]);
                }
                5 => changed["artifacts"]["fk_library"] = changed["artifacts"]["library"].clone(),
                6 => changed["results"]["vectors"][1]["phase"] = json!(0),
                7 => changed["results"]["vectors"][4]["goal_error_rad"] = json!(2e-6),
                8 => {
                    changed["results"]["vectors"][4]
                        .as_object_mut()
                        .unwrap()
                        .remove("goal_error_rad");
                }
                _ => changed["results"]["vectors"][0]["final_q_rad"] = json!(vec![0.0; 7]),
            }
            assert!(
                verify_evidence(&changed, &expected_hash).is_err(),
                "case {case}"
            );
        }
    }
}
