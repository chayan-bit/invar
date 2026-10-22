//! Integration checks for reviewed CPU evidence, not fresh hardware acceptance.
use std::{fs, path::Path};

use serde_json::Value;
use sha2::{Digest, Sha256};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn document(path: &str) -> Value {
    serde_json::from_slice(&fs::read(root().join(path)).expect("fixture exists"))
        .expect("valid evidence JSON")
}

#[test]
fn issue_gnet_05_reviewed_oracle_bytes_remain_immutable() {
    let fixtures = [
        (
            "contract.json",
            "85362985e3729e6f40eac401efc3772d27fb90f93238f3e0174a01517bd2b0b6",
        ),
        (
            "reference.json",
            "8a4cef46ff9aeec67ab51f5c426b219e0d467eb16d0921679586ebc6cceb64c4",
        ),
        (
            "attribution.json",
            "3350f84ca3fb3f0f30d322a76c06a6d978a8cf140e57afb5244f6d1ccb0c5a91",
        ),
        (
            "all_link_contract.json",
            "eb010c784434d8175c63001ac490ec7aa741f041f65400075419daa16159f7f5",
        ),
        (
            "all_link_reference.json",
            "1ef9a54634610e7939f63ac478b656a74bd77ab67495f67e7f7a278870019847",
        ),
    ];
    for (name, expected) in fixtures {
        let bytes = fs::read(
            root()
                .join("tests/fixtures/foreign-components/gpu-parity-oracles")
                .join(name),
        )
        .expect("reviewed oracle exists");
        let actual: String = Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(actual, expected, "{name}");
    }
}

#[test]
fn issue_gnet_05_current_spec_records_body_fk_without_promoting_mass_or_gpu() {
    let spec = document("bench/general-net-v01/parity-spec.json");
    let workload = spec["workloads"]
        .as_array()
        .expect("workloads")
        .iter()
        .find(|row| row["id"] == "panda_ur5e_fk_jacobian_dynamics")
        .expect("robotics workload");
    let gaps = workload["reference_gaps"]
        .as_array()
        .expect("reference gaps");
    assert!(
        !gaps.iter().any(|gap| gap == "all_link_FK_goldens"),
        "reviewed all-body FK evidence must replace the stale missing-evidence claim"
    );
    assert!(
        gaps.iter()
            .any(|gap| gap == "Panda_active_mass_CPU_cross_library_failure_6_of_8")
    );
    assert!(
        gaps.iter()
            .any(|gap| gap == "scored_GPU_floating_error_contract")
    );
    assert_eq!(
        workload["status"],
        "pending_source_primitives_and_reference"
    );
    let evidence = &workload["all_body_fk_cpu_evidence"];
    assert_eq!(
        evidence["scope"],
        "historical_independent_cpu_reference_only"
    );
    assert_eq!(evidence["checked_transforms"], 144);
    assert_eq!(evidence["failures"], 0);
    assert_eq!(
        evidence["contract_sha256"],
        "eb010c784434d8175c63001ac490ec7aa741f041f65400075419daa16159f7f5"
    );
    assert_eq!(
        evidence["reference_sha256"],
        "1ef9a54634610e7939f63ac478b656a74bd77ab67495f67e7f7a278870019847"
    );
    assert_eq!(spec["measurement_status"], "none");
}

fn matrix7(value: &Value) -> Result<[[f64; 7]; 7], &'static str> {
    let rows = value.as_array().ok_or("missing mass matrix")?;
    if rows.len() != 7 {
        return Err("mass row count");
    }
    let mut result = [[0.0_f64; 7]; 7];
    for (target, row) in result.iter_mut().zip(rows) {
        let values = row.as_array().ok_or("invalid mass row")?;
        if values.len() != 7 {
            return Err("mass column count");
        }
        for (slot, value) in target.iter_mut().zip(values) {
            *slot = value
                .as_str()
                .ok_or("mass encoding")?
                .parse()
                .map_err(|_| "mass encoding")?;
            if !slot.is_finite() {
                return Err("nonfinite mass");
            }
        }
    }
    Ok(result)
}

fn compare_selected_mass(
    selection: &Value,
    contract: &Value,
    reference: &Value,
    attribution: &Value,
) -> Result<(f64, f64), &'static str> {
    if selection["schema"] != "invar-panda-mass-reference-selection-v1"
        || selection["scope"] != "historical_cpu_reference_selection_only"
        || selection["model"] != "panda"
        || selection["joint_count"] != 7
        || selection["source_fidelity"]["matrix"] != "mass_pinocchio"
        || selection["source_fidelity"]["witness"] != "source_kinetic_mass"
        || selection["compiled_runtime"]["matrix"] != "mass_mujoco"
        || selection["compiled_runtime"]["witness"] != "compiled_kinetic_mass"
        || selection["witness_check"]["abs_residual_exclusive"] != 1e-10
        || reference["runtime_status"] != "failed"
        || attribution["status"] != "attributed"
    {
        return Err("mass reference selection contract");
    }
    let original = &reference["models"][0];
    if original["name"] != "panda"
        || contract["models"][0]["name"] != "panda"
        || original["source_model_sha256"] != selection["source_model_sha256"]
        || attribution["source_sha256"] != selection["source_model_sha256"]
        || original["active_joints"] != contract["models"][0]["active_joints"]
        || original["fixed_inactive_joints"] != contract["models"][0]["fixed_inactive_joints"]
    {
        return Err("mass model identity");
    }
    let cases = original["cases"].as_array().ok_or("missing cases")?;
    let witnesses = attribution["cases"].as_array().ok_or("missing witnesses")?;
    let order = contract["inputs"]["ordered_cases"]
        .as_array()
        .ok_or("missing order")?;
    if cases.len() != 8 || witnesses.len() != 8 || order.len() != 8 {
        return Err("mass case count");
    }
    let mut maxima = [0.0_f64; 2];
    let mut failed = 0;
    for ((case, witness), name) in cases.iter().zip(witnesses).zip(order) {
        if case["name"] != *name || witness["name"] != *name || case["q_rad"] != witness["q_rad"] {
            return Err("mass input case mismatch");
        }
        if case["comparison"]["active_mass"]["passed"] == false {
            failed += 1;
        }
        for (index, key, reconstruction) in [
            (0, "mass_pinocchio", "source_kinetic_mass"),
            (1, "mass_mujoco", "compiled_kinetic_mass"),
        ] {
            let matrix = matrix7(&case[key])?;
            let energy = matrix7(&witness[reconstruction])?;
            let raw = matrix7(
                &case[if index == 0 {
                    "mass_pinocchio_without_armature"
                } else {
                    "mass_mujoco_raw"
                }],
            )?;
            for i in 0..7 {
                let armature: f64 = contract["models"][0]["source_armature_diag_text"][i]
                    .as_str()
                    .ok_or("armature")?
                    .parse()
                    .map_err(|_| "armature")?;
                let imported: f64 = case["mujoco_imported_armature"][i]
                    .as_str()
                    .ok_or("imported armature")?
                    .parse()
                    .map_err(|_| "imported armature")?;
                if !armature.is_finite() || !imported.is_finite() {
                    return Err("armature finiteness");
                }
                for j in 0..7 {
                    let residual = (matrix[i][j] - energy[i][j]).abs();
                    maxima[index] = maxima[index].max(residual);
                    if residual >= 1e-10 {
                        return Err("kinetic-energy witness mismatch");
                    }
                    let diagonal = if i != j {
                        0.0
                    } else if index == 0 {
                        armature
                    } else {
                        armature - imported
                    };
                    if (matrix[i][j] - raw[i][j] - diagonal).abs() >= 1e-10 {
                        return Err("armature not included once");
                    }
                }
            }
        }
    }
    if failed != 6
        || selection["original_cross_library_comparison"]["status"] != "failed"
        || selection["original_cross_library_comparison"]["failed_cases"] != 6
        || selection["original_cross_library_comparison"]["cases"] != 8
    {
        return Err("historical failure was erased");
    }
    Ok((maxima[0], maxima[1]))
}

#[test]
fn issue_gnet_05_source_exact_mass_selection_keeps_compiled_physics_distinct() {
    let directory = "tests/fixtures/foreign-components/gpu-parity-oracles";
    let selection = document(&format!("{directory}/mass_reference_selection.json"));
    for name in ["contract.json", "reference.json", "attribution.json"] {
        let bytes = fs::read(root().join(directory).join(name)).unwrap();
        let hash: String = Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(selection["inputs"][name], hash, "{name}");
    }
    let contract = document(&format!("{directory}/contract.json"));
    let model_path = contract["models"][0]["file"].as_str().unwrap();
    // The historical selection describes the upstream Menagerie model, kept
    // byte-for-byte here; the cell's vendored copy now uses principal inertias
    // (contract-v2.json / reference-v2.json).
    let model_bytes = fs::read(
        root()
            .join("tests/fixtures/foreign-components/models")
            .join(model_path),
    )
    .unwrap();
    let model_hash: String = Sha256::digest(model_bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(selection["source_model_sha256"], model_hash);
    let reference = document(&format!("{directory}/reference.json"));
    let attribution = document(&format!("{directory}/attribution.json"));
    let (source, compiled) =
        compare_selected_mass(&selection, &contract, &reference, &attribution).unwrap();
    assert!(source <= 8e-15 && compiled <= 9e-16);
    let mut wrong = selection.clone();
    wrong["source_fidelity"]["matrix"] = serde_json::json!("mass_mujoco");
    assert!(compare_selected_mass(&wrong, &contract, &reference, &attribution).is_err());
    let mut wrong = attribution.clone();
    wrong["cases"][0]["source_kinetic_mass"] = wrong["cases"][0]["compiled_kinetic_mass"].clone();
    assert_eq!(
        compare_selected_mass(&selection, &contract, &reference, &wrong),
        Err("kinetic-energy witness mismatch")
    );
    wrong = attribution.clone();
    wrong["cases"][0]["q_rad"][0] = serde_json::json!("1.0");
    assert!(compare_selected_mass(&selection, &contract, &reference, &wrong).is_err());
    let mut wrong_reference = reference.clone();
    wrong_reference["runtime_status"] = serde_json::json!("pass");
    assert!(compare_selected_mass(&selection, &contract, &wrong_reference, &attribution).is_err());
    let mut doubled = reference.clone();
    let mut doubled_witness = attribution.clone();
    let diagonal = doubled["models"][0]["cases"][0]["mass_pinocchio"][0][0]
        .as_str()
        .unwrap()
        .parse::<f64>()
        .unwrap()
        + 0.1;
    doubled["models"][0]["cases"][0]["mass_pinocchio"][0][0] =
        serde_json::json!(diagonal.to_string());
    doubled_witness["cases"][0]["source_kinetic_mass"][0][0] =
        serde_json::json!(diagonal.to_string());
    assert_eq!(
        compare_selected_mass(&selection, &contract, &doubled, &doubled_witness),
        Err("armature not included once")
    );
    wrong = attribution.clone();
    wrong["cases"][0]["source_kinetic_mass"][0][0] = serde_json::json!("NaN");
    assert!(compare_selected_mass(&selection, &contract, &reference, &wrong).is_err());
}
