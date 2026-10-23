//! The vendored Panda MJCF writes every upstream `fullinertia` in principal
//! form (`quat` + `diaginertia`, decision 2026-10-13) so MuJoCo 3.12.0 does not
//! re-diagonalize it. This test keeps the rewrite equal to the upstream tensors
//! and keeps the cell sources equal to the rewritten MJCF text.
use std::{fs, path::Path};

use sha2::{Digest, Sha256};

const UPSTREAM: &str = "tests/fixtures/foreign-components/models/franka_emika_panda/panda.xml";
const VENDORED: &str = "examples/cells/reach_place/assets/menagerie/franka_emika_panda/panda.xml";
/// Relative to the largest tensor entry; the committed rewrite is within 1.2e-15.
const RELATIVE_TOLERANCE: f64 = 4e-15;

fn read(path: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap()
}

fn numbers(text: &str) -> Vec<f64> {
    text.split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

fn rotation(q: &[f64]) -> [[f64; 3]; 3] {
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let [w, x, y, z] = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Largest |R diag(d) R^T - I| relative to the largest |I| entry.
fn relative_error(full: &[f64], quat: &[f64], diagonal: &[f64]) -> f64 {
    let [xx, yy, zz, xy, xz, yz] = [full[0], full[1], full[2], full[3], full[4], full[5]];
    let tensor = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]];
    let r = rotation(quat);
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let rebuilt: f64 = (0..3).map(|k| r[i][k] * diagonal[k] * r[j][k]).sum();
            worst = worst.max((rebuilt - tensor[i][j]).abs());
        }
    }
    let scale = tensor.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
    worst / scale
}

#[test]
fn issue_s1_06_panda_principal_inertia_matches_upstream_tensors() {
    let upstream_text = read(UPSTREAM);
    // The planner contract and the cell import's record check share the pinned
    // (vendored, upstream) pair in `tool_projection`; the files must still hash to it.
    let (vendored_sha, upstream_sha) = (sha256_hex(VENDORED), sha256_hex(UPSTREAM));
    use invar::system::tool_projection::same_kinematics;
    assert!(same_kinematics(&vendored_sha, &upstream_sha));
    assert!(!same_kinematics(&upstream_sha, &vendored_sha));
    assert!(!same_kinematics(&vendored_sha, &"0".repeat(64)));
    let vendored_text = read(VENDORED);
    let upstream = roxmltree::Document::parse(&upstream_text).unwrap();
    let vendored = roxmltree::Document::parse(&vendored_text).unwrap();
    let (a, b): (Vec<_>, Vec<_>) = (
        upstream.descendants().filter(|n| n.is_element()).collect(),
        vendored.descendants().filter(|n| n.is_element()).collect(),
    );
    assert_eq!(a.len(), b.len(), "element structure changed");
    let mut rewritten = 0;
    for (old, new) in a.iter().zip(&b) {
        assert_eq!(old.tag_name(), new.tag_name());
        let attrs = |n: &roxmltree::Node<'_, '_>| {
            n.attributes()
                .filter(|a| !matches!(a.name(), "fullinertia" | "quat" | "diaginertia"))
                .map(|a| (a.name().to_owned(), a.value().to_owned()))
                .collect::<Vec<_>>()
        };
        assert_eq!(attrs(old), attrs(new), "only inertia may change");
        if let Some(full) = old.attribute("fullinertia") {
            assert!(old.attribute("quat").is_none() && new.attribute("fullinertia").is_none());
            let quat = numbers(new.attribute("quat").unwrap());
            let diagonal = numbers(new.attribute("diaginertia").unwrap());
            assert!(diagonal.iter().all(|d| *d > 0.0));
            assert!(diagonal[0] + diagonal[1] >= diagonal[2]);
            let error = relative_error(&numbers(full), &quat, &diagonal);
            assert!(error <= RELATIVE_TOLERANCE, "{:?}: {error:e}", new.parent());
            rewritten += 1;
        } else {
            assert_eq!(old.attribute("quat"), new.attribute("quat"));
            assert_eq!(old.attribute("diaginertia"), new.attribute("diaginertia"));
        }
    }
    assert_eq!(rewritten, 8);
}

#[test]
fn issue_s1_06_panda_cell_sources_declare_the_principal_inertias() {
    let vendored = read(VENDORED);
    let model = roxmltree::Document::parse(&vendored).unwrap();
    for cell in [
        "examples/cells/reach_place/cell.inv",
        "examples/cells/reach_place/cell-ur5e.inv",
    ] {
        let source = read(cell);
        let mut checked = 0;
        for inertial in model.descendants().filter(|n| n.has_tag_name("inertial")) {
            let Some(diagonal) = inertial.attribute("diaginertia") else {
                continue;
            };
            let Some(quat) = inertial.attribute("quat") else {
                continue;
            };
            let link = inertial.parent().unwrap().attribute("name").unwrap();
            let start = source.find(&format!("link {link} {{")).unwrap();
            let block = &source[start..start + source[start..].find("\n        }").unwrap()];
            let comma = |text: &str| text.split_whitespace().collect::<Vec<_>>().join(", ");
            assert!(
                block.contains(&format!(" quat ({});", comma(quat))),
                "{cell} {link} inertial quaternion"
            );
            assert!(
                block.contains(&format!("inertia ({}, 0, 0, 0) kg*m*m;", comma(diagonal))),
                "{cell} {link} principal inertia"
            );
            checked += 1;
        }
        assert_eq!(checked, 8, "{cell}");
    }
}

fn sha256_hex(path: &str) -> String {
    Sha256::digest(read(path).as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Mass golden for the principal-form model (decision 2026-10-13): contract v2
/// changes only the Panda model identity, and its Route 1 reference passes all
/// eight cases per arm with the frozen tolerances.
#[test]
fn issue_s1_06_panda_principal_mass_oracle_passes_contract_v2() {
    let dir = "tests/fixtures/foreign-components/gpu-parity-oracles";
    let contract_path = format!("{dir}/contract-v2.json");
    let reference_path = format!("{dir}/reference-v2.json");
    let contract_hash = sha256_hex(&contract_path);
    let contract: serde_json::Value = serde_json::from_str(&read(&contract_path)).unwrap();
    let reference: serde_json::Value = serde_json::from_str(&read(&reference_path)).unwrap();
    let panda = &contract["models"][0];
    assert_eq!(panda["name"], "panda");
    assert_eq!(panda["sha256"], sha256_hex(VENDORED));
    assert_eq!(panda["derived_from_sha256"], sha256_hex(UPSTREAM));
    assert_eq!(reference["contract_sha256"], contract_hash);
    assert_eq!(reference["runtime_status"], "passed");
    assert_eq!(reference["failures"], serde_json::json!([]));
    let models = reference["models"].as_array().unwrap();
    assert_eq!(models.len(), 2);
    for (model, spec) in models.iter().zip(contract["models"].as_array().unwrap()) {
        assert_eq!(model["name"], spec["name"]);
        assert_eq!(model["source_model_sha256"], spec["sha256"]);
        let cases = model["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 8, "{}", model["name"]);
        for case in cases {
            let comparisons = case["comparison"].as_object().unwrap();
            assert!(comparisons.contains_key("active_mass"));
            assert!(
                comparisons.values().all(|c| c["passed"] == true),
                "{} {}",
                model["name"],
                case["name"]
            );
        }
    }
}
