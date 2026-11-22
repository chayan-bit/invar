//! #199: `invar-explain-v6` (`docs/spec/invar-v0.2-axes.md` section 6). Every
//! region records its axis levels; v6 is v5 plus those fields, so an
//! undeclared program projects back to the v5 document byte for byte.

use super::region_axes::{cli, program, scratch};
use serde_json::{Value, json};

const AXIS_FIELDS: [&str; 11] = [
    "c_level",
    "s_level",
    "n_level",
    "t_level",
    "contract_name",
    "bound",
    "deterministic",
    "n_justification",
    "s_artifact",
    "rule_levels",
    "c_refused",
];

fn schema(name: &str) -> jsonschema::Validator {
    let path = format!("{}/schema/{name}", env!("CARGO_MANIFEST_DIR"));
    let schema: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    jsonschema::validator_for(&schema).unwrap()
}

fn explain(name: &str, source: &str) -> Value {
    let path = scratch(name, source);
    let (status, stdout) = cli(&["explain", path.to_str().unwrap(), "--format", "json"]);
    assert_eq!(status, 0, "{stdout}");
    let document: Value = serde_json::from_str(&stdout).unwrap();
    assert!(
        schema("explain-v6.schema.json").is_valid(&document),
        "{document}"
    );
    document
}

/// The v5 view of a v6 document: the axis fields and level sets removed.
fn project_to_v5(mut document: Value) -> Value {
    document["schema"] = json!("invar-explain-v5");
    document["schema_version"] = json!(5);
    for region in document["regions"].as_array_mut().into_iter().flatten() {
        let fields = region.as_object_mut().unwrap();
        fields.retain(|key, _| !AXIS_FIELDS.contains(&key.as_str()));
    }
    if let Some(contracts) = document
        .get_mut("numeric_contracts")
        .and_then(Value::as_object_mut)
    {
        contracts.remove("levels");
        if contracts.is_empty() {
            document
                .as_object_mut()
                .unwrap()
                .remove("numeric_contracts");
        }
    }
    document
}

/// The goldens are the v5 documents `main` printed before this change.
#[test]
fn undeclared_documents_project_to_the_committed_v5_goldens() {
    let root = env!("CARGO_MANIFEST_DIR");
    let v5 = schema("explain-v5.schema.json");
    for name in ["fsum", "generated-map", "sum"] {
        let golden =
            std::fs::read_to_string(format!("{root}/tests/numeric_tiers/explain_v5/{name}.json"))
                .unwrap();
        let golden_value: Value = serde_json::from_str(&golden).unwrap();
        assert!(v5.is_valid(&golden_value), "{name}");
        let (status, stdout) = cli(&[
            "explain",
            &format!("{root}/examples/{name}.inv"),
            "--format",
            "json",
        ]);
        assert_eq!(status, 0, "{stdout}");
        let document: Value = serde_json::from_str(&stdout).unwrap();
        assert!(
            schema("explain-v6.schema.json").is_valid(&document),
            "{document}"
        );
        for region in document["regions"].as_array().unwrap() {
            assert_eq!(
                (&region["c_level"], &region["n_level"]),
                (&json!("c0"), &json!("n0"))
            );
        }
        assert_eq!(
            project_to_v5(document).to_string(),
            golden.trim_end(),
            "{name}"
        );
    }
}

/// One receipt per honoured level on each axis, field for field.
#[test]
fn every_honoured_level_has_its_receipt() {
    let relaxed = "abs(r - exact) <= (n - 1) * 2^-24 * sum(abs(x_i))";
    let cases = [
        ("", ["c0", "s0", "n0", "t0"], None, None, None),
        (
            r#"#[region(c = "c0", s = "s0", n = "n0", t = "t0")]"#,
            ["c0", "s0", "n0", "t0"],
            None,
            None,
            None,
        ),
        (
            r#"#[region(contract = "f32-tree-v2")]"#,
            ["c1", "s0", "n0", "t0"],
            Some("f32-tree-v2"),
            None,
            None,
        ),
        (
            r#"#[region(contract = "f32-relaxed-v2")]"#,
            ["c2", "s0", "n0", "t0"],
            Some("f32-relaxed-v2"),
            Some(relaxed),
            None,
        ),
        (
            r#"#[region(n = "n1")]"#,
            ["c0", "s0", "n1", "t0"],
            None,
            None,
            Some("serial-reference-bits"),
        ),
        (
            r#"#[region(n = "n2")]"#,
            ["c0", "s0", "n2", "t0"],
            None,
            None,
            Some("serial-reference-bits"),
        ),
    ];
    for (attribute, [c, s, n, t], contract, bound, justification) in cases {
        let document = explain("receipt", &program(attribute));
        let region = &document["regions"][0];
        assert_eq!(region["function"], "fold_sum", "{document}");
        let fields: Value = AXIS_FIELDS
            .iter()
            .map(|field| (field.to_string(), region[field].clone()))
            .collect();
        let expected = json!({"c_level": c, "s_level": s, "n_level": n, "t_level": t,
            "contract_name": contract, "bound": bound, "deterministic": true,
            "n_justification": justification, "s_artifact": null,
            "rule_levels": ["c0"], "c_refused": null});
        assert_eq!(fields, expected, "{attribute}");
        let levels = &document["numeric_contracts"]["levels"];
        if attribute.is_empty() {
            assert_eq!(document.get("numeric_contracts"), None, "{document}");
        } else {
            // `main` is undeclared, so each axis also uses its safe level.
            let with_safe = |safe: &str, level: &str| {
                json!(if safe == level {
                    vec![safe]
                } else {
                    vec![safe, level]
                })
            };
            let used = json!({"c": with_safe("c0", c), "s": with_safe("s0", s),
                "n": with_safe("n0", n), "t": with_safe("t0", t)});
            assert_eq!(levels, &used, "{attribute}");
        }
    }
    // N4 (deterministic: false) and N3 are refused before any receipt exists (NA4.3).
    for attribute in [r#"#[region(n = "n4")]"#, r#"#[region(n = "n3")]"#] {
        let path = scratch("refused", &program(attribute));
        let (status, stdout) = cli(&["explain", path.to_str().unwrap(), "--format", "json"]);
        assert_eq!(status, 1, "{stdout}");
        assert!(stdout.contains("CONTRACT_UNAVAILABLE"), "{stdout}");
    }
}

/// Changing any axis level or naming a contract changes the source and plan
/// identities, so every identity-keyed record and cache misses; restating the
/// safe levels keeps the plan identity.
#[test]
fn changing_any_axis_level_changes_the_plan_identity() {
    let map = |attribute: &str| {
        format!(
            "{attribute}\nfn bump(xs: read Array<u32>) -> Array<u32> {{ var out: Array<u32> = filled_u32(len(xs), 0u32); \
             for i in 0u32..len(xs) {{ out[i] = wrapping_add(xs[i], 1u32); }} return out; }}\n\
             fn main() -> i32 {{ let xs: Array<u32> = [1u32, 2u32, 3u32]; let ys = bump(xs); \
             if ys[2u32] == 4u32 {{ return 4; }} return 0; }}"
        )
    };
    let identities = |attribute: &str| {
        let document = explain("identity", &map(attribute));
        assert_eq!(document["regions"][0]["status"], "admitted", "{document}");
        (
            document["source_identity"].clone(),
            document["regions"][0]["candidate_identity"].clone(),
        )
    };
    let (_, safe) = identities(r#"#[region(c = "c0", n = "n0")]"#);
    let (_, undeclared) = identities(&" ".repeat(r#"#[region(c = "c0", n = "n0")]"#.len()));
    assert_eq!(safe, undeclared);
    let declared = [
        r#"#[region(contract = "f32-tree-v2")]"#,
        r#"#[region(contract = "f32-relaxed-v2")]"#,
        r#"#[region(contract = "strict-f64-v1")]"#,
        r#"#[region(n = "n1")]"#,
        r#"#[region(n = "n2")]"#,
    ];
    let mut seen = vec![undeclared];
    for attribute in declared {
        let padded = format!("{attribute:<40}");
        let (source, candidate) = identities(&padded);
        assert_ne!(source, identities(&" ".repeat(40)).0, "{attribute}");
        seen.push(candidate);
    }
    let distinct: std::collections::BTreeSet<_> = seen.iter().map(Value::to_string).collect();
    assert_eq!(distinct.len(), seen.len(), "{seen:?}");
}
