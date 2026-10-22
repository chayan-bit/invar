//! Explicit P0b deltas over immutable historical fixtures. Existing ABI shapes
//! stay pinned; changed source text/receipt IDs and new ABI13 rows are exact.
use serde_json::{Map, Value};
pub fn apply(legacy: &str, delta: &str) -> Map<String, Value> {
    let mut baseline: Map<String, Value> = serde_json::from_str(legacy).unwrap();
    let delta: Map<String, Value> = serde_json::from_str(delta).unwrap();
    for (key, row) in delta {
        let before = &row["before"];
        let after = &row["after"];
        assert_eq!(
            baseline.get(&key),
            Some(before),
            "{key}: historical fixture changed"
        );
        assert_ne!(before, after, "{key}: redundant delta");
        if before.get("abi").is_some() {
            let mut old = before.as_object().unwrap().clone();
            let mut new = after.as_object().unwrap().clone();
            for field in ["source_sha256", "source_bytes", "normalized_id"] {
                old.remove(field);
                new.remove(field);
            }
            assert_eq!(old, new, "{key}: P0b changed a legacy ABI shape");
        } else {
            assert!(
                key.contains("/emit_device_speculative/")
                    || key.contains("/compile_device_speculative/"),
                "{key}: unrelated admission delta"
            );
            assert!(
                matches!(
                    before["error"].as_str(),
                    Some("UnsupportedType" | "Compiler(UnsupportedType)")
                ),
                "{key}: old refusal"
            );
            if key.ends_with("/Cpu") {
                assert!(
                    matches!(
                        after["error"].as_str(),
                        Some("SpeculationTarget" | "Compiler(SpeculationTarget)")
                    ),
                    "{key}: CPU device exclusion"
                );
            } else {
                assert_eq!(after["abi"], 13, "{key}: new device profile");
            }
        }
        baseline.insert(key, after.clone());
    }
    baseline
}
