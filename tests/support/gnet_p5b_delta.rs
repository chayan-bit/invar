//! P5b changes emitted bytes and receipt IDs while preserving logical cursors.
//! Immutable prior fixtures still pin admission, ABI and every storage shape.
use serde_json::{Map, Value};
pub fn apply(mut baseline: Map<String, Value>, delta: &str) -> Map<String, Value> {
    let rows: Map<String, Value> = serde_json::from_str(delta).unwrap();
    for (key, row) in rows {
        let before = &row["before"];
        let after = &row["after"];
        assert_eq!(
            baseline.get(&key),
            Some(before),
            "{key}: historical fixture changed"
        );
        assert_ne!(before, after, "{key}: redundant delta");
        assert!(
            before["abi"].is_u64() && after["abi"].is_u64(),
            "{key}: P5b cannot change admission"
        );
        let mut old = before.as_object().unwrap().clone();
        let mut new = after.as_object().unwrap().clone();
        for field in ["source_sha256", "source_bytes", "normalized_id"] {
            old.remove(field);
            new.remove(field);
        }
        assert_eq!(old, new, "{key}: P5b changed an ABI/storage shape");
        for field in ["source_bytes"] {
            if let Some(old) = before[field].as_u64() {
                assert!(after[field].as_u64().unwrap() <= old, "{key}: {field} grew");
            }
        }
        baseline.insert(key, after.clone());
    }
    baseline
}
