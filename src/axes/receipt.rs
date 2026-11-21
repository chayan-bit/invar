//! The axis fields of an `invar-explain-v6` region record and the per-axis
//! level sets of its `numeric_contracts` (`docs/spec/invar-v0.2-axes.md`
//! section 6, `schema/explain-v6.schema.json`).

use serde_json::{Map, Value, json};

use super::{AXES, Axes, N_AXIS};

/// The error bound each bounded-error (C2) contract states; C0 and C1
/// contracts have none, because their oracle is bit equality.
const BOUNDS: [(&str, &str); 2] = [
    (
        "f32-relaxed-v2",
        "abs(r - exact) <= (n - 1) * 2^-24 * sum(abs(x_i))",
    ),
    (
        "tc-f16-f32-v1",
        "abs(r - exact) <= 2 * K * 2^-23 * sum(abs(p_i))",
    ),
];

/// Why a region at N1 or N2 meets its level: every plan admitted today is
/// bit-identical to the serial reference (NA4.5), the canonical result.
pub const N_JUSTIFICATION: &str = "serial-reference-bits";

/// The bound a contract states, `None` for a C0 or C1 contract.
pub fn contract_bound(contract: &str) -> Option<&'static str> {
    BOUNDS
        .iter()
        .find(|(name, _)| *name == contract)
        .map(|(_, bound)| *bound)
}

impl Axes {
    /// The v6 fields of one region record: every axis level, the contract and
    /// its bound, whether the run is deterministic and the N and S evidence.
    pub fn receipt(&self) -> Map<String, Value> {
        let n = self.level(&N_AXIS);
        let mut fields: Map<String, Value> = AXES
            .iter()
            .map(|axis| (format!("{}_level", axis.key), json!(self.level(axis))))
            .collect();
        fields.insert("contract_name".into(), json!(self.contract));
        fields.insert(
            "bound".into(),
            json!(self.contract.and_then(contract_bound)),
        );
        // shortcut: N3 is refused today; when it lands, an N3 predicate that admits several results is also false here.
        fields.insert("deterministic".into(), json!(n != "n4"));
        fields.insert(
            "n_justification".into(),
            json!((n != N_AXIS.levels[0]).then_some(N_JUSTIFICATION)),
        );
        fields.insert("s_artifact".into(), Value::Null);
        fields
    }

    /// The bytes a plan identity gains from this declaration: none at the safe
    /// levels without a contract, so an undeclared region keeps its identity.
    pub fn identity_bytes(&self) -> Vec<u8> {
        let safe = AXES.iter().all(|axis| self.level(axis) == axis.levels[0]);
        if safe && self.contract.is_none() {
            return Vec::new();
        }
        let levels = AXES.map(|axis| self.level(axis)).join(",");
        format!("axes:{levels};{}", self.contract.unwrap_or_default()).into_bytes()
    }
}

/// Each axis's levels used by `regions`, in level order, keyed by axis.
pub fn levels_used(regions: impl IntoIterator<Item = Axes> + Clone) -> Value {
    let used = AXES.iter().map(|axis| {
        let levels: Vec<_> = axis
            .levels
            .iter()
            .filter(|level| {
                regions
                    .clone()
                    .into_iter()
                    .any(|r| r.level(axis) == **level)
            })
            .collect();
        (axis.key.to_owned(), json!(levels))
    });
    Value::Object(used.collect())
}

/// Record in `numeric_contracts.levels` the levels each axis uses across the
/// functions of `program`, undeclared ones at the safe levels; a program that
/// declares nothing gains no field, so its document keeps the v5 shape.
pub fn record_levels(document: &mut Value, program: &crate::mir::Program) {
    if program.region_axes.values().any(Axes::is_declared) {
        let axes = program.functions.iter();
        let axes = axes.map(|function| {
            program
                .region_axes
                .get(&function.name)
                .copied()
                .unwrap_or_default()
        });
        document["numeric_contracts"]["levels"] = levels_used(axes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axes::{C_AXIS, CLevel, CONTRACTS};

    fn declared(values: &[(&'static str, &str)]) -> Axes {
        let values: Vec<_> = values
            .iter()
            .map(|(key, value)| (*key, (*value).to_owned()))
            .collect();
        Axes::resolve(&values).unwrap()
    }

    /// Exactly the C2 contracts state a bound, and an undeclared region has
    /// the safe levels, no identity bytes and is deterministic.
    #[test]
    fn bounds_belong_to_c2_and_the_safe_receipt_adds_no_identity() {
        for (name, level) in CONTRACTS {
            assert_eq!(
                contract_bound(name).is_some(),
                level == CLevel::C2,
                "{name}"
            );
        }
        let safe = Axes::default().receipt();
        assert_eq!(safe["c_level"], "c0");
        assert_eq!(safe["n_level"], "n0");
        assert_eq!(safe["deterministic"], true);
        assert_eq!(safe["n_justification"], Value::Null);
        assert!(Axes::default().identity_bytes().is_empty());
        assert!(
            declared(&[("n", "n0"), ("c", "c0")])
                .identity_bytes()
                .is_empty()
        );
    }

    /// Changing any one axis level, or naming a contract, changes the identity bytes.
    #[test]
    fn every_level_and_contract_changes_the_identity() {
        let mut seen = vec![Axes::default().identity_bytes()];
        for axis in AXES {
            for level in axis.honoured.iter().skip(1) {
                let axes = if axis.key == C_AXIS.key {
                    let contract = CONTRACTS
                        .iter()
                        .find(|(_, c)| c.name() == *level)
                        .unwrap()
                        .0;
                    declared(&[("contract", contract)])
                } else {
                    declared(&[(axis.key, level)])
                };
                seen.push(axes.identity_bytes());
            }
        }
        seen.push(declared(&[("contract", "strict-f64-v1")]).identity_bytes());
        let distinct: std::collections::BTreeSet<_> = seen.iter().collect();
        assert_eq!(distinct.len(), seen.len(), "{seen:?}");
    }

    #[test]
    fn level_sets_follow_level_order() {
        let regions = [
            declared(&[("n", "n2")]),
            Axes::default(),
            declared(&[("contract", "f32-tree-v2")]),
        ];
        let used = levels_used(regions);
        assert_eq!(
            used,
            json!({"c": ["c0", "c1"], "s": ["s0"], "n": ["n0", "n2"], "t": ["t0"]})
        );
    }
}
