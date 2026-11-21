//! The contract axes of `docs/spec/invar-v0.2-axes.md`.
//!
//! A region has one level on each axis, and the levels are independent. This
//! file holds the C (numeric contract) levels and the table that maps every
//! numeric contract on `main` to exactly one of them. A level is a rule set
//! over the typed-net rewrite and extraction, never a second pipeline.

use crate::numeric_contract::NumericPolicy;

/// A level of the numeric contract axis C. C3 and C4 do not exist on this
/// axis; older prose uses them for S1/S2 and N4.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CLevel {
    /// Exact laws only: bit-identical to the serial reference, traps and fuel kept.
    C0,
    /// Deterministic float rules: one fixed tree or exact rounding, same bits everywhere.
    C1,
    /// Bounded-error rules, each with a declared bound, deterministic per device.
    C2,
    /// A trusted foreign kernel paired with an Invar reference function (#201).
    C5,
}

impl CLevel {
    pub const ALL: [Self; 4] = [Self::C0, Self::C1, Self::C2, Self::C5];

    /// The source and receipt spelling.
    pub const fn name(self) -> &'static str {
        match self {
            Self::C0 => "c0",
            Self::C1 => "c1",
            Self::C2 => "c2",
            Self::C5 => "c5",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|level| level.name() == name)
    }
}

/// Every numeric contract on `main` and the one C level it needs. A source
/// intrinsic that names its contract (`sum_tree`, `tc_mul_add`) and a fold tier
/// apply it at that site only; they never raise the rest of a region.
pub const CONTRACTS: [(&str, CLevel); 7] = [
    (NumericPolicy::STRICT_F32.id(), CLevel::C0),
    (NumericPolicy::STRICT_F64.id(), CLevel::C0),
    (crate::math::CONTRACT, CLevel::C0),
    (NumericPolicy::TREE_F32.id(), CLevel::C1),
    (NumericPolicy::EXACT_F32.id(), CLevel::C1),
    (NumericPolicy::RELAXED_F32.id(), CLevel::C2),
    (NumericPolicy::TC_F16_F32.id(), CLevel::C2),
];

/// The level a contract id needs; `None` for an id that is not a contract.
pub fn contract_level(id: &str) -> Option<CLevel> {
    CONTRACTS
        .iter()
        .find(|(name, _)| *name == id)
        .map(|(_, level)| *level)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Every contract on `main` maps to exactly one level, and the spec's
    /// mapping table says the same as `CONTRACTS`, row for row.
    #[test]
    fn every_contract_maps_to_exactly_one_level_as_the_spec_says() {
        let on_main = [
            NumericPolicy::STRICT_F32,
            NumericPolicy::STRICT_F64,
            NumericPolicy::TREE_F32,
            NumericPolicy::EXACT_F32,
            NumericPolicy::RELAXED_F32,
            NumericPolicy::TC_F16_F32,
        ]
        .map(NumericPolicy::id);
        for id in on_main.iter().chain([&crate::math::CONTRACT]) {
            let levels = CONTRACTS.iter().filter(|(name, _)| name == id).count();
            assert_eq!(levels, 1, "{id}");
        }
        assert_eq!(CONTRACTS.len(), on_main.len() + 1);
        let spec = include_str!("../docs/spec/invar-v0.2-axes.md");
        let rows: BTreeMap<&str, &str> = spec
            .lines()
            .filter_map(|line| line.strip_prefix("| `"))
            .filter_map(|line| {
                let (contract, rest) = line.split_once('`')?;
                let level = rest.rsplit('|').nth(1)?.trim();
                CLevel::from_name(level).map(|_| (contract, level))
            })
            .collect();
        let table: BTreeMap<&str, &str> = CONTRACTS
            .iter()
            .map(|(name, level)| (*name, level.name()))
            .collect();
        assert_eq!(rows, table);
    }

    #[test]
    fn level_names_round_trip_and_c3_c4_are_not_levels() {
        for level in CLevel::ALL {
            assert_eq!(CLevel::from_name(level.name()), Some(level));
        }
        for name in ["c3", "c4", "C0", "", "c00"] {
            assert_eq!(CLevel::from_name(name), None, "{name}");
        }
    }
}
