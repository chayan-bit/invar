//! The contract axes of `docs/spec/invar-v0.2-axes.md`.
//!
//! A region has one level on each axis, and the levels are independent. This
//! file holds the C (numeric contract) levels and the table that maps every
//! numeric contract on `main` to exactly one of them. A level is a rule set
//! over the typed-net rewrite and extraction, never a second pipeline.

use crate::front::FrontError;
use crate::numeric_contract::{FoldTier, NumericPolicy};

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

/// One axis a region declares: its attribute key, its levels with the safe
/// default first, and the levels this engine honours today. A declared level
/// outside `honoured` is refused, never run at another level.
pub struct Axis {
    pub key: &'static str,
    pub name: &'static str,
    pub levels: &'static [&'static str],
    pub honoured: &'static [&'static str],
}

pub const C_AXIS: Axis = Axis {
    key: "c",
    name: "C",
    levels: &["c0", "c1", "c2", "c5"],
    honoured: &["c0", "c1", "c2"],
};
pub const S_AXIS: Axis = Axis {
    key: "s",
    name: "S",
    levels: &["s0", "s1", "s2"],
    honoured: &["s0"],
};
pub const N_AXIS: Axis = Axis {
    key: "n",
    name: "N",
    levels: &["n0", "n1", "n2", "n3", "n4"],
    honoured: &["n0", "n1", "n2"],
};
pub const T_AXIS: Axis = Axis {
    key: "t",
    name: "T",
    levels: &["t0", "t1", "t2", "t3"],
    honoured: &["t0"],
};
/// The axes `#[region(...)]` declares, in receipt order.
pub const AXES: [&Axis; 4] = [&C_AXIS, &S_AXIS, &N_AXIS, &T_AXIS];
/// The contract names a region may name: exactly the ids of `CONTRACTS`.
pub const CONTRACT_NAMES: [&str; 7] = [
    "strict-f32-v1",
    "strict-f64-v1",
    "invar-math-cr-v1",
    "f32-tree-v2",
    "f32-exact-v2",
    "f32-relaxed-v2",
    "tc-f16-f32-v1",
];

/// The levels one function declares with `#[region(...)]`, already checked
/// against what the engine honours. An absent axis is its safe level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Axes {
    /// Declared level per axis of `AXES`, in that order.
    levels: [Option<&'static str>; 4],
    contract: Option<&'static str>,
}

impl Axes {
    /// The level on `axis`, the safe one when undeclared.
    pub fn level(&self, axis: &Axis) -> &'static str {
        let index = AXES
            .iter()
            .position(|known| known.key == axis.key)
            .unwrap_or(0);
        self.levels[index].unwrap_or(axis.levels[0])
    }

    /// The named contract; `None` is the default strict contract.
    pub fn contract(&self) -> Option<&'static str> {
        self.contract
    }

    /// Whether any axis or contract was declared.
    pub fn is_declared(&self) -> bool {
        self.contract.is_some() || self.levels.iter().any(Option::is_some)
    }

    /// Whether the C level or contract was declared.
    pub fn declares_c(&self) -> bool {
        self.contract.is_some() || self.levels[0].is_some()
    }

    /// The fold tier the named contract selects for the function's folds.
    pub fn fold(&self) -> Option<FoldTier> {
        FoldTier::ALL.into_iter().find(|tier| {
            *tier != FoldTier::Strict && Some(NumericPolicy::f32_fold(*tier).id()) == self.contract
        })
    }

    /// Check the keys of one `#[region(...)]`, whose spellings the attribute
    /// registry already admitted, and return `(code, message)` on refusal.
    pub fn resolve(values: &[(&'static str, String)]) -> Result<Self, (&'static str, String)> {
        let mut axes = Self::default();
        for (key, value) in values {
            if *key == "contract" {
                axes.contract = CONTRACT_NAMES.iter().copied().find(|name| name == value);
                continue;
            }
            let Some(index) = AXES.iter().position(|axis| axis.key == *key) else {
                continue;
            };
            let axis = AXES[index];
            let Some(level) = axis.honoured.iter().copied().find(|level| level == value) else {
                return Err((
                    "CONTRACT_UNAVAILABLE",
                    format!(
                        "level {value} of axis {} is specified but not implemented; this engine honours {}",
                        axis.name,
                        axis.honoured.join(", ")
                    ),
                ));
            };
            axes.levels[index] = Some(level);
        }
        let declared = axes.levels[0].and_then(CLevel::from_name);
        match (declared, axes.contract.and_then(contract_level)) {
            (Some(c), Some(needed)) if c != needed => Err((
                "CONTRACT_MISMATCH",
                format!(
                    "contract {} is level {}, not {}",
                    axes.contract.unwrap_or_default(),
                    needed.name(),
                    c.name()
                ),
            )),
            (Some(c), None) if c != CLevel::C0 => {
                let named: Vec<_> = CONTRACTS
                    .iter()
                    .filter(|(_, level)| *level == c)
                    .map(|(name, _)| *name)
                    .collect();
                Err((
                    "CONTRACT_MISMATCH",
                    format!(
                        "level {} names its contract: one of {}",
                        c.name(),
                        named.join(", ")
                    ),
                ))
            }
            (None, Some(needed)) => {
                axes.levels[0] = Some(needed.name());
                Ok(axes)
            }
            _ => Ok(axes),
        }
    }

    /// The first axis on which this declaration exceeds a CLI cap, as
    /// (axis key, declared level, cap). Caps only narrow: an undeclared axis
    /// is its safe level, which no cap refuses.
    pub fn over_cap(&self, caps: &[(&Axis, &str)]) -> Option<(&'static str, &'static str, String)> {
        caps.iter().find_map(|(axis, cap)| {
            let rank = |level: &str| axis.levels.iter().position(|known| *known == level);
            let level = self.level(axis);
            (rank(level) > rank(cap)).then(|| (axis.key, level, (*cap).to_owned()))
        })
    }
}

/// Refuse the first function declared above a CLI cap (`--max-c`, `--max-n`).
pub fn check_caps(
    program: &crate::front::Program,
    caps: &[(&Axis, &str)],
) -> Result<(), FrontError> {
    for function in &program.functions {
        if let Some((key, level, cap)) = function.axes.over_cap(caps) {
            return Err(FrontError::source(
                "CONTRACT_CAP",
                format!(
                    "function {} declares {level}, above the cap --max-{key} {cap}",
                    function.name
                ),
                function.span,
            ));
        }
    }
    Ok(())
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
    fn region_contract_names_are_the_contracts() {
        let ids: Vec<_> = CONTRACTS.iter().map(|(name, _)| *name).collect();
        assert_eq!(CONTRACT_NAMES.as_slice(), ids.as_slice());
        let levels: Vec<_> = CLevel::ALL.iter().map(|level| level.name()).collect();
        assert_eq!(C_AXIS.levels, levels.as_slice());
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
