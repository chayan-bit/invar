//! P8: certificate-bound, ordered collectives on pre-lifted law elements.
//! See `docs/law-collective-v01.md` for the word ABI and publication contract.

pub mod cost;
mod emit;

pub use cost::{Calibration, Decision, Figures, MARGIN, Residency, Route, TIE, decide, route};

use super::{
    Net, Proof,
    normal_form::{self, Certificate, Guard, LawId, Rep},
};

/// A checked P1b representation. No public constructor accepts target source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LawRepr {
    certificate: Certificate,
    representation: Representation,
    words: u32,
    prefix_bounds: bool,
}

/// Typed emitter IR: identity, ordered combine, apply and guard are lowered
/// from this enum, rather than interpolating caller-supplied source snippets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Representation {
    Carried(Rep),
    /// L9: P4a supplies path-weighted non-negative residues. Raw-root handling
    /// and every certificate guard must precede publication in the caller.
    ResidueSum {
        modulus: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Certificate(normal_form::certificate::Rejection),
    Unsupported(&'static str),
    Layout,
    Limit,
}

/// AoS elements of `LawRepr::word_count()` little-endian u32 words.
/// Each lane owns consecutive elements; groups cover consecutive tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub lanes: u32,
    pub items_per_lane: u32,
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            lanes: 256,
            items_per_lane: 4,
        }
    }
}

/// Standalone target source. Entry points have the fixed ABI documented in
/// `docs/law-collective-v01.md`. No launch or publication is performed here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Emission {
    pub source: String,
    pub entry: &'static str,
    pub words: u32,
    pub tile_elements: u32,
    pub threadgroup_bytes: u32,
}

/// Scratch is never shared across dispatches. All counts are u32 words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub groups: u32,
    pub input_words: usize,
    pub output_words: usize,
    /// Each of the two zero-initialized immutable atomic look-back tables.
    pub table_words: usize,
}

impl Emission {
    /// Validate length and the aggregate 8 GiB storage ceiling before allocation.
    pub fn launch(&self, length: u32) -> Result<Launch, Error> {
        if length > 1 << 26 || self.tile_elements == 0 || self.words == 0 {
            return Err(Error::Limit);
        }
        let groups = length.div_ceil(self.tile_elements).max(1);
        let input_words = (length as usize)
            .checked_mul(self.words as usize)
            .ok_or(Error::Limit)?;
        let scan = self.entry == "invar_law_scan_v1";
        let partial_words = (groups as usize)
            .checked_mul(self.words as usize)
            .ok_or(Error::Limit)?;
        let output_words = if scan { input_words } else { partial_words };
        let table_words = if scan {
            partial_words.checked_mul(2).ok_or(Error::Limit)?
        } else {
            0
        };
        let bytes = input_words
            .checked_add(output_words)
            .and_then(|x| table_words.checked_mul(2).and_then(|t| x.checked_add(t)))
            .and_then(|x| x.checked_mul(4))
            .ok_or(Error::Limit)?;
        if bytes > invar_device::MAX_GNET_STORAGE_BYTES {
            return Err(Error::Limit);
        }
        Ok(Launch {
            groups,
            input_words,
            output_words,
            table_words,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrapRecovery {
    None,
    /// Reduce guard first; scan and locate only on a possible violation.
    PrefixBounds,
    /// Replay each source lane from its exact exclusive-prefix seed. A final
    /// state can hide an overflowing intermediate, so no clean-fold shortcut.
    ReplayEveryLane,
}

impl LawRepr {
    /// Re-derive every certificate fact against this verified net first.
    /// Whole-region users must also call P1b `certificate::check_all`.
    pub fn from_certificate(
        net: &Net,
        proof: &Proof,
        certificate: &Certificate,
    ) -> Result<Self, Error> {
        let partition = super::regions::partition(net, proof).map_err(|error| {
            Error::Certificate(normal_form::certificate::Rejection::Net(format!(
                "{error:?}"
            )))
        })?;
        normal_form::certificate::check_with(&partition, certificate)
            .map_err(Error::Certificate)?;
        match certificate.law {
            LawId::L10RangeSum => {
                return Err(Error::Unsupported(
                    "L10 native fallback: source-tree joins and checked trap order are not a leaf-prefix fold",
                ));
            }
            LawId::L11ScatterCombine => {
                return Err(Error::Unsupported(
                    "L11 native fallback: per-target segmentation, seeds and source-order replay are required",
                ));
            }
            _ => {}
        }
        let representation = match &certificate.rep {
            Some(rep) => Representation::Carried(rep.clone()),
            None if certificate.law == LawId::L9PathSum => {
                let modulus = certificate
                    .guards
                    .iter()
                    .find_map(|guard| match guard {
                        Guard::TableRange { modulus } => Some(*modulus),
                        _ => None,
                    })
                    .ok_or(Error::Unsupported("path sum without modulus"))?;
                Representation::ResidueSum { modulus }
            }
            _ => {
                return Err(Error::Unsupported(
                    "certificate has no collective representation",
                ));
            }
        };
        let words = match &representation {
            Representation::ResidueSum { .. } | Representation::Carried(Rep::Sealed(_)) => 1,
            Representation::Carried(rep) => match rep {
                Rep::LiftedSum(_) => 6,
                Rep::SaturatedProduct(_) => 4,
                Rep::Affine(ring) => 2 * ring_words(*ring),
                Rep::Matrix { ring, vars } if (1..=4).contains(&vars.len()) => {
                    ((vars.len() + 1).pow(2) as u32) * ring_words(*ring)
                }
                Rep::Matrix { .. } => return Err(Error::Unsupported("matrix dimension")),
                Rep::Tropical { .. } => 6,
                Rep::Select { companions, .. } if companions.len() <= 8 => {
                    2 + companions.len() as u32
                }
                Rep::Select { .. } => {
                    return Err(Error::Unsupported("more than eight select companions"));
                }
                Rep::Last => 2,
                Rep::Counter(_) => 2,
                Rep::Sealed(_) => 1,
                Rep::Lifted(_) => {
                    return Err(Error::Unsupported("no collective for a lifted law (#59)"));
                }
            },
        };
        let prefix_bounds = canonical_fold(&partition, certificate, &representation);
        Ok(Self {
            certificate: certificate.clone(),
            representation,
            words,
            prefix_bounds,
        })
    }

    pub fn representation(&self) -> &Representation {
        &self.representation
    }
    pub fn certificate(&self) -> &Certificate {
        &self.certificate
    }
    pub fn guards(&self) -> &[Guard] {
        &self.certificate.guards
    }
    pub fn word_count(&self) -> u32 {
        self.words
    }

    pub fn trap_recovery(&self) -> TrapRecovery {
        use normal_form::{Rep, Ring};
        match &self.representation {
            Representation::Carried(
                Rep::LiftedSum(_) | Rep::SaturatedProduct(_) | Rep::Counter(_),
            ) => {
                if self.prefix_bounds {
                    TrapRecovery::PrefixBounds
                } else {
                    TrapRecovery::ReplayEveryLane
                }
            }
            Representation::Carried(
                Rep::Affine(Ring::Checked64(_) | Ring::Residue { .. })
                | Rep::Matrix {
                    ring: Ring::Checked64(_) | Ring::Residue { .. },
                    ..
                }
                | Rep::Tropical { .. },
            ) => TrapRecovery::ReplayEveryLane,
            _ => TrapRecovery::None,
        }
    }
}

/// A normalized polynomial can hide checked intermediates. Only a single,
/// direct primitive update per element licenses the cheap prefix-bound guard.
/// Alias reads, nested loops and additional uses conservatively require replay.
fn canonical_fold(
    partition: &super::regions::Partition,
    certificate: &Certificate,
    representation: &Representation,
) -> bool {
    use super::regions::tree::{self, Expr, Stmt};
    use crate::front::BinaryOp;
    let op = match representation {
        Representation::Carried(Rep::LiftedSum(_) | Rep::Counter(_)) => BinaryOp::Add,
        Representation::Carried(Rep::SaturatedProduct(_)) => BinaryOp::Mul,
        _ => return false,
    };
    let normal_form::Subject::Var(var) = &certificate.subject else {
        return false;
    };
    let Some(tree) = partition
        .function(certificate.region.function)
        .and_then(|f| f.tree.as_ref().ok())
    else {
        return false;
    };
    let Some(inner) = certificate
        .region
        .head
        .and_then(|head| tree::find_loop(&tree.body, head))
    else {
        return false;
    };
    let mentions = |value: &Expr| {
        let mut names = Vec::new();
        value.names(&mut names);
        names.contains(var)
    };
    let mut updates = 0;
    let mut simple = true;
    tree::walk(&inner.body, &mut Vec::new(), 0, &mut |stmt, _, depth| {
        if let Stmt::Assign { name, value } = stmt
            && name == var
        {
            updates += 1;
            let direct = match value {
                Expr::Binary {
                    op: actual,
                    left,
                    right,
                    ..
                } if *actual == op => {
                    (matches!(left.as_ref(), Expr::Local(name) if name==var) && !mentions(right))
                        || (matches!(right.as_ref(), Expr::Local(name) if name==var)
                            && !mentions(left))
                }
                _ => false,
            };
            simple &= depth == 0 && direct;
        } else if tree::stmt_exprs(stmt).into_iter().any(mentions) {
            simple = false;
        }
    });
    simple && updates == 1
}

fn ring_words(ring: normal_form::Ring) -> u32 {
    match ring {
        normal_form::Ring::Checked64(_) => 2,
        _ => 1,
    }
}

fn validate(repr: &LawRepr, layout: Layout) -> Result<(), Error> {
    if !layout.lanes.is_power_of_two()
        || !(32..=1024).contains(&layout.lanes)
        || !(1..=16).contains(&layout.items_per_lane)
    {
        return Err(Error::Layout);
    }
    if (layout.lanes + 1) * repr.words * 4 + 8 > 32768 {
        return Err(Error::Limit);
    }
    Ok(())
}

pub mod msl {
    use super::*;
    pub fn emit_reduce(repr: &LawRepr, layout: Layout) -> Result<Emission, Error> {
        emit::emit(repr, layout, false, emit::Target::Metal)
    }
    pub fn emit_scan(repr: &LawRepr, layout: Layout) -> Result<Emission, Error> {
        emit::emit(repr, layout, true, emit::Target::Metal)
    }
}

pub mod cuda {
    use super::*;
    pub fn emit_reduce(repr: &LawRepr, layout: Layout) -> Result<Emission, Error> {
        emit::emit(repr, layout, false, emit::Target::Cuda)
    }
    pub fn emit_scan(repr: &LawRepr, layout: Layout) -> Result<Emission, Error> {
        emit::emit(repr, layout, true, emit::Target::Cuda)
    }
}
