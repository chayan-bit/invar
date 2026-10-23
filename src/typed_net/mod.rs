//! Versioned, demand-ordered typed interaction graph for the bounded M1 fragment.
//!
//! This is the production graph contract. Child connections are ordered input
//! ports; changing one changes extraction or fails verification.

use std::collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

use crate::term;

mod bounded_scan;
mod codec;
mod dag;
mod extract;
mod observe;
mod rewrite;
mod verify;

pub use bounded_scan::{BoundedFamily, BoundedObservation, BoundedScanError};
pub use codec::encode;
pub use dag::DagOp;
pub use extract::{Plan, Source};
pub use observe::ArrayObservation;

pub const SCHEMA_VERSION: u32 = 1;
pub const TABLE_SCAN_RULE: &str = "finite-state-table-scan-v1";
pub const MATRIX_SCAN_RULE: &str = "u32-matrix-mod2p32-scan-v1";
pub const BETA_RULE: &str = "affine-beta-v1";
pub const MAP_FUSION_RULE: &str = "total-u32-map-fusion-v1";
pub const DAG_FUSION_RULE: &str = "total-u32-dag-fusion-v1";
/// Most source maps (and roots) one `extract_dag` plan holds: the element-SSA plan the device
/// back ends consume is sized for it, so a longer fused chain is refused as `MapLimit`.
pub const MAX_DAG_MAPS: usize = 16;

pub type NodeId = usize;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Unit,
    Bool,
    Nat,
    Pair(Box<Type>, Box<Type>),
    Function(Box<Type>, Box<Type>),
    ArrayU32,
}

impl Type {
    fn copyable(&self) -> bool {
        match self {
            Self::Unit | Self::Bool | Self::Nat => true,
            Self::Pair(a, b) => a.copyable() && b.copyable(),
            Self::Function(..) | Self::ArrayU32 => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Effect {
    Total,
    MayFail,
    Unknown,
}

impl Effect {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::MayFail, _) | (_, Self::MayFail) => Self::MayFail,
            _ => Self::Total,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MapOp {
    WrappingAdd(u32),
    WrappingMul(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZipOp {
    WrappingAdd,
    WrappingMul,
}

impl ZipOp {
    pub(crate) fn apply(self, left: u32, right: u32) -> u32 {
        match self {
            Self::WrappingAdd => left.wrapping_add(right),
            Self::WrappingMul => left.wrapping_mul(right),
        }
    }
}

impl MapOp {
    pub(crate) fn apply(self, value: u32) -> u32 {
        match self {
            Self::WrappingAdd(n) => value.wrapping_add(n),
            Self::WrappingMul(n) => value.wrapping_mul(n),
        }
    }
}

/// Which states an affine scan publishes as its array result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prefix {
    /// No array; the result is the final state.
    None,
    /// Element `i` is the state after step `i`.
    Inclusive,
    /// Element `i` is the state before step `i`.
    Exclusive,
}

/// One affine coefficient: a constant, or element `i` of the next array input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Coefficient {
    Constant(u32),
    Input,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    Unit,
    Bool(bool),
    Nat(u32),
    Pair,
    Var(term::Binder),
    Lambda {
        binder: term::Binder,
    },
    Apply,
    Let {
        binder: term::Binder,
    },
    Select,
    Call {
        name: String,
    },
    Fail(&'static str),
    ArrayU32(Vec<u32>),
    /// A runtime `Array<u32>` bound to input slot `n` when an extracted plan executes.
    InputU32(u32),
    /// A runtime `u32` scalar bound to scalar slot `n` when an extracted plan executes.
    InputScalar(u32),
    /// Ordered fold of an array (input 1) from a seed (input 0) by a sealed law.
    Reduce {
        law: crate::laws::Law,
        rule: &'static str,
    },
    /// Ordered affine-pair fold: scalar seeds a/b, then coefficient arrays a/b.
    PairReduce {
        rule: &'static str,
    },
    /// Ordered states of `s = a_i * s + b_i` from a seed (input 0); array-valued
    /// coefficients take the following inputs, `a` before `b`.
    AffineScan {
        prefix: Prefix,
        a: Coefficient,
        b: Coefficient,
        rule: &'static str,
    },
    /// Ordered bounded-family scan: seed, coefficients, then step count.
    BoundedScan {
        family: BoundedFamily,
        prefix: Prefix,
        rule: &'static str,
    },
    Map(MapOp),
    FusedMap(Vec<MapOp>),
    ZipMap {
        op: ZipOp,
        len: usize,
    },
    /// A fused two-input map: `left`/`right` apply to each input element before
    /// `op` combines them, then `suffix` applies to the result.
    FusedZipMap {
        op: ZipOp,
        len: usize,
        left: Vec<MapOp>,
        right: Vec<MapOp>,
        suffix: Vec<MapOp>,
    },
}

impl Op {
    fn arity(&self) -> Option<usize> {
        match self {
            Self::Unit
            | Self::Bool(_)
            | Self::Nat(_)
            | Self::Var(_)
            | Self::Fail(_)
            | Self::ArrayU32(_)
            | Self::InputU32(_)
            | Self::InputScalar(_) => Some(0),
            Self::BoundedScan { .. } => Some(3),
            Self::Reduce { .. } => Some(2),
            Self::PairReduce { .. } => Some(4),
            Self::AffineScan { a, b, .. } => Some(
                1 + usize::from(*a == Coefficient::Input) + usize::from(*b == Coefficient::Input),
            ),
            Self::Lambda { .. } | Self::Map(_) | Self::FusedMap(_) => Some(1),
            Self::Pair
            | Self::Apply
            | Self::Let { .. }
            | Self::ZipMap { .. }
            | Self::FusedZipMap { .. } => Some(2),
            Self::Select => Some(3),
            Self::Call { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Node {
    pub source_id: u32,
    pub ty: Type,
    pub effect: Effect,
    /// Ordered input ports. Port 0 is demanded before port 1, except Select
    /// demands only the chosen branch and Lambda defers its body.
    pub inputs: Vec<NodeId>,
    pub op: Op,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub params: Vec<(term::Binder, Type)>,
    pub result: Type,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    pub signature: Signature,
    pub body: NodeId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub nodes: usize,
    pub ports: usize,
    pub depth: usize,
    pub array_len: usize,
    pub map_ops: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Schema,
    NodeLimit,
    PortLimit,
    DepthLimit,
    ArrayLimit,
    MapLimit,
    InvalidNode(NodeId),
    DuplicateSource(u32),
    SharedOrOrphan(NodeId),
    Cycle(NodeId),
    Arity(NodeId),
    Type(NodeId),
    Effect(NodeId),
    MissingBinder(NodeId, term::Binder),
    DuplicatedOwner(term::Binder),
    UnknownFunction(NodeId),
    StaleProof,
    RuleRejected,
}

#[derive(Clone, Debug)]
pub struct Net {
    pub schema_version: u32,
    pub nodes: Vec<Option<Node>>,
    pub entry: NodeId,
    pub functions: BTreeMap<String, Function>,
    pub revision: u64,
    pub rule_history: Vec<&'static str>,
    pub source_identity: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Proof {
    revision: u64,
    fingerprint: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rewrite {
    pub rule: &'static str,
    pub before: u64,
    pub after: u64,
}

fn digest<T: std::fmt::Debug>(value: &T) -> u64 {
    let mut hash = DefaultHasher::new();
    format!("{value:?}").hash(&mut hash);
    hash.finish()
}

fn add_counts(into: &mut BTreeMap<term::Binder, usize>, other: BTreeMap<term::Binder, usize>) {
    for (binder, count) in other {
        *into.entry(binder).or_default() += count;
    }
}

fn max_counts(into: &mut BTreeMap<term::Binder, usize>, other: BTreeMap<term::Binder, usize>) {
    for (binder, count) in other {
        let current = into.entry(binder).or_default();
        *current = (*current).max(count);
    }
}

impl Net {
    fn free_variables(
        &self,
        id: NodeId,
        bound: &mut BTreeSet<term::Binder>,
        free: &mut BTreeSet<term::Binder>,
    ) {
        let node = self.node(id).expect("verified node");
        match node.op {
            Op::Var(binder) => {
                if !bound.contains(&binder) {
                    free.insert(binder);
                }
            }
            Op::Lambda { binder } => {
                bound.insert(binder);
                self.free_variables(node.inputs[0], bound, free);
                bound.remove(&binder);
            }
            Op::Let { binder } => {
                self.free_variables(node.inputs[0], bound, free);
                bound.insert(binder);
                self.free_variables(node.inputs[1], bound, free);
                bound.remove(&binder);
            }
            _ => {
                for &child in &node.inputs {
                    self.free_variables(child, bound, free);
                }
            }
        }
    }

    pub fn empty(entry: NodeId) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            nodes: Vec::new(),
            entry,
            functions: BTreeMap::new(),
            revision: 0,
            rule_history: Vec::new(),
            source_identity: 0,
        }
    }

    pub fn push(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Some(node));
        id
    }

    pub fn seal_source(&mut self) {
        self.source_identity = digest(&(&self.nodes, self.entry, &self.functions));
    }

    fn node(&self, id: NodeId) -> Result<&Node, Error> {
        self.nodes
            .get(id)
            .and_then(Option::as_ref)
            .ok_or(Error::InvalidNode(id))
    }

    fn node_mut(&mut self, id: NodeId) -> Result<&mut Node, Error> {
        self.nodes
            .get_mut(id)
            .and_then(Option::as_mut)
            .ok_or(Error::InvalidNode(id))
    }

    fn fingerprint(&self) -> u64 {
        digest(&(
            self.schema_version,
            &self.nodes,
            self.entry,
            &self.functions,
            self.revision,
            &self.rule_history,
            self.source_identity,
        ))
    }
}

fn effect_for(net: &Net, op: &Op, inputs: &[NodeId]) -> Effect {
    match op {
        Op::Fail(_) | Op::BoundedScan { .. } => Effect::MayFail,
        Op::Call { .. } => Effect::Unknown,
        Op::Lambda { .. } => Effect::Total,
        Op::Apply => {
            let mut effect = inputs.iter().fold(Effect::Total, |e, id| {
                e.join(net.node(*id).expect("encoded child").effect)
            });
            effect = effect.join(match &net.node(inputs[0]).expect("encoded function").op {
                Op::Lambda { .. } => {
                    net.node(net.node(inputs[0]).expect("encoded function").inputs[0])
                        .expect("encoded body")
                        .effect
                }
                _ => Effect::Unknown,
            });
            effect
        }
        _ => inputs.iter().fold(Effect::Total, |e, id| {
            e.join(net.node(*id).expect("encoded child").effect)
        }),
    }
}
