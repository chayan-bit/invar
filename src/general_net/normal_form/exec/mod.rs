//! CPU reference semantics of the parallel normal form (normal-form design sections 3-5).
//!
//! [`run`] interprets the net-derived statement trees, but every region the
//! normal form certifies is executed through its law, not in source order:
//!
//! * a parallel loop is split into lanes (one per iteration). Each carried
//!   group is computed as a lane element in its representation by symbolic
//!   lane execution, the elements are combined by a balanced-tree scan, and
//!   every lane is then replayed concretely from its scanned prefix. Arrays are
//!   read through the law's model (own overlay, per-target combine prefix,
//!   monotone reconstruction from round-start state, fused lane overlay). The
//!   first event (trap or exit) is the minimum lane index (L15), and writes
//!   merge by last writer in lane order;
//! * a path-sum function (L9) enumerates its call tree with self calls as
//!   holes and reduces weighted leaves mod P; a range sum (L10) combines
//!   bottom-up; trap keys are lexicographic `(path, segment)`;
//! * independent sibling calls (L20) are evaluated speculatively in reverse and
//!   committed in source order.
//!
//! Runtime guards are checked before anything is published; a false guard
//! (or a law-state violation) falls back to source-order interpretation of the
//! region. The [`Receipt`] counts law executions, guard fallbacks and
//! violations. [`Mutation`] injects the soundness bugs the differential tests
//! must catch.

mod interp;
mod lanes;
mod paths;
mod rep;
mod sym;
mod table;
mod value;

use std::{
    collections::{BTreeMap, HashMap},
    rc::Rc,
};

use super::{Guard, LawId, NormalForm};
use crate::{
    eval::{self, EvalError, TrapCategory},
    front::{Mode, Span},
    general_net::{
        Net, Proof,
        regions::{self, tree::FunctionTree},
    },
};
use value::Val;

/// Soundness bugs injected on purpose; each must be caught by the tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutation {
    None,
    /// Report the last event (trap or exit) instead of the first.
    LastTrap,
    /// Skip every runtime guard.
    DropGuards,
    /// Combine with the wrong operator (keep only the right operand), merge
    /// ordered writes first-writer-wins, ignore own and fused overlays, ignore
    /// path weights.
    WrongCombine,
}

/// What the executor did, per law.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Receipt {
    /// Regions executed through their law.
    pub applied: BTreeMap<LawId, u64>,
    /// Regions that fell back to source order because a runtime guard failed.
    pub guard_fallbacks: BTreeMap<LawId, u64>,
    /// Regions that fell back because the law state could not be formed
    /// (never expected for certified regions; reported, not hidden).
    pub violations: Vec<String>,
    /// Lanes (iterations or call-tree nodes) executed through laws.
    pub lanes: u64,
    /// Statement ticks of the whole run: one per executed statement and loop
    /// iteration, symbolic lane passes included (not source fuel).
    pub steps: u64,
    /// Deepest call nesting reached by a call frame (`main` is depth 1).
    pub max_depth: usize,
    /// Array elements allocated by source allocations over the whole run,
    /// symbolic lane passes included; `u64::MAX` once a length was symbolic.
    /// Never below the reference evaluator's live element count.
    pub allocated: u64,
}

impl Receipt {
    fn applied(&mut self, laws: impl IntoIterator<Item = LawId>) {
        for law in laws {
            *self.applied.entry(law).or_default() += 1;
        }
    }

    fn fallback(&mut self, laws: impl IntoIterator<Item = LawId>, cause: &Stop) {
        match cause {
            Stop::Guard(_) => {
                for law in laws {
                    *self.guard_fallbacks.entry(law).or_default() += 1;
                }
            }
            Stop::Violation(reason) => self.violations.push(reason.clone()),
            Stop::Poison => self
                .violations
                .push("first event is a poisoned lane".to_owned()),
            _ => {}
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Outcome {
    /// Observation contract v2: the value, or the first source-order trap.
    pub result: Result<eval::Value, EvalError>,
    pub receipt: Receipt,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Regions(regions::Error),
    /// The normal form was derived from a different net.
    Fingerprint,
    UnknownEntry,
}

/// Internal control: events, fallbacks and tool failures.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Stop {
    Trap(TrapCategory, Span),
    Return(Val),
    Guard(Guard),
    Violation(String),
    /// A poisoned (unrepresentable) prefix was used.
    Poison,
    Tool(&'static str, String),
}

impl Stop {
    pub(super) fn internal(message: &str) -> Self {
        Self::Tool("MIR_INVALID", message.to_owned())
    }

    pub(super) fn unsupported(message: &str) -> Self {
        Self::Tool("UNSUPPORTED_FEATURE", message.to_owned())
    }

    pub(super) fn tool(message: &str) -> Self {
        Self::Tool("RESOURCE_LIMIT", message.to_owned())
    }

    pub(super) fn violation(message: &str) -> Self {
        Self::Violation(message.to_owned())
    }

    /// A lane event (trap or exit) rather than a fallback or tool failure.
    pub(super) fn is_event(&self) -> bool {
        matches!(self, Self::Trap(..) | Self::Return(_))
    }
}

/// Statement budget of one run (a tool limit, like the oracle's fuel).
const STEP_LIMIT: u64 = 50_000_000;
/// Call depth limit (a tool limit).
const DEPTH_LIMIT: usize = 400;

pub(super) struct Machine<'a> {
    form: &'a NormalForm,
    trees: Vec<Option<Rc<FunctionTree>>>,
    names: HashMap<String, usize>,
    receipt: Receipt,
    mutation: Mutation,
    steps: u64,
    /// Statement budget of this run, at most [`STEP_LIMIT`].
    step_limit: u64,
    depth: usize,
    /// Functions whose law fell back: their nested self calls run in source order.
    serial: Vec<usize>,
}

impl Machine<'_> {
    fn tick(&mut self) -> Result<(), Stop> {
        self.steps += 1;
        if self.steps > self.step_limit {
            return Err(Stop::tool("step limit"));
        }
        Ok(())
    }

    /// Charge one tick per lane of a law loop before its lanes exist, as the
    /// serial loop ticks once per iteration.
    fn charge(&mut self, lanes: u64) -> Result<(), Stop> {
        self.steps = self.steps.saturating_add(lanes);
        if self.steps > self.step_limit {
            return Err(Stop::tool("step limit"));
        }
        Ok(())
    }

    /// Account for a source allocation of `elements` (`None`: a symbolic
    /// length) before it exists, so the run's memory stays bounded.
    fn allocate(&mut self, elements: Option<u64>) -> Result<(), Stop> {
        self.receipt.allocated = match elements {
            Some(n) => self.receipt.allocated.saturating_add(n),
            None => u64::MAX,
        };
        if elements.is_some()
            && self.receipt.allocated > crate::array_limits::MAX_LIVE_ELEMENTS as u64
        {
            return Err(Stop::tool("live array element limit exceeded"));
        }
        Ok(())
    }

    fn guards_on(&self) -> bool {
        self.mutation != Mutation::DropGuards
    }

    fn function(&self, index: usize) -> Result<Rc<FunctionTree>, Stop> {
        self.trees
            .get(index)
            .and_then(Option::clone)
            .ok_or_else(|| Stop::unsupported("call to an unstructured function"))
    }
}

/// Run `entry` on `args` through the normal form `form` of `net`.
pub fn run(
    net: &Net,
    proof: &Proof,
    form: &NormalForm,
    entry: &str,
    args: &[eval::Value],
    mutation: Mutation,
) -> Result<Outcome, Error> {
    run_bounded(net, proof, form, entry, args, mutation, STEP_LIMIT)
}

/// [`run`] with a smaller statement budget: past `step_limit` ticks the run
/// ends with the `RESOURCE_LIMIT` tool failure.
pub fn run_bounded(
    net: &Net,
    proof: &Proof,
    form: &NormalForm,
    entry: &str,
    args: &[eval::Value],
    mutation: Mutation,
    step_limit: u64,
) -> Result<Outcome, Error> {
    let partition = regions::partition(net, proof).map_err(Error::Regions)?;
    if partition.fingerprint != form.fingerprint
        || partition.functions.len() != form.functions.len()
    {
        return Err(Error::Fingerprint);
    }
    let names: HashMap<String, usize> = partition
        .functions
        .iter()
        .map(|function| (function.name.clone(), function.index))
        .collect();
    let index = *names.get(entry).ok_or(Error::UnknownEntry)?;
    let trees = partition
        .functions
        .into_iter()
        .map(|function| function.tree.ok().map(Rc::new))
        .collect();
    let mut machine = Machine {
        form,
        trees,
        names,
        receipt: Receipt::default(),
        mutation,
        steps: 0,
        step_limit: step_limit.min(STEP_LIMIT),
        depth: 0,
        serial: Vec::new(),
    };
    let result = invoke(&mut machine, index, args);
    machine.receipt.steps = machine.steps;
    Ok(Outcome {
        result,
        receipt: machine.receipt,
    })
}

fn invoke(
    machine: &mut Machine<'_>,
    index: usize,
    args: &[eval::Value],
) -> Result<eval::Value, EvalError> {
    let to_error = |stop: Stop| match stop {
        Stop::Trap(category, span) => EvalError::Trap { category, span },
        Stop::Tool(code, message) => EvalError::Tool { code, message },
        other => EvalError::Tool {
            code: "MIR_INVALID",
            message: format!("unresolved executor state {other:?}"),
        },
    };
    let values = args
        .iter()
        .map(Val::from_eval)
        .collect::<Result<Vec<_>, _>>()
        .map_err(to_error)?;
    let modes: Vec<Option<Mode>> = machine
        .function(index)
        .map_err(to_error)?
        .params
        .iter()
        .map(|param| param.mode)
        .collect();
    let result = machine.call(index, values.clone()).map_err(to_error)?;
    let result = result.to_eval().map_err(to_error)?;
    for ((arg, value), mode) in args.iter().zip(&values).zip(modes) {
        if mode != Some(Mode::Mut) {
            continue;
        }
        commit(arg, value).map_err(to_error)?;
    }
    Ok(result)
}

/// Commit a `mut` array argument back to the caller's value (success only).
fn commit(to: &eval::Value, from: &Val) -> Result<(), Stop> {
    let from = from.to_eval()?;
    match (to, from) {
        (eval::Value::ArrayI32(a), eval::Value::ArrayI32(b)) => {
            *a.borrow_mut() = b.borrow().clone()
        }
        (eval::Value::ArrayU32(a), eval::Value::ArrayU32(b)) => {
            *a.borrow_mut() = b.borrow().clone()
        }
        (eval::Value::ArrayF32(a), eval::Value::ArrayF32(b)) => {
            *a.borrow_mut() = b.borrow().clone()
        }
        _ => return Err(Stop::internal("mut argument kind")),
    }
    Ok(())
}
