//! Checked, conservative same-block independent-child extraction. The emitted
//! runtime consumes these source-derived certificates, never fixture identities.
use super::{Error, scalar};
use crate::mir;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CallSite {
    pub function: usize,
    pub block: usize,
    pub position: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SerializationReason {
    AffineArgument,
    OwnerResult,
    DependentArgument,
    OrderedPrelude,
    ProjectionLimit,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub source: CallSite,
    pub target: CallSite,
    pub callee: usize,
    /// Instruction positions to evaluate in a private scalar projection.
    pub projection: Vec<usize>,
    pub scalar_values: Vec<mir::ValueId>,
    pub scalar_places: Vec<String>,
    /// Local ownership is isolated but requires source-turn replay/accounting.
    pub replay_owners: bool,
}

#[derive(Clone, Debug)]
pub struct Serialization {
    pub source: CallSite,
    pub target: CallSite,
    pub reason: SerializationReason,
}

#[derive(Clone, Debug, Default)]
pub struct ForkPlan {
    pub candidates: Vec<Candidate>,
    pub serialized: Vec<Serialization>,
}

fn scalar_signature(function: &mir::Function) -> Result<(), SerializationReason> {
    if function.params.iter().any(|p| !scalar(&p.ty)) {
        return Err(SerializationReason::AffineArgument);
    }
    if !scalar(&function.result) {
        return Err(SerializationReason::OwnerResult);
    }
    Ok(())
}

fn has_owners(program: &mir::Program) -> Result<Vec<bool>, Error> {
    let mut owners = Vec::new();
    owners
        .try_reserve(program.functions.len())
        .map_err(|_| Error::Limit)?;
    owners.extend(program.functions.iter().map(|f| {
        !scalar(&f.result)
            || f.params.iter().any(|p| !scalar(&p.ty))
            || f.locals.values().any(|p| !scalar(&p.ty))
    }));
    // Least fixed point over call edges, including recursive SCCs.
    loop {
        let mut changed = false;
        for (f, function) in program.functions.iter().enumerate() {
            if owners[f] {
                continue;
            }
            let reachable = function
                .blocks
                .iter()
                .flat_map(|b| &b.instructions)
                .any(|i| {
                    if let mir::Operation::Call { name, .. } = &i.kind {
                        program
                            .functions
                            .iter()
                            .position(|f| f.name == *name)
                            .is_some_and(|t| owners[t])
                    } else {
                        false
                    }
                });
            if reachable {
                owners[f] = true;
                changed = true;
            }
        }
        if !changed {
            return Ok(owners);
        }
    }
}

fn operands(op: &mir::Operation) -> Vec<mir::ValueId> {
    super::coverage::operands(op)
}

/// Input must be the current verified MIR from the checked compiler boundary.
/// The scan is bounded by admitted instruction/state limits and `lookahead`.
pub(super) fn analyze(program: &mir::Program, lookahead: usize) -> Result<ForkPlan, Error> {
    if lookahead == 0 || lookahead > 256 {
        return Err(Error::Limit);
    }
    let owners = has_owners(program)?;
    let mut plan = ForkPlan::default();
    plan.candidates.try_reserve(256).map_err(|_| Error::Limit)?;
    plan.serialized
        .try_reserve(4096)
        .map_err(|_| Error::Limit)?;
    for (f, function) in program.functions.iter().enumerate() {
        let types: std::collections::BTreeMap<_, _> = function
            .blocks
            .iter()
            .flat_map(|b| {
                b.params
                    .iter()
                    .chain(b.instructions.iter().filter_map(|i| i.output.as_ref()))
            })
            .map(|(id, ty)| (*id, ty))
            .collect();
        for (b, block) in function.blocks.iter().enumerate() {
            for (first, instruction) in block.instructions.iter().enumerate() {
                let mir::Operation::Call { name, .. } = &instruction.kind else {
                    continue;
                };
                let Some(first_callee) = program.functions.iter().position(|f| f.name == *name)
                else {
                    continue;
                };
                let source = CallSite {
                    function: f,
                    block: b,
                    position: first,
                };
                let first_reason = scalar_signature(&program.functions[first_callee]).err();
                let mut unavailable = BTreeSet::new();
                if let Some((id, _)) = &instruction.output {
                    unavailable.insert(*id);
                }
                let mut unknown_places = BTreeSet::new();
                let mut projection = Vec::new();
                projection
                    .try_reserve(lookahead)
                    .map_err(|_| Error::Limit)?;
                let mut ordered = false;
                for (pos, i) in block
                    .instructions
                    .iter()
                    .enumerate()
                    .skip(first + 1)
                    .take(lookahead)
                {
                    use mir::Operation as O;
                    if let O::Call { name, args } = &i.kind
                        && let Some(callee) = program.functions.iter().position(|f| f.name == *name)
                    {
                        let target = CallSite {
                            function: f,
                            block: b,
                            position: pos,
                        };
                        let reason = first_reason
                            .or_else(|| scalar_signature(&program.functions[callee]).err())
                            .or(if ordered {
                                Some(SerializationReason::OrderedPrelude)
                            } else {
                                None
                            })
                            .or_else(|| {
                                args.iter()
                                    .any(|id| unavailable.contains(id))
                                    .then_some(SerializationReason::DependentArgument)
                            })
                            .or_else(|| {
                                (plan.candidates.len() >= 256)
                                    .then_some(SerializationReason::ProjectionLimit)
                            });
                        if let Some(reason) = reason {
                            if plan.serialized.len() < 4096 {
                                plan.serialized.push(Serialization {
                                    source,
                                    target,
                                    reason,
                                });
                            }
                        } else {
                            // Capture only scalar-typed bytes. Affine handles in
                            // the parent's value/place store are never copied.
                            let produced: BTreeSet<_> = projection
                                .iter()
                                .filter_map(|&pos: &usize| {
                                    block.instructions[pos].output.as_ref().map(|(id, _)| *id)
                                })
                                .collect();
                            let needed: BTreeSet<_> = projection
                                .iter()
                                .flat_map(|&pos| operands(&block.instructions[pos].kind))
                                .chain(args.iter().copied())
                                .filter(|id| !produced.contains(id))
                                .collect();
                            if needed
                                .iter()
                                .any(|id| !types.get(id).is_some_and(|ty| scalar(ty)))
                            {
                                return Err(Error::State);
                            }
                            let places: BTreeSet<_> = projection
                                .iter()
                                .filter_map(|&pos| match &block.instructions[pos].kind {
                                    O::Read(name) => Some(name.clone()),
                                    _ => None,
                                })
                                .collect();
                            plan.candidates.push(Candidate {
                                source,
                                target,
                                callee,
                                projection: projection.clone(),
                                scalar_values: needed.into_iter().collect(),
                                scalar_places: places.into_iter().collect(),
                                replay_owners: owners[callee],
                            });
                        }
                        if let Some((id, _)) = &i.output {
                            unavailable.insert(*id);
                        }
                        continue;
                    }
                    let available = operands(&i.kind).iter().all(|id| !unavailable.contains(id));
                    let safe = match &i.kind {
                        O::Tick
                        | O::I32(_)
                        | O::U32(_)
                        | O::Bool(_)
                        | O::Unary { .. }
                        | O::Binary { .. } => true,
                        O::Read(name) => {
                            scalar(&function.locals[name].ty) && !unknown_places.contains(name)
                        }
                        O::Bind { name, .. } | O::Store { name, .. } => {
                            scalar(&function.locals[name].ty)
                        }
                        O::Call { name, .. } => {
                            matches!(name.as_str(), "wrapping_add" | "wrapping_mul")
                        }
                        // Cleanup can be observable. It never executes in the
                        // projection; no candidate crosses this ordered barrier.
                        _ => false,
                    };
                    if let O::Bind { name, .. } | O::Store { name, .. } = &i.kind {
                        if available && safe {
                            unknown_places.remove(name);
                        } else {
                            unknown_places.insert(name.clone());
                        }
                    }
                    if !safe && !matches!(i.kind, O::Read(_)) {
                        ordered = true;
                    }
                    if available && safe {
                        if !matches!(i.kind, O::Tick) {
                            projection.push(pos);
                        }
                    } else if let Some((id, _)) = &i.output {
                        unavailable.insert(*id);
                    }
                }
            }
        }
    }
    Ok(plan)
}
