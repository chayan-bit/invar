//! L9 path sums and L10 range sums: the call tree is enumerated node by node
//! with self calls as holes; L9 reduces weighted leaves and node constants mod
//! P in any order, L10 combines bottom-up. The first trap is the minimum
//! lexicographic key `(path, segment)`: segment `s` of a node runs after its
//! children `0 .. s` and before child `s`.

use std::collections::HashMap;

use super::{
    Machine, Mutation, Stop,
    interp::Frame,
    rep::Lin,
    value::{Sym, Val},
};
use crate::general_net::{
    normal_form::{Guard, IntTy, LawId, Recursion, Ring},
    regions::tree::Expr,
};

/// Call-tree nodes enumerated before the law gives up (a tool limit).
const NODE_LIMIT: usize = 1 << 20;

const HOLE_RING: Ring = Ring::Checked64(IntTy::I32);

pub(super) enum Outcome {
    Leaf(Val),
    Tail(Vec<Val>),
    /// L9: `(sum_j c_j hole_j + k) mod P`.
    Combine(Lin),
    /// L10: the combine expression, evaluated once the children are known.
    Deferred {
        vars: HashMap<String, Val>,
        expr: Expr,
    },
}

/// Path-mode state of one node's frame.
pub(super) struct Node {
    pub function: usize,
    name: String,
    recursion: Recursion,
    children: Vec<(String, Vec<Val>)>,
    outcome: Option<Outcome>,
}

fn self_call<'e>(node: &Node, expr: &'e Expr) -> Option<&'e [Expr]> {
    match expr {
        Expr::Call { name, args, .. } if *name == node.name => Some(args),
        _ => None,
    }
}

fn concrete_args(
    machine: &mut Machine<'_>,
    frame: &mut Frame,
    args: &[Expr],
) -> Result<Vec<Val>, Stop> {
    let values = args
        .iter()
        .map(|arg| machine.eval(frame, arg))
        .collect::<Result<Vec<_>, _>>()?;
    if values.iter().any(Val::is_sym) {
        return Err(Stop::violation("self-call argument reads a hole"));
    }
    Ok(values)
}

/// `let x = f(..)` in path mode: record the child, bind a hole.
pub(super) fn bind_child(
    machine: &mut Machine<'_>,
    frame: &mut Frame,
    name: &str,
    value: &Expr,
) -> Result<bool, Stop> {
    let Some(node) = frame.path.as_ref() else {
        return Ok(false);
    };
    let Some(args) = self_call(node, value) else {
        return Ok(false);
    };
    let args = concrete_args(machine, frame, args)?;
    let node = frame
        .path
        .as_mut()
        .ok_or_else(|| Stop::internal("path node"))?;
    let j = node.children.len();
    node.children.push((name.to_owned(), args));
    frame.vars.insert(
        name.to_owned(),
        Val::Sym(Box::new(Sym::Lin(Lin::unit(HOLE_RING, j + 1, j)))),
    );
    Ok(true)
}

/// `return` in path mode: a leaf, a tail call or a combine.
pub(super) fn ret(
    machine: &mut Machine<'_>,
    frame: &mut Frame,
    value: Option<&Expr>,
) -> Result<(), Stop> {
    let node = frame
        .path
        .as_ref()
        .ok_or_else(|| Stop::internal("path node"))?;
    let value = value.ok_or_else(|| Stop::internal("path-sum return without value"))?;
    let holes: Vec<String> = node.children.iter().map(|(n, _)| n.clone()).collect();
    let mentions = holes.iter().any(|h| value.mentions(h));
    let outcome = if let Some(args) = self_call(node, value) {
        Outcome::Tail(concrete_args(machine, frame, args)?)
    } else if !mentions {
        let v = machine.eval(frame, value)?;
        if v.is_sym() {
            return Err(Stop::violation("leaf reads a hole"));
        }
        Outcome::Leaf(v)
    } else {
        match (&node.recursion, value) {
            (Recursion::RangeSum, _) => Outcome::Deferred {
                vars: frame.vars.clone(),
                expr: value.clone(),
            },
            (
                Recursion::PathSum { modulus },
                Expr::Binary {
                    op: crate::front::BinaryOp::Rem,
                    left,
                    right,
                    ..
                },
            ) if right.literal() == Some(i64::from(*modulus)) => match machine.eval(frame, left)? {
                Val::Sym(sym) => match *sym {
                    Sym::Lin(lin) => Outcome::Combine(lin),
                    _ => return Err(Stop::violation("combine is not linear in the holes")),
                },
                _ => return Err(Stop::violation("combine lost its holes")),
            },
            _ => return Err(Stop::violation("combine without the certified modulus")),
        }
    };
    frame
        .path
        .as_mut()
        .ok_or_else(|| Stop::internal("path node"))?
        .outcome = Some(outcome);
    Err(Stop::Return(Val::Unit))
}

struct Visit {
    args: Vec<Val>,
    key: Vec<u32>,
    weight: u64,
    under: bool,
    parent: Option<(usize, usize)>,
}

struct Record {
    event: Option<(Vec<u32>, Stop)>,
    key: Vec<u32>,
    outcome: Option<Outcome>,
    children: Vec<usize>,
    names: Vec<String>,
}

fn run_node(
    machine: &mut Machine<'_>,
    index: usize,
    recursion: &Recursion,
    args: Vec<Val>,
) -> Result<(Option<Stop>, Node), Stop> {
    let tree = machine.function(index)?;
    let vars = tree
        .params
        .iter()
        .map(|p| p.name.clone())
        .zip(args)
        .collect();
    let mut frame = Frame::new(index, vars);
    frame.path = Some(Node {
        function: index,
        name: tree.name.clone(),
        recursion: recursion.clone(),
        children: Vec::new(),
        outcome: None,
    });
    let result = machine.block(&mut frame, &tree.body);
    let node = frame
        .path
        .take()
        .ok_or_else(|| Stop::internal("path node"))?;
    match result {
        Err(Stop::Return(_)) if node.outcome.is_some() => Ok((None, node)),
        Err(stop @ Stop::Trap(..)) => Ok((Some(stop), node)),
        Err(Stop::Return(_)) | Ok(()) => Err(Stop::internal("path node without outcome")),
        Err(other) => Err(other),
    }
}

fn key_of(prefix: &[u32], digit: usize) -> Result<Vec<u32>, Stop> {
    let mut key = prefix.to_vec();
    key.push(u32::try_from(digit).map_err(|_| Stop::tool("call-tree fan-out"))?);
    Ok(key)
}

pub(super) fn call(machine: &mut Machine<'_>, index: usize, args: Vec<Val>) -> Result<Val, Stop> {
    let form = machine
        .form
        .functions
        .get(index)
        .ok_or_else(|| Stop::internal("function form"))?;
    let recursion = form
        .recursion
        .clone()
        .ok_or_else(|| Stop::internal("recursion law"))?;
    let laws: Vec<LawId> = form.certificates.iter().map(|c| c.law).collect();
    match tree_law(machine, index, &recursion, args.clone()) {
        Ok(value) => {
            machine.receipt.applied(laws);
            Ok(value)
        }
        Err(stop) if stop.is_event() || matches!(stop, Stop::Tool(..)) => Err(stop),
        Err(cause) => {
            machine.receipt.fallback(laws, &cause);
            machine.serial.push(index);
            let result = machine.call_serial(index, args);
            machine.serial.pop();
            result
        }
    }
}

fn tree_law(
    machine: &mut Machine<'_>,
    index: usize,
    recursion: &Recursion,
    args: Vec<Val>,
) -> Result<Val, Stop> {
    let modulus = match recursion {
        Recursion::PathSum { modulus } => Some(u64::from(*modulus)),
        Recursion::RangeSum => None,
    };
    let checked = machine.guards_on();
    let wrong = machine.mutation == Mutation::WrongCombine;
    let mut records: Vec<Record> = Vec::new();
    let mut stack = vec![Visit {
        args,
        key: Vec::new(),
        weight: 1,
        under: false,
        parent: None,
    }];
    let mut sum: u64 = 0;
    let mut raw = None;
    while let Some(visit) = stack.pop() {
        if records.len() >= NODE_LIMIT {
            return Err(Stop::tool("call-tree node limit"));
        }
        let (event, node) = run_node(machine, index, recursion, visit.args)?;
        let id = records.len();
        if let Some((parent, slot)) = visit.parent
            && let Some(child) = records
                .get_mut(parent)
                .and_then(|r| r.children.get_mut(slot))
        {
            *child = id;
        }
        let m = node.children.len();
        let event = match event {
            Some(stop) => Some((key_of(&visit.key, 2 * m)?, stop)),
            None => None,
        };
        let mut kids: Vec<(Vec<Val>, usize, u64, bool)> = Vec::new();
        for (j, (_, child)) in node.children.iter().enumerate() {
            kids.push((child.clone(), 2 * j + 1, 0, true));
        }
        if event.is_none() {
            match (&node.outcome, modulus) {
                (Some(Outcome::Leaf(v)), Some(p)) => {
                    let x = v
                        .int()
                        .ok_or_else(|| Stop::violation("path-sum leaf is not an integer"))?;
                    if visit.under {
                        if checked && !(0..i128::from(p)).contains(&x) {
                            return Err(Stop::Guard(Guard::TableRange {
                                modulus: u32::try_from(p).map_err(|_| Stop::internal("modulus"))?,
                            }));
                        }
                        sum = (sum + visit.weight * (x.rem_euclid(i128::from(p)) as u64)) % p;
                    } else {
                        raw = Some(v.clone());
                    }
                }
                (Some(Outcome::Combine(lin)), Some(p)) => {
                    if lin.coeffs.len() > m {
                        return Err(Stop::violation("combine reads an unbound hole"));
                    }
                    let k = i128::from(lin.constant as i64);
                    let coefficients: Vec<i128> = (0..m)
                        .map(|j| i128::from(lin.coefficient(j) as i64))
                        .collect();
                    if checked {
                        if k < 0 || coefficients.iter().any(|c| *c < 0) {
                            return Err(Stop::Guard(Guard::NonNegativeConstants));
                        }
                        let bound = coefficients
                            .iter()
                            .map(|c| c * (i128::from(p) - 1))
                            .sum::<i128>()
                            + k;
                        if bound > i128::from(i32::MAX) {
                            return Err(Stop::Guard(Guard::NoOverflowCombine));
                        }
                    }
                    let residue = |x: i128| x.rem_euclid(i128::from(p)) as u64;
                    sum = (sum + visit.weight * residue(k)) % p;
                    for (kid, c) in kids.iter_mut().zip(&coefficients) {
                        kid.2 = if wrong {
                            visit.weight
                        } else {
                            visit.weight * residue(*c) % p
                        };
                    }
                }
                (Some(Outcome::Tail(child)), _) => {
                    kids.push((child.clone(), 2 * m + 1, visit.weight, visit.under))
                }
                (Some(Outcome::Leaf(_) | Outcome::Deferred { .. }), None) => {}
                _ => return Err(Stop::violation("node outcome does not match its law")),
            }
        }
        records.push(Record {
            event,
            key: visit.key.clone(),
            outcome: node.outcome,
            children: vec![usize::MAX; kids.len()],
            names: node.children.into_iter().map(|(n, _)| n).collect(),
        });
        for (slot, (args, digit, weight, under)) in kids.into_iter().enumerate().rev() {
            stack.push(Visit {
                args,
                key: key_of(&visit.key, digit)?,
                weight,
                under,
                parent: Some((id, slot)),
            });
        }
    }
    machine.receipt.lanes += records.len() as u64;
    let root = match modulus {
        None => range_values(machine, index, &mut records, wrong)?,
        Some(_) => None,
    };
    let events = records.iter().filter_map(|r| r.event.as_ref());
    let chosen = if machine.mutation == Mutation::LastTrap {
        events.max_by(|a, b| a.0.cmp(&b.0))
    } else {
        events.min_by(|a, b| a.0.cmp(&b.0))
    };
    if let Some((_, stop)) = chosen {
        return Err(stop.clone());
    }
    match modulus {
        Some(_) => match raw {
            Some(v) => Ok(v),
            None => Ok(Val::I32(
                i32::try_from(sum).map_err(|_| Stop::internal("path sum"))?,
            )),
        },
        None => root.ok_or_else(|| Stop::internal("range-sum root value")),
    }
}

/// L10: node values bottom-up (children have larger ids than parents); a
/// combine trap becomes the node's event at its final segment.
fn range_values(
    machine: &mut Machine<'_>,
    index: usize,
    records: &mut [Record],
    wrong: bool,
) -> Result<Option<Val>, Stop> {
    let mut values: Vec<Option<Val>> = vec![None; records.len()];
    for id in (0..records.len()).rev() {
        let Some(record) = records.get(id) else {
            continue;
        };
        if record.event.is_some() {
            continue;
        }
        let child = |c: &usize| values.get(*c).cloned().flatten();
        let mut trap = None;
        let value = match &record.outcome {
            Some(Outcome::Leaf(v)) => Some(v.clone()),
            Some(Outcome::Tail(_)) => record.children.last().and_then(child),
            Some(Outcome::Deferred { vars, expr }) => {
                let mut children: Vec<Option<Val>> = record.children.iter().map(child).collect();
                if wrong {
                    // Wrong combine: every hole reads the last child.
                    let last = children.last().cloned().flatten();
                    children.iter_mut().for_each(|c| *c = last.clone());
                }
                if children.iter().any(Option::is_none) {
                    None
                } else {
                    let mut frame = Frame::new(index, vars.clone());
                    for (name, value) in record.names.iter().zip(children.into_iter().flatten()) {
                        frame.vars.insert(name.clone(), value);
                    }
                    match machine.eval(&mut frame, expr) {
                        Ok(v) => Some(v),
                        Err(stop @ Stop::Trap(..)) => {
                            trap = Some((key_of(&record.key, 2 * record.names.len())?, stop));
                            None
                        }
                        Err(other) => return Err(other),
                    }
                }
            }
            _ => return Err(Stop::violation("range-sum node outcome")),
        };
        if let Some(record) = records.get_mut(id) {
            record.event = trap;
        }
        if let Some(slot) = values.get_mut(id) {
            *slot = value;
        }
    }
    Ok(values.first().cloned().flatten())
}
