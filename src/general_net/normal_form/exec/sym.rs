//! Symbolic lane execution (SYMPLE-style): the current carried group holds a
//! value of its law representation; everything that cannot be expressed in
//! the representation becomes poison, and poison that would decide the
//! group's own state selects the fallback.

use std::collections::BTreeSet;

use super::{
    Machine, Stop,
    interp::{Frame, poison, truth},
    lanes::Group,
    rep::{self, Lin},
    value::{Sym, Val, array_id},
};
use crate::{
    front::{BinaryOp, UnaryOp},
    general_net::{
        normal_form::{Guard, Order, Rep, Ring},
        regions::tree::{self, Expr, Stmt},
    },
};

/// The ring of a linear representation.
pub(super) fn ring_of(rep: &Rep) -> Option<Ring> {
    Some(match rep {
        Rep::Sealed(_) => Ring::Wrapping32,
        Rep::LiftedSum(ty) | Rep::SaturatedProduct(ty) | Rep::Counter(ty) => Ring::Checked64(*ty),
        Rep::Affine(ring) | Rep::Matrix { ring, .. } => *ring,
        _ => return None,
    })
}

/// The symbolic start value of group member `j`.
pub(super) fn start(group: &Group, j: usize) -> Val {
    let sym = match (&group.rep, ring_of(&group.rep)) {
        (_, Some(ring)) => Sym::Lin(Lin::unit(ring, group.vars.len(), j)),
        (Rep::Tropical { .. }, _) => Sym::Trop {
            a: Some(0),
            b: None,
        },
        (Rep::Select { .. }, _) => Sym::Select,
        _ => Sym::Last(None),
    };
    Val::Sym(Box::new(sym))
}

fn guard_for(group: &Group) -> Guard {
    Guard::NonNegativeCoefficients(group.vars.first().cloned().unwrap_or_default())
}

/// Assign `value` to `name`, folding it into the group state when `name` is
/// a member of the current group.
pub(super) fn assign(
    machine: &Machine<'_>,
    frame: &mut Frame,
    name: &str,
    value: Val,
) -> Result<(), Stop> {
    let checked = machine.guards_on();
    let value = match frame.lane.as_deref_mut() {
        Some(lane) if lane.phase == super::lanes::Phase::Symbolic && !lane.capturing => {
            match &lane.group {
                Some(group) if group.vars.iter().any(|v| v == name) => {
                    let folded = fold(group, value, checked)?;
                    if let (Rep::LiftedSum(_) | Rep::Counter(_), Some(Sym::Lin(lin))) =
                        (&group.rep, folded.sym())
                        && lin.coefficient(0) == 1
                    {
                        let partial = i128::from(lin.constant as i64);
                        lane.range = (lane.range.0.min(partial), lane.range.1.max(partial));
                    }
                    folded
                }
                Some(group) if group.companions.iter().any(|c| c == name) => {
                    return Err(Stop::violation("companion assigned outside its select"));
                }
                _ => value,
            }
        }
        _ => value,
    };
    frame.vars.insert(name.to_owned(), value);
    Ok(())
}

fn fold(group: &Group, value: Val, checked: bool) -> Result<Val, Stop> {
    if let Some(ring) = ring_of(&group.rep) {
        return match value {
            Val::Sym(sym) => match *sym {
                Sym::Lin(lin) => Ok(Val::Sym(Box::new(Sym::Lin(lin)))),
                _ => Err(Stop::violation("linear law state became non-linear")),
            },
            concrete => {
                let word = rep::embed(ring, &concrete, Some(guard_for(group)), checked)?;
                Ok(Val::Sym(Box::new(Sym::Lin(Lin::constant(
                    ring,
                    group.vars.len(),
                    word,
                )))))
            }
        };
    }
    match (&group.rep, value) {
        (Rep::Tropical { .. }, Val::Sym(sym)) if matches!(*sym, Sym::Trop { .. }) => {
            Ok(Val::Sym(sym))
        }
        (Rep::Tropical { .. }, concrete) if !concrete.is_sym() => {
            let b = concrete
                .int()
                .ok_or_else(|| Stop::violation("tropical reset"))?;
            Ok(Val::Sym(Box::new(Sym::Trop {
                a: None,
                b: Some(b),
            })))
        }
        (Rep::Last, concrete) if !concrete.is_sym() => {
            Ok(Val::Sym(Box::new(Sym::Last(Some(Box::new(concrete))))))
        }
        _ => Err(Stop::violation("group member assigned outside its law")),
    }
}

/// Symbolic binary operation (one operand is symbolic).
pub(super) fn binary(
    machine: &Machine<'_>,
    frame: &Frame,
    op: BinaryOp,
    left: &Val,
    right: &Val,
) -> Result<Val, Stop> {
    poison(frame)?;
    let checked = machine.guards_on();
    let group_guard = frame
        .lane
        .as_ref()
        .and_then(|lane| lane.group.as_ref())
        .map(guard_for);
    let sym = |v: &Val| v.sym().cloned();
    let embed = |ring: Ring, v: &Val| rep::embed(ring, v, group_guard.clone(), checked);
    let lin_val = |l: Lin| Ok(Val::Sym(Box::new(Sym::Lin(l))));
    match (sym(left), sym(right)) {
        (Some(Sym::Poison), _) | (_, Some(Sym::Poison)) => Ok(Val::poison()),
        (Some(Sym::Lin(a)), Some(Sym::Lin(b))) => match op {
            BinaryOp::Add => lin_val(a.add(&b)),
            BinaryOp::Sub => match (a.ring, b.neg()) {
                (Ring::Residue { .. }, _) | (_, None) => {
                    residue_sub(group_guard.clone(), checked, &a, &b)
                }
                (_, Some(nb)) => lin_val(a.add(&nb)),
            },
            BinaryOp::Mul if a.is_constant() => lin_val(b.scale(a.constant)),
            BinaryOp::Mul if b.is_constant() => lin_val(a.scale(b.constant)),
            _ => Ok(Val::poison()),
        },
        (Some(Sym::Lin(a)), None) => {
            let ring = a.ring;
            match op {
                BinaryOp::Add => lin_val(a.add(&Lin::constant(ring, 0, embed(ring, right)?))),
                BinaryOp::Sub => {
                    let c = Lin::constant(ring, 0, embed(ring, right)?);
                    match c.neg() {
                        Some(nc) => lin_val(a.add(&nc)),
                        None => residue_sub(group_guard.clone(), checked, &a, &c),
                    }
                }
                BinaryOp::Mul => lin_val(a.scale(embed(ring, right)?)),
                BinaryOp::Rem => match (ring, right.int()) {
                    (Ring::Residue { modulus, .. }, Some(p)) if p == i128::from(modulus) => {
                        lin_val(a)
                    }
                    _ => Ok(Val::poison()),
                },
                _ => Ok(Val::poison()),
            }
        }
        (None, Some(Sym::Lin(b))) => {
            let ring = b.ring;
            match op {
                BinaryOp::Add => lin_val(b.add(&Lin::constant(ring, 0, embed(ring, left)?))),
                BinaryOp::Sub => match b.neg() {
                    Some(nb) => lin_val(nb.add(&Lin::constant(ring, 0, embed(ring, left)?))),
                    None => residue_sub(
                        group_guard.clone(),
                        checked,
                        &Lin::constant(ring, 0, embed(ring, left)?),
                        &b,
                    ),
                },
                BinaryOp::Mul => lin_val(b.scale(embed(ring, left)?)),
                _ => Ok(Val::poison()),
            }
        }
        (Some(Sym::Trop { a, b }), None) | (None, Some(Sym::Trop { a, b })) => {
            let (c, trop_left) = if left.is_sym() {
                (right, true)
            } else {
                (left, false)
            };
            let Some(c) = c.int() else {
                return Ok(Val::poison());
            };
            let shift = match (op, trop_left) {
                (BinaryOp::Add, _) => c,
                (BinaryOp::Sub, true) => -c,
                _ => return Ok(Val::poison()),
            };
            Ok(Val::Sym(Box::new(Sym::Trop {
                a: a.map(|a| a + shift),
                b: b.map(|b| b + shift),
            })))
        }
        _ => Ok(Val::poison()),
    }
}

/// Subtraction under a modulus breaks Lemma B: the coefficient guard fails
/// (with guards dropped, the residue difference is used instead).
fn residue_sub(guard: Option<Guard>, checked: bool, a: &Lin, b: &Lin) -> Result<Val, Stop> {
    if checked {
        return Err(Stop::Guard(guard.unwrap_or(Guard::NonNegativeConstants)));
    }
    let Ring::Residue { modulus, .. } = a.ring else {
        return Err(Stop::internal("residue subtraction"));
    };
    let negated = b.scale(u64::from(modulus) - 1);
    Ok(Val::Sym(Box::new(Sym::Lin(a.add(&negated)))))
}

pub(super) fn unary(
    machine: &Machine<'_>,
    frame: &Frame,
    op: UnaryOp,
    value: &Val,
) -> Result<Val, Stop> {
    poison(frame)?;
    match (op, value.sym()) {
        (UnaryOp::Neg, Some(Sym::Lin(lin))) => match lin.neg() {
            Some(neg) => Ok(Val::Sym(Box::new(Sym::Lin(neg)))),
            None => residue_sub(
                frame
                    .lane
                    .as_ref()
                    .and_then(|l| l.group.as_ref())
                    .map(guard_for),
                machine.guards_on(),
                &Lin::constant(lin.ring, 0, 0),
                lin,
            ),
        },
        _ => Ok(Val::poison()),
    }
}

/// A call with a symbolic argument: the sealed wrapping intrinsics act on a
/// `Z/2^32` linear state; any other callee yields poison.
pub(super) fn call(
    machine: &Machine<'_>,
    frame: &Frame,
    name: &str,
    args: &[Val],
) -> Result<Val, Stop> {
    poison(frame)?;
    let op = match name {
        "wrapping_add" => BinaryOp::Add,
        "wrapping_mul" => BinaryOp::Mul,
        _ => return Ok(Val::poison()),
    };
    let [left, right] = args else {
        return Ok(Val::poison());
    };
    let wrapping = |v: &Val| match v.sym() {
        Some(Sym::Lin(lin)) => lin.ring == Ring::Wrapping32,
        Some(_) => false,
        None => true,
    };
    if !wrapping(left) || !wrapping(right) {
        return Ok(Val::poison());
    }
    binary(machine, frame, op, left, right)
}

/// A store at a symbolic index poisons the whole array for this lane.
pub(super) fn poison_store(frame: &mut Frame, target: &super::value::Array) -> Result<(), Stop> {
    poison(frame)?;
    if let Some(lane) = frame.lane.as_deref_mut() {
        lane.poisoned.insert(array_id(target));
    }
    Ok(())
}

/// Names assigned or bound, and arrays stored, anywhere in `lists`.
fn effects(lists: &[&[Stmt]]) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut names = BTreeSet::new();
    let mut arrays = BTreeSet::new();
    for list in lists {
        tree::walk(list, &mut Vec::new(), 0, &mut |stmt, _, _| match stmt {
            Stmt::Let { name, .. } | Stmt::Assign { name, .. } => {
                names.insert(name.clone());
            }
            Stmt::Store { array, .. } => {
                arrays.insert(array.clone());
            }
            Stmt::Loop(inner) => {
                if let tree::LoopKind::For { var, .. } = &inner.kind {
                    names.insert(var.clone());
                }
            }
            _ => {}
        });
    }
    (names, arrays)
}

fn poison_effects(frame: &mut Frame, names: &BTreeSet<String>, arrays: &BTreeSet<String>) {
    for name in names {
        frame.vars.insert(name.clone(), Val::poison());
    }
    let ids: Vec<usize> = arrays
        .iter()
        .filter_map(|a| super::interp::local_array_id(frame, a))
        .collect();
    if let Some(lane) = frame.lane.as_deref_mut() {
        lane.poisoned.extend(ids);
    }
}

/// Skip statements whose execution is unknown in a symbolic lane: everything
/// they may assign becomes poison. Unknown control over the current group's
/// own state is a violation (the fallback runs the region in source order).
pub(super) fn skip(frame: &mut Frame, lists: &[&[Stmt]]) -> Result<(), Stop> {
    let (names, arrays) = effects(lists);
    if let Some(group) = frame.lane.as_ref().and_then(|lane| lane.group.as_ref())
        && names
            .iter()
            .any(|n| group.vars.contains(n) || group.companions.contains(n))
    {
        return Err(Stop::violation("unknown control decides the law state"));
    }
    poison_effects(frame, &names, &arrays);
    Ok(())
}

/// `(order, strict)` of a select test `left op right` with the selected
/// variable on the right (`var_right`) or the left.
fn orientation(op: BinaryOp, var_right: bool) -> Option<(Order, bool)> {
    Some(match (op, var_right) {
        (BinaryOp::Lt, true) | (BinaryOp::Gt, false) => (Order::Min, true),
        (BinaryOp::Le, true) | (BinaryOp::Ge, false) => (Order::Min, false),
        (BinaryOp::Gt, true) | (BinaryOp::Lt, false) => (Order::Max, true),
        (BinaryOp::Ge, true) | (BinaryOp::Le, false) => (Order::Max, false),
        _ => return None,
    })
}

/// `if`: select and tropical tests on the current group, poison skipping,
/// or ordinary control.
pub(super) fn branch(
    machine: &mut Machine<'_>,
    frame: &mut Frame,
    cond: &Expr,
    yes: &[Stmt],
    no: &[Stmt],
) -> Result<(), Stop> {
    let value = match cond {
        Expr::Binary {
            op,
            left,
            right,
            span,
        } if frame.symbolic() => {
            let a = machine.eval(frame, left)?;
            let b = machine.eval(frame, right)?;
            if let Some(()) = group_test(machine, frame, *op, &a, &b, yes, no)? {
                return Ok(());
            }
            if a.is_sym() || b.is_sym() {
                Val::poison()
            } else {
                super::value::binary(*op, &a, &b, *span)?
            }
        }
        _ => machine.eval(frame, cond)?,
    };
    match truth(frame, &value)? {
        Some(true) => machine.block(frame, yes),
        Some(false) => machine.block(frame, no),
        None => skip(frame, &[yes, no]),
    }
}

/// A test comparing a concrete candidate with the current select or tropical
/// variable. `Some(())` when handled.
fn group_test(
    machine: &mut Machine<'_>,
    frame: &mut Frame,
    op: BinaryOp,
    a: &Val,
    b: &Val,
    yes: &[Stmt],
    no: &[Stmt],
) -> Result<Option<()>, Stop> {
    let Some(group) = frame.lane.as_ref().and_then(|lane| lane.group.clone()) else {
        return Ok(None);
    };
    let (var_right, candidate, state) = match (a.sym(), b.sym()) {
        (None, Some(s)) => (true, a, s.clone()),
        (Some(s), None) => (false, b, s.clone()),
        _ => return Ok(None),
    };
    let var = group.vars.first().cloned().unwrap_or_default();
    match (&group.rep, state) {
        (
            Rep::Select {
                order,
                strict,
                elem,
                ..
            },
            Sym::Select,
        ) => {
            if orientation(op, var_right) != Some((*order, *strict)) {
                return Err(Stop::violation(
                    "select test disagrees with its representation",
                ));
            }
            if !no.is_empty() {
                return Err(Stop::violation("select with an else arm"));
            }
            if rep::is_nan(candidate, *elem) {
                return Ok(Some(()));
            }
            capture(machine, frame, &group, &var, candidate.clone(), yes)?;
            Ok(Some(()))
        }
        (Rep::Tropical { order, .. }, Sym::Trop { a: ta, b: tb }) => {
            if orientation(op, var_right).map(|(o, _)| o) != Some(*order) {
                return Err(Stop::violation(
                    "tropical test disagrees with its representation",
                ));
            }
            let [Stmt::Assign { name, value }] = yes else {
                return Err(Stop::violation("tropical arm is not a single update"));
            };
            if !no.is_empty() || name != &var {
                return Err(Stop::violation("tropical arm shape"));
            }
            let assigned = machine.eval(frame, value)?;
            if &assigned != candidate {
                return Err(Stop::violation("tropical arm assigns another value"));
            }
            let c = candidate
                .int()
                .ok_or_else(|| Stop::violation("tropical candidate"))?;
            let b = Some(match tb {
                Some(b) if *order == Order::Max => b.max(c),
                Some(b) => b.min(c),
                None => c,
            });
            frame
                .vars
                .insert(var, Val::Sym(Box::new(Sym::Trop { a: ta, b })));
            Ok(Some(()))
        }
        _ => Ok(None),
    }
}

/// Run a select arm "as if taken" with the candidate bound, capture the
/// companions, poison everything else the arm touches, and fold the candidate
/// into the lane's select element.
fn capture(
    machine: &mut Machine<'_>,
    frame: &mut Frame,
    group: &Group,
    var: &str,
    candidate: Val,
    yes: &[Stmt],
) -> Result<(), Stop> {
    frame.vars.insert(var.to_owned(), candidate.clone());
    if let Some(lane) = frame.lane.as_deref_mut() {
        lane.capturing = true;
    }
    let result = machine.block(frame, yes);
    if let Some(lane) = frame.lane.as_deref_mut() {
        lane.capturing = false;
    }
    match result {
        Ok(()) => {}
        Err(stop) if stop.is_event() => return Err(Stop::violation("event inside a select arm")),
        Err(stop) => return Err(stop),
    }
    if frame.get(var)? != candidate {
        return Err(Stop::violation("select arm assigns another value"));
    }
    let companions = group
        .companions
        .iter()
        .map(|c| frame.get(c))
        .collect::<Result<Vec<_>, _>>()?;
    if companions.iter().any(Val::is_sym) {
        return Err(Stop::violation("companion not assigned by its select"));
    }
    let (mut names, arrays) = effects(&[yes]);
    names.remove(var);
    for c in &group.companions {
        names.remove(c);
    }
    poison_effects(frame, &names, &arrays);
    frame
        .vars
        .insert(var.to_owned(), Val::Sym(Box::new(Sym::Select)));
    for c in &group.companions {
        frame
            .vars
            .insert(c.clone(), Val::Sym(Box::new(Sym::Companion)));
    }
    let Rep::Select { order, strict, .. } = &group.rep else {
        return Err(Stop::internal("select group"));
    };
    let lane = frame
        .lane
        .as_deref_mut()
        .ok_or_else(|| Stop::internal("select outside a lane"))?;
    let replaces = match &lane.select {
        None => true,
        Some((best, _)) => rep::replaces(*order, *strict, &candidate, best),
    };
    if replaces {
        lane.select = Some((candidate, companions));
    }
    Ok(())
}
