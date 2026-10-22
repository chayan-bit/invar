use super::*;
use crate::general_net::normal_form::{ArrayLaw, Carried, Rep, Ring};
use crate::laws::Law;

fn operand<'a>(e: &'a Expr, i: &str) -> Option<&'a str> {
    match e {
        Expr::Index { array, index, .. } if named(index, i) => Some(array),
        _ => None,
    }
}

fn map_at<'a>(cx: &Context<'_>, inner: &'a Loop) -> Option<(&'a str, Vec<&'a str>)> {
    let (i, hi) = counted(inner)?;
    if !matches!(hi, Expr::U32(_) | Expr::Len(_)) {
        return None;
    }
    let [
        Stmt::Store {
            array,
            index,
            value,
            ..
        },
    ] = inner.body.as_slice()
    else {
        return None;
    };
    if !named(index, i) || !cx.mutable(array, Type::ArrayU32) {
        return None;
    }
    let Expr::Call { name, args, .. } = value else {
        return None;
    };
    if !matches!(name.as_str(), "wrapping_add" | "wrapping_mul") || args.len() != 2 {
        return None;
    }
    let mut inputs = Vec::new();
    for arg in args {
        match arg {
            Expr::U32(_) => {}
            _ => {
                let input = operand(arg, i)?;
                let local = cx.lowered.locals.get(input)?;
                if input == array
                    || local.ty != Type::ArrayU32
                    || local.mutable
                    || local.mode == Some(Mode::Mut)
                {
                    return None;
                }
                inputs.push(input);
            }
        }
    }
    if inputs.is_empty() {
        return None;
    }
    let fresh = cx
        .lowered
        .blocks
        .iter()
        .take(inner.head)
        .flat_map(|b| &b.instructions)
        .any(|ins| {
            let mir::Operation::Bind { name, value } = &ins.kind else {
                return false;
            };
            name == array
                && cx
                    .lowered
                    .blocks
                    .iter()
                    .flat_map(|b| &b.instructions)
                    .any(|def| {
                        def.output.as_ref().is_some_and(|(id, _)| id == value)
                            && match &def.kind {
                                mir::Operation::Array(_) => true,
                                mir::Operation::Call { name, .. } => name == "filled_u32",
                                _ => false,
                            }
                    })
        });
    fresh.then_some((array, inputs))
}

pub(super) fn map(cx: &Context<'_>) -> bool {
    cx.node.carried.is_empty()
        && cx.node.arrays.iter().all(|(_, l)| *l == ArrayLaw::Own)
        && map_at(cx, cx.inner).is_some()
}

pub(super) fn reduce(cx: &Context<'_>) -> bool {
    let Some((i, _)) = counted(cx.inner) else {
        return false;
    };
    let carried = match cx.node.carried.as_slice() {
        [carried] => carried,
        [x, y] => return pair(cx, i, x, y),
        _ => return false,
    };
    if carried.scan || !matches!(carried.rep, Rep::Sealed(_)) || !cx.node.arrays.is_empty() {
        return false;
    }
    let [Stmt::Assign { name, value }] = cx.inner.body.as_slice() else {
        return false;
    };
    let Expr::Call { name: op, args, .. } = value else {
        return false;
    };
    if !matches!(op.as_str(), "wrapping_add" | "wrapping_mul") || !cx.mutable(name, Type::U32) {
        return false;
    }
    let [a, b] = args.as_slice() else {
        return false;
    };
    [(a, b), (b, a)].into_iter().any(|(state, e)| {
        named(state, name) && operand(e, i).is_some_and(|input| cx.ty(input, Type::ArrayU32))
    })
}

/// A read-only `Array<u32>` local (A0's `local_is(.., ArrayU32, false)`).
fn read_only(cx: &Context<'_>, name: &str) -> bool {
    cx.lowered
        .locals
        .get(name)
        .is_some_and(|l| l.ty == Type::ArrayU32 && !l.mutable && l.mode != Some(Mode::Mut))
}

/// `wrapping_mul(state, x[i])` in either operand order; the array `x`.
fn state_times<'a>(e: &'a Expr, state: &str, i: &str) -> Option<&'a str> {
    let [l, r] = call(e, "wrapping_mul")? else {
        return None;
    };
    match (named(l, state), named(r, state)) {
        (true, false) => operand(r, i),
        (false, true) => operand(l, i),
        _ => None,
    }
}

/// A0's ordered affine-pair reduction (`pair_reduction::reduction_in`):
/// `a = a * x[i]` under an L2 sealed product and `b = b * x[i] + y[i]` under an
/// L4 wrapping affine law, over one read-only coefficient array `x`, in either
/// update order. Each update reads only its own old state.
fn pair(cx: &Context<'_>, i: &str, x: &Carried, y: &Carried) -> bool {
    if !cx.node.arrays.is_empty() || x.scan || y.scan {
        return false;
    }
    let (a, b) = match (&x.rep, &y.rep) {
        (Rep::Sealed(Law::WrappingProduct), Rep::Affine(Ring::Wrapping32)) => (x, y),
        (Rep::Affine(Ring::Wrapping32), Rep::Sealed(Law::WrappingProduct)) => (y, x),
        _ => return false,
    };
    if a.var == b.var || !cx.mutable(&a.var, Type::U32) || !cx.mutable(&b.var, Type::U32) {
        return false;
    }
    let [first, second] = cx.inner.body.as_slice() else {
        return false;
    };
    let (Some(update_a), Some(update_b)) = (
        assign(first, &a.var).or_else(|| assign(second, &a.var)),
        assign(first, &b.var).or_else(|| assign(second, &b.var)),
    ) else {
        return false;
    };
    let Some(coefficients) = state_times(update_a, &a.var, i) else {
        return false;
    };
    let Some([l, r]) = call(update_b, "wrapping_add") else {
        return false;
    };
    let offsets = [(l, r), (r, l)].into_iter().find_map(|(product, offset)| {
        (state_times(product, &b.var, i) == Some(coefficients))
            .then(|| operand(offset, i))
            .flatten()
    });
    offsets.is_some_and(|offsets| read_only(cx, coefficients) && read_only(cx, offsets))
}

fn coefficient(cx: &Context<'_>, e: &Expr, i: &str) -> bool {
    matches!(e, Expr::U32(_)) || operand(e, i).is_some_and(|a| cx.ty(a, Type::ArrayU32))
}

pub(super) fn scan(cx: &Context<'_>) -> bool {
    let Some((i, _)) = counted(cx.inner) else {
        return false;
    };
    let [state] = cx.node.carried.as_slice() else {
        return false;
    };
    if !matches!(state.rep, Rep::Affine(Ring::Wrapping32) | Rep::Sealed(_))
        || !cx.mutable(&state.var, Type::U32)
    {
        return false;
    }
    let (update, publish) = match cx.inner.body.as_slice() {
        [s @ Stmt::Assign { .. }] => (s, None),
        [s @ Stmt::Assign { .. }, p @ Stmt::Store { .. }]
        | [p @ Stmt::Store { .. }, s @ Stmt::Assign { .. }] => (s, Some(p)),
        _ => return false,
    };
    let Some(value) = assign(update, &state.var) else {
        return false;
    };
    let Some([a, b]) = call(value, "wrapping_add") else {
        return false;
    };
    let valid = [(a, b), (b, a)].into_iter().any(|(mul, offset)| {
        let Some([a, b]) = call(mul, "wrapping_mul") else {
            return false;
        };
        coefficient(cx, offset, i)
            && [(a, b), (b, a)]
                .into_iter()
                .any(|(s, c)| named(s, &state.var) && coefficient(cx, c, i))
    });
    valid
        && publish.is_none_or(|p| {
            matches!(p, Stmt::Store { array, index, value, .. }
        if cx.mutable(array,Type::ArrayU32) && named(index,i) && named(value,&state.var)
            && !reads_array(assign(update,&state.var).unwrap_or(value),array))
        })
}

fn reads_array(e: &Expr, array: &str) -> bool {
    match e {
        Expr::Index {
            array: a, index, ..
        } => a == array || reads_array(index, array),
        Expr::Call { args, .. } => args.iter().any(|e| reads_array(e, array)),
        _ => false,
    }
}

fn feeds(cx: &Context<'_>, first: &Loop, second: &Loop) -> bool {
    let Some((output, _)) = map_at(cx, first) else {
        return false;
    };
    let Some((_, inputs)) = map_at(cx, second) else {
        return false;
    };
    let mut owners = vec![output.to_owned()];
    for s in &cx.tree.body {
        if let Stmt::Let {
            name,
            value: Expr::Move(from),
        } = s
            && owners.contains(from)
        {
            owners.push(name.clone());
        }
    }
    inputs
        .iter()
        .filter(|i| owners.iter().any(|o| o == **i))
        .count()
        == 1
}

/// Follow only checked L19 edges. Two stages use pair fusion; a longer run
/// uses the DAG family. Independent maps after a fan-out end a producer run.
pub(super) fn fusion(cx: &Context<'_>, dag: bool) -> bool {
    let forms = &cx.form.functions[cx.tree.index].loops;
    let mut current = RegionId {
        function: cx.tree.index,
        head: Some(cx.inner.head),
    };
    let mut inner = cx.inner;
    let mut predecessors = 0usize;
    for _ in 0..forms.len() {
        let Some(previous) = forms
            .iter()
            .find(|l| matches!(&l.node, Node::Parallel(n) if n.fused_with == Some(current)))
        else {
            break;
        };
        let Some(first) = previous
            .region
            .head
            .and_then(|h| tree::find_loop(&cx.tree.body, h))
        else {
            return false;
        };
        if !feeds(cx, first, inner) {
            break;
        }
        predecessors += 1;
        current = previous.region;
        inner = first;
    }
    // The existing core graph bounds a producer to 16 stages. A longer source
    // run may start another candidate only at the next chunk boundary.
    if !predecessors.is_multiple_of(16) {
        return false;
    }
    let mut node = cx.node;
    inner = cx.inner;
    let mut stages = 1;
    while stages < 16 {
        let Some(next) = node.fused_with else { break };
        let Some(second) = next.head.and_then(|h| tree::find_loop(&cx.tree.body, h)) else {
            return false;
        };
        if !feeds(cx, inner, second) {
            break;
        }
        let Some(form) = cx.form.loop_form(next) else {
            return false;
        };
        let Node::Parallel(next_node) = &form.node else {
            return false;
        };
        node = next_node;
        inner = second;
        stages += 1;
    }
    if dag { stages > 2 } else { stages == 2 }
}
