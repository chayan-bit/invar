use super::*;
use crate::general_net::normal_form::{ArrayLaw, Rep};

fn nest<'a>(cx: &'a Context<'_>, map: bool) -> Option<(&'a str, &'a str, &'a Loop, &'a str)> {
    if !cx.params(
        &[
            (Type::ArrayU32, Some(Mode::Read)),
            (Type::ArrayU32, Some(Mode::Read)),
            (Type::U32, None),
        ],
        Type::ArrayU32,
    ) {
        return None;
    }
    let (out, length) = fresh(cx, "filled_u32")?;
    let p = &cx.tree.params;
    let (input, offsets, count) = (&p[0].name, &p[1].name, &p[2].name);
    if !(if map {
        len(length, input)
    } else {
        named(length, count)
    }) {
        return None;
    }
    let segment = bound(cx.inner, count)?;
    let s = if map {
        let [s] = cx.inner.body.as_slice() else {
            return None;
        };
        s
    } else {
        let [_, s, _] = cx.inner.body.as_slice() else {
            return None;
        };
        s
    };
    let inner = as_loop(s)?;
    let LoopKind::For { var, lo, hi } = &inner.kind else {
        return None;
    };
    if !index(lo, offsets).is_some_and(|e| named(e, segment))
        || !index(hi, offsets).is_some_and(|e| {
            binary(e, BinaryOp::Add).is_some_and(|(a, b)| named(a, segment) && uint(b, 1))
        })
    {
        return None;
    }
    Some((out, input, inner, var))
}

pub(super) fn map(cx: &Context<'_>) -> bool {
    let Some((out, input, inner, i)) = nest(cx, true) else {
        return false;
    };
    let [
        Stmt::Store {
            array,
            index,
            value,
            ..
        },
    ] = inner.body.as_slice()
    else {
        return false;
    };
    if array != out || !named(index, i) {
        return false;
    }
    let Some((segment, _)) = counted(cx.inner) else {
        return false;
    };
    let mut value = value;
    let mut ops = 0;
    loop {
        if element(value, input, i) {
            return ops <= 16;
        }
        let Expr::Call { name, args, .. } = value else {
            return false;
        };
        let [a, b] = args.as_slice() else {
            return false;
        };
        let operand =
            |e: &Expr| matches!(e, Expr::U32(_)) || (name == "wrapping_add" && named(e, segment));
        if !matches!(name.as_str(), "wrapping_add" | "wrapping_mul") {
            return false;
        }
        value = if operand(b) {
            a
        } else if operand(a) {
            b
        } else {
            return false;
        };
        ops += 1;
        if ops > 16 {
            return false;
        }
    }
}

pub(super) fn reduce(cx: &Context<'_>) -> bool {
    let Some((out, input, inner, i)) = nest(cx, false) else {
        return false;
    };
    if !cx.parallel(inner) {
        return false;
    }
    let [
        seed,
        _,
        Stmt::Store {
            array,
            index,
            value,
            ..
        },
    ] = cx.inner.body.as_slice()
    else {
        return false;
    };
    let Some((state, initial)) = binding(seed) else {
        return false;
    };
    let Some((segment, _)) = counted(cx.inner) else {
        return false;
    };
    let [s] = inner.body.as_slice() else {
        return false;
    };
    let Some(update) = assign(s, state) else {
        return false;
    };
    let op = if uint(initial, 0) {
        "wrapping_add"
    } else if uint(initial, 1) {
        "wrapping_mul"
    } else {
        return false;
    };
    let Some([a, b]) = call(update, op) else {
        return false;
    };
    array == out
        && named(index, segment)
        && named(value, state)
        && named(a, state)
        && element(b, input, i)
}

fn increment(s: &Stmt, n: &str) -> bool {
    assign(s, n)
        .and_then(|e| binary(e, BinaryOp::Add))
        .is_some_and(|(a, b)| named(a, n) && uint(b, 1))
}
fn selected<'a>(s: &'a Stmt, input: &str, i: &str, cutoff: &str) -> Option<(BinaryOp, &'a [Stmt])> {
    let Stmt::If { cond, yes, no } = s else {
        return None;
    };
    let Expr::Binary {
        op, left, right, ..
    } = cond
    else {
        return None;
    };
    (no.is_empty()
        && matches!(
            op,
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge | BinaryOp::Eq | BinaryOp::Ne
        )
        && element(left, input, i)
        && named(right, cutoff))
    .then_some((*op, yes))
}

pub(super) fn compaction(cx: &Context<'_>) -> bool {
    if !cx.params(
        &[
            (Type::ArrayU32, Some(Mode::Read)),
            (Type::U32, None),
            (Type::U32, None),
        ],
        Type::ArrayU32,
    ) {
        return false;
    }
    let p = &cx.tree.params;
    let (input, count, cutoff) = (&p[0].name, &p[1].name, &p[2].name);
    let [seed, first, alloc, pos, second, ret] = cx.tree.body.as_slice() else {
        return false;
    };
    let Some((size, Expr::U32(0))) = binding(seed) else {
        return false;
    };
    let Some((position, Expr::U32(0))) = binding(pos) else {
        return false;
    };
    let Some((out, length)) = allocated(alloc, "filled_u32") else {
        return false;
    };
    let (Some(first), Some(second)) = (as_loop(first), as_loop(second)) else {
        return false;
    };
    if ![first.head, second.head].contains(&cx.inner.head)
        || !cx.parallel(first)
        || !cx.parallel(second)
        || !named(length, size)
        || !returned(ret, out)
    {
        return false;
    }
    let (Some(i), Some(j)) = (bound(first, count), bound(second, count)) else {
        return false;
    };
    let ([a], [b]) = (first.body.as_slice(), second.body.as_slice()) else {
        return false;
    };
    let (Some((predicate, [add])), Some((other, [store, next]))) =
        (selected(a, input, i, cutoff), selected(b, input, j, cutoff))
    else {
        return false;
    };
    predicate == other
        && increment(add, size)
        && increment(next, position)
        && matches!(store,Stmt::Store { array, index, value, .. } if array == out && named(index,position) && element(value,input,j))
        && (cx.inner.head == first.head
            || cx.node.arrays.iter().any(|(a, l)| {
                a == out && matches!(l,ArrayLaw::Compaction { counter } if counter == position)
            }))
        && cx
            .node
            .carried
            .iter()
            .all(|s| matches!(s.rep, Rep::Counter(_) | Rep::LiftedSum(_)))
}
