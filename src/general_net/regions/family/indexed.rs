use super::*;

pub(super) fn gather_scatter(cx: &Context<'_>, scatter: bool) -> bool {
    let mut params = vec![(Type::ArrayU32, Some(Mode::Read)); 2];
    if scatter {
        params.push((Type::U32, None));
    }
    if !cx.params(&params, Type::ArrayU32) {
        return false;
    }
    let Some((out, length)) = fresh(cx, "filled_u32") else {
        return false;
    };
    let p = &cx.tree.params;
    let (input, indices) = (&p[0].name, &p[1].name);
    let Some((i, hi)) = counted(cx.inner) else {
        return false;
    };
    if !len(hi, indices)
        || !(if scatter {
            named(length, &p[2].name)
        } else {
            len(length, indices)
        })
    {
        return false;
    }
    let [
        Stmt::Store {
            array,
            index: dst,
            value,
            ..
        },
    ] = cx.inner.body.as_slice()
    else {
        return false;
    };
    let Some(src) = index(value, input) else {
        return false;
    };
    array == out
        && if scatter {
            element(dst, indices, i) && named(src, i)
        } else {
            named(dst, i) && element(src, indices, i)
        }
}

pub(super) fn histogram(cx: &Context<'_>) -> bool {
    if !cx.params(
        &[
            (Type::ArrayU32, Some(Mode::Read)),
            (Type::ArrayU32, Some(Mode::Read)),
            (Type::U32, None),
        ],
        Type::ArrayU32,
    ) {
        return false;
    }
    let Some((out, length)) = fresh(cx, "filled_u32") else {
        return false;
    };
    let p = &cx.tree.params;
    let (values, indices, bins) = (&p[0].name, &p[1].name, &p[2].name);
    let Some((i, hi)) = counted(cx.inner) else {
        return false;
    };
    let [
        Stmt::Store {
            array,
            index: dst,
            value,
            ..
        },
    ] = cx.inner.body.as_slice()
    else {
        return false;
    };
    let Some([a, b]) = call(value, "wrapping_add") else {
        return false;
    };
    named(length, bins)
        && len(hi, indices)
        && array == out
        && element(dst, indices, i)
        && [(a, b), (b, a)].into_iter().any(|(acc, weight)| {
            index(acc, out).is_some_and(|x| element(x, indices, i)) && element(weight, values, i)
        })
}

fn number(e: &Expr) -> Option<u32> {
    if let Expr::U32(n) = e { Some(*n) } else { None }
}
fn linear(e: &Expr, i: &str) -> Option<(u32, u32)> {
    let factor = |e: &Expr| {
        if named(e, i) {
            return Some(1);
        }
        let (a, b) = binary(e, BinaryOp::Mul)?;
        if named(a, i) {
            number(b)
        } else if named(b, i) {
            number(a)
        } else {
            None
        }
    };
    if let Some(a) = factor(e) {
        return Some((a, 0));
    }
    let (a, b) = binary(e, BinaryOp::Add)?;
    factor(a)
        .zip(number(b))
        .or_else(|| factor(b).zip(number(a)))
}

pub(super) fn affine(cx: &Context<'_>) -> bool {
    if !cx.params(&[(Type::ArrayU32, Some(Mode::Read))], Type::ArrayU32) {
        return false;
    }
    let Some((out, length)) = fresh(cx, "filled_u32") else {
        return false;
    };
    let Some(length) = number(length) else {
        return false;
    };
    let LoopKind::For { var: i, lo, hi } = &cx.inner.kind else {
        return false;
    };
    let (Some(lo), Some(hi)) = (number(lo), number(hi)) else {
        return false;
    };
    let [
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
    let Some((a, b)) = linear(index, i) else {
        return false;
    };
    if array != out
        || !element(value, &cx.tree.params[0].name, i)
        || hi.saturating_sub(lo) > 4096
        || length > 4096
        || (hi.saturating_sub(lo) > 1 && a == 0)
    {
        return false;
    }
    // Same finite literal proof as the static indexed family: no wrap and every
    // demanded endpoint in bounds. This is not a runtime injectivity guess.
    hi <= lo
        || hi
            .checked_sub(1)
            .and_then(|last| a.checked_mul(last))
            .and_then(|x| x.checked_add(b))
            .is_some_and(|last| last < length)
}

fn test(e: &Expr, array: &str, i: &str, n: u32) -> bool {
    binary(e, BinaryOp::Eq).is_some_and(|(a, b)| element(a, array, i) && uint(b, n))
}
fn fixed_loop(s: &Stmt) -> Option<(&Loop, &str, u32)> {
    let l = as_loop(s)?;
    let (i, hi) = counted(l)?;
    Some((l, i, number(hi)?))
}
fn condition(s: &Stmt) -> Option<(&Expr, &[Stmt])> {
    if let Stmt::If { cond, yes, no } = s
        && no.is_empty()
    {
        Some((cond, yes))
    } else {
        None
    }
}
fn set(s: &Stmt, array: &str, i: &str, value: u32) -> bool {
    matches!(s,Stmt::Store { array: a, index, value: v, .. } if a == array && named(index,i) && uint(v,value))
}

pub(super) fn frontier(tree: &FunctionTree, inner: &Loop) -> bool {
    if !params(
        tree,
        &[(Type::ArrayU32, Some(Mode::Read)), (Type::U32, None)],
        Type::I32,
    ) {
        return false;
    }
    let [fa, sa, seed, round, ci, count, ret] = tree.body.as_slice() else {
        return false;
    };
    let (Some((frontier, n)), Some((seen, m))) =
        (allocated(fa, "filled_u32"), allocated(sa, "filled_u32"))
    else {
        return false;
    };
    let (Some(n), Some(m)) = (number(n), number(m)) else {
        return false;
    };
    let Some((round, _, _)) = fixed_loop(round) else {
        return false;
    };
    if round.head != inner.head || n != m || !set(seed, frontier, &tree.params[1].name, 1) {
        return false;
    }
    let [na, vertices, copy] = round.body.as_slice() else {
        return false;
    };
    let Some((next, length)) = allocated(na, "filled_u32") else {
        return false;
    };
    let Some((vertices, u, un)) = fixed_loop(vertices) else {
        return false;
    };
    let [s] = vertices.body.as_slice() else {
        return false;
    };
    let Some((
        Expr::Logic {
            and: true,
            left,
            right,
        },
        [mark, neighbors],
    )) = condition(s)
    else {
        return false;
    };
    if !uint(length, n)
        || un != n
        || !test(left, frontier, u, 1)
        || !test(right, seen, u, 0)
        || !set(mark, seen, u, 1)
    {
        return false;
    }
    let Some((neighbors, v, vn)) = fixed_loop(neighbors) else {
        return false;
    };
    let [s] = neighbors.body.as_slice() else {
        return false;
    };
    let Some((
        Expr::Logic {
            and: true,
            left,
            right,
        },
        [store],
    )) = condition(s)
    else {
        return false;
    };
    let Some((edge, one)) = binary(left, BinaryOp::Eq) else {
        return false;
    };
    let Some(offset) = index(edge, &tree.params[0].name) else {
        return false;
    };
    let dense = binary(offset, BinaryOp::Add).is_some_and(|(a, b)| {
        named(b, v) && binary(a, BinaryOp::Mul).is_some_and(|(a, b)| named(a, u) && uint(b, n))
    });
    let Some((copy, i, cn)) = fixed_loop(copy) else {
        return false;
    };
    let [
        Stmt::Store {
            array,
            index: ix,
            value,
            ..
        },
    ] = copy.body.as_slice()
    else {
        return false;
    };
    let Some((count_name, Expr::I32(0))) = binding(ci) else {
        return false;
    };
    let Some((count, j, jn)) = fixed_loop(count) else {
        return false;
    };
    let [s] = count.body.as_slice() else {
        return false;
    };
    let Some((cond, [s])) = condition(s) else {
        return false;
    };
    dense
        && vn == n
        && uint(one, 1)
        && test(right, seen, v, 0)
        && set(store, next, v, 1)
        && cn == n
        && array == frontier
        && named(ix, i)
        && element(value, next, i)
        && jn == n
        && test(cond, seen, j, 1)
        && assign(s, count_name)
            .and_then(|e| binary(e, BinaryOp::Add))
            .is_some_and(|(a, b)| named(a, count_name) && matches!(b, Expr::I32(1)))
        && returned(ret, count_name)
}

/// A dense frontier is a serial sequence of certified frontier rounds and
/// disjoint copy maps. There is no law commuting the rounds themselves.
pub(super) fn frontier_laws(tree: &FunctionTree, inner: &Loop, form: &NormalForm) -> bool {
    let [_, vertices, copy] = inner.body.as_slice() else {
        return false;
    };
    let (Some(vertices), Some(copy)) = (as_loop(vertices), as_loop(copy)) else {
        return false;
    };
    let parallel = |inner: &Loop| {
        form.loop_form(RegionId {
            function: tree.index,
            head: Some(inner.head),
        })
        .and_then(|l| match &l.node {
            Node::Parallel(n) => Some(n),
            _ => None,
        })
    };
    parallel(vertices).is_some_and(|n| n.shape() == "frontier" && !n.first_exit)
        && parallel(copy).is_some_and(|n| n.shape() == "map" && !n.first_exit)
}
