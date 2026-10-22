use super::*;
use crate::general_net::normal_form::{Rep, Ring};

/// Identify only the bounded table summary shape, so its absent L8 law is an
/// explicit refusal. This never constructs a certificate or an admission.
pub(super) fn table(tree: &FunctionTree, lowered: &mir::Function, inner: &Loop) -> bool {
    if !matches!(tree.result, Type::U32 | Type::ArrayU32)
        || !params(
            tree,
            &[
                (Type::ArrayU32, Some(Mode::Read)),
                (Type::U32, None),
                (Type::U32, None),
            ],
            tree.result.clone(),
        )
    {
        return false;
    }
    let p = &tree.params;
    let Some(i) = bound(inner, &p[2].name) else {
        return false;
    };
    let (state, output, ret) = match tree.body.as_slice() {
        [state, Stmt::Loop(l), ret] if l.head == inner.head => (state, None, ret),
        [state, alloc, Stmt::Loop(l), ret] if l.head == inner.head => (state, Some(alloc), ret),
        _ => return false,
    };
    let Some((s, initial)) = binding(state) else {
        return false;
    };
    if !named(initial, &p[1].name)
        || !lowered
            .locals
            .get(s)
            .is_some_and(|l| l.mutable && l.ty == Type::U32)
    {
        return false;
    }
    let Some(update) = inner.body.first().and_then(|stmt| assign(stmt, s)) else {
        return false;
    };
    let Some((mul, state)) = index(update, &p[0].name).and_then(|ix| binary(ix, BinaryOp::Add))
    else {
        return false;
    };
    if !named(state, s)
        || !binary(mul, BinaryOp::Mul).is_some_and(|(iter, width)| {
            named(iter, i) && matches!(width, Expr::U32(k) if (1..=16).contains(k))
        })
    {
        return false;
    }
    match output {
        None => tree.result == Type::U32 && inner.body.len() == 1 && returned(ret, s),
        Some(alloc) => {
            let Some((out, length)) = allocated(alloc, "filled_u32") else {
                return false;
            };
            tree.result == Type::ArrayU32
                && named(length, &p[2].name)
                && returned(ret, out)
                && matches!(inner.body.as_slice(), [_, Stmt::Store { array, index, value, .. }] if array == out && named(index, i) && named(value, s))
        }
    }
}

fn flat_const(e: &Expr, i: &str, stride: u32, offset: u32) -> bool {
    binary(e, BinaryOp::Add).is_some_and(|(a, b)| {
        uint(b, offset)
            && binary(a, BinaryOp::Mul).is_some_and(|(a, b)| named(a, i) && uint(b, stride))
    })
}

fn dot(mut e: &Expr, coeff: &str, i: &str, states: &[&str], row: usize) -> bool {
    let d = states.len();
    for col in (0..d).rev() {
        let Some([prior, term]) = call(e, "wrapping_add") else {
            return false;
        };
        let Some([coefficient, state]) = call(term, "wrapping_mul") else {
            return false;
        };
        if !named(state, states[col])
            || !index(coefficient, coeff)
                .is_some_and(|ix| flat_const(ix, i, (d * d) as u32, (row * d + col) as u32))
        {
            return false;
        }
        e = prior;
    }
    uint(e, 0)
}

pub(super) fn admit(cx: &Context<'_>) -> bool {
    // Finite-state tables need L8, which P1 deliberately does not represent.
    // A source recognizer's table admission cannot license an invented PNF law.
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
    let p = &cx.tree.params;
    let (coeff, seed, count) = (&p[0].name, &p[1].name, &p[2].name);
    let Some(i) = bound(cx.inner, count) else {
        return false;
    };
    let Some(position) = cx
        .tree
        .body
        .iter()
        .position(|s| as_loop(s).is_some_and(|l| l.head == cx.inner.head))
    else {
        return false;
    };
    if cx.tree.body.len() != position + 2 {
        return false;
    }
    let ret = &cx.tree.body[position + 1];
    let (prefix, d, out) = if position > 0 {
        match allocated(&cx.tree.body[position - 1], "filled_u32") {
            Some((out, _)) if returned(ret, out) => (true, position - 1, Some(out)),
            _ => (false, position, None),
        }
    } else {
        return false;
    };
    if !(1..=4).contains(&d) || cx.inner.body.len() != d * (2 + usize::from(prefix)) {
        return false;
    }
    if cx.node.carried.iter().any(|s| {
        !matches!(
            &s.rep,
            Rep::Affine(Ring::Wrapping32)
                | Rep::Sealed(_)
                | Rep::Matrix {
                    ring: Ring::Wrapping32,
                    ..
                }
        )
    }) {
        return false;
    }
    let mut states = Vec::new();
    for (j, s) in cx.tree.body[..d].iter().enumerate() {
        let Some((state, initial)) = binding(s) else {
            return false;
        };
        if !cx.mutable(state, Type::U32) || !index(initial, seed).is_some_and(|e| uint(e, j as u32))
        {
            return false;
        }
        states.push(state);
    }
    for row in 0..d {
        let Some((next, value)) = binding(&cx.inner.body[row]) else {
            return false;
        };
        if !dot(value, coeff, i, &states, row)
            || !assign(&cx.inner.body[d + row], states[row]).is_some_and(|e| named(e, next))
        {
            return false;
        }
    }
    if let Some(out) = out {
        let Some((_, length)) = allocated(&cx.tree.body[d], "filled_u32") else {
            return false;
        };
        if !binary(length, BinaryOp::Mul).is_some_and(|(a, b)| named(a, count) && uint(b, d as u32))
        {
            return false;
        }
        for (row, state) in states.iter().enumerate() {
            if !matches!(&cx.inner.body[2*d+row],Stmt::Store { array, index, value, .. }
                if array == out && flat_const(index,i,d as u32,row as u32) && named(value,state))
            {
                return false;
            }
        }
        true
    } else {
        matches!(ret,Stmt::Return { value: Some(Expr::ArrayLit { elem: Type::U32, items }), .. }
            if items.len() == d && items.iter().zip(states).all(|(e,n)| named(e,n)))
    }
}
