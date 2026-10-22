use super::*;

fn grid_params(cx: &Context<'_>, elem: Type) -> bool {
    cx.params(
        &[
            (elem.clone(), Some(Mode::Read)),
            (Type::U32, None),
            (Type::U32, None),
        ],
        elem,
    )
}

pub(super) fn transpose(cx: &Context<'_>) -> bool {
    let allocator = if grid_params(cx, Type::ArrayU32) {
        "filled_u32"
    } else if grid_params(cx, Type::ArrayF32) {
        "filled_f32"
    } else {
        return false;
    };
    let Some((out, length)) = fresh(cx, allocator) else {
        return false;
    };
    let p = &cx.tree.params;
    let (input, rows, cols) = (&p[0].name, &p[1].name, &p[2].name);
    let Some(i) = bound(cx.inner, rows) else {
        return false;
    };
    let [s] = cx.inner.body.as_slice() else {
        return false;
    };
    let Some(inner) = as_loop(s) else {
        return false;
    };
    let Some(j) = bound(inner, cols) else {
        return false;
    };
    let [
        Stmt::Store {
            array,
            index: dst,
            value,
            ..
        },
    ] = inner.body.as_slice()
    else {
        return false;
    };
    product(length, rows, cols)
        && array == out
        && flat(dst, j, rows, i)
        && index(value, input).is_some_and(|src| flat(src, i, cols, j))
}

pub(super) fn tensor(cx: &Context<'_>) -> bool {
    if !cx.params(
        &[
            (Type::ArrayF32, Some(Mode::Read)),
            (Type::ArrayF32, Some(Mode::Read)),
            (Type::U32, None),
            (Type::U32, None),
            (Type::U32, None),
        ],
        Type::ArrayF32,
    ) {
        return false;
    }
    let Some((out, length)) = fresh(cx, "filled_f32") else {
        return false;
    };
    let p = &cx.tree.params;
    let (a, b, rows, inner, cols) = (&p[0].name, &p[1].name, &p[2].name, &p[3].name, &p[4].name);
    if !product(length, rows, cols) {
        return false;
    }
    let Some(i) = bound(cx.inner, rows) else {
        return false;
    };
    let [column] = cx.inner.body.as_slice() else {
        return false;
    };
    let Some(column) = as_loop(column) else {
        return false;
    };
    let Some(j) = bound(column, cols) else {
        return false;
    };
    // The inner f32 accumulation remains source ordered in each (i,j) lane.
    // It is never offered to the sealed-u32 collective.
    let (reduction, sum) = match column.body.as_slice() {
        [
            Stmt::Let {
                name,
                value: Expr::F32(zero),
            },
            r,
            Stmt::Store {
                array,
                index: dst,
                value,
                ..
            },
        ] if *zero == crate::numeric_contract::F32Bits::POS_ZERO
            && array == out
            && flat(dst, i, cols, j)
            && named(value, name) =>
        {
            (r, Some(name.as_str()))
        }
        [r] => (r, None),
        _ => return false,
    };
    let Some(reduction) = as_loop(reduction) else {
        return false;
    };
    let Some(k) = bound(reduction, inner) else {
        return false;
    };
    let [step] = reduction.body.as_slice() else {
        return false;
    };
    let value = match (sum, step) {
        (Some(s), Stmt::Assign { name, value }) if name == s => value,
        (
            None,
            Stmt::Store {
                array,
                index,
                value,
                ..
            },
        ) if array == out && flat(index, i, cols, j) => value,
        _ => return false,
    };
    // The ordered strict step `acc + a*b`, or the explicit Tensor Core step
    // `tc_mul_add(a, b, acc)` whose policy the source declares.
    let (acc, left, right) = match call(value, "tc_mul_add") {
        Some([left, right, acc]) => (acc, left, right),
        Some(_) => return false,
        None => {
            let Some((acc, product)) = binary(value, BinaryOp::Add) else {
                return false;
            };
            let Some((left, right)) = binary(product, BinaryOp::Mul) else {
                return false;
            };
            (acc, left, right)
        }
    };
    let accumulator = sum.map_or_else(
        || index(acc, out).is_some_and(|x| flat(x, i, cols, j)),
        |s| named(acc, s),
    );
    accumulator
        && index(left, a).is_some_and(|x| flat(x, i, inner, k))
        && index(right, b).is_some_and(|x| flat(x, k, cols, j))
}

/// Radius-one five-point stencil in the branch order north/south/west/east:
/// an f32 grid with the exact left-associated `+` sum, or a u32 grid with the
/// same left-associated `wrapping_add` sum. The u32 form is the device
/// runtime's own source contract, matched explicitly, not inferred from f32.
pub(super) fn stencil(cx: &Context<'_>) -> bool {
    let (allocator, elem) = if grid_params(cx, Type::ArrayF32) {
        ("filled_f32", Type::F32)
    } else if grid_params(cx, Type::ArrayU32) {
        ("filled_u32", Type::U32)
    } else {
        return false;
    };
    let floating = elem == Type::F32;
    let Some((out, length)) = fresh(cx, allocator) else {
        return false;
    };
    let p = &cx.tree.params;
    let (input, rows, cols) = (&p[0].name, &p[1].name, &p[2].name);
    if !product(length, rows, cols) {
        return false;
    }
    let Some(i) = bound(cx.inner, rows) else {
        return false;
    };
    let [s] = cx.inner.body.as_slice() else {
        return false;
    };
    let Some(inner) = as_loop(s) else {
        return false;
    };
    let Some(j) = bound(inner, cols) else {
        return false;
    };
    if inner.body.len() != 9 {
        return false;
    }
    let center = |e: &Expr| index(e, input).is_some_and(|x| flat(x, i, cols, j));
    let mut names = Vec::new();
    let mut clamp = None;
    for direction in 0..4 {
        let Some((name, zero)) = binding(&inner.body[2 * direction]) else {
            return false;
        };
        let zero = match zero {
            Expr::F32(z) => floating && *z == crate::numeric_contract::F32Bits::POS_ZERO,
            _ => !floating && uint(zero, 0),
        };
        if !zero || !cx.mutable(name, elem.clone()) {
            return false;
        }
        let Stmt::If { cond, yes, no } = &inner.body[2 * direction + 1] else {
            return false;
        };
        let [s] = yes.as_slice() else { return false };
        let Some(value) = assign(s, name) else {
            return false;
        };
        let Some(ix) = index(value, input) else {
            return false;
        };
        let axis = if direction < 2 { i } else { j };
        let extent = if direction < 2 { rows } else { cols };
        let positive = direction % 2 == 1;
        let shifted = |e: &Expr| {
            binary(
                e,
                if positive {
                    BinaryOp::Add
                } else {
                    BinaryOp::Sub
                },
            )
            .is_some_and(|(l, r)| named(l, axis) && uint(r, 1))
        };
        let guard = if positive {
            binary(cond, BinaryOp::Lt).is_some_and(|(l, r)| shifted(l) && named(r, extent))
        } else {
            binary(cond, BinaryOp::Gt).is_some_and(|(l, r)| named(l, axis) && uint(r, 0))
        };
        let offset = binary(ix, BinaryOp::Add).is_some_and(|(l, r)| {
            binary(l, BinaryOp::Mul).is_some_and(|(row, stride)| {
                named(stride, cols)
                    && if direction < 2 {
                        shifted(row) && named(r, j)
                    } else {
                        named(row, i) && shifted(r)
                    }
            })
        });
        let this_clamp = match no.as_slice() {
            [] => false,
            [s] if assign(s, name).is_some_and(center) => true,
            _ => return false,
        };
        if !guard || !offset || clamp.is_some_and(|c| c != this_clamp) {
            return false;
        }
        clamp = Some(this_clamp);
        names.push(name);
    }
    let Stmt::Store {
        array,
        index: dst,
        value,
        ..
    } = &inner.body[8]
    else {
        return false;
    };
    let mut sum = value;
    for name in names.iter().rev() {
        let Some((l, r)) = sum_step(sum, floating) else {
            return false;
        };
        if !named(r, name) {
            return false;
        }
        sum = l;
    }
    array == out && flat(dst, i, cols, j) && center(sum)
}

/// One step of a stencil sum: `l + r` in f32, `wrapping_add(l, r)` in u32.
fn sum_step(e: &Expr, floating: bool) -> Option<(&Expr, &Expr)> {
    if floating {
        return binary(e, BinaryOp::Add);
    }
    match call(e, "wrapping_add")? {
        [l, r] => Some((l, r)),
        _ => None,
    }
}
