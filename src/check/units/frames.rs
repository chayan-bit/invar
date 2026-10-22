//! Frame arithmetic: rotation validation and vector, point and transform operators.

use super::*;

fn rotation_entry(expr: &Expr) -> Option<i32> {
    match &expr.kind {
        ExprKind::Integer(raw) => raw.trim_end_matches("i32").parse::<i32>().ok(),
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
            ..
        } => rotation_entry(value)?.checked_neg(),
        _ => None,
    }
}

pub(super) fn proper_signed_permutation(rotation: &[Expr]) -> bool {
    if rotation.len() != 9 {
        return false;
    }
    let Some(values) = rotation
        .iter()
        .map(rotation_entry)
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    if values.iter().any(|value| !(-1..=1).contains(value)) {
        return false;
    }
    for row in 0..3 {
        if (0..3)
            .filter(|column| values[row * 3 + column] != 0)
            .count()
            != 1
            || (0..3)
                .filter(|column| values[column * 3 + row] != 0)
                .count()
                != 1
        {
            return false;
        }
    }
    let determinant = values[0] * (values[4] * values[8] - values[5] * values[7])
        - values[1] * (values[3] * values[8] - values[5] * values[6])
        + values[2] * (values[3] * values[7] - values[4] * values[6]);
    determinant == 1
}

pub(super) fn frame_binary(
    op: BinaryOp,
    left: &PhysicalType,
    right: &PhysicalType,
    span: Span,
) -> Result<(&'static str, PhysicalType), FrontError> {
    use FrameTag::{Point, Transform, Vector};
    let mismatch_frame = || {
        FrontError::source(
            "FRAME_MISMATCH",
            "geometric operation needs matching source and target frames",
            span,
        )
    };
    let same_coordinates = || {
        if left.scalar != right.scalar || left.unit != right.unit {
            Err(FrontError::source(
                "UNIT_MISMATCH",
                "geometric coordinate scales or scalars differ",
                span,
            ))
        } else {
            Ok(())
        }
    };
    match (op, left.frame.as_ref(), right.frame.as_ref()) {
        (BinaryOp::Add | BinaryOp::Sub, Some(Vector(a)), Some(Vector(b))) if a == b => {
            same_coordinates()?;
            let signed = left.scalar == Type::ArrayI32;
            let name = match (op, signed) {
                (BinaryOp::Add, true) => "$vec_add_i32",
                (BinaryOp::Add, false) => "$vec_add_u32",
                (BinaryOp::Sub, true) => "$vec_sub_i32",
                _ => "$vec_sub_u32",
            };
            Ok((name, left.clone()))
        }
        (BinaryOp::Add | BinaryOp::Sub, Some(Point(a)), Some(Vector(b))) if a == b => {
            same_coordinates()?;
            let signed = left.scalar == Type::ArrayI32;
            let name = match (op, signed) {
                (BinaryOp::Add, true) => "$vec_add_i32",
                (BinaryOp::Add, false) => "$vec_add_u32",
                (BinaryOp::Sub, true) => "$vec_sub_i32",
                _ => "$vec_sub_u32",
            };
            Ok((name, left.clone()))
        }
        (BinaryOp::Add, Some(Vector(a)), Some(Point(b))) if a == b => {
            same_coordinates()?;
            let name = if left.scalar == Type::ArrayI32 {
                "$vec_add_i32"
            } else {
                "$vec_add_u32"
            };
            Ok((name, right.clone()))
        }
        (BinaryOp::Sub, Some(Point(a)), Some(Point(b))) if a == b => {
            same_coordinates()?;
            let mut result = left.clone();
            result.frame = Some(Vector(a.clone()));
            Ok((
                if left.scalar == Type::ArrayI32 {
                    "$vec_sub_i32"
                } else {
                    "$vec_sub_u32"
                },
                result,
            ))
        }
        (BinaryOp::Mul, Some(Transform(a, b)), Some(Transform(c, d))) if b == c => Ok((
            "$xf_compose",
            PhysicalType {
                scalar: Type::ArrayI32,
                unit: None,
                frame: Some(Transform(a.clone(), d.clone())),
            },
        )),
        (BinaryOp::Mul, Some(Transform(a, b)), Some(Point(c)))
            if a == c && right.scalar == Type::ArrayI32 =>
        {
            let mut result = right.clone();
            result.frame = Some(Point(b.clone()));
            Ok(("$xf_point", result))
        }
        (BinaryOp::Mul, Some(Transform(a, b)), Some(Vector(c)))
            if a == c && right.scalar == Type::ArrayI32 =>
        {
            let mut result = right.clone();
            result.frame = Some(Vector(b.clone()));
            Ok(("$xf_vec", result))
        }
        _ => Err(mismatch_frame()),
    }
}
