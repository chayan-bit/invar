//! Expression checking with erased unit tags.

use super::*;

pub(super) fn check_expr(
    expr: &mut Expr,
    scope: &HashMap<String, PhysicalType>,
    context: &PhysicalContext,
    expected: Option<&PhysicalType>,
) -> Result<PhysicalType, FrontError> {
    if matches!(&expr.kind, ExprKind::Binary { .. }) {
        return check_binary_chain(expr, scope, context, expected);
    }
    let ty = match &mut expr.kind {
        ExprKind::Float(raw) => PhysicalType::plain(
            if raw.ends_with("f64")
                || (!raw.ends_with("f32") && expected.is_some_and(|ty| ty.scalar == Type::F64))
            {
                Type::F64
            } else {
                Type::F32
            },
        ),
        ExprKind::Integer(raw) => PhysicalType::plain(
            if raw.ends_with("u32") || expected.is_some_and(|ty| ty.scalar == Type::U32) {
                Type::U32
            } else {
                Type::I32
            },
        ),
        ExprKind::Bool(_) => PhysicalType::plain(Type::Bool),
        ExprKind::Text(_) => PhysicalType::plain(Type::Named("String".to_owned())),
        ExprKind::Bytes(_) => PhysicalType::plain(Type::Named("Bytes".to_owned())),
        ExprKind::Name(name) | ExprKind::Move(name) => {
            if !scope.contains_key(name) && Unit::builtin(name).is_some() {
                return Err(FrontError::source(
                    "UNIT_AS_VALUE",
                    format!("unit '{name}' is a type, not a value"),
                    expr.span,
                ));
            }
            scope
                .get(name)
                .cloned()
                .unwrap_or_else(|| PhysicalType::plain(Type::Unit))
        }
        ExprKind::Group(inner) => check_expr(inner, scope, context, expected)?,
        ExprKind::Unary { value, op, .. } => {
            let inner = check_expr(value, scope, context, expected)?;
            if matches!(op, UnaryOp::Not) && (inner.unit.is_some() || inner.frame.is_some()) {
                return Err(FrontError::source(
                    "UNIT_MISMATCH",
                    "logical not requires Bool",
                    expr.span,
                ));
            }
            inner
        }
        ExprKind::Call { name, args }
            if name.starts_with("vector3$<") || name.starts_with("point$<") =>
        {
            let vector = name.starts_with("vector3$<");
            let content = name
                .split_once("$<")
                .and_then(|(_, value)| value.strip_suffix('>'))
                .ok_or_else(|| mismatch("invalid geometric constructor", expr.span))?;
            let (unit, frame) = if vector {
                let (unit, frame) = content
                    .split_once('|')
                    .ok_or_else(|| mismatch("vector3 needs unit and frame", expr.span))?;
                (
                    Unit::parse(unit).ok_or_else(|| {
                        FrontError::source("UNIT_UNKNOWN", "unknown vector unit", expr.span)
                    })?,
                    frame,
                )
            } else {
                (Unit::builtin("um").unwrap(), content)
            };
            let frame = frame.to_owned();
            if !context.frames.contains(&frame) {
                return Err(FrontError::source(
                    "FRAME_UNKNOWN",
                    "coordinate frame is not declared",
                    expr.span,
                ));
            }
            if args.len() != 3 {
                return Err(mismatch(
                    "geometric constructor needs three coordinates",
                    expr.span,
                ));
            }
            let scalar = if expected.is_some_and(|ty| ty.scalar == Type::ArrayU32)
                || args
                    .iter()
                    .any(|arg| matches!(&arg.kind, ExprKind::Integer(raw) if raw.ends_with("u32")))
            {
                Type::U32
            } else {
                Type::I32
            };
            let hint = PhysicalType::plain(scalar.clone());
            for arg in args.iter_mut() {
                let actual = check_expr(arg, scope, context, Some(&hint))?;
                if actual.unit.is_some() || actual.frame.is_some() || actual.scalar != scalar {
                    return Err(FrontError::source(
                        "UNIT_MISMATCH",
                        "geometric coordinate must be a plain matching integer",
                        arg.span,
                    ));
                }
            }
            let helper = if scalar == Type::U32 {
                "$vec3_u32"
            } else {
                "$vec3_i32"
            };
            let rewritten_args = std::mem::take(args);
            expr.kind = ExprKind::Call {
                name: helper.to_owned(),
                args: rewritten_args,
            };
            context.frame_helpers_used.set(true);
            PhysicalType {
                scalar: array_type(&scalar),
                unit: Some(unit),
                frame: Some(if vector {
                    FrameTag::Vector(frame)
                } else {
                    FrameTag::Point(frame)
                }),
            }
        }
        ExprKind::Call { name, args } if name.starts_with("transform$<") => {
            let content = name
                .strip_prefix("transform$<")
                .and_then(|value| value.strip_suffix('>'))
                .ok_or_else(|| mismatch("invalid transform constructor", expr.span))?;
            let (from, to) = content
                .split_once('|')
                .ok_or_else(|| mismatch("transform needs source and target frames", expr.span))?;
            let (from, to) = (from.to_owned(), to.to_owned());
            if !context.frames.contains(&from) || !context.frames.contains(&to) {
                return Err(FrontError::source(
                    "FRAME_UNKNOWN",
                    "transform frame is not declared",
                    expr.span,
                ));
            }
            if args.len() != 12 {
                return Err(mismatch(
                    "transform needs translation and nine rotation entries",
                    expr.span,
                ));
            }
            if !proper_signed_permutation(&args[3..]) {
                return Err(FrontError::source(
                    "ROTATION_INVALID",
                    "rotation must be a proper signed-permutation matrix",
                    expr.span,
                ));
            }
            let hint = PhysicalType::plain(Type::I32);
            for arg in args.iter_mut() {
                let actual = check_expr(arg, scope, context, Some(&hint))?;
                if actual.unit.is_some() || actual.frame.is_some() || actual.scalar != Type::I32 {
                    return Err(FrontError::source(
                        "UNIT_MISMATCH",
                        "transform entries must be plain i32 values",
                        arg.span,
                    ));
                }
            }
            let rewritten_args = std::mem::take(args);
            expr.kind = ExprKind::Call {
                name: "$xf_make".to_owned(),
                args: rewritten_args,
            };
            context.frame_helpers_used.set(true);
            PhysicalType {
                scalar: Type::ArrayI32,
                unit: None,
                frame: Some(FrameTag::Transform(from, to)),
            }
        }
        ExprKind::Call { name, args } if name.starts_with("quantity$<") => {
            let unit = name
                .strip_prefix("quantity$<")
                .and_then(|name| name.strip_suffix('>'))
                .and_then(Unit::parse)
                .ok_or_else(|| {
                    FrontError::source("UNIT_UNKNOWN", "unknown quantity unit", expr.span)
                })?;
            if args.len() != 1 {
                return Err(mismatch(
                    "quantity constructor expects one value",
                    expr.span,
                ));
            }
            let scalar_hint = expected.map(|ty| PhysicalType::plain(ty.scalar.clone()));
            let inner = check_expr(&mut args[0], scope, context, scalar_hint.as_ref())?;
            if inner.unit.is_some()
                || inner.frame.is_some()
                || !(is_integer(&inner.scalar) || inner.scalar == Type::F32)
            {
                return Err(mismatch(
                    "quantity constructor requires an integer or f32 value",
                    expr.span,
                ));
            }
            let replacement = args.remove(0);
            expr.kind = replacement.kind;
            expr.span = replacement.span;
            PhysicalType {
                scalar: inner.scalar,
                unit: Some(unit),
                frame: None,
            }
        }
        ExprKind::Call { name, args } if name.starts_with("to$<") => {
            let target = name
                .strip_prefix("to$<")
                .and_then(|name| name.strip_suffix('>'))
                .and_then(Unit::parse)
                .ok_or_else(|| {
                    FrontError::source("UNIT_UNKNOWN", "unknown conversion target unit", expr.span)
                })?;
            if args.len() != 1 {
                return Err(mismatch("to expects one quantity", expr.span));
            }
            let source = check_expr(&mut args[0], scope, context, None)?;
            if source.frame.is_some() {
                return Err(FrontError::source(
                    "UNSUPPORTED_FEATURE",
                    "vector scale conversion is not available in this subset",
                    expr.span,
                ));
            }
            let Some(source_unit) = source.unit else {
                return Err(mismatch("to requires a quantity", expr.span));
            };
            if source_unit.dimension != target.dimension {
                return Err(FrontError::source(
                    "UNIT_MISMATCH",
                    "conversion target has a different dimension",
                    expr.span,
                ));
            }
            if source.scalar == Type::F32 && source_unit != target {
                return Err(FrontError::source(
                    "UNSUPPORTED_FEATURE",
                    "f32 quantity scale conversion is not implemented",
                    expr.span,
                ));
            }
            let replacement = args.remove(0);
            let deg = Unit::builtin("deg").unwrap();
            let rad = Unit::builtin("rad").unwrap();
            if source_unit == deg && target == rad {
                expr.kind = ExprKind::Call {
                    name: "$angle$<d2r>".to_owned(),
                    args: vec![replacement],
                };
            } else if source_unit == rad && target == deg {
                expr.kind = ExprKind::Call {
                    name: "$angle$<r2d>".to_owned(),
                    args: vec![replacement],
                };
            } else {
                let factor = source_unit.conversion_to(target).ok_or_else(|| {
                    FrontError::source(
                        "UNSUPPORTED_FEATURE",
                        "only explicit deg/rad uses the irrational angle conversion",
                        expr.span,
                    )
                })?;
                if factor == super::unit_math::Rational::ONE {
                    expr.kind = replacement.kind;
                    expr.span = replacement.span;
                    return Ok(PhysicalType {
                        scalar: source.scalar,
                        unit: Some(target),
                        frame: None,
                    });
                }
                let numerator = i64::try_from(factor.numerator).ok();
                let denominator = i64::try_from(factor.denominator).ok();
                let (Some(numerator), Some(denominator)) = (numerator, denominator) else {
                    return Err(FrontError::source(
                        "RESOURCE_LIMIT",
                        "conversion factor exceeds its bound",
                        expr.span,
                    ));
                };
                if numerator > 1_000_000_000 || denominator > 1_000_000_000 {
                    return Err(FrontError::source(
                        "RESOURCE_LIMIT",
                        "conversion factor exceeds one billion",
                        expr.span,
                    ));
                }
                expr.kind = ExprKind::Call {
                    name: format!("$scale$<{numerator}|{denominator}>"),
                    args: vec![replacement],
                };
            }
            PhysicalType {
                scalar: source.scalar,
                unit: Some(target),
                frame: None,
            }
        }
        ExprKind::Call { name, args } => {
            let signature = context.functions.get(name);
            for (index, arg) in args.iter_mut().enumerate() {
                let expected_arg = signature.and_then(|sig| sig.params.get(index));
                let actual = check_expr(arg, scope, context, expected_arg)?;
                if let Some(expected_arg) = expected_arg {
                    actual.require(expected_arg, arg.span)?;
                } else if actual.frame.is_some() {
                    return Err(FrontError::source(
                        "FRAME_MISMATCH",
                        "geometric value passed to an untyped intrinsic or unknown function",
                        arg.span,
                    ));
                } else if actual.unit.is_some() || actual.frame.is_some() {
                    return Err(FrontError::source(
                        "UNIT_MISMATCH",
                        "quantity passed to an untyped intrinsic or unknown function",
                        arg.span,
                    ));
                }
            }
            signature.map_or_else(|| PhysicalType::plain(Type::Unit), |sig| sig.result.clone())
        }
        ExprKind::Binary { .. } => unreachable!("binary expression handled before match"),
        ExprKind::Index { base, index } => {
            let base_ty = check_expr(base, scope, context, None)?;
            let index_ty = check_expr(index, scope, context, None)?;
            if index_ty.unit.is_some() || index_ty.frame.is_some() {
                return Err(FrontError::source(
                    "UNIT_MISMATCH",
                    "array index cannot be a quantity or geometric value",
                    expr.span,
                ));
            }
            match base_ty.frame {
                Some(FrameTag::Vector(_) | FrameTag::Point(_)) => PhysicalType {
                    scalar: if base_ty.scalar == Type::ArrayU32 {
                        Type::U32
                    } else {
                        Type::I32
                    },
                    unit: base_ty.unit,
                    frame: None,
                },
                Some(FrameTag::Transform(..)) => {
                    return Err(FrontError::source(
                        "FRAME_MISMATCH",
                        "transform internals are not directly indexable",
                        expr.span,
                    ));
                }
                None if base_ty.unit.is_some() => {
                    return Err(FrontError::source(
                        "UNIT_MISMATCH",
                        "scalar quantity is not indexable",
                        expr.span,
                    ));
                }
                None => PhysicalType::plain(Type::Unit),
            }
        }
        ExprKind::Array(items) => {
            for item in items {
                let actual = check_expr(item, scope, context, None)?;
                if actual.unit.is_some() || actual.frame.is_some() {
                    return Err(FrontError::source(
                        "UNIT_MISMATCH",
                        "plain arrays cannot contain quantities or geometric values",
                        item.span,
                    ));
                }
            }
            PhysicalType::plain(Type::Unit)
        }
        ExprKind::Record { fields, name } => {
            let declared = context.records.get(name);
            for field in fields {
                let expected_field = declared.and_then(|record| record.get(&field.name));
                let actual = check_expr(&mut field.value, scope, context, expected_field)?;
                if let Some(expected_field) = expected_field {
                    actual.require(expected_field, field.value.span)?;
                } else if actual.unit.is_some() {
                    return Err(FrontError::source(
                        "UNIT_MISMATCH",
                        "quantity field has no declared unit",
                        field.span,
                    ));
                }
            }
            PhysicalType::plain(Type::Named(name.clone()))
        }
        ExprKind::Field { base, name } | ExprKind::MoveField { base, field: name } => {
            let base_ty = check_expr(base, scope, context, None)?;
            if let Type::Named(record_name) = base_ty.scalar {
                context
                    .records
                    .get(&record_name)
                    .and_then(|record| record.get(name))
                    .cloned()
                    .unwrap_or_else(|| PhysicalType::plain(Type::Unit))
            } else {
                PhysicalType::plain(Type::Unit)
            }
        }
        ExprKind::Try(base) => {
            let inner = check_expr(base, scope, context, None)?;
            if inner.unit.is_some() || inner.frame.is_some() {
                return Err(mismatch(
                    "Result propagation requires a Result value",
                    expr.span,
                ));
            }
            PhysicalType::plain(Type::Unit)
        }
        ExprKind::Match { value, arms } => {
            let scrutinee = check_expr(value, scope, context, None)?;
            if scrutinee.unit.is_some() || scrutinee.frame.is_some() {
                return Err(mismatch("match value cannot be a quantity", value.span));
            }
            let mut common = expected.cloned();
            for arm in arms {
                let actual = check_expr(&mut arm.value, scope, context, common.as_ref())?;
                if let Some(common) = &common {
                    actual.require(common, arm.value.span)?;
                } else {
                    common = Some(actual);
                }
            }
            common.unwrap_or_else(|| PhysicalType::plain(Type::Unit))
        }
    };
    if let Some(expected) = expected {
        ty.clone().require(expected, expr.span)?;
    }
    Ok(ty)
}
