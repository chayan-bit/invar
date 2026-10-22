//! Check physical quantities before erasing their compile-time-only unit tags.

use super::unit_math::Unit;
use super::*;

mod binary;
mod block;
mod expr;
mod frames;
mod physical;
use binary::check_binary_chain;
use block::check_block;
use expr::check_expr;
use frames::{frame_binary, proper_signed_permutation};
use physical::{FrameTag, PhysicalType, array_type};

#[derive(Clone)]
struct PhysicalSignature {
    params: Vec<PhysicalType>,
    result: PhysicalType,
}

struct PhysicalContext {
    functions: HashMap<String, PhysicalSignature>,
    records: HashMap<String, HashMap<String, PhysicalType>>,
    frames: HashSet<String>,
    frame_helpers_used: std::cell::Cell<bool>,
}

pub(super) fn check_and_erase(program: &mut Program) -> Result<(), FrontError> {
    let frames = program
        .frames
        .iter()
        .map(|frame| frame.name.clone())
        .collect::<HashSet<_>>();
    let functions = program
        .functions
        .iter()
        .map(|function| {
            Ok((
                function.name.clone(),
                PhysicalSignature {
                    params: function
                        .params
                        .iter()
                        .map(|param| PhysicalType::from_source(&param.ty, param.span, &frames))
                        .collect::<Result<_, _>>()?,
                    result: PhysicalType::from_source(&function.result, function.span, &frames)?,
                },
            ))
        })
        .collect::<Result<HashMap<_, _>, FrontError>>()?;

    let records = program
        .records
        .iter()
        .map(|record| {
            Ok((
                record.name.clone(),
                record
                    .fields
                    .iter()
                    .map(|field| {
                        let type_only = record.module.as_deref() == Some("std::robotics")
                            && matches!(&field.ty, Type::Named(name) if name.starts_with("Quantity<f32,") || name.starts_with("Quantity<f64,") || name.starts_with("Quantity<i64,"));
                        Ok((
                            field.name.clone(),
                            if type_only { PhysicalType::plain(field.ty.clone()) } else { PhysicalType::from_source(&field.ty, field.span, &frames)? },
                        ))
                    })
                    .collect::<Result<HashMap<_, _>, FrontError>>()?,
            ))
        })
        .collect::<Result<HashMap<_, _>, FrontError>>()?;
    let context = PhysicalContext {
        functions,
        records,
        frames,
        frame_helpers_used: std::cell::Cell::new(false),
    };

    for function in &mut program.functions {
        let result = context.functions[&function.name].result.clone();
        let mut scope = HashMap::new();
        for (param, ty) in function
            .params
            .iter_mut()
            .zip(&context.functions[&function.name].params)
        {
            if ty.unit.is_some() && ty.frame.is_none() && param.mode.is_some() {
                return Err(mismatch(
                    "quantities are scalar values and do not take a loan mode",
                    param.span,
                ));
            }
            scope.insert(param.name.clone(), ty.clone());
            param.ty = ty.scalar.clone();
        }
        check_block(&mut function.body, &mut scope, &result, &context)?;
        function.result = result.scalar;
    }
    for record in &mut program.records {
        for field in &mut record.fields {
            if record.module.as_deref() == Some("std::robotics")
                && matches!(&field.ty, Type::Named(name) if name.starts_with("Quantity<f32,") || name.starts_with("Quantity<f64,") || name.starts_with("Quantity<i64,"))
            {
                continue;
            }
            field.ty = PhysicalType::from_source(&field.ty, field.span, &context.frames)?.scalar;
        }
    }
    if context.frame_helpers_used.get() {
        let helpers = crate::front::parse(include_bytes!("frame_helpers.inv"))
            .map_err(|_| FrontError::tool("INTERNAL_TYPE", "built-in frame helpers are invalid"))?;
        program
            .functions
            .extend(helpers.functions.into_iter().map(|mut function| {
                function.name.insert(0, '$');
                function
            }));
    }
    Ok(())
}

pub(crate) fn is_frame_helper(name: &str) -> bool {
    matches!(
        name,
        "$vec3_i32"
            | "$vec3_u32"
            | "$vec_add_i32"
            | "$vec_sub_i32"
            | "$vec_add_u32"
            | "$vec_sub_u32"
            | "$xf_make"
            | "$xf_compose"
            | "$xf_point"
            | "$xf_vec"
    )
}

pub(crate) fn scale_call(name: &str) -> Option<(i64, i64)> {
    let (numerator, denominator) = name
        .strip_prefix("$scale$<")?
        .strip_suffix('>')?
        .split_once('|')?;
    let numerator = numerator.parse::<i64>().ok()?;
    let denominator = denominator.parse::<i64>().ok()?;
    (numerator > 0 && denominator > 0 && numerator <= 1_000_000_000 && denominator <= 1_000_000_000)
        .then_some((numerator, denominator))
}

pub(crate) fn angle_call(name: &str) -> Option<bool> {
    match name {
        "$angle$<d2r>" => Some(true),
        "$angle$<r2d>" => Some(false),
        _ => None,
    }
}
