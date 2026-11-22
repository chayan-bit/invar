//! Loop regions (#355, `docs/spec/invar-v0.2-regions.md` section 9): a
//! `#[region(...)]` before a loop declares that loop and the loops nested in
//! it, and the innermost declaration governs a region.

use super::{Axes, Axis};
use crate::front::{FrontError, Span};

fn within(inner: Span, outer: Span) -> bool {
    outer.start <= inner.start && inner.end <= outer.end
}

/// The levels governing the loop region of `function` at `span`: the innermost
/// declared loop containing it, else the function's declaration, else the
/// safe levels.
pub fn region_levels(program: &crate::mir::Program, function: &str, span: Span) -> Axes {
    let loops = program.loop_axes.get(function).into_iter().flatten();
    let innermost = loops
        .filter(|(declared, _)| within(span, *declared))
        .min_by_key(|(declared, _)| declared.end - declared.start);
    innermost.map_or_else(
        || {
            program
                .region_axes
                .get(function)
                .copied()
                .unwrap_or_default()
        },
        |(_, axes)| *axes,
    )
}

/// Every declaration of `program`: each function's own levels (safe when
/// undeclared) and each declared loop's.
pub fn declarations(program: &crate::mir::Program) -> impl Iterator<Item = Axes> + Clone + '_ {
    let functions = program.functions.iter().map(|function| {
        program
            .region_axes
            .get(&function.name)
            .copied()
            .unwrap_or_default()
    });
    functions.chain(program.loop_axes.values().flatten().map(|(_, axes)| *axes))
}

/// Refuse the first declared loop above a CLI cap, at its attribute.
pub(super) fn check_caps(
    function: &crate::front::Function,
    caps: &[(&Axis, &str)],
) -> Result<(), FrontError> {
    for (_, attribute, axes) in &function.loop_axes {
        if let Some((key, level, cap)) = axes.over_cap(caps) {
            return Err(FrontError::source(
                "CONTRACT_CAP",
                format!(
                    "a loop in function {} declares {level}, above the cap --max-{key} {cap}",
                    function.name
                ),
                *attribute,
            ));
        }
    }
    Ok(())
}
