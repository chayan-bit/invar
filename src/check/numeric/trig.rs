//! `--numeric-contract trig-f32-v1` (#210 C4, `math::select_trig_f32_v1`):
//! after typing, every f32 `sin` and `cos` call takes the name of its
//! `invar-trig-f32-v1` helper, so MIR, explain and receipts record the
//! contract the program ran under. f64 calls and every other function keep
//! `invar-math-cr-v1`.

use super::visit::{Visit, walk_block};
use super::*;

pub(super) fn select(program: &mut Program, types: &HashMap<String, Vec<(Span, Type)>>) {
    if !crate::math::trig_f32_v1_selected() {
        return;
    }
    for function in &mut program.functions {
        let f32_spans = types
            .get(&function.name)
            .into_iter()
            .flatten()
            .filter(|(_, ty)| *ty == Type::F32)
            .map(|(span, _)| *span)
            .collect();
        walk_block(&mut function.body, &mut Rename(f32_spans));
    }
}

/// The spans of the function's f32 expressions.
struct Rename(HashSet<Span>);

impl Visit for Rename {
    fn expr(&mut self, e: &mut Expr) {
        if let ExprKind::Call { name, .. } = &mut e.kind
            && self.0.contains(&e.span)
        {
            match name.as_str() {
                "sin" => *name = crate::math::TRIG_F32_SIN.to_owned(),
                "cos" => *name = crate::math::TRIG_F32_COS.to_owned(),
                _ => {}
            }
        }
    }
}
