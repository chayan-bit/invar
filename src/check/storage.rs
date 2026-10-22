//! Storage-only array element types of numeric policy v2 (#280,
//! `docs/numeric-v2.md`): `Array<f16>`, `Array<bf16>`, `Array<i8>` and
//! `Array<u8>`. A load widens exactly to `f32`, `i32` or `u32`, so no
//! arithmetic ever sees a narrow value. After typing, each storage array runs
//! as its widened array, and every value stored into it (an indexed store or a
//! `filled_*` fill) passes through one generated store conversion: f16 and
//! bf16 round to nearest even, i8 and u8 trap `INTEGER_OVERFLOW`
//! (`numeric_contract::storage`). Storage arrays stay inside one function for
//! now, so no host or caller can hand one an unconverted value.

use super::numeric::visit::{Visit, walk_block};
use super::*;

/// One storage element: its array type, the type a load widens to, its
/// constructor, and its store conversion. A store conversion name is not an
/// identifier, so source cannot call it.
struct Storage {
    array: &'static str,
    item: Type,
    wide: Type,
    filled: &'static str,
    store: &'static str,
}

static STORAGE: [Storage; 4] = [
    Storage {
        array: "Array<f16>",
        item: Type::F32,
        wide: Type::ArrayF32,
        filled: "filled_f16",
        store: "store-f16",
    },
    Storage {
        array: "Array<bf16>",
        item: Type::F32,
        wide: Type::ArrayF32,
        filled: "filled_bf16",
        store: "store-bf16",
    },
    Storage {
        array: "Array<i8>",
        item: Type::I32,
        wide: Type::ArrayI32,
        filled: "filled_i8",
        store: "store-i8",
    },
    Storage {
        array: "Array<u8>",
        item: Type::U32,
        wide: Type::ArrayU32,
        filled: "filled_u8",
        store: "store-u8",
    },
];

fn storage(ty: &Type) -> Option<&'static Storage> {
    let Type::Named(name) = ty else { return None };
    STORAGE.iter().find(|s| s.array == name)
}

/// Whether a named type is a storage array.
pub(super) fn is_array(name: &str) -> bool {
    STORAGE.iter().any(|s| s.array == name)
}

/// The type a load from a storage array widens to.
pub(super) fn element(name: &str) -> Option<Type> {
    STORAGE
        .iter()
        .find(|s| s.array == name)
        .map(|s| s.item.clone())
}

/// Every `filled_*` constructor: (fill value type, array type).
pub(super) fn filled(name: &str) -> Option<(Type, Type)> {
    Some(match name {
        "filled_i32" => (Type::I32, Type::ArrayI32),
        "filled_u32" => (Type::U32, Type::ArrayU32),
        "filled_f32" => (Type::F32, Type::ArrayF32),
        "filled_f64" => (Type::F64, Type::ArrayF64),
        _ => {
            let s = STORAGE.iter().find(|s| s.filled == name)?;
            (s.item.clone(), Type::Named(s.array.to_owned()))
        }
    })
}

/// A generated store conversion, typed like a float conversion: (input,
/// result), both the widened element type.
pub(crate) fn store(name: &str) -> Option<(Type, Type)> {
    let s = STORAGE.iter().find(|s| s.store == name)?;
    Some((s.item.clone(), s.item.clone()))
}

fn unsupported(message: &str, span: Span) -> FrontError {
    FrontError::source("UNSUPPORTED_FEATURE", message, span)
}

/// Refuse a storage array anywhere a value crosses a function boundary.
fn refuse_boundaries(program: &Program) -> Result<(), FrontError> {
    const CROSSING: &str = "a storage array (Array<f16|bf16|i8|u8>) stays inside one function for now: it cannot be a parameter, result, record field or variant payload";
    for function in &program.functions {
        if let Some(param) = function.params.iter().find(|p| storage(&p.ty).is_some()) {
            return Err(unsupported(CROSSING, param.span));
        }
        if storage(&function.result).is_some() {
            return Err(unsupported(CROSSING, function.span));
        }
    }
    for record in &program.records {
        if let Some(field) = record.fields.iter().find(|f| storage(&f.ty).is_some()) {
            return Err(unsupported(CROSSING, field.span));
        }
    }
    for variant in &program.variants {
        if variant
            .arms
            .iter()
            .any(|arm| arm.payload.as_ref().and_then(storage).is_some())
        {
            return Err(unsupported(CROSSING, variant.span));
        }
    }
    Ok(())
}

/// Lower every storage array to its widened array after typing: fills and
/// indexed stores convert, and every recorded type widens.
pub(super) fn lower(
    program: &mut Program,
    types: &mut [(Span, Type)],
    by_function: &mut HashMap<String, Vec<(Span, Type)>>,
) -> Result<(), FrontError> {
    refuse_boundaries(program)?;
    if !types.iter().any(|(_, ty)| storage(ty).is_some()) {
        return Ok(());
    }
    for function in &mut program.functions {
        let typed = by_function.get(&function.name);
        let mut lowering = Lowering {
            types: typed.into_iter().flatten().cloned().collect(),
            arrays: HashMap::new(),
            end: function.span.end,
            added: Vec::new(),
            error: None,
        };
        walk_block(&mut function.body, &mut lowering);
        if let Some(error) = lowering.error {
            return Err(error);
        }
        if !lowering.added.is_empty() {
            let typed = by_function.entry(function.name.clone()).or_default();
            typed.extend(lowering.added);
        }
        walk_block(&mut function.body, &mut Annotations);
    }
    types.iter_mut().for_each(|(_, ty)| widen(ty));
    by_function
        .values_mut()
        .flatten()
        .for_each(|(_, ty)| widen(ty));
    Ok(())
}

fn widen(ty: &mut Type) {
    if let Some(s) = storage(ty) {
        *ty = s.wide.clone();
    }
}

struct Lowering {
    types: HashMap<Span, Type>,
    /// The function's storage array bindings (names are unique per function).
    arrays: HashMap<String, &'static Storage>,
    end: usize,
    added: Vec<(Span, Type)>,
    error: Option<FrontError>,
}

impl Lowering {
    /// `value` passed through `store`. The call takes the value's start, so a
    /// trap points at the stored value, and the first free end after it.
    fn convert(&mut self, value: &mut Expr, s: &Storage) {
        let start = value.span.start;
        let Some(end) = (value.span.end + 1..=self.end)
            .find(|&end| !self.types.contains_key(&Span { start, end }))
        else {
            self.error = Some(unsupported(
                "no span is free for a storage store",
                value.span,
            ));
            return;
        };
        let span = Span { start, end };
        let call = ExprKind::Call {
            name: s.store.to_owned(),
            args: vec![value.clone()],
        };
        match Expr::new(call, span) {
            Ok(call) => *value = call,
            Err(error) => self.error = Some(error),
        }
        self.types.insert(span, s.item.clone());
        self.added.push((span, s.item.clone()));
    }
}

impl Visit for Lowering {
    fn stmt(&mut self, statement: &mut Stmt) {
        match &mut statement.kind {
            StmtKind::Binding {
                name,
                annotation,
                value,
                ..
            } => {
                let ty = annotation.as_ref().or_else(|| self.types.get(&value.span));
                if let Some(s) = ty.and_then(storage) {
                    self.arrays.insert(name.clone(), s);
                }
            }
            StmtKind::Assign {
                place: Place::Index { name, .. },
                value,
            } => {
                if let Some(s) = self.arrays.get(name).copied() {
                    self.convert(value, s);
                }
            }
            _ => {}
        }
    }

    fn expr(&mut self, e: &mut Expr) {
        let ExprKind::Call { name, args } = &mut e.kind else {
            return;
        };
        if let Some(s) = STORAGE.iter().find(|s| s.filled == name) {
            *name = s.wide_filled().to_owned();
            if let Some(value) = args.get_mut(1) {
                self.convert(value, s);
            }
        } else if matches!(name.as_str(), "view" | "view_mut")
            && let Some(owner) = args.first().and_then(|a| self.types.get(&a.span))
            && storage(owner).is_some()
        {
            self.error = Some(unsupported(
                "a storage array has no views yet: a view would store unconverted values",
                e.span,
            ));
        }
    }
}

impl Storage {
    fn wide_filled(&self) -> &'static str {
        match self.wide {
            Type::ArrayI32 => "filled_i32",
            Type::ArrayU32 => "filled_u32",
            _ => "filled_f32",
        }
    }
}

/// Widen every binding annotation.
struct Annotations;

impl Visit for Annotations {
    fn stmt(&mut self, statement: &mut Stmt) {
        if let StmtKind::Binding {
            annotation: Some(ty),
            ..
        } = &mut statement.kind
        {
            widen(ty);
        }
    }
}
