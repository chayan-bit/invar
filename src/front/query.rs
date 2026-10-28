//! Small queries on the AST.

use super::*;

/// Strips any number of enclosing `Group` nodes.
pub fn ungroup(mut e: &Expr) -> &Expr {
    while let ExprKind::Group(inner) = &e.kind {
        e = inner;
    }
    e
}

/// True when `e`, ignoring grouping, is the plain name `name`.
pub fn named(e: &Expr, name: &str) -> bool {
    matches!(&ungroup(e).kind, ExprKind::Name(n) if n == name)
}

impl Program {
    /// The first function named `name`.
    pub fn function(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|f| f.name == name)
    }
}

impl Type {
    /// The type as written in source (`Array<i32>`, `Bool`); a named type is its stored name.
    pub fn source_name(&self) -> &str {
        match self {
            Type::I32 => "i32",
            Type::U32 => "u32",
            Type::F32 => "f32",
            Type::F64 => "f64",
            Type::Bool => "Bool",
            Type::Unit => "Unit",
            Type::ArrayI32 => "Array<i32>",
            Type::ArrayU32 => "Array<u32>",
            Type::ArrayF32 => "Array<f32>",
            Type::ArrayF64 => "Array<f64>",
            Type::ArrayBool => "Array<Bool>",
            Type::Named(name) => name,
        }
    }
}
