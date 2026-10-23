//! Name and type checks for the bootstrap source tree.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::front::{
    BinaryOp, Block, ElseArm, Expr, ExprKind, FrontError, Function, MatchArm, Mode, Place, Program,
    Span, Stmt, StmtKind, Type, UnaryOp,
};

#[derive(Debug)]
pub struct CheckedProgram {
    pub program: Program,
    /// Expression types in source traversal order, with original byte spans.
    pub expression_types: Vec<(Span, Type)>,
    pub function_expression_types: HashMap<String, Vec<(Span, Type)>>,
    pub lowered_folds: crate::numeric_contract::LoweredFolds,
}

impl CheckedProgram {
    pub(crate) fn types_for(&self, function: &str) -> HashMap<Span, Type> {
        self.function_expression_types
            .get(function)
            .into_iter()
            .flatten()
            .cloned()
            .collect()
    }
}

#[derive(Clone)]
struct Signature {
    params: Vec<Type>,
    result: Type,
}

struct Checker {
    functions: HashMap<String, Signature>,
    records: HashMap<String, Vec<(String, Type)>>,
    variants: HashMap<String, Vec<(String, Option<Type>)>>,
    expression_types: Vec<(Span, Type)>,
    function_expression_types: HashMap<String, Vec<(Span, Type)>>,
    generic_templates: HashMap<String, Function>,
    pending_instances: Vec<Function>,
    generic_instances: HashSet<String>,
    resolved_calls: HashMap<(String, Span), String>,
    call_edges: BTreeMap<String, BTreeSet<String>>,
    current_function: String,
    current_module: Option<String>,
    /// Call spans where `tc_mul_add` is admitted: only the accumulation step
    /// of a recognized matrix contraction, pinned by its recognizer.
    tc_steps: Vec<Span>,
}

#[derive(Clone)]
struct Binding {
    ty: Type,
    mutable: bool,
    mode: Option<Mode>,
}

struct FunctionChecker<'a> {
    checker: &'a mut Checker,
    scopes: Vec<HashMap<String, Binding>>,
    declared: HashSet<String>,
    result: Type,
}

fn mismatch(message: impl Into<String>, span: Span) -> FrontError {
    FrontError::source("TYPE_MISMATCH", message, span)
}

fn is_integer(ty: &Type) -> bool {
    matches!(ty, Type::I32 | Type::U32)
}

fn is_float(ty: &Type) -> bool {
    matches!(ty, Type::F32 | Type::F64)
}

fn is_numeric(ty: &Type) -> bool {
    is_integer(ty) || is_float(ty)
}

fn is_array(ty: &Type) -> bool {
    matches!(
        ty,
        Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool
    )
}

/// Fixed-signature float conversions: (input, result).
pub(crate) fn conversion(name: &str) -> Option<(Type, Type)> {
    Some(match name {
        "f32_from_i32" => (Type::I32, Type::F32),
        "f32_from_u32" => (Type::U32, Type::F32),
        "f64_from_i32" => (Type::I32, Type::F64),
        "f64_from_u32" => (Type::U32, Type::F64),
        "f64_from_f32" => (Type::F32, Type::F64),
        "f32_from_f64" => (Type::F64, Type::F32),
        "i32_from_f32" => (Type::F32, Type::I32),
        "u32_from_f32" => (Type::F32, Type::U32),
        "i32_from_f64" => (Type::F64, Type::I32),
        "u32_from_f64" => (Type::F64, Type::U32),
        // v0.1 spelling of `i32_from_f32`, kept for existing sources.
        "trunc_i32" => (Type::F32, Type::I32),
        _ => return storage::store(name),
    })
}

/// An unsuffixed decimal literal, possibly grouped or negated, whose float
/// type comes from context (default `f32`).
fn flexible_float(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Float(raw) => crate::numeric_contract::decimal_digits(raw).1.is_none(),
        ExprKind::Group(inner) => flexible_float(inner),
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
            ..
        } => flexible_float(value),
        ExprKind::Binary {
            op: BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div,
            left,
            right,
            ..
        } => flexible_float(left) || flexible_float(right),
        _ => false,
    }
}

fn element(ty: &Type) -> Option<Type> {
    match ty {
        Type::ArrayI32 => Some(Type::I32),
        Type::ArrayU32 => Some(Type::U32),
        Type::ArrayF32 => Some(Type::F32),
        Type::ArrayF64 => Some(Type::F64),
        Type::ArrayBool => Some(Type::Bool),
        Type::Named(name) if name == "Bytes" => Some(Type::U32),
        Type::Named(name) => crate::front::view_parts(name)
            .map(|(element, _)| element)
            .or_else(|| storage::element(name)),
        _ => None,
    }
}

pub(crate) fn intrinsic(name: &str) -> bool {
    crate::spatial::result(name).is_some()
        || sort_modes(name).is_some()
        || matches!(
            name,
            "len"
                | "view"
                | "view_mut"
                | "fma"
                | "sin"
                | "cos"
                | "sqrt"
                | "tc_mul_add"
                | "wrapping_add"
                | "wrapping_mul"
                | "bytes_slice"
                | "utf8_slice"
                | "utf8_decode"
                | "utf8_encode"
                | "quantity"
                | "to"
                | "sum_tree"
                | "dot_tree"
                | "sum_exact"
                | "dot_exact"
                | "reassoc_add"
        )
        || conversion(name).is_some()
        || storage::filled(name).is_some()
        || crate::math::MathFn::from_name(name).is_some()
}

/// Call-scoped loans for the reserved sort intrinsics, in argument order.
pub(crate) fn sort_modes(name: &str) -> Option<&'static [Mode]> {
    match name {
        "sort" => Some(&[Mode::Mut]),
        "sort_by_key" => Some(&[Mode::Mut, Mode::Mut]),
        "argsort" => Some(&[Mode::Read]),
        "segmented_sort" => Some(&[Mode::Mut, Mode::Read]),
        _ => None,
    }
}

/// Check names, types, and definite returns. Array ownership and loans are later gates.
/// `require_main` is false for library-style `check`, true for an executable entrypoint.
pub fn check(program: Program, require_main: bool) -> Result<CheckedProgram, FrontError> {
    check_impl(program, require_main, false, None)
}

/// Check wire-only robotics declarations without producing an executable program.
pub fn check_port_types(program: Program) -> Result<(), FrontError> {
    check_impl(program, false, true, Some(&[])).map(|_| ())
}

fn check_impl(
    program: Program,
    require_main: bool,
    allow_type_only: bool,
    tc_steps: Option<&[Span]>,
) -> Result<CheckedProgram, FrontError> {
    if !allow_type_only && let Some(item) = program.system_items.first() {
        return Err(FrontError::source(
            "UNSUPPORTED_FEATURE",
            "system declarations require invar system check",
            item.span,
        ));
    }
    let mut program = program;
    let mut standard = crate::front::parse(include_bytes!("../lib/std/robotics/vocabulary.inv"))
        .map_err(|_| {
            FrontError::tool("INTERNAL_TYPE", "standard robotics vocabulary is invalid")
        })?;
    let mut planner = crate::front::parse(include_bytes!("../lib/std/robotics/planner-v1.inv"))
        .map_err(|_| FrontError::tool("INTERNAL_TYPE", "standard planner vocabulary is invalid"))?;
    // Both files contribute records to the same standard module, not duplicate modules.
    standard.records.append(&mut planner.records);
    program.modules.append(&mut standard.modules);
    program.records.append(&mut standard.records);
    frames::validate(&program.frames)?;
    closures::desugar(&mut program)?;
    let numeric_sites = numeric::desugar(&mut program)?;
    resolve_modules(&mut program)?;
    record_generics::expand(&mut program)?;
    synthesize_results(&mut program)?;
    units::check_and_erase(&mut program)?;
    let mut generic_templates = HashMap::new();
    let mut concrete_functions = Vec::new();
    for function in std::mem::take(&mut program.functions) {
        if function.generic_params.is_empty() {
            concrete_functions.push(function);
        } else {
            generics::validate_template(&function)?;
            generic_templates.insert(function.name.clone(), function);
        }
    }
    if concrete_functions.len() > generics::MAX_FUNCTIONS {
        return Err(FrontError::source(
            "RESOURCE_LIMIT",
            "lowered function limit exceeds 128",
            concrete_functions[generics::MAX_FUNCTIONS].span,
        ));
    }
    program.functions = concrete_functions;
    let record_names: HashSet<&str> = program
        .records
        .iter()
        .map(|record| record.name.as_str())
        .collect();
    let mut records = HashMap::new();
    for record in &program.records {
        for field in &record.fields {
            if !matches!(
                field.ty,
                Type::I32
                    | Type::U32
                    | Type::F32
                    | Type::F64
                    | Type::Bool
                    | Type::ArrayI32
                    | Type::ArrayU32
                    | Type::ArrayF32
                    | Type::ArrayF64
                    | Type::ArrayBool
            ) && !matches!(&field.ty, Type::Named(name) if record.module.as_deref() == Some("std::robotics") && (record_names.contains(name.as_str()) || matches!(name.as_str(), "f32" | "f64" | "i64") || name.starts_with("FixedArray<") || name.starts_with("Quantity<f32,") || name.starts_with("Quantity<f64,") || name.starts_with("Quantity<i64,")))
            {
                return Err(FrontError::source(
                    "UNSUPPORTED_FEATURE",
                    "record field type is not supported",
                    field.span,
                ));
            }
        }
        records.insert(
            record.name.clone(),
            record
                .fields
                .iter()
                .map(|field| (field.name.clone(), field.ty.clone()))
                .collect(),
        );
    }
    records.extend(crate::spatial::records());
    let mut variants = HashMap::new();
    for variant in &program.variants {
        for arm in &variant.arms {
            if crate::front::result_parts(&variant.name).is_none()
                && arm.payload.as_ref().is_some_and(|ty| {
                    !matches!(
                        ty,
                        Type::I32 | Type::U32 | Type::F32 | Type::F64 | Type::Bool
                    )
                })
            {
                return Err(FrontError::source(
                    "UNSUPPORTED_FEATURE",
                    "this variant subset requires scalar payloads",
                    arm.span,
                ));
            }
        }
        variants.insert(
            variant.name.clone(),
            variant
                .arms
                .iter()
                .map(|arm| (arm.name.clone(), arm.payload.clone()))
                .collect(),
        );
    }
    let mut functions = HashMap::new();
    for function in &program.functions {
        let base = function.name.rsplit("::").next().unwrap_or(&function.name);
        if crate::math::MathFn::from_name(base).is_some() {
            return Err(FrontError::source(
                "NAME_DUPLICATE",
                format!(
                    "function name '{base}' is reserved for the {} library function '{base}'",
                    crate::math::CONTRACT
                ),
                function.span,
            ));
        }
        if intrinsic(base) || functions.contains_key(&function.name) {
            return Err(FrontError::source(
                "NAME_DUPLICATE",
                format!("duplicate or reserved function '{}'", function.name),
                function.span,
            ));
        }
        functions.insert(
            function.name.clone(),
            Signature {
                params: function
                    .params
                    .iter()
                    .map(|param| param.ty.clone())
                    .collect(),
                result: function.result.clone(),
            },
        );
    }
    for foreign in &program.foreign_functions {
        if intrinsic(&foreign.name) || functions.contains_key(&foreign.name) {
            return Err(FrontError::source(
                "NAME_DUPLICATE",
                "duplicate or reserved foreign function",
                foreign.span,
            ));
        }
        if foreign
            .params
            .iter()
            .map(|param| param.name.as_str())
            .collect::<HashSet<_>>()
            .len()
            != foreign.params.len()
        {
            return Err(FrontError::source(
                "NAME_DUPLICATE",
                "duplicate foreign parameter",
                foreign.span,
            ));
        }
        if foreign.params.len() > 8 {
            return Err(FrontError::source(
                "RESOURCE_LIMIT",
                "foreign function accepts at most eight arguments",
                foreign.span,
            ));
        }
        let Some((ok, err)) = crate::front::result_parts(match &foreign.result {
            Type::Named(name) => name,
            _ => "",
        }) else {
            return Err(mismatch(
                "foreign result must be Result<T,ForeignError>",
                foreign.span,
            ));
        };
        let valid_result = matches!(ok, Type::I32 | Type::U32)
            || matches!(&ok, Type::Named(name) if matches!(name.as_str(), "Bytes" | "String"));
        if err != Type::Named("ForeignError".to_owned()) || !valid_result {
            return Err(mismatch("unsupported foreign result type", foreign.span));
        }
        for param in &foreign.params {
            let valid = match (&param.ty, param.mode) {
                (Type::I32 | Type::U32 | Type::F64, None) => true,
                (Type::Named(name), Some(Mode::Read | Mode::Take))
                    if matches!(name.as_str(), "Bytes" | "String") =>
                {
                    true
                }
                _ => false,
            };
            if !valid {
                return Err(mismatch(
                    "unsupported foreign argument type or mode",
                    param.span,
                ));
            }
        }
        let Some(oracle) = program
            .functions
            .iter()
            .find(|function| function.name == foreign.oracle)
        else {
            return Err(FrontError::source(
                "NAME_UNKNOWN",
                "foreign oracle function is missing",
                foreign.span,
            ));
        };
        if oracle.result != foreign.result
            || oracle
                .params
                .iter()
                .map(|param| (&param.ty, param.mode))
                .ne(foreign.params.iter().map(|param| (&param.ty, param.mode)))
        {
            return Err(mismatch(
                "foreign oracle signature differs from declaration",
                foreign.span,
            ));
        }
        functions.insert(
            foreign.name.clone(),
            Signature {
                params: foreign
                    .params
                    .iter()
                    .map(|param| param.ty.clone())
                    .collect(),
                result: foreign.result.clone(),
            },
        );
    }
    if let Some(main) = program
        .functions
        .iter()
        .find(|function| function.name == "main")
    {
        if !main.params.is_empty() || main.result != Type::I32 {
            return Err(FrontError::source(
                "ENTRYPOINT_SIGNATURE",
                "main must be fn main() -> i32",
                main.span,
            ));
        }
    } else if require_main {
        return Err(FrontError::source(
            "ENTRYPOINT_MISSING",
            "main function is required for execution",
            Span { start: 0, end: 0 },
        ));
    }
    let mut checker = Checker {
        functions,
        records,
        variants,
        expression_types: Vec::new(),
        function_expression_types: HashMap::new(),
        generic_templates,
        pending_instances: Vec::new(),
        generic_instances: HashSet::new(),
        resolved_calls: HashMap::new(),
        call_edges: BTreeMap::new(),
        current_function: String::new(),
        current_module: None,
        tc_steps: tc_steps
            .map(<[Span]>::to_vec)
            .unwrap_or_else(|| crate::contraction_source::tc_steps(&program)),
    };
    for function in &program.functions {
        checker.function(function)?;
    }
    while let Some(instance) = checker.pending_instances.pop() {
        checker.function(&instance)?;
        program.functions.push(instance);
    }
    checker.check_generic_graph()?;
    numeric::finish(&numeric_sites, &mut program, &checker)?;
    storage::lower(
        &mut program,
        &mut checker.expression_types,
        &mut checker.function_expression_types,
    )?;
    for function in &mut program.functions {
        generics::rewrite_calls(&mut function.body, &function.name, &checker.resolved_calls);
    }
    if !allow_type_only {
        record_generics::reject_type_only_runtime(&program)?;
    }
    for (name, fields) in crate::spatial::records() {
        let ty = Type::Named(name.clone());
        let used = checker
            .expression_types
            .iter()
            .any(|(_, actual)| actual == &ty)
            || program
                .functions
                .iter()
                .any(|f| f.result == ty || f.params.iter().any(|p| p.ty == ty))
            || program
                .records
                .iter()
                .any(|r| r.fields.iter().any(|f| f.ty == ty))
            || program
                .variants
                .iter()
                .any(|v| v.arms.iter().any(|a| a.payload.as_ref() == Some(&ty)));
        if !used {
            continue;
        }
        // Materialize only used builtin layouts. Unrelated programs retain an
        // empty record table for existing scalar/array backend admission.
        program.records.push(crate::front::RecordDecl {
            name,
            module: None,
            public: false,
            generic_params: Vec::new(),
            fields: fields
                .into_iter()
                .map(|(name, ty)| crate::front::RecordField {
                    name,
                    ty,
                    span: Span { start: 0, end: 0 },
                })
                .collect(),
            span: Span { start: 0, end: 0 },
        });
    }
    let checked = CheckedProgram {
        program,
        expression_types: checker.expression_types,
        function_expression_types: checker.function_expression_types,
        lowered_folds: numeric_sites.lowered,
    };
    crate::ownership::check(&checked)?;
    views::check(&checked)?;
    Ok(checked)
}

include!("check/modules.rs");
mod closures;
mod frames;
mod generics;
mod literals;
mod numeric;
mod record_generics;
mod reductions;
mod result;
mod storage;
mod unit_math;
mod units;
mod views;
use literals::{literal_type, split_literal};
use result::synthesize_results;
pub(crate) use units::{angle_call, is_frame_helper, scale_call};

pub(crate) fn system_unit_supported(name: &str) -> bool {
    unit_math::Unit::parse(name).is_some()
}

impl Checker {
    fn function(&mut self, function: &Function) -> Result<(), FrontError> {
        self.current_function = function.name.clone();
        self.current_module = function.module.clone();
        let first_type = self.expression_types.len();
        match (&function.result, &function.return_from) {
            (Type::Named(name), Some(source)) if crate::front::view_parts(name).is_some() => {
                let param = function
                    .params
                    .iter()
                    .find(|param| param.name == *source)
                    .ok_or_else(|| {
                        mismatch("view return source must be a parameter", function.span)
                    })?;
                let (_, mutable) = crate::front::view_parts(name).expect("checked view type");
                if !is_array(&param.ty)
                    || (mutable && param.mode != Some(Mode::Mut))
                    || (!mutable && !matches!(param.mode, Some(Mode::Read | Mode::Mut)))
                {
                    return Err(mismatch(
                        "view return source needs a matching borrowed array parameter",
                        param.span,
                    ));
                }
            }
            (Type::Named(name), None) if crate::front::view_parts(name).is_some() => {
                return Err(mismatch(
                    "view return requires 'from' parameter",
                    function.span,
                ));
            }
            (_, Some(_)) => {
                return Err(mismatch(
                    "'from' requires a view return type",
                    function.span,
                ));
            }
            _ => {}
        }
        let mut context = FunctionChecker {
            checker: self,
            scopes: vec![HashMap::new()],
            declared: HashSet::new(),
            result: function.result.clone(),
        };
        for param in &function.params {
            if crate::spatial::is_grid(&param.ty) && param.mode == Some(Mode::Mut) {
                return Err(mismatch(
                    "VoxelGrid permits read or take, never mut",
                    param.span,
                ));
            }
            if param.ty == Type::Unit {
                return Err(mismatch(
                    "Unit is only permitted as a function return type",
                    param.span,
                ));
            }
            if crate::front::view_parts(match &param.ty {
                Type::Named(name) => name,
                _ => "",
            })
            .is_some()
            {
                return Err(mismatch(
                    "view parameters are outside this bounded subset",
                    param.span,
                ));
            }
            if (is_array(&param.ty) || matches!(param.ty, Type::Named(_))) != param.mode.is_some() {
                return Err(mismatch(
                    "owned parameters require read, mut, or take; scalar parameters omit a mode",
                    param.span,
                ));
            }
            context.declare(
                &param.name,
                Binding {
                    ty: param.ty.clone(),
                    mutable: false,
                    mode: param.mode,
                },
                param.span,
            )?;
        }
        let returns = context.block(&function.body)?;
        if function.result != Type::Unit && !returns {
            return Err(FrontError::source(
                "RETURN_MISSING",
                format!(
                    "function '{}' may fall through without returning",
                    function.name
                ),
                function.span,
            ));
        }
        self.function_expression_types.insert(
            function.name.clone(),
            self.expression_types[first_type..].to_vec(),
        );
        Ok(())
    }
}

impl FunctionChecker<'_> {
    fn declare(&mut self, name: &str, binding: Binding, span: Span) -> Result<(), FrontError> {
        // A variable may share a library function's name: calls and values are
        // separate, so `round(x)` still calls the library with `var round` in scope.
        if self.declared.contains(name)
            || self.checker.functions.contains_key(name)
            || (intrinsic(name) && crate::math::MathFn::from_name(name).is_none())
        {
            return Err(FrontError::source(
                "NAME_DUPLICATE",
                format!("duplicate or shadowing name '{name}'"),
                span,
            ));
        }
        self.declared.insert(name.to_owned());
        let scope = self
            .scopes
            .last_mut()
            .ok_or_else(|| FrontError::tool("INTERNAL_SCOPE", "missing checker scope"))?;
        scope.insert(name.to_owned(), binding);
        Ok(())
    }

    fn binding(&self, name: &str, span: Span) -> Result<Binding, FrontError> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .cloned()
            .ok_or_else(|| {
                FrontError::source("NAME_UNKNOWN", format!("unknown binding '{name}'"), span)
            })
    }

    fn block(&mut self, block: &Block) -> Result<bool, FrontError> {
        self.scopes.push(HashMap::new());
        let mut returns = false;
        for statement in &block.statements {
            let statement_returns = self.statement(statement)?;
            // Dead statements are still checked, but cannot alter a completed path.
            returns |= statement_returns;
        }
        self.scopes.pop();
        Ok(returns)
    }

    fn statement(&mut self, statement: &Stmt) -> Result<bool, FrontError> {
        match &statement.kind {
            StmtKind::Binding {
                mutable,
                name,
                annotation,
                value,
            } => {
                if annotation == &Some(Type::Unit) {
                    return Err(mismatch(
                        "Unit local bindings are unsupported",
                        statement.span,
                    ));
                }
                let ty = self.expr(value, annotation.as_ref())?;
                if ty == Type::Unit {
                    return Err(mismatch(
                        "Unit local bindings are unsupported",
                        statement.span,
                    ));
                }
                self.declare(
                    name,
                    Binding {
                        ty,
                        mutable: *mutable,
                        mode: None,
                    },
                    statement.span,
                )?;
                Ok(false)
            }
            StmtKind::Assign { place, value } => {
                match place {
                    Place::Name(name, span) => {
                        let binding = self.binding(name, *span)?;
                        // Assignment checks its RHS before the destination access.
                        self.expr(value, Some(&binding.ty))?;
                        if !binding.mutable {
                            return Err(mismatch("cannot assign to immutable binding", *span));
                        }
                    }
                    Place::Index { name, index, span } => {
                        let binding = self.binding(name, *span)?;
                        let ty = element(&binding.ty).ok_or_else(|| {
                            mismatch("indexed assignment requires an array", *span)
                        })?;
                        self.expr(value, Some(&ty))?;
                        self.expr(index, Some(&Type::U32))?;
                        let view_mut = matches!(&binding.ty, Type::Named(name) if crate::front::view_parts(name).is_some_and(|(_, mutable)| mutable));
                        if !(binding.mutable || binding.mode == Some(Mode::Mut) || view_mut) {
                            return Err(mismatch(
                                "cannot mutate through immutable or read binding",
                                *span,
                            ));
                        }
                    }
                }
                Ok(false)
            }
            StmtKind::Return(value) => {
                match value {
                    Some(value) => {
                        self.expr(value, Some(&self.result.clone()))?;
                    }
                    None if self.result == Type::Unit => {}
                    None => {
                        return Err(mismatch("non-Unit return requires a value", statement.span));
                    }
                }
                Ok(true)
            }
            StmtKind::If { .. } => {
                let mut branch = statement;
                let mut returns = true;
                loop {
                    let StmtKind::If {
                        condition,
                        then_block,
                        else_arm,
                    } = &branch.kind
                    else {
                        unreachable!("else-if arm is an if statement")
                    };
                    self.expr(condition, Some(&Type::Bool))?;
                    returns &= self.block(then_block)?;
                    match else_arm {
                        Some(ElseArm::Block(block)) => {
                            let else_returns = self.block(block)?;
                            return Ok(returns && else_returns);
                        }
                        Some(ElseArm::If(next)) => branch = next,
                        None => return Ok(false),
                    }
                }
            }
            StmtKind::While { condition, body } => {
                self.expr(condition, Some(&Type::Bool))?;
                self.block(body)?;
                Ok(false)
            }
            StmtKind::For {
                name,
                lower,
                upper,
                body,
            } => {
                self.expr(lower, Some(&Type::U32))?;
                self.expr(upper, Some(&Type::U32))?;
                self.scopes.push(HashMap::new());
                self.declare(
                    name,
                    Binding {
                        ty: Type::U32,
                        mutable: false,
                        mode: None,
                    },
                    statement.span,
                )?;
                self.block(body)?;
                self.scopes.pop();
                Ok(false)
            }
            StmtKind::Expr(expr) => {
                self.expr(expr, None)?;
                Ok(false)
            }
        }
    }

    fn hint(&self, expr: &Expr) -> Option<Type> {
        match &expr.kind {
            ExprKind::Integer(raw) => {
                if raw.ends_with("i32") {
                    Some(Type::I32)
                } else if raw.ends_with("u32") {
                    Some(Type::U32)
                } else {
                    None
                }
            }
            ExprKind::Float(raw) => literal_type(raw),
            ExprKind::Text(_) => Some(Type::Named("String".to_owned())),
            ExprKind::Bytes(_) => Some(Type::Named("Bytes".to_owned())),
            ExprKind::Bool(_) => Some(Type::Bool),
            ExprKind::Name(name) | ExprKind::Move(name) => {
                self.binding(name, expr.span).ok().map(|binding| binding.ty)
            }
            ExprKind::MoveField { base, field } => {
                let Type::Named(record) = self.hint(base)? else {
                    return None;
                };
                self.checker
                    .records
                    .get(&record)?
                    .iter()
                    .find(|(name, _)| name == field)
                    .map(|(_, ty)| ty.clone())
            }
            ExprKind::Group(inner) => self.hint(inner),
            ExprKind::Try(inner) => {
                let Type::Named(name) = self.hint(inner)? else {
                    return None;
                };
                crate::front::result_parts(&name).map(|(ok, _)| ok)
            }
            ExprKind::Call { name, args } => match name.as_str() {
                value if scale_call(value).is_some() || angle_call(value).is_some() => {
                    self.hint(args.first()?)
                }
                "len" | "wrapping_add" | "wrapping_mul" => Some(Type::U32),
                "bytes_slice" | "utf8_encode" => Some(Type::Named("Bytes".to_owned())),
                "utf8_decode" | "utf8_slice" => {
                    Some(Type::Named("Result<String,Utf8Error>".to_owned()))
                }
                "argsort" => Some(Type::ArrayU32),
                "sort" | "sort_by_key" | "segmented_sort" => Some(Type::Unit),
                value if storage::filled(value).is_some() => {
                    storage::filled(value).map(|(_, array)| array)
                }
                value if conversion(value).is_some() => conversion(value).map(|(_, result)| result),
                value if crate::spatial::result(value).is_some() => crate::spatial::result(value),
                value if value == "fma" || crate::math::MathFn::from_name(value).is_some() => {
                    args.iter().find_map(|arg| self.hint(arg).filter(is_float))
                }
                "tc_mul_add" => Some(Type::F32),
                reduction if reductions::is_reduction(reduction) => args
                    .first()
                    .and_then(|arg| self.hint(arg))
                    .and_then(|ty| element(&ty)),
                _ => self
                    .checker
                    .functions
                    .get(name)
                    .map(|signature| signature.result.clone())
                    .or_else(|| {
                        name.rsplit_once("::").and_then(|(nominal, arm)| {
                            self.checker
                                .variants
                                .get(nominal)
                                .filter(|arms| arms.iter().any(|(name, _)| name == arm))
                                .map(|_| Type::Named(nominal.to_owned()))
                        })
                    }),
            },
            ExprKind::Array(_) => None,
            ExprKind::Record { name, .. } => Some(Type::Named(name.clone())),
            ExprKind::Match { arms, .. } => arms.first().and_then(|arm| self.hint(&arm.value)),
            ExprKind::Field { base, name } => {
                let Type::Named(record) = self.hint(base)? else {
                    return None;
                };
                self.checker
                    .records
                    .get(&record)?
                    .iter()
                    .find(|(field, _)| field == name)
                    .map(|(_, ty)| ty.clone())
            }
            ExprKind::Index { base, .. } => self.hint(base).as_ref().and_then(element),
            ExprKind::Unary {
                op: UnaryOp::Not, ..
            } => Some(Type::Bool),
            ExprKind::Unary {
                op: UnaryOp::Neg,
                value,
                ..
            } => self
                .hint(value)
                .or_else(|| (!flexible_float(value)).then_some(Type::I32)),
            ExprKind::Binary {
                op:
                    BinaryOp::And
                    | BinaryOp::Or
                    | BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge,
                ..
            } => Some(Type::Bool),
            ExprKind::Binary { .. } => {
                let mut pending = vec![expr];
                while let Some(next) = pending.pop() {
                    if let ExprKind::Binary {
                        op:
                            BinaryOp::Add
                            | BinaryOp::Sub
                            | BinaryOp::Mul
                            | BinaryOp::Div
                            | BinaryOp::Rem,
                        left,
                        right,
                        ..
                    } = &next.kind
                    {
                        pending.push(right);
                        pending.push(left);
                    } else if let Some(ty) = self.hint(next) {
                        return Some(ty);
                    }
                }
                None
            }
        }
    }

    fn expr(&mut self, expr: &Expr, expected: Option<&Type>) -> Result<Type, FrontError> {
        let ty = match &expr.kind {
            ExprKind::Integer(raw) => self.integer(raw, expected, expr.span)?,
            ExprKind::Float(raw) => literal_type(raw)
                .or_else(|| expected.filter(|ty| is_float(ty)).cloned())
                .unwrap_or(Type::F32),
            ExprKind::Bool(_) => Type::Bool,
            ExprKind::Text(_) => Type::Named("String".to_owned()),
            ExprKind::Bytes(_) => Type::Named("Bytes".to_owned()),
            ExprKind::Name(name) => self.binding(name, expr.span)?.ty,
            ExprKind::Move(name) => {
                let ty = self.binding(name, expr.span)?.ty;
                if !is_array(&ty)
                    && !matches!(ty, Type::Named(ref name) if crate::front::view_parts(name).is_none())
                {
                    return Err(mismatch("move requires an owned value", expr.span));
                }
                ty
            }
            ExprKind::MoveField { base, field } => {
                let Type::Named(record) = self.expr(base, None)? else {
                    return Err(mismatch("field move requires a record", expr.span));
                };
                let ty = self
                    .checker
                    .records
                    .get(&record)
                    .and_then(|fields| fields.iter().find(|(name, _)| name == field))
                    .map(|(_, ty)| ty.clone())
                    .ok_or_else(|| {
                        FrontError::source(
                            "NAME_UNKNOWN",
                            format!("unknown field '{field}'"),
                            expr.span,
                        )
                    })?;
                if !is_array(&ty) {
                    return Err(mismatch(
                        "field move requires an owned array field",
                        expr.span,
                    ));
                }
                ty
            }
            ExprKind::Group(inner) => self.expr(inner, expected)?,
            ExprKind::Try(inner) => {
                let Type::Named(name) = self.expr(inner, None)? else {
                    return Err(mismatch("? requires Result<T,E>", expr.span));
                };
                let (ok, err) = crate::front::result_parts(&name)
                    .ok_or_else(|| mismatch("? requires Result<T,E>", expr.span))?;
                let Type::Named(return_name) = &self.result else {
                    return Err(mismatch(
                        "? requires a Result-returning function",
                        expr.span,
                    ));
                };
                let (_, return_err) = crate::front::result_parts(return_name)
                    .ok_or_else(|| mismatch("? requires a Result-returning function", expr.span))?;
                if err != return_err {
                    return Err(mismatch(
                        "? requires the same error type as the function result",
                        expr.span,
                    ));
                }
                ok
            }
            ExprKind::Call { name, args } => self.call(name, args, expr.span, expected)?,
            ExprKind::Array(values) => {
                let array_ty = expected.filter(|ty| is_array(ty)).cloned().ok_or_else(|| {
                    mismatch("array literal requires an expected Array type", expr.span)
                })?;
                let item_ty = element(&array_ty)
                    .ok_or_else(|| mismatch("invalid array element type", expr.span))?;
                for value in values {
                    self.expr(value, Some(&item_ty))?;
                }
                array_ty
            }
            ExprKind::Record { name, fields } => {
                let declared = self.checker.records.get(name).cloned().ok_or_else(|| {
                    FrontError::source(
                        "NAME_UNKNOWN",
                        format!("unknown record '{name}'"),
                        expr.span,
                    )
                })?;
                let mut supplied = HashSet::new();
                for field in fields {
                    if !supplied.insert(field.name.as_str()) {
                        return Err(FrontError::source(
                            "NAME_DUPLICATE",
                            format!("duplicate initializer field '{}'", field.name),
                            field.span,
                        ));
                    }
                    let field_ty = declared
                        .iter()
                        .find(|(field_name, _)| *field_name == field.name)
                        .map(|(_, ty)| ty)
                        .ok_or_else(|| {
                            FrontError::source(
                                "NAME_UNKNOWN",
                                format!("unknown field '{}'", field.name),
                                field.span,
                            )
                        })?;
                    self.expr(&field.value, Some(field_ty))?;
                }
                if fields.len() != declared.len() {
                    return Err(mismatch("record initializer is missing fields", expr.span));
                }
                Type::Named(name.clone())
            }
            ExprKind::Match { value, arms } => self.match_expr(value, arms, expected, expr.span)?,
            ExprKind::Field { base, name } => {
                if !matches!(base.kind, ExprKind::Name(_)) {
                    return Err(FrontError::source(
                        "UNSUPPORTED_FEATURE",
                        "field projection requires a named record binding",
                        base.span,
                    ));
                }
                let Type::Named(record) = self.expr(base, None)? else {
                    return Err(mismatch("field projection requires a record", base.span));
                };
                let ty = self
                    .checker
                    .records
                    .get(&record)
                    .and_then(|fields| fields.iter().find(|(field, _)| field == name))
                    .map(|(_, ty)| ty.clone())
                    .ok_or_else(|| {
                        FrontError::source(
                            "NAME_UNKNOWN",
                            format!("unknown field '{name}'"),
                            expr.span,
                        )
                    })?;
                if is_array(&ty) {
                    return Err(FrontError::source(
                        "OWN_MOVE_REQUIRED",
                        "owned record field requires move",
                        expr.span,
                    ));
                }
                ty
            }
            ExprKind::Index { base, index } => {
                let base_ty = self.expr(base, None)?;
                let item_ty = element(&base_ty)
                    .ok_or_else(|| mismatch("indexing requires an array", base.span))?;
                self.expr(index, Some(&Type::U32))?;
                item_ty
            }
            ExprKind::Unary {
                op: UnaryOp::Not,
                value,
                ..
            } => {
                self.expr(value, Some(&Type::Bool))?;
                Type::Bool
            }
            ExprKind::Unary {
                op: UnaryOp::Neg,
                value,
                ..
            } => {
                let float = match self.hint(value) {
                    Some(ty) if is_float(&ty) => Some(ty),
                    None if flexible_float(value) => Some(
                        expected
                            .filter(|ty| is_float(ty))
                            .cloned()
                            .unwrap_or(Type::F32),
                    ),
                    _ => None,
                };
                if let Some(float) = float {
                    self.expr(value, Some(&float))?;
                    float
                } else {
                    if let ExprKind::Integer(raw) = &value.kind {
                        let (digits, suffix) = split_literal(raw);
                        if suffix == Some(Type::U32) {
                            return Err(mismatch("cannot negate a u32 literal", expr.span));
                        }
                        let magnitude = digits
                            .parse::<u64>()
                            .map_err(|_| mismatch("integer literal out of range", value.span))?;
                        if magnitude > 2_147_483_648 {
                            return Err(mismatch("i32 literal out of range", expr.span));
                        }
                        self.checker.expression_types.push((value.span, Type::I32));
                    } else {
                        self.expr(value, Some(&Type::I32))?;
                    }
                    Type::I32
                }
            }
            ExprKind::Binary {
                left, op, right, ..
            } => match op {
                BinaryOp::And | BinaryOp::Or => {
                    self.expr(left, Some(&Type::Bool))?;
                    self.expr(right, Some(&Type::Bool))?;
                    Type::Bool
                }
                BinaryOp::Eq | BinaryOp::Ne => {
                    let operand_ty = self.operand_type(left, right);
                    self.expr(left, Some(&operand_ty))?;
                    self.expr(right, Some(&operand_ty))?;
                    if !is_numeric(&operand_ty) && operand_ty != Type::Bool {
                        return Err(mismatch(
                            "equality requires matching scalar operands",
                            expr.span,
                        ));
                    }
                    Type::Bool
                }
                BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                    let operand_ty = self.operand_type(left, right);
                    self.expr(left, Some(&operand_ty))?;
                    self.expr(right, Some(&operand_ty))?;
                    if !is_numeric(&operand_ty) {
                        return Err(mismatch(
                            "comparison requires matching numeric operands",
                            expr.span,
                        ));
                    }
                    Type::Bool
                }
                _ => return self.arithmetic_chain(expr, expected),
            },
        };
        if let Some(expected) = expected
            && &ty != expected
        {
            return Err(mismatch(
                format!("expected {expected:?}, found {ty:?}"),
                expr.span,
            ));
        }
        self.checker.expression_types.push((expr.span, ty.clone()));
        Ok(ty)
    }

    fn arithmetic_chain(
        &mut self,
        expr: &Expr,
        expected: Option<&Type>,
    ) -> Result<Type, FrontError> {
        let ExprKind::Binary { left, right, .. } = &expr.kind else {
            return Err(mismatch("arithmetic expression required", expr.span));
        };
        let operand_ty = expected
            .filter(|ty| is_numeric(ty))
            .cloned()
            .unwrap_or_else(|| self.operand_type(left, right));
        let mut chain = Vec::new();
        let mut leaf = expr;
        while let ExprKind::Binary {
            op: BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem,
            left,
            right,
            ..
        } = &leaf.kind
        {
            if is_float(&operand_ty)
                && matches!(
                    &leaf.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Rem,
                        ..
                    }
                )
            {
                return Err(mismatch("float remainder is unsupported", leaf.span));
            }
            chain.push((leaf.span, right.as_ref()));
            leaf = left;
        }
        self.expr(leaf, Some(&operand_ty))?;
        let count = chain.len();
        for (index, (span, right)) in chain.into_iter().rev().enumerate() {
            self.expr(right, Some(&operand_ty))?;
            if !is_numeric(&operand_ty) {
                return Err(mismatch(
                    "arithmetic requires matching numeric operands",
                    span,
                ));
            }
            if index + 1 == count
                && let Some(expected) = expected
                && &operand_ty != expected
            {
                return Err(mismatch(
                    format!("expected {expected:?}, found {operand_ty:?}"),
                    span,
                ));
            }
            self.checker
                .expression_types
                .push((span, operand_ty.clone()));
        }
        Ok(operand_ty)
    }

    fn match_expr(
        &mut self,
        value: &Expr,
        arms: &[MatchArm],
        expected: Option<&Type>,
        span: Span,
    ) -> Result<Type, FrontError> {
        let Type::Named(nominal) = self.expr(value, None)? else {
            return Err(mismatch("match scrutinee must be a variant", value.span));
        };
        let declared = self
            .checker
            .variants
            .get(&nominal)
            .cloned()
            .ok_or_else(|| mismatch("match scrutinee must be a variant", value.span))?;
        let first = arms
            .first()
            .ok_or_else(|| mismatch("match needs arms", span))?;
        let result_ty = expected
            .cloned()
            .or_else(|| self.hint(&first.value))
            .unwrap_or(Type::I32);
        if !matches!(
            result_ty,
            Type::I32 | Type::U32 | Type::F32 | Type::F64 | Type::Bool
        ) {
            return Err(FrontError::source(
                "UNSUPPORTED_FEATURE",
                "this match subset returns a scalar",
                span,
            ));
        }
        let mut seen = HashSet::new();
        for arm in arms {
            let (prefix, case) = arm.variant.rsplit_once("::").ok_or_else(|| {
                FrontError::source(
                    "NAME_UNKNOWN",
                    "match arm needs a qualified variant",
                    arm.span,
                )
            })?;
            if prefix != nominal
                && !(crate::front::result_parts(&nominal).is_some() && prefix == "Result")
            {
                return Err(mismatch("match arm has the wrong variant type", arm.span));
            }
            let payload = declared
                .iter()
                .find(|(name, _)| name == case)
                .ok_or_else(|| {
                    FrontError::source(
                        "NAME_UNKNOWN",
                        format!("unknown variant arm '{}'", arm.variant),
                        arm.span,
                    )
                })?
                .1
                .clone();
            if !seen.insert(case) {
                return Err(FrontError::source(
                    "NAME_DUPLICATE",
                    format!("duplicate match arm '{case}'"),
                    arm.span,
                ));
            }
            if payload.is_some() != arm.binding.is_some() {
                return Err(mismatch(
                    "match payload binding does not match variant arm",
                    arm.span,
                ));
            }
            if payload
                .as_ref()
                .is_some_and(|ty| matches!(ty, Type::Named(_)))
                && !matches!(value.kind, ExprKind::Move(_))
            {
                return Err(FrontError::source(
                    "OWN_MOVE_REQUIRED",
                    "owned pattern payload requires match move",
                    arm.span,
                ));
            }
            self.scopes.push(HashMap::new());
            if let (Some(binding), Some(ty)) = (&arm.binding, payload) {
                self.declare(
                    binding,
                    Binding {
                        ty,
                        mutable: false,
                        mode: None,
                    },
                    arm.span,
                )?;
            }
            self.expr(&arm.value, Some(&result_ty))?;
            self.scopes.pop();
        }
        if seen.len() != declared.len() {
            return Err(FrontError::source(
                "MATCH_NONEXHAUSTIVE",
                "match does not cover every variant arm",
                span,
            ));
        }
        Ok(result_ty)
    }

    /// Operand type of a binary predicate or arithmetic chain: the first
    /// operand with a known type, else `f32` for an unsuffixed float literal
    /// and `i32` otherwise.
    fn operand_type(&self, left: &Expr, right: &Expr) -> Type {
        self.hint(left)
            .or_else(|| self.hint(right))
            .unwrap_or_else(|| {
                if flexible_float(left) || flexible_float(right) {
                    Type::F32
                } else {
                    Type::I32
                }
            })
    }

    fn integer(&self, raw: &str, expected: Option<&Type>, span: Span) -> Result<Type, FrontError> {
        let (digits, suffix) = split_literal(raw);
        let ty = suffix
            .or_else(|| expected.filter(|ty| is_integer(ty)).cloned())
            .unwrap_or(Type::I32);
        if !is_integer(&ty) {
            return Err(mismatch("integer literal requires integer type", span));
        }
        let magnitude = digits
            .parse::<u64>()
            .map_err(|_| mismatch("integer literal out of range", span))?;
        let limit = if ty == Type::I32 {
            i32::MAX as u64
        } else {
            u32::MAX as u64
        };
        if magnitude > limit {
            return Err(mismatch("integer literal out of range", span));
        }
        Ok(ty)
    }

    fn call(
        &mut self,
        name: &str,
        args: &[Expr],
        span: Span,
        expected: Option<&Type>,
    ) -> Result<Type, FrontError> {
        if let Some((params, result)) = crate::spatial::signature(name) {
            if args.len() != params.len() {
                return Err(mismatch("wrong spatial intrinsic arity", span));
            }
            for (arg, ty) in args.iter().zip(params) {
                self.expr(arg, Some(&ty))?;
            }
            return Ok(result);
        }
        if scale_call(name).is_some() || angle_call(name).is_some() {
            if args.len() != 1 {
                return Err(mismatch("checked scale expects one value", span));
            }
            let ty = self.expr(&args[0], expected)?;
            if !is_integer(&ty) {
                return Err(mismatch("checked scale requires an integer", span));
            }
            return Ok(ty);
        }
        if let Some(result) = self.generic_call(name, args, span)? {
            return Ok(result);
        }
        if matches!(name, "Result::Ok" | "Result::Err") {
            let Some(Type::Named(nominal)) = expected else {
                return Err(mismatch(
                    "Result constructor requires an expected Result type",
                    span,
                ));
            };
            let (ok, err) = crate::front::result_parts(nominal).ok_or_else(|| {
                mismatch("Result constructor requires an expected Result type", span)
            })?;
            if args.len() != 1 {
                return Err(mismatch("Result constructor expects one payload", span));
            }
            self.expr(
                &args[0],
                Some(if name == "Result::Ok" { &ok } else { &err }),
            )?;
            return Ok(Type::Named(nominal.clone()));
        }
        if let Some(modes) = sort_modes(name) {
            if args.len() != modes.len() {
                return Err(mismatch("wrong sort intrinsic arity", span));
            }
            for (index, arg) in args.iter().enumerate() {
                if !matches!(arg.kind, ExprKind::Name(_)) {
                    return Err(FrontError::source(
                        "OWN_BORROW_SOURCE",
                        "sort loans require named arrays",
                        arg.span,
                    ));
                }
                let ty = self.expr(arg, None)?;
                let valid = if index == 0 {
                    matches!(ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32)
                } else if name == "segmented_sort" {
                    ty == Type::ArrayU32
                } else {
                    // f64 sort keys and payloads are a C5 follow-up: no MIR or
                    // runtime representation carries them yet.
                    is_array(&ty) && ty != Type::ArrayF64
                };
                if !valid {
                    return Err(mismatch("invalid sort array element type", arg.span));
                }
            }
            return Ok(if name == "argsort" {
                Type::ArrayU32
            } else {
                Type::Unit
            });
        }
        match name {
            "tc_mul_add" => {
                if !self.checker.tc_steps.contains(&span) {
                    return Err(FrontError::source(
                        "UNSUPPORTED_FEATURE",
                        "tc_mul_add is admitted only as the accumulation step of a recognized matrix contraction",
                        span,
                    ));
                }
                if args.len() != 3 {
                    return Err(mismatch(
                        "tc_mul_add expects a, b and the accumulator",
                        span,
                    ));
                }
                for arg in args {
                    self.expr(arg, Some(&Type::F32))?;
                }
                Ok(Type::F32)
            }
            "bytes_slice" | "utf8_slice" => {
                if args.len() != 3 {
                    return Err(mismatch("slice expects owner, start, end", span));
                }
                if !matches!(args[0].kind, ExprKind::Name(_)) {
                    return Err(mismatch(
                        "slice needs a named immutable owner",
                        args[0].span,
                    ));
                }
                let source = if name == "bytes_slice" {
                    "Bytes"
                } else {
                    "String"
                };
                self.expr(&args[0], Some(&Type::Named(source.to_owned())))?;
                self.expr(&args[1], Some(&Type::U32))?;
                self.expr(&args[2], Some(&Type::U32))?;
                Ok(Type::Named(
                    if name == "bytes_slice" {
                        "Bytes"
                    } else {
                        "Result<String,Utf8Error>"
                    }
                    .to_owned(),
                ))
            }
            "utf8_decode" | "utf8_encode" => {
                if args.len() != 1 {
                    return Err(mismatch("UTF-8 conversion expects one argument", span));
                }
                if !matches!(args[0].kind, ExprKind::Name(_)) {
                    return Err(mismatch(
                        "UTF-8 conversion needs a named immutable owner",
                        args[0].span,
                    ));
                }
                let source = if name == "utf8_decode" {
                    "Bytes"
                } else {
                    "String"
                };
                self.expr(&args[0], Some(&Type::Named(source.to_owned())))?;
                Ok(Type::Named(
                    if name == "utf8_decode" {
                        "Result<String,Utf8Error>"
                    } else {
                        "Bytes"
                    }
                    .to_owned(),
                ))
            }
            reduction if reductions::is_reduction(reduction) => {
                self.reduction(reduction, args, span)
            }
            "len" => {
                if args.len() != 1 {
                    return Err(mismatch("len expects one argument", span));
                }
                let ty = self.expr(&args[0], None)?;
                if !is_array(&ty)
                    && element(&ty).is_none()
                    && ty != Type::Named("String".to_owned())
                {
                    return Err(mismatch(
                        "len expects an array, view, Bytes, or String",
                        args[0].span,
                    ));
                }
                Ok(Type::U32)
            }
            "view" | "view_mut" => {
                if !(3..=4).contains(&args.len()) {
                    return Err(mismatch(
                        "view expects owner, start, end, and optional stride",
                        span,
                    ));
                }
                let owner = self.expr(&args[0], None)?;
                if matches!(owner, Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool) {
                    return Err(mismatch(
                        "only integer borrowed views are supported",
                        args[0].span,
                    ));
                }
                let item = element(&owner)
                    .ok_or_else(|| mismatch("view needs an integer array or view", args[0].span))?;
                if name == "view_mut"
                    && matches!(&owner, Type::Named(view) if crate::front::view_parts(view).is_some_and(|(_, mutable)| !mutable))
                {
                    return Err(mismatch("mutable reborrow requires MutView", args[0].span));
                }
                for arg in args.iter().skip(1) {
                    self.expr(arg, Some(&Type::U32))?;
                }
                let kind = if name == "view_mut" {
                    "MutView"
                } else {
                    "View"
                };
                let item = if item == Type::I32 { "i32" } else { "u32" };
                Ok(Type::Named(format!("{kind}<{item}>")))
            }
            _ if conversion(name).is_some() => {
                let Some((input, result)) = conversion(name) else {
                    return Err(mismatch("unknown float conversion", span));
                };
                if args.len() != 1 {
                    return Err(mismatch("float conversion expects one argument", span));
                }
                self.expr(&args[0], Some(&input))?;
                Ok(result)
            }
            _ if name == "fma" || crate::math::MathFn::from_name(name).is_some() => {
                let arity = crate::math::MathFn::from_name(name).map_or(3, |f| f.arity());
                if args.len() != arity {
                    return Err(mismatch(
                        format!("{name} expects {arity} float arguments"),
                        span,
                    ));
                }
                let ty = args
                    .iter()
                    .find_map(|arg| self.hint(arg).filter(is_float))
                    .or_else(|| expected.filter(|ty| is_float(ty)).cloned())
                    .unwrap_or(Type::F32);
                for arg in args {
                    self.expr(arg, Some(&ty))?;
                }
                Ok(ty)
            }
            _ if storage::filled(name).is_some() => {
                let Some((item, array)) = storage::filled(name) else {
                    return Err(mismatch("unknown filled array", span));
                };
                if args.len() != 2 {
                    return Err(mismatch("filled array expects two arguments", span));
                }
                self.expr(&args[0], Some(&Type::U32))?;
                self.expr(&args[1], Some(&item))?;
                Ok(array)
            }
            "wrapping_add" | "wrapping_mul" => {
                if args.len() != 2 {
                    return Err(mismatch("wrapping operation expects two arguments", span));
                }
                self.expr(&args[0], Some(&Type::U32))?;
                self.expr(&args[1], Some(&Type::U32))?;
                Ok(Type::U32)
            }
            _ => {
                if let Some((nominal, arm)) = name.rsplit_once("::")
                    && let Some(declared) = self.checker.variants.get(nominal)
                    && let Some((_, payload)) = declared.iter().find(|(case, _)| case == arm)
                {
                    let payload = payload.clone();
                    if args.len() != usize::from(payload.is_some()) {
                        return Err(mismatch("wrong variant constructor arity", span));
                    }
                    if let Some(payload) = payload {
                        self.expr(&args[0], Some(&payload))?;
                    }
                    return Ok(Type::Named(nominal.to_owned()));
                }
                let signature = self.checker.functions.get(name).cloned().ok_or_else(|| {
                    FrontError::source("NAME_UNKNOWN", format!("unknown function '{name}'"), span)
                })?;
                if args.len() != signature.params.len() {
                    return Err(mismatch("wrong call arity", span));
                }
                for (arg, ty) in args.iter().zip(&signature.params) {
                    self.expr(arg, Some(ty))?;
                }
                self.checker
                    .call_edges
                    .entry(self.checker.current_function.clone())
                    .or_default()
                    .insert(name.to_owned());
                Ok(signature.result)
            }
        }
    }
}
