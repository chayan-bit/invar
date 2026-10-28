//! Source-derived, sealed rank-2 contraction annotations over verified typed MIR.
//! Recognition never changes source execution; unknown forms retain the CPU path.
use crate::{
    check::CheckedProgram,
    front::{
        self, BinaryOp, Expr, ExprKind, Mode, Place, Span, Stmt, StmtKind, Type, named, ungroup,
    },
    general_net, mir,
    numeric_contract::{F32Bits, NumericPolicy},
};
use sha2::{Digest, Sha256};

pub const VERSION: u32 = 1;
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Contract {
    version: u32,
    function: String,
    span: Span,
    loop_span: Span,
    loop_spans: [Span; 3],
    loop_ticks: [u64; 3],
    inputs: [String; 2],
    output: String,
    dimensions: [String; 3],
    numeric_policy: NumericPolicy,
    layout_version: u32,
    function_contract: [u8; 32],
}
impl Contract {
    pub fn function(&self) -> &str {
        &self.function
    }
    pub fn span(&self) -> Span {
        self.span
    }
    pub fn loop_span(&self) -> Span {
        self.loop_span
    }
    pub fn loop_spans(&self) -> [Span; 3] {
        self.loop_spans
    }
    /// Source ticks still owed after the reached outer loop's bounds.
    pub fn loop_fuel(&self, [rows, inner, cols]: [u32; 3]) -> Option<u64> {
        let [row, column, product] = self.loop_ticks;
        u64::from(rows).checked_mul(
            row.checked_add(
                u64::from(cols)
                    .checked_mul(column.checked_add(u64::from(inner).checked_mul(product)?)?)?,
            )?,
        )
    }
    pub fn inputs(&self) -> &[String; 2] {
        &self.inputs
    }
    pub fn output(&self) -> &str {
        &self.output
    }
    pub fn dimensions(&self) -> &[String; 3] {
        &self.dimensions
    }
    pub fn numeric_policy(&self) -> NumericPolicy {
        self.numeric_policy
    }
    pub fn layout_version(&self) -> u32 {
        self.layout_version
    }
    pub fn function_contract(&self) -> [u8; 32] {
        self.function_contract
    }
}

/// CheckedProgram has public annotations: re-run the checker on the actual AST and
/// require the current annotations, so stale caller edits are not authentication.
/// Shared by the stencil, transpose and summary source families with their own refusals.
pub(crate) fn recheck<E>(
    source: &CheckedProgram,
    bounds: E,
    mismatch: E,
    rejected: fn(front::FrontError) -> E,
) -> Result<CheckedProgram, E> {
    let p = &source.program;
    if p.functions.len() > mir::MAX_FUNCTIONS {
        return Err(bounds);
    }
    let checked = crate::check::check(
        front::Program {
            functions: p.functions.clone(),
            foreign_functions: p.foreign_functions.clone(),
            closures: p.closures.clone(),
            frames: p
                .frames
                .iter()
                .map(|f| front::FrameDecl {
                    name: f.name.clone(),
                    parent: f.parent.clone(),
                    span: f.span,
                })
                .collect(),
            modules: p
                .modules
                .iter()
                // check::check injects this reserved standard module each time.
                // Rebuild it, then require exact equality with the checked AST;
                // caller mutation of standard declarations is still rejected.
                .filter(|m| m.name != "std::robotics")
                .map(|m| front::ModuleDecl {
                    name: m.name.clone(),
                    span: m.span,
                })
                .collect(),
            imports: p
                .imports
                .iter()
                .map(|i| front::Import {
                    owner: i.owner.clone(),
                    module: i.module.clone(),
                    item: i.item.clone(),
                    span: i.span,
                })
                .collect(),
            records: p
                .records
                .iter()
                .filter(|r| r.module.as_deref() != Some("std::robotics"))
                .cloned()
                .collect(),
            variants: p.variants.clone(),
            system_items: p.system_items.clone(),
        },
        false,
    )
    .map_err(rejected)?;
    if &checked.program != p
        || checked.expression_types != source.expression_types
        || checked.function_expression_types != source.function_expression_types
    {
        return Err(mismatch);
    }
    Ok(checked)
}

pub(crate) fn function_fingerprint(function: &mir::Function) -> [u8; 32] {
    let mut locals = function.locals.iter().collect::<Vec<_>>();
    locals.sort_by_key(|(name, _)| *name);
    let fields = format!(
        "{:?}",
        (
            &function.name,
            &function.foreign_oracle,
            &function.params,
            &function.result,
            &function.return_from,
            function.span,
            function.entry,
            &function.blocks,
            locals,
            &function.declaration_order
        )
    );
    Sha256::digest(fields.as_bytes()).into()
}
fn zero(expr: &Expr) -> bool {
    match &ungroup(expr).kind {
        ExprKind::Float(raw) => F32Bits::from_decimal(raw).is_ok_and(|v| v == F32Bits::POS_ZERO),
        _ => false,
    }
}
fn zero_index(expr: &Expr) -> bool {
    matches!(&ungroup(expr).kind,ExprKind::Integer(raw) if raw=="0u32" || raw=="0")
}
fn product(expr: &Expr, a: &str, b: &str) -> bool {
    matches!(&ungroup(expr).kind,ExprKind::Binary{left,op:BinaryOp::Mul,right,..} if named(left,a)&&named(right,b))
}
fn flat(expr: &Expr, index: &str, stride: &str, column: &str) -> bool {
    matches!(&ungroup(expr).kind,ExprKind::Binary{left,op:BinaryOp::Add,right,..} if product(left,index,stride)&&named(right,column))
}
fn indexed(expr: &Expr, array: &str, index: &str, stride: &str, column: &str) -> bool {
    matches!(&ungroup(expr).kind,ExprKind::Index{base,index:at} if named(base,array)&&flat(at,index,stride,column))
}
fn counted<'a>(stmt: &'a Stmt, upper: &str) -> Option<(&'a str, &'a front::Block)> {
    let StmtKind::For {
        name,
        lower,
        upper: bound,
        body,
    } = &stmt.kind
    else {
        return None;
    };
    (zero_index(lower) && named(bound, upper)).then_some((name.as_str(), body))
}
pub(crate) fn expr_ticks(expr: &Expr) -> Option<u64> {
    match &expr.kind {
        ExprKind::Group(inner) => expr_ticks(inner),
        ExprKind::Name(_) | ExprKind::Integer(_) | ExprKind::Float(_) => Some(1),
        ExprKind::Binary { left, right, .. }
        | ExprKind::Index {
            base: left,
            index: right,
        } => expr_ticks(left)?
            .checked_add(expr_ticks(right)?)?
            .checked_add(1),
        _ => None,
    }
}
fn statement_ticks(stmt: &Stmt) -> Option<u64> {
    let value = match &stmt.kind {
        StmtKind::Binding { value, .. } => expr_ticks(value)?,
        StmtKind::Assign { value, place } => expr_ticks(value)?.checked_add(match place {
            Place::Name(..) => 0,
            Place::Index { index, .. } => expr_ticks(index)?,
        })?,
        StmtKind::For { lower, upper, .. } => expr_ticks(lower)?.checked_add(expr_ticks(upper)?)?,
        _ => return None,
    };
    value.checked_add(1)
}
fn recognize(function: &front::Function, lowered: &mir::Function) -> Option<Contract> {
    if function.params.len() != 5
        || function.result != Type::ArrayF32
        || function.return_from.is_some()
        || !function.generic_params.is_empty()
    {
        return None;
    }
    if function.params[..2]
        .iter()
        .any(|p| p.ty != Type::ArrayF32 || p.mode != Some(Mode::Read))
        || function.params[2..]
            .iter()
            .any(|p| p.ty != Type::U32 || p.mode.is_some())
    {
        return None;
    }
    let [allocate, outer, return_owner] = function.body.statements.as_slice() else {
        return None;
    };
    let a = &function.params[0].name;
    let b = &function.params[1].name;
    let rows = &function.params[2].name;
    let inner = &function.params[3].name;
    let cols = &function.params[4].name;
    let StmtKind::Binding {
        mutable: true,
        name: output,
        value,
        ..
    } = &allocate.kind
    else {
        return None;
    };
    let ExprKind::Call { name, args } = &ungroup(value).kind else {
        return None;
    };
    if name != "filled_f32" || args.len() != 2 || !product(&args[0], rows, cols) || !zero(&args[1])
    {
        return None;
    }
    let StmtKind::Return(Some(owner)) = &return_owner.kind else {
        return None;
    };
    if !named(owner, output) && !matches!(&owner.kind,ExprKind::Move(name) if name==output) {
        return None;
    }
    let (i, outer_body) = counted(outer, rows)?;
    let [column_loop] = outer_body.statements.as_slice() else {
        return None;
    };
    let (j, column_body) = counted(column_loop, cols)?;
    let [seed, reduction, store] = column_body.statements.as_slice() else {
        return None;
    };
    let StmtKind::Binding {
        mutable: true,
        name: sum,
        value,
        ..
    } = &seed.kind
    else {
        return None;
    };
    if !zero(value) {
        return None;
    }
    let (k, reduction_body) = counted(reduction, inner)?;
    let [step] = reduction_body.statements.as_slice() else {
        return None;
    };
    let StmtKind::Assign {
        place: Place::Name(name, _),
        value,
    } = &step.kind
    else {
        return None;
    };
    if name != sum {
        return None;
    }
    let ExprKind::Binary {
        left,
        op: BinaryOp::Add,
        right,
        ..
    } = &ungroup(value).kind
    else {
        return None;
    };
    if !named(left, sum) {
        return None;
    }
    let ExprKind::Binary {
        left,
        op: BinaryOp::Mul,
        right,
        ..
    } = &ungroup(right).kind
    else {
        return None;
    };
    if !indexed(left, a, i, inner, k) || !indexed(right, b, k, cols, j) {
        return None;
    }
    let StmtKind::Assign {
        place: Place::Index { name, index, .. },
        value,
    } = &store.kind
    else {
        return None;
    };
    if name != output || !flat(index, i, cols, j) || !named(value, sum) {
        return None;
    }
    // The typed lowering and ownership checker have already validated every
    // scalar port, read loan, owner transfer and ordered source effect.
    Some(Contract {
        version: VERSION,
        function: function.name.clone(),
        span: function.span,
        loop_span: outer.span,
        loop_spans: [outer.span, column_loop.span, reduction.span],
        loop_ticks: [
            1_u64.checked_add(statement_ticks(column_loop)?)?,
            1_u64
                .checked_add(statement_ticks(seed)?)?
                .checked_add(statement_ticks(reduction)?)?
                .checked_add(statement_ticks(store)?)?,
            1_u64.checked_add(statement_ticks(step)?)?,
        ],
        inputs: [a.clone(), b.clone()],
        output: output.clone(),
        dimensions: [rows.clone(), inner.clone(), cols.clone()],
        numeric_policy: NumericPolicy::STRICT_F32,
        layout_version: crate::layout_contract::VERSION,
        function_contract: function_fingerprint(lowered),
    })
}

pub mod host;

pub(crate) fn annotate(source: &CheckedProgram, program: &mir::Program) -> Vec<Contract> {
    source
        .program
        .functions
        .iter()
        .filter_map(|f| {
            program
                .function(&f.name)
                .and_then(|lowered| recognize(f, lowered))
        })
        .collect()
}
pub(crate) fn verify(program: &mir::Program) -> Result<(), mir::Error> {
    if program.tensor_contracts.len() > program.functions.len() {
        return Err(mir::invalid("too many tensor contracts"));
    }
    let mut seen = std::collections::HashSet::new();
    for contract in &program.tensor_contracts {
        let function = program
            .function(&contract.function)
            .ok_or_else(|| mir::invalid("tensor contract function missing"))?;
        if contract.version != VERSION
            || !seen.insert(&contract.function)
            || contract.numeric_policy != program.numeric_policy
            || contract.layout_version != program.layout_version
            || contract.span != function.span
            || function_fingerprint(function) != contract.function_contract
        {
            return Err(mir::invalid("stale or mismatched tensor contract"));
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum Error {
    Source(front::FrontError),
    Mir(mir::Error),
    Net(general_net::Error),
    Unrecognized,
    Layout(crate::layout_contract::LayoutError),
    Device(invar_device::floating::Error),
    Cache(crate::tensor_cache::Error),
}
#[derive(Clone, Debug)]
pub struct Contraction {
    net: general_net::Net,
    proof: general_net::Proof,
    contract: Contract,
}
impl Contraction {
    pub fn from_checked(source: &CheckedProgram, entry: &str) -> Result<Self, Error> {
        let net =
            general_net::Net::lower(source, general_net::Limits::default()).map_err(Error::Net)?;
        let proof = net.verify().map_err(Error::Net)?;
        let program = net.compiled_mir(&proof).map_err(Error::Net)?;
        let contract = program
            .tensor_contracts
            .into_iter()
            .find(|c| c.function == entry)
            .ok_or(Error::Unrecognized)?;
        Ok(Self {
            net,
            proof,
            contract,
        })
    }
    pub fn contract(&self) -> &Contract {
        &self.contract
    }
    pub fn source_identity(&self) -> u64 {
        self.net.graph.source_identity
    }
    pub fn graph_revision(&self) -> u64 {
        self.net.graph.revision
    }
    pub fn compiled_mir(&self) -> Result<mir::Program, Error> {
        self.net.compiled_mir(&self.proof).map_err(Error::Net)
    }
}

/// Source-verified code and identity. This object grants no execution permit.
pub struct Emission {
    contraction: Contraction,
    layouts: BoundLayouts,
    plan: invar_device::floating::Verified,
    source: String,
    key: crate::tensor_cache::CacheKey,
}
impl Emission {
    #[cfg(any(test, feature = "cuda", all(feature = "metal", target_os = "macos")))]
    pub(crate) fn into_plan_key(
        self,
    ) -> (
        invar_device::floating::Verified,
        crate::tensor_cache::CacheKey,
    ) {
        (self.plan, self.key)
    }
    pub fn contraction(&self) -> &Contraction {
        &self.contraction
    }
    pub fn layouts(&self) -> &BoundLayouts {
        &self.layouts
    }
    pub fn plan(&self) -> &invar_device::floating::Verified {
        &self.plan
    }
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn key(&self) -> &crate::tensor_cache::CacheKey {
        &self.key
    }
    pub fn matches_hint(&self, hint: &crate::tensor_cache::CacheKey) -> bool {
        &self.key == hint
    }
}

/// Reverify ordinary source before emitting or consulting a content-key hint.
/// Concrete lengths are metadata only; live ownership, source fuel/trap commit
/// and device fences remain the host execution boundary's responsibility.
pub fn emit_cuda(
    source: &[u8],
    entry: &str,
    dimensions: [u32; 3],
    input_lengths: [u64; 2],
    capabilities: invar_device::Capabilities,
) -> Result<Emission, Error> {
    emit_for(
        source,
        entry,
        dimensions,
        input_lengths,
        capabilities,
        false,
    )
}

/// The Metal emission of the same verified plan; identical source admission,
/// layout binding and key fields, with the MSL target and options.
pub fn emit_msl(
    source: &[u8],
    entry: &str,
    dimensions: [u32; 3],
    input_lengths: [u64; 2],
    capabilities: invar_device::Capabilities,
) -> Result<Emission, Error> {
    emit_for(source, entry, dimensions, input_lengths, capabilities, true)
}

/// Emit for whichever device family `capabilities` names.
pub fn emit(
    source: &[u8],
    entry: &str,
    dimensions: [u32; 3],
    input_lengths: [u64; 2],
    capabilities: invar_device::Capabilities,
) -> Result<Emission, Error> {
    let metal = matches!(capabilities, invar_device::Capabilities::Metal { .. });
    emit_for(
        source,
        entry,
        dimensions,
        input_lengths,
        capabilities,
        metal,
    )
}

fn emit_for(
    source: &[u8],
    entry: &str,
    dimensions: [u32; 3],
    input_lengths: [u64; 2],
    capabilities: invar_device::Capabilities,
    metal: bool,
) -> Result<Emission, Error> {
    use invar_device::floating::{self, Buffer, Ir};
    let checked = crate::check::check(front::parse(source).map_err(Error::Source)?, false)
        .map_err(Error::Source)?;
    crate::ownership::check(&checked).map_err(Error::Source)?;
    let contraction = Contraction::from_checked(&checked, entry)?;
    let layouts = contraction
        .bind_layouts(dimensions, input_lengths)
        .map_err(Error::Layout)?;
    let [rows, inner, cols] = dimensions;
    let buffers: [Buffer; 3] = std::array::from_fn(|i| Buffer {
        id: i as u32,
        elements: layouts.allocation_bytes[i] / 4,
        shape: [[rows, inner], [inner, cols], [rows, cols]][i],
        strides: layouts.strides[i],
        offset: 0,
        alignment: 4,
    });
    let [a, b, output] = buffers;
    let contract = contraction.contract();
    let span = contract.span();
    let origin = invar_device::Provenance {
        source_identity: contraction.source_identity(),
        graph_revision: contraction.graph_revision(),
    };
    let plan = floating::verify(
        Ir {
            version: floating::VERSION,
            numeric_version: contract.numeric_policy().version(),
            numeric_policy: contract.numeric_policy().id().to_owned(),
            layout_version: contract.layout_version(),
            span: invar_device::Span {
                start: span.start,
                end: span.end,
            },
            source_identity: origin.source_identity,
            graph_revision: origin.graph_revision,
            function_contract: contract.function_contract(),
            dimensions,
            inputs: [a, b],
            output,
        },
        capabilities,
        origin,
    )
    .map_err(Error::Device)?;
    let (emitted, options, target) = if metal {
        (
            floating::emit_msl(&plan).map_err(Error::Device)?,
            plan.msl_options().to_vec(),
            "metal-msl-3.1",
        )
    } else {
        (
            floating::emit_cuda(&plan).map_err(Error::Device)?,
            plan.nvrtc_options()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            "cuda-nvrtc-c++17",
        )
    };
    let key = crate::tensor_cache::CacheKey::new(&crate::tensor_cache::KeyInput {
        source,
        entry,
        source_identity: origin.source_identity,
        graph_revision: origin.graph_revision,
        function_contract: contract.function_contract(),
        numeric_policy: contract.numeric_policy(),
        layout_version: contract.layout_version(),
        dimensions,
        layouts: layouts.specs(),
        compiler_build: env!("INVAR_BUILD_DIGEST"),
        target,
        compiler_options: &options,
        emitted_source: emitted.as_bytes(),
    })
    .map_err(Error::Cache)?;
    Ok(Emission {
        contraction,
        layouts,
        plan,
        source: emitted,
        key,
    })
}

/// Metadata for the actual logical arrays; it confers no pointer/lease lifetime.
#[derive(Debug)]
pub struct BoundLayouts {
    dimensions: [u32; 3],
    shapes: [[u64; 2]; 3],
    strides: [[u64; 2]; 3],
    allocation_bytes: [u64; 3],
}
impl BoundLayouts {
    pub fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }
    pub fn specs(&self) -> [crate::layout_contract::LayoutSpec<'_>; 3] {
        std::array::from_fn(|i| crate::layout_contract::LayoutSpec {
            element: crate::layout_contract::LayoutElement::F32,
            shape: &self.shapes[i],
            strides: &self.strides[i],
            offset: 0,
            allocation_bytes: self.allocation_bytes[i],
            base_alignment: 4,
        })
    }
    pub fn checked(
        &self,
    ) -> Result<[crate::layout_contract::CheckedLayout; 3], crate::layout_contract::LayoutError>
    {
        let [a, b, c] = self.specs();
        Ok([a.verify()?, b.verify()?, c.verify()?])
    }
}
impl Contraction {
    /// Bind concrete u32 dimensions and live input lengths to the verified ports.
    /// Rejection is optimization-specific: source errors/fuel still use the CPU.
    pub fn bind_layouts(
        &self,
        dimensions: [u32; 3],
        input_lengths: [u64; 2],
    ) -> Result<BoundLayouts, crate::layout_contract::LayoutError> {
        use crate::layout_contract::LayoutError as E;
        let [rows, inner, cols] = dimensions.map(u64::from);
        let output = rows.checked_mul(cols).ok_or(E::ArithmeticOverflow)?;
        if input_lengths
            .iter()
            .any(|&n| n > crate::eval::MAX_ARRAY_LEN as u64)
            || output > crate::eval::MAX_ARRAY_LEN as u64
        {
            return Err(E::BufferLimit);
        }
        let lengths = [input_lengths[0], input_lengths[1], output];
        let bindings = BoundLayouts {
            dimensions,
            shapes: [[rows, inner], [inner, cols], [rows, cols]],
            strides: [[inner, 1], [cols, 1], [cols, 1]],
            allocation_bytes: lengths.map(|n| n * 4),
        };
        bindings.checked()?;
        Ok(bindings)
    }
}
