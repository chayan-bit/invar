//! Source binding for fresh-output, direct-cell binary32 contraction: the
//! strict `c = c + a * b` step, or the explicit `c = tc_mul_add(a, b, c)` step
//! of `tc-f16-f32-v1` (D5 as amended). Reuses the existing verified physical
//! plan and emitters; grants no source replacement, pointer ownership, runtime
//! publication or device-launch permit.
#![forbid(unsafe_code)]

use crate::hex::hex;
use crate::{
    check,
    front::{
        self, BinaryOp, Expr, ExprKind, Mode, Place, Span, Stmt, StmtKind, Type, named, ungroup,
    },
    general_net, layout_contract, mir,
    numeric_contract::{F32Bits, NumericPolicy},
    tensor_cache,
};
use invar_device::{Capabilities, Provenance, floating, gemm, tc};
use sha2::{Digest, Sha256};

/// Matrix-family metadata ceiling; the source evaluator and shared runtimes
/// enforce their own independent resource budgets.
pub const MAX_ELEMENTS: u64 = floating::MAX_MATRIX_ELEMENTS;

#[derive(Debug)]
pub enum Error {
    Source(front::FrontError),
    Net(general_net::Error),
    Unrecognized,
    SourceMismatch,
    MirMismatch,
    CandidateMismatch,
    Bounds,
    Layout(layout_contract::LayoutError),
    Device(floating::Error),
    Cache(tensor_cache::Error),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(e) => write!(f, "contraction source: {e:?}"),
            Self::Net(e) => write!(f, "contraction typed extraction: {e:?}"),
            Self::Layout(e) => write!(f, "contraction layout: {e:?}"),
            Self::Device(e) => match e.code() {
                Some(code) => write!(f, "{code}: contraction physical plan: {e:?}"),
                None => write!(f, "contraction physical plan: {e:?}"),
            },
            Self::Cache(e) => write!(f, "contraction content identity: {e:?}"),
            Self::Unrecognized => f.write_str("unrecognized direct-cell contraction"),
            Self::SourceMismatch => f.write_str("contraction source mismatch"),
            Self::MirMismatch => f.write_str("contraction MIR mismatch"),
            Self::CandidateMismatch => f.write_str("contraction candidate mismatch"),
            Self::Bounds => f.write_str("contraction candidate bounds exceeded"),
        }
    }
}
impl std::error::Error for Error {}

/// Only actual parsed/checked ordinary source can construct this binding.
#[derive(Debug)]
pub struct Contract {
    net: general_net::Net,
    proof: general_net::Proof,
    source_hash: [u8; 32],
    mir_hash: [u8; 32],
    function: String,
    loop_span: Span,
    loop_spans: [Span; 3],
    inputs: [String; 2],
    dimensions: [String; 3],
    output: String,
    ticks: [u64; 3],
    policy: NumericPolicy,
}

/// The recognized family shape of one entry function.
struct Shape {
    inputs: [String; 2],
    dimensions: [String; 3],
    output: String,
    loop_span: Span,
    loop_spans: [Span; 3],
    ticks: [u64; 3],
    /// The `tc_mul_add` call span of a `tc-f16-f32-v1` step.
    tc_step: Option<Span>,
}

fn product(e: &Expr, a: &str, b: &str) -> bool {
    matches!(&ungroup(e).kind, ExprKind::Binary { left, op: BinaryOp::Mul, right, .. }
        if named(left, a) && named(right, b))
}
fn flat(e: &Expr, row: &str, stride: &str, col: &str) -> bool {
    matches!(&ungroup(e).kind, ExprKind::Binary { left, op: BinaryOp::Add, right, .. }
        if product(left, row, stride) && named(right, col))
}
fn indexed(e: &Expr, array: &str, row: &str, stride: &str, col: &str) -> bool {
    matches!(&ungroup(e).kind, ExprKind::Index { base, index }
        if named(base, array) && flat(index, row, stride, col))
}
fn zero(e: &Expr) -> bool {
    matches!(&ungroup(e).kind, ExprKind::Float(raw)
        if F32Bits::from_decimal(raw).is_ok_and(|v| v == F32Bits::POS_ZERO))
}
fn counted<'a>(s: &'a Stmt, bound: &str) -> Option<(&'a str, &'a front::Block)> {
    let StmtKind::For {
        name,
        lower,
        upper,
        body,
    } = &s.kind
    else {
        return None;
    };
    (matches!(&ungroup(lower).kind, ExprKind::Integer(n) if n == "0u32" || n == "0")
        && named(upper, bound))
    .then_some((name, body))
}
fn expr_ticks(e: &Expr) -> Option<u64> {
    match &e.kind {
        ExprKind::Group(inner) => expr_ticks(inner),
        ExprKind::Name(_) | ExprKind::Integer(_) | ExprKind::Float(_) => Some(1),
        ExprKind::Binary { left, right, .. }
        | ExprKind::Index {
            base: left,
            index: right,
        } => expr_ticks(left)?
            .checked_add(expr_ticks(right)?)?
            .checked_add(1),
        ExprKind::Call { args, .. } => args
            .iter()
            .try_fold(1u64, |n, arg| n.checked_add(expr_ticks(arg)?)),
        _ => None,
    }
}
fn statement_ticks(s: &Stmt) -> Option<u64> {
    match &s.kind {
        StmtKind::For { lower, upper, .. } => expr_ticks(lower)?
            .checked_add(expr_ticks(upper)?)?
            .checked_add(1),
        StmtKind::Assign {
            place: Place::Index { index, .. },
            value,
        } => expr_ticks(value)?
            .checked_add(expr_ticks(index)?)?
            .checked_add(1),
        _ => None,
    }
}

/// Recognize the family on a parsed or checked program's entry function.
fn shape(program: &front::Program, entry: &str) -> Result<Shape, Error> {
    let f = program
        .functions
        .iter()
        .find(|f| f.name == entry)
        .ok_or(Error::Unrecognized)?;
    if f.params.len() != 5
        || f.result != Type::ArrayF32
        || f.return_from.is_some()
        || !f.generic_params.is_empty()
        || f.params[..2]
            .iter()
            .any(|p| p.ty != Type::ArrayF32 || p.mode != Some(Mode::Read))
        || f.params[2..]
            .iter()
            .any(|p| p.ty != Type::U32 || p.mode.is_some())
    {
        return Err(Error::Unrecognized);
    }
    let [allocate, outer, ret] = f.body.statements.as_slice() else {
        return Err(Error::Unrecognized);
    };
    let [a, b, rows, inner, cols] = std::array::from_fn::<_, 5, _>(|i| &f.params[i].name);
    let StmtKind::Binding {
        mutable: true,
        name: out,
        value,
        ..
    } = &allocate.kind
    else {
        return Err(Error::Unrecognized);
    };
    if !matches!(&ungroup(value).kind, ExprKind::Call { name, args }
        if name == "filled_f32" && args.len() == 2 && product(&args[0], rows, cols) && zero(&args[1]))
    {
        return Err(Error::Unrecognized);
    }
    if !matches!(&ret.kind, StmtKind::Return(Some(e))
        if named(e,out) || matches!(&ungroup(e).kind, ExprKind::Move(n) if n==out))
    {
        return Err(Error::Unrecognized);
    }
    let (i, ib) = counted(outer, rows).ok_or(Error::Unrecognized)?;
    let [column] = ib.statements.as_slice() else {
        return Err(Error::Unrecognized);
    };
    let (j, jb) = counted(column, cols).ok_or(Error::Unrecognized)?;
    let [reduction] = jb.statements.as_slice() else {
        return Err(Error::Unrecognized);
    };
    let (k, kb) = counted(reduction, inner).ok_or(Error::Unrecognized)?;
    let [step] = kb.statements.as_slice() else {
        return Err(Error::Unrecognized);
    };
    let StmtKind::Assign {
        place: Place::Index { name, index, .. },
        value,
    } = &step.kind
    else {
        return Err(Error::Unrecognized);
    };
    if name != out || !flat(index, i, cols, j) {
        return Err(Error::Unrecognized);
    }
    let tc_step = match &ungroup(value).kind {
        ExprKind::Binary {
            left,
            op: BinaryOp::Add,
            right,
            ..
        } if indexed(left, out, i, cols, j)
            && matches!(&ungroup(right).kind, ExprKind::Binary { left, op: BinaryOp::Mul, right, .. }
                if indexed(left,a,i,inner,k) && indexed(right,b,k,cols,j)) =>
        {
            None
        }
        ExprKind::Call { name, args }
            if name == "tc_mul_add"
                && matches!(args.as_slice(), [x, y, acc]
                    if indexed(x, a, i, inner, k) && indexed(y, b, k, cols, j) && indexed(acc, out, i, cols, j)) =>
        {
            Some(ungroup(value).span)
        }
        _ => return Err(Error::Unrecognized),
    };
    let ticks = [
        1u64.checked_add(statement_ticks(column).ok_or(Error::Bounds)?),
        1u64.checked_add(statement_ticks(reduction).ok_or(Error::Bounds)?),
        1u64.checked_add(statement_ticks(step).ok_or(Error::Bounds)?),
    ]
    .map(|n| n.ok_or(Error::Bounds));
    let [row, col, step] = ticks;
    Ok(Shape {
        inputs: [a.clone(), b.clone()],
        dimensions: [rows.clone(), inner.clone(), cols.clone()],
        output: out.clone(),
        loop_span: outer.span,
        loop_spans: [outer.span, column.span, reduction.span],
        ticks: [row?, col?, step?],
        tc_step,
    })
}

/// Call sites admitted by ordinary checking; every other use stays unsupported.
pub(crate) fn tc_steps(program: &front::Program) -> Vec<Span> {
    program
        .functions
        .iter()
        .filter_map(|f| shape(program, &f.name).ok()?.tc_step)
        .collect()
}

/// Whether a checked function uses the direct-cell TC source contract.
pub(crate) fn is_tc(program: &front::Program, entry: &str) -> bool {
    shape(program, entry).is_ok_and(|s| s.tc_step.is_some())
}

/// Serial MIR and CPU-net completion need the exact D5 sum, like the source oracle.
pub(crate) fn mir_is_tc(function: &mir::Function) -> bool {
    function
        .blocks
        .iter()
        .flat_map(|b| &b.instructions)
        .any(|i| matches!(&i.kind, mir::Operation::Call { name, .. } if name == "tc_mul_add"))
}

/// A cheap shape filter for the registry; never grants source admission.
pub(crate) fn matches_loop(source: &[u8], entry: &str, span: Span) -> bool {
    front::parse(source)
        .ok()
        .and_then(|p| shape(&p, entry).ok())
        .is_some_and(|s| s.loop_span == span)
}

impl Contract {
    /// Recognize only the direct-output-cell family: fresh +0 output, i/j/k
    /// counted loops, and either the ordered strict step `C + A*B` or the
    /// explicit `tc_mul_add(A, B, C)` step, with the same C load/store.
    /// Scalar-accumulator contractions remain owned by `tensor::Contraction`.
    pub fn from_source(source: &[u8], entry: &str) -> Result<Self, Error> {
        let parsed = front::parse(source).map_err(Error::Source)?;
        // Admission belongs to the shared checker, including other recognized
        // TC functions in this program; extraction below selects only entry.
        let checked = check::check(parsed, false).map_err(Error::Source)?;
        let net = general_net::Net::lower(&checked, general_net::Limits::default())
            .map_err(Error::Net)?;
        let proof = net.verify().map_err(Error::Net)?;
        let m = net.compiled_mir(&proof).map_err(Error::Net)?;
        let s = shape(&checked.program, entry)?;
        Ok(Self {
            net,
            proof,
            source_hash: Sha256::digest(source).into(),
            mir_hash: m.contract_digest(),
            function: entry.to_owned(),
            loop_span: s.loop_span,
            loop_spans: s.loop_spans,
            inputs: s.inputs,
            dimensions: s.dimensions,
            output: s.output,
            ticks: s.ticks,
            policy: if s.tc_step.is_some() {
                NumericPolicy::TC_F16_F32
            } else {
                NumericPolicy::STRICT_F32
            },
        })
    }
    /// `strict-f32-v1`, or `tc-f16-f32-v1` for a `tc_mul_add` step.
    pub fn policy(&self) -> NumericPolicy {
        self.policy
    }
    pub fn function(&self) -> &str {
        &self.function
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
    pub fn loop_span(&self) -> Span {
        self.loop_span
    }
    pub fn loop_spans(&self) -> [Span; 3] {
        self.loop_spans
    }
    pub fn source_hash(&self) -> [u8; 32] {
        self.source_hash
    }
    /// Conservative whole-source-MIR binding, not the scalar family's private
    /// per-function fingerprint. Also used in the physical descriptor field.
    pub fn mir_contract(&self) -> [u8; 32] {
        self.mir_hash
    }
    pub fn verified_mir(&self) -> Result<mir::Program, Error> {
        self.net.compiled_mir(&self.proof).map_err(Error::Net)
    }
    pub fn verify_source(&self, source: &[u8]) -> Result<(), Error> {
        if <[u8; 32]>::from(Sha256::digest(source)) != self.source_hash {
            return Err(Error::SourceMismatch);
        }
        Ok(())
    }
    pub fn verify_mir(&self, program: &mir::Program) -> Result<(), Error> {
        mir::verify(program).map_err(|_| Error::MirMismatch)?;
        if program.contract_digest() != self.mir_hash {
            return Err(Error::MirMismatch);
        }
        Ok(())
    }
    /// Remaining source-loop ticks after outer dispatch/bounds; metadata only.
    pub fn loop_fuel(&self, [rows, inner, cols]: [u32; 3]) -> Option<u64> {
        let [row, col, step] = self.ticks;
        u64::from(rows).checked_mul(row.checked_add(
            u64::from(cols).checked_mul(col.checked_add(u64::from(inner).checked_mul(step)?)?)?,
        )?)
    }
    /// Generate CPU code using the existing strict physical verifier/emitter.
    /// `caps` is the frozen verifier's target metadata, not detected hardware;
    /// it opens no GPU. There is currently no CPU-only capability variant.
    pub fn emit_cpu(
        &self,
        source: &[u8],
        dims: [u32; 3],
        lengths: [u64; 2],
        caps: Capabilities,
    ) -> Result<Emission, Error> {
        // The CPU emission's target metadata stays CUDA; Metal and CUDA device
        // code comes from `emit_device` of the same verified plan.
        if !matches!(caps, Capabilities::Cuda { .. }) {
            return Err(Error::Device(floating::Error::Capabilities));
        }
        let (plan, layouts) = self.plan(source, dims, lengths, caps)?;
        let emitted = floating::emit_cpu(&plan).map_err(Error::Device)?;
        let key = self.key(
            source,
            &plan,
            "cpu-c++17-strict-direct-cell-v1",
            plan.cpu_options(),
            &emitted,
        )?;
        Ok(Emission {
            plan,
            layouts,
            source: emitted,
            key,
        })
    }

    /// Generate the tiled device schedule (`gemm`) for the target `caps` names,
    /// from the same verified plan as the CPU emission. Opens no GPU; the
    /// runtime's `run_tiled_contraction` executes exactly this plan.
    pub fn emit_device(
        &self,
        source: &[u8],
        dims: [u32; 3],
        lengths: [u64; 2],
        caps: Capabilities,
    ) -> Result<DeviceEmission, Error> {
        let (plan, layouts) = self.plan(source, dims, lengths, caps)?;
        let cuda = matches!(caps, Capabilities::Cuda { .. });
        let options = if cuda {
            plan.nvrtc_options().to_vec()
        } else {
            plan.msl_options().iter().map(|&o| o.to_owned()).collect()
        };
        let (emitted, target, schedule, tile, geometry) = match (plan.is_tc(), cuda) {
            (false, true) => (
                gemm::emit_cuda(&plan),
                "cuda-nvrtc-c++17-gemm-tiled-v1",
                gemm::SCHEDULE,
                [gemm::TILE, gemm::TILE, gemm::TILE_K],
                gemm::geometry(&plan),
            ),
            (false, false) => (
                gemm::emit_msl(&plan),
                "metal-msl-3.1-gemm-tiled-v1",
                gemm::SCHEDULE,
                [gemm::TILE, gemm::TILE, gemm::TILE_K],
                gemm::geometry(&plan),
            ),
            (true, true) => (
                tc::emit_cuda(&plan),
                "cuda-nvrtc-c++17-tc-mma-m16n8k16-v1",
                tc::CUDA_SCHEDULE,
                tc::CUDA_TILE,
                tc::geometry(&plan),
            ),
            (true, false) => (
                tc::emit_msl(&plan),
                "metal-msl-3.1-tc-simdgroup8x8-v1",
                tc::METAL_SCHEDULE,
                tc::METAL_TILE,
                tc::geometry(&plan),
            ),
        };
        let emitted = emitted.map_err(Error::Device)?;
        let borrowed = options.iter().map(String::as_str).collect::<Vec<_>>();
        let key = self.key(source, &plan, target, &borrowed, &emitted)?;
        Ok(DeviceEmission {
            geometry,
            schedule,
            tile,
            // tc-f16-f32-v1 has no CPU lowering (`floating::emit_cpu` refuses
            // it) and the strict source loop would run unrounded f32 inputs,
            // so a TC plan is always placed on the device.
            fallback: if plan.is_tc() {
                None
            } else {
                gemm::cpu_fallback(dims)
            },
            plan,
            layouts,
            source: emitted,
            options,
            target,
            key,
        })
    }

    fn plan(
        &self,
        source: &[u8],
        dims: [u32; 3],
        lengths: [u64; 2],
        caps: Capabilities,
    ) -> Result<(floating::Verified, [layout_contract::CheckedLayout; 3]), Error> {
        self.verify_source(source)?;
        let m = self.verified_mir()?;
        self.verify_mir(&m)?;
        let [rows, inner, cols] = dims;
        let shapes = [[rows, inner], [inner, cols], [rows, cols]];
        let output = u64::from(rows)
            .checked_mul(u64::from(cols))
            .ok_or(Error::Bounds)?;
        let lens = [lengths[0], lengths[1], output];
        if lens.iter().any(|&n| n > MAX_ELEMENTS) {
            return Err(Error::Bounds);
        }
        let buffers: [floating::Buffer; 3] = std::array::from_fn(|i| floating::Buffer {
            id: i as u32,
            elements: lens[i],
            shape: shapes[i],
            strides: [u64::from(shapes[i][1]), 1],
            offset: 0,
            alignment: 4,
        });
        let layouts: [layout_contract::CheckedLayout; 3] = {
            // Shapes need u64 storage distinct from the physical u32 descriptors.
            let wide = shapes.map(|s| s.map(u64::from));
            let checked: [Result<_, _>; 3] = std::array::from_fn(|i| {
                layout_contract::LayoutSpec {
                    element: layout_contract::LayoutElement::F32,
                    shape: &wide[i],
                    strides: &buffers[i].strides,
                    offset: 0,
                    allocation_bytes: lens[i] * 4,
                    base_alignment: 4,
                }
                .verify()
            });
            let [a, b, c] = checked;
            [
                a.map_err(Error::Layout)?,
                b.map_err(Error::Layout)?,
                c.map_err(Error::Layout)?,
            ]
        };
        let origin = Provenance {
            source_identity: self.net.graph.source_identity,
            graph_revision: self.net.graph.revision,
        };
        let [a, b, c] = buffers;
        let plan = floating::verify(
            floating::Ir {
                version: floating::VERSION,
                numeric_version: self.policy.version(),
                numeric_policy: self.policy.id().to_owned(),
                layout_version: layout_contract::VERSION,
                span: invar_device::Span {
                    start: self.loop_span.start,
                    end: self.loop_span.end,
                },
                source_identity: origin.source_identity,
                graph_revision: origin.graph_revision,
                function_contract: self.mir_hash,
                dimensions: dims,
                inputs: [a, b],
                output: c,
            },
            caps,
            origin,
        )
        .map_err(Error::Device)?;
        Ok((plan, layouts))
    }

    fn key(
        &self,
        source: &[u8],
        plan: &floating::Verified,
        target: &str,
        options: &[&str],
        emitted: &str,
    ) -> Result<tensor_cache::CacheKey, Error> {
        let ir = plan.ir();
        let wide =
            [ir.inputs[0].shape, ir.inputs[1].shape, ir.output.shape].map(|s| s.map(u64::from));
        tensor_cache::CacheKey::new(&tensor_cache::KeyInput {
            source,
            entry: &self.function,
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
            function_contract: self.mir_hash,
            numeric_policy: self.policy,
            layout_version: layout_contract::VERSION,
            dimensions: ir.dimensions,
            layouts: std::array::from_fn(|i| layout_contract::LayoutSpec {
                element: layout_contract::LayoutElement::F32,
                shape: &wide[i],
                strides: &plan_buffer(plan, i).strides,
                offset: 0,
                allocation_bytes: plan_buffer(plan, i).elements * 4,
                base_alignment: 4,
            }),
            compiler_build: env!("INVAR_BUILD_DIGEST"),
            target,
            compiler_options: options,
            emitted_source: emitted.as_bytes(),
        })
        .map_err(Error::Cache)
    }
}
fn plan_buffer(plan: &floating::Verified, i: usize) -> &floating::Buffer {
    match i {
        0 => &plan.ir().inputs[0],
        1 => &plan.ir().inputs[1],
        _ => &plan.ir().output,
    }
}

/// Private verified emission. A digest/descriptor copy is only a candidate.
#[derive(Debug)]
pub struct Emission {
    plan: floating::Verified,
    layouts: [layout_contract::CheckedLayout; 3],
    source: String,
    key: tensor_cache::CacheKey,
}
impl Emission {
    pub fn plan(&self) -> &floating::Verified {
        &self.plan
    }
    pub fn layouts(&self) -> &[layout_contract::CheckedLayout; 3] {
        &self.layouts
    }
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn options(&self) -> &'static [&'static str] {
        self.plan.cpu_options()
    }
    pub fn key(&self) -> &tensor_cache::CacheKey {
        &self.key
    }
    /// A valid but differently sourced/shaped/aliased descriptor is not this
    /// extraction. Options and emitted bytes must match exactly, not just hash.
    pub fn verify_candidate(
        &self,
        ir: &floating::Ir,
        source: &str,
        options: &[&str],
    ) -> Result<(), Error> {
        let origin = Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        floating::verify(ir.clone(), self.plan.capabilities(), origin).map_err(Error::Device)?;
        if ir != self.plan.ir() || source != self.source || options != self.options() {
            return Err(Error::CandidateMismatch);
        }
        Ok(())
    }
}

/// Verified tiled device emission. Like `Emission`, a copied digest or
/// descriptor is only a candidate; `fallback` names why `auto` placement keeps
/// a strict-f32-v1 shape on the CPU (the result contract is identical either
/// way). A tc-f16-f32-v1 plan never has a CPU fallback.
#[derive(Debug)]
pub struct DeviceEmission {
    plan: floating::Verified,
    layouts: [layout_contract::CheckedLayout; 3],
    source: String,
    options: Vec<String>,
    target: &'static str,
    geometry: gemm::Geometry,
    schedule: &'static str,
    tile: [u32; 3],
    fallback: Option<&'static str>,
    key: tensor_cache::CacheKey,
}
impl DeviceEmission {
    #[cfg(any(test, feature = "cuda", all(feature = "metal", target_os = "macos")))]
    pub(crate) fn into_plan_key(self) -> (floating::Verified, tensor_cache::CacheKey) {
        (self.plan, self.key)
    }
    pub fn plan(&self) -> &floating::Verified {
        &self.plan
    }
    pub fn layouts(&self) -> &[layout_contract::CheckedLayout; 3] {
        &self.layouts
    }
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn options(&self) -> &[String] {
        &self.options
    }
    pub fn target(&self) -> &'static str {
        self.target
    }
    pub fn geometry(&self) -> gemm::Geometry {
        self.geometry
    }
    pub fn fallback(&self) -> Option<&'static str> {
        self.fallback
    }
    pub fn key(&self) -> &tensor_cache::CacheKey {
        &self.key
    }
    /// Explain/receipt record: the target, numeric contract, layout version,
    /// schedule and geometry, generated-source hash and placement decision.
    pub fn receipt(&self) -> serde_json::Value {
        let ir = self.plan.ir();
        let (reference, bound) = if self.plan.is_tc() {
            (
                "exact-sum-of-f16-rounded-products-i128",
                "|r - exact| <= 2 * K * 2^-23 * sum |a_ik * b_kj|",
            )
        } else {
            ("source-order-strict-f32", "bit-exact")
        };
        serde_json::json!({
            "family": "direct-cell-matrix-contraction-v1",
            "track": "generated",
            "target": self.target,
            "capabilities": format!("{:?}", self.plan.capabilities()),
            "numeric_policy": ir.numeric_policy,
            "numeric_version": ir.numeric_version,
            "layout_version": ir.layout_version,
            "dimensions": ir.dimensions,
            "reference": reference,
            "bound": bound,
            "schedule": self.schedule,
            "tile": self.tile,
            "groups": self.geometry.groups,
            "threads": self.geometry.threads,
            "tiles": self.geometry.tiles,
            "options": self.options,
            "generated_sha256": hex(&Sha256::digest(self.source.as_bytes())),
            "key": self.key.to_hex(),
            "placement": match self.fallback {
                Some(reason) => format!("cpu:{reason}"),
                None => "device".to_owned(),
            },
        })
    }
}
