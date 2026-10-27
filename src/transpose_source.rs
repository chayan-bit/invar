//! Checked ordinary-source rank-2 transpose extraction. This contract identifies
//! a lawful loop region; it does not authorize launch, fuel debit or publication.
#![forbid(unsafe_code)]

use crate::{
    check::CheckedProgram,
    front::{self, BinaryOp, Expr, ExprKind, Mode, Place, Span, StmtKind, Type, named, ungroup},
    general_net, mir,
};
use invar_device::{Capabilities, Provenance, floating::Buffer, transpose};
pub mod host;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Source(front::FrontError),
    CheckedMismatch,
    Net(general_net::Error),
    Unrecognized,
    Bounds,
    Device(transpose::Error),
    DescriptorMismatch,
}

/// Private source/MIR-bound extraction, never reconstructed from a digest.
#[derive(Debug)]
pub struct Contract {
    net: general_net::Net,
    proof: general_net::Proof,
    function: String,
    loop_span: Span,
    inner_span: Span,
    input: String,
    output: String,
    dimensions: [String; 2],
    element: Type,
    function_contract: [u8; 32],
    ticks: [u64; 2],
}

fn zero(e: &Expr) -> bool {
    matches!(&ungroup(e).kind,ExprKind::Integer(n) if n=="0u32" || n=="0")
}
fn product(e: &Expr, a: &str, b: &str) -> bool {
    matches!(&ungroup(e).kind,ExprKind::Binary {left,op:BinaryOp::Mul,right,..} if named(left,a)&&named(right,b))
}
fn flat(e: &Expr, row: &str, stride: &str, col: &str) -> bool {
    matches!(&ungroup(e).kind,ExprKind::Binary {left,op:BinaryOp::Add,right,..} if product(left,row,stride)&&named(right,col))
}
fn expr_ticks(e: &Expr) -> Option<u64> {
    match &e.kind {
        ExprKind::Group(e) => expr_ticks(e),
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

impl Contract {
    pub fn from_checked(source: &CheckedProgram, entry: &str) -> Result<Self, Error> {
        let checked =
            crate::tensor::recheck(source, Error::Bounds, Error::CheckedMismatch, Error::Source)?;
        let net = general_net::Net::lower(&checked, general_net::Limits::default())
            .map_err(Error::Net)?;
        let proof = net.verify().map_err(Error::Net)?;
        let program = net.compiled_mir(&proof).map_err(Error::Net)?;
        let f = checked.program.function(entry).ok_or(Error::Unrecognized)?;
        if f.params.len() != 3
            || f.return_from.is_some()
            || !f.generic_params.is_empty()
            || f.params[0].mode != Some(Mode::Read)
            || f.params[1..]
                .iter()
                .any(|p| p.ty != Type::U32 || p.mode.is_some())
        {
            return Err(Error::Unrecognized);
        }
        let (element, allocator) = match (&f.params[0].ty, &f.result) {
            (Type::ArrayU32, Type::ArrayU32) => (Type::U32, "filled_u32"),
            (Type::ArrayF32, Type::ArrayF32) => (Type::F32, "filled_f32"),
            _ => return Err(Error::Unrecognized),
        };
        let [allocate, outer, ret] = f.body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::Binding {
            mutable: true,
            name: output,
            value,
            ..
        } = &allocate.kind
        else {
            return Err(Error::Unrecognized);
        };
        let ExprKind::Call { name, args } = &ungroup(value).kind else {
            return Err(Error::Unrecognized);
        };
        let input = &f.params[0].name;
        let rows = &f.params[1].name;
        let cols = &f.params[2].name;
        if name != allocator || args.len() != 2 || !product(&args[0], rows, cols) {
            return Err(Error::Unrecognized);
        }
        let fill_zero = if element == Type::U32 {
            zero(&args[1])
        } else {
            matches!(&ungroup(&args[1]).kind,ExprKind::Float(raw) if crate::numeric_contract::F32Bits::from_decimal(raw).is_ok_and(|n|n==crate::numeric_contract::F32Bits::POS_ZERO))
        };
        if !fill_zero {
            return Err(Error::Unrecognized);
        }
        let StmtKind::Return(Some(owner)) = &ret.kind else {
            return Err(Error::Unrecognized);
        };
        if !named(owner, output) && !matches!(&ungroup(owner).kind,ExprKind::Move(n) if n==output) {
            return Err(Error::Unrecognized);
        }
        let StmtKind::For {
            name: row,
            lower,
            upper,
            body,
        } = &outer.kind
        else {
            return Err(Error::Unrecognized);
        };
        if !zero(lower) || !named(upper, rows) {
            return Err(Error::Unrecognized);
        }
        let [inner] = body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::For {
            name: col,
            lower,
            upper,
            body,
        } = &inner.kind
        else {
            return Err(Error::Unrecognized);
        };
        if !zero(lower) || !named(upper, cols) {
            return Err(Error::Unrecognized);
        }
        let [store] = body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::Assign {
            place: Place::Index { name, index, .. },
            value,
        } = &store.kind
        else {
            return Err(Error::Unrecognized);
        };
        if name != output || !flat(index, col, rows, row) {
            return Err(Error::Unrecognized);
        }
        if !matches!(&ungroup(value).kind,ExprKind::Index {base,index} if named(base,input)&&flat(index,row,cols,col))
        {
            return Err(Error::Unrecognized);
        }
        let row_ticks = 1u64
            .checked_add(1)
            .and_then(|n| expr_ticks(lower).and_then(|t| n.checked_add(t)))
            .and_then(|n| expr_ticks(upper).and_then(|t| n.checked_add(t)))
            .ok_or(Error::Bounds)?;
        let column_ticks = 2u64
            .checked_add(expr_ticks(value).ok_or(Error::Bounds)?)
            .and_then(|n| expr_ticks(index).and_then(|t| n.checked_add(t)))
            .ok_or(Error::Bounds)?;
        let lowered = program.function(entry).ok_or(Error::Unrecognized)?;
        let function_contract = crate::tensor::function_fingerprint(lowered);
        let loop_span = outer.span;
        if loop_span.start >= loop_span.end
            || loop_span.start < f.span.start
            || loop_span.end > f.span.end
        {
            return Err(Error::Unrecognized);
        }
        Ok(Self {
            net,
            proof,
            function: entry.to_owned(),
            loop_span,
            inner_span: inner.span,
            input: input.clone(),
            output: output.clone(),
            dimensions: [rows.clone(), cols.clone()],
            element,
            function_contract,
            ticks: [row_ticks, column_ticks],
        })
    }
    pub fn function(&self) -> &str {
        &self.function
    }
    pub fn loop_span(&self) -> Span {
        self.loop_span
    }
    pub fn input(&self) -> &str {
        &self.input
    }
    pub fn output(&self) -> &str {
        &self.output
    }
    pub fn dimensions(&self) -> &[String; 2] {
        &self.dimensions
    }
    pub fn element(&self) -> Type {
        self.element.clone()
    }
    pub fn function_contract(&self) -> [u8; 32] {
        self.function_contract
    }
    pub fn verified_mir(&self) -> Result<mir::Program, Error> {
        self.net.compiled_mir(&self.proof).map_err(Error::Net)
    }
    /// Remaining ticks after the reached outer loop's statement and two bounds.
    /// Includes each row/column control tick, inner bounds and ordered store.
    pub fn loop_fuel(&self, [rows, cols]: [u32; 2]) -> Option<u64> {
        u64::from(rows)
            .checked_mul(self.ticks[0].checked_add(u64::from(cols).checked_mul(self.ticks[1])?)?)
    }
    pub fn emit_cuda(
        &self,
        dimensions: [u32; 2],
        input_len: u64,
        caps: Capabilities,
    ) -> Result<Emission, Error> {
        let plan = self.device_plan(dimensions, input_len, caps)?;
        let source = transpose::emit_cuda(&plan).map_err(Error::Device)?;
        Ok(Emission { plan, source })
    }
    /// Verify the shared bit-copy layout before choosing a target emitter.
    pub fn device_plan(
        &self,
        dimensions: [u32; 2],
        input_len: u64,
        caps: Capabilities,
    ) -> Result<transpose::Verified, Error> {
        let [rows, cols] = dimensions;
        // Source indices are checked u32. Interpreter allocation admission is
        // independent of this device-plan ceiling.
        let count = rows.checked_mul(cols).ok_or(Error::Bounds)?;
        if u64::from(count) > input_len
            || input_len > transpose::MAX_LOGICAL_ELEMENTS
            || u64::from(count) > transpose::MAX_LOGICAL_ELEMENTS
        {
            return Err(Error::Bounds);
        }
        let origin = Provenance {
            source_identity: self.net.graph.source_identity,
            graph_revision: self.net.graph.revision,
        };
        let ir = transpose::Ir {
            version: transpose::VERSION,
            layout_version: crate::layout_contract::VERSION,
            span: invar_device::Span {
                start: self.loop_span.start,
                end: self.loop_span.end,
            },
            source_identity: origin.source_identity,
            graph_revision: origin.graph_revision,
            function_contract: self.function_contract,
            dimensions,
            input: Buffer {
                id: 0,
                elements: input_len,
                shape: dimensions,
                strides: [u64::from(cols), 1],
                offset: 0,
                alignment: 4,
            },
            output: Buffer {
                id: 1,
                elements: u64::from(count),
                shape: [cols, rows],
                strides: [u64::from(rows), 1],
                offset: 0,
                alignment: 4,
            },
        };
        transpose::verify(ir, caps, origin).map_err(Error::Device)
    }
}

#[derive(Debug)]
pub struct Emission {
    plan: transpose::Verified,
    source: String,
}
impl Emission {
    pub fn plan(&self) -> &transpose::Verified {
        &self.plan
    }
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn options(&self) -> &[String] {
        self.plan.nvrtc_options()
    }
    /// Compare a cache/host descriptor against this source-derived opaque plan.
    /// A standalone physical verifier does not authenticate caller digests.
    pub fn verify_descriptor(&self, ir: &transpose::Ir) -> Result<(), Error> {
        if self.plan.ir() == ir {
            Ok(())
        } else {
            Err(Error::DescriptorMismatch)
        }
    }
}
