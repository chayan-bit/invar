//! Source-bound extraction of fresh radius-1 strict FP32 five-point sums.
//! This component emits one completed-slice schedule. `host` publishes a reached
//! loop only through a completed, fenced CPU ticket with an exact fuel debit;
//! repeated source calls implement multi-step ping-pong.
use crate::{
    check::CheckedProgram,
    front::{self, ElseArm, Expr, ExprKind, Mode, Place, Span, Stmt, StmtKind, Type, ungroup},
    general_net, mir,
};
use invar_device::{
    Capabilities, Provenance,
    floating::Buffer,
    stencil::{self, Boundary},
};

mod device_u32;
pub mod host;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Source(front::FrontError),
    CheckedMismatch,
    Net(general_net::Error),
    Unrecognized,
    Bounds,
    Device(stencil::Error),
    DeviceU32(invar_device::stencil_u32::Error),
    DescriptorMismatch,
}
#[derive(Debug)]
pub struct Contract {
    net: general_net::Net,
    proof: general_net::Proof,
    function: String,
    loop_span: Span,
    inner_span: Span,
    branch_spans: [Span; 4],
    input: String,
    output: String,
    dimensions: [String; 2],
    boundary: Boundary,
    function_contract: [u8; 32],
    ticks: Ticks,
    element: Type,
}
/// Source steps of the reached loop, derived from the matched checked AST with
/// the evaluator's charging rules (one per statement, expression node and loop
/// iteration). Directions are north, south, west, east.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Ticks {
    row: u64,
    cell: u64,
    load: [u64; 4],
    edge: [u64; 4],
}
/// Expression steps: one per node; parentheses are not a step.
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
        ExprKind::Call { name, args } if name == "wrapping_add" && args.len() == 2 => args
            .iter()
            .try_fold(1u64, |n, e| n.checked_add(expr_ticks(e)?)),
        _ => None,
    }
}
/// Steps of one straight-line statement admitted in the cell body.
fn stmt_ticks(s: &Stmt) -> Option<u64> {
    match &s.kind {
        StmtKind::Binding { value, .. } => expr_ticks(value)?.checked_add(1),
        StmtKind::Assign {
            place: Place::Name(..),
            value,
        } => expr_ticks(value)?.checked_add(1),
        StmtKind::Assign {
            place: Place::Index { index, .. },
            value,
        } => expr_ticks(value)?
            .checked_add(expr_ticks(index)?)?
            .checked_add(1),
        _ => None,
    }
}
fn block_ticks(statements: &[Stmt]) -> Option<u64> {
    statements
        .iter()
        .try_fold(0u64, |n, s| n.checked_add(stmt_ticks(s)?))
}
impl Ticks {
    /// `lower`/`upper` bound the inner loop; `cell` is the verified nine-statement
    /// body: four bindings, four guarded loads (north, south, west, east) and a store.
    fn derive(lower: &Expr, upper: &Expr, cell: &[Stmt]) -> Option<Self> {
        // Outer iteration + inner `for` statement + its two bound expressions.
        let row = 2u64
            .checked_add(expr_ticks(lower)?)?
            .checked_add(expr_ticks(upper)?)?;
        // Inner iteration + bindings + guard statements/conditions + store.
        let mut base = 1u64;
        let mut load = [0u64; 4];
        let mut edge = [0u64; 4];
        for (i, s) in cell.iter().enumerate() {
            match (&s.kind, i) {
                (
                    StmtKind::If {
                        condition,
                        then_block,
                        else_arm,
                    },
                    1 | 3 | 5 | 7,
                ) => {
                    base = base.checked_add(expr_ticks(condition)?.checked_add(1)?)?;
                    let d = (i - 1) / 2;
                    load[d] = block_ticks(&then_block.statements)?;
                    edge[d] = match else_arm {
                        None => 0,
                        Some(ElseArm::Block(b)) => block_ticks(&b.statements)?,
                        Some(ElseArm::If(_)) => return None,
                    };
                }
                (_, 0 | 2 | 4 | 6 | 8) => base = base.checked_add(stmt_ticks(s)?)?,
                _ => return None,
            }
        }
        Some(Self {
            row,
            cell: base,
            load,
            edge,
        })
    }
}
fn same_expr(a: &Expr, b: &Expr) -> bool {
    match (&ungroup(a).kind, &ungroup(b).kind) {
        (ExprKind::Name(a), ExprKind::Name(b)) | (ExprKind::Move(a), ExprKind::Move(b)) => a == b,
        (ExprKind::Integer(a), ExprKind::Integer(b)) => {
            a.trim_end_matches("u32") == b.trim_end_matches("u32")
        }
        (ExprKind::Float(a), ExprKind::Float(b)) => {
            crate::numeric_contract::F32Bits::from_decimal(a).is_ok_and(|a| {
                crate::numeric_contract::F32Bits::from_decimal(b).is_ok_and(|b| a == b)
            })
        }
        (
            ExprKind::Binary {
                left: a,
                op: ao,
                right: ar,
                ..
            },
            ExprKind::Binary {
                left: b,
                op: bo,
                right: br,
                ..
            },
        ) => ao == bo && same_expr(a, b) && same_expr(ar, br),
        (ExprKind::Index { base: a, index: ai }, ExprKind::Index { base: b, index: bi }) => {
            same_expr(a, b) && same_expr(ai, bi)
        }
        (ExprKind::Call { name: a, args: aa }, ExprKind::Call { name: b, args: ba }) => {
            a == b && aa.len() == ba.len() && aa.iter().zip(ba).all(|(a, b)| same_expr(a, b))
        }
        _ => false,
    }
}
fn same_place(a: &Place, b: &Place) -> bool {
    match (a, b) {
        (Place::Name(a, _), Place::Name(b, _)) => a == b,
        (
            Place::Index {
                name: a, index: ai, ..
            },
            Place::Index {
                name: b, index: bi, ..
            },
        ) => a == b && same_expr(ai, bi),
        _ => false,
    }
}
fn same_stmts(a: &[Stmt], b: &[Stmt]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_stmt(a, b))
}
fn same_stmt(a: &Stmt, b: &Stmt) -> bool {
    match (&a.kind, &b.kind) {
        (
            StmtKind::Binding {
                mutable: a,
                name: an,
                annotation: at,
                value: av,
            },
            StmtKind::Binding {
                mutable: b,
                name: bn,
                annotation: bt,
                value: bv,
            },
        ) => a == b && an == bn && at == bt && same_expr(av, bv),
        (
            StmtKind::Assign {
                place: a,
                value: av,
            },
            StmtKind::Assign {
                place: b,
                value: bv,
            },
        ) => same_place(a, b) && same_expr(av, bv),
        (StmtKind::Return(Some(a)), StmtKind::Return(Some(b))) => {
            same_expr(a, b)
                || matches!((&ungroup(a).kind,&ungroup(b).kind),(ExprKind::Move(a),ExprKind::Name(b)) if a==b)
        }
        (
            StmtKind::For {
                name: a,
                lower: al,
                upper: au,
                body: ab,
            },
            StmtKind::For {
                name: b,
                lower: bl,
                upper: bu,
                body: bb,
            },
        ) => {
            a == b
                && same_expr(al, bl)
                && same_expr(au, bu)
                && same_stmts(&ab.statements, &bb.statements)
        }
        (
            StmtKind::If {
                condition: a,
                then_block: at,
                else_arm: ae,
            },
            StmtKind::If {
                condition: b,
                then_block: bt,
                else_arm: be,
            },
        ) => {
            same_expr(a, b)
                && same_stmts(&at.statements, &bt.statements)
                && match (ae, be) {
                    (None, None) => true,
                    (Some(ElseArm::Block(a)), Some(ElseArm::Block(b))) => {
                        same_stmts(&a.statements, &b.statements)
                    }
                    _ => false,
                }
        }
        _ => false,
    }
}

impl Contract {
    /// Recognizes the operation structurally, with arbitrary function, parameter,
    /// loop and temporary names. No caller policy/proof flag grants admission.
    /// Unknown expressions, calls, effects or grouping of the sum are refused.
    pub fn from_checked(source: &CheckedProgram, entry: &str) -> Result<Self, Error> {
        let checked =
            crate::tensor::recheck(source, Error::Bounds, Error::CheckedMismatch, Error::Source)?;
        let net = general_net::Net::lower(&checked, general_net::Limits::default())
            .map_err(Error::Net)?;
        let proof = net.verify().map_err(Error::Net)?;
        let program = net.compiled_mir(&proof).map_err(Error::Net)?;
        let f = checked.program.function(entry).ok_or(Error::Unrecognized)?;
        if f.params.len() != 3
            || !matches!(f.params[0].ty, Type::ArrayF32 | Type::ArrayU32)
            || f.params[0].mode != Some(Mode::Read)
            || f.params[1..]
                .iter()
                .any(|p| p.ty != Type::U32 || p.mode.is_some())
            || f.result != f.params[0].ty
            || f.return_from.is_some()
            || !f.generic_params.is_empty()
        {
            return Err(Error::Unrecognized);
        }
        let [allocate, outer, _] = f.body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::Binding { name: output, .. } = &allocate.kind else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::For {
            name: row, body, ..
        } = &outer.kind
        else {
            return Err(Error::Unrecognized);
        };
        let [inner] = body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::For {
            name: col,
            lower: col_lower,
            upper: col_upper,
            body,
        } = &inner.kind
        else {
            return Err(Error::Unrecognized);
        };
        if body.statements.len() != 9 {
            return Err(Error::Unrecognized);
        }
        let mut names = Vec::new();
        for i in [0, 2, 4, 6] {
            let StmtKind::Binding { name, .. } = &body.statements[i].kind else {
                return Err(Error::Unrecognized);
            };
            names.push(name.as_str());
        }
        let [n, s, w, e] = names.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let input = &f.params[0].name;
        let rows = &f.params[1].name;
        let cols = &f.params[2].name;
        let center = format!("{input}[{row}*{cols}+{col}]");
        // Construct an AST pattern using the observed binders, then compare
        // structure without spans/groups. The four branch loads establish halo
        // bounds; the fresh allocator and sole write establish disjoint effects.
        let mut matched = None;
        for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
            let edge = |name: &str| {
                if boundary == Boundary::Clamp {
                    format!("else {{ {name}={center}; }}")
                } else {
                    String::new()
                }
            };
            let pattern = format!(
                "fn pattern({input}:read Array<f32>,{rows}:u32,{cols}:u32)->Array<f32>{{var {output}=filled_f32({rows}*{cols},0.0); for {row} in 0u32..{rows}{{for {col} in 0u32..{cols}{{var {n}=0.0; if {row}>0u32{{{n}={input}[({row}-1u32)*{cols}+{col}];}} {ne} var {s}=0.0; if {row}+1u32<{rows}{{{s}={input}[({row}+1u32)*{cols}+{col}];}} {se} var {w}=0.0; if {col}>0u32{{{w}={input}[{row}*{cols}+({col}-1u32)];}} {we} var {e}=0.0; if {col}+1u32<{cols}{{{e}={input}[{row}*{cols}+({col}+1u32)];}} {ee} {output}[{row}*{cols}+{col}]=((({center}+{n})+{s})+{w})+{e};}}}}return {output};}}",
                ne = edge(n),
                se = edge(s),
                we = edge(w),
                ee = edge(e)
            );
            let pattern = if f.params[0].ty == Type::ArrayU32 {
                pattern.replace("Array<f32>", "Array<u32>").replace("filled_f32", "filled_u32").replace("0.0", "0u32").replace(&format!("((({center}+{n})+{s})+{w})+{e}"), &format!("wrapping_add(wrapping_add(wrapping_add(wrapping_add({center},{n}),{s}),{w}),{e})"))
            } else {
                pattern
            };
            if pattern.len() > invar_device::MAX_SOURCE_BYTES {
                return Err(Error::Bounds);
            }
            let parsed = front::parse(pattern.as_bytes()).map_err(|_| Error::Unrecognized)?;
            let expected = parsed.functions.first().ok_or(Error::Unrecognized)?;
            if same_stmts(&f.body.statements, &expected.body.statements) {
                matched = Some(boundary);
                break;
            }
        }
        let boundary = matched.ok_or(Error::Unrecognized)?;
        let ticks = Ticks::derive(col_lower, col_upper, &body.statements).ok_or(Error::Bounds)?;
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
            function: entry.into(),
            loop_span,
            inner_span: inner.span,
            branch_spans: [1, 3, 5, 7].map(|i| body.statements[i].span),
            input: input.clone(),
            output: output.clone(),
            dimensions: [rows.clone(), cols.clone()],
            boundary,
            function_contract,
            ticks,
            element: f.params[0].ty.clone(),
        })
    }
    /// Exact source steps from the reached outer-loop head to its exit, or
    /// `None` on overflow. Guards decide per cell whether a neighbour load or
    /// the boundary edge statement runs: north needs row > 0, south row + 1 <
    /// rows, west col > 0, east col + 1 < cols.
    pub fn loop_fuel(&self, [rows, cols]: [u32; 2]) -> Option<u64> {
        let (rows, cols) = (u64::from(rows), u64::from(cols));
        let cells = rows.checked_mul(cols)?;
        let vertical = rows.saturating_sub(1).checked_mul(cols)?;
        let horizontal = rows.checked_mul(cols.saturating_sub(1))?;
        let inside = [vertical, vertical, horizontal, horizontal];
        let mut total = rows
            .checked_mul(self.ticks.row)?
            .checked_add(cells.checked_mul(self.ticks.cell)?)?;
        for ((inside, load), edge) in inside.iter().zip(self.ticks.load).zip(self.ticks.edge) {
            total = total
                .checked_add(inside.checked_mul(load)?)?
                .checked_add((cells - inside).checked_mul(edge)?)?;
        }
        Some(total)
    }
    pub fn element(&self) -> Type {
        self.element.clone()
    }
    pub fn function(&self) -> &str {
        &self.function
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
    pub fn boundary(&self) -> Boundary {
        self.boundary
    }
    pub fn loop_span(&self) -> Span {
        self.loop_span
    }
    pub fn function_contract(&self) -> [u8; 32] {
        self.function_contract
    }
    pub fn verified_mir(&self) -> Result<mir::Program, Error> {
        self.net.compiled_mir(&self.proof).map_err(Error::Net)
    }
    pub fn emit_cuda(
        &self,
        dimensions: [u32; 2],
        input_len: u64,
        caps: Capabilities,
    ) -> Result<Emission, Error> {
        if self.element != Type::ArrayF32 {
            return Err(Error::Unrecognized);
        }
        let [rows, cols] = dimensions;
        let count = rows.checked_mul(cols).ok_or(Error::Bounds)?;
        if u64::from(count) > input_len
            || input_len > invar_device::transpose::MAX_LOGICAL_ELEMENTS
            || u64::from(count) > invar_device::transpose::MAX_LOGICAL_ELEMENTS
        {
            return Err(Error::Bounds);
        }
        let origin = Provenance {
            source_identity: self.net.graph.source_identity,
            graph_revision: self.net.graph.revision,
        };
        let ir = stencil::Ir {
            version: stencil::VERSION,
            numeric_version: crate::numeric_contract::NumericPolicy::STRICT_F32.version(),
            numeric_policy: crate::numeric_contract::NumericPolicy::STRICT_F32
                .id()
                .into(),
            layout_version: crate::layout_contract::VERSION,
            boundary_version: stencil::BOUNDARY_VERSION,
            boundary: self.boundary,
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
                shape: dimensions,
                strides: [u64::from(cols), 1],
                offset: 0,
                alignment: 4,
            },
        };
        let plan = stencil::verify(ir, caps, origin).map_err(Error::Device)?;
        let source = stencil::emit_cuda(&plan).map_err(Error::Device)?;
        Ok(Emission { plan, source })
    }
}
#[derive(Debug)]
pub struct Emission {
    plan: stencil::Verified,
    source: String,
}
impl Emission {
    pub fn plan(&self) -> &stencil::Verified {
        &self.plan
    }
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn options(&self) -> &[String] {
        self.plan.nvrtc_options()
    }
    /// Physical verification alone does not authenticate invented provenance or
    /// function hashes. Compare every field with the opaque source-derived plan.
    pub fn verify_descriptor(&self, ir: &stencil::Ir) -> Result<(), Error> {
        if self.plan.ir() == ir {
            Ok(())
        } else {
            Err(Error::DescriptorMismatch)
        }
    }
}
