//! Source-derived rank-1 strict f32 regions over the existing typed net proof.
//! Unknown forms and failed live guards keep the original source loop.
use crate::{
    check::CheckedProgram,
    eval,
    front::{self, BinaryOp, Expr, ExprKind, Place, StmtKind, Type, named, ungroup},
    general_net, mir, mir_eval,
    numeric_contract::{Comparison, F32Bits},
};
use invar_device::{
    f32_ops::{self, Compare, Kind, Node, Program, Scalar},
    v2,
};

fn tool(code: &'static str, message: &str) -> eval::EvalError {
    eval::EvalError::Tool {
        code,
        message: message.into(),
    }
}

#[derive(Debug)]
pub enum Error {
    Source(front::FrontError),
    Net(general_net::Error),
    Mir(mir::Error),
    Unrecognized,
    Device(v2::Error),
    Input,
    IntegerOverflow,
    Worker(crate::schedule::Error),
}
#[derive(Clone, Debug)]
pub struct Region {
    pub(crate) function: String,
    pub(crate) span: front::Span,
    pub(crate) inputs: Vec<(String, Type)>,
    pub(crate) output: String,
    pub(crate) accumulator: Option<String>,
    pub(crate) fingerprint: [u8; 32],
    pub(crate) ticks_per_iteration: u64,
    origin: invar_device::Provenance,
    program: Program,
}
fn same_bound(a: &Expr, b: &Expr) -> bool {
    match (&ungroup(a).kind, &ungroup(b).kind) {
        (ExprKind::Name(a), ExprKind::Name(b)) => a == b,
        (ExprKind::Call { name: a, args: aa }, ExprKind::Call { name: b, args: ba }) => {
            a == "len" && a == b && aa.len() == 1 && ba.len() == 1 && same_bound(&aa[0], &ba[0])
        }
        _ => false,
    }
}
fn zero(e: &Expr) -> bool {
    matches!(&ungroup(e).kind, ExprKind::Integer(n) if n == "0u32" || n == "0")
}
fn is_array(ty: &Type) -> bool {
    matches!(ty, Type::ArrayF32 | Type::ArrayI32 | Type::ArrayU32)
}
fn scalar(ty: &Type) -> Option<Scalar> {
    match ty {
        Type::F32 | Type::ArrayF32 => Some(Scalar::F32),
        Type::U32 | Type::ArrayU32 => Some(Scalar::U32),
        Type::I32 | Type::ArrayI32 => Some(Scalar::I32),
        _ => None,
    }
}
fn constant(e: &Expr) -> Option<(u32, Scalar)> {
    match &ungroup(e).kind {
        ExprKind::Float(raw) => Some((F32Bits::from_decimal(raw).ok()?.to_bits(), Scalar::F32)),
        ExprKind::Integer(raw) => {
            if let Some(n) = raw.strip_suffix("u32") {
                Some((n.parse().ok()?, Scalar::U32))
            } else {
                Some((
                    raw.strip_suffix("i32").unwrap_or(raw).parse::<i32>().ok()? as u32,
                    Scalar::I32,
                ))
            }
        }
        ExprKind::Unary {
            op: front::UnaryOp::Neg,
            value,
            ..
        } => {
            let (w, ty) = constant(value)?;
            Some((
                if ty == Scalar::F32 {
                    w ^ 0x80000000
                } else {
                    (w as i32).checked_neg()? as u32
                },
                ty,
            ))
        }
        _ => None,
    }
}
fn ticks(e: &Expr) -> Option<u64> {
    let children = match &e.kind {
        ExprKind::Group(inner) => return ticks(inner),
        ExprKind::Name(_) | ExprKind::Integer(_) | ExprKind::Float(_) => 0,
        ExprKind::Unary { value, .. } => ticks(value)?,
        ExprKind::Binary { left, right, .. }
        | ExprKind::Index {
            base: left,
            index: right,
        } => ticks(left)?.checked_add(ticks(right)?)?,
        ExprKind::Call { args, .. } => args
            .iter()
            .try_fold(0u64, |n, e| n.checked_add(ticks(e)?))?,
        _ => return None,
    };
    children.checked_add(1)
}
fn expression(
    e: &Expr,
    index: &str,
    inputs: &[(String, Type)],
    nodes: &mut Vec<Node>,
) -> Option<usize> {
    if nodes.len() >= f32_ops::MAX_NODES {
        return None;
    }
    let node = if let Some((w, ty)) = constant(e) {
        Node::Constant(w, ty)
    } else {
        match &ungroup(e).kind {
            ExprKind::Name(n) => {
                let slot = inputs
                    .iter()
                    .position(|(name, ty)| name == n && !is_array(ty))?;
                Node::Input(slot)
            }
            ExprKind::Index { base, index: at } if named(at, index) => {
                let slot = inputs
                    .iter()
                    .position(|(name, ty)| named(base, name) && is_array(ty))?;
                Node::Input(slot)
            }
            ExprKind::Unary {
                op: front::UnaryOp::Neg,
                value,
                ..
            } => Node::Neg(expression(value, index, inputs, nodes)?),
            ExprKind::Binary {
                left, op, right, ..
            } => {
                let a = expression(left, index, inputs, nodes)?;
                let b = expression(right, index, inputs, nodes)?;
                match op {
                    BinaryOp::Add => Node::Add(a, b),
                    BinaryOp::Sub => Node::Sub(a, b),
                    BinaryOp::Mul => Node::Mul(a, b),
                    BinaryOp::Eq => Node::Compare(Compare::Eq, a, b),
                    BinaryOp::Ne => Node::Compare(Compare::Ne, a, b),
                    BinaryOp::Lt => Node::Compare(Compare::Lt, a, b),
                    BinaryOp::Le => Node::Compare(Compare::Le, a, b),
                    BinaryOp::Gt => Node::Compare(Compare::Gt, a, b),
                    BinaryOp::Ge => Node::Compare(Compare::Ge, a, b),
                    _ => return None,
                }
            }
            ExprKind::Call { name, args } => {
                let args = args
                    .iter()
                    .map(|a| expression(a, index, inputs, nodes))
                    .collect::<Option<Vec<_>>>()?;
                match (crate::math::f32_trig_base(name), args.as_slice()) {
                    ("fma", [a, b, c]) => Node::Fma(*a, *b, *c),
                    ("f32_from_u32", [a]) => Node::FromU32(*a),
                    ("f32_from_i32", [a]) => Node::FromI32(*a),
                    ("trunc_i32", [a]) => Node::TruncI32(*a),
                    ("sin", [a]) => Node::Sin(*a),
                    ("cos", [a]) => Node::Cos(*a),
                    ("sqrt", [a]) => Node::Sqrt(*a),
                    _ => return None,
                }
            }
            _ => return None,
        }
    };
    let i = nodes.len();
    nodes.push(node);
    Some(i)
}
impl Region {
    pub fn from_checked(checked: &CheckedProgram, entry: &str) -> Result<Self, Error> {
        crate::ownership::check(checked).map_err(Error::Source)?;
        let net =
            general_net::Net::lower(checked, general_net::Limits::default()).map_err(Error::Net)?;
        let proof = net.verify().map_err(Error::Net)?;
        let mir = net.compiled_mir(&proof).map_err(Error::Net)?;
        let f = checked
            .program
            .functions
            .iter()
            .find(|f| f.name == entry)
            .ok_or(Error::Unrecognized)?;
        let lowered = mir
            .functions
            .iter()
            .find(|f| f.name == entry)
            .ok_or(Error::Unrecognized)?;
        if let [
            front::Stmt {
                kind: StmtKind::Return(Some(value)),
                ..
            },
        ] = f.body.statements.as_slice()
        {
            let inputs = f
                .params
                .iter()
                .map(|p| (p.name.clone(), p.ty.clone()))
                .collect::<Vec<_>>();
            if !inputs.is_empty()
                && inputs.len() <= f32_ops::MAX_INPUTS
                && inputs
                    .iter()
                    .all(|(_, t)| !is_array(t) && scalar(t).is_some())
            {
                let mut nodes = Vec::new();
                expression(value, "", &inputs, &mut nodes).ok_or(Error::Unrecognized)?;
                let program = Program {
                    length: 1,
                    inputs: inputs
                        .iter()
                        .map(|(_, t)| scalar(t).ok_or(Error::Unrecognized))
                        .collect::<Result<_, _>>()?,
                    nodes,
                    kind: Kind::Map,
                };
                program.output_type().map_err(Error::Device)?;
                return Ok(Self {
                    function: entry.into(),
                    span: f.span,
                    inputs,
                    output: String::new(),
                    accumulator: None,
                    fingerprint: crate::tensor::function_fingerprint(lowered),
                    ticks_per_iteration: 0,
                    origin: invar_device::Provenance {
                        source_identity: net.graph.source_identity,
                        graph_revision: net.graph.revision,
                    },
                    program,
                });
            }
        }
        let inputs = f
            .params
            .iter()
            .filter(|p| is_array(&p.ty) || p.ty == Type::F32 || p.ty == Type::I32)
            .map(|p| (p.name.clone(), p.ty.clone()))
            .collect::<Vec<_>>();
        if inputs.is_empty()
            || inputs.len() > f32_ops::MAX_INPUTS
            || inputs.iter().any(|(name, ty)| {
                scalar(ty).is_none()
                    || (is_array(ty)
                        && f.params
                            .iter()
                            .find(|p| p.name == *name)
                            .is_none_or(|p| p.mode != Some(front::Mode::Read)))
            })
        {
            return Err(Error::Unrecognized);
        }
        let loop_stmt = f
            .body
            .statements
            .iter()
            .find(|s| matches!(s.kind, StmtKind::For { .. }))
            .ok_or(Error::Unrecognized)?;
        let StmtKind::For {
            name: index,
            lower,
            upper,
            body,
        } = &loop_stmt.kind
        else {
            return Err(Error::Unrecognized);
        };
        if !zero(lower) {
            return Err(Error::Unrecognized);
        }
        let bound_ok = match &ungroup(upper).kind {
            ExprKind::Name(n) => f.params.iter().any(|p| p.name == *n && p.ty == Type::U32),
            ExprKind::Call { name, args } if name == "len" && args.len() == 1 => inputs
                .iter()
                .any(|(name, ty)| is_array(ty) && named(&args[0], name)),
            _ => false,
        };
        if !bound_ok {
            return Err(Error::Unrecognized);
        }
        let mut accumulator = None;
        let mut seed = 0u32;
        let mut output = String::new();
        // Only allocation and a constant accumulator may precede the loop.
        for s in f
            .body
            .statements
            .iter()
            .take_while(|s| s.span != loop_stmt.span)
        {
            let StmtKind::Binding {
                name,
                mutable: true,
                value,
                ..
            } = &s.kind
            else {
                return Err(Error::Unrecognized);
            };
            if let Some((w, Scalar::F32)) = constant(value) {
                if accumulator.replace(name.clone()).is_some() {
                    return Err(Error::Unrecognized);
                }
                seed = w;
            } else if let ExprKind::Call {
                name: constructor,
                args,
            } = &ungroup(value).kind
            {
                if !matches!(
                    constructor.as_str(),
                    "filled_f32" | "filled_u32" | "filled_i32"
                ) || args.len() != 2
                    || !same_bound(&args[0], upper)
                    || constant(&args[1]).is_none()
                    || !output.is_empty()
                {
                    return Err(Error::Unrecognized);
                }
                output = name.clone();
            } else {
                return Err(Error::Unrecognized);
            }
        }
        let mut nodes = Vec::new();
        let (kind, expression_value, store_ticks) = match (
            accumulator.as_deref(),
            body.statements.as_slice(),
        ) {
            (None, [store]) => {
                if let StmtKind::If {
                    condition,
                    then_block,
                    else_arm: Some(front::ElseArm::Block(other)),
                } = &store.kind
                {
                    let mask = |b: &front::Block, bit: u32| {
                        let [s] = b.statements.as_slice() else {
                            return false;
                        };
                        matches!(&s.kind,StmtKind::Assign{place:Place::Index{name,index:at,..},value} if name==&output && named(at,index) && constant(value)==Some((bit,Scalar::U32)))
                    };
                    if !mask(then_block, 1) || !mask(other, 0) {
                        return Err(Error::Unrecognized);
                    }
                    (Kind::Map, condition, 4)
                } else {
                    let StmtKind::Assign {
                        place:
                            Place::Index {
                                name, index: at, ..
                            },
                        value,
                    } = &store.kind
                    else {
                        return Err(Error::Unrecognized);
                    };
                    if name != &output || !named(at, index) {
                        return Err(Error::Unrecognized);
                    }
                    (Kind::Map, value, ticks(at).ok_or(Error::Unrecognized)? + 1)
                }
            }
            (Some(acc), [step]) | (Some(acc), [step, _]) => {
                let StmtKind::Assign {
                    place: Place::Name(name, _),
                    value,
                } = &step.kind
                else {
                    return Err(Error::Unrecognized);
                };
                let ExprKind::Binary {
                    left,
                    op: BinaryOp::Add,
                    right,
                    ..
                } = &ungroup(value).kind
                else {
                    return Err(Error::Unrecognized);
                };
                if name != acc || !named(left, acc) {
                    return Err(Error::Unrecognized);
                }
                if body.statements.len() == 2 {
                    let StmtKind::Assign {
                        place:
                            Place::Index {
                                name, index: at, ..
                            },
                        value,
                    } = &body.statements[1].kind
                    else {
                        return Err(Error::Unrecognized);
                    };
                    if name != &output || !named(at, index) || !named(value, acc) {
                        return Err(Error::Unrecognized);
                    }
                    (
                        Kind::Scan { initial: seed },
                        &**right,
                        3 + ticks(at).ok_or(Error::Unrecognized)?
                            + ticks(value).ok_or(Error::Unrecognized)?
                            + 1,
                    )
                } else {
                    if !output.is_empty() {
                        return Err(Error::Unrecognized);
                    }
                    output = acc.to_owned();
                    (Kind::Reduce { initial: seed }, &**right, 3)
                }
            }
            _ => return Err(Error::Unrecognized),
        };
        let expected_statements = 1
            + usize::from(accumulator.is_some())
            + usize::from(!matches!(kind, Kind::Reduce { .. }))
            + 1;
        if f.body.statements.len() != expected_statements {
            return Err(Error::Unrecognized);
        }
        let Some(StmtKind::Return(Some(ret))) = f.body.statements.last().map(|s| &s.kind) else {
            return Err(Error::Unrecognized);
        };
        if !named(ret, &output) && !matches!(&ret.kind, ExprKind::Move(n) if n == &output) {
            return Err(Error::Unrecognized);
        }
        expression(expression_value, index, &inputs, &mut nodes).ok_or(Error::Unrecognized)?;
        let program = Program {
            length: 0,
            inputs: inputs
                .iter()
                .map(|(_, ty)| scalar(ty).ok_or(Error::Unrecognized))
                .collect::<Result<_, _>>()?,
            nodes,
            kind,
        };
        program.output_type().map_err(Error::Device)?;
        Ok(Self {
            function: entry.into(),
            span: loop_stmt.span,
            inputs,
            output,
            accumulator,
            fingerprint: crate::tensor::function_fingerprint(lowered),
            ticks_per_iteration: ticks(expression_value).ok_or(Error::Unrecognized)?
                + store_ticks
                + 1,
            origin: invar_device::Provenance {
                source_identity: net.graph.source_identity,
                graph_revision: net.graph.revision,
            },
            program,
        })
    }
    pub fn bind(
        &self,
        length: u32,
        caps: invar_device::Capabilities,
    ) -> Result<v2::Verified, Error> {
        let mut p = self.program.clone();
        p.length = length;
        let ir = f32_ops::descriptor(
            p,
            invar_device::Span {
                start: self.span.start,
                end: self.span.end,
            },
            self.origin,
        )
        .map_err(Error::Device)?;
        v2::verify(ir, self.origin, caps).map_err(Error::Device)
    }
}

mod host;
pub(crate) use host::Completion;
#[cfg(feature = "cuda")]
pub use host::run_cuda_source;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use host::run_metal_source;
pub use host::{Run, cpu, run_cpu_source, run_source};
