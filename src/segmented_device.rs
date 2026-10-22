//! Source-bound lowering of M4-04 regions to the verified segmented device IR:
//! segmented maps (recognized here), segmented reductions
//! (`segmented_source`) and stable compaction (`compaction_source`).
//!
//! The IR identity binds the exact source bytes and verified MIR digests. A
//! lowered IR grants no reached-source loan, fuel, completion or publication
//! authority; inputs still pass `segmented::admit_inputs`, and any refusal is
//! an admission miss that keeps the ordinary CPU path.
#![forbid(unsafe_code)]

use crate::{
    check, compaction_source,
    front::{self, Expr, ExprKind, Mode, Place, Span, StmtKind, Type, named, ungroup},
    general_net, segmented_source,
    typed_net::{self, DagOp, Effect, Net, Node, Op, Plan, ZipOp},
};
use invar_device::{
    Provenance,
    segmented::{self as seg, Family, Ir, Law, MapOp, Predicate},
};
use sha2::{Digest, Sha256};

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Source(front::FrontError),
    Net(general_net::Error),
    Typed(typed_net::Error),
    Unrecognized,
    /// Input lengths or offsets fail device admission; use the CPU path.
    Admission,
}

fn identity(source_hash: [u8; 32], mir_hash: [u8; 32], span: Span) -> (Ir, Provenance) {
    let word = |bytes: [u8; 32]| u64::from_le_bytes(bytes[..8].try_into().unwrap_or([0; 8]));
    let origin = Provenance {
        // Never zero: zero is the verifier's "no source" value.
        source_identity: word(source_hash) | 1,
        graph_revision: word(mir_hash),
    };
    let ir = Ir {
        version: seg::VERSION,
        span: invar_device::Span {
            start: span.start,
            end: span.end,
        },
        source_identity: origin.source_identity,
        graph_revision: origin.graph_revision,
        function_contract: mir_hash,
        family: Family::Reduce(Law::WrappingSum),
        elements: 0,
        segments: 0,
    };
    (ir, origin)
}

fn sizes(input: &[u32], offsets: &[u32]) -> Result<(u32, u32), Error> {
    let elements = u32::try_from(input.len()).map_err(|_| Error::Admission)?;
    let segments = offsets
        .len()
        .checked_sub(1)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or(Error::Admission)?;
    Ok((elements, segments))
}

/// Segmented reduction IR for one prepared source-bound reduction.
pub fn reduce_ir(
    contract: &segmented_source::Contract,
    input: &[u32],
    offsets: &[u32],
) -> Result<(Ir, Provenance), Error> {
    let (mut ir, origin) = identity(
        contract.source_hash(),
        contract.mir_contract(),
        contract.loop_span(),
    );
    contract
        .verify_extraction(contract.graph(), contract.plan())
        .map_err(|_| Error::Admission)?;
    let Plan::Reduce {
        law,
        seed: 0,
        input: 0,
        ..
    } = contract.plan()
    else {
        return Err(Error::Admission);
    };
    ir.family = Family::Reduce(match law {
        crate::laws::Law::WrappingSum => Law::WrappingSum,
        crate::laws::Law::WrappingProduct => Law::WrappingProduct,
    });
    (ir.elements, ir.segments) = sizes(input, offsets)?;
    Ok((ir, origin))
}

/// Stable compaction IR for the counted prefix `input[..count]`.
pub fn compact_ir(
    contract: &compaction_source::Contract,
    count: u32,
    cutoff: u32,
) -> Result<(Ir, Provenance), Error> {
    let [first, second] = contract.loop_spans();
    contract
        .verify_extraction(contract.graph(), contract.plan())
        .map_err(|_| Error::Admission)?;
    if !matches!(
        contract.plan(),
        Plan::AffineScan {
            prefix: typed_net::Prefix::Exclusive,
            seed: 0,
            a: typed_net::Source::Constant(1),
            b: typed_net::Source::Slot(0),
            ..
        }
    ) {
        return Err(Error::Admission);
    }
    let (mut ir, origin) = identity(
        contract.source_hash(),
        contract.mir_contract(),
        Span {
            start: first.start,
            end: second.end,
        },
    );
    use compaction_source::Predicate as P;
    let predicate = match contract.predicate() {
        P::Less => Predicate::Less,
        P::LessEqual => Predicate::LessEqual,
        P::Equal => Predicate::Equal,
        P::NotEqual => Predicate::NotEqual,
        P::GreaterEqual => Predicate::GreaterEqual,
        P::Greater => Predicate::Greater,
    };
    ir.family = Family::Compact { predicate, cutoff };
    ir.elements = count;
    Ok((ir, origin))
}

/// A recognized ordinary-source segmented map:
/// `out[i] = ops(xs[i], s)` for `i in offsets[s]..offsets[s + 1]`, `s in 0..count`,
/// into `out = filled_u32(len(xs), 0u32)`, with `wrapping_add`/`wrapping_mul`
/// by u32 constants or by the segment index.
#[derive(Debug)]
pub struct MapContract {
    source_hash: [u8; 32],
    mir_hash: [u8; 32],
    span: Span,
    inner_span: Span,
    names: [String; 4],
    loop_ticks: [u64; 2],
    ops: Vec<MapOp>,
    graph: Net,
    plan: Plan,
}

/// One node per callback operation, independent of element/segment counts.
/// Slot 1 is the segment index broadcast over the current segment.
fn map_graph(ops: &[MapOp]) -> Result<(Net, Plan), Error> {
    let mut graph = Net::empty(0);
    let push = |graph: &mut Net, op, inputs| {
        let id = graph.nodes.len();
        graph.nodes.push(Some(Node {
            // At most two nodes per admitted callback operation plus its input.
            source_id: id as u32,
            ty: typed_net::Type::ArrayU32,
            effect: Effect::Total,
            op,
            inputs,
        }));
        id
    };
    let mut root = push(&mut graph, Op::InputU32(0), vec![]);
    for op in ops {
        let (op, inputs) = match op {
            MapOp::WrappingAdd(k) => (Op::Map(typed_net::MapOp::WrappingAdd(*k)), vec![root]),
            MapOp::WrappingMul(k) => (Op::Map(typed_net::MapOp::WrappingMul(*k)), vec![root]),
            MapOp::AddSegment => {
                let segment = push(&mut graph, Op::InputU32(1), vec![]);
                (
                    Op::ZipMap {
                        op: ZipOp::WrappingAdd,
                        len: 0,
                    },
                    vec![root, segment],
                )
            }
        };
        root = push(&mut graph, op, inputs);
    }
    graph.entry = root;
    graph.seal_source();
    let proof = graph
        .verify(crate::loops::GRAPH_LIMITS)
        .map_err(Error::Typed)?;
    let plan = graph.extract_dag(&proof, &[root]).map_err(Error::Typed)?;
    Ok((graph, plan))
}

fn extracted_ops(plan: &Plan) -> Result<Vec<MapOp>, Error> {
    let Plan::LiveDag { ops, outputs, .. } = plan else {
        return Err(Error::Admission);
    };
    let [at] = outputs.as_slice() else {
        return Err(Error::Admission);
    };
    let mut at = *at;
    let mut result = Vec::new();
    loop {
        match ops.get(at) {
            Some(DagOp::Input(0)) => {
                result.reverse();
                return Ok(result);
            }
            Some(DagOp::Map { input, op }) if *input < at => {
                result.push(match op {
                    typed_net::MapOp::WrappingAdd(k) => MapOp::WrappingAdd(*k),
                    typed_net::MapOp::WrappingMul(k) => MapOp::WrappingMul(*k),
                });
                at = *input;
            }
            Some(DagOp::Zip {
                left,
                right,
                op: ZipOp::WrappingAdd,
            }) if *left < at && matches!(ops.get(*right), Some(DagOp::Input(1))) => {
                result.push(MapOp::AddSegment);
                at = *left;
            }
            _ => return Err(Error::Admission),
        }
    }
}

fn integer(e: &Expr) -> Option<u32> {
    match &ungroup(e).kind {
        ExprKind::Integer(n) => n.strip_suffix("u32").unwrap_or(n).parse().ok(),
        _ => None,
    }
}
fn index<'a>(e: &'a Expr, name: &str) -> Option<&'a Expr> {
    match &ungroup(e).kind {
        ExprKind::Index { base, index } if named(base, name) => Some(index),
        _ => None,
    }
}

/// Innermost-first operations of `expr`, which must bottom out at `xs[i]`.
fn map_ops(mut expr: &Expr, input: &str, element: &str, segment: &str) -> Option<Vec<MapOp>> {
    let mut ops = Vec::new();
    loop {
        if index(expr, input).is_some_and(|e| named(e, element)) {
            ops.reverse();
            return (!ops.is_empty() && ops.len() <= seg::MAX_MAP_OPS).then_some(ops);
        }
        let ExprKind::Call { name, args } = &ungroup(expr).kind else {
            return None;
        };
        let [inner, operand] = args.as_slice() else {
            return None;
        };
        ops.push(match (name.as_str(), integer(operand)) {
            ("wrapping_add", Some(k)) => MapOp::WrappingAdd(k),
            ("wrapping_mul", Some(k)) => MapOp::WrappingMul(k),
            ("wrapping_add", None) if named(operand, segment) => MapOp::AddSegment,
            _ => return None,
        });
        expr = inner;
    }
}

impl MapContract {
    pub fn from_source(source: &[u8], entry: &str) -> Result<Self, Error> {
        let checked = check::check(front::parse(source).map_err(Error::Source)?, false)
            .map_err(Error::Source)?;
        let f = checked
            .program
            .functions
            .iter()
            .find(|f| f.name == entry)
            .ok_or(Error::Unrecognized)?;
        if f.params.len() != 3
            || f.result != Type::ArrayU32
            || f.return_from.is_some()
            || !f.generic_params.is_empty()
            || f.params[..2]
                .iter()
                .any(|p| p.ty != Type::ArrayU32 || p.mode != Some(Mode::Read))
            || f.params[2].ty != Type::U32
            || f.params[2].mode.is_some()
        {
            return Err(Error::Unrecognized);
        }
        let (input, offsets, count) = (&f.params[0].name, &f.params[1].name, &f.params[2].name);
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
        let allocation = matches!(&ungroup(value).kind, ExprKind::Call { name, args }
            if name == "filled_u32" && matches!(args.as_slice(), [length, fill]
                if integer(fill) == Some(0)
                    && matches!(&ungroup(length).kind, ExprKind::Call { name, args }
                        if name == "len" && matches!(args.as_slice(), [a] if named(a, input)))));
        let returns = matches!(&ret.kind, StmtKind::Return(Some(e))
            if named(e, output) || matches!(&ungroup(e).kind, ExprKind::Move(n) if n == output));
        let StmtKind::For {
            name: segment,
            lower,
            upper,
            body,
        } = &outer.kind
        else {
            return Err(Error::Unrecognized);
        };
        if !allocation || !returns || integer(lower) != Some(0) || !named(upper, count) {
            return Err(Error::Unrecognized);
        }
        let [inner] = body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::For {
            name: element,
            lower,
            upper,
            body,
        } = &inner.kind
        else {
            return Err(Error::Unrecognized);
        };
        let successor = |e: &Expr| {
            matches!(&ungroup(e).kind, ExprKind::Binary { left, op: front::BinaryOp::Add, right, .. }
                if named(left, segment) && integer(right) == Some(1))
        };
        if !index(lower, offsets).is_some_and(|e| named(e, segment))
            || !index(upper, offsets).is_some_and(successor)
        {
            return Err(Error::Unrecognized);
        }
        let [store] = body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::Assign {
            place: Place::Index {
                name, index: at, ..
            },
            value,
        } = &store.kind
        else {
            return Err(Error::Unrecognized);
        };
        if name != output || !named(at, element) {
            return Err(Error::Unrecognized);
        }
        let ops = map_ops(value, input, element, segment).ok_or(Error::Unrecognized)?;
        let net = general_net::Net::lower(&checked, general_net::Limits::default())
            .map_err(Error::Net)?;
        let proof = net.verify().map_err(Error::Net)?;
        let mir = net.compiled_mir(&proof).map_err(Error::Net)?;
        let (graph, plan) = map_graph(&ops)?;
        Ok(Self {
            source_hash: Sha256::digest(source).into(),
            mir_hash: mir.contract_digest(),
            span: outer.span,
            inner_span: inner.span,
            names: [
                input.clone(),
                offsets.clone(),
                count.clone(),
                output.clone(),
            ],
            loop_ticks: [
                segmented_source::stmt_ticks(inner)
                    .and_then(|n| n.checked_add(1))
                    .ok_or(Error::Admission)?,
                segmented_source::stmt_ticks(store)
                    .and_then(|n| n.checked_add(1))
                    .ok_or(Error::Admission)?,
            ],
            ops,
            graph,
            plan,
        })
    }

    pub fn names(&self) -> &[String; 4] {
        &self.names
    }
    pub fn loop_span(&self) -> Span {
        self.span
    }
    pub fn inner_span(&self) -> Span {
        self.inner_span
    }
    pub fn mir_contract(&self) -> [u8; 32] {
        self.mir_hash
    }
    /// Remaining source ticks at the reached outer head, after its bounds.
    pub fn loop_fuel(&self, segments: usize, elements: usize) -> Option<u64> {
        u64::try_from(segments)
            .ok()?
            .checked_mul(self.loop_ticks[0])?
            .checked_add(
                u64::try_from(elements)
                    .ok()?
                    .checked_mul(self.loop_ticks[1])?,
            )
    }

    pub fn ops(&self) -> &[MapOp] {
        &self.ops
    }

    pub fn graph(&self) -> &Net {
        &self.graph
    }
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    pub fn verify_extraction(&self, graph: &Net, plan: &Plan) -> Result<(), Error> {
        let proof = graph
            .verify(crate::loops::GRAPH_LIMITS)
            .map_err(Error::Typed)?;
        let extracted = graph
            .extract_dag(&proof, &[graph.entry])
            .map_err(Error::Typed)?;
        if extracted != self.plan || plan != &self.plan {
            return Err(Error::Admission);
        }
        Ok(())
    }

    /// Segmented map IR for these runtime input and offsets lengths.
    pub fn ir(&self, input: &[u32], offsets: &[u32]) -> Result<(Ir, Provenance), Error> {
        let (mut ir, origin) = identity(self.source_hash, self.mir_hash, self.span);
        self.verify_extraction(&self.graph, &self.plan)?;
        ir.family = Family::Map(extracted_ops(&self.plan)?);
        (ir.elements, ir.segments) = sizes(input, offsets)?;
        Ok((ir, origin))
    }
}
