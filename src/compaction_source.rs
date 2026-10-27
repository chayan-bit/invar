//! Ordinary-source stable u32 compaction with a bounded local CPU candidate.
//! Flags and exclusive positions are bound to checked source and a verified typed
//! scan. This module supplies no reached-source loan, fuel or publication permit.
#![forbid(unsafe_code)]

use crate::{
    check,
    front::{
        self, BinaryOp, Expr, ExprKind, Mode, Place, Span, Stmt, StmtKind, Type, named, ungroup,
    },
    general_net, laws, mir, recurrence,
    typed_net::{self, Coefficient, Effect, Net, Node, Op, Plan, Prefix},
};
use sha2::{Digest, Sha256};

pub const RULE: &str = "source-stable-u32-compaction-v1";
pub mod allocation;

/// Only total unsigned comparisons are admitted; no user callback is assumed total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predicate {
    Less,
    LessEqual,
    Equal,
    NotEqual,
    GreaterEqual,
    Greater,
}
impl Predicate {
    fn from_op(op: BinaryOp) -> Option<Self> {
        match op {
            BinaryOp::Lt => Some(Self::Less),
            BinaryOp::Le => Some(Self::LessEqual),
            BinaryOp::Eq => Some(Self::Equal),
            BinaryOp::Ne => Some(Self::NotEqual),
            BinaryOp::Ge => Some(Self::GreaterEqual),
            BinaryOp::Gt => Some(Self::Greater),
            _ => None,
        }
    }
    fn apply(self, value: u32, cutoff: u32) -> bool {
        match self {
            Self::Less => value < cutoff,
            Self::LessEqual => value <= cutoff,
            Self::Equal => value == cutoff,
            Self::NotEqual => value != cutoff,
            Self::GreaterEqual => value >= cutoff,
            Self::Greater => value > cutoff,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Source(front::FrontError),
    Net(general_net::Error),
    Typed(typed_net::Error),
    Scan(recurrence::Error),
    Unrecognized,
    SourceMismatch,
    MirMismatch,
    ExtractionMismatch,
    Bounds,
    Budget,
    Allocation,
    DescriptorMismatch,
    Positions,
    FuelStructure,
    AllocationDescriptorMismatch,
}

/// Logical payload and work bounds. Bytes include flags, scan coefficients,
/// exclusive positions and output conservatively together; they are not RSS.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub max_elements: usize,
    pub max_bytes: u64,
    pub max_work: u64,
}

#[derive(Debug)]
pub struct Contract {
    net: general_net::Net,
    proof: general_net::Proof,
    source_hash: [u8; 32],
    mir_hash: [u8; 32],
    entry: String,
    names: [String; 4],
    spans: [Span; 2],
    selections: [Span; 2],
    states: [String; 2],
    predicate: Predicate,
    graph: Net,
    plan: Plan,
    fuel: [u64; 4], // per element, retained branch, intervening, first retained branch
    first_loop_ticks: u64,
    allocation: allocation::Site,
}
fn integer(e: &Expr) -> Option<u32> {
    match &ungroup(e).kind {
        ExprKind::Integer(n) => n.strip_suffix("u32").unwrap_or(n).parse().ok(),
        _ => None,
    }
}
fn index(e: &Expr, array: &str, iter: &str) -> bool {
    matches!(&ungroup(e).kind, ExprKind::Index {base,index} if named(base,array) && named(index,iter))
}
fn initialized(s: &Stmt) -> Option<(&str, &Expr)> {
    match &s.kind {
        StmtKind::Binding {
            mutable: true,
            name,
            value,
            ..
        } => Some((name, value)),
        _ => None,
    }
}
fn increment(s: &Stmt, name: &str) -> bool {
    matches!(&s.kind, StmtKind::Assign {place:Place::Name(n,_),value}
        if n==name && matches!(&ungroup(value).kind,ExprKind::Binary {left,op:BinaryOp::Add,right,..}
            if named(left,name) && integer(right)==Some(1)))
}
fn counted<'a>(s: &'a Stmt, count: &str) -> Option<(&'a str, &'a [Stmt])> {
    match &s.kind {
        StmtKind::For {
            name,
            lower,
            upper,
            body,
        } if integer(lower) == Some(0) && named(upper, count) => Some((name, &body.statements)),
        _ => None,
    }
}
fn selected<'a>(
    s: &'a Stmt,
    input: &str,
    iter: &str,
    cutoff: &str,
) -> Option<(Predicate, &'a [Stmt])> {
    let StmtKind::If {
        condition,
        then_block,
        else_arm: None,
    } = &s.kind
    else {
        return None;
    };
    let ExprKind::Binary {
        left, op, right, ..
    } = &ungroup(condition).kind
    else {
        return None;
    };
    if !index(left, input, iter) || !named(right, cutoff) {
        return None;
    }
    Some((Predicate::from_op(*op)?, then_block.statements.as_slice()))
}
fn expr_ticks(e: &Expr) -> Option<u64> {
    match &e.kind {
        ExprKind::Group(e) => expr_ticks(e),
        ExprKind::Name(_) | ExprKind::Integer(_) => Some(1),
        ExprKind::Binary { left, right, .. }
        | ExprKind::Index {
            base: left,
            index: right,
        } => expr_ticks(left)?
            .checked_add(expr_ticks(right)?)?
            .checked_add(1),
        ExprKind::Call { args, .. } => args
            .iter()
            .try_fold(1u64, |n, e| n.checked_add(expr_ticks(e)?)),
        _ => None,
    }
}
fn statement_ticks(s: &Stmt) -> Option<u64> {
    let n = match &s.kind {
        StmtKind::Binding { value, .. } => expr_ticks(value)?,
        StmtKind::Assign { value, place } => expr_ticks(value)?.checked_add(match place {
            Place::Name(..) => 0,
            Place::Index { index, .. } => expr_ticks(index)?,
        })?,
        StmtKind::If { condition, .. } => expr_ticks(condition)?,
        StmtKind::For { lower, upper, .. } => expr_ticks(lower)?.checked_add(expr_ticks(upper)?)?,
        _ => return None,
    };
    n.checked_add(1)
}
fn branch_ticks(select: &Stmt) -> Option<u64> {
    let StmtKind::If {
        then_block,
        else_arm: None,
        ..
    } = &select.kind
    else {
        return None;
    };
    then_block
        .statements
        .iter()
        .try_fold(0u64, |n, s| n.checked_add(statement_ticks(s)?))
}
fn scan_graph() -> Result<(Net, Plan), Error> {
    let mut graph = Net::empty(2);
    for (source_id, ty, op, inputs) in [
        (0, typed_net::Type::Nat, Op::InputScalar(0), vec![]),
        (1, typed_net::Type::ArrayU32, Op::InputU32(0), vec![]),
        (
            2,
            typed_net::Type::ArrayU32,
            Op::AffineScan {
                prefix: Prefix::Exclusive,
                a: Coefficient::Constant(1),
                b: Coefficient::Input,
                rule: recurrence::RULE_REVISION,
            },
            vec![0, 1],
        ),
    ] {
        graph.nodes.push(Some(Node {
            source_id,
            ty,
            effect: Effect::Total,
            inputs,
            op,
        }));
    }
    graph.seal_source();
    let proof = graph
        .verify(crate::loops::GRAPH_LIMITS)
        .map_err(Error::Typed)?;
    let plan = graph.extract(&proof).map_err(Error::Typed)?;
    Ok((graph, plan))
}
impl Contract {
    /// Recognize two matching ordinary counted predicate passes over immutable u32
    /// input, with exact-size fresh allocation between counting and stable copying.
    pub fn from_source(source: &[u8], entry: &str) -> Result<Self, Error> {
        let checked = check::check(front::parse(source).map_err(Error::Source)?, false)
            .map_err(Error::Source)?;
        let f = checked.program.function(entry).ok_or(Error::Unrecognized)?;
        if f.params.len() != 3
            || f.result != Type::ArrayU32
            || f.return_from.is_some()
            || !f.generic_params.is_empty()
            || f.params[0].ty != Type::ArrayU32
            || f.params[0].mode != Some(Mode::Read)
            || f.params[1..]
                .iter()
                .any(|p| p.ty != Type::U32 || p.mode.is_some())
        {
            return Err(Error::Unrecognized);
        }
        let input = &f.params[0].name;
        let count = &f.params[1].name;
        let cutoff = &f.params[2].name;
        let [seed, flag_loop, allocate, position_seed, copy_loop, ret] =
            f.body.statements.as_slice()
        else {
            return Err(Error::Unrecognized);
        };
        let (size, zero) = initialized(seed).ok_or(Error::Unrecognized)?;
        if integer(zero) != Some(0) {
            return Err(Error::Unrecognized);
        }
        let (first, body) = counted(flag_loop, count).ok_or(Error::Unrecognized)?;
        let [select] = body else {
            return Err(Error::Unrecognized);
        };
        let (predicate, body) =
            selected(select, input, first, cutoff).ok_or(Error::Unrecognized)?;
        let [add] = body else {
            return Err(Error::Unrecognized);
        };
        if !increment(add, size) {
            return Err(Error::Unrecognized);
        }
        let (output, value) = initialized(allocate).ok_or(Error::Unrecognized)?;
        if !matches!(&ungroup(value).kind,ExprKind::Call {name,args}
            if name=="filled_u32" && matches!(args.as_slice(),[length,fill]
                if named(length,size) && integer(fill)==Some(0)))
        {
            return Err(Error::Unrecognized);
        }
        let (position, zero) = initialized(position_seed).ok_or(Error::Unrecognized)?;
        if integer(zero) != Some(0) {
            return Err(Error::Unrecognized);
        }
        let (second, body) = counted(copy_loop, count).ok_or(Error::Unrecognized)?;
        let [select] = body else {
            return Err(Error::Unrecognized);
        };
        let (second_predicate, body) =
            selected(select, input, second, cutoff).ok_or(Error::Unrecognized)?;
        let [store, add] = body else {
            return Err(Error::Unrecognized);
        };
        if predicate != second_predicate
            || !increment(add, position)
            || !matches!(&store.kind,StmtKind::Assign {place:Place::Index{name,index:dest,..},value}
                if name==output && named(dest,position) && index(value,input,second))
        {
            return Err(Error::Unrecognized);
        }
        if !matches!(&ret.kind,StmtKind::Return(Some(e))
            if named(e,output) || matches!(&ungroup(e).kind,ExprKind::Move(n) if n==output))
        {
            return Err(Error::Unrecognized);
        }
        let (_, first_body) = counted(flag_loop, count).ok_or(Error::FuelStructure)?;
        let (_, second_body) = counted(copy_loop, count).ok_or(Error::FuelStructure)?;
        let [first_select] = first_body else {
            return Err(Error::FuelStructure);
        };
        let [second_select] = second_body else {
            return Err(Error::FuelStructure);
        };
        let first_loop_ticks = statement_ticks(first_select)
            .and_then(|n| n.checked_add(1))
            .ok_or(Error::FuelStructure)?;
        let per_element = statement_ticks(second_select)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_add(first_loop_ticks))
            .ok_or(Error::FuelStructure)?;
        let first_branch = branch_ticks(first_select).ok_or(Error::FuelStructure)?;
        let per_retained = branch_ticks(second_select)
            .and_then(|n| n.checked_add(first_branch))
            .ok_or(Error::FuelStructure)?;
        let allocation_ticks = statement_ticks(allocate).ok_or(Error::FuelStructure)?;
        let intervening = statement_ticks(position_seed)
            .and_then(|n| n.checked_add(allocation_ticks))
            .and_then(|n| n.checked_add(statement_ticks(copy_loop)?))
            .ok_or(Error::FuelStructure)?;
        let allocation = allocation::Site::from_statement(allocate, 2, allocation_ticks)
            .ok_or(Error::Unrecognized)?;
        let names = [
            input.clone(),
            count.clone(),
            cutoff.clone(),
            output.to_owned(),
        ];
        let spans = [flag_loop.span, copy_loop.span];
        let net = general_net::Net::lower(&checked, general_net::Limits::default())
            .map_err(Error::Net)?;
        let proof = net.verify().map_err(Error::Net)?;
        let m = net.compiled_mir(&proof).map_err(Error::Net)?;
        let (graph, plan) = scan_graph()?;
        Ok(Self {
            net,
            proof,
            source_hash: Sha256::digest(source).into(),
            mir_hash: m.contract_digest(),
            entry: entry.to_owned(),
            names,
            spans,
            selections: [first_select.span, second_select.span],
            states: [size.to_owned(), position.to_owned()],
            predicate,
            graph,
            plan,
            fuel: [per_element, per_retained, intervening, first_branch],
            first_loop_ticks,
            allocation,
        })
    }
    pub fn function(&self) -> &str {
        &self.entry
    }
    pub fn names(&self) -> &[String; 4] {
        &self.names
    }
    pub fn loop_spans(&self) -> [Span; 2] {
        self.spans
    }
    pub fn selection_spans(&self) -> [Span; 2] {
        self.selections
    }
    /// Count and copy scalar roles, in pass order.
    pub fn states(&self) -> &[String; 2] {
        &self.states
    }
    /// Remaining ticks for each pass; MIR owns the intervening allocation/seed.
    pub fn pass_fuel(&self, pass: usize, count: u32, retained: u32) -> Option<u64> {
        if retained > count {
            return None;
        }
        let (element, selected) = match pass {
            0 => (self.first_loop_ticks, self.fuel[3]),
            1 => (
                self.fuel[0].checked_sub(self.first_loop_ticks)?,
                self.fuel[1].checked_sub(self.fuel[3])?,
            ),
            _ => return None,
        };
        u64::from(count)
            .checked_mul(element)?
            .checked_add(u64::from(retained).checked_mul(selected)?)
    }
    pub fn predicate(&self) -> Predicate {
        self.predicate
    }
    pub fn source_hash(&self) -> [u8; 32] {
        self.source_hash
    }
    pub fn mir_contract(&self) -> [u8; 32] {
        self.mir_hash
    }
    pub fn graph(&self) -> &Net {
        &self.graph
    }
    pub fn plan(&self) -> &Plan {
        &self.plan
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
    pub fn verify_mir(&self, m: &mir::Program) -> Result<(), Error> {
        mir::verify(m).map_err(|_| Error::MirMismatch)?;
        if m.contract_digest() != self.mir_hash {
            return Err(Error::MirMismatch);
        }
        Ok(())
    }
    pub fn verify_extraction(&self, graph: &Net, plan: &Plan) -> Result<(), Error> {
        let proof = graph
            .verify(crate::loops::GRAPH_LIMITS)
            .map_err(Error::Typed)?;
        let extracted = graph.extract(&proof).map_err(Error::Typed)?;
        if extracted != self.plan || plan != &self.plan {
            return Err(Error::ExtractionMismatch);
        }
        Ok(())
    }
    /// Count and check logical budgets before any flag/position allocation. Rust
    /// borrows keep inputs immutable; this does not establish Invar host loans.
    /// Source allocation lies inside this region. Preparation cannot authenticate
    /// the interpreter's live ledger or bypass its array/live allocation guards.
    pub fn prepare<'a>(
        &'a self,
        input: &'a [u32],
        count: u32,
        cutoff: u32,
        budget: Budget,
    ) -> Result<Prepared<'a>, Error> {
        let count = usize::try_from(count).map_err(|_| Error::Bounds)?;
        if input.len() > laws::MAX_ELEMENTS
            || count > laws::MAX_ELEMENTS
            || count > budget.max_elements
        {
            return Err(Error::Budget);
        }
        let values = input.get(..count).ok_or(Error::Bounds)?;
        let n = u64::try_from(count).map_err(|_| Error::Budget)?;
        let preparation_bytes = n.checked_mul(16).ok_or(Error::Budget)?;
        let base_work = n.checked_mul(6).ok_or(Error::Budget)?;
        // Refuse a plainly insufficient guard before even scanning the predicate.
        if preparation_bytes > budget.max_bytes || base_work > budget.max_work {
            return Err(Error::Budget);
        }
        let retained = values
            .iter()
            .filter(|&&v| self.predicate.apply(v, cutoff))
            .count();
        let bytes = u64::try_from(retained)
            .ok()
            .and_then(|k| k.checked_mul(4))
            .and_then(|out| preparation_bytes.checked_add(out))
            .ok_or(Error::Budget)?;
        // Conservative payload: flags(4N)+coefficients(8N)+positions(4N)+output(4K).
        // Six full passes bound preflight predicates, flags, coefficients, prefix,
        // position validation and stable copying, plus K output initialization units.
        // These costs do not charge source fuel.
        let work = u64::try_from(retained)
            .ok()
            .and_then(|k| base_work.checked_add(k))
            .ok_or(Error::Budget)?;
        if bytes > budget.max_bytes || work > budget.max_work {
            return Err(Error::Budget);
        }
        self.verify_extraction(&self.graph, &self.plan)?;
        let mut flags = Vec::new();
        flags
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        flags.extend(
            values
                .iter()
                .map(|&v| u32::from(self.predicate.apply(v, cutoff))),
        );
        let coefficients = self
            .plan
            .affine_coefficients(&[&flags], count)
            .map_err(Error::Scan)?;
        let mut positions = Vec::new();
        positions
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        let mut total = 0u32;
        for (coefficient, &flag) in coefficients.iter().zip(&flags) {
            positions.push(total);
            let exact = total.checked_add(flag).ok_or(Error::Positions)?;
            // Use the extracted affine action; bounded binary flags prove it agrees
            // with source checked addition, so modular overflow is never approximated.
            total = coefficient.apply(total);
            if total != exact {
                return Err(Error::Positions);
            }
        }
        if usize::try_from(total).ok() != Some(retained) {
            return Err(Error::Positions);
        }
        Ok(Prepared {
            contract: self,
            input,
            count,
            cutoff,
            flags,
            positions,
            retained: total,
            bytes,
            work,
        })
    }
}

/// Diagnostic candidate descriptor. Public fields confer no execution/publication
/// authority; all values and ordered positions must match the private preparation.
#[derive(Clone, Copy, Debug)]
pub struct Descriptor<'a> {
    pub source_hash: [u8; 32],
    pub mir_hash: [u8; 32],
    pub predicate: Predicate,
    pub input: &'a [u32],
    pub count: u32,
    pub cutoff: u32,
    pub flags: &'a [u32],
    pub positions: &'a [u32],
    pub retained: u32,
}
#[derive(Debug)]
pub struct Prepared<'a> {
    contract: &'a Contract,
    input: &'a [u32],
    count: usize,
    cutoff: u32,
    flags: Vec<u32>,
    positions: Vec<u32>,
    retained: u32,
    bytes: u64,
    work: u64,
}
impl Prepared<'_> {
    pub fn descriptor(&self) -> Descriptor<'_> {
        Descriptor {
            source_hash: self.contract.source_hash,
            mir_hash: self.contract.mir_hash,
            predicate: self.contract.predicate,
            input: self.input,
            count: self.count as u32,
            cutoff: self.cutoff,
            flags: &self.flags,
            positions: &self.positions,
            retained: self.retained,
        }
    }
    pub fn flags(&self) -> &[u32] {
        &self.flags
    }
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }
    pub fn retained(&self) -> u32 {
        self.retained
    }
    pub fn required_bytes(&self) -> u64 {
        self.bytes
    }
    pub fn required_work(&self) -> u64 {
        self.work
    }
    pub fn rule(&self) -> &'static str {
        RULE
    }
    /// Original ticks from just after first-loop dispatch/bounds to second-loop
    /// completion. Includes the intervening fresh allocation and position seed.
    pub fn remaining_region_ticks(&self) -> Option<u64> {
        let [element, retained, intervening, _] = self.contract.fuel;
        u64::try_from(self.count)
            .ok()?
            .checked_mul(element)?
            .checked_add(u64::from(self.retained).checked_mul(retained)?)?
            .checked_add(intervening)
    }
    /// Source-bound allocation obligation; this is not reached runtime admission.
    /// A future adapter must check the real global ledger at this exact intrinsic.
    pub fn allocation_description(&self) -> Option<allocation::Description<'_>> {
        let first_ticks = u64::try_from(self.count)
            .ok()?
            .checked_mul(self.contract.first_loop_ticks)?
            .checked_add(u64::from(self.retained).checked_mul(self.contract.fuel[3])?)?;
        Some(allocation::Description::new(
            &self.contract.allocation,
            &self.contract.entry,
            self.contract.source_hash,
            self.contract.mir_hash,
            self.contract.spans,
            self.retained,
            first_ticks.checked_add(self.contract.allocation.ticks)?,
        ))
    }
    pub fn verify_allocation_descriptor(&self, d: allocation::Descriptor<'_>) -> Result<(), Error> {
        if self
            .allocation_description()
            .ok_or(Error::FuelStructure)?
            .descriptor()
            != d
        {
            return Err(Error::AllocationDescriptorMismatch);
        }
        Ok(())
    }
    pub fn verify_descriptor(&self, d: Descriptor<'_>) -> Result<(), Error> {
        let expected = self.descriptor();
        if d.source_hash != expected.source_hash
            || d.mir_hash != expected.mir_hash
            || d.predicate != expected.predicate
            || d.count != expected.count
            || d.cutoff != expected.cutoff
            || d.retained != expected.retained
            || d.input != expected.input
            || d.flags != expected.flags
            || d.positions != expected.positions
        {
            return Err(Error::DescriptorMismatch);
        }
        Ok(())
    }
    /// Complete privately into a fresh exact-size output, preserving input index
    /// order. Neither supplied descriptors nor results can publish source owners.
    pub fn execute(&self) -> Result<Vec<u32>, Error> {
        let mut total = 0u32;
        if self.flags.len() != self.count || self.positions.len() != self.count {
            return Err(Error::Positions);
        }
        for (&flag, &position) in self.flags.iter().zip(&self.positions) {
            if flag > 1 || position != total {
                return Err(Error::Positions);
            }
            total = total.checked_add(flag).ok_or(Error::Positions)?;
        }
        if total != self.retained {
            return Err(Error::Positions);
        }
        let length = usize::try_from(total).map_err(|_| Error::Bounds)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .map_err(|_| Error::Allocation)?;
        output.resize(length, 0);
        let input = self.input.get(..self.count).ok_or(Error::Bounds)?;
        for ((&value, &flag), &position) in input.iter().zip(&self.flags).zip(&self.positions) {
            if flag == 1 {
                let destination = usize::try_from(position)
                    .ok()
                    .and_then(|i| output.get_mut(i))
                    .ok_or(Error::Positions)?;
                *destination = value;
            }
        }
        Ok(output)
    }
}
