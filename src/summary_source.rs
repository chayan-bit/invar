//! Checked source recognition and immutable preparation for bounded summaries.
//! Verified typed candidate extraction remains separate from reached-source fuel
//! debit, runtime loan admission and publication authority.
#![forbid(unsafe_code)]

mod device;
pub mod host;
mod matcher;
pub use matcher::matcher_footprint;
use matcher::{dot, state_names};

use crate::{
    bounded_summary::{self, Budget, SummaryFamily},
    budget::{BudgetExceeded, ByteBudget},
    check::CheckedProgram,
    front::{
        self, BinaryOp, Expr, ExprKind, Mode, Place, Span, Stmt, StmtKind, Type, named, ungroup,
    },
    general_net, mir, typed_net,
};
use std::{mem::size_of, sync::Arc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Table(u8),
    Matrix(u8),
}
impl Family {
    pub fn id(self) -> &'static str {
        match self {
            Self::Table(_) => "finite-state-table-v1",
            Self::Matrix(_) => "u32-matrix-mod2p32-v1",
        }
    }
    pub fn dimension(self) -> u8 {
        match self {
            Self::Table(k) | Self::Matrix(k) => k,
        }
    }
    pub fn state_width(self) -> usize {
        match self {
            Self::Table(_) => 1,
            Self::Matrix(d) => usize::from(d),
        }
    }
    fn coefficients(self) -> u32 {
        match self {
            Self::Table(k) => u32::from(k),
            Self::Matrix(d) => u32::from(d) * u32::from(d),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observation {
    FinalState,
    InclusivePrefixes,
}
#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Source(front::FrontError),
    CheckedMismatch,
    Net(general_net::Error),
    Typed(typed_net::Error),
    Unrecognized,
    Bounds,
    Shape,
    Seed,
    Alias,
    Allocation,
    RuntimeMismatch,
    Budget(bounded_summary::Error),
    /// The matcher's analysis storage exceeds the [`ByteBudget`] (`BUDGET_EXCEEDED`); the loop
    /// stays serial.
    Exceeded(BudgetExceeded),
    InteractionUnavailable(&'static str),
    /// A complete table or matrix scan form whose K exceeds 16 or d exceeds 4 (the family's own
    /// shape bound, not a size budget).
    FamilyBound,
    /// A coefficient read or operand depends on the evolving state.
    StateDependent,
    Device(invar_device::Error),
}
impl Error {
    /// Stable reason code for a source loop that stays serial.
    pub fn reason(&self) -> Option<&'static str> {
        match self {
            Self::FamilyBound => Some("summary_family_bound"),
            Self::StateDependent => Some("summary_state_dependent"),
            _ => None,
        }
    }
}
#[derive(Debug)]
pub struct Contract {
    net: general_net::Net,
    proof: general_net::Proof,
    function: String,
    coefficients: String,
    seed: String,
    state_names: Vec<String>,
    count: String,
    family: Family,
    observation: Observation,
    output: Option<String>,
    loop_span: Span,
    function_contract: [u8; 32],
    step_ticks: u64,
    token: Arc<()>,
}
fn integer(e: &Expr) -> Option<u32> {
    match &ungroup(e).kind {
        ExprKind::Integer(s) => s.strip_suffix("u32").unwrap_or(s).parse().ok(),
        _ => None,
    }
}
fn binary(e: &Expr, op: BinaryOp) -> Option<(&Expr, &Expr)> {
    match &ungroup(e).kind {
        ExprKind::Binary {
            left,
            op: actual,
            right,
            ..
        } if *actual == op => Some((left, right)),
        _ => None,
    }
}
fn call<'a>(e: &'a Expr, name: &str) -> Option<&'a [Expr]> {
    match &ungroup(e).kind {
        ExprKind::Call { name: n, args } if n == name => Some(args),
        _ => None,
    }
}
fn index<'a>(e: &'a Expr, name: &str) -> Option<&'a Expr> {
    match &ungroup(e).kind {
        ExprKind::Index { base, index } if named(base, name) => Some(index),
        _ => None,
    }
}
fn product(e: &Expr, name: &str, n: u32) -> bool {
    binary(e, BinaryOp::Mul).is_some_and(|(a, b)| named(a, name) && integer(b) == Some(n))
}
fn flat(e: &Expr, name: &str, stride: u32, offset: u32) -> bool {
    binary(e, BinaryOp::Add)
        .is_some_and(|(a, b)| product(a, name, stride) && integer(b) == Some(offset))
}
fn binding(s: &Stmt, mutable: bool) -> Option<(&str, &Expr)> {
    match &s.kind {
        StmtKind::Binding {
            mutable: m,
            name,
            value,
            ..
        } if *m == mutable => Some((name, value)),
        _ => None,
    }
}
fn assign<'a>(s: &'a Stmt, name: &str) -> Option<&'a Expr> {
    match &s.kind {
        StmtKind::Assign {
            place: Place::Name(n, _),
            value,
        } if n == name => Some(value),
        _ => None,
    }
}
fn returned(s: &Stmt) -> Option<&Expr> {
    match &s.kind {
        StmtKind::Return(Some(e)) => Some(e),
        _ => None,
    }
}
fn owner(e: &Expr, n: &str) -> bool {
    named(e, n) || matches!(&ungroup(e).kind,ExprKind::Move(x) if x==n)
}
fn allocation(s: &Stmt, count: &str, width: Option<u32>) -> Option<String> {
    let (name, e) = binding(s, true)?;
    let [len, zero] = call(e, "filled_u32")? else {
        return None;
    };
    if integer(zero) != Some(0)
        || !match width {
            None => named(len, count),
            Some(w) => product(len, count, w),
        }
    {
        return None;
    }
    Some(name.to_owned())
}
fn publish(s: &Stmt, output: &str, iter: &str, width: Option<(u32, u32)>, state: &str) -> bool {
    matches!(&s.kind,StmtKind::Assign {place:Place::Index {name,index,..},value}
        if name==output && named(value,state) && match width {None=>named(index,iter),Some((d,row))=>flat(index,iter,d,row)})
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
fn stmt_ticks(s: &Stmt) -> Option<u64> {
    match &s.kind {
        StmtKind::Binding { value, .. } => expr_ticks(value)?.checked_add(1),
        StmtKind::Assign { place, value } => {
            expr_ticks(value)?.checked_add(1)?.checked_add(match place {
                Place::Name(..) => 0,
                Place::Index { index, .. } => expr_ticks(index)?,
            })
        }
        _ => None,
    }
}
/// Whether `e` reads any of `names`, through any subexpression.
fn mentions(e: &Expr, names: &[String]) -> bool {
    match &e.kind {
        ExprKind::Name(n) | ExprKind::Move(n) => names.contains(n),
        ExprKind::Group(x) | ExprKind::Try(x) | ExprKind::Unary { value: x, .. } => {
            mentions(x, names)
        }
        ExprKind::Field { base, .. } | ExprKind::MoveField { base, .. } => mentions(base, names),
        ExprKind::Binary { left, right, .. }
        | ExprKind::Index {
            base: left,
            index: right,
        } => mentions(left, names) || mentions(right, names),
        ExprKind::Call { args, .. } | ExprKind::Array(args) => {
            args.iter().any(|a| mentions(a, names))
        }
        _ => false,
    }
}
impl Contract {
    pub fn from_checked(source: &CheckedProgram, entry: &str) -> Result<Self, Error> {
        Self::from_checked_within(source, entry, &ByteBudget::default())
    }

    /// [`Self::from_checked`] with the matcher's analysis storage held to `budget`.
    pub fn from_checked_within(
        source: &CheckedProgram,
        entry: &str,
        budget: &ByteBudget,
    ) -> Result<Self, Error> {
        let checked =
            crate::tensor::recheck(source, Error::Bounds, Error::CheckedMismatch, Error::Source)?;
        let net = general_net::Net::lower(&checked, general_net::Limits::default())
            .map_err(Error::Net)?;
        let proof = net.verify().map_err(Error::Net)?;
        let m = net.compiled_mir(&proof).map_err(Error::Net)?;
        if m.layout_version != crate::layout_contract::VERSION
            || m.numeric_policy != crate::numeric_contract::NumericPolicy::STRICT_F32
        {
            return Err(Error::CheckedMismatch);
        }
        let f = checked.program.function(entry).ok_or(Error::Unrecognized)?;
        if f.params.len() != 3
            || f.return_from.is_some()
            || !f.generic_params.is_empty()
            || f.params[0].ty != Type::ArrayU32
            || f.params[0].mode != Some(Mode::Read)
            || f.params[2].ty != Type::U32
            || f.params[2].mode.is_some()
        {
            return Err(Error::Unrecognized);
        }
        let coeff = &f.params[0].name;
        let seed = &f.params[1].name;
        let count = &f.params[2].name;
        let table = f.params[1].ty == Type::U32 && f.params[1].mode.is_none();
        let matrix = f.params[1].ty == Type::ArrayU32 && f.params[1].mode == Some(Mode::Read);
        if !table && !matrix {
            return Err(Error::Unrecognized);
        }
        let stmts = &f.body.statements;
        let position = stmts
            .iter()
            .position(|s| matches!(s.kind, StmtKind::For { .. }))
            .ok_or(Error::Unrecognized)?;
        if stmts.len() != position + 2 {
            return Err(Error::Unrecognized);
        }
        let outer = &stmts[position];
        let StmtKind::For {
            name: iter,
            lower,
            upper,
            body,
        } = &outer.kind
        else {
            return Err(Error::Unrecognized);
        };
        if integer(lower) != Some(0) || !named(upper, count) {
            return Err(Error::Unrecognized);
        }
        let ret = returned(&stmts[position + 1]).ok_or(Error::Unrecognized)?;
        let (family, observation, output, state_names) = if table {
            if !(position == 1 || position == 2) {
                return Err(Error::Unrecognized);
            }
            let (state, initial) = binding(&stmts[0], true).ok_or(Error::Unrecognized)?;
            if !named(initial, seed) {
                return Err(Error::Unrecognized);
            }
            let value = body
                .statements
                .first()
                .and_then(|s| assign(s, state))
                .ok_or(Error::Unrecognized)?;
            let ix = index(value, coeff).ok_or(Error::Unrecognized)?;
            let (mul, s) = binary(ix, BinaryOp::Add).ok_or(Error::Unrecognized)?;
            let (i, k) = binary(mul, BinaryOp::Mul).ok_or(Error::Unrecognized)?;
            if mentions(mul, &[state.to_owned()]) {
                return Err(Error::StateDependent);
            }
            let k = integer(k).filter(|k| *k >= 1).ok_or(Error::Unrecognized)?;
            if !named(i, iter) || !named(s, state) {
                return Err(Error::Unrecognized);
            }
            if position == 1 {
                if body.statements.len() != 1 || !named(ret, state) || f.result != Type::U32 {
                    return Err(Error::Unrecognized);
                }
                (
                    (true, k as usize),
                    Observation::FinalState,
                    None,
                    vec![state.to_owned()],
                )
            } else {
                let output = allocation(&stmts[1], count, None).ok_or(Error::Unrecognized)?;
                if body.statements.len() != 2
                    || !publish(&body.statements[1], &output, iter, None, state)
                    || !owner(ret, &output)
                    || f.result != Type::ArrayU32
                {
                    return Err(Error::Unrecognized);
                }
                (
                    (true, k as usize),
                    Observation::InclusivePrefixes,
                    Some(output),
                    vec![state.to_owned()],
                )
            }
        } else {
            let prefix = matches!(&ungroup(ret).kind, ExprKind::Name(_) | ExprKind::Move(_));
            let d = position
                .checked_sub(usize::from(prefix))
                .filter(|d| *d >= 1)
                .ok_or(Error::Unrecognized)?;
            let states = state_names(stmts, d, seed, budget)?;
            let expected = d * (2 + usize::from(prefix));
            if body.statements.len() != expected || f.result != Type::ArrayU32 {
                return Err(Error::Unrecognized);
            }
            for row in 0..d {
                let (next, e) = binding(&body.statements[row], false).ok_or(Error::Unrecognized)?;
                dot(e, coeff, iter, &states, row, d)?;
                if !assign(&body.statements[d + row], &states[row]).is_some_and(|e| named(e, next))
                {
                    return Err(Error::Unrecognized);
                }
            }
            if prefix {
                let output =
                    allocation(&stmts[d], count, Some(d as u32)).ok_or(Error::Unrecognized)?;
                if !owner(ret, &output) {
                    return Err(Error::Unrecognized);
                }
                for (row, state) in states.iter().enumerate() {
                    if !publish(
                        &body.statements[2 * d + row],
                        &output,
                        iter,
                        Some((d as u32, row as u32)),
                        state,
                    ) {
                        return Err(Error::Unrecognized);
                    }
                }
                (
                    (false, d),
                    Observation::InclusivePrefixes,
                    Some(output),
                    states,
                )
            } else {
                let ExprKind::Array(values) = &ungroup(ret).kind else {
                    return Err(Error::Unrecognized);
                };
                if values.len() != d || !values.iter().zip(&states).all(|(v, s)| named(v, s)) {
                    return Err(Error::Unrecognized);
                }
                ((false, d), Observation::FinalState, None, states)
            }
        };
        let step_ticks = body
            .statements
            .iter()
            .try_fold(1u64, |n, s| n.checked_add(stmt_ticks(s)?))
            .ok_or(Error::Bounds)?;
        let function = m.function(entry).ok_or(Error::Unrecognized)?;
        let function_contract = crate::tensor::function_fingerprint(function);
        if outer.span.start >= outer.span.end
            || outer.span.start < f.span.start
            || outer.span.end > f.span.end
        {
            return Err(Error::Unrecognized);
        }
        // The whole scan form matched; only its shape exceeds the family. These bounds are
        // proof assumptions, not memory limits: `Family` carries the width in a `u8`, the
        // device kernels are unrolled for tables up to 16 states and matrices up to 4x4, and
        // the typed-net law is proved for exactly those shapes. Larger forms stay serial.
        let family = match family {
            (true, k) if k <= 16 => Family::Table(k as u8),
            (false, d) if d <= 4 => Family::Matrix(d as u8),
            _ => return Err(Error::FamilyBound),
        };
        Ok(Self {
            net,
            proof,
            function: entry.to_owned(),
            coefficients: coeff.clone(),
            seed: seed.clone(),
            state_names,
            count: count.clone(),
            family,
            observation,
            output,
            loop_span: outer.span,
            function_contract,
            step_ticks,
            token: Arc::new(()),
        })
    }
    pub fn function(&self) -> &str {
        &self.function
    }
    pub fn coefficients(&self) -> &str {
        &self.coefficients
    }
    pub fn seed(&self) -> &str {
        &self.seed
    }
    pub fn state_names(&self) -> &[String] {
        &self.state_names
    }
    pub fn count(&self) -> &str {
        &self.count
    }
    pub fn family(&self) -> Family {
        self.family
    }
    pub fn observation(&self) -> Observation {
        self.observation
    }
    pub fn output(&self) -> Option<&str> {
        self.output.as_deref()
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
    pub fn checked_net(&self) -> (&general_net::Net, &general_net::Proof) {
        (&self.net, &self.proof)
    }
    /// Require this actual source extraction instance, never an imported digest.
    pub fn verify_prepared(&self, prepared: &Prepared) -> Result<(), Error> {
        if Arc::ptr_eq(&self.token, &prepared.token)
            && self.family == prepared.family
            && self.observation == prepared.observation
            && self.function_contract == prepared.function_contract
            && self.loop_span == prepared.source_span
            && self.net.graph.source_identity == prepared.source_identity
            && self.net.graph.revision == prepared.graph_revision
        {
            Ok(())
        } else {
            Err(Error::RuntimeMismatch)
        }
    }
    /// Bind an extracted candidate to this source instance and exact runtime facts.
    /// This check does not debit source fuel or authorize reached-source publication.
    pub fn verify_plan(
        &self,
        prepared: &Prepared,
        plan: &typed_net::Plan,
        coefficients: &[u32],
        seed: &[u32],
        steps: u32,
        budget: Budget,
    ) -> Result<(), Error> {
        self.verify_prepared(prepared)?;
        prepared.verify_inputs(coefficients, seed, steps, budget)?;
        prepared.verify_plan(plan)
    }
    /// Remaining source ticks after the outer statement and its two bounds.
    pub fn loop_fuel(&self, steps: u32) -> Option<u64> {
        u64::from(steps).checked_mul(self.step_ticks)
    }
    /// Immutable bounded-family preparation, not a typed interaction scan permit.
    pub fn prepare(
        &self,
        coefficients: &[u32],
        seed: &[u32],
        steps: u32,
        budget: Budget,
    ) -> Result<Prepared, Error> {
        let count = steps
            .checked_mul(self.family.coefficients())
            .ok_or(Error::Bounds)?;
        if coefficients.len() != count as usize || coefficients.len() > crate::eval::MAX_ARRAY_LEN {
            return Err(Error::Shape);
        }
        if seed.len() != self.family.state_width()
            || matches!(self.family,Family::Table(k) if seed[0]>=u32::from(k))
        {
            return Err(Error::Seed);
        }
        if matches!(self.family, Family::Matrix(_)) && !coefficients.is_empty() {
            let start = coefficients.as_ptr() as usize;
            let end = start
                .checked_add(
                    coefficients
                        .len()
                        .checked_mul(size_of::<u32>())
                        .ok_or(Error::Bounds)?,
                )
                .ok_or(Error::Bounds)?;
            let seed_start = seed.as_ptr() as usize;
            let seed_end = seed_start
                .checked_add(
                    seed.len()
                        .checked_mul(size_of::<u32>())
                        .ok_or(Error::Bounds)?,
                )
                .ok_or(Error::Bounds)?;
            if start < seed_end && seed_start < end {
                return Err(Error::Alias);
            }
        }
        if self.observation == Observation::InclusivePrefixes
            && steps
                .checked_mul(self.family.state_width() as u32)
                .is_none_or(|n| n as usize > crate::eval::MAX_ARRAY_LEN)
        {
            return Err(Error::Bounds);
        }
        if let Family::Table(k) = self.family
            && coefficients.iter().any(|&n| n >= u32::from(k))
        {
            return Err(Error::Shape);
        }
        let n = steps as usize;
        let (snapshot_bytes, required_bytes, required_work) = cost(self.family, n)?;
        if n > budget.max_steps {
            return Err(Error::Budget(bounded_summary::Error::StepLimit));
        }
        if required_bytes > budget.max_bytes {
            return Err(Error::Budget(bounded_summary::Error::ByteLimit));
        }
        if required_work > budget.max_composition_ops {
            return Err(Error::Budget(bounded_summary::Error::CompositionLimit));
        }
        let mut transitions = Vec::new();
        transitions
            .try_reserve_exact(n)
            .map_err(|_| Error::Allocation)?;
        for row in coefficients.chunks_exact(self.family.coefficients() as usize) {
            let transition = match self.family {
                Family::Table(k) => {
                    let mut t = [0u8; 16];
                    for (to, &from) in t.iter_mut().zip(row) {
                        *to = from as u8;
                    }
                    SummaryFamily::table(&t[..usize::from(k)])
                }
                Family::Matrix(d) => SummaryFamily::matrix(d, row),
            }
            .map_err(Error::Budget)?;
            transitions.push(transition);
        }
        let states = bounded_summary::prefixes(
            &transitions,
            seed,
            Budget {
                max_bytes: budget.max_bytes - snapshot_bytes,
                ..budget
            },
        )
        .map_err(Error::Budget)?;
        let mut immutable = Vec::new();
        immutable
            .try_reserve_exact(coefficients.len())
            .map_err(|_| Error::Allocation)?;
        immutable.extend_from_slice(coefficients);
        let mut fixed_seed = [0; 4];
        fixed_seed[..seed.len()].copy_from_slice(seed);
        Ok(Prepared {
            coefficients: immutable,
            transitions,
            seed: fixed_seed,
            states,
            family: self.family,
            observation: self.observation,
            steps,
            budget,
            source_span: self.loop_span,
            source_fuel: self.loop_fuel(steps).ok_or(Error::Bounds)?,
            function_contract: self.function_contract,
            source_identity: self.net.graph.source_identity,
            graph_revision: self.net.graph.revision,
            required_bytes,
            required_work,
            token: Arc::clone(&self.token),
        })
    }
}

// Prefix scan's frozen logical model plus immutable coefficients/transition
// heap payload owned here. Fixed seed/holder fields, allocator metadata and
// other stack costs are excluded, as in bounded_summary's existing model.
fn cost(family: Family, n: usize) -> Result<(usize, usize, u64), Error> {
    let snapshot = n
        .checked_mul(family.coefficients() as usize)
        .and_then(|n| n.checked_mul(4))
        .and_then(|bytes| {
            n.checked_mul(size_of::<SummaryFamily>())
                .and_then(|t| bytes.checked_add(t))
        })
        .ok_or(Error::Bounds)?;
    if n == 0 {
        return Ok((0, 0, 0));
    }
    let p = n.checked_next_power_of_two().ok_or(Error::Bounds)?;
    let combines = p
        .checked_sub(1)
        .and_then(|n| n.checked_mul(2))
        .and_then(|x| x.checked_add(n))
        .ok_or(Error::Bounds)?;
    let d = u64::from(family.dimension());
    let (compose, action) = match family {
        Family::Table(_) => (3 * d, 2),
        Family::Matrix(_) => (4 * d * d * d + d * d, 4 * d * d + d),
    };
    let work = (combines as u64)
        .checked_mul(compose)
        .and_then(|x| {
            (n as u64)
                .checked_mul(action)
                .and_then(|a| x.checked_add(a))
        })
        .ok_or(Error::Bounds)?;
    let tree = p
        .checked_add(2)
        .and_then(|n| n.checked_mul(size_of::<SummaryFamily>()));
    let output = n.checked_mul(size_of::<Vec<u32>>() + family.state_width() * 4);
    let bytes = tree
        .and_then(|t| output.and_then(|o| t.checked_add(o)))
        .and_then(|b| b.checked_add(snapshot))
        .ok_or(Error::Bounds)?;
    Ok((snapshot, bytes, work))
}

#[derive(Debug)]
pub struct Prepared {
    coefficients: Vec<u32>,
    transitions: Vec<SummaryFamily>,
    seed: [u32; 4],
    states: Vec<Vec<u32>>,
    family: Family,
    observation: Observation,
    steps: u32,
    budget: Budget,
    source_span: Span,
    source_fuel: u64,
    function_contract: [u8; 32],
    source_identity: u64,
    graph_revision: u64,
    required_bytes: usize,
    required_work: u64,
    token: Arc<()>,
}
impl Prepared {
    /// Revalidate the exact immutable runtime facts, independently of a hash.
    pub fn verify_inputs(
        &self,
        coefficients: &[u32],
        seed: &[u32],
        steps: u32,
        budget: Budget,
    ) -> Result<(), Error> {
        if self.coefficients == coefficients
            && self.seed() == seed
            && self.steps == steps
            && self.budget == budget
        {
            Ok(())
        } else {
            Err(Error::RuntimeMismatch)
        }
    }
    pub fn coefficients(&self) -> &[u32] {
        &self.coefficients
    }
    pub fn transitions(&self) -> &[SummaryFamily] {
        &self.transitions
    }
    pub fn seed(&self) -> &[u32] {
        &self.seed[..self.family.state_width()]
    }
    pub fn prefixes(&self) -> &[Vec<u32>] {
        &self.states
    }
    pub fn final_state(&self) -> &[u32] {
        self.states.last().map_or(self.seed(), Vec::as_slice)
    }
    pub fn family(&self) -> Family {
        self.family
    }
    pub fn observation(&self) -> Observation {
        self.observation
    }
    pub fn steps(&self) -> u32 {
        self.steps
    }
    pub fn budget(&self) -> Budget {
        self.budget
    }
    pub fn source_span(&self) -> Span {
        self.source_span
    }
    pub fn source_fuel(&self) -> u64 {
        self.source_fuel
    }
    pub fn function_contract(&self) -> [u8; 32] {
        self.function_contract
    }
    pub fn source_identity(&self) -> u64 {
        self.source_identity
    }
    pub fn graph_revision(&self) -> u64 {
        self.graph_revision
    }
    pub fn required_bytes(&self) -> usize {
        self.required_bytes
    }
    pub fn required_work(&self) -> u64 {
        self.required_work
    }
    /// Encode live ordered ports into a graph bound to this checked source.
    /// The source graph revision is retained in Prepared; this new typed graph
    /// has no rewrite history and starts at revision zero.
    pub fn typed_graph(&self) -> Result<typed_net::Net, Error> {
        use typed_net::{BoundedFamily, Effect, Net, Node, Op, Prefix, Type};
        let family = match self.family {
            Family::Table(k) => BoundedFamily::Table(k),
            Family::Matrix(d) => BoundedFamily::Matrix(d),
        };
        let prefix = match self.observation {
            Observation::FinalState => Prefix::None,
            Observation::InclusivePrefixes => Prefix::Inclusive,
        };
        let (seed_type, seed_op) = match family {
            BoundedFamily::Table(_) => (Type::Nat, Op::InputScalar(0)),
            BoundedFamily::Matrix(_) => (Type::ArrayU32, Op::InputU32(1)),
        };
        let mut graph = Net::empty(3);
        graph.push(Node {
            source_id: 0,
            ty: seed_type,
            effect: Effect::Total,
            inputs: Vec::new(),
            op: seed_op,
        });
        graph.push(Node {
            source_id: 1,
            ty: Type::ArrayU32,
            effect: Effect::Total,
            inputs: Vec::new(),
            op: Op::InputU32(0),
        });
        graph.push(Node {
            source_id: 2,
            ty: Type::Nat,
            effect: Effect::Total,
            inputs: Vec::new(),
            op: Op::InputScalar(1),
        });
        graph.push(Node {
            source_id: 3,
            ty: if matches!(family, BoundedFamily::Table(_)) && prefix == Prefix::None {
                Type::Nat
            } else {
                Type::ArrayU32
            },
            effect: Effect::MayFail,
            inputs: vec![0, 1, 2],
            op: Op::BoundedScan {
                family,
                prefix,
                rule: family.rule(),
            },
        });
        graph.source_identity = self.source_identity;
        // Check before exposing the graph; it remains untrusted after caller mutation.
        graph
            .verify(typed_net::Limits {
                nodes: 4,
                ports: 7,
                depth: 2,
                array_len: crate::eval::MAX_ARRAY_LEN,
                map_ops: 0,
            })
            .map_err(Error::Typed)?;
        Ok(graph)
    }
    /// Extract from actual verified connections, never a caller descriptor.
    pub fn typed_plan(&self) -> Result<typed_net::Plan, Error> {
        let graph = self.typed_graph()?;
        let proof = graph
            .verify(typed_net::Limits {
                nodes: 4,
                ports: 7,
                depth: 2,
                array_len: crate::eval::MAX_ARRAY_LEN,
                map_ops: 0,
            })
            .map_err(Error::Typed)?;
        graph.extract(&proof).map_err(Error::Typed)
    }
    /// Compare every public plan fact against a fresh extraction of private facts.
    pub fn verify_plan(&self, plan: &typed_net::Plan) -> Result<(), Error> {
        if *plan == self.typed_plan()? {
            Ok(())
        } else {
            Err(Error::RuntimeMismatch)
        }
    }
}
