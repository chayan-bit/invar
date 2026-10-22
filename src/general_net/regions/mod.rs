//! P1a: region partition of a verified GNET net (#132 section 3, #133 P1a).
//!
//! Regions are cut on `Net::compiled_mir(proof)` and on the net's own agent and
//! control annotations, never on source text, file names, hashes or benchmark
//! ids. A region is a function body or a natural loop, keyed by
//! `(function, loop head)` and bound to the compiled-MIR contract digest, so any
//! net mutation invalidates it exactly as it invalidates the proof.
//!
//! Each region records its blocks, exits, live-in and live-out locals, array
//! footprint, effect class and ownership facts. [`verify`] re-derives the whole
//! partition from the current net and rejects any difference.

mod cfg;
pub mod family;
pub mod tree;

use std::collections::BTreeSet;

use super::{CallTarget, Net, Proof, coverage::Observation};
use crate::{
    front::Mode,
    mir::{self, BlockId},
};
use tree::{Expr, FunctionTree, LoopKind, Stmt, Unstructured};

pub const VERSION: u32 = 1;

/// `head == None` is the function body region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId {
    pub function: usize,
    pub head: Option<BlockId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Counting {
    /// `for var in lo .. hi`.
    For { var: String },
    /// `while i < E { ...; i = i + 1; }` with `E` and `i` otherwise unchanged.
    CountedWhile { induction: String },
    /// Any other `while`: data-dependent trip count.
    Uncounted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegionKind {
    Function,
    Loop(Counting),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Effects {
    /// A host or foreign call is reachable from the region.
    pub host: bool,
    /// Some agent in the region may trap (checked arithmetic, index, call).
    pub may_trap: bool,
    /// The region allocates arrays.
    pub allocates: bool,
    /// A `return` leaves the region (loops only).
    pub early_exit: bool,
    /// User functions called from the region, by program index.
    pub calls: Vec<usize>,
    /// Arrays lent `mut` to a callee.
    pub mut_loans: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ownership {
    /// Locals declared inside the region (they die at its boundary).
    pub declared: Vec<String>,
    /// Owners moved inside the region.
    pub moved: Vec<String>,
}

/// The region certificate of #132 section 3: every field is re-derived by
/// [`verify`] from the current net.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub id: RegionId,
    pub parent: Option<RegionId>,
    pub kind: RegionKind,
    pub blocks: Vec<BlockId>,
    /// Blocks outside the region entered from inside it.
    pub exits: Vec<BlockId>,
    /// Locals and parameters read in the region and declared outside it.
    pub live_in: Vec<String>,
    /// Locals declared outside the region that it assigns or stores into.
    pub live_out: Vec<String>,
    pub array_reads: Vec<String>,
    pub array_writes: Vec<String>,
    pub effects: Effects,
    pub ownership: Ownership,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionRegions {
    pub index: usize,
    pub name: String,
    /// Statement tree, or why the function stays an opaque serial frame.
    pub tree: Result<FunctionTree, Unstructured>,
    /// No host effect and no `mut` parameter, transitively over callees.
    pub pure: bool,
    /// The body region first, then loops in source pre-order.
    pub regions: Vec<Region>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    pub version: u32,
    /// `mir::Program::contract_digest` of the net-derived MIR.
    pub fingerprint: [u8; 32],
    pub functions: Vec<FunctionRegions>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Net(super::Error),
    /// The CFG and the recovered tree disagree, or annotations do not match.
    Inconsistent(&'static str),
    /// A presented partition differs from the one re-derived from the net.
    Mismatch,
}

impl Partition {
    pub fn function(&self, index: usize) -> Option<&FunctionRegions> {
        self.functions.get(index)
    }

    pub fn region(&self, id: RegionId) -> Option<&Region> {
        self.functions
            .get(id.function)?
            .regions
            .iter()
            .find(|region| region.id == id)
    }
}

/// Partition the verified net into regions.
pub fn partition(net: &Net, proof: &Proof) -> Result<Partition, Error> {
    let program = net.compiled_mir(proof).map_err(Error::Net)?;
    partition_program(net, &program)
}

/// Re-derive the partition from the current net and compare.
pub fn verify(net: &Net, proof: &Proof, presented: &Partition) -> Result<(), Error> {
    if presented.version != VERSION {
        return Err(Error::Mismatch);
    }
    if partition(net, proof)? == *presented {
        Ok(())
    } else {
        Err(Error::Mismatch)
    }
}

pub(crate) fn partition_program(net: &Net, program: &mir::Program) -> Result<Partition, Error> {
    let pure = purity(net, program);
    let mut functions = Vec::with_capacity(program.functions.len());
    for (index, function) in program.functions.iter().enumerate() {
        let tree = tree::decompile(function, index);
        let regions = match &tree {
            Ok(tree) => function_regions(net, function, tree)?,
            Err(_) => Vec::new(),
        };
        functions.push(FunctionRegions {
            index,
            name: function.name.clone(),
            tree,
            pure: pure[index],
            regions,
        });
    }
    Ok(Partition {
        version: VERSION,
        fingerprint: program.contract_digest(),
        functions,
    })
}

/// Transitive purity from the net's call annotations.
fn purity(net: &Net, program: &mir::Program) -> Vec<bool> {
    let count = program.functions.len();
    let mut pure: Vec<bool> = program
        .functions
        .iter()
        .map(|function| {
            function.foreign_oracle.is_none()
                && function
                    .params
                    .iter()
                    .all(|param| param.mode != Some(Mode::Mut))
        })
        .collect();
    let mut callees = vec![BTreeSet::new(); count];
    for agent in &net.agents {
        let caller = agent.token.block.function;
        if caller >= count {
            continue;
        }
        match &agent.call {
            Some(CallTarget::Function(target)) => {
                callees[caller].insert(*target);
            }
            Some(CallTarget::Host { .. }) => pure[caller] = false,
            _ => {}
        }
        if agent.observation == Observation::HostBoundary {
            pure[caller] = false;
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for caller in 0..count {
            if pure[caller]
                && callees[caller]
                    .iter()
                    .any(|&callee| !pure.get(callee).copied().unwrap_or(false))
            {
                pure[caller] = false;
                changed = true;
            }
        }
    }
    pure
}

fn function_regions(
    net: &Net,
    function: &mir::Function,
    tree: &FunctionTree,
) -> Result<Vec<Region>, Error> {
    let cfg = cfg::Cfg::new(function).ok_or(Error::Inconsistent("cfg"))?;
    let heads = cfg.loop_heads();
    let all: Vec<BlockId> = (0..function.blocks.len())
        .filter(|&block| cfg.reachable(block))
        .collect();
    let body_id = RegionId {
        function: tree.index,
        head: None,
    };
    let mut regions = vec![region(
        net,
        &cfg,
        tree,
        body_id,
        None,
        RegionKind::Function,
        all,
        &tree.body,
        &[],
    )];
    let mut loops = Vec::new();
    collect_loops(&tree.body, None, &mut loops);
    if loops.len() != heads.len() {
        return Err(Error::Inconsistent("loop heads"));
    }
    for (inner, parent) in loops {
        let latches = heads
            .iter()
            .find(|(head, _)| *head == inner.head)
            .map(|(_, latches)| latches.as_slice())
            .ok_or(Error::Inconsistent("loop head"))?;
        let blocks = cfg.natural_loop(inner.head, latches);
        let id = RegionId {
            function: tree.index,
            head: Some(inner.head),
        };
        let parent = Some(RegionId {
            function: tree.index,
            head: parent,
        });
        let counting = counting(inner);
        let (header, body) = (header_stmts(inner), inner.body.as_slice());
        regions.push(region(
            net,
            &cfg,
            tree,
            id,
            parent,
            RegionKind::Loop(counting),
            blocks,
            body,
            &header,
        ));
    }
    Ok(regions)
}

fn header_stmts(inner: &tree::Loop) -> Vec<Stmt> {
    match &inner.kind {
        LoopKind::For { lo, hi, .. } => vec![Stmt::Eval(lo.clone()), Stmt::Eval(hi.clone())],
        LoopKind::While { cond } => vec![Stmt::Eval(cond.clone())],
    }
}

fn collect_loops<'a>(
    body: &'a [Stmt],
    parent: Option<BlockId>,
    out: &mut Vec<(&'a tree::Loop, Option<BlockId>)>,
) {
    for stmt in body {
        match stmt {
            Stmt::Loop(inner) => {
                out.push((inner, parent));
                collect_loops(&inner.body, Some(inner.head), out);
            }
            Stmt::If { yes, no, .. } => {
                collect_loops(yes, parent, out);
                collect_loops(no, parent, out);
            }
            _ => {}
        }
    }
}

/// Counted-loop recognition on the tree (the same test the law extractor uses).
pub fn counting(inner: &tree::Loop) -> Counting {
    match &inner.kind {
        LoopKind::For { var, .. } => Counting::For { var: var.clone() },
        LoopKind::While { cond } => {
            counted_while(cond, &inner.body).map_or(Counting::Uncounted, |induction| {
                Counting::CountedWhile {
                    induction: induction.to_owned(),
                }
            })
        }
    }
}

/// `while i < E { ...; i = i + 1; }` where neither `i` nor any name of `E` is
/// assigned elsewhere in the body.
pub fn counted_while<'a>(cond: &'a Expr, body: &[Stmt]) -> Option<&'a str> {
    let Expr::Binary {
        op: crate::front::BinaryOp::Lt,
        left,
        right,
        ..
    } = cond
    else {
        return None;
    };
    let Expr::Local(induction) = left.as_ref() else {
        return None;
    };
    let (Stmt::Assign { name, value }, rest) = body.split_last()? else {
        return None;
    };
    let increments = matches!(value, Expr::Binary { op: crate::front::BinaryOp::Add, left, right, .. }
        if matches!(left.as_ref(), Expr::Local(n) if n == induction) && right.literal() == Some(1));
    if name != induction || !increments {
        return None;
    }
    let mut bound = Vec::new();
    right.names(&mut bound);
    let mut clean = true;
    tree::walk(rest, &mut Vec::new(), 0, &mut |stmt, _, _| match stmt {
        Stmt::Assign { name, .. } | Stmt::Let { name, .. }
            if name == induction || bound.contains(name) =>
        {
            clean = false;
        }
        Stmt::Store { array, .. } if bound.contains(array) => clean = false,
        _ => {}
    });
    clean.then_some(induction.as_str())
}

#[allow(clippy::too_many_arguments)]
fn region(
    net: &Net,
    cfg: &cfg::Cfg,
    tree: &FunctionTree,
    id: RegionId,
    parent: Option<RegionId>,
    kind: RegionKind,
    blocks: Vec<BlockId>,
    body: &[Stmt],
    header: &[Stmt],
) -> Region {
    let mut declared = BTreeSet::new();
    let mut assigned = BTreeSet::new();
    let mut read = BTreeSet::new();
    let mut array_reads = BTreeSet::new();
    let mut array_writes = BTreeSet::new();
    let mut moved = BTreeSet::new();
    let mut mut_loans = BTreeSet::new();
    let mut early_exit = false;
    if let RegionKind::Loop(Counting::For { var }) = &kind {
        declared.insert(var.clone());
    }
    let mut visit = |stmt: &Stmt| {
        for expr in tree::stmt_exprs(stmt) {
            expr_facts(
                expr,
                &mut read,
                &mut array_reads,
                &mut moved,
                &mut mut_loans,
            );
        }
        match stmt {
            Stmt::Let { name, .. } => {
                declared.insert(name.clone());
            }
            Stmt::Assign { name, .. } => {
                assigned.insert(name.clone());
            }
            Stmt::Store { array, .. } => {
                array_writes.insert(array.clone());
            }
            Stmt::Loop(inner) => {
                if let LoopKind::For { var, .. } = &inner.kind {
                    declared.insert(var.clone());
                }
            }
            Stmt::Return { .. } => early_exit = true,
            _ => {}
        }
    };
    for stmt in header {
        visit(stmt);
    }
    tree::walk(body, &mut Vec::new(), 0, &mut |stmt, _, _| visit(stmt));
    for name in &mut_loans {
        array_writes.insert(name.clone());
    }
    let is_function = matches!(kind, RegionKind::Function);
    if is_function {
        for param in &tree.params {
            declared.insert(param.name.clone());
        }
        early_exit = false;
    }
    let outside = |name: &String| !declared.contains(name);
    let live_in = read
        .iter()
        .chain(array_reads.iter())
        .filter(|name| outside(name) || is_function && tree.params.iter().any(|p| &p.name == *name))
        .cloned()
        .collect::<BTreeSet<_>>();
    let live_out = assigned
        .iter()
        .chain(array_writes.iter())
        .filter(|name| outside(name))
        .cloned()
        .collect::<BTreeSet<_>>();
    let inside = |block: BlockId| blocks.binary_search(&block).is_ok();
    let mut exits = BTreeSet::new();
    for &block in &blocks {
        for &target in &cfg.successors[block] {
            if !inside(target) {
                exits.insert(target);
            }
        }
    }
    let mut effects = Effects {
        early_exit,
        mut_loans: mut_loans.into_iter().collect(),
        ..Effects::default()
    };
    let mut calls = BTreeSet::new();
    for agent in &net.agents {
        let site = agent.token.block;
        if site.function != tree.index || !inside(site.block) {
            continue;
        }
        match agent.observation {
            Observation::HostBoundary => effects.host = true,
            Observation::MayTrap => effects.may_trap = true,
            Observation::Allocate => effects.allocates = true,
            _ => {}
        }
        match &agent.call {
            Some(CallTarget::Function(target)) => {
                calls.insert(*target);
                effects.may_trap = true;
            }
            Some(CallTarget::Host { .. }) => effects.host = true,
            Some(CallTarget::Intrinsic(name)) => {
                effects.may_trap = true;
                if name.starts_with("filled_") {
                    effects.allocates = true;
                }
            }
            None => {}
        }
    }
    effects.calls = calls.into_iter().collect();
    Region {
        id,
        parent,
        kind,
        blocks: blocks.clone(),
        exits: exits.into_iter().collect(),
        live_in: live_in.into_iter().collect(),
        live_out: live_out.into_iter().collect(),
        array_reads: array_reads.into_iter().collect(),
        array_writes: array_writes.into_iter().collect(),
        effects,
        ownership: Ownership {
            declared: declared.into_iter().collect(),
            moved: moved.into_iter().collect(),
        },
    }
}

fn expr_facts(
    expr: &Expr,
    read: &mut BTreeSet<String>,
    arrays: &mut BTreeSet<String>,
    moved: &mut BTreeSet<String>,
    mut_loans: &mut BTreeSet<String>,
) {
    let children: Vec<&Expr> = match expr {
        Expr::I32(_) | Expr::U32(_) | Expr::F32(_) | Expr::Bool(_) => Vec::new(),
        Expr::Local(name) => {
            read.insert(name.clone());
            Vec::new()
        }
        Expr::Owner { name, loan } => {
            arrays.insert(name.clone());
            if *loan == Some(Mode::Mut) {
                mut_loans.insert(name.clone());
            }
            Vec::new()
        }
        Expr::Move(name) => {
            moved.insert(name.clone());
            arrays.insert(name.clone());
            Vec::new()
        }
        Expr::Len(name) => {
            arrays.insert(name.clone());
            Vec::new()
        }
        Expr::ArrayLit { items, .. } | Expr::Call { args: items, .. } => items.iter().collect(),
        Expr::Index { array, index, .. } => {
            arrays.insert(array.clone());
            vec![index]
        }
        Expr::Unary { value, .. } => vec![value],
        Expr::Binary { left, right, .. } | Expr::Logic { left, right, .. } => vec![left, right],
    };
    for child in children {
        expr_facts(child, read, arrays, moved, mut_loans);
    }
}
