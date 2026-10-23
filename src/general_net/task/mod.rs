//! P4b: residual fork-join task region T (law L20, region design section 5 "Fork-join T").
//!
//! A program whose certified normal form keeps a fork-join after every law
//! (`NormalForm::is_flat() == false`) runs as one task region:
//!
//! * every reachable function that is recursive, holds an L20 fork site, or
//!   calls such a function becomes a *machine* function: its statement tree is
//!   lowered to a resumable state machine whose frame lives in an arena node
//!   or on a private stack. The other functions are emitted as plain checked
//!   functions;
//! * nodes advance on one of two schedules ([`Schedule`]). A node runs until
//!   it returns, fails, or reaches a call: an L20 fork site spawns one child
//!   node per sibling (nested forks: children fork again), any other machine
//!   call spawns one child. The arena is compacted, so no lane is padded and
//!   no leaf is evaluated twice. Expansion stops at a fixed, calibrated
//!   *width* ([`cost`]); past it, nodes run serially on a private stack, in
//!   source order;
//! * *staged* (MSL 3.1 fallback): a single-group scheduler kernel expands
//!   breadth first until a round holds `width` ready nodes, a multi-group run
//!   kernel executes those as serial tasks, and later scheduler passes join
//!   bottom up;
//! * *cascade* (CUDA, MSL 3.2): one persistent multi-group kernel whose lanes
//!   claim arena nodes in allocation order (a device-wide ready list); the
//!   last child to finish resumes its parent in the same lane;
//! * a join resumes its node in source order, so the first failing sibling
//!   (or the first failing sibling argument) is the node's result, exactly as
//!   the serial program meets it. Calls in a continuation (child-dependent
//!   recursion) spawn new children, which expand again.
//!
//! The observation is the value or the first source-order trap with its span
//! (observation contract v2 without source steps); the 64-frame call depth
//! limit is reproduced as the oracle's `CALL_DEPTH_LIMIT` tool failure. Source
//! fuel is not modelled: a lane step budget only guarantees termination and is
//! reported as a tool failure, never as a value.

pub mod cost;
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub mod host;
mod kernel;
mod lower;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::{
    Net, Proof,
    normal_form::{NormalForm, certificate},
    regions::{
        self,
        tree::{Expr, FunctionTree, Stmt},
    },
};
use crate::{
    eval::{EvalError, TrapCategory, Value},
    front::{Span, Type},
};

pub use cost::{Plan, Profile, Schedule};
pub use cost::{plan, route};
pub use kernel::{
    BUFFERS, CASCADE_KERNEL, RUN_KERNEL, SCHED_KERNEL, STATE_WORDS, Source, buffer_words, emit,
};

/// Version of the task-region kernel ABI (buffers, node and state layout).
pub const ABI: &str = "gnet-task-v1";
/// The CPU net oracle's call depth limit (live frames).
pub const MAX_CALL_DEPTH: u32 = 64;

/// Why a program does not run as a task region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The normal form was not derived from this net, or does not verify.
    Certificates,
    /// No residual fork-join: the program belongs to a flat lowering (D5 tier 1).
    Flat,
    UnknownEntry,
    /// The entry neither forks nor reaches a recursive function.
    EntryNotTask,
    /// A reachable function stays an opaque serial frame (named reason).
    Opaque {
        function: String,
        reason: &'static str,
    },
    /// Arrays, records, owners or f32 values (the task ABI is 32-bit scalar).
    NonScalar {
        function: String,
    },
    /// A reachable function has a host effect or a `mut` parameter.
    Impure {
        function: String,
    },
    /// Functions that call each other recursively (only self recursion).
    MutualRecursion {
        function: String,
    },
    /// A call to a machine function inside an expression.
    NestedTaskCall {
        function: String,
        span: Span,
    },
    /// A call to a machine function inside a loop.
    TaskCallInLoop {
        function: String,
    },
    /// An intrinsic outside the scalar integer set.
    Intrinsic {
        function: String,
        name: String,
    },
    /// Frame, sibling or parameter counts above the ABI bounds.
    Limit,
    /// The device's calibrated cost model measured this region above the D5
    /// residual tier (resident or host-to-host): it runs on the native path.
    CostModel,
}

impl Refusal {
    /// Stable reason code for explain output.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Certificates => "task_certificates",
            Self::Flat => "task_flat_form",
            Self::UnknownEntry => "task_unknown_entry",
            Self::EntryNotTask => "task_entry_not_task",
            Self::Opaque { .. } => "task_opaque_function",
            Self::NonScalar { .. } => "task_non_scalar",
            Self::Impure { .. } => "task_impure",
            Self::MutualRecursion { .. } => "task_mutual_recursion",
            Self::NestedTaskCall { .. } => "task_nested_call",
            Self::TaskCallInLoop { .. } => "task_call_in_loop",
            Self::Intrinsic { .. } => "task_intrinsic",
            Self::Limit => "task_limit",
            Self::CostModel => "task_cost_model",
        }
    }
}

/// Bounds of the task ABI.
pub const MAX_FRAME_WORDS: usize = 64;
pub const MAX_SIBLINGS: usize = 8;
pub const MAX_PARAMS: usize = 8;

/// One reachable function and its role in the task region.
#[derive(Clone, Debug)]
pub(crate) struct Function {
    pub tree: FunctionTree,
    /// Lowered to a resumable state machine (recursive, forking, or calling one).
    pub machine: bool,
    /// Frame slot of every local (parameters first, in declaration order).
    pub slots: BTreeMap<String, usize>,
    /// L20 fork sites outside loops: first sibling span and sibling count.
    pub forks: Vec<(Span, usize)>,
}

/// An admitted task-region program.
#[derive(Clone, Debug)]
pub struct Program {
    pub(crate) functions: BTreeMap<usize, Function>,
    pub(crate) names: HashMap<String, usize>,
    pub(crate) entry: usize,
    pub(crate) frame_words: usize,
    pub(crate) siblings: usize,
    pub(crate) params: usize,
    fingerprint: [u8; 32],
}

impl Program {
    pub fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }
    pub fn entry_params(&self) -> Vec<Type> {
        self.functions[&self.entry]
            .tree
            .params
            .iter()
            .map(|p| p.ty.clone())
            .collect()
    }
    pub fn entry_result(&self) -> Type {
        self.functions[&self.entry].tree.result.clone()
    }
    /// Largest sibling count of a fork site (the expansion radix).
    pub fn max_siblings(&self) -> usize {
        self.siblings
    }
    /// Whether a continuation after a fork or call can call again
    /// (child-dependent recursion, which needs further scheduler rounds).
    pub fn continuation_calls(&self) -> bool {
        self.functions
            .values()
            .filter(|f| f.machine)
            .any(|f| lower::calls_after_call(&f.tree.body, &self.machine_names()))
    }
    pub fn frame_words(&self) -> usize {
        self.frame_words
    }
    pub(crate) fn machine_names(&self) -> BTreeSet<String> {
        self.functions
            .values()
            .filter(|f| f.machine)
            .map(|f| f.tree.name.clone())
            .collect()
    }
}

fn scalar(ty: &Type) -> bool {
    matches!(ty, Type::I32 | Type::U32 | Type::Bool | Type::Unit)
}

fn callees(body: &[Stmt]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    regions::tree::walk(body, &mut Vec::new(), 0, &mut |stmt, _, _| {
        for expr in regions::tree::stmt_exprs(stmt) {
            let mut calls = Vec::new();
            expr.calls(&mut calls);
            for call in calls {
                if let Expr::Call { name, .. } = call {
                    out.insert(name.clone());
                }
            }
        }
    });
    out
}

/// Admit `entry` of a verified net as a task region. The normal form must be
/// the net's own (its certificates are re-checked) and keep a fork-join.
pub fn admit(net: &Net, proof: &Proof, form: &NormalForm, entry: &str) -> Result<Program, Refusal> {
    let partition = regions::partition(net, proof).map_err(|_| Refusal::Certificates)?;
    if partition.fingerprint != form.fingerprint
        || partition.functions.len() != form.functions.len()
        || certificate::check_all(net, proof, form).is_err()
    {
        return Err(Refusal::Certificates);
    }
    if form.is_flat() {
        return Err(Refusal::Flat);
    }
    let names: HashMap<String, usize> = partition
        .functions
        .iter()
        .map(|f| (f.name.clone(), f.index))
        .collect();
    let entry = *names.get(entry).ok_or(Refusal::UnknownEntry)?;
    // Reachable functions, with their trees and direct callees.
    let mut trees: BTreeMap<usize, (FunctionTree, BTreeSet<usize>)> = BTreeMap::new();
    let mut todo = vec![entry];
    while let Some(index) = todo.pop() {
        if trees.contains_key(&index) {
            continue;
        }
        let function = &partition.functions[index];
        let tree = function.tree.clone().map_err(|reason| Refusal::Opaque {
            function: function.name.clone(),
            reason: reason.reason(),
        })?;
        if !function.pure {
            return Err(Refusal::Impure {
                function: function.name.clone(),
            });
        }
        let named = |function: &FunctionTree| function.name.clone();
        if !scalar(&tree.result)
            || tree
                .params
                .iter()
                .any(|p| p.mode.is_some() || !scalar(&p.ty) || p.ty == Type::Unit)
            || tree
                .locals
                .values()
                .any(|ty| !scalar(ty) || *ty == Type::Unit)
        {
            return Err(Refusal::NonScalar {
                function: named(&tree),
            });
        }
        let mut called = BTreeSet::new();
        for name in callees(&tree.body) {
            match names.get(&name) {
                Some(&callee) => {
                    called.insert(callee);
                    todo.push(callee);
                }
                None if matches!(name.as_str(), "wrapping_add" | "wrapping_mul") => {}
                None => {
                    return Err(Refusal::Intrinsic {
                        function: named(&tree),
                        name,
                    });
                }
            }
        }
        trees.insert(index, (tree, called));
    }
    // Transitive reachability for the recursion structure.
    let reach = |from: usize| {
        let mut seen = BTreeSet::new();
        let mut stack: Vec<usize> = trees[&from].1.iter().copied().collect();
        while let Some(f) = stack.pop() {
            if seen.insert(f) {
                stack.extend(trees[&f].1.iter().copied());
            }
        }
        seen
    };
    let reaches: BTreeMap<usize, BTreeSet<usize>> = trees.keys().map(|&f| (f, reach(f))).collect();
    for (&f, set) in &reaches {
        if set.iter().any(|&g| g != f && reaches[&g].contains(&f)) {
            return Err(Refusal::MutualRecursion {
                function: trees[&f].0.name.clone(),
            });
        }
    }
    let mut machine: BTreeSet<usize> = trees
        .keys()
        .copied()
        .filter(|f| reaches[f].contains(f) || !form.functions[*f].forks.is_empty())
        .collect();
    loop {
        let grow: Vec<usize> = trees
            .iter()
            .filter(|(f, (_, called))| {
                !machine.contains(f) && called.iter().any(|g| machine.contains(g))
            })
            .map(|(f, _)| *f)
            .collect();
        if grow.is_empty() {
            break;
        }
        machine.extend(grow);
    }
    if !machine.contains(&entry) {
        return Err(Refusal::EntryNotTask);
    }
    let machine_names: BTreeSet<String> = machine.iter().map(|f| trees[f].0.name.clone()).collect();
    let mut functions = BTreeMap::new();
    let (mut frame_words, mut siblings, mut params) = (1, 1, 1);
    for (index, (tree, _)) in trees {
        let is_machine = machine.contains(&index);
        let mut slots = BTreeMap::new();
        for param in &tree.params {
            let next = slots.len();
            slots.insert(param.name.clone(), next);
        }
        for name in tree.locals.keys() {
            let next = slots.len();
            slots.entry(name.clone()).or_insert(next);
        }
        let forks = if is_machine {
            lower::fork_points(&tree.body, &form.functions[index].forks)
        } else {
            Vec::new()
        };
        if is_machine {
            lower::check_machine(&tree, &machine_names)?;
        }
        // A plain function's frame (as a forked child) holds its parameters.
        frame_words = frame_words.max(if is_machine {
            slots.len()
        } else {
            tree.params.len()
        });
        siblings = forks.iter().map(|(_, n)| *n).fold(siblings, usize::max);
        params = params.max(tree.params.len());
        functions.insert(
            index,
            Function {
                tree,
                machine: is_machine,
                slots,
                forks,
            },
        );
    }
    if frame_words > MAX_FRAME_WORDS || siblings > MAX_SIBLINGS || params > MAX_PARAMS {
        return Err(Refusal::Limit);
    }
    Ok(Program {
        functions,
        names,
        entry,
        frame_words,
        siblings,
        params,
        fingerprint: form.fingerprint,
    })
}

/// Device status words of the task ABI.
pub mod status {
    pub const OK: u32 = 0;
    pub const OVERFLOW: u32 = 1;
    pub const DIVISION_BY_ZERO: u32 = 2;
    pub const CALL_DEPTH: u32 = 4;
    pub const STEP_BUDGET: u32 = 5;
    pub const INVALID_STATE: u32 = 7;
    /// A later sibling of a failed node (never observed by a join).
    pub const CANCELLED: u32 = 8;
}

/// Encode entry arguments as 32-bit words (the params buffer).
pub fn encode_args(program: &Program, args: &[Value]) -> Option<Vec<u32>> {
    let params = program.entry_params();
    if params.len() != args.len() {
        return None;
    }
    params
        .iter()
        .zip(args)
        .map(|(ty, value)| match (ty, value) {
            (Type::I32, Value::I32(x)) => Some(*x as u32),
            (Type::U32, Value::U32(x)) => Some(*x),
            (Type::Bool, Value::Bool(x)) => Some(u32::from(*x)),
            _ => None,
        })
        .collect()
}

/// Decode the root observation `(status, value, span start, span end)`.
pub fn decode(program: &Program, words: [u32; 4]) -> Result<Value, EvalError> {
    let [code, value, start, end] = words;
    let span = Span {
        start: start as usize,
        end: end as usize,
    };
    match code {
        status::OK => Ok(match program.entry_result() {
            Type::I32 => Value::I32(value as i32),
            Type::U32 => Value::U32(value),
            Type::Bool => Value::Bool(value != 0),
            _ => Value::Unit,
        }),
        status::OVERFLOW => Err(EvalError::Trap {
            category: TrapCategory::IntegerOverflow,
            span,
        }),
        status::DIVISION_BY_ZERO => Err(EvalError::Trap {
            category: TrapCategory::DivisionByZero,
            span,
        }),
        status::CALL_DEPTH => Err(EvalError::Tool {
            code: "CALL_DEPTH_LIMIT",
            message: "task region call depth limit exceeded".to_owned(),
        }),
        status::STEP_BUDGET => Err(EvalError::Tool {
            code: "RESOURCE_LIMIT",
            message: "task region lane step budget exhausted".to_owned(),
        }),
        _ => Err(EvalError::Tool {
            code: "MIR_INVALID",
            message: format!("task region status {code}"),
        }),
    }
}

/// Soundness bugs injected on purpose; each must be caught by a named test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutation {
    None,
    /// A join reports the last failing child instead of the first.
    LastFailure,
    /// A join reports a failed sibling argument before earlier child failures.
    EagerArgumentFailure,
    /// Children inherit the parent's call depth (depth limit off by one).
    DepthOffByOne,
    /// A fork join binds the sibling results in reverse order.
    SwapBindings,
    /// An arena-full fallback restarts the node from the step that spawned,
    /// not from the last committed state (loses the join bindings).
    StaleFallback,
    /// A failed child cancels its earlier siblings instead of its later ones.
    CancelEarlier,
    /// The first child to finish resumes its parent (the join does not wait
    /// for the last sibling).
    EarlyJoin,
}
