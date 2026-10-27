//! CPU execution model of the emitted ABI3 CUDA/Metal per-root continuation
//! kernel `invar_gnet_resume_v3` (machinery in `epoch_model`).
//! This checks the device lowering of ordinary-source owned continuations; it
//! is never a GPU, driver, memory-model or performance claim.

use crate::epoch_model::{self, *};

use invar::{
    eval,
    general_net::{Net, compiled, oracle},
};
use std::collections::BTreeSet;

mod cases;

/// ABI3 roots are one `u32` word each, or none for a parameterless entry.
fn harness(
    cpu: &compiled::Artifact,
    dev: &compiled::Artifact,
    roots: &[u32],
    l: oracle::Limits,
) -> String {
    let params = cpu.params().len();
    assert!(params <= 1);
    let roots: Vec<Vec<u32>> = roots.iter().map(|&r| vec![r; params]).collect();
    epoch_model::harness(cpu, dev, "invar_gnet_resume_v3", &roots, l)
}

struct Case {
    name: &'static str,
    source: &'static str,
    entry: &'static str,
    roots: &'static [u32],
    trace_events: usize,
    transitions: u64,
    storage: compiled::OwnedLimits,
    /// Roots whose backend storage limit (status 11-13) has no oracle analogue;
    /// they are checked word-for-word against the CPU artifact every epoch.
    backend: &'static [(u32, i64)],
}
impl Case {
    fn limits(&self) -> oracle::Limits {
        oracle::Limits {
            fuel: 1000,
            transitions: self.transitions,
            trace_events: self.trace_events,
            ..oracle::Limits::default()
        }
    }
}
const STORAGE: compiled::OwnedLimits = compiled::OwnedLimits {
    elements: 64,
    allocations: 16,
    cleanup_events: 32,
};
const CASES: &[Case] = &[
    Case {
        name: "fib",
        source: "fn fib(n: u32) -> i32 { if n < 2u32 { return 1; } return fib(n - 1u32) + fib(n - 2u32); } fn main() -> i32 { return fib(5u32); }",
        entry: "fib",
        roots: &[0, 1, 2, 3, 5, 7, 9, 4, 6, 8],
        trace_events: 2000,
        transitions: 10_000,
        storage: STORAGE,
        backend: &[],
    },
    Case {
        name: "loans",
        source: "fn mutate(xs: mut Array<i32>, n: u32) -> Unit { for i in 0u32..n { let tmp = filled_i32(2u32, 3); xs[0u32] = xs[0u32] + tmp[1u32]; } } fn run(n: u32) -> i32 { var xs: Array<i32> = filled_i32(n + 1u32, 1); mutate(xs, n); return xs[0u32]; } fn main() -> i32 { return run(2u32); }",
        entry: "run",
        roots: &[0, 1, 2, 5, 3, 8],
        trace_events: 2000,
        transitions: 10_000,
        storage: STORAGE,
        backend: &[],
    },
    Case {
        name: "moves",
        source: "fn consume(xs: take Array<i32>, n: u32) -> i32 { if n == 0u32 { return xs[0u32]; } return consume(move xs, n - 1u32); } fn run(n: u32) -> i32 { let xs: Array<i32> = [7]; return consume(move xs, n); } fn main() -> i32 { return run(3u32); }",
        entry: "run",
        roots: &[0, 3, 1, 6, 2],
        trace_events: 2000,
        transitions: 10_000,
        storage: STORAGE,
        backend: &[],
    },
    Case {
        name: "mixed",
        source: "fn run(n: u32) -> i32 { if n == 0u32 { while true {} } if n == 1u32 { return 17; } if n == 4u32 { return 1 / 0; } let xs: Array<i32> = [1]; return xs[n]; } fn main() -> i32 { return run(1u32); }",
        entry: "run",
        roots: &[0, 1, 2, 4, 3, 1],
        trace_events: 10_000,
        transitions: 10_000,
        storage: STORAGE,
        backend: &[],
    },
    Case {
        name: "limits",
        source: "fn deep(n: u32) -> i32 { return deep(n + 1u32); } fn run(n: u32) -> i32 { if n == 0u32 { let x: i32 = 2147483647; return x + 1; } if n == 1u32 { return deep(0u32); } if n == 2u32 { let big: Array<i32> = filled_i32(16777217u32, 0); return big[0u32]; } return 5; } fn main() -> i32 { return run(3u32); }",
        entry: "run",
        roots: &[0, 1, 2, 3],
        trace_events: 2000,
        transitions: 10_000,
        storage: STORAGE,
        backend: &[],
    },
    Case {
        name: "transitions",
        source: "fn fib(n: u32) -> i32 { if n < 2u32 { return 1; } return fib(n - 1u32) + fib(n - 2u32); } fn main() -> i32 { return fib(5u32); }",
        entry: "fib",
        roots: &[1, 6, 9],
        trace_events: 10_000,
        transitions: 150,
        storage: STORAGE,
        backend: &[],
    },
    Case {
        name: "storage",
        source: "fn run(n: u32) -> i32 { if n == 0u32 { let wide: Array<i32> = filled_i32(9u32, 0); return wide[0u32]; } if n == 1u32 { for i in 0u32..3u32 { let t: Array<i32> = filled_i32(1u32, 0); } return 1; } if n == 2u32 { let a: Array<i32> = [1]; let b: Array<i32> = [2]; let c: Array<i32> = [3]; let d: Array<i32> = [4]; let e: Array<i32> = [5]; return 1; } return 2; } fn main() -> i32 { return run(3u32); }",
        entry: "run",
        roots: &[0, 1, 2, 3],
        trace_events: 2000,
        transitions: 10_000,
        storage: compiled::OwnedLimits {
            elements: 8,
            allocations: 4,
            cleanup_events: 2,
        },
        backend: &[(0, 11), (1, 13), (2, 12)],
    },
    Case {
        // More than 32 roots: two CUDA blocks, the second partly past R, so the
        // block offset in the device lane index is exercised.
        name: "blocks",
        source: "fn fib(n: u32) -> i32 { if n < 2u32 { return 1; } return fib(n - 1u32) + fib(n - 2u32); } fn main() -> i32 { return fib(5u32); }",
        entry: "fib",
        roots: &[
            0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0,
            1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 6,
        ],
        trace_events: 2000,
        transitions: 10_000,
        storage: STORAGE,
        backend: &[],
    },
];
fn emit(net: &Net, case: &Case, target: compiled::Target) -> compiled::Artifact {
    let proof = net.verify().unwrap();
    compiled::emit_resumable(
        net,
        &proof,
        case.entry,
        target,
        compiled::Limits::default(),
        case.storage,
    )
    .unwrap()
}
