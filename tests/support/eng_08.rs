//! Shared helpers for the #278 ENG-08 byte-budget tests: a per-thread counting allocator, source
//! assembly, and agreement of the reference, MIR and `cpu-parallel` engines.
#![allow(dead_code)]
use invar::{check, eval, front, mir, schedule};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// Per-thread cumulative requested bytes, so parallel tests cannot disturb a measurement.
pub struct Counting;
thread_local! {
    static TOTAL: Cell<usize> = const { Cell::new(0) };
}
// SAFETY: forwards every call to `System` unchanged and only updates a thread-local counter.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        TOTAL.with(|t| t.set(t.get() + layout.size()));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
pub static ALLOC: Counting = Counting;

/// Bytes requested from the allocator while `f` runs; `f`'s result is returned alongside.
pub fn allocated_during<T>(f: impl FnOnce() -> T) -> (usize, T) {
    let before = TOTAL.with(Cell::get);
    let out = f();
    (TOTAL.with(Cell::get) - before, out)
}

pub fn checked(s: &str) -> check::CheckedProgram {
    check::check(front::parse(s.as_bytes()).unwrap(), false).unwrap()
}
pub fn array(a: &[u32]) -> String {
    let items: Vec<_> = a.iter().map(|v| format!("{v}u32")).collect();
    format!("[{}]", items.join(","))
}
pub fn permutation(n: u32) -> Vec<u32> {
    // A fixed odd stride is coprime with every power-of-two-plus-one size used here only when
    // gcd(stride, n) == 1; assert it so the scatter indices stay injective.
    let stride = 7919u32;
    assert_eq!(gcd(stride, n), 1, "n={n}");
    (0..n).map(|i| (i * stride) % n).collect()
}
pub fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}
pub fn program(
    entry_source: &str,
    call: &str,
    values: &[u32],
    indices: &[u32],
    expected: &[u32],
) -> String {
    format!(
        "{entry_source} pub fn main() -> i32 {{ let a: Array<u32> = {}; let ix: Array<u32> = {}; let wanted: Array<u32> = {}; let result = {call}; if len(result) != len(wanted) {{ return 0; }} for k in 0u32..len(result) {{ if result[k] != wanted[k] {{ return 0; }} }} return 7; }}",
        array(values),
        array(indices),
        array(expected)
    )
}

/// Reference engine, MIR evaluator and `cpu-parallel` (workers 1, 2, 7, 64) agree on `source`.
pub fn agree_on_every_cpu_engine(source: &str) -> usize {
    let p = checked(source);
    let reference = eval::observe(&p, 100_000_000);
    assert_eq!(reference.result, Ok(eval::Value::I32(7)), "reference");
    let m = mir::lower(&p).unwrap();
    assert_eq!(mir_eval_result(&m), reference.result);
    let mut engines = 1;
    for workers in [1, 2, 7, 64] {
        let run = schedule::run_source(source.as_bytes(), &m, 100_000_000, workers).unwrap();
        assert_eq!(
            run.observation.result, reference.result,
            "workers={workers}"
        );
        engines += 1;
    }
    engines
}
pub fn mir_eval_result(m: &mir::Program) -> Result<eval::Value, eval::EvalError> {
    invar::mir_eval::observe(m, 100_000_000).result
}
