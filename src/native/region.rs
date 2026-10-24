//! ENG-18 step 4 (#101): native kernels for admitted PNF map and reduce
//! regions over raw `u32` buffers, so a CPU region runs compiled, vectorized
//! code instead of the plan interpreter.
//!
//! The kernels are total (wrapping `u32` arithmetic only), so they never trap
//! and fuel is charged by the caller exactly as for the interpreted plan.
//! Their promises are tied to facts and checked by [`super::audit`]:
//! - `noalias` on a map's output only: the caller writes a fresh buffer it owns,
//!   distinct from every borrowed input (affine ownership);
//! - `inbounds` on every access: the caller passes slices of exactly `n`
//!   elements, and `n <= invar_foreign::region::MAX_ELEMENTS` (`u32::MAX`);
//! - `nuw` on the index increment: `i < n <= MAX_ELEMENTS`.

use std::fmt::Write as _;

use crate::{
    laws::Law,
    typed_net::{MapOp, Plan, ZipOp},
};

use super::{Fact, IntervalOp, Operand};

/// C ABI symbol of a map kernel: `void(const u32 *in0, const u32 *in1,
/// u32 *out, u64 n)` writes `out[0..n]` from `in0[0..n]` (and `in1[0..n]`);
/// `in1` is ignored for one-input maps. Callers pass chunk slices, so every
/// pointer addresses the start of `n` live elements.
pub const MAP_SYMBOL: &str = "invar_region_map_v1";
/// C ABI symbol of a reduce kernel: `u32(const u32 *in, u64 n)`, the law's
/// fold of `in[0..n]` from its identity, left to right.
pub const REDUCE_SYMBOL: &str = "invar_region_reduce_v1";

/// What the caller checks before it calls a kernel (the `InBounds` guard).
pub const GUARD: &str = "invar_foreign::region: n == len(buffer) <= MAX_ELEMENTS for every buffer";

/// A region kernel: emitted IR plus the facts behind its promises.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kernel {
    pub symbol: &'static str,
    pub ir: String,
    pub facts: Vec<Fact>,
}

/// `getelementptr inbounds i32, ptr BASE, i64 %i` with `%i < n <= MAX_ELEMENTS`.
fn bounds(base: &str) -> Fact {
    let elements = invar_foreign::region::MAX_ELEMENTS as u64;
    Fact::InBounds {
        base: base.to_owned(),
        index: Operand {
            value: "%i".to_owned(),
            range: 0..=i128::from(elements) - 1,
        },
        elements,
        guard: GUARD.to_owned(),
    }
}

/// `%next = add nuw i64 %i, 1` with `%i < n <= MAX_ELEMENTS`.
fn step() -> Fact {
    let limit = invar_foreign::region::MAX_ELEMENTS as i128;
    Fact::Interval {
        op: IntervalOp::Add,
        bits: 64,
        lhs: Operand {
            value: "%i".to_owned(),
            range: 0..=limit - 1,
        },
        rhs: Operand {
            value: "1".to_owned(),
            range: 1..=1,
        },
    }
}

fn map_op(text: &mut String, value: &mut String, temp: &mut usize, op: MapOp) {
    let (opcode, constant) = match op {
        MapOp::WrappingAdd(n) => ("add", n),
        MapOp::WrappingMul(n) => ("mul", n),
    };
    let next = format!("%v{temp}");
    *temp += 1;
    // Wrapping u32 arithmetic: no flags, so the result is defined for every input.
    let _ = writeln!(text, "  {next} = {opcode} i32 {value}, {constant}");
    *value = next;
}

/// The loop skeleton shared by both kernels: `%i` runs over `0..n`.
fn header(text: &mut String) {
    text.push_str(
        "entry:\n  %empty = icmp eq i64 %n, 0\n  br i1 %empty, label %done, label %body\nbody:\n  %i = phi i64 [ 0, %entry ], [ %next, %body ]\n",
    );
}

fn footer(text: &mut String, step: usize) {
    let _ = writeln!(
        text,
        "  %next = add nuw i64 %i, 1, !invar.fact !{step}\n  %more = icmp ult i64 %next, %n\n  br i1 %more, label %body, label %done"
    );
}

/// The kernel of a live map plan: element `i` of each input slot, its `pre`
/// operations, the optional `zip`, then every stage operation in order,
/// exactly as `Plan::live_values` computes it. `inputs` and `output` name the
/// source owners bound to the plan's slots (for the disjointness fact).
pub fn map(plan: &Plan, inputs: &[String], output: &str) -> Result<Kernel, String> {
    let Plan::LiveArray {
        inputs: slots,
        pre,
        zip,
        stages,
        ..
    } = plan
    else {
        return Err("native region kernels lower live map plans only".to_owned());
    };
    let arity = 1 + usize::from(zip.is_some());
    if slots.len() != arity || pre.len() != arity || inputs.len() != arity {
        return Err("malformed live map plan".to_owned());
    }
    let mut text = format!(
        "define void @{MAP_SYMBOL}(ptr %in0, ptr %in1, ptr noalias %out, i64 %n) !invar.fact !4 {{\n"
    );
    header(&mut text);
    let mut temp = 0;
    let mut operands = Vec::new();
    for (k, ops) in pre.iter().enumerate() {
        let _ = writeln!(
            text,
            "  %p{k} = getelementptr inbounds i32, ptr %in{k}, i64 %i, !invar.fact !{k}\n  %x{k} = load i32, ptr %p{k}"
        );
        let mut value = format!("%x{k}");
        for op in ops {
            map_op(&mut text, &mut value, &mut temp, *op);
        }
        operands.push(value);
    }
    let mut value = match zip {
        Some(op) => {
            let opcode = match op {
                ZipOp::WrappingAdd => "add",
                ZipOp::WrappingMul => "mul",
            };
            let _ = writeln!(
                text,
                "  %zip = {opcode} i32 {}, {}",
                operands[0], operands[1]
            );
            "%zip".to_owned()
        }
        None => operands.remove(0),
    };
    for op in stages.iter().flatten() {
        map_op(&mut text, &mut value, &mut temp, *op);
    }
    let _ = writeln!(
        text,
        "  %q = getelementptr inbounds i32, ptr %out, i64 %i, !invar.fact !2\n  store i32 {value}, ptr %q"
    );
    footer(&mut text, 3);
    text.push_str("done:\n  ret void\n}\n");
    // Repeated input owners are fine (inputs are only read); an input that is
    // the output stays a duplicate, so the fact refuses it.
    let mut unique: Vec<String> = Vec::new();
    for input in inputs {
        if !unique.contains(input) {
            unique.push(input.clone());
        }
    }
    let owners = std::iter::once(output.to_owned()).chain(unique).collect();
    // 0 and 1: the inputs' accesses; 2: the output's; 3: the index step; 4:
    // the output is a fresh buffer distinct from every input owner.
    let facts = vec![
        bounds("%in0"),
        bounds("%in1"),
        bounds("%out"),
        step(),
        Fact::Disjoint {
            params: vec!["%out".to_owned()],
            owners,
        },
    ];
    text.push_str(&super::fact_nodes(&facts, 0));
    Ok(Kernel {
        symbol: MAP_SYMBOL,
        ir: text,
        facts,
    })
}

/// The kernel of a lawful reduction over `input`. Wrapping `u32` sum and
/// product are associative and commutative modulo 2^32, so any combine order
/// LLVM picks (vector lanes, interleaved accumulators) gives the left fold.
pub fn reduce(law: Law) -> Kernel {
    let (opcode, identity) = match law {
        Law::WrappingSum => ("add", 0),
        Law::WrappingProduct => ("mul", 1),
    };
    debug_assert_eq!(identity, law.identity());
    let mut text = format!("define i32 @{REDUCE_SYMBOL}(ptr %in, i64 %n) {{\n");
    header(&mut text);
    let _ = writeln!(
        text,
        "  %acc = phi i32 [ {identity}, %entry ], [ %folded, %body ]\n  %p = getelementptr inbounds i32, ptr %in, i64 %i, !invar.fact !0\n  %x = load i32, ptr %p\n  %folded = {opcode} i32 %acc, %x"
    );
    footer(&mut text, 1);
    let _ = writeln!(
        text,
        "done:\n  %result = phi i32 [ {identity}, %entry ], [ %folded, %body ]\n  ret i32 %result\n}}"
    );
    // A reduce kernel writes nothing, so it needs no disjointness fact.
    let facts = vec![bounds("%in"), step()];
    text.push_str(&super::fact_nodes(&facts, 0));
    Kernel {
        symbol: REDUCE_SYMBOL,
        ir: text,
        facts,
    }
}

/// When cpu-parallel regions use compiled kernels (`INVAR_NATIVE_REGIONS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Never: the plan interpreter runs every region (`never`).
    Never,
    /// Regions of at least [`AUTO_MIN_ELEMENTS`] elements, where compiling
    /// once pays for itself (unset, or any other value).
    Auto,
    /// Every eligible region (`always`); tests and A/B measurements.
    Always,
}

/// Smallest region `Policy::Auto` compiles a kernel for. Building one kernel
/// (emit, audit, O3, link, load) took about 40 ms on a test machine, so Auto waits
/// for regions large enough that the build is small beside the pass it
/// replaces. An engineering default, not a measured crossover.
pub const AUTO_MIN_ELEMENTS: usize = 1 << 23;
/// Most distinct kernels one process keeps; later plans use the interpreter.
const MAX_CACHED: usize = 256;

/// Process-wide override of `INVAR_NATIVE_REGIONS`: 0 none, else 1 + the
/// `Policy` discriminant (tests and A/B measurements set it).
static OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Set (or with `None` clear) the process-wide policy override.
pub fn set_policy(policy: Option<Policy>) {
    let value = policy.map_or(0, |policy| 1 + policy as u8);
    OVERRIDE.store(value, std::sync::atomic::Ordering::Relaxed);
}

/// The policy in force: the override, else `INVAR_NATIVE_REGIONS`.
pub fn policy() -> Policy {
    match OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return Policy::Never,
        2 => return Policy::Auto,
        3 => return Policy::Always,
        _ => {}
    }
    match std::env::var("INVAR_NATIVE_REGIONS").as_deref() {
        Ok("never") => Policy::Never,
        Ok("always") => Policy::Always,
        _ => Policy::Auto,
    }
}

/// A kernel compiled for the live CPU and loaded into this process.
pub struct Compiled {
    kernel: invar_foreign::region::Kernel,
    /// Plan input slot of each kernel input, in kernel order.
    slots: Vec<usize>,
    /// The CPU, features and pipeline the kernel was generated for.
    pub machine: super::Machine,
}

impl Compiled {
    /// Elements `range` of the map over `arrays` (indexed by plan input slot,
    /// as `Plan::live_values` takes them), or `None` when a slot is missing or
    /// too short; the caller then runs the interpreter.
    pub fn map(&self, arrays: &[&[u32]], range: std::ops::Range<usize>) -> Option<Vec<u32>> {
        let slices = self
            .slots
            .iter()
            .map(|slot| arrays.get(*slot)?.get(range.clone()))
            .collect::<Option<Vec<_>>>()?;
        let mut out = Vec::new();
        out.try_reserve_exact(range.len()).ok()?;
        out.resize(range.len(), 0);
        self.kernel.map(&slices, &mut out).ok()?;
        CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(out)
    }

    /// The law's fold of `input`, or `None` when the kernel refuses it.
    pub fn reduce(&self, input: &[u32]) -> Option<u32> {
        let folded = self.kernel.reduce(input).ok()?;
        CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(folded)
    }
}

type Cache = std::sync::Mutex<std::collections::HashMap<String, Option<std::sync::Arc<Compiled>>>>;

fn cache() -> &'static Cache {
    static CACHE: std::sync::OnceLock<Cache> = std::sync::OnceLock::new();
    CACHE.get_or_init(Default::default)
}

/// The compiled kernel for `kernel`, compiling and loading it on first use.
/// A failed compile is remembered, so the interpreter keeps running that
/// plan without retrying; any failure only ever selects the interpreter.
static LAST_ERROR: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Why the most recent kernel build failed, if one did (diagnostics only:
/// a failed build just keeps the interpreter).
pub fn last_error() -> Option<String> {
    LAST_ERROR.lock().ok().and_then(|last| last.clone())
}

static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// How many chunks compiled kernels have computed in this process.
pub fn calls() -> usize {
    CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// How many kernels this process compiled and loaded successfully.
pub fn compiled_count() -> usize {
    cache().lock().map_or(0, |cache| {
        cache.values().filter(|entry| entry.is_some()).count()
    })
}

fn compiled(kernel: Kernel, slots: Vec<usize>) -> Option<std::sync::Arc<Compiled>> {
    let mut cache = cache().lock().ok()?;
    if let Some(entry) = cache.get(&kernel.ir) {
        return entry.clone();
    }
    if cache.len() >= MAX_CACHED {
        return None;
    }
    // Compiling under the lock keeps one compile per plan; regions are few.
    let built = build(&kernel, slots);
    if let Err(error) = &built
        && let Ok(mut last) = LAST_ERROR.lock()
    {
        *last = Some(error.clone());
    }
    let entry = built.ok().map(std::sync::Arc::new);
    cache.insert(kernel.ir, entry.clone());
    entry
}

fn eligible(len: usize) -> bool {
    len <= invar_foreign::region::MAX_ELEMENTS
        && match policy() {
            Policy::Never => false,
            Policy::Auto => len >= AUTO_MIN_ELEMENTS,
            Policy::Always => true,
        }
}

/// The compiled kernel of a live map plan over `len` elements, if the policy
/// selects one and it builds; `None` runs the interpreter.
pub fn map_kernel(
    plan: &Plan,
    inputs: &[String],
    output: &str,
    len: usize,
) -> Option<std::sync::Arc<Compiled>> {
    if !eligible(len) {
        return None;
    }
    let Plan::LiveArray { inputs: slots, .. } = plan else {
        return None;
    };
    let slots = slots
        .iter()
        .map(|slot| usize::try_from(*slot).ok())
        .collect::<Option<Vec<_>>>()?;
    compiled(map(plan, inputs, output).ok()?, slots)
}

/// The compiled kernel of a lawful `u32` reduction over `len` elements.
pub fn reduce_kernel(law: Law, len: usize) -> Option<std::sync::Arc<Compiled>> {
    if !eligible(len) {
        return None;
    }
    compiled(reduce(law), vec![0])
}

/// Audit, optimize for the live CPU, link as a shared library and load.
fn build(kernel: &Kernel, slots: Vec<usize>) -> Result<Compiled, String> {
    use sha2::{Digest, Sha256};
    let object = super::optimize::object(
        (kernel.ir.clone(), Vec::new(), Vec::new(), Vec::new()),
        kernel.facts.clone(),
        &super::Build::HOST,
    )?;
    let directory = super::unique_directory()?;
    let result = (|| {
        let object_path = directory.join("region.o");
        let library = directory.join(format!("region.{}", std::env::consts::DLL_EXTENSION));
        std::fs::write(&object_path, &object.bytes).map_err(|error| error.to_string())?;
        let linker = std::env::var_os("CC").unwrap_or_else(|| "cc".into());
        let mut command = std::process::Command::new(linker);
        command
            .arg("-shared")
            .arg("-o")
            .arg(&library)
            .arg(&object_path);
        let linked = super::wait_bounded(command)?;
        if !linked.status.success() {
            return Err(format!(
                "native region link failed: {}",
                String::from_utf8_lossy(&linked.stderr)
            ));
        }
        let bytes = std::fs::read(&library).map_err(|error| error.to_string())?;
        let hash = <[u8; 32]>::from(Sha256::digest(&bytes));
        let loaded = if kernel.symbol == MAP_SYMBOL {
            invar_foreign::region::Kernel::load_map(&library, hash, kernel.symbol, slots.len())
        } else {
            invar_foreign::region::Kernel::load_reduce(&library, hash, kernel.symbol)
        };
        loaded.map_err(|error| format!("native region load failed: {error:?}"))
    })();
    // The loaded mapping outlives its file.
    let _ = std::fs::remove_dir_all(directory);
    Ok(Compiled {
        kernel: result?,
        slots,
        machine: object.machine,
    })
}
