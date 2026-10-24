//! Explicit sort lowering over bounded word buffers. Float keys are encoded bits;
//! payload words are opaque, so signed values, floats and Bool retain their bits.
use crate::sort_cpu;
pub use sort_cpu::{Error, KeyKind};
use std::cell::Cell;

fn parallel_pairs(
    pairs: &mut [[u32; 2]],
    workers: usize,
    joined: &Cell<usize>,
) -> Result<(), Error> {
    if workers == 1 {
        return sort_cpu::serial_pairs(pairs);
    }
    let mut scratch = sort_cpu::reserve(pairs.len())?;
    scratch.resize(pairs.len(), [0; 2]);
    for shift in [0, 8, 16, 24] {
        let (parts, _, count) = crate::schedule::parallel_chunks_with_limit(
            pairs.len(),
            workers,
            &crate::schedule::Hooks::default(),
            sort_cpu::MAX_ELEMENTS,
            |range| {
                let input = &pairs[range];
                let mut output = sort_cpu::reserve(input.len())
                    .map_err(|_| crate::schedule::Error::Allocation)?;
                output.resize(input.len(), [0; 2]);
                let counts = sort_cpu::byte_pass(input, &mut output, shift)
                    .map_err(|_| crate::schedule::Error::Candidate)?;
                Ok((output, counts))
            },
        )
        .map_err(|error| Error::WorkerFailure(error.code()))?;
        joined.set(joined.get() + count);
        let mut destination = 0;
        for digit in 0..256 {
            for (values, counts) in &parts {
                let start: usize = counts[..digit].iter().sum();
                let end = start + counts[digit];
                let target = scratch
                    .get_mut(destination..destination + counts[digit])
                    .ok_or(Error::Internal)?;
                target.copy_from_slice(values.get(start..end).ok_or(Error::Internal)?);
                destination += counts[digit];
            }
        }
        if destination != pairs.len() {
            return Err(Error::Internal);
        }
        pairs.copy_from_slice(&scratch);
    }
    Ok(())
}
fn workers_valid(workers: usize) -> Result<(), Error> {
    if (1..=crate::schedule::MAX_WORKERS).contains(&workers) {
        Ok(())
    } else {
        Err(Error::Workers)
    }
}
/// Stable argsort of u32/i32/f32 word encodings; input words are never modified.
/// At most 2^28 keys and 1..=4 workers; allocation and worker failures are explicit.
pub fn argsort_words(keys: &[u32], kind: KeyKind, workers: usize) -> Result<Vec<u32>, Error> {
    argsort_counted(keys, kind, workers, &Cell::new(0))
}
fn argsort_counted(
    keys: &[u32],
    kind: KeyKind,
    workers: usize,
    joined: &Cell<usize>,
) -> Result<Vec<u32>, Error> {
    workers_valid(workers)?;
    sort_cpu::order_with(keys, kind, None, |pairs| {
        parallel_pairs(pairs, workers, joined)
    })
}
/// Stable key/payload radix sort; supplied boundaries must cover 0..keys.len().
/// All offsets, payload length and staging allocations are checked before writes.
/// Duplicate boundaries describe empty segments; every payload bit is preserved.
pub fn sort_words(
    keys: &mut [u32],
    payload: Option<&mut [u32]>,
    kind: KeyKind,
    offsets: Option<&[u32]>,
    workers: usize,
) -> Result<(), Error> {
    sort_counted(keys, payload, kind, offsets, workers, &Cell::new(0))
}
fn sort_counted(
    keys: &mut [u32],
    payload: Option<&mut [u32]>,
    kind: KeyKind,
    offsets: Option<&[u32]>,
    workers: usize,
    joined: &Cell<usize>,
) -> Result<(), Error> {
    workers_valid(workers)?;
    sort_cpu::sort_with(keys, payload, kind, offsets, |pairs| {
        parallel_pairs(pairs, workers, joined)
    })
}

use crate::{
    eval::{self, EvalError, TrapCategory, Value},
    front::Span,
    numeric_contract::F32Bits,
};
use std::{cell::RefCell, rc::Rc};
fn invalid() -> EvalError {
    EvalError::Tool {
        code: "INTERNAL_TYPE",
        message: "invalid verified sort operands".into(),
    }
}
fn words(value: &Value) -> Result<Vec<u32>, EvalError> {
    let mut output = sort_cpu::reserve(eval::view_len(value).ok_or_else(invalid)? as usize)
        .map_err(|e| failure(e, Span { start: 0, end: 0 }))?;
    match value {
        Value::ArrayU32(v) => output.extend_from_slice(&v.try_borrow().map_err(|_| invalid())?),
        Value::ArrayI32(v) => output.extend(
            v.try_borrow()
                .map_err(|_| invalid())?
                .iter()
                .map(|&v| v as u32),
        ),
        Value::ArrayF32(v) => output.extend(
            v.try_borrow()
                .map_err(|_| invalid())?
                .iter()
                .map(|v| v.to_bits()),
        ),
        Value::ArrayBool(v) => output.extend(
            v.try_borrow()
                .map_err(|_| invalid())?
                .iter()
                .map(|&v| u32::from(v)),
        ),
        _ => return Err(invalid()),
    }
    Ok(output)
}
fn commit(value: &Value, words: &[u32]) -> Result<(), EvalError> {
    macro_rules! write {
        ($v:expr,$convert:expr) => {{
            let mut target = $v.try_borrow_mut().map_err(|_| invalid())?;
            if target.len() != words.len() {
                return Err(invalid());
            }
            for (target, &word) in target.iter_mut().zip(words) {
                *target = ($convert)(word);
            }
        }};
    }
    match value {
        Value::ArrayU32(v) => write!(v, |v| v),
        Value::ArrayI32(v) => write!(v, |v| v as i32),
        Value::ArrayF32(v) => write!(v, F32Bits::from_bits),
        Value::ArrayBool(v) => write!(v, |v| v != 0),
        _ => return Err(invalid()),
    }
    Ok(())
}
pub(crate) fn failure(error: Error, span: Span) -> EvalError {
    match error {
        Error::LengthMismatch | Error::Offset { .. } => EvalError::Trap {
            category: TrapCategory::IndexOutOfBounds,
            span,
        },
        error => EvalError::Tool {
            code: match error {
                Error::Allocation => "RESOURCE_LIMIT",
                Error::Limit => "ARRAY_SIZE_LIMIT",
                Error::Workers => "CPU_WORKERS",
                Error::WorkerFailure(code) => code,
                _ => "INTERNAL_TYPE",
            },
            message: format!("sort failed: {error:?}"),
        },
    }
}
pub(crate) fn invoke_counted(
    name: &str,
    args: &[Value],
    span: Span,
    live: &mut usize,
    workers: usize,
    joined: &Cell<usize>,
) -> Result<Value, EvalError> {
    if crate::check::sort_modes(name).map(|m| m.len()) != Some(args.len()) {
        return Err(invalid());
    }
    let kind = match args.first() {
        Some(Value::ArrayU32(_)) => KeyKind::U32,
        Some(Value::ArrayI32(_)) => KeyKind::I32,
        Some(Value::ArrayF32(_)) => KeyKind::F32,
        _ => return Err(invalid()),
    };
    let mut keys = words(&args[0])?;
    if name == "argsort" {
        let indices =
            argsort_counted(&keys, kind, workers, joined).map_err(|e| failure(e, span))?;
        eval::admit_live(live, indices.len())?;
        return Ok(Value::ArrayU32(Rc::new(RefCell::new(indices))));
    }
    let mut payload = (name == "sort_by_key")
        .then(|| words(&args[1]))
        .transpose()?;
    let offsets = (name == "segmented_sort")
        .then(|| words(&args[1]))
        .transpose()?;
    sort_counted(
        &mut keys,
        payload.as_deref_mut(),
        kind,
        offsets.as_deref(),
        workers,
        joined,
    )
    .map_err(|e| failure(e, span))?;
    commit(&args[0], &keys)?;
    if let Some(payload) = payload {
        commit(&args[1], &payload)?;
    }
    Ok(Value::Unit)
}

pub(crate) fn invoke(
    name: &str,
    args: &[Value],
    span: Span,
    live: &mut usize,
    workers: usize,
) -> Result<Value, EvalError> {
    invoke_counted(name, args, span, live, workers, &Cell::new(0))
}
pub(crate) fn contains(program: &crate::mir::Program) -> bool {
    program.functions.iter().flat_map(|f|&f.blocks).flat_map(|b|&b.instructions)
        .any(|i| matches!(&i.kind,crate::mir::Operation::Call{name,..} if crate::check::sort_modes(name).is_some()))
}
pub(crate) fn run_cpu(
    source: &[u8],
    program: &crate::mir::Program,
    fuel: u64,
    workers: usize,
) -> Result<crate::schedule::Run, crate::schedule::Error> {
    let (mut reductions, mut fault) = (Vec::new(), None);
    let mut reducer = |call: &crate::mir_eval::ReduceCall<'_>| {
        cpu_reduce(call, workers, &mut reductions, &mut fault)
    };
    let run = run_cpu_with(source, program, fuel, workers, &mut reducer)?;
    match fault {
        Some(error) => Err(error),
        None => Ok(crate::schedule::Run { reductions, ..run }),
    }
}

/// Whether `source` checks and lowers to `program`, the binding every
/// intrinsic run verifies before it executes or opens a device.
pub(crate) fn binds(source: &[u8], program: &crate::mir::Program) -> bool {
    crate::front::parse(source)
        .and_then(|p| crate::check::check(p, true))
        .ok()
        .and_then(|checked| crate::mir::lower(&checked).ok())
        .is_some_and(|lowered| lowered.contract_digest() == program.contract_digest())
}

/// [`run_cpu`] with the caller's float reducer (auto passes its cost-model
/// choice); the caller keeps the reduction receipts.
pub(crate) fn run_cpu_with(
    source: &[u8],
    program: &crate::mir::Program,
    fuel: u64,
    workers: usize,
    reducer: &mut crate::mir_eval::Reducer<'_>,
) -> Result<crate::schedule::Run, crate::schedule::Error> {
    if !binds(source, program) {
        return Err(crate::schedule::Error::Candidate);
    }
    let hooks = crate::schedule::Hooks::default();
    let mut cpu = crate::spatial::CpuIntrinsics::new(program, workers, &hooks);
    let mut provider =
        |function: &str, name: &str, args: &[Value], span: Span, live: &mut usize| {
            cpu.invoke(function, name, args, span, live)
        };
    let observation =
        crate::mir_eval::observe_all(program, fuel, None, Some(&mut provider), Some(reducer)).0;
    let crate::spatial::CpuIntrinsics {
        joined, executed, ..
    } = cpu;
    Ok(crate::schedule::Run {
        observation,
        plan: if executed.is_empty() {
            crate::plan::PlanKind::Serial
        } else {
            crate::plan::PlanKind::Sort
        },
        workers_used: joined.get(),
        chunks: Vec::new(),
        joined: joined.get(),
        fallback_reason: None,
        executed,
        reductions: Vec::new(),
    })
}

/// The fsum CPU engine's reducer for sort and spatial runs: a reached
/// float reduction runs on `workers` CPU workers with the receipt a sort-free
/// program records; a worker failure is kept in `fault` for the run's error.
pub(crate) fn cpu_reduce(
    call: &crate::mir_eval::ReduceCall<'_>,
    workers: usize,
    receipts: &mut Vec<crate::schedule::reductions::Reduction>,
    fault: &mut Option<crate::schedule::Error>,
) -> Result<Option<crate::numeric_contract::F32Bits>, EvalError> {
    use crate::schedule::reductions;
    let bits = reductions::cpu(call, workers).map_err(|error| {
        let tool = EvalError::Tool {
            code: error.code(),
            message: format!("CPU reduction failure: {error:?}"),
        };
        *fault = Some(error);
        tool
    })?;
    let mut receipt = reductions::cpu_receipt(call, workers, "forced-cpu-parallel");
    if receipt.backend == reductions::SERIAL_BACKEND {
        receipt.reason = "measured-cpu-cost";
    }
    receipts.push(receipt);
    Ok(Some(bits))
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn run_device(
    source: &[u8],
    program: &crate::mir::Program,
    fuel: u64,
    implementation: &'static str,
    mut launch: impl FnMut(
        &invar_device::sorting::Plan,
        &[u32],
        &[u32],
    ) -> Result<(Vec<u32>, Vec<u32>), EvalError>,
    reducer: &mut crate::mir_eval::Reducer<'_>,
) -> Result<(eval::Observation, Vec<crate::schedule::Executed>), EvalError> {
    let from_source = crate::front::parse(source)
        .and_then(|p| crate::check::check(p, true))
        .map_err(|_| invalid())?;
    let from_source = crate::mir::lower(&from_source).map_err(|_| invalid())?;
    if from_source.contract_digest() != program.contract_digest() {
        return Err(invalid());
    }
    let mut executed = Vec::new();
    let mut provider =
        |function: &str, name: &str, args: &[Value], span: Span, live: &mut usize| {
            let kind = match args.first() {
                Some(Value::ArrayU32(_)) => invar_device::sorting::Kind::U32,
                Some(Value::ArrayI32(_)) => invar_device::sorting::Kind::I32,
                Some(Value::ArrayF32(_)) => invar_device::sorting::Kind::F32,
                _ => return Err(invalid()),
            };
            let mut keys = words(&args[0])?;
            let mut payload = if name == "sort_by_key" {
                words(&args[1])?
            } else {
                let mut ids = sort_cpu::reserve(keys.len()).map_err(|e| failure(e, span))?;
                ids.extend(0..keys.len() as u32);
                ids
            };
            if keys.len() != payload.len() {
                return Err(failure(Error::LengthMismatch, span));
            }
            let offsets = if name == "segmented_sort" {
                Some(words(&args[1])?)
            } else {
                None
            };
            // Finish every offset check before the first upload, let alone a write.
            let ranges =
                sort_cpu::ranges(keys.len(), offsets.as_deref()).map_err(|e| failure(e, span))?;
            let mut dispatched = false;
            for range in ranges {
                if range.is_empty() {
                    continue;
                }
                let plan = invar_device::sorting::Plan::new(range.len() as u32, kind)
                    .map_err(|_| failure(Error::Limit, span))?;
                let (sorted, values) =
                    launch(&plan, &keys[range.clone()], &payload[range.clone()])?;
                if sorted.len() != range.len() || values.len() != range.len() {
                    return Err(invalid());
                }
                keys[range.clone()].copy_from_slice(&sorted);
                payload[range].copy_from_slice(&values);
                dispatched = true;
            }
            let result = if name == "argsort" {
                eval::admit_live(live, payload.len())?;
                Value::ArrayU32(Rc::new(RefCell::new(payload)))
            } else {
                if name == "sort_by_key"
                    && matches!(&args[1], Value::ArrayBool(_))
                    && payload.iter().any(|&v| v > 1)
                {
                    return Err(invalid());
                }
                commit(&args[0], &keys)?;
                if name == "sort_by_key" {
                    commit(&args[1], &payload)?;
                }
                Value::Unit
            };
            executed.push(crate::schedule::Executed {
                function: function.into(),
                span,
                kind: crate::plan::PlanKind::Sort,
                published_array: None,
                implementation: Some(if dispatched {
                    implementation
                } else {
                    "cpu-empty-sort"
                }),
            });
            Ok(result)
        };
    let observation =
        crate::mir_eval::observe_all(program, fuel, None, Some(&mut provider), Some(reducer)).0;
    Ok((observation, executed))
}
#[cfg(feature = "cuda")]
pub(crate) fn run_cuda(
    source: &[u8],
    program: &crate::mir::Program,
    fuel: u64,
    device: &mut invar_cuda::cuda::CudaDevice,
    compiler: &std::path::Path,
    fault: crate::schedule::cuda::Fault,
) -> Result<crate::schedule::cuda::Run, crate::schedule::cuda::Error> {
    let mut failure = None;
    let device = RefCell::new(device);
    let mut reductions = Vec::new();
    // Reached float reductions run on the device, as in `--engine cuda`.
    let mut reducer = |call: &crate::mir_eval::ReduceCall<'_>| {
        crate::schedule::cuda::reduce::forced(
            call,
            &mut device.borrow_mut(),
            compiler,
            &mut reductions,
        )
    };
    let run = run_device(
        source,
        program,
        fuel,
        "cuda-lsd-radix-8x4",
        |plan, keys, values| {
            let mut device = device.borrow_mut();
            match device.run_sort_with_fault(plan, keys, values, compiler, fault.collective()) {
                Ok(result) => Ok((result.keys, result.payload)),
                Err(error) => {
                    let error =
                        crate::schedule::cuda::Error::from_cuda(error, device.is_poisoned());
                    failure = Some(error.clone());
                    Err(EvalError::Tool {
                        code: error.code,
                        message: error.message,
                    })
                }
            }
        },
        &mut reducer,
    )
    .map_err(|e| crate::schedule::cuda::Error::mismatch(format!("sort source binding: {e:?}")))?;
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(crate::schedule::cuda::Run {
        observation: run.0,
        executed: run.1,
        reductions,
        cache_hits: 0,
        hint_hits: 0,
        hint_invalids: 0,
    })
}
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn run_metal(
    source: &[u8],
    program: &crate::mir::Program,
    fuel: u64,
    device: &mut invar_metal::MetalDevice,
    fault: invar_metal::Fault,
) -> Result<crate::schedule::metal::Run, crate::schedule::metal::Error> {
    let mut failure = None;
    let device = RefCell::new(device);
    let mut reductions = Vec::new();
    // Reached float reductions run on the device, as in `--engine metal`.
    let workers = crate::schedule::metal::reduce::workers()?;
    let mut reducer = |call: &crate::mir_eval::ReduceCall<'_>| {
        crate::schedule::metal::reduce::forced(
            call,
            &mut device.borrow_mut(),
            workers,
            &mut reductions,
        )
    };
    let run = run_device(
        source,
        program,
        fuel,
        "metal-lsd-radix-8x4",
        |plan, keys, values| {
            let mut device = device.borrow_mut();
            match device.run_sort_with_fault(plan, keys, values, fault) {
                Ok(result) => Ok((result.keys, result.payload)),
                Err(error) => {
                    let error =
                        crate::schedule::metal::Error::from_metal(error, device.is_poisoned());
                    failure = Some(error.clone());
                    Err(EvalError::Tool {
                        code: error.code,
                        message: error.message,
                    })
                }
            }
        },
        &mut reducer,
    )
    .map_err(|e| crate::schedule::metal::Error::mismatch(format!("sort source binding: {e:?}")))?;
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(crate::schedule::metal::Run {
        observation: run.0,
        executed: run.1,
        reductions,
        cache_hits: 0,
        hint_hits: 0,
        hint_invalids: 0,
    })
}

/// Explicit builtin sites and conservative default selection. This describes
/// checked calls; only execution receipts attest that a call was reached.
pub fn explain(program: &crate::mir::Program) -> serde_json::Value {
    let mut sites = Vec::new();
    for function in &program.functions {
        let types: std::collections::HashMap<_, _> = function
            .blocks
            .iter()
            .flat_map(|block| {
                block.params.iter().map(|(id, ty)| (*id, ty)).chain(
                    block
                        .instructions
                        .iter()
                        .filter_map(|i| i.output.as_ref().map(|(id, ty)| (*id, ty))),
                )
            })
            .collect();
        for instruction in function.blocks.iter().flat_map(|block| &block.instructions) {
            let crate::mir::Operation::Call { name, args } = &instruction.kind else {
                continue;
            };
            if crate::check::sort_modes(name).is_none() {
                continue;
            }
            // Verified sort calls have a numeric array key. Do not invent an id
            // for malformed MIR; the verifier rejects it before CLI explanation.
            let ty = match args.first().and_then(|id| types.get(id)).copied() {
                Some(crate::front::Type::ArrayU32) => "u32",
                Some(crate::front::Type::ArrayI32) => "i32",
                Some(crate::front::Type::ArrayF32) => "f32",
                _ => continue,
            };
            sites.push(serde_json::json!({"function":function.name,"operation":name,"span":{"start":instruction.span.start,"end":instruction.span.end},
                "calibration_id":format!("sort.{name}.{ty}"),
                "default_implementation":"cpu-lsd-radix-4x8","reason":"sort-cost-unmeasured","rule":"explicit-stable-sort-v1"}));
        }
    }
    serde_json::json!(sites)
}
