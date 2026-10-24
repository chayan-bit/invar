//! Exact rank-1 execution and reached-source publication.
use super::*;

fn evaluate(p: &Program, inputs: &[u32], index: usize) -> Result<u32, Error> {
    let mut values = Vec::with_capacity(p.nodes.len());
    for node in &p.nodes {
        let w = |i: usize| values.get(i).copied().ok_or(Error::Input);
        let f = |i| w(i).map(F32Bits::from_bits);
        let value = match *node {
            Node::Input(slot) => *inputs
                .get(slot * p.length as usize + index)
                .ok_or(Error::Input)?,
            Node::Constant(w, _) => w,
            Node::Neg(a) => f(a)?.negate().to_bits(),
            Node::Add(a, b) => f(a)?.add(f(b)?).to_bits(),
            Node::Sub(a, b) => f(a)?.sub(f(b)?).to_bits(),
            Node::Mul(a, b) => f(a)?.mul(f(b)?).to_bits(),
            Node::Fma(a, b, c) => f(a)?.fma(f(b)?, f(c)?).to_bits(),
            Node::FromU32(a) => F32Bits::from_u32(w(a)?).to_bits(),
            Node::FromI32(a) => F32Bits::from_i32(w(a)? as i32).to_bits(),
            Node::TruncI32(a) => f(a)?.trunc_i32().map_err(|_| Error::IntegerOverflow)? as u32,
            Node::Sqrt(a) => f(a)?.sqrt().to_bits(),
            Node::Compare(op, a, b) => u32::from(f(a)?.compare(
                match op {
                    Compare::Eq => Comparison::Eq,
                    Compare::Ne => Comparison::Ne,
                    Compare::Lt => Comparison::Lt,
                    Compare::Le => Comparison::Le,
                    Compare::Gt => Comparison::Gt,
                    Compare::Ge => Comparison::Ge,
                },
                f(b)?,
            )),
        };
        values.push(value);
    }
    values.last().copied().ok_or(Error::Input)
}
/// Exact software CPU oracle; maps distribute independent elements, folds keep
/// the source-order left fold. Every worker joins before publication.
pub fn cpu(plan: &v2::Verified, inputs: &[u32], workers: usize) -> Result<Vec<u32>, Error> {
    cpu_details(plan, inputs, workers).map(|(values, _, _)| values)
}
fn cpu_details(
    plan: &v2::Verified,
    inputs: &[u32],
    workers: usize,
) -> Result<crate::schedule::workers::Scheduled<Vec<u32>>, Error> {
    let p = f32_ops::program(plan).map_err(Error::Device)?;
    if inputs.len() != p.packed_len() || !(1..=crate::schedule::MAX_WORKERS).contains(&workers) {
        return Err(Error::Input);
    }
    if let Kind::Reduce { initial } | Kind::Scan { initial } = p.kind {
        let mut sum = F32Bits::from_bits(initial);
        let mut output = Vec::with_capacity(p.output_len());
        for i in 0..p.length as usize {
            sum = sum.add(F32Bits::from_bits(evaluate(p, inputs, i)?));
            if matches!(p.kind, Kind::Scan { .. }) {
                output.push(sum.to_bits());
            }
        }
        if matches!(p.kind, Kind::Reduce { .. }) {
            output.push(sum.to_bits());
        }
        return Ok((output, Vec::new(), 0));
    }
    let (parts, ranges, joined) = crate::schedule::workers::parallel_chunks(
        p.length as usize,
        workers,
        &crate::schedule::Hooks::default(),
        |range| {
            Ok(range
                .map(|i| evaluate(p, inputs, i))
                .collect::<Result<Vec<_>, _>>())
        },
    )
    .map_err(Error::Worker)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(p.output_len())
        .map_err(|_| Error::Worker(crate::schedule::Error::Allocation))?;
    for part in parts {
        output.extend(part?);
    }
    Ok((output, ranges, joined))
}

pub(crate) struct Completion {
    pub region: Region,
    pub length: usize,
    pub input_lengths: Vec<(String, usize)>,
    pub words: Vec<u32>,
    pub ticks: u64,
}
pub struct Run {
    pub observation: eval::Observation,
    pub completed: usize,
    pub refusals: Vec<&'static str>,
    pub workers_used: usize,
    pub kind: crate::plan::PlanKind,
    pub chunks: Vec<std::ops::Range<usize>>,
    pub joined: usize,
    pub executed: Vec<crate::schedule::Executed>,
}

/// The backend sees only a sealed v2 plan and privately packed words.
/// `None` is a quiescent refusal, which records a reason and preserves source.
pub fn run_source<F>(
    source: &[u8],
    fuel: u64,
    caps: invar_device::Capabilities,
    mut backend: F,
) -> Result<Run, Error>
where
    F: FnMut(&v2::Verified, &[u32]) -> Result<Option<Vec<u32>>, eval::EvalError>,
{
    let checked = crate::check::check(front::parse(source).map_err(Error::Source)?, true)
        .map_err(Error::Source)?;
    crate::ownership::check(&checked).map_err(Error::Source)?;
    let program = mir::lower(&checked).map_err(Error::Mir)?;
    let regions = checked
        .program
        .functions
        .iter()
        .filter_map(|f| Region::from_checked(&checked, &f.name).ok())
        .collect::<Vec<_>>();
    let mut refusals = Vec::new();
    let mut kind = crate::plan::PlanKind::Serial;
    let mut executed = Vec::new();
    let mut provider =
        |live: &mir_eval::LiveLoop<'_>| -> Result<Option<mir_eval::LoopOverride>, eval::EvalError> {
            let Some(region) = regions
                .iter()
                .find(|r| r.function == live.function && r.span == live.span)
            else {
                return Ok(None);
            };
            let Some(ticks) = region
                .ticks_per_iteration
                .checked_mul(live.iterations as u64)
            else {
                return Ok(None);
            };
            if ticks > live.fuel_remaining
                || region.inputs.iter().any(|(name, _)| {
                    live.same_floating(name, &region.output)
                        || live.same_object(name, &region.output)
                })
            {
                return Ok(None);
            }
            let plan = match region.bind(live.iterations as u32, caps) {
                Ok(p) => p,
                Err(_) => {
                    refusals.push("f32_device_contract");
                    return Ok(None);
                }
            };
            let mut packed = Vec::new();
            if packed
                .try_reserve_exact(live.iterations.saturating_mul(region.inputs.len()))
                .is_err()
            {
                return Ok(None);
            }
            let mut lengths = Vec::new();
            for (name, ty) in &region.inputs {
                let Some(words) = live.words(name, ty) else {
                    return Ok(None);
                };
                if is_array(ty) {
                    if words.len() < live.iterations {
                        return Ok(None);
                    }
                    lengths.push((name.clone(), words.len()));
                    packed.extend_from_slice(&words[..live.iterations]);
                } else {
                    let Some(w) = words.first().copied() else {
                        return Ok(None);
                    };
                    packed.extend(std::iter::repeat_n(w, live.iterations));
                }
            }
            let words = match backend(&plan, &packed)? {
                Some(w) => w,
                None => {
                    refusals.push("f32_quiescent_fallback");
                    return Ok(None);
                }
            };
            kind = match f32_ops::program(&plan)
                .map_err(|_| tool("MIR_INVALID", "lost f32 program"))?
                .kind
            {
                Kind::Map => crate::plan::PlanKind::Map,
                Kind::Reduce { .. } => crate::plan::PlanKind::Reduction,
                Kind::Scan { .. } => crate::plan::PlanKind::Scan,
            };
            if words.len()
                != f32_ops::program(&plan)
                    .map_err(|_| tool("DEVICE_TOOLCHAIN_MISMATCH", "lost f32 program"))?
                    .output_len()
            {
                return Err(tool(
                    "DEVICE_TOOLCHAIN_MISMATCH",
                    "f32 output length mismatch",
                ));
            }
            executed.push(crate::schedule::Executed {
                function: region.function.clone(),
                span: region.span,
                kind,
                published_array: if matches!(kind, crate::plan::PlanKind::Reduction) {
                    None
                } else {
                    Some(region.output.clone())
                },
                implementation: None,
            });
            Ok(Some(mir_eval::LoopOverride::tensor(
                crate::tensor::host::Completion::Linear(Completion {
                    region: region.clone(),
                    length: live.iterations,
                    input_lengths: lengths,
                    words,
                    ticks,
                }),
            )))
        };
    let (observation, completed) =
        mir_eval::observe_with_scheduler(&program, fuel, Some(&mut provider));
    Ok(Run {
        observation,
        completed,
        refusals,
        workers_used: 0,
        kind,
        chunks: Vec::new(),
        joined: 0,
        executed,
    })
}

pub fn run_cpu_source(source: &[u8], fuel: u64, workers: usize) -> Result<Run, Error> {
    if !(1..=crate::schedule::MAX_WORKERS).contains(&workers) {
        return Err(Error::Input);
    }
    let mut used = 0;
    let mut chunks = Vec::new();
    let mut joined = 0;
    let mut run = run_source(
        source,
        fuel,
        invar_device::Capabilities::cuda_12_9_sm89(),
        |p, inputs| match cpu_details(p, inputs, workers) {
            Ok((words, ranges, count)) => {
                used = used.max(ranges.len().max(1));
                joined += count;
                if !ranges.is_empty() {
                    chunks = ranges;
                }
                Ok(Some(words))
            }
            Err(Error::IntegerOverflow) => Ok(None),
            Err(Error::Worker(error)) => Err(eval::EvalError::Tool {
                code: error.code(),
                message: format!("f32 CPU worker failed: {error:?}"),
            }),
            Err(_) => Err(tool("CPU_CANDIDATE", "f32 candidate failed")),
        },
    )?;
    run.workers_used = used;
    run.chunks = chunks;
    run.joined = joined;
    Ok(run)
}

#[cfg(feature = "cuda")]
pub fn run_cuda_source(
    source: &[u8],
    fuel: u64,
    device: &mut invar_cuda::cuda::CudaDevice,
    compiler: &std::path::Path,
) -> Result<Run, Error> {
    run_source(
        source,
        fuel,
        device.capabilities(),
        |plan, inputs| match device.run_f32_ops(plan, inputs, compiler) {
            Ok(words) => Ok(Some(words)),
            Err(_) if device.quarantine_len() != 0 => Err(tool(
                "CUDA_PENDING",
                "f32 completion unconfirmed; source replay forbidden",
            )),
            Err(invar_cuda::Error::IntegerOverflow | invar_cuda::Error::Unavailable) => Ok(None),
            Err(error) => Err(eval::EvalError::Tool {
                code: "CUDA_UNAVAILABLE",
                message: format!("f32 CUDA dispatch failed: {error:?}"),
            }),
        },
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn run_metal_source(
    source: &[u8],
    fuel: u64,
    device: &mut invar_metal::MetalDevice,
) -> Result<Run, Error> {
    let mut fallback = 0;
    let mut run = run_source(
        source,
        fuel,
        device.capabilities(),
        |plan, inputs| match device.run_f32_ops(plan, inputs) {
            Ok(words) => Ok(Some(words)),
            Err(invar_metal::Error::SubnormalFallback) => {
                fallback += 1;
                Ok(None)
            }
            Err(invar_metal::Error::IntegerOverflow) => Ok(None),
            Err(invar_metal::Error::Pending) => Err(tool(
                "METAL_PENDING",
                "f32 completion unconfirmed; source replay forbidden",
            )),
            Err(invar_metal::Error::Unavailable(_)) => Ok(None),
            Err(_) => Err(tool("METAL_UNAVAILABLE", "f32 Metal dispatch failed")),
        },
    )?;
    run.refusals
        .extend(std::iter::repeat_n("metal_subnormal_fallback", fallback));
    Ok(run)
}
