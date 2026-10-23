//! Reached source-loop admission and completed, privately staged publication.
//! The generic device owns request staging until quiescence. No GPU is opened
//! here; CpuDevice executes the exact ordered software numeric contract.
use super::{Contract, Error};
use crate::{
    eval::{EvalError, Observation},
    mir_eval::{self, LiveLoop, LoopOverride},
    numeric_contract::F32Bits,
};
use invar_cuda::{Device, Runtime, SubmitError};
use invar_device::Capabilities;
use std::sync::atomic::{AtomicU64, Ordering};

const HEADER: usize = 144;
static GENERATION: AtomicU64 = AtomicU64::new(1);
const CPU_WORK_LIMIT: u64 = 65_536;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
pub(crate) mod packet;

fn pending() -> EvalError {
    EvalError::Tool {
        code: "CUDA_UNAVAILABLE",
        message: "device completion is unconfirmed; retained work blocks source replay".into(),
    }
}

pub(crate) enum Completion {
    Contraction(ContractionCompletion),
    Linear(crate::f32_source::Completion),
}
pub(crate) struct ContractionCompletion {
    contract: Contract,
    dimensions: [u32; 3],
    lengths: [usize; 2],
    values: Vec<F32Bits>,
    ticks: u64,
}
#[cfg(test)]
impl Completion {
    /// Test-only tensor receipt bound to a real contract, for cross-family
    /// ambiguity tests; it carries no values and is never published.
    pub(crate) fn for_ambiguity_test(contract: Contract) -> Self {
        Self::Contraction(ContractionCompletion {
            contract,
            dimensions: [1, 1, 1],
            lengths: [1, 1],
            values: Vec::new(),
            ticks: 0,
        })
    }
}
impl ContractionCompletion {
    pub(crate) fn parts(&self) -> (&Contract, [u32; 3], [usize; 2], &[F32Bits], u64) {
        (
            &self.contract,
            self.dimensions,
            self.lengths,
            &self.values,
            self.ticks,
        )
    }
}

pub struct Run {
    pub observation: Observation,
    pub completed: usize,
    pub refusals: Vec<&'static str>,
}

fn request(plan: &super::Emission, inputs: [&[F32Bits]; 2]) -> Option<Vec<u8>> {
    request_packet(
        plan.key(),
        plan.layouts().dimensions(),
        plan.plan().ir().output.elements,
        inputs,
    )
}

pub(crate) fn request_packet(
    key: &crate::tensor_cache::CacheKey,
    dimensions: [u32; 3],
    output: u64,
    inputs: [&[F32Bits]; 2],
) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    let count = inputs[0].len().checked_add(inputs[1].len())?;
    bytes
        .try_reserve_exact(HEADER.checked_add(count.checked_mul(4)?)?)
        .ok()?;
    bytes.extend_from_slice(b"itf1");
    for n in dimensions {
        bytes.extend_from_slice(&n.to_le_bytes());
    }
    for n in [inputs[0].len() as u64, inputs[1].len() as u64, output] {
        bytes.extend_from_slice(&n.to_le_bytes());
    }
    bytes.extend_from_slice(key.to_hex().as_bytes());
    let generation = GENERATION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .ok()?;
    bytes.extend_from_slice(&generation.to_le_bytes());
    bytes.extend_from_slice(&[0; 32]);
    for values in inputs {
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    let digest = packet::packet_digest(&bytes);
    bytes[112..144].copy_from_slice(&digest);
    Some(bytes)
}
pub(crate) fn response(bytes: &[u8], request: &[u8], output_len: usize) -> Option<Vec<F32Bits>> {
    if bytes.get(..HEADER)? != request.get(..HEADER)?
        || bytes.len() != HEADER.checked_add(output_len.checked_mul(4)?)?
    {
        return None;
    }
    let mut values = Vec::new();
    values.try_reserve_exact(output_len).ok()?;
    for word in bytes[HEADER..].chunks_exact(4) {
        let value = F32Bits::from_bits(u32::from_le_bytes(word.try_into().ok()?));
        if value.is_nan() || value.is_infinite() {
            return None;
        }
        values.push(value);
    }
    Some(values)
}

/// Reverify actual source and reach each loop before preparing any work.
/// A source fuel refusal or optimization-specific miss runs the original loop.
/// Device packets bind the source-derived artifact key, dimensions and lengths;
/// all input bytes are privately copied. Device::Work must own that staging.
pub fn run_source<D: Device>(
    source: &[u8],
    fuel: u64,
    runtime: &mut Runtime<D>,
    capabilities: Capabilities,
) -> Result<Run, Error> {
    let checked = crate::check::check(crate::front::parse(source).map_err(Error::Source)?, true)
        .map_err(Error::Source)?;
    crate::ownership::check(&checked).map_err(Error::Source)?;
    let program = crate::mir::lower(&checked).map_err(Error::Mir)?;
    let mut refusals = Vec::new();
    let mut provider = |live: &LiveLoop<'_>| {
        let Some(contract) = program
            .tensor_contracts
            .iter()
            .find(|c| c.function() == live.function && c.loop_span() == live.span)
        else {
            return Ok(None);
        };
        if runtime.blocks_dispatch() {
            return Err(pending());
        }
        let dimensions = contract
            .dimensions()
            .iter()
            .map(|n| live.scalars.get(n).copied())
            .collect::<Option<Vec<_>>>();
        let Some(dimensions) = dimensions.and_then(|v| <[u32; 3]>::try_from(v).ok()) else {
            return Ok(None);
        };
        let Some(ticks) = contract.loop_fuel(dimensions) else {
            refusals.push("tensor_fuel_overflow");
            return Ok(None);
        };
        if ticks > live.fuel_remaining {
            refusals.push("tensor_source_fuel");
            return Ok(None);
        }
        if live.same_floating(contract.output(), &contract.inputs()[0])
            || live.same_floating(contract.output(), &contract.inputs()[1])
        {
            refusals.push("tensor_alias");
            return Ok(None);
        }
        // These live read loans persist through prepare/submit/poll. No backend
        // receives an original allocation pointer, including the fresh output.
        let Some(a) = live.floating(&contract.inputs()[0]) else {
            return Ok(None);
        };
        let Some(b) = live.floating(&contract.inputs()[1]) else {
            return Ok(None);
        };
        let Some(output) = live.floating(contract.output()) else {
            return Ok(None);
        };
        if a.iter()
            .chain(b.iter())
            .any(|v| v.is_nan() || v.is_infinite())
        {
            refusals.push("tensor_nonfinite_input");
            return Ok(None);
        }
        let plan = match super::emit(
            source,
            contract.function(),
            dimensions,
            [a.len() as u64, b.len() as u64],
            capabilities,
        ) {
            Ok(plan) if plan.contraction().contract() == contract => plan,
            _ => {
                refusals.push("tensor_contract_or_layout");
                return Ok(None);
            }
        };
        if output.len() as u64 != plan.plan().ir().output.elements {
            return Ok(None);
        }
        let Some(packet) = request(&plan, [&a, &b]) else {
            refusals.push("tensor_staging_limit");
            return Ok(None);
        };
        let prepared = match runtime.prepare(&packet) {
            Ok(work) => work,
            // The device refused strict work before any submission (for
            // example a failed D1 numeric probe); the source loop runs instead.
            Err(invar_cuda::Error::Unavailable) => {
                refusals.push("tensor_device_unavailable");
                return Ok(None);
            }
            Err(_) => {
                refusals.push("tensor_prepare");
                return Ok(None);
            }
        };
        let mut submitted = match prepared.submit_classified() {
            Ok(work) => work,
            Err(SubmitError::InFlightUnknown(_)) => return Err(pending()),
            // The Metal underflow guard discarded the device words; the
            // source loop now runs exactly on the CPU, and the receipt says so.
            Err(SubmitError::FailedQuiescent(invar_cuda::Error::SubnormalFallback)) => {
                refusals.push("metal_subnormal_fallback");
                return Ok(None);
            }
            Err(SubmitError::FailedQuiescent(_)) => {
                refusals.push("tensor_submit");
                return Ok(None);
            }
        };
        let bytes = match submitted.poll() {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return Err(pending()),
            Err(_) if !submitted.is_quiescent() => return Err(pending()),
            Err(_) => {
                refusals.push("tensor_unconfirmed_completion");
                return Ok(None);
            }
        };
        let Some(values) = response(&bytes, &packet, output.len()) else {
            refusals.push("tensor_stale_or_malformed_output");
            return Ok(None);
        };
        Ok(Some(LoopOverride {
            iterations: live.iterations,
            tensor: Some(Completion::Contraction(ContractionCompletion {
                contract: contract.clone(),
                dimensions,
                lengths: [a.len(), b.len()],
                values,
                ticks,
            })),
            ..Default::default()
        }))
    };
    let (observation, completed) =
        mir_eval::observe_with_scheduler(&program, fuel, Some(&mut provider));
    Ok(Run {
        observation,
        completed,
        refusals,
    })
}

/// Synchronous real CPU implementation of the private request protocol. It
/// uses strict ordered binary32 operations, not host FP or a generated GPU.
/// Its small private work cap is placement-specific and retains source fallback.
#[derive(Default)]
pub struct CpuDevice;
pub struct CpuWork {
    request: Vec<u8>,
    dimensions: [u32; 3],
    lengths: [usize; 3],
    response: Option<Vec<u8>>,
}
impl Device for CpuDevice {
    type Work = CpuWork;
    fn prepare(&mut self, bytes: &[u8]) -> Result<CpuWork, invar_cuda::Error> {
        use invar_cuda::Error as E;
        let (dimensions, [a, b, c]) = packet::packet_shape(bytes)?;
        // The CPU has no emitted plan to compare: check the shape and the key
        // text directly. C is the dense logical output here.
        let [rows, inner, cols] = dimensions.map(u64::from);
        if rows.checked_mul(inner).is_none_or(|n| n > a as u64)
            || inner.checked_mul(cols).is_none_or(|n| n > b as u64)
            || rows.checked_mul(cols) != Some(c as u64)
            || !bytes[40..104]
                .iter()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
        {
            return Err(E::InvalidHandle);
        }
        if rows
            .checked_mul(inner)
            .and_then(|n| n.checked_mul(cols))
            .is_none_or(|n| n > CPU_WORK_LIMIT)
        {
            return Err(E::Unprofitable);
        }
        let mut request = Vec::new();
        request
            .try_reserve_exact(bytes.len())
            .map_err(|_| E::OutOfMemory)?;
        request.extend_from_slice(bytes);
        Ok(CpuWork {
            request,
            dimensions,
            lengths: [a, b, c],
            response: None,
        })
    }
    fn submit(&mut self, work: &mut CpuWork) -> Result<(), SubmitError> {
        use invar_cuda::Error as E;
        if work.response.is_some() {
            return Err(SubmitError::FailedQuiescent(E::InvalidHandle));
        }
        let mut response = Vec::new();
        response
            .try_reserve_exact(HEADER + work.lengths[2] * 4)
            .map_err(|_| SubmitError::FailedQuiescent(E::OutOfMemory))?;
        response.extend_from_slice(&work.request[..HEADER]);
        let value = |index: usize| {
            F32Bits::from_bits(u32::from_le_bytes(
                work.request[HEADER + index * 4..HEADER + index * 4 + 4]
                    .try_into()
                    .unwrap(),
            ))
        };
        let [_, inner, cols] = work.dimensions.map(|n| n as usize);
        for q in 0..work.lengths[2] {
            let (i, j) = (q / cols, q % cols);
            let mut sum = F32Bits::POS_ZERO;
            for k in 0..inner {
                sum = sum.add(value(i * inner + k).mul(value(work.lengths[0] + k * cols + j)));
            }
            response.extend_from_slice(&sum.to_bits().to_le_bytes());
        }
        work.response = Some(response);
        Ok(())
    }
    fn complete(&mut self, work: &mut CpuWork) -> Result<bool, invar_cuda::Error> {
        Ok(work.response.is_some())
    }
    fn read(&mut self, work: &mut CpuWork) -> Result<Vec<u8>, invar_cuda::Error> {
        work.response.take().ok_or(invar_cuda::Error::InvalidHandle)
    }
}
