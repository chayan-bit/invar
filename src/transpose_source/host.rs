//! Reached ordinary transpose loops, retained read loans and typed publication.
use super::{Contract, Error};
use crate::{
    eval::{EvalError, Observation},
    front::Type,
    mir_eval::{self, LiveLoop, LoopOverride},
    numeric_contract::F32Bits,
};
use invar_cuda::{Device, Runtime, SubmitError};
use invar_device::Capabilities;
use sha2::{Digest, Sha256};
use std::{
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};
#[cfg(feature = "cuda")]
pub mod cuda;
const HEADER: usize = 144;
static GENERATION: AtomicU64 = AtomicU64::new(1);
pub(crate) enum Words {
    U32(Vec<u32>),
    F32(Vec<F32Bits>),
}
pub(crate) struct Completion {
    contract: Rc<Contract>,
    dimensions: [u32; 2],
    input_length: usize,
    values: Words,
    ticks: u64,
}
#[cfg(test)]
impl Completion {
    /// Test-only transpose receipt bound to a real contract, for cross-family
    /// ambiguity tests; it carries no values and is never published.
    pub(crate) fn for_ambiguity_test(contract: Rc<Contract>) -> Self {
        Self {
            contract,
            dimensions: [1, 1],
            input_length: 1,
            values: Words::U32(Vec::new()),
            ticks: 0,
        }
    }
}
impl Completion {
    pub(crate) fn parts(&self) -> (&Contract, [u32; 2], usize, &Words, u64) {
        (
            &self.contract,
            self.dimensions,
            self.input_length,
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
fn pending() -> EvalError {
    EvalError::Tool {
        code: "CUDA_UNAVAILABLE",
        message: "transpose completion is unconfirmed; retained work blocks source replay".into(),
    }
}
fn artifact_key(source: &[u8], contract: &Contract, plan: &super::Emission) -> [u8; 32] {
    let mut hash = Sha256::new();
    // Length framing preserves exact source bytes, names, typed layout/options,
    // emitted source and compiler identity. Root rederives this sealed plan.
    for field in [
        b"invar-private-transpose-host-v1".as_slice(),
        source,
        contract.function().as_bytes(),
        format!("{:?}", contract.element()).as_bytes(),
        format!("{:?}", plan.plan().ir()).as_bytes(),
        format!("{:?}", plan.plan().capabilities()).as_bytes(),
        format!("{:?}", plan.options()).as_bytes(),
        plan.source().as_bytes(),
        env!("INVAR_BUILD_DIGEST").as_bytes(),
    ] {
        hash.update((field.len() as u64).to_le_bytes());
        hash.update(field);
    }
    hash.finalize().into()
}
fn request(
    source: &[u8],
    contract: &Contract,
    plan: &super::Emission,
    input: &[u32],
) -> Option<Vec<u8>> {
    let mut packet = Vec::new();
    packet
        .try_reserve_exact(HEADER.checked_add(input.len().checked_mul(4)?)?)
        .ok()?;
    packet.extend_from_slice(b"itr1");
    for n in plan.plan().ir().dimensions {
        packet.extend_from_slice(&n.to_le_bytes());
    }
    packet.extend_from_slice(&u32::from(contract.element() == Type::F32).to_le_bytes());
    for n in [input.len() as u64, plan.plan().ir().output.elements, 0] {
        packet.extend_from_slice(&n.to_le_bytes());
    }
    for byte in artifact_key(source, contract, plan) {
        packet.extend_from_slice(format!("{byte:02x}").as_bytes());
    }
    let generation = GENERATION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .ok()?;
    packet.extend_from_slice(&generation.to_le_bytes());
    packet.extend_from_slice(&[0; 32]);
    for word in input {
        packet.extend_from_slice(&word.to_le_bytes());
    }
    let hash = Sha256::digest(&packet[HEADER..]);
    packet[112..144].copy_from_slice(&hash);
    Some(packet)
}
fn response(bytes: &[u8], packet: &[u8], count: usize, element: Type) -> Option<Words> {
    if bytes.get(..HEADER)? != packet.get(..HEADER)?
        || bytes.len() != HEADER.checked_add(count.checked_mul(4)?)?
    {
        return None;
    }
    let mut words = Vec::new();
    words.try_reserve_exact(count).ok()?;
    for word in bytes[HEADER..].chunks_exact(4) {
        words.push(u32::from_le_bytes(word.try_into().ok()?));
    }
    match element {
        Type::U32 => Some(Words::U32(words)),
        Type::F32 => {
            let mut floating = Vec::new();
            floating.try_reserve_exact(count).ok()?;
            floating.extend(words.into_iter().map(F32Bits::from_bits));
            Some(Words::F32(floating))
        }
        _ => None,
    }
}
/// Transpose contracts recognized in a checked source (crate-private so a
/// combined provider can share one source run and one runtime).
pub(crate) fn contracts(checked: &crate::check::CheckedProgram) -> Vec<Rc<Contract>> {
    checked
        .program
        .functions
        .iter()
        .filter(|f| f.params.len() == 3)
        .filter_map(|f| Contract::from_checked(checked, &f.name).ok().map(Rc::new))
        .collect()
}

/// One reached-loop decision, exactly as `run_source` applies it. Returns
/// `Ok(None)` without recording anything when no transpose contract matches.
pub(crate) fn provide<D: Device>(
    source: &[u8],
    contracts: &[Rc<Contract>],
    runtime: &mut Runtime<D>,
    caps: Capabilities,
    refusals: &mut Vec<&'static str>,
    live: &LiveLoop<'_>,
) -> Result<Option<LoopOverride>, EvalError> {
    let Some(contract) = contracts
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
    let Some(dimensions) = dimensions.and_then(|n| <[u32; 2]>::try_from(n).ok()) else {
        return Ok(None);
    };
    let Some(ticks) = contract.loop_fuel(dimensions) else {
        refusals.push("transpose_fuel_overflow");
        return Ok(None);
    };
    if ticks > live.fuel_remaining {
        refusals.push("transpose_source_fuel");
        return Ok(None);
    }
    if live.same_object(contract.input(), contract.output())
        || live.same_floating(contract.input(), contract.output())
    {
        refusals.push("transpose_alias");
        return Ok(None);
    }
    // Both input and fresh output stay read borrowed across the device call.
    let integer = live.array(contract.input());
    let floating = live.floating(contract.input());
    let integer_out = live.array(contract.output());
    let floating_out = live.floating(contract.output());
    let mut input = Vec::new();
    let output_len = match contract.element() {
        Type::U32 => match (integer.as_ref(), integer_out.as_ref()) {
            (Some(a), Some(b)) => {
                if input.try_reserve_exact(a.len()).is_err() {
                    refusals.push("transpose_staging_limit");
                    return Ok(None);
                }
                input.extend_from_slice(a);
                b.len()
            }
            _ => return Ok(None),
        },
        Type::F32 => match (floating.as_ref(), floating_out.as_ref()) {
            (Some(a), Some(b)) => {
                if input.try_reserve_exact(a.len()).is_err() {
                    refusals.push("transpose_staging_limit");
                    return Ok(None);
                }
                input.extend(a.iter().map(|n| n.to_bits()));
                b.len()
            }
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };
    let plan = match contract.emit_cuda(dimensions, input.len() as u64, caps) {
        Ok(p) => p,
        Err(_) => {
            refusals.push("transpose_contract_or_layout");
            return Ok(None);
        }
    };
    if output_len as u64 != plan.plan().ir().output.elements {
        return Ok(None);
    }
    let Some(packet) = request(source, contract, &plan, &input) else {
        refusals.push("transpose_staging_limit");
        return Ok(None);
    };
    let prepared = match runtime.prepare(&packet) {
        Ok(p) => p,
        Err(_) => {
            refusals.push("transpose_prepare");
            return Ok(None);
        }
    };
    let mut submitted = match prepared.submit_classified() {
        Ok(s) => s,
        Err(SubmitError::InFlightUnknown(_)) => return Err(pending()),
        Err(SubmitError::FailedQuiescent(_)) => {
            refusals.push("transpose_submit");
            return Ok(None);
        }
    };
    let bytes = match submitted.poll() {
        Ok(Some(b)) => b,
        Ok(None) => return Err(pending()),
        Err(_) if !submitted.is_quiescent() => return Err(pending()),
        Err(_) => {
            refusals.push("transpose_read");
            return Ok(None);
        }
    };
    let Some(values) = response(&bytes, &packet, output_len, contract.element()) else {
        refusals.push("transpose_stale_or_malformed_output");
        return Ok(None);
    };
    Ok(Some(LoopOverride {
        iterations: live.iterations,
        transpose: Some(Completion {
            contract: contract.clone(),
            dimensions,
            input_length: input.len(),
            values,
            ticks,
        }),
        ..Default::default()
    }))
}

pub fn run_source<D: Device>(
    source: &[u8],
    fuel: u64,
    runtime: &mut Runtime<D>,
    caps: Capabilities,
) -> Result<Run, Error> {
    let checked = crate::front::parse(source)
        .and_then(|p| crate::check::check(p, true))
        .map_err(Error::Source)?;
    let program =
        crate::mir::lower(&checked).map_err(|e| Error::Net(crate::general_net::Error::Mir(e)))?;
    let contracts = contracts(&checked);
    let mut refusals = Vec::new();
    let mut provider =
        |live: &LiveLoop<'_>| provide(source, &contracts, runtime, caps, &mut refusals, live);
    let (observation, completed) =
        mir_eval::observe_with_scheduler(&program, fuel, Some(&mut provider));
    Ok(Run {
        observation,
        completed,
        refusals,
    })
}

/// Actual synchronous word-copy CPU backend; never a GPU execution claim.
#[derive(Default)]
pub struct CpuDevice;
pub struct CpuWork {
    packet: Vec<u8>,
    dimensions: [usize; 2],
    input_len: usize,
    output_len: usize,
    response: Option<Vec<u8>>,
}
impl Device for CpuDevice {
    type Work = CpuWork;
    fn prepare(&mut self, bytes: &[u8]) -> Result<CpuWork, invar_cuda::Error> {
        use invar_cuda::Error as E;
        if bytes.get(..4) != Some(b"itr1") || bytes.len() < HEADER {
            return Err(E::InvalidHandle);
        }
        let u32_at = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().expect("header"));
        let u64_at = |i| u64::from_le_bytes(bytes[i..i + 8].try_into().expect("header"));
        let dimensions = [u32_at(4) as usize, u32_at(8) as usize];
        let input_len = usize::try_from(u64_at(16)).map_err(|_| E::Limit)?;
        let output_len = usize::try_from(u64_at(24)).map_err(|_| E::Limit)?;
        if u32_at(12) > 1
            || u64_at(32) != 0
            || u64_at(104) == 0
            || dimensions[0].checked_mul(dimensions[1]) != Some(output_len)
            || output_len > input_len
            || input_len > invar_device::transpose::MAX_LOGICAL_ELEMENTS as usize
            || HEADER.checked_add(input_len.checked_mul(4).ok_or(E::Limit)?) != Some(bytes.len())
            || !bytes[40..104]
                .iter()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
            || Sha256::digest(&bytes[HEADER..]).as_slice() != &bytes[112..144]
        {
            return Err(E::InvalidHandle);
        }
        let mut packet = Vec::new();
        packet
            .try_reserve_exact(bytes.len())
            .map_err(|_| E::OutOfMemory)?;
        packet.extend_from_slice(bytes);
        Ok(CpuWork {
            packet,
            dimensions,
            input_len,
            output_len,
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
            .try_reserve_exact(HEADER + work.output_len * 4)
            .map_err(|_| SubmitError::FailedQuiescent(E::OutOfMemory))?;
        response.extend_from_slice(&work.packet[..HEADER]);
        let [rows, cols] = work.dimensions;
        for output in 0..work.output_len {
            let row = output % rows;
            let col = output / rows;
            let index = row * cols + col;
            if index >= work.input_len {
                return Err(SubmitError::FailedQuiescent(E::InvalidHandle));
            }
            response.extend_from_slice(&work.packet[HEADER + index * 4..HEADER + index * 4 + 4]);
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

impl Contract {
    /// Bind this exact reached source loop before submitting private work.
    pub(crate) fn reached(&self, live: &LiveLoop<'_>) -> Option<([u32; 2], u64)> {
        let dimensions = [
            *live.scalars.get(&self.dimensions[0])?,
            *live.scalars.get(&self.dimensions[1])?,
        ];
        let fuel = self.loop_fuel(dimensions)?;
        (live.function == self.function
            && live.span == self.loop_span
            && live.iterations == dimensions[0] as usize
            && !dimensions.contains(&0)
            && fuel <= live.fuel_remaining
            && !live.same_object(&self.input, &self.output)
            && !live.same_floating(&self.input, &self.output))
        .then_some((dimensions, fuel))
    }

    /// Validate actual completed bits before granting publication authority.
    pub(crate) fn completed_region(
        &self,
        dimensions: [u32; 2],
        input: &[u32],
        output: Vec<u32>,
    ) -> Option<crate::mir_eval::region::CompletedRegion> {
        use crate::mir_eval::region::{Branch, CompletedRegion, Elements, Write};
        let [rows, cols] = dimensions.map(|n| n as usize);
        let count = rows.checked_mul(cols)?;
        if output.len() != count
            || input.len() < count
            || output
                .iter()
                .enumerate()
                .any(|(i, word)| *word != input[(i % rows) * cols + i / rows])
        {
            return None;
        }
        let values = match self.element {
            Type::U32 => Elements::U32(output),
            Type::F32 => {
                let mut values = Vec::new();
                values.try_reserve_exact(output.len()).ok()?;
                values.extend(output.into_iter().map(F32Bits::from_bits));
                Elements::F32(values)
            }
            _ => return None,
        };
        Some(CompletedRegion {
            function: self.function.clone(),
            span: self.loop_span,
            branches: vec![
                Branch {
                    span: self.loop_span,
                    taken: rows as u64,
                    skipped: 1,
                },
                Branch {
                    span: self.inner_span,
                    taken: count as u64,
                    skipped: rows as u64,
                },
            ],
            writes: vec![Write {
                owner: self.output.clone(),
                length: count,
                range: 0..count,
                values,
            }],
            scalars: Vec::new(),
            cleanup: Vec::new(),
            fuel: self.loop_fuel(dimensions)?,
            trap: None,
        })
    }
}
