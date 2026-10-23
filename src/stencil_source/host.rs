//! Reached ordinary stencil loops and synchronous fenced CPU tickets.
//!
//! One ticket per reached source loop: the provider stages the read input,
//! submits a sealed request through the shared `invar_cuda::Runtime`, and
//! publishes only a confirmed, validated response. Unknown in-flight work is
//! quarantined in that runtime and blocks dispatch and source replay for every
//! module dispatching through it. Host staging is outside the language
//! live-element cap and is fallibly reserved within a byte bound; publication
//! creates no new Invar owner. Repeated source calls implement ping-pong; device
//! residency across steps is out of scope.
use super::{Contract, Error};
use crate::{
    eval::{EvalError, Observation},
    mir_eval::{self, LiveLoop, LoopOverride},
    numeric_contract::F32Bits,
};
use invar_cuda::{Device, Runtime, SubmitError};
use invar_device::{Capabilities, stencil};
use sha2::{Digest, Sha256};
use std::{
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

/// Request/response header: magic, dimensions, boundary, lengths, reserved,
/// hex artifact key, generation and request-payload digest.
pub const HEADER: usize = 144;
/// Response trailer: SHA-256 of the output words.
pub const TRAILER: usize = 32;
pub const MAGIC: &[u8; 4] = b"ist1";
/// Default staging bound for one ticket (request + response bytes).
pub const STAGING_LIMIT: usize = invar_cuda::MAX_TRANSFER_BYTES;
static GENERATION: AtomicU64 = AtomicU64::new(1);

pub mod combined;

pub(crate) struct Completion {
    contract: Rc<Contract>,
    dimensions: [u32; 2],
    input_length: usize,
    values: Vec<F32Bits>,
    ticks: u64,
}
#[cfg(test)]
impl Completion {
    /// Test-only stencil receipt bound to a real contract, for cross-family
    /// ambiguity tests; it carries no values and is never published.
    pub(crate) fn for_ambiguity_test(contract: Rc<Contract>) -> Self {
        Self {
            contract,
            dimensions: [1, 1],
            input_length: 1,
            values: Vec::new(),
            ticks: 0,
        }
    }
}
impl Completion {
    pub(crate) fn parts(&self) -> (&Contract, [u32; 2], usize, &[F32Bits], u64) {
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
    /// Published loop overrides.
    pub completed: usize,
    pub refusals: Vec<&'static str>,
}

fn pending() -> EvalError {
    EvalError::Tool {
        code: "CUDA_UNAVAILABLE",
        message:
            "stencil completion is unconfirmed; retained work blocks dispatch and source replay"
                .into(),
    }
}

fn boundary_code(boundary: stencil::Boundary) -> u32 {
    match boundary {
        stencil::Boundary::ConstantZero => 0,
        stencil::Boundary::Clamp => 1,
    }
}

/// Domain-separated from transpose: distinct magic and key domain, and binds
/// the stencil IR and boundary versions and policy.
fn artifact_key(source: &[u8], contract: &Contract, plan: &super::Emission) -> [u8; 32] {
    let mut hash = Sha256::new();
    for field in [
        b"invar-private-stencil-host-v1".as_slice(),
        &stencil::VERSION.to_le_bytes(),
        &stencil::BOUNDARY_VERSION.to_le_bytes(),
        contract.boundary().name().as_bytes(),
        source,
        contract.function().as_bytes(),
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
    input: &[F32Bits],
    output_len: usize,
    limit: usize,
) -> Option<Vec<u8>> {
    let length = HEADER.checked_add(input.len().checked_mul(4)?)?;
    let response = HEADER
        .checked_add(output_len.checked_mul(4)?)?
        .checked_add(TRAILER)?;
    if length.checked_add(response)? > limit {
        return None;
    }
    let mut packet = Vec::new();
    packet.try_reserve_exact(length).ok()?;
    packet.extend_from_slice(MAGIC);
    for n in plan.plan().ir().dimensions {
        packet.extend_from_slice(&n.to_le_bytes());
    }
    packet.extend_from_slice(&boundary_code(contract.boundary()).to_le_bytes());
    for n in [input.len() as u64, output_len as u64, 0] {
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
        packet.extend_from_slice(&word.to_bits().to_le_bytes());
    }
    let digest = Sha256::digest(&packet[HEADER..]);
    packet[112..HEADER].copy_from_slice(&digest);
    Some(packet)
}

/// Accept only the echoed request header, exact length and a matching digest.
fn response(bytes: &[u8], packet: &[u8], count: usize) -> Option<Vec<F32Bits>> {
    let words = count.checked_mul(4)?;
    if bytes.len() != HEADER.checked_add(words)?.checked_add(TRAILER)?
        || bytes.get(..HEADER)? != packet.get(..HEADER)?
    {
        return None;
    }
    let payload = &bytes[HEADER..HEADER + words];
    if Sha256::digest(payload).as_slice() != &bytes[HEADER + words..] {
        return None;
    }
    let mut values = Vec::new();
    values.try_reserve_exact(count).ok()?;
    values.extend(
        payload
            .chunks_exact(4)
            .map(|w| F32Bits::from_bits(u32::from_le_bytes([w[0], w[1], w[2], w[3]]))),
    );
    Some(values)
}

/// Stencil contracts recognized in a checked source.
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
/// `Ok(None)` without recording anything when no stencil contract matches.
pub(crate) fn provide<D: Device>(
    source: &[u8],
    contracts: &[Rc<Contract>],
    runtime: &mut Runtime<D>,
    caps: Capabilities,
    staging_limit: usize,
    refusals: &mut Vec<&'static str>,
    live: &LiveLoop<'_>,
) -> Result<Option<LoopOverride>, EvalError> {
    let Some(contract) = contracts
        .iter()
        .find(|c| c.function() == live.function && c.loop_span() == live.span)
    else {
        return Ok(None);
    };
    // Unknown work anywhere in this runtime blocks dispatch and replay.
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
        refusals.push("stencil_fuel_overflow");
        return Ok(None);
    };
    if ticks > live.fuel_remaining {
        refusals.push("stencil_source_fuel");
        return Ok(None);
    }
    if live.same_floating(contract.input(), contract.output()) {
        refusals.push("stencil_alias");
        return Ok(None);
    }
    let (Some(input_view), Some(output_view)) = (
        live.floating(contract.input()),
        live.floating(contract.output()),
    ) else {
        refusals.push("stencil_source_publication_view_unavailable");
        return Ok(None);
    };
    let output_len = output_view.len();
    drop(output_view);
    let plan = match contract.emit_cuda(dimensions, input_view.len() as u64, caps) {
        Ok(p) => p,
        Err(_) => {
            // Includes short inputs: ordinary source then traps exactly.
            refusals.push("stencil_contract_or_layout");
            return Ok(None);
        }
    };
    if output_len as u64 != plan.plan().ir().output.elements {
        refusals.push("stencil_output_length");
        return Ok(None);
    }
    let Some(packet) = request(
        source,
        contract,
        &plan,
        &input_view,
        output_len,
        staging_limit,
    ) else {
        refusals.push("stencil_staging_limit");
        return Ok(None);
    };
    let input_length = input_view.len();
    drop(input_view);
    let prepared = match runtime.prepare(&packet) {
        Ok(p) => p,
        Err(_) => {
            refusals.push("stencil_prepare");
            return Ok(None);
        }
    };
    let mut submitted = match prepared.submit_classified() {
        Ok(s) => s,
        Err(SubmitError::InFlightUnknown(_)) => return Err(pending()),
        Err(SubmitError::FailedQuiescent(_)) => {
            refusals.push("stencil_submit");
            return Ok(None);
        }
    };
    let bytes = match submitted.poll() {
        Ok(Some(b)) => b,
        Ok(None) => return Err(pending()),
        Err(_) if !submitted.is_quiescent() => return Err(pending()),
        Err(_) => {
            refusals.push("stencil_read");
            return Ok(None);
        }
    };
    let Some(values) = response(&bytes, &packet, output_len) else {
        refusals.push("stencil_stale_or_malformed_output");
        return Ok(None);
    };
    Ok(Some(LoopOverride {
        iterations: live.iterations,
        stencil: Some(Completion {
            contract: contract.clone(),
            dimensions,
            input_length,
            values,
            ticks,
        }),
        ..Default::default()
    }))
}

/// Run `main` publishing reached stencil loops through `runtime`.
pub fn run_source<D: Device>(
    source: &[u8],
    fuel: u64,
    runtime: &mut Runtime<D>,
    caps: Capabilities,
) -> Result<Run, Error> {
    run_source_bounded(source, fuel, runtime, caps, STAGING_LIMIT)
}

/// As `run_source`, with an explicit per-ticket staging byte bound.
pub fn run_source_bounded<D: Device>(
    source: &[u8],
    fuel: u64,
    runtime: &mut Runtime<D>,
    caps: Capabilities,
    staging_limit: usize,
) -> Result<Run, Error> {
    let checked = crate::front::parse(source)
        .and_then(|p| crate::check::check(p, true))
        .map_err(Error::Source)?;
    let program =
        crate::mir::lower(&checked).map_err(|e| Error::Net(crate::general_net::Error::Mir(e)))?;
    let contracts = contracts(&checked);
    let mut refusals = Vec::new();
    let mut provider = |live: &LiveLoop<'_>| {
        provide(
            source,
            &contracts,
            runtime,
            caps,
            staging_limit,
            &mut refusals,
            live,
        )
    };
    let (observation, completed) =
        mir_eval::observe_with_scheduler(&program, fuel, Some(&mut provider));
    Ok(Run {
        observation,
        completed,
        refusals,
    })
}

/// Synchronous CPU executor of the verified stencil request: strict binary32
/// in source order `(((center + n) + s) + w) + e`. Never a GPU claim.
#[derive(Default)]
pub struct CpuDevice;
pub struct CpuWork {
    packet: Vec<u8>,
    dimensions: [usize; 2],
    clamp: bool,
    input_len: usize,
    output_len: usize,
    response: Option<Vec<u8>>,
}
impl Device for CpuDevice {
    type Work = CpuWork;
    fn prepare(&mut self, bytes: &[u8]) -> Result<CpuWork, invar_cuda::Error> {
        use invar_cuda::Error as E;
        if bytes.get(..4) != Some(MAGIC.as_slice()) || bytes.len() < HEADER {
            return Err(E::InvalidHandle);
        }
        let u32_at =
            |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
        let u64_at = |i: usize| {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[i..i + 8]);
            u64::from_le_bytes(b)
        };
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
            || Sha256::digest(&bytes[HEADER..]).as_slice() != &bytes[112..HEADER]
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
            clamp: u32_at(12) == 1,
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
        let words = work.output_len * 4;
        let mut response = Vec::new();
        response
            .try_reserve_exact(HEADER + words + TRAILER)
            .map_err(|_| SubmitError::FailedQuiescent(E::OutOfMemory))?;
        response.extend_from_slice(&work.packet[..HEADER]);
        let [rows, cols] = work.dimensions;
        let input = &work.packet[HEADER..];
        let at = |i: usize| {
            F32Bits::from_bits(u32::from_le_bytes([
                input[i * 4],
                input[i * 4 + 1],
                input[i * 4 + 2],
                input[i * 4 + 3],
            ]))
        };
        let zero = F32Bits::from_bits(0);
        for r in 0..rows {
            for c in 0..cols {
                let i = r * cols + c;
                if i >= work.input_len {
                    return Err(SubmitError::FailedQuiescent(E::InvalidHandle));
                }
                let center = at(i);
                let edge = if work.clamp { center } else { zero };
                let n = if r > 0 { at(i - cols) } else { edge };
                let s = if r + 1 < rows { at(i + cols) } else { edge };
                let w = if c > 0 { at(i - 1) } else { edge };
                let e = if c + 1 < cols { at(i + 1) } else { edge };
                let sum = center.add(n).add(s).add(w).add(e);
                response.extend_from_slice(&sum.to_bits().to_le_bytes());
            }
        }
        let digest = Sha256::digest(&response[HEADER..]);
        response.extend_from_slice(&digest);
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
    /// Bind the reached step, retaining the original source fuel and alias guard.
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

    /// One source call is one completed time slice, even during ping-pong calls.
    pub(crate) fn completed_region(
        &self,
        plan: &invar_device::stencil_u32::Verified,
        input: &[u32],
        output: Vec<u32>,
    ) -> Option<crate::mir_eval::region::CompletedRegion> {
        use crate::mir_eval::region::{Branch, CompletedRegion, Elements, Write};
        let dimensions = plan.ir().dimensions;
        let [rows, cols] = dimensions.map(u64::from);
        let cells = rows.checked_mul(cols)?;
        if self.element != crate::front::Type::ArrayU32
            || plan.ir().steps != 1
            || plan.ir().function_contract != self.function_contract
            || plan.ir().boundary != self.boundary
            || output.len() as u64 != cells
            || input.len() as u64 != cells
        {
            return None;
        }
        for (i, word) in output.iter().enumerate() {
            let reads = plan
                .reads([(i as u64 / cols) as u32, (i as u64 % cols) as u32])
                .ok()?;
            let expected = reads.into_iter().flatten().try_fold(0u32, |sum, at| {
                Some(sum.wrapping_add(*input.get(at as usize)?))
            })?;
            if *word != expected {
                return None;
            }
        }
        let mut branches = vec![
            Branch {
                span: self.loop_span,
                taken: rows,
                skipped: 1,
            },
            Branch {
                span: self.inner_span,
                taken: cells,
                skipped: rows,
            },
        ];
        for (span, taken) in self.branch_spans.into_iter().zip([
            rows.saturating_sub(1).checked_mul(cols)?,
            rows.saturating_sub(1).checked_mul(cols)?,
            rows.checked_mul(cols.saturating_sub(1))?,
            rows.checked_mul(cols.saturating_sub(1))?,
        ]) {
            branches.push(Branch {
                span,
                taken,
                skipped: cells - taken,
            });
        }
        Some(CompletedRegion {
            function: self.function.clone(),
            span: self.loop_span,
            branches,
            writes: vec![Write {
                owner: self.output.clone(),
                length: output.len(),
                range: 0..output.len(),
                values: Elements::U32(output),
            }],
            scalars: Vec::new(),
            cleanup: Vec::new(),
            fuel: self.loop_fuel(dimensions)?,
            trap: None,
        })
    }
}
