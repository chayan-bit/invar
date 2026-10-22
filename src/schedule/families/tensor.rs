//! Source-bound matrix contractions. Only fenced, privately staged output
//! becomes a region; strict source never acquires a Tensor Core policy.
use super::super::*;
use crate::{contraction_source, tensor};

pub(super) struct Candidate {
    #[cfg(any(test, feature = "cuda", all(feature = "metal", target_os = "macos")))]
    source: Vec<u8>,
    #[cfg_attr(
        not(any(test, feature = "cuda", all(feature = "metal", target_os = "macos"))),
        expect(dead_code, reason = "tensor CPU placement retains serial MIR")
    )]
    contract: Contract,
}

#[cfg_attr(
    not(any(test, feature = "cuda", all(feature = "metal", target_os = "macos"))),
    expect(dead_code, reason = "tensor CPU placement retains serial MIR")
)]
enum Contract {
    Scalar(Box<tensor::Contract>),
    Direct(Box<contraction_source::Contract>),
}

/// A PNF tensor admission binds the scalar or direct-output contraction at
/// its span. The recognizer this replaces is the oracle in
/// `tests/support/family_oracle.rs`.
pub(super) fn lower(
    admitted: &lowering::Admitted,
    live: &lowering::Live<'_>,
) -> Result<Option<Candidate>, admission::Error> {
    if !admitted.admits(&[crate::general_net::regions::family::Family::Tensor]) {
        return Ok(None);
    }
    bind(live.source, live.program, live.region)
}

fn bind(
    source: &[u8],
    program: &mir::Program,
    region: &Region,
) -> Result<Option<Candidate>, admission::Error> {
    let mismatch = |message: String| admission::Error {
        code: "ADMISSION_STALE",
        message,
    };
    let contract = if let Some(contract) = program
        .tensor_contracts
        .iter()
        .find(|c| c.function() == region.function() && c.loop_span() == region.span())
    {
        let checked = crate::front::parse(source)
            .and_then(|p| crate::check::check(p, false))
            .map_err(|e| mismatch(format!("tensor source: {e:?}")))?;
        let actual = tensor::Contraction::from_checked(&checked, region.function())
            .map_err(|e| mismatch(format!("tensor extraction: {e:?}")))?;
        if actual.contract() != contract
            || actual
                .compiled_mir()
                .map_err(|e| mismatch(format!("tensor MIR: {e:?}")))?
                .contract_digest()
                != program.contract_digest()
        {
            return Err(mismatch("tensor source/MIR mismatch".into()));
        }
        Contract::Scalar(Box::new(contract.clone()))
    } else {
        if !contraction_source::matches_loop(source, region.function(), region.span()) {
            return Ok(None);
        }
        match contraction_source::Contract::from_source(source, region.function()) {
            Ok(c) if c.loop_span() == region.span() => {
                c.verify_mir(program).map_err(|e| mismatch(e.to_string()))?;
                Contract::Direct(Box::new(c))
            }
            Ok(_) | Err(contraction_source::Error::Unrecognized) => return Ok(None),
            Err(e) => return Err(mismatch(e.to_string())),
        }
    };
    Ok(Some(Candidate {
        #[cfg(any(test, feature = "cuda", all(feature = "metal", target_os = "macos")))]
        source: source.to_vec(),
        contract,
    }))
}

pub(super) fn cpu(
    _candidate: &Candidate,
    _live: &LiveLoop<'_>,
    _pending: &mut VecDeque<Pending>,
    _workers: usize,
    _hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    // MIR is the exact fallback, including strict nonfinite/underflow and fuel.
    Ok(None)
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn is_consumer(_candidate: &Candidate) -> bool {
    false
}
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn is_producer(_candidate: &Candidate) -> bool {
    false
}

#[cfg(any(test, feature = "cuda", all(feature = "metal", target_os = "macos")))]
mod reached {
    use super::*;
    use crate::{
        mir_eval::region::{Branch, CompletedRegion, Elements, Write},
        numeric_contract::{F32Bits, NumericPolicy},
        tensor::host::{packet, request_packet, response},
    };
    use invar_cuda::{Device, Runtime, SubmitError};
    use invar_device::{Capabilities, floating};

    pub(super) struct Bound<'a> {
        pub plan: floating::Verified,
        key: crate::tensor_cache::CacheKey,
        source_hash: u64,
        // Keep the actual live read loans through completion, without sharing
        // source pointers with a device. C is borrowed only for its guard.
        _inputs: [std::cell::Ref<'a, Vec<F32Bits>>; 2],
        _output: std::cell::Ref<'a, Vec<F32Bits>>,
        pub(super) region: CompletedRegion,
    }

    impl Candidate {
        pub(super) fn bind<'a>(
            &self,
            live: &LiveLoop<'a>,
            caps: Capabilities,
        ) -> Option<Bound<'a>> {
            let (function, spans, names, dimensions, output, policy) = match &self.contract {
                Contract::Scalar(c) => (
                    c.function(),
                    c.loop_spans(),
                    c.inputs(),
                    c.dimensions(),
                    c.output(),
                    c.numeric_policy(),
                ),
                Contract::Direct(c) => (
                    c.function(),
                    c.loop_spans(),
                    c.inputs(),
                    c.dimensions(),
                    c.output(),
                    c.policy(),
                ),
            };
            if function != live.function || spans[0] != live.span {
                return None;
            }
            let [Some(rows), Some(inner), Some(cols)] =
                dimensions.each_ref().map(|n| live.scalars.get(n).copied())
            else {
                return None;
            };
            // An empty loop nest needs no device, even when the outer i loop
            // itself has trips. Its source control flow remains MIR's.
            if rows == 0 || cols == 0 || inner == 0 || live.iterations != rows as usize {
                return None;
            }
            let dims = [rows, inner, cols];
            let fuel = match &self.contract {
                Contract::Scalar(c) => c.loop_fuel(dims),
                Contract::Direct(c) => c.loop_fuel(dims),
            }?;
            if fuel > live.fuel_remaining || names.iter().any(|n| live.same_floating(output, n)) {
                return None;
            }
            let a = live.floating(&names[0])?;
            let b = live.floating(&names[1])?;
            let c = live.floating(output)?;
            if a.iter()
                .chain(b.iter())
                .any(|v| v.is_nan() || v.is_infinite())
                || c.iter().any(|v| *v != F32Bits::POS_ZERO)
            {
                return None;
            }
            let lengths = [a.len() as u64, b.len() as u64];
            let (plan, key, source_hash) = match &self.contract {
                Contract::Scalar(_) => {
                    let e = tensor::emit(&self.source, function, dims, lengths, caps).ok()?;
                    let source_hash = invar_device::source_hash(e.source());
                    let (plan, key) = e.into_plan_key();
                    (plan, key, source_hash)
                }
                Contract::Direct(contract) => {
                    let e = contract
                        .emit_device(&self.source, dims, lengths, caps)
                        .ok()?;
                    let source_hash = invar_device::source_hash(e.source());
                    let (plan, key) = e.into_plan_key();
                    (plan, key, source_hash)
                }
            };
            // The explicit source contract is the only gate to relaxed math.
            if plan.is_tc() != (policy == NumericPolicy::TC_F16_F32)
                || plan.ir().numeric_policy != policy.id()
            {
                return None;
            }
            let cells = u64::from(rows).checked_mul(u64::from(cols))?;
            let products = cells.checked_mul(u64::from(inner))?;
            let count = usize::try_from(cells).ok()?;
            if c.len() != count || plan.ir().output.elements != cells {
                return None;
            }

            let region = CompletedRegion {
                function: function.to_owned(),
                span: spans[0],
                branches: vec![
                    Branch {
                        span: spans[0],
                        taken: u64::from(rows),
                        skipped: 1,
                    },
                    Branch {
                        span: spans[1],
                        taken: cells,
                        skipped: u64::from(rows),
                    },
                    Branch {
                        span: spans[2],
                        taken: products,
                        skipped: cells,
                    },
                ],
                writes: vec![Write {
                    owner: output.to_owned(),
                    length: count,
                    range: 0..count,
                    values: Elements::F32(Vec::new()),
                }],
                scalars: Vec::new(),
                cleanup: Vec::new(),
                fuel,
                trap: None,
            };
            Some(Bound {
                plan,
                key,
                source_hash,
                _inputs: [a, b],
                _output: c,
                region,
            })
        }
    }

    impl Bound<'_> {
        pub(super) fn completed_words(
            &self,
            source_hash: u64,
            physical: &[u32],
        ) -> Result<Vec<u32>, invar_cuda::Error> {
            if source_hash != self.source_hash {
                return Err(invar_cuda::Error::StaleBorrow);
            }
            packet::logical_output(&self.plan, physical)
        }
    }

    /// The same private packet and poll boundary as the family host, over an
    /// already-open scheduler device. No whole-program runner is called.
    pub(super) struct Dispatch<F>(pub F);
    pub(super) struct Work {
        packet: Vec<u8>,
        inputs: [Vec<u32>; 2],
        response: Option<Vec<u8>>,
    }
    impl<F: FnMut([&[u32]; 2]) -> Result<Vec<u32>, SubmitError>> Device for Dispatch<F> {
        type Work = Work;
        fn prepare(&mut self, bytes: &[u8]) -> Result<Work, invar_cuda::Error> {
            let (_, [a, b, _]) = packet::packet_shape(bytes)?;
            let mut packet = Vec::new();
            packet
                .try_reserve_exact(bytes.len())
                .map_err(|_| invar_cuda::Error::OutOfMemory)?;
            packet.extend_from_slice(bytes);
            Ok(Work {
                packet,
                inputs: packet::packet_inputs(bytes, [a, b])?,
                response: None,
            })
        }
        fn submit(&mut self, work: &mut Work) -> Result<(), SubmitError> {
            if work.response.is_some() {
                return Err(SubmitError::FailedQuiescent(
                    invar_cuda::Error::InvalidHandle,
                ));
            }
            let words = (self.0)([&work.inputs[0], &work.inputs[1]])?;
            work.response = Some(
                packet::response_packet(&work.packet, &words)
                    .map_err(SubmitError::FailedQuiescent)?,
            );
            Ok(())
        }
        fn complete(&mut self, work: &mut Work) -> Result<bool, invar_cuda::Error> {
            Ok(work.response.is_some())
        }
        fn read(&mut self, work: &mut Work) -> Result<Vec<u8>, invar_cuda::Error> {
            work.response.take().ok_or(invar_cuda::Error::InvalidHandle)
        }
    }

    pub(super) fn finish<D: Device>(
        bound: &Bound<'_>,
        runtime: &mut Runtime<D>,
    ) -> Result<Step, SubmitError> {
        if runtime.blocks_dispatch() {
            return Err(SubmitError::InFlightUnknown(invar_cuda::Error::Poisoned));
        }
        let packet = request_packet(
            &bound.key,
            bound.plan.ir().dimensions,
            bound.plan.ir().output.elements,
            [&bound._inputs[0], &bound._inputs[1]],
        )
        .ok_or(SubmitError::FailedQuiescent(invar_cuda::Error::OutOfMemory))?;
        let prepared = runtime
            .prepare(&packet)
            .map_err(SubmitError::FailedQuiescent)?;
        let mut submitted = prepared.submit_classified()?;
        let bytes = match submitted.poll() {
            Ok(Some(bytes)) => bytes,
            // A pending command never falls through to serial replay.
            _ if !submitted.is_quiescent() => {
                return Err(SubmitError::InFlightUnknown(invar_cuda::Error::Poisoned));
            }
            _ => {
                return Err(SubmitError::FailedQuiescent(
                    invar_cuda::Error::InvalidHandle,
                ));
            }
        };
        let values = response(&bytes, &packet, bound.region.writes[0].length).ok_or(
            SubmitError::FailedQuiescent(invar_cuda::Error::InvalidHandle),
        )?;
        let mut region = bound.region.clone();
        region.writes[0].values = Elements::F32(values);
        Ok(Step {
            plan: LoopOverride::completed(region),
            kind: PlanKind::Tensor,
            chunks: Vec::new(),
            joined: 0,
        })
    }

    pub(super) fn cost(
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        caps: Capabilities,
        model: crate::plan::DeviceCostModel,
    ) -> Option<crate::plan::DeviceCost> {
        let bound = candidate.bind(live, caps)?;
        let [r, k, n] = bound.plan.ir().dimensions.map(|v| v as usize);
        let transfer = bound
            .plan
            .ir()
            .inputs
            .iter()
            // Both owning runtimes initialize private C and read it back.
            .chain([&bound.plan.ir().output, &bound.plan.ir().output])
            .try_fold(0usize, |s, b| {
                s.checked_add(usize::try_from(b.elements).ok()?.checked_mul(4)?)
            })?;
        let work = r.checked_mul(k)?.checked_mul(n)?;
        Some(model.estimate(transfer, work, 2, 1))
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(super) mod metal {
    use super::{reached::*, *};
    use crate::{
        device_cache::HintState,
        schedule::metal::{Error, Fault},
    };
    use invar_cuda::{Runtime, SubmitError};
    pub(in crate::schedule::families) fn cost(
        c: &Candidate,
        l: &LiveLoop<'_>,
        caps: invar_device::Capabilities,
        model: crate::plan::DeviceCostModel,
    ) -> Option<crate::plan::DeviceCost> {
        reached::cost(c, l, caps, model)
    }
    pub(in crate::schedule::families) fn ineligible_reason(
        _c: &Candidate,
        _len: usize,
    ) -> &'static str {
        "tensor-live-guard-false"
    }
    /// The D1 probe and the contraction kernel `step` loads: tiled (or Tensor
    /// Core) for a direct contract, the scalar schedule otherwise.
    pub(in crate::schedule::families) fn kernels(
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        caps: invar_device::Capabilities,
    ) -> Vec<(String, Vec<String>)> {
        use invar_device::{f32_probe, floating, gemm, tc};
        let Some(bound) = candidate.bind(live, caps) else {
            return Vec::new();
        };
        let plan = &bound.plan;
        let msl = match &candidate.contract {
            Contract::Direct(_) if plan.is_tc() => tc::emit_msl(plan).ok(),
            Contract::Direct(_) => gemm::emit_msl(plan).ok(),
            _ => floating::emit_msl(plan).ok(),
        };
        let probe = (f32_probe::emit_msl(), vec![f32_probe::KERNEL_NAME.to_owned()]);
        msl.map(|msl| vec![probe, (msl, vec![floating::KERNEL_NAME.to_owned()])])
            .unwrap_or_default()
    }
    pub(in crate::schedule::families) fn step(
        source: &[u8],
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        _pending: &mut VecDeque<Pending>,
        device: &mut invar_metal::MetalDevice,
        fault: Fault,
    ) -> Result<(Step, bool, HintState), Error> {
        if source != candidate.source {
            return Err(Error::mismatch("tensor source mismatch"));
        }
        let bound = candidate
            .bind(live, device.capabilities())
            .ok_or_else(|| Error::unavailable("tensor live guard is false"))?;
        if matches!(fault, Fault::OutOfMemory | Fault::LaunchFailure) {
            return Err(Error::unavailable(format!(
                "tensor pre-submission fault: {fault:?}"
            )));
        }
        let plan = &bound.plan;
        let tiled = matches!(&candidate.contract, Contract::Direct(_));
        let mut runtime = Runtime::new(
            Dispatch(|inputs: [&[u32]; 2]| {
                let mut result = (if tiled {
                    device.run_tiled_contraction(plan, inputs)
                } else {
                    device.run_contraction(plan, inputs)
                })
                .map_err(|error| {
                    if device.is_poisoned() {
                        SubmitError::InFlightUnknown(invar_cuda::Error::Poisoned)
                    } else {
                        let error = match error {
                            invar_metal::Error::Toolchain(message) => {
                                invar_cuda::Error::Compile(message)
                            }
                            invar_metal::Error::SubnormalFallback => {
                                invar_cuda::Error::SubnormalFallback
                            }
                            invar_metal::Error::InvalidGuard(error) => {
                                invar_cuda::Error::InvalidGuard(error)
                            }
                            invar_metal::Error::InvalidProbe(error) => {
                                invar_cuda::Error::InvalidProbe(error)
                            }
                            error => invar_cuda::Error::Driver(format!("{error:?}")),
                        };
                        SubmitError::FailedQuiescent(error)
                    }
                })?;
                // The owning runtime has confirmed quiescence before this
                // fault corrupts the private completion, never live owners.
                if fault == Fault::PartialCompletion {
                    result.output.pop();
                }
                bound
                    .completed_words(result.source_hash, &result.output)
                    .map_err(SubmitError::FailedQuiescent)
            }),
            1,
        );
        let result = finish(&bound, &mut runtime).map_err(|error| match error {
            SubmitError::InFlightUnknown(_) => Error::from_metal(invar_metal::Error::Pending, true),
            SubmitError::FailedQuiescent(invar_cuda::Error::Compile(message)) => {
                Error::mismatch(message)
            }
            SubmitError::FailedQuiescent(error) => {
                Error::unavailable(format!("tensor completion refused: {error:?}"))
            }
        })?;
        Ok((result, false, HintState::Miss))
    }
}

#[cfg(feature = "cuda")]
pub(super) mod cuda {
    use super::{reached::*, *};
    use crate::{
        device_cache::HintState,
        schedule::cuda::{Error, Fault},
    };
    use invar_cuda::{Runtime, SubmitError};
    pub(in crate::schedule::families) fn cost(
        c: &Candidate,
        l: &LiveLoop<'_>,
        caps: invar_device::Capabilities,
        model: crate::plan::DeviceCostModel,
    ) -> Option<crate::plan::DeviceCost> {
        reached::cost(c, l, caps, model)
    }
    #[allow(clippy::too_many_arguments)] // Registry signature includes the CUDA compiler.
    pub(in crate::schedule::families) fn step(
        source: &[u8],
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        _pending: &mut VecDeque<Pending>,
        device: &mut invar_cuda::cuda::CudaDevice,
        compiler: &std::path::Path,
        fault: Fault,
    ) -> Result<(Step, bool, HintState), Error> {
        if source != candidate.source {
            return Err(Error::mismatch("tensor source mismatch"));
        }
        let bound = candidate
            .bind(live, device.capabilities())
            .ok_or_else(|| Error::unavailable("tensor live guard is false"))?;
        if matches!(fault, Fault::OutOfMemory | Fault::LaunchFailure) {
            return Err(Error::unavailable(format!(
                "tensor pre-submission fault: {fault:?}"
            )));
        }
        let plan = &bound.plan;
        let tiled = matches!(&candidate.contract, Contract::Direct(_));
        let mut runtime = Runtime::new(
            Dispatch(|inputs: [&[u32]; 2]| {
                let mut result = (if tiled {
                    device.run_tiled_contraction(plan, inputs, compiler)
                } else {
                    device.run_contraction(plan, inputs, compiler)
                })
                .map_err(|error| {
                    if device.is_poisoned() {
                        SubmitError::InFlightUnknown(error)
                    } else {
                        SubmitError::FailedQuiescent(error)
                    }
                })?;
                // The owning runtime has confirmed quiescence before this
                // fault corrupts the private completion, never live owners.
                if fault == Fault::PartialCompletion {
                    result.output.pop();
                }
                bound
                    .completed_words(result.source_hash, &result.output)
                    .map_err(SubmitError::FailedQuiescent)
            }),
            1,
        );
        let result = finish(&bound, &mut runtime).map_err(|error| match error {
            SubmitError::InFlightUnknown(error) => Error::from_cuda(error, true),
            SubmitError::FailedQuiescent(error) => Error::from_cuda(error, false),
        })?;
        Ok((result, false, HintState::Miss))
    }
}

#[cfg(test)]
mod tests;
