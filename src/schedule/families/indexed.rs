//! Indexed family: guarded private device work, then one completed MIR region.
use super::super::*;
use super::{Admitted, Live};
use crate::general_net::regions::family::Family;
use crate::indexed_device::SourcePlan;

pub(super) type Candidate = SourcePlan;

/// A PNF gather, scatter, affine scatter, histogram or frontier admission
/// binds the reached-source plan at its span; a plan of another family is no
/// lowering. The recognizer this replaces is the oracle in
/// `tests/support/family_oracle.rs`.
pub(super) fn lower(
    admitted: &Admitted,
    live: &Live<'_>,
) -> Result<Option<Candidate>, admission::Error> {
    use Family::{AffineScatter, Frontier, Gather, Histogram, Scatter};
    if !admitted.admits(&[Gather, Scatter, AffineScatter, Histogram, Frontier]) {
        return Ok(None);
    }
    let (function, span) = (live.region.function(), live.region.span());
    let plan = SourcePlan::recognize(live.source, live.program, function, span).map_err(|e| {
        admission::Error {
            code: "INDEXED_SOURCE_BINDING",
            message: format!("indexed source binding failed: {e:?}"),
        }
    })?;
    Ok(plan.filter(|plan| admitted.admits(&[plan.family()])))
}

pub(super) fn cpu(
    _candidate: &Candidate,
    _live: &LiveLoop<'_>,
    _pending: &mut VecDeque<Pending>,
    _workers: usize,
    _hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    Ok(None) // Keep the reached source loop in serial MIR.
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn is_consumer(_candidate: &Candidate) -> bool {
    false
}
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn is_producer(_candidate: &Candidate) -> bool {
    false
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(super) mod metal {
    use super::*;
    use crate::{
        device_cache::HintState,
        plan::{DeviceCost, DeviceCostModel},
        schedule::metal::{Error, Fault},
    };
    pub(crate) fn cost(
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        caps: invar_device::Capabilities,
        model: DeviceCostModel,
    ) -> Option<DeviceCost> {
        candidate.prepare(live, caps).ok()?.cost(model)
    }
    pub(crate) fn ineligible_reason(_candidate: &Candidate, _len: usize) -> &'static str {
        "indexed-live-guard-false"
    }
    /// The indexed kernel `step` compiles for the dimensions reached at `live`,
    /// so a built library holds it and a run needs no source compiler.
    pub(crate) fn kernels(
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        caps: invar_device::Capabilities,
    ) -> Vec<(String, Vec<String>)> {
        let Ok(prepared) = candidate.prepare(live, caps) else {
            return Vec::new();
        };
        let plan = prepared.bound.plan();
        let Ok(msl) = invar_device::indexed::emit_msl(plan) else {
            return Vec::new();
        };
        let names = invar_device::indexed::kernel_names(plan)
            .into_iter()
            .map(str::to_owned)
            .collect();
        vec![(msl, names)]
    }
    pub(crate) fn step(
        _source: &[u8],
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        _pending: &mut VecDeque<Pending>,
        device: &mut invar_metal::MetalDevice,
        fault: Fault,
    ) -> Result<(Step, bool, HintState), Error> {
        let prepared = candidate
            .prepare(live, device.capabilities())
            .map_err(|e| Error::unavailable(format!("indexed live guard: {e:?}")))?;
        let kernel = invar_device::indexed::emit_msl(prepared.bound.plan())
            .map_err(|e| Error::mismatch(format!("indexed emission: {e:?}")))?;
        let receipt = device
            .run_indexed_with_fault(&prepared.bound, fault)
            .map_err(|e| Error::from_metal(e, device.is_poisoned()))?;
        let completed = prepared
            .complete(receipt, "metal", invar_device::source_hash(&kernel))
            .map_err(|e| Error::unavailable(format!("indexed completion: {e:?}")))?;
        Ok((
            Step {
                plan: LoopOverride::completed(completed),
                kind: PlanKind::Indexed,
                chunks: vec![],
                joined: 0,
            },
            false,
            HintState::Miss,
        ))
    }
}

#[cfg(feature = "cuda")]
pub(super) mod cuda {
    use super::*;
    use crate::{
        device_cache::HintState,
        plan::{DeviceCost, DeviceCostModel},
        schedule::cuda::{Error, Fault},
    };
    pub(crate) fn cost(
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        caps: invar_device::Capabilities,
        model: DeviceCostModel,
    ) -> Option<DeviceCost> {
        candidate.prepare(live, caps).ok()?.cost(model)
    }
    pub(crate) fn step(
        _source: &[u8],
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        _pending: &mut VecDeque<Pending>,
        device: &mut invar_cuda::cuda::CudaDevice,
        compiler: &std::path::Path,
        fault: Fault,
    ) -> Result<(Step, bool, HintState), Error> {
        let prepared = candidate
            .prepare(live, device.capabilities())
            .map_err(|e| Error::unavailable(format!("indexed live guard: {e:?}")))?;
        let kernel = invar_device::indexed::emit_cuda(prepared.bound.plan())
            .map_err(|e| Error::mismatch(format!("indexed emission: {e:?}")))?;
        let receipt = device
            .run_indexed_with_fault(&prepared.bound, compiler, fault.collective())
            .map_err(|e| Error::from_cuda(e, device.is_poisoned()))?;
        let completed = prepared
            .complete(receipt, "cuda", invar_device::source_hash(&kernel))
            .map_err(|e| Error::unavailable(format!("indexed completion: {e:?}")))?;
        Ok((
            Step {
                plan: LoopOverride::completed(completed),
                kind: PlanKind::Indexed,
                chunks: vec![],
                joined: 0,
            },
            false,
            HintState::Miss,
        ))
    }
}
