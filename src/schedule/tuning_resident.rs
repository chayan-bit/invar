//! Placement over real retained map buffers. These raw results are not MIR
//! publication permits; ordinary source still uses its reached-loop guards.
use super::tuning::*;
use crate::{admission::Region, plan::Execution};
use invar_device::Verified;

pub enum Output<T> {
    Host(Vec<Vec<u32>>),
    Resident(T),
}
pub struct Run<T> {
    pub decision: Decision,
    /// The selection record for the resident API (inputs and output on the
    /// device). A CPU decision never submits device work, so its record stays
    /// `NotSubmitted`; it is the basis of the choice, not an account of the
    /// CPU execution, whose result is `Output::Host`.
    pub cost: CostRecord,
    pub output: Output<T>,
    /// One raw resident event, not a qualified steady-state boundary. `None`
    /// when no device ran, or when the device clock gave no interval for work
    /// that did complete.
    pub kernel_ms: Option<f64>,
}

fn validate(region: &Region, plan: &Verified) -> Result<(), Refusal> {
    let mut ir =
        crate::device_ir::from_region(region, plan.ir().length).map_err(|_| Refusal::Admission)?;
    ir.geometry = plan.ir().geometry;
    if &ir != plan.ir() {
        return Err(Refusal::Admission);
    }
    Ok(())
}

fn cpu(region: &Region, plan: &Verified, inputs: &[Vec<u32>]) -> Result<Vec<Vec<u32>>, Refusal> {
    validate(region, plan)?;
    let slices = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let n = plan.ir().length as usize;
    match region.execution() {
        Execution::Map(map) => map.plan.run_live(&slices, n).map(|a| vec![a]),
        Execution::DagProducer(chain) => chain.plan.run_dag(&slices, 0..n),
        Execution::FusedProducer(pair) => {
            let (captured, values) = pair
                .plan
                .live_values(
                    &slices,
                    0..n,
                    (!pair.elide_intermediate).then_some(pair.capture),
                )
                .map_err(|_| Refusal::Admission)?;
            return Ok(if pair.elide_intermediate {
                vec![values]
            } else {
                vec![captured, values]
            });
        }
        _ => return Err(Refusal::UnsupportedSchedule),
    }
    .map_err(|_| Refusal::Admission)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn metal(
    region: &Region,
    device: &mut invar_metal::MetalDevice,
    mut map: invar_metal::ResidentMap,
    model: crate::plan::DeviceCostModel,
    control: Control,
) -> Result<Run<invar_metal::ResidentMap>, Refusal> {
    device.reap();
    if device.is_poisoned() {
        return Err(Refusal::PartialCompletion);
    }
    validate(region, map.plan())?;
    let (decision, mut cost) = select_map(map.plan(), Residency::DEVICE, model, control, 0)?;
    let mut kernel_ms = None;
    let output = if decision.engine == Engine::Gpu {
        let ms = device.run_resident_map(&mut map).map_err(|_| {
            if device.is_poisoned() {
                Refusal::PartialCompletion
            } else {
                Refusal::Launch
            }
        })?;
        cost.completion = Completion::Completed;
        kernel_ms = ms;
        Output::Resident(map)
    } else {
        Output::Host(cpu(region, map.plan(), &map.inputs())?)
    };
    Ok(Run {
        decision,
        cost,
        output,
        kernel_ms,
    })
}

#[cfg(feature = "cuda")]
pub fn cuda(
    region: &Region,
    device: &mut invar_cuda::cuda::CudaDevice,
    map: invar_cuda::cuda::ResidentMap,
    model: crate::plan::DeviceCostModel,
    control: Control,
) -> Result<Run<invar_cuda::cuda::ResidentMap>, Refusal> {
    device.reap();
    if device.is_poisoned() {
        return Err(Refusal::PartialCompletion);
    }
    validate(region, map.plan())?;
    let (decision, mut cost) = select_map(map.plan(), Residency::DEVICE, model, control, 0)?;
    let mut kernel_ms = None;
    let output = if decision.engine == Engine::Gpu {
        let (map, ms) = device.run_resident_map(map).map_err(|_| {
            if device.is_poisoned() {
                Refusal::PartialCompletion
            } else {
                Refusal::Launch
            }
        })?;
        cost.completion = Completion::Completed;
        kernel_ms = ms;
        Output::Resident(map)
    } else {
        let (map, inputs) = device.read_resident_inputs(map).map_err(|_| {
            if device.is_poisoned() {
                Refusal::PartialCompletion
            } else {
                Refusal::Transfer
            }
        })?;
        Output::Host(cpu(region, map.plan(), &inputs)?)
    };
    Ok(Run {
        decision,
        cost,
        output,
        kernel_ms,
    })
}
