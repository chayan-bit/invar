//! Reached v0.2 float reductions on Metal: forced runs and cost-model
//! placement, each recorded as a receipt. Device partials are finished on the
//! host by the same code as the CPU engines, so every backend gives one result.

use invar_metal::MetalDevice;

use super::{Error, EvalError, Kernels};
use crate::{
    mir_eval::ReduceCall,
    numeric_contract::F32Bits,
    plan::DeviceCostModel,
    schedule::reductions::{self, Reduction},
};

fn tool(error: Error) -> EvalError {
    EvalError::Tool {
        code: error.code,
        message: error.message,
    }
}

/// One reduction on the device: the result bits and the GPU time.
pub(super) fn device_reduce(
    call: &ReduceCall<'_>,
    device: &mut MetalDevice,
) -> Result<(F32Bits, f64), invar_metal::Error> {
    let a = reductions::words(call.a);
    let b = call.b.map(reductions::words);
    let op = reductions::device_op(call.kind);
    let result = device.run_fsum(op, &a, b.as_deref(), reductions::LEAF_SHIFT)?;
    let bits = reductions::finish(call, &result.nodes, &result.rows);
    Ok((bits, result.kernel_ms))
}

fn on_cpu(
    call: &ReduceCall<'_>,
    workers: usize,
    reason: &'static str,
    receipts: &mut Vec<Reduction>,
) -> Result<F32Bits, EvalError> {
    let bits = reductions::cpu(call, workers).map_err(|error| EvalError::Tool {
        code: error.code(),
        message: format!("CPU execution failed: {error:?}"),
    })?;
    receipts.push(reductions::cpu_receipt(call, workers, reason));
    Ok(bits)
}

/// `--engine metal`: every nonempty reduction runs on the device or fails
/// typed. When the subnormal guard discards the device words, the CPU
/// workers produce the result and the receipt says so.
pub(crate) fn forced(
    call: &ReduceCall<'_>,
    device: &mut MetalDevice,
    workers: usize,
    receipts: &mut Vec<Reduction>,
) -> Result<Option<F32Bits>, EvalError> {
    if call.a.is_empty() {
        receipts.push(Reduction::new(call, "mir-serial", "device-plan-ineligible"));
        return Ok(None);
    }
    match device_reduce(call, device) {
        Ok((bits, kernel_ms)) => {
            let mut receipt = Reduction::new(call, "metal", "forced-device");
            receipt.kernel_ms = Some(kernel_ms);
            receipts.push(receipt);
            Ok(Some(bits))
        }
        Err(invar_metal::Error::SubnormalFallback) => {
            on_cpu(call, workers, "metal-failed-before-publication", receipts).map(Some)
        }
        Err(error) => Err(tool(Error::from_metal(error, device.is_poisoned()))),
    }
}

/// `--engine auto`: the calibrated cost model picks Metal or CPU workers.
/// A device failure before publication falls back to the CPU result.
pub(crate) fn auto(
    call: &ReduceCall<'_>,
    device: Option<&mut MetalDevice>,
    model: Option<DeviceCostModel>,
    workers: usize,
    receipts: &mut Vec<Reduction>,
) -> Result<Option<F32Bits>, EvalError> {
    let cost = model
        .filter(|_| device.is_some())
        .map(|model| reductions::cost(call, model));
    let mut reason = match (&device, cost) {
        (None, _) => "device-unavailable",
        _ if call.a.is_empty() => "device-plan-ineligible",
        // Below the measured crossover Metal loses even when the bandwidth
        // model prefers it, so the call stays on the CPU.
        (_, Some(cost))
            if cost.choose_device
                && call.a.len() >= reductions::calibration(call.kind).metal_min_terms =>
        {
            "measured-device-cost"
        }
        _ => "measured-cpu-cost",
    };
    if let (Some(device), "measured-device-cost") = (device, reason) {
        match device_reduce(call, device) {
            Ok((bits, kernel_ms)) => {
                let mut receipt = Reduction::new(call, "metal", reason);
                (receipt.cost, receipt.kernel_ms) = (cost, Some(kernel_ms));
                receipts.push(receipt);
                return Ok(Some(bits));
            }
            Err(error) if device.is_poisoned() => {
                return Err(tool(Error::from_metal(error, true)));
            }
            Err(_) => reason = "metal-failed-before-publication",
        }
    }
    let bits = on_cpu(call, workers, reason, receipts)?;
    if let Some(receipt) = receipts.last_mut() {
        receipt.cost = cost;
    }
    Ok(Some(bits))
}

/// Workers for a forced run's CPU fallback, as `--engine metal` picks them.
pub(crate) fn workers() -> Result<usize, Error> {
    super::selected_workers()
}

/// Auto's cost model (`model`, else a cached calibration) and its device,
/// opened on every run whatever calibration cached: a refused library or an
/// unconfirmed completion is an error; only a missing device or library
/// leaves auto on the CPU.
pub(super) fn auto_device(
    source: &[u8],
    kernels: Kernels<'_>,
    model: Option<DeviceCostModel>,
    workers: usize,
) -> Result<(Option<DeviceCostModel>, Option<MetalDevice>), Error> {
    let open = &mut || super::open_device(source, kernels);
    let calibration = match model {
        Some(model) => Some(model),
        None => match super::calibrate_with_workers(workers, open) {
            Ok(model) => Some(model),
            Err(error) if error.code == "METAL_PENDING" => return Err(error),
            // A supplied library that is refused is an error, not a reason to
            // run on the CPU; only an absent device or library degrades.
            Err(error) if error.code == "DEVICE_TOOLCHAIN_MISMATCH" => return Err(error),
            Err(_) => None,
        },
    };
    let device = match calibration.map(|_| super::open_device(source, kernels)) {
        Some(Ok(device)) => Some(device),
        Some(Err(error)) if matches!(error.code, "METAL_PENDING" | "DEVICE_TOOLCHAIN_MISMATCH") => {
            return Err(error);
        }
        Some(Err(_)) | None => None,
    };
    Ok((calibration, device))
}
