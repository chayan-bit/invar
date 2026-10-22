//! Reached v0.2 float reductions on CUDA: forced runs and cost-model
//! placement, each recorded as a receipt. Device partials are finished on the
//! host by the same code as the CPU engines, so every backend gives one result.

use std::path::Path;

use invar_cuda::cuda::CudaDevice;

use super::{Error, EvalError};
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
fn device_reduce(
    call: &ReduceCall<'_>,
    device: &mut CudaDevice,
    compiler: &Path,
) -> Result<(F32Bits, f64), Error> {
    let a = reductions::words(call.a);
    let b = call.b.map(reductions::words);
    let op = reductions::device_op(call.kind);
    let result = device
        .run_fsum(op, &a, b.as_deref(), reductions::LEAF_SHIFT, compiler)
        .map_err(|error| Error::from_cuda(error, device.is_poisoned()))?;
    let bits = reductions::finish(call, &result.nodes, &result.rows);
    Ok((bits, f64::from(result.kernel_ms)))
}

/// `--engine cuda`: every nonempty reduction runs on the device or fails typed.
pub(crate) fn forced(
    call: &ReduceCall<'_>,
    device: &mut CudaDevice,
    compiler: &Path,
    receipts: &mut Vec<Reduction>,
) -> Result<Option<F32Bits>, EvalError> {
    if call.a.is_empty() {
        receipts.push(Reduction::new(call, "mir-serial", "device-plan-ineligible"));
        return Ok(None);
    }
    let (bits, kernel_ms) = device_reduce(call, device, compiler).map_err(tool)?;
    let mut receipt = Reduction::new(call, "cuda", "forced-device");
    receipt.kernel_ms = Some(kernel_ms);
    receipts.push(receipt);
    Ok(Some(bits))
}

/// `--engine auto`: the calibrated cost model picks CUDA or CPU workers.
/// A device failure before publication falls back to the CPU result.
pub(crate) struct Auto<'a> {
    pub compiler: &'a Path,
    pub model: Option<DeviceCostModel>,
    pub workers: usize,
}

impl Auto<'_> {
    pub(crate) fn reduce(
        &self,
        call: &ReduceCall<'_>,
        device: Option<&mut CudaDevice>,
        receipts: &mut Vec<Reduction>,
    ) -> Result<Option<F32Bits>, EvalError> {
        let cost = self
            .model
            .filter(|_| device.is_some())
            .map(|model| reductions::cost(call, model));
        let mut reason = match (&device, cost) {
            (None, _) => "device-unavailable",
            _ if !self.compiler.is_file() => "toolchain-unavailable",
            _ if call.a.is_empty() => "device-plan-ineligible",
            (_, Some(cost)) if cost.choose_device => "measured-device-cost",
            _ => "measured-cpu-cost",
        };
        if let (Some(device), "measured-device-cost") = (device, reason) {
            match device_reduce(call, device, self.compiler) {
                Ok((bits, kernel_ms)) => {
                    let mut receipt = Reduction::new(call, "cuda", reason);
                    (receipt.cost, receipt.kernel_ms) = (cost, Some(kernel_ms));
                    receipts.push(receipt);
                    return Ok(Some(bits));
                }
                Err(error) if error.code == "CUDA_PENDING" => return Err(tool(error)),
                Err(_) => reason = "cuda-failed-before-publication",
            }
        }
        let bits = reductions::cpu(call, self.workers).map_err(|error| EvalError::Tool {
            code: error.code(),
            message: format!("CPU execution failed: {error:?}"),
        })?;
        let mut receipt = reductions::cpu_receipt(call, self.workers, reason);
        receipt.cost = cost;
        receipts.push(receipt);
        Ok(Some(bits))
    }
}
