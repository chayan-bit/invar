//! CUDA source bridge. Mapping, stable radix passes, boundaries and query rows
//! execute on the device; immutable source owners are published after validation.
use crate::{
    eval::{self, EvalError},
    plan::PlanKind,
    schedule::{self, Executed},
};
use invar_cuda::cuda::CudaDevice;
use std::{
    cell::{Cell, RefCell},
    path::Path,
};

pub(crate) fn run(
    source: &[u8],
    program: &crate::mir::Program,
    fuel: u64,
    device: &mut CudaDevice,
    compiler: &Path,
    fault: schedule::cuda::Fault,
) -> Result<schedule::cuda::Run, schedule::cuda::Error> {
    let checked = crate::front::parse(source)
        .and_then(|p| crate::check::check(p, true))
        .map_err(|_| schedule::cuda::Error::mismatch("spatial source binding failed"))?;
    let lowered = crate::mir::lower(&checked)
        .map_err(|_| schedule::cuda::Error::mismatch("spatial MIR binding failed"))?;
    if lowered.contract_digest() != program.contract_digest() {
        return Err(schedule::cuda::Error::mismatch(
            "spatial MIR differs from checked source",
        ));
    }
    let device = RefCell::new(device);
    let failure = RefCell::new(None);
    let launches = Cell::new(0usize);
    let map_error = |error: invar_cuda::Error, poisoned: bool| {
        let error = schedule::cuda::Error::from_cuda(error, poisoned);
        *failure.borrow_mut() = Some(error.clone());
        EvalError::Tool {
            code: error.code,
            message: error.message,
        }
    };
    let mut launch = |plan: &invar_device::spatial::Plan, inputs: [&[u32]; 4]| {
        let mut device = device.borrow_mut();
        match device.run_spatial_with_fault(plan, inputs, compiler, fault.collective()) {
            Ok(r) => {
                launches.set(launches.get() + usize::from(plan.work() != 0));
                Ok([r.counts, r.indices, r.distances])
            }
            Err(e) => Err(map_error(e, device.is_poisoned())),
        }
    };
    let mut sort = |keys: &[u32], ids: &[u32]| {
        let plan =
            invar_device::sorting::Plan::new(keys.len() as u32, invar_device::sorting::Kind::I32)
                .map_err(|_| EvalError::Tool {
                code: "RESOURCE_LIMIT",
                message: "spatial radix plan limit".into(),
            })?;
        let mut device = device.borrow_mut();
        match device.run_sort_with_fault(&plan, keys, ids, compiler, fault.collective()) {
            Ok(r) => {
                launches.set(launches.get() + usize::from(!keys.is_empty()));
                Ok((r.keys, r.payload))
            }
            Err(e) => Err(map_error(e, device.is_poisoned())),
        }
    };
    let mut reductions = Vec::new();
    // Reached float reductions run on the device, as in `--engine cuda`.
    let mut reducer = |call: &crate::mir_eval::ReduceCall<'_>| {
        schedule::cuda::reduce::forced(call, &mut device.borrow_mut(), compiler, &mut reductions)
    };
    let mut executed = Vec::new();
    let mut provider =
        |function: &str, name: &str, args: &[eval::Value], span, live: &mut usize| {
            let before = launches.get();
            let spatial = super::result(name).is_some();
            let result = if spatial {
                super::runtime::device::invoke(name, args, span, live, &mut launch, &mut sort)?
            } else {
                // Explicit non-spatial sorts retain the existing serial semantics;
                // receipts identify this host work instead of claiming a GPU launch.
                crate::sorting::invoke(name, args, span, live, 1)?
            };
            executed.push(Executed {
                function: function.into(),
                span,
                kind: if spatial {
                    PlanKind::Spatial
                } else {
                    PlanKind::Sort
                },
                published_array: None,
                implementation: Some(if !spatial {
                    "cpu-lsd-radix-4x8"
                } else if launches.get() == before {
                    "host-spatial-metadata-or-empty"
                } else {
                    "cuda-sorted-grid-strict-f32"
                }),
            });
            Ok(result)
        };
    let observation =
        crate::mir_eval::observe_all(program, fuel, None, Some(&mut provider), Some(&mut reducer))
            .0;
    if let Some(error) = failure.into_inner() {
        return Err(error);
    }
    Ok(schedule::cuda::Run {
        observation,
        executed,
        cache_hits: 0,
        hint_hits: 0,
        hint_invalids: 0,
        reductions,
    })
}
