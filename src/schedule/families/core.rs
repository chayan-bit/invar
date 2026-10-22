//! Existing map/fusion/reduction/scan family. Backend bodies are moved unchanged.
use super::super::*;
pub(super) type Candidate = Region;

/// A PNF map, fusion, reduction or scan admission at an A0 loop whose plan is
/// not serial; the recognizer it replaces is the oracle in tests/support.
pub(super) fn lower(admitted: &lowering::Admitted, live: &lowering::Live<'_>) -> Result<Option<Candidate>, admission::Error> {
    Ok((admitted.admits(lowering::CORE) && live.region.plan_kind() != PlanKind::Serial).then(|| live.region.clone()))
}

pub(super) fn cpu(
    candidate: &Candidate,
    live: &LiveLoop<'_>,
    pending: &mut VecDeque<Pending>,
    workers: usize,
    hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    region_step(candidate, live, pending, workers, hooks)
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn is_consumer(candidate: &Candidate) -> bool {
    matches!(
        candidate.execution(),
        Execution::FusedConsumer { .. } | Execution::DagConsumer { .. }
    )
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn is_producer(candidate: &Candidate) -> bool {
    matches!(
        candidate.execution(),
        Execution::FusedProducer(_) | Execution::DagProducer(_)
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(super) mod metal {
    use super::*;
    use crate::schedule::metal::{Error, Fault};
    use crate::{
        device_cache::{self, HintState},
        device_ir,
        plan::{DeviceCost, DeviceCostModel},
        typed_net::Prefix,
    };
    use invar_device::{Capabilities, Provenance, Verified};
    use invar_metal::MetalDevice;
    pub(crate) fn cost(
        region: &Region,
        live: &LiveLoop<'_>,
        caps: Capabilities,
        model: DeviceCostModel,
    ) -> Option<DeviceCost> {
        let len = live.iterations;
        let plan = verified(region, len, caps).ok()?;
        let ir = plan.ir();
        let (inputs, outputs, operations, launches) = if let Some(dag) = &ir.dag {
            (dag.inputs.len(), dag.outputs.len(), dag.nodes.len(), 1)
        } else if let Some(collective) = &ir.collective {
            (collective.inputs.len(), 1, 2, 3)
        } else {
            (1, 1, ir.ops.len(), 1)
        };
        let transfer_bytes = len.checked_mul(inputs + outputs)?.checked_mul(4)?;
        Some(model.estimate(transfer_bytes, len, operations, launches))
    }

    /// None here: `metal_aot::run_kernels` builds every core region from
    /// admission, because its MSL does not depend on the reached length.
    pub(crate) fn kernels(_: &Region, _: &LiveLoop<'_>, _: Capabilities) -> Vec<(String, Vec<String>)> {
        Vec::new()
    }

    /// The auto reason for a region with no Metal estimate.
    pub(crate) fn ineligible_reason(region: &Region, len: usize) -> &'static str {
        let planned = u32::try_from(len)
            .ok()
            .and_then(|len| device_ir::from_region(region, len).ok())
            .map(|mut ir| device_ir::plan_metal(&mut ir));
        match planned {
            Some(Err(device_ir::Error::MetalBufferSlots { .. })) => "metal-buffer-slot-limit",
            _ => "device-plan-ineligible",
        }
    }

    fn verified(region: &Region, len: usize, caps: Capabilities) -> Result<Verified, Error> {
        let len = u32::try_from(len).map_err(|_| Error::unavailable("Metal length exceeds u32"))?;
        let ir = device_ir::from_region(region, len)
            .and_then(|mut ir| device_ir::plan_metal(&mut ir).map(|()| ir))
            .map_err(|error| Error::unavailable(format!("region has no Metal plan: {error}")))?;
        let origin = Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        invar_device::verify(ir, origin, caps)
            .map_err(|error| Error::mismatch(format!("device plan verification failed: {error:?}")))
    }

    fn inputs<'a>(
        live: &'a LiveLoop<'a>,
        names: &[String],
        output: Option<&str>,
    ) -> Result<Vec<std::cell::Ref<'a, Vec<u32>>>, Error> {
        if live.iterations > laws::MAX_ELEMENTS
            || names.iter().any(|name| {
                live.lengths
                    .get(name)
                    .is_none_or(|len| *len < live.iterations)
            })
            || output.is_some_and(|name| {
                live.lengths
                    .get(name)
                    .is_none_or(|len| *len < live.iterations)
                    || names.iter().any(|input| live.same_object(name, input))
            })
        {
            return Err(Error::unavailable("live Metal array guard is false"));
        }
        names
            .iter()
            .map(|name| {
                live.array(name)
                    .ok_or_else(|| Error::unavailable("live Metal input is unavailable"))
            })
            .collect()
    }

    fn lengths(
        names: &[String],
        borrowed: &[std::cell::Ref<'_, Vec<u32>>],
    ) -> Vec<(String, usize)> {
        names
            .iter()
            .zip(borrowed)
            .map(|(name, values)| (name.clone(), values.len()))
            .collect()
    }

    fn independent_step(
        source: &[u8],
        region: &Region,
        live: &LiveLoop<'_>,
        pending: &mut VecDeque<Pending>,
        device: &mut MetalDevice,
        fault: Fault,
    ) -> Result<(Step, bool, HintState), Error> {
        let len = live.iterations;
        let plan = verified(region, len, device.capabilities())?;
        let artifact = invar_device::emit_msl(&plan)
            .map_err(|error| Error::mismatch(format!("Metal emission failed: {error:?}")))?;
        let hint = device_cache::expected(source, region, &plan, &artifact);
        let hint_state = device_cache::load(&hint);
        let result = match region.execution() {
            Execution::Map(map) => {
                let borrowed = inputs(live, &map.inputs, Some(&map.output))?;
                let input_lengths = lengths(&map.inputs, &borrowed);
                let (values, cache_hit) = if plan.ir().dag.is_some() {
                    // A two-input map runs as the one-output case of the DAG kernel.
                    let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                    let result = device
                        .run_fused_map_with_fault(&plan, &slices, fault)
                        .map_err(|error| Error::from_metal(error, device.is_poisoned()))?;
                    let values = result
                        .outputs
                        .into_iter()
                        .next()
                        .ok_or_else(|| Error::mismatch("map kernel returned no output"))?;
                    (values, result.cache_hit)
                } else {
                    if borrowed.len() != 1 {
                        return Err(Error::unavailable("Metal map requires one input"));
                    }
                    let result = device
                        .run_map_with_fault(&plan, &borrowed[0][..len], fault)
                        .map_err(|error| Error::from_metal(error, device.is_poisoned()))?;
                    (result.values, result.cache_hit)
                };
                Ok((
                    Step {
                        plan: LoopOverride {
                            histogram: None,
                            scatter: None,
                            gather: None,
                            summary: None,
                            region: None,
                            tensor: None,
                            transpose: None,
                            stencil: None,
                            iterations: len,
                            input_lengths,
                            scalar: None,
                            array: Some((map.output.clone(), values)),
                        },
                        kind: PlanKind::Map,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    cache_hit,
                ))
            }
            Execution::FusedProducer(pair) => {
                pending.clear();
                let borrowed = inputs(
                    live,
                    &pair.inputs,
                    (!pair.elide_intermediate).then_some(pair.intermediate.as_str()),
                )?;
                let input_lengths = lengths(&pair.inputs, &borrowed);
                let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                let result = device
                    .run_fused_map_with_fault(&plan, &slices, fault)
                    .map_err(|error| Error::from_metal(error, device.is_poisoned()))?;
                let [first, second]: [Vec<u32>; 2] = result
                    .outputs
                    .try_into()
                    .map_err(|_| Error::mismatch("pair kernel returned wrong output count"))?;
                let mut operands = Vec::new();
                if !pair.elide_intermediate {
                    operands.push((pair.consumer_input.clone(), first.clone()));
                }
                if let Some(other) = &pair.consumer_other {
                    let index = pair
                        .inputs
                        .iter()
                        .position(|name| name == other)
                        .ok_or_else(|| Error::mismatch("pair operand missing from graph plan"))?;
                    operands.push((other.clone(), borrowed[index][..len].to_vec()));
                }
                pending.push_back(Pending {
                    function: live.function.clone(),
                    consumer: pair.source_spans[1],
                    stage: 1,
                    operands,
                    output: second,
                    required: pair.elide_intermediate,
                });
                Ok((
                    Step {
                        plan: LoopOverride {
                            histogram: None,
                            scatter: None,
                            gather: None,
                            summary: None,
                            region: None,
                            tensor: None,
                            transpose: None,
                            stencil: None,
                            iterations: len,
                            input_lengths,
                            scalar: None,
                            array: (!pair.elide_intermediate)
                                .then_some((pair.intermediate.clone(), first)),
                        },
                        kind: PlanKind::FusedMap,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    result.cache_hit,
                ))
            }
            Execution::DagProducer(chain) => {
                pending.clear();
                let first = chain
                    .outputs
                    .first()
                    .ok_or_else(|| Error::mismatch("DAG has no output"))?;
                let borrowed = inputs(live, &chain.inputs, Some(first))?;
                let input_lengths = lengths(&chain.inputs, &borrowed);
                let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                let result = device
                    .run_fused_map_with_fault(&plan, &slices, fault)
                    .map_err(|error| Error::from_metal(error, device.is_poisoned()))?;
                if result.outputs.len() != chain.outputs.len() {
                    return Err(Error::mismatch("DAG kernel returned wrong output count"));
                }
                let mut values = result.outputs;
                for (index, link) in chain.links.iter().enumerate() {
                    let operands = chain
                        .link_operands(index, &values, &slices, len)
                        .ok_or_else(|| Error::mismatch("DAG operand missing from graph plan"))?;
                    pending.push_back(Pending {
                        function: live.function.clone(),
                        consumer: link.span,
                        stage: index + 1,
                        operands,
                        output: values[index + 1].clone(),
                        required: false,
                    });
                }
                Ok((
                    Step {
                        plan: LoopOverride {
                            histogram: None,
                            scatter: None,
                            gather: None,
                            summary: None,
                            region: None,
                            tensor: None,
                            transpose: None,
                            stencil: None,
                            iterations: len,
                            input_lengths,
                            scalar: None,
                            array: Some((first.clone(), values.remove(0))),
                        },
                        kind: PlanKind::FusedMap,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    result.cache_hit,
                ))
            }
            Execution::PairReduction(candidate) => {
                let borrowed = inputs(live, &candidate.inputs, None)?;
                let seeds = candidate
                    .states
                    .iter()
                    .map(|name| live.scalars.get(name).copied())
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| Error::unavailable("live pair seeds are unavailable"))?;
                let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                let result = device
                    .run_collective_with_fault(&plan, &slices, 0, fault)
                    .map_err(|error| Error::from_metal(error, device.is_poisoned()))?;
                let [a, b] = result
                    .pair
                    .ok_or_else(|| Error::mismatch("pair kernel returned no pair"))?;
                let total = crate::laws::Affine {
                    a: seeds[0],
                    b: seeds[1],
                }
                .then(crate::laws::Affine { a, b });
                Ok((
                    Step {
                        plan: LoopOverride {
                            region: None,
                            histogram: None,
                            scatter: None,
                            gather: None,
                            summary: None,
                            tensor: None,
                            transpose: None,
                            stencil: None,
                            iterations: len,
                            input_lengths: lengths(&candidate.inputs, &borrowed),
                            scalar: Some(vec![
                                (candidate.states[0].clone(), seeds[0], total.a),
                                (candidate.states[1].clone(), seeds[1], total.b),
                            ]),
                            array: None,
                        },
                        kind: PlanKind::Reduction,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    result.cache_hit,
                ))
            }
            Execution::Reduction(candidate) => {
                let borrowed = inputs(live, std::slice::from_ref(&candidate.input), None)?;
                let seed = live
                    .scalars
                    .get(&candidate.state)
                    .copied()
                    .ok_or_else(|| Error::unavailable("live reduction seed is unavailable"))?;
                let result = device
                    .run_collective_with_fault(&plan, &[&borrowed[0][..len]], seed, fault)
                    .map_err(|error| Error::from_metal(error, device.is_poisoned()))?;
                Ok((
                    Step {
                        plan: LoopOverride {
                            histogram: None,
                            scatter: None,
                            gather: None,
                            summary: None,
                            region: None,
                            tensor: None,
                            transpose: None,
                            stencil: None,
                            iterations: len,
                            input_lengths: lengths(
                                std::slice::from_ref(&candidate.input),
                                &borrowed,
                            ),
                            scalar: Some(vec![(candidate.state.clone(), seed, result.final_state)]),
                            array: None,
                        },
                        kind: PlanKind::Reduction,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    result.cache_hit,
                ))
            }
            Execution::Scan(candidate) => {
                let crate::typed_net::Plan::AffineScan { prefix, .. } = &candidate.plan else {
                    return Err(Error::mismatch("scan graph plan missing"));
                };
                let output = if *prefix == Prefix::None {
                    None
                } else {
                    candidate.output.as_deref()
                };
                let borrowed = inputs(live, &candidate.arrays, output)?;
                let seed = live
                    .scalars
                    .get(&candidate.state)
                    .copied()
                    .ok_or_else(|| Error::unavailable("live scan seed is unavailable"))?;
                let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                let result = device
                    .run_collective_with_fault(&plan, &slices, seed, fault)
                    .map_err(|error| Error::from_metal(error, device.is_poisoned()))?;
                Ok((
                    Step {
                        plan: LoopOverride {
                            histogram: None,
                            scatter: None,
                            gather: None,
                            summary: None,
                            region: None,
                            tensor: None,
                            transpose: None,
                            stencil: None,
                            iterations: len,
                            input_lengths: lengths(&candidate.arrays, &borrowed),
                            scalar: Some(vec![(candidate.state.clone(), seed, result.final_state)]),
                            array: output.map(|name| (name.to_owned(), result.prefixes)),
                        },
                        kind: PlanKind::Scan,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    result.cache_hit,
                ))
            }
            _ => Err(Error::unavailable("region has no independent Metal kernel")),
        };
        if result.is_ok() && hint_state != HintState::Hit {
            let _ = device_cache::save(&hint);
        }
        result.map(|(step, hit)| (step, hit, hint_state))
    }

    pub(crate) fn step(
        source: &[u8],
        region: &Region,
        live: &LiveLoop<'_>,
        pending: &mut VecDeque<Pending>,
        device: &mut MetalDevice,
        fault: Fault,
    ) -> Result<(Step, bool, HintState), Error> {
        match region.execution() {
            Execution::FusedConsumer { standalone, .. } => {
                staged_consumer(standalone, live, pending.pop_front())
                    .map(|step| (step, false, HintState::Miss))
                    .ok_or_else(|| Error::unavailable("fused Metal publication guard is false"))
            }
            Execution::DagConsumer {
                stage, standalone, ..
            } => {
                let staged = pending.pop_front().filter(|item| item.stage == *stage);
                staged_consumer(standalone, live, staged)
                    .map(|step| (step, false, HintState::Miss))
                    .ok_or_else(|| Error::unavailable("DAG Metal publication guard is false"))
            }
            _ => independent_step(source, region, live, pending, device, fault),
        }
    }
}

#[cfg(feature = "cuda")]
pub(super) mod cuda {
    use super::*;
    use crate::schedule::cuda::{Error, Fault};
    use crate::{
        device_cache::{self, HintState},
        device_ir,
        plan::{DeviceCost, DeviceCostModel},
        typed_net::Prefix,
    };
    use invar_cuda::cuda::CudaDevice;
    use invar_device::{Capabilities, Provenance, Verified};
    use std::path::Path;
    pub(crate) fn cost(
        region: &Region,
        live: &LiveLoop<'_>,
        caps: Capabilities,
        model: DeviceCostModel,
    ) -> Option<DeviceCost> {
        let len = live.iterations;
        let plan = verified(region, len, caps).ok()?;
        let ir = plan.ir();
        let (inputs, outputs, operations, launches) = if let Some(dag) = &ir.dag {
            (dag.inputs.len(), dag.outputs.len(), dag.nodes.len(), 1)
        } else if let Some(collective) = &ir.collective {
            (collective.inputs.len(), 1, 2, 3)
        } else {
            (1, 1, ir.ops.len(), 1)
        };
        let transfer_bytes = len.checked_mul(inputs + outputs)?.checked_mul(4)?;
        Some(model.estimate(transfer_bytes, len, operations, launches))
    }

    fn verified(region: &Region, len: usize, caps: Capabilities) -> Result<Verified, Error> {
        let len = u32::try_from(len).map_err(|_| Error::unavailable("CUDA length exceeds u32"))?;
        let ir = device_ir::from_region(region, len)
            .map_err(|error| Error::unavailable(format!("region has no CUDA plan: {error:?}")))?;
        let origin = Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        invar_device::verify(ir, origin, caps)
            .map_err(|error| Error::mismatch(format!("device plan verification failed: {error:?}")))
    }

    fn inputs<'a>(
        live: &'a LiveLoop<'a>,
        names: &[String],
        output: Option<&str>,
    ) -> Result<Vec<std::cell::Ref<'a, Vec<u32>>>, Error> {
        if live.iterations > laws::MAX_ELEMENTS
            || names.iter().any(|name| {
                live.lengths
                    .get(name)
                    .is_none_or(|len| *len < live.iterations)
            })
            || output.is_some_and(|name| {
                live.lengths
                    .get(name)
                    .is_none_or(|len| *len < live.iterations)
                    || names.iter().any(|input| live.same_object(name, input))
            })
        {
            return Err(Error::unavailable("live CUDA array guard is false"));
        }
        names
            .iter()
            .map(|name| {
                live.array(name)
                    .ok_or_else(|| Error::unavailable("live CUDA input is unavailable"))
            })
            .collect()
    }

    fn lengths(
        names: &[String],
        borrowed: &[std::cell::Ref<'_, Vec<u32>>],
    ) -> Vec<(String, usize)> {
        names
            .iter()
            .zip(borrowed)
            .map(|(name, values)| (name.clone(), values.len()))
            .collect()
    }

    fn independent_step(
        source: &[u8],
        region: &Region,
        live: &LiveLoop<'_>,
        pending: &mut VecDeque<Pending>,
        device: &mut CudaDevice,
        compiler: &Path,
        fault: Fault,
    ) -> Result<(Step, bool, HintState), Error> {
        let len = live.iterations;
        let plan = verified(region, len, device.capabilities())?;
        let artifact = invar_device::emit_cuda(&plan)
            .map_err(|error| Error::mismatch(format!("CUDA emission failed: {error:?}")))?;
        let hint = device_cache::expected(source, region, &plan, &artifact);
        let hint_state = device_cache::load(&hint);
        let result = match region.execution() {
            Execution::Map(map) => {
                let borrowed = inputs(live, &map.inputs, Some(&map.output))?;
                let input_lengths = lengths(&map.inputs, &borrowed);
                let (values, cache_hit) = if plan.ir().dag.is_some() {
                    // A two-input map runs as the one-output case of the DAG kernel.
                    let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                    let result = device
                        .run_fused_map_with_fault(&plan, &slices, compiler, fault.fused())
                        .map_err(|error| Error::from_cuda(error, device.is_poisoned()))?;
                    let values = result
                        .outputs
                        .into_iter()
                        .next()
                        .ok_or_else(|| Error::mismatch("map kernel returned no output"))?;
                    (values, result.cache_hit)
                } else {
                    if borrowed.len() != 1 {
                        return Err(Error::unavailable("CUDA map requires one input"));
                    }
                    let result = device
                        .run_map_with_fault(&plan, &borrowed[0][..len], compiler, fault.fused())
                        .map_err(|error| Error::from_cuda(error, device.is_poisoned()))?;
                    (result.values, result.cache_hit)
                };
                Ok((
                    Step {
                        plan: LoopOverride {
                            histogram: None,
                            scatter: None,
                            gather: None,
                            summary: None,
                            region: None,
                            tensor: None,
                            transpose: None,
                            stencil: None,
                            iterations: len,
                            input_lengths,
                            scalar: None,
                            array: Some((map.output.clone(), values)),
                        },
                        kind: PlanKind::Map,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    cache_hit,
                ))
            }
            Execution::FusedProducer(pair) => {
                pending.clear();
                let borrowed = inputs(
                    live,
                    &pair.inputs,
                    (!pair.elide_intermediate).then_some(pair.intermediate.as_str()),
                )?;
                let input_lengths = lengths(&pair.inputs, &borrowed);
                let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                let result = device
                    .run_fused_map_with_fault(&plan, &slices, compiler, fault.fused())
                    .map_err(|error| Error::from_cuda(error, device.is_poisoned()))?;
                let [first, second]: [Vec<u32>; 2] = result
                    .outputs
                    .try_into()
                    .map_err(|_| Error::mismatch("pair kernel returned wrong output count"))?;
                let mut operands = Vec::new();
                if !pair.elide_intermediate {
                    operands.push((pair.consumer_input.clone(), first.clone()));
                }
                if let Some(other) = &pair.consumer_other {
                    let index = pair
                        .inputs
                        .iter()
                        .position(|name| name == other)
                        .ok_or_else(|| Error::mismatch("pair operand missing from graph plan"))?;
                    operands.push((other.clone(), borrowed[index][..len].to_vec()));
                }
                pending.push_back(Pending {
                    function: live.function.clone(),
                    consumer: pair.source_spans[1],
                    stage: 1,
                    operands,
                    output: second,
                    required: pair.elide_intermediate,
                });
                Ok((
                    Step {
                        plan: LoopOverride {
                            histogram: None,
                            scatter: None,
                            gather: None,
                            summary: None,
                            region: None,
                            tensor: None,
                            transpose: None,
                            stencil: None,
                            iterations: len,
                            input_lengths,
                            scalar: None,
                            array: (!pair.elide_intermediate)
                                .then_some((pair.intermediate.clone(), first)),
                        },
                        kind: PlanKind::FusedMap,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    result.cache_hit,
                ))
            }
            Execution::DagProducer(chain) => {
                pending.clear();
                let first = chain
                    .outputs
                    .first()
                    .ok_or_else(|| Error::mismatch("DAG has no output"))?;
                let borrowed = inputs(live, &chain.inputs, Some(first))?;
                let input_lengths = lengths(&chain.inputs, &borrowed);
                let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                let result = device
                    .run_fused_map_with_fault(&plan, &slices, compiler, fault.fused())
                    .map_err(|error| Error::from_cuda(error, device.is_poisoned()))?;
                if result.outputs.len() != chain.outputs.len() {
                    return Err(Error::mismatch("DAG kernel returned wrong output count"));
                }
                let mut values = result.outputs;
                for (index, link) in chain.links.iter().enumerate() {
                    let operands = chain
                        .link_operands(index, &values, &slices, len)
                        .ok_or_else(|| Error::mismatch("DAG operand missing from graph plan"))?;
                    pending.push_back(Pending {
                        function: live.function.clone(),
                        consumer: link.span,
                        stage: index + 1,
                        operands,
                        output: values[index + 1].clone(),
                        required: false,
                    });
                }
                Ok((
                    Step {
                        plan: LoopOverride {
                            histogram: None,
                            scatter: None,
                            gather: None,
                            summary: None,
                            region: None,
                            tensor: None,
                            transpose: None,
                            stencil: None,
                            iterations: len,
                            input_lengths,
                            scalar: None,
                            array: Some((first.clone(), values.remove(0))),
                        },
                        kind: PlanKind::FusedMap,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    result.cache_hit,
                ))
            }
            Execution::PairReduction(candidate) => {
                let borrowed = inputs(live, &candidate.inputs, None)?;
                let seeds = candidate
                    .states
                    .iter()
                    .map(|name| live.scalars.get(name).copied())
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| Error::unavailable("live pair seeds are unavailable"))?;
                let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                let result = device
                    .run_collective_with_fault(&plan, &slices, 0, compiler, fault.collective())
                    .map_err(|error| Error::from_cuda(error, device.is_poisoned()))?;
                let [a, b] = result
                    .pair
                    .ok_or_else(|| Error::mismatch("pair kernel returned no pair"))?;
                let total = crate::laws::Affine {
                    a: seeds[0],
                    b: seeds[1],
                }
                .then(crate::laws::Affine { a, b });
                Ok((
                    Step {
                        plan: LoopOverride {
                            region: None,
                            histogram: None,
                            scatter: None,
                            gather: None,
                            summary: None,
                            tensor: None,
                            transpose: None,
                            stencil: None,
                            iterations: len,
                            input_lengths: lengths(&candidate.inputs, &borrowed),
                            scalar: Some(vec![
                                (candidate.states[0].clone(), seeds[0], total.a),
                                (candidate.states[1].clone(), seeds[1], total.b),
                            ]),
                            array: None,
                        },
                        kind: PlanKind::Reduction,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    result.cache_hit,
                ))
            }
            Execution::Reduction(candidate) => {
                let borrowed = inputs(live, std::slice::from_ref(&candidate.input), None)?;
                let seed = live
                    .scalars
                    .get(&candidate.state)
                    .copied()
                    .ok_or_else(|| Error::unavailable("live reduction seed is unavailable"))?;
                let result = device
                    .run_collective_with_fault(
                        &plan,
                        &[&borrowed[0][..len]],
                        seed,
                        compiler,
                        fault.collective(),
                    )
                    .map_err(|error| Error::from_cuda(error, device.is_poisoned()))?;
                Ok((
                    Step {
                        plan: LoopOverride {
                            histogram: None,
                            scatter: None,
                            gather: None,
                            summary: None,
                            region: None,
                            tensor: None,
                            transpose: None,
                            stencil: None,
                            iterations: len,
                            input_lengths: lengths(
                                std::slice::from_ref(&candidate.input),
                                &borrowed,
                            ),
                            scalar: Some(vec![(candidate.state.clone(), seed, result.final_state)]),
                            array: None,
                        },
                        kind: PlanKind::Reduction,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    result.cache_hit,
                ))
            }
            Execution::Scan(candidate) => {
                let crate::typed_net::Plan::AffineScan { prefix, .. } = &candidate.plan else {
                    return Err(Error::mismatch("scan graph plan missing"));
                };
                let output = if *prefix == Prefix::None {
                    None
                } else {
                    candidate.output.as_deref()
                };
                let borrowed = inputs(live, &candidate.arrays, output)?;
                let seed = live
                    .scalars
                    .get(&candidate.state)
                    .copied()
                    .ok_or_else(|| Error::unavailable("live scan seed is unavailable"))?;
                let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                let result = device
                    .run_collective_with_fault(&plan, &slices, seed, compiler, fault.collective())
                    .map_err(|error| Error::from_cuda(error, device.is_poisoned()))?;
                Ok((
                    Step {
                        plan: LoopOverride {
                            histogram: None,
                            scatter: None,
                            gather: None,
                            summary: None,
                            region: None,
                            tensor: None,
                            transpose: None,
                            stencil: None,
                            iterations: len,
                            input_lengths: lengths(&candidate.arrays, &borrowed),
                            scalar: Some(vec![(candidate.state.clone(), seed, result.final_state)]),
                            array: output.map(|name| (name.to_owned(), result.prefixes)),
                        },
                        kind: PlanKind::Scan,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    result.cache_hit,
                ))
            }
            _ => Err(Error::unavailable("region has no independent CUDA kernel")),
        };
        if result.is_ok() && hint_state != HintState::Hit {
            let _ = device_cache::save(&hint);
        }
        result.map(|(step, hit)| (step, hit, hint_state))
    }

    pub(crate) fn step(
        source: &[u8],
        region: &Region,
        live: &LiveLoop<'_>,
        pending: &mut VecDeque<Pending>,
        device: &mut CudaDevice,
        compiler: &Path,
        fault: Fault,
    ) -> Result<(Step, bool, HintState), Error> {
        match region.execution() {
            Execution::FusedConsumer { standalone, .. } => {
                staged_consumer(standalone, live, pending.pop_front())
                    .map(|step| (step, false, HintState::Miss))
                    .ok_or_else(|| Error::unavailable("fused CUDA publication guard is false"))
            }
            Execution::DagConsumer {
                stage, standalone, ..
            } => {
                let staged = pending.pop_front().filter(|item| item.stage == *stage);
                staged_consumer(standalone, live, staged)
                    .map(|step| (step, false, HintState::Miss))
                    .ok_or_else(|| Error::unavailable("DAG CUDA publication guard is false"))
            }
            _ => independent_step(source, region, live, pending, device, compiler, fault),
        }
    }
}
