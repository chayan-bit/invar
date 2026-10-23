//! Existing map/fusion/reduction/scan family. One generic step runs it on CUDA or Metal.
use super::super::*;
pub(super) type Candidate = Region;

/// A PNF map, fusion, reduction or scan admission at an A0 loop whose plan is
/// not serial; the recognizer it replaces is the oracle in tests/support.
pub(super) fn lower(
    admitted: &lowering::Admitted,
    live: &lowering::Live<'_>,
) -> Result<Option<Candidate>, admission::Error> {
    Ok(
        (admitted.admits(lowering::CORE) && live.region.plan_kind() != PlanKind::Serial)
            .then(|| live.region.clone()),
    )
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

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use self::device::{Collected, Launch, estimate, independent};

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
/// The CUDA and Metal modules below differ only in their launches and plan checks.
mod device {
    use super::*;
    use crate::{
        plan::{DeviceCost, DeviceCostModel},
        typed_net::Prefix,
    };
    use invar_device::Verified;

    /// One backend's kernel launches for the independent core regions. Each launch
    /// returns completed values and whether the kernel cache hit.
    pub(super) trait Launch {
        type Error;
        /// The backend as named in refusal messages.
        const NAME: &'static str;
        fn unavailable(message: impl Into<String>) -> Self::Error;
        fn mismatch(message: impl Into<String>) -> Self::Error;
        fn fused(
            &mut self,
            plan: &Verified,
            inputs: &[&[u32]],
        ) -> Result<(Vec<Vec<u32>>, bool), Self::Error>;
        fn map(&mut self, plan: &Verified, input: &[u32]) -> Result<(Vec<u32>, bool), Self::Error>;
        fn collective(
            &mut self,
            plan: &Verified,
            inputs: &[&[u32]],
            seed: u32,
        ) -> Result<Collected, Self::Error>;
    }

    pub(super) struct Collected {
        pub(super) final_state: u32,
        pub(super) pair: Option<[u32; 2]>,
        pub(super) prefixes: Vec<u32>,
        pub(super) cache_hit: bool,
    }

    /// Transfer and launch shape of a verified core plan for the auto cost model.
    pub(super) fn estimate(
        plan: &Verified,
        len: usize,
        model: DeviceCostModel,
    ) -> Option<DeviceCost> {
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

    fn inputs<'a, L: Launch>(
        live: &'a LiveLoop<'a>,
        names: &[String],
        output: Option<&str>,
    ) -> Result<Vec<std::cell::Ref<'a, Vec<u32>>>, L::Error> {
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
            return Err(L::unavailable(format!(
                "live {} array guard is false",
                L::NAME
            )));
        }
        names
            .iter()
            .map(|name| {
                live.array(name)
                    .ok_or_else(|| L::unavailable(format!("live {} input is unavailable", L::NAME)))
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

    /// Run one independent core region on a device; staged consumers are not here.
    pub(super) fn independent<L: Launch>(
        launch: &mut L,
        plan: &Verified,
        region: &Region,
        live: &LiveLoop<'_>,
        pending: &mut VecDeque<Pending>,
    ) -> Result<(Step, bool), L::Error> {
        let len = live.iterations;
        match region.execution() {
            Execution::Map(map) => {
                let borrowed = inputs::<L>(live, &map.inputs, Some(&map.output))?;
                let input_lengths = lengths(&map.inputs, &borrowed);
                let (values, cache_hit) = if plan.ir().dag.is_some() {
                    // A two-input map runs as the one-output case of the DAG kernel.
                    let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                    let (outputs, cache_hit) = launch.fused(plan, &slices)?;
                    let values = outputs
                        .into_iter()
                        .next()
                        .ok_or_else(|| L::mismatch("map kernel returned no output"))?;
                    (values, cache_hit)
                } else {
                    if borrowed.len() != 1 {
                        return Err(L::unavailable(format!(
                            "{} map requires one input",
                            L::NAME
                        )));
                    }
                    launch.map(plan, &borrowed[0][..len])?
                };
                Ok((
                    Step {
                        plan: LoopOverride {
                            iterations: len,
                            input_lengths,
                            array: Some((map.output.clone(), values)),
                            ..Default::default()
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
                let borrowed = inputs::<L>(
                    live,
                    &pair.inputs,
                    (!pair.elide_intermediate).then_some(pair.intermediate.as_str()),
                )?;
                let input_lengths = lengths(&pair.inputs, &borrowed);
                let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                let (outputs, cache_hit) = launch.fused(plan, &slices)?;
                let [first, second]: [Vec<u32>; 2] = outputs
                    .try_into()
                    .map_err(|_| L::mismatch("pair kernel returned wrong output count"))?;
                let mut operands = Vec::new();
                if !pair.elide_intermediate {
                    operands.push((pair.consumer_input.clone(), first.clone()));
                }
                if let Some(other) = &pair.consumer_other {
                    let index = pair
                        .inputs
                        .iter()
                        .position(|name| name == other)
                        .ok_or_else(|| L::mismatch("pair operand missing from graph plan"))?;
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
                            iterations: len,
                            input_lengths,
                            array: (!pair.elide_intermediate)
                                .then_some((pair.intermediate.clone(), first)),
                            ..Default::default()
                        },
                        kind: PlanKind::FusedMap,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    cache_hit,
                ))
            }
            Execution::DagProducer(chain) => {
                pending.clear();
                let first = chain
                    .outputs
                    .first()
                    .ok_or_else(|| L::mismatch("DAG has no output"))?;
                let borrowed = inputs::<L>(live, &chain.inputs, Some(first))?;
                let input_lengths = lengths(&chain.inputs, &borrowed);
                let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                let (mut values, cache_hit) = launch.fused(plan, &slices)?;
                if values.len() != chain.outputs.len() {
                    return Err(L::mismatch("DAG kernel returned wrong output count"));
                }
                for (index, link) in chain.links.iter().enumerate() {
                    let operands = chain
                        .link_operands(index, &values, &slices, len)
                        .ok_or_else(|| L::mismatch("DAG operand missing from graph plan"))?;
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
                            iterations: len,
                            input_lengths,
                            array: Some((first.clone(), values.remove(0))),
                            ..Default::default()
                        },
                        kind: PlanKind::FusedMap,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    cache_hit,
                ))
            }
            Execution::PairReduction(candidate) => {
                let borrowed = inputs::<L>(live, &candidate.inputs, None)?;
                let seeds = candidate
                    .states
                    .iter()
                    .map(|name| live.scalars.get(name).copied())
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| L::unavailable("live pair seeds are unavailable"))?;
                let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                let result = launch.collective(plan, &slices, 0)?;
                let [a, b] = result
                    .pair
                    .ok_or_else(|| L::mismatch("pair kernel returned no pair"))?;
                let total = crate::laws::Affine {
                    a: seeds[0],
                    b: seeds[1],
                }
                .then(crate::laws::Affine { a, b });
                Ok((
                    Step {
                        plan: LoopOverride {
                            iterations: len,
                            input_lengths: lengths(&candidate.inputs, &borrowed),
                            scalar: Some(vec![
                                (candidate.states[0].clone(), seeds[0], total.a),
                                (candidate.states[1].clone(), seeds[1], total.b),
                            ]),
                            ..Default::default()
                        },
                        kind: PlanKind::Reduction,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    result.cache_hit,
                ))
            }
            Execution::Reduction(candidate) => {
                let borrowed = inputs::<L>(live, std::slice::from_ref(&candidate.input), None)?;
                let seed = live
                    .scalars
                    .get(&candidate.state)
                    .copied()
                    .ok_or_else(|| L::unavailable("live reduction seed is unavailable"))?;
                let result = launch.collective(plan, &[&borrowed[0][..len]], seed)?;
                Ok((
                    Step {
                        plan: LoopOverride {
                            iterations: len,
                            input_lengths: lengths(
                                std::slice::from_ref(&candidate.input),
                                &borrowed,
                            ),
                            scalar: Some(vec![(candidate.state.clone(), seed, result.final_state)]),
                            ..Default::default()
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
                    return Err(L::mismatch("scan graph plan missing"));
                };
                let output = if *prefix == Prefix::None {
                    None
                } else {
                    candidate.output.as_deref()
                };
                let borrowed = inputs::<L>(live, &candidate.arrays, output)?;
                let seed = live
                    .scalars
                    .get(&candidate.state)
                    .copied()
                    .ok_or_else(|| L::unavailable("live scan seed is unavailable"))?;
                let slices = borrowed.iter().map(|a| &a[..len]).collect::<Vec<_>>();
                let result = launch.collective(plan, &slices, seed)?;
                Ok((
                    Step {
                        plan: LoopOverride {
                            iterations: len,
                            input_lengths: lengths(&candidate.arrays, &borrowed),
                            scalar: Some(vec![(candidate.state.clone(), seed, result.final_state)]),
                            array: output.map(|name| (name.to_owned(), result.prefixes)),
                            ..Default::default()
                        },
                        kind: PlanKind::Scan,
                        chunks: Vec::new(),
                        joined: 0,
                    },
                    result.cache_hit,
                ))
            }
            _ => Err(L::unavailable(format!(
                "region has no independent {} kernel",
                L::NAME
            ))),
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(super) mod metal {
    use super::*;
    use crate::schedule::metal::{Error, Fault};
    use crate::{
        device_cache::{self, HintState},
        device_ir,
        plan::{DeviceCost, DeviceCostModel},
    };
    use invar_device::{Capabilities, Provenance, Verified};
    use invar_metal::MetalDevice;
    pub(crate) fn cost(
        region: &Region,
        live: &LiveLoop<'_>,
        caps: Capabilities,
        model: DeviceCostModel,
    ) -> Option<DeviceCost> {
        estimate(
            &verified(region, live.iterations, caps).ok()?,
            live.iterations,
            model,
        )
    }

    /// None here: `metal_aot::run_kernels` builds every core region from
    /// admission, because its MSL does not depend on the reached length.
    pub(crate) fn kernels(
        _: &Region,
        _: &LiveLoop<'_>,
        _: Capabilities,
    ) -> Vec<(String, Vec<String>)> {
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

    struct Launcher<'a> {
        device: &'a mut MetalDevice,
        fault: Fault,
    }

    impl Launch for Launcher<'_> {
        type Error = Error;
        const NAME: &'static str = "Metal";
        fn unavailable(message: impl Into<String>) -> Error {
            Error::unavailable(message)
        }
        fn mismatch(message: impl Into<String>) -> Error {
            Error::mismatch(message)
        }
        fn fused(
            &mut self,
            plan: &Verified,
            inputs: &[&[u32]],
        ) -> Result<(Vec<Vec<u32>>, bool), Error> {
            let result = self
                .device
                .run_fused_map_with_fault(plan, inputs, self.fault)
                .map_err(|error| Error::from_metal(error, self.device.is_poisoned()))?;
            Ok((result.outputs, result.cache_hit))
        }
        fn map(&mut self, plan: &Verified, input: &[u32]) -> Result<(Vec<u32>, bool), Error> {
            let result = self
                .device
                .run_map_with_fault(plan, input, self.fault)
                .map_err(|error| Error::from_metal(error, self.device.is_poisoned()))?;
            Ok((result.values, result.cache_hit))
        }
        fn collective(
            &mut self,
            plan: &Verified,
            inputs: &[&[u32]],
            seed: u32,
        ) -> Result<Collected, Error> {
            let result = self
                .device
                .run_collective_with_fault(plan, inputs, seed, self.fault)
                .map_err(|error| Error::from_metal(error, self.device.is_poisoned()))?;
            Ok(Collected {
                final_state: result.final_state,
                pair: result.pair,
                prefixes: result.prefixes,
                cache_hit: result.cache_hit,
            })
        }
    }

    fn independent_step(
        source: &[u8],
        region: &Region,
        live: &LiveLoop<'_>,
        pending: &mut VecDeque<Pending>,
        device: &mut MetalDevice,
        fault: Fault,
    ) -> Result<(Step, bool, HintState), Error> {
        let plan = verified(region, live.iterations, device.capabilities())?;
        let artifact = invar_device::emit_msl(&plan)
            .map_err(|error| Error::mismatch(format!("Metal emission failed: {error:?}")))?;
        let hint = device_cache::expected(source, region, &plan, &artifact);
        let hint_state = device_cache::load(&hint);
        let result = independent(
            &mut Launcher { device, fault },
            &plan,
            region,
            live,
            pending,
        );
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
        estimate(
            &verified(region, live.iterations, caps).ok()?,
            live.iterations,
            model,
        )
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

    struct Launcher<'a> {
        device: &'a mut CudaDevice,
        compiler: &'a Path,
        fault: Fault,
    }

    impl Launch for Launcher<'_> {
        type Error = Error;
        const NAME: &'static str = "CUDA";
        fn unavailable(message: impl Into<String>) -> Error {
            Error::unavailable(message)
        }
        fn mismatch(message: impl Into<String>) -> Error {
            Error::mismatch(message)
        }
        fn fused(
            &mut self,
            plan: &Verified,
            inputs: &[&[u32]],
        ) -> Result<(Vec<Vec<u32>>, bool), Error> {
            let result = self
                .device
                .run_fused_map_with_fault(plan, inputs, self.compiler, self.fault.fused())
                .map_err(|error| Error::from_cuda(error, self.device.is_poisoned()))?;
            Ok((result.outputs, result.cache_hit))
        }
        fn map(&mut self, plan: &Verified, input: &[u32]) -> Result<(Vec<u32>, bool), Error> {
            let result = self
                .device
                .run_map_with_fault(plan, input, self.compiler, self.fault.fused())
                .map_err(|error| Error::from_cuda(error, self.device.is_poisoned()))?;
            Ok((result.values, result.cache_hit))
        }
        fn collective(
            &mut self,
            plan: &Verified,
            inputs: &[&[u32]],
            seed: u32,
        ) -> Result<Collected, Error> {
            let result = self
                .device
                .run_collective_with_fault(
                    plan,
                    inputs,
                    seed,
                    self.compiler,
                    self.fault.collective(),
                )
                .map_err(|error| Error::from_cuda(error, self.device.is_poisoned()))?;
            Ok(Collected {
                final_state: result.final_state,
                pair: result.pair,
                prefixes: result.prefixes,
                cache_hit: result.cache_hit,
            })
        }
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
        let plan = verified(region, live.iterations, device.capabilities())?;
        let artifact = invar_device::emit_cuda(&plan)
            .map_err(|error| Error::mismatch(format!("CUDA emission failed: {error:?}")))?;
        let hint = device_cache::expected(source, region, &plan, &artifact);
        let hint_state = device_cache::load(&hint);
        let launch = &mut Launcher {
            device,
            compiler,
            fault,
        };
        let result = independent(launch, &plan, region, live, pending);
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
