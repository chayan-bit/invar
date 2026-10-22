//! Bounded, deterministic schedule admission and backend-independent placement.
//! No measurements or searches run on production output. Estimates are not timings.

use invar_device::{Capabilities, Geometry, Provenance, Verified};

use crate::{
    admission::Region,
    device_ir,
    plan::{DeviceCost, DeviceCostModel},
};

pub const MAX_VARIANTS: usize = 32;
pub const SCHEDULE_VERSION: &str = "bounded-schedule-v1";
pub const SEED: u64 = 0x494e_5641_5206;
pub const WARMUPS: usize = 10;
pub const SAMPLES: usize = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    TooManyVariants,
    NoCandidates,
    UnsupportedSchedule,
    Admission,
    NumericPolicy,
    StaleOrForgedHint,
    UnknownCost,
    Guard,
    MissingDevice,
    Transfer,
    Launch,
    PartialCompletion,
    ProductionOutput,
    RuntimeTuning,
    FreezeRequired,
    Budget,
}

impl Refusal {
    pub fn code(self) -> &'static str {
        match self {
            Self::TooManyVariants => "TUNING_VARIANT_LIMIT",
            Self::NoCandidates => "TUNING_NO_CANDIDATES",
            Self::UnsupportedSchedule => "TUNING_UNSUPPORTED_SCHEDULE",
            Self::Admission => "TUNING_ADMISSION",
            Self::NumericPolicy => "TUNING_NUMERIC_POLICY",
            Self::StaleOrForgedHint => "TUNING_HINT_INVALID",
            Self::UnknownCost => "TUNING_COST_UNKNOWN",
            Self::Guard => "TUNING_GUARD_FALSE",
            Self::MissingDevice => "TUNING_DEVICE_UNAVAILABLE",
            Self::Transfer => "TUNING_TRANSFER_FAILED",
            Self::Launch => "TUNING_LAUNCH_FAILED",
            Self::PartialCompletion => "TUNING_COMPLETION_UNKNOWN",
            Self::ProductionOutput => "TUNING_PRODUCTION_OUTPUT",
            Self::RuntimeTuning => "TUNING_RUNTIME_FORBIDDEN",
            Self::FreezeRequired => "TUNING_FREEZE_REQUIRED",
            Self::Budget => "TUNING_BUDGET",
        }
    }
}

/// M3 supports scalar vectors and a one-dimensional tile. Other axes are
/// explicit refusals until a family adapter supplies its verified device path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub geometry: Geometry,
    pub tile: [u32; 3],
    pub vector_width: u32,
    pub fusion_cutoff: Option<usize>,
    pub frontier_chunk: Option<u32>,
    pub cpu_threshold: usize,
}

impl Schedule {
    pub fn m3(geometry: Geometry) -> Self {
        Self {
            geometry,
            tile: [geometry.lanes, 1, 1],
            vector_width: 1,
            fusion_cutoff: None,
            frontier_chunk: None,
            cpu_threshold: 0,
        }
    }
}

pub struct Candidate {
    index: usize,
    schedule: Schedule,
    plan: Verified,
}

impl Candidate {
    pub fn index(&self) -> usize {
        self.index
    }
    pub fn schedule(&self) -> Schedule {
        self.schedule
    }
    pub fn plan(&self) -> &Verified {
        &self.plan
    }
    pub fn into_plan(self) -> Verified {
        self.plan
    }
}

/// Fixed caller order, no RNG, no compile or device work. Observe at most 33
/// entries, even from an infinite iterator. Every variant is freshly verified.
pub fn enumerate(
    region: &Region,
    length: u32,
    caps: Capabilities,
    schedules: impl IntoIterator<Item = Schedule>,
) -> Result<Vec<Candidate>, Refusal> {
    let schedules = schedules
        .into_iter()
        .take(MAX_VARIANTS + 1)
        .collect::<Vec<_>>();
    if schedules.len() > MAX_VARIANTS {
        return Err(Refusal::TooManyVariants);
    }
    if schedules.is_empty() {
        return Err(Refusal::NoCandidates);
    }
    let base = device_ir::from_region(region, length).map_err(|_| Refusal::Admission)?;
    let origin = Provenance {
        source_identity: base.source_identity,
        graph_revision: base.graph_revision,
    };
    let mut candidates = Vec::with_capacity(schedules.len());
    for (index, schedule) in schedules.into_iter().enumerate() {
        if schedule.vector_width != 1
            || schedule.tile != [schedule.geometry.lanes, 1, 1]
            || schedule.fusion_cutoff.is_some()
            || schedule.frontier_chunk.is_some()
        {
            return Err(Refusal::UnsupportedSchedule);
        }
        let mut ir = base.clone();
        ir.geometry = schedule.geometry;
        let plan = invar_device::verify(ir, origin, caps).map_err(|error| {
            if error == invar_device::Error::Numeric {
                Refusal::NumericPolicy
            } else {
                Refusal::Admission
            }
        })?;
        candidates.push(Candidate {
            index,
            schedule,
            plan,
        });
    }
    Ok(candidates)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Location {
    Host,
    Device,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Residency {
    pub inputs: Location,
    pub output: Location,
}

impl Residency {
    pub const HOST: Self = Self {
        inputs: Location::Host,
        output: Location::Host,
    };
    pub const DEVICE: Self = Self {
        inputs: Location::Device,
        output: Location::Device,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completion {
    NotSubmitted,
    Completed,
    FailedQuiescent,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Track {
    Generated,
    LibraryAssisted,
}

/// All three boundaries are unmeasured by default. Cold process includes
/// compilation, allocation, transfers and completion; never infer it from a kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    pub steady_state_ms: Option<f64>,
    pub warm_host_to_host_ms: Option<f64>,
    pub cold_process_ms: Option<f64>,
}

/// Descriptive facts, not a publication permit. Residency must describe the
/// actual execution API; ordinary MIR adapters currently publish host buffers.
#[derive(Clone, Debug, PartialEq)]
pub struct CostRecord {
    pub shape: Vec<usize>,
    pub layout: String,
    pub numeric: String,
    pub elements: usize,
    pub operations: usize,
    pub model: DeviceCostModel,
    pub residency: Residency,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub transfer_bytes: usize,
    pub launches: usize,
    pub temporary_bytes: usize,
    pub completion: Completion,
    pub timings: Timings,
    pub track: Track,
    pub estimate: DeviceCost,
}

impl CostRecord {
    /// M3 facts from a physical capability, including backend-specific
    /// collective workspace and the real chunked Metal scan launch count.
    pub fn from_plan(
        plan: &Verified,
        residency: Residency,
        model: DeviceCostModel,
    ) -> Result<Self, Refusal> {
        let ir = plan.ir();
        let n = ir.length as usize;
        let facts = plan.cost_facts().ok_or(Refusal::UnknownCost)?;
        let input_bytes = facts.input_bytes;
        let output_bytes = facts.output_bytes;
        let all_bytes = input_bytes
            .checked_add(output_bytes)
            .ok_or(Refusal::UnknownCost)?;
        let transfer_bytes = (if residency.inputs == Location::Host {
            input_bytes
        } else {
            0usize
        })
        .checked_add(if residency.output == Location::Host {
            output_bytes
        } else {
            0
        })
        .ok_or(Refusal::UnknownCost)?;
        let mut estimate = model.estimate(transfer_bytes, n, facts.operations, facts.launches);
        // CPU always computes the same work; resident input additionally needs
        // retrieval. Removing H2D must never make the CPU's computation cheaper.
        estimate.cpu_ms = model
            .estimate(all_bytes, n, facts.operations, facts.launches)
            .cpu_ms;
        if residency.inputs == Location::Device {
            estimate.cpu_ms += input_bytes as f64 / model.device_bytes_per_ms;
        }
        estimate.choose_device = estimate.device_ms * 1.1 < estimate.cpu_ms;
        let record = Self {
            shape: vec![n],
            layout: format!("{:?}", ir.input.layout),
            numeric: format!("{:?}", ir.numeric),
            elements: n,
            operations: facts.operations,
            model,
            residency,
            input_bytes,
            output_bytes,
            transfer_bytes,
            launches: facts.launches,
            temporary_bytes: facts.temporary_bytes,
            completion: Completion::NotSubmitted,
            timings: Timings::default(),
            track: Track::Generated,
            estimate,
        };
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<(), Refusal> {
        if self.shape.is_empty()
            || self.shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d)) != Some(self.elements)
            || self.layout.is_empty()
            || self.numeric.is_empty()
            || !self.estimate.cpu_ms.is_finite()
            || !self.estimate.device_ms.is_finite()
            || self.estimate.cpu_ms < 0.0
            || self.estimate.device_ms < 0.0
            || [
                self.timings.steady_state_ms,
                self.timings.warm_host_to_host_ms,
                self.timings.cold_process_ms,
            ]
            .into_iter()
            .flatten()
            .any(|ms| !ms.is_finite() || ms < 0.0)
        {
            return Err(Refusal::UnknownCost);
        }
        let bytes = (if self.residency.inputs == Location::Host {
            self.input_bytes
        } else {
            0usize
        })
        .checked_add(if self.residency.output == Location::Host {
            self.output_bytes
        } else {
            0
        })
        .ok_or(Refusal::UnknownCost)?;
        if self.transfer_bytes != bytes {
            return Err(Refusal::UnknownCost);
        }
        let expected = self.model.estimate(
            self.transfer_bytes,
            self.elements,
            self.operations,
            self.launches,
        );
        let all_bytes = self
            .input_bytes
            .checked_add(self.output_bytes)
            .ok_or(Refusal::UnknownCost)?;
        let mut cpu_ms = self
            .model
            .estimate(all_bytes, self.elements, self.operations, self.launches)
            .cpu_ms;
        if self.residency.inputs == Location::Device {
            cpu_ms += self.input_bytes as f64 / self.model.device_bytes_per_ms;
        }
        if !expected.device_ms.is_finite()
            || !cpu_ms.is_finite()
            || expected.device_ms != self.estimate.device_ms
            || cpu_ms != self.estimate.cpu_ms
        {
            return Err(Refusal::UnknownCost);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Auto,
    Serial,
    Cpu,
    Gpu,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Serial,
    Cpu,
    Gpu,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    pub engine: Engine,
    pub variant: Option<usize>,
    pub refusal: Option<Refusal>,
}

/// Selection cannot authorize replay after unknown completion. Forced GPU
/// refuses missing facts/guards/devices; automatic dispatch retains lawful CPU.
pub fn decide(
    control: Control,
    record: Option<&CostRecord>,
    variant: usize,
    cpu_threshold: usize,
    guard: bool,
    device_available: bool,
) -> Result<Decision, Refusal> {
    if record.is_some_and(|r| r.completion == Completion::Unknown) {
        return Err(Refusal::PartialCompletion);
    }
    if control == Control::Serial {
        return Ok(Decision {
            engine: Engine::Serial,
            variant: None,
            refusal: None,
        });
    }
    let refusal = if !guard {
        Some(Refusal::Guard)
    } else if control == Control::Cpu {
        None
    } else if !device_available {
        Some(Refusal::MissingDevice)
    } else if record.is_none_or(|r| r.validate().is_err()) {
        Some(Refusal::UnknownCost)
    } else {
        None
    };
    if control == Control::Gpu
        && let Some(refusal) = refusal
    {
        return Err(refusal);
    }
    let gpu = control == Control::Gpu
        || (control == Control::Auto
            && refusal.is_none()
            && record.is_some_and(|r| {
                r.estimate.device_ms * 1.1 < r.estimate.cpu_ms
                    && !(r.residency.inputs == Location::Host
                        && r.shape
                            .iter()
                            .try_fold(1usize, |n, d| n.checked_mul(*d))
                            .is_none_or(|n| n <= cpu_threshold))
            }));
    Ok(Decision {
        engine: if !guard {
            Engine::Serial
        } else if gpu {
            Engine::Gpu
        } else {
            Engine::Cpu
        },
        variant: gpu.then_some(variant),
        refusal,
    })
}

/// Only confirmed failure permits a fresh CPU fallback; no partial output is
/// published and no unknown submission is replayed.
pub fn failed(
    control: Control,
    refusal: Refusal,
    completion: Completion,
) -> Result<Decision, Refusal> {
    if completion == Completion::Unknown || refusal == Refusal::PartialCompletion {
        return Err(Refusal::PartialCompletion);
    }
    if completion != Completion::FailedQuiescent && completion != Completion::NotSubmitted {
        return Err(Refusal::Admission);
    }
    if control == Control::Gpu {
        return Err(refusal);
    }
    Ok(Decision {
        engine: if control == Control::Serial {
            Engine::Serial
        } else {
            Engine::Cpu
        },
        variant: None,
        refusal: Some(refusal),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuningUse {
    Runtime,
    ProductionVisible,
    OfflineTuning,
}

/// Before the permanent freeze, enumeration and synthetic correctness checks
/// work, but actual schedule measurement cannot start. A later M5 change must
/// wire its reviewed freeze into this boundary rather than accepting a flag.
pub fn measurement_gate(use_: TuningUse) -> Result<(), Refusal> {
    Err(match use_ {
        TuningUse::Runtime => Refusal::RuntimeTuning,
        TuningUse::ProductionVisible => Refusal::ProductionOutput,
        TuningUse::OfflineTuning => Refusal::FreezeRequired,
    })
}

/// Enforced limits for a future authorized offline driver. No runtime search.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub compile_ms: u64,
    pub tuning_ms: u64,
    pub temporary_bytes: usize,
}

impl Budget {
    pub fn check(
        &self,
        compile_ms: u64,
        tuning_ms: u64,
        temporary_bytes: usize,
    ) -> Result<(), Refusal> {
        if self.compile_ms == 0
            || self.tuning_ms == 0
            || compile_ms > self.compile_ms
            || tuning_ms > self.tuning_ms
            || temporary_bytes > self.temporary_bytes
        {
            Err(Refusal::Budget)
        } else {
            Ok(())
        }
    }
}

/// Map placement used by host and retained-buffer callers. No device work or
/// timing occurs here. A resident choice is meaningful only with a real handle.
pub fn select_map(
    plan: &Verified,
    residency: Residency,
    model: DeviceCostModel,
    control: Control,
    cpu_threshold: usize,
) -> Result<(Decision, CostRecord), Refusal> {
    if plan.ir().collective.is_some() {
        return Err(Refusal::UnsupportedSchedule);
    }
    let record = CostRecord::from_plan(plan, residency, model)?;
    let decision = decide(control, Some(&record), 0, cpu_threshold, true, true)?;
    Ok((decision, record))
}

impl CostRecord {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({"shape": self.shape, "layout": self.layout, "numeric": self.numeric,
            "elements": self.elements, "operations": self.operations,
            "model": {"cpu_bytes_per_ms": self.model.cpu_bytes_per_ms,
                "device_bytes_per_ms": self.model.device_bytes_per_ms, "device_fixed_ms": self.model.device_fixed_ms},
            "residency": {"inputs": format!("{:?}", self.residency.inputs), "output": format!("{:?}", self.residency.output)},
            "transfer_bytes": self.transfer_bytes, "input_bytes": self.input_bytes, "output_bytes": self.output_bytes,
            "launches": self.launches, "temporary_bytes": self.temporary_bytes, "completion": format!("{:?}", self.completion),
            "track": match self.track { Track::Generated => "generated", Track::LibraryAssisted => "library-assisted" },
            "timing_ms": {"steady_state": self.timings.steady_state_ms, "warm_host_to_host": self.timings.warm_host_to_host_ms, "cold_process": self.timings.cold_process_ms},
            "estimate_ms": {"cpu": self.estimate.cpu_ms, "device": self.estimate.device_ms}})
    }
}

pub fn receipt(candidates: &[Candidate], budget: Budget) -> serde_json::Value {
    serde_json::json!({"schedule_version": SCHEDULE_VERSION, "max_variants": MAX_VARIANTS,
        "seed": SEED, "warmups": WARMUPS, "samples": SAMPLES,
        "compile_limit_ms": budget.compile_ms, "tuning_limit_ms": budget.tuning_ms,
        "temporary_limit_bytes": budget.temporary_bytes,
        "candidates": candidates.iter().map(|c| serde_json::json!({"index": c.index,
            "blocks": c.schedule.geometry.blocks, "lanes": c.schedule.geometry.lanes,
            "tile": c.schedule.tile, "vector_width": c.schedule.vector_width,
            "fusion_cutoff": c.schedule.fusion_cutoff, "frontier_chunk": c.schedule.frontier_chunk,
            "cpu_threshold": c.schedule.cpu_threshold})).collect::<Vec<_>>()})
}

/// Stable minimum estimate, then original enumeration index on ties. The
/// current analytic M3 model cannot distinguish equal-work geometry variants;
/// offline measurement may replace these hints only after the permanent freeze.
pub fn select(
    candidates: &[Candidate],
    residency: Residency,
    model: DeviceCostModel,
    control: Control,
    guard: bool,
    available: bool,
) -> Result<(Decision, Vec<CostRecord>), Refusal> {
    if candidates.len() > MAX_VARIANTS {
        return Err(Refusal::TooManyVariants);
    }
    if candidates.is_empty() {
        return Err(Refusal::NoCandidates);
    }
    let costs = candidates
        .iter()
        .map(|c| CostRecord::from_plan(c.plan(), residency, model))
        .collect::<Result<Vec<_>, _>>()?;
    let index = (0..costs.len())
        .min_by(|a, b| {
            costs[*a]
                .estimate
                .device_ms
                .total_cmp(&costs[*b].estimate.device_ms)
        })
        .ok_or(Refusal::NoCandidates)?;
    let choice = decide(
        control,
        Some(&costs[index]),
        candidates[index].index(),
        candidates[index].schedule().cpu_threshold,
        guard,
        available,
    )?;
    Ok((choice, costs))
}
