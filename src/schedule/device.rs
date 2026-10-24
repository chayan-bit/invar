//! Run records and auto bookkeeping shared by the CUDA and Metal schedulers.

use super::*;
use crate::plan::DeviceCost;
use reductions::Reduction;

/// A forced CUDA or Metal run.
pub struct Run {
    pub observation: Observation,
    pub executed: Vec<Executed>,
    /// Every reached float reduction, in execution order.
    pub reductions: Vec<Reduction>,
    pub cache_hits: usize,
    pub hint_hits: usize,
    pub hint_invalids: usize,
}

/// One auto placement decision.
#[derive(Clone, Debug)]
pub struct Choice {
    pub function: String,
    pub span: crate::front::Span,
    pub backend: &'static str,
    pub reason: &'static str,
    pub cost: Option<DeviceCost>,
    pub record: Option<tuning::CostRecord>,
}

/// An auto CUDA or Metal run.
pub struct AutoRun {
    pub observation: Observation,
    pub executed: Vec<Executed>,
    pub choices: Vec<Choice>,
    /// Every reached float reduction and where it ran, in execution order.
    pub reductions: Vec<Reduction>,
    pub cache_hits: usize,
    pub hint_hits: usize,
    pub hint_invalids: usize,
}

/// Auto on a program with spatial or sort intrinsics: those families are
/// unmeasured on devices, so every published region stays on the CPU.
pub(super) fn intrinsic_cpu_auto(cpu: super::Run, receipts: Vec<Reduction>) -> AutoRun {
    let choices = cpu
        .executed
        .iter()
        .map(|step| Choice {
            function: step.function.clone(),
            span: step.span,
            backend: "cpu-parallel",
            reason: if step.kind == PlanKind::Sort {
                "sort-cost-unmeasured"
            } else {
                crate::spatial::placement::UNMEASURED
            },
            cost: None,
            record: None,
        })
        .collect();
    AutoRun {
        observation: cpu.observation,
        executed: cpu.executed,
        reductions: receipts,
        choices,
        cache_hits: 0,
        hint_hits: 0,
        hint_invalids: 0,
    }
}

/// Offered choices are exactly the non-serial ones, in offer order; one whose
/// result the publication guard refused ran as serial MIR.
pub(super) fn settle_choices(choices: &mut [Choice], published: &[bool]) {
    let mut outcomes = published.iter();
    for choice in choices
        .iter_mut()
        .filter(|choice| choice.backend != "mir-serial")
    {
        if outcomes.next() != Some(&true) {
            choice.backend = "mir-serial";
            choice.reason = "publication-guard-false";
        }
    }
}

/// Median wall milliseconds of the real cpu-parallel element executor on a
/// one-operation map over `elements` words, used to calibrate `auto`.
pub(crate) fn cpu_map_ms(elements: usize, workers: usize) -> Option<f64> {
    let plan = Plan::LiveDag {
        inputs: vec![0],
        ops: vec![
            crate::typed_net::DagOp::Input(0),
            crate::typed_net::DagOp::Map {
                input: 0,
                op: crate::typed_net::MapOp::WrappingAdd(1),
            },
        ],
        outputs: vec![1],
        source_identity: 1,
        graph_revision: 0,
    };
    let input = vec![7u32; elements];
    let hooks = Hooks::default();
    let mut samples = Vec::new();
    for trial in 0..6 {
        let started = std::time::Instant::now();
        let parts = parallel_chunks(elements, workers, &hooks, |range| {
            plan.run_dag(&[&input], range).map_err(|_| Error::Candidate)
        })
        .ok()?;
        std::hint::black_box(&parts);
        if trial > 0 {
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    samples.sort_by(f64::total_cmp);
    samples.get(samples.len() / 2).copied()
}
