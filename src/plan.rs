//! One planning decision per loop region, shared by `explain` and the CPU scheduler.

use std::{collections::HashMap, sync::Arc};

use serde_json::{Value, json};

use crate::{
    admission::{self, Obligation, Region, Status, fact},
    front::Span,
    laws, loops,
    mir::{self, Function, Operation, ValueId},
    recurrence, typed_net,
};

/// The executed plan family of one region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanKind {
    Serial,
    Map,
    FusedMap,
    Reduction,
    Scan,
    // M4 labels are reserved here so family adapters only edit their registry line.
    Segmented,
    Compaction,
    Indexed,
    Transpose,
    Stencil,
    Tensor,
    Summary,
    Sort,
    Spatial,
}

/// Transfer-inclusive estimates from live, per-host calibration samples.
/// `device_bytes_per_ms` measures a host/device round trip, so it deliberately
/// overestimates device cost for a resident kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeviceCostModel {
    pub cpu_bytes_per_ms: f64,
    pub device_bytes_per_ms: f64,
    pub device_fixed_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeviceCost {
    pub cpu_ms: f64,
    pub device_ms: f64,
    pub choose_device: bool,
}

impl DeviceCostModel {
    pub fn estimate(
        self,
        transfer_bytes: usize,
        elements: usize,
        operations: usize,
        launches: usize,
    ) -> DeviceCost {
        let valid = self.cpu_bytes_per_ms.is_finite()
            && self.device_bytes_per_ms.is_finite()
            && self.device_fixed_ms.is_finite()
            && self.cpu_bytes_per_ms > 0.0
            && self.device_bytes_per_ms > 0.0
            && self.device_fixed_ms >= 0.0;
        if !valid {
            return DeviceCost {
                cpu_ms: f64::INFINITY,
                device_ms: f64::INFINITY,
                choose_device: false,
            };
        }
        let bytes = transfer_bytes as f64;
        let compute = elements as f64 * 4.0;
        let ops = operations.max(1) as f64;
        let cpu_ms = (bytes + compute * ops) / self.cpu_bytes_per_ms;
        let device_ms = self.device_fixed_ms * launches.max(1) as f64
            + bytes / self.device_bytes_per_ms
            + compute * (1.0 + ops / 8.0) / self.device_bytes_per_ms;
        DeviceCost {
            cpu_ms,
            device_ms,
            choose_device: device_ms * 1.1 < cpu_ms,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Execution {
    Serial,
    Map(Box<loops::LiveMap>),
    /// Computes its own output and the consumer's fused output in one pass.
    FusedProducer(Box<loops::Fused>),
    /// Publishes the producer's staged output; runs `standalone` if that guard fails.
    FusedConsumer {
        producer: Span,
        standalone: Box<loops::LiveMap>,
    },
    DagProducer(Box<loops::FusedChain>),
    DagConsumer {
        producer: Span,
        stage: usize,
        standalone: Box<loops::LiveMap>,
    },
    Reduction(Box<laws::Reduction>),
    PairReduction(Box<crate::pair_reduction::Reduction>),
    Scan(Box<recurrence::Candidate>),
}

impl Execution {
    /// Analysis bytes this plan retains beyond its region (`Serial` holds nothing).
    pub(crate) fn held_bytes(&self) -> usize {
        match self {
            Self::Serial => 0,
            Self::Map(map)
            | Self::FusedConsumer {
                standalone: map, ..
            }
            | Self::DagConsumer {
                standalone: map, ..
            } => map.held_bytes(),
            Self::FusedProducer(fused) => fused.held_bytes(),
            Self::DagProducer(chain) => chain.held_bytes(),
            Self::Reduction(reduction) => size_of::<laws::Reduction>()
                .saturating_add(loops::net_bytes(&reduction.graph))
                .saturating_add(reduction.input.capacity())
                .saturating_add(reduction.state.capacity()),
            Self::PairReduction(reduction) => size_of::<crate::pair_reduction::Reduction>()
                .saturating_add(loops::net_bytes(&reduction.graph))
                .saturating_add(loops::names_bytes(&reduction.states))
                .saturating_add(loops::names_bytes(&reduction.inputs)),
            Self::Scan(scan) => size_of::<recurrence::Candidate>()
                .saturating_add(loops::net_bytes(&scan.graph))
                .saturating_add(scan.state.capacity())
                .saturating_add(loops::names_bytes(&scan.arrays))
                .saturating_add(scan.output.as_ref().map_or(0, String::capacity)),
        }
    }

    pub fn kind(&self) -> PlanKind {
        match self {
            Self::Serial => PlanKind::Serial,
            Self::Map(_) => PlanKind::Map,
            Self::FusedProducer(_)
            | Self::FusedConsumer { .. }
            | Self::DagProducer(_)
            | Self::DagConsumer { .. } => PlanKind::FusedMap,
            Self::Reduction(_) | Self::PairReduction(_) => PlanKind::Reduction,
            Self::Scan(_) => PlanKind::Scan,
        }
    }

    pub fn rules(&self) -> Vec<&'static str> {
        match self {
            Self::Serial | Self::Map(_) => vec![admission::RULE_REVISION],
            Self::FusedProducer(_) | Self::FusedConsumer { .. } => {
                vec![admission::RULE_REVISION, typed_net::MAP_FUSION_RULE]
            }
            Self::DagProducer(_) | Self::DagConsumer { .. } => {
                vec![admission::RULE_REVISION, typed_net::DAG_FUSION_RULE]
            }
            Self::Reduction(_) => vec![laws::RULE_REVISION],
            Self::PairReduction(_) => {
                vec![crate::pair_reduction::RULE_REVISION, laws::RULE_REVISION]
            }
            Self::Scan(_) => vec![recurrence::RULE_REVISION, laws::RULE_REVISION],
        }
    }

    pub fn chosen_plan(&self) -> &'static str {
        match self {
            Self::Serial => "mir-serial",
            Self::Map(_) => "wrapping-map-candidate-if-guard",
            Self::FusedProducer(_) => "fused-wrapping-map-if-guard",
            Self::FusedConsumer { .. } => "fused-wrapping-map-consumer-if-guard",
            Self::DagProducer(_) => "fused-wrapping-map-if-guard",
            Self::DagConsumer { .. } => "fused-wrapping-map-consumer-if-guard",
            Self::Reduction(_) => "wrapping-reduction-if-guard",
            Self::PairReduction(_) => "ordered-affine-pair-reduction-if-guard",
            Self::Scan(_) => "affine-scan-if-guard",
        }
    }

    pub(crate) fn json(&self) -> Value {
        let span = |span: Span| json!({"start": span.start, "end": span.end});
        let (kind, partner, law, prefix) = match self {
            Self::Serial => ("serial", None, None, None),
            Self::Map(_) => ("map", None, None, None),
            Self::FusedProducer(fused) => (
                "fused-map-producer",
                Some(span(fused.source_spans[1])),
                None,
                None,
            ),
            Self::FusedConsumer { producer, .. } => {
                ("fused-map-consumer", Some(span(*producer)), None, None)
            }
            Self::DagProducer(chain) => (
                "fused-map-producer",
                chain.source_spans.get(1).copied().map(span),
                None,
                None,
            ),
            Self::DagConsumer { producer, .. } => {
                ("fused-map-consumer", Some(span(*producer)), None, None)
            }
            Self::Reduction(reduction) => (
                "reduction",
                None,
                Some(match reduction.law {
                    laws::Law::WrappingSum => "wrapping-sum",
                    laws::Law::WrappingProduct => "wrapping-product",
                }),
                None,
            ),
            Self::PairReduction(_) => ("pair-reduction", None, Some("ordered-affine-pair"), None),
            Self::Scan(candidate) => (
                "scan",
                None,
                None,
                Some(
                    match (candidate.publishes_prefixes, candidate.exclusive_prefix) {
                        (false, _) => "none",
                        (true, false) => "inclusive",
                        (true, true) => "exclusive",
                    },
                ),
            ),
        };
        let intermediate = match self {
            Self::FusedProducer(fused) if fused.elide_intermediate => Some("elided"),
            Self::FusedProducer(_) => Some("materialized"),
            Self::DagProducer(_) => Some("materialized"),
            _ => None,
        };
        json!({"kind": kind, "partner": partner, "law": law, "prefix": prefix, "intermediate": intermediate})
    }
}

fn admit(region: &mut Region, execution: Execution, obligations: Vec<Obligation>) {
    region.status = Status::Admitted;
    region.first_rejected_obligation = None;
    region.guard = None;
    region.obligations = obligations;
    region.execution = execution;
}

fn family_facts(dependence: &'static str, publication: Option<&'static str>) -> Vec<Obligation> {
    let mut facts = vec![
        fact(
            "RANGE",
            "checked_invariant",
            "zero-based counted for; trip count and live input lengths checked at loop entry",
        ),
        fact("DEPENDENCE", "checked_invariant", dependence),
        fact(
            "TOTALITY",
            "trusted_rule",
            "sealed wrapping u32 operations do not trap",
        ),
    ];
    if let Some(detail) = publication {
        facts.push(fact("PUBLICATION", "checked_invariant", detail));
    }
    facts.push(fact(
        "SEED",
        "runtime_guard",
        "seed read from the live state at loop entry; no write before the region completes",
    ));
    facts
}

fn scan_facts(candidate: &recurrence::Candidate) -> Vec<Obligation> {
    let publication = match (candidate.publishes_prefixes, candidate.exclusive_prefix) {
        (false, _) => None,
        (true, false) => Some("one store per step of the state after the update"),
        (true, true) => Some("one store per step of the state before the update"),
    };
    family_facts(
        "one state updated by state-independent affine coefficients",
        publication,
    )
}

/// Per-analysis context shared by every region: one definition map per function
/// and one fallback program for all recurrence candidates.
struct Context<'a> {
    program: &'a mir::Program,
    definitions: HashMap<&'a str, HashMap<ValueId, &'a Operation>>,
    fallback: Option<Arc<mir::Program>>,
}

impl<'a> Context<'a> {
    fn function(&self, name: &str) -> Option<&'a Function> {
        self.program
            .functions
            .iter()
            .find(|function| function.name == name)
    }
}

fn classify(context: &mut Context<'_>, region: &mut Region) {
    if region.status == Status::Admitted {
        match loops::single(region) {
            Ok(map) => region.execution = Execution::Map(Box::new(map)),
            Err(_) => {
                region.status = Status::Rejected;
                region.first_rejected_obligation = Some("EXTRACTION");
                region.obligations.push(fact(
                    "EXTRACTION",
                    "unknown",
                    "typed graph extraction failed",
                ));
            }
        }
        return;
    }
    let Some(function) = context.function(region.function()) else {
        return;
    };
    let program = context.program;
    let defs = context
        .definitions
        .entry(function.name.as_str())
        .or_insert_with(|| crate::counted::definitions(function));
    let fallback = context
        .fallback
        .get_or_insert_with(|| Arc::new(program.clone()));
    if let Ok(pair) = crate::pair_reduction::reduction_in(function, defs, region.head) {
        let facts = family_facts(
            "two components composed by the sealed ordered affine-pair law; identity (1,0)",
            None,
        );
        admit(region, Execution::PairReduction(Box::new(pair)), facts);
        return;
    }
    let scan = recurrence::candidate_in(fallback, function, defs, region.head);
    if let Ok(candidate) = scan {
        let facts = scan_facts(&candidate);
        admit(region, Execution::Scan(Box::new(candidate)), facts);
    } else if let Some(reduction) = laws::reduction_in(function, defs, region.head) {
        let facts = family_facts("one accumulator combined by a sealed law", None);
        admit(region, Execution::Reduction(Box::new(reduction)), facts);
    } else if let Err(reason) = scan
        && region.first_rejected_obligation == Some("EFFECT_ORDER")
        && !matches!(reason, "RANGE" | "EFFECT_ORDER")
    {
        // A scalar-state loop the map rule cannot describe: the recurrence rule's
        // failure is the more precise first rejected obligation.
        region.first_rejected_obligation = Some(reason);
        region.obligations.pop();
        region.obligations.push(fact(
            reason,
            "unknown",
            "required recurrence fact not established for this region",
        ));
    }
}

fn fuse(program: &mir::Program, first: &mut Region, second: &mut Region) -> bool {
    if !matches!(
        (&first.execution, &second.execution),
        (Execution::Map(_), Execution::Map(_))
    ) || first.function() != second.function()
    {
        return false;
    }
    let Some(fused) = program
        .functions
        .iter()
        .find(|function| function.name == first.function())
        .and_then(|function| loops::fuse_pair(function, first, second).ok().flatten())
    else {
        return false;
    };
    let Execution::Map(standalone) = std::mem::replace(&mut second.execution, Execution::Serial)
    else {
        return false;
    };
    second.execution = Execution::FusedConsumer {
        producer: first.span(),
        standalone,
    };
    first.execution = Execution::FusedProducer(Box::new(fused));
    let segment = fact(
        "FUSION",
        "checked_invariant",
        "producer exit reaches consumer head by a straight-line segment that does not write the intermediate",
    );
    let staged = fact(
        "FUSION_GUARD",
        "runtime_guard",
        "consumer input equals the published intermediate before the staged output is written",
    );
    first.obligations.extend([segment.clone(), staged.clone()]);
    second.obligations.extend([segment, staged]);
    true
}

fn fuse_dag(program: &mir::Program, regions: &mut [Region]) -> usize {
    let max = regions.len().min(loops::MAX_FUSED_STAGES);
    for count in (2..=max).rev() {
        let run = &regions[..count];
        if !run
            .iter()
            .all(|region| matches!(region.execution, Execution::Map(_)))
        {
            continue;
        }
        let Some(function) = program
            .functions
            .iter()
            .find(|function| function.name == run[0].function())
        else {
            continue;
        };
        // Preserve the M2 pair path, including its proven intermediate elision.
        if count == 2
            && loops::fuse_pair(function, &run[0], &run[1])
                .ok()
                .flatten()
                .is_some()
        {
            continue;
        }
        let Some(chain) = loops::fuse_chain(function, run).ok().flatten() else {
            continue;
        };
        let producer = run[0].span();
        for (stage, region) in regions[1..count].iter_mut().enumerate() {
            let Execution::Map(standalone) =
                std::mem::replace(&mut region.execution, Execution::Serial)
            else {
                return 0;
            };
            region.execution = Execution::DagConsumer {
                producer,
                stage: stage + 1,
                standalone,
            };
            region.obligations.push(fact(
                "FUSION_GUARD",
                "runtime_guard",
                "staged input prefix must still match the prior published stage",
            ));
        }
        regions[0].execution = Execution::DagProducer(Box::new(chain));
        regions[0].obligations.push(fact(
            "FUSION",
            "checked_invariant",
            "straight source-map chain has equal trip counts and no intervening writes to each dependency",
        ));
        return count;
    }
    0
}

/// Choose each region's plan: admitted maps (fused in pairs where legal), then
/// affine scans, then lawful reductions; everything else stays MIR serial.
pub(crate) fn decide(program: &mir::Program, regions: &mut [Region]) {
    let mut context = Context {
        program,
        definitions: HashMap::new(),
        fallback: None,
    };
    for region in regions.iter_mut() {
        classify(&mut context, region);
    }
    let mut index = 0;
    while index + 1 < regions.len() {
        let dag = fuse_dag(program, &mut regions[index..]);
        if dag > 0 {
            index += dag;
            continue;
        }
        let (left, right) = regions.split_at_mut(index + 1);
        index += if fuse(program, &mut left[index], &mut right[0]) {
            2
        } else {
            1
        };
    }
}
