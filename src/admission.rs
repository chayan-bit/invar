//! Conservative, versioned admission facts for bounded source loops.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::{
    budget::{ByteBudget, Footprint, array_bytes},
    front::{BinaryOp, Mode, Span, Type},
    laws, loops,
    mir::{self, BlockId, Function, Operation, Terminator, ValueId},
    plan::{self, Execution},
    typed_net::{MapOp, ZipOp},
};

mod fold_hint;

pub const SCHEMA_VERSION: u32 = 1;
pub const RULE_REVISION: &str = "counted-wrapping-map-v1";
/// Largest serialized admission record or report: an interchange-format bound on the document, not
/// on analysis memory, so it stays (`ADMISSION_RECORD_LIMIT`).
const MAX_RECORD_BYTES: usize = 1_000_000;
const MAX_SOURCE_BYTES: usize = 1_048_576;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub code: &'static str,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Admitted,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Guard {
    upper: Upper,
    inputs: Vec<String>,
    output: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Upper {
    Constant(u32),
    Length(String),
}

/// The single element operation of an admitted map region, in source operand order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapShape {
    /// One indexed input combined with a constant.
    Unary(MapOp),
    /// Two indexed inputs, left then right.
    Zip(ZipOp),
}

impl Guard {
    pub(crate) fn upper(&self) -> &Upper {
        &self.upper
    }
    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }
    pub fn output(&self) -> &str {
        &self.output
    }
    /// Host bytes the map holds live at its peak for `len` elements: every input, the destination
    /// the loop already owns, the workers' chunk results and the joined values buffer built from
    /// them before publication (`schedule::live_pass`).
    pub fn footprint(&self, len: usize) -> Footprint {
        Footprint::ZERO.host(
            array_bytes::<u32>(len).and_then(|array| array.checked_mul(self.inputs.len() + 3)),
        )
    }
    /// `len` is within the length the CPU scheduler and the device IR execute
    /// (`laws::MAX_ELEMENTS`; they refuse a longer map and run it serially) and the peak
    /// footprint fits `budget`.
    fn fits(&self, len: usize, budget: &ByteBudget) -> bool {
        len <= laws::MAX_ELEMENTS && budget.admit("admission", &self.footprint(len)).is_ok()
    }
    /// Guard inputs must remain unchanged until the selected plan completes.
    pub fn evaluate(&self, lengths: &[(&str, usize)]) -> bool {
        self.evaluate_within(lengths, &ByteBudget::standard())
    }
    /// `evaluate`, with the elements the map would hold live admitted against `budget`.
    pub fn evaluate_within(&self, lengths: &[(&str, usize)], budget: &ByteBudget) -> bool {
        let len = |name: &str| {
            lengths
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, n)| *n)
        };
        let upper = match &self.upper {
            Upper::Constant(n) => Some(*n as usize),
            Upper::Length(name) => len(name),
        };
        matches!((upper, len(&self.output)),
            (Some(hi), Some(output)) if hi <= output && self.fits(hi, budget) && self.inputs.iter().all(|name| len(name).is_some_and(|input| hi <= input)))
    }

    pub fn select_plan(&self, lengths: &[(&str, usize)]) -> &'static str {
        self.select_plan_within(lengths, &ByteBudget::standard())
    }
    pub fn select_plan_within(
        &self,
        lengths: &[(&str, usize)],
        budget: &ByteBudget,
    ) -> &'static str {
        if self.evaluate_within(lengths, budget) {
            "wrapping-map-candidate"
        } else {
            "mir-serial"
        }
    }

    /// The current graph map rewrites every element, so partial loops must stay serial.
    pub fn covers_whole_arrays(&self, lengths: &[(&str, usize)]) -> bool {
        self.covers_whole_arrays_within(lengths, &ByteBudget::standard())
    }
    pub fn covers_whole_arrays_within(
        &self,
        lengths: &[(&str, usize)],
        budget: &ByteBudget,
    ) -> bool {
        let len = |name: &str| {
            lengths
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, n)| *n)
        };
        let upper = match &self.upper {
            Upper::Constant(n) => Some(*n as usize),
            Upper::Length(name) => len(name),
        };
        matches!(upper, Some(hi) if self.fits(hi, budget)
            && len(&self.output) == Some(hi)
            && self.inputs.iter().all(|name| len(name) == Some(hi)))
    }

    pub fn upper_len(&self, lengths: &[(&str, usize)]) -> Option<usize> {
        match &self.upper {
            Upper::Constant(n) => Some(*n as usize),
            Upper::Length(name) => lengths
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, n)| *n),
        }
    }

    fn json(&self) -> Value {
        json!({"upper": match &self.upper { Upper::Constant(n) => json!({"constant": n}), Upper::Length(n) => json!({"length": n}) }, "inputs": self.inputs, "output": self.output})
    }
}

#[derive(Clone, Debug)]
pub struct Region {
    function: String,
    span: Span,
    pub(crate) head: BlockId,
    candidate_identity: String,
    pub(crate) status: Status,
    pub(crate) first_rejected_obligation: Option<&'static str>,
    pub(crate) guard: Option<Guard>,
    pub(crate) shape: Option<MapShape>,
    pub(crate) obligations: Vec<Obligation>,
    pub(crate) execution: Execution,
}

#[derive(Clone, Debug)]
pub(crate) struct Obligation {
    name: &'static str,
    evidence: &'static str,
    detail: &'static str,
}

impl Region {
    /// Analysis bytes this region retains: the header, its names and guard, the obligation list
    /// at its capacity, and the chosen plan's graphs and names (`Execution::held_bytes`).
    fn held_bytes(&self) -> usize {
        let guard = self.guard.as_ref().map_or(0, |guard| {
            let upper = match &guard.upper {
                Upper::Constant(_) => 0,
                Upper::Length(name) => name.capacity(),
            };
            loops::names_bytes(&guard.inputs)
                .saturating_add(guard.output.capacity())
                .saturating_add(upper)
        });
        size_of::<Self>()
            .saturating_add(self.function.capacity())
            .saturating_add(self.candidate_identity.capacity())
            .saturating_add(guard)
            .saturating_add(
                self.obligations
                    .capacity()
                    .saturating_mul(size_of::<Obligation>()),
            )
            .saturating_add(self.execution.held_bytes())
    }

    /// Stable region identity shared with serialized admission/explain records.
    pub(crate) fn identity(&self) -> u64 {
        u64::from_str_radix(self.candidate_identity.trim_start_matches("fnv1a64:"), 16).unwrap_or(0)
    }
    pub fn function(&self) -> &str {
        &self.function
    }
    pub fn span(&self) -> Span {
        self.span
    }
    pub fn status(&self) -> Status {
        self.status
    }
    pub fn first_rejected_obligation(&self) -> Option<&'static str> {
        self.first_rejected_obligation
    }
    pub fn guard(&self) -> Option<&Guard> {
        self.guard.as_ref()
    }
    pub fn shape(&self) -> Option<MapShape> {
        self.shape
    }
    /// The plan the CPU scheduler selects for this region; shared with `explain`.
    pub fn execution(&self) -> &Execution {
        &self.execution
    }
    pub fn plan_kind(&self) -> plan::PlanKind {
        self.execution.kind()
    }
    pub fn fallback_plan(&self) -> &'static str {
        "mir-serial"
    }
    fn json(&self) -> Value {
        let rules = self.execution.rules();
        let footprints = self.guard.as_ref().map(|guard| {
            json!({
                "reads": guard.inputs.iter().map(|name| json!({"owner": name, "element_type": "u32", "offset": "induction"})).collect::<Vec<_>>(),
                "writes": [{"owner": guard.output, "element_type": "u32", "offset": "induction"}],
                "ownership": "immutable_read_capture_and_fresh_mutable_output",
                "loan_state": "no_conflicting_loan",
                "effect_order": "source_element_read_then_unique_store",
                "trap_summary": "total_if_guard_true",
                "termination": "bounded_counted_for",
                "physical_alias": "distinct_owned_arrays",
                "cleanup": "original_serial_fallback_retained"
            })
        });
        json!({
            "function": self.function,
            "span": {"start": self.span.start, "end": self.span.end},
            "candidate_identity": self.candidate_identity,
            "status": if self.status == Status::Admitted { "admitted" } else { "rejected" },
            "first_rejected_obligation": self.first_rejected_obligation,
            "rule_revision": rules[0],
            "rule_identities": rules,
            "numeric_policy": "wrapping-u32-v1",
            "target": "cpu",
            "chosen_plan": self.execution.chosen_plan(),
            "fallback_plan": self.fallback_plan(),
            "execution": self.execution.json(),
            "runtime_guard": self.guard.as_ref().map(Guard::json),
            "facts": footprints,
            "obligations": self.obligations.iter().map(|fact| json!({"name": fact.name, "evidence": fact.evidence, "detail": fact.detail})).collect::<Vec<_>>()
        })
    }
}

#[derive(Clone, Debug)]
pub struct Report {
    source_identity: String,
    regions: Vec<Region>,
}

impl Report {
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }
    /// Rebuild a region `keep` refuses as a fresh rejection, as `analyze` would
    /// record it: its plan was not selected by the normal-form dispatcher
    /// (#54). The checked normal form is the dependence proof of record
    /// (decision L1), so a loop it does not certify fails that obligation.
    pub(crate) fn demote(&mut self, source: &[u8], keep: impl Fn(&Region) -> bool) {
        for region in &mut self.regions {
            if region.plan_kind() == plan::PlanKind::Serial || keep(region) {
                continue;
            }
            region.obligations.retain(|fact| fact.name != "DEPENDENCE");
            region.obligations.push(fact(
                "DEPENDENCE",
                "unknown",
                "required fact not established for this region",
            ));
            region.status = Status::Rejected;
            region.first_rejected_obligation = Some("DEPENDENCE");
            (region.guard, region.shape) = (None, None);
            region.execution = Execution::Serial;
            region.candidate_identity = candidate_identity(source, region).unwrap_or_default();
        }
    }
    pub fn to_json(&self) -> Value {
        json!({"schema_version": SCHEMA_VERSION, "status": "ok", "diagnostics": [],
            "source_identity": self.source_identity, "regions": self.regions.iter().map(Region::json).collect::<Vec<_>>()})
    }
}

/// FNV-1a over exact source bytes. This is an identity checksum, not authentication.
fn identity(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    format!("fnv1a64:{hash:016x}")
}

fn error(code: &'static str, message: impl Into<String>) -> Error {
    Error {
        code,
        message: message.into(),
    }
}

/// A persisted admission record is accepted only when it exactly matches a fresh analysis.
pub fn validate_serialized(
    value: &Value,
    source: &[u8],
    program: &mir::Program,
) -> Result<(), Error> {
    let bytes = value.to_string().into_bytes();
    if bytes.len() > MAX_RECORD_BYTES {
        return Err(error(
            "ADMISSION_RECORD_LIMIT",
            "admission record too large",
        ));
    }
    if value.get("schema_version") != Some(&json!(SCHEMA_VERSION)) {
        return Err(error("ADMISSION_SCHEMA", "unknown admission schema"));
    }
    // `explain` documents the dispatcher's regions (#54), so a record is
    // checked against that view.
    let current = crate::schedule::regions(source, program)?.to_json();
    if *value != current {
        return Err(error(
            "ADMISSION_STALE",
            "admission facts differ from current source, MIR, or rules",
        ));
    }
    Ok(())
}

pub fn analyze(source: &[u8], program: &mir::Program) -> Result<Report, Error> {
    analyze_within(source, program, &ByteBudget::standard())
}

/// `analyze`, with the region list admitted against `budget.analysis_bytes`.
pub fn analyze_within(
    source: &[u8],
    program: &mir::Program,
    budget: &ByteBudget,
) -> Result<Report, Error> {
    if source.len() > MAX_SOURCE_BYTES {
        return Err(error(
            "ADMISSION_SOURCE_LIMIT",
            "source bytes exceed analysis limit",
        ));
    }
    mir::verify(program).map_err(|e| error(e.code, e.message))?;
    if program.functions.iter().any(|f| f.span.end > source.len()) {
        return Err(error("ADMISSION_SOURCE", "MIR span exceeds source bytes"));
    }
    let mut regions = Vec::new();
    let mut hints = Vec::new();
    let mut held = 0usize;
    for function in &program.functions {
        let defs: HashMap<ValueId, (&Operation, BlockId)> = function
            .blocks
            .iter()
            .enumerate()
            .flat_map(|(block, basic)| {
                basic.instructions.iter().filter_map(move |instruction| {
                    instruction
                        .output
                        .as_ref()
                        .map(|(id, _)| (*id, (&instruction.kind, block)))
                })
            })
            .collect();
        for (head, block) in function.blocks.iter().enumerate() {
            let Some(Terminator::Branch {
                condition,
                yes,
                span,
                ..
            }) = &block.end
            else {
                continue;
            };
            let is_for = block.params.iter().any(|(id, ty)| *ty == Type::U32 && matches!(defs.get(condition), Some((Operation::Binary { op: BinaryOp::Lt, left, .. }, _)) if left == id));
            let is_while = !is_for && function.blocks.iter().enumerate().any(|(id, body)| {
                id > head
                    && matches!(&body.end, Some(Terminator::Goto { target, .. }) if *target == head)
            });
            if !is_for && !is_while {
                continue;
            }
            let mut region = if is_for {
                analyze_region(function, &defs, head, *yes, *condition, *span)
            } else {
                reject(function, *span, "RANGE", Vec::new())
            };
            if let Some(hint) = fold_hint::hint(function, &defs, *yes) {
                hints.push((regions.len(), hint));
            }
            region.head = head;
            held = held.saturating_add(region.held_bytes());
            admit_held(budget, held)?;
            regions.push(region);
        }
    }
    plan::decide(program, &mut regions);
    for (index, hint) in hints {
        if regions[index].status == Status::Rejected {
            regions[index].obligations.push(hint);
        }
    }
    for region in &mut regions {
        region.candidate_identity = candidate_identity(source, region)?;
    }
    // Planning attached graphs, fusion obligations and identities; the settled list must fit.
    admit_held(
        budget,
        regions.iter().fold(0usize, |sum, region| {
            sum.saturating_add(region.held_bytes())
        }),
    )?;
    let report = Report {
        source_identity: identity(source),
        regions,
    };
    let bytes = report.to_json().to_string().into_bytes();
    if bytes.len() > MAX_RECORD_BYTES {
        return Err(error(
            "ADMISSION_RECORD_LIMIT",
            "admission report too large",
        ));
    }
    Ok(report)
}

/// Admit `held` bytes of retained region storage against the analysis budget.
fn admit_held(budget: &ByteBudget, held: usize) -> Result<(), Error> {
    budget
        .admit("admission", &Footprint::ZERO.analysis(Some(held)))
        .map_err(|e| error("RESOURCE_LIMIT", e.to_string()))
}

/// The region's source bytes and the rules of its chosen plan.
fn candidate_identity(source: &[u8], region: &Region) -> Result<String, Error> {
    let region_bytes = source
        .get(region.span.start..region.span.end)
        .ok_or_else(|| error("ADMISSION_SOURCE", "region span exceeds source bytes"))?;
    let mut identity_bytes = region_bytes.to_vec();
    for rule in region.execution.rules() {
        identity_bytes.extend_from_slice(rule.as_bytes());
    }
    identity_bytes.extend_from_slice(b"wrapping-u32-v1/cpu");
    Ok(identity(&identity_bytes))
}

fn reject(
    function: &Function,
    span: Span,
    name: &'static str,
    mut facts: Vec<Obligation>,
) -> Region {
    facts.push(fact(
        name,
        "unknown",
        "required fact not established for this region",
    ));
    Region {
        function: function.name.clone(),
        span,
        head: 0,
        candidate_identity: String::new(),
        status: Status::Rejected,
        first_rejected_obligation: Some(name),
        guard: None,
        shape: None,
        obligations: facts,
        execution: Execution::Serial,
    }
}

pub(crate) fn fact(name: &'static str, evidence: &'static str, detail: &'static str) -> Obligation {
    Obligation {
        name,
        evidence,
        detail,
    }
}

fn read_name<'a>(
    defs: &HashMap<ValueId, (&'a Operation, BlockId)>,
    id: ValueId,
) -> Option<&'a str> {
    match defs.get(&id)?.0 {
        Operation::Read(name) | Operation::ArrayRef { name, .. } => Some(name),
        _ => None,
    }
}

fn upper(defs: &HashMap<ValueId, (&Operation, BlockId)>, id: ValueId) -> Option<Upper> {
    match defs.get(&id)?.0 {
        Operation::U32(n) => Some(Upper::Constant(*n)),
        Operation::Length(name) => Some(Upper::Length(name.clone())),
        _ => None,
    }
}

fn analyze_region(
    function: &Function,
    defs: &HashMap<ValueId, (&Operation, BlockId)>,
    head: BlockId,
    body: BlockId,
    condition: ValueId,
    span: Span,
) -> Region {
    let mut facts = Vec::new();
    let Some((induction, _)) = function.blocks[head].params.first() else {
        return reject(function, span, "RANGE", facts);
    };
    let Some((
        Operation::Binary {
            op: BinaryOp::Lt,
            left,
            right,
            ..
        },
        _,
    )) = defs.get(&condition)
    else {
        return reject(function, span, "RANGE", facts);
    };
    if *left != *induction {
        return reject(function, span, "RANGE", facts);
    }
    let Some(hi) = upper(defs, *right) else {
        return reject(function, span, "RANGE", facts);
    };
    let incoming = function
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(block_id, block)| match &block.end {
            Some(Terminator::Goto { target, args, .. }) if *target == head && block_id != body => {
                Some(args.as_slice())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let lower_zero = incoming.len() == 1
        && incoming[0]
            .first()
            .is_some_and(|id| matches!(defs.get(id), Some((Operation::U32(0), _))));
    if !lower_zero {
        return reject(function, span, "RANGE", facts);
    }
    facts.push(fact(
        "RANGE",
        "checked_invariant",
        "zero-based counted for with fixed upper value",
    ));
    let body_block = &function.blocks[body];
    if !matches!(&body_block.end, Some(Terminator::Goto { target, args, .. }) if *target == head && args.first().is_some_and(|id| matches!(defs.get(id), Some((Operation::ForNext(n), _)) if n == induction)))
    {
        return reject(function, span, "TERMINATION", facts);
    }
    facts.push(fact(
        "TERMINATION",
        "checked_invariant",
        "induction advances once on completed body",
    ));
    let induction_name = body_block.instructions.iter().find_map(|i| match &i.kind {
        Operation::Bind { name, value } if value == induction => Some(name.as_str()),
        _ => None,
    });
    let Some(induction_name) = induction_name else {
        return reject(function, span, "DEPENDENCE", facts);
    };
    let mut store = None;
    let mut indexed_ids = Vec::new();
    let mut call_ids = Vec::new();
    for instruction in &body_block.instructions {
        match &instruction.kind {
            Operation::Tick | Operation::U32(_) | Operation::ForNext(_) => {}
            Operation::Read(name) if name == induction_name => {}
            Operation::ArrayRef { .. } => {}
            Operation::Drop(names) if names.iter().all(|name| name == induction_name) => {}
            Operation::Bind { name, value } if name == induction_name && value == induction => {}
            Operation::Index { .. } => {
                indexed_ids.push(instruction.output.as_ref().map(|(id, _)| *id))
            }
            Operation::Call { name, .. } if name == "wrapping_add" || name == "wrapping_mul" => {
                call_ids.push(instruction.output.as_ref().map(|(id, _)| *id))
            }
            Operation::StoreIndex { name, index, value } if store.is_none() => {
                store = Some((name.as_str(), *index, *value))
            }
            Operation::Binary { .. } | Operation::Unary { .. } => {
                // Arithmetic that only computes an element offset is a dependence question first.
                let offset_only = instruction.output.as_ref().is_some_and(|(id, _)| {
                    body_block.instructions.iter().any(|use_site| {
                        matches!(&use_site.kind, Operation::Index { index, .. } | Operation::StoreIndex { index, .. } if index == id)
                    })
                });
                let reason = if offset_only {
                    "DEPENDENCE"
                } else {
                    "TOTALITY"
                };
                return reject(function, span, reason, facts);
            }
            Operation::Call { .. } => return reject(function, span, "UNKNOWN_EFFECT", facts),
            _ => return reject(function, span, "EFFECT_ORDER", facts),
        }
    }
    facts.push(fact(
        "TOTALITY",
        "trusted_rule",
        "wrapping u32 element operations do not trap",
    ));
    let Some((output, write_index, value)) = store else {
        return reject(function, span, "FOOTPRINT", facts);
    };
    if read_name(defs, write_index) != Some(induction_name) {
        return reject(function, span, "FOOTPRINT", facts);
    }
    let Some((Operation::Call { name, args }, _)) = defs.get(&value) else {
        return reject(function, span, "TOTALITY", facts);
    };
    if !matches!(name.as_str(), "wrapping_add" | "wrapping_mul") || args.len() != 2 {
        return reject(function, span, "TOTALITY", facts);
    }
    if call_ids != [Some(value)] || indexed_ids.is_empty() || indexed_ids.len() > 2 {
        return reject(function, span, "EFFECT_ORDER", facts);
    }
    let indexed = args
        .iter()
        .filter_map(|id| match defs.get(id) {
            Some((Operation::Index { array, index }, _))
                if read_name(defs, *index) == Some(induction_name) =>
            {
                read_name(defs, *array)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let arg_index_ids = args
        .iter()
        .filter(|id| matches!(defs.get(id), Some((Operation::Index { .. }, _))))
        .copied()
        .map(Some)
        .collect::<Vec<_>>();
    if indexed.len() != indexed_ids.len()
        || indexed_ids != arg_index_ids
        || args.iter().any(|id| {
            !matches!(
                defs.get(id),
                Some((Operation::Index { .. } | Operation::U32(_), _))
            )
        })
    {
        return reject(function, span, "DEPENDENCE", facts);
    }
    if indexed.contains(&output) {
        return reject(function, span, "ALIAS", facts);
    }
    if indexed.iter().any(|input| !matches!(function.locals.get(*input), Some(local) if local.ty == Type::ArrayU32 && !local.mutable && local.mode != Some(Mode::Mut))) {
        return reject(function, span, "OWNERSHIP", facts);
    }
    if !matches!(function.locals.get(output), Some(local) if local.ty == Type::ArrayU32 && local.mutable)
    {
        return reject(function, span, "OWNERSHIP", facts);
    }
    let fresh = function
        .blocks
        .iter()
        .take(head)
        .flat_map(|block| &block.instructions)
        .any(|instruction| match &instruction.kind {
            Operation::Bind { name, value } if name == output => match defs.get(value) {
                Some((Operation::Array(_), _)) => true,
                Some((Operation::Call { name, .. }, _)) => name == "filled_u32",
                _ => false,
            },
            _ => false,
        });
    if !fresh {
        return reject(function, span, "FRESH_OUTPUT", facts);
    }
    facts.push(fact(
        "OWNERSHIP",
        "checked_invariant",
        "immutable read capture and mutable output owner",
    ));
    facts.push(fact(
        "ALIAS",
        "checked_invariant",
        "distinct source-language owners",
    ));
    facts.push(fact(
        "FRESH_OUTPUT",
        "checked_invariant",
        "output originates from a fresh array allocation",
    ));
    facts.push(fact(
        "DEPENDENCE",
        "checked_invariant",
        "read and write offsets equal induction; writes are injective",
    ));
    facts.push(fact(
        "EFFECT_ORDER",
        "checked_invariant",
        "one total element store per iteration",
    ));
    facts.push(fact(
        "CLEANUP",
        "trusted_rule",
        "staged publication never releases or allocates; MIR serial performs all cleanup",
    ));
    let guard = Guard {
        upper: hi,
        inputs: indexed.iter().map(|name| (*name).to_string()).collect(),
        output: output.into(),
    };
    let constant = args.iter().find_map(|id| match defs.get(id) {
        Some((Operation::U32(n), _)) => Some(*n),
        _ => None,
    });
    let shape = match (name.as_str(), constant) {
        ("wrapping_add", Some(n)) => MapShape::Unary(MapOp::WrappingAdd(n)),
        ("wrapping_mul", Some(n)) => MapShape::Unary(MapOp::WrappingMul(n)),
        ("wrapping_add", None) => MapShape::Zip(ZipOp::WrappingAdd),
        _ => MapShape::Zip(ZipOp::WrappingMul),
    };
    facts.push(fact(
        "PHYSICAL_BOUNDS",
        "runtime_guard",
        "upper is at most both array lengths before any write",
    ));
    Region {
        function: function.name.clone(),
        span,
        head: 0,
        candidate_identity: String::new(),
        status: Status::Admitted,
        first_rejected_obligation: None,
        guard: Some(guard),
        shape: Some(shape),
        obligations: facts,
        execution: Execution::Serial,
    }
}

#[cfg(test)]
mod ai_01_fault_tests {
    #[test]
    fn issue_ai_01_machine_diagnostics_planner_extraction_fault() {
        let source = include_bytes!("../examples/generated-map.inv");
        let checked = crate::check::check(crate::front::parse(source).unwrap(), true).unwrap();
        let mir = crate::mir::lower(&checked).unwrap();
        let mut report = super::analyze(source, &mir).unwrap();
        assert_eq!(report.regions[0].status, super::Status::Admitted);
        report.regions[0].shape = None;
        crate::plan::decide(&mir, &mut report.regions);
        let code = report.regions[0].first_rejected_obligation.unwrap();
        assert_eq!(code, "EXTRACTION");
        let document = crate::diagnostics::document(
            "tool_error",
            vec![crate::diagnostics::diagnostic(
                code,
                "forged admitted region",
                "fixture.inv",
                source,
                None,
            )],
        );
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/invar-diagnostics-v1.schema.json"))
                .unwrap();
        assert!(jsonschema::is_valid(&schema, &document), "{document}");
    }
}
