//! Live loop guards and atomic publication for staged CPU results.

use super::*;

/// A prevalidated bulk result, published only when the live loop guard still matches.
#[derive(Default)]
pub(crate) struct LoopOverride {
    pub histogram: Option<crate::histogram_source::host::Completion>,
    pub scatter: Option<crate::scatter_proof::host::Completion>,
    pub gather: Option<crate::indexed_source::host::Completion>,
    pub iterations: usize,
    pub input_lengths: Vec<(String, usize)>,
    pub scalar: Option<Vec<(String, u32, u32)>>, // owners, expected seeds, final values
    pub array: Option<(String, Vec<u32>)>,
    pub tensor: Option<crate::tensor::host::Completion>,
    pub transpose: Option<crate::transpose_source::host::Completion>,
    pub stencil: Option<crate::stencil_source::host::Completion>,
    pub summary: Option<crate::summary_source::host::Completion>,
    /// The generic completed-region contract; see [`super::region`].
    pub region: Option<Box<super::region::CompletedRegion>>,
}

impl LoopOverride {
    /// A plan whose only authority is one completed region.
    pub(crate) fn completed(region: super::region::CompletedRegion) -> Self {
        Self {
            region: Some(Box::new(region)),
            ..Default::default()
        }
    }

    /// The array owner this plan writes, for run receipts.
    pub(crate) fn published_array(&self) -> Option<String> {
        match (&self.array, &self.region) {
            (Some((name, _)), _) => Some(name.clone()),
            (None, Some(region)) => region.writes.first().map(|w| w.owner.clone()),
            (None, None) => None,
        }
    }
}

/// The trip count of a counted loop head reached with a fresh induction.
pub(super) fn live_trips(head: &mir::BasicBlock, condition: ValueId, frame: &Frame) -> Option<u32> {
    let Operation::Binary {
        op: BinaryOp::Lt,
        left,
        right,
        ..
    } = &head
        .instructions
        .iter()
        .find(|instruction| {
            instruction
                .output
                .as_ref()
                .is_some_and(|(id, _)| *id == condition)
        })?
        .kind
    else {
        return None;
    };
    if !matches!(frame.value(*left), Ok(Value::U32(0))) {
        return None;
    }
    match frame.value(*right) {
        Ok(Value::U32(trips)) => Some(trips),
        _ => None,
    }
}

pub(crate) struct LiveLoop<'a> {
    slots: &'a HashMap<String, Slot>,
    pub function: String,
    pub span: Span,
    pub iterations: usize,
    pub lengths: HashMap<String, usize>,
    pub scalars: HashMap<String, u32>,
    pub fuel_remaining: u64,
    arrays: HashMap<String, &'a RefCell<Vec<u32>>>,
    floating: HashMap<String, &'a RefCell<Vec<crate::numeric_contract::F32Bits>>>,
}

impl<'a> LiveLoop<'a> {
    pub(crate) fn words(&self, name: &str, ty: &Type) -> Option<Vec<u32>> {
        match (self.slots.get(name)?.value.as_ref()?, ty) {
            (Value::ArrayF32(a), Type::ArrayF32) => {
                Some(a.try_borrow().ok()?.iter().map(|v| v.to_bits()).collect())
            }
            (Value::ArrayU32(a), Type::ArrayU32) => Some(a.try_borrow().ok()?.clone()),
            (Value::ArrayI32(a), Type::ArrayI32) => {
                Some(a.try_borrow().ok()?.iter().map(|v| *v as u32).collect())
            }
            (Value::F32(v), Type::F32) => Some(vec![v.to_bits()]),
            (Value::U32(v), Type::U32) => Some(vec![*v]),
            (Value::I32(v), Type::I32) => Some(vec![*v as u32]),
            _ => None,
        }
    }
    pub(crate) fn owned(&self, name: &str) -> Option<bool> {
        self.slots.get(name).map(|s| s.owned)
    }
    pub(crate) fn array_handle(&self, name: &str) -> Option<Rc<RefCell<Vec<u32>>>> {
        match self.slots.get(name)?.value.as_ref()? {
            Value::ArrayU32(a) => Some(a.clone()),
            _ => None,
        }
    }
    /// Current elements of a live `Array<u32>` owner, borrowed for this loop head only.
    pub fn array(&self, name: &str) -> Option<std::cell::Ref<'a, Vec<u32>>> {
        self.arrays.get(name)?.try_borrow().ok()
    }

    /// Every live `Array<u32>` owner and its elements, sorted by name.
    pub fn snapshot(&self) -> Vec<(String, Vec<u32>)> {
        let mut arrays = self
            .arrays
            .iter()
            .filter_map(|(name, values)| Some((name.clone(), values.try_borrow().ok()?.clone())))
            .collect::<Vec<_>>();
        arrays.sort();
        arrays
    }

    /// Whether two live owners hold the same array object.
    pub fn same_object(&self, left: &str, right: &str) -> bool {
        matches!((self.arrays.get(left), self.arrays.get(right)), (Some(a), Some(b)) if std::ptr::eq(*a, *b))
    }
    pub fn floating(
        &self,
        name: &str,
    ) -> Option<std::cell::Ref<'a, Vec<crate::numeric_contract::F32Bits>>> {
        self.floating.get(name)?.try_borrow().ok()
    }
    pub fn same_floating(&self, left: &str, right: &str) -> bool {
        matches!((self.floating.get(left), self.floating.get(right)), (Some(a), Some(b)) if std::ptr::eq(*a, *b))
    }
}

impl LoopOverride {
    pub(crate) fn tensor(completion: crate::tensor::host::Completion) -> Self {
        Self {
            region: None,
            histogram: None,
            scatter: None,
            gather: None,
            iterations: 0,
            input_lengths: Vec::new(),
            scalar: None,
            array: None,
            tensor: Some(completion),
            transpose: None,
            stencil: None,
            summary: None,
        }
    }
}

pub(super) fn live_loop<'a>(
    function: &Function,
    head: &mir::BasicBlock,
    body_id: usize,
    condition: ValueId,
    span: Span,
    frame: &'a Frame,
    fuel_remaining: u64,
) -> Option<LiveLoop<'a>> {
    let iterations = live_trips(head, condition, frame)?;
    let body = function.blocks.get(body_id)?;
    let ticks = head
        .instructions
        .iter()
        .chain(&body.instructions)
        .filter(|instruction| matches!(instruction.kind, Operation::Tick))
        .count();
    let needed = u64::try_from((iterations as usize).checked_mul(ticks)?).ok()?;
    if needed > fuel_remaining {
        return None;
    }
    let mut lengths = HashMap::new();
    let mut scalars = HashMap::new();
    let mut arrays = HashMap::new();
    let mut floating = HashMap::new();
    for (name, slot) in &frame.slots {
        match slot.value.as_ref() {
            Some(Value::U32(value)) => {
                scalars.insert(name.clone(), *value);
            }
            Some(Value::ArrayU32(values)) => {
                arrays.insert(name.clone(), &**values);
                if let Some(value) = slot
                    .value
                    .as_ref()
                    .and_then(|value| array_length(value).ok())
                {
                    lengths.insert(name.clone(), value);
                }
            }
            Some(Value::ArrayI32(_)) => {
                if let Some(value) = slot
                    .value
                    .as_ref()
                    .and_then(|value| array_length(value).ok())
                {
                    lengths.insert(name.clone(), value);
                }
            }
            Some(Value::ArrayF32(values)) => {
                floating.insert(name.clone(), &**values);
                lengths.insert(name.clone(), values.try_borrow().ok()?.len());
            }
            _ => {}
        }
    }
    Some(LiveLoop {
        slots: &frame.slots,
        function: function.name.clone(),
        span,
        iterations: iterations as usize,
        lengths,
        scalars,
        arrays,
        floating,
        fuel_remaining,
    })
}

impl Interpreter<'_> {
    pub(super) fn apply_override(
        &mut self,
        function: &Function,
        head: &mir::BasicBlock,
        condition: ValueId,
        body_id: usize,
        frame: &mut Frame,
        plan: &LoopOverride,
    ) -> Result<bool, EvalError> {
        // Nine families (the reviewed seven, the consolidated stencil family and
        // the generic completed region). Ordinary bulk is one family: a prefix-scan step's
        // single receipt legitimately carries both its output array and its
        // final scalar, which the ordinary path below publishes together.
        let families = [
            plan.histogram.is_some(),
            plan.scatter.is_some(),
            plan.gather.is_some(),
            plan.summary.is_some(),
            plan.tensor.is_some(),
            plan.transpose.is_some(),
            plan.stencil.is_some(),
            plan.array.is_some() || plan.scalar.is_some(),
            plan.region.is_some(),
        ];
        if families.into_iter().filter(|present| *present).count() > 1 {
            // These are already-produced receipts. Refuse publication rather
            // than letting field order select an authority. A provider-level
            // selection phase is still required to refuse before producer entry.
            // Retain every unresolved indexed payload before any validation;
            // the first quarantine takes the full reached frame, subsequent
            // quarantines retain their own possibly-active private work.
            use crate::general_net::compiled::FenceObservation;
            if let Some(c) = &plan.histogram
                && matches!(
                    c.status,
                    FenceObservation::Unknown | FenceObservation::Pending
                )
            {
                let _ = self.apply_histogram_override(function, head, condition, frame, c);
            }
            if let Some(c) = &plan.scatter
                && matches!(
                    c.status,
                    FenceObservation::Unknown | FenceObservation::Pending
                )
            {
                let _ = self.apply_scatter_override(function, head, condition, frame, c);
            }
            if let Some(c) = &plan.gather
                && matches!(
                    c.status,
                    FenceObservation::Unknown | FenceObservation::Pending
                )
            {
                let _ = self.apply_gather_override(function, head, condition, frame, c);
            }
            return Err(tool(
                "BULK_AMBIGUOUS_COMPLETION",
                "multiple family receipts; no publication or source replay",
            ));
        }
        if let Some(histogram) = &plan.histogram {
            return self.apply_histogram_override(function, head, condition, frame, histogram);
        }
        if let Some(scatter) = &plan.scatter {
            return self.apply_scatter_override(function, head, condition, frame, scatter);
        }
        if let Some(gather) = &plan.gather {
            return self.apply_gather_override(function, head, condition, frame, gather);
        }
        if let Some(summary) = &plan.summary {
            return self.apply_summary_override(function, head, condition, frame, summary);
        }
        if let Some(tensor) = &plan.tensor {
            return self.apply_tensor_override(function, head, condition, frame, tensor);
        }
        if let Some(transpose) = &plan.transpose {
            return self.apply_transpose_override(function, head, condition, frame, transpose);
        }
        if let Some(stencil) = &plan.stencil {
            return self.apply_stencil_override(function, head, condition, frame, stencil);
        }
        if let Some(region) = &plan.region {
            return self.apply_region(function, head, condition, frame, region);
        }
        if live_trips(head, condition, frame).is_none_or(|n| n as usize != plan.iterations)
            || plan.input_lengths.iter().any(|(name, expected)| {
                !matches!(frame.slot(name).and_then(array_length), Ok(actual) if actual == *expected)
            })
            || plan.scalar.as_ref().is_some_and(|updates| {
                updates.is_empty() || updates.len() > 2
                    || (updates.len() == 2 && updates[0].0 == updates[1].0)
            })
            || plan.scalar.iter().flatten().any(|(name, expected, _)| {
                !matches!(frame.slot(name), Ok(Value::U32(value)) if value == expected)
            })
            || plan.array.as_ref().is_some_and(|(name, values)| {
                values.len() != plan.iterations
                    || !matches!(frame.slot(name).and_then(array_length), Ok(actual) if actual >= values.len())
            })
        {
            return Ok(false);
        }
        let body = function
            .blocks
            .get(body_id)
            .ok_or_else(|| tool("MIR_BLOCK", "missing loop body"))?;
        let ticks_per_iteration = head
            .instructions
            .iter()
            .chain(&body.instructions)
            .filter(|instruction| matches!(instruction.kind, Operation::Tick))
            .count();
        let needed = plan
            .iterations
            .checked_mul(ticks_per_iteration)
            .and_then(|ticks| u64::try_from(ticks).ok())
            .ok_or_else(|| tool("RESOURCE_LIMIT", "loop tick accounting overflow"))?;
        if needed > self.fuel_limit.saturating_sub(self.steps_used) {
            self.steps_used = self.fuel_limit;
            return Err(EvalError::EvaluationLimit {
                fuel_limit: self.fuel_limit,
                steps_used: self.steps_used,
            });
        }
        self.steps_used += needed;
        if let Some((name, values)) = &plan.array {
            let Value::ArrayU32(target) = frame.slot(name)? else {
                return Ok(false);
            };
            // The loop writes exactly elements `0..iterations`; any tail is untouched.
            target
                .try_borrow_mut()
                .map_err(|_| tool("MIR_LOAN", "bulk output borrow conflict"))?
                .get_mut(..values.len())
                .ok_or_else(|| tool("MIR_PLACE", "bulk output shorter than trip count"))?
                .copy_from_slice(values);
        }
        for (name, _, final_value) in plan.scalar.iter().flatten() {
            let slot = frame
                .slots
                .get_mut(name)
                .ok_or_else(|| tool("MIR_PLACE", "missing bulk scalar"))?;
            slot.value = Some(Value::U32(*final_value));
        }
        self.used_overrides += 1;
        Ok(true)
    }
}

#[cfg(test)]
mod hardening_tests {
    use super::*;
    use crate::{
        check, front, general_net::compiled::FenceObservation, histogram_source, indexed_source,
        scatter_proof,
    };

    const SOURCES: [&[u8]; 3] = [
        b"fn work(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out = filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i] = a[ix[i]]; } return out; }",
        b"fn work(a: read Array<u32>) -> Array<u32> { var out = filled_u32(2u32,0u32); for i in 0u32..2u32 { out[i] = a[i]; } return out; }",
        b"fn work(a: read Array<u32>, ix: read Array<u32>, n: u32) -> Array<u32> { var out = filled_u32(n,0u32); for i in 0u32..len(ix) { out[ix[i]] = wrapping_add(out[ix[i]],a[i]); } return out; }",
    ];

    struct Reached {
        program: Rc<mir::Program>,
        frame: Frame,
        head: usize,
        body: usize,
        condition: ValueId,
        output: Rc<RefCell<Vec<u32>>>,
    }

    // Real cold plans and producers create these receipts. The first populated
    // family would successfully publish without the ambiguity rejection.
    fn completed(kind: usize) -> (Reached, LoopOverride) {
        let source = SOURCES[kind];
        let checked = check::check(front::parse(source).unwrap(), false).unwrap();
        let program = Rc::new(mir::lower(&checked).unwrap());
        let function = &program.functions[0];
        let (head, body, condition, span) = function
            .blocks
            .iter()
            .enumerate()
            .find_map(|(id, b)| match b.end {
                Some(Terminator::Branch {
                    yes,
                    condition,
                    span,
                    ..
                }) => Some((id, yes, condition, span)),
                _ => None,
            })
            .unwrap();
        let (left, right) = function.blocks[head]
            .instructions
            .iter()
            .find_map(|i| match i.kind {
                Operation::Binary {
                    op: BinaryOp::Lt,
                    left,
                    right,
                    ..
                } => Some((left, right)),
                _ => None,
            })
            .unwrap();
        let values = Rc::new(RefCell::new(vec![7, 11]));
        let indices = Rc::new(RefCell::new(vec![0, 1]));
        let output = Rc::new(RefCell::new(vec![0; 2]));
        let mut args = vec![Value::ArrayU32(values)];
        if kind != 1 {
            args.push(Value::ArrayU32(indices));
        }
        if kind == 2 {
            args.push(Value::U32(2));
        }
        let mut frame = Frame::new(function, args);
        frame.slots.insert(
            "out".into(),
            Slot {
                value: Some(Value::ArrayU32(output.clone())),
                owned: true,
            },
        );
        frame.values.insert(left, Value::U32(0));
        frame.values.insert(right, Value::U32(2));
        let live = live_loop(
            function,
            &function.blocks[head],
            body,
            condition,
            span,
            &frame,
            1000,
        )
        .unwrap();
        let mut begun = 0;
        let mut refusals = Vec::new();
        let mut result = LoopOverride {
            iterations: 2,
            ..Default::default()
        };
        match kind {
            0 => {
                let plan = Rc::new(
                    indexed_source::host::Plan::bind(source.into(), &checked, "work", &program)
                        .unwrap(),
                );
                result.gather = indexed_source::host::prepare(
                    plan,
                    &live,
                    Default::default(),
                    Rc::new(RefCell::new(None)),
                    &mut begun,
                    &mut refusals,
                    indexed_source::host::produce_sync,
                )
                .unwrap();
                assert!(result.gather.is_some(), "{refusals:?}");
            }
            1 => {
                let plan = Rc::new(
                    scatter_proof::host::Plan::bind(source.into(), &checked, "work", &program)
                        .unwrap(),
                );
                result.scatter = scatter_proof::host::prepare(
                    plan,
                    &live,
                    Default::default(),
                    Rc::new(RefCell::new(None)),
                    &mut begun,
                    &mut refusals,
                    scatter_proof::host::produce_sync,
                )
                .unwrap();
                assert!(result.scatter.is_some(), "{refusals:?}");
            }
            2 => {
                let plan = Rc::new(
                    histogram_source::host::Plan::bind(source, &checked, "work", program.clone())
                        .unwrap(),
                );
                let ledger = Rc::new(RefCell::new(histogram_source::host::Ledger::new(
                    vec![plan.clone()],
                    Default::default(),
                )));
                result.histogram = histogram_source::host::prepare(
                    plan,
                    ledger,
                    &live,
                    Default::default(),
                    Rc::new(RefCell::new(None)),
                    histogram_source::host::Progress {
                        begun: &mut begun,
                        refusals: &mut refusals,
                    },
                    histogram_source::Contract::runtime_produce,
                )
                .unwrap();
                assert!(result.histogram.is_some(), "{refusals:?}");
            }
            _ => unreachable!(),
        }
        assert_eq!(begun, 1); // These are already-entered receipts, not pre-entry claims.
        (
            Reached {
                program,
                frame,
                head,
                body,
                condition,
                output,
            },
            result,
        )
    }

    fn publish(
        reached: &mut Reached,
        plan: &LoopOverride,
    ) -> (
        Result<bool, EvalError>,
        u64,
        usize,
        Vec<crate::eval::CleanupEvent>,
    ) {
        let mut interpreter = Interpreter {
            program: &reached.program,
            provider: None,
            sorts: None,
            reducer: None,
            fuel_limit: 1000,
            steps_used: 17,
            call_depth: 0,
            trace: Vec::new(),
            live_elements: 6,
            used_overrides: 0,
            published: Vec::new(),
        };
        let function = &reached.program.functions[0];
        let result = interpreter.apply_override(
            function,
            &function.blocks[reached.head],
            reached.condition,
            reached.body,
            &mut reached.frame,
            plan,
        );
        (
            result,
            interpreter.steps_used,
            interpreter.used_overrides,
            interpreter.trace,
        )
    }

    #[test]
    fn issue_m4_05_hardening_all_indexed_receipt_pairs_triple_no_publication() {
        let statuses = [
            FenceObservation::Complete,
            FenceObservation::FailedQuiescent,
            FenceObservation::Unknown,
            FenceObservation::Pending,
        ];
        let mut cases = 0;
        for mask in [3u32, 5, 6, 7] {
            for encoded in 0..4usize.pow(mask.count_ones()) {
                let mut remaining = encoded;
                let mut current = [FenceObservation::Complete; 3];
                for (kind, status) in current.iter_mut().enumerate() {
                    if mask & (1 << kind) != 0 {
                        *status = statuses[remaining % 4];
                        remaining /= 4;
                    }
                }
                let (g_frame, g) = completed(0);
                let (s_frame, s) = completed(1);
                let (h_frame, h) = completed(2);
                let mut reached = if mask & 4 != 0 {
                    h_frame
                } else if mask & 2 != 0 {
                    s_frame
                } else {
                    g_frame
                };
                let mut plan = LoopOverride {
                    histogram: if mask & 4 != 0 { h.histogram } else { None },
                    scatter: if mask & 2 != 0 { s.scatter } else { None },
                    gather: if mask & 1 != 0 { g.gather } else { None },
                    iterations: 2,
                    ..Default::default()
                };
                if let Some(c) = plan.histogram.as_mut() {
                    c.status = current[2];
                }
                if let Some(c) = plan.scatter.as_mut() {
                    c.status = current[1];
                }
                if let Some(c) = plan.gather.as_mut() {
                    c.status = current[0];
                }
                let (result, steps, applied, trace) = publish(&mut reached, &plan);
                assert!(
                    matches!(
                        result,
                        Err(EvalError::Tool {
                            code: "BULK_AMBIGUOUS_COMPLETION",
                            ..
                        })
                    ),
                    "mask={mask}, status={current:?}, result={result:?}"
                );
                assert_eq!(*reached.output.borrow(), [0, 0]);
                assert_eq!(steps, 17);
                assert_eq!(applied, 0);
                assert!(trace.is_empty());
                let unresolved = |status| {
                    matches!(
                        status,
                        FenceObservation::Unknown | FenceObservation::Pending
                    )
                };
                if current.into_iter().any(unresolved) {
                    if let Some(c) = &plan.histogram {
                        assert_eq!(c.quarantine.borrow().is_some(), unresolved(c.status));
                    }
                    if let Some(c) = &plan.scatter {
                        assert_eq!(c.quarantine.borrow().is_some(), unresolved(c.status));
                    }
                    if let Some(c) = &plan.gather {
                        assert_eq!(c.quarantine.borrow().is_some(), unresolved(c.status));
                    }
                    assert!(reached.frame.slots.is_empty());
                    assert!(reached.frame.values.is_empty());
                }
                cases += 1;
            }
        }
        assert_eq!(cases, 112);
        println!(
            "all three indexed receipt pairs and triple: {cases} independent completion-state combinations"
        );
    }

    #[test]
    fn issue_m4_05_hardening_single_family_receipts_still_publish() {
        for kind in 0..3 {
            let (mut reached, plan) = completed(kind);
            let (result, steps, applied, trace) = publish(&mut reached, &plan);
            assert_eq!(result, Ok(true));
            assert_eq!(*reached.output.borrow(), [7, 11]);
            assert!(steps > 17);
            assert_eq!(applied, 1);
            assert!(trace.is_empty());
        }
    }

    /// Review3 R1: ordinary array/scalar bulk is one family, yet a scan-shaped
    /// (array + scalar), array-only or scalar-only receipt next to any indexed
    /// family is still refused before publication, with unresolved indexed
    /// work quarantined exactly as for indexed pairs.
    #[test]
    fn issue_m4_05_hardening_ordinary_bulk_with_indexed_family_is_ambiguous() {
        type Bulk = (Option<(String, Vec<u32>)>, Option<Vec<(String, u32, u32)>>);
        let shapes: [Bulk; 3] = [
            (
                Some(("out".into(), vec![1, 2])),
                Some(vec![("state".into(), 17, 683)]),
            ),
            (Some(("out".into(), vec![1, 2])), None),
            (None, Some(vec![("state".into(), 17, 683)])),
        ];
        for kind in 0..3 {
            for status in [FenceObservation::Complete, FenceObservation::Unknown] {
                for (array, scalar) in shapes.clone() {
                    let (mut reached, mut plan) = completed(kind);
                    if let Some(c) = plan.histogram.as_mut() {
                        c.status = status;
                    }
                    if let Some(c) = plan.scatter.as_mut() {
                        c.status = status;
                    }
                    if let Some(c) = plan.gather.as_mut() {
                        c.status = status;
                    }
                    plan.array = array;
                    plan.scalar = scalar;
                    let (result, steps, applied, trace) = publish(&mut reached, &plan);
                    assert!(
                        matches!(
                            result,
                            Err(EvalError::Tool {
                                code: "BULK_AMBIGUOUS_COMPLETION",
                                ..
                            })
                        ),
                        "kind {kind} {status:?}: {result:?}"
                    );
                    assert_eq!(*reached.output.borrow(), [0, 0]);
                    assert_eq!((steps, applied), (17, 0));
                    assert!(trace.is_empty());
                    let quarantined = plan
                        .histogram
                        .as_ref()
                        .map(|c| c.quarantine.borrow().is_some())
                        .or(plan
                            .scatter
                            .as_ref()
                            .map(|c| c.quarantine.borrow().is_some()))
                        .or(plan
                            .gather
                            .as_ref()
                            .map(|c| c.quarantine.borrow().is_some()));
                    assert_eq!(quarantined, Some(status == FenceObservation::Unknown));
                }
            }
        }
    }

    fn stencil_completion() -> crate::stencil_source::host::Completion {
        // Constant-zero boundary five-point stencil, as in stencil_source.
        const STENCIL: &str = "fn step(a:read Array<f32>,rows:u32,cols:u32)->Array<f32>{var b=filled_f32(rows*cols,0.0);for r in 0u32..rows{for c in 0u32..cols{var n=0.0; if r>0u32{n=a[(r-1u32)*cols+c];} var s=0.0; if r+1u32<rows{s=a[(r+1u32)*cols+c];} var w=0.0; if c>0u32{w=a[r*cols+(c-1u32)];} var e=0.0; if c+1u32<cols{e=a[r*cols+(c+1u32)];} b[r*cols+c]=(((a[r*cols+c]+n)+s)+w)+e;}}return b;}";
        let checked = check::check(front::parse(STENCIL.as_bytes()).unwrap(), false).unwrap();
        let contract = crate::stencil_source::Contract::from_checked(&checked, "step").unwrap();
        crate::stencil_source::host::Completion::for_ambiguity_test(Rc::new(contract))
    }

    /// The consolidated stencil family is the eighth override family: a stencil
    /// receipt together with any indexed or ordinary bulk receipt is refused
    /// before publication, with nothing written or traced.
    #[test]
    fn issue_m4_05_hardening_stencil_with_other_family_is_ambiguous() {
        for kind in 0..4 {
            let (mut reached, mut plan) = completed(kind.min(2));
            if kind == 3 {
                plan.gather = None;
                plan.scatter = None;
                plan.histogram = None;
                plan.array = Some(("out".into(), vec![1, 2]));
                plan.scalar = Some(vec![("state".into(), 17, 683)]);
            }
            plan.stencil = Some(stencil_completion());
            let (result, steps, applied, trace) = publish(&mut reached, &plan);
            assert!(
                matches!(
                    result,
                    Err(EvalError::Tool {
                        code: "BULK_AMBIGUOUS_COMPLETION",
                        ..
                    })
                ),
                "kind {kind}: {result:?}"
            );
            assert_eq!(*reached.output.borrow(), [0, 0]);
            assert_eq!((steps, applied), (17, 0));
            assert!(trace.is_empty());
        }
    }

    fn transpose_completion() -> crate::transpose_source::host::Completion {
        // u32 flip, as in transpose_source.
        const FLIP: &str = "fn flip(a: read Array<u32>, rows: u32, cols: u32) -> Array<u32> { var b = filled_u32(rows * cols,0u32); for r in 0u32..rows { for c in 0u32..cols { b[c * rows + r] = a[r * cols + c]; } } return b; }";
        let checked = check::check(front::parse(FLIP.as_bytes()).unwrap(), false).unwrap();
        let contract = crate::transpose_source::Contract::from_checked(&checked, "flip").unwrap();
        crate::transpose_source::host::Completion::for_ambiguity_test(Rc::new(contract))
    }

    fn tensor_completion() -> crate::tensor::host::Completion {
        // f32 matrix product, as in tensor_host.
        const MAT: &str = "fn mat(a: read Array<f32>, b: read Array<f32>, rows: u32, inner: u32, cols: u32) -> Array<f32> { var c = filled_f32(rows * cols,0.0); for i in 0u32..rows { for j in 0u32..cols { var sum: f32 = 0.0; for k in 0u32..inner { sum = sum + a[i * inner + k] * b[k * cols + j]; } c[i * cols + j] = sum; } } return c; }";
        let checked = check::check(front::parse(MAT.as_bytes()).unwrap(), false).unwrap();
        let contraction = crate::tensor::Contraction::from_checked(&checked, "mat").unwrap();
        crate::tensor::host::Completion::for_ambiguity_test(contraction.contract().clone())
    }

    /// Stencil, transpose and tensor are distinct families. In particular the
    /// combined stencil/transpose route must never produce both receipts for
    /// one head; if it did, publication is refused. Each pair is refused with
    /// nothing written or traced.
    #[test]
    fn issue_m4_05_hardening_stencil_transpose_tensor_pairs_are_ambiguous() {
        for pair in 0..3 {
            let (mut reached, mut plan) = completed(0);
            plan.gather = None;
            match pair {
                0 => {
                    plan.stencil = Some(stencil_completion());
                    plan.transpose = Some(transpose_completion());
                }
                1 => {
                    plan.stencil = Some(stencil_completion());
                    plan.tensor = Some(tensor_completion());
                }
                _ => {
                    plan.transpose = Some(transpose_completion());
                    plan.tensor = Some(tensor_completion());
                }
            }
            let (result, steps, applied, trace) = publish(&mut reached, &plan);
            assert!(
                matches!(
                    result,
                    Err(EvalError::Tool {
                        code: "BULK_AMBIGUOUS_COMPLETION",
                        ..
                    })
                ),
                "pair {pair}: {result:?}"
            );
            assert_eq!(*reached.output.borrow(), [0, 0]);
            assert_eq!((steps, applied), (17, 0));
            assert!(trace.is_empty());
        }
    }

    /// M4-07 summary pairings: a real bounded-summary receipt next to every
    /// indexed family, in every fence state, is refused before publication.
    /// Unresolved indexed work is still quarantined (taking the reached frame);
    /// resolved work is not.
    #[test]
    fn issue_m4_07_hardening_summary_with_indexed_family_is_ambiguous() {
        let mut cases = 0;
        for kind in 0..3 {
            for status in [
                FenceObservation::Complete,
                FenceObservation::FailedQuiescent,
                FenceObservation::Unknown,
                FenceObservation::Pending,
            ] {
                let (mut reached, mut plan) = completed(kind);
                for c in [
                    plan.histogram.as_mut().map(|c| &mut c.status),
                    plan.scatter.as_mut().map(|c| &mut c.status),
                    plan.gather.as_mut().map(|c| &mut c.status),
                ]
                .into_iter()
                .flatten()
                {
                    *c = status;
                }
                plan.summary = Some(crate::summary_source::host::Completion::for_ambiguity_test());
                let (result, steps, applied, trace) = publish(&mut reached, &plan);
                assert!(
                    matches!(
                        result,
                        Err(EvalError::Tool {
                            code: "BULK_AMBIGUOUS_COMPLETION",
                            ..
                        })
                    ),
                    "kind {kind} {status:?}: {result:?}"
                );
                assert_eq!(*reached.output.borrow(), [0, 0]);
                assert_eq!((steps, applied), (17, 0));
                assert!(trace.is_empty());
                let unresolved = matches!(
                    status,
                    FenceObservation::Unknown | FenceObservation::Pending
                );
                let quarantined = [
                    plan.histogram
                        .as_ref()
                        .map(|c| c.quarantine.borrow().is_some()),
                    plan.scatter
                        .as_ref()
                        .map(|c| c.quarantine.borrow().is_some()),
                    plan.gather
                        .as_ref()
                        .map(|c| c.quarantine.borrow().is_some()),
                ];
                assert_eq!(
                    quarantined.into_iter().flatten().collect::<Vec<_>>(),
                    [unresolved],
                    "kind {kind} {status:?}"
                );
                assert_eq!(reached.frame.slots.is_empty(), unresolved);
                cases += 1;
            }
        }
        assert_eq!(cases, 12);
    }

    /// M4-07 summary pairings with the remaining families: ordinary bulk in
    /// each receipt shape (array + scalar, array only, scalar only), tensor,
    /// transpose and stencil. Each pair is refused with nothing written,
    /// debited or traced.
    #[test]
    fn issue_m4_07_hardening_summary_with_ordinary_tensor_transpose_stencil_is_ambiguous() {
        for pair in 0..6 {
            let (mut reached, mut plan) = completed(0);
            plan.gather = None;
            plan.summary = Some(crate::summary_source::host::Completion::for_ambiguity_test());
            match pair {
                0 => {
                    plan.array = Some(("out".into(), vec![1, 2]));
                    plan.scalar = Some(vec![("state".into(), 17, 683)]);
                }
                1 => plan.array = Some(("out".into(), vec![1, 2])),
                2 => plan.scalar = Some(vec![("state".into(), 17, 683)]),
                3 => plan.tensor = Some(tensor_completion()),
                4 => plan.transpose = Some(transpose_completion()),
                _ => plan.stencil = Some(stencil_completion()),
            }
            let (result, steps, applied, trace) = publish(&mut reached, &plan);
            assert!(
                matches!(
                    result,
                    Err(EvalError::Tool {
                        code: "BULK_AMBIGUOUS_COMPLETION",
                        ..
                    })
                ),
                "pair {pair}: {result:?}"
            );
            assert_eq!(*reached.output.borrow(), [0, 0]);
            assert_eq!((steps, applied), (17, 0));
            assert!(trace.is_empty());
            assert!(!reached.frame.slots.is_empty());
        }
    }

    #[test]
    fn issue_ai_01_machine_diagnostics_pair_reduction_faults_validate() {
        let source = include_bytes!("../../examples/pair-reduction.inv");
        let checked = check::check(front::parse(source).unwrap(), true).unwrap();
        let mut program = mir::lower(&checked).unwrap();
        let mut pair = crate::pair_reduction::extract(&program).unwrap();
        program.functions[0].entry = usize::MAX;
        let invalid_mir = crate::pair_reduction::extract(&program).unwrap_err();
        assert_eq!(invalid_mir.code(), "MIR_INVALID");
        pair.graph = crate::typed_net::Net::empty(0);
        let invalid_plan = pair
            .run(&[], &[], crate::laws::Affine::IDENTITY, 0)
            .unwrap_err();
        assert_eq!(invalid_plan.code(), "CPU_CANDIDATE");
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../schema/invar-diagnostics-v3.schema.json"
        ))
        .unwrap();
        for failure in [invalid_mir, invalid_plan] {
            let document = crate::diagnostics::document(
                "tool_error",
                vec![crate::diagnostics::diagnostic(
                    failure.code(),
                    "forged pair candidate",
                    "pair.inv",
                    source,
                    None,
                )],
            );
            assert!(jsonschema::is_valid(&schema, &document));
        }
    }

    #[test]
    fn pair_reduction_scalar_batch_validates_both_seeds_before_publication() {
        let (mut reached, mut plan) = completed(1);
        plan.scatter = None;
        for (name, value) in [("sa", 1), ("sb", 0)] {
            reached.frame.slots.insert(
                name.into(),
                Slot {
                    value: Some(Value::U32(value)),
                    owned: false,
                },
            );
        }
        plan.scalar = Some(vec![("sa".into(), 1, 10), ("sb".into(), 99, 22)]);
        let (result, steps, applied, _) = publish(&mut reached, &plan);
        assert_eq!(result, Ok(false));
        assert_eq!((steps, applied), (17, 0));
        assert!(matches!(reached.frame.slot("sa"), Ok(Value::U32(1))));
        assert!(matches!(reached.frame.slot("sb"), Ok(Value::U32(0))));
        plan.scalar = Some(vec![("sa".into(), 1, 10), ("sb".into(), 0, 22)]);
        let (result, _, applied, _) = publish(&mut reached, &plan);
        assert_eq!(result, Ok(true));
        assert_eq!(applied, 1);
        assert!(matches!(reached.frame.slot("sa"), Ok(Value::U32(10))));
        assert!(matches!(reached.frame.slot("sb"), Ok(Value::U32(22))));
    }

    #[test]
    fn issue_ai_01_machine_diagnostics_bulk_ambiguous_completion() {
        let (mut reached, gather) = completed(0);
        let (_, scatter) = completed(1);
        let plan = LoopOverride {
            scatter: scatter.scatter,
            ..gather
        };
        let (result, _, applied, _) = publish(&mut reached, &plan);
        assert_eq!(applied, 0);
        let Err(error @ EvalError::Tool { code, .. }) = &result else {
            panic!("{result:?}");
        };
        assert_eq!(*code, "BULK_AMBIGUOUS_COMPLETION");
        let document = error.to_diagnostics_json("bulk.inv", b"");
        assert_eq!(document["status"], "tool_error");
        assert_eq!(document["diagnostics"][0]["code"], *code);
        assert_eq!(document["diagnostics"][0]["severity"], "error");
        assert_eq!(
            document["diagnostics"][0]["suggested_fixes"],
            serde_json::json!([])
        );
    }
}
