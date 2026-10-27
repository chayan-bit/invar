//! Synthetic completions for the four missing family bridges (compaction,
//! frontier, direct-cell contraction, summary prefixes), each built from the
//! serial MIR oracle of the family's real source shape, plus one refusal per
//! check. Positive cases publish and must reproduce the serial observation
//! (value, source steps and cleanup trace); refusals must leave the reached
//! frame, fuel, trace and allocation accounting untouched.
use super::*;
use crate::{check, front};

const LIMIT: u64 = 1_000_000;

const CONTRACTION: &str = "fn mat(a: read Array<f32>, b: read Array<f32>, rows: u32, inner: u32, cols: u32) -> Array<f32> { var c = filled_f32(rows * cols,0.0); for i in 0u32..rows { for j in 0u32..cols { for k in 0u32..inner { c[i * cols + j] = c[i * cols + j] + a[i * inner + k] * b[k * cols + j]; } } } return move c; }
fn main() -> i32 { let a: Array<f32> = [1.0,2.0,3.0,4.0,5.0,6.0]; let b: Array<f32> = [0.5,1.5,-2.0,0.25,3.0,1.0]; let c = mat(a,b,2u32,3u32,2u32); let expected: Array<f32> = [5.5,5.0,10.0,13.25]; for k in 0u32..4u32 { if c[k] != expected[k] { return 0; } } return 7; }";

const COMPACTION: &str = "fn run(xs: read Array<u32>, count: u32, cutoff: u32) -> Array<u32> { var selected=0u32; for i in 0u32..count { if xs[i] < cutoff { selected=selected+1u32; } } var out=filled_u32(selected,0u32); var pos=0u32; for j in 0u32..count { if xs[j] < cutoff { out[pos]=xs[j]; pos=pos+1u32; } } return move out; }
fn main() -> i32 { let xs: Array<u32> = [5u32,1u32,9u32,2u32,7u32,0u32]; let out = run(xs,6u32,6u32); let expected: Array<u32> = [5u32,1u32,2u32,0u32]; if len(out) != 4u32 { return 0; } for k in 0u32..4u32 { if out[k] != expected[k] { return 0; } } return 7; }";

const FRONTIER: &str = include_str!("../../../bench/general-net-v01/current/frontier_graph_a.inv");

const SUMMARY: &str = "fn run(coeff: read Array<u32>, seed: read Array<u32>, count: u32) -> Array<u32> { var x0 = seed[0u32]; var x1 = seed[1u32]; var out = filled_u32(count * 2u32,0u32); for i in 0u32..count { let y0 = wrapping_add(wrapping_mul(coeff[i * 4u32 + 0u32],x0),wrapping_mul(coeff[i * 4u32 + 1u32],x1)); let y1 = wrapping_add(wrapping_mul(coeff[i * 4u32 + 2u32],x0),wrapping_mul(coeff[i * 4u32 + 3u32],x1)); x0 = y0; x1 = y1; out[i * 2u32 + 0u32] = x0; out[i * 2u32 + 1u32] = x1; } return out; }
fn main() -> i32 { let coeff: Array<u32> = [1u32,1u32,1u32,0u32,1u32,1u32,1u32,0u32,4294967295u32,2u32,3u32,4294967295u32]; let seed: Array<u32> = [1u32,0u32]; let out = run(coeff,seed,3u32); let expected: Array<u32> = [1u32,1u32,2u32,1u32,0u32,5u32]; for k in 0u32..6u32 { if out[k] != expected[k] { return 0; } } return 7; }";

const TRAPPING: &str = "fn accumulate(xs: read Array<i32>, n: u32) -> i32 { var s: i32 = 0; for i in 0u32..n { s = s + xs[i]; } return s; }
fn main() -> i32 { let xs: Array<i32> = [5, 2147483640, 3, 9]; return accumulate(xs, 4u32); }";

fn lower(source: &str) -> mir::Program {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
    mir::lower(&checked).unwrap()
}

fn function<'p>(program: &'p mir::Program, name: &str) -> &'p Function {
    program.function(name).unwrap()
}

/// The span of the `nth` counted loop of `name`, in block order.
fn loop_span(program: &mir::Program, name: &str, nth: usize) -> Span {
    function(program, name)
        .blocks
        .iter()
        .filter_map(
            |b| match (&b.end, b.instructions.first().map(|i| &i.kind)) {
                (
                    Some(Terminator::Branch { span, .. }),
                    Some(Operation::Binary {
                        op: BinaryOp::Lt, ..
                    }),
                ) if b.params.len() == 1 => Some(*span),
                _ => None,
            },
        )
        .nth(nth)
        .unwrap()
}

fn u32s(values: &[u32]) -> Value {
    Value::ArrayU32(Rc::new(RefCell::new(values.to_vec())))
}
fn f32s(values: &[f32]) -> Value {
    Value::ArrayF32(Rc::new(RefCell::new(
        values
            .iter()
            .map(|v| F32Bits::from_bits(v.to_bits()))
            .collect(),
    )))
}

/// One frame driven block by block, as `Interpreter::blocks` drives it.
struct Machine<'p> {
    interpreter: Interpreter<'p>,
    function: &'p Function,
    frame: Frame,
    block: usize,
}

impl<'p> Machine<'p> {
    fn new(program: &'p mir::Program, name: &str, args: Vec<Value>) -> Self {
        let function = function(program, name);
        Self {
            interpreter: Interpreter {
                program,
                provider: None,
                sorts: None,
                reducer: None,
                fuel_limit: LIMIT,
                steps_used: 0,
                call_depth: 0,
                trace: Vec::new(),
                live_elements: 0,
                used_overrides: 0,
                published: Vec::new(),
            },
            function,
            frame: Frame::new(function, args),
            block: function.entry,
        }
    }

    fn instructions(&mut self) -> Result<(), EvalError> {
        for instruction in &self.function.blocks[self.block].instructions {
            let output = instruction.output.as_ref().map(|(_, ty)| ty);
            let value = self.interpreter.operation(
                &self.function.name,
                &instruction.kind,
                output,
                instruction.span,
                &mut self.frame,
            )?;
            if let (Some((id, _)), Some(value)) = (&instruction.output, value) {
                self.frame.values.insert(*id, value);
            }
        }
        Ok(())
    }

    /// Follow the current terminator, reporting a branch outcome.
    fn terminate(&mut self) -> Option<(Span, bool)> {
        match self.function.blocks[self.block].end.as_ref().unwrap() {
            Terminator::Goto { target, args, .. } => {
                let values = args.iter().map(|id| self.frame.value(*id).unwrap());
                let params = &self.function.blocks[*target].params;
                for ((id, _), value) in params.iter().zip(values.collect::<Vec<_>>()) {
                    self.frame.values.insert(*id, value);
                }
                self.block = *target;
                None
            }
            Terminator::Branch {
                condition,
                yes,
                no,
                span,
            } => {
                let truth = matches!(self.frame.value(*condition), Ok(Value::Bool(true)));
                self.block = if truth { *yes } else { *no };
                Some((*span, truth))
            }
            Terminator::Return { .. } => panic!("returned before the region"),
        }
    }

    /// Stop at the first true test of the loop at `span`, before its branch.
    fn reach(&mut self, span: Span) -> (usize, ValueId, usize, usize) {
        loop {
            self.instructions().unwrap();
            if let Some(Terminator::Branch {
                span: at,
                condition,
                yes,
                no,
            }) = &self.function.blocks[self.block].end
                && *at == span
                && matches!(self.frame.value(*condition), Ok(Value::Bool(true)))
            {
                return (self.block, *condition, *yes, *no);
            }
            self.terminate();
        }
    }

    fn apply(&mut self, region: CompletedRegion) -> Result<bool, EvalError> {
        let (head, condition, yes, _) = self.reach(region.span);
        let plan = LoopOverride::completed(region);
        self.interpreter.apply_override(
            self.function,
            &self.function.blocks[head],
            condition,
            yes,
            &mut self.frame,
            &plan,
        )
    }

    fn validate(&mut self, region: &CompletedRegion) -> Result<u64, Refusal> {
        let (head, condition, _, _) = self.reach(region.span);
        let head = &self.function.blocks[head];
        self.interpreter
            .validate_region(self.function, head, condition, &self.frame, region)
            .map(|validated| validated.fuel)
    }

    /// Every slot's deep value, plus fuel, trace and allocation accounting.
    fn snapshot(&self) -> String {
        let mut slots = self
            .frame
            .slots
            .iter()
            .map(|(name, slot)| format!("{name}={:?}/{}", slot.value, slot.owned))
            .collect::<Vec<_>>();
        slots.sort();
        let i = &self.interpreter;
        format!(
            "{slots:?} steps={} trace={:?} live={} used={}",
            i.steps_used, i.trace, i.live_elements, i.used_overrides
        )
    }
}

fn scalar(value: &Value) -> Scalar {
    Scalar::of(value).unwrap()
}

/// Run `name` serially through the loop at `span` and describe exactly what
/// it did, as a correct device completion must.
fn oracle(program: &mir::Program, name: &str, args: Vec<Value>, span: Span) -> CompletedRegion {
    let mut machine = Machine::new(program, name, args);
    let (head, _, _, exit) = machine.reach(span);
    let shape = shape(program, machine.function, head).unwrap();
    let steps = machine.interpreter.steps_used;
    let traced = machine.interpreter.trace.len();
    let before = |m: &Machine, n: &String| scalar(m.frame.slot(n).unwrap());
    let seeds = shape
        .assigned
        .iter()
        .map(|n| (n.clone(), before(&machine, n)))
        .collect::<Vec<_>>();
    let mut counts: HashMap<Span, (u64, u64)> = HashMap::new();
    let mut trap = None;
    loop {
        if let Some((at, truth)) = machine.terminate() {
            let entry = counts.entry(at).or_default();
            if truth { entry.0 += 1 } else { entry.1 += 1 }
        }
        if machine.block == exit {
            break;
        }
        match machine.instructions() {
            Ok(()) => {}
            Err(EvalError::Trap { category, span }) => {
                trap = Some(FirstTrap { category, span });
                break;
            }
            Err(error) => panic!("{error:?}"),
        }
    }
    let writes = shape
        .stored
        .iter()
        .map(|owner| {
            let values = match machine.frame.slot(owner).unwrap() {
                Value::ArrayU32(a) => Elements::U32(a.borrow().clone()),
                Value::ArrayI32(a) => Elements::I32(a.borrow().clone()),
                Value::ArrayF32(a) => Elements::F32(a.borrow().clone()),
                other => panic!("{other:?}"),
            };
            Write {
                owner: owner.clone(),
                length: values.len(),
                range: 0..values.len(),
                values,
            }
        })
        .collect();
    CompletedRegion {
        function: name.to_owned(),
        span,
        branches: counts
            .into_iter()
            .map(|(span, (taken, skipped))| Branch {
                span,
                taken,
                skipped,
            })
            .collect(),
        writes,
        scalars: seeds
            .into_iter()
            .map(|(name, before)| {
                let after = scalar(machine.frame.slot(&name).unwrap());
                ScalarState {
                    name,
                    before,
                    after,
                }
            })
            .collect(),
        cleanup: machine.interpreter.trace[traced..]
            .iter()
            .map(|e| (e.name.clone(), e.length))
            .collect(),
        fuel: machine.interpreter.steps_used - steps,
        trap,
    }
}

struct Observed {
    result: Result<Value, EvalError>,
    steps: u64,
    trace: Vec<CleanupEvent>,
    applied: usize,
    offered: Vec<Span>,
}

/// Run `main` with a provider that offers `regions` at their reached heads.
fn observe(program: &mir::Program, regions: &[CompletedRegion]) -> Observed {
    let mut offered = Vec::new();
    let (observation, applied) = {
        let mut provider = |live: &LiveLoop<'_>| -> Result<Option<LoopOverride>, EvalError> {
            offered.push(live.span);
            Ok(regions
                .iter()
                .find(|r| r.function == live.function && r.span == live.span)
                .cloned()
                .map(LoopOverride::completed))
        };
        observe_with_scheduler(program, LIMIT, Some(&mut provider))
    };
    Observed {
        result: observation.result,
        steps: observation.steps_used,
        trace: observation.cleanup_trace,
        applied,
        offered,
    }
}

/// Publication reproduces the serial observation exactly.
fn assert_exact(
    program: &mir::Program,
    regions: &[CompletedRegion],
    expected: Result<Value, EvalError>,
) -> Observed {
    let serial = super::super::observe(program, LIMIT);
    let published = observe(program, regions);
    assert_eq!(serial.result, expected);
    assert_eq!(published.result, serial.result);
    assert_eq!(published.steps, serial.steps_used);
    assert_eq!(published.trace, serial.cleanup_trace);
    assert_eq!(published.applied, regions.len());
    published
}

fn contraction() -> (mir::Program, CompletedRegion) {
    let program = lower(CONTRACTION);
    let args = vec![
        f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        f32s(&[0.5, 1.5, -2.0, 0.25, 3.0, 1.0]),
        Value::U32(2),
        Value::U32(3),
        Value::U32(2),
    ];
    let span = loop_span(&program, "mat", 0);
    let region = oracle(&program, "mat", args, span);
    (program, region)
}

fn contraction_machine(program: &mir::Program) -> Machine<'_> {
    let args = vec![
        f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        f32s(&[0.5, 1.5, -2.0, 0.25, 3.0, 1.0]),
        Value::U32(2),
        Value::U32(3),
        Value::U32(2),
    ];
    Machine::new(program, "mat", args)
}

fn compaction() -> (mir::Program, [CompletedRegion; 2]) {
    let program = lower(COMPACTION);
    let args = || vec![u32s(&[5, 1, 9, 2, 7, 0]), Value::U32(6), Value::U32(6)];
    let count = oracle(&program, "run", args(), loop_span(&program, "run", 0));
    let copy = oracle(&program, "run", args(), loop_span(&program, "run", 1));
    (program, [count, copy])
}

#[test]
fn direct_cell_contraction_publishes_outer_and_suppresses_inner_loops() {
    let (program, region) = contraction();
    assert_eq!(region.writes.len(), 1);
    assert_eq!(region.writes[0].owner, "c");
    let words = [5.5f32, 5.0, 10.0, 13.25].map(|v| F32Bits::from_bits(v.to_bits()));
    assert_eq!(region.writes[0].values, Elements::F32(words.to_vec()));
    let mut loops = region
        .branches
        .iter()
        .map(|b| (b.taken, b.skipped))
        .collect::<Vec<_>>();
    loops.sort();
    // i: 2 trips, j: 2 per row, k: 3 per cell; each nested head exits once per entry.
    assert_eq!(loops, [(2, 1), (4, 2), (12, 4)]);
    let published = assert_exact(&program, std::slice::from_ref(&region), Ok(Value::I32(7)));
    let inner = [1, 2].map(|n| loop_span(&program, "mat", n));
    assert_eq!(
        published
            .offered
            .iter()
            .filter(|s| **s == region.span)
            .count(),
        1
    );
    assert!(!published.offered.iter().any(|s| inner.contains(s)));
    // A serial outer loop still offers every nested head.
    let serial = observe(&program, &[]);
    assert_eq!(serial.offered.iter().filter(|s| **s == inner[0]).count(), 2);
    assert_eq!(serial.offered.iter().filter(|s| **s == inner[1]).count(), 4);
    let plan = LoopOverride::completed(region);
    assert_eq!(plan.published_array().as_deref(), Some("c"));
}

#[test]
fn compaction_count_then_copy_into_the_allocated_output() {
    let (program, [count, copy]) = compaction();
    assert!(count.writes.is_empty());
    assert_eq!(
        count.scalars,
        [ScalarState {
            name: "selected".into(),
            before: Scalar::U32(0),
            after: Scalar::U32(4)
        }]
    );
    // The copy pass writes the whole fresh allocation sized by the count pass.
    assert_eq!(
        copy.writes,
        [Write {
            owner: "out".into(),
            length: 4,
            range: 0..4,
            values: Elements::U32(vec![5, 1, 2, 0]),
        }]
    );
    let flagged = |r: &CompletedRegion| r.branches.iter().any(|b| (b.taken, b.skipped) == (4, 2));
    assert!(flagged(&count) && flagged(&copy));
    assert_exact(&program, &[count.clone(), copy.clone()], Ok(Value::I32(7)));
    // Either pass alone also publishes; MIR runs the other pass and the allocation.
    assert_exact(&program, std::slice::from_ref(&copy), Ok(Value::I32(7)));
    assert_exact(&program, std::slice::from_ref(&count), Ok(Value::I32(7)));
}

#[test]
fn frontier_reached_set_with_per_round_transient_owners() {
    let program = lower(FRONTIER);
    let edges = [
        0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
    ];
    let args = || vec![u32s(&edges), Value::U32(0)];
    let rounds = oracle(&program, "reach", args(), loop_span(&program, "reach", 0));
    let mut owners = rounds
        .writes
        .iter()
        .map(|w| w.owner.as_str())
        .collect::<Vec<_>>();
    owners.sort();
    assert_eq!(owners, ["frontier", "seen"]);
    let seen = rounds.writes.iter().find(|w| w.owner == "seen").unwrap();
    assert_eq!(seen.values, Elements::U32(vec![1; 5]));
    assert_eq!(rounds.cleanup, vec![("next".to_owned(), 5); 5]);
    // The count pass after the rounds is an ordinary i32 region of its own.
    let count = oracle(&program, "reach", args(), loop_span(&program, "reach", 4));
    assert_eq!(count.scalars[0].after, Scalar::I32(5));
    assert_exact(&program, &[rounds, count], Ok(Value::I32(5)));
}

#[test]
fn summary_prefix_words_and_final_state() {
    let program = lower(SUMMARY);
    let coeff = [1, 1, 1, 0, 1, 1, 1, 0, u32::MAX, 2, 3, u32::MAX];
    let args = vec![u32s(&coeff), u32s(&[1, 0]), Value::U32(3)];
    let region = oracle(&program, "run", args, loop_span(&program, "run", 0));
    assert_eq!(
        region.writes[0].values,
        Elements::U32(vec![1, 1, 2, 1, 0, 5])
    );
    let mut finals = region
        .scalars
        .iter()
        .map(|s| (s.name.as_str(), s.after))
        .collect::<Vec<_>>();
    finals.sort_by_key(|(name, _)| *name);
    assert_eq!(finals, [("x0", Scalar::U32(0)), ("x1", Scalar::U32(5))]);
    assert!(region.cleanup.is_empty());
    assert_exact(&program, &[region], Ok(Value::I32(7)));
}

#[test]
fn first_trap_publishes_the_source_prefix_then_traps() {
    let program = lower(TRAPPING);
    let args = vec![
        Value::ArrayI32(Rc::new(RefCell::new(vec![5, 2147483640, 3, 9]))),
        Value::U32(4),
    ];
    let region = oracle(
        &program,
        "accumulate",
        args,
        loop_span(&program, "accumulate", 0),
    );
    let trap = region.trap.unwrap();
    assert_eq!(trap.category, TrapCategory::IntegerOverflow);
    assert_eq!(region.scalars[0].after, Scalar::I32(2147483645));
    assert_eq!(
        (region.branches[0].taken, region.branches[0].skipped),
        (3, 0)
    );
    let expected = Err(EvalError::Trap {
        category: trap.category,
        span: trap.span,
    });
    assert_exact(&program, std::slice::from_ref(&region), expected);
    // A completion that hides the trap is refused by flow, not published.
    let mut hidden = region;
    hidden.trap = None;
    assert_refused(
        &program,
        "accumulate",
        trapping_args(),
        hidden,
        Refusal::Flow,
    );
}

fn trapping_args() -> Vec<Value> {
    vec![
        Value::ArrayI32(Rc::new(RefCell::new(vec![5, 2147483640, 3, 9]))),
        Value::U32(4),
    ]
}

/// The refusal is the expected check, and nothing at all is published.
fn assert_refused(
    program: &mir::Program,
    name: &str,
    args: Vec<Value>,
    region: CompletedRegion,
    expected: Refusal,
) {
    let mut probe = Machine::new(program, name, clone_args(&args));
    assert_eq!(probe.validate(&region), Err(expected));
    let mut machine = Machine::new(program, name, args);
    machine.reach(region.span);
    let before = machine.snapshot();
    assert_eq!(machine.apply(region), Ok(false));
    assert_eq!(machine.snapshot(), before);
}

fn clone_args(args: &[Value]) -> Vec<Value> {
    args.iter()
        .map(|value| match value {
            Value::ArrayU32(a) => Value::ArrayU32(Rc::new(RefCell::new(a.borrow().clone()))),
            Value::ArrayI32(a) => Value::ArrayI32(Rc::new(RefCell::new(a.borrow().clone()))),
            Value::ArrayF32(a) => Value::ArrayF32(Rc::new(RefCell::new(a.borrow().clone()))),
            other => copied(other),
        })
        .collect()
}

fn compaction_args() -> Vec<Value> {
    vec![u32s(&[5, 1, 9, 2, 7, 0]), Value::U32(6), Value::U32(6)]
}

fn contraction_args() -> Vec<Value> {
    vec![
        f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        f32s(&[0.5, 1.5, -2.0, 0.25, 3.0, 1.0]),
        Value::U32(2),
        Value::U32(3),
        Value::U32(2),
    ]
}

#[test]
fn refuses_wrong_spans_and_forged_regions_over_unreached_loops() {
    let (program, region) = contraction();
    let inner = loop_span(&program, "mat", 1);
    // A claim for a branch outside this region.
    let mut stray = region.clone();
    stray.branches.push(Branch {
        span: Span { start: 0, end: 1 },
        taken: 0,
        skipped: 0,
    });
    assert_refused(&program, "mat", contraction_args(), stray, Refusal::Span);
    // The same nest offered at its inner head, which no fresh outer entry reaches.
    let mut inside = region.clone();
    inside.span = inner;
    let mut machine = contraction_machine(&program);
    let (outer_head, condition, _, _) = machine.reach(region.span);
    let head = &machine.function.blocks[outer_head];
    let refusal = machine
        .interpreter
        .validate_region(machine.function, head, condition, &machine.frame, &inside)
        .map(|_| ());
    assert_eq!(refusal, Err(Refusal::Head));
    // Compaction's copy pass offered at the count pass head.
    let (program, [count, copy]) = compaction();
    let mut forged = copy.clone();
    forged.span = count.span;
    assert_refused(&program, "run", compaction_args(), forged, Refusal::Span);
    let mut wrong_function = count;
    wrong_function.function = "main".into();
    assert_refused(
        &program,
        "run",
        compaction_args(),
        wrong_function,
        Refusal::Head,
    );
}

#[test]
fn refuses_aliased_and_duplicate_owners() {
    let (program, region) = contraction();
    let mut machine = contraction_machine(&program);
    machine.reach(region.span);
    let Ok(Value::ArrayF32(c)) = machine.frame.slot("c") else {
        panic!()
    };
    let alias = Value::ArrayF32(c.clone());
    machine.frame.slots.insert(
        "alias".into(),
        Slot {
            value: Some(alias),
            owned: false,
        },
    );
    let before = machine.snapshot();
    let (head, condition, _, _) = machine.reach(region.span);
    let refusal = machine
        .interpreter
        .validate_region(
            machine.function,
            &machine.function.blocks[head],
            condition,
            &machine.frame,
            &region,
        )
        .map(|_| ());
    assert_eq!(refusal, Err(Refusal::Alias));
    assert_eq!(machine.apply(region.clone()), Ok(false));
    assert_eq!(machine.snapshot(), before);
    let mut twice = region;
    twice.writes.push(twice.writes[0].clone());
    assert_refused(&program, "mat", contraction_args(), twice, Refusal::Alias);
}

#[test]
fn refuses_out_of_bounds_ranges_and_count_mismatches() {
    let (program, [count, copy]) = compaction();
    let mut outside = copy.clone();
    outside.writes[0].range = 1..5;
    assert_refused(&program, "run", compaction_args(), outside, Refusal::Bounds);
    let mut short = copy.clone();
    short.writes[0].values = Elements::U32(vec![5, 1, 2]);
    assert_refused(&program, "run", compaction_args(), short, Refusal::Count);
    // An allocation the count pass did not size.
    let mut resized = copy.clone();
    resized.writes[0].length = 5;
    assert_refused(&program, "run", compaction_args(), resized, Refusal::Owner);
    // Branch counts that disagree with the live trip count or with each other.
    let mut trips = count.clone();
    for branch in &mut trips.branches {
        if branch.span == count.span {
            branch.taken += 1;
        }
    }
    assert_refused(&program, "run", compaction_args(), trips, Refusal::Flow);
    let mut flags = count.clone();
    for branch in &mut flags.branches {
        if branch.span != count.span {
            branch.taken -= 1;
            branch.skipped += 1;
        }
    }
    // Conserved flow, so the derived fuel exposes the miscount.
    assert_refused(&program, "run", compaction_args(), flags, Refusal::Fuel);
    let mut layout = copy;
    layout.writes[0].values = Elements::I32(vec![5, 1, 2, 0]);
    assert_refused(&program, "run", compaction_args(), layout, Refusal::Owner);
}

#[test]
fn refuses_fuel_mismatches_and_exhausted_budgets() {
    let (program, region) = contraction();
    for fuel in [region.fuel - 1, region.fuel + 1, 0] {
        let mut wrong = region.clone();
        wrong.fuel = fuel;
        assert_refused(&program, "mat", contraction_args(), wrong, Refusal::Fuel);
    }
    let mut machine = contraction_machine(&program);
    let (head, condition, _, _) = machine.reach(region.span);
    machine.interpreter.fuel_limit = machine.interpreter.steps_used + region.fuel - 1;
    let refusal = machine
        .interpreter
        .validate_region(
            machine.function,
            &machine.function.blocks[head],
            condition,
            &machine.frame,
            &region,
        )
        .map(|_| ());
    assert_eq!(refusal, Err(Refusal::Budget));
}

#[test]
fn refuses_scalar_owner_cleanup_and_trap_forgeries() {
    let (program, [count, copy]) = compaction();
    let mut seed = count.clone();
    seed.scalars[0].before = Scalar::U32(1);
    assert_refused(&program, "run", compaction_args(), seed, Refusal::Scalar);
    let mut missing = count.clone();
    missing.scalars.clear();
    assert_refused(&program, "run", compaction_args(), missing, Refusal::Scalar);
    let mut retyped = count.clone();
    retyped.scalars[0].after = Scalar::I32(4);
    assert_refused(&program, "run", compaction_args(), retyped, Refusal::Scalar);
    // A write to the read-only input the copy pass never stores to.
    let mut input = copy.clone();
    input.writes.push(Write {
        owner: "xs".into(),
        length: 6,
        range: 0..1,
        values: Elements::U32(vec![0]),
    });
    assert_refused(&program, "run", compaction_args(), input, Refusal::Owner);
    let mut unwritten = copy.clone();
    unwritten.writes.clear();
    assert_refused(
        &program,
        "run",
        compaction_args(),
        unwritten,
        Refusal::Owner,
    );
    let mut trapped = copy;
    trapped.trap = Some(FirstTrap {
        category: TrapCategory::IndexOutOfBounds,
        span: Span { start: 0, end: 1 },
    });
    assert_refused(&program, "run", compaction_args(), trapped, Refusal::Trap);

    let program = lower(FRONTIER);
    let edges = [
        0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
    ];
    let args = || vec![u32s(&edges), Value::U32(0)];
    let rounds = oracle(&program, "reach", args(), loop_span(&program, "reach", 0));
    let mut dropped = rounds.clone();
    dropped.cleanup.pop();
    assert_refused(&program, "reach", args(), dropped, Refusal::Cleanup);
}

#[test]
fn a_region_next_to_any_other_receipt_is_ambiguous() {
    let (program, region) = contraction();
    let mut machine = contraction_machine(&program);
    let (head, condition, yes, _) = machine.reach(region.span);
    let before = machine.snapshot();
    let mut apply = |plan: &LoopOverride| {
        machine.interpreter.apply_override(
            machine.function,
            &machine.function.blocks[head],
            condition,
            yes,
            &mut machine.frame,
            plan,
        )
    };
    let mut paired = LoopOverride::completed(region);
    paired.array = Some(("c".into(), vec![1, 2, 3, 4]));
    let result = apply(&paired);
    // The same refusal as any existing pair of family receipts (summary plus
    // ordinary bulk); its registered code stays in the publication module.
    let mut existing = LoopOverride::completed(contraction().1);
    existing.region = None;
    existing.array = Some(("c".into(), vec![1, 2, 3, 4]));
    existing.summary = Some(crate::summary_source::host::Completion::for_ambiguity_test());
    let expected = apply(&existing);
    assert!(
        matches!(expected, Err(EvalError::Tool { .. })),
        "{expected:?}"
    );
    assert_eq!(result, expected);
    assert_eq!(machine.snapshot(), before);
}

#[test]
fn refuses_nests_the_contract_cannot_describe() {
    let cases = [
        // Control: a plain counted nest is describable.
        (
            "var s = 0u32; for i in 0u32..n { s = wrapping_add(s, i); }",
            None,
        ),
        // A user call can write owners and spend fuel the CFG does not show.
        (
            "var s = 0u32; for i in 0u32..n { s = wrapping_add(s, one()); }",
            Some(Refusal::Shape),
        ),
        // An early return leaves the region by another edge.
        (
            "var s = 0u32; for i in 0u32..n { if i == 2u32 { return i; } s = wrapping_add(s, i); }",
            Some(Refusal::Shape),
        ),
        // An allocation that is not itself a named region owner (an array
        // moved into a record) is refused: only named owners' drops pay it back.
        (
            "var s = 0u32; for i in 0u32..n { let p = P { items: [1u32], code: i }; s = wrapping_add(s, p.code); }",
            Some(Refusal::Shape),
        ),
    ];
    for (body, expected) in cases {
        let source = format!(
            "struct P {{ items: Array<u32>, code: u32 }} fn one() -> u32 {{ return 1u32; }} fn work(n: u32) -> u32 {{ {body} return s; }}"
        );
        let program = lower(&source);
        let span = loop_span(&program, "work", 0);
        let function = function(&program, "work");
        let head = function
            .blocks
            .iter()
            .position(
                |b| matches!(&b.end, Some(Terminator::Branch { span: at, .. }) if *at == span),
            )
            .unwrap();
        let refusal = shape(&program, function, head).err();
        assert_eq!(refusal, expected, "{body}");
    }
}
