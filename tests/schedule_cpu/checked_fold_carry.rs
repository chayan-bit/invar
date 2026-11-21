//! #204: checked integer sums run in parallel through the exact
//! (total, min prefix, max prefix) carry of PNF law L3, with no per-lane
//! replay. Value, trap kind, trap position and fuel prefix must equal the
//! serial fold on the reference evaluator, MIR and every worker count.
use invar::{
    check::check,
    eval::{self, EvalError},
    front::parse,
    mir, mir_eval,
    schedule::{self, Fault, Hooks, PlanKind},
};

#[path = "../../test-support/split_mix.rs"]
mod split_mix;
use split_mix::Rng;

const FUEL: u64 = 1_000_000;
const WORKERS: [usize; 4] = [1, 2, 7, 64];

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn between(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo + 1) as u64) as i64
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ty {
    I32,
    U32,
}

impl Ty {
    fn range(self) -> (i64, i64) {
        match self {
            Self::I32 => (i64::from(i32::MIN), i64::from(i32::MAX)),
            Self::U32 => (0, i64::from(u32::MAX)),
        }
    }
}

struct Case {
    ty: Ty,
    seed: i64,
    values: Vec<i64>,
}

impl Case {
    fn source(&self) -> String {
        let (scalar, suffix) = match self.ty {
            Ty::I32 => ("i32", ""),
            Ty::U32 => ("u32", "u32"),
        };
        let literal = self
            .values
            .iter()
            .map(|v| format!("{v}{suffix}"))
            .collect::<Vec<_>>()
            .join(", ");
        // u32 results leave through an injective code (low 31 bits, top bit
        // as the sign) without floats, which the normal form does not encode.
        let (helper, result) = match self.ty {
            Ty::I32 => ("", "s"),
            Ty::U32 => (
                "fn code(s: u32) -> i32 { var r: i32 = 0; var p: i32 = 1; var q: u32 = s; for b in 0u32 .. 31u32 { if q % 2u32 == 1u32 { r = r + p; } q = q / 2u32; if b < 30u32 { p = p * 2; } } if q == 1u32 { return -1 - r; } return r; } ",
                "code(s)",
            ),
        };
        format!(
            "{helper}fn main() -> i32 {{ let a: Array<{scalar}> = [{literal}]; var s: {scalar} = {}{suffix}; for i in 0u32 .. len(a) {{ s = s + a[i]; }} return {result}; }}",
            self.seed
        )
    }

    /// Index of the first trapping element, by the plain serial definition.
    fn trap(&self) -> Option<usize> {
        let (min, max) = self.ty.range();
        let mut s = self.seed;
        for (k, x) in self.values.iter().enumerate() {
            s += x;
            if s < min || s > max {
                return Some(k);
            }
        }
        None
    }

    /// The chunk of `trap` under the scheduler's split for `workers`, and
    /// whether that chunk's prefixes leave the range on both sides.
    fn chunk(&self, workers: usize) -> Option<(usize, usize, bool)> {
        let k = self.trap()?;
        let n = self.values.len();
        let count = if n < 64 { 1 } else { workers.min(n) };
        let width = n.div_ceil(count);
        let chunk = k / width;
        let (min, max) = self.ty.range();
        let mut s = self.seed + self.values[..chunk * width].iter().sum::<i64>();
        let (mut low, mut high) = (false, false);
        for x in &self.values[chunk * width..((chunk + 1) * width).min(n)] {
            s += x;
            low |= s < min;
            high |= s > max;
        }
        Some((chunk, n.div_ceil(width), low && high))
    }
}

/// Small non-negative noise, then an exact crossing at a uniform index
/// `t + 1`; for i32 the same chunk may also fall below the minimum later.
fn planted(rng: &mut Rng, ty: Ty, n: usize) -> Case {
    let (min, max) = ty.range();
    let t = rng.below(n as u64 - 4) as usize;
    let mut values: Vec<i64> = (0..n).map(|_| rng.between(0, 1000)).collect();
    let before: i64 = values[..t].iter().sum();
    values[t] = max - before;
    values[t + 1] = 1 + rng.between(0, 1000);
    if ty == Ty::I32 && rng.below(2) == 0 {
        values[t + 2] = min;
        values[t + 3] = min;
        values[t + 4] = min;
    }
    Case {
        ty,
        seed: 0,
        values,
    }
}

/// A random walk whose drift and spread place the first crossing anywhere
/// in the array, sometimes nowhere and sometimes on both sides of a chunk.
fn case(rng: &mut Rng) -> Case {
    let ty = if rng.below(3) == 0 { Ty::U32 } else { Ty::I32 };
    let (min, max) = ty.range();
    let n = 64 + rng.below(129) as usize;
    if rng.below(2) == 0 {
        return planted(rng, ty, n);
    }
    let seed = match rng.below(4) {
        0 => 0,
        1 => rng.between(min, max),
        2 => max - rng.between(0, 1000),
        _ => min + rng.between(0, 1000),
    };
    let target = 1 + rng.below(n as u64 + n as u64 / 3);
    let span = max - min;
    let drift = (span / 2 / target as i64).max(1);
    let spread = match rng.below(3) {
        0 => drift / 4,
        1 => drift * 3,
        _ => span / 4,
    };
    let sign = if ty == Ty::U32 || rng.below(2) == 0 {
        1
    } else {
        -1
    };
    let values = (0..n)
        .map(|_| {
            let x = sign * drift + rng.between(-spread, spread);
            match ty {
                Ty::I32 => x.clamp(min, max),
                Ty::U32 => x.clamp(0, max),
            }
        })
        .collect();
    Case { ty, seed, values }
}

fn compile(source: &str) -> (invar::check::CheckedProgram, mir::Program) {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    (checked, program)
}

/// Every engine equals the reference evaluator, and every cpu-parallel run
/// published the loop through the carried fold.
fn assert_agrees(source: &str, fuel: u64) -> eval::Observation {
    assert_agrees_on(source, fuel, &WORKERS)
}

fn assert_agrees_on(source: &str, fuel: u64, workers: &[usize]) -> eval::Observation {
    let (checked, program) = compile(source);
    let reference = eval::observe(&checked, fuel);
    let lowered = mir_eval::observe(&program, fuel);
    assert_eq!(lowered.result, reference.result, "MIR {source}");
    assert_eq!(
        lowered.steps_used, reference.steps_used,
        "MIR fuel {source}"
    );
    for &workers in workers {
        let run = schedule::run_source(source.as_bytes(), &program, fuel, workers).unwrap();
        let observed = &run.observation;
        assert_eq!(
            observed.result, reference.result,
            "{workers} workers {source}"
        );
        assert_eq!(
            observed.steps_used, reference.steps_used,
            "{workers} fuel {source}"
        );
        assert_eq!(observed.cleanup_trace, reference.cleanup_trace);
        // With the fuel for the whole loop at its head, the carry must publish;
        // a tighter limit leaves the loop to serial MIR, which stops exactly.
        if fuel == FUEL {
            let published: Vec<_> = run.executed.iter().map(|e| e.kind).collect();
            assert_eq!(
                published,
                [PlanKind::Reduction],
                "{workers} workers {source}"
            );
        }
    }
    reference
}

#[test]
fn checked_fold_carry_matches_the_serial_fold_over_3000_generated_cases() {
    let mut rng = Rng(0x204_c0de);
    // first / middle / last chunk at 7 workers, both sides in one chunk, no trap
    let mut seen = [0usize; 5];
    for index in 0..3000 {
        let case = case(&mut rng);
        // Every worker count on the first 400 cases, one in rotation after.
        let workers = if index < 400 {
            &WORKERS[..]
        } else {
            &WORKERS[index % 4..=index % 4]
        };
        let reference = assert_agrees_on(&case.source(), FUEL, workers);
        assert_eq!(
            matches!(reference.result, Err(EvalError::Trap { .. })),
            case.trap().is_some()
        );
        match case.chunk(7) {
            None => seen[4] += 1,
            Some((chunk, count, both)) => {
                seen[if chunk == 0 {
                    0
                } else if chunk + 1 == count {
                    2
                } else {
                    1
                }] += 1;
                seen[3] += usize::from(both);
            }
        }
    }
    assert!(seen.iter().all(|n| *n >= 50), "coverage {seen:?}");
}

#[test]
fn checked_fold_carry_edge_positions_and_exact_limits() {
    let max = i64::from(i32::MAX);
    let min = i64::from(i32::MIN);
    let mut cases = vec![
        // Reaches the limits exactly without crossing.
        Case {
            ty: Ty::I32,
            seed: max - 100,
            values: [vec![100], vec![-1; 99], vec![1; 99]].concat(),
        },
        Case {
            ty: Ty::I32,
            seed: min + 64,
            values: vec![-1; 64],
        },
        Case {
            ty: Ty::U32,
            seed: i64::from(u32::MAX) - 70,
            values: vec![1; 70],
        },
        // Traps on the first and on the last element.
        Case {
            ty: Ty::I32,
            seed: max,
            values: vec![1; 100],
        },
        Case {
            ty: Ty::I32,
            seed: 0,
            values: [vec![0; 99], vec![min], vec![-1]].concat(),
        },
        Case {
            ty: Ty::U32,
            seed: i64::from(u32::MAX),
            values: vec![1; 80],
        },
    ];
    // Crossings exactly at every chunk boundary of a 128-element split.
    for k in [16usize, 32, 63, 64, 65, 96, 127] {
        let mut values = vec![1; 128];
        values[k] = max;
        cases.push(Case {
            ty: Ty::I32,
            seed: 0,
            values,
        });
    }
    for case in &cases {
        let source = case.source();
        let reference = assert_agrees(&source, FUEL);
        // The fuel prefix is exact: the reference's own count suffices and one
        // step less exhausts the limit at the same point on every engine.
        assert_agrees(&source, reference.steps_used);
        assert_agrees(&source, reference.steps_used - 1);
    }
}

#[test]
fn checked_fold_carry_mutation_that_loses_the_minimum_prefix_is_caught() {
    // The walk dips below i32::MIN inside the last chunk and recovers, so a
    // combine without the minimum prefix misses the first trap.
    let min = i64::from(i32::MIN);
    let mut values = vec![0; 128];
    values[100] = min;
    values[101] = -5;
    values[102] = i64::from(i32::MAX);
    let source = Case {
        ty: Ty::I32,
        seed: 0,
        values,
    }
    .source();
    let (checked, program) = compile(&source);
    let reference = eval::observe(&checked, FUEL);
    assert!(matches!(reference.result, Err(EvalError::Trap { .. })));
    let honest = schedule::run_source(source.as_bytes(), &program, FUEL, 7).unwrap();
    assert_eq!(honest.observation.result, reference.result);
    let hooks = Hooks {
        fault: Fault::LoseMinPrefix,
        ..Hooks::default()
    };
    let mutated =
        schedule::run_source_with_hooks(source.as_bytes(), &program, FUEL, 7, &hooks).unwrap();
    assert_ne!(
        mutated.observation.result, reference.result,
        "mutation survived"
    );
}

#[test]
fn checked_fold_carry_admits_a_2_pow_23_sum_and_the_receipt_names_the_law() {
    let source = "fn main() -> i32 { let a = filled_i32(8388608u32, 1); var s: i32 = -8388608; for i in 0u32 .. 8388608u32 { s = s + a[i]; } return s; }";
    let (_, program) = compile(source);
    let report = schedule::regions(source.as_bytes(), &program)
        .unwrap()
        .to_json();
    let region = &report["regions"][0];
    assert_eq!(region["status"], "admitted");
    assert_eq!(region["chosen_plan"], "checked-sum-carry-if-guard");
    assert_eq!(region["execution"]["kind"], "reduction");
    assert_eq!(region["execution"]["law"], "checked-sum-carry");
    assert_eq!(
        region["rule_identities"],
        serde_json::json!([invar::checked_fold::RULE_REVISION])
    );
    let run = schedule::run_source(source.as_bytes(), &program, 200_000_000, 16).unwrap();
    assert_eq!(run.observation.result, Ok(eval::Value::I32(0)));
    assert_eq!(run.executed.len(), 1);
    assert_eq!(run.executed[0].kind, PlanKind::Reduction);
    assert_eq!(run.chunks.len(), 16, "the carry ran on 16 workers");
}

#[test]
fn checked_fold_carry_refuses_shapes_outside_the_law() {
    let refused = [
        // A product is not a lifted sum.
        "fn main() -> i32 { let a: Array<i32> = [1, 2]; var s: i32 = 1; for i in 0u32 .. len(a) { s = s * a[i]; } return s; }",
        // The summed term is not the element itself.
        "fn main() -> i32 { let a: Array<i32> = [1, 2]; var s: i32 = 0; for i in 0u32 .. len(a) { s = s + a[i] * 2; } return s; }",
        // Checked subtraction is not the admitted sum shape.
        "fn main() -> i32 { let a: Array<i32> = [1, 2]; var s: i32 = 0; for i in 0u32 .. len(a) { s = s - a[i]; } return s; }",
    ];
    for source in refused {
        let (_, program) = compile(source);
        let report = schedule::regions(source.as_bytes(), &program)
            .unwrap()
            .to_json();
        assert_ne!(
            report["regions"][0]["chosen_plan"], "checked-sum-carry-if-guard",
            "{source}"
        );
        assert_agrees_without_publication(source);
    }
}

fn assert_agrees_without_publication(source: &str) {
    let (checked, program) = compile(source);
    let reference = eval::observe(&checked, FUEL);
    for workers in WORKERS {
        let run = schedule::run_source(source.as_bytes(), &program, FUEL, workers).unwrap();
        assert_eq!(run.observation.result, reference.result);
        assert_eq!(run.observation.steps_used, reference.steps_used);
        assert!(run.executed.iter().all(|e| e.kind != PlanKind::Reduction));
    }
}

#[cfg(feature = "native")]
#[test]
fn checked_fold_carry_native_serial_matches_the_trap_and_fuel_prefix() {
    let mut rng = Rng(0x204_4a71);
    for _ in 0..24 {
        let case = case(&mut rng);
        let source = case.source();
        let (checked, program) = compile(&source);
        let reference = eval::observe(&checked, FUEL);
        let (_, doc) = invar::native::run_program(&program, FUEL).unwrap();
        assert_eq!(doc["steps_used"], reference.steps_used, "{source}");
        match &reference.result {
            Ok(eval::Value::I32(value)) => assert_eq!(doc["result"]["value"], *value),
            Err(EvalError::Trap { category, span }) => {
                assert_eq!(*category, eval::TrapCategory::IntegerOverflow);
                assert_eq!(doc["status"], "trap");
                assert_eq!(doc["trap"]["category"], "INTEGER_OVERFLOW");
                assert_eq!(doc["trap"]["span"]["start"], span.start);
                assert_eq!(doc["trap"]["span"]["end"], span.end);
            }
            other => panic!("unexpected reference outcome {other:?}"),
        }
    }
}
