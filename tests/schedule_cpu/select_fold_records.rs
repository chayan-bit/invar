//! #207: L7 select loops (`if a[i] < m { m = a[i]; at = i; }` and the other
//! compare forms) run on CPU workers through per-chunk local records and an
//! ordered combine. Value bits, companions, the if-taken count (and so fuel)
//! must equal the serial loop on the reference evaluator, MIR and every
//! worker count, including NaN, signed zero, ties, empty inputs and seeds
//! nothing beats.
use invar::{
    check::check,
    eval::{self, EvalError},
    front::parse,
    mir, mir_eval,
    schedule::{self, Fault, Hooks, PlanKind},
};

use crate::split_mix::Rng;

const FUEL: u64 = 1_000_000;
const WORKERS: [usize; 5] = [1, 2, 7, 16, 64];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    I32,
    U32,
    F32,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::F32 => "f32",
        }
    }

    /// Element and seed literals. The last entries are seeds only for the
    /// integer types (the extremes nothing beats); for f32 every entry,
    /// NaN included, is also an element.
    fn table(self) -> &'static [&'static str] {
        match self {
            Self::I32 => &["-3", "-1", "0", "2", "5", "-2147483647 - 1", "2147483647"],
            Self::U32 => &["0u32", "1u32", "2u32", "7u32", "9u32", "4294967295u32"],
            Self::F32 => &["-1.5", "-0.0", "0.0", "2.0", "inf", "-inf", "nan"],
        }
    }

    fn elements(self) -> usize {
        match self {
            Self::I32 | Self::U32 => 5,
            Self::F32 => 7,
        }
    }
}

struct Case {
    ty: Ty,
    /// `<`, `<=`, `>` or `>=`.
    op: &'static str,
    element_left: bool,
    seed: usize,
    values: Vec<usize>,
    companions: usize,
    /// Extra trips past `len(a)`: the loop traps out of bounds at `len(a)`.
    overrun: u32,
}

impl Case {
    fn literal(&self, k: usize) -> String {
        match self.ty.table()[k] {
            "inf" => "inf".into(),
            "-inf" => "ninf".into(),
            "nan" => "nan".into(),
            other => format!("({other})"),
        }
    }

    fn source(&self) -> String {
        let ty = self.ty.name();
        let items = self
            .values
            .iter()
            .map(|&k| self.literal(k))
            .collect::<Vec<_>>()
            .join(", ");
        let array = if self.values.is_empty() {
            match self.ty {
                Ty::I32 => "filled_i32(0u32, 0)".into(),
                Ty::U32 => "filled_u32(0u32, 0u32)".into(),
                Ty::F32 => "filled_f32(0u32, 0.0)".into(),
            }
        } else {
            format!("[{items}]")
        };
        let floats = if self.ty == Ty::F32 {
            "let inf: f32 = 1.0 / 0.0; let ninf: f32 = -1.0 / 0.0; let nan: f32 = 0.0 / 0.0; "
        } else {
            ""
        };
        let test = if self.element_left {
            format!("a[i] {} m", self.op)
        } else {
            let flipped = match self.op {
                "<" => ">",
                "<=" => ">=",
                ">" => "<",
                _ => "<=",
            };
            format!("m {flipped} a[i]")
        };
        let names = ["at", "bt"];
        let declare: String = names[..self.companions]
            .iter()
            .map(|c| format!("var {c}: u32 = 999u32; "))
            .collect();
        let assign: String = names[..self.companions]
            .iter()
            .map(|c| format!(" {c} = i;"))
            .collect();
        let code: String = names[..self.companions]
            .iter()
            .map(|c| format!("r = r * 1000 + i32_from_f32(f32_from_u32({c})); "))
            .collect();
        let hi = match self.overrun {
            0 => "len(a)".to_owned(),
            k => format!("len(a) + {k}u32"),
        };
        format!(
            "{} fn main() -> i32 {{ {floats}let a: Array<{ty}> = {array}; var m: {ty} = {}; {declare}for i in 0u32 .. {hi} {{ if {test} {{ m = a[i];{assign} }} }} var r = class(m); {code}return r; }}",
            self.class_fn(),
            self.literal(self.seed),
        )
    }

    /// `class(m)`: the table index of `m`, distinguishing -0, +0 and NaN.
    fn class_fn(&self) -> String {
        let ty = self.ty.name();
        let mut body = String::new();
        if self.ty == Ty::F32 {
            body += "if m != m { return 6; } if m == 0.0 { if 1.0 / m < 0.0 { return 1; } return 2; } if m == 1.0 / 0.0 { return 4; } if m == -1.0 / 0.0 { return 5; } ";
        }
        for (k, value) in self.ty.table().iter().enumerate() {
            if !matches!(*value, "-0.0" | "0.0" | "inf" | "-inf" | "nan") {
                body += &format!("if m == ({value}) {{ return {k}; }} ");
            }
        }
        format!("fn class(m: {ty}) -> i32 {{ {body}return 99; }}")
    }
}

fn case(rng: &mut Rng) -> Case {
    let ty = [Ty::I32, Ty::U32, Ty::F32][rng.below(3) as usize];
    let n = match rng.below(16) {
        0 => 0,
        1 => rng.below(64) as usize,
        _ => 64 + rng.below(160) as usize,
    };
    // A narrow palette makes long runs of ties.
    let palette: Vec<usize> = (0..1 + rng.below(3))
        .map(|_| rng.below(ty.elements() as u64) as usize)
        .collect();
    let values = (0..n)
        .map(|_| palette[rng.below(palette.len() as u64) as usize])
        .collect();
    Case {
        ty,
        op: ["<", "<=", ">", ">="][rng.below(4) as usize],
        element_left: rng.below(2) == 0,
        seed: rng.below(ty.table().len() as u64) as usize,
        values,
        companions: rng.below(3) as usize,
        overrun: 0,
    }
}

fn compile(source: &str) -> (invar::check::CheckedProgram, mir::Program) {
    let checked = check(parse(source.as_bytes()).unwrap(), true)
        .unwrap_or_else(|e| panic!("{source}: {e:?}"));
    let program = mir::lower(&checked).unwrap();
    (checked, program)
}

/// Every engine equals the reference evaluator in value, fuel and cleanup.
/// `published`, when given, is whether every cpu-parallel run published the
/// loop as a reduction. Returns the reference and whether the runs published
/// (all worker counts must decide alike).
fn assert_agrees(
    source: &str,
    fuel: u64,
    workers: &[usize],
    published: Option<bool>,
) -> (eval::Observation, bool) {
    let (checked, program) = compile(source);
    let reference = eval::observe(&checked, fuel);
    let lowered = mir_eval::observe(&program, fuel);
    let mut decided = None;
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
        let kinds: Vec<_> = run.executed.iter().map(|e| e.kind).collect();
        let this = match kinds[..] {
            [] => false,
            [PlanKind::Reduction] => true,
            _ => panic!("{workers} workers ran {kinds:?} for {source}"),
        };
        if let Some(expected) = published {
            assert_eq!(
                this, expected,
                "{workers} workers published={this} {source}"
            );
        }
        assert!(
            decided.replace(this).is_none_or(|before| before == this),
            "worker counts disagree on publishing {source}"
        );
    }
    (reference, decided.unwrap_or(false))
}

#[test]
fn select_records_match_the_serial_loop_over_2000_generated_cases() {
    let mut rng = Rng(0x207_5e1e);
    // NaN seed, NaN element, signed zero element, a tie between chunks, empty
    let mut seen = [0usize; 5];
    for index in 0..2000 {
        let case = case(&mut rng);
        let workers = if index < 300 {
            &WORKERS[..]
        } else {
            &WORKERS[index % 5..=index % 5]
        };
        let source = case.source();
        assert_agrees(&source, FUEL, workers, Some(!case.values.is_empty()));
        let table = case.ty.table();
        seen[0] += usize::from(table[case.seed] == "nan");
        seen[1] += usize::from(case.values.iter().any(|&k| table[k] == "nan"));
        seen[2] += usize::from(case.values.iter().any(|&k| table[k].ends_with("0.0")));
        let n = case.values.len();
        seen[3] += usize::from(n >= 64 && case.values[0] == case.values[n - 1]);
        seen[4] += usize::from(n == 0);
    }
    assert!(seen.iter().all(|n| *n >= 20), "coverage {seen:?}");
}

#[test]
fn select_records_fuel_prefix_is_exact() {
    let mut rng = Rng(0x207_f0e1);
    for _ in 0..40 {
        let mut case = case(&mut rng);
        if case.values.is_empty() {
            case.values = vec![0; 70];
        }
        let source = case.source();
        let (reference, _) = assert_agrees(&source, FUEL, &WORKERS, Some(true));
        // The fuel prefix is exact: the reference's own count suffices and one
        // step less exhausts the limit at the same point on every engine. The
        // loop is followed by `class(m)`, so the region may still fit.
        assert_agrees(&source, reference.steps_used, &WORKERS, Some(true));
        assert_agrees(&source, reference.steps_used - 1, &WORKERS, None);
    }
}

#[test]
fn select_records_publish_exactly_when_the_loop_fits_the_fuel() {
    // Ties across chunks, a NaN and signed zeros. Every fuel limit up to the
    // reference's count agrees on every engine, and the region is published
    // for exactly the limits from some threshold on: never while the limit
    // ends inside the loop, always once the whole loop fits.
    let values = [0, 2, 6, 1, 1, 6, 3, 2]
        .into_iter()
        .cycle()
        .take(96)
        .collect();
    let source = Case {
        ty: Ty::F32,
        op: "<=",
        element_left: true,
        seed: 4,
        values,
        companions: 1,
        overrun: 0,
    }
    .source();
    let (reference, _) = assert_agrees(&source, FUEL, &[1, 7], Some(true));
    let published: Vec<bool> = (0..=reference.steps_used)
        .map(|fuel| assert_agrees(&source, fuel, &[1, 7], None).1)
        .collect();
    let threshold = published
        .iter()
        .position(|&p| p)
        .expect("published at the full count");
    assert!(published[threshold..].iter().all(|&p| p), "{published:?}");
    // Every one of the 96 trips costs at least two steps below the threshold.
    assert!(threshold >= 96 * 2, "threshold {threshold}");
}

#[test]
fn select_records_an_out_of_bounds_trip_stays_serial_and_traps_at_len() {
    for ty in [Ty::I32, Ty::U32, Ty::F32] {
        for overrun in [1, 5] {
            let case = Case {
                ty,
                op: "<",
                element_left: true,
                seed: 1,
                values: vec![0, 2, 1, 0, 3],
                companions: 1,
                overrun,
            };
            let source = case.source();
            let (reference, _) = assert_agrees(&source, FUEL, &WORKERS, Some(false));
            assert!(
                matches!(
                    reference.result,
                    Err(EvalError::Trap {
                        category: eval::TrapCategory::IndexOutOfBounds,
                        ..
                    })
                ),
                "{source}"
            );
        }
    }
}

#[test]
fn select_records_mutation_that_takes_the_rightmost_tie_is_caught() {
    // Equal minima at 10 and 100: under 7 workers they sit in different chunks.
    let mut values = vec![4; 128];
    values[10] = 0;
    values[100] = 0;
    let source = Case {
        ty: Ty::I32,
        op: "<",
        element_left: true,
        seed: 6,
        values,
        companions: 1,
        overrun: 0,
    }
    .source();
    let (checked, program) = compile(&source);
    let reference = eval::observe(&checked, FUEL);
    let honest = schedule::run_source(source.as_bytes(), &program, FUEL, 7).unwrap();
    assert_eq!(honest.observation.result, reference.result);
    let hooks = Hooks {
        fault: Fault::RightmostTie,
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
fn select_records_admit_a_2_pow_23_argmin_and_the_receipt_names_the_law() {
    let source = "fn main() -> i32 { let a = filled_i32(8388608u32, 3); var m: i32 = 7; var at: u32 = 0u32; for i in 0u32 .. 8388608u32 { if a[i] < m { m = a[i]; at = i; } } return m * 10 + i32_from_f32(f32_from_u32(at)); }";
    let (_, program) = compile(source);
    let report = schedule::regions(source.as_bytes(), &program)
        .unwrap()
        .to_json();
    let region = &report["regions"][0];
    assert_eq!(region["status"], "admitted");
    assert_eq!(region["chosen_plan"], "select-records-if-guard");
    assert_eq!(region["execution"]["kind"], "reduction");
    assert_eq!(region["execution"]["law"], "select-min-first");
    assert_eq!(
        region["rule_identities"],
        serde_json::json!([invar::select_fold::RULE_REVISION])
    );
    let run = schedule::run_source(source.as_bytes(), &program, 200_000_000, 16).unwrap();
    assert_eq!(run.observation.result, Ok(eval::Value::I32(30)));
    assert_eq!(run.executed.len(), 1);
    assert_eq!(run.executed[0].kind, PlanKind::Reduction);
    assert_eq!(run.chunks.len(), 16, "the records ran on 16 workers");
}

#[test]
fn select_records_refuse_shapes_outside_the_law() {
    let refused = [
        // The selected value is not the element itself.
        "fn main() -> i32 { let a: Array<i32> = [3, 1, 2]; var m: i32 = 9; for i in 0u32 .. len(a) { if a[i] < m { m = a[i] + 1; } } return m; }",
        // An else arm.
        "fn main() -> i32 { let a: Array<i32> = [3, 1, 2]; var m: i32 = 9; var at: u32 = 0u32; for i in 0u32 .. len(a) { if a[i] < m { m = a[i]; } else { at = i; } } return m; }",
        // A companion that is not the induction.
        "fn main() -> i32 { let a: Array<i32> = [3, 1, 2]; var m: i32 = 9; var at: u32 = 0u32; for i in 0u32 .. len(a) { if a[i] < m { m = a[i]; at = i + 1u32; } } return m; }",
        // The compare is against another element.
        "fn main() -> i32 { let a: Array<i32> = [3, 1, 2]; let b: Array<i32> = [1, 1, 1]; var m: i32 = 9; for i in 0u32 .. len(a) { if a[i] < m { m = b[i]; } } return m; }",
        // Equality is not an order.
        "fn main() -> i32 { let a: Array<i32> = [3, 1, 2]; var m: i32 = 1; var at: u32 = 0u32; for i in 0u32 .. len(a) { if a[i] == m { m = a[i]; at = i; } } return m; }",
    ];
    for source in refused {
        let (_, program) = compile(source);
        let report = schedule::regions(source.as_bytes(), &program)
            .unwrap()
            .to_json();
        assert_ne!(
            report["regions"][0]["chosen_plan"], "select-records-if-guard",
            "{source}"
        );
        let (checked, _) = compile(source);
        let reference = eval::observe(&checked, FUEL);
        for workers in WORKERS {
            let run = schedule::run_source(source.as_bytes(), &program, FUEL, workers).unwrap();
            assert_eq!(run.observation.result, reference.result, "{source}");
            assert_eq!(run.observation.steps_used, reference.steps_used, "{source}");
        }
    }
}

#[cfg(feature = "native")]
#[test]
fn select_records_native_serial_matches_value_and_fuel() {
    let mut rng = Rng(0x207_4a71);
    for _ in 0..24 {
        let source = case(&mut rng).source();
        let (checked, program) = compile(&source);
        let reference = eval::observe(&checked, FUEL);
        let (_, doc) = invar::native::run_program(&program, FUEL).unwrap();
        assert_eq!(doc["steps_used"], reference.steps_used, "{source}");
        match &reference.result {
            Ok(eval::Value::I32(value)) => assert_eq!(doc["result"]["value"], *value, "{source}"),
            other => panic!("unexpected reference outcome {other:?}"),
        }
    }
}
