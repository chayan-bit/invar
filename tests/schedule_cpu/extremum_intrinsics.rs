//! #207: `min_of`, `max_of`, `argmin` and `argmax` over `Array<i32>`,
//! `Array<u32>`, `Array<f32>` and `Array<f64>`. The oracle is the spec
//! definition, computed here independently of the engines: `min_of` is the
//! left fold of the library `min` (minimumNumber, `-0 < +0`) with a
//! canonical NaN, `argmin` the lowest index with its bits (0 when all are
//! NaN), and an empty input traps INDEX_OUT_OF_BOUNDS at the call. The
//! reference evaluator, MIR and cpu-parallel at 1, 2, 7, 16 and 64 workers
//! must equal it in value, fuel and cleanup.
use invar::{
    check::check,
    eval::{self, EvalError, TrapCategory},
    front::parse,
    mir, mir_eval,
    numeric_contract::{F32Bits, F64Bits},
    schedule::{self, Fault, Hooks},
};

use crate::split_mix::Rng;

const FUEL: u64 = 10_000_000;
const WORKERS: [usize; 5] = [1, 2, 7, 16, 64];
const KINDS: [&str; 4] = ["min_of", "max_of", "argmin", "argmax"];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    I32,
    U32,
    F32,
    F64,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    /// Element literals; floats name `inf`, `ninf` and `nan` locals.
    fn table(self) -> &'static [&'static str] {
        match self {
            Self::I32 => &["-3", "-1", "0", "2", "5", "(-2147483647 - 1)", "2147483647"],
            Self::U32 => &["0u32", "1u32", "2u32", "7u32", "9u32", "4294967295u32"],
            Self::F32 | Self::F64 => &["-1.5", "-0.0", "0.0", "2.0", "inf", "ninf", "nan"],
        }
    }

    fn float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }
}

struct Case {
    ty: Ty,
    kind: &'static str,
    values: Vec<usize>,
}

impl Case {
    fn source(&self) -> String {
        let ty = self.ty.name();
        let table = self.ty.table();
        let floats = if self.ty.float() {
            format!(
                "let inf: {ty} = 1.0 / 0.0; let ninf: {ty} = -1.0 / 0.0; let nan: {ty} = 0.0 / 0.0; "
            )
        } else {
            String::new()
        };
        let array = if self.values.is_empty() {
            format!(
                "filled_{ty}(0u32, {})",
                if self.ty.float() { "0.0" } else { table[2] }
            )
        } else {
            let items: Vec<_> = self.values.iter().map(|&k| table[k]).collect();
            format!("[{}]", items.join(", "))
        };
        let (result, class) = if self.kind.starts_with("arg") {
            (
                "u32",
                "if r > 4000u32 { return 9999; } return i32_from_f32(f32_from_u32(r));".to_owned(),
            )
        } else {
            (ty, self.class())
        };
        format!(
            "fn class(r: {result}) -> i32 {{ {class} }} fn main() -> i32 {{ {floats}let a: Array<{ty}> = {array}; let r = {}(a); return class(r); }}",
            self.kind
        )
    }

    /// The table index of a value result, telling -0, +0 and NaN apart.
    fn class(&self) -> String {
        let table = self.ty.table();
        let mut body = String::new();
        if self.ty.float() {
            body += "if r != r { return 6; } if r == 0.0 { if 1.0 / r < 0.0 { return 1; } return 2; } if r == 1.0 / 0.0 { return 4; } if r == -1.0 / 0.0 { return 5; } ";
        }
        for (k, value) in table.iter().enumerate() {
            if !matches!(*value, "-0.0" | "0.0" | "inf" | "ninf" | "nan") {
                body += &format!("if r == {value} {{ return {k}; }} ");
            }
        }
        body + "return 99;"
    }

    /// The spec result as `main` returns it, or `None` for the empty trap.
    fn expected(&self) -> Option<i32> {
        let table = self.ty.table();
        let n = self.values.len();
        if n == 0 {
            return None;
        }
        let max = self.kind.contains("max");
        if !self.ty.float() {
            let parse = |k: usize| -> i64 {
                match table[k].trim_matches(|c| c == '(' || c == ')') {
                    "-2147483647 - 1" => i64::from(i32::MIN),
                    other => other.trim_end_matches("u32").parse().unwrap(),
                }
            };
            let numbers: Vec<i64> = self.values.iter().map(|&k| parse(k)).collect();
            let best = if max {
                numbers.iter().max()
            } else {
                numbers.iter().min()
            };
            let best = *best.unwrap();
            let at = numbers.iter().position(|&x| x == best).unwrap();
            let class = (0..table.len()).position(|k| parse(k) == best).unwrap();
            return Some(if self.kind.starts_with("arg") {
                at
            } else {
                class
            } as i32);
        }
        let encodings = |ks: &mut dyn Iterator<Item = usize>| -> Vec<u64> {
            ks.map(|k| match self.ty {
                Ty::F32 => u64::from((literal(table[k]) as f32).to_bits()),
                _ => literal(table[k]).to_bits(),
            })
            .collect()
        };
        let bits = encodings(&mut self.values.iter().copied());
        let classes = encodings(&mut (0..table.len()));
        let fold = self.float_fold(max);
        let at = bits.iter().position(|&b| Some(b) == fold).unwrap_or(0);
        let class = fold.map_or(6, |b| classes.iter().position(|&c| c == b).unwrap_or(99));
        Some(if self.kind.starts_with("arg") {
            at
        } else {
            class
        } as i32)
    }

    /// The library fold's encoding, or `None` when every element is NaN.
    fn float_fold(&self, max: bool) -> Option<u64> {
        let table = self.ty.table();
        if self.ty == Ty::F32 {
            let items: Vec<F32Bits> = self
                .values
                .iter()
                .map(|&k| F32Bits::from_bits((literal(table[k]) as f32).to_bits()))
                .collect();
            let fold = items[1..].iter().fold(
                items[0],
                |acc, &x| if max { acc.max(x) } else { acc.min(x) },
            );
            (!fold.is_nan()).then(|| u64::from(fold.to_bits()))
        } else {
            let items: Vec<F64Bits> = self
                .values
                .iter()
                .map(|&k| F64Bits::from_bits(literal(table[k]).to_bits()))
                .collect();
            let fold = items[1..].iter().fold(
                items[0],
                |acc, &x| if max { acc.max(x) } else { acc.min(x) },
            );
            (!fold.is_nan()).then(|| fold.to_bits())
        }
    }
}

/// A float table literal; every entry is exact in `f32` and `f64`.
fn literal(text: &str) -> f64 {
    match text {
        "inf" => f64::INFINITY,
        "ninf" => f64::NEG_INFINITY,
        "nan" => f64::NAN,
        other => other.parse().unwrap(),
    }
}

fn case(rng: &mut Rng) -> Case {
    let ty = [Ty::I32, Ty::U32, Ty::F32, Ty::F64][rng.below(4) as usize];
    let n = match rng.below(12) {
        0 => 0,
        1 => 1 + rng.below(8) as usize,
        _ => 64 + rng.below(200) as usize,
    };
    // A narrow palette makes long runs of ties.
    let elements = ty.table().len() as u64;
    let palette: Vec<usize> = (0..1 + rng.below(3))
        .map(|_| rng.below(elements) as usize)
        .collect();
    let values = (0..n)
        .map(|_| palette[rng.below(palette.len() as u64) as usize])
        .collect();
    Case {
        ty,
        kind: KINDS[rng.below(4) as usize],
        values,
    }
}

fn compile(source: &str) -> (invar::check::CheckedProgram, mir::Program) {
    let checked = check(parse(source.as_bytes()).unwrap(), true)
        .unwrap_or_else(|e| panic!("{source}: {e:?}"));
    let program = mir::lower(&checked).unwrap();
    (checked, program)
}

/// Every engine equals the reference, which equals the spec result.
fn assert_agrees(case: &Case, workers: &[usize]) {
    let source = case.source();
    let (checked, program) = compile(&source);
    let reference = eval::observe(&checked, FUEL);
    match case.expected() {
        Some(value) => assert_eq!(reference.result, Ok(eval::Value::I32(value)), "{source}"),
        None => assert!(
            matches!(
                reference.result,
                Err(EvalError::Trap {
                    category: TrapCategory::IndexOutOfBounds,
                    ..
                })
            ),
            "{source}"
        ),
    }
    let lowered = mir_eval::observe(&program, FUEL);
    assert_eq!(lowered.result, reference.result, "MIR {source}");
    assert_eq!(
        lowered.steps_used, reference.steps_used,
        "MIR fuel {source}"
    );
    for &workers in workers {
        let run = schedule::run_source(source.as_bytes(), &program, FUEL, workers).unwrap();
        assert_eq!(
            run.observation.result, reference.result,
            "{workers} workers {source}"
        );
        assert_eq!(
            run.observation.steps_used, reference.steps_used,
            "{workers} fuel {source}"
        );
        assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
        assert_eq!(
            run.selections.len(),
            1,
            "{workers} workers ran the call {source}"
        );
        assert_eq!(run.selections[0].elements, case.values.len());
    }
}

#[test]
fn extremum_intrinsics_match_the_spec_over_2000_generated_cases() {
    let mut rng = Rng(0x207_1471);
    // NaN element, all NaN, signed zero, a tie, empty, per type
    let mut seen = [0usize; 6];
    let mut types = [0usize; 4];
    for index in 0..2000 {
        let case = case(&mut rng);
        let workers = if index < 300 {
            &WORKERS[..]
        } else {
            &WORKERS[index % 5..=index % 5]
        };
        assert_agrees(&case, workers);
        let table = case.ty.table();
        let name = |k: &usize| table[*k];
        seen[0] += usize::from(case.values.iter().any(|k| name(k) == "nan"));
        seen[1] +=
            usize::from(!case.values.is_empty() && case.values.iter().all(|k| name(k) == "nan"));
        seen[2] += usize::from(
            case.values.iter().any(|k| name(k) == "-0.0")
                && case.values.iter().any(|k| name(k) == "0.0"),
        );
        seen[3] += usize::from(
            case.values.len() >= 64 && case.values[0] == case.values[case.values.len() - 1],
        );
        seen[4] += usize::from(case.values.is_empty());
        seen[5] += usize::from(case.values.len() >= 64);
        types[case.ty as usize] += 1;
    }
    assert!(seen.iter().all(|n| *n >= 20), "coverage {seen:?}");
    assert!(types.iter().all(|n| *n >= 300), "types {types:?}");
}

#[test]
fn extremum_intrinsics_empty_input_traps_at_the_call_on_every_engine() {
    for ty in [Ty::I32, Ty::U32, Ty::F32, Ty::F64] {
        for kind in KINDS {
            let case = Case {
                ty,
                kind,
                values: Vec::new(),
            };
            let source = case.source();
            let (checked, _) = compile(&source);
            let reference = eval::observe(&checked, FUEL);
            let call = source.find(&format!("{kind}(a)")).unwrap();
            match reference.result {
                Err(EvalError::Trap {
                    category: TrapCategory::IndexOutOfBounds,
                    span,
                }) => {
                    assert_eq!(
                        (span.start, span.end),
                        (call, call + kind.len() + 3),
                        "{source}"
                    );
                }
                other => panic!("{source}: {other:?}"),
            }
            assert_agrees(&case, &WORKERS);
        }
    }
}

#[test]
fn extremum_intrinsics_run_on_16_workers_for_2_pow_22_elements() {
    let source = "fn main() -> i32 { var a = filled_f32(4194304u32, 2.5); a[1000u32] = 0.0 / 0.0; a[3000000u32] = -1.0; a[4000000u32] = -1.0; let at = argmin(a); let m = min_of(a); if m == -1.0 { return i32_from_f32(f32_from_u32(at)); } return 0; }";
    let (checked, program) = compile(source);
    let reference = eval::observe(&checked, 200_000_000);
    assert_eq!(reference.result, Ok(eval::Value::I32(3_000_000)));
    let run = schedule::run_source(source.as_bytes(), &program, 200_000_000, 16).unwrap();
    assert_eq!(run.observation.result, reference.result);
    assert_eq!(run.observation.steps_used, reference.steps_used);
    assert_eq!(run.selections.len(), 2);
    assert!(
        run.selections.iter().all(|s| s.workers == 16),
        "{:?}",
        run.selections
    );
}

#[test]
fn extremum_intrinsics_mutation_that_takes_the_rightmost_tie_is_caught() {
    // Equal minima at 10 and 100: under 7 workers they sit in different chunks.
    let mut values = vec![4; 128];
    values[10] = 0;
    values[100] = 0;
    let case = Case {
        ty: Ty::I32,
        kind: "argmin",
        values,
    };
    let source = case.source();
    let (checked, program) = compile(&source);
    let reference = eval::observe(&checked, FUEL);
    assert_eq!(reference.result, Ok(eval::Value::I32(10)));
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
fn extremum_intrinsics_reject_ill_formed_calls() {
    let rejected = [
        // Two operands.
        "fn main() -> i32 { let a: Array<i32> = [1, 2]; return min_of(a, a); }",
        // Not a named array.
        "fn main() -> i32 { return min_of(filled_i32(3u32, 1)); }",
        // Not a numeric array.
        "fn main() -> i32 { let a: Array<Bool> = [true]; if max_of(a) { return 1; } return 0; }",
        // The index is u32.
        "fn main() -> i32 { let a: Array<i32> = [1, 2]; let k: i32 = argmin(a); return k; }",
        // The value has the element type.
        "fn main() -> i32 { let a: Array<u32> = [1u32]; let k: i32 = max_of(a); return k; }",
        // No operand.
        "fn main() -> i32 { return i32_from_f32(f32_from_u32(argmax())); }",
    ];
    for source in rejected {
        assert!(
            check(parse(source.as_bytes()).unwrap(), true).is_err(),
            "{source}"
        );
    }
    // The names are reserved.
    let shadow = "fn argmin(x: i32) -> i32 { return x; } fn main() -> i32 { return argmin(1); }";
    assert!(check(parse(shadow.as_bytes()).unwrap(), true).is_err());
}

#[test]
fn extremum_intrinsics_gnet_refuses_as_a_pending_intrinsic() {
    use invar::general_net::{Limits, Net, oracle};
    let source = "fn main() -> i32 { let a: Array<i32> = [3, 1, 2]; return min_of(a); }";
    let (checked, _) = compile(source);
    let refused = match Net::lower(&checked, Limits::default()) {
        Err(_) => true,
        Ok(net) => {
            let observed =
                oracle::observe(&net, &net.verify().unwrap(), oracle::Limits::default()).unwrap();
            observed.failure == Some(oracle::ToolFailure::Unsupported)
        }
    };
    assert!(
        refused,
        "GNET must refuse the call until it has an oracle rule"
    );
}

#[cfg(feature = "native")]
#[test]
fn extremum_intrinsics_native_serial_matches_value_trap_and_fuel() {
    let mut rng = Rng(0x207_4a72);
    for index in 0..40 {
        let mut case = case(&mut rng);
        if index < 4 {
            case.values.clear();
        }
        let source = case.source();
        let (checked, program) = compile(&source);
        let reference = eval::observe(&checked, FUEL);
        let (_, doc) = invar::native::run_program(&program, FUEL).unwrap();
        assert_eq!(doc["steps_used"], reference.steps_used, "{source}");
        match &reference.result {
            Ok(eval::Value::I32(value)) => assert_eq!(doc["result"]["value"], *value, "{source}"),
            Err(EvalError::Trap { category, span }) => {
                assert_eq!(doc["status"], "trap", "{source}");
                assert_eq!(doc["trap"]["category"], category.code(), "{source}");
                assert_eq!(doc["trap"]["span"]["start"], span.start);
                assert_eq!(doc["trap"]["span"]["end"], span.end);
            }
            other => panic!("unexpected reference outcome {other:?}"),
        }
    }
}

#[test]
fn extremum_intrinsics_beside_sort_and_spatial_calls_run_on_workers() {
    // Sort and spatial programs take their own CPU runners; the call must
    // still reach the workers and record its selection there.
    let tail = "var a = filled_f32(100000u32, 2.5); a[70000u32] = -1.0; a[90000u32] = -1.0; \
                a[5u32] = 0.0 / 0.0; return i32_from_f32(f32_from_u32(argmin(a)));";
    let sources = [
        format!("fn main() -> i32 {{ var s: Array<u32> = [3u32, 1u32, 2u32]; sort(s); {tail} }}"),
        format!(
            "fn main() -> i32 {{ let p: Array<f32> = [0.0, 1.0]; let grid = voxel_grid_build(p, p, p, 1.0); \
             let q: Array<f32> = [0.0]; let near = knn(grid, q, q, q, 1u32); {tail} }}"
        ),
    ];
    for source in &sources {
        let (checked, program) = compile(source);
        let reference = eval::observe(&checked, FUEL);
        assert_eq!(reference.result, Ok(eval::Value::I32(70_000)), "{source}");
        let run = schedule::run_source(source.as_bytes(), &program, FUEL, 7).unwrap();
        assert_eq!(run.observation.result, reference.result, "{source}");
        assert_eq!(run.observation.steps_used, reference.steps_used, "{source}");
        assert_eq!(run.selections.len(), 1, "{source}: {:?}", run.selections);
        assert!(
            run.selections[0].workers > 1,
            "{source}: {:?}",
            run.selections
        );
    }
}
