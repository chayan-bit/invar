#![forbid(unsafe_code)]
//! #54 scope addition: families are lowerings of checked normal-form nodes.
//! Every recognizer choice on the corpus must be the dispatcher's choice or a
//! named refusal, every run must equal the reference evaluator, a forged form
//! must be refused before any step, and no production path may reach a
//! family's recognizer.
#[allow(dead_code, reason = "only the stencil fixture is used here")]
#[path = "support/a1_grid_source.rs"]
mod grid_source;
#[path = "support/family_oracle.rs"]
mod oracle;

use invar::{
    eval,
    general_net::normal_form::{NormalForm, Rep, Ring},
    schedule::{self, Hooks},
};
use std::collections::BTreeMap;

const FUEL: u64 = 1_000_000;

fn hooks(forge: Option<fn(&mut NormalForm)>) -> Hooks {
    Hooks {
        forge,
        ..Hooks::default()
    }
}

/// One program's parity outcome: recognizer choices the dispatcher matched
/// (by family), choices it refused by name, and every disagreement.
struct Parity {
    matched: Vec<&'static str>,
    refused: usize,
    differences: Vec<String>,
}

fn parity(program: &oracle::Program) -> Parity {
    let dispatched = schedule::selection(&program.source, &program.mir, &Hooks::default())
        .unwrap_or_else(|e| panic!("{}: dispatcher error {e:?}", program.id));
    let production: Vec<_> = dispatched
        .selected
        .iter()
        .map(|s| (s.function.clone(), s.span, s.family))
        .collect();
    let expected = oracle::oracle(program);
    let mut out = Parity {
        matched: Vec::new(),
        refused: 0,
        differences: Vec::new(),
    };
    for choice in &expected {
        let named = dispatched.refusal.is_some()
            || dispatched
                .gaps
                .iter()
                .any(|g| g.function == choice.0 && g.span == choice.1);
        if production.contains(choice) {
            out.matched.push(choice.2);
        } else if named {
            out.refused += 1;
        } else {
            out.differences
                .push(format!("{}: recognizer {choice:?} not lowered", program.id));
        }
    }
    for choice in production.iter().filter(|c| !expected.contains(c)) {
        out.differences.push(format!(
            "{}: lowered {choice:?} without a recognizer",
            program.id
        ));
    }
    // explain, auto and calibrate describe the dispatcher's regions; on a
    // parity program that is the A0 document unchanged.
    let described = schedule::regions(&program.source, &program.mir).unwrap();
    let a0 = invar::admission::analyze(&program.source, &program.mir).unwrap();
    if out.differences.is_empty() && described.to_json() != a0.to_json() {
        out.differences
            .push(format!("{}: explain document changed", program.id));
    }
    out
}

fn selection_parity(dirs: &[&str], least: usize) {
    let results = oracle::each(dirs, parity);
    assert!(results.len() >= least, "corpus shrank to {}", results.len());
    let mut matched = BTreeMap::<&str, usize>::new();
    for family in results.iter().flat_map(|r| &r.matched) {
        *matched.entry(family).or_default() += 1;
    }
    let refused: usize = results.iter().map(|r| r.refused).sum();
    let differences: Vec<_> = results
        .iter()
        .flat_map(|r| &r.differences)
        .cloned()
        .collect();
    println!(
        "programs={} matched={matched:?} refused={refused}",
        results.len()
    );
    assert!(differences.is_empty(), "{}", differences.join("\n"));
    for family in oracle::FAMILIES {
        assert!(
            matched.get(family).copied().unwrap_or(0) > 0,
            "no {family} choice matched"
        );
    }
}

#[test]
fn issue_54_selection_parity_on_the_corpus() {
    selection_parity(oracle::CORPUS, 600);
}

/// The fast Tier 1 slice of the corpus pass above.
#[test]
fn issue_54_selection_parity_on_the_examples() {
    selection_parity(&["examples"], 40);
}

/// Results, traps and fuel equal; a tool error (an engine limit) by its code.
fn same(a: &eval::Observation, b: &eval::Observation) -> bool {
    let result = match (&a.result, &b.result) {
        (
            Err(eval::EvalError::Tool { code: x, .. }),
            Err(eval::EvalError::Tool { code: y, .. }),
        ) => x == y,
        (x, y) => x == y,
    };
    result && a.steps_used == b.steps_used && a.cleanup_trace == b.cleanup_trace
}

#[test]
fn issue_54_equal_to_reference_at_1_2_7_64_workers() {
    let results = oracle::each(oracle::CORPUS, |program| {
        let reference = eval::observe(&program.checked, FUEL);
        let mut published = 0;
        let mut differences = Vec::new();
        for workers in [1, 2, 7, 64] {
            match schedule::run_source(&program.source, &program.mir, FUEL, workers) {
                Ok(run) if same(&run.observation, &reference) => {
                    published += usize::from(!run.executed.is_empty());
                }
                Ok(run) => differences.push(format!(
                    "{} at {workers}: {:?} != {:?}",
                    program.id, run.observation.result, reference.result
                )),
                Err(e) => differences.push(format!("{} at {workers}: {e:?}", program.id)),
            }
        }
        (published, differences)
    });
    let differences: Vec<_> = results.iter().flat_map(|r| &r.1).cloned().collect();
    assert!(differences.is_empty(), "{}", differences.join("\n"));
    let published: usize = results.iter().map(|r| r.0).sum();
    println!("programs={} published_runs={published}", results.len());
    assert!(published > 0, "no corpus run published a region");
}

fn wrong_fingerprint(form: &mut NormalForm) {
    form.fingerprint[0] ^= 1;
}

fn mutated_ring(form: &mut NormalForm) {
    for cert in form.functions.iter_mut().flat_map(|f| {
        f.certificates
            .iter_mut()
            .chain(f.loops.iter_mut().flat_map(|l| &mut l.certificates))
    }) {
        if let Some(Rep::Affine(ring) | Rep::Matrix { ring, .. }) = &mut cert.rep {
            *ring = match *ring {
                Ring::Wrapping32 => Ring::Checked64(invar::general_net::normal_form::IntTy::U32),
                _ => Ring::Wrapping32,
            };
            return;
        }
    }
}

fn dropped_guard(form: &mut NormalForm) {
    for cert in form.functions.iter_mut().flat_map(|f| {
        f.certificates
            .iter_mut()
            .chain(f.loops.iter_mut().flat_map(|l| &mut l.certificates))
    }) {
        if !cert.guards.is_empty() {
            cert.guards.pop();
            return;
        }
    }
}

/// The checked form a forge would change, so a no-op forge is not counted.
fn changes(program: &oracle::Program, forge: fn(&mut NormalForm)) -> bool {
    use invar::general_net::{Limits, Net, normal_form};
    let Ok(net) = Net::encode(&program.mir, Limits::default()) else {
        return false;
    };
    let Ok(proof) = net.verify() else {
        return false;
    };
    let Ok(form) = normal_form::extract(&net, &proof) else {
        return false;
    };
    let mut forged = form.clone();
    forge(&mut forged);
    forged != form
}

#[test]
fn issue_54_forged_certificates_are_refused_before_any_step() {
    let forges = [
        (
            "wrong-fingerprint",
            wrong_fingerprint as fn(&mut NormalForm),
        ),
        ("mutated-ring", mutated_ring),
        ("dropped-guard", dropped_guard),
    ];
    let results = oracle::each(oracle::CORPUS, |program| {
        let honest = schedule::selection(&program.source, &program.mir, &Hooks::default()).unwrap();
        let mut forged = Vec::new();
        for (name, forge) in forges {
            if !changes(program, forge) {
                continue;
            }
            let hooks = hooks(Some(forge));
            let dispatched = schedule::selection(&program.source, &program.mir, &hooks).unwrap();
            let id = format!("{name} {}", program.id);
            assert_eq!(dispatched.refusal, Some("certificate-rejected"), "{id}");
            assert!(dispatched.selected.is_empty(), "{id}");
            let run =
                schedule::run_source_with_hooks(&program.source, &program.mir, FUEL, 4, &hooks)
                    .unwrap_or_else(|e| panic!("{id}: {e:?}"));
            // Builtin sort, spatial and f32 paths are not family regions.
            assert!(
                !run.executed.iter().any(|e| honest
                    .selected
                    .iter()
                    .any(|s| s.function == e.function && s.span == e.span)),
                "{id}: a forged form published a family region"
            );
            assert!(
                same(&run.observation, &eval::observe(&program.checked, FUEL)),
                "{id}"
            );
            forged.push(name);
        }
        forged
    });
    for (name, _) in forges {
        let n = results.iter().flatten().filter(|f| **f == name).count();
        println!("{name}: {n} programs");
        assert!(n > 0, "{name} changed no corpus form");
    }
}

#[test]
fn issue_54_a_certified_loop_without_a_lowering_is_a_typed_gap() {
    // L3 certifies the checked sum; no family lowers a checked fold.
    let source = "fn run(a: read Array<u32>, n: u32) -> u32 { var s = 0u32; for i in 0u32..n { s = s + a[i]; } return s; } fn main() -> i32 { let a = filled_u32(8u32, 2u32); if run(a, 8u32) != 16u32 { return 1; } return 0; }";
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), false).unwrap();
    let program = invar::mir::lower(&checked).unwrap();
    let dispatched = schedule::selection(source.as_bytes(), &program, &Hooks::default()).unwrap();
    assert!(dispatched.selected.is_empty());
    let reasons: Vec<_> = dispatched.gaps.iter().map(|g| g.reason.as_str()).collect();
    assert_eq!(reasons, ["no-lowering:reduce/lifted-sum"]);
    let run = schedule::run_source(source.as_bytes(), &program, FUEL, 4).unwrap();
    assert_eq!(run.observation.result, eval::observe(&checked, FUEL).result);
}

#[test]
fn issue_54_u32_and_f32_stencils_lower_to_the_grid_family() {
    // The PNF stencil admission once took only f32 grids, so a u32 stencil
    // the source contract binds lost its region under the dispatcher.
    for floating in [false, true] {
        for clamp in [false, true] {
            let source = grid_source::stencil(clamp, floating, 3, 5, 2);
            let checked =
                invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
            let program = invar::mir::lower(&checked).unwrap();
            let dispatched =
                schedule::selection(source.as_bytes(), &program, &Hooks::default()).unwrap();
            let families: Vec<_> = dispatched.selected.iter().map(|s| s.family).collect();
            assert_eq!(families, ["grid"], "floating={floating} clamp={clamp}");
            let run = schedule::run_source(source.as_bytes(), &program, FUEL, 4).unwrap();
            assert!(same(&run.observation, &eval::observe(&checked, FUEL)));
        }
    }
}

#[test]
fn issue_54_a_grid_contract_bound_to_other_mir_is_refused_by_both() {
    // Production binds the grid contract to the run's MIR by its function
    // fingerprint; the oracle must refuse the same mismatch, or parity could
    // not see that binding regress.
    let source = grid_source::stencil(false, true, 3, 5, 1);
    let program = |forge: bool| {
        let checked =
            invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let mut mir = invar::mir::lower(&checked).unwrap();
        if forge {
            // Still valid MIR, but no longer the function the contract binds.
            let step = mir.functions.iter_mut().find(|f| f.name == "step").unwrap();
            let local = invar::mir::Local {
                ty: invar::front::Type::U32,
                mutable: true,
                mode: None,
            };
            step.locals.insert("unused".into(), local);
            step.declaration_order.push("unused".into());
        }
        oracle::Program {
            id: "stencil".into(),
            source: source.clone().into_bytes(),
            checked,
            mir,
        }
    };
    let grid = |p: &oracle::Program| {
        let choices = oracle::oracle(p);
        choices.iter().filter(|c| c.2 == "grid").count()
    };
    let (honest, forged) = (program(false), program(true));
    assert_eq!(grid(&honest), 1);
    let error = schedule::selection(source.as_bytes(), &forged.mir, &Hooks::default()).unwrap_err();
    assert_eq!(error.code, "GRID_MIR_MISMATCH");
    assert_eq!(grid(&forged), 0);
}

#[test]
fn issue_54_scalar_direct_and_explicit_tc_contractions_lower_to_tensor() {
    // The PNF tensor admission once took only the strict `acc + a*b` step, so
    // an explicit `tc_mul_add` contraction lost its region under the dispatcher.
    for source in [
        include_str!("a1_tensor/scalar.inv"),
        include_str!("a1_tensor/direct.inv"),
        include_str!("a1_tensor/tc.inv"),
    ] {
        let checked =
            invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = invar::mir::lower(&checked).unwrap();
        let dispatched =
            schedule::selection(source.as_bytes(), &program, &Hooks::default()).unwrap();
        let families: Vec<_> = dispatched.selected.iter().map(|s| s.family).collect();
        assert_eq!(families, ["tensor"], "{source}");
        let run = schedule::run_source(source.as_bytes(), &program, FUEL, 4).unwrap();
        assert!(same(&run.observation, &eval::observe(&checked, FUEL)));
    }
}

/// L19's extent follows `len` through moves and allocations. The extractor
/// once followed an array literal one hop further than its checker, so a
/// chain of 9 or more maps from a literal had its whole form refused.
#[test]
fn issue_54_long_fusion_chains_keep_a_checkable_form() {
    for stages in 1..=20 {
        let mut source = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; let ws: Array<u32> = [4u32,0u32,6u32]; ".to_owned();
        let mut input = "xs".to_owned();
        for s in 0..stages {
            source.push_str(&format!("var out{s}: Array<u32> = filled_u32(len({input}),0u32); for i{s} in 0u32 .. len({input}) {{ out{s}[i{s}] = wrapping_add({input}[i{s}],ws[i{s}]); }} let stable{s}: Array<u32> = move out{s}; "));
            input = format!("stable{s}");
        }
        source.push_str(&format!(
            "if {input}[0u32] == 0u32 {{ return 1; }} return 0; }}"
        ));
        let checked =
            invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = invar::mir::lower(&checked).unwrap();
        let dispatched =
            schedule::selection(source.as_bytes(), &program, &Hooks::default()).unwrap();
        assert_eq!(dispatched.refusal, None, "stages={stages}");
        assert_eq!(dispatched.selected.len(), stages, "stages={stages}");
        let run = schedule::run_source(source.as_bytes(), &program, FUEL, 2).unwrap();
        assert_eq!(run.executed.len(), stages, "stages={stages}");
        assert!(same(&run.observation, &eval::observe(&checked, FUEL)));
    }
}

#[test]
fn issue_54_no_production_path_reaches_a_family_recognizer() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let registry = std::fs::read_to_string(root.join("src/schedule/families.rs")).unwrap();
    for family in oracle::FAMILIES {
        let file = std::fs::read_to_string(root.join(format!("src/schedule/families/{family}.rs")))
            .unwrap();
        // Production code ends where the file's test module begins.
        let production = file.split("#[cfg(test)]").next().unwrap();
        assert!(
            !production.contains("fn recognize"),
            "{family} still defines recognize"
        );
        assert!(production.contains("fn lower("), "{family} has no lowering");
        assert!(
            registry.contains(&format!("=> {family},")),
            "{family} is not registered as a lowering"
        );
    }
    let mut stack = vec![root.join("src")];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let text = std::fs::read_to_string(&path).unwrap();
                for family in oracle::FAMILIES {
                    assert!(
                        !text.contains(&format!("{family}::recognize(")),
                        "{} calls {family}::recognize",
                        path.display()
                    );
                }
            }
        }
    }
}
