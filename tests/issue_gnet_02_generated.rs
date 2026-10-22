//! GNET-02 randomized well-typed compositions. A seeded generator composes
//! independent source fragments (checked arithmetic with possible traps,
//! branches, counted and while loops, recursion across the frame limit, owned
//! arrays, records, variants, UTF-8 results, borrowed mutable views, byte
//! slices, scoped cleanup, early returns, untaken traps and short-circuiting)
//! into one program. Every program must check; the reference evaluator, the MIR
//! evaluator and the bounded CPU net oracle must agree exactly on result, source
//! steps and cleanup, and on every fuel prefix of a sampled subset.
#![forbid(unsafe_code)]

#[path = "support/gnet_agree.rs"]
mod agree;
#[path = "support/gnet_generator.rs"]
mod generator;

use agree::same;
use general_net::{Limits, Net, oracle};
use generator::{PROGRAMS, Rng, SEED, program};
use invar::{check, eval, front, general_net, mir, mir_eval};
use std::collections::BTreeMap;

fn oracle_run(net: &Net, fuel: u64) -> oracle::Observation {
    oracle::observe(
        net,
        &net.verify().unwrap(),
        oracle::Limits {
            fuel,
            ..oracle::Limits::default()
        },
    )
    .unwrap()
}

#[test]
fn issue_gnet_02_generated_compositions_agree_with_both_reference_engines() {
    let mut rng = Rng(SEED);
    let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
    let mut outcomes: BTreeMap<String, usize> = BTreeMap::new();
    let mut prefixes = 0usize;
    for index in 0..PROGRAMS {
        let (source, used) = program(&mut rng);
        let checked = check::check(
            front::parse(source.as_bytes()).unwrap_or_else(|e| panic!("{e:?}\n{source}")),
            true,
        )
        .unwrap_or_else(|e| panic!("{e:?}\n{source}"));
        let mir = mir::lower(&checked).unwrap();
        let net = Net::lower(&checked, Limits::default()).unwrap();
        let reference = eval::observe(&checked, 100_000);
        same(&mir_eval::observe(&mir, 100_000), &reference, &source);
        let observed = oracle_run(&net, 100_000);
        assert_eq!(observed.failure, None, "{source}");
        same(&observed.semantic, &reference, &source);
        if index % 8 == 0 {
            for fuel in 1..=reference.steps_used + 1 {
                let expected = eval::observe(&checked, fuel);
                same(&mir_eval::observe(&mir, fuel), &expected, &source);
                same(&oracle_run(&net, fuel).semantic, &expected, &source);
                prefixes += 1;
            }
        }
        for kind in used {
            *kinds.entry(kind).or_default() += 1;
        }
        let outcome = match &reference.result {
            Ok(_) => "value".to_owned(),
            Err(eval::EvalError::Trap { category, .. }) => format!("trap_{category:?}"),
            Err(eval::EvalError::Tool { code, .. }) => format!("tool_{code}"),
            Err(error) => format!("{error:?}"),
        };
        *outcomes.entry(outcome).or_default() += 1;
    }
    println!(
        "gnet_02_generated seed={SEED:#x} programs={PROGRAMS} fuel_prefixes={prefixes} outcomes={outcomes:?}"
    );
    println!("gnet_02_generated fragment_kinds={} {kinds:?}", kinds.len());
    // Every fragment kind and the trap/limit classes are actually exercised.
    assert_eq!(kinds.len(), 16, "{kinds:?}");
    for outcome in [
        "value",
        "trap_IntegerOverflow",
        "trap_DivisionByZero",
        "trap_IndexOutOfBounds",
        "tool_CALL_DEPTH_LIMIT",
    ] {
        assert!(outcomes.contains_key(outcome), "{outcome}: {outcomes:?}");
    }
    assert!(prefixes > 1000, "{prefixes}");
}
