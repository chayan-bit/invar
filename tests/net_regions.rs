#![forbid(unsafe_code)]
//! P1a region partition: regions are cut on the verified net's compiled MIR,
//! carry boundaries, live-ins and live-outs, effects and ownership, and
//! `verify` rejects any partition that differs from the one re-derived.

#[path = "support/normal_form.rs"]
mod support;

use invar::general_net::regions::{self, Counting, RegionKind};

const SOURCE: &str = "\
fn scale(xs: mut Array<i32>, n: u32, k: i32) -> Unit { for i in 0u32 .. n { xs[i] = xs[i] * k; } }
fn first(xs: read Array<i32>, n: u32, key: i32) -> u32 { var i = 0u32; while i < n { if xs[i] == key { return i; } i = i + 1u32; } return n; }
fn tally(n: u32) -> u32 { var i = 0u32; var s = 0u32; while i < n { s = s + i; i = i + 1u32; } return s; }
fn drain(n: i32) -> i32 { var x = n; var steps = 0; while x > 1 { if x % 2 == 0 { x = x / 2; } else { x = 3 * x + 1; } steps = steps + 1; } return steps; }
fn run(n: u32) -> i32 { var xs = filled_i32(n, 3); scale(xs, n, 2); if first(xs, n, 6) == n { return 0; } if tally(n) == 3u32 { return 1; } return drain(7); }
";

#[test]
fn partition_records_regions_and_verifies() {
    let (net, proof) = support::compile(SOURCE);
    let partition = regions::partition(&net, &proof).expect("partition");
    assert_eq!(partition.version, regions::VERSION);
    let function = |name: &str| {
        partition
            .functions
            .iter()
            .find(|f| f.name == name)
            .expect("function")
    };

    let scale = function("scale");
    assert!(scale.tree.is_ok());
    assert!(!scale.pure, "a mut parameter makes the function impure");
    let [body, lp] = &scale.regions[..] else {
        panic!("scale regions: {:?}", scale.regions)
    };
    assert_eq!(body.kind, RegionKind::Function);
    assert_eq!(lp.kind, RegionKind::Loop(Counting::For { var: "i".into() }));
    assert_eq!(lp.parent, Some(body.id));
    assert!(lp.live_out.contains(&"xs".to_owned()));
    assert!(lp.live_in.contains(&"k".to_owned()));
    assert!(lp.array_writes.contains(&"xs".to_owned()));
    assert!(lp.effects.may_trap);
    assert!(!lp.effects.early_exit);
    assert!(!lp.exits.is_empty());

    let first = function("first");
    assert!(first.pure);
    let lp = &first.regions[1];
    // The increment after an `if` that returns is placed in the surviving arm,
    // so a `while` with an early exit is not a counted loop.
    assert_eq!(lp.kind, RegionKind::Loop(Counting::Uncounted));
    assert!(lp.effects.early_exit);

    let tally = function("tally");
    assert_eq!(
        tally.regions[1].kind,
        RegionKind::Loop(Counting::CountedWhile {
            induction: "i".into()
        })
    );
    assert_eq!(
        tally.regions[1].live_out,
        vec!["i".to_owned(), "s".to_owned()]
    );

    let drain = function("drain");
    assert_eq!(drain.regions[1].kind, RegionKind::Loop(Counting::Uncounted));

    let run = function("run");
    let body = &run.regions[0];
    assert_eq!(body.effects.mut_loans, vec!["xs".to_owned()]);
    assert!(body.effects.allocates);
    assert_eq!(body.effects.calls.len(), 4);
    assert!(!run.pure, "calls an impure function");

    assert_eq!(regions::verify(&net, &proof, &partition), Ok(()));
}

#[test]
fn verify_rejects_a_tampered_partition() {
    let (net, proof) = support::compile(SOURCE);
    let partition = regions::partition(&net, &proof).expect("partition");
    let tamper: [&dyn Fn(&mut regions::Partition); 6] = [
        &|p| p.version += 1,
        &|p| p.fingerprint[0] ^= 1,
        &|p| p.functions[0].regions[1].live_in.push("ghost".into()),
        &|p| p.functions[0].regions[1].effects.host = true,
        &|p| p.functions[0].regions[1].kind = RegionKind::Loop(Counting::Uncounted),
        &|p| p.functions[0].pure = true,
    ];
    for (k, change) in tamper.iter().enumerate() {
        let mut forged = partition.clone();
        change(&mut forged);
        assert_eq!(
            regions::verify(&net, &proof, &forged),
            Err(regions::Error::Mismatch),
            "tamper {k}"
        );
    }
    let (other_net, other_proof) = support::compile("fn f(n: u32) -> u32 { return n; }");
    assert!(regions::verify(&other_net, &other_proof, &partition).is_err());
}
