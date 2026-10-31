//! GNET-03 parallel slice and device fault injection, shared by the CPU model,
//! CUDA and Metal suites. The slice (`tests/fixtures/gnet_parallel`) is
//! ordinary source with independent call sites and wide, input-dependent
//! frontiers; every run is compared word for word with the CPU net oracle.
#![allow(dead_code)]

use crate::device::{Frontier, Source, agree, config, corpus, drive, reference};
use invar::{
    eval::Value,
    front::Type,
    general_net::{
        compiled::Target,
        device_host::{Backend, Limits, Plan},
    },
};

/// Every slice fixture defines `run(n, seed)`; these runtime inputs vary the
/// depth, range and branch structure. They are test data, not compiler facts.
const INPUTS: &[(&str, &[(i64, i64)])] = &[
    ("discovered_frontier", &[(6, 7), (5, 3), (7, 11), (4, 2)]),
    ("divide_conquer", &[(256, 3), (100, 0), (513, 17), (9, 5)]),
    ("fanout_collatz", &[(27, 97), (7, 3), (63, 255), (1, 1)]),
    ("owner_fan", &[(12, 30), (0, 1), (40, 7), (3, 64)]),
    ("quad_branch", &[(5, 2), (4, 11), (6, 1), (3, 5)]),
    ("trap_in_fan", &[(40, 3), (40, 0), (5, 1), (90, 0)]),
];
pub const SLOTS: usize = Limits::CERTIFIED_SLOTS;
const QUANTUM: u64 = 16;
const ROUNDS: u64 = 64;

/// The certified limits `invar gnet-device` also uses.
pub fn limits() -> Limits {
    Limits::certified()
}

pub fn slice() -> Vec<Source> {
    let slice: Vec<_> = corpus()
        .into_iter()
        .filter(|s| std::path::Path::new(&s.name).starts_with("tests/fixtures/gnet_parallel"))
        .collect();
    assert_eq!(slice.len(), INPUTS.len(), "every slice fixture has inputs");
    slice
}

fn stem(source: &Source) -> &str {
    let file = std::path::Path::new(&source.name);
    assert_eq!(file.extension().and_then(|e| e.to_str()), Some("inv"));
    file.file_stem().and_then(|s| s.to_str()).unwrap()
}

fn values(plan: &Plan, pair: (i64, i64)) -> Vec<Value> {
    plan.artifact()
        .params()
        .iter()
        .zip([pair.0, pair.1])
        .map(|(ty, n)| match ty {
            Type::I32 => Value::I32(n as i32),
            Type::U32 => Value::U32(n as u32),
            other => panic!("slice parameter {other:?}"),
        })
        .collect()
}

pub fn inputs(source: &Source, plan: &Plan) -> Vec<Vec<Value>> {
    let (_, pairs) = INPUTS
        .iter()
        .find(|(name, _)| *name == stem(source))
        .unwrap_or_else(|| panic!("{} has no slice inputs", source.name));
    pairs.iter().map(|&pair| values(plan, pair)).collect()
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Totals {
    pub runs: usize,
    pub traps: usize,
    pub spawned: u64,
    pub committed: u64,
    pub replayed: u64,
    pub capacity_serialized: u64,
    pub owned_replays: u64,
    pub peak_width: u64,
    pub epochs: u64,
}
impl Totals {
    fn add(&mut self, f: Frontier, trapped: bool, epochs: u64) {
        self.runs += 1;
        self.traps += usize::from(trapped);
        self.spawned += f.spawned;
        self.committed += f.committed;
        self.replayed += f.replayed;
        self.capacity_serialized += f.capacity_serialized;
        self.owned_replays += f.owned_replays;
        self.peak_width = self.peak_width.max(f.peak_width);
        self.epochs += epochs;
    }
}

/// One exact device run against the oracle; returns its frontier counters.
pub fn exact<B: Backend>(
    context: &str,
    plan: &Plan,
    backend: &mut impl FnMut() -> B,
    args: &[Value],
    capacity: usize,
) -> (Frontier, bool, u64) {
    let sized = reference(
        plan,
        args,
        config(100_000, 1_000_000, QUANTUM, ROUNDS, capacity),
    );
    let cfg = config(
        100_000,
        sized.events.len().max(1),
        QUANTUM,
        ROUNDS,
        capacity,
    );
    let expected = reference(plan, args, cfg);
    let (got, epochs) =
        drive(plan, &mut backend(), args, cfg).unwrap_or_else(|e| panic!("{context}: {e:?}"));
    agree(context, plan, &got, &expected);
    (Frontier::of(&got), got.result.is_none(), epochs)
}

/// The slice at full width and as a serial control (capacity 0). Every
/// fixture must execute on the device and expose a frontier.
pub fn denominator<B: Backend>(
    target: Target,
    label: &str,
    mut backend: impl FnMut() -> B,
) -> (Totals, Totals) {
    let (mut wide, mut serial) = (Totals::default(), Totals::default());
    for source in slice() {
        let plan = Plan::from_source(&source.bytes, "run", target, limits())
            .unwrap_or_else(|e| panic!("{label} {}: refused {e:?}", source.name));
        assert!(
            !plan.artifact().fork_plan().unwrap().candidates.is_empty(),
            "{}: independent call sites",
            source.name
        );
        let mut spawned = 0;
        for args in inputs(&source, &plan) {
            let context = format!("{label} {} {args:?}", source.name);
            let (f, trapped, epochs) = exact(&context, &plan, &mut backend, &args, SLOTS);
            spawned += f.spawned;
            wide.add(f, trapped, epochs);
            let (f, trapped, epochs) =
                exact(&format!("{context} serial"), &plan, &mut backend, &args, 0);
            assert_eq!(f.spawned, 0, "{context}: serial control");
            serial.add(f, trapped, epochs);
        }
        assert!(spawned > 0, "{label} {}: frontier executed", source.name);
    }
    println!("general_net_abi5 {label} parallel slice wide={wide:?}");
    println!("general_net_abi5 {label} parallel slice serial={serial:?}");
    assert!(wide.committed > 0 && wide.peak_width >= 4, "{wide:?}");
    assert!(wide.traps > 0 && wide.traps == serial.traps);
    (wide, serial)
}

// Shared by test crates that each use a subset.
#[cfg(feature = "gnet-fault-injection")]
#[allow(unused_imports)]
pub use faults::{FAULTS, injected};

#[cfg(feature = "gnet-fault-injection")]
mod faults {
    use super::*;
    use invar::general_net::compiled::{FAULT_HOOK, Fault};

    pub const FAULTS: [Fault; 4] = [
        Fault::PoisonedStale,
        Fault::StaleCompletion,
        Fault::ContestedClaim,
        Fault::CancelRace,
    ];

    /// Every slice fixture and input with the device hook rewriting slot words,
    /// paired with the same runs without it. Both must be exact; the injected
    /// counters must show the fault reached the protocol.
    pub fn injected<B: Backend>(
        target: Target,
        label: &str,
        fault: Fault,
        mut backend: impl FnMut() -> B,
    ) -> (Totals, Totals) {
        let (mut base, mut hit) = (Totals::default(), Totals::default());
        for source in slice() {
            let plan = Plan::from_source(&source.bytes, "run", target, limits()).unwrap();
            let faulted =
                Plan::from_source_with_fault(&source.bytes, "run", target, limits(), fault)
                    .unwrap();
            assert!(!plan.artifact().source().contains(FAULT_HOOK));
            assert!(faulted.artifact().source().contains(FAULT_HOOK));
            for args in inputs(&source, &plan) {
                let context = format!("{label} {fault:?} {} {args:?}", source.name);
                let (f, trapped, epochs) = exact(
                    &format!("{context} base"),
                    &plan,
                    &mut backend,
                    &args,
                    SLOTS,
                );
                base.add(f, trapped, epochs);
                let (f, trapped, epochs) = exact(&context, &faulted, &mut backend, &args, SLOTS);
                hit.add(f, trapped, epochs);
            }
        }
        println!("general_net_abi5 {label} fault {fault:?} base={base:?}");
        println!("general_net_abi5 {label} fault {fault:?} injected={hit:?}");
        assert_eq!(base.traps, hit.traps);
        match fault {
            Fault::StaleCompletion | Fault::PoisonedStale => {
                assert!(hit.committed < base.committed && hit.replayed > base.replayed)
            }
            Fault::ContestedClaim => assert!(
                (hit.spawned, hit.committed, hit.capacity_serialized)
                    != (base.spawned, base.committed, base.capacity_serialized)
            ),
            // A cancelled running lane is free again and discovery claims it
            // anew; which children then commit depends on device timing.
            Fault::CancelRace => assert!(hit.spawned > base.spawned),
        }
        (base, hit)
    }
}
