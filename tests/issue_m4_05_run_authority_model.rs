#![forbid(unsafe_code)]
#[path = "../src/indexed_source/run_authority_model.rs"]
mod model;

use model::{
    Completion, Cost, Family, Finish, Limits, MAX_BYTES, MAX_VISITS, MAX_WITNESSES, Phase, Refusal,
    Run,
};
use std::{
    cell::Cell,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

fn new_run() -> Run {
    Run::new(Limits {
        visits: MAX_VISITS,
        bytes: MAX_BYTES,
    })
    .unwrap()
}
fn witness(run: &mut Run, family: Family, key: u8) -> model::Witness {
    run.register(
        family,
        [key; 32],
        Cost {
            visits: 64,
            bytes: 32,
        },
    )
    .unwrap()
}
fn ready(run: &mut Run, entered: &mut model::Entered) -> model::PublicationPermit {
    let Finish::Ready(ready) = run.finish(entered, Completion::Complete).unwrap() else {
        panic!("expected completed authority")
    };
    ready
}

#[test]
fn issue_m4_05_run_authority_model_caps_and_full_success() {
    assert_eq!(MAX_VISITS, invar::indexed_source::host::MAX_VISITS);
    assert_eq!(MAX_BYTES, invar::indexed_source::host::MAX_STAGING_BYTES);
    let mut run = Run::new(Limits {
        visits: usize::MAX,
        bytes: usize::MAX,
    })
    .unwrap();
    assert_eq!(run.snapshot().visits_limit, MAX_VISITS);
    assert_eq!(run.snapshot().bytes_limit, MAX_BYTES);
    let w = witness(&mut run, Family::Gather, 1);
    let mut permit = run.select(&[&w]).unwrap();
    let reserved = run.snapshot();
    assert_eq!(reserved.entries, 0);
    assert_eq!(reserved.publications, 0);
    assert_eq!(reserved.reserved_visits, 160);
    assert_eq!(reserved.reserved_bytes, 4128);
    run.prepare_work(&permit, 8).unwrap();
    let native = Cell::new(0);
    let published = Cell::new(0);
    let mut entered = run
        .enter(&mut permit, || {
            native.set(native.get() + 1);
            Ok(())
        })
        .unwrap();
    run.work(&entered, 16).unwrap();
    let mut completed = ready(&mut run, &mut entered);
    run.validate_work(&completed, 8).unwrap();
    run.publish(&mut completed, || published.set(published.get() + 1))
        .unwrap();
    let final_state = run.snapshot();
    assert_eq!((native.get(), published.get()), (1, 1));
    assert_eq!(final_state.reserved_visits, reserved.reserved_visits);
    assert_eq!(final_state.reserved_bytes, reserved.reserved_bytes);
    assert_eq!(final_state.actual_visits, 140);
    assert_eq!((final_state.entries, final_state.publications), (1, 1));
    assert_eq!(final_state.phase, Phase::Published);
}

#[test]
fn issue_m4_05_run_authority_model_every_pair_triple_order_and_duplicate_claim() {
    let cases: &[&[usize]] = &[
        &[0, 1],
        &[1, 0],
        &[0, 2],
        &[2, 0],
        &[1, 2],
        &[2, 1],
        &[0, 1, 2],
        &[0, 2, 1],
        &[1, 0, 2],
        &[1, 2, 0],
        &[2, 0, 1],
        &[2, 1, 0],
        &[0, 0],
        &[1, 1, 1],
        &[0, 1, 2, 0],
    ];
    for indices in cases {
        let mut run = new_run();
        let ws = [
            witness(&mut run, Family::Gather, 1),
            witness(&mut run, Family::Scatter, 2),
            witness(&mut run, Family::Histogram, 3),
        ];
        let claims = indices.iter().map(|&i| &ws[i]).collect::<Vec<_>>();
        let native = Cell::new(0);
        let result = run.select(&claims);
        if let Ok(mut p) = result {
            let _ = run.enter(&mut p, || {
                native.set(native.get() + 1);
                Ok(())
            });
            panic!("ambiguous claims admitted: {indices:?}");
        }
        assert_eq!(result.unwrap_err(), Refusal::Ambiguous);
        assert_eq!(native.get(), 0);
        assert_eq!(run.snapshot().entries, 0);
        assert_eq!(run.snapshot().publications, 0);
        assert_eq!(run.snapshot().attempts, 0);
        assert_eq!(
            run.snapshot().reserved_visits,
            128 + 8
                + if indices.len() <= 3 {
                    40 * indices.len()
                } else {
                    0
                }
        );
    }
    let mut run = new_run();
    let a = witness(&mut run, Family::Gather, 1);
    let b = witness(&mut run, Family::Gather, 2);
    assert_eq!(run.select(&[&a, &b]).unwrap_err(), Refusal::Ambiguous);
    assert_eq!(run.select(&[]).unwrap_err(), Refusal::NoClaim);
    assert_eq!(run.snapshot().entries, 0);
    println!(
        "12 family pair/triple permutations, duplicate claims and bounded overlong-list refusal; entry/publication=0"
    );
}

#[test]
fn issue_m4_05_run_authority_model_repeated_failed_selection_exhausts_without_reset() {
    let mut run = Run::new(Limits {
        visits: 500,
        bytes: MAX_BYTES,
    })
    .unwrap();
    let a = witness(&mut run, Family::Gather, 1);
    let b = witness(&mut run, Family::Scatter, 2);
    let _c = witness(&mut run, Family::Histogram, 3);
    let mut before = run.snapshot();
    for _ in 0..4 {
        assert_eq!(run.select(&[&a, &b]).unwrap_err(), Refusal::Ambiguous);
        let after = run.snapshot();
        assert_eq!(after.reserved_visits, before.reserved_visits + 88);
        assert_eq!(after.actual_visits, after.reserved_visits);
        before = after;
    }
    assert_eq!(before.reserved_visits, 480);
    assert_eq!(run.select(&[&a, &b]).unwrap_err(), Refusal::Exhausted);
    let exhausted = run.snapshot();
    assert_eq!(exhausted.reserved_visits, 488);
    assert!(exhausted.closed);
    for _ in 0..4 {
        assert!(run.select(&[&a]).is_err());
        assert!(run.snapshot().reserved_visits <= 500);
    }
    assert_eq!(run.snapshot().entries, 0);
    assert_eq!(run.snapshot().publications, 0);
}

#[test]
fn issue_m4_05_run_authority_model_reservation_overflow_exhaustion_no_entry() {
    for limits in [
        Limits {
            visits: 7,
            bytes: MAX_BYTES,
        },
        Limits {
            visits: MAX_VISITS,
            bytes: 4095,
        },
    ] {
        assert_eq!(Run::new(limits).unwrap_err(), Refusal::Exhausted);
    }
    for (limits, cost, error) in [
        (
            Limits {
                visits: 159,
                bytes: MAX_BYTES,
            },
            Cost {
                visits: 64,
                bytes: 32,
            },
            Refusal::Exhausted,
        ),
        (
            Limits {
                visits: MAX_VISITS,
                bytes: 4127,
            },
            Cost {
                visits: 64,
                bytes: 32,
            },
            Refusal::Exhausted,
        ),
        (
            Limits {
                visits: MAX_VISITS,
                bytes: MAX_BYTES,
            },
            Cost {
                visits: usize::MAX,
                bytes: 0,
            },
            Refusal::Overflow,
        ),
        (
            Limits {
                visits: MAX_VISITS,
                bytes: MAX_BYTES,
            },
            Cost {
                visits: 64,
                bytes: usize::MAX,
            },
            Refusal::Overflow,
        ),
    ] {
        let mut run = Run::new(limits).unwrap();
        let w = run.register(Family::Gather, [1; 32], cost).unwrap();
        assert_eq!(run.select(&[&w]).unwrap_err(), error);
        let s = run.snapshot();
        assert_eq!(s.entries, 0);
        assert_eq!(s.publications, 0);
        assert_eq!(s.attempts, 0);
        assert_eq!(s.reserved_visits, 96);
        assert_eq!(s.reserved_bytes, 4096);
        assert!(s.closed);
    }
    for cost in [
        Cost {
            visits: usize::MAX,
            bytes: 0,
        },
        Cost {
            visits: 0,
            bytes: usize::MAX,
        },
    ] {
        let mut run = new_run();
        let before = run.snapshot();
        assert_eq!(run.charge(cost), Err(Refusal::Overflow));
        assert_eq!(run.snapshot().reserved_visits, before.reserved_visits);
        assert!(run.snapshot().closed);
    }
}

#[test]
fn issue_m4_05_run_authority_model_witness_capacity_and_funding_are_bounded() {
    let mut run = new_run();
    assert_eq!(
        run.register(
            Family::Gather,
            [99; 32],
            Cost {
                visits: 11,
                bytes: 0
            }
        )
        .unwrap_err(),
        Refusal::Unfunded
    );
    for key in 0..MAX_WITNESSES {
        let _ = witness(&mut run, Family::Gather, key as u8);
    }
    assert_eq!(
        run.register(
            Family::Gather,
            [0; 32],
            Cost {
                visits: 64,
                bytes: 32
            }
        )
        .unwrap_err(),
        Refusal::DuplicateWitness
    );
    assert_eq!(
        run.register(
            Family::Gather,
            [99; 32],
            Cost {
                visits: 64,
                bytes: 32
            }
        )
        .unwrap_err(),
        Refusal::WitnessLimit
    );
    assert_eq!(run.snapshot().entries, 0);
}

#[test]
fn issue_m4_05_run_authority_model_rejection_and_abandonment_do_not_refund() {
    let mut run = new_run();
    let w = witness(&mut run, Family::Scatter, 1);
    let mut p = run.select(&[&w]).unwrap();
    let reserved = run.snapshot();
    run.prepare_work(&p, 8).unwrap();
    run.reject(&mut p).unwrap();
    assert_eq!(run.snapshot().reserved_visits, reserved.reserved_visits);
    assert_eq!(run.snapshot().reserved_bytes, reserved.reserved_bytes);
    assert_eq!(
        run.enter(&mut p, || panic!("rejected permit replayed"))
            .unwrap_err(),
        Refusal::Consumed
    );
    let p = run.select(&[&w]).unwrap();
    let before_drop = run.snapshot();
    drop(p);
    assert_eq!(run.snapshot().phase, Phase::Abandoned);
    assert_eq!(run.snapshot().reserved_visits, before_drop.reserved_visits);
    assert_eq!(run.snapshot().reserved_bytes, before_drop.reserved_bytes);
    let _p = run.select(&[&w]).unwrap();
    assert_eq!(run.snapshot().attempts, 3);
    assert_eq!(run.snapshot().entries, 0);
    assert!(run.snapshot().reserved_visits > before_drop.reserved_visits);
}

#[test]
fn issue_m4_05_run_authority_model_repeated_stale_entry_consumes_once_no_entry() {
    let mut run = new_run();
    let native = Cell::new(0);
    let mut last = run.snapshot();
    for key in 0..4 {
        let w = witness(&mut run, Family::Gather, key);
        let mut p = run.select(&[&w]).unwrap();
        let reserved = run.snapshot();
        run.revoke(&w).unwrap();
        assert_eq!(
            run.enter(&mut p, || {
                native.set(native.get() + 1);
                Ok(())
            })
            .unwrap_err(),
            Refusal::StaleWitness
        );
        assert_eq!(
            run.enter(&mut p, || {
                native.set(native.get() + 1);
                Ok(())
            })
            .unwrap_err(),
            Refusal::Consumed
        );
        assert_eq!(run.select(&[&w]).unwrap_err(), Refusal::StaleWitness);
        let now = run.snapshot();
        assert!(now.reserved_visits >= reserved.reserved_visits);
        assert!(now.reserved_visits > last.reserved_visits);
        last = now;
    }
    assert_eq!(run.snapshot().attempts, 4);
    assert_eq!(native.get(), 0);
    assert_eq!(run.snapshot().entries, 0);
}

#[test]
fn issue_m4_05_run_authority_model_run_identity_is_not_an_address() {
    let mut first = new_run();
    let old = witness(&mut first, Family::Gather, 1);
    let mut old_permit = first.select(&[&old]).unwrap();
    let mut second = new_run();
    let new = witness(&mut second, Family::Gather, 1);
    let _new_permit = second.select(&[&new]).unwrap();
    assert!(second.snapshot().run_id > first.snapshot().run_id);
    assert_eq!(
        second
            .enter(&mut old_permit, || panic!("cross-run entry"))
            .unwrap_err(),
        Refusal::StaleRun
    );
    // Cross-run rejection did not consume the original authority or reset either ledger.
    let mut entry = first.enter(&mut old_permit, || Ok(())).unwrap();
    let _ready = ready(&mut first, &mut entry);
    drop(second);
    let mut last = first.snapshot().run_id;
    for _ in 0..32 {
        let mut next = new_run();
        assert!(next.snapshot().run_id > last);
        last = next.snapshot().run_id;
        let _matching = witness(&mut next, Family::Gather, 1);
        assert_eq!(next.select(&[&old]).unwrap_err(), Refusal::StaleRun);
        assert_eq!(next.snapshot().entries, 0);
    }
}

#[test]
fn issue_m4_05_run_authority_model_entry_and_publication_consume_exactly_once() {
    let mut run = new_run();
    let w = witness(&mut run, Family::Histogram, 1);
    let mut p = run.select(&[&w]).unwrap();
    let native = Cell::new(0);
    let publication = Cell::new(0);
    let mut entry = run
        .enter(&mut p, || {
            native.set(native.get() + 1);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        run.enter(&mut p, || {
            native.set(native.get() + 1);
            Ok(())
        })
        .unwrap_err(),
        Refusal::Consumed
    );
    let mut completed = ready(&mut run, &mut entry);
    assert_eq!(
        run.finish(&mut entry, Completion::Complete).unwrap_err(),
        Refusal::Consumed
    );
    run.publish(&mut completed, || publication.set(publication.get() + 1))
        .unwrap();
    assert_eq!(
        run.publish(&mut completed, || publication.set(publication.get() + 1)),
        Err(Refusal::Consumed)
    );
    assert_eq!((native.get(), publication.get()), (1, 1));
    assert_eq!(
        (run.snapshot().entries, run.snapshot().publications),
        (1, 1)
    );
}

#[test]
fn issue_m4_05_run_authority_model_post_entry_errors_are_terminal_not_replayable() {
    for entry_error in [true, false] {
        let mut run = new_run();
        let w = witness(&mut run, Family::Gather, 1);
        let mut p = run.select(&[&w]).unwrap();
        let reserved = run.snapshot();
        let native = Cell::new(0);
        let result = run.enter(&mut p, || {
            native.set(native.get() + 1);
            if entry_error { Err(()) } else { Ok(()) }
        });
        if entry_error {
            assert_eq!(result.unwrap_err(), Refusal::EntryFailed);
        } else {
            let mut entered = result.unwrap();
            assert!(matches!(
                run.finish(&mut entered, Completion::FailedQuiescent)
                    .unwrap(),
                Finish::Failed
            ));
        }
        assert_eq!(run.snapshot().phase, Phase::FailedQuiescent);
        assert!(run.snapshot().closed);
        assert_eq!(run.snapshot().reserved_bytes, reserved.reserved_bytes);
        assert_eq!(run.select(&[&w]).unwrap_err(), Refusal::Terminal);
        assert_eq!(native.get(), 1);
        assert_eq!(run.snapshot().entries, 1);
        assert_eq!(run.snapshot().publications, 0);
    }
}

#[test]
fn issue_m4_05_run_authority_model_every_work_prefix_preserves_future_steps() {
    for (stage, prefix) in (0..3).flat_map(|stage| (0..=53).map(move |prefix| (stage, prefix))) {
        let prepare = if stage == 0 { prefix } else { 0 };
        let work = if stage == 1 { prefix } else { 0 };
        let validate = if stage == 2 { prefix } else { 0 };
        let admitted = prefix <= 52;
        let mut run = new_run();
        let w = witness(&mut run, Family::Gather, 1);
        let mut p = run.select(&[&w]).unwrap();
        let reserve = run.snapshot();
        let preparation = run.prepare_work(&p, prepare);
        if preparation.is_err() {
            assert!(!admitted);
            assert_eq!(
                run.enter(&mut p, || panic!("unfunded entry")).unwrap_err(),
                Refusal::Terminal
            );
            assert_eq!(run.snapshot().entries, 0);
            continue;
        }
        let mut entry = run.enter(&mut p, || Ok(())).unwrap();
        let worked = run.work(&entry, work);
        if worked.is_err() {
            assert!(!admitted);
            assert!(run.finish(&mut entry, Completion::Complete).is_err());
            continue;
        }
        let mut completed = ready(&mut run, &mut entry);
        let validated = run.validate_work(&completed, validate);
        let published = Cell::new(0);
        let publication = run.publish(&mut completed, || published.set(published.get() + 1));
        assert_eq!(validated.is_ok() && publication.is_ok(), admitted);
        assert_eq!(published.get(), usize::from(admitted));
        assert_eq!(run.snapshot().reserved_visits, reserve.reserved_visits);
        assert_eq!(run.snapshot().reserved_bytes, reserve.reserved_bytes);
    }
    let mut run = new_run();
    let w = witness(&mut run, Family::Gather, 1);
    let p = run.select(&[&w]).unwrap();
    assert_eq!(run.prepare_work(&p, usize::MAX), Err(Refusal::Overflow));
    assert_eq!(run.snapshot().entries, 0);
    println!(
        "162 symbolic work prefixes: preparation, production and validation; no publication beyond prepaid allowance"
    );
}

#[test]
fn issue_m4_05_run_authority_model_budget_failure_completion_matrix() {
    for completion in [
        Completion::Complete,
        Completion::FailedQuiescent,
        Completion::Unknown,
        Completion::Pending,
    ] {
        let mut run = new_run();
        let w = witness(&mut run, Family::Gather, 1);
        let mut p = run.select(&[&w]).unwrap();
        let reservation = run.snapshot();
        let mut entry = run.enter(&mut p, || Ok(())).unwrap();
        assert_eq!(run.work(&entry, usize::MAX), Err(Refusal::Overflow));
        let finish = run.finish(&mut entry, completion);
        match completion {
            Completion::Complete => assert_eq!(finish.unwrap_err(), Refusal::Terminal),
            Completion::FailedQuiescent => assert!(matches!(finish.unwrap(), Finish::Failed)),
            Completion::Unknown | Completion::Pending => {
                assert!(matches!(finish.unwrap(), Finish::Quarantined(_)))
            }
        }
        assert!(run.snapshot().closed);
        assert_eq!(run.snapshot().publications, 0);
        assert_eq!(run.snapshot().reserved_visits, reservation.reserved_visits);
        assert_eq!(run.snapshot().reserved_bytes, reservation.reserved_bytes);
    }
}

#[test]
fn issue_m4_05_run_authority_model_unknown_pending_and_drop_retain_reservations() {
    for completion in [Completion::Unknown, Completion::Pending] {
        let mut run = new_run();
        let inspect = run.inspection();
        let w = witness(&mut run, Family::Histogram, 1);
        let mut p = run.select(&[&w]).unwrap();
        let reserved = run.snapshot();
        let mut entry = run.enter(&mut p, || Ok(())).unwrap();
        let Finish::Quarantined(q) = run.finish(&mut entry, completion).unwrap() else {
            panic!("quarantine required")
        };
        let state = q.snapshot();
        assert_eq!(state.phase, Phase::Quarantined);
        assert!(state.closed);
        assert_eq!(state.reserved_visits, reserved.reserved_visits);
        assert_eq!(state.reserved_bytes, reserved.reserved_bytes);
        assert_eq!(run.select(&[&w]).unwrap_err(), Refusal::Terminal);
        assert_eq!(run.snapshot().publications, 0);
        drop(entry);
        drop(p);
        drop(run);
        assert!(inspect.snapshot().is_some());
        drop(q);
        let retained = inspect.snapshot().unwrap();
        assert_eq!(retained.phase, Phase::Quarantined);
        assert_eq!(retained.reserved_bytes, reserved.reserved_bytes);
    }
    let mut run = new_run();
    let inspect = run.inspection();
    let w = witness(&mut run, Family::Gather, 1);
    let mut p = run.select(&[&w]).unwrap();
    let entry = run.enter(&mut p, || Ok(())).unwrap();
    let reserved = run.snapshot();
    drop(entry);
    assert_eq!(run.snapshot().phase, Phase::Quarantined);
    assert_eq!(run.select(&[&w]).unwrap_err(), Refusal::Terminal);
    drop(p);
    drop(run);
    assert_eq!(
        inspect.snapshot().unwrap().reserved_bytes,
        reserved.reserved_bytes
    );
}

#[test]
fn issue_m4_05_run_authority_model_unknown_retains_before_wrong_run_refusal() {
    let mut first = new_run();
    let inspect = first.inspection();
    let w = witness(&mut first, Family::Gather, 1);
    let mut p = first.select(&[&w]).unwrap();
    let mut entry = first.enter(&mut p, || Ok(())).unwrap();
    let mut wrong = new_run();
    assert_eq!(
        wrong.finish(&mut entry, Completion::Unknown).unwrap_err(),
        Refusal::StaleRun
    );
    assert_eq!(first.snapshot().phase, Phase::Quarantined);
    assert!(first.snapshot().closed);
    assert_eq!(wrong.snapshot().entries, 0);
    drop(entry);
    drop(p);
    drop(first);
    assert!(inspect.snapshot().is_some());
}

#[test]
fn issue_m4_05_run_authority_model_quiescent_abandoned_and_panic_terminals() {
    let mut run = new_run();
    let inspect = run.inspection();
    let w = witness(&mut run, Family::Gather, 1);
    let mut p = run.select(&[&w]).unwrap();
    let mut entry = run.enter(&mut p, || Ok(())).unwrap();
    let completed = ready(&mut run, &mut entry);
    let reserved = run.snapshot();
    drop(completed);
    assert_eq!(run.snapshot().phase, Phase::Abandoned);
    assert!(run.snapshot().closed);
    assert_eq!(run.snapshot().reserved_visits, reserved.reserved_visits);
    drop(entry);
    drop(p);
    drop(run);
    assert!(inspect.snapshot().is_none());
    for publishing in [false, true] {
        let mut run = new_run();
        let inspect = run.inspection();
        let w = witness(&mut run, Family::Gather, 1);
        let mut p = run.select(&[&w]).unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if !publishing {
                let _ = run.enter(&mut p, || panic!("simulated entry unwind"));
            } else {
                let mut entry = run.enter(&mut p, || Ok(())).unwrap();
                let mut completed = ready(&mut run, &mut entry);
                let _ = run.publish(&mut completed, || panic!("simulated publication unwind"));
            }
        }));
        assert!(panic.is_err());
        assert_eq!(run.snapshot().phase, Phase::Quarantined);
        assert!(run.snapshot().closed);
        assert_eq!(run.snapshot().publications, 0);
        drop(p);
        drop(run);
        assert!(inspect.snapshot().is_some());
    }
}

#[test]
fn issue_m4_05_run_authority_model_busy_and_revocation_cannot_replace_authority() {
    let mut run = new_run();
    let w = witness(&mut run, Family::Gather, 1);
    let mut p = run.select(&[&w]).unwrap();
    assert_eq!(run.select(&[&w]).unwrap_err(), Refusal::Busy);
    let mut entry = run.enter(&mut p, || Ok(())).unwrap();
    assert_eq!(run.revoke(&w), Err(Refusal::Busy));
    let mut completed = ready(&mut run, &mut entry);
    run.publish(&mut completed, || {}).unwrap();
    assert_eq!(run.snapshot().publications, 1);
}

static CLIENT_SEQUENCE: AtomicU64 = AtomicU64::new(1);
#[test]
fn issue_m4_05_run_authority_model_safe_clients_cannot_clone_forge_or_reuse_moved_tokens() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let module = root.join("src/indexed_source/run_authority_model.rs");
    let cases = [
        ("witness-clone", "let _=w.clone();", "E0599"),
        ("run-clone", "let _=run.clone();", "E0599"),
        (
            "permit-clone",
            "let p=run.select(&[&w]).unwrap(); let _=p.clone();",
            "E0599",
        ),
        (
            "entered-clone",
            "let mut p=run.select(&[&w]).unwrap(); let e=run.enter(&mut p,||Ok(())).unwrap(); let _=e.clone();",
            "E0599",
        ),
        (
            "publication-clone",
            "let mut p=run.select(&[&w]).unwrap(); let mut e=run.enter(&mut p,||Ok(())).unwrap(); if let model::Finish::Ready(p)=run.finish(&mut e,model::Completion::Complete).unwrap(){let _=p.clone();}",
            "E0599",
        ),
        (
            "quarantine-clone",
            "let mut p=run.select(&[&w]).unwrap(); let mut e=run.enter(&mut p,||Ok(())).unwrap(); if let model::Finish::Quarantined(q)=run.finish(&mut e,model::Completion::Unknown).unwrap(){let _=q.clone();}",
            "E0599",
        ),
        (
            "permit-forge",
            "let _=model::Permit {ledger:todo!(),authority:None};",
            "E0451",
        ),
        (
            "witness-forge",
            "let _=model::Witness {run_id:run.snapshot().run_id,slot:0,family:model::Family::Gather,key:[1;32]};",
            "E0451",
        ),
        ("run-forge", "let _=model::Run {ledger:todo!()};", "E0451"),
        (
            "quarantine-publish",
            "let mut p=run.select(&[&w]).unwrap(); let mut e=run.enter(&mut p,||Ok(())).unwrap(); if let model::Finish::Quarantined(mut q)=run.finish(&mut e,model::Completion::Unknown).unwrap(){run.publish(&mut q,||{}).unwrap();}",
            "E0308",
        ),
        (
            "moved-permit",
            "let p=run.select(&[&w]).unwrap(); drop(p); drop(p);",
            "E0382",
        ),
    ];
    let seq = CLIENT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "invar-run-model-clients-{}-{seq}",
        std::process::id()
    ));
    std::fs::create_dir(&dir).unwrap();
    for (name, code, diagnostic) in cases {
        let source = format!(
            "#![forbid(unsafe_code)]\n#[path={:?}] mod model;\nfn main() {{let mut run=model::Run::new(model::Limits {{visits:model::MAX_VISITS,bytes:model::MAX_BYTES}}).unwrap();let w=run.register(model::Family::Gather,[1;32],model::Cost {{visits:64,bytes:32}}).unwrap();{code}}}\n",
            module
        );
        let input = dir.join(format!("{name}.rs"));
        std::fs::write(&input, source).unwrap();
        let compiler = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let result = Command::new(compiler)
            .arg("--edition=2024")
            .arg("--crate-name")
            .arg("model_client")
            .arg("--emit=metadata")
            .arg("-o")
            .arg(dir.join("client.rmeta"))
            .arg(&input)
            .output()
            .unwrap();
        let stderr = String::from_utf8(result.stderr).unwrap();
        assert!(!result.status.success(), "{name} unexpectedly compiled");
        assert!(stderr.contains(diagnostic), "{name}: {stderr}");
        println!("safe-client refusal {name}: {diagnostic}\n{stderr}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}
