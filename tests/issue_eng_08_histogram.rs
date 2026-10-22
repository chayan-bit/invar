//! #57 ENG-08 part A: the weighted histogram admits by bytes, not by 4096 elements or 128
//! contributions. Every test asserts a nonzero case count. Device rows above the old caps are
//! pending until run on hardware.
use invar::budget::{BudgetExceeded, ByteBudget, Resource};
use invar::histogram_source::{self, Budget, Contract, Cost, Error};

#[path = "support/eng_08.rs"]
mod eng_08;
use eng_08::{agree_on_every_cpu_engine, allocated_during, checked, permutation, program};

const HIST: &str = "fn hist(a: read Array<u32>, ix: read Array<u32>, n: u32) -> Array<u32> { var out = filled_u32(n,0u32); for i in 0u32..len(ix) { out[ix[i]] = wrapping_add(out[ix[i]],a[i]); } return out; }";

fn contract() -> Contract {
    Contract::from_checked(HIST.as_bytes(), &checked(HIST), "hist").unwrap()
}
fn reference(values: &[u32], indices: &[u32], bins: usize) -> Vec<u32> {
    let mut out = vec![0u32; bins];
    for (i, &bin) in indices.iter().enumerate() {
        out[bin as usize] = out[bin as usize].wrapping_add(values[i]);
    }
    out
}

#[test]
fn issue_eng_08_histogram_past_the_old_caps_is_admitted_and_exact() {
    let c = contract();
    let mut cases = 0;
    // 4097 elements and bins (old cap 4096) and 129+ contributions (old cap 128), spread and piled.
    for (n, bins) in [(4097u32, 4097usize), (6000, 100), (10_007, 10_007)] {
        let values: Vec<u32> = (0..n).map(|i| i.wrapping_mul(2_654_435_761)).collect();
        for indices in [
            permutation(n)
                .into_iter()
                .map(|i| i % bins as u32)
                .collect::<Vec<_>>(),
            vec![0; n as usize],
        ] {
            let expected = reference(&values, &indices, bins);
            let candidate = c.candidate(&values, &indices, bins as u32).unwrap();
            assert_eq!(candidate.output(), expected, "n={n} bins={bins}");
            assert_eq!(candidate.source_accesses().len(), n as usize);
            assert_eq!(candidate.grouped_accesses().len(), n as usize);
            cases += 1;
        }
    }
    assert_eq!(cases, 6);
}

#[test]
fn issue_eng_08_histogram_matches_every_cpu_engine_past_the_old_caps() {
    let mut engines = 0;
    for (n, bins) in [(4097u32, 97usize), (5000, 4500)] {
        let values: Vec<u32> = (0..n).map(|i| i ^ 0xa5a5).collect();
        let indices: Vec<u32> = permutation(n)
            .into_iter()
            .map(|i| i % bins as u32)
            .collect();
        let expected = reference(&values, &indices, bins);
        let call = format!("hist(a,ix,{bins}u32)");
        let source = program(HIST, &call, &values, &indices, &expected);
        engines += agree_on_every_cpu_engine(&source);
    }
    assert_eq!(engines, 10);
}

#[test]
fn issue_eng_08_one_byte_below_the_footprint_is_refused_and_at_it_is_admitted() {
    let c = contract();
    let mut cases = 0;
    for (a, i, n) in [
        (1usize, 1usize, 1usize),
        (4097, 4097, 4097),
        (200, 5000, 300),
    ] {
        let values = vec![1u32; a];
        let indices: Vec<u32> = (0..i).map(|k| (k % n) as u32).collect();
        // Every iteration reads values[iteration], so keep the values long enough.
        let values = if a < i { vec![1u32; i] } else { values };
        let a = values.len();
        let need = histogram_source::footprint(a, i, n)
            .bytes(Resource::Host)
            .unwrap();
        assert_eq!(
            need,
            4 * (a + i + n) + 2 * i * size_of::<histogram_source::Access>()
        );
        assert!(
            c.candidate_within(&values, &indices, n as u32, &ByteBudget::new(need, 0, 0))
                .is_ok()
        );
        assert_eq!(
            c.candidate_within(
                &values,
                &indices,
                n as u32,
                &ByteBudget::new(need - 1, 0, 0)
            )
            .unwrap_err(),
            Error::Exceeded(BudgetExceeded {
                family: "histogram",
                resource: Resource::Host,
                needed: Some(need),
                available: need - 1,
            })
        );
        cases += 1;
    }
    assert_eq!(cases, 3);
}

#[test]
fn issue_eng_08_footprint_matches_measured_allocation() {
    let c = contract();
    let (fixed, _) = allocated_during(|| c.candidate(&[], &[], 0).unwrap());
    assert!(fixed > 0);
    let mut cases = 0;
    for (n, bins) in [(1usize, 1usize), (17, 5), (4097, 300), (9000, 9000)] {
        let values: Vec<u32> = (0..n as u32).collect();
        let indices: Vec<u32> = (0..n as u32).map(|k| k % bins as u32).collect();
        let (total, _kept) =
            allocated_during(|| c.candidate(&values, &indices, bins as u32).unwrap());
        let footprint = histogram_source::footprint(n, n, bins)
            .bytes(Resource::Host)
            .unwrap();
        // The footprint also counts the caller's resident values and indices.
        assert_eq!(total - fixed, footprint - 4 * 2 * n, "n={n} bins={bins}");
        cases += 1;
    }
    assert_eq!(cases, 4);
}

#[test]
fn issue_eng_08_overflow_and_zero_budget_refuse_without_panicking() {
    let c = contract();
    assert_eq!(
        histogram_source::footprint(usize::MAX, 1, 1).bytes(Resource::Host),
        None
    );
    assert_eq!(
        histogram_source::footprint(0, usize::MAX, 0).bytes(Resource::Host),
        None
    );
    let err = c
        .candidate_within(&[], &[], u32::MAX, &ByteBudget::default())
        .unwrap_err();
    assert!(matches!(err, Error::Exceeded(e) if e.resource == Resource::Host));
    assert!(matches!(
        c.candidate_within(&[1], &[0], 1, &ByteBudget::ZERO),
        Err(Error::Exceeded(_))
    ));
    // A region that needs no bytes is not refused by a zero budget.
    assert!(c.candidate_within(&[], &[], 0, &ByteBudget::ZERO).is_ok());
}

#[test]
fn issue_eng_08_bounded_cost_has_no_element_or_contribution_cap() {
    // Past the old caps only the staging bytes, the visit budget and overflow refuse.
    let mut cases = 0;
    for (a, i, n, admitted) in [
        (4097, 0, 0, true),
        (0, 129, 1, true),
        (0, 0, 4097, true),
        (50_000, 0, 0, false),
        (0, 2000, 1, false),
        (usize::MAX, 0, 0, false),
    ] {
        assert_eq!(
            Cost::bounded(a, i, n, Budget::default()).is_ok(),
            admitted,
            "{a} {i} {n}"
        );
        cases += 1;
    }
    assert_eq!(cases, 6);
}

#[test]
fn issue_eng_08_deleted_caps_no_longer_exist_in_the_histogram_modules() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut scanned = 0;
    for file in [
        "histogram_source.rs",
        "histogram_source/bounded.rs",
        "histogram_source/host.rs",
    ] {
        let text = std::fs::read_to_string(root.join(file)).unwrap();
        for deleted in ["MAX_ELEMENTS", "MAX_CONTRIBUTIONS"] {
            assert!(!text.contains(deleted), "{file} {deleted}");
        }
        scanned += 1;
    }
    assert_eq!(scanned, 3);
}

#[test]
fn issue_eng_08_language_array_limit_still_refuses_below_the_byte_budget() {
    let c = contract();
    let over = invar::array_limits::MAX_ARRAY_LEN as u32 + 1;
    let (bytes, refused) = allocated_during(|| c.candidate(&[], &[], over));
    assert!(matches!(refused, Err(Error::Budget)), "{refused:?}");
    assert!(bytes < 1 << 20, "refusal allocated {bytes} bytes");
    let big_values = vec![0u32; invar::array_limits::MAX_ARRAY_LEN + 1];
    assert!(matches!(
        c.candidate(&big_values, &[], 1),
        Err(Error::Budget)
    ));
    // The byte budget alone would admit the oversized bins; only the language limit refuses it.
    let footprint = histogram_source::footprint(0, 0, over as usize);
    assert!(ByteBudget::default().admit("histogram", &footprint).is_ok());
}
