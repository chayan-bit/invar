//! #57 ENG-08 part A: gather, injective scatter and affine scatter admit by bytes, not by a
//! 4096-element cap. Every test asserts a nonzero case count. `native-serial` never consults these
//! source modules, and device rows above the old caps are pending until run on hardware.
use invar::budget::{BudgetExceeded, ByteBudget, Resource};
use invar::indexed_source::{self, Contract, Error, Family};
use invar::scatter_proof;

use crate::eng_08;
use eng_08::{agree_on_every_cpu_engine, allocated_during, checked, permutation, program};

const GATHER: &str = "fn gather(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out = filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i] = a[ix[i]]; } return out; }";
const SCATTER: &str = "fn scatter(a: read Array<u32>, ix: read Array<u32>, n: u32) -> Array<u32> { var out = filled_u32(n,0u32); for i in 0u32..len(ix) { out[ix[i]] = a[i]; } return out; }";

fn contract(s: &str, entry: &str) -> Contract {
    Contract::from_checked(s.as_bytes(), &checked(s), entry).unwrap()
}

fn gather_reference(values: &[u32], indices: &[u32]) -> Vec<u32> {
    indices.iter().map(|&i| values[i as usize]).collect()
}

fn scatter_reference(values: &[u32], indices: &[u32], n: u32) -> Vec<u32> {
    let mut out = vec![0; n as usize];
    for (i, &d) in indices.iter().enumerate() {
        out[d as usize] = values[i];
    }
    out
}

#[test]
fn issue_eng_08_gather_past_the_old_cap_is_admitted_and_exact() {
    let mut cases = 0;
    let gather = contract(GATHER, "gather");
    for n in [4097u32, 6000, 10_007] {
        let values: Vec<u32> = (0..n).map(|i| 17 * i + 3).collect();
        let indices = permutation(n);
        let expected = gather_reference(&values, &indices);
        let candidate = gather.candidate(&values, &indices, None).unwrap();
        assert_eq!(candidate.output(), expected, "n={n}");
        assert_eq!(candidate.accesses().len(), n as usize);
        if n <= 6000 {
            let source = program(GATHER, "gather(a,ix)", &values, &indices, &expected);
            assert_eq!(agree_on_every_cpu_engine(&source), 5);
        }
        cases += 1;
    }
    assert_eq!(cases, 3);
}

#[test]
fn issue_eng_08_injective_scatter_past_the_old_cap_is_admitted_and_exact() {
    let mut cases = 0;
    let scatter = contract(SCATTER, "scatter");
    assert_eq!(scatter.family(), Family::InjectiveScatter);
    for n in [4097u32, 6000, 10_007] {
        let values: Vec<u32> = (0..n).map(|i| 31 * i + 5).collect();
        let indices = permutation(n);
        let expected = scatter_reference(&values, &indices, n);
        let candidate = scatter.candidate(&values, &indices, Some(n)).unwrap();
        assert_eq!(candidate.output(), expected, "n={n}");
        if n <= 6000 {
            let call = format!("scatter(a,ix,{n}u32)");
            let source = program(SCATTER, &call, &values, &indices, &expected);
            assert_eq!(agree_on_every_cpu_engine(&source), 5);
        }
        cases += 1;
    }
    assert_eq!(cases, 3);
}

#[test]
fn issue_eng_08_affine_scatter_past_the_old_cap_is_certified_and_exact() {
    let mut cases = 0;
    for n in [4097u32, 6000, 20_000] {
        let source = format!(
            "fn scatter(a: read Array<u32>) -> Array<u32> {{ var out: Array<u32> = filled_u32({n}u32,0u32); for i in 0u32..{n}u32 {{ out[i] = a[i]; }} return out; }}"
        );
        let p = checked(&source);
        let c = scatter_proof::Contract::from_checked(source.as_bytes(), &p, "scatter").unwrap();
        assert_eq!(c.certificate().iterations(), n);
        let input: Vec<u32> = (0..n).map(|i| i * 3 + 1).collect();
        assert_eq!(c.candidate(&input).unwrap(), input, "n={n}");
        cases += 1;
    }
    assert_eq!(cases, 3);
}

#[test]
fn issue_eng_08_one_byte_below_the_footprint_is_refused_and_at_it_is_admitted() {
    let gather = contract(GATHER, "gather");
    let scatter = contract(SCATTER, "scatter");
    let mut cases = 0;
    for (a, ix, out) in [
        (1usize, 1usize, 1usize),
        (4097, 4097, 4097),
        (100, 5000, 5000),
    ] {
        let values = vec![0u32; a];
        let indices = vec![0u32; ix];
        let need = indexed_source::footprint(Family::Gather, a, ix, out)
            .bytes(Resource::Host)
            .unwrap();
        assert_eq!(need, 4 * (a + ix + out) + 24 * ix);
        assert!(
            gather
                .candidate_within(&values, &indices, None, &ByteBudget::new(need, 0, 0))
                .is_ok()
        );
        assert_eq!(
            gather
                .candidate_within(&values, &indices, None, &ByteBudget::new(need - 1, 0, 0))
                .unwrap_err(),
            Error::Exceeded(BudgetExceeded {
                family: "gather",
                resource: Resource::Host,
                needed: Some(need),
                available: need - 1,
            })
        );
        cases += 1;
    }
    for (a, ix, n) in [(2usize, 2usize, 3usize), (4097, 4097, 5000)] {
        let values = vec![9u32; a];
        let indices: Vec<u32> = (0..ix as u32).collect();
        let need = indexed_source::footprint(Family::InjectiveScatter, a, ix, n)
            .bytes(Resource::Host)
            .unwrap();
        assert_eq!(need, 4 * (a + ix + n) + 24 * ix + n);
        let ok = scatter.candidate_within(
            &values,
            &indices,
            Some(n as u32),
            &ByteBudget::new(need, 0, 0),
        );
        assert!(ok.is_ok());
        let err = scatter
            .candidate_within(
                &values,
                &indices,
                Some(n as u32),
                &ByteBudget::new(need - 1, 0, 0),
            )
            .unwrap_err();
        assert!(
            matches!(&err, Error::Exceeded(e) if e.to_string().starts_with("BUDGET_EXCEEDED {") && e.family == "injective_scatter")
        );
        cases += 1;
    }
    assert_eq!(cases, 5);
}

#[test]
fn issue_eng_08_refusal_keeps_the_source_path_with_identical_results() {
    // A refused region runs the retained source path, so the program's result is unchanged.
    let n = 5000u32;
    let values: Vec<u32> = (0..n).map(|i| i ^ 0x55).collect();
    let indices = permutation(n);
    let expected = gather_reference(&values, &indices);
    let gather = contract(GATHER, "gather");
    let err = gather
        .candidate_within(&values, &indices, None, &ByteBudget::ZERO)
        .unwrap_err();
    assert!(matches!(err, Error::Exceeded(_)));
    let source = program(GATHER, "gather(a,ix)", &values, &indices, &expected);
    assert_eq!(agree_on_every_cpu_engine(&source), 5);
}

#[test]
fn issue_eng_08_footprint_overflow_is_a_refusal_not_a_panic() {
    let gather = contract(GATHER, "gather");
    let scatter = contract(SCATTER, "scatter");
    // The largest u32 scatter length needs about 20 GB of host bytes: refused, never allocated.
    let err = scatter
        .candidate_within(&[], &[], Some(u32::MAX), &ByteBudget::default())
        .unwrap_err();
    assert!(matches!(err, Error::Exceeded(e) if e.resource == Resource::Host));
    let huge =
        indexed_source::footprint(Family::InjectiveScatter, usize::MAX, usize::MAX, usize::MAX);
    assert_eq!(huge.bytes(Resource::Host), None);
    assert!(
        ByteBudget::new(usize::MAX, 0, 0)
            .admit("gather", &huge)
            .is_err_and(|e| e.needed.is_none())
    );
    assert!(
        gather
            .candidate_within(&[], &[], None, &ByteBudget::ZERO)
            .is_ok()
    );
    let overflowed = scatter_proof::footprint(usize::MAX, usize::MAX);
    assert_eq!(overflowed.bytes(Resource::Host), None);
}

#[test]
fn issue_eng_08_footprints_match_measured_candidate_allocations() {
    let gather = contract(GATHER, "gather");
    let scatter = contract(SCATTER, "scatter");
    // `candidate` re-verifies the contract's MIR first; that fixed cost is the empty-input total.
    let (gather_fixed, _) = allocated_during(|| gather.candidate(&[], &[], None).unwrap());
    let (scatter_fixed, _) = allocated_during(|| scatter.candidate(&[], &[], Some(0)).unwrap());
    assert!(gather_fixed > 0 && scatter_fixed > 0);
    let mut cases = 0;
    for n in [1usize, 17, 4097, 9000] {
        let values: Vec<u32> = (0..n as u32).collect();
        let indices: Vec<u32> = (0..n as u32).rev().collect();
        // The footprint also counts the caller's resident inputs, which `candidate` borrows.
        let owned = |f: invar::budget::Footprint| f.bytes(Resource::Host).unwrap() - 4 * (2 * n);
        let (total, _kept) =
            allocated_during(|| gather.candidate(&values, &indices, None).unwrap());
        assert_eq!(
            total - gather_fixed,
            owned(indexed_source::footprint(Family::Gather, n, n, n)),
            "gather n={n}"
        );
        let (total, _kept) = allocated_during(|| {
            scatter
                .candidate(&values, &indices, Some(n as u32))
                .unwrap()
        });
        assert_eq!(
            total - scatter_fixed,
            owned(indexed_source::footprint(Family::InjectiveScatter, n, n, n)),
            "scatter n={n}"
        );
        cases += 2;
    }
    assert_eq!(cases, 8);
}

#[test]
fn issue_eng_08_affine_scatter_footprint_matches_measured_allocation() {
    // The output length is part of the source, so each size is its own contract. Re-verifying a
    // contract costs the same for literals of equal width, so within a width group the allocation
    // difference between two sizes is exactly the footprint difference.
    let mut cases = 0;
    for group in [[1000u32, 4097, 9999], [10_000, 12_000, 99_999]] {
        let mut totals = Vec::new();
        for n in group {
            let source = format!(
                "fn scatter(a: read Array<u32>) -> Array<u32> {{ var out: Array<u32> = filled_u32({n}u32,0u32); for i in 0u32..{n}u32 {{ out[i] = a[i]; }} return out; }}"
            );
            let p = checked(&source);
            let c =
                scatter_proof::Contract::from_checked(source.as_bytes(), &p, "scatter").unwrap();
            let input = vec![1u32; n as usize];
            let (total, _kept) = allocated_during(|| c.candidate(&input).unwrap());
            let fp = scatter_proof::footprint(input.len(), n as usize);
            let owned = fp.bytes(Resource::Host).unwrap() - 4 * n as usize;
            totals.push((total as i64, owned as i64));
            cases += 1;
        }
        let fixed: Vec<i64> = totals.iter().map(|(t, o)| t - o).collect();
        assert!(
            fixed[0] > 0 && fixed.windows(2).all(|w| w[0] == w[1]),
            "{totals:?}"
        );
    }
    assert_eq!(cases, 6);
}

#[test]
fn issue_eng_08_affine_device_bind_expands_past_the_old_cap_and_refuses_short_values() {
    let n = 20_000u32;
    let source = format!(
        "fn scatter(a: read Array<u32>) -> Array<u32> {{ var out: Array<u32> = filled_u32({n}u32,0u32); for i in 0u32..{n}u32 {{ out[i] = a[i]; }} return out; }}"
    );
    let plan = invar::indexed_device::Plan::from_source(source.as_bytes(), "scatter").unwrap();
    let caps = invar_device::Capabilities::cuda_12_9_sm89();
    let values: Vec<u32> = (0..n).collect();
    assert!(plan.bind(&values, &[], None, caps).is_ok());
    let short = plan.bind(&values[..10], &[], None, caps);
    assert!(
        matches!(
            short,
            Err(invar::indexed_device::Error::Device(
                invar_device::indexed::Error::Index { .. }
            ))
        ),
        "{short:?}"
    );
}
