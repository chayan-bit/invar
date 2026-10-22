//! #278 ENG-08 part A: the summary matcher has no read-size cap of 16. A scan form of any
//! dimension is matched; the family's own shape bound (d <= 4, K <= 16) stays a typed
//! `FamilyBound` refusal with the `summary_family_bound` reason, and the matcher's analysis storage
//! is held to the byte budget. Every test asserts a nonzero case count.
use invar::budget::{BudgetExceeded, ByteBudget, Resource};
use invar::summary_source::{Contract, Error, Family, matcher_footprint};

#[path = "support/eng_08.rs"]
mod eng_08;
use eng_08::{agree_on_every_cpu_engine, checked, program};

fn matrix_source(d: usize) -> String {
    let mut s =
        "fn run(coeff: read Array<u32>, seed: read Array<u32>, count: u32) -> Array<u32> { "
            .to_owned();
    for row in 0..d {
        s.push_str(&format!("var x{row} = seed[{row}u32]; "));
    }
    s.push_str("for i in 0u32..count { ");
    for row in 0..d {
        let mut dot = "0u32".to_owned();
        for col in 0..d {
            dot = format!(
                "wrapping_add({dot},wrapping_mul(coeff[i * {}u32 + {}u32],x{col}))",
                d * d,
                row * d + col
            );
        }
        s.push_str(&format!("let y{row} = {dot}; "));
    }
    for row in 0..d {
        s.push_str(&format!("x{row} = y{row}; "));
    }
    s.push_str("} return [");
    s.push_str(
        &(0..d)
            .map(|j| format!("x{j}"))
            .collect::<Vec<_>>()
            .join(","),
    );
    s.push_str("]; }");
    s
}
/// Independent model of the scan: `count` steps of `state = M_i * state` modulo 2^32.
fn model(d: usize, coeff: &[u32], seed: &[u32], count: usize) -> Vec<u32> {
    let mut x = seed.to_vec();
    for i in 0..count {
        x = (0..d)
            .map(|row| {
                (0..d).fold(0u32, |acc, col| {
                    acc.wrapping_add(coeff[i * d * d + row * d + col].wrapping_mul(x[col]))
                })
            })
            .collect();
    }
    x
}

#[test]
fn issue_eng_08_dimensions_past_the_old_matcher_cap_are_a_typed_family_bound() {
    let mut cases = 0;
    for d in [4usize, 5, 16, 17, 24] {
        let source = matrix_source(d);
        let result = Contract::from_checked(&checked(&source), "run");
        if d <= 4 {
            assert_eq!(result.unwrap().family(), Family::Matrix(d as u8), "d={d}");
        } else {
            // Past 16 this used to read as `Unrecognized` (no reason); it is now the family bound.
            let error = result.unwrap_err();
            assert_eq!(error, Error::FamilyBound, "d={d}");
            assert_eq!(error.reason(), Some("summary_family_bound"));
        }
        cases += 1;
    }
    assert_eq!(cases, 5);
}

#[test]
fn issue_eng_08_refused_forms_run_identically_on_every_cpu_engine() {
    let mut engines = 0;
    for d in [5usize, 17] {
        let count = 2;
        let coeff: Vec<u32> = (0..count * d * d)
            .map(|k| (k as u32).wrapping_mul(2_654_435_761) | 1)
            .collect();
        let seed: Vec<u32> = (0..d as u32).map(|k| k + 3).collect();
        let expected = model(d, &coeff, &seed, count);
        let source = program(
            &matrix_source(d),
            "run(a,ix,2u32)",
            &coeff,
            &seed,
            &expected,
        );
        engines += agree_on_every_cpu_engine(&source);
    }
    assert_eq!(engines, 10);
}

#[test]
fn issue_eng_08_matcher_analysis_bytes_are_budgeted() {
    let mut cases = 0;
    let long = |d: usize| {
        (0..d).fold(matrix_source(d), |s, j| {
            s.replace(&format!("x{j}"), &format!("s{}{j}", "a".repeat(2000)))
        })
    };
    for source in [matrix_source(1), matrix_source(4), long(2)] {
        let p = checked(&source);
        let names = Contract::from_checked(&p, "run").unwrap();
        let d = names.state_names().len();
        let name_bytes: usize = names.state_names().iter().map(String::len).sum();
        let need = matcher_footprint(d, name_bytes)
            .bytes(Resource::Analysis)
            .unwrap();
        assert_eq!(need, size_of_val(names.state_names()) + name_bytes);
        assert!(Contract::from_checked_within(&p, "run", &ByteBudget::new(0, 0, need)).is_ok());
        assert_eq!(
            Contract::from_checked_within(&p, "run", &ByteBudget::new(0, 0, need - 1)).unwrap_err(),
            Error::Exceeded(BudgetExceeded {
                family: "summary_matcher",
                resource: Resource::Analysis,
                needed: Some(need),
                available: need - 1,
            })
        );
        // Host and device bytes are not the matcher's: a zero host and device budget is fine.
        assert!(Contract::from_checked_within(&p, "run", &ByteBudget::new(0, 0, need)).is_ok());
        cases += 1;
    }
    assert_eq!(cases, 3);
    assert_eq!(
        matcher_footprint(usize::MAX, 0).bytes(Resource::Analysis),
        None
    );
    assert_eq!(
        matcher_footprint(1, usize::MAX).bytes(Resource::Analysis),
        None
    );
    let tight = ByteBudget::new(usize::MAX, usize::MAX, 0);
    let source = matrix_source(2);
    assert!(matches!(
        Contract::from_checked_within(&checked(&source), "run", &tight),
        Err(Error::Exceeded(e)) if e.resource == Resource::Analysis
    ));
}

#[test]
fn issue_eng_08_deleted_cap_no_longer_exists() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut scanned = 0;
    for file in [
        "summary_source.rs",
        "summary_source/matcher.rs",
        "summary_source/host.rs",
    ] {
        let text = std::fs::read_to_string(root.join(file)).unwrap();
        assert!(!text.contains("MAX_MATCHED_DIMENSION"), "{file}");
        scanned += 1;
    }
    assert_eq!(scanned, 3);
}
