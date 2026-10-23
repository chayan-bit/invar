//! #57 ENG-08 part A follow-up: with the 4096-element caps gone, the language's per-array limit
//! (`array_limits::MAX_ARRAY_LEN`) is still enforced by the indexed and affine scatter families, so
//! a region the checked source would trap on (`ARRAY_SIZE_LIMIT`) is refused, not computed. Every
//! test asserts a nonzero case count.
use invar::array_limits::MAX_ARRAY_LEN;
use invar::budget::ByteBudget;
use invar::indexed_source::{self, Contract};
use invar::{check, front, scatter_proof};

const GATHER: &str = "fn gather(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out = filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i] = a[ix[i]]; } return out; }";
const SCATTER: &str = "fn scatter(a: read Array<u32>, ix: read Array<u32>, n: u32) -> Array<u32> { var out = filled_u32(n,0u32); for i in 0u32..len(ix) { out[ix[i]] = a[i]; } return out; }";

fn contract(source: &str, entry: &str) -> Contract {
    let program = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
    Contract::from_checked(source.as_bytes(), &program, entry).unwrap()
}

#[test]
fn issue_eng_08_indexed_families_refuse_past_the_language_array_limit_under_the_byte_budget() {
    let over = MAX_ARRAY_LEN + 1;
    let budget = ByteBudget::default();
    let scatter = contract(SCATTER, "scatter");
    let gather = contract(GATHER, "gather");
    let mut cases = 0;
    assert!(matches!(
        scatter.candidate_within(&[], &[], Some(over as u32), &budget),
        Err(indexed_source::Error::Budget)
    ));
    cases += 1;
    let big = vec![0u32; over];
    assert!(matches!(
        gather.candidate_within(&big, &[], None, &budget),
        Err(indexed_source::Error::Budget)
    ));
    cases += 1;
    assert!(
        gather
            .candidate_within(&big[..MAX_ARRAY_LEN], &[], None, &budget)
            .is_ok()
    );
    cases += 1;
    assert_eq!(cases, 3);
}

#[test]
fn issue_eng_08_affine_scatter_refuses_past_the_language_array_limit() {
    let over = MAX_ARRAY_LEN as u64 + 1;
    let source = format!(
        "fn scatter(a: read Array<u32>) -> Array<u32> {{ var out: Array<u32> = filled_u32({over}u32,0u32); for i in 0u32..4u32 {{ out[i] = a[i]; }} return out; }}"
    );
    let program = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
    assert!(matches!(
        scatter_proof::Contract::from_checked(source.as_bytes(), &program, "scatter"),
        Err(scatter_proof::Error::Budget)
    ));
    let at_limit = source.replace(&format!("{over}u32"), &format!("{MAX_ARRAY_LEN}u32"));
    let program = check::check(front::parse(at_limit.as_bytes()).unwrap(), false).unwrap();
    let c =
        scatter_proof::Contract::from_checked(at_limit.as_bytes(), &program, "scatter").unwrap();
    let big = vec![0u32; MAX_ARRAY_LEN + 1];
    assert!(matches!(
        c.candidate_within(&big, &ByteBudget::default()),
        Err(scatter_proof::Error::Budget)
    ));
}
