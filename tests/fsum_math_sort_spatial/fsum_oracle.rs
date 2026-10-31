//! v0.2 deterministic reductions: the fast engine code (`invar::fsum`) and the
//! reference semantics (`invar::fsum_reference`) against an oracle written
//! independently here: a top-down recursive tree and an exact big-rational sum
//! rounded by binary search over binary32 encodings.

use super::fsum_cases_support as cases;

use cases::{CANONICAL_NAN, Rng, adversarial_vectors, bits};
use invar::{
    front::Span,
    fsum::{self, Kind},
    fsum_reference as reference,
    mir_eval::ReduceCall,
    numeric_contract::F32Bits,
    schedule::reductions,
};
use num_bigint::BigInt;
use num_rational::BigRational;

/// Top-down aligned tree: the root over `m` leaves splits at `2^(H-1)`.
fn oracle_tree(terms: &[f32]) -> u32 {
    fn leaf(block: &[f32]) -> f32 {
        block[1..].iter().fold(block[0], |acc, term| acc + term)
    }
    fn node(leaves: &[f32]) -> f32 {
        if leaves.len() == 1 {
            return leaves[0];
        }
        let half = leaves.len().next_power_of_two() / 2;
        node(&leaves[..half]) + node(&leaves[half..])
    }
    let leaves = terms.chunks(8).map(leaf).collect::<Vec<_>>();
    if leaves.is_empty() {
        return 0;
    }
    let root = node(&leaves);
    if root.is_nan() {
        CANONICAL_NAN
    } else {
        root.to_bits()
    }
}

fn exact_value(word: u32) -> BigRational {
    let exponent = ((word >> 23) & 0xff) as i64;
    let fraction = i64::from(word & 0x7f_ffff);
    let (significand, power) = if exponent == 0 {
        (fraction, -149)
    } else {
        (fraction | 1 << 23, exponent - 150)
    };
    let value = if power >= 0 {
        BigRational::from_integer(BigInt::from(significand) << power as usize)
    } else {
        BigRational::new(
            BigInt::from(significand),
            BigInt::from(1) << (-power) as usize,
        )
    };
    if word >> 31 == 1 { -value } else { value }
}

/// Nearest-even binary32 by binary search over positive encodings.
fn oracle_round(value: &BigRational, negative_zero: bool) -> u32 {
    let origin = BigRational::from_integer(BigInt::from(0));
    if *value == origin {
        return if negative_zero { 0x8000_0000 } else { 0 };
    }
    let negative = *value < origin;
    let sign = if negative { 0x8000_0000 } else { 0 };
    let magnitude = if negative {
        -value.clone()
    } else {
        value.clone()
    };
    let (mut low, mut high) = (0u32, 0x7f80_0000u32);
    // Invariant: value(low) <= magnitude < value(high), with 0x7f800000 = 2^128.
    let weight = |word: u32| {
        if word == 0x7f80_0000 {
            exact_value(0x7f00_0000) * BigInt::from(2)
        } else {
            exact_value(word)
        }
    };
    if magnitude >= weight(high) {
        return sign | 0x7f80_0000;
    }
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if weight(middle) <= magnitude {
            low = middle;
        } else {
            high = middle;
        }
    }
    let midpoint = (weight(low) + weight(high)) / BigInt::from(2);
    let chosen = match magnitude.cmp(&midpoint) {
        std::cmp::Ordering::Less => low,
        std::cmp::Ordering::Greater => high,
        std::cmp::Ordering::Equal if low & 1 == 0 => low,
        std::cmp::Ordering::Equal => high,
    };
    sign | chosen
}

fn nan(word: u32) -> bool {
    word & 0x7fff_ffff > 0x7f80_0000
}

fn inf(word: u32) -> Option<bool> {
    (word & 0x7fff_ffff == 0x7f80_0000).then_some(word >> 31 == 1)
}

fn zero(word: u32) -> bool {
    word & 0x7fff_ffff == 0
}

fn oracle_specials(terms: &[(u32, u32)], dot: bool) -> Option<u32> {
    let mut seen = (false, false);
    for &(a, b) in terms {
        let invalid = dot && ((inf(a).is_some() && zero(b)) || (zero(a) && inf(b).is_some()));
        if nan(a) || (dot && nan(b)) || invalid {
            return Some(CANONICAL_NAN);
        }
        let negative = if dot {
            (a ^ b) >> 31 == 1
        } else {
            a >> 31 == 1
        };
        if inf(a).is_some() || (dot && inf(b).is_some()) {
            if negative {
                seen.1 = true
            } else {
                seen.0 = true
            }
        }
    }
    match seen {
        (true, true) => Some(CANONICAL_NAN),
        (true, false) => Some(0x7f80_0000),
        (false, true) => Some(0xff80_0000),
        _ => None,
    }
}

fn oracle_exact(a: &[u32], b: Option<&[u32]>) -> u32 {
    let pairs = match b {
        Some(b) => a.iter().copied().zip(b.iter().copied()).collect::<Vec<_>>(),
        None => a.iter().map(|&x| (x, 0x3f80_0000)).collect(),
    };
    if pairs.is_empty() {
        return 0;
    }
    if let Some(special) = oracle_specials(&pairs, b.is_some()) {
        return special;
    }
    let negative_zero = pairs
        .iter()
        .all(|&(x, y)| (zero(x) || zero(y)) && (x ^ y) >> 31 == 1);
    let total = pairs
        .iter()
        .map(|&(x, y)| exact_value(x) * exact_value(y))
        .fold(BigRational::from_integer(BigInt::from(0)), |acc, term| {
            acc + term
        });
    oracle_round(&total, negative_zero)
}

fn as_f32(words: &[u32]) -> Vec<f32> {
    words.iter().map(|&w| f32::from_bits(w)).collect()
}

fn as_bits(words: &[u32]) -> Vec<F32Bits> {
    words.iter().map(|&w| F32Bits::from_bits(w)).collect()
}

fn products(a: &[u32], b: &[u32]) -> Vec<f32> {
    as_f32(a)
        .iter()
        .zip(as_f32(b))
        .map(|(x, y)| x * y)
        .collect()
}

fn check_vector(a: &[u32], b: &[u32], label: &str) {
    let (ra, rb) = (as_bits(a), as_bits(b));
    let tree = oracle_tree(&as_f32(a));
    assert_eq!(fsum::sum_tree(a), tree, "{label}: fast sum_tree");
    assert_eq!(
        reference::sum_tree(&ra).to_bits(),
        tree,
        "{label}: reference sum_tree"
    );
    let dot = oracle_tree(&products(a, b));
    assert_eq!(fsum::dot_tree(a, b), dot, "{label}: fast dot_tree");
    assert_eq!(
        reference::dot_tree(&ra, &rb).to_bits(),
        dot,
        "{label}: reference dot_tree"
    );
    let exact = oracle_exact(a, None);
    assert_eq!(fsum::sum_exact(a), exact, "{label}: fast sum_exact");
    assert_eq!(
        reference::sum_exact(&ra).to_bits(),
        exact,
        "{label}: reference sum_exact"
    );
    let exact_dot = oracle_exact(a, Some(b));
    assert_eq!(fsum::dot_exact(a, b), exact_dot, "{label}: fast dot_exact");
    assert_eq!(
        reference::dot_exact(&ra, &rb).to_bits(),
        exact_dot,
        "{label}: reference dot_exact"
    );
    for (kind, want) in [
        (Kind::SumTree, tree),
        (Kind::DotTree, dot),
        (Kind::SumExact, exact),
        (Kind::DotExact, exact_dot),
    ] {
        check_workers(kind, &ra, &rb, want, label);
    }
}

/// Every forced CPU worker count returns the oracle bits, whatever the cost
/// model would pick for this size.
fn check_workers(kind: Kind, a: &[F32Bits], b: &[F32Bits], want: u32, label: &str) {
    let call = ReduceCall {
        function: "main",
        span: Span { start: 0, end: 0 },
        kind,
        a,
        b: matches!(kind, Kind::DotTree | Kind::DotExact).then_some(b),
    };
    for workers in [2, 3, 8] {
        let got = reductions::cpu_on(&call, workers).unwrap().to_bits();
        assert_eq!(got, want, "{label}: {kind:?} on {workers} workers");
    }
}

#[test]
fn fsum_adversarial_vectors_match_independent_oracle() {
    for (label, a, b) in adversarial_vectors() {
        check_vector(&a, &b, &label);
    }
}

#[test]
fn fsum_random_lengths_match_independent_oracle() {
    let mut rng = Rng(0x005e_edf5_u64);
    for round in 0..300 {
        let n = (rng.next() % 140) as usize;
        let a = (0..n).map(|_| rng.word()).collect::<Vec<_>>();
        let b = (0..n).map(|_| rng.word()).collect::<Vec<_>>();
        check_vector(&a, &b, &format!("random round {round} n={n}"));
    }
}

#[test]
fn fsum_exact_results_are_order_independent() {
    let mut rng = Rng(77);
    for _ in 0..40 {
        let a = (0..200).map(|_| rng.word()).collect::<Vec<_>>();
        let b = (0..200).map(|_| rng.word()).collect::<Vec<_>>();
        let mut order = (0..200).collect::<Vec<_>>();
        rng.shuffle(&mut order);
        let pa = order.iter().map(|&i| a[i]).collect::<Vec<_>>();
        let pb = order.iter().map(|&i| b[i]).collect::<Vec<_>>();
        assert_eq!(fsum::sum_exact(&pa), fsum::sum_exact(&a));
        assert_eq!(fsum::dot_exact(&pa, &pb), fsum::dot_exact(&a, &b));
    }
}

/// Every aligned chunk level gives the serial bits: the parallel contract.
#[test]
fn fsum_tree_chunking_is_bit_identical_at_every_level() {
    let mut rng = Rng(3);
    for n in [1usize, 7, 8, 9, 63, 64, 65, 1000, 4097, 33_333] {
        let a = (0..n).map(|_| rng.word() & 0xbfff_ffff).collect::<Vec<_>>();
        let serial = fsum::sum_tree(&a);
        let term = |i: usize| f32::from_bits(a[i]);
        let leaves = fsum::leaf_count(n);
        for level in 0..=usize::BITS - leaves.leading_zeros() {
            let chunks = leaves.div_ceil(1 << level);
            let nodes = (0..chunks)
                .map(|j| fsum::tree_node(n, level, j, &term))
                .collect::<Vec<_>>();
            let root = fsum::canonical(fsum::fold_aligned(chunks, |j| nodes[j]));
            assert_eq!(root, serial, "n={n} level={level}");
        }
        for parts in 1..=6 {
            let level = fsum::chunk_level(leaves, parts);
            assert!(leaves.div_ceil(1 << level) <= parts, "n={n} parts={parts}");
        }
    }
}

#[test]
fn fsum_documented_special_values() {
    let one = bits(1.0);
    let inf = 0x7f80_0000;
    let ninf = 0xff80_0000;
    let nan_payload = 0x7f80_0001;
    for kind in ["sum_tree", "sum_exact"] {
        let run = |xs: &[u32]| match kind {
            "sum_tree" => fsum::sum_tree(xs),
            _ => fsum::sum_exact(xs),
        };
        assert_eq!(run(&[]), 0, "{kind}: empty is +0");
        assert_eq!(run(&[0x8000_0000]), 0x8000_0000, "{kind}: single -0");
        assert_eq!(run(&[0x8000_0000; 20]), 0x8000_0000, "{kind}: all -0");
        assert_eq!(run(&[0x8000_0000, 0]), 0, "{kind}: mixed zeros");
        assert_eq!(
            run(&[one, bits(-1.0)]),
            0,
            "{kind}: exact cancellation is +0"
        );
        assert_eq!(run(&[one, nan_payload, one]), CANONICAL_NAN, "{kind}: NaN");
        assert_eq!(
            run(&[inf, one, ninf]),
            CANONICAL_NAN,
            "{kind}: +inf with -inf"
        );
        assert_eq!(run(&[one, ninf, one]), ninf, "{kind}: one infinity");
    }
    let max = 0x7f7f_ffff;
    assert_eq!(
        fsum::sum_exact(&[max, max]),
        inf,
        "exact overflow rounds to inf"
    );
    assert_eq!(
        fsum::sum_exact(&[max, max, 0xff7f_ffff]),
        max,
        "no spurious overflow"
    );
    assert_eq!(
        fsum::sum_tree(&[max, max, 0xff7f_ffff]),
        inf,
        "tree overflow is IEEE"
    );
    assert_eq!(fsum::dot_exact(&[inf], &[0]), CANONICAL_NAN, "inf * 0");
    assert_eq!(
        fsum::dot_exact(&[inf, one], &[0x8000_0000 | one, one]),
        ninf
    );
    assert_eq!(
        fsum::dot_exact(&[0x8000_0000], &[one]),
        0x8000_0000,
        "-0 product"
    );
    assert_eq!(
        fsum::dot_exact(&[1], &[1]),
        0,
        "product below half the least subnormal"
    );
    assert_eq!(
        fsum::dot_exact(&[1], &[0x8000_0000 | 0x3eff_ffff]),
        0x8000_0000
    );
}
