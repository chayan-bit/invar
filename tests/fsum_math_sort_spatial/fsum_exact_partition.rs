//! #206: `sum_exact` and `dot_exact` admit any partition (law
//! `fsum-exact-merge-v1`). The banked bulk deposit must leave exactly the bins,
//! flags and term count of the per-term deposit, and every worker count must
//! return the bits of an oracle that does not use the accumulator: inputs that
//! cancel by construction around a short tail rounded by the BigInt reference.
use crate::fsum_cases_support::{CANONICAL_NAN, INF, NEG, Rng, adversarial_vectors};
use invar::{
    front::Span,
    fsum::{self, Exact, Kind},
    fsum_reference as reference,
    mir_eval::ReduceCall,
    numeric_contract::F32Bits,
    schedule::reductions,
};

const WORKERS: [usize; 4] = [1, 2, 7, 64];

fn per_term(a: &[u32], b: Option<&[u32]>) -> Exact {
    let mut exact = Exact::new(b.is_some());
    match b {
        Some(b) => a.iter().zip(b).for_each(|(&x, &y)| exact.add_product(x, y)),
        None => a.iter().for_each(|&x| exact.add(x)),
    }
    exact
}

fn bulk(a: &[u32], b: Option<&[u32]>, split: usize) -> Exact {
    let mut exact = Exact::new(b.is_some());
    let (head, tail) = a.split_at(split);
    for (part, range) in [(head, 0..split), (tail, split..a.len())] {
        match b {
            Some(b) => exact.add_products(part, &b[range], |word| word),
            None => exact.add_words(part, |word| word),
        }
    }
    exact
}

/// Bins must match unless a NaN or infinity decides the result: the bulk
/// deposit then leaves a bounded value in bin 254 that `finish` never reads.
fn assert_same(got: &Exact, want: &Exact, label: &str) {
    let flags = want.flags;
    if !(flags.nan || flags.pos_inf || flags.neg_inf) {
        assert_eq!(got.bins, want.bins, "{label}: bins");
    }
    assert_eq!(got.flags, want.flags, "{label}: flags");
    assert_eq!(got.terms, want.terms, "{label}: terms");
    assert_eq!(got.finish(), want.finish(), "{label}: bits");
}

#[test]
fn fsum_exact_bulk_deposit_equals_per_term_deposit() {
    let mut rng = Rng(0x0206_b0c5);
    let mut cases = adversarial_vectors();
    for round in 0..200 {
        let n = (rng.next() % 3000) as usize;
        let a = (0..n).map(|_| rng.word()).collect::<Vec<_>>();
        let b = (0..n).map(|_| rng.word()).collect::<Vec<_>>();
        cases.push((format!("random {round} n={n}"), a, b));
    }
    for (label, a, b) in cases {
        let split = if a.is_empty() {
            0
        } else {
            (rng.next() % a.len() as u64) as usize
        };
        let label = format!("{label} split={split}");
        assert_same(&bulk(&a, None, split), &per_term(&a, None), &label);
        let dot = format!("{label} dot");
        assert_same(&bulk(&a, Some(&b), split), &per_term(&a, Some(&b)), &dot);
    }
}

/// A finite word from the adversarial distribution (subnormals, zeros,
/// extreme exponents); infinities and NaNs become their payload bits.
fn finite(rng: &mut Rng) -> u32 {
    let word = rng.word();
    if word & 0x7f80_0000 == 0x7f80_0000 {
        word & 0x807f_ffff
    } else {
        word
    }
}

/// `n` terms: `m` finite terms, the tail, then the `m` negations in reverse,
/// so every split of the terms leaves partials that cancel only when merged.
fn cancelling(rng: &mut Rng, n: usize, tail: &[u32]) -> Vec<u32> {
    let m = (n - tail.len()) / 2;
    let head = (0..m).map(|_| finite(rng)).collect::<Vec<_>>();
    let mut words = head.clone();
    words.extend_from_slice(tail);
    words.extend(head.iter().rev().map(|word| word ^ NEG));
    words.resize(n, 0);
    words
}

fn call<'a>(kind: Kind, a: &'a [F32Bits], b: Option<&'a [F32Bits]>) -> ReduceCall<'a> {
    ReduceCall {
        function: "main",
        span: Span { start: 0, end: 0 },
        kind,
        a,
        b,
    }
}

fn check_workers(kind: Kind, a: &[F32Bits], b: Option<&[F32Bits]>, want: u32, label: &str) {
    for workers in WORKERS {
        let got = reductions::cpu_on(&call(kind, a, b), workers)
            .unwrap()
            .to_bits();
        assert_eq!(got, want, "{label}: {kind:?} on {workers} workers");
    }
}

fn floats(words: &[u32]) -> Vec<F32Bits> {
    words.iter().map(|&word| F32Bits::from_bits(word)).collect()
}

/// The specials override any finite total: one NaN, one infinity, both signs.
fn specials(words: &[u32], at: usize) -> [(Vec<u32>, u32); 3] {
    let with = |values: &[(usize, u32)]| {
        let mut copy = words.to_vec();
        values
            .iter()
            .for_each(|&(index, value)| copy[index] = value);
        copy
    };
    let last = words.len() - 1;
    [
        (with(&[(at, 0xffc0_0bad)]), CANONICAL_NAN),
        (with(&[(at, INF)]), INF),
        (with(&[(at, NEG | INF), (last - at, INF)]), CANONICAL_NAN),
    ]
}

#[test]
fn fsum_exact_sums_admit_any_partition_up_to_2_pow_26() {
    let mut rng = Rng(0x0206_5e3d);
    for log2 in [20u32, 23, 26] {
        let n = 1usize << log2;
        let tail = (0..64).map(|_| finite(&mut rng)).collect::<Vec<_>>();
        let words = cancelling(&mut rng, n, &tail);
        let want = reference::sum_exact(&floats(&tail)).to_bits();
        assert_eq!(fsum::sum_exact(&words), want, "n=2^{log2} serial");
        check_workers(Kind::SumExact, &floats(&words), None, want, "cancel");
        let at = (rng.next() % n as u64) as usize;
        for (index, (special, want)) in specials(&words, at).into_iter().enumerate() {
            let label = format!("n=2^{log2} special {index}");
            check_workers(Kind::SumExact, &floats(&special), None, want, &label);
        }
        // Random hard words: bit equality with the per-term accumulator.
        let random = (0..n).map(|_| finite(&mut rng)).collect::<Vec<_>>();
        let want = per_term(&random, None).finish();
        let label = format!("n=2^{log2} random");
        check_workers(Kind::SumExact, &floats(&random), None, want, &label);
    }
}

#[test]
fn fsum_exact_dots_admit_any_partition_up_to_2_pow_24() {
    let mut rng = Rng(0x0020_6d07);
    for log2 in [20u32, 24] {
        let n = 1usize << log2;
        let (ta, tb) = (
            (0..64).map(|_| finite(&mut rng)).collect::<Vec<_>>(),
            (0..64).map(|_| finite(&mut rng)).collect::<Vec<_>>(),
        );
        let a = cancelling(&mut rng, n, &ta);
        // `b` mirrors `a`'s layout without negation: products cancel in pairs.
        let m = (n - 64) / 2;
        let head = (0..m).map(|_| finite(&mut rng)).collect::<Vec<_>>();
        let mut b = head.clone();
        b.extend_from_slice(&tb);
        b.extend(head.iter().rev());
        b.resize(n, 0);
        let want = reference::dot_exact(&floats(&ta), &floats(&tb)).to_bits();
        assert_eq!(fsum::dot_exact(&a, &b), want, "n=2^{log2} serial dot");
        let fb = floats(&b);
        check_workers(Kind::DotExact, &floats(&a), Some(&fb), want, "cancel dot");
        // Unit partners, so a special `a` word is the product's own special.
        let at = (rng.next() % n as u64) as usize;
        let mut unit = b.clone();
        (unit[at], unit[n - 1 - at]) = (0x3f80_0000, 0x3f80_0000);
        let unit = floats(&unit);
        for (index, (special, want)) in specials(&a, at).into_iter().enumerate() {
            let label = format!("n=2^{log2} dot special {index}");
            check_workers(Kind::DotExact, &floats(&special), Some(&unit), want, &label);
        }
    }
}
