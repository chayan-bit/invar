#![forbid(unsafe_code)]
//! ENG-02 (#59): the in-repo decision procedure for law formulas. Ring
//! identities are proved by exact polynomial normal forms over `Z` or
//! `Z/2^32`, and `max`/`min`/`ite` over `Z` by case splits refuted with
//! Fourier-Motzkin; anything else is unproved. Where `z3` is on PATH, every verdict
//! is cross-checked against it (not required: no external solver is trusted).

use invar::laws_synth::{
    SOLVER, SOLVER_VERSION,
    formula::{Cond, Formula, Sort, Term},
    prove::{MAX_MONOMIALS, Mutation, Verdict, decide, decide_mutated},
};
use std::{
    io::Write,
    process::{Command, Stdio},
};

fn v(name: &str) -> Term {
    Term::var(name)
}

/// Gaussian integers `(re, im)` with `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`.
fn gauss(x: (Term, Term), y: (Term, Term)) -> (Term, Term) {
    let ((a, b), (c, d)) = (x, y);
    (a.clone() * c.clone() - b.clone() * d.clone(), a * d + b * c)
}

type Quat = (Term, Term, Term, Term);

/// The Hamilton product of quaternions `w + xi + yj + zk`.
fn hamilton(p: Quat, q: Quat) -> Quat {
    let (a1, b1, c1, d1) = p;
    let (a2, b2, c2, d2) = q;
    let m = |x: &Term, y: &Term| x.clone() * y.clone();
    (
        m(&a1, &a2) - m(&b1, &b2) - m(&c1, &c2) - m(&d1, &d2),
        m(&a1, &b2) + m(&b1, &a2) + m(&c1, &d2) - m(&d1, &c2),
        m(&a1, &c2) - m(&b1, &d2) + m(&c1, &a2) + m(&d1, &b2),
        m(&a1, &d2) + m(&b1, &c2) - m(&c1, &b2) + m(&d1, &a2),
    )
}

fn quat(prefix: &str) -> Quat {
    (
        v(&format!("{prefix}w")),
        v(&format!("{prefix}x")),
        v(&format!("{prefix}y")),
        v(&format!("{prefix}z")),
    )
}

fn quat_equations(l: Quat, r: Quat) -> Vec<(Term, Term)> {
    vec![(l.0, r.0), (l.1, r.1), (l.2, r.2), (l.3, r.3)]
}

/// Max-subarray segments `(sum, best prefix, best suffix, best)`.
type Seg = (Term, Term, Term, Term);

fn seg(prefix: &str) -> Seg {
    (
        v(&format!("{prefix}s")),
        v(&format!("{prefix}p")),
        v(&format!("{prefix}q")),
        v(&format!("{prefix}m")),
    )
}

/// The max-subarray combine; `cross: false` drops the spanning candidate.
fn combine(x: Seg, y: Seg, cross: bool) -> Seg {
    let (s1, p1, q1, m1) = x;
    let (s2, p2, q2, m2) = y;
    let best = m1.max(m2);
    (
        s1.clone() + s2.clone(),
        p1.max(s1 + p2.clone()),
        q2.max(q1.clone() + s2),
        if cross { best.max(q1 + p2) } else { best },
    )
}

fn seg_equations(l: Seg, r: Seg) -> Vec<(Term, Term)> {
    vec![(l.0, r.0), (l.1, r.1), (l.2, r.2), (l.3, r.3)]
}

/// The case-split formulas with the verdict each must get.
fn split_cases() -> Vec<(&'static str, Formula, bool)> {
    let int = |l: Term, r: Term| Formula::new(Sort::Int, vec![(l, r)]);
    let (a, b, c) = (v("a"), v("b"), v("c"));
    let xy = v("x") * v("y");
    let lt = Term::ite(Cond::Lt(v("x"), Term::Lit(1)), Term::Lit(0), Term::Lit(1));
    let le = Term::ite(Cond::Le(v("x"), Term::Lit(0)), Term::Lit(0), Term::Lit(1));
    let assoc = |cross| {
        Formula::new(
            Sort::Int,
            seg_equations(
                combine(combine(seg("a"), seg("b"), true), seg("c"), true),
                combine(seg("a"), combine(seg("b"), seg("c"), true), cross),
            ),
        )
    };
    vec![
        (
            "max is associative",
            int(
                a.clone().max(b.clone()).max(c.clone()),
                a.clone().max(b.clone().max(c)),
            ),
            true,
        ),
        (
            "max commutes",
            int(a.clone().max(b.clone()), b.clone().max(a.clone())),
            true,
        ),
        (
            "max(a, b) = a fails",
            int(a.clone().max(b.clone()), a.clone()),
            false,
        ),
        (
            "max(a, b) + min(a, b) = a + b",
            int(a.clone().max(b.clone()) + a.clone().min(b.clone()), a + b),
            true,
        ),
        ("x < 1 is x <= 0 over Z", int(lt, le), true),
        (
            "max(xy, 0) + min(xy, 0) = xy",
            int(
                xy.clone().max(Term::Lit(0)) + xy.clone().min(Term::Lit(0)),
                xy,
            ),
            true,
        ),
        ("max-subarray combine is associative", assoc(true), true),
        (
            "max-subarray combine without the spanning candidate",
            assoc(false),
            false,
        ),
        (
            "max over Z/2^32 is outside the fragment",
            Formula::new(Sort::Bv32, vec![(v("a").max(v("b")), v("b").max(v("a")))]),
            false,
        ),
    ]
}

/// The formulas of this file with the verdict each must get.
fn cases() -> Vec<(&'static str, Formula, bool)> {
    let g = |p: &str| (v(&format!("{p}re")), v(&format!("{p}im")));
    let (gl, gr) = (
        gauss(gauss(g("a"), g("b")), g("c")),
        gauss(g("a"), gauss(g("b"), g("c"))),
    );
    let gauss_assoc = Formula::new(Sort::Bv32, vec![(gl.0, gr.0), (gl.1, gr.1)]);
    let (gab, gba) = (gauss(g("a"), g("b")), gauss(g("b"), g("a")));
    let gauss_commutes = Formula::new(Sort::Bv32, vec![(gab.0, gba.0), (gab.1, gba.1)]);
    let cancels = Formula::new(
        Sort::Int,
        vec![(
            v("a") * (v("b") + v("c")) - v("a") * v("c"),
            v("a") * v("b"),
        )],
    );
    let gi = gauss(g("a"), (Term::Lit(1), Term::Lit(0)));
    let gauss_identity = Formula::new(Sort::Bv32, vec![(gi.0, v("are")), (gi.1, v("aim"))]);
    let quat_assoc = Formula::new(
        Sort::Int,
        quat_equations(
            hamilton(hamilton(quat("p"), quat("q")), quat("r")),
            hamilton(quat("p"), hamilton(quat("q"), quat("r"))),
        ),
    );
    let quat_commutes = Formula::new(
        Sort::Int,
        quat_equations(
            hamilton(quat("p"), quat("q")),
            hamilton(quat("q"), quat("p")),
        ),
    );
    // A wrong sign in one component of the right-hand product (mutation).
    let (r0, r1, r2, r3) = hamilton(quat("p"), hamilton(quat("q"), quat("r")));
    let quat_mutant = Formula::new(
        Sort::Int,
        quat_equations(
            hamilton(hamilton(quat("p"), quat("q")), quat("r")),
            (r0, r1, -r2, r3),
        ),
    );
    let sub_assoc = Formula::new(
        Sort::Int,
        vec![(v("a") - v("b") - v("c"), v("a") - (v("b") - v("c")))],
    );
    let word = |sort| Formula::new(sort, vec![(Term::Lit(4_294_967_295) * v("x"), -v("x"))]);
    vec![
        ("gaussian u32 multiply is associative", gauss_assoc, true),
        ("(1, 0) is the gaussian identity", gauss_identity, true),
        ("gaussian multiply commutes", gauss_commutes, true),
        ("a (b + c) - a c = a b", cancels, true),
        ("quaternion multiply is associative", quat_assoc, true),
        ("quaternion multiply does not commute", quat_commutes, false),
        ("mutated quaternion product", quat_mutant, false),
        ("sub is not associative", sub_assoc, false),
        ("(2^32 - 1) x = -x in Z/2^32", word(Sort::Bv32), true),
        ("(2^32 - 1) x = -x fails over Z", word(Sort::Int), false),
    ]
    .into_iter()
    .chain(split_cases())
    .collect()
}

#[test]
fn prove_ring_identities() {
    let mut proved = 0;
    for (name, formula, valid) in cases() {
        let verdict = decide(&formula);
        assert_eq!(verdict == Verdict::Proved, valid, "{name}: {verdict:?}");
        proved += usize::from(valid);
    }
    assert_eq!(proved, 12);
    assert_eq!((SOLVER, SOLVER_VERSION), ("invar-prove", 1));
}

#[test]
fn prove_smtlib_text() {
    let formula = Formula::new(
        Sort::Int,
        vec![(
            v("a") * (v("b") + Term::Lit(-2)),
            v("a") * v("b") - v("a") - v("a"),
        )],
    );
    assert_eq!(
        formula.smtlib(),
        "(set-logic QF_NIA)\n(declare-const a Int)\n(declare-const b Int)\n\
         (assert (not (= (* a (+ b (- 2))) (- (- (* a b) a) a))))\n(check-sat)\n"
    );
    assert_eq!(decide(&formula), Verdict::Proved);
    let words = Formula::new(
        Sort::Bv32,
        vec![(v("x") + Term::Lit(-1), v("x")), (-v("x"), Term::Lit(0))],
    );
    assert_eq!(
        words.smtlib(),
        "(set-logic QF_BV)\n(declare-const x (_ BitVec 32))\n\
         (assert (not (and (= (bvadd x (_ bv4294967295 32)) x) (= (bvneg x) (_ bv0 32)))))\n\
         (check-sat)\n"
    );
    assert!(matches!(decide(&words), Verdict::Unproved(_)));
}

/// Distinct names get distinct symbols, never a builtin; a sum past
/// [`MAX_MONOMIALS`] is unproved.
#[test]
fn prove_symbols_and_limits() {
    let names = ["|", "_", "#p", "\\", "true", "bvadd", "x"];
    let sum = names
        .iter()
        .map(|n| v(n))
        .reduce(|a, b| a + b)
        .expect("names");
    let text = Formula::new(Sort::Int, vec![(sum.clone(), sum)]).smtlib();
    let declared: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("(declare-const ")?.strip_suffix(" Int)"))
        .collect();
    let expected = [
        "|###p|", "|##b|", "|#_|", "|#bvadd|", "|#true|", "x", "|##p|",
    ];
    assert_eq!(declared, expected);
    // A balanced sum of distinct variables, one past the bound.
    let mut terms: Vec<Term> = (0..=MAX_MONOMIALS).map(|i| v(&format!("x{i}"))).collect();
    while terms.len() > 1 {
        let mut pairs = terms.into_iter();
        let mut next = Vec::new();
        while let Some(a) = pairs.next() {
            next.push(match pairs.next() {
                Some(b) => a + b,
                None => a,
            });
        }
        terms = next;
    }
    let big = terms.pop().expect("sum");
    let verdict = decide(&Formula::new(Sort::Int, vec![(big.clone(), big)]));
    assert_eq!(
        verdict,
        Verdict::Unproved("equation 0: polynomial too large".to_owned())
    );
}

/// A deep `max` chain normalises and prints in linear size (each operand
/// once), and its SMT-LIB2 form binds operands with `let`.
#[test]
fn prove_nested_max_stays_linear() {
    let mut chain = v("x");
    for _ in 0..40 {
        chain = chain.max(v("x"));
    }
    let deep = Formula::new(Sort::Int, vec![(chain, v("x"))]);
    assert_eq!(decide(&deep), Verdict::Proved);
    assert!(deep.smtlib().len() < 40 * 80, "{}", deep.smtlib().len());
    let swap = Formula::new(Sort::Int, vec![(v("a").max(v("b")), v("b").min(v("a")))]);
    let text = swap.smtlib();
    assert!(
        text.contains(
            "(= (let ((|#l| a) (|#r| b)) (ite (<= |#r| |#l|) |#l| |#r|)) \
             (let ((|#l| b) (|#r| a)) (ite (<= |#l| |#r|) |#l| |#r|)))"
        ),
        "{text}"
    );
    assert!(matches!(decide(&swap), Verdict::Unproved(_)));
}

/// Each named rule mutation changes some verdict of [`cases`].
#[test]
fn prove_kills_every_rule_mutation() {
    for mutation in Mutation::ALL {
        let killed = cases()
            .into_iter()
            .find(|(_, formula, valid)| {
                (decide_mutated(formula, mutation) == Verdict::Proved) != *valid
            })
            .map(|(name, ..)| name);
        assert!(killed.is_some(), "{mutation:?} survives every case");
        eprintln!("{mutation:?} killed by: {}", killed.unwrap_or_default());
    }
    for (name, formula, _) in cases() {
        assert_eq!(
            decide_mutated(&formula, Mutation::None),
            decide(&formula),
            "{name}"
        );
    }
}

/// Cross-check against z3 where it is installed; it is never required.
#[test]
fn prove_agrees_with_z3_when_present() {
    let Ok(probe) = Command::new("z3").arg("-version").output() else {
        eprintln!("z3 not on PATH: cross-check not run");
        return;
    };
    assert!(probe.status.success());
    for (name, formula, valid) in cases() {
        let mut child = Command::new("z3")
            .args(["-in", "-T:60"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn z3");
        child
            .stdin
            .take()
            .expect("stdin")
            .write_all(formula.smtlib().as_bytes())
            .expect("write");
        let out = child.wait_with_output().expect("z3 output");
        let answer = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        // A z3 timeout or `unknown` is no verdict; only a contradiction fails.
        let contradicts = if valid { "sat" } else { "unsat" };
        assert_ne!(answer, contradicts, "{name}: z3 disagrees");
        eprintln!("{name}: z3 {answer}");
    }
}
