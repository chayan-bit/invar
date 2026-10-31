#![forbid(unsafe_code)]
//! ENG-02 (#59): lifting certificates. Kadane's loop carries `cur` and
//! `best`; the certificate lifts it with the running sum `s` and the best
//! non-empty prefix `p`, so chunks combine with the max-subarray join. The
//! checker re-derives the obligation from the net and refuses every tampered
//! file: an edited formula, an edited result or solver, a dropped join
//! candidate, a changed loop, and a state that is not the loop's.

use crate::support;

use invar::{
    general_net::regions::{self, Partition},
    laws_synth::{
        certificate::{Component, LawCertificate, Refusal},
        formula::Term,
    },
};

const KADANE: &str = "fn f(xs: read Array<i32>, n: u32) -> i32 { var best = 0 - 2147483647; var cur = 0; for i in 0u32 .. n { cur = cur + xs[i]; if cur < 0 { cur = 0; } if cur > best { best = cur; } } return best; }";
/// The same loop clamped at 1: a different net.
const CLAMPED: &str = "fn f(xs: read Array<i32>, n: u32) -> i32 { var best = 0 - 2147483647; var cur = 0; for i in 0u32 .. n { cur = cur + xs[i]; if cur < 1 { cur = 0; } if cur > best { best = cur; } } return best; }";

fn partition(source: &str) -> Partition {
    let (net, proof) = support::compile(source);
    regions::partition(&net, &proof).expect("partition")
}

fn v(name: &str) -> Term {
    Term::var(name)
}

/// `(sum, best prefix, best suffix = cur, best)`; `spanning: false` drops
/// the `l.cur + r.p` candidate from the join of `best`.
fn kadane_state(spanning: bool) -> Vec<Component> {
    let x = || v("xs[i]");
    let best = v("l.best").max(v("r.best"));
    vec![
        Component {
            name: "s".to_owned(),
            step: Some(v("s") + x()),
            element: x(),
            join: v("l.s") + v("r.s"),
        },
        Component {
            name: "p".to_owned(),
            step: Some(v("p").max(v("s") + x())),
            element: x(),
            join: v("l.p").max(v("l.s") + v("r.p")),
        },
        Component {
            name: "cur".to_owned(),
            step: None,
            element: Term::Lit(0),
            join: v("r.cur").max(v("l.cur") + v("r.s")),
        },
        Component {
            name: "best".to_owned(),
            step: None,
            element: Term::Lit(0),
            join: if spanning {
                best.max(v("l.cur") + v("r.p"))
            } else {
                best
            },
        },
    ]
}

/// An edit applied to a sealed certificate.
type Tamper<'a> = &'a dyn Fn(&mut LawCertificate);

fn sealed() -> (Partition, LawCertificate) {
    let partition = partition(KADANE);
    let cert = LawCertificate::seal(&partition, "f", 0, kadane_state(true)).expect("seal");
    (partition, cert)
}

#[test]
fn certificate_max_subarray_is_accepted() {
    let (partition, cert) = sealed();
    assert_eq!(
        (cert.solver.as_str(), cert.result.as_str()),
        ("invar-prove", "unsat")
    );
    assert_eq!(cert.state.len(), 4);
    assert!(cert.formula.starts_with("(set-logic QF_NIA)\n"));
    let text = cert.to_json();
    let read = LawCertificate::from_json(&text).expect("round trip");
    assert_eq!(read, cert);
    assert_eq!(read.check(&partition), Ok(()));
}

#[test]
fn certificate_tampering_is_refused() {
    let (partition, cert) = sealed();
    let edit = |f: &dyn Fn(&mut LawCertificate)| {
        let mut forged = cert.clone();
        f(&mut forged);
        forged.check(&partition)
    };
    let cases: [(&str, Tamper, Refusal); 6] = [
        (
            "edited formula",
            &|c| c.formula = c.formula.replacen("(+ ", "(- ", 1),
            Refusal::FormulaDiffers,
        ),
        (
            "edited result",
            &|c| c.result = "sat".to_owned(),
            Refusal::Result,
        ),
        ("edited solver", &|c| c.solver_version = 2, Refusal::Solver),
        ("edited trust root", &|c| c.trust.push('!'), Refusal::Solver),
        (
            "join drops the spanning candidate",
            &|c| c.state = kadane_state(false),
            Refusal::FormulaDiffers,
        ),
        (
            "a loop that does not exist",
            &|c| c.loop_index = 1,
            Refusal::Loop("f has no loop 1".to_owned()),
        ),
    ];
    for (name, tamper, refusal) in cases {
        assert_eq!(edit(tamper), Err(refusal), "{name}");
    }
    // A state that does not carry the loop's scalars, or reads a stray name.
    let mut missing = cert.clone();
    missing.state.retain(|c| c.name != "best");
    assert!(matches!(missing.check(&partition), Err(Refusal::State(_))));
    let mut stray = cert.clone();
    stray.state[0].element = v("cur");
    assert!(matches!(stray.check(&partition), Err(Refusal::State(_))));
    // The certificate states a different net than the current one.
    assert_eq!(
        cert.check(&self::partition(CLAMPED)),
        Err(Refusal::FormulaDiffers)
    );
    // Resealing the broken join fails: the re-decided obligation is unproved.
    let broken = LawCertificate::seal(&partition, "f", 0, kadane_state(false));
    assert!(matches!(broken, Err(Refusal::Unproved(_))), "{broken:?}");
    assert!(matches!(
        LawCertificate::from_json("{\"format\": \"other\"}"),
        Err(Refusal::Malformed(_))
    ));
}
