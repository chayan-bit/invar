#![forbid(unsafe_code)]
//! Lowering annotations (P4a profile items 1-3 and 6): re-derived demand order,
//! fallback state and modulus boundaries are accepted; malformed, altered or
//! non-legality annotations are rejected; well-formed bounds and narrowing
//! are reported as unverified unless a named analysis the caller supplies
//! re-derives them.

use crate::support;

use invar::{
    front::Span,
    general_net::{
        Net, Proof,
        normal_form::{
            self, Annotated, Annotation, Certificate, Evidence, LawId, Provenance, RangeGuard,
            Signedness, Source, Subject, Target,
            certificate::{self, Analysis, Rejection},
        },
        regions::{self, Partition, RegionId},
    },
};

const L4_RESIDUE: &str = "fn f(xs: read Array<i32>, ys: read Array<i32>, n: u32, s0: i32) -> i32 { var s = s0; for i in 0u32 .. n { s = (s * xs[i] + ys[i]) % 10007; } return s; }";
const L9_TAIL: &str = "fn w(x: i32) -> i32 { return (x * 7 + 3) % 1009; } fn t(n: i32, s: i32) -> i32 { if n <= 0 { return w(s); } let sel = (s * 29 + n * 6) % 4; if sel == 0 { let v0 = t(n - 1, (s * 14 + 60) % 1009); let v1 = t(n - 1, (s * 12 + 66) % 1009); return (v0 * 7 + v1 + s) % 10007; } if sel == 1 { return (w(s) + n) % 10007; } return t(n - 1, (s * 4 + 22) % 1009); }";

const TRIVIAL_REMAINDER: &str = "fn f(xs: read Array<i32>, ys: read Array<i32>, n: u32, s0: i32) -> i32 { var s = s0; for i in 0u32 .. n { s = s + xs[i] % 1 + ys[i] % 7; } return s; }";

fn certified(source: &str, law: LawId) -> (Net, Proof, Certificate, Vec<Annotated>) {
    let (net, proof) = support::compile(source);
    let form = normal_form::extract(&net, &proof).expect("extract");
    let cert = form
        .certificates()
        .find(|c| c.law == law)
        .cloned()
        .expect("certificate");
    let partition = regions::partition(&net, &proof).expect("partition");
    let derived = certificate::rederive(&partition, cert.region).expect("rederive");
    (net, proof, cert, derived)
}

fn with(cert: &Certificate, annotations: Vec<Annotated>) -> Certificate {
    let mut forged = cert.clone();
    forged.annotations = annotations;
    forged
}

fn ops(derived: &[Annotated]) -> Vec<Span> {
    match &derived[0].annotation {
        Annotation::Demand(order) => order.clone(),
        other => panic!("demand first: {other:?}"),
    }
}

#[test]
fn rederived_annotations_are_accepted() {
    let (net, proof, cert, derived) = certified(L4_RESIDUE, LawId::L4Affine);
    assert_eq!(cert.subject, Subject::Var("s".into()));
    // `*`, `xs[i]`, `+`, `ys[i]`, `%`: five sites, one modulus boundary.
    assert_eq!(ops(&derived).len(), 5, "{derived:?}");
    assert!(
        derived.contains(&Annotated::rederived(Annotation::Fallback {
            vars: vec!["s".into()],
            arrays: Vec::new(),
        }))
    );
    let modulus: Vec<_> = derived
        .iter()
        .filter(|a| matches!(a.annotation, Annotation::Modulus { modulus: 10007, .. }))
        .collect();
    assert_eq!(modulus.len(), 1);
    assert_eq!(
        certificate::check(&net, &proof, &with(&cert, derived)),
        Ok(())
    );

    let (net, proof, cert, derived) = certified(L9_TAIL, LawId::L9PathSum);
    assert!(derived.len() > 2);
    assert_eq!(
        certificate::check(&net, &proof, &with(&cert, derived)),
        Ok(())
    );
}

#[test]
fn altered_or_malformed_annotations_are_rejected() {
    let (net, proof, cert, derived) = certified(L4_RESIDUE, LawId::L4Affine);
    let sites = ops(&derived);
    let op = sites[0];
    let modulus_site = derived
        .iter()
        .find_map(|a| match a.annotation {
            Annotation::Modulus {
                target: Target::Op(span),
                ..
            } => Some(span),
            _ => None,
        })
        .expect("modulus site");
    let legality = Annotated::rederived;
    let analysis = |annotation| Annotated {
        annotation,
        provenance: Provenance {
            evidence: Evidence::Legality,
            source: Source::Analysis {
                name: "ranges".into(),
                version: 1,
            },
        },
    };
    let bounds = |target, low, high, signedness| Annotation::Bounds {
        target,
        low,
        high,
        signedness,
    };
    let narrow = |target, bits, license| Annotation::Narrow {
        target,
        bits,
        signedness: Signedness::Signed,
        license,
    };
    let guard = |var: &str| RangeGuard {
        var: var.into(),
        low: 0,
        high: 1000,
    };
    let mut reversed = sites.clone();
    reversed.reverse();
    let mut dropped = sites.clone();
    dropped.pop();
    let mut repeated = sites.clone();
    repeated.push(op);
    let s = || Target::Var("s".into());

    let malformed: Vec<(&str, Annotated)> = vec![
        ("demand reversed", legality(Annotation::Demand(reversed))),
        ("demand dropped", legality(Annotation::Demand(dropped))),
        ("demand repeated", legality(Annotation::Demand(repeated))),
        (
            "demand foreign span",
            legality(Annotation::Demand(vec![Span { start: 0, end: 1 }])),
        ),
        (
            "fallback missing var",
            legality(Annotation::Fallback {
                vars: Vec::new(),
                arrays: Vec::new(),
            }),
        ),
        (
            "fallback extra array",
            legality(Annotation::Fallback {
                vars: vec!["s".into()],
                arrays: vec!["xs".into()],
            }),
        ),
        (
            "fallback unsorted",
            legality(Annotation::Fallback {
                vars: vec!["s".into(), "i".into()],
                arrays: Vec::new(),
            }),
        ),
        (
            "modulus value",
            legality(Annotation::Modulus {
                target: Target::Op(modulus_site),
                modulus: 10009,
            }),
        ),
        (
            "modulus on a non-% site",
            legality(Annotation::Modulus {
                target: Target::Op(op),
                modulus: 10007,
            }),
        ),
        (
            "modulus on a variable",
            legality(Annotation::Modulus {
                target: s(),
                modulus: 10007,
            }),
        ),
        (
            "bounds out of order",
            analysis(bounds(s(), 5, 4, Signedness::Signed)),
        ),
        (
            "unsigned below zero",
            analysis(bounds(Target::Op(op), -1, 4, Signedness::Unsigned)),
        ),
        (
            "signedness differs from i32",
            analysis(bounds(s(), 0, 4, Signedness::Unsigned)),
        ),
        (
            "bounds on an unknown variable",
            analysis(bounds(Target::Var("zz".into()), 0, 4, Signedness::Signed)),
        ),
        (
            "bounds on an unknown array",
            analysis(bounds(Target::Array("zz".into()), 0, 4, Signedness::Signed)),
        ),
        (
            "path weight outside L9",
            analysis(bounds(Target::PathWeight, 0, 4, Signedness::Signed)),
        ),
        ("narrow to 64 bits", analysis(narrow(s(), 64, Vec::new()))),
        (
            "narrow an array",
            analysis(narrow(Target::Array("xs".into()), 32, Vec::new())),
        ),
        (
            "licence on a non-live-in",
            analysis(narrow(Target::Op(op), 16, vec![guard("zz")])),
        ),
        (
            "licence repeated",
            analysis(narrow(Target::Op(op), 16, vec![guard("n"), guard("n")])),
        ),
    ];
    for (label, annotation) in malformed {
        assert!(
            matches!(
                certificate::check(&net, &proof, &with(&cert, vec![annotation])),
                Err(Rejection::Annotation(_))
            ),
            "{label}"
        );
    }

    for evidence in [
        Evidence::Profitability,
        Evidence::PlanningCost,
        Evidence::DeviceAcceptance,
    ] {
        let mut annotation = derived[0].clone();
        annotation.provenance.evidence = evidence;
        assert!(
            matches!(
                certificate::check(&net, &proof, &with(&cert, vec![annotation])),
                Err(Rejection::Annotation(_))
            ),
            "{evidence:?} is not legality"
        );
    }

    // Well-formed, but v1 cannot verify ranges: never accepted.
    let unverified = [
        analysis(bounds(s(), -5, 10006, Signedness::Signed)),
        analysis(bounds(Target::Op(op), 0, 4, Signedness::Unsigned)),
        legality(narrow(Target::Op(op), 32, vec![guard("n")])),
        analysis(derived[0].annotation.clone()),
    ];
    for annotation in unverified {
        let mut annotations = derived.clone();
        annotations.push(annotation.clone());
        assert_eq!(
            certificate::check(&net, &proof, &with(&cert, annotations)),
            Err(Rejection::Unverified),
            "{annotation:?}"
        );
    }
}

/// `% 1` is an ordinary operator site, never a modulus boundary: the rule is
/// `P >= 2` everywhere (rings, path sums, annotations), so `rederive` must not
/// emit what the checker rejects.
#[test]
fn trivial_remainders_are_sites_not_modulus_boundaries() {
    let (net, proof, cert, derived) = certified(TRIVIAL_REMAINDER, LawId::L3TrapLiftedFold);
    let moduli: Vec<u32> = derived
        .iter()
        .filter_map(|a| match a.annotation {
            Annotation::Modulus { modulus, .. } => Some(modulus),
            _ => None,
        })
        .collect();
    assert_eq!(moduli, vec![7], "{derived:?}");
    // `xs[i]`, `% 1`, `+`, `ys[i]`, `% 7`, `+`.
    assert_eq!(ops(&derived).len(), 6, "{derived:?}");
    assert_eq!(
        certificate::check(&net, &proof, &with(&cert, derived)),
        Ok(())
    );
}

/// A toy range analysis: every `i32` local lies in the `i32` range. It reads
/// only the net-derived statement tree.
struct DeclaredRanges {
    version: u32,
}

impl Analysis for DeclaredRanges {
    fn name(&self) -> &str {
        "declared-ranges"
    }

    fn version(&self) -> u32 {
        self.version
    }

    fn rederive(&self, partition: &Partition, region: RegionId) -> Vec<Annotation> {
        let Some(Ok(tree)) = partition.function(region.function).map(|f| f.tree.as_ref()) else {
            return Vec::new();
        };
        tree.locals
            .iter()
            .filter(|(_, ty)| **ty == invar::front::Type::I32)
            .map(|(name, _)| Annotation::Bounds {
                target: Target::Var(name.clone()),
                low: i64::from(i32::MIN),
                high: i64::from(i32::MAX),
                signedness: Signedness::Signed,
            })
            .collect()
    }
}

#[test]
fn named_analyses_verify_their_own_annotations() {
    let (net, proof, cert, _) = certified(L4_RESIDUE, LawId::L4Affine);
    let partition = regions::partition(&net, &proof).expect("partition");
    let bounds = |low: i64| Annotation::Bounds {
        target: Target::Var("s".into()),
        low,
        high: i64::from(i32::MAX),
        signedness: Signedness::Signed,
    };
    let from = |annotation, name: &str, version| Annotated {
        annotation,
        provenance: Provenance {
            evidence: Evidence::Legality,
            source: Source::Analysis {
                name: name.into(),
                version,
            },
        },
    };
    let ranges = DeclaredRanges { version: 1 };
    let analyses: [&dyn Analysis; 1] = [&ranges];
    let check = |annotated: Annotated| {
        certificate::check_with_analyses(&partition, &with(&cert, vec![annotated]), &analyses)
    };
    let low = i64::from(i32::MIN);
    assert_eq!(check(from(bounds(low), "declared-ranges", 1)), Ok(()));
    assert_eq!(
        certificate::check(
            &net,
            &proof,
            &with(&cert, vec![from(bounds(low), "declared-ranges", 1)])
        ),
        Err(Rejection::Unverified),
        "no analysis supplied"
    );
    assert_eq!(
        check(from(bounds(low), "declared-ranges", 2)),
        Err(Rejection::Unverified),
        "other version"
    );
    assert_eq!(
        check(from(bounds(low), "other", 1)),
        Err(Rejection::Unverified),
        "other analysis"
    );
    assert!(
        matches!(
            check(from(bounds(low + 1), "declared-ranges", 1)),
            Err(Rejection::Annotation(_))
        ),
        "a bound the analysis does not derive"
    );
    assert_eq!(
        check(Annotated::rederived(bounds(low))),
        Err(Rejection::Unverified),
        "P1 itself derives no bounds"
    );
}
