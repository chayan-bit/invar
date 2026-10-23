use super::*;

#[test]
fn every_law_agrees_with_the_net_oracle() {
    let mut per_law: BTreeMap<LawId, Tally> = BTreeMap::new();
    let mut report = Vec::new();
    let mut failures = Vec::new();
    for (p, program) in programs().iter().enumerate() {
        let (net, proof, form) = compiled(program);
        let mut rng = Rng(0x5eed_0000 + p as u64);
        let mut tally = Tally::default();
        for _ in 0..CASES {
            let args = (program.args)(&mut rng);
            let Some((same, receipt, trapped)) =
                differential(&net, &proof, &form, program, &args, Mutation::None)
            else {
                continue;
            };
            tally.cases += 1;
            tally.traps += u64::from(trapped);
            let used = program
                .laws
                .iter()
                .any(|law| receipt.applied.contains_key(law));
            tally.law_path += u64::from(used);
            tally.fallbacks +=
                u64::from(!receipt.guard_fallbacks.is_empty() || !receipt.violations.is_empty());
            if !receipt.violations.is_empty() && tally.mismatches.len() < 5 {
                tally
                    .mismatches
                    .push(format!("violation {:?} on {args:?}", receipt.violations));
            }
            if !same && tally.mismatches.len() < 5 {
                tally.mismatches.push(format!("mismatch on {args:?}"));
            }
        }
        report.push(format!(
            "{:<16} cases={} traps={} law_path={} fallbacks={}",
            program.name, tally.cases, tally.traps, tally.law_path, tally.fallbacks
        ));
        if !tally.mismatches.is_empty() {
            failures.push(format!("{}: {:#?}", program.name, tally.mismatches));
        }
        if tally.traps == 0 {
            failures.push(format!("{}: no trapping cases", program.name));
        }
        if tally.law_path * 2 <= tally.cases {
            failures.push(format!(
                "{}: law path used in only {} of {} cases",
                program.name, tally.law_path, tally.cases
            ));
        }
        for law in program.laws {
            let t = per_law.entry(*law).or_default();
            t.cases += tally.cases;
            t.traps += tally.traps;
            t.law_path += tally.law_path;
            t.fallbacks += tally.fallbacks;
        }
    }
    for (law, t) in &per_law {
        report.push(format!(
            "{:<4} cases={} traps={} law_path={} fallbacks={}",
            law.to_string(),
            t.cases,
            t.traps,
            t.law_path,
            t.fallbacks
        ));
    }
    eprintln!("{}", report.join("\n"));
    assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
    let covered: Vec<LawId> = per_law.keys().copied().collect();
    assert_eq!(covered, LawId::ALL.to_vec(), "every v0.1 law is exercised");
    for (law, t) in &per_law {
        assert!(t.cases >= 3000, "{law:?}: {} cases", t.cases);
    }
}

/// Each mutation of the reference semantics must be caught by the oracle on
/// the programs of the law it breaks: the last trap instead of the first, a
/// dropped runtime guard, and a wrong combine (the right operand only, or a
/// reversed merge, or a stale round-start read).
#[test]
fn every_law_has_a_caught_mutation() {
    const MUTATION_CASES: usize = 800;
    let mutations = [
        Mutation::LastTrap,
        Mutation::DropGuards,
        Mutation::WrongCombine,
    ];
    let mut caught: BTreeMap<(LawId, &str), u64> = BTreeMap::new();
    let mut report = Vec::new();
    for (p, program) in programs().iter().enumerate() {
        let (net, proof, form) = compiled(program);
        let primary = *program.laws.last().expect("laws");
        let mut row = Vec::new();
        for mutation in mutations {
            let mut rng = Rng(0xbad_0000 + p as u64);
            let mut wrong = 0;
            for _ in 0..MUTATION_CASES {
                let args = (program.args)(&mut rng);
                if let Some((false, _, _)) =
                    differential(&net, &proof, &form, program, &args, mutation)
                {
                    wrong += 1;
                }
            }
            *caught
                .entry((primary, mutation_name(mutation)))
                .or_default() += wrong;
            row.push(format!("{}={wrong}", mutation_name(mutation)));
        }
        report.push(format!("{:<16} {}", program.name, row.join(" ")));
    }
    eprintln!("{}", report.join("\n"));
    let mut missing = Vec::new();
    for law in LawId::ALL {
        let total: u64 = mutations
            .iter()
            .map(|m| caught.get(&(law, mutation_name(*m))).copied().unwrap_or(0))
            .sum();
        if total == 0 {
            missing.push(law);
        }
    }
    assert!(missing.is_empty(), "no mutation caught for {missing:?}");
    for law in [LawId::L4Affine, LawId::L5Matrix, LawId::L9PathSum] {
        assert!(
            caught.get(&(law, "drop-guards")).copied().unwrap_or(0) > 0,
            "{law:?}: dropped guard not caught"
        );
    }
    let stride = programs()
        .into_iter()
        .find(|p| p.name == "l1-stride")
        .expect("l1-stride");
    let (net, proof, form) = compiled(&stride);
    let mut rng = Rng(0xbad_1000);
    let dropped = (0..MUTATION_CASES)
        .filter(|_| {
            let args = (stride.args)(&mut rng);
            matches!(
                differential(&net, &proof, &form, &stride, &args, Mutation::DropGuards),
                Some((false, _, _))
            )
        })
        .count();
    assert!(dropped > 0, "L1 stride: dropped guard not caught");
}

fn mutation_name(mutation: Mutation) -> &'static str {
    match mutation {
        Mutation::None => "none",
        Mutation::LastTrap => "last-trap",
        Mutation::DropGuards => "drop-guards",
        Mutation::WrongCombine => "wrong-combine",
    }
}

fn form_of(source: &str) -> (Net, Proof, NormalForm) {
    let (net, proof) = support::compile(source);
    let form = normal_form::extract(&net, &proof).expect("extract");
    (net, proof, form)
}

fn cert_where(form: &NormalForm, law: LawId, subject: &Subject) -> Certificate {
    form.certificates()
        .find(|c| c.law == law && c.subject == *subject)
        .cloned()
        .unwrap_or_else(|| panic!("no {law:?} certificate for {subject:?}"))
}

fn var(name: &str) -> Subject {
    Subject::Var(name.to_owned())
}

/// The checker re-derives every certified fact from the net: each forgery
/// below must be rejected (never accepted by name, hash or benchmark id).
#[test]
fn forged_certificates_are_rejected() {
    let mut accepted = Vec::new();
    let mut reject = |label: &str, net: &Net, proof: &Proof, cert: &Certificate| {
        if certificate::check(net, proof, cert).is_ok() {
            accepted.push(label.to_owned());
        }
    };
    let (net, proof, form) = form_of(L3_SUM);
    let sum = cert_where(&form, LawId::L3TrapLiftedFold, &var("s"));
    assert_eq!(certificate::check(&net, &proof, &sum), Ok(()));
    let forge = |f: &dyn Fn(&mut Certificate)| {
        let mut c = sum.clone();
        f(&mut c);
        c
    };
    reject(
        "fingerprint",
        &net,
        &proof,
        &forge(&|c| c.fingerprint[0] ^= 1),
    );
    reject("version", &net, &proof, &forge(&|c| c.version += 1));
    reject(
        "region function",
        &net,
        &proof,
        &forge(&|c| c.region.function += 1),
    );
    reject(
        "region head",
        &net,
        &proof,
        &forge(&|c| c.region.head = None),
    );
    reject(
        "law swap",
        &net,
        &proof,
        &forge(&|c| c.law = LawId::L4Affine),
    );
    reject(
        "rep swap",
        &net,
        &proof,
        &forge(&|c| c.rep = Some(Rep::SaturatedProduct(IntTy::I32))),
    );
    reject(
        "rep type",
        &net,
        &proof,
        &forge(&|c| c.rep = Some(Rep::LiftedSum(IntTy::U32))),
    );
    reject(
        "guard added",
        &net,
        &proof,
        &forge(&|c| c.guards.push(Guard::ReducedSeed("s".into()))),
    );
    reject(
        "subject renamed",
        &net,
        &proof,
        &forge(&|c| c.subject = var("zz")),
    );
    reject("facts altered", &net, &proof, &forge(&|c| c.facts.clear()));
    reject(
        "annotation added",
        &net,
        &proof,
        &forge(&|c| {
            c.annotations.push(Annotated::rederived(Annotation::Narrow {
                target: Target::Var("s".into()),
                bits: 32,
                signedness: Signedness::Signed,
                license: Vec::new(),
            }))
        }),
    );
    let (other_net, other_proof, _) = form_of(L3_PRODUCT);
    reject("certificate of another net", &other_net, &other_proof, &sum);

    let (net, proof, form) = form_of(L3_PRODUCT);
    let product = cert_where(&form, LawId::L3TrapLiftedFold, &var("p"));
    let mut lifted = product.clone();
    lifted.rep = Some(Rep::LiftedSum(IntTy::I32));
    reject("product claimed as a sum", &net, &proof, &lifted);

    let (net, proof, form) = form_of(L4_RESIDUE);
    let residue = cert_where(&form, LawId::L4Affine, &var("s"));
    assert_eq!(certificate::check(&net, &proof, &residue), Ok(()));
    let forge = |f: &dyn Fn(&mut Certificate)| {
        let mut c = residue.clone();
        f(&mut c);
        c
    };
    reject(
        "ring",
        &net,
        &proof,
        &forge(&|c| c.rep = Some(Rep::Affine(Ring::Checked64(IntTy::I32)))),
    );
    reject(
        "modulus",
        &net,
        &proof,
        &forge(&|c| {
            c.rep = Some(Rep::Affine(Ring::Residue {
                modulus: 10009,
                ty: IntTy::I32,
            }))
        }),
    );
    reject(
        "guard dropped",
        &net,
        &proof,
        &forge(&|c| {
            c.guards.retain(|g| !matches!(g, Guard::ReducedSeed(_)));
        }),
    );
    reject(
        "guard duplicated",
        &net,
        &proof,
        &forge(&|c| {
            let g = c.guards[0].clone();
            c.guards.push(g);
            c.guards.remove(1);
        }),
    );
    reject(
        "affine offset",
        &net,
        &proof,
        &forge(&|c| {
            if let Some(Fact::Affine { offset, .. }) = c.facts.first_mut() {
                *offset = Expr::I32(1);
            }
        }),
    );

    let (net, proof, form) = form_of(L1_STRIDE);
    let stride = cert_where(&form, LawId::L1Map, &Subject::Array("out".into()));
    assert_eq!(stride.guards, vec![Guard::StrideNonZero("k".into())]);
    let mut unguarded = stride.clone();
    unguarded.guards.clear();
    reject("stride guard dropped", &net, &proof, &unguarded);

    let (net, proof, form) = form_of(L2_SEALED);
    let mut sealed = cert_where(&form, LawId::L2SealedFold, &var("s"));
    sealed.rep = Some(Rep::Sealed(invar::laws::Law::WrappingProduct));
    reject("sealed sum as product", &net, &proof, &sealed);

    let (net, proof, form) = form_of(L7_ARGMIN);
    let select = cert_where(&form, LawId::L7Select, &var("m"));
    let mut flipped = select.clone();
    if let Some(Rep::Select { strict, .. }) = &mut flipped.rep {
        *strict = !*strict;
    }
    reject("select strictness", &net, &proof, &flipped);
    let mut dropped = form.clone();
    for function in &mut dropped.functions {
        for loop_form in &mut function.loops {
            for c in &mut loop_form.certificates {
                if let Some(Rep::Select { companions, .. }) = &mut c.rep {
                    companions.clear();
                }
            }
            if let Node::Parallel(node) = &mut loop_form.node {
                for carried in &mut node.carried {
                    if let Rep::Select { companions, .. } = &mut carried.rep {
                        companions.clear();
                    }
                }
            }
        }
    }
    assert!(
        certificate::check_all(&net, &proof, &dropped).is_err(),
        "companions dropped"
    );

    let (net, proof, form) = form_of(L11_CHECKED);
    let mut combine = cert_where(&form, LawId::L11ScatterCombine, &Subject::Array("h".into()));
    combine.law = LawId::L1Map;
    reject("scatter claimed owned", &net, &proof, &combine);
    let (net, proof, form) = form_of(L12_ORDERED);
    let mut ordered = cert_where(&form, LawId::L12OrderedWrite, &Subject::Array("out".into()));
    ordered.law = LawId::L14MonotoneFrontier;
    reject("ordered claimed monotone", &net, &proof, &ordered);

    let (net, proof, form) = form_of(L9_PATHSUM);
    let path = cert_where(&form, LawId::L9PathSum, &Subject::Recursion);
    let mut table = path.clone();
    table.guards = vec![
        Guard::TableRange { modulus: 100_019 },
        Guard::NonNegativeConstants,
        Guard::NoOverflowCombine,
    ];
    reject("path-sum table modulus", &net, &proof, &table);
    let mut coefficient = path.clone();
    if let Some(Fact::Combine { coefficients, .. }) = coefficient.facts.first_mut() {
        coefficients[0].1 += 1;
    }
    reject("path-sum coefficient", &net, &proof, &coefficient);
    let mut as_range = path.clone();
    as_range.law = LawId::L10RangeSum;
    as_range.guards.clear();
    reject("path sum claimed range sum", &net, &proof, &as_range);

    let (net, proof, form) = form_of(L20_FORK);
    let mut fork = cert_where(
        &form,
        LawId::L20ForkJoin,
        &form
            .certificates()
            .find(|c| c.law == LawId::L20ForkJoin)
            .expect("fork")
            .subject,
    );
    fork.facts = vec![Fact::Siblings(4)];
    reject("fork siblings + 1", &net, &proof, &fork);

    let (net, proof, form) = form_of(L19_FUSED);
    let fused = form
        .certificates()
        .find(|c| c.law == LawId::L19MapFusion)
        .cloned()
        .expect("fused");
    let mut self_fused = fused.clone();
    self_fused.subject = Subject::Fused(fused.region);
    reject("fusion with a non-adjacent loop", &net, &proof, &self_fused);

    let (net, proof, form) = form_of(L15_SCAN);
    let mut stripped = form.clone();
    for function in &mut stripped.functions {
        for loop_form in &mut function.loops {
            loop_form
                .certificates
                .retain(|c| !matches!(c.subject, Subject::Var(_)));
            if let Node::Parallel(node) = &mut loop_form.node {
                node.carried.clear();
            }
        }
    }
    assert!(
        certificate::check_all(&net, &proof, &stripped).is_err(),
        "carried state removed"
    );
    assert!(
        accepted.is_empty(),
        "forged certificates accepted: {accepted:?}"
    );
}
