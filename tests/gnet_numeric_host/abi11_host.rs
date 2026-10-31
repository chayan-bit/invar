//! Actual loaded ABI11 owned numeric continuations. The independent owned net
//! oracle is the comparator; it never supplies a result.
//! Issue label GNET-11.

use invar::{
    eval,
    general_net::{
        compiled,
        numeric_owned_resume_host::{Error, Plan, Progress},
        oracle,
    },
    numeric_contract::F32Bits,
};
use std::{path::Path, rc::Rc};

const FIXTURES: [&str; 12] = [
    "literals",
    "filled_empty",
    "update_fma",
    "loans_moves",
    "replacement",
    "mixed_relocation",
    "root_return",
    "element_copy",
    "index_trap",
    "store_trap",
    "older_integer_trap",
    "array_size_trap",
];
fn storage(elements: usize) -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements,
        allocations: 128,
        cleanup_events: 128,
    }
}
fn limits(elements: usize) -> oracle::Limits {
    oracle::Limits {
        fuel: 1000,
        transitions: 100_000,
        trace_events: 10_000,
        array_elements: elements,
    }
}
fn plan(source: &str, entry: &str, elements: usize) -> Plan {
    Plan::from_source(
        source.as_bytes(),
        entry,
        Default::default(),
        storage(elements),
        Path::new("c++"),
    )
    .unwrap()
}
fn equal(a: &Result<eval::Value, eval::EvalError>, b: &Result<eval::Value, eval::EvalError>) {
    match (a, b) {
        (Ok(eval::Value::F32(a)), Ok(eval::Value::F32(b))) => assert!(a.strict_eq(*b)),
        (Ok(eval::Value::ArrayF32(a)), Ok(eval::Value::ArrayF32(b))) => {
            let (a, b) = (a.borrow(), b.borrow());
            assert_eq!(a.len(), b.len());
            assert!(a.iter().zip(b.iter()).all(|(a, b)| a.strict_eq(*b)));
        }
        (
            Err(eval::EvalError::Tool { code: a, .. }),
            Err(eval::EvalError::Tool { code: b, .. }),
        ) => assert_eq!(a, b),
        _ => assert_eq!(a, b),
    }
}
fn same(actual: &oracle::Observation, expected: &oracle::Observation) {
    equal(&actual.semantic.result, &expected.semantic.result);
    assert_eq!(actual.semantic.steps_used, expected.semantic.steps_used);
    assert_eq!(
        actual.semantic.cleanup_trace,
        expected.semantic.cleanup_trace
    );
    assert_eq!(actual.events, expected.events);
    assert_eq!(actual.transitions, expected.transitions);
    assert_eq!(actual.max_frames, expected.max_frames);
    assert_eq!(actual.failure, expected.failure);
}
/// Drive one fresh task to completion with a fixed quantum.
fn drive(
    p: &mut Plan,
    args: &[eval::Value],
    l: oracle::Limits,
    quantum: u64,
) -> (oracle::Observation, usize) {
    let evidence = p.evidence_directory().to_owned();
    let mut run = p
        .start(args, l)
        .unwrap_or_else(|e| panic!("{e:?}; evidence {}", evidence.display()));
    let mut yields = 0;
    loop {
        match run
            .resume(quantum)
            .unwrap_or_else(|e| panic!("{e:?}; evidence {}", evidence.display()))
        {
            Progress::Pending(pending) => {
                assert!(pending.depth > 0);
                assert_eq!(pending.transitions, quantum * (yields as u64 + 1));
                yields += 1;
            }
            Progress::Complete(o) => return (o, yields),
        }
    }
}
fn compare(p: &mut Plan, args: &[eval::Value], l: oracle::Limits) -> oracle::Observation {
    let (net, proof) = p.artifact().checked_net();
    let entry = p.artifact().manifest()["entry"]
        .as_str()
        .unwrap()
        .to_owned();
    let expected = oracle::observe_call(net, proof, &entry, args, l).unwrap();
    let mut first = None;
    for quantum in [1, 2, 3, 17, 1_000_000] {
        let (actual, yields) = drive(p, args, l, quantum);
        same(&actual, &expected);
        // Completion finishes inside its last slice; a transition-limit stop is
        // observed only at the next attempted transition, after a full slice.
        let t = expected.transitions;
        assert!(
            yields as u64 == t.saturating_sub(1) / quantum || yields as u64 == t / quantum,
            "quantum {quantum}: {yields} yields for {t} transitions"
        );
        if first.is_none() {
            first = Some(actual);
        }
    }
    first.unwrap()
}

#[test]
fn loaded_fixture_quanta_match_oracle() {
    for name in FIXTURES {
        let source = std::fs::read_to_string(format!(
            "tests/issue_gnet_10_f32_arrays_fixtures/{name}.inv"
        ))
        .unwrap();
        let elements = if name == "mixed_relocation" { 3 } else { 128 };
        let mut p = plan(&source, "main", elements);
        let done = compare(&mut p, &[], limits(elements));
        for fuel in 1..=done.semantic.steps_used + 1 {
            compare(
                &mut p,
                &[],
                oracle::Limits {
                    fuel,
                    ..limits(elements)
                },
            );
        }
        let evidence = p.evidence_directory().to_owned();
        drop(p);
        assert!(!evidence.exists(), "{name} directory survived success");
    }
}

#[test]
fn loaded_recursion_loans_and_budget_failures() {
    // Nested frames hold loan aliases and owners across yields.
    let s = "fn fill(xs:mut Array<f32>,n:u32)->Unit{if n>0u32{let t=filled_f32(1u32,1.0);xs[0u32]=fma(xs[0u32],t[0u32],1.0);fill(xs,n-1u32);}} fn run(x:f32,n:u32)->Array<f32>{var xs:Array<f32> = [x,-0.0];fill(xs,n);let ys=move xs;return ys;}";
    let mut p = plan(s, "run", 128);
    for (bits, n) in [(0x3f800001, 5), (0x80000000, 0), (0x7fa00001, 3)] {
        let args = [
            eval::Value::F32(F32Bits::from_bits(bits)),
            eval::Value::U32(n),
        ];
        compare(&mut p, &args, limits(128));
        for cap in [0, 1, 2, 3, 7, 40] {
            compare(
                &mut p,
                &args,
                oracle::Limits {
                    transitions: cap,
                    ..limits(128)
                },
            );
            compare(
                &mut p,
                &args,
                oracle::Limits {
                    trace_events: cap as usize,
                    ..limits(128)
                },
            );
        }
    }
    let s = "fn run(x:f32)->f32{return run(x);}";
    let mut p = plan(s, "run", 128);
    let o = compare(
        &mut p,
        &[eval::Value::F32(F32Bits::from_bits(1))],
        limits(128),
    );
    assert_eq!(o.max_frames, 64);
    for (allocations, cleanup_events) in [(1, 128), (128, 0)] {
        let s = "fn run()->i32{let a:Array<f32> = [1.0];let b:Array<f32> = [2.0];return 7;}";
        let limits_for = compiled::OwnedLimits {
            elements: 128,
            allocations,
            cleanup_events,
        };
        let mut p = Plan::from_source(
            s.as_bytes(),
            "run",
            Default::default(),
            limits_for,
            Path::new("c++"),
        )
        .unwrap();
        let (o, _) = drive(&mut p, &[], limits(128), 1);
        assert_eq!(o.failure, Some(oracle::ToolFailure::Allocation));
    }
}

#[test]
fn loaded_terminal_owners_are_fresh_on_every_retrieval() {
    let s = "fn run(x:f32)->Array<f32>{let a:Array<f32> = [x,-0.0,0.0/0.0];return a;}";
    let mut p = plan(s, "run", 128);
    for bits in [0, 1, 0x80000000, 0x7f800000, 0x7fa00001, 0xffc12345] {
        let args = [eval::Value::F32(F32Bits::from_bits(bits))];
        let mut run = p.start(&args, limits(128)).unwrap();
        let first = loop {
            if let Progress::Complete(o) = run.resume(2).unwrap() {
                break o;
            }
        };
        let Progress::Complete(second) = run.resume(1_000_000).unwrap() else {
            panic!("terminal replay yielded")
        };
        let (Ok(eval::Value::ArrayF32(a)), Ok(eval::Value::ArrayF32(b))) =
            (first.semantic.result, second.semantic.result)
        else {
            panic!()
        };
        assert!(!Rc::ptr_eq(&a, &b));
        assert_eq!(a.borrow()[0].to_bits(), bits);
        assert_eq!(a.borrow()[1].to_bits(), 0x80000000);
        a.borrow_mut()[0] = F32Bits::from_bits(0x3f800000);
        assert_eq!(b.borrow()[0].to_bits(), bits);
        assert_eq!(second.events, first.events);
        // Invalid quanta are refused even after completion.
        assert!(matches!(run.resume(0), Err(Error::Limit)));
    }
}

#[test]
fn loaded_drop_pending_refusals_and_isolation() {
    let s = "fn run(n:u32)->Array<u32>{var xs:Array<u32> = filled_u32(n,7u32);for i in 0u32..n{let t:Array<i32> = [1,2];xs[i]=xs[i]+i;}return xs;}";
    let t = "fn run(n:u32)->Array<i32>{let ys:Array<i32> = filled_i32(n,-1);return ys;}";
    let mut a = plan(s, "run", 128);
    let mut b = plan(t, "run", 128);
    let args = [eval::Value::U32(4)];
    for _ in 0..3 {
        // Abandon pending tasks; destruction precedes any further use.
        let mut run = a.start(&args, limits(128)).unwrap();
        for _ in 0..5 {
            assert!(matches!(run.resume(3).unwrap(), Progress::Pending(_)));
        }
        for invalid in [0, 1_000_001, u64::MAX] {
            assert!(matches!(run.resume(invalid), Err(Error::Limit)));
        }
        let Progress::Pending(p) = run.resume(1).unwrap() else {
            panic!("finished early")
        };
        assert_eq!(p.transitions, 16);
        drop(run);
    }
    // Interleave two simultaneously loaded artifacts with live tasks.
    let mut x = a.start(&args, limits(128)).unwrap();
    let mut y = b.start(&args, limits(128)).unwrap();
    let (mut xo, mut yo) = (None, None);
    while xo.is_none() || yo.is_none() {
        if xo.is_none()
            && let Progress::Complete(o) = x.resume(2).unwrap()
        {
            xo = Some(o);
        }
        if yo.is_none()
            && let Progress::Complete(o) = y.resume(3).unwrap()
        {
            yo = Some(o);
        }
    }
    let Ok(eval::Value::ArrayU32(xs)) = xo.unwrap().semantic.result else {
        panic!()
    };
    let Ok(eval::Value::ArrayI32(ys)) = yo.unwrap().semantic.result else {
        panic!()
    };
    assert_eq!(*xs.borrow(), [7, 8, 9, 10]);
    assert_eq!(*ys.borrow(), [-1, -1, -1, -1]);
    drop((x, y));
    assert!(matches!(a.start(&[], limits(128)), Err(Error::Input)));
    assert!(matches!(
        a.start(&[eval::Value::I32(1)], limits(128)),
        Err(Error::Input)
    ));
    assert!(matches!(a.start(&args, limits(127)), Err(Error::Limit)));
    for l in [
        oracle::Limits {
            fuel: 0,
            ..limits(128)
        },
        oracle::Limits {
            transitions: 1_000_001,
            ..limits(128)
        },
        // Three trace copies alone exceed the aggregate 128 MiB buffer cap.
        oracle::Limits {
            trace_events: 1_000_000,
            ..limits(128)
        },
    ] {
        assert!(matches!(a.start(&args, l), Err(Error::Limit)));
    }
    // Refusals did not poison the plan.
    let (o, _) = drive(&mut a, &args, limits(128), 17);
    assert!(o.semantic.result.is_ok());
    let evidence = [
        a.evidence_directory().to_owned(),
        b.evidence_directory().to_owned(),
    ];
    drop((a, b));
    assert!(evidence.iter().all(|e| !e.exists()));
}

#[test]
fn loaded_receipt_binds_identity_without_library_in_prelink() {
    let p = plan("fn run(x:i32)->i32{return x;}", "run", 16);
    let receipt = p.receipt();
    assert_eq!(receipt["profile"], "cpu-owned-numeric-continuation-v1");
    assert_eq!(receipt["prelink"]["abi"], 11);
    assert_eq!(receipt["prelink"]["policy"], "strict-f32-v1");
    assert!(receipt["prelink"].get("library_sha256").is_none());
    assert_eq!(receipt["library_sha256"].as_str().unwrap().len(), 64);
    let symbol = receipt["symbol"].as_str().unwrap();
    assert_eq!(
        symbol,
        format!(
            "invar_{}_owned_resume_v1",
            receipt["identity"].as_str().unwrap()
        )
    );
    assert_eq!(p.artifact().artifact().abi_version(), 11);
}

#[test]
fn loaded_refuses_non_abi11_admission() {
    // External owner arguments remain outside this profile.
    assert!(
        Plan::from_source(
            b"fn run(xs:take Array<f32>)->i32{return 0;}",
            "run",
            Default::default(),
            storage(16),
            Path::new("c++"),
        )
        .is_err()
    );
    let missing = Plan::from_source(
        b"fn run()->i32{return 1;}",
        "run",
        Default::default(),
        storage(16),
        Path::new("invar-no-such-compiler-11"),
    );
    assert!(matches!(missing, Err(Error::Input)));
}

#[cfg(unix)]
#[test]
fn loader_refuses_changed_library_bytes_and_identity() {
    use invar_foreign::numeric_owned_resume::{Driver, Error as Loader, file_digest};
    let dir = std::env::temp_dir().join(format!("gnet11-loader-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (cpp, lib) = (dir.join("id.cpp"), dir.join("id.so"));
    std::fs::write(&cpp, "#include <cstddef>\nextern \"C\" unsigned int invar_fault_identity(unsigned char* p,std::size_t n){if(n!=32)return 2;for(int i=0;i<32;i++)p[i]=0;return 0;}").unwrap();
    let shared = if cfg!(target_os = "macos") {
        "-dynamiclib"
    } else {
        "-shared"
    };
    let out = std::process::Command::new("c++")
        .args(["-fPIC", shared])
        .arg(&cpp)
        .arg("-o")
        .arg(&lib)
        .output()
        .unwrap();
    crate::child_deadline::assert_success(&out);
    let hash = file_digest(&lib).unwrap();
    let load = |path: &Path, id| {
        Driver::load(path, hash, id, "invar_fault", vec![], 144 + 64, [4, 4]).map(drop)
    };
    // A wrong embedded identity and changed library bytes are both refused.
    assert!(matches!(load(&lib, [17; 32]), Err(Loader::Identity)));
    std::fs::write(&lib, b"forged library").unwrap();
    assert!(matches!(load(&lib, [0; 32]), Err(Loader::Identity)));
    std::fs::remove_dir_all(&dir).unwrap();
}
