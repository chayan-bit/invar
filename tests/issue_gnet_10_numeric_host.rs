//! Actual loaded ABI10 execution; source and oracle are independent comparators.
#[cfg(unix)]
#[path = "support/executable.rs"]
mod executable;
use invar::{
    check, eval, front,
    general_net::{
        compiled,
        numeric_owned_host::{Error, Plan},
        oracle,
    },
    numeric_contract::F32Bits,
};
use invar_foreign::numeric_owned::{self as numeric, Carrier, Driver};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};
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
fn compare(p: &mut Plan, args: &[eval::Value], l: oracle::Limits) -> oracle::Observation {
    let (net, proof) = p.artifact().checked_net();
    let entry = p.artifact().manifest()["entry"]
        .as_str()
        .unwrap()
        .to_owned();
    let expected = oracle::observe_call(net, proof, &entry, args, l).unwrap();
    let checked = check::check(front::parse(p.artifact().source()).unwrap(), false).unwrap();
    let source = eval::named::observe_numeric_named(&checked, &entry, args, l.fuel);
    let actual = p
        .run(args, l)
        .unwrap_or_else(|e| panic!("{e:?}; evidence {}", p.evidence_directory().display()));
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
    // The named source API deliberately admits scalar roots only. Array-root
    // cases use the independent owned oracle plus hand-derived bit assertions;
    // the fixture corpus separately compares scalar source observations.
    let array_root = matches!(
        actual.semantic.result,
        Ok(eval::Value::ArrayI32(_) | eval::Value::ArrayU32(_) | eval::Value::ArrayF32(_))
    );
    if expected.failure.is_none() && !array_root {
        equal(&actual.semantic.result, &source.result);
        assert_eq!(actual.semantic.steps_used, source.steps_used);
        assert_eq!(actual.semantic.cleanup_trace, source.cleanup_trace);
    }
    actual
}
#[test]
fn issue_gnet_10_loaded_source_all_fixture_fuel_prefixes() {
    for name in [
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
    ] {
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
    }
}
#[test]
fn issue_gnet_10_loaded_roots_are_fresh_and_preserve_bits() {
    let s = "fn run(x:f32)->Array<f32>{let a:Array<f32> = [x,-0.0,0.0/0.0];return a;}";
    let mut p = plan(s, "run", 128);
    for bits in [0, 1, 0x80000000, 0x7f800000, 0x7fa00001, 0xffc12345] {
        let args = [eval::Value::F32(F32Bits::from_bits(bits))];
        let a = compare(&mut p, &args, limits(128));
        let b = compare(&mut p, &args, limits(128));
        let (Ok(eval::Value::ArrayF32(a)), Ok(eval::Value::ArrayF32(b))) =
            (a.semantic.result, b.semantic.result)
        else {
            panic!()
        };
        assert!(!Rc::ptr_eq(&a, &b));
        assert_eq!(a.borrow()[0].to_bits(), bits);
        a.borrow_mut()[0] = F32Bits::from_bits(0x3f800000);
        assert_eq!(b.borrow()[0].to_bits(), bits);
    }
    for s in [
        "fn run()->Array<i32>{let a:Array<i32> = [-1,2];return a;}",
        "fn run()->Array<u32>{return filled_u32(0u32,0u32);}",
        "fn run()->Array<i32>{return filled_i32(0u32,0);}",
        "fn run()->Array<u32>{let a:Array<u32> = [0u32,4294967295u32];return a;}",
        "fn run()->Array<f32>{return [1.0,-0.0];}",
        "fn run()->Array<i32>{return [-2147483648,2147483647];}",
        "fn run()->Array<f32>{return filled_f32(0u32,-0.0);}",
    ] {
        compare(&mut plan(s, "run", 128), &[], limits(128));
    }
}
#[test]
fn issue_gnet_10_loaded_failure_budgets_and_typed_refusals() {
    let s = "fn run(x:f32)->f32{let a:Array<f32> = [x,1.0];return a[0u32];}";
    let mut p = plan(s, "run", 128);
    let args = [eval::Value::F32(F32Bits::from_bits(1))];
    for cap in [0, 1, 2, 3, 7] {
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
    assert!(matches!(p.run(&[], limits(128)), Err(Error::Input)));
    assert!(matches!(
        p.run(
            &args,
            oracle::Limits {
                fuel: 0,
                ..limits(128)
            }
        ),
        Err(Error::Limit)
    ));
    assert!(matches!(p.run(&args, limits(1)), Err(Error::Limit)));
    compare(&mut p, &args, limits(128));
    for (allocations, cleanup_events, status) in [(1, 128, 12), (128, 0, 13)] {
        let s = "fn run()->i32{let a:Array<f32> = [1.0];let b:Array<f32> = [2.0];return 7;}";
        let mut p = Plan::from_source(
            s.as_bytes(),
            "run",
            Default::default(),
            compiled::OwnedLimits {
                elements: 128,
                allocations,
                cleanup_events,
            },
            Path::new("c++"),
        )
        .unwrap();
        let got = p.run(&[], limits(128)).unwrap();
        assert_eq!(got.failure, Some(oracle::ToolFailure::Allocation));
        assert!(
            matches!(
                got.semantic.result,
                Err(eval::EvalError::Tool {
                    code: "RESOURCE_LIMIT",
                    ..
                })
            ),
            "{status}"
        );
    }
}
#[test]
fn issue_gnet_10_loaded_artifacts_are_isolated_and_unload_cleanly() {
    let mut a = plan(
        "fn run()->Array<f32>{return filled_f32(1u32,1.0);}",
        "run",
        128,
    );
    let mut b = plan(
        "fn run()->Array<f32>{return filled_f32(1u32,2.0);}",
        "run",
        128,
    );
    assert_ne!(a.artifact().id(), b.artifact().id());
    let path = a.evidence_directory().to_path_buf();
    compare(&mut a, &[], limits(128));
    compare(&mut b, &[], limits(128));
    drop(a);
    assert!(!path.exists());
    compare(&mut b, &[], limits(128));
    let error = Plan::from_source(
        b"fn run()->i32{return 7;}",
        "run",
        Default::default(),
        storage(128),
        Path::new("/bin/false"),
    );
    let Err(Error::Compile { evidence, .. }) = error else {
        panic!("compiler failure must retain evidence")
    };
    assert!(evidence.join("compiler-version.stdout").is_file());
}

struct Fixture {
    dir: PathBuf,
    path: PathBuf,
    hash: [u8; 32],
}
impl Fixture {
    fn new(body: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "gnet10-loader-fixture-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let path = dir.join(format!("fixture.{}", std::env::consts::DLL_EXTENSION));
        fs::write(dir.join("fixture.cpp"),format!(r#"#include <cstddef>
extern "C" unsigned int invar_fixture_identity(unsigned char* out,std::size_t n){{if(n!=32)return 2;for(unsigned i=0;i<32;++i)out[i]=17;return 0;}}
extern "C" unsigned int invar_fixture(const long long* in,std::size_t n,long long* out,std::size_t on,unsigned long long* trace,std::size_t tn,long long* heap,std::size_t hn,unsigned long long* cleanup,std::size_t cn,unsigned long long fuel,unsigned long long trans,unsigned long long cap){{(void)in;(void)n;(void)out;(void)on;(void)trace;(void)tn;(void)heap;(void)hn;(void)cleanup;(void)cn;(void)fuel;(void)trans;(void)cap;{body}}}"#)).unwrap();
        let out = Command::new("c++")
            .args([
                "-std=c++17",
                "-fPIC",
                if cfg!(target_os = "macos") {
                    "-dynamiclib"
                } else {
                    "-shared"
                },
            ])
            .arg(dir.join("fixture.cpp"))
            .arg("-o")
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let hash = numeric::file_digest(&path).unwrap();
        Self { dir, path, hash }
    }
    fn load(&self) -> Driver {
        Driver::load(
            &self.path,
            self.hash,
            [17; 32],
            "invar_fixture",
            vec![Carrier::Word],
            [4, 0],
        )
        .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}
#[test]
fn issue_gnet_10_numeric_host_reviewed_loader_identity_symbols_carriers_and_returning_failures() {
    let fixture = Fixture::new("return 3;");
    let mut bad = fixture.hash;
    bad[0] ^= 1;
    assert!(matches!(
        Driver::load(
            &fixture.path,
            bad,
            [17; 32],
            "invar_fixture",
            vec![],
            [4, 0]
        ),
        Err(numeric::Error::Identity)
    ));
    assert!(matches!(
        Driver::load(
            &fixture.path,
            fixture.hash,
            [18; 32],
            "invar_fixture",
            vec![],
            [4, 0]
        ),
        Err(numeric::Error::Identity)
    ));
    assert!(matches!(
        Driver::load(
            &fixture.path,
            fixture.hash,
            [17; 32],
            "invar_missing",
            vec![],
            [4, 0]
        ),
        Err(numeric::Error::Symbol(_))
    ));
    let mut driver = fixture.load();
    for input in [vec![-1], vec![u32::MAX as i64 + 1], vec![], vec![0, 1]] {
        assert!(matches!(
            driver.run(&input, [1, 1, 1]),
            Err(numeric::Error::Limit)
        ));
    }
    for _ in 0..3 {
        assert!(matches!(
            driver.run(&[1], [1, 1, 1]),
            Err(numeric::Error::Protocol(3))
        ));
    }
    let changed = fixture.dir.join("changed.so");
    fs::copy(&fixture.path, &changed).unwrap();
    fs::write(&changed, b"forged library").unwrap();
    assert!(matches!(
        Driver::load(
            &changed,
            fixture.hash,
            [17; 32],
            "invar_fixture",
            vec![],
            [4, 0]
        ),
        Err(numeric::Error::Identity)
    ));
}
#[cfg(unix)]
#[test]
fn issue_gnet_10_numeric_host_failed_compiler_preserves_original_artifacts_and_logs() {
    let script = executable::fixture("abi10-diagnostic.sh");
    let Err(Error::Compile { message, evidence }) = Plan::from_source(
        b"fn run()->f32{return 0.0;}",
        "run",
        Default::default(),
        storage(128),
        &script,
    ) else {
        panic!("expected compiler failure")
    };
    assert!(message.contains("ORIGINAL_ABI10_COMPILER_DIAGNOSTIC"));
    assert!(evidence.join("numeric.cpp").is_file());
    assert_eq!(
        fs::read_to_string(evidence.join("compiler.stderr")).unwrap(),
        "ORIGINAL_ABI10_COMPILER_DIAGNOSTIC"
    );
    // Preserve original failed-attempt artifacts.
}

#[test]
fn issue_gnet_10_numeric_host_loaded_recursion_call_depth_returns_and_reuses() {
    let mut p = plan("fn run(x:f32)->f32{return run(x);}", "run", 128);
    for bits in [0x80000000, 1, 0xffc12345] {
        let o = compare(
            &mut p,
            &[eval::Value::F32(F32Bits::from_bits(bits))],
            limits(128),
        );
        assert!(matches!(
            o.semantic.result,
            Err(eval::EvalError::Tool {
                code: "CALL_DEPTH_LIMIT",
                ..
            })
        ));
        assert_eq!(o.max_frames, 64);
    }
}

#[test]
fn issue_gnet_10_numeric_host_library_size_and_loader_failure_preserve_evidence() {
    let fixture = Fixture::new("return 0;");
    let sparse = fixture.dir.join("oversize.so");
    fs::File::create(&sparse)
        .unwrap()
        .set_len(64 * 1024 * 1024 + 1)
        .unwrap();
    assert!(matches!(
        numeric::file_digest(&sparse),
        Err(numeric::Error::Limit)
    ));
    #[cfg(unix)]
    {
        let script = executable::fixture("bad-library.sh");
        let Err(Error::Load {
            cause: numeric::Error::Library(message),
            evidence,
        }) = Plan::from_source(
            b"fn run()->f32{return 0.0;}",
            "run",
            Default::default(),
            storage(128),
            &script,
        )
        else {
            panic!("expected returning load error")
        };
        assert!(!message.is_empty());
        assert!(evidence.join("numeric.cpp").exists());
        assert!(evidence.join("receipt.json").exists());
        // Preserve original failed-attempt artifacts.
    }
}

#[test]
fn issue_gnet_10_loaded_heap_exhaustion_and_source_limits() {
    let source =
        b"fn run()->i32{let a:Array<f32> = [1.0,2.0];let b:Array<f32> = [3.0,4.0];return 7;}";
    let mut p = Plan::from_source(
        source,
        "run",
        Default::default(),
        storage(2),
        Path::new("c++"),
    )
    .unwrap();
    for _ in 0..2 {
        let o = p.run(&[], limits(2)).unwrap();
        assert_eq!(o.failure, Some(oracle::ToolFailure::Allocation));
        assert!(matches!(
            o.semantic.result,
            Err(eval::EvalError::Tool {
                code: "RESOURCE_LIMIT",
                ..
            })
        ));
    }
    for which in 0..8 {
        let mut l = invar::general_net::provenance::Limits::default();
        match which {
            0 => l.compiler.values = 0,
            1 => l.compiler.places = 0,
            2 => l.compiler.states = 0,
            3 => l.compiler.source_bytes = 1,
            4 => l.net.functions = 0,
            5 => l.net.blocks = 0,
            6 => l.net.instructions = 0,
            7 => l.net.wires = 0,
            _ => unreachable!(),
        }
        assert!(
            matches!(
                Plan::from_source(
                    source,
                    "run",
                    l,
                    storage(128),
                    Path::new("nonexistent-compiler")
                ),
                Err(Error::Provenance(_))
            ),
            "budget {which}"
        );
    }
    for source in [
        "fn run(xs:read Array<f32>)->f32{return xs[0u32];}",
        "fn run(xs:take Array<i32>)->i32{return xs[0u32];}",
    ] {
        assert!(matches!(
            Plan::from_source(
                source.as_bytes(),
                "run",
                Default::default(),
                storage(128),
                Path::new("nonexistent-compiler")
            ),
            Err(Error::Provenance(_))
        ));
    }
}
