#![forbid(unsafe_code)]
#[cfg(unix)]
#[path = "support/executable.rs"]
mod executable;
use invar::{
    check, eval, front,
    general_net::{
        numeric_host::{Error, Plan},
        oracle, provenance,
    },
    mir, mir_eval,
    numeric_contract::F32Bits,
};
use invar_foreign::numeric::{self, Carrier, Driver};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
fn limits() -> oracle::Limits {
    oracle::Limits {
        fuel: 10000,
        transitions: 10000,
        trace_events: 10000,
        ..Default::default()
    }
}
fn plan(source: &str) -> Plan {
    Plan::from_source(
        source.as_bytes(),
        "run",
        Default::default(),
        Path::new("c++"),
    )
    .unwrap()
}
fn f(bits: u32) -> eval::Value {
    eval::Value::F32(F32Bits::from_bits(bits))
}
fn same_result(a: &Result<eval::Value, eval::EvalError>, b: &Result<eval::Value, eval::EvalError>) {
    match (a, b) {
        (Ok(eval::Value::F32(a)), Ok(eval::Value::F32(b))) => {
            assert!(a.strict_eq(*b), "{a:?} != {b:?}")
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
    let expected = oracle::observe_call(
        net,
        proof,
        p.artifact().manifest()["entry"].as_str().unwrap(),
        args,
        l,
    )
    .unwrap();
    let actual = p.run(args, l).unwrap_or_else(|e| {
        panic!(
            "loaded result {e:?}, limits {l:?}, evidence {}",
            p.evidence_directory().display()
        )
    });
    same_result(&actual.semantic.result, &expected.semantic.result);
    assert_eq!(actual.semantic.steps_used, expected.semantic.steps_used);
    assert_eq!(
        actual.semantic.cleanup_trace,
        expected.semantic.cleanup_trace
    );
    assert_eq!(actual.failure, expected.failure);
    assert_eq!(actual.events, expected.events);
    assert_eq!(actual.transitions, expected.transitions);
    assert_eq!(actual.max_frames, expected.max_frames);
    actual
}
#[test]
fn issue_gnet_08_numeric_host_runtime_word_classes_copies_arithmetic_and_conversion() {
    let bits = [
        0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x3f800001, 0x3f7ffffe, 0xbf800000, 0x7f7fffff,
        0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0xffc12345,
    ];
    let mut copy = plan("fn run(x:f32)->f32{return x;}");
    for x in bits {
        let got = compare(&mut copy, &[f(x)], limits());
        assert_eq!(got.semantic.result.unwrap(), f(x));
    }
    for op in ["+", "-", "*", "/", "<", "==", "!="] {
        let result = if ["<", "==", "!="].contains(&op) {
            "Bool"
        } else {
            "f32"
        };
        let mut p = plan(&format!("fn run(a:f32,b:f32)->{result}{{return a{op}b;}}"));
        for (i, a) in bits.iter().enumerate() {
            compare(
                &mut p,
                &[f(*a), f(bits[(i * 7 + 3) % bits.len()])],
                limits(),
            );
        }
    }
    let mut p = plan("fn run(a:f32,b:f32,c:f32)->f32{return fma(a,b,c);}");
    for xs in [
        [0x3f800001, 0x3f7ffffe, 0xbf800000],
        [1, 0x3f000000, 1],
        [0x7f7fffff, 0x40000000, 0xff7fffff],
        [0x7f800000, 0, 0x3f800000],
    ] {
        compare(&mut p, &xs.map(f), limits());
    }
    for (ty, intrinsic) in [("i32", "f32_from_i32"), ("u32", "f32_from_u32")] {
        let mut p = plan(&format!("fn run(x:{ty})->f32{{return {intrinsic}(x);}}"));
        let args = if ty == "i32" {
            vec![
                eval::Value::I32(i32::MIN),
                eval::Value::I32(-16777217),
                eval::Value::I32(i32::MAX),
            ]
        } else {
            vec![eval::Value::U32(16777217), eval::Value::U32(u32::MAX)]
        };
        for a in args {
            compare(&mut p, &[a], limits());
        }
    }
}
#[test]
fn issue_gnet_08_numeric_host_source_mir_net_loaded_every_fuel_and_event_prefix() {
    let sources = [
        "fn step(x:f32,n:u32)->f32{if n==0u32{return x;}return step(x+0.000000059604644775390625,n-1u32);}fn run()->i32{let n=step(1.0,3u32);if n==1.0{return 7;}return 0;}",
        "fn run()->i32{var x=0.0;for i in 0u32..3u32{x=x+f32_from_u32(i);}if x==3.0{return 7;}return 0;}",
        "fn run()->i32{let x=fma(1.00000011920928955078125,0.99999988079071044921875,-1.0);if x<0.0{return 7;}return 0;}",
        "fn run()->i32{let x=1/0;let later=1.0/0.0;return x;}",
        "fn run()->i32{let x=2147483647+1;let later=0.0/0.0;return x;}",
        "fn run()->i32{if false{let dead=1/0;}let x=0.0/0.0;if x==x{return 0;}return 7;}",
    ];
    let mut count = 0;
    for source in sources {
        let main_source = source.replace("fn run()", "fn main()");
        let mut p = Plan::from_source(
            main_source.as_bytes(),
            "main",
            Default::default(),
            Path::new("c++"),
        )
        .unwrap();
        let done = compare(&mut p, &[], limits());
        let c = check::check(front::parse(main_source.as_bytes()).unwrap(), true).unwrap();
        let m = mir::lower(&c).unwrap();
        for fuel in 1..=done.semantic.steps_used + 1 {
            let l = oracle::Limits { fuel, ..limits() };
            let got = compare(&mut p, &[], l);
            let source = eval::observe(&c, fuel);
            let mid = mir_eval::observe(&m, fuel);
            same_result(&source.result, &mid.result);
            same_result(&got.semantic.result, &source.result);
            assert_eq!(source.steps_used, mid.steps_used);
            assert_eq!(got.semantic.steps_used, source.steps_used);
            assert_eq!(got.semantic.cleanup_trace, source.cleanup_trace);
            count += 1;
        }
        for transitions in 0..=done.transitions + 1 {
            compare(
                &mut p,
                &[],
                oracle::Limits {
                    transitions,
                    ..limits()
                },
            );
            count += 1;
        }
        for trace_events in 0..=done.events.len() + 1 {
            compare(
                &mut p,
                &[],
                oracle::Limits {
                    trace_events,
                    ..limits()
                },
            );
            count += 1;
        }
    }
    assert!(count > 100);
    eprintln!("actual loaded prefix comparisons={count}");
}
#[test]
fn issue_gnet_08_numeric_host_wrong_arguments_and_private_limits_before_call() {
    let mut p = plan("fn run(x:f32)->f32{return x;}");
    for args in [vec![], vec![eval::Value::I32(0)], vec![f(0), f(1)]] {
        assert!(matches!(p.run(&args, limits()), Err(Error::Input)));
    }
    for l in [
        oracle::Limits {
            fuel: 0,
            ..limits()
        },
        oracle::Limits {
            fuel: 1000001,
            ..limits()
        },
        oracle::Limits {
            transitions: 1000001,
            ..limits()
        },
        oracle::Limits {
            trace_events: 1000001,
            ..limits()
        },
    ] {
        assert!(matches!(p.run(&[f(1)], l), Err(Error::Limit)));
    }
    assert_eq!(
        p.run(&[f(1)], limits()).unwrap().semantic.result.unwrap(),
        f(1)
    );
    let mut l = provenance::Limits::default();
    l.compiler.frame_bytes = 128 * 1024 + 1;
    assert!(matches!(
        Plan::from_source(
            b"fn run()->f32{return 0.0;}",
            "run",
            l,
            Path::new("unavailable")
        ),
        Err(Error::Limit)
    ));
    l.compiler.frame_bytes = 0;
    assert!(matches!(
        Plan::from_source(b"fn run()->f32{return 0.0;}", "run", l, Path::new("c++")),
        Err(Error::Provenance(_))
    ));
    for source in [
        "fn run()->Array<f32>{return filled_f32(1u32,0.0);}",
        "struct X{x:f32}fn run()->i32{return 0;}",
    ] {
        assert!(matches!(
            Plan::from_source(
                source.as_bytes(),
                "run",
                Default::default(),
                Path::new("c++")
            ),
            Err(Error::Provenance(_))
        ));
    }
}
#[test]
fn issue_gnet_08_numeric_host_two_loaded_artifacts_library_lifetime_and_receipts() {
    let mut a = plan("fn run(x:f32)->f32{return x;}");
    let mut b = plan("fn run(x:f32)->f32{return -x;}");
    assert_ne!(a.receipt()["identity"], b.receipt()["identity"]);
    assert_ne!(a.receipt()["library_sha256"], b.receipt()["library_sha256"]);
    assert_eq!(a.artifact().artifact().abi_version(), 8);
    assert_eq!(
        a.artifact().manifest()["numeric"]["semantic_id"],
        "strict-f32-v1"
    );
    for _ in 0..4 {
        assert_eq!(
            compare(&mut a, &[f(0xffc12345)], limits())
                .semantic
                .result
                .unwrap(),
            f(0xffc12345)
        );
        assert_eq!(
            compare(&mut b, &[f(0x80000000)], limits())
                .semantic
                .result
                .unwrap(),
            f(0)
        );
    }
    let dir = a.evidence_directory().to_owned();
    let library = dir.join(format!("numeric.{}", std::env::consts::DLL_EXTENSION));
    #[cfg(unix)]
    {
        fs::remove_file(&library).unwrap();
        assert_eq!(
            a.run(&[f(1)], limits()).unwrap().semantic.result.unwrap(),
            f(1)
        );
    }
    drop(a);
    assert!(!dir.exists());
    assert_eq!(
        b.run(&[f(0)], limits()).unwrap().semantic.result.unwrap(),
        f(0x80000000)
    );
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
            "gnet8-loader-fixture-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let path = dir.join(format!("fixture.{}", std::env::consts::DLL_EXTENSION));
        fs::write(dir.join("fixture.cpp"),format!(r#"#include <cstddef>
extern "C" unsigned int invar_fixture_identity(unsigned char* out,std::size_t n){{if(n!=32)return 2;for(unsigned i=0;i<32;++i)out[i]=17;return 0;}}
extern "C" unsigned int invar_fixture(const long long* in,std::size_t n,long long* out,std::size_t on,unsigned long long* trace,std::size_t tn,unsigned long long fuel,unsigned long long trans,unsigned long long cap){{(void)in;(void)n;(void)out;(void)on;(void)trace;(void)tn;(void)fuel;(void)trans;(void)cap;{body}}}"#)).unwrap();
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
fn issue_gnet_08_numeric_host_reviewed_loader_identity_symbols_carriers_and_returning_failures() {
    let fixture = Fixture::new("return 3;");
    let mut bad = fixture.hash;
    bad[0] ^= 1;
    assert!(matches!(
        Driver::load(&fixture.path, bad, [17; 32], "invar_fixture", vec![]),
        Err(numeric::Error::Identity)
    ));
    assert!(matches!(
        Driver::load(
            &fixture.path,
            fixture.hash,
            [18; 32],
            "invar_fixture",
            vec![]
        ),
        Err(numeric::Error::Identity)
    ));
    assert!(matches!(
        Driver::load(
            &fixture.path,
            fixture.hash,
            [17; 32],
            "invar_missing",
            vec![]
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
        Driver::load(&changed, fixture.hash, [17; 32], "invar_fixture", vec![]),
        Err(numeric::Error::Identity)
    ));
}
#[cfg(unix)]
#[test]
fn issue_gnet_08_numeric_host_failed_compiler_preserves_original_artifacts_and_logs() {
    let script = executable::fixture("abi8-diagnostic.sh");
    let Err(Error::Compile { message, evidence }) = Plan::from_source(
        b"fn run()->f32{return 0.0;}",
        "run",
        Default::default(),
        &script,
    ) else {
        panic!("expected compiler failure")
    };
    assert!(message.contains("ORIGINAL_ABI8_COMPILER_DIAGNOSTIC"));
    assert!(evidence.join("numeric.cpp").is_file());
    assert_eq!(
        fs::read_to_string(evidence.join("compiler.stderr")).unwrap(),
        "ORIGINAL_ABI8_COMPILER_DIAGNOSTIC"
    );
    fs::remove_dir_all(evidence).unwrap();
}

#[test]
fn issue_gnet_08_numeric_host_loaded_recursion_call_depth_returns_and_reuses() {
    let mut p = plan("fn run(x:f32)->f32{return run(x);}");
    for bits in [0x80000000, 1, 0xffc12345] {
        let o = compare(&mut p, &[f(bits)], limits());
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
fn issue_gnet_08_numeric_host_library_size_and_loader_failure_preserve_evidence() {
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
            &script,
        )
        else {
            panic!("expected returning load error")
        };
        assert!(!message.is_empty());
        assert!(evidence.join("numeric.cpp").exists());
        assert!(evidence.join("receipt.json").exists());
        fs::remove_dir_all(evidence).unwrap();
    }
}

#[test]
fn issue_gnet_08_numeric_host_all_source_compile_budgets_and_private_scalar_carriers() {
    let source = b"fn run(x:f32)->f32{return x;}";
    for which in 0..9 {
        let mut l = provenance::Limits::default();
        match which {
            0 => l.compiler.values = 0,
            1 => l.compiler.places = 0,
            2 => l.compiler.states = 0,
            3 => l.compiler.source_bytes = 1,
            4 => l.net.functions = 0,
            5 => l.net.blocks = 0,
            6 => l.net.instructions = 0,
            7 => l.net.wires = 0,
            8 => l.net.literal_bytes = 0,
            _ => unreachable!(),
        }
        assert!(
            matches!(
                Plan::from_source(
                    if which == 8 {
                        br#"fn run()->i32{let x=b"z";return 0;}"#
                    } else {
                        source
                    },
                    "run",
                    l,
                    Path::new("unavailable")
                ),
                Err(Error::Provenance(_))
            ),
            "compile budget {which}"
        );
    }
    assert!(matches!(
        Plan::from_source(
            &vec![b' '; 1024 * 1024 + 1],
            "run",
            Default::default(),
            Path::new("unavailable")
        ),
        Err(Error::Provenance(provenance::Error::Limit))
    ));
    assert!(matches!(
        Plan::from_source(source, "", Default::default(), Path::new("unavailable")),
        Err(Error::Provenance(provenance::Error::Limit))
    ));
    let params = (0..129)
        .map(|n| format!("x{n}:Bool"))
        .collect::<Vec<_>>()
        .join(",");
    let source = format!("fn run({params})->i32{{return 0;}}");
    let mut l = provenance::Limits::default();
    l.compiler.places = 129;
    assert!(matches!(
        Plan::from_source(source.as_bytes(), "run", l, Path::new("unavailable")),
        Err(Error::Limit)
    ));
    for (source, expected) in [
        (
            "fn run()->i32{return -2147483648;}",
            eval::Value::I32(i32::MIN),
        ),
        (
            "fn run()->u32{return 4294967295u32;}",
            eval::Value::U32(u32::MAX),
        ),
        ("fn run()->Bool{return true;}", eval::Value::Bool(true)),
        ("fn run()->Unit{return;}", eval::Value::Unit),
    ] {
        let mut p = plan(source);
        assert_eq!(
            compare(&mut p, &[], limits()).semantic.result.unwrap(),
            expected
        );
    }
    let fixture = Fixture::new("return 3;");
    for (carrier, bad) in [
        (Carrier::I32, i32::MIN as i64 - 1),
        (Carrier::I32, i32::MAX as i64 + 1),
        (Carrier::Bool, 2),
        (Carrier::Bool, -1),
    ] {
        let mut driver = Driver::load(
            &fixture.path,
            fixture.hash,
            [17; 32],
            "invar_fixture",
            vec![carrier],
        )
        .unwrap();
        assert!(matches!(
            driver.run(&[bad], [1, 1, 1]),
            Err(numeric::Error::Limit)
        ));
    }
}
