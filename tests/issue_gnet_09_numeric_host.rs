#![forbid(unsafe_code)]
#[cfg(unix)]
#[path = "support/executable.rs"]
mod executable;
use invar::{
    check, eval, front,
    general_net::{
        numeric_resume_host::{Error, Plan, Progress},
        oracle, provenance,
    },
    mir, mir_eval,
    numeric_contract::F32Bits,
};
use invar_foreign::{numeric, numeric_resume::Driver};
use std::{fs, path::Path, process::Command, rc::Rc};
fn limits() -> oracle::Limits {
    oracle::Limits {
        fuel: 10000,
        transitions: 10000,
        trace_events: 1000,
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
    let checked = check::check(front::parse(p.artifact().source()).unwrap(), false).unwrap();
    let reference = eval::named::observe_numeric_named(
        &checked,
        p.artifact().manifest()["entry"].as_str().unwrap(),
        args,
        l.fuel,
    );
    let mut actual = None;
    for quantum in [1, 3, 17] {
        let got = execute(p, args, l, quantum).unwrap_or_else(|e| {
            panic!(
                "loaded result {e:?}, limits {l:?}, evidence {}",
                p.evidence_directory().display()
            )
        });
        same_result(&got.semantic.result, &expected.semantic.result);
        assert_eq!(got.semantic.steps_used, expected.semantic.steps_used);
        assert_eq!(got.events, expected.events);
        assert_eq!(got.transitions, expected.transitions);
        assert_eq!(got.failure, expected.failure);
        assert_eq!(got.max_frames, expected.max_frames);
        actual = Some(got);
    }
    let actual = actual.unwrap();
    if expected.failure.is_none() {
        same_result(&actual.semantic.result, &reference.result);
        assert_eq!(actual.semantic.steps_used, reference.steps_used);
        assert_eq!(actual.semantic.cleanup_trace, reference.cleanup_trace);
    }
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
fn execute(
    p: &mut Plan,
    args: &[eval::Value],
    l: oracle::Limits,
    quantum: u64,
) -> Result<oracle::Observation, Error> {
    let mut run = p.start(args, l)?;
    let observation = loop {
        match run.resume(quantum)? {
            Progress::Pending(p) => {
                assert!(p.depth > 0);
            }
            Progress::Complete(o) => break o,
        }
    };
    let replay = match run.resume(17)? {
        Progress::Complete(o) => o,
        _ => panic!("terminal replay yielded"),
    };
    assert!(Rc::ptr_eq(&observation, &replay));
    drop(replay);
    drop(run);
    Ok(Rc::try_unwrap(observation).unwrap())
}
trait Execute {
    fn run(
        &mut self,
        args: &[eval::Value],
        l: oracle::Limits,
    ) -> Result<oracle::Observation, Error>;
}
impl Execute for Plan {
    fn run(
        &mut self,
        args: &[eval::Value],
        l: oracle::Limits,
    ) -> Result<oracle::Observation, Error> {
        execute(self, args, l, 3)
    }
}
#[test]
fn issue_gnet_09_numeric_host_runtime_word_classes_copies_arithmetic_and_conversion() {
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
fn issue_gnet_09_numeric_host_source_mir_net_loaded_every_fuel_and_event_prefix() {
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
fn issue_gnet_09_numeric_host_wrong_arguments_and_private_limits_before_call() {
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
fn issue_gnet_09_numeric_host_two_loaded_artifacts_library_lifetime_and_receipts() {
    let mut a = plan("fn run(x:f32)->f32{return x;}");
    let mut b = plan("fn run(x:f32)->f32{return -x;}");
    assert_ne!(a.receipt()["identity"], b.receipt()["identity"]);
    assert_ne!(a.receipt()["library_sha256"], b.receipt()["library_sha256"]);
    assert_eq!(a.artifact().artifact().abi_version(), 9);
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
    #[cfg(target_os = "linux")]
    {
        let symbols = Command::new("nm")
            .args(["-D", "--defined-only"])
            .arg(&library)
            .output()
            .unwrap();
        assert!(symbols.status.success());
        let defined = String::from_utf8(symbols.stdout).unwrap();
        let mut actual: Vec<_> = defined
            .lines()
            .map(|l| l.split_whitespace().last().unwrap().to_owned())
            .collect();
        actual.sort();
        let symbol = a.receipt()["symbol"].as_str().unwrap();
        let mut expected: Vec<_> = ["capture", "create", "destroy", "identity", "resume"]
            .map(|s| format!("{symbol}_{s}"))
            .to_vec();
        expected.sort();
        assert_eq!(actual, expected, "private helpers exported: {defined}");
    }
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
#[test]
fn issue_gnet_09_numeric_host_loaded_recursion_call_depth_returns_and_reuses() {
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
fn issue_gnet_09_numeric_host_all_source_compile_budgets_and_private_scalar_carriers() {
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
    let fixture = LoaderFixture::new(3);
    for (carrier, bad) in [
        (numeric::Carrier::I32, i32::MIN as i64 - 1),
        (numeric::Carrier::I32, i32::MAX as i64 + 1),
        (numeric::Carrier::Bool, 2),
        (numeric::Carrier::Bool, -1),
    ] {
        let mut driver = Driver::load(
            &fixture.path,
            fixture.hash,
            [17; 32],
            "invar_fixture",
            vec![carrier],
            64 * 48 + 144,
        )
        .unwrap();
        assert!(matches!(
            driver.start(&[bad], [1, 1, 1]),
            Err(numeric::Error::Limit)
        ));
    }
}

#[test]
fn issue_gnet_09_numeric_host_captured_inputs_independent_runs_and_invalid_quanta() {
    let mut a = plan("fn run(x:f32,n:u32)->f32{var y=x;for i in 0u32..n{y=y+0.0;}return y;}");
    let mut b = plan("fn run(x:f32)->f32{return -x;}");
    let mut args = vec![f(0x80000000), eval::Value::U32(4)];
    let mut ra = a.start(&args, limits()).unwrap();
    let mut rb = b.start(&[f(0x80000000)], limits()).unwrap();
    let Progress::Pending(first) = ra.resume(1).unwrap() else {
        panic!("expected yield")
    };
    args[0] = f(0x7fc12345);
    args[1] = eval::Value::U32(0);
    drop(args);
    for q in [0, 1_000_001, u64::MAX] {
        assert!(matches!(ra.resume(q), Err(Error::Limit)));
    }
    let Progress::Pending(next) = ra.resume(1).unwrap() else {
        panic!("expected second yield")
    };
    assert_eq!(next.transitions, first.transitions + 1);
    let ob = loop {
        if let Progress::Complete(o) = rb.resume(3).unwrap() {
            break o;
        }
    };
    assert_eq!(ob.semantic.result, Ok(f(0)));
    let oa = loop {
        if let Progress::Complete(o) = ra.resume(17).unwrap() {
            break o;
        }
    };
    assert_eq!(oa.semantic.result, Ok(f(0)));
    assert!(matches!(ra.resume(0), Err(Error::Limit)));
    let Progress::Complete(replay) = ra.resume(1).unwrap() else {
        panic!("expected terminal replay")
    };
    assert!(Rc::ptr_eq(&oa, &replay));
    drop(ra);
    drop(rb);
    // Dropping a yielded owner also allows a fresh independent task in its plan.
    let mut yielded = a.start(&[f(1), eval::Value::U32(3)], limits()).unwrap();
    assert!(matches!(yielded.resume(1).unwrap(), Progress::Pending(_)));
    drop(yielded);
    assert_eq!(
        a.run(&[f(0x7f800001), eval::Value::U32(0)], limits())
            .unwrap()
            .semantic
            .result,
        Ok(f(0x7f800001))
    );
}

#[test]
fn issue_gnet_09_numeric_host_named_runtime_fma_loop_recursion_and_separate_rounding() {
    let source = "fn recur(a:f32,b:f32,c:f32,n:u32)->f32{if n==0u32{return a;}return recur(fma(a,b,c),b,c,n-1u32);}fn run(a:f32,b:f32,c:f32,n:u32)->f32{var x=a;for i in 0u32..n{x=recur(x,b,c,2u32);}return x;}";
    let mut p = plan(source);
    for xs in [
        [0x7f7fffff, 0x40000000, 0xff7fffff, 1],
        [0x80000000, 0x3f800000, 0x80000000, 2],
        [0x7f800001, 0x3f800001, 1, 3],
    ] {
        compare(
            &mut p,
            &[f(xs[0]), f(xs[1]), f(xs[2]), eval::Value::U32(xs[3])],
            limits(),
        );
    }
    let mut fused = plan("fn run(a:f32,b:f32,c:f32)->f32{return fma(a,b,c);}");
    let mut separate = plan("fn run(a:f32,b:f32,c:f32)->f32{return a*b+c;}");
    let args = [f(0x3f800001), f(0x3f7ffffe), f(0xbf800000)];
    let fused = compare(&mut fused, &args, limits());
    let separate = compare(&mut separate, &args, limits());
    assert_ne!(fused.semantic.result, separate.semantic.result);
}

struct LoaderFixture {
    dir: std::path::PathBuf,
    path: std::path::PathBuf,
    hash: [u8; 32],
}
impl LoaderFixture {
    fn new(code: u32) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "gnet9-loader-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let path = dir.join(format!("fixture.{}", std::env::consts::DLL_EXTENSION));
        fs::write(dir.join("fixture.cpp"),format!(r#"#include <cstddef>
extern "C" unsigned int invar_fixture_identity(unsigned char* out,std::size_t n){{if(n!=32)return 2;for(unsigned i=0;i<32;++i)out[i]=17;return 0;}}
extern "C" unsigned int invar_fixture_create(const long long*,std::size_t,unsigned long long,unsigned long long,unsigned long long,void** p){{*p=nullptr;return {code};}}
extern "C" unsigned int invar_fixture_resume(void*,unsigned long long){{return 2;}}
extern "C" unsigned int invar_fixture_capture(void*,long long*,std::size_t,long long*,std::size_t,unsigned long long*,std::size_t,unsigned char*,std::size_t,unsigned long long*,std::size_t,unsigned long long*){{return 2;}}
extern "C" unsigned int invar_fixture_destroy(void*){{return 2;}}
"#)).unwrap();
        let output = Command::new("c++")
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
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let hash = numeric::file_digest(&path).unwrap();
        Self { dir, path, hash }
    }
    fn load(
        &self,
        hash: [u8; 32],
        identity: [u8; 32],
        symbol: &str,
    ) -> Result<Driver, numeric::Error> {
        Driver::load(
            &self.path,
            hash,
            identity,
            symbol,
            vec![numeric::Carrier::Word],
            64 * 48 + 144,
        )
    }
}
#[test]
fn issue_gnet_09_numeric_host_loader_identity_carriers_and_returning_protocol_failures() {
    for code in [1, 2, 3] {
        let fixture = LoaderFixture::new(code);
        let mut bad = fixture.hash;
        bad[0] ^= 1;
        assert!(matches!(
            fixture.load(bad, [17; 32], "invar_fixture"),
            Err(numeric::Error::Identity)
        ));
        assert!(matches!(
            fixture.load(fixture.hash, [18; 32], "invar_fixture"),
            Err(numeric::Error::Identity)
        ));
        assert!(matches!(
            fixture.load(fixture.hash, [17; 32], "invar_missing"),
            Err(numeric::Error::Symbol(_))
        ));
        let mut driver = fixture
            .load(fixture.hash, [17; 32], "invar_fixture")
            .unwrap();
        for args in [vec![], vec![0, 1], vec![-1], vec![i64::from(u32::MAX) + 1]] {
            assert!(matches!(
                driver.start(&args, [1, 1, 1]),
                Err(numeric::Error::Limit)
            ));
        }
        for caps in [
            [0, 1, 1],
            [1_000_001, 1, 1],
            [1, 1_000_001, 1],
            [1, 1, 1_000_001],
        ] {
            assert!(matches!(
                driver.start(&[0], caps),
                Err(numeric::Error::Limit)
            ));
        }
        if code == 1 {
            assert_eq!(
                driver.start(&[0], [1, 1, 1]),
                Err(numeric::Error::Allocation)
            );
        } else {
            assert_eq!(
                driver.start(&[0], [1, 1, 1]),
                Err(numeric::Error::Protocol(code))
            );
        }
        drop(driver);
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
                64 * 48 + 144
            ),
            Err(numeric::Error::Identity)
        ));
        let sparse = fixture.dir.join("oversize.so");
        fs::File::create(&sparse)
            .unwrap()
            .set_len(64 * 1024 * 1024 + 1)
            .unwrap();
        assert_eq!(numeric::file_digest(&sparse), Err(numeric::Error::Limit));
        fs::remove_file(sparse).unwrap();
        // Keep deliberate failure fixtures for the lane evidence, including exact C source.
        eprintln!("retained loader failure fixture {}", fixture.dir.display());
    }
}
#[cfg(unix)]
#[test]
fn issue_gnet_09_numeric_host_real_compiler_and_loader_failures_retain_artifacts() {
    for loader in [false, true] {
        let script = executable::fixture(if loader {
            "abi9-bad-library.sh"
        } else {
            "abi9-diagnostic.sh"
        });
        let error = Plan::from_source(
            b"fn run()->f32{return 0.0;}",
            "run",
            Default::default(),
            &script,
        )
        .err()
        .unwrap();
        let evidence = match error {
            Error::Compile { message, evidence } => {
                assert!(!loader);
                assert!(message.contains("ORIGINAL_ABI9_COMPILER_DIAGNOSTIC"));
                assert_eq!(
                    fs::read_to_string(evidence.join("compiler.stderr")).unwrap(),
                    "ORIGINAL_ABI9_COMPILER_DIAGNOSTIC"
                );
                evidence
            }
            Error::Load {
                cause: numeric::Error::Library(_),
                evidence,
            } => {
                assert!(loader);
                evidence
            }
            e => panic!("unexpected {e:?}"),
        };
        assert!(evidence.join("numeric.cpp").exists());
        assert!(evidence.join("prelink.json").exists());
        eprintln!("retained native failure {}", evidence.display());
    }
}

#[cfg(unix)]
#[test]
fn issue_gnet_09_numeric_host_native_destruction_precedes_unload_and_failure_retains_mapping() {
    for fail in [false, true] {
        let fixture = LoaderFixture::new(2);
        let audit = fixture.dir.join("lifetime.txt");
        let path_literal = serde_json::to_string(audit.to_str().unwrap()).unwrap();
        fs::write(fixture.dir.join("lifetime.cpp"),format!(r#"#include <cstddef>
#include <cstdio>
#include <cstring>
#include <new>
static void note(char c){{auto* f=std::fopen({path_literal},"a");if(f){{std::fputc(c,f);std::fclose(f);}}}}
__attribute__((destructor)) static void unloaded(){{note('U');}}
struct Owner{{unsigned long long calls=0;}};
extern "C" unsigned int invar_lifetime_identity(unsigned char* p,std::size_t n){{if(n!=32)return 2;std::memset(p,17,n);return 0;}}
extern "C" unsigned int invar_lifetime_create(const long long*,std::size_t n,unsigned long long,unsigned long long,unsigned long long,void** p){{if(n!=1)return 2;*p=new(std::nothrow) Owner;if(!*p)return 1;note('C');return 0;}}
extern "C" unsigned int invar_lifetime_resume(void* p,unsigned long long){{++static_cast<Owner*>(p)->calls;note('R');return 0;}}
extern "C" unsigned int invar_lifetime_capture(void* p,long long* in,std::size_t n,long long* out,std::size_t on,unsigned long long* trace,std::size_t tn,unsigned char* task,std::size_t task_n,unsigned long long* c,std::size_t cn,unsigned long long* calls){{if(n!=1||on!=8||tn!=7||task_n!=3216||cn!=18)return 2;in[0]=0;std::memset(out,0,on*8);std::memset(trace,0,tn*8);std::memset(task,0,task_n);std::memset(c,0,cn*8);*calls=static_cast<Owner*>(p)->calls;note('P');return 0;}}
extern "C" unsigned int invar_lifetime_destroy(void* p){{note('D');if({fail})return 3;delete static_cast<Owner*>(p);return 0;}}
"#,fail=u8::from(fail))).unwrap();
        let library = fixture
            .dir
            .join(format!("lifetime.{}", std::env::consts::DLL_EXTENSION));
        let output = Command::new("c++")
            .args([
                "-std=c++17",
                "-fPIC",
                if cfg!(target_os = "macos") {
                    "-dynamiclib"
                } else {
                    "-shared"
                },
            ])
            .arg(fixture.dir.join("lifetime.cpp"))
            .arg("-o")
            .arg(&library)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let mut driver = Driver::load(
            &library,
            numeric::file_digest(&library).unwrap(),
            [17; 32],
            "invar_lifetime",
            vec![numeric::Carrier::Word],
            64 * 48 + 144,
        )
        .unwrap();
        driver.start(&[0], [1, 1, 1]).unwrap();
        assert_eq!(driver.snapshot().unwrap().native_calls, 0);
        assert!(matches!(driver.resume(0), Err(numeric::Error::Limit)));
        assert_eq!(driver.resume(1).unwrap().native_calls, 1);
        assert_eq!(fs::read_to_string(&audit).unwrap(), "CPRP");
        if fail {
            assert_eq!(driver.finish(), Err(numeric::Error::Protocol(3)));
            assert_eq!(driver.finish(), Err(numeric::Error::Protocol(3)));
            assert!(matches!(driver.resume(1), Err(numeric::Error::Protocol(3))));
            drop(driver);
            assert_eq!(fs::read_to_string(&audit).unwrap(), "CPRPD");
            eprintln!("retained destruction failure {}", fixture.dir.display());
        } else {
            // Mapping remains usable after unlink; its owner dies before dlclose.
            fs::remove_file(&library).unwrap();
            drop(driver);
            assert_eq!(fs::read_to_string(&audit).unwrap(), "CPRPDU");
        }
    }
}
