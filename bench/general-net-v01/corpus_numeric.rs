//! Source-derived ABI8 strict-f32 CPU corpus with explicit host-mode capabilities.
use super::{child_deadline, corpus_sample};

use super::corpus_cpu as base;
use invar::{
    check, eval, front,
    general_net::{self, Net, compiled, oracle},
    mir, mir_eval,
    numeric_contract::{F32Bits, NumericPolicy},
};
use serde_json::{Value as Json, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    process::Stdio,
    sync::atomic::{AtomicUsize, Ordering},
};
fn checked(bytes: &[u8]) -> Result<check::CheckedProgram, front::FrontError> {
    let p = check::check(front::parse(bytes)?, true)?;
    invar::ownership::check(&p)?;
    Ok(p)
}
fn limits() -> oracle::Limits {
    oracle::Limits {
        fuel: 1000,
        transitions: 100000,
        trace_events: 10000,
        array_elements: 128,
    }
}
fn args(row: &Json) -> Vec<eval::Value> {
    row.as_array()
        .unwrap()
        .iter()
        .map(|v| match v["type"].as_str().unwrap() {
            "f32" => eval::Value::F32(F32Bits::from_bits(
                u32::try_from(v["bits"].as_u64().unwrap()).unwrap(),
            )),
            "i32" => eval::Value::I32(i32::try_from(v["value"].as_i64().unwrap()).unwrap()),
            "u32" => eval::Value::U32(u32::try_from(v["value"].as_u64().unwrap()).unwrap()),
            "bool" => eval::Value::Bool(v["value"].as_bool().unwrap()),
            _ => panic!("argument type"),
        })
        .collect()
}
fn carriers(xs: &[eval::Value]) -> Vec<i64> {
    xs.iter()
        .map(|v| match v {
            eval::Value::F32(x) => i64::from(x.to_bits()),
            eval::Value::I32(x) => i64::from(*x),
            eval::Value::U32(x) => i64::from(*x),
            eval::Value::Bool(x) => i64::from(*x),
            _ => panic!("numeric scalar input"),
        })
        .collect()
}
fn semantic(o: &eval::Observation) -> Json {
    match &o.result {
        Ok(eval::Value::F32(x)) => {
            json!({"status":"ok","result":{"type":"f32","bits":x.to_bits(),"nan_class":x.is_nan()}})
        }
        Ok(eval::Value::Bool(x)) => json!({"status":"ok","result":{"type":"bool","value":x}}),
        Ok(eval::Value::Unit) => json!({"status":"ok","result":{"type":"unit"}}),
        _ => base::semantic(o),
    }
}
fn recorded(o: &eval::Observation) -> Json {
    json!({"status":"executed","semantic":semantic(o),"steps_used":o.steps_used,"cleanup":base::cleanup(o)})
}
fn equal(a: &eval::Observation, b: &eval::Observation) {
    match (&a.result, &b.result) {
        (Ok(eval::Value::F32(x)), Ok(eval::Value::F32(y))) => {
            assert!(x.strict_eq(*y), "{x:?} != {y:?}")
        }
        _ => assert_eq!(semantic(a), semantic(b)),
    };
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace)
}
fn expectation(o: &eval::Observation, e: &Json) {
    if e["result"]["type"] == "f32" {
        let Ok(eval::Value::F32(x)) = &o.result else {
            panic!("{o:?}")
        };
        if e["result"]["nan"] == true {
            assert!(x.is_nan())
        } else {
            assert_eq!(
                u64::from(x.to_bits()),
                e["result"]["bits"].as_u64().unwrap()
            )
        }
    } else if e["result"]["type"] == "bool" {
        assert_eq!(semantic(o), *e)
    } else {
        base::expectation(o, e, "ABI8 source expectation")
    }
}
fn oracle_record(o: &oracle::Observation) -> Json {
    json!({"status":if o.failure==Some(oracle::ToolFailure::Unsupported){"unsupported"}else{"executed"},"reason":o.failure.map(|f|format!("{f:?}")),"semantic":semantic(&o.semantic),"steps_used":o.semantic.steps_used,"transitions":o.transitions,"max_frames":o.max_frames,"events":o.events.len(),"cleanup":base::cleanup(&o.semantic)})
}
fn emit(net: &Net, entry: &str) -> Result<compiled::Artifact, compiled::Error> {
    emit_for(net, entry, compiled::Target::Cpu)
}
/// The ABI11 owned-numeric continuation is the only CPU numeric artifact.
fn emit_for(
    net: &Net,
    entry: &str,
    target: compiled::Target,
) -> Result<compiled::Artifact, compiled::Error> {
    compiled::emit_resumable_owned_numeric(
        net,
        &net.verify().unwrap(),
        entry,
        target,
        compiled::Limits::default(),
        compiled::OwnedLimits {
            elements: 128,
            allocations: 128,
            cleanup_events: 128,
        },
    )
}
fn report(name: &str, value: Json) {
    if let Some(dir) = std::env::var_os("INVAR_GNET_CORPUS_REPORT_DIR") {
        let dir = PathBuf::from(dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&value).unwrap(),
        )
        .unwrap()
    }
}
fn profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}
struct Cpu {
    dir: PathBuf,
    metadata: Json,
    x86_ftz_daz: bool,
}
impl Cpu {
    fn new(a: compiled::Artifact) -> Self {
        Self::with_sse_mask(a, false)
    }
    fn with_sse_mask(a: compiled::Artifact, force_no_sse: bool) -> Self {
        assert_eq!(a.abi_version(), 11);
        assert_eq!(a.numeric_policy(), Some(NumericPolicy::STRICT_F32));
        assert!(
            a.continuation_layout().is_some()
                && a.speculation_layout().is_none()
                && a.device_layout().is_none()
        );
        let owned = a.owned_layout().unwrap();
        let (heap, clean) = (owned.heap_words(), owned.cleanup_words());
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "invar-corpus-numeric-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let driver = format!(
            r#"
#include <iostream>
#include <vector>
#include <cfenv>
#include <string>
#if defined(__SSE__) && !defined(INVAR_CORPUS_NO_SSE)
#include <xmmintrin.h>
#define INVAR_CORPUS_HAS_SSE 1
#else
#define INVAR_CORPUS_HAS_SSE 0
#endif
int main(int argc,char** argv){{
 if(argc==2 && std::string(argv[1])=="--capabilities"){{std::cout<<"sse-ftz-daz="<<INVAR_CORPUS_HAS_SSE<<'\n';return 0;}}
 if(argc!=3)return 2;int mode=argv[1][0]-'0';
 if(mode<0||mode>3)return 2;
 if(std::fesetround(mode==1?FE_UPWARD:mode==2?FE_DOWNWARD:mode==3?FE_TOWARDZERO:FE_TONEAREST)){{std::cerr<<"unsupported fenv rounding mode "<<mode<<'\n';return 77;}}
#if INVAR_CORPUS_HAS_SSE
 if(argv[2][0]=='1')_mm_setcsr(_mm_getcsr()|0x8040);
#else
 if(argv[2][0]=='1'){{std::cerr<<"unsupported x86 FTZ/DAZ: compiled target has no SSE\n";return 77;}}
#endif
 gnet_u fuel,trans,cap;while(std::cin>>fuel>>trans>>cap){{if(cap>1000000)return 2;std::vector<gnet_w> input({n});for(auto& x:input)if(!(std::cin>>x))return 2;gnet_w output[10]={{}};std::vector<gnet_u> trace(cap*7);std::vector<gnet_w> heap({heap});std::vector<gnet_u> cleanup({clean});GnetTask task={{}};gnet_u rounds=0;while(!gnet_resume(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data(),&task,1000000))if(++rounds>1000001)return 4;if(output[7]<0||(gnet_u)output[7]>cap||output[8]!=0)return 3;for(gnet_u i=0;i<8;++i)std::cout<<output[i]<<' ';for(gnet_u i=0;i<(gnet_u)output[7]*7;++i)std::cout<<trace[i]<<' ';std::cout<<'\n';}}}}
"#,
            n = a.params().len()
        );
        fs::write(dir.join("source.cpp"), format!("{}\n{driver}", a.source())).unwrap();
        let mut flags = child_deadline::CXX17_UBSAN_O2.to_vec();
        if force_no_sse {
            flags.push("-DINVAR_CORPUS_NO_SSE=1");
        }
        child_deadline::cxx(&flags, &dir.join("source.cpp"), &dir.join("run"));
        let probe = child_deadline::command(5)
            .arg(dir.join("run"))
            .arg("--capabilities")
            .output()
            .unwrap();
        assert!(
            probe.status.success(),
            "numeric capability probe: {probe:?}"
        );
        let x86_ftz_daz = match probe.stdout.as_slice() {
            b"sse-ftz-daz=0\n" => false,
            b"sse-ftz-daz=1\n" => true,
            other => panic!("unexpected numeric capability response: {other:?}"),
        };
        let metadata = json!({"host_environment_capabilities":{"x86_FTZ_DAZ":x86_ftz_daz,"forced_no_sse_for_test":force_no_sse},"abi":a.abi_version(),"numeric_policy":"strict-f32-v1","target":format!("{:?}",a.target()),"params":a.params().iter().map(|p|format!("{p:?}")).collect::<Vec<_>>(),"result":format!("{:?}",a.result()),"states":a.states(),"frame_bytes":a.frame_bytes(),"emitted_source_sha256":base::digest(a.source().as_bytes()),"harness_sha256":base::digest(&fs::read(dir.join("source.cpp")).unwrap()),"native_executable_sha256":base::digest(&fs::read(dir.join("run")).unwrap()),"owned_layout":true,"continuation_layout":true,"speculation_layout":false,"device_layout":false});
        if let Some(r) = std::env::var_os("INVAR_GNET_CORPUS_REPORT_DIR") {
            let p = PathBuf::from(r)
                .join("numeric-artifacts")
                .join(base::digest(a.source().as_bytes()));
            // The forced no-SSE regression must not overwrite a real-host receipt.
            let p = if force_no_sse {
                p.join("forced-no-sse")
            } else {
                p
            };
            fs::create_dir_all(&p).unwrap();
            fs::write(p.join("emitted-source.txt"), a.source()).unwrap();
            fs::copy(dir.join("source.cpp"), p.join("cpu-harness.cpp")).unwrap();
            fs::write(
                p.join("artifact.json"),
                serde_json::to_vec_pretty(&metadata).unwrap(),
            )
            .unwrap();
        }
        Self {
            dir,
            metadata,
            x86_ftz_daz,
        }
    }
    fn environment(&self, mode: u8) -> Json {
        json!({"rounding_mode":mode,"x86_FTZ_DAZ_requested":true,"x86_FTZ_DAZ":self.x86_ftz_daz,"x86_FTZ_DAZ_status":if self.x86_ftz_daz {"executed"} else {"unsupported_no_sse"}})
    }
    fn batch(&self, rows: &[(oracle::Limits, Vec<i64>)], mode: u8, ftz: bool) -> Vec<Vec<i64>> {
        assert!(
            !ftz || self.x86_ftz_daz,
            "x86 FTZ/DAZ is unsupported by this compiled driver"
        );
        let mut input = String::new();
        for (l, xs) in rows {
            input.push_str(&format!("{} {} {}", l.fuel, l.transitions, l.trace_events));
            for x in xs {
                input.push_str(&format!(" {x}"))
            }
            input.push('\n')
        }
        let path = self.dir.join("input.txt");
        fs::write(&path, input).unwrap();
        let out = child_deadline::command(30)
            .arg(self.dir.join("run"))
            .arg(mode.to_string())
            .arg(if ftz { "1" } else { "0" })
            .stdin(Stdio::from(fs::File::open(path).unwrap()))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "numeric run {:?}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
        let result: Vec<Vec<i64>> = String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
            .collect();
        assert_eq!(result.len(), rows.len());
        result
    }
    fn compare(&self, o: &oracle::Observation, w: &[i64]) -> Json {
        let (status, start, end) = match &o.semantic.result {
            Ok(_) => (0, 0, 0),
            Err(eval::EvalError::EvaluationLimit { .. }) => (3, 0, 0),
            Err(eval::EvalError::Trap { category, span }) => (
                match category {
                    eval::TrapCategory::IntegerOverflow => 1,
                    eval::TrapCategory::DivisionByZero => 2,
                    _ => panic!("numeric trap {o:?}"),
                },
                span.start as i64,
                span.end as i64,
            ),
            Err(eval::EvalError::Tool {
                code: "CALL_DEPTH_LIMIT",
                ..
            }) => (compiled::Status::CallDepthLimit as i64, 0, 0),
            Err(_) => (
                match o.failure {
                    Some(oracle::ToolFailure::TransitionLimit) => 5,
                    Some(oracle::ToolFailure::TraceLimit) => 6,
                    _ => panic!("numeric tool failure {o:?}"),
                },
                0,
                0,
            ),
        };
        assert_eq!((w[0], w[2], w[3]), (status, start, end));
        assert_eq!(
            &w[4..8],
            &[
                o.semantic.steps_used as i64,
                o.transitions as i64,
                o.max_frames as i64,
                o.events.len() as i64
            ]
        );
        assert!(o.semantic.cleanup_trace.is_empty());
        match &o.semantic.result {
            Ok(eval::Value::F32(x)) => {
                assert!((0..=u32::MAX as i64).contains(&w[1]));
                assert!(x.strict_eq(F32Bits::from_bits(w[1] as u32)))
            }
            Ok(eval::Value::I32(x)) => assert_eq!(w[1], i64::from(*x)),
            Ok(eval::Value::U32(x)) => assert_eq!(w[1], i64::from(*x)),
            Ok(eval::Value::Bool(x)) => assert_eq!(w[1], i64::from(*x)),
            Ok(eval::Value::Unit) | Err(_) => assert_eq!(w[1], 0),
            x => panic!("scalar result {x:?}"),
        };
        assert_eq!(w.len(), 8 + o.events.len() * 7);
        for (actual, e) in w[8..].chunks_exact(7).zip(&o.events) {
            assert_eq!(
                actual,
                &[
                    i64::from(e.kind == oracle::EventKind::Complete),
                    e.frame as i64,
                    e.activation as i64,
                    e.token.block.function as i64,
                    e.token.block.block as i64,
                    e.token.position as i64,
                    e.rule as i64
                ]
            )
        }
        json!({"status":"executed","artifact":self.metadata,"header":w[..8],"all_words_le_i64_sha256":base::digest(&w.iter().flat_map(|w|w.to_le_bytes()).collect::<Vec<_>>()),"events":o.events.len(),"nan_agreement":"class only; non-NaN exact bits; echo/negate/selected vectors additionally assert NaN payload/sign bits","gpu_execution":"not_performed"})
    }
}
impl Drop for Cpu {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap()
    }
}
fn observe(
    c: &check::CheckedProgram,
    entry: &str,
    xs: &[eval::Value],
    fuel: u64,
) -> eval::Observation {
    let o = if entry == "main" && xs.is_empty() {
        eval::observe(c, fuel)
    } else {
        eval::named::observe_numeric_named(c, entry, xs, fuel)
    };
    assert!(
        !matches!(
            o.result,
            Err(eval::EvalError::Tool {
                code: "NATIVE_ABI" | "INVALID_FUEL",
                ..
            })
        ),
        "source reference refused {entry}: {o:?}"
    );
    o
}
#[path = "corpus_numeric/denominator.rs"]
mod denominator;
pub fn full_denominator() {
    denominator::run();
}
pub fn rounding_environment() {
    let values = [
        0, 0x80000000, 1, 3, 0x007fffff, 0x00800000, 0x3f800000, 0x3f800001, 0x33800000,
        0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fa00001, 0xffc12345,
    ];
    let mut seed = 0x474e_4638u32;
    let mut rows = Vec::new();
    let mut comparisons = 0;
    for (file, entries) in [
        (
            "arithmetic",
            vec![
                "add", "sub", "mul", "div", "fused", "separate", "negate", "echo",
            ],
        ),
        ("predicates", vec!["eq", "ne", "lt", "le", "gt", "ge"]),
    ] {
        let bytes = fs::read(base::root().join(format!("numeric/{file}.inv"))).unwrap();
        let c = checked(&bytes).unwrap();
        let net = Net::lower(&c, general_net::Limits::default()).unwrap();
        for entry in entries {
            let mut inputs = Vec::<Vec<eval::Value>>::new();
            if matches!(entry, "echo" | "negate") {
                for a in values {
                    inputs.push(vec![eval::Value::F32(F32Bits::from_bits(a))])
                }
            } else {
                for a in values {
                    for b in values {
                        let mut xs = vec![
                            eval::Value::F32(F32Bits::from_bits(a)),
                            eval::Value::F32(F32Bits::from_bits(b)),
                        ];
                        if matches!(entry, "fused" | "separate") {
                            xs.push(eval::Value::F32(F32Bits::from_bits(
                                values[(inputs.len() + 3) % values.len()],
                            )))
                        };
                        inputs.push(xs)
                    }
                }
            };
            for _ in 0..64 {
                let n = if matches!(entry, "echo" | "negate") {
                    1
                } else if matches!(entry, "fused" | "separate") {
                    3
                } else {
                    2
                };
                let mut xs = Vec::new();
                for _ in 0..n {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    xs.push(eval::Value::F32(F32Bits::from_bits(seed)))
                }
                inputs.push(xs)
            }
            let h = Cpu::new(emit(&net, entry).unwrap());
            let l = limits();
            let batch = inputs
                .iter()
                .map(|xs| (l, carriers(xs)))
                .collect::<Vec<_>>();
            let mut sources = Vec::new();
            let expected = inputs
                .iter()
                .map(|xs| {
                    let s = observe(&c, entry, xs, l.fuel);
                    let o =
                        oracle::observe_call(&net, &net.verify().unwrap(), entry, xs, l).unwrap();
                    equal(&s, &o.semantic);
                    sources.push(recorded(&s));
                    if matches!(entry, "echo" | "negate") {
                        let eval::Value::F32(x) = xs[0] else {
                            panic!("f32 input")
                        };
                        let bits = x.to_bits() ^ if entry == "negate" { 0x80000000 } else { 0 };
                        for result in [&s.result, &o.semantic.result] {
                            let Ok(eval::Value::F32(y)) = result else {
                                panic!("f32 copy")
                            };
                            assert_eq!(y.to_bits(), bits);
                        }
                    }
                    o
                })
                .collect::<Vec<_>>();
            let baseline = h.batch(&batch, 0, false);
            for (w, o) in baseline.iter().zip(&expected) {
                h.compare(o, w);
                if matches!(entry, "echo" | "negate") {
                    let Ok(eval::Value::F32(x)) = o.semantic.result else {
                        panic!("f32 copy")
                    };
                    assert_eq!(w[1], i64::from(x.to_bits()));
                }
                comparisons += 1
            }
            let mut environments = Vec::new();
            for mode in [1, 2, 3] {
                let actual = h.batch(&batch, mode, h.x86_ftz_daz);
                assert_eq!(
                    actual, baseline,
                    "all encodings and trace words unchanged by host flags"
                );
                for (w, o) in actual.iter().zip(&expected) {
                    h.compare(o, w);
                    comparisons += 1
                }
                environments.push(json!({"environment":h.environment(mode),"rounding_mode":mode,"x86_FTZ_DAZ":h.x86_ftz_daz,"observations":actual.len(),"all_words_identical_to_nearest":true}))
            }
            rows.push(json!({"source":format!("numeric/{file}.inv"),"source_sha256":base::digest(&bytes),"entry":entry,"artifact":h.metadata,"source_named_reference":"eval::named::observe_numeric_named; original checked function and fuel","source_observations":sources,"actual_source_oracle_comparisons":inputs.len(),"actual_source_native_comparisons":inputs.len()*4,"input_vectors":inputs.len(),"input_carriers_sha256":base::digest(&batch.iter().flat_map(|(_,xs)|xs.iter().flat_map(|x|x.to_le_bytes())).collect::<Vec<_>>()),"baseline_all_words_sha256":base::digest(&baseline.iter().flatten().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>()),"environments":environments}));
        }
    }
    eprintln!(
        "ABI11 rounding/environment:{comparisons} actual native/reference comparisons across14 source entries"
    );
    report(
        "numeric-environment",
        json!({"profile":profile(),"actual_native_reference_comparisons":comparisons,"rows":rows,"special_bit_values":values,"random":{"initial_seed":0x474e4638u32,"generator":"u32 wrapping state=state*1664525+1013904223","random_vectors_per_entry":64,"shared_source_order_stream":true},"host_environment_scope":"process-local fenv upward/downward/toward-zero on every host; x86 FTZ/DAZ only where the compiled driver reports SSE, otherwise explicitly unsupported; not a host settings mutation","gpu_execution":"not_performed"}),
    );
}
pub fn fuel_and_admission() {
    let doc = base::document("numeric-vectors.json");
    let mut rows = Vec::new();
    let mut native = 0;
    let mut reference = 0;
    let mut source_native = 0;
    let mut rejection = 0;
    for case in doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["entry"] == "main")
    {
        let path = case["source"].as_str().unwrap();
        let bytes = fs::read(base::root().join(path)).unwrap();
        let c = match checked(&bytes) {
            Ok(c) => c,
            Err(e) => {
                assert_eq!(case["expected_check"], e.code);
                rejection += 1;
                rows.push(json!({"source":path,"source_rejection":e.code}));
                continue;
            }
        };
        let net = Net::lower(&c, general_net::Limits::default()).unwrap();
        let m = mir::lower(&c).unwrap();
        let done = eval::observe(&c, 1000);
        expectation(&done, &case["expected_source"]);
        let emitted = emit(&net, "main");
        if path.contains("excluded") {
            // Nominal f32 members stay refused. f32 arrays and owners are ABI11
            // programs but keep the oracle-only rows recorded for them here.
            let admission = if path.contains("nominal") {
                assert_eq!(emitted.unwrap_err(), compiled::Error::UnsupportedType);
                rejection += 1;
                "UnsupportedType"
            } else {
                assert!(emitted.is_ok(), "{path} is an ABI11 program");
                "admitted"
            };
            let mut prefixes = Vec::new();
            for fuel in 1..=done.steps_used + 1 {
                let l = oracle::Limits { fuel, ..limits() };
                let s = eval::observe(&c, fuel);
                let mid = mir_eval::observe(&m, fuel);
                equal(&s, &mid);
                let o = oracle::observe(&net, &net.verify().unwrap(), l).unwrap();
                // Every excluded form, including nominal f32 members, has
                // exact CPU oracle agreement.
                assert_eq!(o.failure, None);
                equal(&s, &o.semantic);
                reference += 1;
                prefixes.push(json!({"fuel":fuel,"source":recorded(&s),"oracle":oracle_record(&o)}))
            }
            rows.push(json!({"source":path,"abi11":admission,"prefixes":prefixes}));
            continue;
        }
        let h = Cpu::new(emitted.unwrap());
        let mut batch = Vec::new();
        let mut expected = Vec::new();
        let mut prefixes = Vec::new();
        for fuel in 1..=done.steps_used + 1 {
            let l = oracle::Limits { fuel, ..limits() };
            let s = eval::observe(&c, fuel);
            equal(&s, &mir_eval::observe(&m, fuel));
            let o = oracle::observe(&net, &net.verify().unwrap(), l).unwrap();
            equal(&s, &o.semantic);
            reference += 1;
            source_native += 1;
            batch.push((l, vec![]));
            expected.push(o);
            prefixes.push(json!({"fuel":fuel,"source":recorded(&s)}))
        }
        for cap in [0, 1, 2, 7, 31] {
            for trace in [false, true] {
                let l = if trace {
                    oracle::Limits {
                        trace_events: cap as usize,
                        ..limits()
                    }
                } else {
                    oracle::Limits {
                        transitions: cap,
                        ..limits()
                    }
                };
                batch.push((l, vec![]));
                expected.push(oracle::observe(&net, &net.verify().unwrap(), l).unwrap())
            }
        }
        let actual = h.batch(&batch, 2, h.x86_ftz_daz);
        let receipts = actual
            .iter()
            .zip(&expected)
            .map(|(w, o)| {
                native += 1;
                h.compare(o, w)
            })
            .collect::<Vec<_>>();
        rows.push(
            json!({"source":path,"environment":h.environment(2),"positive_fuel_prefixes":prefixes,"native_observations":receipts}),
        );
    }
    let mut cache = BTreeMap::<(String, String), Cpu>::new();
    let mut named_reference = 0;
    let mut named_rows = 0;
    for case in doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["entry"] != "main")
    {
        named_rows += 1;
        let path = case["source"].as_str().unwrap();
        let bytes = fs::read(base::root().join(path)).unwrap();
        let c = checked(&bytes).unwrap();
        let n = Net::lower(&c, general_net::Limits::default()).unwrap();
        let entry = case["entry"].as_str().unwrap();
        let xs = args(&case["arguments"]);
        let done = observe(&c, entry, &xs, case["fuel"].as_u64().unwrap());
        expectation(&done, &case["expected_source"]);
        let h = cache
            .entry((path.to_owned(), entry.to_owned()))
            .or_insert_with(|| Cpu::new(emit(&n, entry).unwrap()));
        let mut batch = Vec::new();
        let mut expected = Vec::new();
        let mut prefixes = Vec::new();
        for fuel in 1..=done.steps_used + 1 {
            let l = oracle::Limits { fuel, ..limits() };
            let s = observe(&c, entry, &xs, fuel);
            let o = oracle::observe_call(&n, &n.verify().unwrap(), entry, &xs, l).unwrap();
            equal(&s, &o.semantic);
            named_reference += 1;
            source_native += 1;
            prefixes.push(json!({"fuel":fuel,"source":recorded(&s)}));
            batch.push((l, carriers(&xs)));
            expected.push(o)
        }
        let receipts = h
            .batch(&batch, 3, h.x86_ftz_daz)
            .iter()
            .zip(&expected)
            .map(|(w, o)| {
                native += 1;
                h.compare(o, w)
            })
            .collect::<Vec<_>>();
        rows.push(json!({"id":case["id"],"source":path,"entry":entry,"environment":h.environment(3),"runtime_arguments":case["arguments"],"source_named_reference":"eval::named::observe_numeric_named","positive_fuel_prefixes":prefixes,"native_observations":receipts}));
    }
    assert_eq!(named_rows, 85);
    let bytes = fs::read(base::root().join("numeric/arithmetic.inv")).unwrap();
    let n = Net::lower(&checked(&bytes).unwrap(), general_net::Limits::default()).unwrap();
    let h = Cpu::new(emit(&n, "echo").unwrap());
    let bad = h.batch(
        &[(limits(), vec![-1]), (limits(), vec![u32::MAX as i64 + 1])],
        0,
        false,
    );
    for w in &bad {
        assert_eq!(w[0], 8);
        assert_eq!(w[4], 0);
        assert_eq!(w[7], 0);
        assert_eq!(w.len(), 8);
        rejection += 1
    }
    assert_eq!(
        emit_for(&n, "echo", compiled::Target::Metal).unwrap_err(),
        compiled::Error::NumericTarget
    );
    rejection += 1;
    assert_eq!(
        compiled::emit(
            &n,
            &n.verify().unwrap(),
            "echo",
            compiled::Target::Cpu,
            compiled::Limits::default()
        )
        .unwrap_err(),
        compiled::Error::UnsupportedType
    );
    rejection += 1;
    let signature = oracle::observe_call(
        &n,
        &n.verify().unwrap(),
        "echo",
        &[eval::Value::I32(0)],
        limits(),
    )
    .unwrap();
    assert!(matches!(
        signature.semantic.result,
        Err(eval::EvalError::Tool {
            code: "ENTRYPOINT_SIGNATURE",
            ..
        })
    ));
    assert_eq!(signature.semantic.steps_used, 0);
    assert!(signature.events.is_empty());
    rejection += 1;
    eprintln!(
        "ABI11 failure/admission:{native} native comparisons,{reference} source/MIR/oracle and {named_reference} named source/oracle fuel observations,{rejection} explicit exclusions/rejections"
    );
    report(
        "numeric-failures",
        json!({"profile":profile(),"actual_native_reference_comparisons":native,"source_mir_oracle_fuel_observations":reference,"named_source_oracle_fuel_comparisons":named_reference,"named_source_native_fuel_comparisons":named_reference,"source_native_fuel_comparisons":source_native,"named_runtime_rows":named_rows,"explicit_admission_rejections":rejection,"invalid_carrier_headers":bad,"wrong_signature":oracle_record(&signature),"rows":rows,"gpu_execution":"not_performed"}),
    );
}

/// Exercise the real compiled no-SSE driver branch even on an x86 test host.
/// This is a harness portability regression, not an ARM hardware claim.
pub fn environment_capabilities() {
    let bytes = fs::read(base::root().join("numeric/arithmetic.inv")).unwrap();
    let checked = checked(&bytes).unwrap();
    let net = Net::lower(&checked, general_net::Limits::default()).unwrap();
    let h = Cpu::with_sse_mask(emit(&net, "echo").unwrap(), true);
    assert!(!h.x86_ftz_daz);
    assert_eq!(
        h.metadata["host_environment_capabilities"]["forced_no_sse_for_test"],
        true
    );
    let rows = [0u32, 1, 0x80000000, 0x7fa00001].map(|bits| (limits(), vec![i64::from(bits)]));
    let baseline = h.batch(&rows, 0, false);
    for mode in [0, 1, 2, 3] {
        let actual = h.batch(&rows, mode, false);
        assert_eq!(actual, baseline);
        assert_eq!(
            h.environment(mode)["x86_FTZ_DAZ_status"],
            "unsupported_no_sse"
        );
        for ((limits, input), words) in rows.iter().zip(&actual) {
            let args = vec![eval::Value::F32(F32Bits::from_bits(input[0] as u32))];
            let expected =
                oracle::observe_call(&net, &net.verify().unwrap(), "echo", &args, *limits).unwrap();
            h.compare(&expected, words);
            assert_eq!(words[1], input[0]);
        }
    }
    let rejected = child_deadline::command(5)
        .arg(h.dir.join("run"))
        .args(["0", "1"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(rejected.status.code(), Some(77));
    assert!(rejected.stdout.is_empty());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("unsupported x86 FTZ/DAZ"));
    report(
        "numeric-no-sse-regression",
        json!({"profile":profile(),"simulated_no_sse":true,"architecture_qualification":"not_claimed_by_forced_no_sse_regression","fenv_modes_executed":[0,1,2,3],"vectors_per_mode":rows.len(),"x86_FTZ_DAZ":"unsupported_no_sse","explicit_ftz_request_exit":77,"artifact":h.metadata}),
    );
}
