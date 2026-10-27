#![forbid(unsafe_code)]

use crate::child_deadline;
use invar::{
    check, eval, front,
    general_net::{self, Net, compiled, oracle},
    mir, mir_eval,
    numeric_contract::{F32Bits, NumericPolicy},
};
use std::{
    fs,
    path::PathBuf,
    process::{Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
};
fn storage() -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: 128,
        allocations: 128,
        cleanup_events: 128,
    }
}
/// The only CPU numeric emission: the ABI11 owned-numeric continuation.
fn emit(
    net: &Net,
    proof: &general_net::Proof,
    entry: &str,
    target: compiled::Target,
) -> Result<compiled::Artifact, compiled::Error> {
    compiled::emit_resumable_owned_numeric(
        net,
        proof,
        entry,
        target,
        compiled::Limits::default(),
        storage(),
    )
}
fn checked(source: &str, main: bool) -> check::CheckedProgram {
    check::check(front::parse(source.as_bytes()).unwrap(), main).unwrap()
}
fn lowered(source: &str) -> Net {
    Net::lower(&checked(source, false), general_net::Limits::default()).unwrap()
}
struct Cpu {
    dir: PathBuf,
}
impl Cpu {
    fn new(net: &Net, entry: &str) -> Self {
        Self::resumable(net, entry, 1_000_000)
    }
    fn resumable(net: &Net, entry: &str, quantum: u64) -> Self {
        let a = emit(net, &net.verify().unwrap(), entry, compiled::Target::Cpu).unwrap();
        assert_eq!(a.abi_version(), 11);
        assert_eq!(a.numeric_policy(), Some(NumericPolicy::STRICT_F32));
        assert!(a.device_layout().is_none());
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "gnet-f32-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let n = a.params().len();
        let bytes = a.continuation_layout().unwrap().task_bytes();
        let storage = a.owned_layout().unwrap();
        let (heap, cleanup) = (storage.heap_words(), storage.cleanup_words().max(1));
        let run = format!(
            "static_assert(sizeof(GnetTask)=={bytes});GnetTask task={{}};std::vector<gnet_w> heap({heap});std::vector<gnet_u> cleanup({cleanup});gnet_u rounds=0;while(!gnet_resume(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data(),&task,{quantum})){{if(++rounds>1000001) return 3;}}auto before=task.transitions;if(!gnet_resume(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data(),&task,{quantum})||task.transitions!=before||task.allocated||task.live||task.cleanup_used) return 4;"
        );
        let driver = format!(
            r#"
#include <iostream>
#include <vector>
#include <cfenv>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
int main(int argc,char** argv) {{
 if(argc>1) {{ int mode=argv[1][0]-'0'; if(std::fesetround(mode==1?FE_UPWARD:mode==2?FE_DOWNWARD:mode==3?FE_TOWARDZERO:FE_TONEAREST)) return 77; }}
#if defined(__SSE__)
 if(argc>2) _mm_setcsr(_mm_getcsr()|0x8040);
#endif
 gnet_u fuel,trans,cap;
 while(std::cin>>fuel>>trans>>cap) {{ std::vector<gnet_w> input({n});for(auto& x:input) std::cin>>x; gnet_w output[10]={{}};std::vector<gnet_u> trace(cap*7);{run}for(int i=0;i<8;++i)std::cout<<output[i]<<' ';for(gnet_u i=0;i<(gnet_u)output[7]*7;++i)std::cout<<trace[i]<<' ';std::cout<<'\n'; }}
}}
"#
        );
        fs::write(dir.join("source.cpp"), format!("{}\n{driver}", a.source())).unwrap();
        child_deadline::cxx(
            child_deadline::CXX17_UBSAN_O2,
            &dir.join("source.cpp"),
            &dir.join("run"),
        );
        Self { dir }
    }
    fn batch(&self, rows: &[(oracle::Limits, Vec<i64>)], mode: u8, ftz: bool) -> Vec<Vec<i64>> {
        let mut input = String::new();
        for (l, args) in rows {
            input.push_str(&format!("{} {} {}", l.fuel, l.transitions, l.trace_events));
            for x in args {
                input.push_str(&format!(" {x}"));
            }
            input.push('\n');
        }
        let path = self.dir.join("input.txt");
        fs::write(&path, input).unwrap();
        let mut cmd = Command::new(self.dir.join("run"));
        cmd.arg(mode.to_string());
        if ftz {
            cmd.arg("ftz");
        }
        let out = cmd
            .stdin(Stdio::from(fs::File::open(path).unwrap()))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
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
    fn compare(
        &self,
        net: &Net,
        entry: &str,
        args: &[eval::Value],
        limits: oracle::Limits,
        words: &[i64],
    ) {
        let o = oracle::observe_call(net, &net.verify().unwrap(), entry, args, limits).unwrap();
        let (status, start, end) = match &o.semantic.result {
            Ok(_) => (0, 0, 0),
            Err(eval::EvalError::EvaluationLimit { .. }) => (3, 0, 0),
            Err(eval::EvalError::Trap { category, span }) => (
                match category {
                    eval::TrapCategory::IntegerOverflow => 1,
                    eval::TrapCategory::DivisionByZero => 2,
                    _ => panic!("{o:?}"),
                },
                span.start as i64,
                span.end as i64,
            ),
            Err(_) => (
                match o.failure {
                    Some(oracle::ToolFailure::TransitionLimit) => 5,
                    Some(oracle::ToolFailure::TraceLimit) => 6,
                    _ => panic!("{o:?}"),
                },
                0,
                0,
            ),
        };
        assert_eq!((words[0], words[2], words[3]), (status, start, end));
        assert_eq!(
            &words[4..8],
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
                assert!((0..=u32::MAX as i64).contains(&words[1]));
                assert!(
                    x.strict_eq(F32Bits::from_bits(words[1] as u32)),
                    "expected {x:?},got {:x}",
                    words[1]
                );
            }
            Ok(eval::Value::I32(x)) => assert_eq!(words[1], i64::from(*x)),
            Ok(eval::Value::U32(x)) => assert_eq!(words[1], i64::from(*x)),
            Ok(eval::Value::Bool(x)) => assert_eq!(words[1], i64::from(*x)),
            Err(_) => assert_eq!(words[1], 0),
            x => panic!("{x:?}"),
        }
        assert_eq!(words.len(), 8 + o.events.len() * 7);
        for (actual, event) in words[8..].chunks_exact(7).zip(&o.events) {
            assert_eq!(
                actual,
                &[
                    i64::from(event.kind == oracle::EventKind::Complete),
                    event.frame as i64,
                    event.activation as i64,
                    event.token.block.function as i64,
                    event.token.block.block as i64,
                    event.token.position as i64,
                    event.rule as i64
                ]
            );
        }
    }
}
impl Drop for Cpu {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap();
    }
}
fn limits() -> oracle::Limits {
    oracle::Limits {
        fuel: 10000,
        transitions: 100000,
        trace_events: 10000,
        ..Default::default()
    }
}
fn floats(xs: &[u32]) -> Vec<eval::Value> {
    xs.iter()
        .map(|x| eval::Value::F32(F32Bits::from_bits(*x)))
        .collect()
}
fn bits(xs: &[u32]) -> Vec<i64> {
    xs.iter().map(|x| i64::from(*x)).collect()
}
fn samples() -> Vec<u32> {
    vec![
        0, 0x80000000, 1, 2, 3, 0x007fffff, 0x00800000, 0x00800001, 0x807fffff, 0x80800000,
        0x3f000000, 0x40000000, 0x40400000, 0xc0000000, 0xc0400000, 0x3f800000, 0x3f800001,
        0x3f7fffff, 0xbf800000, 0x33800000, 0x33000000, 0x4b800000, 0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000, 0x7fc00000, 0x7fa00001, 0xffc12345,
    ]
}
fn random(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    *state
}
#[test]
fn issue_gnet_08_runtime_bits_all_arithmetic_predicates_and_environment_independence() {
    for op in ["+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">="] {
        let ty = if ["+", "-", "*", "/"].contains(&op) {
            "f32"
        } else {
            "Bool"
        };
        let net = lowered(&format!("fn run(a:f32,b:f32)->{ty} {{return a{op}b;}}"));
        let cpu = Cpu::new(&net, "run");
        let mut args = Vec::new();
        for a in samples() {
            for b in samples() {
                args.push(vec![a, b]);
            }
        }
        let mut seed = 0x43525432;
        for _ in 0..1200 {
            args.push(vec![random(&mut seed), random(&mut seed)]);
        }
        let rows: Vec<_> = args.iter().map(|x| (limits(), bits(x))).collect();
        let outputs = cpu.batch(&rows, 0, false);
        for (x, w) in args.iter().zip(&outputs) {
            cpu.compare(&net, "run", &floats(x), limits(), w);
        }
        for mode in 1..=3 {
            assert_eq!(cpu.batch(&rows, mode, true), outputs, "{op},mode{mode}");
        }
    }
}
#[test]
fn issue_gnet_08_fma_negation_and_integer_conversion_rounding() {
    let net = lowered("fn run(a:f32,b:f32,c:f32)->f32{return fma(a,b,c);}");
    let cpu = Cpu::new(&net, "run");
    let mut args = vec![
        vec![0x3f800001, 0x3f7ffffe, 0xbf800000],
        vec![1, 0x3f000000, 0],
        vec![1, 0x3f000000, 1],
        vec![0x7f7fffff, 0x7f7fffff, 1],
        vec![0x7f7fffff, 0x7f7fffff, 0xff7fffff],
        vec![0x7f7fffff, 0x40000000, 0xff7fffff],
        vec![0xff7fffff, 0x40000000, 0x7f7fffff],
        vec![1, 0x3f000000, 0x80000001],
        vec![1, 1, 0x7f7fffff],
    ];
    for a in samples() {
        for b in samples() {
            args.push(vec![a, b, 0x80000000]);
            args.push(vec![a, b, 0x3f800000]);
        }
    }
    let mut seed = 0x464d4100;
    for _ in 0..3000 {
        args.push(vec![
            random(&mut seed),
            random(&mut seed),
            random(&mut seed),
        ]);
    }
    let rows: Vec<_> = args.iter().map(|x| (limits(), bits(x))).collect();
    let outputs = cpu.batch(&rows, 0, false);
    for (x, w) in args.iter().zip(&outputs) {
        cpu.compare(&net, "run", &floats(x), limits(), w);
    }
    assert_eq!(cpu.batch(&rows, 2, true), outputs);
    let separate = lowered("fn run(a:f32,b:f32,c:f32)->f32{return a*b+c;}");
    let sep = Cpu::new(&separate, "run");
    let w = sep.batch(&[(limits(), bits(&args[0]))], 0, false);
    assert_ne!(w[0][1], outputs[0][1]);
    sep.compare(&separate, "run", &floats(&args[0]), limits(), &w[0]);
    let neg = lowered("fn run(a:f32)->f32{return -a;}");
    let cpu = Cpu::new(&neg, "run");
    let rows: Vec<_> = samples()
        .iter()
        .map(|x| (limits(), vec![i64::from(*x)]))
        .collect();
    for (row, w) in rows.iter().zip(cpu.batch(&rows, 1, true)) {
        cpu.compare(&neg, "run", &floats(&[row.1[0] as u32]), limits(), &w);
        assert_eq!(w[1] as u32, row.1[0] as u32 ^ 0x80000000);
    }
    for (ty, intrinsic, values) in [
        (
            "i32",
            "f32_from_i32",
            vec![
                i32::MIN as i64,
                -16777219,
                -16777217,
                -1,
                0,
                1,
                16777215,
                16777217,
                16777219,
                i32::MAX as i64,
            ],
        ),
        (
            "u32",
            "f32_from_u32",
            vec![0, 1, 16777215, 16777217, 16777219, u32::MAX as i64],
        ),
    ] {
        let net = lowered(&format!("fn run(x:{ty})->f32{{return {intrinsic}(x);}}"));
        let cpu = Cpu::new(&net, "run");
        let rows: Vec<_> = values.iter().map(|x| (limits(), vec![*x])).collect();
        for (row, w) in rows.iter().zip(cpu.batch(&rows, 3, true)) {
            let arg = if ty == "i32" {
                eval::Value::I32(row.1[0] as i32)
            } else {
                eval::Value::U32(row.1[0] as u32)
            };
            cpu.compare(&net, "run", &[arg], limits(), &w);
        }
    }
}
#[test]
fn issue_gnet_08_actual_source_mir_net_and_cpu_every_failure_prefix() {
    for source in [
        "fn step(x:f32,n:u32)->f32 {if n==0u32{return x;}return step(x+0.000000059604644775390625,n-1u32);} fn main()->i32 {let n=step(1.0,3u32);if n==1.0{return 7;}return 0;}",
        "fn main()->i32 {let x=fma(1.00000011920928955078125,0.99999988079071044921875,-1.0);if x<0.0{return 7;}return 0;}",
        "fn main()->i32 {let inf=1.0/0.0;let nan=0.0/0.0;let negative=-0.0;if inf>0.0 and nan!=nan and negative==0.0{return 7;}return 0;}",
        "fn main()->i32 {let tiny=0.000000000000000000000000000000000000000000001;let half=tiny*0.5;if half==0.0 and tiny+tiny>tiny{return 7;}return 0;}",
        "fn main()->i32 {let x=1/0;let later=1.0/0.0;return x;}",
        "fn main()->i32 {let x=2147483647+1;let later=1.0/0.0;return x;}",
        "fn main()->i32 {if false {let dead=1/0;}let x=0.0/0.0;if x==x{return 0;}return 7;}",
        "fn main()->i32 {let x=f32_from_u32(4294967295u32);var acc=0.0;for i in 0u32..3u32{acc=acc+x;}if acc>0.0{return 7;}return 0;}",
    ] {
        let checked = checked(source, true);
        let m = mir::lower(&checked).unwrap();
        let net = Net::lower(&checked, general_net::Limits::default()).unwrap();
        let cpu = Cpu::new(&net, "main");
        let done = eval::observe(&checked, 10000);
        let mut rows = Vec::new();
        for fuel in 1..=done.steps_used + 1 {
            let l = oracle::Limits { fuel, ..limits() };
            let source_o = eval::observe(&checked, fuel);
            let mir_o = mir_eval::observe(&m, fuel);
            let net_o = oracle::observe(&net, &net.verify().unwrap(), l).unwrap();
            assert_eq!(source_o.result, mir_o.result);
            assert_eq!(source_o.steps_used, mir_o.steps_used);
            assert_eq!(net_o.semantic.result, source_o.result);
            assert_eq!(net_o.semantic.steps_used, source_o.steps_used);
            assert_eq!(net_o.semantic.cleanup_trace, source_o.cleanup_trace);
            rows.push((l, vec![]));
        }
        for cap in [0, 1, 2, 7, 31] {
            rows.push((
                oracle::Limits {
                    transitions: cap,
                    ..limits()
                },
                vec![],
            ));
            rows.push((
                oracle::Limits {
                    trace_events: cap as usize,
                    ..limits()
                },
                vec![],
            ));
        }
        for (row, w) in rows.iter().zip(cpu.batch(&rows, 2, true)) {
            cpu.compare(&net, "main", &[], row.0, &w);
        }
    }
}
#[test]
fn issue_gnet_08_metadata_proof_inputs_and_exclusions_stay_explicit() {
    let net = lowered("fn run(x:f32)->f32{return x;}");
    let proof = net.verify().unwrap();
    let cpu = Cpu::new(&net, "run");
    let valid: Vec<_> = samples()
        .iter()
        .map(|x| (limits(), vec![i64::from(*x)]))
        .collect();
    for (row, w) in valid.iter().zip(cpu.batch(&valid, 3, true)) {
        cpu.compare(&net, "run", &floats(&[row.1[0] as u32]), limits(), &w);
        assert_eq!(w[1], row.1[0]);
    }
    let bad = oracle::observe_call(&net, &proof, "run", &[eval::Value::I32(0)], limits()).unwrap();
    assert!(matches!(
        bad.semantic.result,
        Err(eval::EvalError::Tool {
            code: "ENTRYPOINT_SIGNATURE",
            ..
        })
    ));
    assert!(bad.events.is_empty());
    assert_eq!(bad.semantic.steps_used, 0);
    let rows = vec![(limits(), vec![-1]), (limits(), vec![u32::MAX as i64 + 1])];
    for w in cpu.batch(&rows, 0, false) {
        assert_eq!(w[0], 8);
        assert_eq!(w[4], 0);
        assert_eq!(w[7], 0);
    }
    let resumed = Cpu::resumable(&net, "run", 1);
    for (row, words) in valid.iter().zip(resumed.batch(&valid, 3, true)) {
        resumed.compare(&net, "run", &floats(&[row.1[0] as u32]), limits(), &words);
        assert_eq!(words[1], row.1[0]);
    }
    for words in resumed.batch(&rows, 0, false) {
        assert_eq!(&words[..8], &[8, 0, 0, 0, 0, 0, 0, 0]);
    }
    assert_eq!(
        emit(&net, &proof, "run", compiled::Target::Metal).unwrap_err(),
        compiled::Error::NumericTarget
    );
    assert_eq!(
        compiled::emit(
            &net,
            &proof,
            "run",
            compiled::Target::Cpu,
            compiled::Limits::default()
        )
        .unwrap_err(),
        compiled::Error::UnsupportedType
    );
    let other = Net::lower(
        &checked("fn run(x:f32)->f32{return x+1.0;}", false),
        general_net::Limits::default(),
    )
    .unwrap();
    assert!(matches!(
        emit(&other, &proof, "run", compiled::Target::Cpu),
        Err(compiled::Error::Boundary(_))
    ));
    for (source, admitted) in [
        ("fn main()->i32 {let a:Array<f32> = [1.0];return 7;}", true),
        ("struct Numeric {x:f32} fn main()->i32{return 7;}", false),
        (
            "fn unused()->Array<f32>{return filled_f32(1u32,1.0);} fn main()->i32{return 7;}",
            true,
        ),
    ] {
        let n = lowered(source);
        let p = n.verify().unwrap();
        // f32 arrays are direct owners; nominal numeric fields stay refused.
        let emitted = emit(&n, &p, "main", compiled::Target::Cpu);
        if admitted {
            assert_eq!(emitted.unwrap().abi_version(), 11);
        } else {
            assert_eq!(emitted.unwrap_err(), compiled::Error::UnsupportedType);
        }
        // The oracle observes every form exactly (GNET-02 closed the nominal
        // f32 member gap).
        let observed = oracle::observe(&n, &p, limits()).unwrap();
        assert_eq!(observed.failure, None);
        let c = checked(source, true);
        let expected = eval::observe(&c, limits().fuel);
        assert_eq!(observed.semantic.result, expected.result);
        assert_eq!(observed.semantic.steps_used, expected.steps_used);
        assert_eq!(observed.semantic.cleanup_trace, expected.cleanup_trace);
    }
}

#[test]
fn issue_gnet_08_numeric_variant_payload_source_exclusion_remains_explicit() {
    // strict-f64-v1 (v0.2) admits scalar f32 payloads; aggregate payloads stay
    // outside the variant subset.
    let parsed = front::parse(b"enum Numeric {X(Array<f32>)} fn main()->i32{return 7;}").unwrap();
    assert_eq!(
        check::check(parsed, true).unwrap_err().code,
        "UNSUPPORTED_FEATURE"
    );
}

#[test]
fn issue_gnet_08_scalar_f32_variant_payload_agrees_on_every_reference_engine() {
    let source = "enum Numeric {X(f32), Missing} \
                  fn get(n: read Numeric) -> f32 { return match n { Numeric::X(v) => v, Numeric::Missing => -1.0 }; } \
                  fn main()->i32 { let a = Numeric::X(2.5); let b = Numeric::Missing(); \
                  if get(a) == 2.5 and get(b) == -1.0 { return 7; } return 0; }";
    let c = checked(source, true);
    let m = mir::lower(&c).unwrap();
    let n = Net::lower(&c, general_net::Limits::default()).unwrap();
    let p = n.verify().unwrap();
    let done = eval::observe(&c, 10000);
    assert_eq!(done.result, Ok(eval::Value::I32(7)));
    for fuel in 1..=done.steps_used + 1 {
        let source = eval::observe(&c, fuel);
        let mid = mir_eval::observe(&m, fuel);
        let net = oracle::observe(&n, &p, oracle::Limits { fuel, ..limits() }).unwrap();
        assert_eq!(net.failure, None, "{fuel}");
        assert_eq!(net.semantic.result, source.result, "{fuel}");
        assert_eq!(mid.result, source.result, "{fuel}");
        assert_eq!(net.semantic.steps_used, source.steps_used, "{fuel}");
        assert_eq!(mid.steps_used, source.steps_used, "{fuel}");
    }
}

#[test]
fn issue_gnet_08_existing_integer_owner_cleanup_and_numeric_cpu_admission() {
    let source = "fn main()->i32 {if true {let a:Array<i32> = [1,2];let x=1.0/0.0;}let y=0.0/0.0;if y!=y{return 7;}return 0;}";
    let c = checked(source, true);
    let m = mir::lower(&c).unwrap();
    let n = Net::lower(&c, general_net::Limits::default()).unwrap();
    let p = n.verify().unwrap();
    let done = eval::observe(&c, 10000);
    for fuel in 1..=done.steps_used + 1 {
        let source = eval::observe(&c, fuel);
        let mid = mir_eval::observe(&m, fuel);
        let net = oracle::observe(&n, &p, oracle::Limits { fuel, ..limits() }).unwrap();
        assert_eq!(net.semantic.result, source.result);
        assert_eq!(mid.result, source.result);
        assert_eq!(net.semantic.steps_used, source.steps_used);
        assert_eq!(mid.steps_used, source.steps_used);
        assert_eq!(net.semantic.cleanup_trace, source.cleanup_trace);
        assert_eq!(mid.cleanup_trace, source.cleanup_trace);
    }
    // An integer owner beside f32 is a direct ABI11 owner.
    assert_eq!(
        emit(&n, &p, "main", compiled::Target::Cpu)
            .unwrap()
            .abi_version(),
        11
    );
}

#[test]
fn issue_gnet_09_numeric_continuations_preserve_bits_every_fuel_and_tool_prefix() {
    for source in [
        "fn step(x:f32,n:u32)->f32 {if n==0u32{return x;}return step(fma(x,1.00000011920928955078125,-0.0),n-1u32);} fn main()->i32 {let x=step(1.0,3u32);if x>1.0{return 7;}return 0;}",
        "fn main()->i32 {var acc=-0.0;for i in 0u32..4u32{acc=acc+0.000000000000000000000000000000000000000000001;}if acc>0.0{return 7;}return 0;}",
        "fn main()->i32 {let z=0.0/0.0;if z!=z {let trapped=2147483647+1;return trapped;}return 0;}",
        "fn main()->i32 {if false {let dead=1/0;}let x=f32_from_u32(4294967295u32);if x>0.0{return 7;}return 0;}",
    ] {
        let c = checked(source, true);
        let n = Net::lower(&c, general_net::Limits::default()).unwrap();
        let mid = mir::lower(&c).unwrap();
        let complete = eval::observe(&c, 10000);
        let mut rows = Vec::new();
        for fuel in 1..=complete.steps_used + 1 {
            let l = oracle::Limits { fuel, ..limits() };
            let source = eval::observe(&c, fuel);
            let mirrored = mir_eval::observe(&mid, fuel);
            let net = oracle::observe(&n, &n.verify().unwrap(), l).unwrap();
            assert_eq!(source.result, mirrored.result);
            assert_eq!(source.result, net.semantic.result);
            assert_eq!(source.steps_used, mirrored.steps_used);
            assert_eq!(source.steps_used, net.semantic.steps_used);
            assert_eq!(source.cleanup_trace, net.semantic.cleanup_trace);
            rows.push((l, vec![]));
        }
        for cap in [0, 1, 2, 7, 31] {
            rows.push((
                oracle::Limits {
                    transitions: cap,
                    ..limits()
                },
                vec![],
            ));
            rows.push((
                oracle::Limits {
                    trace_events: cap as usize,
                    ..limits()
                },
                vec![],
            ));
        }
        for quantum in [1, 3, 17] {
            let cpu = Cpu::resumable(&n, "main", quantum);
            for (row, w) in rows.iter().zip(cpu.batch(&rows, 2, true)) {
                cpu.compare(&n, "main", &[], row.0, &w);
            }
        }
    }
    let n = lowered(
        "fn run(a:f32,b:f32,c:f32,n:u32)->f32 {var x=a;for i in 0u32..n{x=fma(x,b,c);}return x;}",
    );
    let mut cases = Vec::new();
    for x in samples() {
        cases.push(vec![i64::from(x), 0x3f800000, 0x80000000, 3]);
    }
    cases.extend([
        vec![0x7f7fffff, 0x40000000, 0xff7fffff, 1],
        vec![1, 0x3f000000, 0, 1],
    ]);
    for quantum in [1, 3, 17] {
        let cpu = Cpu::resumable(&n, "run", quantum);
        let rows: Vec<_> = cases.iter().map(|x| (limits(), x.clone())).collect();
        for (row, w) in rows.iter().zip(cpu.batch(&rows, 3, true)) {
            let mut args = floats(&row.1[..3].iter().map(|x| *x as u32).collect::<Vec<_>>());
            args.push(eval::Value::U32(row.1[3] as u32));
            cpu.compare(&n, "run", &args, row.0, &w);
        }
    }
    let packet = lowered("struct Packet{x:i32} fn main()->i32{return 7;}");
    assert_eq!(
        emit(
            &packet,
            &packet.verify().unwrap(),
            "main",
            compiled::Target::Cpu
        )
        .unwrap_err(),
        compiled::Error::UnsupportedType
    );
    assert_eq!(
        emit(&n, &n.verify().unwrap(), "run", compiled::Target::Metal).unwrap_err(),
        compiled::Error::NumericTarget
    );
}
