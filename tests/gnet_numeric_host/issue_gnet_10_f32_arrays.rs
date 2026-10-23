#![forbid(unsafe_code)]

use crate::child_deadline;

use invar::{
    check, eval, front,
    general_net::{self, Net, compiled, oracle},
    mir, mir_eval,
    numeric_contract::{F32Bits, NumericPolicy},
};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    process::Stdio,
    sync::atomic::{AtomicUsize, Ordering},
};
fn digest(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn source(name: &str) -> String {
    fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/issue_gnet_10_f32_arrays_fixtures")
            .join(name),
    )
    .unwrap()
}
fn checked(s: &str) -> check::CheckedProgram {
    let c = check::check(front::parse(s.as_bytes()).unwrap(), false).unwrap();
    invar::ownership::check(&c).unwrap();
    c
}
fn net(s: &str) -> Net {
    Net::lower(&checked(s), general_net::Limits::default()).unwrap()
}
fn limits() -> oracle::Limits {
    oracle::Limits {
        fuel: 1000,
        transitions: 100000,
        trace_events: 10000,
        array_elements: 128,
    }
}
fn storage() -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: 128,
        allocations: 128,
        cleanup_events: 128,
    }
}
fn result(o: &eval::Observation) -> Json {
    match &o.result {
        Ok(eval::Value::F32(x)) => json!({"f32":x.to_bits()}),
        Ok(eval::Value::ArrayF32(v)) => {
            json!({"ArrayF32":v.borrow().iter().map(|x|x.to_bits()).collect::<Vec<_>>()})
        }
        Ok(eval::Value::I32(x)) => json!({"i32":x}),
        Ok(eval::Value::U32(x)) => json!({"u32":x}),
        Ok(eval::Value::Bool(x)) => json!({"bool":x}),
        Err(e) => e.to_json("array-test"),
        x => panic!("{x:?}"),
    }
}
fn equal(a: &eval::Observation, b: &eval::Observation) {
    assert_eq!(result(a), result(b));
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace)
}
fn report(name: &str, rows: Json) {
    if let Some(dir) = std::env::var_os("INVAR_GNET_ARRAY_REPORT_DIR") {
        let dir = PathBuf::from(dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(format!("{name}.json")),serde_json::to_vec_pretty(&json!({"profile":if cfg!(debug_assertions){"debug"}else{"release"},"rows":rows,"gpu_execution":"not_performed"})).unwrap()).unwrap()
    }
}
fn carriers(xs: &[eval::Value]) -> Vec<i64> {
    xs.iter()
        .map(|x| match x {
            eval::Value::F32(x) => i64::from(x.to_bits()),
            eval::Value::U32(x) => i64::from(*x),
            eval::Value::I32(x) => i64::from(*x),
            _ => panic!("typed inputs"),
        })
        .collect()
}
struct Cpu {
    dir: PathBuf,
    layout: compiled::OwnedLayout,
    metadata: Json,
}
impl Cpu {
    fn new(n: &Net, entry: &str, s: compiled::OwnedLimits) -> Self {
        let a = compiled::emit_resumable_owned_numeric(
            n,
            &n.verify().unwrap(),
            entry,
            compiled::Target::Cpu,
            compiled::Limits::default(),
            s,
        )
        .unwrap();
        assert_eq!(a.abi_version(), 11);
        assert_eq!(a.numeric_policy(), Some(NumericPolicy::STRICT_F32));
        assert!(
            a.continuation_layout().is_some()
                && a.speculation_layout().is_none()
                && a.device_layout().is_none()
                && a.aggregate_layout().is_none()
        );
        let layout = a.owned_layout().unwrap().clone();
        assert_eq!(layout.output_words(), 10);
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "gnet-f32-arrays-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let driver = format!(
            r#"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
int main(int argc,char** argv){{
 if(argc!=2)return 2;std::string mode=argv[1];gnet_u fuel,trans,cap;
 while(std::cin>>fuel>>trans>>cap){{if(cap>1000000)return 2;std::vector<gnet_w> input({params});for(auto& x:input)if(!(std::cin>>x))return 2;
 std::vector<gnet_w> output_guard(12,-31337),heap_guard({heap}+2,-31337);gnet_w* output=output_guard.data()+1;gnet_w* heap=heap_guard.data()+1;std::memset(output,0,80);std::memset(heap,0,{heap}*8);
 std::vector<gnet_u> trace_guard(cap*7+2,912345),cleanup_guard({clean}+2,912345);gnet_u* trace=trace_guard.data()+1;gnet_u* cleanup=cleanup_guard.data()+1;
 GnetTask task={{}};gnet_u rounds=0;while(!gnet_resume(input.data(),output,trace,fuel,trans,cap,heap,cleanup,&task,1000000))if(++rounds>1000001)return 7;
 if(output_guard.front()!=-31337||output_guard.back()!=-31337||heap_guard.front()!=-31337||heap_guard.back()!=-31337||trace_guard.front()!=912345||trace_guard.back()!=912345||cleanup_guard.front()!=912345||cleanup_guard.back()!=912345)return 3;
 if(output[7]<0||(gnet_u)output[7]>cap||output[8]<0||(gnet_u)output[8]>{cleanup_cap}||output[9]<0||(gnet_u)output[9]>{slots})return 4;
 if(mode!="normal"){{if(output[0]!=0||output[1]<=0)return 5;gnet_w h=output[1];gnet_u d=((gnet_u)h-1)*4,allocated=(gnet_u)output[9],live=(gnet_u)heap[d+1];gnet_w status=0,value=777;gnet_u index=0;
 if(mode=="kind")heap[d+2]=0;if(mode=="offset")heap[d]={elements}+1;if(mode=="length")heap[d+1]=-1;if(mode=="flag")heap[d+3]=2;if(mode=="bits")heap[{pool}+(gnet_u)heap[d]]=-1;if(mode=="allocated")allocated={slots}+1;if(mode=="handle")h=-1;
 if(mode=="bounds")index=(gnet_u)heap[d+1];std::vector<gnet_w> before(heap,heap+{heap});
 bool ok=mode=="store"?gnet_array_f32_store(heap,h,index,4294967296LL,allocated,status):gnet_array_f32_index(heap,h,index,allocated,value,status);
 if(ok||status!=(mode=="bounds"?9:7)||value!=777||std::memcmp(before.data(),heap,{heap}*8))return 6;
 std::cout<<status<<' '<<value<<'\n';continue;
 }}
 for(unsigned j=0;j<10;++j)std::cout<<output[j]<<' ';for(gnet_u j=0;j<(gnet_u)output[7]*7;++j)std::cout<<trace[j]<<' ';for(gnet_u j=0;j<(gnet_u)output[8]*4;++j)std::cout<<cleanup[j]<<' ';for(gnet_u j=0;j<{heap};++j)std::cout<<heap[j]<<' ';std::cout<<'\n';}}
}}
"#,
            params = a.params().len(),
            heap = layout.heap_words(),
            clean = layout.cleanup_words(),
            cleanup_cap = layout.cleanup_events(),
            slots = layout.allocations(),
            elements = layout.elements(),
            pool = layout.allocations() * 4
        );
        fs::write(dir.join("source.cpp"), format!("{}\n{driver}", a.source())).unwrap();
        let out = child_deadline::command(60)
            .args([
                "c++",
                "-std=c++17",
                "-O2",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
            ])
            .arg(dir.join("source.cpp"))
            .arg("-o")
            .arg(dir.join("run"))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{:?}:{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
        let metadata = json!({"abi":a.abi_version(),"numeric_policy":"strict-f32-v1","params":a.params().iter().map(|x|format!("{x:?}")).collect::<Vec<_>>(),"result":format!("{:?}",a.result()),"states":a.states(),"frame_bytes":a.frame_bytes(),"emitted_source_sha256":digest(a.source().as_bytes()),"harness_sha256":digest(&fs::read(dir.join("source.cpp")).unwrap()),"native_sha256":digest(&fs::read(dir.join("run")).unwrap()),"owned":{"elements":layout.elements(),"allocations":layout.allocations(),"cleanup_events":layout.cleanup_events(),"heap_words":layout.heap_words(),"cleanup_words":layout.cleanup_words(),"storage_bytes":layout.storage_bytes()},"array_kinds":{"i32":0,"u32":1,"f32":258},"no_speculation_device":true});
        if let Some(r) = std::env::var_os("INVAR_GNET_ARRAY_REPORT_DIR") {
            let p = PathBuf::from(r)
                .join("artifacts")
                .join(digest(a.source().as_bytes()));
            fs::create_dir_all(&p).unwrap();
            fs::write(p.join("emitted-source.txt"), a.source()).unwrap();
            fs::copy(dir.join("source.cpp"), p.join("cpu-harness.cpp")).unwrap();
            fs::write(
                p.join("artifact.json"),
                serde_json::to_vec_pretty(&metadata).unwrap(),
            )
            .unwrap()
        };
        Self {
            dir,
            layout,
            metadata,
        }
    }
    fn batch(&self, rows: &[(oracle::Limits, Vec<i64>)], mode: &str) -> Vec<Vec<i64>> {
        let mut input = String::new();
        for (l, xs) in rows {
            input.push_str(&format!("{} {} {}", l.fuel, l.transitions, l.trace_events));
            for x in xs {
                input.push_str(&format!(" {x}"))
            }
            input.push('\n')
        }
        let path = self.dir.join("inputs.txt");
        fs::write(&path, input).unwrap();
        let out = child_deadline::command(30)
            .arg(self.dir.join("run"))
            .arg(mode)
            .stdin(Stdio::from(fs::File::open(path).unwrap()))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "array run {:?} mode={mode}:{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
        let rows_out: Vec<Vec<i64>> = String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|x| x.parse().unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(rows_out.len(), rows.len());
        rows_out
    }
    fn compare(&self, n: &Net, o: &oracle::Observation, w: &[i64]) -> Json {
        let (status, start, end) = match &o.semantic.result {
            Ok(_) => (0, 0, 0),
            Err(eval::EvalError::Trap { category, span }) => (
                match category {
                    eval::TrapCategory::IntegerOverflow => 1,
                    eval::TrapCategory::DivisionByZero => 2,
                    eval::TrapCategory::IndexOutOfBounds => 9,
                    _ => panic!("trap"),
                },
                span.start as i64,
                span.end as i64,
            ),
            Err(eval::EvalError::EvaluationLimit { .. }) => (3, 0, 0),
            Err(eval::EvalError::Tool {
                code: "ARRAY_SIZE_LIMIT",
                ..
            }) => (10, 0, 0),
            Err(eval::EvalError::Tool {
                code: "CALL_DEPTH_LIMIT",
                ..
            }) => (4, 0, 0),
            Err(_) => (
                match o.failure {
                    Some(oracle::ToolFailure::TransitionLimit) => 5,
                    Some(oracle::ToolFailure::TraceLimit) => 6,
                    Some(oracle::ToolFailure::Allocation) => 11,
                    _ => panic!("{o:?}"),
                },
                0,
                0,
            ),
        };
        assert_eq!((w[0], w[2], w[3]), (status, start, end));
        assert_eq!(
            &w[4..9],
            &[
                o.semantic.steps_used as i64,
                o.transitions as i64,
                o.max_frames as i64,
                o.events.len() as i64,
                o.semantic.cleanup_trace.len() as i64
            ]
        );
        let mut at = 10;
        for e in &o.events {
            assert_eq!(
                &w[at..at + 7],
                &[
                    i64::from(e.kind == oracle::EventKind::Complete),
                    e.frame as i64,
                    e.activation as i64,
                    e.token.block.function as i64,
                    e.token.block.block as i64,
                    e.token.position as i64,
                    e.rule as i64
                ]
            );
            at += 7
        }
        let p = n.compiled_mir(&n.verify().unwrap()).unwrap();
        for c in &o.semantic.cleanup_trace {
            let f = w[at] as usize;
            let place = w[at + 1] as usize;
            let frame = w[at + 3] as u64;
            assert_eq!(p.functions[f].declaration_order[place], c.name);
            assert_eq!(w[at + 2], c.length as i64);
            assert!(
                o.events.iter().any(|e| e.kind == oracle::EventKind::Demand
                    && e.frame == frame
                    && e.token.block.function == f
                    && cleanup_site(&p, e, &c.name)),
                "cleanup {} must belong to a matching source cleanup/store/return site in frame {}",
                c.name,
                frame
            );
            at += 4
        }
        let heap = &w[at..];
        assert_eq!(heap.len(), self.layout.heap_words());
        let mut allocation_events = o
            .events
            .iter()
            .filter(|e| e.kind == oracle::EventKind::Complete && allocates(&p, e))
            .count();
        if o.failure == Some(oracle::ToolFailure::TraceLimit)
            && o.events
                .last()
                .is_some_and(|e| e.kind == oracle::EventKind::Demand && allocates(&p, e))
        {
            allocation_events += 1
        };
        assert_eq!(
            w[9], allocation_events as i64,
            "all stable allocation IDs accounted by actual logical allocation events"
        );
        let root = match &o.semantic.result {
            Ok(eval::Value::F32(x)) => {
                assert!((0..=u32::MAX as i64).contains(&w[1]));
                assert!(x.strict_eq(F32Bits::from_bits(w[1] as u32)));
                Json::Null
            }
            Ok(eval::Value::ArrayF32(xs)) => {
                let d = usize::try_from(w[1] - 1).unwrap() * 4;
                assert_eq!(heap[d + 2], 258);
                assert_eq!(heap[d + 3], 1);
                let start = self.layout.allocations() * 4 + heap[d] as usize;
                let len = heap[d + 1] as usize;
                assert_eq!(len, xs.borrow().len());
                for (a, b) in heap[start..start + len].iter().zip(xs.borrow().iter()) {
                    assert!((0..=u32::MAX as i64).contains(a));
                    assert!(b.strict_eq(F32Bits::from_bits(*a as u32)))
                }
                json!({"handle":w[1],"kind":heap[d+2],"length":len,"bits":heap[start..start+len]})
            }
            Ok(eval::Value::I32(x)) => {
                assert_eq!(w[1], i64::from(*x));
                Json::Null
            }
            Ok(eval::Value::U32(x)) => {
                assert_eq!(w[1], i64::from(*x));
                Json::Null
            }
            Err(_) => {
                assert_eq!(w[1], 0);
                Json::Null
            }
            x => panic!("{x:?}"),
        };
        json!({"artifact":self.metadata,"header":w[..10],"root":root,"semantic":result(&o.semantic),"cleanup":o.semantic.cleanup_trace.iter().map(|x|json!({"name":x.name,"length":x.length})).collect::<Vec<_>>(),"all_words_le_i64_sha256":digest(&w.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>())})
    }
}
fn cleanup_site(p: &mir::Program, e: &oracle::Event, name: &str) -> bool {
    let b = &p.functions[e.token.block.function].blocks[e.token.block.block];
    match b.instructions.get(e.token.position).map(|i| &i.kind) {
        Some(mir::Operation::Drop(names)) => names.iter().any(|n| n == name),
        Some(mir::Operation::Bind { name: n, .. } | mir::Operation::Store { name: n, .. }) => {
            n == name
        }
        None => {
            matches!(&b.end,Some(mir::Terminator::Return{cleanup,..}) if cleanup.iter().any(|n|n==name))
        }
        _ => false,
    }
}
fn allocates(p: &mir::Program, e: &oracle::Event) -> bool {
    let Some(i) = p.functions[e.token.block.function].blocks[e.token.block.block]
        .instructions
        .get(e.token.position)
    else {
        return false;
    };
    matches!(i.kind, mir::Operation::Array(_))
        || matches!(&i.kind,mir::Operation::Call{name,..} if matches!(name.as_str(),"filled_i32"|"filled_u32"|"filled_f32"))
}
impl Drop for Cpu {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap()
    }
}

#[test]
fn issue_gnet_10_source_mir_oracle_native_every_fuel_prefix() {
    let names = [
        "literals.inv",
        "filled_empty.inv",
        "update_fma.inv",
        "loans_moves.inv",
        "replacement.inv",
        "mixed_relocation.inv",
        "root_return.inv",
        "element_copy.inv",
        "index_trap.inv",
        "store_trap.inv",
        "older_integer_trap.inv",
        "array_size_trap.inv",
    ];
    let mut report_rows = Vec::new();
    let mut comparisons = 0;
    for name in names {
        let s = source(name);
        let c = checked(&s);
        let m = mir::lower(&c).unwrap();
        let n = net(&s);
        let done = eval::observe(&c, 1000);
        if name.contains("trap") {
            assert!(done.result.is_err())
        } else {
            assert_eq!(done.result, Ok(eval::Value::I32(7)))
        };
        let elements = if name == "mixed_relocation.inv" {
            3
        } else {
            128
        };
        let h = Cpu::new(
            &n,
            "main",
            compiled::OwnedLimits {
                elements,
                ..storage()
            },
        );
        let mut batch = Vec::new();
        let mut expected = Vec::new();
        for fuel in 1..=done.steps_used + 1 {
            let s = eval::observe(&c, fuel);
            equal(&s, &mir_eval::observe(&m, fuel));
            let l = oracle::Limits {
                fuel,
                array_elements: elements,
                ..limits()
            };
            let o = oracle::observe(&n, &n.verify().unwrap(), l).unwrap();
            assert_eq!(o.failure, None);
            equal(&s, &o.semantic);
            batch.push((l, vec![]));
            expected.push(o)
        }
        let mut observations = Vec::new();
        for (w, o) in h.batch(&batch, "normal").iter().zip(&expected) {
            observations.push(h.compare(&n, o, w));
            comparisons += 1
        }
        report_rows.push(json!({"source":name,"source_sha256":digest(s.as_bytes()),"source_complete":result(&done),"positive_fuel_prefixes":observations}));
    }
    eprintln!(
        "ABI10 source/MIR/oracle/native:{comparisons} exact positive fuel observations across12 source programs"
    );
    report(
        "source-prefixes",
        json!({"comparisons":comparisons,"programs":12,"cases":report_rows}),
    );
}
#[test]
fn issue_gnet_10_runtime_bits_root_returns_and_relocation() {
    let s = source("runtime.inv");
    let c = checked(&s);
    let n = net(&s);
    let values = [
        0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x3f800000, 0x7f7fffff, 0xff7fffff, 0x7f800000,
        0xff800000, 0x7fa00001, 0xffc12345,
    ];
    let mut rows = Vec::new();
    let mut comparisons = 0;
    for entry in ["run", "make"] {
        let h = Cpu::new(&n, entry, storage());
        let mut cases = Vec::new();
        let mut expected = Vec::new();
        let mut inputs = Vec::new();
        for bits in values {
            for length in [1, 3, 7] {
                let args = vec![
                    eval::Value::U32(length),
                    eval::Value::F32(F32Bits::from_bits(bits)),
                ];
                let o =
                    oracle::observe_call(&n, &n.verify().unwrap(), entry, &args, limits()).unwrap();
                assert_eq!(o.failure, None);
                if entry == "run" {
                    let a = eval::named::observe_numeric_named(&c, entry, &args, 1000);
                    equal(&a, &o.semantic);
                    assert!(matches!(a.result,Ok(eval::Value::F32(x)) if x.to_bits()==bits))
                }
                cases.push((limits(), carriers(&args)));
                expected.push(o);
                inputs.push(json!({"length":length,"bits":bits}))
            }
        }
        if entry == "make" {
            let args = vec![eval::Value::U32(0), eval::Value::F32(F32Bits::NEG_ZERO)];
            cases.push((limits(), carriers(&args)));
            expected.push(
                oracle::observe_call(&n, &n.verify().unwrap(), entry, &args, limits()).unwrap(),
            );
            inputs.push(json!({"length":0,"bits":2147483648u32}))
        };
        for ((w, o), input) in h.batch(&cases, "normal").iter().zip(&expected).zip(&inputs) {
            let receipt = h.compare(&n, o, w);
            if entry == "run" {
                assert_eq!(w[1] as u64, input["bits"].as_u64().unwrap())
            } else {
                assert!(
                    receipt["root"]["bits"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|x| x.as_u64() == input["bits"].as_u64())
                )
            };
            rows.push(json!({"entry":entry,"input":input,"source_named_reference":if entry=="run"{"actual numeric scalar source observation"}else{"array return signature unavailable; ordinary source root fixture qualifies AST/MIR returns"},"observation":receipt}));
            comparisons += 1
        }
    }
    let root_net = net(&source("root_return.inv"));
    let h = Cpu::new(&root_net, "make", storage());
    let o = oracle::observe_call(
        &root_net,
        &root_net.verify().unwrap(),
        "make",
        &[],
        limits(),
    )
    .unwrap();
    assert_eq!(
        result(&o.semantic),
        json!({"ArrayF32":[2147483648u32,2143289344u32,1065353216u32]})
    );
    let w = h.batch(&[(limits(), vec![])], "normal").pop().unwrap();
    let receipt = h.compare(&root_net, &o, &w);
    assert_eq!(
        receipt["root"]["bits"],
        json!([2147483648u32, 2143289344u32, 1065353216u32])
    );
    rows.push(json!({"entry":"make","source":"root_return.inv","source_reference":"source/MIR owner-producing function qualified through ordinary main at every fuel; direct owner root snapshot uses oracle and manual bits","observation":receipt}));
    comparisons += 1;
    eprintln!(
        "ABI10 runtime copy/roots:{comparisons} actual native observations including exact signaling-NaN payloads and empty owner"
    );
    report(
        "runtime-bits",
        json!({"comparisons":comparisons,"rows":rows}),
    );
}
#[test]
fn issue_gnet_10_private_storage_trace_transition_and_cleanup_caps() {
    let s = source("mixed_relocation.inv");
    let n = net(&s);
    let h = Cpu::new(&n, "main", storage());
    let mut rows = Vec::new();
    let mut comparisons = 0;
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
            let o = oracle::observe(&n, &n.verify().unwrap(), l).unwrap();
            let w = h.batch(&[(l, vec![])], "normal").pop().unwrap();
            rows.push(h.compare(&n, &o, &w));
            comparisons += 1
        }
    }
    let heap = Cpu::new(
        &n,
        "main",
        compiled::OwnedLimits {
            elements: 2,
            ..storage()
        },
    );
    let l = oracle::Limits {
        array_elements: 2,
        ..limits()
    };
    let o = oracle::observe(&n, &n.verify().unwrap(), l).unwrap();
    assert_eq!(o.failure, Some(oracle::ToolFailure::Allocation));
    let w = heap.batch(&[(l, vec![])], "normal").pop().unwrap();
    rows.push(heap.compare(&n, &o, &w));
    comparisons += 1;
    let complete = oracle::observe(&n, &n.verify().unwrap(), limits()).unwrap();
    let p = n.compiled_mir(&n.verify().unwrap()).unwrap();
    let mut failures = Vec::new();
    for (s, code) in [
        (
            compiled::OwnedLimits {
                allocations: 2,
                ..storage()
            },
            12,
        ),
        (
            compiled::OwnedLimits {
                cleanup_events: 0,
                ..storage()
            },
            13,
        ),
        (
            compiled::OwnedLimits {
                cleanup_events: 1,
                ..storage()
            },
            13,
        ),
    ] {
        let h = Cpu::new(&n, "main", s);
        let w = h.batch(&[(limits(), vec![])], "normal").pop().unwrap();
        assert_eq!(w[0], code);
        let event_count = w[7] as usize;
        assert!(event_count < complete.events.len());
        for (actual, e) in w[10..10 + event_count * 7]
            .chunks_exact(7)
            .zip(&complete.events)
        {
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
        let at = 10 + event_count * 7;
        for (q, c) in w[at..at + w[8] as usize * 4]
            .chunks_exact(4)
            .zip(&complete.semantic.cleanup_trace)
        {
            assert_eq!(
                p.functions[q[0] as usize].declaration_order[q[1] as usize],
                c.name
            );
            assert_eq!(q[2], c.length as i64)
        }
        failures.push(json!({"status":code,"header":w[..10],"full_source_and_oracle_status":"success; private backend storage failure is separate","exact_committed_trace_cleanup_prefix":true,"artifact":h.metadata}))
    }
    report(
        "budgets",
        json!({"exact_oracle_comparisons":comparisons,"private_storage_failures":failures,"observations":rows}),
    );
}
#[test]
fn issue_gnet_10_malformed_private_heap_and_bit_writes_fail_closed() {
    let s = source("runtime.inv");
    let n = net(&s);
    let h = Cpu::new(&n, "make", storage());
    let batch = [(limits(), vec![1, 1065353216])];
    let mut rows = Vec::new();
    for mode in [
        "kind",
        "offset",
        "length",
        "flag",
        "bits",
        "allocated",
        "handle",
        "bounds",
        "store",
    ] {
        let w = h.batch(&batch, mode).pop().unwrap();
        assert_eq!(w, [if mode == "bounds" { 9 } else { 7 }, 777]);
        rows.push(json!({"mode":mode,"status":w[0],"heap_and_output_sentinel_unchanged":true,"artifact":h.metadata}))
    }
    report(
        "malformed-heap",
        json!({"actual_native_fail_closed_cases":rows.len(),"rows":rows}),
    );
}
#[test]
fn issue_gnet_10_profile_types_proofs_limits_and_legacy_rejections() {
    let n = net(&source("literals.inv"));
    let proof = n.verify().unwrap();
    let numeric = |n: &Net, proof: &general_net::Proof, entry: &str, target, s| {
        compiled::emit_resumable_owned_numeric(
            n,
            proof,
            entry,
            target,
            compiled::Limits::default(),
            s,
        )
    };
    let a = numeric(&n, &proof, "main", compiled::Target::Cpu, storage()).unwrap();
    assert_eq!(a.abi_version(), 11);
    let mut rejections = 0;
    assert_eq!(
        numeric(&n, &proof, "main", compiled::Target::Metal, storage()).unwrap_err(),
        compiled::Error::NumericTarget
    );
    rejections += 1;
    for result in [
        compiled::emit(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            compiled::Limits::default(),
        ),
        compiled::emit_owned(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            compiled::Limits::default(),
            storage(),
        ),
        compiled::emit_resumable(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            compiled::Limits::default(),
            storage(),
        ),
        compiled::emit_speculative(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            compiled::Limits::default(),
            storage(),
            compiled::SpeculationLimits::default(),
        ),
        compiled::emit_aggregates(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            compiled::Limits::default(),
            storage(),
        ),
    ] {
        assert_eq!(result.unwrap_err(), compiled::Error::UnsupportedType);
        rejections += 1
    }
    for source in [
        "fn entry(xs:read Array<f32>)->i32{return 7;}",
        "struct Numeric{x:f32} fn main()->i32{return 7;}",
        "struct Old{x:i32} fn main()->i32{return 7;}",
    ] {
        let c = checked(source);
        let n = Net::lower(&c, general_net::Limits::default()).unwrap();
        assert_eq!(
            numeric(
                &n,
                &n.verify().unwrap(),
                if source.contains("entry") {
                    "entry"
                } else {
                    "main"
                },
                compiled::Target::Cpu,
                storage()
            )
            .unwrap_err(),
            compiled::Error::UnsupportedType
        );
        rejections += 1
    }
    let other = net("fn main()->i32{return 7;}");
    assert!(matches!(
        numeric(&other, &proof, "main", compiled::Target::Cpu, storage()),
        Err(compiled::Error::Boundary(_))
    ));
    rejections += 1;
    for s in [
        compiled::OwnedLimits {
            allocations: 0,
            ..storage()
        },
        compiled::OwnedLimits {
            elements: invar::array_limits::MAX_LIVE_ELEMENTS + 1,
            ..storage()
        },
        compiled::OwnedLimits {
            allocations: 10001,
            ..storage()
        },
        compiled::OwnedLimits {
            cleanup_events: 100001,
            ..storage()
        },
    ] {
        assert_eq!(
            numeric(&n, &proof, "main", compiled::Target::Cpu, s).unwrap_err(),
            compiled::Error::Limit
        );
        rejections += 1
    }
    let h = Cpu::new(&net(&source("runtime.inv")), "run", storage());
    for xs in [vec![1, -1], vec![1, u32::MAX as i64 + 1]] {
        let w = h.batch(&[(limits(), xs)], "normal").pop().unwrap();
        assert_eq!(w[0], 8);
        assert_eq!(w[4], 0);
        assert_eq!(w[7], 0);
        assert_eq!(w[8], 0);
        assert_eq!(w[9], 0);
        rejections += 1
    }
    report(
        "admission",
        json!({"actual_rejections":rejections,"abi11_metadata":h.metadata}),
    );
}
/// A record that owns an `Array<f32>` and is released by reassignment never reaches emitted
/// code: every compiled entry refuses it, so `gnet_leaf_length` (record array fields, kinds 0
/// and 1) never sees an f32 array (kind 258). The `Array<i32>` control shows the refusal comes
/// from the f32 field, not from the record shape.
#[test]
fn issue_gnet_10_record_owning_f32_array_is_refused_by_every_compiled_entry() {
    let shape = |ty: &str, a: &str, b: &str| {
        format!(
            "struct Signal {{ xs: Array<{ty}>, code: i32 }} fn main() -> i32 {{ var s = Signal {{ xs: [{a}], code: 7 }}; s = Signal {{ xs: [{b}], code: 9 }}; return s.code; }}"
        )
    };
    let control = net(&shape("i32", "1, 2", "3"));
    let proof = control.verify().unwrap();
    compiled::emit_aggregates(
        &control,
        &proof,
        "main",
        compiled::Target::Cpu,
        compiled::Limits::default(),
        storage(),
    )
    .unwrap();
    let n = net(&shape("f32", "1.0, 2.0", "3.0"));
    let proof = n.verify().unwrap();
    let limits = compiled::Limits::default;
    let mut results = vec![
        compiled::emit_aggregates(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            limits(),
            storage(),
        ),
        compiled::emit_resumable_aggregates(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            limits(),
            storage(),
        ),
        compiled::emit_resumable_owned_numeric(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            limits(),
            storage(),
        ),
    ];
    for target in [
        compiled::Target::Cpu,
        compiled::Target::Cuda,
        compiled::Target::Metal,
    ] {
        results.push(compiled::emit_resumable_owned_numeric(
            &n,
            &proof,
            "main",
            target,
            limits(),
            storage(),
        ));
        results.push(compiled::emit_device_speculative(
            &n,
            &proof,
            "main",
            target,
            limits(),
            storage(),
            compiled::SpeculationLimits::default(),
        ));
    }
    // Device targets may refuse the numeric target before any layout is built; either way no
    // code is emitted.
    for (i, result) in results.into_iter().enumerate() {
        let error = result.expect_err("a record owning Array<f32> must not compile");
        // The first four are the CPU aggregate and owned-numeric layouts the leaf helper serves.
        if i < 4 {
            assert_eq!(error, compiled::Error::UnsupportedType, "entry {i}");
        }
        assert!(
            matches!(
                error,
                compiled::Error::UnsupportedType | compiled::Error::NumericTarget
            ),
            "entry {i}: {error:?}"
        );
    }
}
