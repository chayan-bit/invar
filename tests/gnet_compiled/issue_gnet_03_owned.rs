#![forbid(unsafe_code)]

use crate::child_deadline;
use general_net::{Net, compiled, oracle};
use invar::{check, eval, front, general_net};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

fn lowered(source: &str) -> Net {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    Net::lower(&checked, general_net::Limits::default()).unwrap()
}

struct Cpu {
    dir: PathBuf,
    layout: compiled::OwnedLayout,
}
impl Cpu {
    fn new(net: &Net, entry: &str, storage: compiled::OwnedLimits) -> Self {
        Self::build(net, entry, storage, None)
    }
    fn resumable(net: &Net, entry: &str, quantum: u64) -> Self {
        Self::build(
            net,
            entry,
            compiled::OwnedLimits {
                elements: 128,
                allocations: 64,
                cleanup_events: 128,
            },
            Some(quantum),
        )
    }
    fn build(net: &Net, entry: &str, storage: compiled::OwnedLimits, quantum: Option<u64>) -> Self {
        let emitter = if quantum.is_some() {
            compiled::emit_resumable
        } else {
            compiled::emit_owned
        };
        let artifact = emitter(
            net,
            &net.verify().unwrap(),
            entry,
            compiled::Target::Cpu,
            compiled::Limits::default(),
            storage,
        )
        .unwrap();
        assert_eq!(
            artifact.abi_version(),
            if quantum.is_some() { 3 } else { 2 }
        );
        let run = if let Some(quantum) = quantum {
            let bytes = artifact.continuation_layout().unwrap().task_bytes();
            format!(
                "static_assert(sizeof(GnetTask)=={bytes}); GnetTask task={{}}; unsigned rounds=0; while(!gnet_resume(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data(),&task,{quantum})) {{ if(++rounds>1000001) return 3; }} auto before=task.transitions; if(!gnet_resume(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data(),&task,{quantum}) || task.transitions!=before) return 4;"
            )
        } else {
            "gnet_run(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data());"
                .to_owned()
        };
        let layout = artifact.owned_layout().unwrap().clone();
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "invar-owned-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let driver = format!(
            r#"
#include <iostream>
#include <vector>
#include <string>
int main(int argc,char** argv) {{
    gnet_u fuel=std::stoull(argv[1]),trans=std::stoull(argv[2]),cap=std::stoull(argv[3]);
    std::vector<gnet_w> input; for(int i=4;i<argc;++i) input.push_back(std::stoll(argv[i]));
    gnet_w output[10]={{}}; std::vector<gnet_u> trace(cap*7),cleanup({clean}); std::vector<gnet_w> heap({heap});
    {run}
    for(auto word:output) std::cout<<word<<' ';
    for(gnet_u i=0;i<(gnet_u)output[7]*7;++i) std::cout<<trace[i]<<' ';
    for(gnet_u i=0;i<(gnet_u)output[8]*4;++i) std::cout<<cleanup[i]<<' ';
    for(auto word:heap) std::cout<<word<<' ';
}}
"#,
            run = run,
            clean = layout.cleanup_words(),
            heap = layout.heap_words()
        );
        fs::write(
            dir.join("source.cpp"),
            format!("{}\n{driver}", artifact.source()),
        )
        .unwrap();
        child_deadline::cxx(
            child_deadline::CXX17_UBSAN_O1,
            &dir.join("source.cpp"),
            &dir.join("run"),
        );
        Self { dir, layout }
    }
    fn run(&self, args: &[i64], limits: oracle::Limits) -> Vec<i64> {
        let output = Command::new(self.dir.join("run"))
            .args([
                limits.fuel.to_string(),
                limits.transitions.to_string(),
                limits.trace_events.to_string(),
            ])
            .args(args.iter().map(ToString::to_string))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .unwrap()
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect()
    }
    fn compare(
        &self,
        net: &Net,
        entry: &str,
        args: &[eval::Value],
        limits: oracle::Limits,
    ) -> oracle::Observation {
        let expected =
            oracle::observe_call(net, &net.verify().unwrap(), entry, args, limits).unwrap();
        let inputs: Vec<_> = args
            .iter()
            .map(|x| match x {
                eval::Value::U32(n) => i64::from(*n),
                eval::Value::I32(n) => i64::from(*n),
                _ => panic!("scalar"),
            })
            .collect();
        let words = self.run(&inputs, limits);
        let (status, start, end) = match &expected.semantic.result {
            Ok(_) => (0, 0, 0),
            Err(eval::EvalError::Trap { category, span }) => (
                match category {
                    eval::TrapCategory::IndexOutOfBounds => 9,
                    eval::TrapCategory::IntegerOverflow => 1,
                    eval::TrapCategory::DivisionByZero => 2,
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
            Err(_) => (
                match expected.failure {
                    Some(oracle::ToolFailure::TransitionLimit) => 5,
                    Some(oracle::ToolFailure::TraceLimit) => 6,
                    _ => panic!("{expected:?}"),
                },
                0,
                0,
            ),
        };
        assert_eq!((words[0], words[2], words[3]), (status, start, end));
        assert_eq!(
            &words[4..9],
            &[
                expected.semantic.steps_used as i64,
                expected.transitions as i64,
                expected.max_frames as i64,
                expected.events.len() as i64,
                expected.semantic.cleanup_trace.len() as i64
            ]
        );
        let mut offset = 10;
        for event in &expected.events {
            assert_eq!(
                &words[offset..offset + 7],
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
            offset += 7;
        }
        let program = net.compiled_mir(&net.verify().unwrap()).unwrap();
        for event in &expected.semantic.cleanup_trace {
            let f = words[offset] as usize;
            let place = words[offset + 1] as usize;
            assert_eq!(program.functions[f].declaration_order[place], event.name);
            assert_eq!(words[offset + 2], event.length as i64);
            offset += 4;
        }
        let heap = &words[offset..];
        assert_eq!(heap.len(), self.layout.heap_words());
        match &expected.semantic.result {
            Ok(eval::Value::I32(n)) => assert_eq!(words[1], i64::from(*n)),
            Ok(eval::Value::U32(n)) => assert_eq!(words[1], i64::from(*n)),
            Ok(eval::Value::ArrayI32(v)) => assert_eq!(
                self.result_array(words[1], heap),
                v.borrow().iter().map(|x| i64::from(*x)).collect::<Vec<_>>()
            ),
            Ok(eval::Value::ArrayU32(v)) => assert_eq!(
                self.result_array(words[1], heap),
                v.borrow().iter().map(|x| i64::from(*x)).collect::<Vec<_>>()
            ),
            Err(_) => assert_eq!(words[1], 0),
            x => panic!("{x:?}"),
        }
        expected
    }
    fn result_array(&self, handle: i64, heap: &[i64]) -> Vec<i64> {
        assert!(handle > 0);
        let d = (handle as usize - 1) * 4;
        assert_eq!(heap[d + 3], 1);
        let start = self.layout.allocations() * 4 + heap[d] as usize;
        heap[start..start + heap[d + 1] as usize].to_vec()
    }
}
impl Drop for Cpu {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap();
    }
}

#[test]
fn issue_gnet_03_worklist_is_fair_bounded_and_keeps_independent_root_observations() {
    let net = lowered(
        "fn run(n: u32) -> i32 { if n == 0u32 { while true {} } if n == 1u32 { return 17; } let xs: Array<i32> = [1]; return xs[1u32]; } fn main() -> i32 { return run(1u32); }",
    );
    let storage = compiled::OwnedLimits {
        elements: 16,
        allocations: 8,
        cleanup_events: 8,
    };
    let limits = oracle::Limits {
        fuel: 100,
        transitions: 10000,
        trace_events: 10000,
        ..oracle::Limits::default()
    };
    let artifact = compiled::emit_resumable(
        &net,
        &net.verify().unwrap(),
        "run",
        compiled::Target::Cpu,
        compiled::Limits::default(),
        storage,
    )
    .unwrap();
    let layout = artifact.owned_layout().unwrap();
    let dir = std::env::temp_dir().join(format!("invar-worklist-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let mut checks = String::new();
    for lane in 0..3u32 {
        let observation = oracle::observe_call(
            &net,
            &net.verify().unwrap(),
            "run",
            &[eval::Value::U32(lane)],
            limits,
        )
        .unwrap();
        let (status, result, start, end) = match &observation.semantic.result {
            Ok(eval::Value::I32(n)) => (0, i64::from(*n), 0, 0),
            Err(eval::EvalError::Trap { span, .. }) => (9, 0, span.start as i64, span.end as i64),
            Err(eval::EvalError::EvaluationLimit { .. }) => (3, 0, 0, 0),
            x => panic!("{x:?}"),
        };
        let header = [
            status,
            result,
            start,
            end,
            observation.semantic.steps_used as i64,
            observation.transitions as i64,
            observation.max_frames as i64,
            observation.events.len() as i64,
            observation.semantic.cleanup_trace.len() as i64,
        ];
        for (j, word) in header.iter().enumerate() {
            checks.push_str(&format!("if(output[{lane}*10+{j}]!={word}) return 20;\n"));
        }
        for (i, event) in observation.events.iter().enumerate() {
            let words = [
                i64::from(event.kind == oracle::EventKind::Complete),
                event.frame as i64,
                event.activation as i64,
                event.token.block.function as i64,
                event.token.block.block as i64,
                event.token.position as i64,
                event.rule as i64,
            ];
            for (j, word) in words.iter().enumerate() {
                checks.push_str(&format!(
                    "if(trace[{lane}*10000*7+{i}*7+{j}]!={word}) return 21;\n"
                ));
            }
        }
    }
    let driver = format!(
        r#"
#include <vector>
int main() {{
    gnet_w input[3]={{0,1,2}}, output[30]={{}}; GnetTask tasks[3]={{}};
    std::vector<gnet_u> trace(3*10000*7),cleanup(3*{clean}); std::vector<gnet_w> heap(3*{heap});
    gnet_u submitted=0,completed=0;
    if(gnet_worklist(input,output,trace.data(),3,100,10000,10000,heap.data(),cleanup.data(),tasks,1,6,submitted,completed)!=14 || submitted!=6 || completed!=0) return 10;
    for(auto& task:tasks) if(task.transitions!=2 || task.done) return 11;
    for(auto word:output) if(word!=0) return 12;
    if(gnet_worklist(input,output,trace.data(),3,100,10000,10000,heap.data(),cleanup.data(),tasks,3,10000,submitted,completed)!=0 || completed!=3) return 13;
    {checks}
    if(gnet_worklist(input,output,trace.data(),3,100,10000,10000,heap.data(),cleanup.data(),tasks,3,0,submitted,completed)!=0 || submitted!=0 || completed!=3) return 14;
    if(gnet_worklist(input,output,trace.data(),0,100,10000,10000,heap.data(),cleanup.data(),tasks,3,0,submitted,completed)!=0) return 15;
    if(gnet_worklist(input,output,trace.data(),1025,100,10000,10000,heap.data(),cleanup.data(),tasks,3,0,submitted,completed)!=8) return 16;
    GnetTask zero_trace={{}}; gnet_w limited[10]={{}};
    if(gnet_worklist(input,limited,nullptr,1,100,10000,0,heap.data(),cleanup.data(),&zero_trace,1,100,submitted,completed)!=0 || limited[0]!=6 || limited[7]!=0) return 17;
    return 0;
}}
"#,
        clean = layout.cleanup_words(),
        heap = layout.heap_words()
    );
    fs::write(
        dir.join("source.cpp"),
        format!("{}\n{driver}", artifact.source()),
    )
    .unwrap();
    child_deadline::cxx(
        child_deadline::CXX17_UBSAN_O1,
        &dir.join("source.cpp"),
        &dir.join("run"),
    );
    let result = Command::new(dir.join("run")).status().unwrap();
    assert!(result.success(), "worklist {result}");
    fs::remove_dir_all(dir).unwrap();
    for target in [compiled::Target::Cuda, compiled::Target::Metal] {
        let artifact = compiled::emit_resumable(
            &net,
            &net.verify().unwrap(),
            "run",
            target,
            compiled::Limits::default(),
            storage,
        )
        .unwrap();
        assert!(artifact.source().contains("invar_gnet_resume_v3"));
        assert!(artifact.source().contains("tasks+lane"));
        assert_eq!(artifact.continuation_layout().unwrap().config_words(), 5);
    }
}

#[test]
fn issue_gnet_03_owned_operations_loans_moves_calls_returns_cleanup() {
    let net = lowered(
        "fn make() -> Array<u32> { let xs = filled_u32(2u32, 4294967295u32); return xs; } fn main() -> i32 { let xs = make(); return 0; }",
    );
    Cpu::new(&net, "make", compiled::OwnedLimits::default()).compare(
        &net,
        "make",
        &[],
        oracle::Limits::default(),
    );

    for source in [
        "fn main() -> i32 { var xs: Array<i32> = [1,2]; xs[0u32] = xs[1u32] + 3; return xs[0u32]; }",
        "fn mutate(xs: mut Array<i32>) -> Unit { xs[0u32] = 9; } fn peek(xs: read Array<i32>) -> i32 { return xs[0u32]; } fn main() -> i32 { var xs: Array<i32> = [1]; mutate(xs); return peek(xs); }",
        "fn consume(xs: take Array<i32>) -> i32 { return xs[0u32]; } fn main() -> i32 { let a: Array<i32> = [4]; let b = move a; return consume(move b); }",
        "fn make() -> Array<i32> { let xs: Array<i32> = [3,5]; return xs; } fn main() -> i32 { let a = make(); return a[1u32]; }",
        "fn main() -> i32 { var xs = filled_i32(2u32, 3); xs = filled_i32(1u32, 9); return xs[0u32]; }",
        "fn main() -> i32 { let xs: Array<u32> = []; if len(xs) == 0u32 { return 7; } return 1 / 0; }",
        include_str!("../fixtures/l1_03_generic_array.inv"),
        include_str!("../fixtures/l1_03_closures.inv"),
    ] {
        let net = lowered(source);
        let cpu = Cpu::new(&net, "main", compiled::OwnedLimits::default());
        cpu.compare(&net, "main", &[], oracle::Limits::default());
    }
}

#[test]
fn issue_gnet_03_owned_relocation_preserves_loan_handles_and_reclaims_storage() {
    let source = "fn loan(xs: mut Array<i32>) -> i32 { for i in 0u32..20u32 { let ys = filled_i32(2u32, 3); xs[0u32] = xs[0u32] + ys[0u32]; } return xs[0u32]; } fn main() -> i32 { if true { let dead = filled_i32(2u32, 0); } var xs: Array<i32> = [1]; return loan(xs); }";
    let net = lowered(source);
    let cpu = Cpu::new(
        &net,
        "main",
        compiled::OwnedLimits {
            elements: 3,
            ..compiled::OwnedLimits::default()
        },
    );
    let result = cpu.compare(&net, "main", &[], oracle::Limits::default());
    assert_eq!(result.semantic.result, Ok(eval::Value::I32(61)));
    assert_eq!(result.semantic.cleanup_trace.len(), 22);
}

#[test]
fn issue_gnet_03_owned_traps_fuel_and_logical_event_prefixes() {
    for source in [
        "fn main() -> i32 { if true { let done: Array<i32> = [1,2]; } let xs: Array<i32> = [1]; return xs[1u32]; }",
        "fn main() -> i32 { var xs: Array<i32> = [1]; xs[1u32] = 9; return 0; }",
        "fn main() -> i32 { let xs = filled_i32(16777217u32, 0); return 0; }",
        "fn main() -> i32 { var xs: Array<i32> = [1,2]; for i in 0u32..2u32 { xs[i] = xs[i] + 3; } return xs[0u32]; }",
        "fn make() -> Array<i32> { let xs: Array<i32> = [3,5]; return xs; } fn main() -> i32 { let a = make(); return a[1u32]; }",
    ] {
        let net = lowered(source);
        let cpu = Cpu::new(&net, "main", compiled::OwnedLimits::default());
        let full = cpu.compare(&net, "main", &[], oracle::Limits::default());
        for fuel in 1..=full.semantic.steps_used {
            cpu.compare(
                &net,
                "main",
                &[],
                oracle::Limits {
                    fuel,
                    ..oracle::Limits::default()
                },
            );
        }
        for cap in [0, 1, 2, 7, 21] {
            cpu.compare(
                &net,
                "main",
                &[],
                oracle::Limits {
                    transitions: cap,
                    ..oracle::Limits::default()
                },
            );
            cpu.compare(
                &net,
                "main",
                &[],
                oracle::Limits {
                    trace_events: cap as usize,
                    ..oracle::Limits::default()
                },
            );
        }
    }
}

#[test]
fn issue_gnet_03_owned_backend_storage_limits_have_distinct_statuses() {
    for (source, limits, status) in [
        (
            "fn main() -> i32 { let xs = filled_i32(4u32, 0); return 0; }",
            compiled::OwnedLimits {
                elements: 3,
                ..compiled::OwnedLimits::default()
            },
            11,
        ),
        (
            "fn main() -> i32 { for i in 0u32..3u32 { let xs: Array<i32> = []; } return 0; }",
            compiled::OwnedLimits {
                allocations: 2,
                ..compiled::OwnedLimits::default()
            },
            12,
        ),
        (
            "fn main() -> i32 { let xs: Array<i32> = [1]; return 0; }",
            compiled::OwnedLimits {
                cleanup_events: 0,
                ..compiled::OwnedLimits::default()
            },
            13,
        ),
    ] {
        let net = lowered(source);
        let cpu = Cpu::new(&net, "main", limits);
        let words = cpu.run(&[], oracle::Limits::default());
        assert_eq!(words[0], status);
        assert_eq!(words[1], 0);
    }
    let net = lowered("fn main() -> i32 { return 1; }");
    let proof = net.verify().unwrap();
    assert_eq!(
        compiled::emit_owned(
            &net,
            &proof,
            "main",
            compiled::Target::Cpu,
            compiled::Limits::default(),
            compiled::OwnedLimits {
                allocations: 0,
                ..compiled::OwnedLimits::default()
            }
        )
        .unwrap_err(),
        compiled::Error::Limit
    );
    for target in [compiled::Target::Cuda, compiled::Target::Metal] {
        let artifact = compiled::emit_owned(
            &net,
            &proof,
            "main",
            target,
            compiled::Limits::default(),
            compiled::OwnedLimits::default(),
        )
        .unwrap();
        assert!(artifact.source().contains("invar_gnet_owned_v2"));
        assert!(artifact.source().contains("heap"));
    }
}

#[test]
fn issue_gnet_03_resumable_frames_heap_and_event_prefixes_match_every_quantum() {
    for source in [
        "fn fib(n: u32) -> i32 { if n < 2u32 { return 1; } return fib(n - 1u32) + fib(n - 2u32); } fn main() -> i32 { return fib(5u32); }",
        "fn consume(xs: take Array<i32>, n: u32) -> i32 { if n == 0u32 { return xs[0u32]; } return consume(move xs,n - 1u32); } fn main() -> i32 { let xs: Array<i32> = [7]; return consume(move xs,3u32); }",
        "fn mutate(xs: mut Array<i32>) -> Unit { for i in 0u32..3u32 { let tmp=filled_i32(2u32,3); xs[0u32]=xs[0u32]+tmp[1u32]; } } fn main() -> i32 { var xs: Array<i32> = [1]; mutate(xs); return xs[0u32]; }",
        "fn main() -> i32 { if true { let done: Array<i32> = [1]; } let xs: Array<i32> = [2]; return xs[1u32]; }",
        "fn oldest() -> i32 { while true {} return 0; } fn later() -> i32 { return 1/0; } fn main() -> i32 { return oldest()+later(); }",
    ] {
        let net = lowered(source);
        for quantum in [1, 3, 17] {
            let cpu = Cpu::resumable(&net, "main", quantum);
            let limits = oracle::Limits {
                fuel: 1000,
                trace_events: 10000,
                transitions: 10000,
                ..oracle::Limits::default()
            };
            cpu.compare(&net, "main", &[], limits);
            for fuel in [1, 2, 7, 18, 40] {
                cpu.compare(&net, "main", &[], oracle::Limits { fuel, ..limits });
            }
            for cap in [0, 1, 7, 20, 80] {
                cpu.compare(
                    &net,
                    "main",
                    &[],
                    oracle::Limits {
                        transitions: cap,
                        ..limits
                    },
                );
                cpu.compare(
                    &net,
                    "main",
                    &[],
                    oracle::Limits {
                        trace_events: cap as usize,
                        ..limits
                    },
                );
            }
        }
    }
}
