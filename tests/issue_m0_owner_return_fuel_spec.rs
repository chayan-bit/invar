#![forbid(unsafe_code)]
use invar::{
    check, eval, front,
    general_net::{self, oracle},
    mir, mir_eval, ownership,
};
fn checked(s: &str) -> check::CheckedProgram {
    let p = check::check(front::parse(s.as_bytes()).unwrap(), true).unwrap();
    ownership::check(&p).unwrap();
    p
}
fn same(a: &eval::Observation, b: &eval::Observation) {
    assert_eq!(a.result, b.result);
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace);
}
fn qualify(source: &str) -> eval::Observation {
    let p = checked(source);
    let m = mir::lower(&p).unwrap();
    let net = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let full = eval::observe(&p, 10000);
    for fuel in 1..=full.steps_used + 1 {
        let r = eval::observe(&p, fuel);
        same(&mir_eval::observe(&m, fuel), &r);
        same(
            &mir_eval::observe(&net.compiled_mir(&proof).unwrap(), fuel),
            &r,
        );
        let g = oracle::observe(
            &net,
            &proof,
            oracle::Limits {
                fuel,
                ..oracle::Limits::default()
            },
        )
        .unwrap();
        same(&g.semantic, &r);
    }
    full
}
fn owner_cases() -> Vec<(&'static str, String)> {
    [
  ("i32-array","","Array<i32>","[3,5]","return result[1u32];"),
  ("u32-array","","Array<u32>","[7u32]","if result[0u32] == 7u32 {return 7;}return 0;"),
  ("f32-array","","Array<f32>","[7.0]","if result[0u32] == 7.0 {return 7;}return 0;"),
  ("bytes","","Bytes","b\"ok\"","if len(result) == 2u32 {return 7;}return 0;"),
  ("string","","String","\"ok\"","if len(result) == 2u32 {return 7;}return 0;"),
  ("record","struct Packet { x: i32 }","Packet","Packet {x:7}","return result.x;"),
  ("variant","enum Choice { Low, High(i32) }","Choice","Choice::High(7)","return match move result {Choice::Low=>0,Choice::High(x)=>x};"),
 ].into_iter().map(|(id,decl,ty,value,body)| (id,format!("{decl} fn make()->{ty} {{let item:{ty} = {value};return item;}} fn main()->i32 {{let result=make();{body}}}"))).collect()
}
#[test]
fn issue_m0_owner_return_fuel_spec_bare_and_explicit_names_have_same_charge() {
    for (id, bare) in owner_cases() {
        let explicit = bare.replace("return item;", "return move item;");
        let a = qualify(&bare);
        let b = qualify(&explicit);
        assert_eq!(a.result, b.result, "{id}");
        assert_eq!(
            a.steps_used, b.steps_used,
            "{id}: every evaluated name and Move costs one"
        );
        assert_eq!(a.cleanup_trace, b.cleanup_trace, "{id}");
    }
    let s = owner_cases()[0].1.clone();
    assert_eq!(
        qualify(&s).steps_used,
        12,
        "2 caller-entry + 4 array construction + 2 return + 4 caller index/return"
    );
}
#[test]
fn issue_m0_owner_return_fuel_spec_charge_precedes_transfer_and_cleanup() {
    let bare = "fn make()->Array<i32> {if true {let done=filled_i32(1u32,9);}let keep=filled_i32(1u32,8);let item:Array<i32> = [3,5];return item;}fn main()->i32 {let result=make();return result[1u32];}";
    let p = checked(bare);
    let full = qualify(bare);
    assert_eq!(full.steps_used, 22);
    for (fuel, names) in [(17, vec!["done"]), (18, vec!["done", "keep"])] {
        let a = eval::observe(&p, fuel);
        assert!(matches!(
            a.result,
            Err(eval::EvalError::EvaluationLimit { .. })
        ));
        assert_eq!(a.steps_used, fuel);
        assert_eq!(
            a.cleanup_trace
                .iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>(),
            names
        );
    }
    assert_eq!(
        full.cleanup_trace
            .iter()
            .map(|e| e.name.as_str())
            .collect::<Vec<_>>(),
        vec!["done", "keep", "result"]
    );
    let net = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    for (fuel, moves) in [(17, 0), (18, 2)] {
        let run = oracle::observe(
            &net,
            &proof,
            oracle::Limits {
                fuel,
                ..oracle::Limits::default()
            },
        )
        .unwrap();
        assert_eq!(
            run.events
                .iter()
                .filter(|e| e.rule == general_net::coverage::Rule::Move)
                .count(),
            moves,
            "no Move demand/completion before its name Tick succeeds"
        );
    }
    let explicit = bare.replace("return item;", "return move item;");
    let q = checked(&explicit);
    for fuel in 1..=23 {
        same(&eval::observe(&p, fuel), &eval::observe(&q, fuel));
    }
}
#[test]
fn issue_m0_owner_return_fuel_spec_nested_and_failed_returns() {
    let bare = "fn relay(xs:take Array<i32>)->Array<i32>{return xs;}fn make()->Array<i32>{let xs:Array<i32> = [3,5];return xs;}fn outer()->Array<i32>{let a=make();return relay(move a);}fn main()->i32{let result=outer();return result[1u32];}";
    let explicit = bare.replace("return xs;", "return move xs;");
    let a = qualify(bare);
    let b = qualify(&explicit);
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.result, b.result);
    assert_eq!(a.cleanup_trace, b.cleanup_trace);
    for source in [
        "fn make()->Array<i32>{let keep=filled_i32(1u32,8);let item:Array<i32> = [3,5];return item;}fn main()->i32{let result=make();return result[9u32];}",
        "fn make()->Array<i32>{let keep=filled_i32(1u32,8);let item:Array<i32> = [3,5];return item;}fn crash(xs:take Array<i32>)->i32{return 1/0;}fn main()->i32{return crash(make());}",
        "fn bad()->Array<i32>{let item:Array<i32> = [3,5];return filled_i32(1u32,1/0);}fn main()->i32{let result=bad();return 7;}",
    ] {
        let run = qualify(source);
        assert!(matches!(run.result, Err(eval::EvalError::Trap { .. })));
        assert!(
            !run.cleanup_trace
                .iter()
                .any(|e| e.name == "result" || e.name == "item")
        );
    }
    for mode in ["read", "mut"] {
        let s = format!(
            "fn bad(xs:{mode} Array<i32>)->Array<i32>{{return xs;}}fn main()->i32{{return 0;}}"
        );
        match check::check(front::parse(s.as_bytes()).unwrap(), true) {
            Err(e) => assert_eq!(e.code, "OWN_RETURN_BORROW"),
            Ok(p) => assert!(ownership::check(&p).is_err()),
        }
    }
}
#[cfg(feature = "native")]
#[test]
fn issue_m0_owner_return_fuel_spec_actual_project_native() {
    for (_, source) in owner_cases() {
        let p = checked(&source);
        let m = mir::lower(&p).unwrap();
        let full = eval::observe(&p, 10000);
        for fuel in [1, full.steps_used - 1, full.steps_used, full.steps_used + 1] {
            let r = eval::observe(&p, fuel);
            let (status, v) = invar::native::run_program(&m, fuel).unwrap();
            match &r.result {
                Ok(eval::Value::I32(value)) => {
                    assert_eq!(status, 0);
                    assert_eq!(v["result"]["value"], *value);
                }
                Err(eval::EvalError::EvaluationLimit { .. }) => {
                    assert_eq!(status, 4);
                    assert_eq!(v["status"], "evaluation_limit");
                }
                other => panic!("unexpected native regression outcome: {other:?}"),
            }
            assert_eq!(v["steps_used"], r.steps_used);
            assert_eq!(
                v["cleanup_trace"],
                serde_json::json!(
                    r.cleanup_trace
                        .iter()
                        .map(|e| serde_json::json!({"name":e.name,"length":e.length}))
                        .collect::<Vec<_>>()
                )
            );
        }
    }
}

#[path = "support/owner_return_cpu.rs"]
mod cpu_process;

#[test]
fn issue_m0_owner_return_fuel_spec_actual_compiled_cpu_and_resumption() {
    use general_net::compiled;
    use std::fs;
    let mut cases = owner_cases()
        .into_iter()
        .filter(|(id, _)| !["bytes", "string"].contains(id))
        .collect::<Vec<_>>();
    cases.push(("cleanup", "fn make()->Array<i32> {if true {let done=filled_i32(1u32,9);}let keep=filled_i32(1u32,8);let item:Array<i32> = [3,5];return item;}fn main()->i32 {let result=make();return result[1u32];}".to_owned()));
    cases.push(("nested","fn relay(xs:take Array<i32>)->Array<i32>{return xs;}fn make()->Array<i32>{let xs:Array<i32> = [3,5];return xs;}fn outer()->Array<i32>{let a=make();return relay(move a);}fn main()->i32{let result=outer();return result[1u32];}".to_owned()));
    cases.push(("trap","fn make()->Array<i32>{let keep=filled_i32(1u32,8);let item:Array<i32> = [3,5];return item;}fn main()->i32{let result=make();return result[9u32];}".to_owned()));
    cases.push(("trap-call", "fn make()->Array<i32>{let keep=filled_i32(1u32,8);let item:Array<i32> = [3,5];return item;}fn crash(xs:take Array<i32>)->i32{return 1/0;}fn main()->i32{return crash(make());}".to_owned()));
    cases.push(("trap-return", "fn bad()->Array<i32>{let item:Array<i32> = [3,5];return filled_i32(1u32,1/0);}fn main()->i32{let result=bad();return 7;}".to_owned()));
    let mut prefixes = 0;
    let mut programs = 0;
    let mut tools = cpu_process::CpuTools::new();
    let dir = tools.dir.clone();
    for (id, bare) in cases {
        for source in [
            bare.clone(),
            bare.replace("return item;", "return move item;")
                .replace("return xs;", "return move xs;"),
        ] {
            let p = checked(&source);
            let net = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
            let proof = net.verify().unwrap();
            let m = net.compiled_mir(&proof).unwrap();
            let done = eval::observe(&p, 10000);
            let maximum = done.steps_used + 1;
            for resumable in [false, true] {
                let storage = compiled::OwnedLimits {
                    elements: 128,
                    allocations: 64,
                    cleanup_events: 64,
                };
                let emitter = match (id, resumable) {
                    ("record" | "variant", false) => compiled::emit_aggregates,
                    ("record" | "variant", true) => compiled::emit_resumable_aggregates,
                    // ABI11 owner continuations are the only CPU numeric artifacts.
                    (_, false) => continue,
                    (_, true) => compiled::emit_resumable_owned_numeric,
                };
                let artifact = emitter(
                    &net,
                    &proof,
                    "main",
                    compiled::Target::Cpu,
                    compiled::Limits::default(),
                    storage,
                )
                .unwrap();
                let layout = artifact.owned_layout().unwrap();
                let run = if resumable {
                    "GnetTask task={}; for(gnet_u round=0;!gnet_resume(nullptr,output,trace.data()+1,fuel,10000,10000,heap.data()+1,cleanup.data()+1,&task,1);++round) if(round>10000) return 31;"
                } else {
                    "gnet_run(nullptr,output,trace.data()+1,fuel,10000,10000,heap.data()+1,cleanup.data()+1);"
                };
                let driver = format!(
                    r#"
#include <iostream>
#include <vector>
int main(){{for(gnet_u fuel=1;fuel<={maximum};++fuel){{
 constexpr gnet_u G=0x12345678;
 gnet_w guarded[12]={{}};guarded[0]=guarded[11]=G;gnet_w* output=guarded+1;
 std::vector<gnet_u> trace(70002,G),cleanup({clean}+2,G);std::vector<gnet_w> heap({heap}+2,G);
 {run}
 if(guarded[0]!=G||guarded[11]!=G||trace.front()!=G||trace.back()!=G||cleanup.front()!=G||cleanup.back()!=G||heap.front()!=G||heap.back()!=G) return 32;
 if(output[8]<0||output[8]>64) return 33;
 for(gnet_u i=0;i<10;++i)std::cout<<output[i]<<' ';
 for(gnet_u i=0;i<(gnet_u)output[8]*4;++i)std::cout<<cleanup[i+1]<<' ';
 std::cout<<'\n';
}}return 0;}}
"#,
                    clean = layout.cleanup_words(),
                    heap = layout.heap_words()
                );
                fs::write(
                    dir.join("candidate.cpp"),
                    format!("{}\n{driver}", artifact.source()),
                )
                .unwrap();
                // Resolve the compiler like the other compiled-CPU harnesses
                // (CXX, else `c++` on PATH) instead of a host-specific path.
                let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".to_owned());
                let built = tools.run(
                    &compiler,
                    &[
                        "-std=c++17",
                        "-O1",
                        "-fsanitize=undefined",
                        "-fno-sanitize-recover=all",
                        "candidate.cpp",
                        "-o",
                        "candidate",
                    ],
                    20,
                    2 * 1024 * 1024 * 1024,
                    16 * 1024 * 1024,
                );
                assert!(
                    built.status.success(),
                    "{id}/{resumable}: {}",
                    String::from_utf8_lossy(&built.stderr)
                );
                let executed = tools.run("./candidate", &[], 3, 256 * 1024 * 1024, 65536);
                assert!(
                    executed.status.success(),
                    "{id}: {}",
                    String::from_utf8_lossy(&executed.stderr)
                );
                let text = String::from_utf8(executed.stdout).unwrap();
                assert_eq!(text.lines().count(), maximum as usize);
                for (index, line) in text.lines().enumerate() {
                    let fuel = index as u64 + 1;
                    let expected = eval::observe(&p, fuel);
                    let graph = oracle::observe(
                        &net,
                        &proof,
                        oracle::Limits {
                            fuel,
                            transitions: 10000,
                            trace_events: 10000,
                            ..oracle::Limits::default()
                        },
                    )
                    .unwrap();
                    same(&graph.semantic, &expected);
                    let words = line
                        .split_ascii_whitespace()
                        .map(|v| v.parse::<i128>().unwrap())
                        .collect::<Vec<_>>();
                    let (status, value, start, end) = match expected.result {
                        Ok(eval::Value::I32(v)) => (0, i128::from(v), 0, 0),
                        Err(eval::EvalError::EvaluationLimit { .. }) => (3, 0, 0, 0),
                        Err(eval::EvalError::Trap {
                            category: eval::TrapCategory::IndexOutOfBounds,
                            span,
                        }) => (9, 0, span.start as i128, span.end as i128),
                        Err(eval::EvalError::Trap {
                            category: eval::TrapCategory::DivisionByZero,
                            span,
                        }) => (2, 0, span.start as i128, span.end as i128),
                        other => panic!("{other:?}"),
                    };
                    assert_eq!(
                        &words[..5],
                        &[status, value, start, end, fuel.min(done.steps_used) as i128],
                        "{id} fuel{fuel} resumable{resumable}"
                    );
                    assert_eq!(words[5], graph.transitions as i128);
                    assert_eq!(words[6], graph.max_frames as i128);
                    assert_eq!(words[7], graph.events.len() as i128);
                    assert_eq!(words[8], expected.cleanup_trace.len() as i128);
                    for (i, event) in expected.cleanup_trace.iter().enumerate() {
                        let f = words[10 + i * 4] as usize;
                        let place = words[11 + i * 4];
                        let name = if place == u64::MAX as i128 {
                            "<result>"
                        } else {
                            &m.functions[f].declaration_order[place as usize]
                        };
                        assert_eq!(name, event.name);
                        assert_eq!(words[12 + i * 4], event.length as i128);
                    }
                    assert_eq!(words.len(), 10 + expected.cleanup_trace.len() * 4);
                    prefixes += 1;
                }
                programs += 1;
            }
        }
    }
    println!(
        "actual_compiled_cpu_programs={programs} exact_result_fuel_cleanup_prefixes={prefixes} includes_resumable_quantum1=true"
    );
    drop(tools);
    cpu_process::absent(&dir);
}
