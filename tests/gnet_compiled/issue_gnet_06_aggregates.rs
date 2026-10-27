#![forbid(unsafe_code)]

use crate::{child_deadline, gnet_words};
use general_net::{Net, compiled, oracle};
use invar::{check, eval, front, general_net};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

fn lowered(source: &str) -> Net {
    let parsed =
        front::parse(source.as_bytes()).unwrap_or_else(|e| panic!("parse {e:?}: {source}"));
    let checked = check::check(parsed, true).unwrap_or_else(|e| panic!("check {e:?}: {source}"));
    Net::lower(&checked, general_net::Limits::default()).unwrap()
}

struct Cpu {
    dir: PathBuf,
    layout: compiled::OwnedLayout,
    aggregate: compiled::AggregateLayout,
    result: front::Type,
}
impl Cpu {
    fn new(net: &Net, entry: &str, storage: compiled::OwnedLimits) -> Self {
        Self::build(net, entry, storage, None)
    }
    fn resumable(net: &Net, entry: &str, storage: compiled::OwnedLimits, quantum: u64) -> Self {
        Self::build(net, entry, storage, Some(quantum))
    }
    fn build(net: &Net, entry: &str, storage: compiled::OwnedLimits, quantum: Option<u64>) -> Self {
        let emitter = if quantum.is_some() {
            compiled::emit_resumable_aggregates
        } else {
            compiled::emit_aggregates
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
            if quantum.is_some() { 7 } else { 6 }
        );
        let aggregate = artifact.aggregate_layout().unwrap().clone();
        let result = artifact.result().clone();
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
            "invar-aggregates-{}-{}",
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
        Self {
            dir,
            layout,
            aggregate,
            result,
        }
    }
    fn run(&self, args: &[i64], limits: oracle::Limits) -> Vec<i128> {
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
        let inputs: Vec<i64> = args
            .iter()
            .map(|x| match x {
                eval::Value::U32(n) => i64::from(*n),
                eval::Value::I32(n) => i64::from(*n),
                eval::Value::Bool(n) => i64::from(*n),
                _ => panic!("scalar"),
            })
            .collect();
        let words = self.run(&inputs, limits);
        let (status, start, end) = gnet_words::status(&expected);
        assert_eq!(
            (words[0], words[2], words[3]),
            (i128::from(status), i128::from(start), i128::from(end))
        );
        assert_eq!(
            &words[4..9],
            &[
                expected.semantic.steps_used as i128,
                expected.transitions as i128,
                expected.max_frames as i128,
                expected.events.len() as i128,
                expected.semantic.cleanup_trace.len() as i128
            ]
        );
        let mut offset = 10;
        for event in &expected.events {
            assert_eq!(
                &words[offset..offset + 7],
                &gnet_words::event(event).map(i128::from)
            );
            offset += 7;
        }
        let program = net.compiled_mir(&net.verify().unwrap()).unwrap();
        for event in &expected.semantic.cleanup_trace {
            let f = words[offset] as usize;
            let place = words[offset + 1] as usize;
            let name = if words[offset + 1] == i128::from(u64::MAX) {
                "<result>"
            } else {
                &program.functions[f].declaration_order[place]
            };
            assert_eq!(name, event.name);
            assert_eq!(words[offset + 2], event.length as i128);
            offset += 4;
        }
        let heap = &words[offset..];
        assert_eq!(heap.len(), self.layout.heap_words());
        match &expected.semantic.result {
            Ok(expected) => assert_eq!(self.decode(&self.result, words[1], heap), *expected),
            Err(_) => assert_eq!(words[1], 0),
        }
        expected
    }
    fn decode(&self, ty: &front::Type, word: i128, heap: &[i128]) -> eval::Value {
        use front::Type as T;
        use std::{cell::RefCell, rc::Rc};
        match ty {
            T::I32 => eval::Value::I32(i32::try_from(word).unwrap()),
            T::U32 => eval::Value::U32(u32::try_from(word).unwrap()),
            T::Bool => eval::Value::Bool(word != 0),
            T::Unit => eval::Value::Unit,
            T::ArrayI32 | T::ArrayU32 | T::Named(_) => {
                assert!(word > 0);
                let d = (word as usize - 1) * 4;
                assert_eq!(heap[d + 3] & 1, 1);
                let start = self.layout.allocations() * 4 + heap[d] as usize;
                match ty {
                    T::ArrayI32 => {
                        assert_eq!(heap[d + 2], 0);
                        eval::Value::ArrayI32(Rc::new(RefCell::new(
                            heap[start..start + heap[d + 1] as usize]
                                .iter()
                                .map(|n| i32::try_from(*n).unwrap())
                                .collect(),
                        )))
                    }
                    T::ArrayU32 => {
                        assert_eq!(heap[d + 2], 1);
                        eval::Value::ArrayU32(Rc::new(RefCell::new(
                            heap[start..start + heap[d + 1] as usize]
                                .iter()
                                .map(|n| u32::try_from(*n).unwrap())
                                .collect(),
                        )))
                    }
                    T::Named(name) => {
                        let kind = self
                            .aggregate
                            .kinds()
                            .iter()
                            .find(|k| k.name() == name)
                            .unwrap();
                        assert_eq!(heap[d + 2], i128::from(kind.id()));
                        match kind.form() {
                            compiled::AggregateForm::Record(fields) => {
                                eval::Value::Record(Rc::new(RefCell::new(
                                    fields
                                        .iter()
                                        .enumerate()
                                        .map(|(i, t)| {
                                            if heap[d + 3] & (1i128 << (i + 1)) != 0 {
                                                self.decode(t, heap[start + i], heap)
                                            } else {
                                                eval::Value::Unit
                                            }
                                        })
                                        .collect(),
                                )))
                            }
                            compiled::AggregateForm::Variant(arms) => {
                                let tag = (heap[start] & 0xffff_ffff) as usize;
                                let payload = if heap[start] & (1i128 << 32) != 0 {
                                    Some(self.decode(
                                        arms[tag].as_ref().unwrap(),
                                        heap[start + 1],
                                        heap,
                                    ))
                                } else {
                                    None
                                };
                                eval::Value::Variant(Rc::new(RefCell::new((tag, payload))))
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            #[allow(unreachable_patterns)]
            _ => panic!("unsupported decode"),
        }
    }
}
impl Drop for Cpu {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap();
    }
}

fn storage() -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: 128,
        allocations: 64,
        cleanup_events: 128,
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

#[test]
fn issue_gnet_06_records_variants_moves_borrows_and_overwrite() {
    let cases = [
        include_str!("../fixtures/l1_01/record-moved-array.inv"),
        include_str!("../fixtures/l1_01/variant-owned-pattern.inv"),
        "struct Packet { flag: Bool, count: u32, items: Array<u32>, code: i32 } fn inspect(p: read Packet) -> i32 { if p.flag and p.count == 2u32 { return p.code; } return 0; } fn main() -> i32 { let packet = Packet { code: 9, items: [3u32,4u32], count: 2u32, flag: true }; return inspect(packet); }",
        "struct Packet { items: Array<i32>, code: i32 } fn make() -> Packet { let p = Packet { items: [1,2,3], code: 7 }; return p; } fn consume(p: take Packet) -> i32 { return p.code; } fn main() -> i32 { let p = make(); return consume(move p); }",
        "struct Packet { items: Array<i32>, code: i32 } fn main() -> i32 { var p = Packet { items: [1,2], code: 7 }; p = Packet { items: [3], code: 9 }; return p.code; }",
        "enum Choice { Empty, Number(i32) } fn inspect(c: read Choice) -> i32 { return match c { Choice::Empty => 1, Choice::Number(x) => x }; } fn main() -> i32 { let a = Choice::Empty(); let b = Choice::Number(0); return inspect(a) + inspect(b); }",
        "fn make(ok: Bool) -> Result<i32,u32> { if ok { return Result::Ok(7); } return Result::Err(9u32); } fn wrap(ok: Bool) -> Result<i32,u32> { let scratch: Array<i32> = [1,2]; let a: i32 = make(ok)?; return Result::Ok(a); } fn run(ok: Bool) -> i32 { let x = wrap(ok); return match move x { Result::Ok(a) => a, Result::Err(e) => 9 }; } fn main() -> i32 { return run(true)+run(false); }",
        "struct Zero {n:i32} fn main()->i32 {let zero=Zero{n:0};return zero.n;}",
    ];
    for source in cases {
        let net = lowered(source);
        let cpu = Cpu::new(&net, "main", storage());
        assert!(
            cpu.compare(&net, "main", &[], limits())
                .semantic
                .result
                .is_ok(),
            "{source}"
        );
    }
}

#[test]
fn issue_gnet_06_returned_handles_snapshot_moved_fields_and_payload_zero() {
    for source in [
        "struct Packet { items: Array<i32>, code: i32 } fn output() -> Packet { let p=Packet {items:[-1,2],code:0}; return p; }",
        "enum Choice { Empty, Number(i32) } fn output() -> Choice { return Choice::Number(0); }",
        "fn output() -> Result<u32,i32> { return Result::Ok(4294967295u32); }",
        "enum Choice { Empty, Number(i32) } fn output() -> Choice { return Choice::Empty(); }",
    ] {
        let parsed = front::parse(source.as_bytes()).unwrap();
        let checked = check::check(parsed, false).unwrap();
        let net = Net::lower(&checked, general_net::Limits::default()).unwrap();
        Cpu::new(&net, "output", storage()).compare(&net, "output", &[], limits());
    }
}

#[test]
fn issue_gnet_06_exact_fuel_trap_and_tool_prefixes() {
    for source in [
        "struct Packet {items:Array<i32>,code:i32} fn main()->i32 { if true {let done=Packet{items:[1,2],code:0};} let p=Packet{items:[3],code:7}; return p.code/0; }",
        "enum Choice {Empty,Number(i32)} fn main()->i32 {let x=Choice::Number(7);return match move x {Choice::Empty=>0,Choice::Number(n)=>n/0};}",
        "struct Packet {items:Array<i32>,code:i32} fn main()->i32 {var p=Packet{items:[1,2],code:0}; for i in 0u32..4u32 {p=Packet{items:[3],code:7};} return p.code; }",
    ] {
        let net = lowered(source);
        let cpu = Cpu::new(&net, "main", storage());
        let done = cpu.compare(&net, "main", &[], limits());
        for fuel in 1..=done.semantic.steps_used {
            cpu.compare(&net, "main", &[], oracle::Limits { fuel, ..limits() });
        }
        for cap in [0, 1, 2, 7, 31] {
            cpu.compare(
                &net,
                "main",
                &[],
                oracle::Limits {
                    trace_events: cap,
                    ..limits()
                },
            );
            cpu.compare(
                &net,
                "main",
                &[],
                oracle::Limits {
                    transitions: cap as u64,
                    ..limits()
                },
            );
        }
    }
}

#[test]
fn issue_gnet_06_metadata_admission_and_old_apis_stay_explicit() {
    let net = lowered(
        "struct Packet {a:Array<i32>,flag:Bool} fn main()->i32 {let p=Packet{a:[1],flag:true};return 7;}",
    );
    let proof = net.verify().unwrap();
    for target in [compiled::Target::Cuda, compiled::Target::Metal] {
        assert_eq!(
            compiled::emit_aggregates(
                &net,
                &proof,
                "main",
                target,
                compiled::Limits::default(),
                storage()
            )
            .unwrap_err(),
            compiled::Error::AggregateTarget
        );
    }
    assert_eq!(
        compiled::emit_owned(
            &net,
            &proof,
            "main",
            compiled::Target::Cpu,
            compiled::Limits::default(),
            storage()
        )
        .unwrap_err(),
        compiled::Error::UnsupportedType
    );
    let artifact = compiled::emit_aggregates(
        &net,
        &proof,
        "main",
        compiled::Target::Cpu,
        compiled::Limits::default(),
        storage(),
    )
    .unwrap();
    let layout = artifact.aggregate_layout().unwrap();
    assert_eq!(layout.kinds()[0].id(), 2);
    assert_eq!(layout.kinds()[0].name(), "Packet");
    assert_eq!(layout.max_record_fields(), 16);
    assert_eq!(layout.temporary_cleanup_place(), u64::MAX);
    assert!(artifact.continuation_layout().is_none());
    assert!(artifact.device_layout().is_none());
    assert!(artifact.speculation_layout().is_none());
    for source in [
        "fn main()->i32 {let inner: Result<i32,u32> = Result::Ok(7); let outer: Result<Result<i32,u32>,u32> = Result::Ok(move inner); return 7;}",
        "fn main()->i32 {let b:Bytes=b\"x\";return 7;}",
        "fn main()->i32 {let result: Result<Bytes,u32> = Result::Ok(b\"x\");return 7;}",
    ] {
        let net = lowered(source);
        assert_eq!(
            compiled::emit_aggregates(
                &net,
                &net.verify().unwrap(),
                "main",
                compiled::Target::Cpu,
                compiled::Limits::default(),
                storage()
            )
            .unwrap_err(),
            compiled::Error::UnsupportedType
        );
    }
}

#[test]
fn issue_gnet_06_source_partial_owner_and_payload_constraints_stay_frozen() {
    for source in [
        "struct Packet {items:Array<i32>,code:i32} fn make()->Packet {return Packet{items:[1,2],code:0};} fn main()->i32 {make();return 7;}",
        "enum Choice {Empty,Number(i32)} fn make()->Choice {return Choice::Number(0);} fn main()->i32 {make();return 7;}",
        "struct Packet {items:Array<i32>,code:i32} fn output()->Packet {let p=Packet{items:[1],code:0};let a=move p.items;return p;}",
        "enum Choice {Items(Array<i32>)} fn main()->i32 {return 0;}",
        "fn main()->i32 {let r:Result<Array<i32>,u32> = Result::Ok([1]);return 0;}",
    ] {
        let parsed = front::parse(source.as_bytes());
        assert!(
            parsed
                .and_then(|p| check::check(p, false).map(|_| ()))
                .is_err()
        );
    }
}

#[test]
fn issue_gnet_06_runtime_inputs_storage_budgets_and_presence_boundary() {
    let source = "struct Packet {items:Array<i32>,flag:Bool} fn run(ok:Bool)->i32 {let p=Packet{items:[3,7],flag:ok}; let a=move p.items; if p.flag {return a[1u32];} return a[0u32];} fn main()->i32{return run(true);}";
    let net = lowered(source);
    let cpu = Cpu::new(&net, "run", storage());
    for ok in [false, true] {
        cpu.compare(&net, "run", &[eval::Value::Bool(ok)], limits());
    }
    let words = cpu.run(&[2], limits());
    assert_eq!(words[0], 8);
    assert_eq!(words[4], 0);
    assert_eq!(words[7], 0);
    let net = lowered(
        "struct Packet {items:Array<i32>,code:i32} fn main()->i32 {let p=Packet{items:[1,2],code:7};return p.code;}",
    );
    let normal = oracle::observe(&net, &net.verify().unwrap(), limits()).unwrap();
    for (bounded, status) in [
        (
            compiled::OwnedLimits {
                elements: 3,
                ..storage()
            },
            11,
        ),
        (
            compiled::OwnedLimits {
                allocations: 1,
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
    ] {
        let cpu = Cpu::new(&net, "main", bounded);
        let words = cpu.run(&[], limits());
        assert_eq!(words[0], status);
        assert_eq!(words[1], 0);
        assert_eq!(words[8], 0);
        let count = words[7] as usize;
        for (i, event) in normal.events[..count].iter().enumerate() {
            assert_eq!(
                &words[10 + i * 7..17 + i * 7],
                &gnet_words::event(event).map(i128::from)
            );
        }
        if status == 13 {
            let heap = &words[10 + count * 7..];
            assert_eq!(heap[3], 1);
            assert_eq!(heap[7], 7);
        }
    }
    for fields in [1, 16, 17] {
        let types = (0..fields)
            .map(|i| format!("f{i}:i32"))
            .collect::<Vec<_>>()
            .join(",");
        let values = (0..fields)
            .map(|i| format!("f{i}:{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let source = format!(
            "struct Wide {{{types}}} fn main()->i32 {{let p=Wide{{{values}}};return p.f{};}}",
            fields - 1
        );
        if fields == 17 {
            assert_eq!(
                check::check(front::parse(source.as_bytes()).unwrap(), true)
                    .unwrap_err()
                    .code,
                "RECORD_LIMIT"
            );
        } else {
            let net = lowered(&source);
            Cpu::new(&net, "main", storage()).compare(&net, "main", &[], limits());
        }
    }
}

#[test]
fn issue_gnet_07_aggregate_continuations_match_every_quantum_and_fuel_prefix() {
    for source in [
        "struct Packet {items:Array<i32>,code:i32} fn walk(n:u32)->i32 {let p=Packet{items:[1,2],code:7};if n==0u32{return p.code;}return p.code+walk(n-1u32);} fn main()->i32{return walk(3u32);}",
        "struct Packet {items:Array<i32>,code:i32} fn main()->i32 {var p=Packet{items:[1,2],code:0};for i in 0u32..3u32 {p=Packet{items:[3],code:7};}let a=move p.items;return p.code+a[0u32];}",
        "struct Packet {items:Array<i32>,code:i32} fn main()->i32 {if true {let done=Packet{items:[1,2],code:0};}let p=Packet{items:[3],code:7};return p.code/0;}",
        "fn leaf(ok:Bool)->Result<i32,u32> {if ok{return Result::Ok(7);}return Result::Err(9u32);} fn wrap(ok:Bool)->Result<i32,u32> {let scratch:Array<i32> = [1,2];let n:i32=leaf(ok)?;return Result::Ok(n);} fn main()->i32 {let a=wrap(true);let b=wrap(false);return match move b {Result::Ok(n)=>n,Result::Err(e)=>9};}",
    ] {
        let net = lowered(source);
        for quantum in [1, 3, 17] {
            let cpu = Cpu::resumable(&net, "main", storage(), quantum);
            let done = cpu.compare(&net, "main", &[], limits());
            if source.contains("fn walk(") {
                let entry = Cpu::resumable(&net, "walk", storage(), quantum);
                for n in [0, 1, 3] {
                    entry.compare(&net, "walk", &[eval::Value::U32(n)], limits());
                }
            }
            if source.contains("fn wrap(") {
                let entry = Cpu::resumable(&net, "wrap", storage(), quantum);
                for ok in [false, true] {
                    entry.compare(&net, "wrap", &[eval::Value::Bool(ok)], limits());
                }
                let invalid = entry.run(&[2], limits());
                assert_eq!(invalid[0], 8);
                assert_eq!(invalid[4], 0);
                assert_eq!(invalid[7], 0);
            }
            for fuel in 1..=done.semantic.steps_used {
                cpu.compare(&net, "main", &[], oracle::Limits { fuel, ..limits() });
            }
            for cap in [0, 1, 7, 31] {
                cpu.compare(
                    &net,
                    "main",
                    &[],
                    oracle::Limits {
                        trace_events: cap,
                        ..limits()
                    },
                );
                cpu.compare(
                    &net,
                    "main",
                    &[],
                    oracle::Limits {
                        transitions: cap as u64,
                        ..limits()
                    },
                );
            }
        }
    }
}

#[test]
fn issue_gnet_07_aggregate_continuation_results_and_storage_failure_prefixes() {
    for source in [
        "struct Packet {items:Array<i32>,code:i32} fn output()->Packet {let p=Packet{items:[-1,2],code:0};return p;}",
        "enum Choice {Empty,Number(i32)} fn output()->Choice {return Choice::Number(0);}",
        "fn output()->Result<u32,i32> {return Result::Ok(4294967295u32);}",
    ] {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
        let net = Net::lower(&checked, general_net::Limits::default()).unwrap();
        for quantum in [1, 3, 17] {
            Cpu::resumable(&net, "output", storage(), quantum).compare(
                &net,
                "output",
                &[],
                limits(),
            );
        }
    }
    let net = lowered(
        "struct Packet {items:Array<i32>,code:i32} fn main()->i32 {let p=Packet{items:[1,2],code:7};return p.code;}",
    );
    for (budget, status) in [
        (
            compiled::OwnedLimits {
                elements: 3,
                ..storage()
            },
            11,
        ),
        (
            compiled::OwnedLimits {
                allocations: 1,
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
    ] {
        let plain = Cpu::new(&net, "main", budget).run(&[], limits());
        assert_eq!(plain[0], status);
        for quantum in [1, 3, 17] {
            assert_eq!(
                Cpu::resumable(&net, "main", budget, quantum).run(&[], limits()),
                plain
            );
        }
    }
    for target in [compiled::Target::Cuda, compiled::Target::Metal] {
        assert_eq!(
            compiled::emit_resumable_aggregates(
                &net,
                &net.verify().unwrap(),
                "main",
                target,
                compiled::Limits::default(),
                storage()
            )
            .unwrap_err(),
            compiled::Error::AggregateTarget
        );
    }
}
