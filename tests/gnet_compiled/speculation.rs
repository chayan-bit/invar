//! Issue label GNET-04.
#![forbid(unsafe_code)]

#[path = "../support/gnet_speculation.rs"]
mod speculation;

use crate::child_deadline;
use invar::eval;
use invar::general_net::{Net, compiled, oracle};
use speculation::{limits, lowered, spec, storage};
use std::{fs, path::PathBuf, process::Command};

struct Cpu {
    dir: PathBuf,
    artifact: compiled::Artifact,
    mode: &'static str,
}
impl Cpu {
    fn new(
        net: &Net,
        entry: &str,
        storage: compiled::OwnedLimits,
        spec: compiled::SpeculationLimits,
    ) -> Self {
        let artifact = compiled::emit_speculative(
            net,
            &net.verify().unwrap(),
            entry,
            compiled::Target::Cpu,
            compiled::Limits::default(),
            storage,
            spec,
        )
        .unwrap();
        assert_eq!(artifact.abi_version(), 4);
        let owned = artifact.owned_layout().unwrap();
        let scratch = artifact.speculation_layout().unwrap();
        let dir = child_deadline::scratch("invar-speculate");
        let driver = format!(
            r#"
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
int main(int argc,char** argv) {{
    gnet_u fuel=std::stoull(argv[1]),trans=std::stoull(argv[2]),cap=std::stoull(argv[3]),quantum=std::stoull(argv[4]),capacity=std::stoull(argv[5]);
    std::vector<gnet_w> input; for(int i=6;i<argc;++i) input.push_back(std::stoll(argv[i]));
    gnet_w output[10]={{}}; GnetTask task={{}};
    std::vector<gnet_u> trace(cap*7),cleanup({clean}),private_trace({ptrace}),private_cleanup({pclean});
    std::vector<gnet_w> heap({heap}),private_heap({pheap}); std::vector<GnetSpecSlot> slots({slots});
    GnetSpecState spec={{}};spec.capacity=capacity;spec.slots=slots.data();spec.heap=private_heap.data();spec.trace=private_trace.data();spec.cleanup=private_cleanup.data();
    std::string mode=std::getenv("GNET_TEST_MODE");
    if(mode=="missing") spec.heap=nullptr;
    if(mode=="alias") spec.heap=heap.data();
    if(mode=="generation") spec.next_generation=UINT64_MAX;
    if(mode=="stale") {{
        for(gnet_u i=0;i<1000000&&!task.done;++i) {{
            gnet_w state=gnet_speculate(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data(),&task,&spec,quantum,1);
            if(state!=0&&state!=14) return 3;
            for(auto& slot:slots) if(slot.valid&&slot.task.done) slot.completion_generation=0;
        }}
        if(!task.done) return 3;
    }} else if(gnet_speculate(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data(),&task,&spec,quantum,1000000)!=0) return 3;
    for(auto word:output) std::cout<<word<<' ';
    gnet_u stats[11]={{spec.spawned,spec.committed,spec.cancelled,spec.replayed,spec.guard_failed,spec.capacity_serialized,spec.spec_transitions,spec.owned_replays,spec.last_reason,task.next_frame,task.depth}};
    for(auto word:stats) std::cout<<word<<' ';
    for(gnet_u i=0;i<(gnet_u)output[7]*7;++i) std::cout<<trace[i]<<' ';
    for(gnet_u i=0;i<(gnet_u)output[8]*4;++i) std::cout<<cleanup[i]<<' ';
}}
"#,
            clean = owned.cleanup_words(),
            heap = owned.heap_words(),
            ptrace = scratch.trace_words(),
            pclean = scratch.cleanup_words(),
            pheap = scratch.heap_words(),
            slots = scratch.slots()
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
            artifact,
            mode: "",
        }
    }
    fn run(
        &self,
        args: &[eval::Value],
        limits: oracle::Limits,
        quantum: u64,
        capacity: usize,
    ) -> Vec<i64> {
        let command = Command::new(self.dir.join("run"));
        speculation::run(command, self.mode, args, limits, quantum, capacity)
    }
    fn compare(
        &self,
        net: &Net,
        entry: &str,
        args: &[eval::Value],
        limits: oracle::Limits,
        quantum: u64,
        capacity: usize,
    ) -> Vec<i64> {
        let words = self.run(args, limits, quantum, capacity);
        let offset = speculation::compare(net, entry, args, limits, &words);
        assert_eq!(offset, words.len());
        words
    }
}
impl Drop for Cpu {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap();
    }
}
#[test]
fn recursive_independent_children_are_extracted_and_committed() {
    let net = lowered(
        "fn fib(n: u32) -> i32 { if n < 2u32 { return 1; } return fib(n - 1u32) + fib(n - 2u32); } fn main() -> i32 { return fib(6u32); }",
    );
    let cpu = Cpu::new(&net, "fib", storage(), spec());
    assert!(!cpu.artifact.fork_plan().unwrap().candidates.is_empty());
    let mut commits = 0;
    for n in 0..=7u32 {
        for quantum in [1, 3, 11] {
            let words = cpu.compare(&net, "fib", &[eval::Value::U32(n)], limits(), quantum, 2);
            commits += words[11];
            if n < 2 {
                assert_eq!(words[10], 0);
            }
            cpu.compare(&net, "fib", &[eval::Value::U32(n)], limits(), quantum, 0);
        }
    }
    assert!(
        commits > 0,
        "real child transcripts must be consumed, not just computed and discarded"
    );
}

#[test]
fn oldest_progress_hides_later_traps_and_private_resource_failures() {
    for source in [
        "fn f() -> i32 { while true {} return 0; } fn g() -> i32 { return 1/0; } fn main() -> i32 { return f()+g(); }",
        "fn f() -> i32 { while true {} return 0; } fn g() -> i32 { let xs=filled_i32(65u32,0); return 0; } fn main() -> i32 { return f()+g(); }",
        "fn f() -> i32 { while true {} return 0; } fn g() -> i32 { for i in 0u32..10000u32 {} return 1; } fn main() -> i32 { return f()+g(); }",
    ] {
        let net = lowered(source);
        let cpu = Cpu::new(&net, "main", storage(), spec());
        let words = cpu.compare(
            &net,
            "main",
            &[],
            oracle::Limits {
                fuel: 100,
                ..limits()
            },
            1,
            2,
        );
        assert!(words[10] > 0);
        assert_eq!(words[11], 0);
        assert!(words[12] > 0);
        assert_eq!(words[0], 3);
    }
}

#[test]
fn committed_traps_and_every_fuel_tool_prefix_are_source_ordered() {
    let source = "fn f() -> i32 { for i in 0u32..10u32 {} return 7; } fn g() -> i32 { return 1/0; } fn main() -> i32 { return f()+g(); }";
    let net = lowered(source);
    let cpu = Cpu::new(&net, "main", storage(), spec());
    let full = cpu.compare(&net, "main", &[], limits(), 1, 2);
    assert_eq!(full[11], 1);
    for fuel in 1..=full[4] as u64 {
        cpu.compare(&net, "main", &[], oracle::Limits { fuel, ..limits() }, 1, 2);
    }
    for cap in [0, 1, 10, 40, 100, 150, 200, 400] {
        cpu.compare(
            &net,
            "main",
            &[],
            oracle::Limits {
                transitions: cap,
                ..limits()
            },
            1,
            2,
        );
        cpu.compare(
            &net,
            "main",
            &[],
            oracle::Limits {
                trace_events: cap as usize,
                ..limits()
            },
            1,
            2,
        );
    }
}

#[test]
fn owner_scratch_replays_without_consuming_serial_live_budget() {
    let source = "fn f() -> i32 { for i in 0u32..10u32 {} return 7; } fn g() -> i32 { let xs=filled_i32(4u32,3); return xs[0u32]; } fn main() -> i32 { let keep=filled_i32(4u32,1); return f()+g(); }";
    let net = lowered(source);
    let mut bound = storage();
    bound.elements = 4;
    let cpu = Cpu::new(&net, "main", bound, spec());
    let words = cpu.compare(
        &net,
        "main",
        &[],
        oracle::Limits {
            array_elements: 4,
            ..limits()
        },
        1,
        2,
    );
    assert_eq!(words[0], 11);
    assert_eq!(words[11], 0);
    assert!(words[17] > 0);
    assert_eq!(words[8], 0);
    let net = lowered(
        "fn f() -> i32 { for i in 0u32..10u32 {} return 7; } fn g() -> i32 { let xs: Array<i32> = [3]; return xs[0u32]; } fn main() -> i32 { return f()+g(); }",
    );
    let cpu = Cpu::new(&net, "main", storage(), spec());
    let words = cpu.compare(&net, "main", &[], limits(), 1, 2);
    assert_eq!(words[0], 0);
    assert_eq!(words[8], 1);
    assert!(words[17] > 0);
}

#[test]
fn dependence_and_affine_arguments_serialize_without_disabling_source() {
    for source in [
        "fn f() -> i32 { return 7; } fn g(x: i32) -> i32 { return x+1; } fn main() -> i32 { let a=f(); return g(a); }",
        "fn f(xs: mut Array<i32>) -> i32 { xs[0u32]=9; return 1; } fn g(xs: read Array<i32>) -> i32 { return xs[0u32]; } fn main() -> i32 { var xs: Array<i32> = [1]; return f(xs)+g(xs); }",
    ] {
        let net = lowered(source);
        let cpu = Cpu::new(&net, "main", storage(), spec());
        assert!(!cpu.artifact.fork_plan().unwrap().serialized.is_empty());
        assert_eq!(cpu.compare(&net, "main", &[], limits(), 1, 2)[10], 0);
    }
}

#[test]
fn missing_alias_stale_and_generation_guards_replay_exactly() {
    let net = lowered(
        "fn f() -> i32 { for i in 0u32..10u32 {} return 7; } fn g() -> i32 { return 8; } fn main() -> i32 { return f()+g(); }",
    );
    let mut cpu = Cpu::new(&net, "main", storage(), spec());
    for mode in ["missing", "alias", "stale", "generation"] {
        cpu.mode = mode;
        let words = cpu.compare(&net, "main", &[], limits(), 1, 2);
        assert_eq!(words[11], 0, "{mode}");
        if matches!(mode, "missing" | "alias") {
            assert_eq!(words[10], 0);
            assert_eq!(words[14], 1);
            assert_eq!(words[18], 10);
        } else if mode == "stale" {
            assert_eq!(words[13], 1);
            assert_eq!(words[18], 7);
        } else {
            assert_eq!(words[10], 0);
            assert_eq!(words[18], 9);
        }
    }
}

#[test]
fn private_limits_and_zero_length_scratch_preserve_source() {
    let net = lowered(
        "fn f() -> i32 { for i in 0u32..10u32 {} return 7; } fn g() -> i32 { return 8; } fn main() -> i32 { return f()+g(); }",
    );
    let mut owned = storage();
    owned.cleanup_events = 0;
    for private in [
        compiled::SpeculationLimits { fuel: 1, ..spec() },
        compiled::SpeculationLimits {
            transitions: 0,
            ..spec()
        },
        compiled::SpeculationLimits {
            trace_events: 0,
            ..spec()
        },
    ] {
        let cpu = Cpu::new(&net, "main", owned, private);
        let words = cpu.compare(&net, "main", &[], limits(), 1, 2);
        assert_eq!(words[11], 0);
        assert_eq!(words[13], 1);
        assert_eq!(words[18], 4);
    }
}

#[test]
fn projection_trap_and_unselected_branch_have_no_observation() {
    let net = lowered(
        "fn f() -> i32 { for i in 0u32..10u32 {} return 7; } fn g(x: i32) -> i32 { return x; } fn main() -> i32 { return f()+g(1/0); }",
    );
    let cpu = Cpu::new(&net, "main", storage(), spec());
    let words = cpu.compare(&net, "main", &[], limits(), 1, 2);
    assert_eq!(words[10], 0);
    assert_eq!(words[14], 1);
    assert_eq!(words[0], 2);
    let net = lowered(
        "fn f() -> i32 { while true {} return 1; } fn g() -> i32 { return 1/0; } fn main() -> i32 { if false { return f()+g(); } return 9; }",
    );
    let cpu = Cpu::new(&net, "main", storage(), spec());
    assert!(!cpu.artifact.fork_plan().unwrap().candidates.is_empty());
    assert_eq!(cpu.compare(&net, "main", &[], limits(), 1, 2)[10], 0);
}

#[test]
fn depth_exhaustion_replays_and_multiple_children_commit_in_order() {
    let net = lowered(
        "fn f() -> i32 { for i in 0u32..100u32 {} return 7; } fn g(n: u32) -> i32 { if n==0u32 { return 1; } return g(n-1u32); } fn main() -> i32 { return f()+g(63u32); }",
    );
    let cpu = Cpu::new(&net, "main", storage(), spec());
    let words = cpu.compare(&net, "main", &[], limits(), 1, 2);
    assert_eq!(words[0], 4);
    assert_eq!(words[11], 0);
    assert_eq!(words[13], 1);
    assert_eq!(words[18], 5);
    let net = lowered(
        "fn f() -> i32 { for i in 0u32..10u32 {} return 7; } fn g(x: i32) -> i32 { return x; } fn main() -> i32 { return f()+g(8)+g(9); }",
    );
    let cpu = Cpu::new(&net, "main", storage(), spec());
    let words = cpu.compare(&net, "main", &[], limits(), 1, 2);
    assert_eq!(words[11], 2);
    assert_eq!(words[1], 24);
    assert_eq!(words[12], 0);
}

#[test]
fn scratch_bounds_and_device_admission_are_explicit() {
    let net = lowered("fn main() -> i32 { return 1; }");
    for private in [
        compiled::SpeculationLimits { slots: 9, ..spec() },
        compiled::SpeculationLimits {
            scratch_bytes: 0,
            ..spec()
        },
        compiled::SpeculationLimits {
            lookahead: 0,
            ..spec()
        },
    ] {
        assert!(matches!(
            compiled::emit_speculative(
                &net,
                &net.verify().unwrap(),
                "main",
                compiled::Target::Cpu,
                compiled::Limits::default(),
                storage(),
                private
            ),
            Err(compiled::Error::Limit)
        ));
    }
    for target in [compiled::Target::Cuda, compiled::Target::Metal] {
        assert!(matches!(
            compiled::emit_speculative(
                &net,
                &net.verify().unwrap(),
                "main",
                target,
                compiled::Limits::default(),
                storage(),
                spec()
            ),
            Err(compiled::Error::SpeculationTarget)
        ));
    }
}

#[test]
fn scalar_projection_and_loop_activation_capture_current_values() {
    let net = lowered(
        "fn f(n: u32) -> u32 { for i in 0u32..10u32 {} return n; } fn g(n: u32) -> u32 { return n*2u32; } fn compute(n: u32) -> u32 { var sum=0u32; for i in 0u32..n { let a=f(i); var x=i; x=x+1u32; let b=g(x); sum=sum+a+b; } return sum; } fn main() -> i32 { return 0; }",
    );
    let cpu = Cpu::new(&net, "compute", storage(), spec());
    for n in 0..=4u32 {
        let words = cpu.compare(&net, "compute", &[eval::Value::U32(n)], limits(), 1, 2);
        assert_eq!(words[11], i64::from(n));
        cpu.compare(&net, "compute", &[eval::Value::U32(n)], limits(), 7, 2);
    }
}

#[test]
fn certificate_budget_is_explicit_and_keeps_source_execution() {
    let mut source = String::from("fn f() -> i32 { return 1; } fn main() -> i32 {");
    for i in 0..24 {
        source.push_str(&format!("let a{i}=f();"));
    }
    source.push_str("return 1; }");
    let net = lowered(&source);
    let cpu = Cpu::new(&net, "main", storage(), spec());
    let plan = cpu.artifact.fork_plan().unwrap();
    assert_eq!(plan.candidates.len(), 256);
    assert!(
        plan.serialized
            .iter()
            .any(|s| s.reason == compiled::SerializationReason::ProjectionLimit)
    );
    cpu.compare(&net, "main", &[], limits(), 1, 2);
}
