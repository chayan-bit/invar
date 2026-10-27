#![forbid(unsafe_code)]

use crate::child_deadline;

#[path = "../support/gnet_speculation.rs"]
mod speculation;

use invar::eval;
use invar::general_net::{Net, compiled, oracle};
use speculation::{limits, lowered, spec, storage};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Device {
    dir: PathBuf,
    artifact: compiled::Artifact,
    mode: &'static str,
}
impl Device {
    fn new(
        net: &Net,
        entry: &str,
        storage: compiled::OwnedLimits,
        spec: compiled::SpeculationLimits,
    ) -> Self {
        Self::with_target(net, entry, storage, spec, compiled::Target::Cuda)
    }
    fn with_target(
        net: &Net,
        entry: &str,
        storage: compiled::OwnedLimits,
        spec: compiled::SpeculationLimits,
        target: compiled::Target,
    ) -> Self {
        let artifact = compiled::emit_device_speculative(
            net,
            &net.verify().unwrap(),
            entry,
            target,
            compiled::Limits::default(),
            storage,
            spec,
        )
        .unwrap();
        assert_eq!(artifact.abi_version(), 5);
        let owned = artifact.owned_layout().unwrap();
        let scratch = artifact.speculation_layout().unwrap();
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "invar-speculate-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let driver = format!(
            r#"
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <thread>
#include <unistd.h>
int main(int argc,char** argv) {{
    gnet_u fuel=std::stoull(argv[1]),trans=std::stoull(argv[2]),cap=std::stoull(argv[3]),quantum=std::stoull(argv[4]),capacity=std::stoull(argv[5]);
    std::vector<gnet_w> input; for(int i=6;i<argc;++i) input.push_back(std::stoll(argv[i]));
    gnet_w output[10]={{}}; GnetTask task={{}}; gnet_u state[14]={{}};
    std::vector<gnet_u> trace(cap*7),cleanup({clean}),private_trace({ptrace}),private_cleanup({pclean});
    std::vector<gnet_w> heap({heap}),private_heap({pheap}); std::vector<GnetSpecSlot> slots({slots});
    gnet_u config[6]={{fuel,trans,cap,quantum,64,capacity}};
    const char* mode_env=std::getenv("GNET_TEST_MODE"); std::string mode=mode_env?mode_env:"";
    if(mode=="generation") state[1]=UINT64_MAX;
    if(mode=="geometry") blockDim.x=16;
    for(gnet_u epoch=0;epoch<1024&&!task.done;++epoch) {{
        std::vector<std::thread> group;
        for(unsigned i=0;i<32;++i) group.emplace_back([&,i] {{threadIdx.x=i;
            host_entry(input.data(),output,trace.data(),config,heap.data(),cleanup.data(),&task,slots.data(),private_heap.data(),private_trace.data(),private_cleanup.data(),state);
        }});
        for(auto& thread:group) thread.join();
        if(state[12]!=0&&state[12]!=14) break;
        if(mode=="stale") for(auto& slot:slots) if(slot.valid&&slot.task.done) slot.completion_generation=0;
    }}
    for(auto word:output) std::cout<<word<<' ';
    gnet_u stats[11]={{state[2],state[3],state[4],state[5],state[6],state[7],state[8],state[9],state[10],task.next_frame,task.depth}};
    for(auto word:stats) std::cout<<word<<' ';
    for(gnet_u i=0;i<(gnet_u)output[7]*7;++i) std::cout<<trace[i]<<' ';
    for(gnet_u i=0;i<(gnet_u)output[8]*4;++i) std::cout<<cleanup[i]<<' ';
    std::cout<<state[12]<<' '<<state[11]<<' '<<state[13]<<' ';
}}
"#,
            clean = owned.cleanup_words(),
            heap = owned.heap_words(),
            ptrace = scratch.trace_words(),
            pclean = scratch.cleanup_words(),
            pheap = scratch.heap_words(),
            slots = scratch.slots()
        );
        let mut shim = String::from(
            r#"
#include <barrier>
std::barrier group_barrier(32);
#define __device__
#define __global__
#define __shared__ static
#define __syncthreads() group_barrier.arrive_and_wait()
struct Dim { unsigned x=1,y=1,z=1; };
Dim gridDim,blockDim{32,1,1}; thread_local Dim threadIdx{0,0,0},blockIdx{0,0,0};
"#,
        );
        let source = if target == compiled::Target::Metal {
            shim.push_str(
                r#"
#define device
#define thread
#define threadgroup
#define constant
#define kernel
using uint=unsigned; using ulong=unsigned long;
struct uint3 { uint x,y,z; };
namespace metal { enum class mem_flags {mem_device=1,mem_threadgroup=2};
inline unsigned operator|(mem_flags a,mem_flags b){return unsigned(a)|unsigned(b);}
inline void threadgroup_barrier(unsigned){group_barrier.arrive_and_wait();} }
"#,
            );
            artifact
                .source()
                .replace("#include <metal_stdlib>", "")
                .replace(
                    "threadgroup gnet_u root_done; threadgroup gnet_u deltas[8];",
                    "static gnet_u root_done; static gnet_u deltas[8];",
                )
        } else {
            shim.push_str("\n#define host_entry invar_gnet_children_v5\n");
            artifact.source().to_owned()
        };
        let bridge = if target == compiled::Target::Metal {
            r#"
#undef thread
#undef device
#undef threadgroup
#undef constant
#undef kernel
inline void host_entry(const gnet_w* input,gnet_w* output,gnet_u* trace,const gnet_u* config,gnet_w* heap,gnet_u* cleanup,GnetTask* task,GnetSpecSlot* slots,gnet_w* private_heap,gnet_u* private_trace,gnet_u* private_cleanup,gnet_u* state) {
 invar_gnet_children_v5(input,output,trace,config,heap,cleanup,task,slots,private_heap,private_trace,private_cleanup,state,threadIdx.x,uint3{0,0,0},uint3{1,1,1},uint3{blockDim.x,1,1});
}
"#
        } else {
            ""
        };
        fs::write(
            dir.join("source.cpp"),
            format!("{shim}\n{source}\n{bridge}\n{driver}"),
        )
        .unwrap();
        child_deadline::cxx(
            &[
                "-std=c++20",
                "-pthread",
                "-O1",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
            ],
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
        let mut command = child_deadline::command(20);
        command.arg(self.dir.join("run"));
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
        assert_eq!(offset + 3, words.len());
        assert_eq!(words[offset], 0);
        words
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap();
    }
}
#[test]
fn issue_gnet_04_device_group_executes_recursive_children_and_exact_prefixes() {
    let net = lowered(
        "fn fib(n: u32) -> i32 { if n<2u32 { return 1; } return fib(n-1u32)+fib(n-2u32); } fn main() -> i32 { return fib(6u32); }",
    );
    let device = Device::new(&net, "fib", storage(), spec());
    assert!(!device.artifact.fork_plan().unwrap().candidates.is_empty());
    let mut committed = 0;
    let mut width = 0;
    for n in 0..=7 {
        for quantum in [1, 8, 32] {
            let words = device.compare(&net, "fib", &[eval::Value::U32(n)], limits(), quantum, 2);
            committed += words[11];
            width = width.max(*words.last().unwrap());
        }
    }
    assert!(committed > 0);
    assert!(width >= 2);
    let full = device.compare(&net, "fib", &[eval::Value::U32(3)], limits(), 1, 2);
    for fuel in 1..=full[4] as u64 {
        device.compare(
            &net,
            "fib",
            &[eval::Value::U32(3)],
            oracle::Limits { fuel, ..limits() },
            1,
            2,
        );
    }
    for cap in [0, 1, 7, 20, 40, 100] {
        device.compare(
            &net,
            "fib",
            &[eval::Value::U32(3)],
            oracle::Limits {
                trace_events: cap,
                ..limits()
            },
            1,
            2,
        );
        device.compare(
            &net,
            "fib",
            &[eval::Value::U32(3)],
            oracle::Limits {
                transitions: cap as u64,
                ..limits()
            },
            1,
            2,
        );
    }
}

#[test]
fn issue_gnet_04_device_ordered_trap_divergence_private_limits_and_serial_control() {
    for source in [
        "fn f() -> i32 { while true {} return 1; } fn g() -> i32 { return 1/0; } fn main() -> i32 { return f()+g(); }",
        "fn f() -> i32 { for i in 0u32..10u32 {} return 7; } fn g() -> i32 { return 1/0; } fn main() -> i32 { return f()+g(); }",
        "fn f() -> i32 { for i in 0u32..10u32 {} return 7; } fn g() -> i32 { while true {} return 1; } fn main() -> i32 { return f()+g(); }",
    ] {
        let net = lowered(source);
        let device = Device::new(
            &net,
            "main",
            storage(),
            compiled::SpeculationLimits { fuel: 20, ..spec() },
        );
        let bound = oracle::Limits {
            fuel: 100,
            ..limits()
        };
        let words = device.compare(&net, "main", &[], bound, 1, 2);
        assert!(words[10] > 0);
        device.compare(&net, "main", &[], bound, 1, 0);
    }
}

#[test]
fn issue_gnet_04_device_geometry_generation_and_metal_abi_are_explicit() {
    let net = lowered(
        "fn f() -> i32 { for i in 0u32..10u32 {} return 7; } fn g() -> i32 { return 8; } fn main() -> i32 { return f()+g(); }",
    );
    let mut device = Device::new(&net, "main", storage(), spec());
    device.mode = "generation";
    let words = device.compare(&net, "main", &[], limits(), 1, 2);
    assert_eq!(words[10], 0);
    assert_eq!(words[18], 9);
    device.mode = "geometry";
    let words = device.run(&[], limits(), 1, 2);
    assert_eq!(words[words.len() - 3], 8);
    assert_eq!(&words[..10], &[0; 10]);
    assert_eq!(words[19], 0);
    let metal = compiled::emit_device_speculative(
        &net,
        &net.verify().unwrap(),
        "main",
        compiled::Target::Metal,
        compiled::Limits::default(),
        storage(),
        spec(),
    )
    .unwrap();
    assert_eq!(metal.abi_version(), 5);
    assert!(
        metal
            .source()
            .contains("threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)")
    );
    assert!(metal.source().contains("[[buffer(11)]]"));
    assert!(!metal.source().contains("std::uintptr_t"));
    assert!(!metal.source().contains("#include <cstdint>"));
    let simulation = Device::with_target(&net, "main", storage(), spec(), compiled::Target::Metal);
    assert_eq!(simulation.compare(&net, "main", &[], limits(), 1, 2)[11], 1);
    let owned = lowered(
        "fn f() -> i32 { let xs: Array<i32> = [1]; return xs[0u32]; } fn main() -> i32 { return f(); }",
    );
    // GNET-03: owners inside the program run on the device heap with exact
    // cleanup; owner results use ABI13 and its fenced root heap.
    for target in [compiled::Target::Cuda, compiled::Target::Metal] {
        let device = Device::with_target(&owned, "main", storage(), spec(), target);
        device.compare(&owned, "main", &[], limits(), 1, 2);
    }
    let result = lowered(
        "fn run(n: u32) -> Array<i32> { return filled_i32(n, 1); } fn main() -> i32 { let xs = run(1u32); return xs[0u32]; }",
    );
    let returned = compiled::emit_device_speculative(
        &result,
        &result.verify().unwrap(),
        "run",
        compiled::Target::Cuda,
        compiled::Limits::default(),
        storage(),
        spec(),
    )
    .unwrap();
    assert_eq!(returned.abi_version(), 13);
    assert!(returned.entry_layout().is_some());
}

fn admitted() -> compiled::Workset {
    let net = lowered("fn main() -> i32 { return 1; }");
    let artifact = compiled::emit_device_speculative(
        &net,
        &net.verify().unwrap(),
        "main",
        compiled::Target::Cuda,
        compiled::Limits::default(),
        storage(),
        spec(),
    )
    .unwrap();
    let layout = artifact.device_layout().unwrap();
    let config = compiled::DeviceConfig {
        fuel: 100,
        transitions: 1000,
        trace_events: 1000,
        quantum: 8,
        rounds: 64,
        capacity: 2,
    };
    let bytes = layout.buffer_bytes(config).unwrap();
    let ranges = std::array::from_fn(|i| compiled::BufferRange {
        device: 1,
        allocation: i as u64 + 1,
        offset: 0,
        bytes: bytes[i],
    });
    layout.workset(config, ranges).unwrap()
}

#[test]
fn issue_gnet_04_device_workset_rejects_alias_short_unknown_overflow_and_config() {
    let good = admitted();
    let net = lowered("fn main() -> i32 { return 1; }");
    let artifact = compiled::emit_device_speculative(
        &net,
        &net.verify().unwrap(),
        "main",
        compiled::Target::Cuda,
        compiled::Limits::default(),
        storage(),
        spec(),
    )
    .unwrap();
    let layout = artifact.device_layout().unwrap();
    for mode in 0..6 {
        let mut ranges = *good.ranges();
        match mode {
            0 => ranges[7].allocation = ranges[6].allocation,
            1 => ranges[2].bytes -= 8,
            2 => ranges[9].allocation = 0,
            3 => ranges[9].offset = usize::MAX - 7,
            4 => ranges[9].device = 2,
            _ => ranges[9].offset = 1,
        }
        assert!(layout.workset(good.config(), ranges).is_err());
    }
    assert!(
        layout
            .buffer_bytes(compiled::DeviceConfig {
                rounds: 65,
                ..good.config()
            })
            .is_err()
    );
    assert!(
        layout
            .buffer_bytes(compiled::DeviceConfig {
                capacity: 3,
                ..good.config()
            })
            .is_err()
    );
    assert!(
        layout
            .buffer_bytes(compiled::DeviceConfig {
                fuel: 0,
                ..good.config()
            })
            .is_err()
    );
}

#[test]
fn issue_gnet_04_device_stale_fenced_completion_and_zero_trace_replay() {
    let net = lowered(
        "fn f() -> i32 { for i in 0u32..100u32 {} return 7; } fn g() -> i32 { return 8; } fn main() -> i32 { return f()+g(); }",
    );
    let mut device = Device::new(&net, "main", storage(), spec());
    device.mode = "stale";
    let words = device.compare(&net, "main", &[], limits(), 1, 2);
    assert_eq!(words[11], 0);
    assert_eq!(words[13], 1);
    assert_eq!(words[18], 7);
    let device = Device::new(
        &net,
        "main",
        storage(),
        compiled::SpeculationLimits {
            trace_events: 0,
            ..spec()
        },
    );
    let words = device.compare(&net, "main", &[], limits(), 1, 2);
    assert_eq!(words[11], 0);
    assert_eq!(words[13], 1);
    assert_eq!(words[18], 4);
}

#[test]
fn issue_gnet_04_device_fence_retains_leases_blocks_observation_and_cancellation() {
    use compiled::{
        ExecutionState as E, FenceError, FenceObservation as F, FencedExecution, Submission as S,
    };
    use std::{cell::Cell, rc::Rc};
    struct Lease(Rc<Cell<usize>>);
    impl Drop for Lease {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    for observation in [F::Pending, F::Unknown] {
        let dropped = Rc::new(Cell::new(0));
        let mut run = FencedExecution::new(admitted(), Lease(dropped.clone()), 2).unwrap();
        assert!(run.observe(|_| ()).is_err());
        run.submit(|_| S::Submitted).unwrap();
        assert!(run.submit(|_| panic!("must not relaunch")).is_err());
        assert!(run.poll(|_| observation).is_err());
        assert!(run.observe(|_| panic!("must not read")).is_err());
        drop(run);
        assert_eq!(dropped.get(), 0);
    }
    let dropped = Rc::new(Cell::new(0));
    let mut run = FencedExecution::new(admitted(), Lease(dropped.clone()), 1).unwrap();
    assert_eq!(run.submit(|_| S::NotSubmitted), Err(FenceError::Submission));
    assert!(run.observe(|_| ()).is_err());
    drop(run);
    assert_eq!(dropped.get(), 1);
    let dropped = Rc::new(Cell::new(0));
    let mut run = FencedExecution::new(admitted(), Lease(dropped.clone()), 2).unwrap();
    run.submit(|_| S::Submitted).unwrap();
    run.cancel();
    assert_eq!(run.state(), E::CancelPending);
    assert_eq!(run.poll(|_| F::Complete), Err(FenceError::Cancelled));
    assert!(run.observe(|_| ()).is_err());
    drop(run.reclaim().unwrap());
    assert_eq!(dropped.get(), 1);
    let dropped = Rc::new(Cell::new(0));
    let mut run = FencedExecution::new(admitted(), Lease(dropped.clone()), 1).unwrap();
    run.submit(|_| S::Submitted).unwrap();
    run.poll(|_| F::Complete).unwrap();
    run.observe(|_| ()).unwrap();
    assert_eq!(run.completed(), 1);
    assert_eq!(
        run.submit(|_| panic!("budget exhausted")),
        Err(FenceError::Budget)
    );
    drop(run);
    assert_eq!(dropped.get(), 1);
}

#[test]
fn issue_gnet_04_device_public_bundle_is_source_derived_and_cpu_qualified() {
    let source = "fn branch(depth: u32) -> i32 { if depth<2u32 { return 1; } return branch(depth-1u32)+branch(depth-2u32); } fn main() -> i32 { return branch(6u32); }";
    let net = lowered(source);
    let private = compiled::SpeculationLimits::default();
    let device = Device::new(&net, "branch", storage(), private);
    let words = device.compare(&net, "branch", &[eval::Value::U32(6)], limits(), 32, 2);
    assert!(words[10] > 0 && words[11] > 0 && words[16] > 0);
    assert!(*words.last().unwrap() >= 2);
    assert!(words[words.len() - 2] <= 64);
    if let Ok(dir) = std::env::var("INVAR_GNET_DEVICE_BUNDLE_DIR") {
        let dir = PathBuf::from(dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("children.inv"), source).unwrap();
        fs::write(dir.join("children.cu"), device.artifact.source()).unwrap();
        fs::copy(device.dir.join("source.cpp"), dir.join("children.cpp")).unwrap();
        let metal = compiled::emit_device_speculative(
            &net,
            &net.verify().unwrap(),
            "branch",
            compiled::Target::Metal,
            compiled::Limits::default(),
            storage(),
            private,
        )
        .unwrap();
        fs::write(dir.join("children.metal"), metal.source()).unwrap();
        let config = compiled::DeviceConfig {
            fuel: 10000,
            transitions: 100000,
            trace_events: 10000,
            quantum: 32,
            rounds: 64,
            capacity: 2,
        };
        let bytes = device
            .artifact
            .device_layout()
            .unwrap()
            .buffer_bytes(config)
            .unwrap();
        let expected = serde_json::json!({"kernel":"invar_gnet_children_v5","abi":5,"input":[6],"config":config.words(),"buffer_bytes":bytes,"task_frame_words":device.artifact.frame_bytes()/8,"header":&words[..10],"events":&words[21..words.len()-3],"minimum_private_width":2,"cpu_schedule_stats":&words[10..21]});
        fs::write(
            dir.join("expected.json"),
            serde_json::to_vec_pretty(&expected).unwrap(),
        )
        .unwrap();
    }
}
