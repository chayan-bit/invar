#![forbid(unsafe_code)]

use crate::{child_deadline, gnet_words};

use invar::{
    eval,
    general_net::{compiled, oracle, provenance},
    numeric_contract::{F32Bits, NumericPolicy},
};
use provenance::{Error, Limits, ProvenancedArtifact};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

mod abi45;
mod numeric;
mod owned;

const SOURCE: &[u8] = b"fn calc(n: u32) -> i32 { var xs = filled_i32(n, 3); xs[0u32] = xs[0u32] + 2; return xs[0u32]; }";
fn storage() -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: 16,
        allocations: 8,
        cleanup_events: 16,
    }
}

const NUMERIC: &[u8] = b"fn calc(a: f32, b: f32, c: f32) -> f32 { return fma(a,b,c); }";
fn numeric(source: &[u8]) -> ProvenancedArtifact {
    numeric_with(source, compiled::Target::Cpu, Limits::default()).unwrap()
}
fn numeric_with(
    source: &[u8],
    target: compiled::Target,
    limits: Limits,
) -> Result<ProvenancedArtifact, Error> {
    provenance::compile_resumable_owned_numeric(source, "calc", target, limits, storage())
}
fn numeric_separate(source: &[u8]) -> Vec<u8> {
    std::str::from_utf8(source)
        .unwrap()
        .replace("fma(a,b,c)", "a*b+c")
        .into_bytes()
}
fn numeric_args(bits: &[u32]) -> Vec<eval::Value> {
    bits.iter()
        .map(|&n| eval::Value::F32(F32Bits::from_bits(n)))
        .collect()
}
fn numeric_limits() -> oracle::Limits {
    oracle::Limits {
        fuel: 1000,
        transitions: 1000,
        trace_events: 1000,
        array_elements: 16,
    }
}

fn compile(source: &[u8], resumable: bool, limits: compiled::OwnedLimits) -> ProvenancedArtifact {
    let emit = if resumable {
        provenance::compile_resumable
    } else {
        provenance::compile_owned
    };
    emit(
        source,
        "calc",
        compiled::Target::Cpu,
        Limits::default(),
        limits,
    )
    .unwrap()
}

/// Runs real emitted CPU instructions. Hashes identify captured continuation
/// bytes; they do not decide computation or certify arbitrary imported tasks.
struct Cpu {
    dir: PathBuf,
}
impl Cpu {
    fn new(artifact: &ProvenancedArtifact) -> Self {
        let dir = child_deadline::scratch("invar-provenance-owned");
        let owned = artifact.artifact().owned_layout().unwrap();
        let run = if let Some(layout) = artifact.artifact().continuation_layout() {
            format!(
                r#"
    static_assert(sizeof(GnetTask)=={bytes});
    GnetTask task={{}}; unsigned rounds=0;
    // Capture a real yielded prefix, including persistent frames and owned data.
    if (gnet_resume(input,output,trace.data(),fuel,10000,10000,heap.data(),cleanup.data(),&task,1)) return 4;
    std::ofstream checkpoint(argv[3],std::ios::binary);
    checkpoint.write(reinterpret_cast<const char*>(&task),sizeof(task)); checkpoint.close();
    if (mode==1) fuel+=1; // illegal mutation of a started task's source budget
    while (!gnet_resume(input,output,trace.data(),fuel,10000,10000,heap.data(),cleanup.data(),&task,quantum)) {{ if (++rounds>100000) return 5; }}
    auto completed=task;
    if (!gnet_resume(input,output,trace.data(),fuel,10000,10000,heap.data(),cleanup.data(),&task,quantum) || std::memcmp(&completed,&task,sizeof(task))) return 6;
    std::ofstream final_task(argv[4],std::ios::binary); final_task.write(reinterpret_cast<const char*>(&task),sizeof(task));
"#,
                bytes = layout.task_bytes()
            )
        } else {
            "gnet_run(input,output,trace.data(),fuel,10000,10000,heap.data(),cleanup.data());"
                .to_owned()
        };
        let driver = format!(
            r#"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
int main(int argc,char** argv) {{
    if(argc!=5) return 2;
    gnet_u quantum=std::stoull(argv[1]); int mode=std::stoi(argv[2]); gnet_u fuel=1000;
    gnet_w input[1]={{3}},output[10]={{}};
    std::vector<gnet_u> trace(70000),cleanup({clean}); std::vector<gnet_w> heap({heap});
    {run}
    for(auto w:output) std::cout<<w<<' ';
    for(gnet_u i=0;i<(gnet_u)output[7]*7;++i) std::cout<<trace[i]<<' ';
    for(gnet_u i=0;i<(gnet_u)output[8]*4;++i) std::cout<<cleanup[i]<<' ';
    for(auto w:heap) std::cout<<w<<' ';
}}
"#,
            clean = owned.cleanup_words(),
            heap = owned.heap_words()
        );
        fs::write(
            dir.join("candidate.cpp"),
            format!("{}\n{driver}", artifact.artifact().source()),
        )
        .unwrap();
        child_deadline::cxx(
            child_deadline::CXX17_UBSAN_O1,
            &dir.join("candidate.cpp"),
            &dir.join("candidate"),
        );
        Self { dir }
    }
    fn run(&self, quantum: u64, mutate_fuel: bool) -> (Vec<i64>, Vec<u8>, Vec<u8>) {
        let output = child_deadline::command(20)
            .arg(self.dir.join("candidate"))
            .arg(quantum.to_string())
            .arg(if mutate_fuel { "1" } else { "0" })
            .arg(self.dir.join("prefix.bin"))
            .arg(self.dir.join("completed.bin"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let words = String::from_utf8(output.stdout)
            .unwrap()
            .split_whitespace()
            .map(|n| n.parse().unwrap())
            .collect();
        let prefix = fs::read(self.dir.join("prefix.bin")).unwrap_or_default();
        let completed = fs::read(self.dir.join("completed.bin")).unwrap_or_default();
        (words, prefix, completed)
    }
}
impl Drop for Cpu {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap();
    }
}

fn check_actual(artifact: &ProvenancedArtifact, words: &[i64], result: i32) {
    let oracle = artifact
        .observe_oracle(
            &[eval::Value::U32(3)],
            oracle::Limits {
                fuel: 1000,
                transitions: 10000,
                trace_events: 10000,
                ..oracle::Limits::default()
            },
        )
        .unwrap();
    let observed = oracle.observation();
    assert_eq!(observed.semantic.result, Ok(eval::Value::I32(result)));
    assert_eq!(
        &words[..10],
        &[
            0,
            i64::from(result),
            0,
            0,
            observed.semantic.steps_used as i64,
            observed.transitions as i64,
            observed.max_frames as i64,
            observed.events.len() as i64,
            observed.semantic.cleanup_trace.len() as i64,
            1
        ]
    );
    let mut offset = 10;
    for event in &observed.events {
        assert_eq!(&words[offset..offset + 7], &gnet_words::event_i64(event));
        offset += 7;
    }
    let (net, proof) = artifact.checked_net();
    let mir = net.compiled_mir(proof).unwrap();
    for event in &observed.semantic.cleanup_trace {
        assert_eq!(
            mir.functions[words[offset] as usize].declaration_order[words[offset + 1] as usize],
            event.name
        );
        assert_eq!(words[offset + 2], event.length as i64);
        offset += 4;
    }
    assert_eq!(
        words.len() - offset,
        artifact.artifact().owned_layout().unwrap().heap_words()
    );
}

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

const CHILDREN: &[u8] = b"fn branch(depth: u32) -> i32 { if depth<2u32 { return 1; } return branch(depth-1u32)+branch(depth-2u32); } fn calc(n: u32) -> i32 { return branch(n); }";
fn private_limits() -> compiled::SpeculationLimits {
    compiled::SpeculationLimits {
        slots: 2,
        fuel: 10000,
        transitions: 10000,
        trace_events: 1024,
        quantum: 32,
        lookahead: 128,
        scratch_bytes: 1024 * 1024,
    }
}
fn children(target: compiled::Target, spec: compiled::SpeculationLimits) -> ProvenancedArtifact {
    let emit = if target == compiled::Target::Cpu {
        provenance::compile_speculative
    } else {
        provenance::compile_device_speculative
    };
    emit(CHILDREN, "calc", target, Limits::default(), storage(), spec).unwrap()
}
fn device_config() -> compiled::DeviceConfig {
    compiled::DeviceConfig {
        fuel: 10000,
        transitions: 10000,
        trace_events: 1024,
        quantum: 32,
        rounds: 64,
        capacity: 2,
    }
}
fn workset(a: &ProvenancedArtifact, c: compiled::DeviceConfig) -> compiled::Workset {
    let l = a.artifact().device_layout().unwrap();
    let bytes = l.buffer_bytes(c).unwrap();
    l.workset(
        c,
        std::array::from_fn(|i| compiled::BufferRange {
            device: 1,
            allocation: (i + 1) as u64,
            offset: 0,
            bytes: bytes[i],
        }),
    )
    .unwrap()
}
fn buffers(w: &compiled::Workset) -> provenance::DeviceBuffers {
    {
        let mut data: provenance::DeviceBuffers =
            std::array::from_fn(|i| vec![0; w.ranges()[i].bytes]);
        for (bytes, word) in data[3][..48].chunks_exact_mut(8).zip(w.config().words()) {
            bytes.copy_from_slice(&word.to_le_bytes());
        }
        data
    }
}
fn complete(e: &mut provenance::DeviceExecution<provenance::DeviceBuffers>) {
    e.submit(|_| compiled::Submission::Submitted).unwrap();
    e.poll(|_| compiled::FenceObservation::Complete).unwrap();
}
fn forged_leaves(value: &Value) -> Vec<Value> {
    fn paths(v: &Value, path: Vec<String>, out: &mut Vec<Vec<String>>) {
        match v {
            Value::Object(m) => {
                for (k, v) in m {
                    let mut p = path.clone();
                    p.push(k.clone());
                    paths(v, p, out);
                }
            }
            // Mutation of an entire ordered array also detects plans/layouts and ranges.
            _ => out.push(path),
        }
    }
    let mut p = Vec::new();
    paths(value, Vec::new(), &mut p);
    p.into_iter()
        .map(|path| {
            let mut forged = value.clone();
            let mut node = &mut forged;
            for key in path {
                node = &mut node[&key];
            }
            *node = json!("forged");
            forged
        })
        .collect()
}

/// Executes source emission under a concurrent CPU group model. No GPU compiler
/// or driver is loaded. For ABI5, actual child-process exit is the model fence.
struct GroupModel {
    dir: PathBuf,
    child: Option<std::process::Child>,
    bytes: [usize; 12],
    completed: bool,
}
impl GroupModel {
    fn new(a: &ProvenancedArtifact, c: compiled::DeviceConfig) -> Self {
        let artifact = a.artifact();
        let owned = artifact.owned_layout().unwrap();
        let scratch = artifact.speculation_layout().unwrap();
        let bytes = if let Some(d) = artifact.device_layout() {
            d.buffer_bytes(c).unwrap()
        } else {
            [
                (artifact.params().len() * 8).max(8),
                80,
                (c.trace_events * 56).max(8),
                48,
                (owned.heap_words() * 8).max(8),
                (owned.cleanup_words() * 8).max(8),
                artifact.continuation_layout().unwrap().task_bytes(),
                (scratch.slots() * scratch.slot_bytes()).max(8),
                (scratch.heap_words() * 8).max(8),
                (scratch.trace_words() * 8).max(8),
                (scratch.cleanup_words() * 8).max(8),
                112,
            ]
        };
        let dir = child_deadline::scratch("invar-provenance-group");
        let mut shim = String::from(
            "#include <thread>\n#include <barrier>\n#include <fstream>\n#include <vector>\n#include <unistd.h>\n",
        );
        let mut source = artifact.source().to_owned();
        let mut bridge = String::new();
        let run = if artifact.target() == compiled::Target::Cpu {
            format!(
                r#"GnetSpecState spec={{}}; spec.slots=slots.data();spec.heap=private_heap.data();spec.trace=private_trace.data();spec.cleanup=private_cleanup.data();spec.capacity={};
                gnet_w status=14;for(unsigned i=0;i<1024 && !task.done;++i) {{status=gnet_speculate(input.data(),output,trace.data(),config[0],config[1],config[2],heap.data(),cleanup.data(),&task,&spec,config[3],config[4]);++state[11];}}
                state[0]=spec.capacity;state[1]=spec.next_generation;state[2]=spec.spawned;state[3]=spec.committed;state[4]=spec.cancelled;state[5]=spec.replayed;state[6]=spec.guard_failed;state[7]=spec.capacity_serialized;state[8]=spec.spec_transitions;state[9]=spec.owned_replays;state[10]=spec.last_reason;state[12]=status;"#,
                c.capacity
            )
        } else {
            shim.push_str(
                r#"std::barrier group_barrier(32);
#define __device__
#define __global__
#define __shared__ static
#define __syncthreads() group_barrier.arrive_and_wait()
struct Dim {unsigned x=1,y=1,z=1;};
Dim gridDim,blockDim{32,1,1};thread_local Dim threadIdx{0,0,0},blockIdx{0,0,0};
"#,
            );
            if artifact.target() == compiled::Target::Metal {
                shim.push_str(
                    r#"
#define device
#define thread
#define threadgroup
#define constant
#define kernel
using uint=unsigned;using ulong=__UINT64_TYPE__;
#if __SIZEOF_LONG__ == 4
#define long long long
#endif
struct uint3 {uint x,y,z;};
namespace metal {enum class mem_flags {mem_device=1,mem_threadgroup=2};
inline unsigned operator|(mem_flags a,mem_flags b){return unsigned(a)|unsigned(b);}
inline void threadgroup_barrier(unsigned){group_barrier.arrive_and_wait();}}
"#,
                );
                source = source.replace("#include <metal_stdlib>", "").replace(
                    "threadgroup gnet_u root_done; threadgroup gnet_u deltas[8];",
                    "static gnet_u root_done; static gnet_u deltas[8];",
                );
                bridge.push_str(r#"
#undef thread
#undef device
#undef threadgroup
#undef constant
#undef kernel
#undef long
inline void host_entry(const gnet_w* input,gnet_w* output,gnet_u* trace,const gnet_u* config,gnet_w* heap,gnet_u* cleanup,GnetTask* task,GnetSpecSlot* slots,gnet_w* private_heap,gnet_u* private_trace,gnet_u* private_cleanup,gnet_u* state) {
 invar_gnet_children_v5(input,output,trace,config,heap,cleanup,task,slots,private_heap,private_trace,private_cleanup,state,threadIdx.x,uint3{0,0,0},uint3{1,1,1},uint3{32,1,1});}
"#);
            } else {
                shim.push_str("#define host_entry invar_gnet_children_v5\n");
            }
            "std::vector<std::thread> workers;for(unsigned i=0;i<32;++i)workers.emplace_back([&,i]{threadIdx.x=i;host_entry(input.data(),output,trace.data(),config,heap.data(),cleanup.data(),&task,slots.data(),private_heap.data(),private_trace.data(),private_cleanup.data(),state);});for(auto& worker:workers)worker.join();".to_owned()
        };
        let driver = format!(
            r#"
int main(int argc,char** argv) {{
    if(argc!=3)return 2;
    std::vector<gnet_w> input({input}),heap({heap}),private_heap({pheap});
    std::vector<gnet_u> trace({trace}),cleanup({clean}),private_trace({ptrace}),private_cleanup({pclean});
    input[0]=std::stoll(argv[1]);gnet_w output[10]={{}};GnetTask task={{}};
    std::vector<GnetSpecSlot> slots({slots});gnet_u state[14]={{}};
    gnet_u config[6]={{{fuel},{transitions},{events},{quantum},{rounds},{capacity}}};
    {run}
    const void* data[12]={{input.data(),output,trace.data(),config,heap.data(),cleanup.data(),&task,slots.data(),private_heap.data(),private_trace.data(),private_cleanup.data(),state}};
    const unsigned sizes[12]={{{sizes}}};
    for(unsigned i=0;i<12;++i) {{std::ofstream out(std::string(argv[2])+"/"+std::to_string(i)+".bin",std::ios::binary);out.write(reinterpret_cast<const char*>(data[i]),sizes[i]);if(!out)return 3;}}
    return 0;
}}
"#,
            input = bytes[0] / 8,
            heap = bytes[4] / 8,
            pheap = bytes[8] / 8,
            trace = bytes[2] / 8,
            clean = bytes[5] / 8,
            ptrace = bytes[9] / 8,
            pclean = bytes[10] / 8,
            slots = scratch.slots(),
            fuel = c.fuel,
            transitions = c.transitions,
            events = c.trace_events,
            quantum = c.quantum,
            rounds = c.rounds,
            capacity = c.capacity,
            sizes = bytes
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        fs::write(
            dir.join("group.cpp"),
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
            &dir.join("group.cpp"),
            &dir.join("group"),
        );
        Self {
            dir,
            child: None,
            bytes,
            completed: false,
        }
    }
    fn launch(&mut self, input: u32) -> compiled::Submission {
        assert!(self.child.is_none());
        self.completed = false;
        self.child = Some(
            child_deadline::command(20)
                .arg(self.dir.join("group"))
                .arg(input.to_string())
                .arg(&self.dir)
                .spawn()
                .unwrap(),
        );
        compiled::Submission::Submitted
    }
    fn wait(&mut self) -> compiled::FenceObservation {
        let status = self.child.take().unwrap().wait().unwrap();
        self.completed = status.success();
        if self.completed {
            compiled::FenceObservation::Complete
        } else {
            compiled::FenceObservation::FailedQuiescent
        }
    }
    fn read(&self) -> provenance::DeviceBuffers {
        assert!(self.completed);
        std::array::from_fn(|i| {
            let b = fs::read(self.dir.join(format!("{i}.bin"))).unwrap();
            assert_eq!(b.len(), self.bytes[i]);
            b
        })
    }
}
impl Drop for GroupModel {
    fn drop(&mut self) {
        assert!(self.child.is_none());
        fs::remove_dir_all(&self.dir).unwrap();
    }
}
fn words(b: &[u8]) -> Vec<u64> {
    b.chunks_exact(8)
        .map(|w| u64::from_ne_bytes(w.try_into().unwrap()))
        .collect()
}
fn compare_scalar_checkpoint(a: &ProvenancedArtifact, b: &provenance::DeviceBuffers, input: u32) {
    let observed = a
        .observe_oracle(
            &[eval::Value::U32(input)],
            oracle::Limits {
                fuel: 10000,
                transitions: 10000,
                trace_events: 1024,
                ..oracle::Limits::default()
            },
        )
        .unwrap()
        .observation();
    let Ok(eval::Value::I32(value)) = observed.semantic.result else {
        panic!("expected scalar success")
    };
    let header = words(&b[1]);
    assert_eq!(
        header,
        vec![
            0,
            value as u64,
            0,
            0,
            observed.semantic.steps_used,
            observed.transitions,
            observed.max_frames as u64,
            observed.events.len() as u64,
            0,
            0
        ]
    );
    let trace = words(&b[2]);
    for (i, event) in observed.events.iter().enumerate() {
        assert_eq!(&trace[i * 7..i * 7 + 7], &gnet_words::event(event));
    }
    let state = words(&b[11]);
    assert_eq!(state[12], 0);
    assert!(state[2] > 0 && state[3] > 0 && state[8] > 0);
}
