//! ABI4 emitted CPU execution and ABI5/13 CPU workgroup models. No device launch
//! or driver fence is inferred from these source-derived reference checks.
use super::{child_deadline, corpus_sample};

use super::{corpus_cpu as base, corpus_owned as owned};
use invar::{
    check, eval, front,
    general_net::{self, Net, compiled, oracle},
};
use serde_json::{Value as Json, json};
use std::{collections::BTreeMap, fs, path::PathBuf, process::Command};

#[path = "corpus_children/denominator.rs"]
mod denominator;

fn checked(bytes: &[u8]) -> Result<check::CheckedProgram, front::FrontError> {
    let p = check::check(front::parse(bytes)?, true)?;
    invar::ownership::check(&p)?;
    Ok(p)
}
fn lower(bytes: &[u8]) -> Net {
    Net::lower(&checked(bytes).unwrap(), general_net::Limits::default()).unwrap()
}
fn args(row: &Json) -> Vec<eval::Value> {
    row.as_array()
        .unwrap()
        .iter()
        .map(|v| match v["type"].as_str().unwrap() {
            "i32" => eval::Value::I32(i32::try_from(v["value"].as_i64().unwrap()).unwrap()),
            "u32" => eval::Value::U32(u32::try_from(v["value"].as_u64().unwrap()).unwrap()),
            _ => panic!("scalar input"),
        })
        .collect()
}
fn limits(row: &Json) -> oracle::Limits {
    oracle::Limits {
        fuel: row["fuel"].as_u64().unwrap(),
        transitions: row["limits"]["transitions"].as_u64().unwrap_or(100000),
        trace_events: row["limits"]["trace_events"].as_u64().unwrap_or(100000) as usize,
        array_elements: row["limits"]["array_elements"].as_u64().unwrap_or(128) as usize,
    }
}
fn storage() -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: 128,
        allocations: 64,
        cleanup_events: 128,
    }
}
fn private() -> compiled::SpeculationLimits {
    compiled::SpeculationLimits {
        fuel: 1000,
        transitions: 10000,
        trace_events: 2048,
        quantum: 32,
        ..compiled::SpeculationLimits::default()
    }
}
fn emit(
    net: &Net,
    entry: &str,
    target: compiled::Target,
    spec: compiled::SpeculationLimits,
    storage: compiled::OwnedLimits,
) -> Result<compiled::Artifact, compiled::Error> {
    let emitter = if target == compiled::Target::Cpu {
        compiled::emit_speculative
    } else {
        compiled::emit_device_speculative
    };
    emitter(
        net,
        &net.verify().unwrap(),
        entry,
        target,
        compiled::Limits::default(),
        storage,
        spec,
    )
}
fn rejected(error: compiled::Error) -> Json {
    assert!(
        matches!(
            error,
            compiled::Error::UnsupportedType | compiled::Error::DeviceSpeculationOwner
        ),
        "unexpected admission error {error:?}"
    );
    json!({"status":"unsupported","reason":format!("{error:?}")})
}
fn fork(artifact: &compiled::Artifact) -> Json {
    let p = artifact.fork_plan().unwrap();
    let site = |s: compiled::CallSite| json!({"function":s.function,"block":s.block,"position":s.position});
    json!({"candidates":p.candidates.iter().map(|c|json!({"source":site(c.source),"target":site(c.target),"callee":c.callee,"projection":c.projection,"scalar_values":c.scalar_values,"scalar_places":c.scalar_places,"replay_owners":c.replay_owners})).collect::<Vec<_>>(),"serialized":p.serialized.iter().map(|s|json!({"source":site(s.source),"target":site(s.target),"reason":format!("{:?}",s.reason)})).collect::<Vec<_>>()})
}
fn artifact_record(a: &compiled::Artifact) -> Json {
    let s = a.speculation_layout().unwrap();
    let c = a.continuation_layout().unwrap();
    let l = s.limits();
    let o = a.owned_layout().unwrap();
    json!({"abi":a.abi_version(),"target":format!("{:?}",a.target()),"source_sha256":base::digest(a.source().as_bytes()),"states":a.states(),"frame_bytes":a.frame_bytes(),"params":a.params().iter().map(|p|format!("{p:?}")).collect::<Vec<_>>(),"result":format!("{:?}",a.result()),"task_bytes":c.task_bytes(),"owned":{"elements":o.elements(),"allocations":o.allocations(),"cleanup_events":o.cleanup_events(),"heap_words":o.heap_words(),"cleanup_words":o.cleanup_words(),"storage_bytes":o.storage_bytes()},"speculation":{"slots":s.slots(),"slot_bytes":s.slot_bytes(),"heap_words":s.heap_words(),"trace_words":s.trace_words(),"cleanup_words":s.cleanup_words(),"scratch_bytes":s.scratch_bytes(),"limits":{"slots":l.slots,"fuel":l.fuel,"transitions":l.transitions,"trace_events":l.trace_events,"quantum":l.quantum,"lookahead":l.lookahead,"scratch_bytes":l.scratch_bytes}},"fork_plan":fork(a),"device_layout":a.device_layout().map(|d|json!({"kernel":d.kernel_name(),"groups":d.groups(),"threads":d.group_threads(),"state_words":d.state_words(),"max_rounds":d.max_rounds()}))})
}

const SHIM: &str = r#"
#include <barrier>
std::barrier group_barrier(32);
#define __device__
#define __global__
#define __shared__ static
#define __syncthreads() group_barrier.arrive_and_wait()
struct Dim {unsigned x=1,y=1,z=1;};
Dim gridDim,blockDim{32,1,1};thread_local Dim threadIdx{0,0,0},blockIdx{0,0,0};
"#;
const METAL_SHIM: &str = r#"
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
"#;
const METAL_BRIDGE: &str = r#"
#undef thread
#undef device
#undef threadgroup
#undef constant
#undef kernel
#undef long
inline void host_entry(const gnet_w* input,gnet_w* output,gnet_u* trace,const gnet_u* config,gnet_w* heap,gnet_u* cleanup,GnetTask* task,GnetSpecSlot* slots,gnet_w* private_heap,gnet_u* private_trace,gnet_u* private_cleanup,gnet_u* state){
 invar_gnet_children_v5(input,output,trace,config,heap,cleanup,task,slots,private_heap,private_trace,private_cleanup,state,threadIdx.x,uint3{0,0,0},uint3{1,1,1},uint3{blockDim.x,1,1});}
"#;
struct Harness {
    cpu: owned::Cpu,
    artifact: compiled::Artifact,
}
struct Observation {
    run: owned::Run,
    stats: Vec<i64>,
    state: i64,
    width: i64,
}
impl Harness {
    fn new(artifact: compiled::Artifact) -> Self {
        let dir = child_deadline::scratch("invar-corpus-children");
        let owned = artifact.owned_layout().unwrap();
        let scratch = artifact.speculation_layout().unwrap();
        let device = artifact.device_layout().is_some();
        let mut source = if !device {
            artifact.source().to_owned()
        } else if artifact.target() == compiled::Target::Cuda {
            format!(
                "{SHIM}\n#define host_entry invar_gnet_children_v5\n{}",
                artifact.source()
            )
        } else {
            format!(
                "{SHIM}\n{METAL_SHIM}\n{}\n{METAL_BRIDGE}",
                artifact
                    .source()
                    .replace("#include <metal_stdlib>", "")
                    .replace(
                        "threadgroup gnet_u root_done; threadgroup gnet_u deltas[8];",
                        "static gnet_u root_done; static gnet_u deltas[8];"
                    )
            )
        };
        let setup = if device {
            r#"gnet_u config[6]={fuel,trans,cap,1,64,capacity};if(mode=="generation")state[1]=UINT64_MAX;if(mode=="geometry")blockDim.x=16;if(mode=="config")config[4]=65;"#
        } else {
            r#"GnetSpecState spec={};spec.capacity=capacity;spec.slots=slots.data();spec.heap=private_heap.data();spec.trace=private_trace.data();spec.cleanup=private_cleanup.data();if(mode=="missing")spec.heap=nullptr;if(mode=="alias")spec.heap=heap.data();if(mode=="generation")spec.next_generation=UINT64_MAX;"#
        };
        let step = if device {
            r#"config[3]=q;std::vector<std::thread> group;for(unsigned i=0;i<32;++i)group.emplace_back([&,i]{threadIdx.x=i;host_entry(input.data(),output,trace.data(),config,heap.data(),cleanup.data(),&task,slots.data(),private_heap.data(),private_trace.data(),private_cleanup.data(),state);});for(auto& t:group)t.join();code=state[12];width=state[13];"#
        } else {
            r#"std::vector<gnet_u> before;for(const auto& slot:slots)before.push_back(slot.task.transitions);code=gnet_speculate(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data(),&task,&spec,q,1);gnet_u advanced=0;for(gnet_u i=0;i<slots.size();++i)if(slots[i].task.transitions>before[i])++advanced;if(advanced>width)width=advanced;"#
        };
        let stats = if device {
            "state[2],state[3],state[4],state[5],state[6],state[7],state[8],state[9],state[10],task.next_frame,task.depth"
        } else {
            "spec.spawned,spec.committed,spec.cancelled,spec.replayed,spec.guard_failed,spec.capacity_serialized,spec.spec_transitions,spec.owned_replays,spec.last_reason,task.next_frame,task.depth"
        };
        let driver = format!(
            r#"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>
int main(int argc,char** argv){{
 if(argc<6)return 2;
 gnet_u fuel=std::stoull(argv[1]),trans=std::stoull(argv[2]),cap=std::stoull(argv[3]),capacity=std::stoull(argv[5]);if(cap>1000000)return 2;
 std::vector<gnet_u> quanta;std::stringstream schedule(argv[4]);std::string piece;while(std::getline(schedule,piece,',')){{auto q=std::stoull(piece);if(q==0||q>1000000)return 2;quanta.push_back(q);}}if(quanta.empty())return 2;
 std::vector<gnet_w> input;for(int i=6;i<argc;++i)input.push_back(std::stoll(argv[i]));if(input.empty())input.push_back(0);input.resize(std::max<size_t>(input.size(),{input_words}),0);
 gnet_w output[10]={{}};GnetTask task={{}};gnet_u state[14]={{}};
 static_assert(sizeof(GnetTask)=={task_bytes});static_assert(sizeof(GnetSpecSlot)=={slot_bytes});
 std::vector<gnet_u> trace(std::max<gnet_u>(1,cap*7)),cleanup({clean}),private_trace({ptrace}),private_cleanup({pclean});
 std::vector<gnet_w> heap({heap}),private_heap({pheap});std::vector<GnetSpecSlot> slots({slots});
 std::string mode=std::getenv("GNET_CORPUS_MODE")?std::getenv("GNET_CORPUS_MODE"):"";
 if(mode=="fence_pause")std::this_thread::sleep_for(std::chrono::milliseconds(100));
 {setup}
 gnet_u epochs=0,width=0;gnet_w code=14;std::vector<gnet_u> committed_trace,committed_cleanup;
 auto advance=[&](gnet_u q){{{step}}};
 while(!task.done&&epochs<{}ull){{
  auto previous_steps=task.steps,previous_trans=task.transitions;advance(quanta[epochs%quanta.size()]);++epochs;
  if(code!=0&&code!=14)break;
  if(task.steps<previous_steps||task.transitions<previous_trans||task.used>cap||task.cleanup_used>{clean_cap}||task.allocated>{alloc_cap})return 20;
  for(size_t i=0;i<committed_trace.size();++i)if(committed_trace[i]!=trace[i])return 21;
  for(size_t i=0;i<committed_cleanup.size();++i)if(committed_cleanup[i]!=cleanup[i])return 22;
  committed_trace.assign(trace.begin(),trace.begin()+task.used*7);committed_cleanup.assign(cleanup.begin(),cleanup.begin()+task.cleanup_used*4);
  if(!task.done)for(auto word:output)if(word!=0)return 23;
  if(mode=="stale")for(auto& slot:slots)if(slot.valid&&slot.task.done)slot.completion_generation=0;
 }}
 gnet_u replay=0;
 if(task.done){{auto output_before=std::vector<gnet_w>(output,output+10),heap_before=heap;auto trace_before=trace,cleanup_before=cleanup;GnetTask task_before=task;
  advance(quanta[0]);if(code!=0||output_before!=std::vector<gnet_w>(output,output+10)||heap_before!=heap||trace_before!=trace||cleanup_before!=cleanup||std::memcmp(&task,&task_before,sizeof(task)))return 24;replay=1;
 }}else if(code==0||code==14){{std::cerr<<"incomplete epochs="<<epochs<<" steps="<<task.steps<<" transitions="<<task.transitions<<"\n";return 25;}}
 for(auto word:output)std::cout<<word<<' ';
 for(gnet_u i=0;i<(gnet_u)output[7]*7;++i)std::cout<<trace[i]<<' ';
 for(gnet_u i=0;i<(gnet_u)output[8]*4;++i)std::cout<<cleanup[i]<<' ';
 for(auto word:heap)std::cout<<word<<' ';
 gnet_u stats[11]={{{stats}}};for(auto word:stats)std::cout<<word<<' ';
 std::cout<<code<<' '<<epochs<<' '<<width<<' '<<replay<<'\n';
}}
"#,
            if device { 1024 } else { 1000000 },
            input_words = artifact
                .entry_layout()
                .map_or(0, |entry| entry.input_words()),
            task_bytes = artifact.continuation_layout().unwrap().task_bytes(),
            slot_bytes = scratch.slot_bytes(),
            clean = owned.cleanup_words().max(1),
            ptrace = scratch.trace_words().max(1),
            pclean = scratch.cleanup_words().max(1),
            heap = owned.heap_words().max(1),
            pheap = scratch.heap_words().max(1),
            slots = scratch.slots().max(1),
            clean_cap = owned.cleanup_events(),
            alloc_cap = owned.allocations()
        );
        source.push_str(&driver);
        fs::write(dir.join("source.cpp"), &source).unwrap();
        let std = if device { "-std=c++20" } else { "-std=c++17" };
        let flags = [
            std,
            "-pthread",
            "-O1",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
        ];
        child_deadline::cxx(&flags, &dir.join("source.cpp"), &dir.join("run"));
        if let Some(report_dir) = std::env::var_os("INVAR_GNET_CORPUS_REPORT_DIR") {
            let recorded = PathBuf::from(report_dir).join("artifacts").join(format!(
                "abi{}-{:?}-{}",
                artifact.abi_version(),
                artifact.target(),
                base::digest(artifact.source().as_bytes())
            ));
            fs::create_dir_all(&recorded).unwrap();
            fs::write(recorded.join("emitted-source.txt"), artifact.source()).unwrap();
            fs::copy(dir.join("source.cpp"), recorded.join("cpu-harness.cpp")).unwrap();
            fs::write(
                recorded.join("artifact.json"),
                serde_json::to_vec_pretty(&artifact_record(&artifact)).unwrap(),
            )
            .unwrap();
        }
        Self {
            cpu: owned::Cpu::for_harness(&artifact, dir),
            artifact,
        }
    }
    fn command(
        &self,
        inputs: &[eval::Value],
        limits: oracle::Limits,
        quanta: &[u64],
        capacity: usize,
        mode: &str,
    ) -> Command {
        let mut cmd = child_deadline::command(10);
        cmd.arg(self.cpu.dir.join("run"))
            .env("GNET_CORPUS_MODE", mode)
            .args([
                limits.fuel.to_string(),
                limits.transitions.to_string(),
                limits.trace_events.to_string(),
                quanta
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                capacity.to_string(),
            ]);
        for input in inputs {
            cmd.arg(match input {
                eval::Value::I32(n) => n.to_string(),
                eval::Value::U32(n) => n.to_string(),
                _ => panic!("scalar"),
            });
        }
        cmd
    }
    fn execute(
        &self,
        inputs: &[eval::Value],
        limits: oracle::Limits,
        quanta: &[u64],
        capacity: usize,
        mode: &str,
    ) -> Observation {
        let output = self
            .command(inputs, limits, quanta, capacity, mode)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "children run {:?} {mode} target={:?} inputs={inputs:?} limits={limits:?} quanta={quanta:?}: {}",
            output.status,
            self.artifact.target(),
            String::from_utf8_lossy(&output.stderr)
        );
        let mut words = String::from_utf8(output.stdout)
            .unwrap()
            .split_whitespace()
            .map(|w| w.parse().unwrap())
            .collect::<Vec<i64>>();
        let tail = words.split_off(words.len() - 15);
        Observation {
            run: owned::Run {
                words,
                submissions: tail[12] as u64,
                yields: 0,
                replay_checked: tail[14] == 1,
            },
            stats: tail[..11].to_vec(),
            state: tail[11],
            width: tail[13],
        }
    }
    fn compare(&self, net: &Net, expected: &oracle::Observation, run: &Observation) -> Json {
        assert_eq!(run.state, 0);
        assert!(run.run.replay_checked);
        let mut receipt = self.cpu.compare(net, expected, &run.run);
        receipt["artifact"] = artifact_record(&self.artifact);
        receipt["execution_kind"] = json!(if self.artifact.device_layout().is_none() {
            "emitted_cpu"
        } else {
            "cpu_workgroup_model"
        });
        receipt["driver_fence"] = json!("not_performed");
        receipt["source_derived_stats"] = json!({"spawned":run.stats[0],"committed":run.stats[1],"cancelled":run.stats[2],"replayed":run.stats[3],"guard_failed":run.stats[4],"capacity_serialized":run.stats[5],"private_transitions":run.stats[6],"owner_replays":run.stats[7],"last_reason":run.stats[8],"next_frame":run.stats[9],"depth":run.stats[10],"maximum_private_width":run.width});
        receipt
    }
}
fn observed(o: &oracle::Observation) -> Json {
    json!({"status":if o.failure==Some(oracle::ToolFailure::Unsupported){"unsupported"}else{"executed"},"reason":o.failure.map(|f|format!("{f:?}")),"semantic":base::semantic(&o.semantic),"source_steps":o.semantic.steps_used,"transitions":o.transitions,"events":o.events.len(),"cleanup":base::cleanup(&o.semantic)})
}
fn report(name: &str, rows: Vec<Json>, extra: Json) {
    if let Some(dir) = std::env::var_os("INVAR_GNET_CORPUS_REPORT_DIR") {
        let dir = PathBuf::from(dir);
        fs::create_dir_all(&dir).unwrap();
        let counts = |k: &str| {
            let mut c = BTreeMap::<String, usize>::new();
            for r in &rows {
                *c.entry(r[k]["status"].as_str().unwrap().to_owned())
                    .or_default() += 1;
            }
            c
        };
        let d = json!({"schema_version":1,"profile":if cfg!(debug_assertions){"debug"}else{"release"},"case_count":rows.len(),"counts":{"source":counts("source"),"oracle":counts("oracle"),"abi4_cpu":counts("abi4_cpu"),"abi5_cuda_cpu_model":counts("abi5_cuda_cpu_model"),"abi5_metal_cpu_model":counts("abi5_metal_cpu_model")},"gpu_execution":"not_performed","driver_fence":"not_performed","rows":rows,"extra":extra});
        fs::write(
            dir.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&d).unwrap(),
        )
        .unwrap();
    }
}

pub fn full_denominator() {
    denominator::run();
}

fn target_key(target: compiled::Target) -> &'static str {
    match target {
        compiled::Target::Cpu => "abi4_cpu",
        compiled::Target::Cuda => "abi5_cuda_cpu_model",
        compiled::Target::Metal => "abi5_metal_cpu_model",
    }
}
fn row_for_run(
    id: &str,
    target: compiled::Target,
    source: &eval::Observation,
    expected: &oracle::Observation,
    receipt: Json,
) -> Json {
    let mut row = json!({"id":id,"target":format!("{target:?}"),"source":base::recorded(source),"oracle":observed(expected),"abi4_cpu":{"status":"not_attempted"},"abi5_cuda_cpu_model":{"status":"not_attempted"},"abi5_metal_cpu_model":{"status":"not_attempted"}});
    row[target_key(target)] = receipt;
    row
}
fn compare_source(
    h: &Harness,
    net: &Net,
    entry: &str,
    inputs: &[eval::Value],
    limits: oracle::Limits,
    mode: &str,
    capacity: usize,
) -> (Json, Observation) {
    let p = net.compiled_mir(&net.verify().unwrap()).unwrap();
    let expected =
        oracle::observe_call(net, &net.verify().unwrap(), entry, inputs, limits).unwrap();
    // Named checked-source observation is independent of net lowering. The source
    // is selected by caller; p here additionally binds function names in receipts.
    let run = h.execute(inputs, limits, &[1], capacity, mode);
    let mut receipt = h.compare(net, &expected, &run);
    receipt["entry"] = json!(entry);
    receipt["function_names"] = json!(p.functions.iter().map(|f| &f.name).collect::<Vec<_>>());
    receipt["mode"] = json!(mode);
    (receipt, run)
}
pub fn failures_and_source_order() {
    let mut rows = Vec::new();
    let mut comparisons = 0;
    let mut fuel_prefixes = 0;
    let mut maximum_width = 0;
    let branch_bytes = fs::read(base::root().join("children/runtime_branch.inv")).unwrap();
    let branch_checked = checked(&branch_bytes).unwrap();
    let branch = lower(&branch_bytes);
    let ordered_bytes = fs::read(base::root().join("children/ordered_runtime.inv")).unwrap();
    let ordered_checked = checked(&ordered_bytes).unwrap();
    let ordered = lower(&ordered_bytes);
    let inputs = [
        eval::Value::U32(30),
        eval::Value::U32(0),
        eval::Value::I32(3),
    ];
    let bound = oracle::Limits {
        fuel: 1000,
        transitions: 100000,
        trace_events: 10000,
        array_elements: 128,
    };
    for target in [
        compiled::Target::Cpu,
        compiled::Target::Cuda,
        compiled::Target::Metal,
    ] {
        let h = Harness::new(emit(&branch, "run", target, private(), storage()).unwrap());
        assert!(!h.artifact.fork_plan().unwrap().candidates.is_empty());
        for mode in ["", "generation", "stale", "missing", "alias"] {
            if target != compiled::Target::Cpu && matches!(mode, "missing" | "alias") {
                continue;
            }
            let source = eval::named::observe_named(&branch_checked, "run", &inputs, bound.fuel);
            let expected =
                oracle::observe_call(&branch, &branch.verify().unwrap(), "run", &inputs, bound)
                    .unwrap();
            base::equal(&source, &expected.semantic, "child guard source");
            let (receipt, run) = compare_source(&h, &branch, "run", &inputs, bound, mode, 2);
            if mode.is_empty() {
                assert!(
                    run.stats[0] > 0 && run.stats[1] > 0,
                    "actual child must commit"
                );
            } else {
                assert_eq!(run.stats[1], 0, "{mode} cannot consume private transcript");
                if mode == "stale" {
                    assert!(run.stats[3] > 0);
                    assert_eq!(run.stats[8], 7);
                }
                if mode == "generation" {
                    assert_eq!(run.stats[0], 0);
                    assert_eq!(run.stats[8], 9);
                }
                if matches!(mode, "missing" | "alias") {
                    assert_eq!(run.stats[0], 0);
                    assert_eq!(run.stats[8], 10);
                }
            }
            rows.push(row_for_run(
                &format!("guard_{mode}"),
                target,
                &source,
                &expected,
                receipt,
            ));
            comparisons += 1;
        }
        let input = [
            eval::Value::U32(30),
            eval::Value::U32(1),
            eval::Value::I32(i32::MAX),
        ];
        let source = eval::named::observe_named(&branch_checked, "run", &input, bound.fuel);
        let expected =
            oracle::observe_call(&branch, &branch.verify().unwrap(), "run", &input, bound).unwrap();
        base::equal(&source, &expected.semantic, "projection guard source");
        let run = h.execute(&input, bound, &[1], 2, "");
        assert_eq!(run.stats[0], 0);
        assert!(run.stats[4] > 0);
        rows.push(row_for_run(
            "projection_checked_overflow",
            target,
            &source,
            &expected,
            h.compare(&branch, &expected, &run),
        ));
        comparisons += 1;
        if target != compiled::Target::Cpu {
            for mode in ["geometry", "config"] {
                let run = h.execute(&inputs, bound, &[1], 2, mode);
                assert_eq!(run.state, 8);
                assert_eq!(&run.run.words[..10], &[0; 10]);
                assert_eq!(run.stats[9], 0);
                let mut row = json!({"id":format!("invalid_{mode}"),"source":{"status":"not_attempted","reason":"deliberately invalid backend geometry/config is not source evaluation"},"oracle":{"status":"not_attempted"},"abi4_cpu":{"status":"not_attempted"},"abi5_cuda_cpu_model":{"status":"not_attempted"},"abi5_metal_cpu_model":{"status":"not_attempted"}});
                row[target_key(target)] = json!({"status":"backend_admission_failure","state":run.state,"output_unpublished":true,"primary_task_not_started":true});
                rows.push(row);
            }
        }
        for (label, spec) in [
            (
                "private_fuel",
                compiled::SpeculationLimits {
                    fuel: 1,
                    ..private()
                },
            ),
            (
                "private_transitions",
                compiled::SpeculationLimits {
                    transitions: 0,
                    ..private()
                },
            ),
            (
                "private_trace",
                compiled::SpeculationLimits {
                    trace_events: 0,
                    ..private()
                },
            ),
        ] {
            let h = Harness::new(emit(&branch, "run", target, spec, storage()).unwrap());
            let source = eval::named::observe_named(&branch_checked, "run", &inputs, bound.fuel);
            let expected =
                oracle::observe_call(&branch, &branch.verify().unwrap(), "run", &inputs, bound)
                    .unwrap();
            base::equal(&source, &expected.semantic, label);
            let run = h.execute(&inputs, bound, &[1], 2, "");
            assert_eq!(run.stats[1], 0);
            assert!(run.stats[3] > 0);
            rows.push(row_for_run(
                label,
                target,
                &source,
                &expected,
                h.compare(&branch, &expected, &run),
            ));
            comparisons += 1;
        }
        let h = Harness::new(emit(&ordered, "run", target, private(), storage()).unwrap());
        for value in [3, i32::MAX] {
            let input = [
                eval::Value::U32(0),
                eval::Value::U32(0),
                eval::Value::I32(value),
            ];
            let limits = oracle::Limits { fuel: 200, ..bound };
            let source = eval::named::observe_named(&ordered_checked, "run", &input, limits.fuel);
            let expected =
                oracle::observe_call(&ordered, &ordered.verify().unwrap(), "run", &input, limits)
                    .unwrap();
            base::equal(&source, &expected.semantic, "older divergent prefix");
            let later = ordered
                .compiled_mir(&ordered.verify().unwrap())
                .unwrap()
                .functions
                .iter()
                .position(|f| f.name == "later")
                .unwrap();
            assert!(
                expected
                    .events
                    .iter()
                    .all(|e| e.token.block.function != later)
            );
            let run = h.execute(&input, limits, &[1], 2, "");
            assert!(run.stats[0] > 0);
            assert_eq!(run.stats[1], 0);
            assert!(run.stats[2] > 0);
            rows.push(row_for_run(
                "oldest_divergence_hides_later_trap",
                target,
                &source,
                &expected,
                h.compare(&ordered, &expected, &run),
            ));
            comparisons += 1;
        }
        let input = [
            eval::Value::U32(1),
            eval::Value::U32(0),
            eval::Value::I32(3),
        ];
        let complete = eval::named::observe_named(&ordered_checked, "run", &input, bound.fuel);
        let mut prefixes = Vec::new();
        for fuel in 1..=complete.steps_used {
            let limits = oracle::Limits { fuel, ..bound };
            let source = eval::named::observe_named(&ordered_checked, "run", &input, fuel);
            let expected =
                oracle::observe_call(&ordered, &ordered.verify().unwrap(), "run", &input, limits)
                    .unwrap();
            base::equal(
                &source,
                &expected.semantic,
                "all ordered trap fuel prefixes",
            );
            let run = h.execute(&input, limits, &[1], 2, "");
            prefixes.push(h.compare(&ordered, &expected, &run));
            comparisons += 1;
            fuel_prefixes += 1;
        }
        let expected =
            oracle::observe_call(&ordered, &ordered.verify().unwrap(), "run", &input, bound)
                .unwrap();
        let mut row = row_for_run(
            "every_positive_ordered_trap_fuel_prefix",
            target,
            &complete,
            &expected,
            json!({"status":"executed","prefix_count":prefixes.len(),"prefixes":prefixes}),
        );
        row["source_ordered_fuel_denominator"] = json!(complete.steps_used);
        rows.push(row);
        for cap in [0, 1, 7, 20] {
            for trace in [false, true] {
                let limits = if trace {
                    oracle::Limits {
                        trace_events: cap,
                        ..bound
                    }
                } else {
                    oracle::Limits {
                        transitions: cap as u64,
                        ..bound
                    }
                };
                let expected = oracle::observe_call(
                    &ordered,
                    &ordered.verify().unwrap(),
                    "run",
                    &input,
                    limits,
                )
                .unwrap();
                let source =
                    eval::named::observe_named(&ordered_checked, "run", &input, bound.fuel);
                let run = h.execute(&input, limits, &[1], 2, "");
                let mut receipt = h.compare(&ordered, &expected, &run);
                receipt["bound_kind"] = json!(if trace {
                    "private_protocol_trace_cap"
                } else {
                    "private_protocol_transition_cap"
                });
                receipt["source_equivalence"] = json!(
                    "explicit backend tool prefix differs from completing source; no source bound introduced"
                );
                rows.push(row_for_run(
                    "explicit_primary_tool_prefix",
                    target,
                    &source,
                    &expected,
                    receipt,
                ));
                comparisons += 1;
            }
        }
        let bytes = fs::read(base::root().join("children/runtime_frontier.inv")).unwrap();
        let net = lower(&bytes);
        let checked = checked(&bytes).unwrap();
        let h = Harness::new(emit(&net, "fanout", target, private(), storage()).unwrap());
        let frontier_bound = oracle::Limits {
            fuel: 5000,
            ..bound
        };
        let input = [eval::Value::U32(4), eval::Value::U32(1)];
        let source = eval::named::observe_named(&checked, "fanout", &input, frontier_bound.fuel);
        let expected = oracle::observe_call(
            &net,
            &net.verify().unwrap(),
            "fanout",
            &input,
            frontier_bound,
        )
        .unwrap();
        base::equal(&source, &expected.semantic, "runtime-shaped frontier");
        let run = h.execute(&input, frontier_bound, &[1], 2, "");
        assert!(run.stats[0] > 0 && run.stats[1] > 0);
        maximum_width = maximum_width.max(run.width);
        rows.push(row_for_run(
            "runtime_frontier_actual_child_commits",
            target,
            &source,
            &expected,
            h.compare(&net, &expected, &run),
        ));
        comparisons += 1;
    }
    assert!(
        maximum_width >= 2,
        "actual CPU/model private frontier has width at least2"
    );
    let bytes = fs::read(base::root().join("children/owner_replay.inv")).unwrap();
    let net = lower(&bytes);
    let p = checked(&bytes).unwrap();
    let input = [eval::Value::U32(30), eval::Value::I32(3)];
    for (entry, storage, limits) in [
        ("run", storage(), bound),
        (
            "priority",
            compiled::OwnedLimits {
                elements: 2,
                ..storage()
            },
            oracle::Limits {
                array_elements: 2,
                ..bound
            },
        ),
    ] {
        let h = Harness::new(emit(&net, entry, compiled::Target::Cpu, private(), storage).unwrap());
        let source = eval::named::observe_named(&p, entry, &input, bound.fuel);
        let expected =
            oracle::observe_call(&net, &net.verify().unwrap(), entry, &input, limits).unwrap();
        if entry == "run" {
            base::equal(&source, &expected.semantic, "owner replay");
        } else {
            assert_eq!(expected.failure, Some(oracle::ToolFailure::Allocation));
            assert_eq!(base::semantic(&source)["result"]["value"], 13);
        }
        let run = h.execute(&input, limits, &[1], 2, "");
        assert!(
            run.stats[7] > 0,
            "local owner child must replay at source demand"
        );
        let mut receipt = h.compare(&net, &expected, &run);
        receipt["source_equivalence"] = json!(if entry == "run" {
            "exact including cleanup"
        } else {
            "primary storage priority intentionally hits private allocation budget; completing source remains13"
        });
        rows.push(row_for_run(
            "owner_replay_primary_storage_priority",
            compiled::Target::Cpu,
            &source,
            &expected,
            receipt,
        ));
        comparisons += 1;
    }
    for path in [
        "children/dependent_serial.inv",
        "children/affine_serial.inv",
    ] {
        let bytes = fs::read(base::root().join(path)).unwrap();
        let net = lower(&bytes);
        let checked = checked(&bytes).unwrap();
        let h =
            Harness::new(emit(&net, "run", compiled::Target::Cpu, private(), storage()).unwrap());
        assert!(!h.artifact.fork_plan().unwrap().serialized.is_empty());
        let input = [eval::Value::I32(4)];
        let source = eval::named::observe_named(&checked, "run", &input, bound.fuel);
        let expected =
            oracle::observe_call(&net, &net.verify().unwrap(), "run", &input, bound).unwrap();
        base::equal(&source, &expected.semantic, "explicit serialization");
        let run = h.execute(&input, bound, &[1], 2, "");
        assert_eq!(run.stats[0], 0);
        rows.push(row_for_run(
            path,
            compiled::Target::Cpu,
            &source,
            &expected,
            h.compare(&net, &expected, &run),
        ));
        comparisons += 1;
    }
    eprintln!(
        "child failure/source-order matrix: {comparisons} actual comparisons; {fuel_prefixes} exhaustive fuel observations; measured CPU/model width {maximum_width}"
    );
    report(
        "children-failures",
        rows,
        json!({"actual_cpu_model_comparisons":comparisons,"ordered_fuel_prefix_observations":fuel_prefixes,"maximum_observed_private_width":maximum_width,"full_source_capability_gate":"pending","device_launches":0}),
    );
}

pub fn workset_and_fence_admission() {
    use compiled::{
        BufferRange, DeviceConfig, ExecutionState as State, FenceError, FenceObservation as Fence,
        FencedExecution, Submission,
    };
    use std::{
        cell::Cell,
        io::Read,
        process::{Child, Stdio},
        rc::Rc,
        time::{Duration, Instant},
    };
    struct Lease {
        buffers: Vec<Box<[u64]>>,
        dropped: Rc<Cell<usize>>,
        child: Option<Child>,
        words: Vec<i64>,
    }
    impl Drop for Lease {
        fn drop(&mut self) {
            self.dropped.set(self.dropped.get() + 1);
        }
    }
    let bytes = fs::read(base::root().join("children/runtime_branch.inv")).unwrap();
    let net = lower(&bytes);
    let p = checked(&bytes).unwrap();
    let artifact = emit(&net, "run", compiled::Target::Cuda, private(), storage()).unwrap();
    let layout = artifact.device_layout().unwrap().clone();
    let config = DeviceConfig {
        fuel: 1000,
        transitions: 100000,
        trace_events: 10000,
        quantum: 1,
        rounds: 64,
        capacity: 2,
    };
    let sizes = layout.buffer_bytes(config).unwrap();
    let allocate = |dropped: Rc<Cell<usize>>| Lease {
        buffers: sizes
            .iter()
            .map(|n| vec![0; (*n).div_ceil(8)].into_boxed_slice())
            .collect(),
        dropped,
        child: None,
        words: Vec::new(),
    };
    // These IDs are issued by the bounded CPU test allocation registry. They
    // are explicitly not CUDA/Metal allocation identities or a GPU permit.
    let ranges = std::array::from_fn(|i| BufferRange {
        device: 1,
        allocation: i as u64 + 1,
        offset: 0,
        bytes: sizes[i],
    });
    let admitted = layout.workset(config, ranges).unwrap();
    assert_eq!(admitted.buffer_bytes(), &sizes);
    assert_eq!(admitted.config().words(), config.words());
    assert_eq!(admitted.target(), compiled::Target::Cuda);
    let mut mutations = 0;
    for i in 0..12 {
        for mode in 0..6 {
            let mut bad = ranges;
            match mode {
                0 => bad[i].bytes -= 1,
                1 => bad[i].allocation = 0,
                2 => bad[i].device = 0,
                3 => bad[i].device = 2,
                4 => bad[i].offset = 1,
                _ => bad[i].offset = usize::MAX - 7,
            }
            assert!(layout.workset(config, bad).is_err(), "buffer{i} mode{mode}");
            mutations += 1;
        }
    }
    for i in 0..12 {
        for j in 0..i {
            let mut bad = ranges;
            bad[i].allocation = bad[j].allocation;
            assert!(layout.workset(config, bad).is_err());
            mutations += 1;
        }
    }
    let mut adjacent = ranges;
    adjacent[1].allocation = adjacent[0].allocation;
    adjacent[1].offset = sizes[0];
    layout.workset(config, adjacent).unwrap();
    adjacent[1].offset -= 8;
    assert!(layout.workset(config, adjacent).is_err());
    mutations += 1;
    for change in 0..9 {
        let mut bad = config;
        match change {
            0 => bad.fuel = 0,
            1 => bad.fuel = 1000001,
            2 => bad.transitions = 1000001,
            3 => bad.trace_events = usize::MAX,
            4 => bad.quantum = 0,
            5 => bad.quantum = 1000001,
            6 => bad.rounds = 0,
            7 => bad.rounds = 65,
            _ => bad.capacity = 3,
        }
        assert!(layout.buffer_bytes(bad).is_err());
        mutations += 1;
    }
    let h = Harness::new(artifact);
    let inputs = [
        eval::Value::U32(1),
        eval::Value::U32(0),
        eval::Value::I32(3),
    ];
    let bound = oracle::Limits {
        fuel: config.fuel,
        transitions: config.transitions,
        trace_events: config.trace_events,
        array_elements: 128,
    };
    let source = eval::named::observe_named(&p, "run", &inputs, bound.fuel);
    let expected =
        oracle::observe_call(&net, &net.verify().unwrap(), "run", &inputs, bound).unwrap();
    base::equal(&source, &expected.semantic, "CPU process fence source");
    let mut process_receipts = Vec::new();
    for cancel in [false, true] {
        let dropped = Rc::new(Cell::new(0));
        let resources = allocate(dropped.clone());
        assert!(
            resources
                .buffers
                .iter()
                .zip(sizes)
                .all(|(b, n)| b.len() * 8 >= n)
        );
        let mut run = FencedExecution::new(admitted.clone(), resources, 1).unwrap();
        assert_eq!(
            run.observe(|_| panic!("no initial observation")),
            Err::<(), _>(FenceError::Pending)
        );
        run.submit(|r| {
            r.child = Some(
                h.command(&inputs, bound, &[1], 2, "fence_pause")
                    .stdout(Stdio::piped())
                    .spawn()
                    .unwrap(),
            );
            Submission::Submitted
        })
        .unwrap();
        assert_eq!(run.state(), State::InFlight);
        assert_eq!(run.submitted(), 1);
        assert_eq!(dropped.get(), 0);
        assert_eq!(
            run.submit(|_| panic!("no unfenced replay/relaunch")),
            Err(FenceError::Busy)
        );
        assert_eq!(
            run.observe(|_| panic!("no inflight result read")),
            Err::<(), _>(FenceError::Pending)
        );
        if cancel {
            run.cancel();
            assert_eq!(run.state(), State::CancelPending);
        }
        // Let the child's two-second TERM/KILL grace finish before the observer expires.
        let deadline =
            Instant::now() + child_deadline::wall(Duration::from_secs(10)) + Duration::from_secs(3);
        let mut pending = 0;
        loop {
            let result = run.poll(|r| match r.child.as_mut().unwrap().try_wait().unwrap() {
                None => Fence::Pending,
                Some(status) => {
                    assert!(status.success());
                    let mut out = String::new();
                    r.child
                        .as_mut()
                        .unwrap()
                        .stdout
                        .as_mut()
                        .unwrap()
                        .read_to_string(&mut out)
                        .unwrap();
                    r.words = out.split_whitespace().map(|n| n.parse().unwrap()).collect();
                    Fence::Complete
                }
            });
            match result {
                Err(FenceError::Pending) => {
                    pending += 1;
                    assert!(Instant::now() < deadline);
                    std::thread::sleep(Duration::from_millis(2));
                }
                Ok(()) => {
                    assert!(!cancel);
                    break;
                }
                Err(FenceError::Cancelled) => {
                    assert!(cancel);
                    break;
                }
                other => panic!("unexpected CPU process fence {other:?}"),
            }
        }
        assert!(pending > 0);
        assert_eq!(run.completed(), 1);
        assert_eq!(dropped.get(), 0);
        if cancel {
            assert_eq!(run.state(), State::Cancelled);
            assert_eq!(
                run.observe(|_| panic!("cancel cannot publish")),
                Err::<(), _>(FenceError::Pending)
            );
        } else {
            let receipt = run
                .observe(|r| {
                    let mut words = r.words.clone();
                    let tail = words.split_off(words.len() - 15);
                    let observed = Observation {
                        run: owned::Run {
                            words,
                            submissions: tail[12] as u64,
                            yields: 0,
                            replay_checked: tail[14] == 1,
                        },
                        stats: tail[..11].to_vec(),
                        state: tail[11],
                        width: tail[13],
                    };
                    h.compare(&net, &expected, &observed)
                })
                .unwrap();
            process_receipts.push(receipt);
            assert_eq!(
                run.submit(|_| panic!("submission budget exhausted")),
                Err(FenceError::Budget)
            );
        }
        let resources = run.reclaim().unwrap();
        assert_eq!(dropped.get(), 0);
        drop(resources);
        assert_eq!(dropped.get(), 1);
    }
    let mut injected = 0;
    for observation in [Fence::Pending, Fence::Unknown] {
        let dropped = Rc::new(Cell::new(0));
        let mut run = FencedExecution::new(admitted.clone(), allocate(dropped.clone()), 2).unwrap();
        run.submit(|_| Submission::Submitted).unwrap();
        assert!(run.poll(|_| observation).is_err());
        assert!(
            run.observe(|_| panic!("unquiescent cannot observe"))
                .is_err()
        );
        assert!(
            run.submit(|_| panic!("unquiescent cannot relaunch"))
                .is_err()
        );
        drop(run);
        assert_eq!(
            dropped.get(),
            0,
            "entire model lease bundle retained on unknown quiescence"
        );
        injected += 1;
    }
    let dropped = Rc::new(Cell::new(0));
    let mut run = FencedExecution::new(admitted.clone(), allocate(dropped.clone()), 1).unwrap();
    assert_eq!(
        run.submit(|_| Submission::Unknown),
        Err(FenceError::Unknown)
    );
    assert_eq!(run.state(), State::Quarantined);
    assert!(run.reclaim().is_err());
    assert_eq!(dropped.get(), 0);
    injected += 1;
    for submitted in [false, true] {
        let dropped = Rc::new(Cell::new(0));
        let mut run = FencedExecution::new(admitted.clone(), allocate(dropped.clone()), 1).unwrap();
        if submitted {
            run.submit(|_| Submission::Submitted).unwrap();
            assert_eq!(
                run.poll(|_| Fence::FailedQuiescent),
                Err(FenceError::Failed)
            );
            assert_eq!(run.state(), State::Failed);
            assert!(run.observe(|_| panic!("failure cannot publish")).is_err());
        } else {
            assert_eq!(
                run.submit(|_| Submission::NotSubmitted),
                Err(FenceError::Submission)
            );
            assert!(
                run.observe(|_| panic!("not submitted cannot publish"))
                    .is_err()
            );
        }
        drop(run.reclaim().unwrap());
        assert_eq!(dropped.get(), 1);
        injected += 1;
    }
    for budget in [0, 1025] {
        let dropped = Rc::new(Cell::new(0));
        assert!(matches!(
            FencedExecution::new(admitted.clone(), allocate(dropped.clone()), budget),
            Err(FenceError::Budget)
        ));
        assert_eq!(dropped.get(), 1);
        injected += 1;
    }
    if let Some(dir) = std::env::var_os("INVAR_GNET_CORPUS_REPORT_DIR") {
        let dir = PathBuf::from(dir);
        fs::create_dir_all(&dir).unwrap();
        let receipt = json!({"schema_version":1,"profile":if cfg!(debug_assertions){"debug"}else{"release"},"artifact":artifact_record(&h.artifact),"source":base::recorded(&source),"oracle":observed(&expected),"actual_cpu_process_completion_cases":2,"successful_completed_cpu_observations":process_receipts,"complete_live_model_buffers":12,"buffer_bytes":sizes,"admission_mutation_rejections":mutations,"injected_fence_failure_cases":injected,"model_allocation_namespace":"bounded CPU registry IDs; not CUDA/Metal allocations","resource_retention":"all model buffer boxes, CPU child handle and completion record retained until observed OS process exit; injected unknown quiescence deliberately retains entire bundles","fence_authority":"actual CPU process completion for success/cancellation; synthetic fault observations only for unknown/failure state transitions","gpu_execution":"not_performed","driver_fence":"not_performed"});
        fs::write(
            dir.join("children-fences.json"),
            serde_json::to_vec_pretty(&receipt).unwrap(),
        )
        .unwrap();
    }
    eprintln!(
        "12-buffer workset: {mutations} actual verifier rejections;2 CPU process completion/cancellation cases;{injected} explicit fault-model lease/fence cases"
    );
}

pub fn source_reference_vectors() {
    let vectors = base::document("child-vectors.json");
    let mut observations = Vec::new();
    for case in vectors["cases"].as_array().unwrap() {
        let bytes = fs::read(base::root().join(case["source"].as_str().unwrap())).unwrap();
        assert_eq!(base::digest(&bytes), case["source_sha256"]);
        let p = checked(&bytes).unwrap();
        let input = args(&case["arguments"]);
        let entry = case["entry"].as_str().unwrap();
        let bound = limits(case);
        let source = eval::named::observe_named(&p, entry, &input, bound.fuel);
        base::expectation(
            &source,
            &case["expected_source"],
            case["id"].as_str().unwrap(),
        );
        let net = Net::lower(&p, general_net::Limits::default()).unwrap();
        let oracle =
            oracle::observe_call(&net, &net.verify().unwrap(), entry, &input, bound).unwrap();
        base::equal(
            &source,
            &oracle.semantic,
            "independent new source reference vectors",
        );
        observations.push(
            json!({"id":case["id"],"source":base::recorded(&source),"oracle":observed(&oracle)}),
        );
    }
    assert_eq!(observations.len(), 39);
    if let Some(dir) = std::env::var_os("INVAR_GNET_CORPUS_REPORT_DIR") {
        let dir = PathBuf::from(dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("children-reference.json"),serde_json::to_vec_pretty(&json!({"profile":if cfg!(debug_assertions){"debug"}else{"release"},"case_count":observations.len(),"rows":observations})).unwrap()).unwrap();
    }
    eprintln!(
        "39 hand-derived source/reference vectors match exactly, including operator-byte traps"
    );
}

/// ABI13 keeps the workgroup protocol while extending the entry packet.
/// Check both empty and scalar-parameter packets against the independent oracle.
pub fn extended_record_entry_cpu_models() {
    let net = lower(b"struct Pair { left:i32, right:i32 } fn fields(x:i32)->i32 { let p:Pair=Pair { left:x, right:4 }; return p.left*2+p.right; } fn main()->i32 { return fields(9); }");
    let limits = oracle::Limits {
        fuel: 10000,
        transitions: 10000,
        trace_events: 2048,
        array_elements: 128,
    };
    for (entry, inputs) in [("main", vec![]), ("fields", vec![eval::Value::I32(9)])] {
        let expected =
            oracle::observe_call(&net, &net.verify().unwrap(), entry, &inputs, limits).unwrap();
        for target in [compiled::Target::Cuda, compiled::Target::Metal] {
            let artifact = emit(&net, entry, target, private(), storage()).unwrap();
            assert_eq!(artifact.abi_version(), 13);
            assert!(artifact.device_layout().is_some());
            let harness = Harness::new(artifact);
            for capacity in [2, 0] {
                let run = harness.execute(&inputs, limits, &[17], capacity, "");
                harness.compare(&net, &expected, &run);
            }
        }
    }
}
