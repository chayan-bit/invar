//! Actual ABI2/3 CPU execution. Caller-owned heaps/tasks retain their full
//! denominator; source and backend budgets are accounted separately.
use super::{child_deadline, corpus_sample, gnet_words};

use super::corpus_cpu::{
    self as base, cleanup, compiled_rejection, digest, document, equal, expectation, recorded,
    root, semantic,
};
use invar::{
    check, eval, front,
    general_net::{self, Net, compiled, oracle},
};
use serde_json::{Value as Json, json};
use std::{collections::BTreeMap, fs, path::PathBuf};

fn checked(bytes: &[u8]) -> Result<check::CheckedProgram, front::FrontError> {
    let checked = check::check(front::parse(bytes)?, true)?;
    invar::ownership::check(&checked)?;
    Ok(checked)
}
fn lower(bytes: &[u8]) -> Net {
    Net::lower(&checked(bytes).unwrap(), general_net::Limits::default()).unwrap()
}
fn owned_limits(doc: &Json) -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: doc["elements"].as_u64().unwrap() as usize,
        allocations: doc["allocations"].as_u64().unwrap() as usize,
        cleanup_events: doc["cleanup_events"].as_u64().unwrap() as usize,
    }
}
fn args(doc: &Json) -> Vec<eval::Value> {
    doc.as_array()
        .unwrap()
        .iter()
        .map(|arg| match arg["type"].as_str().unwrap() {
            "i32" => eval::Value::I32(i32::try_from(arg["value"].as_i64().unwrap()).unwrap()),
            "u32" => eval::Value::U32(u32::try_from(arg["value"].as_u64().unwrap()).unwrap()),
            _ => panic!("scalar inputs"),
        })
        .collect()
}
fn limits(case: &Json) -> oracle::Limits {
    oracle::Limits {
        fuel: case["fuel"].as_u64().unwrap(),
        transitions: case["limits"]["transitions"].as_u64().unwrap(),
        trace_events: case["limits"]["trace_events"].as_u64().unwrap() as usize,
        array_elements: case["limits"]["array_elements"].as_u64().unwrap() as usize,
    }
}
fn report(name: &str, rows: Vec<Json>, extra: Json) {
    let Some(dir) = std::env::var_os("INVAR_GNET_CORPUS_REPORT_DIR") else {
        return;
    };
    fs::create_dir_all(&dir).unwrap();
    let counts = |key: &str| {
        let mut c = BTreeMap::<String, usize>::new();
        for r in &rows {
            *c.entry(r[key]["status"].as_str().unwrap().to_owned())
                .or_default() += 1;
        }
        c
    };
    let output = json!({"schema_version":1,"profile":if cfg!(debug_assertions){"debug"}else{"release"},"case_count":rows.len(),"counts":{"source":counts("source"),"oracle":counts("oracle"),"owned_cpu":counts("owned_cpu"),"resumable_cpu":counts("resumable_cpu")},"gpu_execution":"not_performed","extra":extra,"rows":rows});
    fs::write(
        PathBuf::from(dir).join(format!("{name}.json")),
        serde_json::to_vec_pretty(&output).unwrap(),
    )
    .unwrap();
}
fn emit(
    net: &Net,
    entry: &str,
    resumable: bool,
    storage: compiled::OwnedLimits,
) -> Result<compiled::Artifact, compiled::Error> {
    let emitter = if resumable {
        compiled::emit_resumable
    } else {
        compiled::emit_owned
    };
    emitter(
        net,
        &net.verify().unwrap(),
        entry,
        compiled::Target::Cpu,
        compiled::Limits::default(),
        storage,
    )
}

pub(super) struct Cpu {
    pub(super) dir: PathBuf,
    layout: compiled::OwnedLayout,
    abi: u32,
    artifact: Json,
}
pub(super) struct Run {
    pub(super) words: Vec<i64>,
    pub(super) submissions: u64,
    pub(super) yields: u64,
    pub(super) replay_checked: bool,
}
impl Cpu {
    fn compile(artifact: &compiled::Artifact, driver: &str) -> Self {
        let dir = child_deadline::scratch("invar-corpus-owned");
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
        Self::for_harness(artifact, dir)
    }
    pub(super) fn for_harness(artifact: &compiled::Artifact, dir: PathBuf) -> Self {
        Self {
            dir,
            layout: artifact.owned_layout().unwrap().clone(),
            abi: artifact.abi_version(),
            artifact: json!({"abi":artifact.abi_version(),"source_sha256":digest(artifact.source().as_bytes()),"states":artifact.states(),"frame_bytes":artifact.frame_bytes(),"task_bytes":artifact.continuation_layout().map(|c|c.task_bytes()),"storage_bytes":artifact.owned_layout().unwrap().storage_bytes()}),
        }
    }
    fn new(artifact: &compiled::Artifact) -> Self {
        let layout = artifact.owned_layout().unwrap();
        let task_bytes = artifact.continuation_layout().map(|c| c.task_bytes());
        let body = if let Some(bytes) = task_bytes {
            format!(
                r#"
        static_assert(sizeof(GnetTask)=={bytes}); GnetTask task={{}};
        std::vector<gnet_u> committed_trace,committed_cleanup;
        while(true) {{
            auto before=task.transitions, previous_steps=task.steps;
            auto q=quanta[submissions%quanta.size()]; ++submissions;
            bool done=gnet_resume(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data(),&task,q);
            if(task.transitions<before || task.transitions-before>q || task.steps<previous_steps) return 5;
            if(task.used>cap || task.cleanup_used>{clean_cap} || task.allocated>{alloc_cap}) return 6;
            for(size_t i=0;i<committed_trace.size();++i) if(trace[i]!=committed_trace[i]) return 7;
            for(size_t i=0;i<committed_cleanup.size();++i) if(cleanup[i]!=committed_cleanup[i]) return 8;
            committed_trace.assign(trace.begin(),trace.begin()+task.used*7);
            committed_cleanup.assign(cleanup.begin(),cleanup.begin()+task.cleanup_used*4);
            if(done) break;
            ++yields;for(auto word:output) if(word!=0) return 9;
            if(submissions>1000002) return 10;
        }}
        auto output_before=std::vector<gnet_w>(output,output+10),heap_before=heap;
        auto trace_before=trace,cleanup_before=cleanup;auto before=task.transitions;
        if(!gnet_resume(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data(),&task,quanta[0]) || task.transitions!=before) return 11;
        if(output_before!=std::vector<gnet_w>(output,output+10) || heap_before!=heap || trace_before!=trace || cleanup_before!=cleanup) return 12;
        replay_checked=1;
        "#,
                clean_cap = layout.cleanup_events(),
                alloc_cap = layout.allocations()
            )
        } else {
            "gnet_run(input.data(),output,trace.data(),fuel,trans,cap,heap.data(),cleanup.data()); submissions=1;".to_owned()
        };
        let driver = format!(
            r#"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
int main(int argc,char** argv) {{
    if(argc<5) return 2;
    gnet_u fuel=std::stoull(argv[1]),trans=std::stoull(argv[2]),cap=std::stoull(argv[3]);
    if(cap>1000000) return 2;
    std::vector<gnet_u> quanta; std::stringstream schedule(argv[4]);std::string piece;
    while(std::getline(schedule,piece,',')) {{ auto q=std::stoull(piece);if(q==0 || q>1000000 || quanta.size()>3) return 2;quanta.push_back(q); }}
    if(quanta.empty()) return 2;
    std::vector<gnet_w> input;for(int i=5;i<argc;++i)input.push_back(std::stoll(argv[i]));
    gnet_w output[10]={{}};std::vector<gnet_u> trace(cap*7),cleanup({clean});std::vector<gnet_w> heap({heap});
    gnet_u submissions=0,yields=0,replay_checked=0;
    {body}
    for(auto word:output)std::cout<<word<<' ';
    for(gnet_u i=0;i<(gnet_u)output[7]*7;++i)std::cout<<trace[i]<<' ';
    for(gnet_u i=0;i<(gnet_u)output[8]*4;++i)std::cout<<cleanup[i]<<' ';
    for(auto word:heap)std::cout<<word<<' ';
    std::cout<<submissions<<' '<<yields<<' '<<replay_checked<<'\n';
}}
"#,
            clean = layout.cleanup_words(),
            heap = layout.heap_words()
        );
        Self::compile(artifact, &driver)
    }
    fn execute(&self, inputs: &[eval::Value], limits: oracle::Limits, quanta: &[u64]) -> Run {
        let mut command = child_deadline::command(10);
        command.arg(self.dir.join("run")).args([
            limits.fuel.to_string(),
            limits.transitions.to_string(),
            limits.trace_events.to_string(),
            quanta
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        ]);
        for value in inputs {
            command.arg(match value {
                eval::Value::I32(n) => n.to_string(),
                eval::Value::U32(n) => n.to_string(),
                _ => panic!("scalar"),
            });
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "CPU run {:?}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        let mut words: Vec<i64> = String::from_utf8(output.stdout)
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let metadata = words.split_off(words.len() - 3);
        let run = Run {
            words,
            submissions: metadata[0] as u64,
            yields: metadata[1] as u64,
            replay_checked: metadata[2] == 1,
        };
        if self.abi == 3 {
            assert!(run.replay_checked);
            assert_eq!(run.submissions, run.yields + 1);
        }
        run
    }
    pub(super) fn compare(&self, net: &Net, expected: &oracle::Observation, run: &Run) -> Json {
        let w = &run.words;
        let mut expected_words = owned_expected_words(expected);
        let array_result = matches!(
            expected.semantic.result,
            Ok(eval::Value::ArrayI32(_) | eval::Value::ArrayU32(_))
        );
        if array_result {
            expected_words[1] = w[1];
        }
        assert_eq!(&w[..8], &expected_words[..8], "ABI{} header", self.abi);
        assert_eq!(w[8] as usize, expected.semantic.cleanup_trace.len());
        let trace_end = 10 + expected.events.len() * 7;
        assert_eq!(
            &w[10..trace_end],
            &expected_words[8..],
            "all ordered demand/completion events"
        );
        let program = net.compiled_mir(&net.verify().unwrap()).unwrap();
        let mut offset = trace_end;
        for event in &expected.semantic.cleanup_trace {
            let function = usize::try_from(w[offset]).unwrap();
            let place = usize::try_from(w[offset + 1]).unwrap();
            let frame = w[offset + 3] as u64;
            assert_eq!(
                program.functions[function].declaration_order[place],
                event.name
            );
            assert_eq!(w[offset + 2] as usize, event.length);
            assert!(
                expected
                    .events
                    .iter()
                    .any(|e| e.frame == frame && e.token.block.function == function),
                "cleanup frame must belong to the observed invocation"
            );
            offset += 4;
        }
        let heap = &w[offset..];
        assert_eq!(heap.len(), self.layout.heap_words());
        let allocated = usize::try_from(w[9]).unwrap();
        assert!(allocated <= self.layout.allocations());
        let live = heap[..self.layout.allocations() * 4]
            .chunks_exact(4)
            .enumerate()
            .filter_map(|(i, d)| (d[3] == 1).then_some(i + 1))
            .collect::<Vec<_>>();
        let root_value = match &expected.semantic.result {
            Ok(eval::Value::ArrayI32(values)) => {
                let actual = self.array(w[1], heap, 0);
                let expected = values
                    .borrow()
                    .iter()
                    .map(|x| i64::from(*x))
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected);
                assert_eq!(live, vec![w[1] as usize]);
                json!(actual)
            }
            Ok(eval::Value::ArrayU32(values)) => {
                let actual = self.array(w[1], heap, 1);
                let expected = values
                    .borrow()
                    .iter()
                    .map(|x| i64::from(*x))
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected);
                assert_eq!(live, vec![w[1] as usize]);
                json!(actual)
            }
            Ok(_) => {
                assert!(
                    live.is_empty(),
                    "normal scalar result leaks no owned allocation"
                );
                Json::Null
            }
            Err(_) => Json::Null,
        };
        if expected.failure.is_none() {
            let completed_allocations=expected.events.iter().filter(|e|e.kind==oracle::EventKind::Complete && (matches!(e.rule, general_net::coverage::Rule::Array | general_net::coverage::Rule::Record) || net.agents.iter().any(|a|a.token==e.token && matches!(&a.call,Some(general_net::CallTarget::Intrinsic(name)) if name=="filled_i32" || name=="filled_u32")))).count();
            assert_eq!(
                allocated, completed_allocations,
                "stable allocation IDs follow actual completed allocations"
            );
        }
        json!({"status":"executed","abi":self.abi,"artifact":self.artifact,"header":&w[..10],"full_observation_equal":true,"cleanup":cleanup(&expected.semantic),"allocated_ids":allocated,"live_handles":live,"root_array_snapshot":root_value,"submissions":run.submissions,"yields":run.yields,"completed_replay_unchanged":run.replay_checked,"all_words_le_i64_sha256":digest(&w.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())})
    }
    fn array(&self, handle: i64, heap: &[i64], kind: i64) -> Vec<i64> {
        let handle = usize::try_from(handle).unwrap();
        assert!(handle > 0 && handle <= self.layout.allocations());
        let d = (handle - 1) * 4;
        assert_eq!(heap[d + 3], 1);
        assert_eq!(heap[d + 2], kind);
        let start = self.layout.allocations() * 4 + usize::try_from(heap[d]).unwrap();
        let len = usize::try_from(heap[d + 1]).unwrap();
        heap[start..start + len].to_vec()
    }
}
impl Drop for Cpu {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap();
    }
}
fn owned_expected_words(expected: &oracle::Observation) -> Vec<i64> {
    match &expected.semantic.result {
        Ok(eval::Value::ArrayI32(_) | eval::Value::ArrayU32(_)) => {
            let mut words = vec![
                0,
                0,
                0,
                0,
                expected.semantic.steps_used as i64,
                expected.transitions as i64,
                expected.max_frames as i64,
                expected.events.len() as i64,
            ];
            append_events(&mut words, expected);
            words
        }
        Err(_) if expected.failure == Some(oracle::ToolFailure::Allocation) => {
            let mut words = vec![
                compiled::Status::HeapLimit as i64,
                0,
                0,
                0,
                expected.semantic.steps_used as i64,
                expected.transitions as i64,
                expected.max_frames as i64,
                expected.events.len() as i64,
            ];
            append_events(&mut words, expected);
            words
        }
        _ => base::words(expected),
    }
}
fn append_events(words: &mut Vec<i64>, expected: &oracle::Observation) {
    for e in &expected.events {
        words.extend(gnet_words::event_i64(e));
    }
}
fn schedules(doc: &Json) -> Vec<Vec<u64>> {
    let mut schedules = doc["quanta"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| vec![q.as_u64().unwrap()])
        .collect::<Vec<_>>();
    schedules.push(vec![1, 3, 17]);
    schedules
}
fn observe(
    net: &Net,
    entry: &str,
    inputs: &[eval::Value],
    limits: oracle::Limits,
) -> oracle::Observation {
    oracle::observe_call(net, &net.verify().unwrap(), entry, inputs, limits).unwrap()
}
fn observed(run: &oracle::Observation) -> Json {
    if run.failure == Some(oracle::ToolFailure::Unsupported) {
        json!({"status":"unsupported","reason":format!("{:?}",run.semantic.result)})
    } else {
        json!({"status":"executed","semantic":semantic(&run.semantic),"steps_used":run.semantic.steps_used,"cleanup":cleanup(&run.semantic),"failure":run.failure.map(|f|format!("{f:?}")),"transitions":run.transitions,"max_frames":run.max_frames,"events":run.events.len()})
    }
}

#[path = "corpus_owned/denominator.rs"]
mod denominator;

pub fn full_denominator() {
    denominator::run();
}

pub fn owner_prefixes_and_returned_roots() {
    let manifest = document("expanded-manifest.json");
    let config = document("owned-vectors.json");
    let storage = owned_limits(&config["storage"]);
    let schedules = schedules(&config);
    let mut rows = Vec::new();
    let mut prefixes = 0;
    for id in [
        "array_mutation",
        "array_owner_implicit_return",
        "array_owner_explicit_return",
        "array_take_reborrow",
        "array_replacement",
        "array_loop_cleanup",
        "array_empty_u32",
    ] {
        let case = manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == id)
            .unwrap();
        let bytes = fs::read(root().join(case["source"].as_str().unwrap())).unwrap();
        let checked = checked(&bytes).unwrap();
        let complete = eval::observe(&checked, 1000);
        let net = lower(&bytes);
        let owned = Cpu::new(&emit(&net, "main", false, storage).unwrap());
        let resume = Cpu::new(&emit(&net, "main", true, storage).unwrap());
        let mut runs = Vec::new();
        for fuel in 1..=complete.steps_used + 1 {
            let limits = oracle::Limits {
                fuel,
                trace_events: 10000,
                ..oracle::Limits::default()
            };
            let expected = observe(&net, "main", &[], limits);
            equal(&expected.semantic, &eval::observe(&checked, fuel), id);
            let run = owned.execute(&[], limits, &[1]);
            let receipt = owned.compare(&net, &expected, &run);
            let mut resumed = Vec::new();
            for q in &schedules {
                let run = resume.execute(&[], limits, q);
                let mut receipt = resume.compare(&net, &expected, &run);
                receipt["quanta"] = json!(q);
                resumed.push(receipt);
            }
            runs.push(json!({"fuel":fuel,"owned":receipt,"resumed":resumed}));
            prefixes += 1;
        }
        rows.push(json!({"id":id,"source":{"status":"executed"},"oracle":{"status":"executed"},"owned_cpu":{"status":"executed"},"resumable_cpu":{"status":"executed"},"positive_fuel_prefixes":complete.steps_used+1,"runs":runs}));
    }
    for case in config["returned_roots"].as_array().unwrap() {
        let bytes = fs::read(root().join(case["source"].as_str().unwrap())).unwrap();
        let net = lower(&bytes);
        let inputs = args(&case["arguments"]);
        let entry = case["entry"].as_str().unwrap();
        let limits = oracle::Limits {
            fuel: 1000,
            trace_events: 10000,
            ..oracle::Limits::default()
        };
        let expected = observe(&net, entry, &inputs, limits);
        expectation(
            &expected.semantic,
            &case["expected"],
            case["id"].as_str().unwrap(),
        );
        assert_eq!(cleanup(&expected.semantic), case["expected_cleanup"]);
        let mut runs = Vec::new();
        for resumable in [false, true] {
            let cpu = Cpu::new(&emit(&net, entry, resumable, storage).unwrap());
            let qs = if resumable {
                schedules.clone()
            } else {
                vec![vec![1]]
            };
            for q in qs {
                let run = cpu.execute(&inputs, limits, &q);
                assert_eq!(run.words[9], case["expected_allocations"].as_i64().unwrap());
                let mut receipt = cpu.compare(&net, &expected, &run);
                receipt["quanta"] = json!(q);
                runs.push(receipt);
            }
        }
        rows.push(json!({"id":case["id"],"source":{"status":"named_result_unavailable","reason":"named source API admits scalar results only; original source main is separately executed"},"oracle":observed(&expected),"owned_cpu":{"status":"executed"},"resumable_cpu":{"status":"executed"},"expected":case["expected"],"runs":runs}));
    }
    assert_eq!(prefixes, 169);
    eprintln!(
        "169 owner fuel prefixes x ABI2/3 and four quantum schedules; 3 owned root arrays snapshot retained live handles"
    );
    report(
        "owned-prefixes",
        rows,
        json!({"fuel_prefix_denominator":prefixes,"matched_cpu_observations":prefixes*5+15,"returned_root_denominator":3}),
    );
}

pub fn fair_worklist_and_backend_budgets() {
    let config = document("owned-vectors.json");
    let mut rows = Vec::new();
    let spec = &config["worklist"];
    let bytes = fs::read(root().join(spec["source"].as_str().unwrap())).unwrap();
    let net = lower(&bytes);
    let entry = spec["entry"].as_str().unwrap();
    let storage = owned_limits(&config["storage"]);
    let artifact = emit(&net, entry, true, storage).unwrap();
    let layout = artifact.owned_layout().unwrap();
    let task_bytes = artifact.continuation_layout().unwrap().task_bytes();
    assert_eq!(artifact.continuation_layout().unwrap().max_roots(), 1024);
    let driver = format!(
        r#"
#include <iostream>
#include <vector>
#include <array>
int main() {{
    static_assert(sizeof(GnetTask)=={task_bytes});
    gnet_w input[4]={{0,1,2,3}},output[40]={{}};std::vector<GnetTask> tasks(4);
    std::vector<gnet_u> trace(4*10000*7),cleanup(4*{clean});std::vector<gnet_w> heap(4*{heap});
    gnet_u submitted=0,completed=0;std::vector<gnet_u> stages;
    auto result=gnet_worklist(input,output,trace.data(),4,1000,10000,10000,heap.data(),cleanup.data(),tasks.data(),1,8,submitted,completed);
    if(result!=14 || submitted!=8 || completed!=0)return 20;stages.insert(stages.end(),{{(gnet_u)result,submitted,completed}});
    for(auto& task:tasks)if(task.transitions!=2 || task.done)return 21;
    for(auto word:output)if(word!=0)return 22;
    result=gnet_worklist(input,output,trace.data(),4,1000,10000,10000,heap.data(),cleanup.data(),tasks.data(),1,512,submitted,completed);
    if(result!=14 || submitted!=512 || completed!=3 || tasks[0].done)return 23;
    for(unsigned lane=1;lane<4;++lane)if(!tasks[lane].done)return 24;
    for(unsigned i=0;i<10;++i)if(output[i]!=0)return 25;
    if(output[10]!=0 || output[11]!=17 || output[20]!=9 || output[30]!=0 || output[31]!=23)return 26;
    stages.insert(stages.end(),{{(gnet_u)result,submitted,completed,tasks[0].transitions}});
    auto output_before=std::vector<gnet_w>(output,output+40),heap_before=heap;
    auto trace_before=trace,cleanup_before=cleanup;
    result=gnet_worklist(input,output,trace.data(),4,1000,10000,10000,heap.data(),cleanup.data(),tasks.data(),17,10000,submitted,completed);
    if(result!=0 || completed!=4)return 27;stages.insert(stages.end(),{{(gnet_u)result,submitted,completed}});
    for(unsigned lane=1;lane<4;++lane) {{
        for(unsigned i=0;i<10;++i)if(output[lane*10+i]!=output_before[lane*10+i])return 28;
        for(unsigned i=0;i<{heap};++i)if(heap[lane*{heap}+i]!=heap_before[lane*{heap}+i])return 29;
        for(unsigned i=0;i<{clean};++i)if(cleanup[lane*{clean}+i]!=cleanup_before[lane*{clean}+i])return 30;
        for(unsigned i=0;i<10000*7;++i)if(trace[lane*10000*7+i]!=trace_before[lane*10000*7+i])return 31;
    }}
    auto final_output=std::vector<gnet_w>(output,output+40),final_heap=heap;auto final_trace=trace,final_cleanup=cleanup;
    std::array<gnet_u,4> before;for(unsigned i=0;i<4;++i)before[i]=tasks[i].transitions;
    result=gnet_worklist(input,output,trace.data(),4,1000,10000,10000,heap.data(),cleanup.data(),tasks.data(),3,0,submitted,completed);
    if(result!=0 || submitted!=0 || completed!=4)return 32;
    if(final_output!=std::vector<gnet_w>(output,output+40) || final_heap!=heap || final_trace!=trace || final_cleanup!=cleanup)return 33;
    for(unsigned i=0;i<4;++i)if(tasks[i].transitions!=before[i])return 34;
    stages.insert(stages.end(),{{(gnet_u)result,submitted,completed}});
    for(auto word:stages)std::cout<<word<<' ';
    for(unsigned lane=0;lane<4;++lane) {{
        for(unsigned i=0;i<10;++i)std::cout<<output[lane*10+i]<<' ';
        for(gnet_u i=0;i<(gnet_u)output[lane*10+7]*7;++i)std::cout<<trace[lane*10000*7+i]<<' ';
        for(gnet_u i=0;i<(gnet_u)output[lane*10+8]*4;++i)std::cout<<cleanup[lane*{clean}+i]<<' ';
        for(unsigned i=0;i<{heap};++i)std::cout<<heap[lane*{heap}+i]<<' ';
    }}
    std::cout<<'\n';
}}
"#,
        heap = layout.heap_words(),
        clean = layout.cleanup_words()
    );
    let cpu = Cpu::compile(&artifact, &driver);
    let output = child_deadline::command(10)
        .arg(cpu.dir.join("run"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "fair worklist {:?}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let w = String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse::<i64>().unwrap())
        .collect::<Vec<_>>();
    let stages = &w[..13];
    assert_eq!(&stages[..6], &[14, 8, 0, 14, 512, 3]);
    assert_eq!(&stages[10..13], &[0, 0, 4]);
    let mut offset = 13;
    let runtime = document("runtime-vectors.json");
    let mut roots = Vec::new();
    let limits = oracle::Limits {
        fuel: 1000,
        transitions: 10000,
        trace_events: 10000,
        ..oracle::Limits::default()
    };
    for (lane, id) in [
        "fair_divergent",
        "fair_value",
        "fair_trap_prefix",
        "fair_owned_cleanup",
    ]
    .into_iter()
    .enumerate()
    {
        let case = runtime["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == id)
            .unwrap();
        let inputs = vec![eval::Value::U32(lane as u32)];
        let expected = observe(&net, entry, &inputs, limits);
        expectation(&expected.semantic, &case["expected_source"], id);
        assert_eq!(cleanup(&expected.semantic), case["expected_cleanup"]);
        let size = 10
            + expected.events.len() * 7
            + expected.semantic.cleanup_trace.len() * 4
            + layout.heap_words();
        let run = Run {
            words: w[offset..offset + size].to_vec(),
            submissions: 0,
            yields: 0,
            replay_checked: true,
        };
        offset += size;
        let mut receipt = cpu.compare(&net, &expected, &run);
        receipt["submissions"] = Json::Null;
        receipt["yields"] = Json::Null;
        roots.push(json!({"id":id,"observation":receipt}));
    }
    assert_eq!(offset, w.len());
    rows.push(json!({"id":"four_independent_roots","source":{"status":"executed"},"oracle":{"status":"executed"},"owned_cpu":{"status":"not_attempted"},"resumable_cpu":{"status":"executed"},"root_denominator":4,"stages":stages,"fair_completed_while_oldest_root_yielded":3,"roots":roots,"quantum_changes_preserve_budgets":true,"completed_roots_replay_unchanged":true}));
    for case in config["backend_budget_cases"].as_array().unwrap() {
        let bytes = fs::read(root().join(case["source"].as_str().unwrap())).unwrap();
        let net = lower(&bytes);
        let checked = checked(&bytes).unwrap();
        let entry = case["entry"].as_str().unwrap();
        let inputs = args(&case["arguments"]);
        let limits = oracle::Limits {
            fuel: 10000,
            trace_events: 10000,
            ..oracle::Limits::default()
        };
        let source = eval::named::observe_named(&checked, entry, &inputs, limits.fuel);
        assert!(source.result.is_ok());
        let expected = observe(&net, entry, &inputs, limits);
        equal(&expected.semantic, &source, case["id"].as_str().unwrap());
        let storage = owned_limits(&case["storage"]);
        let mut runs = Vec::new();
        for resumable in [false, true] {
            let cpu = Cpu::new(&emit(&net, entry, resumable, storage).unwrap());
            let run = cpu.execute(&inputs, limits, &[1, 3, 17]);
            assert_eq!(run.words[0], case["expected_status"].as_i64().unwrap());
            assert_eq!(run.words[1], 0);
            let (cleanup_count, allocations, live_count) = match case["id"].as_str().unwrap() {
                "heap_limit" => (0, 0, 0),
                "allocation_limit" => (2, 2, 0),
                "cleanup_limit" => (0, 1, 1),
                _ => panic!("authored budget case"),
            };
            assert_eq!(run.words[8], cleanup_count);
            assert_eq!(run.words[9], allocations);
            let heap_offset = 10 + run.words[7] as usize * 7 + run.words[8] as usize * 4;
            let heap = &run.words[heap_offset..];
            assert_eq!(heap.len(), cpu.layout.heap_words());
            assert_eq!(
                heap[..cpu.layout.allocations() * 4]
                    .chunks_exact(4)
                    .filter(|d| d[3] == 1)
                    .count(),
                live_count
            );
            runs.push(json!({"abi":cpu.abi,"status":run.words[0],"header":&run.words[..10],"submissions":run.submissions,"yields":run.yields,"completed_replay_unchanged":run.replay_checked,"distinct_from_completing_source":true,"all_words_le_i64_sha256":digest(&run.words.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())}));
        }
        rows.push(json!({"id":case["id"],"source":recorded(&source),"oracle":observed(&expected),"owned_cpu":{"status":"backend_budget_failure"},"resumable_cpu":{"status":"backend_budget_failure"},"storage":case["storage"],"runs":runs}));
    }
    // Three live elements suffice despite a dead allocation and four temporary
    // borrowers: descriptors retain identity through every compacting allocate.
    let bytes = fs::read(root().join("expanded/owner_compaction_mut_loan.inv")).unwrap();
    let net = lower(&bytes);
    let inputs = vec![eval::Value::U32(4)];
    let limits = oracle::Limits {
        fuel: 1000,
        trace_events: 10000,
        ..oracle::Limits::default()
    };
    let expected = observe(&net, "compact", &inputs, limits);
    expectation(
        &expected.semantic,
        &json!({"status":"ok","result":{"type":"i32","value":13}}),
        "tight_compaction",
    );
    let mut runs = Vec::new();
    for resumable in [false, true] {
        let cpu = Cpu::new(
            &emit(
                &net,
                "compact",
                resumable,
                compiled::OwnedLimits {
                    elements: 3,
                    allocations: 16,
                    cleanup_events: 16,
                },
            )
            .unwrap(),
        );
        let run = cpu.execute(&inputs, limits, &[1, 3, 17]);
        assert_eq!(run.words[9], 6);
        runs.push(cpu.compare(&net, &expected, &run));
    }
    rows.push(json!({"id":"tight_compaction_mutable_loan","source":{"status":"executed"},"oracle":observed(&expected),"owned_cpu":{"status":"executed"},"resumable_cpu":{"status":"executed"},"storage":{"elements":3,"allocations":16,"cleanup_events":16},"runs":runs}));
    eprintln!(
        "4 fair roots resume after two exact worklist budgets; 3 distinct backend storage failures; tight 3-element mutable-loan compaction"
    );
    report(
        "worklist-and-budgets",
        rows,
        json!({"fair_root_denominator":4,"backend_budget_case_denominator":3,"matched_cpu_observations":6,"backend_budget_executions":6}),
    );
}
