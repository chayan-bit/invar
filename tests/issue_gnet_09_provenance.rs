#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "../test-support/child_deadline.rs"]
mod child_deadline;

use invar::{
    eval::Value,
    general_net::{
        compiled::Target,
        oracle,
        provenance::{self, Error, Limits, NumericContinuationComponents, ProvenancedArtifact},
    },
    numeric_contract::F32Bits,
};
use std::{
    fs,
    sync::atomic::{AtomicUsize, Ordering},
};

const SOURCE: &[u8] = b"fn run(x:f32,n:u32)->f32{if n==0u32{return x;}return run(x,n-1u32);}";
fn artifact() -> ProvenancedArtifact {
    provenance::compile_resumable_numeric(SOURCE, "run", Target::Cpu, Limits::default()).unwrap()
}
fn limits() -> oracle::Limits {
    oracle::Limits {
        fuel: 1000,
        transitions: 1000,
        trace_events: 1000,
        array_elements: 16,
    }
}
fn args(bits: u32) -> [Value; 2] {
    [Value::F32(F32Bits::from_bits(bits)), Value::U32(2)]
}

#[test]
fn numeric_continuation_compile_receipt_binds_profile_layout_and_exact_source() {
    let a = artifact();
    let b = artifact();
    let scalar =
        provenance::compile_numeric(SOURCE, "run", Target::Cpu, Limits::default()).unwrap();
    assert_eq!(a.id(), b.id());
    assert_ne!(a.id(), scalar.id());
    assert_eq!(a.artifact().abi_version(), 9);
    assert_eq!(a.manifest()["abi_version"], 9);
    assert_eq!(a.manifest()["numeric"]["semantic_id"], "strict-f32-v1");
    let bytes = a.artifact().continuation_layout().unwrap().task_bytes();
    assert_eq!(bytes, a.artifact().frame_bytes() + 18 * 8);
    assert_eq!(a.manifest()["layouts"]["continuation"]["task_bytes"], bytes);
    assert!(a.artifact().owned_layout().is_none());
    assert!(a.artifact().device_layout().is_none());
    assert!(a.artifact().speculation_layout().is_none());
    for field in [
        "abi_version",
        "numeric",
        "layouts",
        "source",
        "artifact",
        "params",
        "limits",
    ] {
        let mut forged = a.manifest().clone();
        forged[field] = serde_json::json!("forged");
        assert_eq!(a.verify_manifest(&forged), Err(Error::ManifestMismatch));
    }
    let changed = [SOURCE, b" "].concat();
    let other =
        provenance::compile_resumable_numeric(&changed, "run", Target::Cpu, Limits::default())
            .unwrap();
    assert_ne!(a.id(), other.id());
    let task = vec![0; bytes];
    let id = a.continuation_identity(&task).unwrap();
    a.verify_continuation(&id, &task).unwrap();
    assert_eq!(
        other.verify_continuation(&id, &task),
        Err(Error::ContinuationMismatch)
    );
    for bits in [0x80000000, 0x7fa12345] {
        let o = a.observe_oracle(&args(bits), limits()).unwrap();
        assert_eq!(o.manifest()["result_bits"], bits);
        assert_eq!(o.manifest()["continuation_execution"], false);
        assert_eq!(o.manifest()["input_words"][0], bits);
    }
    for target in [Target::Cuda, Target::Metal] {
        assert!(matches!(
            provenance::compile_resumable_numeric(SOURCE, "run", target, Limits::default()),
            Err(Error::Compiler(
                invar::general_net::compiled::Error::NumericTarget
            ))
        ));
    }
}

const DRIVER: &str = r#"
#include <iostream>
#include <cstdlib>
void record(bool done,const GnetTask& task,const gnet_w* output,const gnet_u* trace) {
 std::cout<<done<<' '<<task.used<<' ';
 for(unsigned i=0;i<8;++i) std::cout<<output[i]<<' ';
 for(gnet_u i=0;i<task.used*7;++i) std::cout<<trace[i]<<' ';
 auto bytes=reinterpret_cast<const unsigned char*>(&task);
 for(unsigned i=0;i<sizeof(task);++i) std::cout<<unsigned(bytes[i])<<' ';
 std::cout<<'\n';
}
int main(int argc,char** argv) {
 if(argc!=2) return 2;
 GnetTask task={}; gnet_w input[2]={gnet_w(std::strtoull(argv[1],nullptr,10)),2};
 gnet_w output[8]={111,112,113,114,115,116,117,118}; gnet_u trace[7000]={};
 bool done=gnet_resume(input,output,trace,1000,1000,1000,nullptr,nullptr,&task,1);
 record(done,task,output,trace);
 input[0]=17;input[1]=0;
 unsigned rounds=0;
 while(!done && rounds++<1000) done=gnet_resume(input,output,trace,1000,1000,1000,nullptr,nullptr,&task,3);
 if(!done) return 3;
 record(done,task,output,trace);
 return 0;
}
"#;

#[test]
fn numeric_continuation_supplied_identity_binds_actual_cpu_yield_and_terminal_components() {
    let a = artifact();
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "gnet9-provenance-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).unwrap();
    let source = dir.join("source.cpp");
    let exe = dir.join("run");
    fs::write(&source, format!("{}\n{DRIVER}", a.artifact().source())).unwrap();
    let compiled = child_deadline::command(60)
        .args([
            "c++",
            "-std=c++17",
            "-O2",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
        ])
        .arg(&source)
        .arg("-o")
        .arg(&exe)
        .output()
        .unwrap();
    fs::write(dir.join("compiler.stderr"), &compiled.stderr).unwrap();
    assert!(
        compiled.status.success(),
        "{}: {}",
        dir.display(),
        String::from_utf8_lossy(&compiled.stderr)
    );
    let executable = fs::read(&exe).unwrap();
    for bits in [0x80000000, 0x7fa12345] {
        let input = args(bits);
        let observed = a.observe_oracle(&input, limits()).unwrap().observation();
        let expected: Vec<u64> = observed
            .events
            .iter()
            .flat_map(|e| {
                [
                    u64::from(e.kind == oracle::EventKind::Complete),
                    e.frame,
                    e.activation,
                    e.token.block.function as u64,
                    e.token.block.block as u64,
                    e.token.position as u64,
                    e.rule as u64,
                ]
            })
            .collect();
        let run = child_deadline::command(10)
            .arg(&exe)
            .arg(bits.to_string())
            .output()
            .unwrap();
        fs::write(dir.join(format!("run-{bits}.stdout")), &run.stdout).unwrap();
        fs::write(dir.join(format!("run-{bits}.stderr")), &run.stderr).unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let text = String::from_utf8(run.stdout).unwrap();
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        for (phase, line) in lines.into_iter().enumerate() {
            let words: Vec<i64> = line
                .split_whitespace()
                .map(|n| n.parse().unwrap())
                .collect();
            let done = words[0] != 0;
            assert_eq!(done, phase == 1);
            let events = usize::try_from(words[1]).unwrap();
            let output = &words[2..10];
            let trace: Vec<u64> = words[10..10 + events * 7]
                .iter()
                .map(|&n| u64::try_from(n).unwrap())
                .collect();
            let task: Vec<u8> = words[10 + events * 7..]
                .iter()
                .map(|&n| u8::try_from(n).unwrap())
                .collect();
            assert_eq!(trace, &expected[..trace.len()]);
            if done {
                assert_eq!(output[0], 0);
                assert_eq!(output[1], i64::from(bits));
                assert_eq!(output[4], observed.semantic.steps_used as i64);
                assert_eq!(output[5], observed.transitions as i64);
                assert_eq!(trace, expected);
            } else {
                assert_eq!(output, [111, 112, 113, 114, 115, 116, 117, 118]);
            }
            let components = || NumericContinuationComponents {
                executable: &executable,
                args: &input,
                limits: limits(),
                quantum: if done { 3 } else { 1 },
                task: &task,
                output: done.then_some(output),
                trace: &trace,
            };
            let id = a.numeric_continuation_identity(components()).unwrap();
            id.verify_artifact(&a).unwrap();
            id.verify_components(components()).unwrap();
            assert_eq!(id.manifest()["execution_observed"], false);
            assert_eq!(id.manifest()["task_semantics_validated"], false);
            assert_eq!(id.manifest()["source_completion_certified"], false);
            assert_eq!(id.manifest()["resume_authorized"], false);
            assert_eq!(id.manifest()["input"]["words"][0], bits);
            let mut changed_task = task.clone();
            changed_task[0] ^= 1;
            let mut c = components();
            c.task = &changed_task;
            assert_eq!(id.verify_components(c), Err(Error::ExecutionMismatch));
            let changed_input = args(bits ^ 0x80000000);
            let mut c = components();
            c.args = &changed_input;
            assert_eq!(id.verify_components(c), Err(Error::ExecutionMismatch));
            let mut c = components();
            c.quantum += 1;
            assert_eq!(id.verify_components(c), Err(Error::ExecutionMismatch));
            for field in 0..4 {
                let mut c = components();
                match field {
                    0 => c.limits.fuel += 1,
                    1 => c.limits.transitions += 1,
                    2 => c.limits.trace_events += 1,
                    _ => c.limits.array_elements += 1,
                };
                assert_eq!(id.verify_components(c), Err(Error::ExecutionMismatch));
            }
            let mut library = executable.clone();
            library[0] ^= 1;
            let mut c = components();
            c.executable = &library;
            assert_eq!(id.verify_components(c), Err(Error::ExecutionMismatch));
            if done {
                let mut changed_output = output.to_vec();
                changed_output[1] ^= 1;
                let mut c = components();
                c.output = Some(&changed_output);
                assert_eq!(id.verify_components(c), Err(Error::ExecutionMismatch));
                let mut c = components();
                c.output = None;
                assert_eq!(id.verify_components(c), Err(Error::ExecutionMismatch));
            }
            if !trace.is_empty() {
                let mut changed_trace = trace.clone();
                changed_trace[0] ^= 1;
                let mut c = components();
                c.trace = &changed_trace;
                assert_eq!(id.verify_components(c), Err(Error::ExecutionMismatch));
            }
            id.verify_components(components()).unwrap();
            let mut forged = id.manifest().clone();
            forged["resume_authorized"] = serde_json::json!(true);
            assert_eq!(id.verify_manifest(&forged), Err(Error::ManifestMismatch));
            let scalar =
                provenance::compile_numeric(SOURCE, "run", Target::Cpu, Limits::default()).unwrap();
            assert_eq!(id.verify_artifact(&scalar), Err(Error::ArtifactMismatch));
        }
    }
}

#[test]
fn numeric_continuation_identity_rejects_profile_and_malformed_component_framing() {
    let a = artifact();
    let input = args(0x80000000);
    let task = vec![0; a.artifact().continuation_layout().unwrap().task_bytes()];
    let components = || NumericContinuationComponents {
        executable: b"supplied-component-only",
        args: &input,
        limits: limits(),
        quantum: 1,
        task: &task,
        output: None,
        trace: &[],
    };
    let scalar =
        provenance::compile_numeric(SOURCE, "run", Target::Cpu, Limits::default()).unwrap();
    assert!(matches!(
        scalar.numeric_continuation_identity(components()),
        Err(Error::ExecutionMismatch)
    ));
    let mut c = components();
    c.task = &task[..task.len() - 1];
    assert!(matches!(
        a.numeric_continuation_identity(c),
        Err(Error::ContinuationMismatch)
    ));
    let mut c = components();
    c.executable = &[];
    assert!(matches!(
        a.numeric_continuation_identity(c),
        Err(Error::ExecutionMismatch)
    ));
    let bad_args = [Value::U32(0), Value::U32(2)];
    let mut c = components();
    c.args = &bad_args;
    assert!(matches!(
        a.numeric_continuation_identity(c),
        Err(Error::ExecutionMismatch)
    ));
    let mut c = components();
    c.output = Some(&[0; 7]);
    assert!(matches!(
        a.numeric_continuation_identity(c),
        Err(Error::ExecutionMismatch)
    ));
    let mut c = components();
    c.output = Some(&[0, 0, 0, 0, 0, 0, 0, -1]);
    assert!(matches!(
        a.numeric_continuation_identity(c),
        Err(Error::ExecutionMismatch)
    ));
    let mut c = components();
    c.trace = &[0];
    assert!(matches!(
        a.numeric_continuation_identity(c),
        Err(Error::ExecutionMismatch)
    ));
    let mut c = components();
    c.trace = &[0; 7];
    c.limits.trace_events = 0;
    assert!(matches!(
        a.numeric_continuation_identity(c),
        Err(Error::ExecutionMismatch)
    ));
    // A failed-runtime quantum may be recorded, without admitting resume.
    let mut c = components();
    c.quantum = 0;
    assert_eq!(
        a.numeric_continuation_identity(c).unwrap().manifest()["resume_authorized"],
        false
    );
}
