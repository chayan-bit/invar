#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "../test-support/child_deadline.rs"]
mod child_deadline;
use invar::{
    eval,
    general_net::{compiled, oracle, provenance},
    numeric_contract::{F32Bits, NumericPolicy},
};
use provenance::{Error, Limits, ProvenancedArtifact};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

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

#[test]
fn issue_gnet_05_provenance_numeric_exact_policy_bits_layout_and_all_compile_budgets() {
    let a = numeric(NUMERIC);
    let b = numeric(NUMERIC);
    assert_eq!(a.id(), b.id());
    assert_eq!(a.manifest(), b.manifest());
    assert_eq!(a.artifact().abi_version(), 11);
    assert_eq!(
        a.artifact().numeric_policy(),
        Some(NumericPolicy::STRICT_F32)
    );
    assert_eq!(
        a.manifest()["numeric"]["policy"],
        NumericPolicy::STRICT_F32.to_json()
    );
    assert_eq!(a.manifest()["numeric"]["semantic_id"], "strict-f32-v1");
    assert_eq!(
        a.manifest()["numeric"]["layout_version"],
        invar::layout_contract::VERSION
    );
    assert_eq!(a.manifest()["numeric"]["f32"]["maximum"], u32::MAX);
    assert_eq!(a.manifest()["numeric"]["output_words"], 10);
    assert_eq!(
        a.manifest()["numeric"]["event_words"],
        compiled::EVENT_WORDS
    );
    assert_eq!(a.manifest()["params"], json!(["F32", "F32", "F32"]));
    assert_eq!(a.manifest()["result"], "F32");
    assert!(a.artifact().owned_layout().is_some());
    assert!(a.artifact().continuation_layout().is_some());
    assert!(a.artifact().speculation_layout().is_none());
    assert!(a.artifact().device_layout().is_none());
    assert!(a.artifact().aggregate_layout().is_none());
    for field in 0..10 {
        let mut l = Limits::default();
        match field {
            0 => l.net.functions += 1,
            1 => l.net.blocks += 1,
            2 => l.net.instructions += 1,
            3 => l.net.wires += 1,
            4 => l.net.literal_bytes += 1,
            5 => l.compiler.values += 1,
            6 => l.compiler.places += 1,
            7 => l.compiler.states += 1,
            8 => l.compiler.frame_bytes += 1,
            9 => l.compiler.source_bytes += 1,
            _ => unreachable!(),
        }
        let b = numeric_with(NUMERIC, compiled::Target::Cpu, l).unwrap();
        assert_ne!(a.id(), b.id(), "compile budget {field}");
        assert_eq!(a.artifact().source(), b.artifact().source());
    }
    // Non-numeric manifests carry no numeric block; numeric policy is added
    // only when the actual emitter identifies a numeric artifact.
    let legacy = provenance::compile(
        b"fn calc() -> i32 { return 7; }",
        "calc",
        compiled::Target::Cpu,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(legacy.manifest()["version"], 3);
    assert!(legacy.manifest().get("numeric").is_none());
    assert!(
        legacy
            .observe_oracle(&[], numeric_limits())
            .unwrap()
            .manifest()
            .get("numeric")
            .is_none()
    );
}

#[test]
fn issue_gnet_05_provenance_numeric_stale_mutations_and_unsupported_modes_reject() {
    let a = numeric(NUMERIC);
    for forged in forged_leaves(a.manifest()) {
        assert_eq!(a.verify_manifest(&forged), Err(Error::ManifestMismatch));
    }
    let changed = numeric_separate(NUMERIC);
    let b = numeric(&changed);
    assert_ne!(a.id(), b.id());
    assert_eq!(a.verify_source(&changed), Err(Error::SourceMismatch));
    assert_eq!(
        a.verify_artifact_bytes(b.artifact().source().as_bytes()),
        Err(Error::ArtifactMismatch)
    );
    let (net, proof) = b.checked_net();
    assert_eq!(a.verify_net(net, proof), Err(Error::NetMismatch));
    let (net, proof) = a.checked_net();
    let mut stale = net.clone();
    stale.graph.revision += 1;
    assert!(matches!(a.verify_net(&stale, proof), Err(Error::Net(_))));
    assert!(matches!(
        numeric_with(NUMERIC, compiled::Target::Metal, Limits::default()),
        Err(Error::Compiler(compiled::Error::NumericTarget))
    ));
    assert!(matches!(
        provenance::compile(NUMERIC, "calc", compiled::Target::Cpu, Limits::default()),
        Err(Error::Compiler(compiled::Error::UnsupportedType))
    ));
    assert!(matches!(
        provenance::compile_owned(
            NUMERIC,
            "calc",
            compiled::Target::Cpu,
            Limits::default(),
            storage()
        ),
        Err(Error::Compiler(compiled::Error::UnsupportedType))
    ));
    assert!(matches!(
        numeric_with(
            b"struct Numeric{x:f32} fn calc() -> i32 { return 7; }",
            compiled::Target::Cpu,
            Limits::default()
        ),
        Err(_)
    ));
    assert!(matches!(
        numeric_with(
            b"fn calc() -> f32 { return 1u32; }",
            compiled::Target::Cpu,
            Limits::default()
        ),
        Err(Error::Source(_))
    ));
}

#[test]
fn issue_gnet_05_provenance_numeric_oracle_preserves_exact_nan_zero_and_runtime_budgets() {
    let echo = numeric(b"fn calc(a: f32) -> f32 { return a; }");
    let mut ids = std::collections::BTreeSet::new();
    for bits in [
        0,
        0x80000000,
        1,
        0x007fffff,
        0x00800000,
        0x3f800000,
        0x7f800000,
        0xff800000,
        0x7fc12345,
        0x7f812345,
        0xffc54321,
        u32::MAX,
    ] {
        let args = numeric_args(&[bits]);
        let e = echo.observe_oracle(&args, numeric_limits()).unwrap();
        assert!(ids.insert(e.id().to_owned()));
        assert_eq!(
            e.observation().semantic.result,
            Ok(eval::Value::F32(F32Bits::from_bits(bits)))
        );
        assert_eq!(e.manifest()["input_words"], json!([bits]));
        assert_eq!(e.manifest()["result_bits"], bits);
        let Ok(eval::Value::F32(snapshot)) = e.observation().semantic.result else {
            panic!("expected an actual f32 snapshot");
        };
        assert_eq!(snapshot.to_bits(), bits);
        let mut observed = e.observation();
        observed.semantic.result = Ok(eval::Value::F32(F32Bits::POS_ZERO));
        assert_eq!(
            e.observation().semantic.result,
            Ok(eval::Value::F32(F32Bits::from_bits(bits)))
        );
        for forged in forged_leaves(e.manifest()) {
            assert_eq!(e.verify_manifest(&forged), Err(Error::ManifestMismatch));
        }
    }
    let args = numeric_args(&[0x80000000]);
    let full = echo.observe_oracle(&args, numeric_limits()).unwrap();
    for field in 0..4 {
        let mut l = numeric_limits();
        match field {
            0 => l.fuel += 1,
            1 => l.transitions += 1,
            2 => l.trace_events += 1,
            3 => l.array_elements += 1,
            _ => unreachable!(),
        }
        let changed = echo.observe_oracle(&args, l).unwrap();
        assert_ne!(full.id(), changed.id());
        assert_eq!(
            changed.observation().semantic.result,
            full.observation().semantic.result
        );
    }
    for fuel in 1..=full.observation().semantic.steps_used + 1 {
        let l = oracle::Limits {
            fuel,
            ..numeric_limits()
        };
        let e = echo.observe_oracle(&args, l).unwrap();
        assert_eq!(e.manifest()["limits"]["fuel"], fuel);
        if fuel < full.observation().semantic.steps_used {
            assert!(matches!(
                e.observation().semantic.result,
                Err(eval::EvalError::EvaluationLimit { .. })
            ));
        }
    }
    assert!(
        echo.observe_oracle(&[eval::Value::U32(0)], numeric_limits())
            .is_err()
    );
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

#[test]
fn issue_gnet_05_provenance_actual_abi_owned_and_continuation_layouts() {
    let scalar = b"fn calc(n: u32) -> u32 { return n + 1u32; }";
    let v1 = provenance::compile(scalar, "calc", compiled::Target::Cpu, Limits::default()).unwrap();
    let v2 = compile(scalar, false, storage());
    let v3 = compile(scalar, true, storage());
    assert_ne!(v1.id(), v2.id());
    assert_ne!(v2.id(), v3.id());
    for artifact in [&v1, &v2, &v3] {
        let manifest = artifact.manifest();
        assert_eq!(manifest["abi_version"], artifact.artifact().abi_version());
        if let Some(layout) = artifact.artifact().owned_layout() {
            assert_eq!(
                manifest["layouts"]["owned"],
                json!({
                "elements":layout.elements(), "allocations":layout.allocations(),
                "cleanup_events":layout.cleanup_events(), "heap_words":layout.heap_words(),
                "cleanup_words":layout.cleanup_words(), "storage_bytes":layout.storage_bytes(),
                "output_words":layout.output_words(), "kernel_name":layout.kernel_name()})
            );
            assert_eq!(manifest["storage"]["output_words"], 10);
        } else {
            assert!(manifest["layouts"]["owned"].is_null());
            assert_eq!(manifest["storage"]["output_words"], 8);
        }
        if let Some(layout) = artifact.artifact().continuation_layout() {
            assert_eq!(
                manifest["layouts"]["continuation"],
                json!({
                "task_bytes":layout.task_bytes(), "kernel_name":layout.kernel_name(),
                "config_words":layout.config_words(), "max_roots":layout.max_roots(),
                "max_slices":layout.max_slices()})
            );
            assert_eq!(
                layout.task_bytes(),
                artifact.artifact().frame_bytes() + 18 * 8
            );
        } else {
            assert!(manifest["layouts"]["continuation"].is_null());
        }
    }
    assert_eq!(v2.manifest()["layouts"]["owned"]["heap_words"], 48);
    assert_eq!(v2.manifest()["layouts"]["owned"]["cleanup_words"], 64);
    assert_eq!(v2.manifest()["layouts"]["owned"]["storage_bytes"], 896);
}

#[test]
fn issue_gnet_05_provenance_owned_budgets_target_and_source_mutations() {
    for resumable in [false, true] {
        let artifact = compile(SOURCE, resumable, storage());
        for field in 0..3 {
            let mut limits = storage();
            match field {
                0 => limits.elements += 1,
                1 => limits.allocations += 1,
                2 => limits.cleanup_events += 1,
                _ => unreachable!(),
            }
            assert_ne!(artifact.id(), compile(SOURCE, resumable, limits).id());
        }
        let emit = if resumable {
            provenance::compile_resumable
        } else {
            provenance::compile_owned
        };
        let cuda = emit(
            SOURCE,
            "calc",
            compiled::Target::Cuda,
            Limits::default(),
            storage(),
        )
        .unwrap();
        let metal = emit(
            SOURCE,
            "calc",
            compiled::Target::Metal,
            Limits::default(),
            storage(),
        )
        .unwrap();
        assert_ne!(artifact.id(), cuda.id());
        assert_ne!(cuda.id(), metal.id());
        assert_eq!(cuda.manifest()["layouts"], artifact.manifest()["layouts"]);
        let changed = String::from_utf8(SOURCE.to_vec())
            .unwrap()
            .replace("n, 3", "n, 7");
        assert_ne!(
            artifact.id(),
            compile(changed.as_bytes(), resumable, storage()).id()
        );
        assert_eq!(
            artifact.verify_source(changed.as_bytes()),
            Err(Error::SourceMismatch)
        );
        assert_eq!(
            artifact.verify_artifact_bytes(cuda.artifact().source().as_bytes()),
            Err(Error::ArtifactMismatch)
        );
    }
    assert!(matches!(
        provenance::compile_owned(
            SOURCE,
            "calc",
            compiled::Target::Cpu,
            Limits::default(),
            compiled::OwnedLimits {
                allocations: 0,
                ..storage()
            }
        ),
        Err(Error::Compiler(compiled::Error::Limit))
    ));
}

#[test]
fn issue_gnet_05_provenance_layout_forgery_rejected_and_oracle_not_resume_or_gpu() {
    let artifact = compile(SOURCE, true, storage());
    for (section, fields) in [
        (
            "owned",
            vec![
                "elements",
                "allocations",
                "cleanup_events",
                "heap_words",
                "cleanup_words",
                "storage_bytes",
                "output_words",
                "kernel_name",
            ],
        ),
        (
            "continuation",
            vec![
                "task_bytes",
                "kernel_name",
                "config_words",
                "max_roots",
                "max_slices",
            ],
        ),
    ] {
        for field in fields {
            let mut forged = artifact.manifest().clone();
            forged["layouts"][section][field] = json!("forged");
            assert_eq!(
                artifact.verify_manifest(&forged),
                Err(Error::ManifestMismatch)
            );
        }
    }
    let mut forged = artifact.manifest().clone();
    forged["abi_version"] = json!(2);
    assert_eq!(
        artifact.verify_manifest(&forged),
        Err(Error::ManifestMismatch)
    );
    let cuda = provenance::compile_resumable(
        SOURCE,
        "calc",
        compiled::Target::Cuda,
        Limits::default(),
        storage(),
    )
    .unwrap();
    let oracle = cuda
        .observe_oracle(&[eval::Value::U32(3)], oracle::Limits::default())
        .unwrap();
    assert_eq!(
        oracle.observation().semantic.result,
        Ok(eval::Value::I32(5))
    );
    assert_eq!(oracle.manifest()["backend"], "cpu_net_oracle");
    assert_eq!(oracle.manifest()["artifact_target"], "Cuda");
    assert_eq!(oracle.manifest()["abi_version"], 3);
    assert_eq!(oracle.manifest()["continuation_execution"], false);
    assert_eq!(oracle.manifest()["layouts"], cuda.manifest()["layouts"]);
    let mut forged = oracle.manifest().clone();
    forged["continuation_execution"] = json!(true);
    assert_eq!(
        oracle.verify_manifest(&forged),
        Err(Error::ManifestMismatch)
    );
}

/// Runs real emitted CPU instructions. Hashes identify captured continuation
/// bytes; they do not decide computation or certify arbitrary imported tasks.
struct Cpu {
    dir: PathBuf,
}
impl Cpu {
    fn new(artifact: &ProvenancedArtifact) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "invar-provenance-owned-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
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
        let output = child_deadline::command(60)
            .arg("c++")
            .args([
                "-std=c++17",
                "-O1",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
            ])
            .arg(dir.join("candidate.cpp"))
            .arg("-o")
            .arg(dir.join("candidate"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
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

#[test]
fn issue_gnet_05_provenance_real_owned_cpu_and_resume_quantum_mutations() {
    for (source, result) in [
        (SOURCE.to_vec(), 5),
        (
            String::from_utf8(SOURCE.to_vec())
                .unwrap()
                .replace("n, 3", "n, 7")
                .into_bytes(),
            9,
        ),
    ] {
        for resumable in [false, true] {
            let artifact = compile(&source, resumable, storage());
            let cpu = Cpu::new(&artifact);
            for quantum in if resumable { vec![1, 3, 17] } else { vec![1] } {
                let (words, prefix, completed) = cpu.run(quantum, false);
                check_actual(&artifact, &words, result);
                if resumable {
                    let bytes = artifact
                        .artifact()
                        .continuation_layout()
                        .unwrap()
                        .task_bytes();
                    assert_eq!(prefix.len(), bytes);
                    assert_eq!(completed.len(), bytes);
                    assert_ne!(prefix, completed);
                    let identity = artifact.continuation_identity(&prefix).unwrap();
                    artifact.verify_continuation(&identity, &prefix).unwrap();
                    let mut changed = prefix.clone();
                    changed[0] ^= 1;
                    assert_eq!(
                        artifact.verify_continuation(&identity, &changed),
                        Err(Error::ContinuationMismatch)
                    );
                    assert_eq!(
                        artifact.verify_continuation(&identity, &completed),
                        Err(Error::ContinuationMismatch)
                    );
                    assert_eq!(
                        artifact
                            .continuation_identity(&prefix[..prefix.len() - 1])
                            .unwrap_err(),
                        Error::ContinuationMismatch
                    );
                    let other = provenance::compile_resumable(
                        &source,
                        "calc",
                        compiled::Target::Cuda,
                        Limits::default(),
                        storage(),
                    )
                    .unwrap();
                    assert_eq!(
                        other.verify_continuation(&identity, &prefix),
                        Err(Error::ContinuationMismatch)
                    );
                    let mut forged = identity.manifest().clone();
                    forged["task"]["sha256"] = json!("forged");
                    assert_eq!(
                        identity.verify_manifest(&forged),
                        Err(Error::ManifestMismatch)
                    );
                }
                println!(
                    "{}",
                    json!({"backend":"emitted_cpu_ubsan", "abi_version":artifact.artifact().abi_version(),
                    "artifact_id":artifact.id(),"layouts":artifact.manifest()["layouts"],"quantum":quantum,
                    "input":[3],"fuel":1000,"transition_cap":10000,"trace_cap":10000,
                    "header":&words[..10],"output_sha256":hash(format!("{words:?}").as_bytes()),
                    "yielded_task_sha256":hash(&prefix),"completed_task_sha256":hash(&completed)})
                );
            }
            if resumable {
                let (words, _, _) = cpu.run(3, true);
                assert_eq!(words[0], compiled::Status::InvalidInput as i64);
                assert_eq!(words[1], 0);
                assert_eq!(words[8], 0);
            }
        }
    }
}

#[test]
fn issue_gnet_05_provenance_owned_receipt_profile_identity() {
    let artifact = compile(SOURCE, true, storage());
    println!(
        "{}",
        json!({"compiler_build":env!("INVAR_BUILD_DIGEST"),"debug_assertions":cfg!(debug_assertions),
        "artifact_id":artifact.id(),"manifest":artifact.manifest()})
    );
    assert_eq!(artifact.manifest()["abi_version"], Value::from(3));
}

#[test]
fn issue_gnet_05_provenance_owned_results_do_not_alias_receipt_state() {
    for unsigned in [false, true] {
        let source = if unsigned {
            b"fn calc(n: u32) -> Array<u32> { let xs = filled_u32(n, 3u32); return xs; }".as_slice()
        } else {
            b"fn calc(n: u32) -> Array<i32> { let xs = filled_i32(n, 3); return xs; }".as_slice()
        };
        let artifact = compile(source, true, storage());
        let receipt = artifact
            .observe_oracle(&[eval::Value::U32(3)], oracle::Limits::default())
            .unwrap();
        let manifest = receipt.manifest().clone();
        let identity = receipt.id().to_owned();
        match receipt.observation().semantic.result.unwrap() {
            eval::Value::ArrayI32(xs) => xs.borrow_mut()[0] = 999,
            eval::Value::ArrayU32(xs) => xs.borrow_mut()[0] = 999,
            other => panic!("expected owned result: {other:?}"),
        }
        match receipt.observation().semantic.result.unwrap() {
            eval::Value::ArrayI32(xs) => assert_eq!(*xs.borrow(), vec![3, 3, 3]),
            eval::Value::ArrayU32(xs) => assert_eq!(*xs.borrow(), vec![3, 3, 3]),
            other => panic!("expected owned result: {other:?}"),
        }
        assert_eq!(receipt.id(), identity);
        receipt.verify_manifest(&manifest).unwrap();
        assert_eq!(
            artifact.manifest()["result"],
            if unsigned { "ArrayU32" } else { "ArrayI32" }
        );
    }
}

#[test]
fn issue_gnet_05_provenance_real_cpu_backend_limits_are_not_oracle_success() {
    for resumable in [false, true] {
        let artifact = compile(
            SOURCE,
            resumable,
            compiled::OwnedLimits {
                elements: 2,
                ..storage()
            },
        );
        let cpu = Cpu::new(&artifact);
        let (words, _, _) = cpu.run(3, false);
        assert_eq!(words[0], compiled::Status::HeapLimit as i64);
        assert_eq!(words[1], 0);
        let oracle = artifact
            .observe_oracle(&[eval::Value::U32(3)], oracle::Limits::default())
            .unwrap();
        assert_eq!(
            oracle.observation().semantic.result,
            Ok(eval::Value::I32(5))
        );
        assert_eq!(oracle.manifest()["backend"], "cpu_net_oracle");
        println!(
            "{}",
            json!({"backend":"emitted_cpu_ubsan", "artifact_id":artifact.id(),
            "abi_version":artifact.artifact().abi_version(), "limits":artifact.manifest()["layouts"],
            "header":&words[..10],"status":"backend_heap_limit"})
        );
        if resumable {
            for quantum in [0, 1_000_001] {
                let (words, _, _) = cpu.run(quantum, false);
                assert_eq!(words[0], compiled::Status::InvalidInput as i64);
            }
        }
    }
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

#[test]
fn issue_gnet_05_provenance_abi45_complete_layouts_and_fork_facts() {
    for target in [
        compiled::Target::Cpu,
        compiled::Target::Cuda,
        compiled::Target::Metal,
    ] {
        let a = children(target, private_limits());
        let artifact = a.artifact();
        let m = a.manifest();
        assert_eq!(m["version"], 3);
        assert_eq!(
            m["abi_version"],
            if target == compiled::Target::Cpu {
                4
            } else {
                5
            }
        );
        let l = artifact.speculation_layout().unwrap();
        let limits = l.limits();
        assert_eq!(
            m["layouts"]["speculation"],
            json!({"slots":l.slots(),"slot_bytes":l.slot_bytes(),
            "heap_words":l.heap_words(),"trace_words":l.trace_words(),"cleanup_words":l.cleanup_words(),
            "scratch_bytes":l.scratch_bytes(),"entry_point":l.entry_point(),"limits":{
            "slots":limits.slots,"fuel":limits.fuel,"transitions":limits.transitions,
            "trace_events":limits.trace_events,"quantum":limits.quantum,"lookahead":limits.lookahead,
            "scratch_bytes":limits.scratch_bytes}})
        );
        let plan = artifact.fork_plan().unwrap();
        assert!(!plan.candidates.is_empty());
        assert_eq!(
            m["layouts"]["fork_plan"]["candidates"]
                .as_array()
                .unwrap()
                .len(),
            plan.candidates.len()
        );
        for (actual, c) in m["layouts"]["fork_plan"]["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&plan.candidates)
        {
            let site = |s: compiled::CallSite| json!({"function":s.function,"block":s.block,"position":s.position});
            assert_eq!(
                *actual,
                json!({"source":site(c.source),"target":site(c.target),"callee":c.callee,
                "projection":c.projection,"scalar_values":c.scalar_values,"scalar_places":c.scalar_places,"replay_owners":c.replay_owners})
            );
        }
        if let Some(d) = artifact.device_layout() {
            let c = compiled::DeviceConfig {
                fuel: 1,
                transitions: 0,
                trace_events: 0,
                quantum: 1,
                rounds: 1,
                capacity: 0,
            };
            assert_eq!(
                m["layouts"]["device"],
                json!({"kernel_name":d.kernel_name(),"group_threads":d.group_threads(),
                "groups":d.groups(),"max_rounds":d.max_rounds(),"state_words":d.state_words(),
                "minimum_config_words":c.words(),"minimum_buffer_bytes":d.buffer_bytes(c).unwrap(),"buffers":12})
            );
        } else {
            assert!(m["layouts"]["device"].is_null());
        }
        for forged in forged_leaves(m) {
            assert_eq!(a.verify_manifest(&forged), Err(Error::ManifestMismatch));
        }
        let oracle = a
            .observe_oracle(&[eval::Value::U32(3)], oracle::Limits::default())
            .unwrap();
        assert_eq!(oracle.manifest()["backend"], "cpu_net_oracle");
        assert_eq!(oracle.manifest()["continuation_execution"], false);
        assert_eq!(oracle.manifest()["layouts"], m["layouts"]);
    }
    assert!(matches!(
        provenance::compile_speculative(
            CHILDREN,
            "calc",
            compiled::Target::Cuda,
            Limits::default(),
            storage(),
            private_limits()
        ),
        Err(Error::Compiler(compiled::Error::SpeculationTarget))
    ));
    assert!(matches!(
        provenance::compile_device_speculative(
            CHILDREN,
            "calc",
            compiled::Target::Cpu,
            Limits::default(),
            storage(),
            private_limits()
        ),
        Err(Error::Compiler(compiled::Error::SpeculationTarget))
    ));
    // GNET-03: device children admit owners inside the program; only an
    // owner results cross only the validated ABI13 root-heap boundary.
    let owned = provenance::compile_device_speculative(
        SOURCE,
        "calc",
        compiled::Target::Cuda,
        Limits::default(),
        storage(),
        private_limits(),
    )
    .unwrap();
    assert_eq!(owned.artifact().abi_version(), 5);
    assert!(owned.artifact().device_layout().is_some());
    let returned = provenance::compile_device_speculative(
        b"fn calc(n: u32) -> Array<i32> { return filled_i32(n, 3); }",
        "calc",
        compiled::Target::Cuda,
        Limits::default(),
        storage(),
        private_limits(),
    )
    .unwrap();
    assert_eq!(returned.artifact().abi_version(), 13);
    assert!(returned.artifact().entry_layout().is_some());
}

#[test]
fn issue_gnet_05_provenance_abi45_every_private_budget_and_source_is_bound() {
    for target in [
        compiled::Target::Cpu,
        compiled::Target::Cuda,
        compiled::Target::Metal,
    ] {
        let a = children(target, private_limits());
        for i in 0..7 {
            let mut l = private_limits();
            match i {
                0 => l.slots += 1,
                1 => l.fuel += 1,
                2 => l.transitions += 1,
                3 => l.trace_events += 1,
                4 => l.quantum += 1,
                5 => l.lookahead += 1,
                6 => l.scratch_bytes += 1,
                _ => unreachable!(),
            };
            assert_ne!(a.id(), children(target, l).id(), "private field {i}");
        }
        let changed = String::from_utf8(CHILDREN.to_vec())
            .unwrap()
            .replace("return 1;", "return 2;");
        let emit = if target == compiled::Target::Cpu {
            provenance::compile_speculative
        } else {
            provenance::compile_device_speculative
        };
        let b = emit(
            changed.as_bytes(),
            "calc",
            target,
            Limits::default(),
            storage(),
            private_limits(),
        )
        .unwrap();
        assert_ne!(a.id(), b.id());
        assert_ne!(a.artifact().source(), b.artifact().source());
        assert_eq!(
            a.verify_manifest(b.manifest()),
            Err(Error::ManifestMismatch)
        );
        let task = vec![0; a.artifact().continuation_layout().unwrap().task_bytes()];
        let token = a.continuation_identity(&task).unwrap();
        assert_eq!(token.manifest()["scope"], "caller_supplied_task_component");
        assert_eq!(
            b.verify_continuation(&token, &task),
            Err(Error::ContinuationMismatch)
        );
    }
    assert_ne!(
        children(compiled::Target::Cuda, private_limits()).id(),
        children(compiled::Target::Metal, private_limits()).id()
    );
    let mut invalid = private_limits();
    invalid.slots = 9;
    assert!(matches!(
        provenance::compile_device_speculative(
            CHILDREN,
            "calc",
            compiled::Target::Cuda,
            Limits::default(),
            storage(),
            invalid
        ),
        Err(Error::Compiler(compiled::Error::Limit))
    ));
}

#[test]
fn issue_gnet_05_provenance_fenced_checkpoints_bind_all_ranges_bytes_and_epochs() {
    let a = children(compiled::Target::Cuda, private_limits());
    let w = workset(&a, device_config());
    let bytes = buffers(&w);
    let mut e = a
        .fenced_execution(
            w.clone(),
            bytes.clone(),
            2,
            provenance::FenceBackend::CpuGroupModel,
        )
        .unwrap();
    complete(&mut e);
    let r = e.capture(Clone::clone).unwrap();
    e.verify_checkpoint(&r, Clone::clone).unwrap();
    assert_eq!(
        r.manifest()["fence"],
        json!({"state":"Ready","submitted":1,"completed":1,"counter_semantics":"core_host_attempt_ordinals","successful_submissions":1,"successful_completions":1})
    );
    assert_eq!(r.manifest()["config_words"], json!(device_config().words()));
    assert_eq!(r.manifest()["evidence_backend"], "CpuGroupModel");
    assert_eq!(r.manifest()["source_completion_certified"], false);
    assert_eq!(r.manifest()["resume_authorized"], false);
    for i in 0..12 {
        let range = w.ranges()[i];
        assert_eq!(
            r.manifest()["ranges"][i],
            json!({"device":range.device,"allocation":range.allocation,"offset":range.offset,"bytes":range.bytes})
        );
        let mut mutated = bytes.clone();
        mutated[i][0] ^= 1;
        assert_eq!(
            e.verify_checkpoint(&r, |_| mutated),
            Err(Error::ExecutionMismatch),
            "buffer {i}"
        );
        let mut short = bytes.clone();
        short[i].pop();
        assert_eq!(e.capture(|_| short).unwrap_err(), Error::ExecutionMismatch);
        let mut long = bytes.clone();
        long[i].push(0);
        assert_eq!(e.capture(|_| long).unwrap_err(), Error::ExecutionMismatch);
    }
    for forged in forged_leaves(r.manifest()) {
        assert_eq!(r.verify_manifest(&forged), Err(Error::ManifestMismatch));
    }
    let mut other = a
        .fenced_execution(w, bytes.clone(), 2, provenance::FenceBackend::CpuGroupModel)
        .unwrap();
    complete(&mut other);
    assert_eq!(
        other.verify_checkpoint(&r, Clone::clone),
        Err(Error::ExecutionMismatch)
    );
    e.submit(|_| compiled::Submission::Submitted).unwrap();
    assert_eq!(
        e.verify_checkpoint(&r, |_| panic!("in-flight read")),
        Err(Error::Fence(compiled::FenceError::Pending))
    );
    e.poll(|_| compiled::FenceObservation::Complete).unwrap();
    assert_eq!(
        e.verify_checkpoint(&r, Clone::clone),
        Err(Error::ExecutionMismatch)
    );
    assert_ne!(e.capture(Clone::clone).unwrap().id(), r.id());
    assert_eq!(
        e.submit(|_| panic!("over-budget launch")),
        Err(Error::Fence(compiled::FenceError::Budget))
    );
    assert_eq!(r.buffers(), &bytes);
}

#[test]
fn issue_gnet_05_provenance_execution_rejects_foreign_layout_backend_and_large_capture() {
    let a = children(compiled::Target::Cuda, private_limits());
    let w = workset(&a, device_config());
    assert!(matches!(
        a.fenced_execution(w.clone(), (), 1, provenance::FenceBackend::MetalCommand),
        Err(Error::ExecutionMismatch)
    ));
    for budget in [0, 1025] {
        assert!(matches!(
            a.fenced_execution(
                w.clone(),
                (),
                budget,
                provenance::FenceBackend::CpuGroupModel
            ),
            Err(Error::Fence(compiled::FenceError::Budget))
        ));
    }
    let m = children(compiled::Target::Metal, private_limits());
    assert!(matches!(
        m.fenced_execution(w.clone(), (), 1, provenance::FenceBackend::CpuGroupModel),
        Err(Error::ExecutionMismatch)
    ));
    let mut l = private_limits();
    l.slots = 1;
    let b = children(compiled::Target::Cuda, l);
    assert!(matches!(
        b.fenced_execution(w.clone(), (), 1, provenance::FenceBackend::CpuGroupModel),
        Err(Error::Compiler(compiled::Error::Limit))
    ));
    let mut l = private_limits();
    l.trace_events += 1;
    let b = children(compiled::Target::Cuda, l);
    assert!(matches!(
        b.fenced_execution(w.clone(), (), 1, provenance::FenceBackend::CpuGroupModel),
        Err(Error::Compiler(compiled::Error::State))
    ));
    let mut ranges = *w.ranges();
    ranges[0].bytes = 256 * 1024 * 1024 + 1;
    let huge = a
        .artifact()
        .device_layout()
        .unwrap()
        .workset(device_config(), ranges)
        .unwrap();
    assert!(matches!(
        a.fenced_execution(huge, (), 1, provenance::FenceBackend::CpuGroupModel),
        Err(Error::Limit)
    ));
    let mut padded = *w.ranges();
    padded[0].bytes += 8;
    let p = a
        .artifact()
        .device_layout()
        .unwrap()
        .workset(device_config(), padded)
        .unwrap();
    let mut e = a
        .fenced_execution(p, buffers(&w), 1, provenance::FenceBackend::CpuGroupModel)
        .unwrap();
    complete(&mut e);
    assert_eq!(
        e.capture(Clone::clone).unwrap_err(),
        Error::ExecutionMismatch
    );
    let mut good = e.reclaim().unwrap();
    good[0].extend_from_slice(&[0; 8]);
    let p = a
        .artifact()
        .device_layout()
        .unwrap()
        .workset(device_config(), padded)
        .unwrap();
    let mut e = a
        .fenced_execution(p, good, 1, provenance::FenceBackend::CpuGroupModel)
        .unwrap();
    complete(&mut e);
    e.capture(Clone::clone).unwrap();
}

#[test]
fn issue_gnet_05_provenance_fence_failure_paths_never_read_or_publish() {
    let a = children(compiled::Target::Cuda, private_limits());
    for observation in [
        compiled::FenceObservation::Pending,
        compiled::FenceObservation::Unknown,
        compiled::FenceObservation::FailedQuiescent,
    ] {
        let w = workset(&a, device_config());
        let mut e = a
            .fenced_execution(w, (), 1, provenance::FenceBackend::CpuGroupModel)
            .unwrap();
        assert_eq!(
            e.capture(|_| panic!("unsubmitted read")).unwrap_err(),
            Error::Fence(compiled::FenceError::Pending)
        );
        e.submit(|_| compiled::Submission::Submitted).unwrap();
        assert!(e.poll(|_| observation).is_err());
        assert_eq!(
            e.capture(|_| panic!("unsafe read")).unwrap_err(),
            Error::Fence(compiled::FenceError::Pending)
        );
        assert_eq!(
            e.submit(|_| panic!("unsafe replay")),
            Err(Error::Fence(compiled::FenceError::Busy))
        );
        if observation == compiled::FenceObservation::Pending {
            e.cancel();
            assert_eq!(
                e.poll(|_| compiled::FenceObservation::Complete),
                Err(Error::Fence(compiled::FenceError::Cancelled))
            );
        }
        if observation == compiled::FenceObservation::Unknown {
            assert_eq!(
                e.poll(|_| compiled::FenceObservation::Complete),
                Err(Error::Fence(compiled::FenceError::Cancelled))
            );
        }
        assert_eq!(
            e.capture(|_| panic!("terminal read")).unwrap_err(),
            Error::Fence(compiled::FenceError::Pending)
        );
    }
    for submission in [
        compiled::Submission::Unknown,
        compiled::Submission::NotSubmitted,
    ] {
        let w = workset(&a, device_config());
        let mut e = a
            .fenced_execution(w, (), 1, provenance::FenceBackend::CpuGroupModel)
            .unwrap();
        assert!(e.submit(|_| submission).is_err());
        assert!(e.capture(|_| panic!("failed launch read")).is_err());
        if submission == compiled::Submission::Unknown {
            assert_eq!(
                e.reclaim().unwrap_err(),
                Error::Fence(compiled::FenceError::Pending)
            );
        } else {
            assert_eq!(
                e.submit(|_| panic!("retry exceeds attempts")),
                Err(Error::Fence(compiled::FenceError::Budget))
            );
            e.reclaim().unwrap();
        }
    }
    let w = workset(&a, device_config());
    let data = buffers(&w);
    let mut e = a
        .fenced_execution(w, data, 2, provenance::FenceBackend::CpuGroupModel)
        .unwrap();
    assert_eq!(
        e.submit(|_| compiled::Submission::NotSubmitted),
        Err(Error::Fence(compiled::FenceError::Submission))
    );
    complete(&mut e);
    let r = e.capture(Clone::clone).unwrap();
    assert_eq!(r.manifest()["fence"]["submitted"], 2);
    assert_eq!(r.manifest()["fence"]["completed"], 2);
    assert_eq!(r.manifest()["fence"]["successful_submissions"], 1);
    assert_eq!(r.manifest()["fence"]["successful_completions"], 1);
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
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "invar-provenance-group-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
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
using uint=unsigned;using ulong=unsigned long;
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
        let output = child_deadline::command(60)
            .arg("c++")
            .args([
                "-std=c++20",
                "-pthread",
                "-O1",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
            ])
            .arg(dir.join("group.cpp"))
            .arg("-o")
            .arg(dir.join("group"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
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
        assert_eq!(
            &trace[i * 7..i * 7 + 7],
            &[
                u64::from(event.kind == oracle::EventKind::Complete),
                event.frame,
                event.activation,
                event.token.block.function as u64,
                event.token.block.block as u64,
                event.token.position as u64,
                event.rule as u64
            ]
        );
    }
    let state = words(&b[11]);
    assert_eq!(state[12], 0);
    assert!(state[2] > 0 && state[3] > 0 && state[8] > 0);
}

#[test]
fn issue_gnet_05_provenance_real_abi45_cpu_execution_and_observed_model_fences() {
    for target in [
        compiled::Target::Cpu,
        compiled::Target::Cuda,
        compiled::Target::Metal,
    ] {
        let a = children(target, private_limits());
        let mut model = GroupModel::new(&a, device_config());
        if target == compiled::Target::Cpu {
            for input in [5, 6] {
                assert_eq!(model.launch(input), compiled::Submission::Submitted);
                assert_eq!(model.wait(), compiled::FenceObservation::Complete);
                let b = model.read();
                compare_scalar_checkpoint(&a, &b, input);
                println!(
                    "{}",
                    json!({"evidence_kind":"actual_emitted_abi4_cpu_ubsan","artifact_id":a.id(),"abi_version":4,"input":input,
                    "header":words(&b[1]),"harness_state":words(&b[11]),"execution_receipt":null,"device_execution":false})
                );
            }
        } else {
            let w = workset(&a, device_config());
            let mut e = a
                .fenced_execution(w, model, 2, provenance::FenceBackend::CpuGroupModel)
                .unwrap();
            assert!(e.capture(|_| panic!("premature actual readback")).is_err());
            let mut previous = None;
            for input in [5, 6] {
                e.submit(|m| m.launch(input)).unwrap();
                assert!(e.capture(|_| panic!("in-flight actual readback")).is_err());
                e.poll(GroupModel::wait).unwrap();
                let receipt = e.capture(GroupModel::read).unwrap();
                compare_scalar_checkpoint(&a, receipt.buffers(), input);
                e.verify_checkpoint(&receipt, GroupModel::read).unwrap();
                if let Some(old) = previous {
                    assert_eq!(
                        e.verify_checkpoint(&old, GroupModel::read),
                        Err(Error::ExecutionMismatch)
                    );
                }
                let task = a.continuation_identity(&receipt.buffers()[6]).unwrap();
                a.verify_continuation(&task, &receipt.buffers()[6]).unwrap();
                let generation_base =
                    a.artifact().continuation_layout().unwrap().task_bytes() + 10 * 8;
                for generation_word in [1, 2] {
                    let mut mutated = receipt.buffers()[7].clone();
                    mutated[generation_base + generation_word * 8] ^= 1;
                    assert_eq!(
                        e.verify_checkpoint(&receipt, |m| {
                            let mut b = m.read();
                            b[7] = mutated;
                            b
                        }),
                        Err(Error::ExecutionMismatch)
                    );
                }
                println!(
                    "{}",
                    json!({"evidence_kind":"actual_emitted_abi5_cpu_group_ubsan","input":input,"header":words(&receipt.buffers()[1]),
                    "state":words(&receipt.buffers()[11]),"receipt_id":receipt.id(),"receipt":receipt.manifest(),"task_component_id":task.id(),"device_execution":false})
                );
                previous = Some(receipt);
            }
            e.reclaim().unwrap();
        }
    }
}

#[test]
fn issue_gnet_05_provenance_all_runtime_config_range_fields_and_unwind_are_bound() {
    let a = children(compiled::Target::Cuda, private_limits());
    for field in 0..6 {
        let mut c = device_config();
        match field {
            0 => c.fuel += 1,
            1 => c.transitions += 1,
            2 => c.trace_events += 1,
            3 => c.quantum += 1,
            4 => c.rounds -= 1,
            5 => c.capacity -= 1,
            _ => unreachable!(),
        };
        let w = workset(&a, c);
        let data = buffers(&w);
        let mut e = a
            .fenced_execution(w, data, 1, provenance::FenceBackend::CpuGroupModel)
            .unwrap();
        complete(&mut e);
        let r = e.capture(Clone::clone).unwrap();
        assert_eq!(r.manifest()["config_words"], json!(c.words()));
        for i in 0..12 {
            for key in ["device", "allocation", "offset", "bytes"] {
                let mut forged = r.manifest().clone();
                forged["ranges"][i][key] = json!(999999);
                assert_eq!(r.verify_manifest(&forged), Err(Error::ManifestMismatch));
            }
        }
        let mut changed = r.buffers().clone();
        changed[3][field * 8] ^= 1;
        assert_eq!(
            e.capture(|_| changed).unwrap_err(),
            Error::ExecutionMismatch
        );
    }
    let w = workset(&a, device_config());
    let mut e = a
        .fenced_execution(w, (), 1, provenance::FenceBackend::CpuGroupModel)
        .unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || e.submit(|_| panic!("controlled adapter unwind"))
        ))
        .is_err()
    );
    assert_eq!(e.state(), compiled::ExecutionState::InFlight);
    assert_eq!(
        e.capture(|_| panic!("unwound read")).unwrap_err(),
        Error::Fence(compiled::FenceError::Pending)
    );
    assert_eq!(
        e.submit(|_| panic!("unwound replay")),
        Err(Error::Fence(compiled::FenceError::Busy))
    );
    e.cancel();
    assert_eq!(
        e.poll(|_| compiled::FenceObservation::Complete),
        Err(Error::Fence(compiled::FenceError::Cancelled))
    );
    e.reclaim().unwrap();
}
