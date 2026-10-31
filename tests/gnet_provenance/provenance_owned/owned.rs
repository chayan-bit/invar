use super::*;

#[test]
fn provenance_actual_abi_owned_and_continuation_layouts() {
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
fn provenance_owned_budgets_target_and_source_mutations() {
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
fn provenance_layout_forgery_rejected_and_oracle_not_resume_or_gpu() {
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

#[test]
fn provenance_real_owned_cpu_and_resume_quantum_mutations() {
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
fn provenance_owned_receipt_profile_identity() {
    let artifact = compile(SOURCE, true, storage());
    println!(
        "{}",
        json!({"compiler_build":env!("INVAR_BUILD_DIGEST"),"debug_assertions":cfg!(debug_assertions),
        "artifact_id":artifact.id(),"manifest":artifact.manifest()})
    );
    assert_eq!(artifact.manifest()["abi_version"], Value::from(3));
}

#[test]
fn provenance_owned_results_do_not_alias_receipt_state() {
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
fn provenance_real_cpu_backend_limits_are_not_oracle_success() {
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
