use super::*;

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
