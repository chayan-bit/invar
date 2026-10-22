use invar::{
    admission, check::check, device_ir, eval, front::parse, mir, mir_eval, plan::PlanKind, schedule,
};
use invar_device::{Capabilities, DagOp, Error as VerifyError, Layout, Provenance};

const CHAIN: &str = "fn main() -> i32 { \
    let xs: Array<u32> = [1u32,2u32,3u32]; \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],3u32); } \
    let first: Array<u32> = move a; \
    var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],7u32); } \
    let second: Array<u32> = move b; \
    var c: Array<u32> = filled_u32(len(second),0u32); \
    for k in 0u32 .. len(second) { c[k] = wrapping_add(second[k],11u32); } \
    if c[0u32] != 39u32 { return 1; } \
    if c[1u32] != 46u32 { return 1; } \
    if c[2u32] != 53u32 { return 1; } return 0; }";

const ZIP_DAG: &str = "fn main() -> i32 { \
    let xs: Array<u32> = [1u32,2u32,3u32]; \
    let ys: Array<u32> = [4u32,5u32,6u32]; \
    let zs: Array<u32> = [7u32,8u32,9u32]; \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],ys[i]); } \
    let first: Array<u32> = move a; \
    var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],zs[j]); } \
    if b[0u32] != 35u32 { return 1; } \
    if b[1u32] != 56u32 { return 1; } \
    if b[2u32] != 81u32 { return 1; } return 0; }";

const LARGE_CHAIN: &str = "fn main() -> i32 { \
    let xs: Array<u32> = filled_u32(67108864u32,1u32); \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],3u32); } \
    let first: Array<u32> = move a; \
    var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],7u32); } \
    let second: Array<u32> = move b; \
    var c: Array<u32> = filled_u32(len(second),0u32); \
    for k in 0u32 .. len(second) { c[k] = wrapping_add(second[k],11u32); } return 0; }";

fn checked_program(source: &str) -> (invar::check::CheckedProgram, mir::Program) {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    (checked, program)
}

fn source_for_len(len: usize, zip: bool) -> String {
    let prefix = if zip {
        format!(
            "let xs: Array<u32> = filled_u32({len}u32,1u32); \
             let ys: Array<u32> = filled_u32({}u32,4u32); \
             let zs: Array<u32> = filled_u32({}u32,7u32); \
             var a: Array<u32> = filled_u32(len(xs),0u32); \
             for i in 0u32 .. len(xs) {{ a[i] = wrapping_add(xs[i],ys[i]); }} \
             let first: Array<u32> = move a; \
             var b: Array<u32> = filled_u32(len(first),0u32); \
             for j in 0u32 .. len(first) {{ b[j] = wrapping_mul(first[j],zs[j]); }}",
            len + 1,
            len + 2
        )
    } else {
        format!(
            "let xs: Array<u32> = filled_u32({len}u32,1u32); \
             var a: Array<u32> = filled_u32(len(xs),0u32); \
             for i in 0u32 .. len(xs) {{ a[i] = wrapping_add(xs[i],3u32); }} \
             let first: Array<u32> = move a; \
             var b: Array<u32> = filled_u32(len(first),0u32); \
             for j in 0u32 .. len(first) {{ b[j] = wrapping_mul(first[j],7u32); }} \
             let second: Array<u32> = move b; \
             var c: Array<u32> = filled_u32(len(second),0u32); \
             for k in 0u32 .. len(second) {{ c[k] = wrapping_add(second[k],11u32); }}"
        )
    };
    let output = if zip { "b" } else { "c" };
    let expected = if zip { 35 } else { 39 };
    let checks = if len == 0 {
        String::new()
    } else {
        format!(
            "if {output}[0u32] != {expected}u32 {{ return 1; }} \
             if {output}[{}u32] != {expected}u32 {{ return 1; }}",
            len - 1
        )
    };
    format!("fn main() -> i32 {{ {prefix} {checks} return 0; }}")
}

fn sixteen_stage_source() -> String {
    let mut source = String::from("fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; ");
    for stage in 0..16 {
        let input = if stage == 0 {
            "xs".to_owned()
        } else {
            format!("stable{}", stage - 1)
        };
        source.push_str(&format!(
            "var out{stage}: Array<u32> = filled_u32(len({input}),0u32); \
             for i{stage} in 0u32 .. len({input}) {{ out{stage}[i{stage}] = wrapping_add({input}[i{stage}],1u32); }} "
        ));
        if stage < 15 {
            source.push_str(&format!(
                "let stable{stage}: Array<u32> = move out{stage}; "
            ));
        }
    }
    source.push_str("if out15[0u32] != 17u32 { return 1; } return 0; }");
    source
}

#[test]
fn issue_m3_03_cuda_fused_maps_three_stage_graph_and_cpu_parity() {
    let (checked, program) = checked_program(CHAIN);
    let report = admission::analyze(CHAIN.as_bytes(), &program).unwrap();
    assert_eq!(report.regions().len(), 3);
    assert!(
        report
            .regions()
            .iter()
            .all(|r| r.plan_kind() == PlanKind::FusedMap)
    );
    let explained = report.to_json();
    assert_eq!(
        explained["regions"][0]["execution"]["kind"],
        "fused-map-producer"
    );
    assert_eq!(
        explained["regions"][1]["execution"]["kind"],
        "fused-map-consumer"
    );
    assert_eq!(
        explained["regions"][2]["execution"]["kind"],
        "fused-map-consumer"
    );
    let reference = eval::observe(&checked, 100_000);
    let serial = mir_eval::observe(&program, 100_000);
    assert_eq!(reference.result, Ok(eval::Value::I32(0)));
    assert_eq!(serial.result, reference.result);
    for workers in [1, 2, 4] {
        let run = schedule::run_source(CHAIN.as_bytes(), &program, 100_000, workers).unwrap();
        assert_eq!(run.observation.result, reference.result);
        assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
        assert_eq!(run.executed.len(), 3);
        assert!(
            run.executed
                .iter()
                .all(|step| step.kind == PlanKind::FusedMap)
        );
    }
}

#[test]
fn issue_m3_03_cuda_fused_maps_zip_dag_and_cpu_parity() {
    let (checked, program) = checked_program(ZIP_DAG);
    let report = admission::analyze(ZIP_DAG.as_bytes(), &program).unwrap();
    assert_eq!(report.regions().len(), 2);
    assert!(
        report
            .regions()
            .iter()
            .all(|r| r.plan_kind() == PlanKind::FusedMap)
    );
    let reference = eval::observe(&checked, 100_000);
    let serial = mir_eval::observe(&program, 100_000);
    assert_eq!(reference.result, Ok(eval::Value::I32(0)));
    assert_eq!(serial.result, reference.result);
    for workers in [1, 2, 4] {
        let run = schedule::run_source(ZIP_DAG.as_bytes(), &program, 100_000, workers).unwrap();
        assert_eq!(run.observation.result, reference.result);
        assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
    }
}

#[test]
fn issue_m3_03_cuda_fused_maps_length_corpus_reference_serial_cpu() {
    for len in [0usize, 1, 31, 32, 33, 255, 256, 257, 4093] {
        for zip in [false, true] {
            let source = source_for_len(len, zip);
            if let Some(dir) = std::env::var_os("INVAR_AI_01_M3_CORPUS_DIR") {
                let name = format!("fused-{len}-{}.inv", if zip { "zip" } else { "unary" });
                std::fs::write(std::path::Path::new(&dir).join(name), &source).unwrap();
            }
            let (checked, program) = checked_program(&source);
            let report = admission::analyze(source.as_bytes(), &program).unwrap();
            assert!(
                report
                    .regions()
                    .iter()
                    .all(|r| r.plan_kind() == PlanKind::FusedMap)
            );
            let reference = eval::observe(&checked, 1_000_000);
            let serial = mir_eval::observe(&program, 1_000_000);
            assert_eq!(
                reference.result,
                Ok(eval::Value::I32(0)),
                "len={len} zip={zip}"
            );
            assert_eq!(serial.result, reference.result, "len={len} zip={zip}");
            assert_eq!(serial.cleanup_trace, reference.cleanup_trace);
            for workers in [1, 2, 4] {
                let run =
                    schedule::run_source(source.as_bytes(), &program, 1_000_000, workers).unwrap();
                assert_eq!(
                    run.observation.result, reference.result,
                    "len={len} zip={zip} workers={workers}"
                );
                assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
            }
        }
    }
}

#[test]
fn issue_m3_03_cuda_fused_maps_sixteen_stage_limit() {
    let source = sixteen_stage_source();
    let (checked, program) = checked_program(&source);
    let report = admission::analyze(source.as_bytes(), &program).unwrap();
    assert_eq!(report.regions().len(), 16);
    assert!(
        report
            .regions()
            .iter()
            .all(|r| r.plan_kind() == PlanKind::FusedMap)
    );
    let ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let verified = invar_device::verify(ir, origin, Capabilities::cuda_12_9_sm89()).unwrap();
    assert_eq!(verified.ir().dag.as_ref().unwrap().outputs.len(), 16);
    assert_eq!(
        eval::observe(&checked, 100_000).result,
        Ok(eval::Value::I32(0))
    );
    let serial = mir_eval::observe(&program, 100_000);
    let parallel = schedule::run_source(source.as_bytes(), &program, 100_000, 4).unwrap();
    assert_eq!(parallel.observation.result, serial.result);
    assert_eq!(parallel.observation.cleanup_trace, serial.cleanup_trace);
}

/// Stage 2 zips the stage-0 intermediate, which is not live at the producer head.
const STAGE_REUSE_DAG: &str = "fn main() -> i32 { \
    let xs: Array<u32> = [1u32,2u32,3u32,4u32,5u32]; \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],3u32); } \
    let first: Array<u32> = move a; \
    var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],7u32); } \
    let second: Array<u32> = move b; \
    var c: Array<u32> = filled_u32(len(second),0u32); \
    for k in 0u32 .. len(second) { c[k] = wrapping_add(second[k],first[k]); } \
    if c[1u32] != 40u32 { return 1; } return 0; }";

/// Stage 2 zips `w`, which is declared after the producer loop head.
const LATE_OPERAND_DAG: &str = "fn main() -> i32 { \
    let xs: Array<u32> = [1u32,2u32,3u32,4u32,5u32]; \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],3u32); } \
    let first: Array<u32> = move a; \
    var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],7u32); } \
    let second: Array<u32> = move b; \
    let w: Array<u32> = [9u32,9u32,9u32,9u32,9u32]; \
    var c: Array<u32> = filled_u32(len(second),0u32); \
    for k in 0u32 .. len(second) { c[k] = wrapping_add(second[k],w[k]); } \
    if c[1u32] != 44u32 { return 1; } return 0; }";

/// Every stage zips the same live array `ws` into the running chain.
fn shared_input_chain(stages: usize, distinct: bool) -> String {
    let mut source = String::from("fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; ");
    let others = if distinct { stages } else { 1 };
    for other in 0..others {
        source.push_str(&format!(
            "let ws{other}: Array<u32> = [4u32,{other}u32,6u32]; "
        ));
    }
    let mut input = "xs".to_owned();
    for stage in 0..stages {
        let other = if distinct { stage } else { 0 };
        source.push_str(&format!(
            "var out{stage}: Array<u32> = filled_u32(len({input}),0u32); \
             for i{stage} in 0u32 .. len({input}) {{ out{stage}[i{stage}] = wrapping_add({input}[i{stage}],ws{other}[i{stage}]); }} \
             let stable{stage}: Array<u32> = move out{stage}; "
        ));
        input = format!("stable{stage}");
    }
    source.push_str(&format!(
        "if {input}[0u32] == 0u32 {{ return 1; }} return 0; }}"
    ));
    source
}

#[test]
fn issue_m3_03_cuda_fused_maps_zip_dag_reuses_stage_and_shared_inputs() {
    let (checked, program) = checked_program(STAGE_REUSE_DAG);
    let report = admission::analyze(STAGE_REUSE_DAG.as_bytes(), &program).unwrap();
    assert!(
        report
            .regions()
            .iter()
            .all(|r| r.plan_kind() == PlanKind::FusedMap)
    );
    let ir = device_ir::from_region(&report.regions()[0], 5).unwrap();
    assert_eq!(ir.dag.as_ref().unwrap().inputs.len(), 1);
    let reference = eval::observe(&checked, 100_000);
    assert_eq!(reference.result, Ok(eval::Value::I32(0)));
    for workers in [1, 2, 4] {
        let run =
            schedule::run_source(STAGE_REUSE_DAG.as_bytes(), &program, 100_000, workers).unwrap();
        assert_eq!(run.observation.result, reference.result);
        assert_eq!(run.observation.steps_used, reference.steps_used);
        assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
        // The producer pass publishes every stage; nothing replays as a plain map.
        assert_eq!(run.executed.len(), 3, "workers={workers}");
        assert!(
            run.executed
                .iter()
                .all(|step| step.kind == PlanKind::FusedMap)
        );
    }
    // An operand declared after the producer head ends the chain before it.
    let (_, program) = checked_program(LATE_OPERAND_DAG);
    let report = admission::analyze(LATE_OPERAND_DAG.as_bytes(), &program).unwrap();
    let kinds = report
        .regions()
        .iter()
        .map(|r| r.plan_kind())
        .collect::<Vec<_>>();
    assert_eq!(
        kinds,
        [PlanKind::FusedMap, PlanKind::FusedMap, PlanKind::Map]
    );
    let run = schedule::run_source(LATE_OPERAND_DAG.as_bytes(), &program, 100_000, 2).unwrap();
    assert_eq!(run.observation.result, Ok(eval::Value::I32(0)));
    assert_eq!(run.executed.len(), 3);
    for stages in [15, 16] {
        let source = shared_input_chain(stages, false);
        let (checked, program) = checked_program(&source);
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        let ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
        // One array read by every stage is one device input, not one per stage.
        assert_eq!(ir.dag.as_ref().unwrap().inputs.len(), 2, "stages={stages}");
        let origin = Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let metal = Capabilities::Metal {
            apple_family: 9,
            msl_major: 3,
            msl_minor: 1,
        };
        assert!(invar_device::verify(ir.clone(), origin, metal).is_ok());
        assert!(invar_device::verify(ir, origin, Capabilities::cuda_12_9_sm89()).is_ok());
        let reference = eval::observe(&checked, 100_000);
        let run = schedule::run_source(source.as_bytes(), &program, 100_000, 2).unwrap();
        assert_eq!(run.observation.result, reference.result);
        assert_eq!(run.executed.len(), stages);
    }
}

#[test]
fn issue_m3_03_cuda_fused_maps_metal_verifier_bounds_buffer_slots() {
    // 16 inputs plus 15 outputs plus `n` needs MSL buffer index 31; Apple GPUs stop at 30.
    let source = shared_input_chain(15, true);
    let (_, program) = checked_program(&source);
    let report = admission::analyze(source.as_bytes(), &program).unwrap();
    let ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
    let dag = ir.dag.as_ref().unwrap();
    assert_eq!((dag.inputs.len(), dag.outputs.len()), (16, 15));
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let metal = Capabilities::Metal {
        apple_family: 9,
        msl_major: 3,
        msl_minor: 1,
    };
    assert!(matches!(
        invar_device::verify(ir.clone(), origin, metal),
        Err(VerifyError::Operations)
    ));
    assert!(invar_device::verify(ir, origin, Capabilities::cuda_12_9_sm89()).is_ok());
}

/// A standalone two-input map region, with no neighbouring map to fuse.
const TWO_INPUT_MAP: &str = "fn main() -> i32 { \
    let xs: Array<u32> = [1u32,2u32,3u32]; \
    let zs: Array<u32> = [7u32,8u32,9u32]; \
    var ys: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { ys[i] = wrapping_mul(zs[i],xs[i]); } \
    if ys[2u32] != 27u32 { return 1; } return 0; }";

#[test]
fn issue_m3_03_cuda_fused_maps_two_input_map_region_lowers_to_dag() {
    let (_, program) = checked_program(TWO_INPUT_MAP);
    let report = admission::analyze(TWO_INPUT_MAP.as_bytes(), &program).unwrap();
    assert_eq!(report.regions()[0].plan_kind(), PlanKind::Map);
    let ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
    let dag = ir.dag.as_ref().unwrap();
    assert_eq!((dag.inputs.len(), dag.outputs.len()), (2, 1));
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let metal = Capabilities::Metal {
        apple_family: 9,
        msl_major: 3,
        msl_minor: 1,
    };
    let msl =
        invar_device::emit_msl(&invar_device::verify(ir.clone(), origin, metal).unwrap()).unwrap();
    let cuda = invar_device::emit_cuda(
        &invar_device::verify(ir, origin, Capabilities::cuda_12_9_sm89()).unwrap(),
    )
    .unwrap();
    for source in [msl, cuda] {
        assert!(source.contains("output0[i] = v2;"), "{source}");
        assert!(source.contains("v2 = v0 * v1;"), "{source}");
    }
}

#[test]
fn issue_m3_03_cuda_fused_maps_metal_slot_overflow_is_ineligible_at_planning() {
    let source = shared_input_chain(15, true);
    let (_, program) = checked_program(&source);
    let report = admission::analyze(source.as_bytes(), &program).unwrap();
    let mut ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
    // The Metal planning step refuses the region before verification, naming
    // the slot count; CUDA has no such limit.
    assert_eq!(
        device_ir::plan_metal(&mut ir),
        Err(device_ir::Error::MetalBufferSlots {
            needed: 32,
            limit: invar_device::METAL_MAX_BUFFERS,
        })
    );
    let (_, program) = checked_program(CHAIN);
    let report = admission::analyze(CHAIN.as_bytes(), &program).unwrap();
    let mut ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
    assert_eq!(device_ir::plan_metal(&mut ir), Ok(()));
    assert_eq!(invar_device::metal_buffer_slots(&ir), 5);
}

#[test]
fn issue_m3_03_cuda_fused_maps_graph_plan_lowers_to_shared_ir() {
    let (_, program) = checked_program(CHAIN);
    let report = admission::analyze(CHAIN.as_bytes(), &program).unwrap();
    let ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
    assert_eq!(ir.length, 3);
    let dag = ir.dag.as_ref().unwrap();
    assert_eq!(dag.inputs.len(), 1);
    assert_eq!(dag.outputs.len(), 3);
    assert_eq!(dag.stage_outputs.len(), 3);
    assert!(dag.nodes.iter().all(|node| node.span.start < node.span.end));
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let verified = invar_device::verify(ir, origin, Capabilities::cuda_12_9_sm89()).unwrap();
    let source = invar_device::emit_cuda(&verified).unwrap();
    assert_eq!(source, invar_device::emit_cuda(&verified).unwrap());
    assert_eq!(invar_device::source_hash(&source), 0x0c25_4f0c_5545_28d4);
    assert!(source.contains("for (unsigned int i = base; i < n; i += stride)"));
    assert!(source.contains("output2[i]"));
    eprintln!(
        "chain_source_hash={:016x}\n{source}",
        invar_device::source_hash(&source)
    );
}

#[test]
fn issue_m3_03_cuda_fused_maps_shared_verifier_rejects_bad_dags() {
    let (_, program) = checked_program(ZIP_DAG);
    let report = admission::analyze(ZIP_DAG.as_bytes(), &program).unwrap();
    let base = device_ir::from_region(&report.regions()[0], 3).unwrap();
    let origin = Provenance {
        source_identity: base.source_identity,
        graph_revision: base.graph_revision,
    };
    let dag = base.dag.as_ref().unwrap();
    assert_eq!((dag.inputs.len(), dag.outputs.len()), (3, 2));
    let verified =
        invar_device::verify(base.clone(), origin, Capabilities::cuda_12_9_sm89()).unwrap();
    let source = invar_device::emit_cuda(&verified).unwrap();
    assert!(source.contains("input2[i]"));
    assert_eq!(invar_device::source_hash(&source), 0xdfac_53cf_e29e_923a);
    eprintln!(
        "zip_source_hash={:016x}\n{source}",
        invar_device::source_hash(&source)
    );

    let mut alias = base.clone();
    alias.dag.as_mut().unwrap().outputs[0].id = alias.dag.as_ref().unwrap().inputs[0].id;
    assert_eq!(
        invar_device::verify(alias, origin, Capabilities::cuda_12_9_sm89()).err(),
        Some(VerifyError::Alias)
    );

    let mut layout = base.clone();
    layout.dag.as_mut().unwrap().inputs[0].layout = Layout::Unknown;
    layout.input.layout = Layout::Unknown;
    assert_eq!(
        invar_device::verify(layout, origin, Capabilities::cuda_12_9_sm89()).err(),
        Some(VerifyError::Layout)
    );

    let mut overread = base.clone();
    overread.dag.as_mut().unwrap().inputs[1].elements = 2;
    assert_eq!(
        invar_device::verify(overread, origin, Capabilities::cuda_12_9_sm89()).err(),
        Some(VerifyError::Bounds)
    );

    let mut invalid_node = base.clone();
    invalid_node.dag.as_mut().unwrap().nodes[1].op = DagOp::ZipAdd { left: 0, right: 99 };
    assert_eq!(
        invar_device::verify(invalid_node, origin, Capabilities::cuda_12_9_sm89()).err(),
        Some(VerifyError::Operations)
    );

    let mut incomplete = base.clone();
    incomplete.completion = false;
    assert_eq!(
        invar_device::verify(incomplete, origin, Capabilities::cuda_12_9_sm89()).err(),
        Some(VerifyError::Completion)
    );
}

#[test]
fn issue_m3_03_cuda_fused_maps_large_grid_stride_plan_bounded() {
    let (_, program) = checked_program(LARGE_CHAIN);
    let report = admission::analyze(LARGE_CHAIN.as_bytes(), &program).unwrap();
    let ir = device_ir::from_region(&report.regions()[0], 64 * 1024 * 1024).unwrap();
    assert_eq!((ir.geometry.blocks, ir.geometry.lanes), (1024, 256));
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let verified = invar_device::verify(ir, origin, Capabilities::cuda_12_9_sm89()).unwrap();
    let source = invar_device::emit_cuda(&verified).unwrap();
    assert!(source.contains("i += stride"));
    assert!(source.len() <= invar_device::MAX_SOURCE_BYTES);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn issue_m3_03_gpu_fused_maps_lengths_and_zip_dag() {
    use invar_cuda::{
        Error as CudaError,
        cuda::{CudaDevice, FusedFault},
    };
    use std::{process::Command, time::Instant};

    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let mut device =
        CudaDevice::open().expect("CUDA_UNAVAILABLE: requires CUDA GPU and NVRTC 12.4+");
    let cases = [(CHAIN, false), (ZIP_DAG, true)];
    for (source, zip) in cases {
        let (_, program) = checked_program(source);
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        let region = &report.regions()[0];
        for len in [0usize, 1, 31, 32, 33, 255, 256, 257, 4093] {
            let ir = device_ir::from_region(region, len as u32).unwrap();
            let origin = Provenance {
                source_identity: ir.source_identity,
                graph_revision: ir.graph_revision,
            };
            let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
            let inputs = (0..if zip { 3 } else { 1 })
                .map(|input| {
                    (0..len)
                        .map(|index| (index as u32).wrapping_mul(17).wrapping_add(input * 3 + 1))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let borrowed = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let result = device
                .run_fused_map(&verified, &borrowed, compiler)
                .unwrap();
            let expected = if zip {
                let first = (0..len)
                    .map(|i| inputs[0][i].wrapping_add(inputs[1][i]))
                    .collect::<Vec<_>>();
                let second = (0..len)
                    .map(|i| first[i].wrapping_mul(inputs[2][i]))
                    .collect::<Vec<_>>();
                vec![first, second]
            } else {
                let first = inputs[0]
                    .iter()
                    .map(|n| n.wrapping_add(3))
                    .collect::<Vec<_>>();
                let second = first.iter().map(|n| n.wrapping_mul(7)).collect::<Vec<_>>();
                let third = second
                    .iter()
                    .map(|n| n.wrapping_add(11))
                    .collect::<Vec<_>>();
                vec![first, second, third]
            };
            assert_eq!(result.outputs, expected, "zip={zip} len={len}");
            assert!(result.host_to_host_ms.is_finite());
            assert!(
                result.h2d_ms.is_finite()
                    && result.kernel_ms.is_finite()
                    && result.d2h_ms.is_finite()
            );
            eprintln!(
                "family={} len={} device={} caps={:?} source_hash={:016x} cubin_bytes={} geometry={}x{} transfer_bytes={} compile_ms={} host_to_host_ms={} h2d_ms={} kernel_ms={} d2h_ms={} residency=host-to-host completion=event-confirmed nvrtc_options=--gpu-architecture=sm_{},--std=c++17 nvrtc_log={:?}",
                if zip { "zip-dag" } else { "chain" },
                len,
                device.name(),
                device.capabilities(),
                result.source_hash,
                result.cubin_bytes,
                result.blocks,
                result.lanes,
                result.transfer_bytes,
                result.compile_ms,
                result.host_to_host_ms,
                result.h2d_ms,
                result.kernel_ms,
                result.d2h_ms,
                match device.capabilities() {
                    Capabilities::Cuda { sm, .. } => sm,
                    _ => unreachable!(),
                },
                result.nvrtc_log
            );
        }
    }
    let cold_started = Instant::now();
    let cold = Command::new(env!("CARGO_BIN_EXE_invar-m3-03-cold"))
        .arg(CHAIN)
        .arg(compiler)
        .output()
        .unwrap();
    assert!(
        cold.status.success(),
        "{}",
        String::from_utf8_lossy(&cold.stderr)
    );
    eprintln!(
        "cold_process_wall_ms={} {}",
        cold_started.elapsed().as_secs_f64() * 1000.0,
        String::from_utf8_lossy(&cold.stdout).trim()
    );

    let (_, program) = checked_program(CHAIN);
    let report = admission::analyze(CHAIN.as_bytes(), &program).unwrap();
    let ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
    let input = [1u32, 2, 3];
    let expected_cpu = match &report.regions()[0].execution() {
        invar::plan::Execution::DagProducer(chain) => chain.plan.run_dag(&[&input], 0..3).unwrap(),
        _ => panic!("missing graph-extracted chain"),
    };
    let mut unknown_layout = verified.ir().clone();
    unknown_layout.dag.as_mut().unwrap().inputs[0].layout = Layout::Unknown;
    unknown_layout.input.layout = Layout::Unknown;
    assert_eq!(
        invar_device::verify(unknown_layout, origin, device.capabilities()).err(),
        Some(VerifyError::Layout)
    );
    assert_eq!(
        device
            .run_fused_map(&verified, &[&input], compiler)
            .unwrap()
            .outputs,
        expected_cpu
    );
    eprintln!(
        "unknown_layout=pre_submit_reject cpu_fallback=reference-equal stale_device_use=none"
    );
    for (fault, expected) in [
        (FusedFault::FalseProfitability, CudaError::Unprofitable),
        (FusedFault::TransferOutOfMemory, CudaError::OutOfMemory),
        (FusedFault::LaunchFailure, CudaError::InvalidHandle),
    ] {
        assert_eq!(
            device
                .run_fused_map_with_fault(&verified, &[&input], compiler, fault)
                .err(),
            Some(expected)
        );
        assert_eq!(
            device
                .run_fused_map(&verified, &[&input], compiler)
                .unwrap()
                .outputs,
            expected_cpu
        );
        eprintln!(
            "pre_submit_fault={fault:?} cpu_fallback=reference-equal completion=no-device-output"
        );
    }
    let mut unknown = CudaDevice::open().unwrap();
    assert_eq!(
        unknown
            .run_fused_map_with_fault(
                &verified,
                &[&input],
                compiler,
                FusedFault::PartialCompletion,
            )
            .err(),
        Some(CudaError::Timeout)
    );
    // Unconfirmed work stays quarantined until its stream fence completes;
    // only then may a fresh submission run.
    assert!(unknown.is_poisoned());
    assert_eq!(unknown.quarantine_len(), 1);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while unknown.quarantine_len() > 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "CUDA_PENDING: fence never completed"
        );
        unknown.reap();
        std::thread::yield_now();
    }
    assert!(!unknown.is_poisoned());
    let fresh = unknown
        .run_fused_map(&verified, &[&input], compiler)
        .unwrap();
    assert_eq!(
        fresh.outputs.len(),
        verified.ir().dag.as_ref().unwrap().outputs.len()
    );
    eprintln!(
        "partial_completion=unknown quarantine=reaped-after-fence publication=none cpu_replay=none"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn issue_m3_03_gpu_bandwidth_gate() {
    use invar_cuda::cuda::CudaDevice;

    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let (_, program) = checked_program(LARGE_CHAIN);
    let report = admission::analyze(LARGE_CHAIN.as_bytes(), &program).unwrap();
    let ir = device_ir::from_region(&report.regions()[0], 64 * 1024 * 1024).unwrap();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let mut device =
        CudaDevice::open().expect("CUDA_UNAVAILABLE: requires CUDA GPU and NVRTC 12.4+");
    let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
    let result = device.benchmark_fused_chain(&verified, compiler).unwrap();
    eprintln!("resident_benchmark={result:?}");
    assert!(result.outputs_correct);
    assert!(
        result.bandwidth_fraction >= 0.80,
        "copy bandwidth fraction={}",
        result.bandwidth_fraction
    );
    assert!(
        result.generated_to_handwritten_ratio <= 1.25,
        "handwritten ratio={}",
        result.generated_to_handwritten_ratio
    );
}
