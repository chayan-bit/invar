#![forbid(unsafe_code)]
use invar::{check, eval, front, mir, mir_eval, tensor};
#[path = "../issue_m4_01_device/ops.rs"]
mod ops;

#[test]
fn issue_m4_01_numeric_layout_contracts_v2_shared_and_adversarial() {
    use invar_device::{Capabilities, v2};
    let artifact = tensor::emit_cuda(
        MAT.as_bytes(),
        "mat",
        [3, 5, 3],
        [15, 15],
        Capabilities::cuda_12_9_sm89(),
    )
    .unwrap();
    let shared = artifact.plan().device_ir();
    assert_eq!(shared.ir().version, 2);
    assert!(matches!(
        shared.ir().kernel,
        v2::Kernel::Contraction {
            dimensions: [3, 5, 3]
        }
    ));
    assert_eq!(shared.access(2, &[2, 2], 4).unwrap(), 32..36);
    assert_eq!(shared.access(2, &[2, 2], 8), Err(v2::Error::AccessWidth));
    let input = invar_device::Buffer {
        id: 0,
        elements: 3,
        address: invar_device::AddressSpace::Device,
        layout: invar_device::Layout::ContiguousU32,
    };
    let legacy = invar_device::Ir {
        version: 1,
        span: invar_device::Span { start: 0, end: 1 },
        source_identity: 1,
        graph_revision: 1,
        input,
        output: invar_device::Buffer { id: 1, ..input },
        length: 3,
        geometry: invar_device::Geometry {
            blocks: 1,
            lanes: 32,
        },
        lane_mask: invar_device::LaneMask::IndexLtLength,
        barrier: invar_device::Barrier::None,
        requires_barrier: false,
        transfer: invar_device::Transfer::Owned,
        numeric: invar_device::Numeric::WrappingU32,
        completion: true,
        ops: vec![invar_device::Op::WrappingAdd(1)],
        dag: None,
        collective: None,
    };
    let origin = invar_device::Provenance {
        source_identity: 1,
        graph_revision: 1,
    };
    let normalized =
        invar_device::verify(legacy.clone(), origin, Capabilities::cuda_12_9_sm89()).unwrap();
    assert_eq!(normalized.device_ir().ir().version, 2);
    assert_eq!(
        normalized.device_ir().ir().buffers[0].element,
        v2::ElemType::U32
    );
    assert_eq!(normalized.device_ir().access(0, &[2], 4).unwrap(), 8..12);
    let mut bad = legacy;
    bad.version = 99;
    assert!(matches!(
        invar_device::verify(bad, origin, Capabilities::cuda_12_9_sm89()),
        Err(invar_device::Error::Version)
    ));
    for mutation in 0..8 {
        let mut ir = shared.ir().clone();
        let origin = ir.origin;
        match mutation {
            0 => ir.version = 1,
            1 => ir.numeric.version = 2,
            2 => ir.layout_version = 2,
            3 => ir.buffers[2].allocation_bytes = 4,
            4 => ir.buffers[2].alignment = 2,
            5 => ir.buffers[2].id = ir.buffers[0].id,
            6 => {
                ir.buffers[2].layout = v2::Layout::Strided {
                    shape: vec![3, 3],
                    strides_in_elements: vec![1, 1],
                    offset: 0,
                }
            }
            _ => {
                ir.buffers[2].layout = v2::Layout::Strided {
                    shape: vec![3, 3],
                    strides_in_elements: vec![u64::MAX, 1],
                    offset: 0,
                }
            }
        }
        assert!(
            v2::verify(ir, origin, shared.capabilities()).is_err(),
            "mutant {mutation}"
        );
    }
    for p in ops::programs() {
        let plan = ops::plan(p.clone(), Capabilities::cuda_12_9_sm89());
        let inputs = ops::packed(&p);
        let serial = invar::f32_source::cpu(&plan, &inputs, 1).unwrap();
        for workers in [2, 4] {
            ops::same(
                &invar::f32_source::cpu(&plan, &inputs, workers).unwrap(),
                &serial,
                p.output_type().unwrap() == invar_device::f32_ops::Scalar::F32,
            );
        }
        assert!(
            invar_device::f32_ops::emit(&plan)
                .unwrap()
                .contains("device-ir-version=2")
        );
    }
}

#[test]
fn issue_m4_01_numeric_layout_contracts_rank1_source_cpu_native_parity() {
    for source in ops::sources() {
        let expected = ops::reference(&source);
        assert_eq!(expected.result, Ok(eval::Value::I32(7)), "{source}");
        let program = ops::mir_program(&source);
        let region =
            invar::f32_source::Region::from_checked(&ops::checked(&source), "kernel").unwrap();
        let _ = region
            .bind(3, invar_device::Capabilities::cuda_12_9_sm89())
            .unwrap();
        for workers in [1, 2, 4] {
            let run =
                invar::f32_source::run_cpu_source(source.as_bytes(), 1_000_000, workers).unwrap();
            ops::parity(&run, &expected);
            // Empty loops may have no reached branch to replace.
            if !source.contains("filled_f32(0u32") {
                assert!(run.completed > 0, "{source}");
            }
            let schedule =
                invar::schedule::run_source(source.as_bytes(), &program, 1_000_000, workers)
                    .unwrap();
            assert_eq!(schedule.observation.result, expected.result);
            assert_eq!(schedule.observation.steps_used, expected.steps_used);
            if workers == 4 && source.contains("filled_f32(65u32") && !source.contains("var sum=") {
                assert_eq!(schedule.workers_used, 4);
                assert_eq!(schedule.joined, 4);
                assert!(!schedule.executed.is_empty());
            }
        }
        #[cfg(feature = "native")]
        {
            let (status, doc) = invar::native::run_program(&program, 1_000_000).unwrap();
            assert_eq!(status, 0, "{doc}: {source}");
            assert_eq!(doc["result"]["value"], 7);
            assert_eq!(doc["steps_used"], expected.steps_used);
        }
    }
}

#[test]
fn issue_m4_01_numeric_layout_contracts_rank1_fuel_traps_and_mutation_order() {
    let source = &ops::sources()[14];
    let checked = ops::checked(source);
    let complete = eval::observe(&checked, 1_000_000);
    for fuel in 0..=complete.steps_used + 1 {
        let reference = eval::observe(&checked, fuel);
        ops::parity(
            &invar::f32_source::run_cpu_source(source.as_bytes(), fuel, 2).unwrap(),
            &reference,
        );
    }
    let source = "fn kernel(a:read Array<f32>,n:u32)->Array<i32>{var out=filled_i32(n,0);for i in 0u32..n{out[i]=i32_from_f32(a[i]);}return out;} fn main()->i32 {let a:Array<f32> = [1.0,2147483648.0];let out=kernel(a,2u32);return 7;}";
    let expected = ops::reference(source);
    assert!(matches!(expected.result, Err(eval::EvalError::Trap { .. })));
    #[cfg(feature = "native")]
    {
        let (status, doc) =
            invar::native::run_program(&ops::mir_program(source), 1_000_000).unwrap();
        assert_eq!(status, 3, "{doc}");
        assert_eq!(doc["trap"]["category"], "INTEGER_OVERFLOW");
        assert_eq!(doc["steps_used"], expected.steps_used);
    }

    ops::parity(
        &invar::f32_source::run_cpu_source(source.as_bytes(), 1_000_000, 4).unwrap(),
        &expected,
    );
    use invar_device::f32_ops::{Kind, Node, Program, Scalar};
    let p = Program {
        length: 3,
        inputs: vec![Scalar::F32],
        nodes: vec![Node::Input(0)],
        kind: Kind::Reduce { initial: 0 },
    };
    let plan = ops::plan(p, invar_device::Capabilities::cuda_12_9_sm89());
    let input = [0x4b800000, 0x3f800000, 0xcb800000];
    assert_eq!(invar::f32_source::cpu(&plan, &input, 4).unwrap(), [0]);
    let reordered = [input[0], input[2], input[1]];
    assert_eq!(
        invar::f32_source::cpu(&plan, &reordered, 4).unwrap(),
        [0x3f800000]
    );
}

#[test]
fn issue_m4_01_numeric_layout_contracts_quantity_f32_erasure() {
    let typed = "fn kernel(a:Quantity<f32,m>,b:Quantity<f32,m>)->Quantity<f32,m>{return a+b;} fn main()->i32 {let a:Quantity<f32,m> = quantity::<m>(1.0);let b:Quantity<f32,m> = quantity::<m>(2.0);if kernel(a,b)==quantity::<m>(3.0){return 7;}return 0;}";
    let plain = typed
        .replace(
            "Quantity<f32,m>",
            &format!("f32{}", " ".repeat("Quantity<f32,m>".len() - 3)),
        )
        .replace(
            "quantity::<m>(",
            &format!("({}", " ".repeat("quantity::<m>(".len() - 1)),
        );
    assert_eq!(typed.len(), plain.len());
    let a = ops::mir_program(typed);
    let b = ops::mir_program(&plain);
    for (a, b) in a.functions.iter().zip(&b.functions) {
        assert_eq!(format!("{:?}", a.blocks), format!("{:?}", b.blocks));
    }
    for caps in [
        invar_device::Capabilities::cuda_12_9_sm89(),
        invar_device::Capabilities::Metal {
            apple_family: 9,
            msl_major: 3,
            msl_minor: 1,
        },
    ] {
        let emit = |source: &str| {
            invar_device::f32_ops::emit(
                &invar::f32_source::Region::from_checked(&ops::checked(source), "kernel")
                    .unwrap()
                    .bind(1, caps)
                    .unwrap(),
            )
            .unwrap()
            .lines()
            .collect::<Vec<_>>()
            .join("\n")
        };
        assert_eq!(emit(typed), emit(&plain));
    }
    #[cfg(feature = "native")]
    assert_eq!(
        invar::native::emit_mir(&a).unwrap().ir,
        invar::native::emit_mir(&b).unwrap().ir
    );
    assert_eq!(ops::reference(typed).result, Ok(eval::Value::I32(7)));
    for source in [
        typed.replace("Quantity<f32,m>", "Quantity<f32,s>"),
        typed.replace("quantity::<m>(2.0)", "quantity::<s>(2.0)"),
    ] {
        // Replacing every annotation with s while constructors retain m, or
        // changing one constructor, must still fail before erasure.
        assert!(check::check(front::parse(source.as_bytes()).unwrap(), true).is_err());
    }
}
const MAT: &str = "fn mat(a: read Array<f32>, b: read Array<f32>, rows: u32, inner: u32, cols: u32) -> Array<f32> { var c = filled_f32(rows * cols,0.0); for i in 0u32..rows { for j in 0u32..cols { var sum: f32 = 0.0; for k in 0u32..inner { sum = sum + a[i * inner + k] * b[k * cols + j]; } c[i * cols + j] = sum; } } return c; }";
fn checked(source: &str) -> check::CheckedProgram {
    let p = check::check(
        front::parse(source.as_bytes()).unwrap(),
        source.contains("fn main"),
    )
    .unwrap();
    invar::ownership::check(&p).unwrap();
    p
}
#[test]
fn issue_m4_01_numeric_layout_contracts_source_rank2_and_typed_extraction() {
    for (rows, inner, cols) in [(0, 3, 2), (2, 0, 3), (1, 1, 1), (3, 5, 3)] {
        let source = format!(
            "{MAT} fn main() -> i32 {{ let a = filled_f32({rows}u32*{inner}u32,1.0); let b = filled_f32({inner}u32*{cols}u32,2.0); let c=mat(a,b,{rows}u32,{inner}u32,{cols}u32); for index in 0u32..len(c) {{ if c[index] != f32_from_u32({inner}u32*2u32) {{ return 0; }} }} return 7; }}"
        );
        let checked = checked(&source);
        let region = tensor::Contraction::from_checked(&checked, "mat").unwrap();
        let contract = region.contract();
        assert_eq!(contract.inputs(), &["a".to_owned(), "b".to_owned()]);
        assert_eq!(contract.output(), "c");
        assert_eq!(
            contract.dimensions(),
            &["rows".to_owned(), "inner".to_owned(), "cols".to_owned()]
        );
        assert_eq!(contract.numeric_policy().id(), "strict-f32-v1");
        assert_eq!(contract.layout_version(), 1);
        let program = region.compiled_mir().unwrap();
        assert_eq!(program.tensor_contracts.len(), 1);
        let reference = eval::observe(&checked, 10000);
        let actual = mir_eval::observe(&program, 10000);
        assert_eq!(reference.result, Ok(eval::Value::I32(7)));
        assert_eq!(actual.result, reference.result);
        assert_eq!(actual.steps_used, reference.steps_used);
        assert_eq!(actual.cleanup_trace, reference.cleanup_trace);
        #[cfg(feature = "native")]
        {
            let (status, doc) = invar::native::run_program(&program, 10000).unwrap();
            assert_eq!(status, 0, "{doc}");
            assert_eq!(doc["result"]["value"], 7);
            assert_eq!(doc["steps_used"], reference.steps_used);
        }
    }
}
#[test]
fn issue_m4_01_numeric_layout_contracts_unknown_shapes_keep_cpu_execution() {
    for changed in [
        MAT.replace(
            "sum + a[i * inner + k] * b[k * cols + j]",
            "fma(a[i * inner + k],b[k * cols + j],sum)",
        ),
        MAT.replace("a[i * inner + k]", "a[k * rows + i]"),
        MAT.replace("var sum: f32 = 0.0", "var sum: f32 = -0.0"),
        MAT.replace("c[i * cols + j]", "c[j * rows + i]"),
    ] {
        let source = format!(
            "{changed} fn main() -> i32 {{ let a = filled_f32(1u32,1.0);let b=filled_f32(1u32,2.0);let c=mat(a,b,1u32,1u32,1u32); if c[0u32]==2.0 {{return 7;}}return 0;}}"
        );
        let checked = checked(&source);
        assert!(matches!(
            tensor::Contraction::from_checked(&checked, "mat"),
            Err(tensor::Error::Unrecognized)
        ));
        assert_eq!(
            eval::observe(&checked, 1000).result,
            Ok(eval::Value::I32(7))
        );
        let program = mir::lower(&checked).unwrap();
        assert!(program.tensor_contracts.is_empty());
        assert_eq!(
            mir_eval::observe(&program, 1000).result,
            Ok(eval::Value::I32(7))
        );
    }
}
#[test]
fn issue_m4_01_numeric_layout_contracts_mutated_mir_cannot_keep_source_contract() {
    let checked = checked(MAT);
    let original = mir::lower(&checked).unwrap();
    assert_eq!(original.tensor_contracts.len(), 1);
    let mut changed = original.clone();
    let mat = changed
        .functions
        .iter_mut()
        .find(|f| f.name == "mat")
        .unwrap();
    let multiplication = mat
        .blocks
        .iter_mut()
        .flat_map(|b| &mut b.instructions)
        .find(|i| {
            matches!(
                i.kind,
                mir::Operation::Binary {
                    op: front::BinaryOp::Mul,
                    ..
                }
            ) && i
                .output
                .as_ref()
                .is_some_and(|(_, ty)| *ty == front::Type::F32)
        })
        .unwrap();
    let mir::Operation::Binary { op, .. } = &mut multiplication.kind else {
        unreachable!()
    };
    *op = front::BinaryOp::Add;
    assert_eq!(mir::verify(&changed).unwrap_err().code, "MIR_INVALID");
    changed.tensor_contracts.clear();
    mir::verify(&changed).unwrap(); // A lawful fallback has no contraction annotation.
    let mut changed = original.clone();
    changed.layout_version = 2;
    assert_eq!(mir::verify(&changed).unwrap_err().code, "MIR_INVALID");
    let mut changed = original.clone();
    changed
        .tensor_contracts
        .push(original.tensor_contracts[0].clone());
    assert_eq!(mir::verify(&changed).unwrap_err().code, "MIR_INVALID");
}

#[test]
fn issue_m4_01_numeric_layout_contracts_rebuild_identity_is_stable() {
    let first = tensor::Contraction::from_checked(&checked(MAT), "mat").unwrap();
    for _ in 0..16 {
        let next = tensor::Contraction::from_checked(&checked(MAT), "mat").unwrap();
        assert_eq!(
            next.contract().function_contract(),
            first.contract().function_contract()
        );
        assert_eq!(next.source_identity(), first.source_identity());
    }
}

#[test]
fn issue_m4_01_numeric_layout_contracts_live_lengths_and_physical_maps() {
    let region = tensor::Contraction::from_checked(&checked(MAT), "mat").unwrap();
    for (dims, lengths) in [
        ([0, 3, 2], [0, 6]),
        ([2, 0, 3], [0, 0]),
        ([1, 1, 1], [1, 1]),
        ([3, 5, 3], [15, 15]),
    ] {
        let bound = region.bind_layouts(dims, lengths).unwrap();
        assert_eq!(bound.dimensions(), dims);
        let [a, b, c] = bound.checked().unwrap();
        assert_eq!(a.shape(), &[u64::from(dims[0]), u64::from(dims[1])]);
        assert_eq!(b.shape(), &[u64::from(dims[1]), u64::from(dims[2])]);
        assert_eq!(c.shape(), &[u64::from(dims[0]), u64::from(dims[2])]);
        for i in 0..dims[0] {
            for j in 0..dims[2] {
                let index = u64::from(i) * u64::from(dims[2]) + u64::from(j);
                assert_eq!(
                    c.access(&[u64::from(i), u64::from(j)], 4).unwrap(),
                    index * 4..index * 4 + 4
                );
            }
        }
    }
    assert!(region.bind_layouts([3, 5, 3], [14, 15]).is_err());
    assert!(region.bind_layouts([3, 5, 3], [15, 14]).is_err());
    assert!(region.bind_layouts([1001, 1, 1000], [1001, 1000]).is_ok());
    let edge = 1 << 12; // edge^2 is the 2^24 per-array language cap.
    assert!(
        region
            .bind_layouts([edge, 1, edge], [edge as u64, edge as u64])
            .is_ok()
    );
    assert!(
        region
            .bind_layouts([edge + 1, 1, edge], [(edge + 1) as u64, edge as u64])
            .is_err()
    );
    assert!(region.bind_layouts([u32::MAX, 0, 0], [0, 0]).is_err());
}

#[test]
fn issue_m4_01_numeric_layout_contracts_emission_binds_actual_source_and_options() {
    let caps = invar_device::Capabilities::cuda_12_9_sm89();
    let emit = |source: &str, lengths| {
        tensor::emit_cuda(source.as_bytes(), "mat", [3, 5, 3], lengths, caps).unwrap()
    };
    let artifact = emit(MAT, [15, 15]);
    assert_eq!(
        artifact.plan().ir().function_contract,
        artifact.contraction().contract().function_contract()
    );
    assert_eq!(
        artifact.plan().ir().source_identity,
        artifact.contraction().source_identity()
    );
    assert_eq!(
        artifact.plan().ir().span.start,
        artifact.contraction().contract().span().start
    );
    assert_eq!(artifact.layouts().dimensions(), [3, 5, 3]);
    assert!(artifact.source().contains("__fmul_rn"));
    assert!(artifact.source().contains("__fadd_rn"));
    for option in artifact.plan().nvrtc_options() {
        assert!(artifact.source().contains(option));
    }
    for _ in 0..4 {
        assert!(artifact.matches_hint(emit(MAT, [15, 15]).key()));
    }
    assert!(!artifact.matches_hint(emit(&format!(" {MAT}"), [15, 15]).key()));
    assert!(!artifact.matches_hint(emit(MAT, [16, 15]).key()));
    let different = tensor::emit_cuda(
        MAT.as_bytes(),
        "mat",
        [3, 5, 3],
        [15, 15],
        invar_device::Capabilities::Cuda {
            major: 12,
            minor: 4,
            sm: 86,
        },
    )
    .unwrap();
    assert!(!artifact.matches_hint(different.key()));
    assert!(matches!(
        tensor::emit_cuda(MAT.as_bytes(), "mat", [3, 5, 3], [14, 15], caps),
        Err(tensor::Error::Layout(_))
    ));
    assert!(matches!(
        tensor::emit_cuda(
            MAT.as_bytes(),
            "mat",
            [3, 5, 3],
            [15, 15],
            invar_device::Capabilities::Metal {
                apple_family: 8,
                msl_major: 3,
                msl_minor: 1
            }
        ),
        Err(tensor::Error::Device(
            invar_device::floating::Error::Capabilities
        ))
    ));
    assert!(matches!(
        tensor::emit_cuda(
            MAT.replace(
                "sum + a[i * inner + k] * b[k * cols + j]",
                "fma(a[i * inner + k],b[k * cols + j],sum)"
            )
            .as_bytes(),
            "mat",
            [3, 5, 3],
            [15, 15],
            caps
        ),
        Err(tensor::Error::Unrecognized)
    ));
}

#[test]
fn issue_m4_01_numeric_layout_contracts_actual_emitted_cpu_matches_numeric_oracle() {
    use invar::numeric_contract::F32Bits;
    use std::{fs, process::Command};
    let caps = invar_device::Capabilities::cuda_12_9_sm89();
    for dims in [[0, 3, 2], [2, 0, 3], [1, 1, 1], [3, 5, 3]] {
        let [rows, inner, cols] = dims.map(|n| n as usize);
        let artifact = tensor::emit_cuda(
            MAT.as_bytes(),
            "mat",
            dims,
            [(rows * inner) as u64, (inner * cols) as u64],
            caps,
        )
        .unwrap();
        let patterns = [1, 0x3f800001, 0x3f7fffff, 0xbf800000, 0x00800001];
        let a = (0..rows * inner)
            .map(|i| patterns[i % patterns.len()])
            .collect::<Vec<_>>();
        let b = (0..inner * cols)
            .map(|i| patterns[(i + 2) % patterns.len()])
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = F32Bits::POS_ZERO;
                for k in 0..inner {
                    sum = sum.add(
                        F32Bits::from_bits(a[i * inner + k])
                            .mul(F32Bits::from_bits(b[k * cols + j])),
                    );
                }
                expected.push(sum.to_bits());
            }
        }
        let list = |values: &[u32]| {
            values
                .iter()
                .map(|n| format!("{n}u"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let source = format!(
            "{}\n#include <cstdio>\nint main() {{ unsigned ab[{}]={{ {} }},bb[{}]={{ {} }}; float a[{}],b[{}],c[{}]; std::memcpy(a,ab,sizeof(ab)); std::memcpy(b,bb,sizeof(bb)); int status=invar_f32_contract_v1(a,b,c); if(status) return status; for(unsigned i=0;i<{};i++) {{ unsigned bits;std::memcpy(&bits,c+i,4);std::printf(\"%u\\n\",bits); }} return 0; }}",
            invar_device::floating::emit_cpu(artifact.plan()).unwrap(),
            a.len().max(1),
            list(&a),
            b.len().max(1),
            list(&b),
            a.len().max(1),
            b.len().max(1),
            expected.len().max(1),
            expected.len()
        );
        let dir = std::env::temp_dir().join(format!(
            "invar-source-tensor-{}-{rows}-{inner}-{cols}",
            std::process::id()
        ));
        fs::create_dir(&dir).unwrap();
        fs::write(dir.join("kernel.cpp"), source).unwrap();
        let built = Command::new(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()))
            .args(artifact.plan().cpu_options())
            .args(["-O2", "-fsanitize=undefined", "-fno-sanitize-recover=all"])
            .arg(dir.join("kernel.cpp"))
            .arg("-o")
            .arg(dir.join("run"))
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let run = Command::new(dir.join("run")).output().unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let actual = String::from_utf8(run.stdout)
            .unwrap()
            .lines()
            .map(|s| s.parse::<u32>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{dims:?}");
        fs::remove_dir_all(dir).unwrap();
    }
}
