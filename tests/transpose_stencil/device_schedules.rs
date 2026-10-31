//! Issue #2 (M4-02).
#[allow(dead_code)]
#[path = "../support/m402.rs"]
mod support;
use invar::{eval, stencil_source};
use invar_device::{Capabilities, Provenance, stencil::Boundary, stencil_u32, transpose};
const METAL: Capabilities = Capabilities::Metal {
    apple_family: 9,
    msl_major: 3,
    msl_minor: 1,
};
const CUDA: Capabilities = Capabilities::Cuda {
    major: 12,
    minor: 9,
    sm: 120,
};
#[test]
fn schedules_device_ir_emitters_and_phases() {
    for caps in [CUDA, METAL] {
        for &shape in support::SHAPES.iter().chain(&[[4096, 4096]]) {
            let plan = support::transpose_plan(shape, caps);
            let text = if caps == METAL {
                transpose::emit_msl(&plan)
            } else {
                transpose::emit_cuda(&plan)
            }
            .unwrap();
            support::archive(
                &format!(
                    "transpose-{}-{}-{}",
                    if caps == METAL { "metal" } else { "cuda" },
                    shape[0],
                    shape[1]
                ),
                &format!(
                    "{:#?}\nmap: input[offset+r*cols+c] -> output[offset+c*rows+r]\nbarriers: all 256 lanes, load/store/reuse\n",
                    plan.ir()
                ),
                &text,
            );
            assert!(text.contains("16x17"));
            if !shape.contains(&0) {
                assert_eq!(
                    text.matches(if caps == METAL {
                        "threadgroup_barrier("
                    } else {
                        "__syncthreads("
                    })
                    .count(),
                    2
                );
            }
            assert!(
                if caps == METAL {
                    transpose::emit_cuda(&plan)
                } else {
                    transpose::emit_msl(&plan)
                }
                .is_err()
            );
        }
        for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
            for &shape in support::SHAPES {
                for steps in [0, 1, 2, 3, 4] {
                    let plan = support::stencil_plan(shape, steps, boundary, caps);
                    assert_eq!(plan.final_buffer(), (steps % 2) as usize);
                    for step in 0..steps {
                        assert_eq!(
                            plan.phase(step).unwrap(),
                            [(step % 2) as usize, 1 - (step % 2) as usize]
                        );
                    }
                    assert!(plan.phase(steps).is_err());
                    let text = if caps == METAL {
                        stencil_u32::emit_msl(&plan)
                    } else {
                        stencil_u32::emit_cuda(&plan)
                    }
                    .unwrap();
                    support::archive(
                        &format!(
                            "stencil-{}-{}-{}-{}-{steps}",
                            if caps == METAL { "metal" } else { "cuda" },
                            boundary.name(),
                            shape[0],
                            shape[1]
                        ),
                        &format!(
                            "{:#?}\nmap: offset+r*cols+c, radius=1 center,north,south,west,east\nmask: constant_zero loads none outside; clamp loads nearest boundary cell\nphases: separate completed time slices, final={}\nbarriers: all 256 lanes including edge masks\n",
                            plan.ir(),
                            plan.final_buffer()
                        ),
                        &text,
                    );
                    assert!(text.contains("tile=18x19"));
                    if !shape.contains(&0) {
                        assert_eq!(
                            text.matches(if caps == METAL {
                                "threadgroup_barrier("
                            } else {
                                "__syncthreads("
                            })
                            .count(),
                            2
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn schedules_mutations_refuse() {
    let plan = support::stencil_plan([31, 33], 3, Boundary::Clamp, CUDA);
    let ir = plan.ir();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    for i in 0..15 {
        let mut bad = ir.clone();
        match i {
            0 => bad.input.id = bad.output.id,
            1 => bad.input.strides = [1, 1],
            2 => bad.input.offset = u64::MAX,
            3 => bad.input.elements = u64::MAX,
            4 => bad.output.shape = [33, 31],
            5 => bad.numeric_policy = "relaxed".into(),
            6 => bad.boundary_version += 1,
            7 => bad.steps = 65,
            8 => bad.source_identity += 1,
            9 => bad.graph_revision += 1,
            10 => bad.function_contract = [0; 32],
            11 => bad.span.end = bad.span.start,
            12 => bad.output.offset = 1,
            13 => bad.output.alignment = 3,
            14 => {
                bad.dimensions = [u32::MAX, u32::MAX];
                bad.steps = 64;
            }
            _ => unreachable!(),
        }
        let refusal = stencil_u32::verify(bad, CUDA, origin).unwrap_err();
        println!("mutation={i} target=cuda refusal={refusal:?}");
    }
    assert!(plan.validate_launch([16, 15, 1], [1, 1, 1]).is_err());
    assert!(plan.validate_launch([16, 16, 1], [0, 1, 1]).is_err());
    assert!(plan.validate_launch([16, 16, 1], [65536, 1, 1]).is_err());
    assert_eq!(
        plan.reads([0, 0]).unwrap(),
        [Some(0), Some(0), Some(33), Some(0), Some(1)]
    );
    assert!(plan.reads([31, 0]).is_err());
    let zero = support::stencil_plan([1, 1], 1, Boundary::ConstantZero, CUDA);
    assert_eq!(
        zero.reads([0, 0]).unwrap(),
        [Some(0), None, None, None, None]
    );
    for minor in [4, 7] {
        assert!(
            transpose::target_options(Capabilities::Cuda {
                major: 12,
                minor,
                sm: 120
            })
            .is_err()
        );
    }
    assert!(
        transpose::target_options(Capabilities::Cuda {
            major: 12,
            minor: 8,
            sm: 120
        })
        .is_ok()
    );
}
#[test]
fn schedules_wrapping_source_oracle_and_rejection() {
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        for shape in [[1, 1], [1, 5], [5, 1], [3, 5]] {
            let words = support::input(shape);
            for steps in [0, 1, 2, 3, 4] {
                let expected = support::stencil_oracle(shape, steps, boundary, &words);
                support::archive_oracle(shape, steps, boundary, &words, &expected);
                let array = words
                    .iter()
                    .map(|n| format!("{n}u32"))
                    .collect::<Vec<_>>()
                    .join(",");
                let assertions = expected
                    .iter()
                    .enumerate()
                    .map(|(i, n)| format!("if cur[{i}u32]!={n}u32{{return 1;}}"))
                    .collect::<String>();
                let source=support::stencil_source(boundary).replace("fn main()->i32{return 7;}",&format!("fn main()->i32{{var cur:Array<u32> =[{array}];for t in 0u32..{steps}u32{{let next=step(cur,{}u32,{}u32);cur=move next;}}{assertions}return 7;}}",shape[0],shape[1]));
                let checked = support::checked(&source);
                let c = stencil_source::Contract::from_checked(&checked, "step").unwrap();
                assert!(c.device_u32(shape, steps, CUDA).is_ok());
                assert_eq!(
                    eval::observe(&checked, 1_000_000).result,
                    Ok(eval::Value::I32(7))
                );
            }
        }
        let source = support::stencil_source(boundary);
        for bad in [
            source.replace(
                "wrapping_add(wrapping_add(wrapping_add(wrapping_add(xs[r*cols+c],n),s),w),e)",
                "(((xs[r*cols+c]+n)+s)+w)+e",
            ),
            source.replace("(r-1u32)*cols+c", "r*cols+c"),
            source.replace(
                "var out=filled_u32(rows*cols,0u32);",
                "var out=filled_u32(rows*cols,1u32);",
            ),
        ] {
            assert!(
                stencil_source::Contract::from_checked(&support::checked(&bad), "step").is_err()
            );
        }
    }
}

#[test]
fn schedules_u32_source_fallback_is_exact_and_explicit() {
    use invar_cuda::Runtime;
    let source=support::stencil_source(Boundary::Clamp).replace("fn main()->i32{return 7;}","fn main()->i32{let a:Array<u32> = [4294967295u32];let b=step(a,1u32,1u32);if b[0u32]!=4294967291u32{return 1;}return 7;}");
    let checked = support::checked(&source);
    let reference = eval::observe(&checked, 100_000);
    let mut runtime = Runtime::new(stencil_source::host::CpuDevice, 2);
    let got =
        stencil_source::host::run_source(source.as_bytes(), 100_000, &mut runtime, CUDA).unwrap();
    assert_eq!(got.observation.result, reference.result);
    assert_eq!(got.observation.steps_used, reference.steps_used);
    assert_eq!(got.observation.cleanup_trace, reference.cleanup_trace);
    assert_eq!(got.completed, 0);
    assert_eq!(
        got.refusals,
        ["stencil_source_publication_view_unavailable"]
    );
}

#[test]
fn cap_2_28_verifier_and_overflow_boundaries() {
    for caps in [CUDA, METAL] {
        for shape in [[16384, 16384], [1, 1 << 28], [1 << 28, 1]] {
            let p = support::transpose_plan(shape, caps);
            let mut ir = p.ir().clone();
            let origin = Provenance {
                source_identity: ir.source_identity,
                graph_revision: ir.graph_revision,
            };
            // Physical padding is independent from the logical cap; last byte
            // offset exceeds 2^30 and is emitted with 64-bit arithmetic.
            for b in [&mut ir.input, &mut ir.output] {
                b.offset = 17;
                b.elements += 34;
            }
            let p = transpose::verify(ir, caps, origin).unwrap();
            assert_eq!((p.ir().input.offset + (1 << 28) - 1) * 4, (1 << 30) + 64);
            let text = if caps == METAL {
                transpose::emit_msl(&p)
            } else {
                transpose::emit_cuda(&p)
            }
            .unwrap();
            assert!(text.contains(if caps == METAL {
                "ulong"
            } else {
                "unsigned long long"
            }));
            for boundary in [Boundary::Clamp, Boundary::ConstantZero] {
                let s = support::stencil_plan(shape, 64, boundary, caps);
                let [r, c] = shape.map(|n| n - 1);
                assert_eq!(s.reads([r, c]).unwrap()[0], Some((1 << 28) - 1));
                assert_eq!(s.phase(63).unwrap(), [1, 0]);
                let mut ir = s.ir().clone();
                ir.steps = 65;
                assert_eq!(
                    stencil_u32::verify(ir, caps, origin).unwrap_err(),
                    stencil_u32::Error::Steps
                );
                let mut ir = s.ir().clone();
                ir.dimensions = [1, (1 << 28) + 1];
                ir.steps = 1;
                for b in [&mut ir.input, &mut ir.output] {
                    b.shape = ir.dimensions;
                    b.strides = [u64::from(ir.dimensions[1]), 1];
                    b.elements = (1 << 28) + 1;
                }
                assert_eq!(
                    stencil_u32::verify(ir, caps, origin).unwrap_err(),
                    stencil_u32::Error::Physical(transpose::Error::Bounds)
                );
            }
        }
        let p = support::transpose_plan([1, 1], caps);
        let mut ir = p.ir().clone();
        ir.dimensions = [1, (1 << 28) + 1];
        ir.input.shape = ir.dimensions;
        ir.input.strides = [u64::from(ir.dimensions[1]), 1];
        ir.input.elements = (1 << 28) + 1;
        ir.output.shape = [ir.dimensions[1], 1];
        ir.output.elements = (1 << 28) + 1;
        let origin = Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        assert_eq!(
            transpose::verify(ir, caps, origin).unwrap_err(),
            transpose::Error::Bounds
        );
    }
}
