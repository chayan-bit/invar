#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "../test-support/child_deadline.rs"]
mod child_deadline;

// Proposed shared allocation: pub mod contraction_source;. Until allocated,
// compile the owned module against the actual existing library boundaries.
#[allow(dead_code)]
#[path = "../src/contraction_source.rs"]
mod contraction_source;
use contraction_source::{Contract, Emission, Error};
use invar::{
    check, eval, front, general_net, hex, layout_contract, mir, numeric_contract, tensor_cache,
};
use invar::{mir_eval, tensor};
use invar_device::{Capabilities, Provenance, floating};
use numeric_contract::F32Bits;
use std::{
    fs,
    sync::atomic::{AtomicUsize, Ordering},
};

const DIRECT: &str = "fn mat(a: read Array<f32>, b: read Array<f32>, rows: u32, inner: u32, cols: u32) -> Array<f32> { var c = filled_f32(rows * cols,0.0); for i in 0u32..rows { for j in 0u32..cols { for k in 0u32..inner { c[i * cols + j] = c[i * cols + j] + a[i * inner + k] * b[k * cols + j]; } } } return move c; }";
fn caps() -> Capabilities {
    Capabilities::cuda_12_9_sm89()
}
fn origin(ir: &floating::Ir) -> Provenance {
    Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    }
}
fn grouped() -> String {
    DIRECT
        .replace("rows * cols", "((rows) * (cols))")
        .replace("0u32..rows", "(0u32)..(rows)")
        .replace("0u32..cols", "(0u32)..(cols)")
        .replace("0u32..inner", "(0u32)..(inner)")
        .replace("c[i * cols + j]", "c[((i) * (cols)) + (j)]")
        .replace("a[i * inner + k]", "((a)[((i) * (inner)) + (k)])")
        .replace("b[k * cols + j]", "((b)[((k) * (cols)) + (j)])")
        .replace(
            "filled_f32(((rows) * (cols)),0.0)",
            "(filled_f32(((rows) * (cols)),(0.0)))",
        )
}
fn renamed(s: &str) -> String {
    s.replace("mat", "contract_cells")
        .replace("rows", "height")
        .replace("inner", "depth")
        .replace("cols", "width")
        .replace("a:", "left:")
        .replace("b:", "right:")
        .replace("a[", "left[")
        .replace("b[", "right[")
        .replace("(a)", "(left)")
        .replace("(b)", "(right)")
        .replace("var c", "var output")
        .replace("c[", "output[")
        .replace("move c;", "move output;")
}
fn checked(s: &str) -> check::CheckedProgram {
    check::check(front::parse(s.as_bytes()).unwrap(), s.contains("fn main")).unwrap()
}
fn literal(bits: u32) -> String {
    let v = F32Bits::from_bits(bits);
    if v.is_nan() {
        "(0.0 / 0.0)".into()
    } else if v.is_infinite() {
        if v.is_negative() {
            "(-1.0 / 0.0)".into()
        } else {
            "(1.0 / 0.0)".into()
        }
    } else {
        format!("{:.149}", f32::from_bits(bits))
    }
}
fn scalar_reference(a: &[u32], b: &[u32], [rows, inner, cols]: [u32; 3]) -> Vec<u32> {
    // Exact integer/rational binary32 operations, independent of the physical
    // verifier and C++ emitter. Increasing k; separately rounded mul then add.
    let mut result = Vec::new();
    for row in 0..rows as usize {
        for col in 0..cols as usize {
            let mut sum = F32Bits::POS_ZERO;
            for k in 0..inner as usize {
                let product = F32Bits::from_bits(a[row * inner as usize + k])
                    .mul(F32Bits::from_bits(b[k * cols as usize + col]));
                sum = sum.add(product);
            }
            result.push(sum.to_bits());
        }
    }
    result
}
fn caller(f: &str, dims: [u32; 3], a: &[u32], b: &[u32], values: bool) -> String {
    let list = |v: &[u32]| v.iter().map(|&n| literal(n)).collect::<Vec<_>>().join(",");
    let entry = if f.contains("contract_cells(") {
        "contract_cells"
    } else {
        "mat"
    };
    let [r, k, c] = dims;
    let comparisons = if values {
        scalar_reference(a, b, dims)
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                if F32Bits::from_bits(n).is_nan() {
                    format!("if answer[{i}u32] == answer[{i}u32] {{ return 0; }}")
                } else {
                    format!("if answer[{i}u32] != {} {{ return 0; }}", literal(n))
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        String::new()
    };
    format!(
        "{f} fn main() -> i32 {{ let x: Array<f32> = [{}]; let y: Array<f32> = [{}]; let answer = {entry}(x,y,{r}u32,{k}u32,{c}u32); {comparisons} return 7; }}",
        list(a),
        list(b)
    )
}
fn parity(a: &eval::Observation, b: &eval::Observation) {
    assert_eq!(a.result, b.result);
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace);
}
fn observations(s: &str, prefixes: bool) -> eval::Observation {
    let p = checked(s);
    let net = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let m = net.compiled_mir(&proof).unwrap();
    let full = eval::observe(&p, 1_000_000);
    let end = if prefixes {
        full.steps_used + 1
    } else {
        1_000_000
    };
    let begin = if prefixes { 1 } else { end };
    for fuel in begin..=end {
        let a = eval::observe(&p, fuel);
        parity(&a, &mir_eval::observe(&m, fuel));
        let got = general_net::oracle::observe(
            &net,
            &proof,
            general_net::oracle::Limits {
                fuel,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(got.failure, None);
        parity(&a, &got.semantic);
    }
    full
}
static CHILD: AtomicUsize = AtomicUsize::new(0);
fn run_cpu(e: &Emission, a: &[u32], b: &[u32], opt: &str, extra: &str) -> Vec<u32> {
    let count = e.plan().ir().output.elements as usize;
    let dir = std::env::temp_dir().join(format!(
        "invar-m4-03-direct-{}-{}",
        std::process::id(),
        CHILD.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).unwrap();
    let list = |v: &[u32]| {
        v.iter()
            .map(|n| format!("{n}u"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let harness = format!(
        r#"
#include <cstdio>
#include <vector>
int main() {{
  unsigned ab[{}]={{ {} }},bb[{}]={{ {} }};
  float A[{}], B[{}]; std::memcpy(A,ab,sizeof(ab)); std::memcpy(B,bb,sizeof(bb));
  std::vector<float> C({}+2,19.0f);
  {extra}
  int status=invar_f32_contract_v1(A,B,C.data()+1); if(status) return status+20;
  if(C.front()!=19.0f || C.back()!=19.0f) return 30;
  for(unsigned q=0;q<{};q++) {{ unsigned bits;std::memcpy(&bits,C.data()+1+q,4);std::printf("%u\n",bits); }}
  return 0;
}}
"#,
        a.len().max(1),
        list(a),
        b.len().max(1),
        list(b),
        a.len().max(1),
        b.len().max(1),
        count,
        count
    );
    fs::write(dir.join("kernel.cpp"), format!("{}\n{harness}", e.source())).unwrap();
    let built = child_deadline::command(30)
        .arg(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()))
        .args(e.options())
        .args([opt, "-fsanitize=undefined", "-fno-sanitize-recover=all"])
        .arg(dir.join("kernel.cpp"))
        .arg("-o")
        .arg(dir.join("run"))
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "CPU compiler failed; preserved {dir:?}: {}",
        String::from_utf8_lossy(&built.stderr)
    );
    let ran = child_deadline::command(10)
        .arg(dir.join("run"))
        .output()
        .unwrap();
    assert!(
        ran.status.success(),
        "CPU status {:?}; preserved {dir:?}: {}",
        ran.status,
        String::from_utf8_lossy(&ran.stderr)
    );
    let actual = String::from_utf8(ran.stdout)
        .unwrap()
        .lines()
        .map(|s| s.parse().unwrap())
        .collect();
    fs::remove_dir_all(dir).unwrap();
    actual
}
fn strict_equal(actual: &[u32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(
            F32Bits::from_bits(a).strict_eq(F32Bits::from_bits(b)),
            "{a:08x} != {b:08x}"
        );
    }
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_direct_source_binding() {
    for s in [
        DIRECT.to_owned(),
        grouped(),
        renamed(DIRECT),
        renamed(&grouped()),
        DIRECT.replace("return move c;", "return c;"),
    ] {
        let entry = if s.contains("contract_cells(") {
            "contract_cells"
        } else {
            "mat"
        };
        let p = checked(&s);
        // This is a concrete missing source family, not a second scalar recognizer.
        assert!(matches!(
            tensor::Contraction::from_checked(&p, entry),
            Err(tensor::Error::Unrecognized)
        ));
        let c = Contract::from_source(s.as_bytes(), entry).unwrap();
        assert_eq!(c.function(), entry);
        assert_eq!(c.output(), if entry == "mat" { "c" } else { "output" });
        assert_eq!(c.inputs()[0], if entry == "mat" { "a" } else { "left" });
        assert_eq!(
            c.dimensions()[2],
            if entry == "mat" { "cols" } else { "width" }
        );
        assert_eq!(
            c.loop_span(),
            p.program.functions[0].body.statements[1].span
        );
        assert_ne!(c.source_hash(), [0; 32]);
        assert_ne!(c.mir_contract(), [0; 32]);
        let m = c.verified_mir().unwrap();
        mir::verify(&m).unwrap();
        assert!(m.tensor_contracts.is_empty());
        c.verify_source(s.as_bytes()).unwrap();
        c.verify_mir(&m).unwrap();
    }
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_shapes_layouts_and_identity() {
    let c = Contract::from_source(DIRECT.as_bytes(), "mat").unwrap();
    for dims in [
        [0, 3, 2],
        [3, 5, 0],
        [0, 0, 0],
        [2, 0, 3],
        [1, 1, 1],
        [3, 5, 2],
        [5, 3, 7],
        [17, 7, 19],
        [31, 1, 33],
        [256, 3, 257],
    ] {
        let [r, k, n] = dims.map(u64::from);
        let e = c
            .emit_cpu(DIRECT.as_bytes(), dims, [r * k, k * n], caps())
            .unwrap();
        assert_eq!(e.plan().ir().dimensions, dims);
        assert_eq!(e.plan().ir().function_contract, c.mir_contract());
        assert_eq!(e.plan().ir().span.start, c.loop_span().start);
        assert_eq!(e.plan().ir().output.elements, r * n);
        assert_eq!(e.source(), floating::emit_cpu(e.plan()).unwrap());
        assert!(e.options().contains(&"-ffp-contract=off"));
        assert!(!e.source().contains("__global__"));
        e.verify_candidate(e.plan().ir(), e.source(), e.options())
            .unwrap();
        for (slot, shape) in [[r, k], [k, n], [r, n]].iter().enumerate() {
            assert_eq!(e.layouts()[slot].shape(), shape);
            if !shape.contains(&0) {
                assert!(
                    e.layouts()[slot]
                        .access(&[shape[0] - 1, shape[1] - 1], 4)
                        .is_ok()
                );
            } else {
                assert!(e.layouts()[slot].access(&[0, 0], 4).is_err());
            }
        }
    }
    let e = c
        .emit_cpu(DIRECT.as_bytes(), [3, 5, 2], [15, 10], caps())
        .unwrap();
    let padded = c
        .emit_cpu(DIRECT.as_bytes(), [3, 5, 2], [16, 11], caps())
        .unwrap();
    assert_ne!(e.key(), padded.key());
    let again = c
        .emit_cpu(DIRECT.as_bytes(), [3, 5, 2], [15, 10], caps())
        .unwrap();
    assert_eq!(again.key(), e.key());
    let cuda = tensor_cache::CacheKey::new(&tensor_cache::KeyInput {
        source: DIRECT.as_bytes(),
        entry: "mat",
        source_identity: e.plan().ir().source_identity,
        graph_revision: e.plan().ir().graph_revision,
        function_contract: c.mir_contract(),
        numeric_policy: numeric_contract::NumericPolicy::STRICT_F32,
        layout_version: 1,
        dimensions: [3, 5, 2],
        layouts: [
            layout_contract::LayoutSpec {
                element: layout_contract::LayoutElement::F32,
                shape: &[3, 5],
                strides: &[5, 1],
                offset: 0,
                allocation_bytes: 60,
                base_alignment: 4,
            },
            layout_contract::LayoutSpec {
                element: layout_contract::LayoutElement::F32,
                shape: &[5, 2],
                strides: &[2, 1],
                offset: 0,
                allocation_bytes: 40,
                base_alignment: 4,
            },
            layout_contract::LayoutSpec {
                element: layout_contract::LayoutElement::F32,
                shape: &[3, 2],
                strides: &[2, 1],
                offset: 0,
                allocation_bytes: 24,
                base_alignment: 4,
            },
        ],
        compiler_build: env!("INVAR_BUILD_DIGEST"),
        target: "cuda-nvrtc-c++17",
        compiler_options: e.options(),
        emitted_source: e.source().as_bytes(),
    })
    .unwrap();
    assert_ne!(&cuda, e.key());
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_source_mir_proof_mutations() {
    let c = Contract::from_source(DIRECT.as_bytes(), "mat").unwrap();
    let other = renamed(DIRECT);
    let d = Contract::from_source(other.as_bytes(), "contract_cells").unwrap();
    assert!(matches!(
        c.verify_source(other.as_bytes()),
        Err(Error::SourceMismatch)
    ));
    assert!(matches!(
        c.verify_mir(&d.verified_mir().unwrap()),
        Err(Error::MirMismatch)
    ));
    let mut m = c.verified_mir().unwrap();
    let op = m.functions[0]
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
                .is_some_and(|(_, t)| *t == front::Type::F32)
        })
        .unwrap();
    if let mir::Operation::Binary { op, .. } = &mut op.kind {
        *op = front::BinaryOp::Add;
    }
    mir::verify(&m).unwrap();
    assert!(matches!(c.verify_mir(&m), Err(Error::MirMismatch)));
    let mut m = c.verified_mir().unwrap();
    let tick = m.functions[0]
        .blocks
        .iter_mut()
        .flat_map(|b| &mut b.instructions)
        .find(|i| matches!(i.kind, mir::Operation::Tick) && i.span == c.loop_span())
        .unwrap();
    tick.span.end += 1;
    mir::verify(&m).unwrap();
    assert!(matches!(c.verify_mir(&m), Err(Error::MirMismatch)));
    let mut m = c.verified_mir().unwrap();
    m.layout_version = 2;
    assert!(matches!(c.verify_mir(&m), Err(Error::MirMismatch)));
    assert!(matches!(
        c.emit_cpu(other.as_bytes(), [1, 1, 1], [1, 1], caps()),
        Err(Error::SourceMismatch)
    ));
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_valid_candidate_mutations() {
    let c = Contract::from_source(DIRECT.as_bytes(), "mat").unwrap();
    let e = c
        .emit_cpu(DIRECT.as_bytes(), [2, 2, 2], [4, 4], caps())
        .unwrap();
    let ir = e.plan().ir();
    let mut variants = Vec::new();
    let mut v = ir.clone();
    v.function_contract[0] ^= 1;
    variants.push(v);
    let mut v = ir.clone();
    v.source_identity += 1;
    variants.push(v);
    let mut v = ir.clone();
    v.graph_revision += 1;
    variants.push(v);
    let mut v = ir.clone();
    v.span.end += 1;
    variants.push(v);
    let mut v = ir.clone();
    v.output.offset = 1;
    v.output.elements = 5;
    variants.push(v);
    let mut v = ir.clone();
    v.inputs[0].elements = 5;
    variants.push(v);
    let mut v = ir.clone();
    v.inputs[1].alignment = 8;
    variants.push(v);
    let mut v = ir.clone();
    v.inputs[1].id = v.inputs[0].id;
    variants.push(v);
    let other = c
        .emit_cpu(DIRECT.as_bytes(), [1, 4, 1], [4, 4], caps())
        .unwrap();
    variants.push(other.plan().ir().clone());
    for v in variants {
        floating::verify(v.clone(), caps(), origin(&v)).unwrap();
        assert!(matches!(
            e.verify_candidate(&v, e.source(), e.options()),
            Err(Error::CandidateMismatch)
        ));
    }
    assert!(matches!(
        e.verify_candidate(
            ir,
            &e.source().replace("sum + product", "sum - product"),
            e.options()
        ),
        Err(Error::CandidateMismatch)
    ));
    let mut options = e.options().to_vec();
    options.push("-ffast-math");
    assert!(matches!(
        e.verify_candidate(ir, e.source(), &options),
        Err(Error::CandidateMismatch)
    ));
    options = e.options().to_vec();
    options.reverse();
    assert!(matches!(
        e.verify_candidate(ir, e.source(), &options),
        Err(Error::CandidateMismatch)
    ));
    let mut bad = ir.clone();
    bad.output.id = bad.inputs[0].id;
    assert!(matches!(
        e.verify_candidate(&bad, e.source(), e.options()),
        Err(Error::Device(floating::Error::Alias))
    ));
    // `tc-f16-f32-v1` is now a verifiable policy (D5 as amended), so a strict
    // emission relabeled with it is refused as a different candidate.
    let mut bad = ir.clone();
    bad.numeric_policy = "tc-f16-f32-v1".into();
    assert!(matches!(
        e.verify_candidate(&bad, e.source(), e.options()),
        Err(Error::CandidateMismatch)
    ));
    bad.numeric_policy = "relaxed-f32-v1".into();
    assert!(matches!(
        e.verify_candidate(&bad, e.source(), e.options()),
        Err(Error::Device(floating::Error::Numeric))
    ));
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_bounds_and_refusal_paths() {
    let c = Contract::from_source(DIRECT.as_bytes(), "mat").unwrap();
    for (d, l) in [
        ([2, 2, 2], [3, 4]),
        ([2, 2, 2], [4, 3]),
        ([1, 1, (1 << 28) + 1], [1, (1 << 28) + 1]),
        ([1, 1, 1], [(1 << 28) + 1, 1]),
        ([0, u32::MAX, 0], [0, 0]),
        ([1 << 16, 1 << 16, 1 << 16], [1 << 32, 1 << 32]),
    ] {
        assert!(
            c.emit_cpu(DIRECT.as_bytes(), d, l, caps()).is_err(),
            "{d:?} {l:?}"
        );
    }
    assert!(matches!(
        c.emit_cpu(
            DIRECT.as_bytes(),
            [1, 1, 1],
            [1, 1],
            Capabilities::Metal {
                apple_family: 8,
                msl_major: 3,
                msl_minor: 1
            }
        ),
        Err(Error::Device(floating::Error::Capabilities))
    ));
    let step = "c[i * cols + j] + a[i * inner + k] * b[k * cols + j]";
    for s in [
        DIRECT.replace(
            step,
            "fma(a[i * inner + k],b[k * cols + j],c[i * cols + j])",
        ),
        DIRECT.replace(step, "a[i * inner + k] * b[k * cols + j] + c[i * cols + j]"),
        DIRECT.replace(step, "c[i * cols + j] + b[k * cols + j] * a[i * inner + k]"),
        DIRECT.replace(
            "filled_f32(rows * cols,0.0)",
            "filled_f32(rows * cols,-0.0)",
        ),
        DIRECT.replace("a[i * inner + k]", "a[k * rows + i]"),
        DIRECT.replace("c[i * cols + j] =", "c[j * rows + i] ="),
        DIRECT.replace("for k in 0u32..inner", "for k in 0u32..inner-1u32"),
        DIRECT.replace("for i in 0u32..rows", "for i in 1u32..rows"),
        DIRECT.replace("c[i * cols + j] +", "c[j * rows + i] +"),
        DIRECT.replace("for k in", "let unused:u32=0u32; for k in"),
        DIRECT.replace("rows * cols,0.0", "rows * cols + 1u32,0.0"),
    ] {
        assert!(matches!(
            Contract::from_source(s.as_bytes(), "mat"),
            Err(Error::Unrecognized)
        ));
        let serial = caller(&s, [1, 1, 1], &[0x3f800000], &[0x40000000], false);
        assert_eq!(observations(&serial, false).result, Ok(eval::Value::I32(7)));
    }
    let scalar = DIRECT.replace(
        "for k in 0u32..inner",
        "var sum:f32=0.0; for k in 0u32..inner",
    );
    assert!(matches!(
        Contract::from_source(scalar.as_bytes(), "mat"),
        Err(Error::Unrecognized)
    ));
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_actual_emitted_cpu_special_values() {
    let patterns = [
        0, 0x80000000, 1, 0x007fffff, 0x00800001, 0x3f800001, 0xbf800000, 0x7f7fffff, 0x7f800000,
        0xff800000, 0x7fc12345, 0x7fa12345,
    ];
    let mut cases = 0;
    for (f, entry) in [
        (DIRECT.to_owned(), "mat"),
        (renamed(&grouped()), "contract_cells"),
    ] {
        let c = Contract::from_source(f.as_bytes(), entry).unwrap();
        for dims in [
            [0, 3, 2],
            [3, 5, 0],
            [0, 0, 0],
            [2, 0, 3],
            [1, 1, 1],
            [3, 5, 2],
            [5, 3, 7],
            [17, 7, 19],
        ] {
            let [r, k, n] = dims.map(|n| n as usize);
            let a = (0..r * k)
                .map(|i| patterns[i % patterns.len()])
                .collect::<Vec<_>>();
            let b = (0..k * n)
                .map(|i| patterns[(i + 3) % patterns.len()])
                .collect::<Vec<_>>();
            let expected = scalar_reference(&a, &b, dims);
            let e = c
                .emit_cpu(f.as_bytes(), dims, [a.len() as u64, b.len() as u64], caps())
                .unwrap();
            for opt in ["-O0", "-O2"] {
                strict_equal(&run_cpu(&e, &a, &b, opt, ""), &expected);
                cases += 1;
            }
            // Bounded ordinary-source checks include special-value semantics.
            if expected.len() <= 35 {
                assert_eq!(
                    observations(&caller(&f, dims, &a, &b, true), false).result,
                    Ok(eval::Value::I32(7))
                );
            }
        }
    }
    assert_eq!(cases, 32);
    println!(
        "actual strict CPU runs: {cases} (O0/O2 with UBSan); special FP32 patterns: {}",
        patterns.len()
    );
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_no_fma_or_reassociation() {
    let c = Contract::from_source(DIRECT.as_bytes(), "mat").unwrap();
    for (a, b) in [
        (vec![0x3f800000, 0x3f800001], vec![0xbf800000, 0x3f7ffffe]),
        (
            vec![0x4b800000, 0x3f800000, 0xcb800000],
            vec![0x3f800000; 3],
        ),
    ] {
        let d = [1, a.len() as u32, 1];
        assert_eq!(scalar_reference(&a, &b, d), vec![0]);
        let e = c
            .emit_cpu(
                DIRECT.as_bytes(),
                d,
                [a.len() as u64, b.len() as u64],
                caps(),
            )
            .unwrap();
        for opt in ["-O0", "-O2"] {
            assert_eq!(run_cpu(&e, &a, &b, opt, ""), vec![0]);
        }
        assert_eq!(
            observations(&caller(DIRECT, d, &a, &b, true), false).result,
            Ok(eval::Value::I32(7))
        );
    }
    // Independent hand discriminator: the fused second step retains -2^-46.
    assert_eq!(
        F32Bits::from_bits(0x3f800001)
            .fma(
                F32Bits::from_bits(0x3f7ffffe),
                F32Bits::from_bits(0xbf800000)
            )
            .to_bits(),
        0xa8800000
    );
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_fuel_prefixes_traps_cleanup() {
    let mut fixtures = 0;
    let mut prefixes = 0;
    for f in [DIRECT.to_owned(), renamed(&grouped())] {
        let entry = if f.contains("contract_cells(") {
            "contract_cells"
        } else {
            "mat"
        };
        for dims in [[0, 2, 2], [2, 0, 3], [1, 1, 1], [2, 2, 3]] {
            let [r, k, n] = dims.map(|v| v as usize);
            let a = vec![0x3f800000; r * k];
            let b = vec![0x40000000; k * n];
            let s = caller(&f, dims, &a, &b, false);
            let c = Contract::from_source(s.as_bytes(), entry).unwrap();
            // Independent hand count for this direct-cell body, not emitter weights.
            assert_eq!(
                c.loop_fuel(dims),
                Some(r as u64 * (4 + n as u64 * (4 + k as u64 * 30)))
            );
            let full = observations(&s, true);
            assert_eq!(full.result, Ok(eval::Value::I32(7)));
            assert_eq!(
                full.steps_used,
                24 + a.len() as u64 + b.len() as u64 + c.loop_fuel(dims).unwrap()
            );
            prefixes += full.steps_used + 1;
            fixtures += 1;
        }
    }
    for (a, b) in [(vec![], vec![0x3f800000]), (vec![0x3f800000], vec![])] {
        let s = caller(DIRECT, [1, 1, 1], &a, &b, false);
        let full = observations(&s, true);
        assert!(matches!(
            full.result,
            Err(eval::EvalError::Trap {
                category: eval::TrapCategory::IndexOutOfBounds,
                ..
            })
        ));
        assert!(full.cleanup_trace.is_empty());
    }
    println!(
        "ordinary fuel-prefix fixtures: {fixtures}; normal prefixes: {prefixes}; plus two index-trap fixtures"
    );
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_runtime_alias_and_environment_guards() {
    let c = Contract::from_source(DIRECT.as_bytes(), "mat").unwrap();
    let e = c
        .emit_cpu(DIRECT.as_bytes(), [1, 1, 1], [1, 1], caps())
        .unwrap();
    let extra = r#"
  if(invar_f32_contract_v1(A,B,A)!=2) return 40;
  if(invar_f32_contract_v1(A,B,B)!=2) return 41;
  float shared[2]={1.0f,19.0f};
  if(invar_f32_contract_v1(shared,B,shared)!=2) return 42;
  if(invar_f32_contract_v1(A,A,C.data()+1)!=0 || C[1]!=1.0f) return 43;
  int saved=std::fegetround();std::fesetround(FE_UPWARD);
  C[1]=19.0f; if(invar_f32_contract_v1(A,B,C.data()+1)!=1 || C[1]!=19.0f) return 44;
  std::fesetround(saved);
  std::feraiseexcept(FE_DIVBYZERO); int flags=std::fetestexcept(FE_ALL_EXCEPT);
  if(invar_f32_contract_v1(A,B,C.data()+1)!=0 || std::fetestexcept(FE_ALL_EXCEPT)!=flags) return 45;
"#;
    assert_eq!(
        run_cpu(&e, &[0x3f800000], &[0x40000000], "-O2", extra),
        vec![0x40000000]
    );
    let padded = c
        .emit_cpu(DIRECT.as_bytes(), [1, 1, 1], [2, 2], caps())
        .unwrap();
    let overlap = "if(invar_f32_contract_v1(A,B,A+1)!=2) return 46;";
    assert_eq!(
        run_cpu(&padded, &[0x3f800000, 0], &[0x40000000, 0], "-O2", overlap),
        vec![0x40000000]
    );
}
