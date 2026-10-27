#![forbid(unsafe_code)]
use crate::child_deadline;
use invar::{
    check, eval, front,
    general_net::{self, oracle},
    mir, mir_eval,
    transpose_source::{Contract, Error},
};
use invar_device::{Capabilities, Provenance, transpose};
use std::{
    fs,
    sync::atomic::{AtomicUsize, Ordering},
};

const U32: &str = "fn flip(a: read Array<u32>, rows: u32, cols: u32) -> Array<u32> { var b = filled_u32(rows * cols,0u32); for r in 0u32..rows { for c in 0u32..cols { b[c * rows + r] = a[r * cols + c]; } } return b; }";
const F32: &str = "fn flip(a: read Array<f32>, rows: u32, cols: u32) -> Array<f32> { var b = filled_f32(rows * cols,0.0); for r in 0u32..rows { for c in 0u32..cols { b[c * rows + r] = a[r * cols + c]; } } return b; }";
fn caps() -> Capabilities {
    Capabilities::Cuda {
        major: 12,
        minor: 4,
        sm: 86,
    }
}
fn checked(s: &str) -> check::CheckedProgram {
    check::check(front::parse(s.as_bytes()).unwrap(), false).unwrap()
}
fn origin(ir: &transpose::Ir) -> Provenance {
    Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    }
}
fn grouped(s: &str) -> String {
    s.replace("rows * cols", "((rows) * (cols))")
        .replace("0u32..rows", "(0u32)..(rows)")
        .replace("0u32..cols", "(0u32)..(cols)")
        .replace(
            "b[c * rows + r] = a[r * cols + c]",
            "b[((c) * (rows)) + (r)] = (a[((r) * (cols)) + (c)])",
        )
}
fn bits(i: usize) -> u32 {
    let special = [
        0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x3f800000, 0x7f800000, 0xff800000, 0x7fc12345,
        0x7f812345,
    ];
    if i < special.len() {
        special[i]
    } else {
        (i as u32).wrapping_mul(747796405).wrapping_add(2891336453)
    }
}
fn inverse_index(input: &[u32], [rows, cols]: [u32; 2]) -> Vec<u32> {
    (0..u64::from(rows) * u64::from(cols))
        .map(|index| {
            let row = index / u64::from(rows);
            let col = index % u64::from(rows);
            input[(col * u64::from(cols) + row) as usize]
        })
        .collect()
}

#[test]
fn issue_m4_02_transpose_source_checked_identity_names_groups_and_owner_return() {
    for s in [
        U32.to_owned(),
        F32.to_owned(),
        grouped(U32),
        grouped(F32),
        U32.replace("return b;", "return move b;"),
    ] {
        let p = checked(&s);
        let c = Contract::from_checked(&p, "flip").unwrap();
        assert_eq!(c.function(), "flip");
        assert_eq!(c.input(), "a");
        assert_eq!(c.output(), "b");
        assert_eq!(c.dimensions(), &["rows".to_owned(), "cols".to_owned()]);
        assert_eq!(
            c.element(),
            if s.contains("Array<f32>") {
                front::Type::F32
            } else {
                front::Type::U32
            }
        );
        assert!(s[c.loop_span().start..c.loop_span().end].starts_with("for r"));
        let m = c.verified_mir().unwrap();
        mir::verify(&m).unwrap();
        assert_ne!(c.function_contract(), [0; 32]);
        let e = c.emit_cuda([3, 5], 15, caps()).unwrap();
        let ir = e.plan().ir();
        assert_eq!(ir.function_contract, c.function_contract());
        assert_eq!(ir.span.start, c.loop_span().start);
        assert_eq!(ir.input.id, 0);
        assert_eq!(ir.output.id, 1);
        assert_eq!(ir.output.elements, 15);
        assert_eq!(ir.input.shape, [3, 5]);
        assert_eq!(ir.output.shape, [5, 3]);
        assert_eq!(ir.output.strides, [3, 1]);
        assert_eq!(e.source(), transpose::emit_cuda(e.plan()).unwrap());
        assert_eq!(e.options(), e.plan().nvrtc_options());
        let mut modified = m.clone();
        modified.functions[0].name = "other".to_owned();
        assert_ne!(modified.contract_digest(), m.contract_digest());
        assert_eq!(
            c.verified_mir().unwrap().contract_digest(),
            m.contract_digest()
        );
    }
    let renamed = U32
        .replace("flip", "rotate")
        .replace("rows", "height")
        .replace("cols", "width")
        .replace("a:", "src:")
        .replace("a[", "src[")
        .replace("var b", "var dst")
        .replace("b[", "dst[")
        .replace("return b", "return dst");
    let c = Contract::from_checked(&checked(&renamed), "rotate").unwrap();
    assert_eq!(c.input(), "src");
    assert_eq!(c.output(), "dst");
    assert_eq!(c.dimensions(), &["height".to_owned(), "width".to_owned()]);
    let original = Contract::from_checked(&checked(U32), "flip")
        .unwrap()
        .emit_cuda([3, 5], 15, caps())
        .unwrap();
    let changed = c.emit_cuda([3, 5], 15, caps()).unwrap();
    assert_eq!(
        original.verify_descriptor(changed.plan().ir()),
        Err(Error::DescriptorMismatch)
    );
}

#[test]
fn issue_m4_02_transpose_source_unknown_effects_alias_and_index_variants_refuse() {
    for s in [
        U32.replace("c * rows + r", "r * cols + c"),
        U32.replace("a[r * cols + c]", "a[r * rows + c]"),
        U32.replace("a[r * cols + c]", "a[r * cols + c] + 1u32"),
        U32.replace("0u32..rows", "1u32..rows"),
        U32.replace("0u32..cols", "0u32..rows"),
        U32.replace("rows * cols", "rows + cols"),
        U32.replace(
            "var b = filled_u32(rows * cols,0u32);",
            "var b = filled_u32(rows * cols,0u32); let unused = len(a);",
        ),
        U32.replace(
            "b[c * rows + r] = a[r * cols + c];",
            "let value = a[r * cols + c]; b[c * rows + r] = value;",
        ),
        U32.replace("return b;", "return filled_u32(0u32,0u32);"),
        U32.replace("0u32);", "1u32);"),
        U32.replace("a: read", "a: mut"),
        U32.replace(
            "b[c * rows + r] = a[r * cols + c]",
            "a[c * rows + r] = b[r * cols + c]",
        )
        .replace("a: read", "a: mut"),
    ] {
        let p = checked(&s);
        assert!(
            matches!(Contract::from_checked(&p, "flip"), Err(Error::Unrecognized)),
            "{s}"
        );
    }
    let alias = U32.replace("var b = filled_u32(rows * cols,0u32);", "var b = a;");
    let error = check::check(front::parse(alias.as_bytes()).unwrap(), false).unwrap_err();
    assert_eq!(error.code, "OWN_MOVE_REQUIRED");
    assert!(matches!(
        Contract::from_checked(&checked(U32), "missing"),
        Err(Error::Unrecognized)
    ));
    let mut p = checked(U32);
    p.function_expression_types.get_mut("flip").unwrap().pop();
    assert!(matches!(
        Contract::from_checked(&p, "flip"),
        Err(Error::CheckedMismatch)
    ));
    let mut p = checked(U32);
    p.expression_types.clear();
    assert!(matches!(
        Contract::from_checked(&p, "flip"),
        Err(Error::CheckedMismatch)
    ));
}

#[test]
fn issue_m4_02_transpose_source_physical_bounds_and_stale_descriptors() {
    let c = Contract::from_checked(&checked(U32), "flip").unwrap();
    for (dims, len) in [
        ([2, 3], 5),
        ([1001, 1000], 1_000_000),
        ([u32::MAX, 2], 1_000_000),
        ([0, 1], (1_u64 << 28) + 1),
    ] {
        assert_eq!(c.emit_cuda(dims, len, caps()).unwrap_err(), Error::Bounds);
    }
    assert_eq!(
        c.emit_cuda([0, u32::MAX], 0, caps()).unwrap_err(),
        Error::Device(transpose::Error::Bounds)
    );
    let source = main_source(U32, [0, u32::MAX], 0);
    let p = checked(&source);
    let empty = Contract::from_checked(&p, "flip").unwrap();
    let reference = eval::observe(&p, 100);
    assert_eq!(reference.result, Ok(eval::Value::I32(7)));
    let actual = mir_eval::observe(&empty.verified_mir().unwrap(), 100);
    assert_eq!(actual.result, reference.result);
    assert_eq!(actual.steps_used, reference.steps_used);
    assert_eq!(
        c.emit_cuda(
            [1, 1],
            1,
            Capabilities::Metal {
                apple_family: 7,
                msl_major: 3,
                msl_minor: 1
            }
        )
        .unwrap_err(),
        Error::Device(transpose::Error::Capabilities)
    );
    let e = c.emit_cuda([3, 5], 20, caps()).unwrap();
    let ir = e.plan().ir().clone();
    e.verify_descriptor(&ir).unwrap();
    for i in 0..8 {
        let mut wrong = ir.clone();
        match i {
            0 => wrong.version += 1,
            1 => wrong.layout_version += 1,
            2 => wrong.source_identity ^= 1,
            3 => wrong.graph_revision += 1,
            4 => wrong.function_contract[0] ^= 1,
            5 => wrong.input.strides = [1, 1],
            6 => wrong.output.id = wrong.input.id,
            7 => wrong.dimensions = [5, 3],
            _ => unreachable!(),
        };
        assert_eq!(e.verify_descriptor(&wrong), Err(Error::DescriptorMismatch));
        if i != 4 {
            assert!(transpose::verify(wrong, caps(), origin(&ir)).is_err());
        }
    }
    // Nonzero invented function digests are metadata to the physical verifier;
    // the opaque source-derived comparison rejects them without trusting a hash.
    let mut forged = ir.clone();
    forged.function_contract = [7; 32];
    let standalone = transpose::verify(forged, caps(), origin(&ir)).unwrap();
    assert_eq!(
        e.verify_descriptor(standalone.ir()),
        Err(Error::DescriptorMismatch)
    );
}

#[test]
fn issue_m4_02_transpose_source_every_source_mir_word_matches_independent_indexing() {
    for f in [U32, F32] {
        for dims in [[0, 33], [33, 0], [1, 1], [3, 5], [31, 33]] {
            let count = (dims[0] * dims[1]) as usize;
            // Real ordinary-source array arguments: the public oracle external
            // entry deliberately admits scalars only. Positive finite f32 values
            // have unique bits; arbitrary NaN payloads are tested by emitted CPU.
            let input: Vec<u32> = (0..count + 5)
                .map(|i| if f == U32 { bits(i) } else { (i + 1) as u32 })
                .collect();
            let expected = inverse_index(&input, dims);
            let literal = |v: u32| {
                if f == U32 {
                    format!("{v}u32")
                } else {
                    format!("{v}.0")
                }
            };
            let ty = if f == U32 { "u32" } else { "f32" };
            let values = input
                .iter()
                .map(|&v| literal(v))
                .collect::<Vec<_>>()
                .join(",");
            let expect = expected
                .iter()
                .map(|&v| literal(v))
                .collect::<Vec<_>>()
                .join(",");
            let body = format!(
                "let input: Array<{ty}> = [{values}]; let original: Array<{ty}> = [{values}]; let expected: Array<{ty}> = [{expect}]; let output = flip(input,{}u32,{}u32); for index in 0u32..len(output) {{ if output[index] != expected[index] {{ return 0; }} }} for retained in 0u32..len(input) {{ if input[retained] != original[retained] {{ return 0; }} }}",
                dims[0], dims[1]
            );
            let source = format!("{f} fn main() -> i32 {{ {body} return 7; }}");
            let p = checked(&source);
            let c = Contract::from_checked(&p, "flip").unwrap();
            let net = general_net::Net::encode(
                &c.verified_mir().unwrap(),
                general_net::Limits::default(),
            )
            .unwrap();
            let proof = net.verify().unwrap();
            let observed = oracle::observe(
                &net,
                &proof,
                oracle::Limits {
                    fuel: 1_000_000,
                    ..oracle::Limits::default()
                },
            )
            .unwrap();
            let reference = eval::observe(&p, 1_000_000);
            let serial = mir_eval::observe(&c.verified_mir().unwrap(), 1_000_000);
            assert_eq!(reference.result, Ok(eval::Value::I32(7)));
            assert_eq!(serial.result, reference.result);
            assert_eq!(serial.steps_used, reference.steps_used);
            assert_eq!(serial.cleanup_trace, reference.cleanup_trace);
            // ABI10 admits direct f32 array rules to the independent oracle.
            assert_eq!(observed.failure, None);
            assert_eq!(observed.semantic.result, reference.result);
            assert_eq!(observed.semantic.steps_used, reference.steps_used);
            assert_eq!(observed.semantic.cleanup_trace, reference.cleanup_trace);
            println!(
                "source_mir_element={:?} dimensions={dims:?} words={count} steps={}",
                c.element(),
                reference.steps_used
            );
        }
    }
}

fn main_source(function: &str, dims: [u32; 2], input_len: u32) -> String {
    let [rows, cols] = dims;
    let floating = function.contains("Array<f32>");
    let allocator = if floating { "filled_f32" } else { "filled_u32" };
    let fill = if floating { "2.0" } else { "2u32" };
    format!(
        "{function} fn main() -> i32 {{ let input = {allocator}({input_len}u32,{fill}); let output = flip(input,{rows}u32,{cols}u32); return 7; }}"
    )
}
#[test]
fn issue_m4_02_transpose_source_exact_loop_fuel_and_every_small_fallback_prefix() {
    for f in [U32.to_owned(), F32.to_owned(), grouped(U32), grouped(F32)] {
        for dims in [[0, 3], [3, 0], [1, 1], [2, 3]] {
            let s = main_source(&f, dims, dims[0] * dims[1]);
            let p = checked(&s);
            let c = Contract::from_checked(&p, "flip").unwrap();
            let m = c.verified_mir().unwrap();
            let complete = eval::observe(&p, 10000);
            assert_eq!(complete.result, Ok(eval::Value::I32(7)));
            let mut baseline = s.clone();
            baseline.replace_range(c.loop_span().start..c.loop_span().end, "");
            let no_loop = eval::observe(&checked(&baseline), 10000);
            assert_eq!(no_loop.result, Ok(eval::Value::I32(7)));
            assert_eq!(
                complete.steps_used - no_loop.steps_used - 3,
                c.loop_fuel(dims).unwrap()
            );
            assert_eq!(
                c.loop_fuel(dims),
                Some(u64::from(dims[0]) * (4 + 14 * u64::from(dims[1])))
            );
            for fuel in 1..=complete.steps_used + 1 {
                let reference = eval::observe(&p, fuel);
                let actual = mir_eval::observe(&m, fuel);
                assert_eq!(actual.result, reference.result, "dims={dims:?} fuel={fuel}");
                assert_eq!(actual.steps_used, reference.steps_used);
                assert_eq!(actual.cleanup_trace, reference.cleanup_trace);
            }
        }
    }
    let c = Contract::from_checked(&checked(U32), "flip").unwrap();
    assert_eq!(c.loop_fuel([u32::MAX, u32::MAX]), None);
}

#[test]
fn issue_m4_02_transpose_source_refused_layouts_keep_original_ordered_traps() {
    for (dims, len) in [([2, 3], 1), ([u32::MAX, 2], 0), ([1001, 1000], 0)] {
        let s = main_source(U32, dims, len);
        let p = checked(&s);
        let c = Contract::from_checked(&p, "flip").unwrap();
        assert!(c.emit_cuda(dims, u64::from(len), caps()).is_err());
        let m = c.verified_mir().unwrap();
        let complete = eval::observe(&p, 1000);
        assert!(complete.result.is_err());
        for fuel in 1..=complete.steps_used + 1 {
            let reference = eval::observe(&p, fuel);
            let actual = mir_eval::observe(&m, fuel);
            assert_eq!(actual.result, reference.result);
            assert_eq!(actual.steps_used, reference.steps_used);
            assert_eq!(actual.cleanup_trace, reference.cleanup_trace);
        }
        println!(
            "refused dims={dims:?} len={len} source_result={:?}",
            complete.result
        );
    }
}

#[test]
fn issue_m4_02_transpose_source_real_emitted_cpu_bits_and_guard_words() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    for s in [U32, F32] {
        let c = Contract::from_checked(&checked(s), "flip").unwrap();
        for dims in [[0, 33], [33, 0], [1, 1], [3, 5], [31, 33], [127, 257]] {
            let count = (dims[0] * dims[1]) as usize;
            let input: Vec<_> = (0..count + 5).map(bits).collect();
            let expected = inverse_index(&input, dims);
            let e = c.emit_cuda(dims, input.len() as u64, caps()).unwrap();
            let cpu = transpose::emit_cpu(e.plan()).unwrap();
            let dir = std::env::temp_dir().join(format!(
                "invar-source-transpose-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&dir).unwrap();
            let driver = format!(
                r#"
#include <vector>
#include <iostream>
#include <unistd.h>
int main() {{ std::vector<std::uint32_t> input={{{input}}};auto original=input;
std::vector<std::uint32_t> output({count}+8,0xdeadbeef);if(invar_transpose_bits_v1(input.data(),output.data()))return 2;
if(input!=original)return 3;for(unsigned i={count};i<output.size();++i)if(output[i]!=0xdeadbeef)return 4;
for(unsigned i=0;i<{count};++i)std::cout<<output[i]<<' ';return 0; }}
"#,
                input = input
                    .iter()
                    .map(|n| format!("{n}u"))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            fs::write(dir.join("transpose.cpp"), format!("{cpu}\n{driver}")).unwrap();
            child_deadline::cxx(
                child_deadline::CXX17_UBSAN_O2,
                &dir.join("transpose.cpp"),
                &dir.join("transpose"),
            );
            let run = child_deadline::command(10)
                .arg(dir.join("transpose"))
                .output()
                .unwrap();
            child_deadline::assert_success(&run);
            let actual: Vec<u32> = String::from_utf8(run.stdout)
                .unwrap()
                .split_whitespace()
                .map(|n| n.parse().unwrap())
                .collect();
            assert_eq!(actual, expected);
            println!(
                "emitted_cpu element={:?} dimensions={dims:?} words={count} guards=8 source_identity={} function_contract={:?}",
                c.element(),
                e.plan().ir().source_identity,
                c.function_contract()
            );
            fs::remove_dir_all(dir).unwrap();
        }
    }
}

#[test]
fn issue_m4_02_transpose_cap_2_28_source_plan() {
    for source in [U32, F32] {
        let c = Contract::from_checked(&checked(source), "flip").unwrap();
        for shape in [[16384, 16384], [1, 1 << 28], [1 << 28, 1]] {
            let p = c.emit_cuda(shape, 1 << 28, caps()).unwrap();
            assert_eq!(p.plan().ir().input.elements, 1 << 28);
            assert!(c.loop_fuel(shape).unwrap() > 1 << 28);
        }
        assert_eq!(
            c.emit_cuda([1, (1 << 28) + 1], (1 << 28) + 1, caps())
                .unwrap_err(),
            Error::Bounds
        );
    }
}
