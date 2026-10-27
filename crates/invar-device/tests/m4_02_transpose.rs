#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "../../../test-support/child_deadline.rs"]
mod child_deadline;
use floating::Buffer;
use invar_device::transpose;
pub use invar_device::{Capabilities, MAX_SOURCE_BYTES, Provenance, Span, floating};
use std::{
    fmt::Write,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use transpose::{Error, Ir, MAX_LOGICAL_ELEMENTS, Verified, emit_cpu, emit_cuda, verify};

fn origin() -> Provenance {
    Provenance {
        source_identity: 71,
        graph_revision: 3,
    }
}
fn plan(dimensions: [u32; 2], offsets: [u64; 2], padding: u64) -> Ir {
    let [rows, cols] = dimensions;
    let make = |id, shape, offset| Buffer {
        id,
        elements: u64::from(rows) * u64::from(cols) + offset + padding,
        shape,
        strides: [u64::from(shape[1]), 1],
        offset,
        alignment: 4,
    };
    Ir {
        version: transpose::VERSION,
        layout_version: floating::LAYOUT_VERSION,
        span: Span { start: 7, end: 93 },
        source_identity: origin().source_identity,
        graph_revision: origin().graph_revision,
        function_contract: [42; 32],
        dimensions,
        input: make(1, [rows, cols], offsets[0]),
        output: make(2, [cols, rows], offsets[1]),
    }
}
fn admit(ir: Ir) -> Verified {
    verify(ir, Capabilities::cuda_12_9_sm89(), origin()).unwrap()
}
fn refused(ir: Ir, error: Error) {
    assert_eq!(
        verify(ir, Capabilities::cuda_12_9_sm89(), origin()).unwrap_err(),
        error
    );
}

#[test]
fn issue_m4_02_transpose_version_provenance_capability_mutations() {
    for (change, error) in [
        (0, Error::Version),
        (1, Error::Layout),
        (2, Error::Source),
        (3, Error::Source),
        (4, Error::Graph),
        (5, Error::Graph),
        (6, Error::Contract),
    ] {
        let mut ir = plan([3, 5], [1, 2], 3);
        match change {
            0 => ir.version = 2,
            1 => ir.layout_version = 2,
            2 => ir.span.end = ir.span.start,
            3 => ir.source_identity = 0,
            4 => ir.source_identity += 1,
            5 => ir.graph_revision += 1,
            6 => ir.function_contract = [0; 32],
            _ => unreachable!(),
        }
        refused(ir, error);
    }
    let mut ir = plan([3, 5], [1, 2], 3);
    ir.span.start = usize::MAX;
    refused(ir, Error::Source);
    for caps in [
        Capabilities::Metal {
            apple_family: 6,
            msl_major: 3,
            msl_minor: 1,
        },
        Capabilities::Cuda {
            major: 11,
            minor: 9,
            sm: 89,
        },
        Capabilities::Cuda {
            major: 13,
            minor: 0,
            sm: 89,
        },
        Capabilities::Cuda {
            major: 12,
            minor: 3,
            sm: 89,
        },
        Capabilities::Cuda {
            major: 12,
            minor: 10,
            sm: 89,
        },
        Capabilities::Cuda {
            major: 12,
            minor: 9,
            sm: 999,
        },
    ] {
        assert_eq!(
            verify(plan([1, 1], [0, 0], 0), caps, origin()).unwrap_err(),
            Error::Capabilities
        );
    }
    for sm in [50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90] {
        let caps = Capabilities::Cuda {
            major: 12,
            minor: 4,
            sm,
        };
        let p = verify(plan([1, 1], [0, 0], 0), caps, origin()).unwrap();
        assert_eq!(p.capabilities(), caps);
        assert!(
            p.nvrtc_options()
                .contains(&format!("--gpu-architecture=compute_{sm}"))
        );
    }
}

#[test]
fn issue_m4_02_transpose_layout_bounds_empty_and_alias_mutations() {
    for slot in 0..2 {
        for change in 0..10 {
            let mut ir = plan([3, 5], [1, 2], 3);
            let b = if slot == 0 {
                &mut ir.input
            } else {
                &mut ir.output
            };
            let error = match change {
                0 => {
                    b.shape[0] += 1;
                    Error::Shape
                }
                1 => {
                    b.shape[1] += 1;
                    Error::Shape
                }
                2 => {
                    b.strides[0] += 1;
                    Error::Layout
                }
                3 => {
                    b.strides[1] = 2;
                    Error::Layout
                }
                4 => {
                    b.alignment = 0;
                    Error::Alignment
                }
                5 => {
                    b.alignment = 3;
                    Error::Alignment
                }
                6 => {
                    b.alignment = 2;
                    Error::Alignment
                }
                7 => {
                    b.elements = b.offset + 14;
                    Error::Bounds
                }
                8 => {
                    b.offset = u64::MAX;
                    Error::Bounds
                }
                9 => {
                    b.elements = floating::MAX_ELEMENTS + 1;
                    Error::Bounds
                }
                _ => unreachable!(),
            };
            refused(ir, error);
        }
    }
    for dimensions in [[3, 5], [0, 33], [33, 0], [0, 0]] {
        let mut ir = plan(dimensions, [1, 2], 3);
        ir.output.id = ir.input.id;
        refused(ir, Error::Alias);
    }
    for dimensions in [
        [0, 33],
        [33, 0],
        [0, 0],
        [0, floating::MAX_EXTENT],
        [floating::MAX_EXTENT, 0],
    ] {
        let mut ir = plan(dimensions, [0, 0], 0);
        ir.input.offset = ir.input.elements;
        ir.output.offset = ir.output.elements;
        let p = admit(ir.clone());
        for source in [emit_cpu(&p).unwrap(), emit_cuda(&p).unwrap()] {
            assert!(!source.contains("input["));
            assert!(!source.contains("output["));
            assert!(!source.contains("__syncthreads"));
        }
        ir.input.offset += 1;
        refused(ir, Error::Bounds);
        let mut ir = plan(dimensions, [0, 0], 0);
        ir.output.offset = 1;
        refused(ir, Error::Bounds);
    }
    for dimensions in [
        [0, u32::MAX],
        [u32::MAX, 0],
        [u32::MAX, u32::MAX],
        [32768, 32768],
    ] {
        refused(plan(dimensions, [0, 0], 0), Error::Bounds);
    }
    let maximum = admit(plan(
        [1, u32::try_from(MAX_LOGICAL_ELEMENTS).unwrap()],
        [0, 0],
        0,
    ));
    assert!(emit_cuda(&maximum).unwrap().len() < MAX_SOURCE_BYTES);
    refused(
        plan([1, MAX_LOGICAL_ELEMENTS as u32 + 1], [0, 0], 0),
        Error::Bounds,
    );
    let mut ir = maximum.ir().clone();
    ir.input.offset = 1;
    refused(ir, Error::Bounds);
    let mut ir = plan([2, 3], [0, 0], 0);
    ir.dimensions = [3, 2];
    refused(ir, Error::Shape);
}

fn pattern(index: usize) -> u32 {
    const EDGE: [u32; 16] = [
        0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7f800001,
        0x7fc12345, 0xffc54321, 0xffffffff, 0x3f800000, 0xbf800000, 0x12345678, 0xdeadbeef,
    ];
    if index < EDGE.len() {
        EDGE[index]
    } else {
        u32::try_from(index).unwrap().wrapping_mul(0x9e3779b9) ^ 0xa5a5a5a5
    }
}

#[test]
fn issue_m4_02_transpose_cuda_uniform_barriers_edge_masks_and_tile_access_map() {
    for dimensions in [
        [0, 33],
        [1, 1],
        [31, 33],
        [127, 257],
        [16, 16],
        [17, 17],
        [1, 37],
        [37, 1],
    ] {
        let p = admit(plan(dimensions, [3, 7], 5));
        let source = emit_cuda(&p).unwrap();
        if dimensions.contains(&0) {
            continue;
        }
        assert_eq!(source.matches("__syncthreads();").count(), 2);
        let mut depth = 0;
        for line in source.lines().filter(|line| !line.starts_with("//")) {
            if line.contains("__syncthreads();") {
                assert_eq!(
                    depth, 2,
                    "barrier must be in the uniform tile loop, outside every lane mask"
                );
            }
            depth += line.matches('{').count();
            depth -= line.matches('}').count();
        }
        assert_eq!(depth, 0);

        assert!(source.contains("__shared__ unsigned int tile[16][17]"));
        assert!(source.contains("if (blockDim.x != 16 || blockDim.y != 16 || blockDim.z != 1"));
        assert!(source.contains("gridDim.y != 1 || gridDim.z != 1 || gridDim.x == 0) return;"));
        assert_eq!(source.matches("return;").count(), 1);
        assert!(source.contains("tile[threadIdx.y][threadIdx.x] = input[at]"));
        assert!(source.contains("output[at] = tile[threadIdx.x][threadIdx.y]"));
        assert!(source.contains("}\n    __syncthreads();"));
        assert!(source.contains(&format!(
            "if (y < {}ull && x < {}ull)",
            dimensions[0], dimensions[1]
        )));
        assert!(source.contains(&format!(
            "if (out_row < {}ull && out_col < {}ull)",
            dimensions[1], dimensions[0]
        )));
        assert!(source.contains("unsigned long long"));
        assert!(!source.contains("float"));
        assert!(!source.contains("__fm"));
        // A separate logical tile model checks each store's source cell was
        // loaded, each physical address is in bounds, and output is covered
        // exactly once, including edge tiles and grid-strided reuse.
        let [rows, cols] = dimensions.map(|n| usize::try_from(n).unwrap());
        let tile_cols = cols.div_ceil(16);
        let tile_rows = rows.div_ceil(16);
        for grid in [1, 3, 17] {
            let mut visits = vec![0; rows * cols];
            for block in 0..grid {
                for t in (block..tile_cols * tile_rows).step_by(grid) {
                    let (tr, tc) = (t / tile_cols, t % tile_cols);
                    let mut shared = [[None; 17]; 16];
                    for (ty, line) in shared.iter_mut().enumerate() {
                        for (tx, cell) in line.iter_mut().take(16).enumerate() {
                            let (r, c) = (tr * 16 + ty, tc * 16 + tx);
                            if r < rows && c < cols {
                                let at = 3 + r * cols + c;
                                assert!(at < p.ir().input.elements as usize);
                                *cell = Some(pattern(at));
                            }
                        }
                    }
                    for ty in 0..16 {
                        for (tx, line) in shared.iter().enumerate() {
                            let (r, c) = (tc * 16 + ty, tr * 16 + tx);
                            if r < cols && c < rows {
                                let out = 7 + r * rows + c;
                                assert!(out < p.ir().output.elements as usize);
                                assert_eq!(line[ty], Some(pattern(3 + c * cols + r)));
                                visits[r * rows + c] += 1;
                            }
                        }
                    }
                }
            }
            assert!(visits.iter().all(|n| *n == 1));
        }
    }
}

#[test]
fn issue_m4_02_transpose_artifact_identity_and_exact_options() {
    let ir = plan([3, 5], [1, 2], 5);
    let p = admit(ir.clone());
    let cuda = emit_cuda(&p).unwrap();
    let cpu = emit_cpu(&p).unwrap();
    assert!(cuda.contains(transpose::KERNEL_NAME));
    assert!(cpu.contains(transpose::KERNEL_NAME));
    for opt in p.nvrtc_options() {
        assert!(cuda.contains(opt));
    }
    for opt in p.cpu_options() {
        assert!(cpu.contains(opt));
    }
    for change in 0..12 {
        let mut changed = ir.clone();
        match change {
            0 => changed.span.end += 1,
            1 => changed.function_contract[0] ^= 1,
            2 => changed.input.id += 10,
            3 => changed.output.id += 10,
            4 => changed.input.alignment = 16,
            5 => changed.output.alignment = 16,
            6 => changed.input.offset += 1,
            7 => changed.output.offset += 1,
            8 => changed.input.elements += 1,
            9 => changed.output.elements += 1,
            10 => {
                changed.source_identity += 1;
            }
            11 => {
                changed.graph_revision += 1;
            }
            _ => unreachable!(),
        }
        let origin = Provenance {
            source_identity: changed.source_identity,
            graph_revision: changed.graph_revision,
        };
        let changed = verify(changed, Capabilities::cuda_12_9_sm89(), origin).unwrap();
        assert_ne!(emit_cuda(&changed).unwrap(), cuda);
        assert_ne!(emit_cpu(&changed).unwrap(), cpu);
    }
    let different = verify(
        ir,
        Capabilities::Cuda {
            major: 12,
            minor: 4,
            sm: 86,
        },
        origin(),
    )
    .unwrap();
    assert_ne!(emit_cuda(&different).unwrap(), cuda);
    assert!(emit_cuda(&admit(plan([3, 6], [1, 2], 5))).unwrap() != cuda);
}

struct Temp(PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn cpu_run(
    p: &Verified,
    second: Option<&Verified>,
    harness: &str,
    optimization: &str,
    label: &str,
) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "invar-transpose-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).unwrap();
    let temp = Temp(dir);
    let mut source = emit_cpu(p).unwrap();
    if let Some(second) = second {
        source.push_str(
            &emit_cpu(second)
                .unwrap()
                .replace(transpose::KERNEL_NAME, "invar_transpose_roundtrip_v1"),
        );
    }
    source.push_str(harness);
    fs::write(temp.0.join("transpose.cpp"), &source).unwrap();
    let flags = [
        optimization,
        "-fsanitize=undefined",
        "-fno-sanitize-recover=all",
        "-Wall",
        "-Wextra",
        "-Werror",
    ];
    let flags = [p.cpu_options(), &flags].concat();
    child_deadline::cxx(&flags, &temp.0.join("transpose.cpp"), &temp.0.join("run"));
    let output = child_deadline::command(10)
        .arg(temp.0.join("run"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{label}: run {:?}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    if let Some(dir) = std::env::var_os("INVAR_TRANSPOSE_REPORT_DIR") {
        let dir =
            PathBuf::from(dir).join(format!("{label}-{}", optimization.trim_start_matches('-')));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cpu.cpp"), source).unwrap();
        fs::write(dir.join("cuda.cu"), emit_cuda(p).unwrap()).unwrap();
        fs::write(dir.join("cpu.stdout"), &output.stdout).unwrap();
    }
    String::from_utf8(output.stdout).unwrap()
}
fn expected(input: &[u32], p: &Verified) -> Vec<u32> {
    let mut result = vec![0xcafebabe; usize::try_from(p.ir().output.elements).unwrap()];
    let [rows, cols] = p.ir().dimensions.map(|n| usize::try_from(n).unwrap());
    // Output-major traversal independently maps inverse (r,c) to input (c,r).
    for r in 0..cols {
        for c in 0..rows {
            result[p.ir().output.offset as usize + r * rows + c] =
                input[p.ir().input.offset as usize + c * cols + r];
        }
    }
    result
}
fn binary(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|n| n.to_le_bytes()).collect()
}

#[test]
fn issue_m4_02_transpose_stencil_schedules_cpu_bit_transpose_roundtrip() {
    for (dimensions, offsets, padding) in [
        ([0, 33], [0, 0], 0),
        ([33, 0], [2, 5], 7),
        ([0, 0], [1, 3], 3),
        ([1, 1], [0, 0], 0),
        ([31, 33], [1, 2], 3),
        ([127, 257], [3, 7], 5),
        ([16, 16], [0, 0], 0),
        ([17, 17], [5, 9], 7),
        ([64, 64], [1, 3], 3),
        ([1, 37], [1, 5], 7),
        ([37, 1], [3, 1], 3),
        ([31, 33], [7, 11], 13),
    ] {
        let p = admit(plan(dimensions, offsets, padding));
        let mut reversed = plan(
            [dimensions[1], dimensions[0]],
            [offsets[1], offsets[0]],
            padding,
        );
        reversed.input = p.ir().output.clone();
        reversed.output.id = 3;
        let reversed = admit(reversed);
        let input = (0..usize::try_from(p.ir().input.elements).unwrap())
            .map(pattern)
            .collect::<Vec<_>>();
        let wanted = expected(&input, &p);
        let roundtrip = expected(&wanted, &reversed);
        let mut harness = String::from(
            "\n#include <vector>\n#include <iostream>\n#include <cstring>\nint main(){\n",
        );
        writeln!(
            harness,
            "std::vector<std::uint32_t>A({}ull),B({}ull,0xcafebabeu),C({}ull,0xcafebabeu);",
            p.ir().input.elements,
            p.ir().output.elements,
            reversed.ir().output.elements
        )
        .unwrap();
        harness.push_str("const std::uint32_t edge[]={0u,0x80000000u,1u,0x007fffffu,0x00800000u,0x7f7fffffu,0x7f800000u,0xff800000u,0x7f800001u,0x7fc12345u,0xffc54321u,0xffffffffu,0x3f800000u,0xbf800000u,0x12345678u,0xdeadbeefu};\nfor(std::size_t i=0;i<A.size();++i)A[i]=i<16?edge[i]:(static_cast<std::uint32_t>(i)*0x9e3779b9u)^0xa5a5a5a5u;\nauto before=A;\nif(invar_transpose_bits_v1(A.data(),B.data())!=0)return 10;\nif(invar_transpose_roundtrip_v1(B.data(),C.data())!=0)return 11;\nif(A!=before)return 12;\n");
        if dimensions.contains(&0) {
            harness.push_str("if(invar_transpose_bits_v1(nullptr,nullptr)!=0)return 13;\n");
        }
        harness.push_str("for(auto v:B){std::cout<<v<<' ';}std::cout<<'\\n';for(auto v:C){std::cout<<v<<' ';}std::cout<<'\\n';return 0;}\n");
        let label = format!(
            "{}x{}-offset{}-{}-pad{}",
            dimensions[0], dimensions[1], offsets[0], offsets[1], padding
        );
        for opt in ["-O0", "-O2"] {
            let output = cpu_run(&p, Some(&reversed), &harness, opt, &label);
            let lines = output.lines().collect::<Vec<_>>();
            assert_eq!(lines.len(), 2);
            let parse = |s: &str| {
                s.split_whitespace()
                    .map(|n| n.parse::<u32>().unwrap())
                    .collect::<Vec<_>>()
            };
            let actual = parse(lines[0]);
            let actual_round = parse(lines[1]);
            assert_eq!(actual, wanted, "{label} transpose bits/padding");
            assert_eq!(actual_round, roundtrip, "{label} roundtrip bits/padding");
            if let Some(dir) = std::env::var_os("INVAR_TRANSPOSE_REPORT_DIR") {
                let dir =
                    PathBuf::from(dir).join(format!("{label}-{}", opt.trim_start_matches('-')));
                for (name, words) in [
                    ("input", &input),
                    ("expected", &wanted),
                    ("actual", &actual),
                    ("roundtrip-expected", &roundtrip),
                    ("roundtrip-actual", &actual_round),
                ] {
                    fs::write(dir.join(format!("{name}.u32le")), binary(words)).unwrap();
                }
            }
            eprintln!(
                "{opt} {label}: exact {} output / {} roundtrip words, immutable input",
                actual.len(),
                actual_round.len()
            );
        }
    }
}

#[test]
fn issue_m4_02_transpose_cpu_refuses_runtime_overlap_alignment_and_null_before_writes() {
    let p = admit(plan([3, 5], [1, 2], 3));
    let harness = r#"
#include <vector>
#include <cstring>
#include <limits>
int main(){
std::vector<std::uint32_t>A(64,0xdeadbeefu),B(64,0xcafebabeu);
auto beforeA=A,beforeB=B;
if(invar_transpose_bits_v1(nullptr,B.data())!=2)return 1;
if(invar_transpose_bits_v1(A.data(),nullptr)!=2)return 2;
if(invar_transpose_bits_v1(A.data(),A.data())!=2)return 3;
if(invar_transpose_bits_v1(A.data(),A.data()+1)!=2)return 4;
if(invar_transpose_bits_v1(A.data()+1,A.data())!=2)return 5;
auto misaligned=reinterpret_cast<std::uint32_t*>(reinterpret_cast<unsigned char*>(B.data())+1);
if(invar_transpose_bits_v1(A.data(),misaligned)!=2)return 6;
if(invar_transpose_bits_v1(misaligned,B.data())!=2)return 7;
auto high=reinterpret_cast<std::uint32_t*>(UINTPTR_MAX&~std::uintptr_t(3));
if(invar_transpose_bits_v1(high,B.data())!=2)return 8;
if(invar_transpose_bits_v1(A.data(),high)!=2)return 9;
if(A!=beforeA||B!=beforeB)return 10;
// Adjacent whole allocations are disjoint, even in one larger host array.
if(invar_transpose_bits_v1(A.data(),A.data()+19)!=0)return 11;
return 0;}
"#;
    for opt in ["-O0", "-O2"] {
        cpu_run(&p, None, harness, opt, "runtime-pointer-negatives");
    }
    let mut ir = plan([1, 1], [0, 0], 0);
    ir.input.alignment = 64;
    ir.output.alignment = 64;
    let p = admit(ir);
    let harness = r#"
int main(){alignas(64) std::uint32_t A[64]={0x7f800001u},B[64]={};
if(invar_transpose_bits_v1(A,B)!=0||B[0]!=A[0])return 1;
B[0]=0xdeadbeefu;
if(invar_transpose_bits_v1(A+1,B)!=2||B[0]!=0xdeadbeefu)return 2;
if(invar_transpose_bits_v1(A,B+1)!=2||B[0]!=0xdeadbeefu)return 3;
return 0;}
"#;
    for opt in ["-O0", "-O2"] {
        cpu_run(&p, None, harness, opt, "runtime-declared-alignment");
    }
}
