//! CPU execution model of emitted ABI12 (`cuda-owned-numeric-continuation-v1`)
//! CUDA text. The exact emitted kernel `invar_gnet_owned_numeric_resume_v1` is
//! compiled with a minimal C++ shim and launched over every lane of full
//! 32-thread blocks in host-fenced quantum epochs. After every epoch each lane's
//! task bytes, output, trace, heap and cleanup must equal the ABI11 CPU
//! artifact's own `gnet_resume`; tails and a red zone must keep sentinels.
//! Terminal values are checked against the independent net oracle. This is
//! never NVRTC compilation, device execution, host support or a performance
//! claim; ABI12 has no runtime loader, and every execution entry refuses it.
//! Issue label GNET-12.
#![cfg(unix)]

use crate::epoch_model::{self, build, expected, lowered};

use invar::{
    eval, front,
    general_net::{
        Net, compiled,
        device_host::{Backend, cpu},
        oracle,
        provenance::{self, FenceBackend},
    },
    numeric_contract::F32Bits,
};
use std::{collections::BTreeSet, path::Path};

const KERNEL: &str = "invar_gnet_owned_numeric_resume_v1";

/// Lane-by-lane epoch model of the header-free ABI12 CUDA text against the ABI11 CPU text.
fn harness(
    cpu: &compiled::Artifact,
    dev: &compiled::Artifact,
    roots: &[Vec<u32>],
    l: oracle::Limits,
) -> String {
    assert_eq!(cpu.abi_version(), 11);
    assert_eq!(dev.abi_version(), 12);
    assert!(
        !dev.source().lines().any(|l| l.starts_with("#include")),
        "ABI12 text must be header-free"
    );
    epoch_model::harness(cpu, dev, KERNEL, roots, l)
}

const SPECIAL: [u32; 20] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x0080_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f80_0001,
    0x4b80_0000,
    0xcb80_0000,
    0x7f7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0x7fa1_2345,
    0xffc0_0001,
    0x3400_0000,
    0x3eaa_aaab,
    0x4040_0000,
];

struct Case {
    name: &'static str,
    source: &'static str,
    entry: &'static str,
    roots: Vec<Vec<u32>>,
    trace_events: usize,
    transitions: u64,
    storage: compiled::OwnedLimits,
    /// Roots (by first word) whose backend storage status has no oracle analogue.
    backend: &'static [(u32, i64)],
}
const STORAGE: compiled::OwnedLimits = compiled::OwnedLimits {
    elements: 64,
    allocations: 16,
    cleanup_events: 32,
};
const SPECIAL_SOURCE: &str = "fn mix(xs: mut Array<f32>, n: u32) -> Unit { if n > 0u32 { xs[0u32] = fma(xs[0u32], xs[1u32], xs[2u32]); mix(xs, n - 1u32); } } fn run(a: f32, b: f32, c: f32, n: u32) -> f32 { var xs: Array<f32> = [a, b, c]; mix(xs, n % 4u32); let t: f32 = xs[0u32]; if t < b { return t / b; } if t == t { return t - c + f32_from_u32(n) - f32_from_i32(0 - 3); } return xs[n]; } fn main() -> i32 { return 0; }";
const LIMITS_SOURCE: &str = "fn deep(x: f32) -> f32 { return deep(x + 1.0); } fn run(a: f32, n: u32) -> f32 { if n == 0u32 { return deep(a); } if n == 1u32 { let big: Array<f32> = filled_f32(16777217u32, a); return big[0u32]; } if n == 2u32 { var k: i32 = 2147483647; k = k + 1; return a; } if n == 3u32 { let z: u32 = n - 3u32; let q: u32 = 7u32 / z; return a; } if n == 4u32 { while true {} } var s: f32 = a; for i in 0u32..n { s = fma(s, a, a); } return s; } fn main() -> i32 { return 0; }";
const STORAGE_SOURCE: &str = "fn run(a: f32, n: u32) -> f32 { if n == 0u32 { let wide: Array<f32> = filled_f32(9u32, a); return wide[0u32]; } if n == 1u32 { for i in 0u32..3u32 { let t: Array<f32> = filled_f32(1u32, a); } return a; } if n == 2u32 { let p: Array<f32> = [a]; let q: Array<f32> = [a]; let r: Array<f32> = [a]; let s: Array<f32> = [a]; let u: Array<f32> = [a]; return a; } return a; } fn main() -> i32 { return 0; }";

fn special_roots() -> Vec<Vec<u32>> {
    // 40 roots (two CUDA blocks): special-bit triples with n covering every
    // recursion depth, NaN-payload copies (n < 3) and index traps (n >= 3).
    (0..40u32)
        .map(|i| {
            let s = |k: u32| SPECIAL[((i * 7 + k * 3) % SPECIAL.len() as u32) as usize];
            vec![s(0), s(1), s(2), i % 6]
        })
        .collect()
}
fn cases() -> Vec<Case> {
    let one = 0x3f80_0000u32;
    vec![
        Case {
            name: "special",
            source: SPECIAL_SOURCE,
            entry: "run",
            roots: special_roots(),
            trace_events: 2000,
            transitions: 10_000,
            storage: STORAGE,
            backend: &[],
        },
        Case {
            name: "limits",
            source: LIMITS_SOURCE,
            entry: "run",
            roots: (0..6u32)
                .map(|n| vec![one, n])
                .chain([vec![0x7fa1_2345, 9], vec![0x8000_0001, 7]])
                .collect(),
            trace_events: 10_000,
            transitions: 10_000,
            storage: STORAGE,
            backend: &[],
        },
        Case {
            name: "transitions",
            source: LIMITS_SOURCE,
            entry: "run",
            roots: vec![vec![one, 5], vec![one, 50]],
            trace_events: 10_000,
            transitions: 60,
            storage: STORAGE,
            backend: &[],
        },
        Case {
            name: "trace",
            source: LIMITS_SOURCE,
            entry: "run",
            roots: vec![vec![one, 5], vec![one, 50]],
            trace_events: 30,
            transitions: 10_000,
            storage: STORAGE,
            backend: &[],
        },
        Case {
            name: "storage",
            source: STORAGE_SOURCE,
            entry: "run",
            roots: (0..4u32).map(|n| vec![one, n]).collect(),
            trace_events: 2000,
            transitions: 10_000,
            storage: compiled::OwnedLimits {
                elements: 8,
                allocations: 4,
                cleanup_events: 2,
            },
            backend: &[(0, 11), (1, 13), (2, 12)],
        },
    ]
}
impl Case {
    fn limits(&self) -> oracle::Limits {
        oracle::Limits {
            fuel: 1000,
            transitions: self.transitions,
            trace_events: self.trace_events,
            array_elements: self.storage.elements,
        }
    }
    fn artifacts(&self, net: &Net) -> (compiled::Artifact, compiled::Artifact) {
        let proof = net.verify().unwrap();
        let emit = |target| {
            compiled::emit_resumable_owned_numeric(
                net,
                &proof,
                self.entry,
                target,
                compiled::Limits::default(),
                self.storage,
            )
            .unwrap()
        };
        (emit(compiled::Target::Cpu), emit(compiled::Target::Cuda))
    }
    fn args(&self, root: &[u32], net: &Net) -> Vec<eval::Value> {
        let proof = net.verify().unwrap();
        let mir = net.compiled_mir(&proof).unwrap();
        let f = mir.function(self.entry).unwrap();
        f.params
            .iter()
            .zip(root)
            .map(|(p, w)| match p.ty {
                front::Type::F32 => eval::Value::F32(F32Bits::from_bits(*w)),
                front::Type::U32 => eval::Value::U32(*w),
                ref other => panic!("{other:?}"),
            })
            .collect()
    }
}

#[test]
fn model_matches_abi11_every_epoch_and_oracle() {
    let mut statuses = BTreeSet::new();
    let mut observed = Vec::new();
    for case in cases() {
        let l = case.limits();
        let net = lowered(case.source);
        let (cpu, dev) = case.artifacts(&net);
        let want: Vec<_> = case
            .roots
            .iter()
            .map(|root| {
                if case.backend.iter().any(|(n, _)| *n == root[root.len() - 1]) {
                    None
                } else {
                    Some(expected(&net, case.entry, &case.args(root, &net), l))
                }
            })
            .collect();
        let name = format!("abi12-{}", case.name);
        let model = build(&name, &harness(&cpu, &dev, &case.roots, l));
        // Sampled schedules: a yield after every step, and a coarse quantum.
        for quantum in [1, 17] {
            let label = format!("{}-{quantum}", case.name);
            let run = model.execute(&label, quantum);
            assert_eq!(run.outputs.len(), case.roots.len(), "{label}");
            for (lane, root) in case.roots.iter().enumerate() {
                let output = &run.outputs[lane];
                match &want[lane] {
                    Some((header, trace)) => {
                        assert_eq!(&output[..9], &header[..], "{label} root {lane} {root:x?}");
                        assert_eq!(&run.traces[lane], trace, "{label} root {lane}");
                    }
                    None => {
                        let n = root[root.len() - 1];
                        let (_, status) = case.backend.iter().find(|(r, _)| *r == n).unwrap();
                        assert_eq!(output[0], *status, "{label} root {lane}");
                    }
                }
                statuses.insert(output[0]);
            }
            assert!(run.pending.windows(2).all(|w| w[1] <= w[0]), "{label}");
            assert_eq!(run.pending.first(), Some(&(case.roots.len() as u64)));
            assert_eq!(run.pending.last(), Some(&0));
            assert_eq!(run.pending.len() as u64, run.epochs + 1);
            if case.name == "special" {
                special_bits(&case.roots, &run.outputs);
            }
            observed.push((label, run.epochs));
        }
        // An invalid quantum finishes every lane in one epoch with status 8.
        let invalid: &[u64] = if case.name == "special" {
            &[0, 1_000_001]
        } else {
            &[]
        };
        for &quantum in invalid {
            let run = model.execute(&format!("{}-status8-{quantum}", case.name), quantum);
            assert_eq!(run.epochs, 1);
            assert_eq!(run.pending, [case.roots.len() as u64, 0]);
            assert!(run.outputs.iter().all(|o| o[0] == 8 && o[7] == 0));
        }
    }
    // Success; overflow, division and index traps; fuel, call-depth,
    // transition, trace and source array-size limits; backend element,
    // allocation-slot and cleanup limits. Status 7 has no ordinary-source
    // producer; status 8 (invalid quantum) is checked on the special case above.
    assert_eq!(statuses, [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13].into());
    eprintln!("ABI12 model runs (label, epochs): {observed:?}");
}

/// Terminal f32 results include NaN-payload copies (moves are bit-exact) and
/// canonical NaN from arithmetic, both read back from the device model.
fn special_bits(roots: &[Vec<u32>], outputs: &[Vec<i64>]) {
    let (mut payload_copies, mut canonical, mut traps) = (0, 0, 0);
    for (root, o) in roots.iter().zip(outputs) {
        match o[0] {
            0 => {
                let bits = o[1] as u32;
                if bits == 0x7fc0_0000 {
                    canonical += 1;
                } else if F32Bits::from_bits(bits).is_nan() {
                    assert!(
                        root[..3].contains(&bits),
                        "non-canonical NaN must be a copy"
                    );
                    payload_copies += 1;
                }
            }
            9 => traps += 1,
            other => panic!("unexpected status {other} for {root:x?}"),
        }
    }
    assert!(payload_copies > 0 && canonical > 0 && traps > 0);
}

/// ABI12 is a compile receipt only: every fence backend and the CPU device
/// model refuse it explicitly, the model before any compiler call.
#[test]
fn execution_entries_refuse_abi12() {
    let l = provenance::Limits::default();
    let source = SPECIAL_SOURCE.as_bytes();
    let abi12 = provenance::compile_resumable_owned_numeric(
        source,
        "run",
        compiled::Target::Cuda,
        l,
        STORAGE,
    )
    .unwrap();
    assert_eq!(abi12.artifact().abi_version(), 12);
    // Any admitted workset will do: ABI12 is refused before it is inspected.
    let children = "fn branch(depth: u32) -> i32 { if depth<2u32 { return 1; } return branch(depth-1u32)+branch(depth-2u32); } fn run(n: u32) -> i32 { return branch(n); }";
    let speculation = compiled::SpeculationLimits {
        slots: 2,
        fuel: 10000,
        transitions: 10000,
        trace_events: 1024,
        quantum: 32,
        lookahead: 128,
        scratch_bytes: 1024 * 1024,
    };
    let owned = compiled::OwnedLimits {
        elements: 16,
        allocations: 8,
        cleanup_events: 16,
    };
    let abi5 = provenance::compile_device_speculative(
        children.as_bytes(),
        "run",
        compiled::Target::Cuda,
        l,
        owned,
        speculation,
    )
    .unwrap();
    let config = compiled::DeviceConfig {
        fuel: 10000,
        transitions: 10000,
        trace_events: 1024,
        quantum: 32,
        rounds: 64,
        capacity: 2,
    };
    let layout = abi5.artifact().device_layout().unwrap();
    let bytes = layout.buffer_bytes(config).unwrap();
    let workset = layout
        .workset(
            config,
            std::array::from_fn(|i| compiled::BufferRange {
                device: 1,
                allocation: (i + 1) as u64,
                offset: 0,
                bytes: bytes[i],
            }),
        )
        .unwrap();
    for backend in [
        FenceBackend::CudaDriver,
        FenceBackend::MetalCommand,
        FenceBackend::CpuGroupModel,
    ] {
        let refused = abi12.fenced_execution(workset.clone(), (), 1, backend);
        assert!(
            matches!(refused, Err(provenance::Error::ExecutionMismatch)),
            "{backend:?}"
        );
    }
    // A compile attempt would fail differently: this compiler does not exist.
    let mut model = cpu::CpuModel::with_compiler(Path::new("/nonexistent/invar-abi12-cxx"));
    let prepared = model.prepare(abi12.artifact(), [8; 12], &[], [0; 6]);
    assert_eq!(prepared.err().as_deref(), Some(cpu::UNSUPPORTED_ABI12));
}
