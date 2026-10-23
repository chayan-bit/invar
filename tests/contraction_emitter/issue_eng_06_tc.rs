//! ENG-06 (#61) matrix-atom schedules through the generic contraction
//! emitter: the `tc_cuda_v1` and `tc_metal_v1` move sequences emit exactly the
//! `tc.rs` text on every built-in target, and a CPU emulation of the emitted
//! fragment index math covers every A, B and C element of each atom tile
//! exactly once and gives results admissible under `tc/reference.rs`.
//! Device rows are not run here.
#![forbid(unsafe_code)]

use crate::corpus;
use invar::contraction_source;

use std::collections::BTreeMap;

use contraction_source::{Contract, DeviceEmission};
use corpus::{DIRECT, SHAPES, TC_DIRECT, tc_cases};
use invar_device::contraction::{self, Error, atoms, gemm_tiled_v1};
use invar_device::moves::{Axis, Move, Refusal};
use invar_device::target::Registry;
use invar_device::{Capabilities, tc};

fn cuda(minor: u32, sm: u32) -> Capabilities {
    Capabilities::Cuda {
        major: 12,
        minor,
        sm,
    }
}

fn metal(apple_family: u32) -> Capabilities {
    Capabilities::Metal {
        apple_family,
        msl_major: 3,
        msl_minor: 1,
    }
}

fn emission(source: &str, dims: [u32; 3], caps: Capabilities) -> DeviceEmission {
    let [r, k, n] = dims.map(u64::from);
    Contract::from_source(source.as_bytes(), "mat")
        .unwrap()
        .emit_device(source.as_bytes(), dims, [r * k, k * n], caps)
        .unwrap()
}

#[test]
fn issue_eng_06_tc_moves_emit_exactly_the_tc_rs_text() {
    let registry = Registry::builtin().unwrap();
    let mut compared = 0;
    for caps in [
        cuda(4, 80),
        cuda(4, 86),
        cuda(4, 87),
        cuda(4, 89),
        cuda(4, 90),
        cuda(9, 120),
        metal(7),
        metal(8),
        metal(9),
    ] {
        let target = registry.select(caps).unwrap();
        for dims in SHAPES {
            let e = emission(TC_DIRECT, dims, caps);
            let plan = e.plan();
            let (generic, fixed, schedule) = match caps {
                Capabilities::Cuda { .. } => (
                    atoms::emit_cuda(plan, &atoms::tc_cuda_v1(), target).unwrap(),
                    tc::emit_cuda(plan).unwrap(),
                    atoms::tc_cuda_v1(),
                ),
                Capabilities::Metal { .. } => (
                    atoms::emit_msl(plan, &atoms::tc_metal_v1(), target).unwrap(),
                    tc::emit_msl(plan).unwrap(),
                    atoms::tc_metal_v1(),
                ),
            };
            assert_eq!(generic, fixed, "{caps:?} {dims:?}");
            assert_eq!(generic, e.source());
            let v = atoms::verify(plan, &schedule, target).unwrap();
            assert_eq!(atoms::geometry(plan, &v), tc::geometry(plan));
            compared += 1;
        }
    }
    assert_eq!(compared, 9 * SHAPES.len());
}

/// The (A, B, C) coordinates one lane of the m16n8k16 lowering touches in a
/// k-tile, copied from the emitted fragment code (`CUDA_K_LOOP`, `CUDA_TAIL`).
type Fragment = (Vec<(u64, u64)>, Vec<(u64, u64)>, Vec<(u64, u64)>);

fn mma_fragment(lane: u64, row0: u64, col0: u64, k0: u64) -> Fragment {
    let (g, q) = (lane / 4, lane % 4);
    let (r0, r1, ka, n) = (row0 + g, row0 + g + 8, k0 + 2 * q, col0 + g);
    let a = vec![
        (r0, ka),
        (r0, ka + 1),
        (r1, ka),
        (r1, ka + 1),
        (r0, ka + 8),
        (r0, ka + 9),
        (r1, ka + 8),
        (r1, ka + 9),
    ];
    let b = vec![(ka, n), (ka + 1, n), (ka + 8, n), (ka + 9, n)];
    let j = col0 + 2 * q;
    let c = vec![(r0, j), (r0, j + 1), (r1, j), (r1, j + 1)];
    (a, b, c)
}

#[test]
fn issue_eng_06_tc_fragments_cover_each_atom_tile_exactly_once() {
    let registry = Registry::builtin().unwrap();
    let sm90 = registry.get("cuda-sm90").unwrap();
    let e = emission(TC_DIRECT, [33, 40, 17], cuda(4, 90));
    let text = atoms::emit_cuda(e.plan(), &atoms::tc_cuda_v1(), sm90).unwrap();
    // The emulated formulas are the emitted ones.
    for line in [
        "g = lane / 4u, q = lane % 4u",
        "r0 = row0 + g, r1 = row0 + g + 8u, ka = k0 + 2u * q, n = col0 + g;",
        "invar_a(A, r0, ka + 8u), x5 = invar_a(A, r0, ka + 9u);",
        "invar_b(B, ka + 8u, n), y3 = invar_b(B, ka + 9u, n);",
        "j = col0 + 2u * q;",
    ] {
        assert!(text.contains(line), "{line}");
    }
    let (mut a, mut b, mut c) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    for lane in 0..32 {
        let (fa, fb, fc) = mma_fragment(lane, 0, 0, 0);
        for p in fa {
            *a.entry(p).or_insert(0) += 1;
        }
        for p in fb {
            *b.entry(p).or_insert(0) += 1;
        }
        for p in fc {
            *c.entry(p).or_insert(0) += 1;
        }
    }
    let full = |map: &BTreeMap<(u64, u64), u32>, rows: u64, cols: u64| {
        map.len() as u64 == rows * cols
            && map.values().all(|&n| n == 1)
            && map.keys().all(|&(r, c)| r < rows && c < cols)
    };
    assert!(full(&a, 16, 16), "A fragment");
    assert!(full(&b, 16, 8), "B fragment");
    assert!(full(&c, 16, 8), "C fragment");
}

/// Emulate the CUDA lowering through the fragment index math: per atom tile
/// and k-tile, the 32 lanes' A and B fragments (bounds-checked loads, zero
/// outside the matrix) form the 16x16 and 16x8 operand tiles; each lane's C
/// cells take the mma sum of the products of zero and normal binary16
/// operands (increasing k here, one admissible order) and the correction
/// sum of every product with a subnormal operand. Each stored cell is the
/// mma sum plus the correction, one nearest addition.
fn emulate_mma([rows, inner, cols]: [u32; 3], a: &[u32], b: &[u32]) -> Vec<u32> {
    let [rows, inner, cols] = [rows, inner, cols].map(u64::from);
    let sub = |w: u32| {
        let m = w & 0x7fff_ffff;
        m != 0 && m < 0x3880_0000
    };
    let load_a = |i: u64, k: u64| {
        if i < rows && k < inner {
            a[(i * inner + k) as usize]
        } else {
            0
        }
    };
    let load_b = |k: u64, j: u64| {
        if k < inner && j < cols {
            b[(k * cols + j) as usize]
        } else {
            0
        }
    };
    let mut out = vec![0u32; (rows * cols) as usize];
    let tiles_n = cols.div_ceil(8);
    for tile in 0..rows.div_ceil(16) * tiles_n {
        let (row0, col0) = (tile / tiles_n * 16, tile % tiles_n * 8);
        let (mut acc, mut fix) = (BTreeMap::new(), BTreeMap::new());
        for k0 in (0..inner).step_by(16) {
            let (mut ta, mut tb) = (BTreeMap::new(), BTreeMap::new());
            for lane in 0..32 {
                let (fa, fb, _) = mma_fragment(lane, row0, col0, k0);
                ta.extend(fa.into_iter().map(|(i, k)| ((i, k), load_a(i, k))));
                tb.extend(fb.into_iter().map(|(k, j)| ((k, j), load_b(k, j))));
            }
            for lane in 0..32 {
                for (i, j) in mma_fragment(lane, row0, col0, k0).2 {
                    let (mut mma, mut s) = (0.0f32, 0.0f32);
                    for k in k0..k0 + 16 {
                        let (x, y) = (ta[&(i, k)], tb[&(k, j)]);
                        let p = f32::from_bits(x) * f32::from_bits(y);
                        if sub(x) || sub(y) { s += p } else { mma += p }
                    }
                    *acc.entry((i, j)).or_insert(0.0f32) += mma;
                    *fix.entry((i, j)).or_insert(0.0f32) += s;
                }
            }
        }
        for ((i, j), c) in acc {
            if i < rows && j < cols {
                out[(i * cols + j) as usize] = (c + fix[&(i, j)]).to_bits();
            }
        }
    }
    out
}

#[test]
fn issue_eng_06_tc_emulated_results_are_admissible_under_the_reference() {
    let mut checked = 0;
    for (label, dims, a, b) in tc_cases() {
        if dims[1] > 4096 {
            continue; // The 2^20 counterexample is covered by the device suites.
        }
        let (a, b) = (tc::round_inputs(0, &a), tc::round_inputs(1, &b));
        let (Ok(a), Ok(b)) = (a, b) else {
            continue; // Refused before any work, as on the device.
        };
        let out = emulate_mma(dims, &a, &b);
        let report =
            tc::verify(dims, &a, &b, &out).unwrap_or_else(|f| panic!("{label} {dims:?}: {f:?}"));
        assert_eq!(report.cells, (dims[0] * dims[2]) as usize);
        checked += 1;
    }
    assert!(checked >= 15, "{checked}");
}

#[test]
fn issue_eng_06_tc_refuses_what_it_does_not_lower() {
    let registry = Registry::builtin().unwrap();
    let sm90 = registry.get("cuda-sm90").unwrap();
    let apple9 = registry.get("metal-apple9").unwrap();
    let tc_plan = emission(TC_DIRECT, [65, 17, 63], cuda(4, 90));
    let strict = emission(DIRECT, [65, 17, 63], cuda(4, 90));
    let plan = tc_plan.plan();
    // A strict plan never takes a matrix atom, and the gemm sequence cannot
    // claim the matrix contract.
    assert_eq!(
        atoms::emit_cuda(strict.plan(), &atoms::tc_cuda_v1(), sm90),
        Err(Error::Plan(invar_device::floating::Error::Numeric))
    );
    assert_eq!(
        atoms::emit_cuda(plan, &gemm_tiled_v1(), sm90),
        Err(Error::Schedule(Refusal::Contract))
    );
    // The Metal atom does not exist on a CUDA target, and the target must
    // describe the plan's device.
    assert!(matches!(
        atoms::emit_cuda(plan, &atoms::tc_metal_v1(), sm90),
        Err(Error::Schedule(Refusal::AtomUnknown(_)))
    ));
    assert_eq!(
        atoms::emit_msl(plan, &atoms::tc_metal_v1(), apple9),
        Err(Error::Target)
    );
    // A plan and its own target still emit only their own language, so a
    // Metal atom never lands in the CUDA template.
    let tc_metal = emission(TC_DIRECT, [65, 17, 63], metal(9));
    assert_eq!(
        atoms::emit_cuda(tc_metal.plan(), &atoms::tc_metal_v1(), apple9),
        Err(Error::Plan(invar_device::floating::Error::Capabilities))
    );
    assert_eq!(
        atoms::emit_msl(plan, &atoms::tc_cuda_v1(), sm90),
        Err(Error::Plan(invar_device::floating::Error::Capabilities))
    );
    // A data-only atom for another contract.
    let mut bf16 = atoms::tc_cuda_v1();
    bf16.moves[1] = Move::UseAtom {
        atom: "mma-sync-m16n8k16-bf16-f32".to_owned(),
    };
    assert!(matches!(
        atoms::emit_cuda(plan, &bf16, sm90),
        Err(Error::Schedule(Refusal::AtomContract { .. }))
    ));
    // Verified, but not one atom per warp tile: this lowering refuses it.
    let mut wide = atoms::tc_cuda_v1();
    wide.moves[0] = Move::Tile {
        axes: vec![Axis::M, Axis::N, Axis::K],
        sizes: vec![32, 16, 16],
    };
    assert_eq!(
        atoms::emit_cuda(plan, &wide, sm90),
        Err(Error::Unsupported("atom-lowering"))
    );
    // And the strict emitter refuses the matrix sequence.
    assert!(contraction::emit_cuda(plan, &atoms::tc_cuda_v1(), sm90).is_err());
}
