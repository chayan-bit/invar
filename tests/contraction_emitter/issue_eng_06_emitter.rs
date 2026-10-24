//! ENG-06 (#61) generic contraction emitter: the move sequence that
//! reproduces `invar-f32-gemm-tiled-v1` emits exactly the `gemm.rs` text for
//! every corpus shape on every built-in GPU target. A CPU emulation of the
//! emitted index math, for that sequence and for other verified sequences,
//! equals the strict oracle bit for bit on zero, one, odd, rectangular,
//! signed-zero and subnormal shapes. Device rows are not run here.
#![forbid(unsafe_code)]

use crate::corpus;
use invar::contraction_source;

use contraction_source::{Contract, DeviceEmission};
use corpus::{
    DIRECT, NORMAL, SHAPES, SUBNORMAL, TC_DIRECT, oracle, random_words, strict_equal, words,
};
use invar_device::contraction::{self, Error, gemm_tiled_v1};
use invar_device::moves::{self, Axis, Layout, Move, Operand, Schedule};
use invar_device::target::{Level, Registry, Target};
use invar_device::{Capabilities, floating, gemm};

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

/// Every built-in GPU target with capabilities that select it.
fn devices() -> Vec<(Capabilities, &'static Target)> {
    let registry = Registry::builtin().unwrap();
    let mut out = Vec::new();
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
        out.push((caps, registry.select(caps).unwrap()));
    }
    out
}

fn emission(source: &str, dims: [u32; 3], caps: Capabilities) -> DeviceEmission {
    let [r, k, n] = dims.map(u64::from);
    Contract::from_source(source.as_bytes(), "mat")
        .unwrap()
        .emit_device(source.as_bytes(), dims, [r * k, k * n], caps)
        .unwrap()
}

/// Shapes beyond the corpus: one tile minus one, two tiles plus one in every
/// axis, and K below one slice.
const EXTRA: [[u32; 3]; 4] = [[63, 15, 63], [129, 33, 129], [3, 2, 200], [200, 5, 3]];

/// Run the emitted kernel's index math on the CPU: every group in launch
/// order, every thread of the load phase, then of the multiply-add phase, per
/// k slice, with the kernel's masks and bounds checks (a failed check is the
/// kernel's `__trap`). Inputs and output are whole allocations of binary32
/// words; the returned output starts as `c`.
fn emulate(
    plan: &floating::Verified,
    v: &moves::Verified,
    a: &[u32],
    b: &[u32],
    c: &[u32],
) -> Result<Vec<u32>, &'static str> {
    let ir = plan.ir();
    let [rows, inner, cols] = ir.dimensions.map(u64::from);
    let ([tm, tn, tk], [fm, fnn]) = (v.tile.map(u64::from), v.micro.map(u64::from));
    let (rm, rn, threads) = (tm / fm, tn / fnn, u64::from(v.threads));
    let (la, lb) = (layout(v, Operand::A), layout(v, Operand::B));
    let [ab, bb] = &ir.inputs;
    let cb = &ir.output;
    let load = |words: &[u32], buf: &floating::Buffer, row: u64, col: u64| {
        let index = buf.offset + row * buf.strides[0] + col;
        let word = if index < buf.elements {
            words.get(index as usize)
        } else {
            None
        };
        word.copied().ok_or("bounds")
    };
    // Staged (row, column) of an element of the A tile (m, k) or B tile (k, n).
    let a_at = |m: u64, k: u64| {
        if la.k_major {
            (k, m, tm + u64::from(la.pad))
        } else {
            (m, k, tk + u64::from(la.pad))
        }
    };
    let b_at = |k: u64, n: u64| {
        if lb.k_major {
            (n, k, tk + u64::from(lb.pad))
        } else {
            (k, n, tn + u64::from(lb.pad))
        }
    };
    let mut out = c.to_vec();
    let g = contraction::geometry(plan, v);
    // The staged tiles as declared: `As$AS$` and `Bs$BS$`.
    let mut as_ =
        vec![0u32; (tm * tk + u64::from(la.pad) * if la.k_major { tk } else { tm }) as usize];
    let mut bs =
        vec![0u32; (tk * tn + u64::from(lb.pad) * if lb.k_major { tn } else { tk }) as usize];
    for group in 0..u64::from(g.groups) {
        let mut tile_index = group;
        while tile_index < g.tiles {
            let tiles_n = cols.div_ceil(tn);
            let (row0, col0) = (tile_index / tiles_n * tm, tile_index % tiles_n * tn);
            let mut acc = vec![0.0f32; (threads * fm * fnn) as usize];
            let mut k0 = 0;
            while k0 < inner {
                for t in 0..threads {
                    for e in (t..tm * tk).step_by(threads as usize) {
                        let (i, k) = (row0 + e / tk, k0 + e % tk);
                        let w = if i < rows && k < inner {
                            load(a, ab, i, k)?
                        } else {
                            0
                        };
                        let (r, s, width) = a_at(e / tk, e % tk);
                        as_[(r * width + s) as usize] = w;
                    }
                    for e in (t..tk * tn).step_by(threads as usize) {
                        let (k, j) = (k0 + e / tn, col0 + e % tn);
                        let w = if k < inner && j < cols {
                            load(b, bb, k, j)?
                        } else {
                            0
                        };
                        let (r, s, width) = b_at(e / tn, e % tn);
                        bs[(r * width + s) as usize] = w;
                    }
                }
                let span = (inner - k0).min(tk);
                for t in 0..threads {
                    let (ty, tx) = (t / rn, t % rn);
                    for kk in 0..span {
                        for r in 0..fm {
                            let (ar, asc, aw) = a_at(ty + rm * r, kk);
                            let x = f32::from_bits(as_[(ar * aw + asc) as usize]);
                            for cc in 0..fnn {
                                let (br, bsc, bw) = b_at(kk, tx + rn * cc);
                                let y = f32::from_bits(bs[(br * bw + bsc) as usize]);
                                let slot = &mut acc[((t * fm + r) * fnn + cc) as usize];
                                *slot += x * y;
                            }
                        }
                    }
                }
                k0 += tk;
            }
            for t in 0..threads {
                let (ty, tx) = (t / rn, t % rn);
                for r in 0..fm {
                    for cc in 0..fnn {
                        let (i, j) = (row0 + ty + rm * r, col0 + tx + rn * cc);
                        if i < rows && j < cols {
                            let index = cb.offset + i * cb.strides[0] + j;
                            let slot = out.get_mut(index as usize).filter(|_| index < cb.elements);
                            *slot.ok_or("bounds")? =
                                acc[((t * fm + r) * fnn + cc) as usize].to_bits();
                        }
                    }
                }
            }
            tile_index += u64::from(g.groups);
        }
    }
    Ok(out)
}

fn layout(v: &moves::Verified, operand: Operand) -> Layout {
    v.staged.iter().find(|(o, _)| *o == operand).map_or(
        Layout {
            k_major: false,
            pad: 0,
        },
        |(_, l)| *l,
    )
}

fn tile(m: u32, n: u32, k: u32) -> Move {
    Move::Tile {
        axes: vec![Axis::M, Axis::N, Axis::K],
        sizes: vec![m, n, k],
    }
}

/// Strict scalar sequences other than the gemm.rs template: other tiles,
/// cells per thread, layouts and pads.
fn other_schedules() -> Vec<Schedule> {
    let seq =
        |name: &str, [m, n, k]: [u32; 3], [fm, fnn]: [u32; 2], a: Layout, b: Layout| Schedule {
            name: name.to_owned(),
            moves: vec![
                tile(m, n, k),
                Move::Split {
                    axis: Axis::M,
                    factor: fm,
                },
                Move::Split {
                    axis: Axis::N,
                    factor: fnn,
                },
                Move::Stage {
                    buffer: Operand::A,
                    level: Level::Shared,
                },
                Move::Permute {
                    buffer: Operand::A,
                    layout: a,
                },
                Move::Stage {
                    buffer: Operand::B,
                    level: Level::Shared,
                },
                Move::Permute {
                    buffer: Operand::B,
                    layout: b,
                },
                Move::Privatize { buffer: Operand::C },
            ],
            guards: Vec::new(),
        };
    let row = |pad| Layout {
        k_major: false,
        pad,
    };
    let kmaj = |pad| Layout { k_major: true, pad };
    vec![
        seq("eng-06-32x32x8", [32, 32, 8], [2, 2], row(0), row(0)),
        seq("eng-06-128x64x32", [128, 64, 32], [8, 4], kmaj(1), kmaj(1)),
        seq("eng-06-16x64x16", [16, 64, 16], [1, 4], row(1), kmaj(0)),
        seq("eng-06-64x32x64", [64, 32, 64], [4, 2], kmaj(4), row(2)),
    ]
}

/// The output cells of an emulated allocation, row-major.
fn cells(plan: &floating::Verified, out: &[u32]) -> Vec<u32> {
    let [rows, _, cols] = plan.ir().dimensions;
    let mut cells = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            cells.push(out[plan.physical_index(2, [i, j]).unwrap() as usize]);
        }
    }
    cells
}

/// Input words for every pattern the strict corpus covers.
fn inputs([r, k, n]: [u32; 3]) -> Vec<(&'static str, Vec<u32>, Vec<u32>)> {
    let (la, lb) = ((r * k) as usize, (k * n) as usize);
    vec![
        ("normal", words(&NORMAL, la, 3), words(&NORMAL, lb, 5)),
        (
            "subnormal",
            words(&SUBNORMAL, la, 3),
            words(&SUBNORMAL, lb, 7),
        ),
        ("random", random_words(la, 0x9e37), random_words(lb, 0x7f4a)),
        // Every product is -0: the sum from +0 must stay +0.
        ("signed-zero", vec![0x8000_0000; la], vec![0x3f80_0000; lb]),
    ]
}

#[test]
fn issue_eng_06_emitter_gemm_moves_emit_exactly_the_gemm_rs_text() {
    let schedule = gemm_tiled_v1();
    let mut compared = 0;
    for (caps, target) in devices() {
        for dims in SHAPES.iter().chain(&EXTRA).copied() {
            let e = emission(DIRECT, dims, caps);
            let plan = e.plan();
            let (generic, fixed) = match caps {
                Capabilities::Cuda { .. } => (
                    contraction::emit_cuda(plan, &schedule, target).unwrap(),
                    gemm::emit_cuda(plan).unwrap(),
                ),
                Capabilities::Metal { .. } => (
                    contraction::emit_msl(plan, &schedule, target).unwrap(),
                    gemm::emit_msl(plan).unwrap(),
                ),
            };
            assert_eq!(generic, fixed, "{caps:?} {dims:?}");
            assert_eq!(generic, e.source());
            let v = contraction::verify(plan, &schedule, target).unwrap();
            assert_eq!(contraction::geometry(plan, &v), gemm::geometry(plan));
            compared += 1;
        }
    }
    assert_eq!(compared, 9 * (SHAPES.len() + EXTRA.len()));
    // The other sequences emit fully substituted MSL too.
    let apple9 = Registry::builtin().unwrap().get("metal-apple9").unwrap();
    let e = emission(DIRECT, [65, 17, 63], metal(9));
    for s in other_schedules() {
        let text = contraction::emit_msl(e.plan(), &s, apple9).unwrap();
        assert!(!text.contains('$') && text.contains(&format!("schedule={}", s.name)));
    }
}

/// Emulation shapes: one, K = 0, a column, odd and rectangular edges in every
/// axis, one tile minus one. Index math is the same for CUDA and MSL (the MSL
/// text equals gemm.rs above), so emulation runs on the sm_90 plan only and
/// stays within the fast test budget.
const EMULATED: [[u32; 3]; 9] = [
    [1, 1, 1],
    [5, 0, 7],
    [1, 37, 1],
    [17, 13, 33],
    [65, 17, 63],
    [70, 1, 70],
    [2, 3, 130],
    [130, 3, 2],
    [63, 15, 63],
];

#[test]
fn issue_eng_06_emitter_emulated_index_math_equals_the_strict_oracle() {
    let registry = Registry::builtin().unwrap();
    let (caps, target) = (cuda(4, 90), registry.get("cuda-sm90").unwrap());
    let mut schedules = vec![gemm_tiled_v1()];
    schedules.extend(other_schedules());
    let mut checked = 0;
    for dims in EMULATED {
        let e = emission(DIRECT, dims, caps);
        let plan = e.plan();
        let c0 = vec![0x7fc0_0000; plan.ir().output.elements as usize];
        for (index, schedule) in schedules.iter().enumerate() {
            let v = contraction::verify(plan, schedule, target).unwrap();
            let text = contraction::emit_cuda(plan, schedule, target).unwrap();
            assert!(
                !text.contains('$'),
                "unsubstituted token in {}",
                schedule.name
            );
            // Every pattern on the template sequence, random words on the others.
            for (label, a, b) in inputs(dims)
                .into_iter()
                .filter(|(l, ..)| index == 0 || *l == "random")
            {
                let out = emulate(plan, &v, &a, &b, &c0).unwrap();
                assert!(
                    strict_equal(&cells(plan, &out), &oracle(dims, &a, &b)),
                    "{} {dims:?} {label}",
                    schedule.name
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, EMULATED.len() * (4 + 4));
}

#[test]
fn issue_eng_06_emitter_refuses_what_it_does_not_lower() {
    let registry = Registry::builtin().unwrap();
    let sm90 = registry.get("cuda-sm90").unwrap();
    let apple9 = registry.get("metal-apple9").unwrap();
    let strict = emission(DIRECT, [65, 17, 63], cuda(4, 90));
    let tc = emission(TC_DIRECT, [65, 17, 63], cuda(4, 90));
    let schedule = gemm_tiled_v1();
    // A tc-f16-f32-v1 plan is never spelled as strict code.
    assert_eq!(
        contraction::emit_cuda(tc.plan(), &schedule, sm90),
        Err(Error::Plan(floating::Error::Numeric))
    );
    // The target must describe the plan's device.
    assert_eq!(
        contraction::emit_cuda(strict.plan(), &schedule, apple9),
        Err(Error::Target)
    );
    let sm86 = registry.get("cuda-sm86").unwrap();
    assert_eq!(
        contraction::emit_cuda(strict.plan(), &schedule, sm86),
        Err(Error::Target)
    );
    // A plan and its own target still emit only their own language.
    let on_metal = emission(DIRECT, [65, 17, 63], metal(9));
    assert_eq!(
        contraction::emit_cuda(on_metal.plan(), &schedule, apple9),
        Err(Error::Plan(floating::Error::Capabilities))
    );
    assert_eq!(
        contraction::emit_msl(strict.plan(), &schedule, sm90),
        Err(Error::Plan(floating::Error::Capabilities))
    );
    // A refused move sequence never reaches emission.
    let mut split_k = schedule.clone();
    split_k.moves.push(Move::Split {
        axis: Axis::K,
        factor: 2,
    });
    assert_eq!(
        contraction::emit_cuda(strict.plan(), &split_k, sm90),
        Err(Error::Schedule(moves::Refusal::SplitK))
    );
    // Verified moves this emitter does not lower yet.
    let mut piped = schedule.clone();
    piped.moves.push(Move::Pipeline { stages: 2 });
    let mut vector = schedule.clone();
    vector.moves.push(Move::Vectorize { width: 4 });
    vector.guards = [Operand::A, Operand::B]
        .map(|operand| moves::Guard::Divisible { operand, by: 4 })
        .to_vec();
    let mut staged_c = schedule.clone();
    staged_c.moves.push(Move::Stage {
        buffer: Operand::C,
        level: Level::Shared,
    });
    let mut unsupported = 0;
    for s in [&piped, &vector, &staged_c] {
        assert!(
            moves::verify(
                s,
                sm90,
                &moves::Region {
                    contract: moves::STRICT.to_owned(),
                    element_bytes: 4
                }
            )
            .is_ok()
        );
        assert!(matches!(
            contraction::emit_cuda(strict.plan(), s, sm90),
            Err(Error::Unsupported(_))
        ));
        unsupported += 1;
    }
    assert_eq!(unsupported, 3);
}
