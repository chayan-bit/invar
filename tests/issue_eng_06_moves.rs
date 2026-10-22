//! ENG-06 (#283) physical move set: every schedule of a generated grid over
//! every built-in GPU target either verifies within the target's limits or is
//! refused with a named reason before any emission; contract, guard and
//! split-K forgeries are refused; and a target added only as a JSON file
//! (sm_87) verifies schedules with no Rust change.

use std::collections::BTreeMap;
use std::path::Path;

use invar_device::Capabilities;
use invar_device::moves::{
    Axis, Guard, Layout, Move, Operand, Refusal, Region, STRICT, Schedule, Verified, verify,
};
use invar_device::target::{Level, Registry, Target};

const TC: &str = "tc-f16-f32-v1";

fn region(contract: &str) -> Region {
    Region {
        contract: contract.to_owned(),
        element_bytes: 4,
    }
}

fn tile(m: u32, n: u32, k: u32) -> Move {
    Move::Tile {
        axes: vec![Axis::M, Axis::N, Axis::K],
        sizes: vec![m, n, k],
    }
}

fn stage(buffer: Operand) -> Move {
    Move::Stage {
        buffer,
        level: Level::Shared,
    }
}

/// The grid's schedule: scalar cells (4x4 per thread, A K-major with a pad)
/// or one matrix atom per warp tile, with A and B staged and C in registers.
fn grid_schedule(
    [m, n, k]: [u32; 3],
    vector: u32,
    atom: Option<&str>,
    stages: u32,
    guarded: bool,
) -> Schedule {
    let mut moves = vec![tile(m, n, k)];
    match atom {
        Some(name) => moves.push(Move::UseAtom {
            atom: name.to_owned(),
        }),
        None => {
            moves.push(Move::Split {
                axis: Axis::M,
                factor: 4,
            });
            moves.push(Move::Split {
                axis: Axis::N,
                factor: 4,
            });
        }
    }
    moves.extend([
        stage(Operand::A),
        Move::Permute {
            buffer: Operand::A,
            layout: Layout {
                k_major: true,
                pad: 1,
            },
        },
        stage(Operand::B),
        Move::Privatize { buffer: Operand::C },
        Move::Vectorize { width: vector },
        Move::Pipeline { stages },
    ]);
    let guards = if guarded && vector > 1 {
        [Operand::A, Operand::B]
            .map(|operand| Guard::Divisible {
                operand,
                by: vector,
            })
            .to_vec()
    } else {
        Vec::new()
    };
    Schedule {
        name: "grid".to_owned(),
        moves,
        guards,
    }
}

/// Independent recomputation of the limits a verified schedule must meet.
fn check_within_target(v: &Verified, t: &Target) {
    let gpu = t.gpu().unwrap();
    assert!(v.threads <= gpu.max_threads_per_group && v.threads.is_multiple_of(gpu.subgroup_width));
    let [m, n, k] = v.tile.map(u64::from);
    let a = k * (m + 1) * 4 * u64::from(v.stages);
    let b = k * n * 4 * u64::from(v.stages);
    assert_eq!(v.shared_bytes, a + b, "{v:?}");
    assert!(v.shared_bytes <= u64::from(gpu.shared_bytes_per_group));
    if let Some(r) = gpu.registers_per_thread {
        assert!(v.registers <= r);
    }
    if v.vector > 1 {
        assert_eq!(v.guards.len(), 2);
    }
    if v.stages > 1 {
        assert!(v.copy.starts_with("cp-async") || v.copy.starts_with("tma"));
    }
    if let Some(atom) = &v.atom {
        assert!(
            atom.reference.is_some()
                && v.tile[0].is_multiple_of(atom.m)
                && v.tile[2].is_multiple_of(atom.k)
        );
    }
}

#[test]
fn issue_eng_06_moves_grid_verifies_or_refuses_by_name_on_every_target() {
    let registry = Registry::builtin().unwrap();
    let mut verified = 0;
    let mut reasons: BTreeMap<&'static str, usize> = BTreeMap::new();
    for t in registry.targets().iter().filter(|t| t.gpu().is_some()) {
        let atoms: Vec<Option<&str>> = std::iter::once(None)
            .chain(
                t.gpu()
                    .unwrap()
                    .mma_atoms
                    .iter()
                    .map(|a| Some(a.name.as_str())),
            )
            .collect();
        for m in [16, 32, 64, 128] {
            for n in [16, 32, 64, 128] {
                for k in [8, 16, 32, 64] {
                    for vector in 1..=4 {
                        for &atom in &atoms {
                            for stages in [1, 2] {
                                let contract = atom
                                    .and_then(|a| {
                                        t.gpu().unwrap().mma_atoms.iter().find(|x| x.name == a)
                                    })
                                    .map_or(STRICT, |a| a.contract.as_str());
                                let s = grid_schedule([m, n, k], vector, atom, stages, true);
                                match verify(&s, t, &region(contract)) {
                                    Ok(v) => {
                                        check_within_target(&v, t);
                                        verified += 1;
                                    }
                                    Err(refusal) => {
                                        *reasons.entry(refusal.reason()).or_default() += 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Every refusal is named, and the grid reaches each budget.
    for reason in [
        "shared-budget",
        "group-threads",
        "vector-width",
        "atom-no-reference",
        "pipeline-needs-async-copy",
    ] {
        assert!(
            reasons.get(reason).is_some_and(|&c| c > 0),
            "{reason}: {reasons:?}"
        );
    }
    assert!(verified > 1000, "{verified} {reasons:?}");
    assert!(!reasons.contains_key("malformed-schedule"), "{reasons:?}");
}

#[test]
fn issue_eng_06_moves_budget_exceeding_sequences_are_refused_before_emission() {
    let sm86 = Registry::builtin().unwrap().get("cuda-sm86").unwrap();
    // 128x128x64 scalar: A 64*(128+1)*4 + B 64*128*4 = 65792 bytes > 48 KiB.
    let big = grid_schedule([128, 128, 64], 1, None, 1, true);
    assert_eq!(
        verify(&big, sm86, &region(STRICT)),
        Err(Refusal::SharedBudget {
            bytes: 65_792,
            limit: 49_152
        })
    );
    // Double buffering doubles the staged inputs: 64x64x32 fits once, not twice.
    let once = grid_schedule([64, 64, 32], 1, None, 1, true);
    let twice = grid_schedule([64, 64, 32], 1, None, 2, true);
    assert_eq!(
        verify(&once, sm86, &region(STRICT)).unwrap().shared_bytes,
        16_512
    );
    assert_eq!(
        verify(&twice, sm86, &region(STRICT)).unwrap().shared_bytes,
        33_024
    );
    let deep = grid_schedule([64, 64, 64], 1, None, 2, true);
    assert!(matches!(
        verify(&deep, sm86, &region(STRICT)),
        Err(Refusal::SharedBudget { .. })
    ));
    // 4x4 cells over a 256x128 tile need 2048 threads.
    let wide = grid_schedule([256, 128, 8], 1, None, 1, true);
    assert!(matches!(
        verify(&wide, sm86, &region(STRICT)),
        Err(Refusal::Threads { threads: 2048, .. })
    ));
}

#[test]
fn issue_eng_06_moves_forged_schedules_are_refused() {
    let registry = Registry::builtin().unwrap();
    let sm90 = registry.get("cuda-sm90").unwrap();
    let apple9 = registry.get("metal-apple9").unwrap();
    const MMA: &str = "mma-sync-m16n8k16-f16-f32";
    let mma = Some(MMA);
    // An atom whose contract differs from the region's.
    let tc = grid_schedule([64, 64, 16], 1, mma, 1, true);
    assert_eq!(
        verify(&tc, sm90, &region(STRICT)),
        Err(Refusal::AtomContract {
            atom: MMA.to_owned(),
            contract: STRICT.to_owned()
        })
    );
    assert!(verify(&tc, sm90, &region(TC)).is_ok());
    // A scalar schedule cannot claim a matrix contract.
    let scalar = grid_schedule([64, 64, 16], 1, None, 1, true);
    assert_eq!(verify(&scalar, sm90, &region(TC)), Err(Refusal::Contract));
    // A forged schedule that skips the divisibility guard its vector loads need.
    let forged = grid_schedule([64, 64, 16], 4, None, 1, false);
    assert_eq!(
        verify(&forged, sm90, &region(STRICT)),
        Err(Refusal::MissingGuard(Guard::Divisible {
            operand: Operand::A,
            by: 4
        }))
    );
    let mut half = grid_schedule([64, 64, 16], 4, None, 1, true);
    half.guards.remove(1);
    assert!(matches!(
        verify(&half, sm90, &region(STRICT)),
        Err(Refusal::MissingGuard(Guard::Divisible {
            operand: Operand::B,
            ..
        }))
    ));
    assert_eq!(
        verify(
            &grid_schedule([64, 64, 16], 4, None, 1, true),
            sm90,
            &region(STRICT)
        )
        .unwrap()
        .guards
        .len(),
        2
    );
    // Splitting K reorders a strict sum.
    let mut split_k = scalar.clone();
    split_k.moves.push(Move::Split {
        axis: Axis::K,
        factor: 2,
    });
    assert_eq!(
        verify(&split_k, sm90, &region(STRICT)),
        Err(Refusal::SplitK)
    );
    // Atoms that are data only, unknown, or do not tile.
    let bf16 = grid_schedule([64, 64, 16], 1, Some("mma-sync-m16n8k16-bf16-f32"), 1, true);
    assert!(matches!(
        verify(&bf16, sm90, &region("tc-bf16-f32-v1")),
        Err(Refusal::AtomNoReference(_))
    ));
    let unknown = grid_schedule([64, 64, 16], 1, Some("mma-sync-m8n8k4-f64"), 1, true);
    assert!(matches!(
        verify(&unknown, sm90, &region(TC)),
        Err(Refusal::AtomUnknown(_))
    ));
    let odd = grid_schedule([16, 8, 8], 1, mma, 1, true);
    assert!(matches!(
        verify(&odd, sm90, &region(TC)),
        Err(Refusal::AtomShape(_))
    ));
    // Metal has no asynchronous copy, so it cannot pipeline.
    let piped = grid_schedule([64, 64, 16], 1, None, 2, true);
    assert_eq!(
        verify(&piped, apple9, &region(STRICT)),
        Err(Refusal::PipelineNeedsAsyncCopy)
    );
    assert_eq!(
        verify(&piped, sm90, &region(STRICT)).unwrap().copy,
        "cp-async-4"
    );
    // Malformed shapes: no tile, repeated tile, odd vector, no accumulator.
    let mut no_tile = scalar.clone();
    no_tile.moves.remove(0);
    let mut two_tiles = scalar.clone();
    two_tiles.moves.push(tile(32, 32, 8));
    let mut no_acc = scalar.clone();
    no_acc
        .moves
        .retain(|m| !matches!(m, Move::Privatize { .. }));
    for s in [&no_tile, &two_tiles, &no_acc] {
        assert!(matches!(
            verify(s, sm90, &region(STRICT)),
            Err(Refusal::Malformed(_))
        ));
    }
    assert_eq!(
        verify(
            &grid_schedule([64, 64, 16], 3, None, 1, true),
            sm90,
            &region(STRICT)
        ),
        Err(Refusal::VectorWidth(3))
    );
    // A staged operand needs the plain copies into and out of shared memory;
    // pipelined inputs land there by async copy and need only the read.
    let sm90_text = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("targets/cuda-sm90.json"),
    )
    .unwrap();
    let without = |atom: &str| {
        let text: String = sm90_text
            .lines()
            .filter(|l| !l.contains(&format!("\"{atom}\"")))
            .map(|l| format!("{l}\n"))
            .collect();
        assert!(text.len() < sm90_text.len(), "{atom}");
        Target::parse("cuda-sm90", &text).unwrap()
    };
    let (no_store, no_read) = (without("st-shared-4"), without("ld-shared-4"));
    for t in [&no_store, &no_read] {
        assert_eq!(
            verify(&scalar, t, &region(STRICT)),
            Err(Refusal::NoStagedCopy { bytes: 4 })
        );
    }
    assert_eq!(
        verify(&piped, &no_store, &region(STRICT)).unwrap().copy,
        "cp-async-4"
    );
    assert_eq!(
        verify(&piped, &no_read, &region(STRICT)),
        Err(Refusal::NoStagedCopy { bytes: 4 })
    );
    // CPU targets carry no GPU schedule.
    let zen4 = registry.get("cpu-x86_64-zen4").unwrap();
    assert_eq!(verify(&scalar, zen4, &region(STRICT)), Err(Refusal::NotGpu));
}

#[test]
fn issue_eng_06_moves_a_new_target_file_verifies_schedules_with_no_rust_change() {
    let dir = std::env::temp_dir().join(format!("invar-eng-06-moves-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let targets = Path::new(env!("CARGO_MANIFEST_DIR")).join("targets");
    let jetson = Capabilities::Cuda {
        major: 12,
        minor: 4,
        sm: 87,
    };
    for entry in std::fs::read_dir(&targets).unwrap() {
        let path = entry.unwrap().path();
        if path.file_stem().is_some_and(|s| s != "cuda-sm87") {
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
    }
    assert!(Registry::from_dir(&dir).unwrap().select(jetson).is_err());
    std::fs::copy(targets.join("cuda-sm87.json"), dir.join("cuda-sm87.json")).unwrap();
    let registry = Registry::from_dir(&dir).unwrap();
    let orin = registry.select(jetson).unwrap();
    let mut verified = 0;
    for (s, contract) in [
        (grid_schedule([64, 64, 16], 1, None, 1, true), STRICT),
        (grid_schedule([64, 64, 32], 4, None, 2, true), STRICT),
        (
            grid_schedule([64, 64, 16], 1, Some("mma-sync-m16n8k16-f16-f32"), 1, true),
            TC,
        ),
    ] {
        let v = verify(&s, orin, &region(contract)).unwrap();
        assert_eq!(v.target, "cuda-sm87");
        check_within_target(&v, orin);
        verified += 1;
    }
    assert_eq!(verified, 3);
    std::fs::remove_dir_all(&dir).unwrap();
}
