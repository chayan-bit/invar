//! ENG-06 physical move set (#61): a contraction schedule
//! `C[M,N] = sum_k A[M,K] * B[K,N]` is a sequence of moves over its
//! iteration space, checked against a target description before any code is
//! emitted. `verify` refuses a move the target cannot hold (shared-memory
//! budget, group size, registers, copy or matrix atoms, alignment), a move
//! that would change a strict contract's summation order (splitting K), an
//! atom whose numeric contract is not the region's, and a schedule that omits
//! a runtime guard its moves need. A guard that fails at run time sends the
//! region to its serial CPU path; it never changes the result.

use crate::target::{CopyKind, Level, MmaAtom, Target};

/// Contraction axes, and `Tiles`: independent output tiles one group runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    M,
    N,
    K,
    Tiles,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Operand {
    A,
    B,
    C,
}

/// Element order of a staged tile. `k_major` stores a tile with K as its
/// outer index (`As[k][m]`); `pad` adds elements to every inner row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub k_major: bool,
    pub pad: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Move {
    /// The group tile: `axes` must be exactly `[M, N, K]`.
    Tile {
        axes: Vec<Axis>,
        sizes: Vec<u32>,
    },
    /// `factor` cells of `M` or `N` per thread (strided across the tile), or
    /// `factor` independent tiles per group (`Tiles`).
    Split {
        axis: Axis,
        factor: u32,
    },
    Stage {
        buffer: Operand,
        level: Level,
    },
    /// Buffers per staged operand; above 1 the copies must be asynchronous.
    Pipeline {
        stages: u32,
    },
    /// Elements per global load of A and B.
    Vectorize {
        width: u32,
    },
    Permute {
        buffer: Operand,
        layout: Layout,
    },
    /// Keep the operand (the accumulator, C) in registers.
    Privatize {
        buffer: Operand,
    },
    UseAtom {
        atom: String,
    },
}

/// A runtime check the emitted host code makes before launch: the operand's
/// offset, row stride and contiguous extent are multiples of `by` elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Guard {
    Divisible { operand: Operand, by: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub name: String,
    pub moves: Vec<Move>,
    pub guards: Vec<Guard>,
}

/// What the region asks of every schedule for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    /// `strict-f32-v1` or the matrix contract the region declares.
    pub contract: String,
    /// Bytes per staged and loaded element.
    pub element_bytes: u32,
}

pub const STRICT: &str = "strict-f32-v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    NotGpu,
    Malformed(&'static str),
    SplitK,
    Contract,
    Threads { threads: u64, limit: u32 },
    Subgroup { threads: u64, width: u32 },
    SharedBudget { bytes: u64, limit: u32 },
    Registers { needed: u32, limit: u32 },
    VectorWidth(u32),
    NoCopyAtom { bytes: u32 },
    NoStagedCopy { bytes: u32 },
    Alignment,
    PipelineNeedsAsyncCopy,
    AtomUnknown(String),
    AtomNoReference(String),
    AtomContract { atom: String, contract: String },
    AtomShape(String),
    MissingGuard(Guard),
}

impl Refusal {
    /// Stable name of the refusal, for receipts and tests.
    pub fn reason(&self) -> &'static str {
        match self {
            Self::NotGpu => "not-a-gpu-target",
            Self::Malformed(_) => "malformed-schedule",
            Self::SplitK => "split-k-reorders-strict-sum",
            Self::Contract => "schedule-contract",
            Self::Threads { .. } => "group-threads",
            Self::Subgroup { .. } => "subgroup-multiple",
            Self::SharedBudget { .. } => "shared-budget",
            Self::Registers { .. } => "register-budget",
            Self::VectorWidth(_) => "vector-width",
            Self::NoCopyAtom { .. } => "no-copy-atom",
            Self::NoStagedCopy { .. } => "no-staged-copy",
            Self::Alignment => "alignment",
            Self::PipelineNeedsAsyncCopy => "pipeline-needs-async-copy",
            Self::AtomUnknown(_) => "atom-unknown",
            Self::AtomNoReference(_) => "atom-no-reference",
            Self::AtomContract { .. } => "atom-contract",
            Self::AtomShape(_) => "atom-shape",
            Self::MissingGuard(_) => "missing-guard",
        }
    }
}

/// A schedule the target can run, with everything an emitter needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verified {
    pub schedule: String,
    pub target: String,
    /// Group tile `[M, N, K]`.
    pub tile: [u32; 3],
    /// Cells per thread along `[M, N]` (1 with an atom).
    pub micro: [u32; 2],
    pub tiles_per_group: u32,
    pub threads: u32,
    /// Staged operands in `A, B, C` order with their layouts.
    pub staged: Vec<(Operand, Layout)>,
    pub privatized: bool,
    pub stages: u32,
    pub vector: u32,
    pub atom: Option<MmaAtom>,
    /// The copy atom that fills staged tiles.
    pub copy: String,
    pub shared_bytes: u64,
    pub registers: u32,
    pub guards: Vec<Guard>,
}

const ROW_MAJOR: Layout = Layout {
    k_major: false,
    pad: 0,
};

/// (rows, elements per row including the pad) of a staged tile of
/// `operand` for the group tile `[m, n, k]`.
fn staged_shape(operand: Operand, layout: Layout, [m, n, k]: [u32; 3]) -> (u64, u64) {
    let (rows, inner) = match (operand, layout.k_major) {
        (Operand::A, true) => (k, m),
        (Operand::A, false) => (m, k),
        (Operand::B, false) => (k, n),
        (Operand::B, true) => (n, k),
        (Operand::C, _) => (m, n),
    };
    (u64::from(rows), u64::from(inner) + u64::from(layout.pad))
}

pub fn verify(schedule: &Schedule, target: &Target, region: &Region) -> Result<Verified, Refusal> {
    let gpu = target.gpu().ok_or(Refusal::NotGpu)?;
    let malformed = |what| Err(Refusal::Malformed(what));
    if !matches!(region.element_bytes, 1 | 2 | 4 | 8) {
        return malformed("element-bytes");
    }
    let mut tile = None;
    let (mut micro, mut tiles_per_group) = ([None, None], None);
    let (mut stages, mut vector, mut atom) = (None, None, None);
    let mut staged: Vec<(Operand, Layout)> = Vec::new();
    let mut permutes: Vec<(Operand, Layout)> = Vec::new();
    let mut privatized = false;
    fn once<T>(slot: &mut Option<T>, value: T) -> Result<(), Refusal> {
        match slot.replace(value) {
            None => Ok(()),
            Some(_) => Err(Refusal::Malformed("repeated-move")),
        }
    }
    for m in &schedule.moves {
        match m {
            Move::Tile { axes, sizes } => {
                if axes != &[Axis::M, Axis::N, Axis::K] || sizes.len() != 3 {
                    return malformed("tile-axes");
                }
                if sizes.iter().any(|s| !(1..=1024).contains(s)) {
                    return malformed("tile-size");
                }
                once(&mut tile, [sizes[0], sizes[1], sizes[2]])?;
            }
            Move::Split { axis: Axis::K, .. } => return Err(Refusal::SplitK),
            Move::Split { axis, factor } => {
                if !(1..=64).contains(factor) {
                    return malformed("split-factor");
                }
                match axis {
                    Axis::M => once(&mut micro[0], *factor)?,
                    Axis::N => once(&mut micro[1], *factor)?,
                    _ => once(&mut tiles_per_group, *factor)?,
                }
            }
            Move::Stage { buffer, level } => {
                if *level != Level::Shared {
                    return malformed("stage-level");
                }
                if staged.iter().any(|(o, _)| o == buffer) {
                    return malformed("repeated-move");
                }
                staged.push((*buffer, ROW_MAJOR));
            }
            Move::Pipeline { stages: s } => {
                if !(1..=8).contains(s) {
                    return malformed("pipeline-stages");
                }
                once(&mut stages, *s)?;
            }
            Move::Vectorize { width } => once(&mut vector, *width)?,
            Move::Permute { buffer, layout } => {
                if layout.pad > 32 || permutes.iter().any(|(o, _)| o == buffer) {
                    return malformed("permute");
                }
                permutes.push((*buffer, *layout));
            }
            Move::Privatize { buffer: Operand::C } if !privatized => privatized = true,
            Move::Privatize { .. } => return malformed("privatize"),
            Move::UseAtom { atom: name } => once(&mut atom, name.clone())?,
        }
    }
    let tile = tile.ok_or(Refusal::Malformed("no-tile"))?;
    for (operand, layout) in permutes {
        match staged.iter_mut().find(|(o, _)| *o == operand) {
            Some(slot) => slot.1 = layout,
            None => return malformed("permute-unstaged"),
        }
    }
    staged.sort_by_key(|(o, _)| *o);
    if !privatized && !staged.iter().any(|(o, _)| *o == Operand::C) {
        return malformed("no-accumulator");
    }
    let (stages, vector) = (stages.unwrap_or(1), vector.unwrap_or(1));
    let tiles_per_group = tiles_per_group.unwrap_or(1);

    // Matrix atom, or scalar cells per thread.
    let atom = match atom {
        None => {
            if region.contract != STRICT {
                return Err(Refusal::Contract);
            }
            None
        }
        Some(name) => {
            let a = gpu.mma_atoms.iter().find(|a| a.name == name);
            let a = a.ok_or_else(|| Refusal::AtomUnknown(name.clone()))?;
            if a.contract != region.contract {
                return Err(Refusal::AtomContract {
                    atom: name,
                    contract: region.contract.clone(),
                });
            }
            if a.reference.is_none() {
                return Err(Refusal::AtomNoReference(name));
            }
            if !tile[0].is_multiple_of(a.m)
                || !tile[1].is_multiple_of(a.n)
                || !tile[2].is_multiple_of(a.k)
            {
                return Err(Refusal::AtomShape(name));
            }
            if micro.iter().any(Option::is_some) {
                return malformed("split-with-atom");
            }
            Some(a.clone())
        }
    };
    let micro = [micro[0].unwrap_or(1), micro[1].unwrap_or(1)];
    if !tile[0].is_multiple_of(micro[0]) || !tile[1].is_multiple_of(micro[1]) {
        return malformed("split-divides-tile");
    }
    let per_tile = match &atom {
        Some(a) => u64::from(a.threads) * u64::from(tile[0] / a.m) * u64::from(tile[1] / a.n),
        None => u64::from(tile[0] / micro[0]) * u64::from(tile[1] / micro[1]),
    };
    let threads = per_tile * u64::from(tiles_per_group);
    if threads > u64::from(gpu.max_threads_per_group) {
        return Err(Refusal::Threads {
            threads,
            limit: gpu.max_threads_per_group,
        });
    }
    if !threads.is_multiple_of(u64::from(gpu.subgroup_width)) {
        return Err(Refusal::Subgroup {
            threads,
            width: gpu.subgroup_width,
        });
    }

    // Global loads of A and B: one plain copy atom of `vector` elements.
    if !matches!(vector, 1 | 2 | 4) {
        return Err(Refusal::VectorWidth(vector));
    }
    let load_bytes = vector * region.element_bytes;
    let load = gpu.copy_atoms.iter().find(|c| {
        c.kind == CopyKind::Plain
            && (c.from, c.to) == (Level::Global, Level::Registers)
            && c.bytes == load_bytes
    });
    let load = load.ok_or(Refusal::NoCopyAtom { bytes: load_bytes })?;
    // A is contiguous along K, B along N.
    if !tile[2].is_multiple_of(vector) || !tile[1].is_multiple_of(vector) {
        return Err(Refusal::VectorWidth(vector));
    }
    let copy = if stages > 1 {
        if !staged.iter().any(|(o, _)| *o != Operand::C) {
            return malformed("pipeline-unstaged");
        }
        let async_copy = gpu
            .copy_atoms
            .iter()
            .find(|c| matches!(c.kind, CopyKind::CpAsync | CopyKind::Tma) && c.bytes == load_bytes);
        async_copy
            .ok_or(Refusal::PipelineNeedsAsyncCopy)?
            .name
            .clone()
    } else {
        load.name.clone()
    };
    // A staged operand is read back from shared memory, and written into it
    // by a plain store unless an async copy lands it there (pipelined
    // inputs): both as one plain copy atom of one element, aligned to it.
    let plain = |from, to| {
        gpu.copy_atoms.iter().any(|c| {
            c.kind == CopyKind::Plain
                && (c.from, c.to) == (from, to)
                && c.bytes == region.element_bytes
                && c.alignment_bytes <= region.element_bytes
        })
    };
    let stored = staged.iter().any(|(o, _)| *o == Operand::C || stages == 1);
    if !staged.is_empty()
        && (!plain(Level::Shared, Level::Registers)
            || (stored && !plain(Level::Registers, Level::Shared)))
    {
        return Err(Refusal::NoStagedCopy {
            bytes: region.element_bytes,
        });
    }

    // Shared memory: every staged tile, inputs once per pipeline stage, and
    // vector stores into a staged row aligned.
    let shared_align = gpu
        .memory_levels
        .iter()
        .find(|l| l.level == Level::Shared)
        .map_or(1, |l| l.alignment_bytes);
    let mut shared_bytes = 0u64;
    for &(operand, layout) in &staged {
        let (rows, row) = staged_shape(operand, layout, tile);
        let copies = if operand == Operand::C { 1 } else { stages };
        shared_bytes += rows * row * u64::from(region.element_bytes) * u64::from(copies);
        let row_bytes = row * u64::from(region.element_bytes);
        // A vector load lands in one staged row only when the operand's
        // contiguous axis (K for A, N for B) is the row: row-major A or B.
        let row_contiguous = operand != Operand::C && !layout.k_major;
        if row_contiguous
            && vector > 1
            && !row_bytes.is_multiple_of(u64::from(load_bytes.min(shared_align)))
        {
            return Err(Refusal::Alignment);
        }
    }
    let limit = gpu.shared_bytes_per_group;
    if shared_bytes > u64::from(limit) {
        return Err(Refusal::SharedBudget {
            bytes: shared_bytes,
            limit,
        });
    }

    // Registers per thread: accumulators plus one operand slice, plus a
    // fixed allowance for addresses and loop state.
    let registers = match &atom {
        Some(a) => {
            let tiles = (tile[0] / a.m) * (tile[1] / a.n);
            tiles * (a.m * a.n).div_ceil(a.threads)
                + (a.m * a.k + a.k * a.n).div_ceil(a.threads)
                + 32
        }
        None => micro[0] * micro[1] + micro[0] + micro[1] + 32,
    };
    if let Some(limit) = gpu.registers_per_thread
        && registers > limit
    {
        return Err(Refusal::Registers {
            needed: registers,
            limit,
        });
    }

    // Guards the moves need; the schedule must carry every one.
    let mut guards = Vec::new();
    if vector > 1 {
        for operand in [Operand::A, Operand::B] {
            guards.push(Guard::Divisible {
                operand,
                by: vector,
            });
        }
    }
    for guard in &guards {
        if !schedule.guards.contains(guard) {
            return Err(Refusal::MissingGuard(*guard));
        }
    }
    let mut carried = schedule.guards.clone();
    carried.sort();
    carried.dedup();

    Ok(Verified {
        schedule: schedule.name.clone(),
        target: target.name.clone(),
        tile,
        micro,
        tiles_per_group,
        threads: u32::try_from(threads).map_err(|_| Refusal::Malformed("threads"))?,
        staged,
        privatized,
        stages,
        vector,
        atom,
        copy,
        shared_bytes,
        registers,
        guards: carried,
    })
}
