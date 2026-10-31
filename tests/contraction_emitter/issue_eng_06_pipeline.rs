//! ENG-06 (#61) part D: `Pipeline { stages }` lowers as real `cp.async`
//! multi-buffering on CUDA targets with an async copy (sm_80 and newer,
//! including sm_90), and is refused where no async copy exists. A model of
//! the emitted buffer ring and commit/wait protocol shows each k-tile is
//! complete before it is read and never overwritten while it can still be
//! read, and the per-cell accumulation is the tiled schedule's (so results
//! stay bit-identical to the strict oracle). Device rows run in
//! `tests/gpu_cuda/cp_async_pipeline.rs`.
#![forbid(unsafe_code)]

use crate::corpus;
use invar::contraction_source::{Contract, DeviceEmission};

use corpus::{DIRECT, SHAPES};
use invar_device::Capabilities;
use invar_device::contraction::{self, Error, gemm_pipelined_v1, gemm_tiled_v1};
use invar_device::moves::{Move, Refusal};
use invar_device::target::Registry;

fn emission(dims: [u32; 3], caps: Capabilities) -> DeviceEmission {
    let [r, k, n] = dims.map(u64::from);
    Contract::from_source(DIRECT.as_bytes(), "mat")
        .unwrap()
        .emit_device(DIRECT.as_bytes(), dims, [r * k, k * n], caps)
        .unwrap()
}

fn cuda(minor: u32, sm: u32) -> Capabilities {
    Capabilities::Cuda {
        major: 12,
        minor,
        sm,
    }
}

/// The accumulation statement both schedules emit: one rounded multiply and
/// one rounded add per product, in increasing k.
const ACCUMULATE: &str = "acc[r][c] = __fadd_rn(acc[r][c], __fmul_rn(a[r], b[c]));";

#[test]
fn issue_eng_06_pipeline_emits_cp_async_multi_buffering_where_the_target_has_it() {
    let registry = Registry::builtin().unwrap();
    let mut emitted = 0;
    for caps in [
        cuda(4, 80),
        cuda(4, 86),
        cuda(4, 87),
        cuda(4, 89),
        cuda(4, 90),
        cuda(8, 100),
        cuda(9, 120),
        cuda(9, 121),
    ] {
        let target = registry.select(caps).unwrap();
        for dims in SHAPES {
            let e = emission(dims, caps);
            let plan = e.plan();
            for stages in [2, 3, 4] {
                let mut schedule = gemm_tiled_v1();
                schedule.moves.push(Move::Pipeline { stages });
                let text = contraction::emit_cuda(plan, &schedule, target).unwrap();
                let tiled = contraction::emit_cuda(plan, &gemm_tiled_v1(), target).unwrap();
                let [rows, inner, cols] = dims;
                if rows == 0 || cols == 0 || inner == 0 {
                    // No k loop: nothing to pipeline beyond the schedule line.
                    assert!(!text.contains("cp.async"));
                    continue;
                }
                for needle in [
                    format!("__shared__ float As[{stages}][16][65];"),
                    format!("__shared__ float Bs[{stages}][16][64];"),
                    "cp.async.ca.shared.global [%0], [%1], 4, %2;".to_owned(),
                    "cp.async.commit_group;".to_owned(),
                    format!("cp.async.wait_group {};", stages - 1),
                    "a[r] = As[cur][kk][ty + 16u * r];".to_owned(),
                    "b[c] = Bs[cur][kk][tx + 16u * c];".to_owned(),
                ] {
                    assert!(text.contains(&needle), "{caps:?} {dims:?}: {needle}");
                }
                assert_eq!(text.matches(ACCUMULATE).count(), 1);
                assert_eq!(tiled.matches(ACCUMULATE).count(), 1);
                // Every bounds check of the tiled loads is kept.
                for check in ["if (in && ai >= ", "if (in && bi >= ", "if (ci >= "] {
                    assert!(text.contains(check), "{check}");
                }
                emitted += 1;
            }
        }
    }
    assert!(emitted >= 8 * 3 * 10, "{emitted}");
    // The named sequence is the two-stage one.
    let sm90 = registry.get("cuda-sm90").unwrap();
    let e = emission([65, 17, 63], cuda(4, 90));
    let text = contraction::emit_cuda(e.plan(), &gemm_pipelined_v1(), sm90).unwrap();
    assert!(text.contains("schedule=invar-f32-gemm-tiled-cp-async-v1 tile=64x64x16"));
    assert!(text.contains("cp.async.wait_group 1;"));
}

#[test]
fn issue_eng_06_pipeline_is_refused_without_an_async_copy() {
    let registry = Registry::builtin().unwrap();
    let schedule = gemm_pipelined_v1();
    // Turing and older have no cp.async; Metal has no async copy at all.
    let sm75 = registry.get("cuda-sm75").unwrap();
    let e = emission([65, 17, 63], cuda(4, 75));
    assert_eq!(
        contraction::emit_cuda(e.plan(), &schedule, sm75),
        Err(Error::Schedule(Refusal::PipelineNeedsAsyncCopy))
    );
    let apple9 = registry.get("metal-apple9").unwrap();
    let metal = Capabilities::Metal {
        apple_family: 9,
        msl_major: 3,
        msl_minor: 1,
    };
    let e = emission([65, 17, 63], metal);
    assert_eq!(
        contraction::emit_msl(e.plan(), &schedule, apple9),
        Err(Error::Schedule(Refusal::PipelineNeedsAsyncCopy))
    );
}

/// The emitted protocol for `stages` buffers and `ktiles` k-tiles: the
/// prologue issues k-tiles 0..stages-1 (one commit group each), iteration
/// `kt` issues k-tile `kt + stages - 1` into buffer `(kt + stages - 1) %
/// stages`, commits, waits until at most `stages - 1` groups are pending,
/// barriers, reads buffer `kt % stages`, and barriers again.
fn model(stages: u64, ktiles: u64) -> u64 {
    let mut buffer: Vec<Option<u64>> = vec![None; stages as usize];
    // Per commit group: the k-tile it copies, if any.
    let mut groups: Vec<Option<u64>> = Vec::new();
    let mut read = 0;
    for step in 0..stages - 1 + ktiles {
        // Prologue steps issue k-tile `step`; iteration `kt` issues
        // `kt + stages - 1`, which is the same `step`.
        let next = step;
        if next < ktiles {
            let slot = (next % stages) as usize;
            // Overwriting is safe only once every reader passed the barrier
            // after reading the slot's previous k-tile.
            if let Some(old) = buffer[slot] {
                assert!(old < read, "overwrote k-tile {old} before it was read");
            }
            buffer[slot] = Some(next);
        }
        groups.push((next < ktiles).then_some(next));
        if step < stages - 1 {
            continue;
        }
        let kt = step - (stages - 1);
        // wait_group(stages - 1): all but the newest stages - 1 groups done.
        let complete = groups.len() - (stages - 1) as usize;
        assert!(
            groups[..complete].contains(&Some(kt)),
            "k-tile {kt} read before its copy completed"
        );
        assert_eq!(buffer[(kt % stages) as usize], Some(kt));
        read += 1;
    }
    read
}

#[test]
fn issue_eng_06_pipeline_protocol_reads_each_k_tile_once_after_its_copy() {
    let mut runs = 0;
    for stages in 2..=8 {
        for ktiles in 0..40 {
            assert_eq!(model(stages, ktiles), ktiles);
            runs += 1;
        }
    }
    assert_eq!(runs, 7 * 40);
}
