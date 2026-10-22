//! ABI12 stage 2a: device workset layout and admission bounds for
//! `cuda-owned-numeric-continuation-v1`. Layout and bounds only: no allocation,
//! loader, receipt or CUDA call exists here. Every bound is checked with
//! checked arithmetic before any caller may allocate.
#![allow(
    dead_code,
    reason = "ABI12 stage 2a: consumed by the stage-2c device host; exercised by owned tests now"
)]
use super::{Artifact, BufferRange, Error, Target};

/// Buffer order: input, output, trace, config, heap, cleanup, tasks.
pub const BUFFERS: usize = 7;
pub const LANE_BLOCK: usize = 32;
pub const RED_ZONE_WORDS: usize = 64;
pub const MAX_ROOTS: usize = 1024;
/// Checked device payload bound (including red zones), before allocation.
pub const MAX_DEVICE_BYTES: usize = 64 * 1024 * 1024;
/// Checked bound on host snapshot copies retained by a run, before allocation.
pub const MAX_HOST_SNAPSHOT_BYTES: usize = 256 * 1024 * 1024;
const OUTPUT_WORDS: usize = 10;
const EVENT_WORDS: usize = 7;
const CONFIG_WORDS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnedNumericConfig {
    pub roots: usize,
    pub fuel: u64,
    pub transitions: u64,
    pub trace_events: usize,
    pub quantum: u64,
}
impl OwnedNumericConfig {
    /// The five shared config words in kernel order.
    pub fn words(self) -> [u64; CONFIG_WORDS] {
        [
            self.roots as u64,
            self.fuel,
            self.transitions,
            self.trace_events as u64,
            self.quantum,
        ]
    }
    fn check(self) -> Result<(), Error> {
        if self.roots == 0
            || self.roots > MAX_ROOTS
            || self.fuel == 0
            || self.fuel > 1_000_000
            || self.transitions > 1_000_000
            || self.trace_events > 1_000_000
            || self.quantum == 0
            || self.quantum > 1_000_000
        {
            return Err(Error::Limit);
        }
        Ok(())
    }
}

/// Per-root layout taken from an ABI12 CUDA artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedNumericLayout {
    params: usize,
    heap_words: usize,
    cleanup_words: usize,
    task_bytes: usize,
}
impl OwnedNumericLayout {
    /// Only an ABI12 CUDA artifact (owned numeric continuation) is admitted.
    pub fn from_artifact(artifact: &Artifact) -> Result<Self, Error> {
        let (Some(owned), Some(continuation)) =
            (artifact.owned_layout(), artifact.continuation_layout())
        else {
            return Err(Error::NumericTarget);
        };
        if artifact.abi_version() != 12 || artifact.target() != Target::Cuda {
            return Err(Error::NumericTarget);
        }
        let task_bytes = continuation.task_bytes();
        if task_bytes == 0 || !task_bytes.is_multiple_of(8) {
            return Err(Error::State);
        }
        Ok(Self {
            params: artifact.params().len(),
            heap_words: owned.heap_words(),
            cleanup_words: owned.cleanup_words(),
            task_bytes,
        })
    }
    /// Launched lanes: roots rounded up to whole 32-thread blocks.
    pub fn lanes(roots: usize) -> Result<usize, Error> {
        roots
            .checked_add(LANE_BLOCK - 1)
            .map(|n| n / LANE_BLOCK * LANE_BLOCK)
            .ok_or(Error::Limit)
    }
    /// Exact bytes per buffer for every launched lane plus a red zone.
    pub fn buffer_bytes(&self, config: OwnedNumericConfig) -> Result<[usize; BUFFERS], Error> {
        config.check()?;
        let lanes = Self::lanes(config.roots)?;
        let trace = config
            .trace_events
            .checked_mul(EVENT_WORDS)
            .ok_or(Error::Limit)?;
        let per_lane_words = [
            self.params.max(1),
            OUTPUT_WORDS,
            trace,
            0,
            self.heap_words,
            self.cleanup_words,
            self.task_bytes / 8,
        ];
        let mut bytes = [0usize; BUFFERS];
        for (i, words) in per_lane_words.into_iter().enumerate() {
            let payload = if i == 3 {
                Some(CONFIG_WORDS)
            } else {
                words.checked_mul(lanes)
            };
            bytes[i] = payload
                .and_then(|n| n.checked_add(RED_ZONE_WORDS))
                .and_then(|n| n.checked_mul(8))
                .ok_or(Error::Limit)?;
        }
        Ok(bytes)
    }
    /// Total device bytes, refused above [`MAX_DEVICE_BYTES`].
    pub fn device_bytes(&self, config: OwnedNumericConfig) -> Result<usize, Error> {
        let total = self
            .buffer_bytes(config)?
            .iter()
            .try_fold(0usize, |n, b| n.checked_add(*b))
            .ok_or(Error::Limit)?;
        if total > MAX_DEVICE_BYTES {
            return Err(Error::Limit);
        }
        Ok(total)
    }
    /// Host bytes for `retained` full snapshots, refused above
    /// [`MAX_HOST_SNAPSHOT_BYTES`]. A run retains at least the previous
    /// validated and the newly captured snapshot.
    pub fn host_snapshot_bytes(
        &self,
        config: OwnedNumericConfig,
        retained: usize,
    ) -> Result<usize, Error> {
        if retained < 2 {
            return Err(Error::Limit);
        }
        let total = self
            .device_bytes(config)?
            .checked_mul(retained)
            .ok_or(Error::Limit)?;
        if total > MAX_HOST_SNAPSHOT_BYTES {
            return Err(Error::Limit);
        }
        Ok(total)
    }
    /// Admit backend-issued ranges after both bounds, with the ABI5 range rules.
    pub fn workset(
        &self,
        config: OwnedNumericConfig,
        retained: usize,
        ranges: [BufferRange; BUFFERS],
    ) -> Result<OwnedNumericWorkset, Error> {
        self.host_snapshot_bytes(config, retained)?;
        let bytes = self.buffer_bytes(config)?;
        for (i, r) in ranges.iter().enumerate() {
            if r.device == 0
                || r.device != ranges[0].device
                || r.allocation == 0
                || r.bytes < bytes[i]
                || r.offset % 8 != 0
                || r.offset.checked_add(r.bytes).is_none()
            {
                return Err(Error::State);
            }
            for other in &ranges[..i] {
                if r.allocation == other.allocation
                    && r.offset < other.offset + other.bytes
                    && other.offset < r.offset + r.bytes
                {
                    return Err(Error::State);
                }
            }
        }
        Ok(OwnedNumericWorkset {
            config,
            retained,
            bytes,
            ranges,
        })
    }
}

/// Read-only ABI12 workset facts for the stage-2b compile receipt, derived
/// through [`OwnedNumericLayout::from_artifact`]. Grants no allocation, loader
/// or execution authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnedNumericFacts {
    pub input_words: usize,
    pub output_words: usize,
    pub event_words: usize,
    pub config_words: usize,
    pub heap_words: usize,
    pub cleanup_words: usize,
    pub task_words: usize,
    pub lane_block: usize,
    pub red_zone_words: usize,
    pub max_roots: usize,
    pub max_fuel: u64,
    pub max_transitions: u64,
    pub max_trace_events: usize,
    pub max_quantum: u64,
    pub max_device_bytes: usize,
    pub max_host_snapshot_bytes: usize,
    pub min_retained_snapshots: usize,
}
impl Artifact {
    /// Only an ABI12 CUDA artifact has these facts.
    pub(crate) fn owned_numeric_cuda_facts(&self) -> Result<OwnedNumericFacts, Error> {
        let layout = OwnedNumericLayout::from_artifact(self)?;
        Ok(OwnedNumericFacts {
            input_words: layout.params.max(1),
            output_words: OUTPUT_WORDS,
            event_words: EVENT_WORDS,
            config_words: CONFIG_WORDS,
            heap_words: layout.heap_words,
            cleanup_words: layout.cleanup_words,
            task_words: layout.task_bytes / 8,
            lane_block: LANE_BLOCK,
            red_zone_words: RED_ZONE_WORDS,
            max_roots: MAX_ROOTS,
            max_fuel: 1_000_000,
            max_transitions: 1_000_000,
            max_trace_events: 1_000_000,
            max_quantum: 1_000_000,
            max_device_bytes: MAX_DEVICE_BYTES,
            max_host_snapshot_bytes: MAX_HOST_SNAPSHOT_BYTES,
            min_retained_snapshots: 2,
        })
    }
}

/// An admitted ABI12 workset description; it owns no memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedNumericWorkset {
    config: OwnedNumericConfig,
    retained: usize,
    bytes: [usize; BUFFERS],
    ranges: [BufferRange; BUFFERS],
}
impl OwnedNumericWorkset {
    pub fn config(&self) -> OwnedNumericConfig {
        self.config
    }
    pub fn retained_snapshots(&self) -> usize {
        self.retained
    }
    pub fn buffer_bytes(&self) -> &[usize; BUFFERS] {
        &self.bytes
    }
    pub fn ranges(&self) -> &[BufferRange; BUFFERS] {
        &self.ranges
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        ExecutionState as E, FenceError, FenceObservation as F, FencedExecution, Limits,
        OwnedLimits, Submission as S, emit_resumable_owned_numeric,
    };
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    fn artifact(target: Target) -> Artifact {
        let source = "fn run(a: f32, n: u32) -> f32 { var xs: Array<f32> = filled_f32(n + 1u32, a); xs[0u32] = xs[0u32] * a + a; return xs[0u32]; } fn main() -> i32 { return 0; }";
        let checked =
            crate::check::check(crate::front::parse(source.as_bytes()).unwrap(), false).unwrap();
        crate::ownership::check(&checked).unwrap();
        let net = crate::general_net::Net::lower(&checked, crate::general_net::Limits::default())
            .unwrap();
        let proof = net.verify().unwrap();
        emit_resumable_owned_numeric(
            &net,
            &proof,
            "run",
            target,
            Limits::default(),
            OwnedLimits {
                elements: 64,
                allocations: 16,
                cleanup_events: 32,
            },
        )
        .unwrap()
    }
    fn config() -> OwnedNumericConfig {
        OwnedNumericConfig {
            roots: 40,
            fuel: 1000,
            transitions: 10_000,
            trace_events: 100,
            quantum: 3,
        }
    }
    fn ranges(bytes: [usize; BUFFERS]) -> [BufferRange; BUFFERS] {
        std::array::from_fn(|i| BufferRange {
            device: 7,
            allocation: 100 + i as u64,
            offset: 0,
            bytes: bytes[i],
        })
    }

    #[test]
    fn abi12_layout_admits_only_cuda_abi12_artifacts() {
        let cuda = artifact(Target::Cuda);
        let layout = OwnedNumericLayout::from_artifact(&cuda).unwrap();
        assert_eq!(layout.params, 2);
        assert_eq!(
            layout.task_bytes,
            cuda.continuation_layout().unwrap().task_bytes()
        );
        let cpu = artifact(Target::Cpu);
        assert_eq!(cpu.abi_version(), 11);
        assert_eq!(
            OwnedNumericLayout::from_artifact(&cpu),
            Err(Error::NumericTarget)
        );
    }

    #[test]
    fn abi12_lane_rounding_and_exact_buffer_bytes() {
        for (roots, lanes) in [(1, 32), (32, 32), (33, 64), (1024, 1024)] {
            assert_eq!(OwnedNumericLayout::lanes(roots), Ok(lanes));
        }
        let layout = OwnedNumericLayout::from_artifact(&artifact(Target::Cuda)).unwrap();
        let c = config();
        let lanes = 64;
        let want = [
            (2 * lanes + RED_ZONE_WORDS) * 8,
            (10 * lanes + RED_ZONE_WORDS) * 8,
            (700 * lanes + RED_ZONE_WORDS) * 8,
            (5 + RED_ZONE_WORDS) * 8,
            (layout.heap_words * lanes + RED_ZONE_WORDS) * 8,
            (layout.cleanup_words * lanes + RED_ZONE_WORDS) * 8,
            (layout.task_bytes / 8 * lanes + RED_ZONE_WORDS) * 8,
        ];
        assert_eq!(layout.buffer_bytes(c), Ok(want));
        assert_eq!(c.words(), [40, 1000, 10_000, 100, 3]);
    }

    #[test]
    fn abi12_config_bounds_are_refused_before_allocation() {
        let layout = OwnedNumericLayout::from_artifact(&artifact(Target::Cuda)).unwrap();
        let c = config();
        for bad in [
            OwnedNumericConfig { roots: 0, ..c },
            OwnedNumericConfig { roots: 1025, ..c },
            OwnedNumericConfig { fuel: 0, ..c },
            OwnedNumericConfig {
                fuel: 1_000_001,
                ..c
            },
            OwnedNumericConfig {
                transitions: 1_000_001,
                ..c
            },
            OwnedNumericConfig {
                trace_events: 1_000_001,
                ..c
            },
            OwnedNumericConfig { quantum: 0, ..c },
            OwnedNumericConfig {
                quantum: 1_000_001,
                ..c
            },
        ] {
            assert_eq!(layout.buffer_bytes(bad), Err(Error::Limit), "{bad:?}");
            assert_eq!(layout.workset(bad, 2, ranges([8; 7])), Err(Error::Limit));
        }
    }

    #[test]
    fn abi12_device_and_host_snapshot_bounds_are_checked() {
        let layout = OwnedNumericLayout::from_artifact(&artifact(Target::Cuda)).unwrap();
        // Largest admitted trace for 1024 roots and the next one, as literals
        // derived in closed form (no search through the function under test):
        // per-lane words S(T) = 2 params + 10 output + 7 T trace + 128 heap
        // (16 descriptors * 4 + 64 elements) + 128 cleanup (32 * 4) + 1490
        // task words (64 frames * 23 words + 18 counters), and
        // total = 8 * (1024 * S(T) + 5 config + 7 * 64 red zone).
        assert_eq!(
            (
                layout.params,
                layout.heap_words,
                layout.cleanup_words,
                layout.task_bytes
            ),
            (2, 128, 128, 64 * 8 * 23 + 144)
        );
        let base = OwnedNumericConfig {
            roots: 1024,
            ..config()
        };
        let edge = OwnedNumericConfig {
            trace_events: 919,
            ..base
        };
        let total = 67_104_296;
        assert_eq!(total, 8 * (1024 * (1758 + 7 * 919) + 453));
        assert_eq!(layout.device_bytes(edge), Ok(total));
        let over = OwnedNumericConfig {
            trace_events: 920,
            ..base
        };
        assert_eq!(8 * (1024 * (1758 + 7 * 920) + 453), 67_161_640);
        assert_eq!(layout.device_bytes(over), Err(Error::Limit));
        // Maximal inputs never overflow or panic; they are refused.
        let huge = OwnedNumericConfig {
            roots: 1024,
            trace_events: 1_000_000,
            ..config()
        };
        assert_eq!(layout.device_bytes(huge), Err(Error::Limit));
        // Host snapshots: at least two retained copies; product bounded.
        assert_eq!(layout.host_snapshot_bytes(edge, 1), Err(Error::Limit));
        let max_retained = MAX_HOST_SNAPSHOT_BYTES / total;
        assert_eq!(
            layout.host_snapshot_bytes(edge, max_retained),
            Ok(total * max_retained)
        );
        assert_eq!(
            layout.host_snapshot_bytes(edge, max_retained + 1),
            Err(Error::Limit)
        );
        assert_eq!(
            layout.host_snapshot_bytes(edge, usize::MAX),
            Err(Error::Limit)
        );
        // The workset applies both bounds before checking ranges.
        let bytes = layout.buffer_bytes(edge).unwrap();
        assert!(layout.workset(edge, max_retained, ranges(bytes)).is_ok());
        assert_eq!(
            layout.workset(edge, max_retained + 1, ranges(bytes)),
            Err(Error::Limit)
        );
    }

    /// Contract limit and its representable boundary, independent of the code
    /// under test. Every buffer except config holds `words * lanes` with
    /// `lanes = 32 k`, so for any layout and config
    /// `total = 8 * (32 k * S + 5 + 7 * 64) = 256 k S + 3624 ≡ 40 (mod 256)`,
    /// where S is the per-lane word sum. 64 MiB ≡ 0 (mod 256), so the nearest
    /// representable totals are 64 MiB - 216 (admitted) and 64 MiB + 40
    /// (refused). A cap threshold anywhere in [64 MiB - 216, 64 MiB + 39],
    /// e.g. `total > MAX_DEVICE_BYTES + 8` or `total >= MAX_DEVICE_BYTES`, is
    /// therefore behaviorally equivalent for every valid layout and cannot be
    /// distinguished by any input; the constant itself is pinned literally.
    #[test]
    fn abi12_device_cap_is_exact_at_representable_boundaries() {
        assert_eq!(MAX_DEVICE_BYTES, 67_108_864);
        assert_eq!(MAX_HOST_SNAPSHOT_BYTES, 268_435_456);
        // Layouts with the from_artifact invariants (task bytes non-zero,
        // 8-aligned), sized so k * S hits the boundary exactly.
        let synthetic = |heap_words| OwnedNumericLayout {
            params: 1,
            heap_words,
            cleanup_words: 0,
            task_bytes: 8,
        };
        let minimal = OwnedNumericConfig {
            roots: 1,
            fuel: 1,
            transitions: 0,
            trace_events: 0,
            quantum: 1,
        };
        // k = 1, S = 1 + 10 + heap + 1: S = 262_129 -> 64 MiB - 216.
        assert_eq!(8 * (32 * 262_129 + 453), 67_108_864 - 216);
        assert_eq!(
            synthetic(262_117).device_bytes(minimal),
            Ok(67_108_864 - 216)
        );
        // S = 262_130 -> 64 MiB + 40, the smallest representable over the cap.
        assert_eq!(8 * (32 * 262_130 + 453), 67_108_864 + 40);
        assert_eq!(synthetic(262_118).device_bytes(minimal), Err(Error::Limit));
        // The same two totals through other lane counts and the trace term:
        // k = 7 (roots 193..=224), S = 37_447 = 12 + heap + 7 T.
        let k7 = OwnedNumericConfig {
            roots: 224,
            trace_events: 5_000,
            ..minimal
        };
        assert_eq!(
            synthetic(37_447 - 12 - 35_000).device_bytes(k7),
            Ok(67_108_864 - 216)
        );
        // k = 22 (roots 673..=704), S = 11_915 -> 64 MiB + 40.
        let k22 = OwnedNumericConfig {
            roots: 673,
            trace_events: 1_000,
            ..minimal
        };
        assert_eq!(8 * (32 * 22 * 11_915 + 453), 67_108_864 + 40);
        assert_eq!(
            synthetic(11_915 - 12 - 7_000).device_bytes(k22),
            Err(Error::Limit)
        );
        // Every admitted total keeps the residue, for the real artifact and
        // synthetic layouts over every lane count.
        let real = OwnedNumericLayout::from_artifact(&artifact(Target::Cuda)).unwrap();
        for layout in [real, synthetic(0), synthetic(12_345)] {
            for roots in 1..=MAX_ROOTS {
                for trace_events in [0, 1, 7, 100] {
                    let c = OwnedNumericConfig {
                        roots,
                        trace_events,
                        ..minimal
                    };
                    if let Ok(total) = layout.device_bytes(c) {
                        assert_eq!(total % 256, 40, "{layout:?} {c:?}");
                    }
                }
            }
        }
    }

    /// The receipt facts agree with the bounds and formulas actually enforced.
    #[test]
    fn abi12_receipt_facts_match_enforced_layout_and_bounds() {
        let cuda = artifact(Target::Cuda);
        let layout = OwnedNumericLayout::from_artifact(&cuda).unwrap();
        let f = cuda.owned_numeric_cuda_facts().unwrap();
        assert_eq!(
            artifact(Target::Cpu).owned_numeric_cuda_facts(),
            Err(Error::NumericTarget)
        );
        let edge = OwnedNumericConfig {
            roots: f.max_roots,
            fuel: f.max_fuel,
            transitions: f.max_transitions,
            trace_events: 1,
            quantum: f.max_quantum,
        };
        let lanes = f.max_roots;
        let per_lane = [
            f.input_words,
            f.output_words,
            f.event_words,
            0,
            f.heap_words,
            f.cleanup_words,
            f.task_words,
        ];
        let want: [usize; BUFFERS] = std::array::from_fn(|i| {
            let payload = if i == 3 {
                f.config_words
            } else {
                per_lane[i] * lanes
            };
            (payload + f.red_zone_words) * 8
        });
        assert_eq!(layout.buffer_bytes(edge), Ok(want));
        let max_trace = OwnedNumericConfig {
            trace_events: f.max_trace_events,
            ..edge
        };
        assert!(layout.buffer_bytes(max_trace).is_ok());
        for bad in [
            OwnedNumericConfig {
                roots: f.max_roots + 1,
                ..edge
            },
            OwnedNumericConfig {
                fuel: f.max_fuel + 1,
                ..edge
            },
            OwnedNumericConfig {
                transitions: f.max_transitions + 1,
                ..edge
            },
            OwnedNumericConfig {
                trace_events: f.max_trace_events + 1,
                ..edge
            },
            OwnedNumericConfig {
                quantum: f.max_quantum + 1,
                ..edge
            },
        ] {
            assert_eq!(layout.buffer_bytes(bad), Err(Error::Limit), "{bad:?}");
        }
        assert_eq!(
            (f.lane_block, f.max_device_bytes, f.max_host_snapshot_bytes),
            (32, 67_108_864, 268_435_456)
        );
        let c = config();
        assert_eq!(
            layout.host_snapshot_bytes(c, f.min_retained_snapshots - 1),
            Err(Error::Limit)
        );
        assert!(
            layout
                .host_snapshot_bytes(c, f.min_retained_snapshots)
                .is_ok()
        );
    }

    #[test]
    fn abi12_workset_ranges_follow_abi5_rules() {
        let layout = OwnedNumericLayout::from_artifact(&artifact(Target::Cuda)).unwrap();
        let c = config();
        let bytes = layout.buffer_bytes(c).unwrap();
        let good = ranges(bytes);
        let workset = layout.workset(c, 2, good).unwrap();
        assert_eq!(workset.buffer_bytes(), &bytes);
        assert_eq!(workset.ranges(), &good);
        assert_eq!(workset.retained_snapshots(), 2);
        let mutate = |f: &dyn Fn(&mut [BufferRange; BUFFERS])| {
            let mut r = good;
            f(&mut r);
            layout.workset(c, 2, r)
        };
        assert_eq!(mutate(&|r| r[0].device = 0), Err(Error::State));
        assert_eq!(mutate(&|r| r[4].device = 8), Err(Error::State));
        assert_eq!(mutate(&|r| r[2].allocation = 0), Err(Error::State));
        assert_eq!(mutate(&|r| r[6].bytes -= 8), Err(Error::State));
        assert_eq!(mutate(&|r| r[1].offset = 4), Err(Error::State));
        assert_eq!(mutate(&|r| r[5].offset = usize::MAX - 4), Err(Error::State));
        // Overlap within one allocation is refused; disjoint subranges are fine.
        assert_eq!(
            mutate(&|r| {
                r[1].allocation = r[0].allocation;
                r[1].offset = 8;
            }),
            Err(Error::State)
        );
        assert!(
            mutate(&|r| {
                r[1].allocation = r[0].allocation;
                r[1].offset = r[0].bytes;
            })
            .is_ok()
        );
        // Larger backend allocations (padding) are admitted.
        assert!(mutate(&|r| r[2].bytes += 64).is_ok());
    }

    /// Resource whose destruction is observable.
    struct Lease(Arc<AtomicUsize>);
    impl Drop for Lease {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn gate(dropped: &Arc<AtomicUsize>, max: u64) -> FencedExecution<Lease, OwnedNumericWorkset> {
        let layout = OwnedNumericLayout::from_artifact(&artifact(Target::Cuda)).unwrap();
        let c = config();
        let workset = layout
            .workset(c, 2, ranges(layout.buffer_bytes(c).unwrap()))
            .unwrap();
        FencedExecution::new(workset, Lease(dropped.clone()), max).unwrap()
    }

    #[test]
    fn abi12_gate_keeps_fence_quarantine_and_lifetime_rules() {
        // Completed epoch: observe allowed, reclaim returns the resources.
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut run = gate(&dropped, 2);
        assert_eq!(run.observe(|_| ()), Err(FenceError::Pending));
        run.submit(|_| S::Submitted).unwrap();
        assert_eq!(run.state(), E::InFlight);
        assert_eq!(run.observe(|_| ()), Err(FenceError::Pending));
        assert_eq!(run.poll(|_| F::Pending), Err(FenceError::Pending));
        run.poll(|_| F::Complete).unwrap();
        assert_eq!((run.submitted(), run.completed()), (1, 1));
        run.observe(|_| ()).unwrap();
        assert_eq!(run.workset().config(), config());
        drop(run.reclaim().unwrap());
        assert_eq!(dropped.load(Ordering::SeqCst), 1);

        // Unknown submission: quarantined, never observed, reclaimed or freed.
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut run = gate(&dropped, 2);
        assert_eq!(run.submit(|_| S::Unknown), Err(FenceError::Unknown));
        assert_eq!(run.state(), E::Quarantined);
        assert_eq!(run.observe(|_| ()), Err(FenceError::Pending));
        assert_eq!(run.submit(|_| S::Submitted), Err(FenceError::Busy));
        let run = match run.reclaim() {
            Err(FenceError::Pending) => gate(&dropped, 1),
            other => panic!("quarantined reclaim: {:?}", other.err()),
        };
        drop(run);
        // The quarantined gate leaked its lease; only the fresh gate dropped.
        assert_eq!(dropped.load(Ordering::SeqCst), 1);

        // Unknown fence after submission: quarantined; dropping forgets.
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut run = gate(&dropped, 2);
        run.submit(|_| S::Submitted).unwrap();
        assert_eq!(run.poll(|_| F::Unknown), Err(FenceError::Unknown));
        assert_eq!(run.state(), E::Quarantined);
        drop(run);
        assert_eq!(
            dropped.load(Ordering::SeqCst),
            0,
            "unknown work must not be freed"
        );

        // Dropping in-flight work also forgets it (no replay, no free).
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut run = gate(&dropped, 2);
        run.submit(|_| S::Submitted).unwrap();
        drop(run);
        assert_eq!(dropped.load(Ordering::SeqCst), 0);

        // Failed but quiescent: no observation, reclaim allowed.
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut run = gate(&dropped, 2);
        run.submit(|_| S::Submitted).unwrap();
        assert_eq!(run.poll(|_| F::FailedQuiescent), Err(FenceError::Failed));
        assert_eq!(run.state(), E::Failed);
        assert_eq!(run.observe(|_| ()), Err(FenceError::Pending));
        drop(run.reclaim().unwrap());
        assert_eq!(dropped.load(Ordering::SeqCst), 1);

        // Cancel in flight: completion ends cancelled, nothing observable.
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut run = gate(&dropped, 2);
        run.submit(|_| S::Submitted).unwrap();
        run.cancel();
        assert_eq!(run.state(), E::CancelPending);
        assert_eq!(run.poll(|_| F::Complete), Err(FenceError::Cancelled));
        assert_eq!(run.state(), E::Cancelled);
        assert_eq!(run.observe(|_| ()), Err(FenceError::Pending));
        drop(run.reclaim().unwrap());
        assert_eq!(dropped.load(Ordering::SeqCst), 1);

        // NotSubmitted keeps ownership; the submission budget is enforced.
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut run = gate(&dropped, 1);
        assert_eq!(run.submit(|_| S::NotSubmitted), Err(FenceError::Submission));
        assert_eq!(run.state(), E::Ready);
        assert_eq!(run.submit(|_| S::Submitted), Err(FenceError::Budget));
        assert!(
            FencedExecution::<Lease, OwnedNumericWorkset>::new(
                run.workset().clone(),
                Lease(dropped.clone()),
                0
            )
            .is_err()
        );
    }
}
