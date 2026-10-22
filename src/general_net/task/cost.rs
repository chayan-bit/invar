//! Cost model of the fixed-width expansion.
//!
//! Two schedules share the node machine and the arena:
//!
//! * [`Schedule::Staged`] (MSL 3.1 fallback): a single-group scheduler expands
//!   ready nodes breadth first until one round holds at least `width` ready
//!   nodes, which then run as serial tasks; joins run in later scheduler
//!   passes, one dispatch boundary per pass.
//! * [`Schedule::Cascade`] (CUDA, MSL 3.2): one persistent multi-group kernel
//!   whose lanes claim arena nodes in allocation order (the device-wide ready
//!   list). A claimed node forks while fewer than `width` nodes exist and runs
//!   serially afterwards; the last child to finish resumes its parent in the
//!   same lane (cascade join).
//!
//! Expansion is compacted (each spawned child is one arena node), so no lane
//! is padded. The width is calibrated per device on the C1 tuning split and
//! frozen in the [`Profile`].
//!
//! For contrast, the P4a spike's digit form expands every fork site to a fixed
//! depth `D` and launches `radix^D` lanes whether or not a branch reached that
//! depth; [`digit_padding`] reports the padded share that form would pay.

use super::{Program, Refusal};

/// How the scheduler advances nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    /// Single-group scheduler passes and a run kernel, one dispatch boundary
    /// per pass (needs no cross-group coherence inside a dispatch).
    Staged,
    /// One persistent multi-group kernel with a device-wide ready list and
    /// cascade joins (needs device-scope fences and coherent device memory).
    Cascade,
}

/// Device characteristics the plan is chosen from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    pub schedule: Schedule,
    /// Threads of the persistent cascade grid (unused by the staged
    /// schedule). One lane per 32-lane SIMD group is a worker: the step
    /// machines diverge, so nodes sharing a SIMD group would serialize.
    pub lanes: u32,
    /// Expansion width, calibrated on the C1 tuning split: ready nodes of one
    /// staged round, or arena nodes before cascade lanes run serially.
    pub width: u32,
    /// Threads of the single staged scheduler group.
    pub sched_threads: u32,
    /// Threads per group of the run (or cascade) kernel.
    pub run_threads: u32,
    /// Arena nodes (bounded by the buffer budget).
    pub arena: u32,
    /// Fingerprints (hex) of the regions the scored D5 bench measured above
    /// the residual tier on this device; [`route`] refuses them to the native
    /// path. Calibrated with the width and checked against the committed
    /// reports by `tests/net_task_region.rs`.
    pub refused: &'static [&'static str],
}

impl Profile {
    /// Apple M4 class GPU, MSL 3.2 (cascade); lanes and width from the
    /// scored pilot (`archived-evidence/gnet-p4b/bench-metal/pilot-metal.json`).
    pub const METAL: Self = Self {
        schedule: Schedule::Cascade,
        lanes: 2048,
        width: 16384,
        sched_threads: 256,
        run_threads: 64,
        arena: 1 << 16,
        refused: METAL_REFUSED,
    };
    /// Apple7+ GPU, MSL 3.1 fallback (staged); width from the scored pilot
    /// (`archived-evidence/gnet-p4b/bench-metal-staged/pilot-metal-staged.json`).
    pub const METAL_STAGED: Self = Self {
        schedule: Schedule::Staged,
        lanes: 1024,
        width: 64,
        sched_threads: 256,
        run_threads: 64,
        arena: 1 << 16,
        refused: METAL_STAGED_REFUSED,
    };
    /// RTX 5070 Ti class (70 SMs), cascade; lanes and width from the scored
    /// pilot (`archived-evidence/gnet-p4b/bench-cuda/pilot-cuda.json`). The inlined
    /// step machines can use up to 255 registers per thread, so groups stay
    /// at 64 threads.
    pub const CUDA: Self = Self {
        schedule: Schedule::Cascade,
        lanes: 8960,
        width: 16384,
        sched_threads: 256,
        run_threads: 64,
        arena: 1 << 17,
        refused: CUDA_REFUSED,
    };
    /// CPU model used by the differential tests (staged): small widths so
    /// expansion, joins and arena fallbacks all run on short inputs.
    pub const CPU_MODEL: Self = Self {
        schedule: Schedule::Staged,
        lanes: 4,
        width: 8,
        sched_threads: 4,
        run_threads: 4,
        arena: 256,
        refused: &[],
    };
    /// CPU model of the cascade schedule (lanes claim in allocation order).
    pub const CPU_MODEL_CASCADE: Self = Self {
        schedule: Schedule::Cascade,
        ..Self::CPU_MODEL
    };
}

/// Fingerprints of the C1 fork-join family (GNET-05 corpus request C1); `C1[i]` is C1-i.
const C1: [&str; 12] = [
    "ab5862dc1292cfa056ebb4f726b67eb9b0a2b1ae424b720e3bda082166caa4f6",
    "8db40fd2892187423857f4077534aac90aaf71a352aac04cc29bde22bbd2bc1a",
    "80c74dc8283178136819c20b4185156233e7439e36874d8c0e9299d525af1e97",
    "ebd985cfdc2df6c366232b818889b987cb3bc22311a04a52beea682a94482e11",
    "1aac720d2219e36b3a6f65226d11402eb7dfb67128edf8b409018e36f97bd062",
    "6d4e2b4ca4f216eab4e0bf8ede9203a36a3c025a31a01be366b84febd9c14c91",
    "242f1ac7f7b24b79801280dc62eac0f2fd238918dd7e10d2ee7ff974125b328d",
    "142a89b21a1551a45e7c2dff146e61aeca336ae7197288d7cc7ceea6fe6205ab",
    "d66ffab6b5d8a750791f6a70b09123962caeec580b457439fd7bd057a238db38",
    "538e4837efba3043e0e84c4f245f03f4ca06c999a63c7903362492a6c0697668",
    "8a087fcd9e2784c09532ceeb04b298cfd20efaede5955e81f0cda67cd0ad1959",
    "30828f24c7bbcfd1d3bbcf68fb21c6373904af35c4a813c4dddb3b67fc37b3ba",
];
/// Apple M4, MSL 3.2 cascade (`archived-evidence/gnet-p4b/bench-metal`): every C1
/// program has a medium input above 3.0x resident.
const METAL_REFUSED: &[&str] = &C1;
/// Apple7+, MSL 3.1 staged (`archived-evidence/gnet-p4b/bench-metal-staged`).
const METAL_STAGED_REFUSED: &[&str] = &C1;
/// RTX 5070 Ti cascade (`archived-evidence/gnet-p4b/bench-cuda`): every C1 program
/// but C1-01 has an input above 3.0x resident or 2.0x host-to-host.
const CUDA_REFUSED: &[&str] = &[
    C1[0], C1[2], C1[3], C1[4], C1[5], C1[6], C1[7], C1[8], C1[9], C1[10], C1[11],
];

/// Lane step budget: loop iterations plus calls of one lane. Only bounds
/// termination; exhausting it is a tool failure, never a value.
pub const STEP_BUDGET: u32 = 1 << 28;

/// The chosen launch shape of a task region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub schedule: Schedule,
    /// Staged: ready nodes in one round that stop expansion. Cascade: arena
    /// nodes after which claimed nodes run serially.
    pub width: u32,
    pub arena: u32,
    /// Lanes of the cascade grid.
    pub lanes: u32,
    pub sched_threads: u32,
    pub run_threads: u32,
    pub budget: u32,
    /// Cascade: polls an idle worker makes, while no lane is running a node,
    /// before it marks the call stalled (a defect, never a result).
    pub poll_limit: u32,
    /// Staged: `(run, sched)` pairs encoded per host submission after the
    /// first scheduler pass. Programs without continuation calls finish in
    /// one. The cascade schedule always finishes in one dispatch.
    pub epochs: u32,
}

/// Widest expansion a plan uses: the arena never exceeds 2^20 nodes, so a
/// wider profile width could not be used.
pub const MAX_WIDTH: u32 = 1 << 20;

/// The plan width for a profile's requested `width`: a power of two in
/// `1..=MAX_WIDTH`, so no profile can overflow it or round it to zero.
fn plan_width(width: u32) -> u32 {
    width.clamp(1, MAX_WIDTH).next_power_of_two()
}

/// Choose the plan of `program` on a device described by `profile`.
pub fn plan(program: &Program, profile: &Profile) -> Plan {
    let width = plan_width(profile.width);
    // A staged round may hold `width - 1` ready nodes and expand each by the
    // largest sibling count; keep room for a few such rounds before falling
    // back. A cascade fork may start just below `width`.
    let need = width
        .saturating_mul(program.max_siblings() as u32)
        .saturating_mul(4);
    Plan {
        schedule: profile.schedule,
        width,
        arena: profile.arena.max(need.min(1 << 20)).max(16),
        lanes: profile.lanes.max(1),
        sched_threads: profile.sched_threads,
        run_threads: profile.run_threads,
        budget: STEP_BUDGET,
        poll_limit: STEP_BUDGET,
        epochs: if program.continuation_calls() { 8 } else { 1 },
    }
}

/// Choose the plan of `program` on `profile`, or refuse a region the
/// device's calibrated cost model measured above the D5 residual tier (the
/// caller runs it on the native path).
pub fn route(program: &Program, profile: &Profile) -> Result<Plan, Refusal> {
    let print: String = program
        .fingerprint()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if profile.refused.contains(&print.as_str()) {
        return Err(Refusal::CostModel);
    }
    Ok(plan(program, profile))
}

/// Whether a Metal device belongs to the class the 32 x 64 cascade grid was
/// measured on (Apple M4, including Pro and Max). Metal has no occupancy query
/// and no forward-progress guarantee for persistent threadgroups, and cascade
/// workers spin on a device-wide claim list, so other devices take the staged
/// MSL 3.1 schedule (P4b cascade liveness).
pub fn metal_cascade_measured(device_name: &str) -> bool {
    device_name == "Apple M4" || device_name.starts_with("Apple M4 ")
}

/// Padded share of the digit form: a full `radix`-ary tree of the given
/// depth launches `radix^depth` lanes, of which only `valid` lead to a call.
pub fn digit_padding(radix: u32, depth: u32, valid: u64) -> f64 {
    let lanes = u64::from(radix).saturating_pow(depth).max(1);
    1.0 - valid.min(lanes) as f64 / lanes as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Review finding: a public profile wider than 2^31 overflowed
    /// `next_power_of_two` (a debug panic, or a zero width in release).
    #[test]
    fn plan_width_is_a_bounded_power_of_two_for_every_profile() {
        for (requested, width) in [
            (0, 1),
            (1, 1),
            (8, 8),
            (64, 64),
            (16384, 16384),
            (16385, 32768),
            (MAX_WIDTH, MAX_WIDTH),
            (MAX_WIDTH + 1, MAX_WIDTH),
            ((1 << 31) + 1, MAX_WIDTH),
            (u32::MAX, MAX_WIDTH),
        ] {
            assert_eq!(plan_width(requested), width, "{requested}");
        }
    }

    /// Review finding: only the measured device class runs the cascade.
    #[test]
    fn only_the_measured_metal_class_runs_the_cascade() {
        for name in ["Apple M4", "Apple M4 Pro", "Apple M4 Max"] {
            assert!(metal_cascade_measured(name), "{name}");
        }
        for name in [
            "Apple M1",
            "Apple M2 Max",
            "Apple M3",
            "Apple M40",
            "AMD Radeon Pro",
            "",
        ] {
            assert!(!metal_cascade_measured(name), "{name}");
        }
    }

    #[test]
    fn digit_padding_matches_the_full_tree() {
        assert_eq!(digit_padding(2, 3, 8), 0.0);
        assert!((digit_padding(2, 3, 2) - 0.75).abs() < 1e-12);
    }
}
