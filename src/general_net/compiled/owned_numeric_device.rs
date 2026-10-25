//! ABI12 device workset facts for `cuda-owned-numeric-continuation-v1`, read
//! by the compile receipt. Layout and bounds only: no allocation, loader,
//! session or CUDA call exists here.
use super::{Artifact, Error, Target};

const LANE_BLOCK: usize = 32;
const RED_ZONE_WORDS: usize = 64;
const MAX_ROOTS: usize = 1024;
/// Upper bound of the fuel, transitions, trace-event and quantum config words.
const MAX_CONFIG: u64 = 1_000_000;
/// Declared device payload bound (including red zones).
const MAX_DEVICE_BYTES: usize = 64 * 1024 * 1024;
/// Declared bound on host snapshot copies retained by a run.
const MAX_HOST_SNAPSHOT_BYTES: usize = 256 * 1024 * 1024;
const OUTPUT_WORDS: usize = 10;
const EVENT_WORDS: usize = 7;
const CONFIG_WORDS: usize = 5;

/// Read-only ABI12 workset facts for the compile receipt. Grants no
/// allocation, loader or execution authority.
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
        let (Some(owned), Some(continuation)) = (self.owned_layout(), self.continuation_layout())
        else {
            return Err(Error::NumericTarget);
        };
        if self.abi_version() != 12 || self.target() != Target::Cuda {
            return Err(Error::NumericTarget);
        }
        let task_bytes = continuation.task_bytes();
        if task_bytes == 0 || !task_bytes.is_multiple_of(8) {
            return Err(Error::State);
        }
        Ok(OwnedNumericFacts {
            input_words: self.params().len().max(1),
            output_words: OUTPUT_WORDS,
            event_words: EVENT_WORDS,
            config_words: CONFIG_WORDS,
            heap_words: owned.heap_words(),
            cleanup_words: owned.cleanup_words(),
            task_words: task_bytes / 8,
            lane_block: LANE_BLOCK,
            red_zone_words: RED_ZONE_WORDS,
            max_roots: MAX_ROOTS,
            max_fuel: MAX_CONFIG,
            max_transitions: MAX_CONFIG,
            max_trace_events: MAX_CONFIG as usize,
            max_quantum: MAX_CONFIG,
            max_device_bytes: MAX_DEVICE_BYTES,
            max_host_snapshot_bytes: MAX_HOST_SNAPSHOT_BYTES,
            min_retained_snapshots: 2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Limits, OwnedLimits, emit_resumable_owned_numeric};
    use super::*;

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
    #[test]
    fn abi12_facts_admit_only_cuda_abi12_artifacts() {
        let cuda = artifact(Target::Cuda);
        let f = cuda.owned_numeric_cuda_facts().unwrap();
        assert_eq!(f.input_words, 2);
        assert_eq!(
            f.task_words * 8,
            cuda.continuation_layout().unwrap().task_bytes()
        );
        assert_eq!(
            (f.lane_block, f.max_device_bytes, f.max_host_snapshot_bytes),
            (32, 67_108_864, 268_435_456)
        );
        let cpu = artifact(Target::Cpu);
        assert_eq!(cpu.abi_version(), 11);
        assert_eq!(cpu.owned_numeric_cuda_facts(), Err(Error::NumericTarget));
    }
}
