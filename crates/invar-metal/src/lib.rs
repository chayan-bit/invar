#![deny(unsafe_op_in_unsafe_fn)]

//! Owned Metal boundary. Only a completed command buffer may publish output.

#[cfg(any(target_os = "macos", test))]
mod cache_file;

#[derive(Debug)]
pub enum Error {
    Unavailable(String),
    Toolchain(String),
    Limit,
    /// Device preflight refusal with a stable resource reason.
    ResourceLimit(&'static str),
    /// Matrix preflight refusal retaining checked byte accounting.
    MatrixResourceLimit {
        resource: &'static str,
        requested: usize,
        limit: usize,
    },
    OutOfMemory,
    Pending,
    Device(String),
    /// The strict-f32 underflow guard fired: the device words were discarded
    /// before publication and the exact CPU path must produce the result.
    SubnormalFallback,
    /// A completed f32 dispatch refused an out-of-range integer conversion.
    IntegerOverflow,
    InvalidGuard(invar_device::floating::guard::InvalidWord),
    InvalidProbe(invar_device::f32_probe::Error),
    /// A kernel has no usable compiled-ahead library; it is never compiled
    /// from source instead.
    Library(LibraryError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LibraryError {
    /// No loaded library was built from this kernel's exact MSL.
    Missing { msl_sha256: String },
    /// The library's functions are not exactly the ones it records.
    EntryPoint(String),
    /// The Metal runtime refused the library bytes.
    Rejected(String),
    /// A library built from the same MSL is already loaded.
    Duplicate,
}

/// Setting this variable makes any runtime MSL source compilation abort the
/// process. Fresh-process tests arm it to prove the compiled-ahead route.
pub const SOURCE_TRIPWIRE: &str = "INVAR_METAL_SOURCE_TRIPWIRE";

// Eight full-size map buffers plus one four-byte runtime scalar.
#[cfg(any(target_os = "macos", test))]
const MAX_DEVICE_BYTES: usize = 8 * 1024 * 1024 * 1024 + 4;

#[cfg(any(target_os = "macos", test))]
pub(crate) fn check_aggregate_limit(total: usize) -> Result<(), Error> {
    if total > MAX_DEVICE_BYTES {
        Err(Error::ResourceLimit(
            "Metal submission exceeds 8 GiB plus 4-byte scalar limit",
        ))
    } else {
        Ok(())
    }
}

/// Check aggregate policy and live Metal capacity before allocation.
#[cfg(any(target_os = "macos", test))]
fn check_capacity(
    largest: usize,
    total: usize,
    max_buffer: u64,
    working_set: u64,
    allocated: u64,
) -> Result<(), Error> {
    check_aggregate_limit(total)?;
    invar_device::check_memory(
        largest as u64,
        total as u64,
        max_buffer,
        working_set.saturating_sub(allocated),
    )
    .map_err(Error::ResourceLimit)
}

/// Internal failure injection; never exposed through source or CLI options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    None,
    OutOfMemory,
    LaunchFailure,
    PartialCompletion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkState {
    Prepared,
    Submitted,
    Completed,
    FailedQuiescent,
    InFlightUnknown,
}

#[cfg(any(target_os = "macos", test))]
impl WorkState {
    pub(crate) fn submitted(self) -> Option<Self> {
        (self == Self::Prepared).then_some(Self::Submitted)
    }

    pub(crate) fn observed(self, completed: Option<bool>) -> Self {
        match completed {
            Some(true) => Self::Completed,
            Some(false) => Self::FailedQuiescent,
            None => self,
        }
    }

    pub(crate) fn uncertain(self) -> Self {
        Self::InFlightUnknown
    }
}

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "macos")]
pub use macos::gnet;
#[cfg(target_os = "macos")]
pub use macos::task;
#[cfg(target_os = "macos")]
pub use macos::{
    ArchiveRefusal, ArchiveReport, ArchiveState, BitsMap, BitsMapRun, CacheReport, Diagnostic,
    FloatingResult, FsumBench, FsumResult, GeneratedBenchResult, GridResult, HostPages,
    MAX_LIBRARIES, MAX_PIPELINES, MetalDevice, PATHSUM_GROUP, PathSum, PipelineMeasurement,
    ResidentMap, SegmentedResult, SortResult, SpatialResult, SummaryBench, SummaryResult,
    WarmMutation,
};

#[cfg(not(target_os = "macos"))]
pub struct MetalDevice;

#[cfg(not(target_os = "macos"))]
impl MetalDevice {
    pub fn open() -> Result<Self, Error> {
        Err(Error::Unavailable(
            "Metal requires an Apple Silicon Mac".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, WorkState, check_capacity};

    #[test]
    fn shared_metal_budget_and_injected_device_capacity() {
        const GIB: usize = 1024 * 1024 * 1024;
        for total in [3 * GIB, 7 * GIB, 8 * GIB, 8 * GIB + 4] {
            assert!(
                check_capacity(GIB, total, 2 * GIB as u64, 10 * GIB as u64, GIB as u64).is_ok()
            );
        }
        assert!(matches!(
            check_capacity(GIB, 8 * GIB + 5, u64::MAX, u64::MAX, 0),
            Err(Error::ResourceLimit(_))
        ));
        assert!(matches!(
            check_capacity(GIB + 1, 3 * GIB, GIB as u64, 10 * GIB as u64, 0),
            Err(Error::ResourceLimit(_))
        ));
        assert!(matches!(
            check_capacity(GIB, 7 * GIB, 2 * GIB as u64, 8 * GIB as u64, 2 * GIB as u64),
            Err(Error::ResourceLimit(_))
        ));
    }

    #[test]
    fn issue_m3_06_metal_state_machine_only_releases_terminal_work() {
        let prepared = WorkState::Prepared;
        assert_eq!(prepared.submitted(), Some(WorkState::Submitted));
        assert_eq!(prepared.submitted().unwrap().submitted(), None);
        let unknown = WorkState::Submitted.uncertain();
        assert_eq!(unknown.observed(None), WorkState::InFlightUnknown);
        assert_eq!(unknown.observed(Some(true)), WorkState::Completed);
        assert_eq!(unknown.observed(Some(false)), WorkState::FailedQuiescent);
    }
}
