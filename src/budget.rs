//! One byte budget for every admission decision that used to be an element cap.
//!
//! # Model
//!
//! A [`ByteBudget`] holds three ceilings in bytes, one per [`Resource`]:
//! `host_bytes` (live payload in host memory), `device_bytes` (live payload on a
//! device) and `analysis_bytes` (compile-time structures: proofs, summaries,
//! frontier storage, matcher state). A [`Footprint`] is what one region needs of
//! each resource. [`ByteBudget::admit`] compares the two and either admits or
//! returns a typed [`BudgetExceeded`] refusal (displayed as `BUDGET_EXCEEDED`).
//!
//! Both types are plain `Copy` values with no global state, so every consumer
//! shares one rule and one refusal shape:
//!
//! - **Families** (#57) compute an exact `footprint(..)` next to their
//!   admission from element counts times element size, plus staging, summaries
//!   and per-round storage, then call `admit`. The old element-count constants
//!   are gone; a larger input is refused only when it needs more bytes than the
//!   budget holds.
//! - **Run budget** (#37): `--memory-limit` becomes
//!   [`ByteBudget::from_run_budget`]; every family then sees the run's host
//!   ceiling without its own plumbing.
//! - **Source limits** (#64): module, record, view and MIR sizes are charged
//!   as [`Footprint::analysis`] against `analysis_bytes`.
//! - **Growable collections** (#72): a collection keeps a running
//!   `Footprint`, extends it with [`Footprint::plus`] before each growth and
//!   calls `admit` again. `admit` is pure and cheap, so it needs no ledger.
//!
//! # Rules
//!
//! - Arithmetic never panics and never wraps. A footprint that overflows
//!   `usize` records that resource as unknown (`None`), and `admit` refuses it
//!   with `needed: None`.
//! - A footprint is admitted when `needed <= available`, so a budget of exactly
//!   the footprint admits and one byte less refuses.
//! - A footprint that asks for a resource the budget holds none of (for
//!   example device bytes against a zero budget) is refused; a footprint that
//!   asks for zero bytes of a resource is never refused on that resource.
//! - Callers may tighten a budget ([`ByteBudget::capped_by`],
//!   [`ByteBudget::with_live_device_bytes`]) and never widen one.
//! - The budget bounds payload and analysis storage that Invar itself
//!   allocates. It excludes allocator overhead, resident-set size and the
//!   storage of the checked source, AST, net and MIR unless a family charges
//!   them as analysis bytes.
//! - The CPU map and fused-chain passes and the admission guard are bounded by
//!   host bytes only (#57). `laws::MAX_ELEMENTS` still bounds scans,
//!   reductions and the segmented, compaction and pair-reduction families,
//!   whose proofs assume it; device back ends keep their own plan-size bounds
//!   (`typed_net::MAX_DAG_MAPS`, `invar_device::MAX_ELEMENTS`) and fall back to
//!   the CPU pass past them.

use crate::array_limits::{MAX_DEVICE_BYTES, MAX_LIVE_ELEMENTS};
use std::fmt;

/// Default analysis ceiling: 256 MiB of compile-time structures.
pub const DEFAULT_ANALYSIS_BYTES: usize = 256 * 1024 * 1024;

/// Default host ceiling: every live element of the language policy, as `u32`.
pub const DEFAULT_HOST_BYTES: usize = MAX_LIVE_ELEMENTS * size_of::<u32>();

/// Bytes of `count` values of `T`, or `None` when the product overflows `usize`.
pub fn array_bytes<T>(count: usize) -> Option<usize> {
    count.checked_mul(size_of::<T>())
}

/// The kinds of memory a budget bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Resource {
    Host,
    Device,
    Analysis,
}
impl Resource {
    pub const ALL: [Resource; 3] = [Resource::Host, Resource::Device, Resource::Analysis];

    pub fn name(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Device => "device",
            Self::Analysis => "analysis",
        }
    }
}

/// What one region needs, per resource. `None` means the sum overflowed `usize`
/// and is therefore larger than any budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footprint {
    host: Option<usize>,
    device: Option<usize>,
    analysis: Option<usize>,
}
impl Default for Footprint {
    fn default() -> Self {
        Self::ZERO
    }
}
impl Footprint {
    pub const ZERO: Self = Self {
        host: Some(0),
        device: Some(0),
        analysis: Some(0),
    };

    pub fn host(self, bytes: Option<usize>) -> Self {
        self.charge(Resource::Host, bytes)
    }
    pub fn device(self, bytes: Option<usize>) -> Self {
        self.charge(Resource::Device, bytes)
    }
    pub fn analysis(self, bytes: Option<usize>) -> Self {
        self.charge(Resource::Analysis, bytes)
    }
    /// Add `bytes` of `resource`; `None` (an overflowed input) poisons that resource.
    pub fn charge(mut self, resource: Resource, bytes: Option<usize>) -> Self {
        let slot = match resource {
            Resource::Host => &mut self.host,
            Resource::Device => &mut self.device,
            Resource::Analysis => &mut self.analysis,
        };
        *slot = slot.zip(bytes).and_then(|(a, b)| a.checked_add(b));
        self
    }
    /// Per-resource sum of two footprints, poisoning on overflow.
    pub fn plus(self, other: Self) -> Self {
        Resource::ALL
            .into_iter()
            .fold(self, |f, r| f.charge(r, other.bytes(r)))
    }
    /// Bytes needed of `resource`, or `None` when the sum overflowed.
    pub fn bytes(&self, resource: Resource) -> Option<usize> {
        match resource {
            Resource::Host => self.host,
            Resource::Device => self.device,
            Resource::Analysis => self.analysis,
        }
    }
}

/// Typed refusal: the region needs more of `resource` than the budget holds.
/// Execution falls back to the CPU path; it is not a source trap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetExceeded {
    pub family: &'static str,
    pub resource: Resource,
    /// `None` when computing the footprint overflowed `usize`.
    pub needed: Option<usize>,
    pub available: usize,
}
impl fmt::Display for BudgetExceeded {
    // The `BUDGET_EXCEEDED` tool-error code is registered when a receipt or explain first
    // reports it (#57 part B); until then it appears only in this message.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BUDGET_EXCEEDED {{ family: {}, resource: {}, needed: ",
            self.family,
            self.resource.name()
        )?;
        match self.needed {
            Some(n) => write!(f, "{n}")?,
            None => f.write_str("overflow")?,
        }
        write!(f, ", available: {} }}", self.available)
    }
}
impl std::error::Error for BudgetExceeded {}

/// Ceilings, in bytes, for host payload, device payload and analysis storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteBudget {
    pub host_bytes: usize,
    pub device_bytes: usize,
    pub analysis_bytes: usize,
}
impl Default for ByteBudget {
    fn default() -> Self {
        Self::standard()
    }
}
impl ByteBudget {
    /// Refuses every region that needs any byte.
    pub const ZERO: Self = Self::new(0, 0, 0);

    pub const fn new(host_bytes: usize, device_bytes: usize, analysis_bytes: usize) -> Self {
        Self {
            host_bytes,
            device_bytes,
            analysis_bytes,
        }
    }

    /// The default when no run budget is present: host from `MAX_LIVE_ELEMENTS`,
    /// device from `MAX_DEVICE_BYTES` (backends tighten it with
    /// [`Self::with_live_device_bytes`]), analysis [`DEFAULT_ANALYSIS_BYTES`].
    pub const fn standard() -> Self {
        Self::new(DEFAULT_HOST_BYTES, MAX_DEVICE_BYTES, DEFAULT_ANALYSIS_BYTES)
    }

    /// The budget of a run limited to `host_bytes` (#37's `--memory-limit`).
    /// Device and analysis ceilings keep their defaults.
    pub const fn from_run_budget(host_bytes: usize) -> Self {
        Self::new(host_bytes, MAX_DEVICE_BYTES, DEFAULT_ANALYSIS_BYTES)
    }

    pub const fn with_host_bytes(mut self, bytes: usize) -> Self {
        self.host_bytes = bytes;
        self
    }
    pub const fn with_device_bytes(mut self, bytes: usize) -> Self {
        self.device_bytes = bytes;
        self
    }
    pub const fn with_analysis_bytes(mut self, bytes: usize) -> Self {
        self.analysis_bytes = bytes;
        self
    }

    /// Clamp the device ceiling to what a live device query reports free.
    pub fn with_live_device_bytes(mut self, free: usize) -> Self {
        self.device_bytes = self.device_bytes.min(free);
        self
    }

    /// The per-resource minimum: a caller-supplied budget may tighten this one,
    /// never widen it.
    pub fn capped_by(self, ceiling: Self) -> Self {
        Self::new(
            self.host_bytes.min(ceiling.host_bytes),
            self.device_bytes.min(ceiling.device_bytes),
            self.analysis_bytes.min(ceiling.analysis_bytes),
        )
    }

    pub fn available(&self, resource: Resource) -> usize {
        match resource {
            Resource::Host => self.host_bytes,
            Resource::Device => self.device_bytes,
            Resource::Analysis => self.analysis_bytes,
        }
    }

    /// Admit `need` for `family`, or name the first resource it exceeds.
    pub fn admit(&self, family: &'static str, need: &Footprint) -> Result<(), BudgetExceeded> {
        for resource in Resource::ALL {
            let available = self.available(resource);
            match need.bytes(resource) {
                Some(n) if n <= available => {}
                needed => {
                    return Err(BudgetExceeded {
                        family,
                        resource,
                        needed,
                        available,
                    });
                }
            }
        }
        Ok(())
    }
}
