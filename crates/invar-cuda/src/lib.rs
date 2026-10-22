#![deny(unsafe_op_in_unsafe_fn)]

//! CUDA ownership boundary. A `Work` value owns its context, stream, event,
//! allocations, and host staging until completion is confirmed.

#[cfg(test)]
#[allow(dead_code)]
#[path = "../../../test-support/child_deadline.rs"]
mod child_deadline;

pub use invar_device::Quarantine;
pub mod pages;

#[cfg(feature = "runtime")]
pub mod cuda;

pub const MAX_TRANSFER_BYTES: usize = 8 * 1024 * 1024 * 1024;

/// Bound one transfer and its concurrently owned host staging.
pub(crate) fn validate_transfer_budget(transfer: usize, owned: usize) -> Result<(), Error> {
    if transfer > MAX_TRANSFER_BYTES || owned > MAX_TRANSFER_BYTES {
        Err(Error::ResourceLimit(
            "CUDA transfer or owned staging exceeds 8 GiB",
        ))
    } else {
        Ok(())
    }
}

pub fn validate_transfer_len(len: usize) -> Result<(), Error> {
    validate_transfer_budget(len, len)
}

#[cfg(any(feature = "runtime", test))]
pub(crate) fn validate_roundtrip_len(len: usize) -> Result<(), Error> {
    validate_transfer_budget(
        len,
        len.checked_mul(4)
            .ok_or(Error::ResourceLimit("CUDA staging byte count overflow"))?,
    )
}

/// Reserve ten percent of reported free memory for the driver and other users.
#[cfg(any(feature = "runtime", test))]
pub(crate) fn validate_device_capacity(bytes: usize, free: usize) -> Result<(), Error> {
    let available = free.saturating_sub((free / 10).max(1));
    invar_device::check_memory(0, bytes as u64, u64::MAX, available as u64)
        .map_err(Error::ResourceLimit)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Unavailable,
    InvalidHandle,
    OutOfMemory,
    ResourceLimit(&'static str),
    StaleBorrow,
    Timeout,
    Canceled,
    Poisoned,
    Unprofitable,
    Limit,
    /// Matrix preflight refusal retaining checked byte accounting.
    MatrixResourceLimit {
        resource: &'static str,
        requested: usize,
        limit: usize,
    },
    Driver(String),
    Compile(String),
    /// A strict-f32 underflow guard discarded the device result before
    /// publication; the caller must produce the result on the exact CPU path.
    SubnormalFallback,
    /// A completed f32 conversion dispatch failed its checked range bound.
    IntegerOverflow,
    InvalidGuard(invar_device::floating::guard::InvalidWord),
    InvalidProbe(invar_device::f32_probe::Error),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubmitError {
    FailedQuiescent(Error),
    InFlightUnknown(Error),
}

/// The backend owns all storage touched by submitted work. `complete` may
/// return true only after device quiescence; false or error retains ownership.
pub trait Device {
    type Work;

    fn prepare(&mut self, bytes: &[u8]) -> Result<Self::Work, Error>;
    fn submit(&mut self, work: &mut Self::Work) -> Result<(), SubmitError>;
    fn complete(&mut self, work: &mut Self::Work) -> Result<bool, Error>;
    fn read(&mut self, work: &mut Self::Work) -> Result<Vec<u8>, Error>;
}

/// One stream and a bounded set of unresolved operations. A poisoned runtime
/// accepts no new work until all quarantined operations confirm completion.
pub struct Runtime<D: Device> {
    device: D,
    quarantine: Quarantine<D::Work>,
    max_quarantine: usize,
}

pub struct Prepared<'a, D: Device> {
    runtime: &'a mut Runtime<D>,
    work: D::Work,
}

pub struct Submitted<'a, D: Device> {
    runtime: &'a mut Runtime<D>,
    work: Option<D::Work>,
    completed: bool,
}

impl<D: Device> Runtime<D> {
    pub fn new(device: D, max_quarantine: usize) -> Self {
        Self {
            device,
            quarantine: Quarantine::new(max_quarantine),
            max_quarantine,
        }
    }

    pub fn prepare(&mut self, bytes: &[u8]) -> Result<Prepared<'_, D>, Error> {
        if self.quarantine.blocks_dispatch() || self.max_quarantine == 0 {
            return Err(Error::Poisoned);
        }
        validate_transfer_len(bytes.len())?;
        let work = self.device.prepare(bytes)?;
        Ok(Prepared {
            runtime: self,
            work,
        })
    }

    pub fn quarantine_len(&self) -> usize {
        self.quarantine.len()
    }

    /// Unknown work or an overflowed quarantine forbids dispatch and replay.
    pub fn blocks_dispatch(&self) -> bool {
        self.quarantine.blocks_dispatch()
    }

    /// Reap only operations for which the backend confirms device quiescence.
    pub fn reap(&mut self) {
        let device = &mut self.device;
        self.quarantine
            .reap(|work| matches!(device.complete(work), Ok(true)));
    }

    fn quarantine(&mut self, work: D::Work) {
        self.quarantine.push(work);
    }
}

impl<'a, D: Device> Prepared<'a, D> {
    pub fn submit(self) -> Result<Submitted<'a, D>, Error> {
        self.submit_classified().map_err(|error| match error {
            SubmitError::FailedQuiescent(error) | SubmitError::InFlightUnknown(error) => error,
        })
    }

    /// Preserve the backend's actual quiescence distinction for source callers.
    pub fn submit_classified(mut self) -> Result<Submitted<'a, D>, SubmitError> {
        match self.runtime.device.submit(&mut self.work) {
            Ok(()) => Ok(Submitted {
                runtime: self.runtime,
                work: Some(self.work),
                completed: false,
            }),
            Err(SubmitError::FailedQuiescent(error)) => Err(SubmitError::FailedQuiescent(error)),
            Err(SubmitError::InFlightUnknown(error)) => {
                self.runtime.quarantine(self.work);
                Err(SubmitError::InFlightUnknown(error))
            }
        }
    }
}

impl<D: Device> Submitted<'_, D> {
    pub fn is_quiescent(&self) -> bool {
        self.completed
    }
    /// Return output only after confirmed completion. An error before then
    /// leaves the token owning the work; dropping it quarantines the work.
    pub fn poll(&mut self) -> Result<Option<Vec<u8>>, Error> {
        if self.completed {
            return Err(Error::Poisoned);
        }
        let Some(work) = self.work.as_mut() else {
            return Err(Error::Poisoned);
        };
        if !self.runtime.device.complete(work)? {
            return Ok(None);
        }
        self.completed = true;
        self.runtime.device.read(work).map(Some)
    }

    /// Cancellation is only a request. Dropping the owning token retains all
    /// resources until later device quiescence is confirmed.
    pub fn cancel(self) -> Error {
        Error::Canceled
    }

    /// A deadline expiring does not imply that submitted work has stopped.
    pub fn timeout(self) -> Error {
        Error::Timeout
    }
}

impl<D: Device> Drop for Submitted<'_, D> {
    fn drop(&mut self) {
        if !self.completed
            && let Some(work) = self.work.take()
        {
            self.runtime.quarantine(work);
        }
    }
}

#[cfg(test)]
mod capacity_tests {
    use super::*;

    #[test]
    fn shared_cuda_budget_and_injected_free_capacity() {
        const GIB: usize = 1024 * 1024 * 1024;
        for bytes in [3 * GIB, 7 * GIB, 8 * GIB] {
            assert!(validate_transfer_budget(bytes, bytes).is_ok());
        }
        assert!(validate_roundtrip_len(2 * GIB).is_ok());
        assert!(matches!(
            validate_roundtrip_len(2 * GIB + 1),
            Err(Error::ResourceLimit(_))
        ));
        assert!(matches!(
            validate_transfer_budget(8 * GIB + 1, 0),
            Err(Error::ResourceLimit(_))
        ));
        assert!(matches!(
            validate_transfer_budget(0, 8 * GIB + 1),
            Err(Error::ResourceLimit(_))
        ));
        assert!(validate_device_capacity(8 * GIB, 9 * GIB).is_ok());
        let available = 9 * GIB - 9 * GIB / 10;
        assert!(validate_device_capacity(available, 9 * GIB).is_ok());
        assert!(matches!(
            validate_device_capacity(available + 1, 9 * GIB),
            Err(Error::ResourceLimit(_))
        ));
        assert!(matches!(
            validate_device_capacity(3 * GIB, GIB),
            Err(Error::ResourceLimit(_))
        ));
        assert!(matches!(
            validate_device_capacity(1, 1),
            Err(Error::ResourceLimit(_))
        ));
    }
}
