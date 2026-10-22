//! Owning host-side epoch gate. The adapter must report actual driver/command
//! fence observations. Unknown quiescence retains resources and forbids replay.
use super::Workset;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Submission {
    Submitted,
    NotSubmitted,
    Unknown,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceObservation {
    Pending,
    Complete,
    FailedQuiescent,
    Unknown,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionState {
    Ready,
    InFlight,
    CancelPending,
    Quarantined,
    Cancelled,
    Failed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceError {
    Busy,
    Budget,
    Submission,
    Pending,
    Cancelled,
    Failed,
    Unknown,
}

/// Resources include all buffers, module/pipeline, queue/stream, fence and
/// context/device needed by the backend. Dropping unfenced work quarantines it
/// by retaining the entire workset; this never replays or publishes it.
/// `W` is the admitted workset description (ABI5 `Workset` by default); the
/// lifecycle, budget and quarantine rules are identical for every workset type.
pub struct FencedExecution<R, W = Workset> {
    resources: Option<R>,
    workset: W,
    state: ExecutionState,
    submitted: u64,
    completed: u64,
    max_submissions: u64,
}
impl<R, W> FencedExecution<R, W> {
    pub fn new(workset: W, resources: R, max_submissions: u64) -> Result<Self, FenceError> {
        if max_submissions == 0 || max_submissions > 1024 {
            return Err(FenceError::Budget);
        }
        Ok(Self {
            resources: Some(resources),
            workset,
            state: ExecutionState::Ready,
            submitted: 0,
            completed: 0,
            max_submissions,
        })
    }
    pub fn state(&self) -> ExecutionState {
        self.state
    }
    pub fn workset(&self) -> &W {
        &self.workset
    }
    pub fn submitted(&self) -> u64 {
        self.submitted
    }
    pub fn completed(&self) -> u64 {
        self.completed
    }
    pub fn submit(&mut self, launch: impl FnOnce(&mut R) -> Submission) -> Result<(), FenceError> {
        if self.state != ExecutionState::Ready {
            return Err(FenceError::Busy);
        }
        if self.submitted == self.max_submissions {
            return Err(FenceError::Budget);
        }
        // Set ownership before invoking the adapter, including during unwinding.
        self.state = ExecutionState::InFlight;
        self.submitted += 1;
        match launch(self.resources.as_mut().expect("retained resources")) {
            Submission::Submitted => Ok(()),
            Submission::NotSubmitted => {
                self.state = ExecutionState::Ready;
                Err(FenceError::Submission)
            }
            Submission::Unknown => {
                self.state = ExecutionState::Quarantined;
                Err(FenceError::Unknown)
            }
        }
    }
    pub fn poll(
        &mut self,
        query: impl FnOnce(&mut R) -> FenceObservation,
    ) -> Result<(), FenceError> {
        if !matches!(
            self.state,
            ExecutionState::InFlight | ExecutionState::CancelPending | ExecutionState::Quarantined
        ) {
            return Err(FenceError::Busy);
        }
        match query(self.resources.as_mut().expect("retained resources")) {
            FenceObservation::Pending => Err(FenceError::Pending),
            FenceObservation::Unknown => {
                self.state = ExecutionState::Quarantined;
                Err(FenceError::Unknown)
            }
            FenceObservation::FailedQuiescent => {
                self.completed = self.submitted;
                self.state = ExecutionState::Failed;
                Err(FenceError::Failed)
            }
            FenceObservation::Complete => {
                self.completed = self.submitted;
                if self.state == ExecutionState::InFlight {
                    self.state = ExecutionState::Ready;
                    Ok(())
                } else {
                    self.state = ExecutionState::Cancelled;
                    Err(FenceError::Cancelled)
                }
            }
        }
    }
    pub fn cancel(&mut self) {
        self.state = match self.state {
            ExecutionState::Ready => ExecutionState::Cancelled,
            ExecutionState::InFlight => ExecutionState::CancelPending,
            other => other,
        };
    }
    /// Observation access requires an actual completed submission, not initial
    /// allocation, launch success, an empty ready queue, or timeout.
    pub fn observe<T>(&self, read: impl FnOnce(&R) -> T) -> Result<T, FenceError> {
        if self.state != ExecutionState::Ready || self.completed == 0 {
            return Err(FenceError::Pending);
        }
        Ok(read(self.resources.as_ref().expect("retained resources")))
    }
    /// Reclaim only when no submission can still reference these resources.
    pub fn reclaim(mut self) -> Result<R, FenceError> {
        if matches!(
            self.state,
            ExecutionState::InFlight | ExecutionState::CancelPending | ExecutionState::Quarantined
        ) {
            return Err(FenceError::Pending);
        }
        Ok(self.resources.take().expect("retained resources"))
    }
}
impl<R, W> Drop for FencedExecution<R, W> {
    fn drop(&mut self) {
        if matches!(
            self.state,
            ExecutionState::InFlight | ExecutionState::CancelPending | ExecutionState::Quarantined
        ) && let Some(resources) = self.resources.take()
        {
            std::mem::forget(resources);
        }
    }
}
