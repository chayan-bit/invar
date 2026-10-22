//! Bounded ownership of device work whose completion is unconfirmed.

use std::mem;

/// A bounded set of submitted operations whose completion is unconfirmed.
/// Entries are released only after `reap` sees confirmed quiescence; work
/// beyond the bound is leaked (never freed) and disables dispatch for good.
pub struct Quarantine<W> {
    entries: Vec<W>,
    max: usize,
    leaked: bool,
}

impl<W> Quarantine<W> {
    pub fn new(max: usize) -> Self {
        Self {
            entries: Vec::new(),
            max,
            leaked: false,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether new work must be refused: unreaped entries remain, or some
    /// work was leaked when the bound was exhausted.
    pub fn blocks_dispatch(&self) -> bool {
        !self.entries.is_empty() || self.leaked
    }

    pub fn push(&mut self, work: W) {
        if self.entries.len() < self.max {
            self.entries.push(work);
        } else {
            // No device quiescence was confirmed. The work owns every live
            // resource, so leaking it is safer than running any destructor.
            mem::forget(work);
            self.leaked = true;
        }
    }

    /// Release, exactly once, every entry for which `done` confirms quiescence.
    pub fn reap(&mut self, mut done: impl FnMut(&mut W) -> bool) {
        self.entries.retain_mut(|work| !done(work));
    }
}

impl<W> Drop for Quarantine<W> {
    fn drop(&mut self) {
        // A pending device operation may outlive this owner; neither stream
        // destruction nor Rust drop proves completion.
        for work in self.entries.drain(..) {
            mem::forget(work);
        }
    }
}
