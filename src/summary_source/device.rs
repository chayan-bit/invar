//! Device lowering for prepared bounded summary scans (M4-07). The device plan
//! comes only from this candidate's freshly verified typed extraction, so its
//! family, observation and provenance are the checked source's, never a
//! caller's. Publication authority stays with the CPU host.
use super::{Error, Prepared};
use crate::typed_net::{BoundedFamily, Plan, Prefix};
use invar_device::{Capabilities, Provenance, summary};

impl Prepared {
    /// Verified device plan for this candidate's own steps and typed scan.
    pub fn device_plan(&self, caps: Capabilities) -> Result<summary::Verified, Error> {
        let Plan::BoundedScan {
            family,
            prefix,
            source_identity,
            graph_revision,
            ..
        } = self.typed_plan()?
        else {
            return Err(Error::RuntimeMismatch);
        };
        let family = match family {
            BoundedFamily::Table(k) => summary::Family::Table(k),
            BoundedFamily::Matrix(d) => summary::Family::Matrix(d),
        };
        let observation = match prefix {
            Prefix::None => summary::Observation::FinalState,
            Prefix::Inclusive => summary::Observation::InclusivePrefixes,
            Prefix::Exclusive => return Err(Error::RuntimeMismatch),
        };
        let plan = summary::Plan {
            family,
            observation,
            steps: self.steps(),
            source_identity,
            graph_revision,
        };
        let origin = Provenance {
            source_identity: self.source_identity(),
            graph_revision,
        };
        summary::verify(plan, origin, caps).map_err(Error::Device)
    }

    /// The CPU candidate in device output layout: flattened inclusive
    /// prefixes, or the final state.
    pub fn device_words(&self) -> Vec<u32> {
        match self.observation() {
            super::Observation::FinalState => self.final_state().to_vec(),
            super::Observation::InclusivePrefixes => self.prefixes().concat(),
        }
    }
}
