//! HA-1.2: the only native host-array invocation, a method consuming the
//! guard (HA-1 section 12.2).
use super::{HostArrayCall, MAX_ARRAYS, decode_descriptor_tag};
use crate::exports::{Descriptor, ExportLoadError, Invocation, LoadedExports};

/// Why [`HostArrayCall::invoke`] failed; every `mut` array was restored.
#[derive(Debug, PartialEq, Eq)]
pub enum HostCallError<E> {
    /// The loader refused or the transaction protocol failed.
    Loader(ExportLoadError),
    /// The caller's decode reported a non-normal outcome (trap, limit, tool).
    Decode(E),
}

impl HostArrayCall<'_> {
    /// Invoke one native array export (HA-1.2) and decide commit or restore.
    ///
    /// The guard is consumed: it builds the record view itself, so no
    /// detached record, record pointer or borrowed guard reaches the call.
    /// `decode` receives the completed invocation and the record addresses
    /// handed to native code (0 for an empty array). Every `mut` array is
    /// committed only if the loader succeeded **and** `decode` returned `Ok`;
    /// otherwise (loader error, trap, limit, tool failure, decode error or a
    /// panic) it is restored to its exact pre-call bits before returning.
    pub fn invoke<T, E>(
        mut self,
        loaded: &mut LoadedExports,
        descriptor: &Descriptor,
        scalars: &[crate::Arg<'_>],
        fuel: u64,
        decode: impl FnOnce(Invocation, &[usize]) -> Result<T, E>,
    ) -> Result<T, HostCallError<E>> {
        let mut addresses = [0usize; MAX_ARRAYS];
        let invocation = {
            let records = self.records();
            for (slot, record) in addresses.iter_mut().zip(records.iter()) {
                *slot = record.address();
            }
            let count = records.len();
            let tags = descriptor
                .parameter_tags
                .iter()
                .filter_map(|tag| decode_descriptor_tag(*tag));
            let matching = tags.clone().count() == count
                && tags.zip(records.iter()).all(|((element, mode), record)| {
                    record.element() == element && record.mode() == mode
                });
            if matching {
                // SAFETY: the view was built by this live guard from its
                // validated borrowed slices (in argument order, matching the
                // descriptor's array tags as checked above); it stays borrowed
                // and the slices untouched until the synchronous call returns.
                unsafe {
                    loaded.invoke_arrays(descriptor, scalars, records.as_ptr().cast(), count, fuel)
                }
                .map(|invocation| (invocation, count))
            } else {
                Err(ExportLoadError::Boundary(crate::Error::Layout))
            }
        };
        let outcome = match invocation {
            Ok((invocation, count)) => {
                decode(invocation, &addresses[..count]).map_err(HostCallError::Decode)
            }
            Err(error) => Err(HostCallError::Loader(error)),
        };
        self.finish(outcome)
    }
}
