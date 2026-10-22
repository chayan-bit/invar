//! Device admission reuses the source/net/extraction proof of the parent matcher.
use super::*;
use invar_device::stencil_u32;

impl Contract {
    /// Build an immutable u32 multi-phase plan from this source-extracted step.
    /// The caller repeats this exact step; no ordinary checked-add is wrapped.
    pub fn device_u32(
        &self,
        dimensions: [u32; 2],
        steps: u32,
        caps: Capabilities,
    ) -> Result<stencil_u32::Verified, Error> {
        if self.element != Type::ArrayU32 {
            return Err(Error::Unrecognized);
        }
        self.verified_mir()?;
        let [rows, cols] = dimensions;
        let count = u64::from(rows)
            .checked_mul(u64::from(cols))
            .ok_or(Error::Bounds)?;
        let origin = Provenance {
            source_identity: self.net.graph.source_identity,
            graph_revision: self.net.graph.revision,
        };
        let buffer = |id| Buffer {
            id,
            elements: count,
            shape: dimensions,
            strides: [u64::from(cols), 1],
            offset: 0,
            alignment: 4,
        };
        stencil_u32::verify(
            stencil_u32::Ir {
                version: stencil_u32::VERSION,
                numeric_policy: stencil_u32::NUMERIC_POLICY.into(),
                layout_version: crate::layout_contract::VERSION,
                boundary_version: stencil::BOUNDARY_VERSION,
                boundary: self.boundary,
                span: invar_device::Span {
                    start: self.loop_span.start,
                    end: self.loop_span.end,
                },
                source_identity: origin.source_identity,
                graph_revision: origin.graph_revision,
                function_contract: self.function_contract,
                dimensions,
                input: buffer(0),
                output: buffer(1),
                steps,
            },
            caps,
            origin,
        )
        .map_err(Error::DeviceU32)
    }
}
