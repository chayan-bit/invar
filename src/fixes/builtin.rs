//! Generators that predate the fix engine: a missing entrypoint and a unit
//! conversion.

use super::{Candidate, Edit};
use crate::front::Span;

const ENTRYPOINT: &str = "\nfn main() -> i32 { return 0; }\n";

/// Append an entrypoint: an insertion at the end of the source is safe for
/// any checked module without `main`.
pub(super) fn entrypoint(source: &[u8]) -> Vec<Candidate> {
    let at_end = Span {
        start: source.len(),
        end: source.len(),
    };
    vec![Candidate {
        description: "Add an entrypoint".to_owned(),
        edits: vec![Edit {
            span: at_end,
            replacement: ENTRYPOINT.to_owned(),
        }],
    }]
}

/// Wrap the expression at `span` in `to::<unit>(..)`.
pub fn unit_conversion(span: Span, unit: &str) -> Candidate {
    Candidate {
        description: format!("Convert to {unit}"),
        edits: vec![
            Edit {
                span: Span {
                    start: span.start,
                    end: span.start,
                },
                replacement: format!("to::<{unit}>("),
            },
            Edit {
                span: Span {
                    start: span.end,
                    end: span.end,
                },
                replacement: ")".to_owned(),
            },
        ],
    }
}
