//! Exact GNET-02 observation agreement. Tool errors compare by registered
//! diagnostic code only: each engine words its own message (for example
//! "reference/MIR/CPU oracle call depth limit exceeded").

use invar::eval::{EvalError, Observation};

pub fn same(actual: &Observation, expected: &Observation, context: &str) {
    match (&actual.result, &expected.result) {
        (Err(EvalError::Tool { code: a, .. }), Err(EvalError::Tool { code: b, .. })) => {
            assert_eq!(a, b, "{context}")
        }
        (a, b) => assert_eq!(a, b, "{context}"),
    }
    assert_eq!(actual.steps_used, expected.steps_used, "{context}");
    assert_eq!(actual.cleanup_trace, expected.cleanup_trace, "{context}");
}
