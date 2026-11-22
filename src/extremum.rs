//! `min_of`, `max_of`, `argmin` and `argmax` over `Array<i32>`, `Array<u32>`,
//! `Array<f32>` and `Array<f64>` (#207, v0.2 NA3.15 to NA3.17).
//!
//! `min_of(a)` is the left fold of the library `min` over `a`: exact for the
//! integers, and for `f32` and `f64` IEEE 754-2019 minimumNumber with
//! `-0 < +0` (NB.4.9), so a NaN element is ignored and an all-NaN input gives
//! the canonical quiet NaN. `argmin(a)` is the lowest index whose element has
//! the bits of `min_of(a)`, or 0 when every element is NaN; `max_of` and
//! `argmax` are the same with `max`. An empty input traps
//! `INDEX_OUT_OF_BOUNDS` at the call, as reading `a[0]` would. The fold
//! ([`fold`]) keeps the first of equal elements and is associative, so chunk
//! results combine in chunk order: the bits and the index never depend on
//! the split.

pub mod fold;

pub use fold::{Elem, Kind, Outcome, combine, extremum, outcome, reference};

use crate::{
    eval::{EvalError, TrapCategory, Value},
    front::Span,
    numeric_contract::{F32Bits, F64Bits},
};

impl Elem for F32Bits {
    fn key(self) -> Option<i128> {
        fold::F32Word(self.to_bits()).key()
    }
    fn canonical(self) -> Self {
        if self.is_nan() { F32Bits::NAN } else { self }
    }
}

impl Elem for F64Bits {
    fn key(self) -> Option<i128> {
        fold::F64Word(self.to_bits()).key()
    }
    fn canonical(self) -> Self {
        if self.is_nan() { F64Bits::NAN } else { self }
    }
}

/// The engine value of a result; `None` is the empty-input trap at `span`.
pub(crate) fn value<T: Elem>(
    found: Option<Outcome<T>>,
    element: impl Fn(T) -> Value,
    span: Span,
) -> Result<Value, EvalError> {
    match found {
        Some(Outcome::Index(index)) => Ok(Value::U32(index)),
        Some(Outcome::Value(x)) => Ok(element(x)),
        None => Err(EvalError::Trap {
            category: TrapCategory::IndexOutOfBounds,
            span,
        }),
    }
}

/// The serial result of a checked call on its borrowed array operand; the
/// reference evaluator and MIR share it.
pub(crate) fn evaluate(kind: Kind, operand: &Value, span: Span) -> Result<Value, EvalError> {
    let conflict = |_| EvalError::Tool {
        code: "INTERNAL_BORROW",
        message: "extremum borrow conflict".into(),
    };
    match operand {
        Value::ArrayI32(v) => value(
            reference(kind, &v.try_borrow().map_err(conflict)?),
            Value::I32,
            span,
        ),
        Value::ArrayU32(v) => value(
            reference(kind, &v.try_borrow().map_err(conflict)?),
            Value::U32,
            span,
        ),
        Value::ArrayF32(v) => value(
            reference(kind, &v.try_borrow().map_err(conflict)?),
            Value::F32,
            span,
        ),
        Value::ArrayF64(v) => value(
            reference(kind, &v.try_borrow().map_err(conflict)?),
            Value::F64,
            span,
        ),
        _ => Err(EvalError::Tool {
            code: "INTERNAL_TYPE",
            message: "checked extremum operand is not an array".into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: f32) -> F32Bits {
        F32Bits::from_bits(x.to_bits())
    }

    /// A NaN with a payload and the sign bit set, distinct from the canonical one.
    const ODD_NAN: F32Bits = F32Bits::from_bits(0xffc0_0001);

    fn chunked<T: Elem>(
        kind: Kind,
        values: &[T],
        width: usize,
        right_ties: bool,
    ) -> Option<Outcome<T>> {
        let parts: Vec<_> = values
            .chunks(width)
            .enumerate()
            .map(|(c, chunk)| extremum(kind.is_max(), chunk, c * width))
            .collect();
        outcome(kind, combine(kind.is_max(), &parts, right_ties))
    }

    #[test]
    fn issue_207_ties_go_to_the_lowest_index_and_signed_zero_is_ordered() {
        let ints = [3, 1, 4, 1, 5, 9, 2, 6, 9];
        assert_eq!(reference(Kind::Argmin, &ints), Some(Outcome::Index(1)));
        assert_eq!(reference(Kind::Argmax, &ints), Some(Outcome::Index(5)));
        assert_eq!(reference(Kind::MinOf, &ints), Some(Outcome::Value(1)));
        assert_eq!(
            reference(Kind::MaxOf, &[7u32, 0, u32::MAX]),
            Some(Outcome::Value(u32::MAX))
        );
        assert_eq!(
            reference(Kind::MinOf, &[i32::MAX, i32::MIN, 0]),
            Some(Outcome::Value(i32::MIN))
        );
        let zeros = [f(0.0), f(-0.0), f(0.0), f(-0.0)];
        assert_eq!(reference(Kind::Argmin, &zeros), Some(Outcome::Index(1)));
        assert_eq!(reference(Kind::Argmax, &zeros), Some(Outcome::Index(0)));
        assert_eq!(
            reference(Kind::MinOf, &zeros),
            Some(Outcome::Value(f(-0.0)))
        );
        assert_eq!(reference(Kind::MaxOf, &zeros), Some(Outcome::Value(f(0.0))));
    }

    #[test]
    fn issue_207_nan_is_ignored_and_all_nan_gives_the_canonical_nan_at_index_0() {
        let values = [ODD_NAN, f(2.0), F32Bits::NAN, f(-1.0), ODD_NAN];
        assert_eq!(reference(Kind::Argmin, &values), Some(Outcome::Index(3)));
        assert_eq!(reference(Kind::Argmax, &values), Some(Outcome::Index(1)));
        let all = [ODD_NAN, F32Bits::NAN, ODD_NAN];
        assert_eq!(
            reference(Kind::MinOf, &all),
            Some(Outcome::Value(F32Bits::NAN))
        );
        assert_eq!(
            reference(Kind::MaxOf, &all),
            Some(Outcome::Value(F32Bits::NAN))
        );
        assert_eq!(reference(Kind::Argmin, &all), Some(Outcome::Index(0)));
        assert_eq!(reference(Kind::Argmax, &all), Some(Outcome::Index(0)));
        let inf = [f(f32::INFINITY), ODD_NAN, f(f32::NEG_INFINITY)];
        assert_eq!(
            reference(Kind::MinOf, &inf),
            Some(Outcome::Value(f(f32::NEG_INFINITY)))
        );
        assert_eq!(reference(Kind::Argmax, &inf), Some(Outcome::Index(0)));
        let odd64 = F64Bits::from_bits(0xfff0_0000_0000_0001);
        let wide = [
            odd64,
            F64Bits::from_bits((-0.0f64).to_bits()),
            F64Bits::from_bits(0),
        ];
        assert_eq!(reference(Kind::Argmax, &wide), Some(Outcome::Index(2)));
        assert_eq!(
            reference(Kind::MinOf, &[odd64]),
            Some(Outcome::Value(F64Bits::NAN))
        );
    }

    #[test]
    fn issue_207_empty_input_has_no_result() {
        for kind in Kind::ALL {
            assert_eq!(reference::<i32>(kind, &[]), None);
            assert_eq!(chunked::<F32Bits>(kind, &[], 3, false), None);
        }
    }

    /// The key order is the library's minimumNumber and maximumNumber: over
    /// every ordered pair of a value set rich in edge cases, and over every
    /// suffix of a mixed array, the fold equals the library fold.
    #[test]
    fn issue_207_min_of_is_the_left_fold_of_the_library_min() {
        let edge = [
            f(0.0),
            f(-0.0),
            f(1.0),
            f(-1.0),
            f(f32::INFINITY),
            f(f32::NEG_INFINITY),
            f(f32::MIN_POSITIVE),
            F32Bits::from_bits(1),
            F32Bits::from_bits(0x8000_0001),
            f(f32::MAX),
            f(f32::MIN),
            ODD_NAN,
            F32Bits::NAN,
            F32Bits::from_bits(0x7f80_0001),
        ];
        for max in [false, true] {
            let kind = if max { Kind::MaxOf } else { Kind::MinOf };
            for &x in &edge {
                for &y in &edge {
                    let library = if max { x.max(y) } else { x.min(y) };
                    let expected = if library.is_nan() {
                        F32Bits::NAN
                    } else {
                        library
                    };
                    assert_eq!(
                        reference(kind, &[x, y]),
                        Some(Outcome::Value(expected)),
                        "{x:?} {y:?}"
                    );
                }
            }
            let values = [
                f(1.5),
                ODD_NAN,
                f(-0.0),
                f(0.0),
                f(-2.0),
                F32Bits::NAN,
                f(-2.0),
                f(7.0),
            ];
            for start in 0..values.len() {
                let slice = &values[start..];
                let fold =
                    slice[1..].iter().fold(
                        slice[0],
                        |acc, &x| if max { acc.max(x) } else { acc.min(x) },
                    );
                let expected = if fold.is_nan() { F32Bits::NAN } else { fold };
                assert_eq!(
                    reference(kind, slice),
                    Some(Outcome::Value(expected)),
                    "{start}"
                );
            }
        }
        let wide = [0.5f64, -0.0, 0.0, f64::NEG_INFINITY, f64::NAN, -3.0]
            .map(|x| F64Bits::from_bits(x.to_bits()));
        for &x in &wide {
            for &y in &wide {
                let library = x.min(y);
                let expected = if library.is_nan() {
                    F64Bits::NAN
                } else {
                    library
                };
                assert_eq!(
                    reference(Kind::MinOf, &[x, y]),
                    Some(Outcome::Value(expected))
                );
            }
        }
    }

    #[test]
    fn issue_207_chunked_result_equals_the_serial_result_at_every_split() {
        let floats = [
            f(3.0),
            ODD_NAN,
            f(-0.0),
            f(1.0),
            f(0.0),
            f(-0.0),
            F32Bits::NAN,
            f(1.0),
            f(3.0),
        ];
        let ints = [4, -2, 7, -2, 7, 0, 4, -2];
        for kind in Kind::ALL {
            for width in 1..=floats.len() {
                assert_eq!(
                    chunked(kind, &floats, width, false),
                    reference(kind, &floats)
                );
            }
            for width in 1..=ints.len() {
                assert_eq!(chunked(kind, &ints, width, false), reference(kind, &ints));
            }
        }
    }

    #[test]
    fn issue_207_the_rightmost_tie_mutation_changes_the_index() {
        let values = [5, 1, 6, 1, 5];
        assert_eq!(
            chunked(Kind::Argmin, &values, 3, false),
            Some(Outcome::Index(1))
        );
        assert_eq!(
            chunked(Kind::Argmin, &values, 3, true),
            Some(Outcome::Index(3))
        );
    }
}
