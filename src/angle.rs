//! Certified integer degree/radian conversion shared with the native host.
//! The constants bound pi via Machin's formula, not a host floating-point value.

const Q: i128 = 10_000_000_000_000_000_000_000_000;
const PI_LOW: i128 = 31_415_926_535_897_932_384_626_252;
const PI_HIGH: i128 = 31_415_926_535_897_932_384_626_616;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AngleError {
    Overflow,
    Ambiguous,
}

fn nearest_even(numerator: i128, denominator: i128) -> i128 {
    let whole = numerator.div_euclid(denominator);
    let remainder = numerator.rem_euclid(denominator);
    match (remainder * 2).cmp(&denominator) {
        std::cmp::Ordering::Less => whole,
        std::cmp::Ordering::Greater => whole + 1,
        std::cmp::Ordering::Equal if whole.rem_euclid(2) == 0 => whole,
        std::cmp::Ordering::Equal => whole + 1,
    }
}

/// `degrees_to_radians` selects x*pi/180; false selects x*180/pi.
/// Inputs and outputs are the currently supported i32/u32 quantity scalars.
pub(crate) fn convert(
    value: i64,
    signed: bool,
    degrees_to_radians: bool,
) -> Result<i64, AngleError> {
    if (signed && i32::try_from(value).is_err()) || (!signed && u32::try_from(value).is_err()) {
        return Err(AngleError::Overflow);
    }
    let value = i128::from(value);
    let (lower, upper) = if degrees_to_radians {
        let (low_pi, high_pi) = if value < 0 {
            (PI_HIGH, PI_LOW)
        } else {
            (PI_LOW, PI_HIGH)
        };
        (
            nearest_even(value * low_pi, 180 * Q),
            nearest_even(value * high_pi, 180 * Q),
        )
    } else {
        let numerator = value * 180 * Q;
        let (low_pi, high_pi) = if value < 0 {
            (PI_LOW, PI_HIGH)
        } else {
            (PI_HIGH, PI_LOW)
        };
        (
            nearest_even(numerator, low_pi),
            nearest_even(numerator, high_pi),
        )
    };
    let (min, max) = if signed {
        (i128::from(i32::MIN), i128::from(i32::MAX))
    } else {
        (0, i128::from(u32::MAX))
    };
    if (lower < min && upper < min) || (lower > max && upper > max) {
        return Err(AngleError::Overflow);
    }
    if lower != upper {
        return Err(AngleError::Ambiguous);
    }
    if lower < min || lower > max {
        return Err(AngleError::Overflow);
    }
    i64::try_from(lower).map_err(|_| AngleError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arctan_bounds(base: i128, terms: u32) -> (i128, i128) {
        let mut low = 0;
        let mut high = 0;
        for index in 0..terms {
            let denominator = (2 * i128::from(index) + 1) * base.pow(2 * index + 1);
            let floor = Q / denominator;
            if index % 2 == 0 {
                low += floor;
                high += floor + 1;
            } else {
                low -= floor + 1;
                high -= floor;
            }
        }
        let next = Q / ((2 * i128::from(terms) + 1) * base.pow(2 * terms + 1)) + 1;
        if terms.is_multiple_of(2) {
            high += next;
        } else {
            low -= next;
        }
        (low, high)
    }

    #[test]
    fn machin_bounds_and_no_small_denominator_midpoint() {
        // pi = 16 atan(1/5) - 4 atan(1/239); alternating remainders are bounded by their next terms.
        let a = arctan_bounds(5, 20);
        let b = arctan_bounds(239, 6);
        assert_eq!((16 * a.0 - 4 * b.1, 16 * a.1 - 4 * b.0), (PI_LOW, PI_HIGH));

        // Stern-Brocot mediants have the least denominator between their neighbors.
        // A rounding midpoint for any in-range i32/u32 deg/rad conversion would
        // make pi rational with denominator at most 2*u32::MAX+1 < 10^10.
        let (mut left_n, mut left_d, mut right_n, mut right_d) = (0i128, 1i128, 1i128, 0i128);
        for _ in 0..1_000 {
            let (middle_n, middle_d) = (left_n + right_n, left_d + right_d);
            if middle_d > 10_000_000_000 {
                return;
            }
            if middle_n * Q <= PI_LOW * middle_d {
                (left_n, left_d) = (middle_n, middle_d);
            } else if middle_n * Q >= PI_HIGH * middle_d {
                (right_n, right_d) = (middle_n, middle_d);
            } else {
                panic!("a rational midpoint lies in the certified pi interval");
            }
        }
        panic!("Stern-Brocot proof did not terminate");
    }

    #[test]
    fn angle_rounding_sign_ties_and_overflow() {
        assert_eq!(nearest_even(5, 2), 2);
        assert_eq!(nearest_even(-5, 2), -2);
        assert_eq!(nearest_even(7, 2), 4);
        assert_eq!(nearest_even(-7, 2), -4);
        assert_eq!(convert(180, true, true), Ok(3));
        assert_eq!(convert(-180, true, true), Ok(-3));
        assert_eq!(convert(1, true, false), Ok(57));
        assert_eq!(
            convert(1_000_000_000, true, false),
            Err(AngleError::Overflow)
        );
    }
}
