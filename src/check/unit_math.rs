//! Exact rational scale and bounded SI dimension arithmetic for R1-01.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Dimension(pub(super) [i8; 8]);

impl Dimension {
    pub(super) fn mul(self, other: Self) -> Option<Self> {
        let mut result = self.0;
        for (value, rhs) in result.iter_mut().zip(other.0) {
            *value = value.checked_add(rhs)?;
        }
        Some(Self(result))
    }

    pub(super) fn div(self, other: Self) -> Option<Self> {
        let mut result = self.0;
        for (value, rhs) in result.iter_mut().zip(other.0) {
            *value = value.checked_sub(rhs)?;
        }
        Some(Self(result))
    }

    pub(super) fn pow(self, exponent: i8) -> Option<Self> {
        let mut result = self.0;
        for value in &mut result {
            *value = value.checked_mul(exponent)?;
        }
        Some(Self(result))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Rational {
    pub(super) numerator: i128,
    pub(super) denominator: i128,
}

impl Rational {
    pub(super) const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    pub(super) fn new(numerator: i128, denominator: i128) -> Option<Self> {
        if numerator <= 0 || denominator <= 0 {
            return None;
        }
        let mut a = numerator;
        let mut b = denominator;
        while b != 0 {
            (a, b) = (b, a % b);
        }
        Some(Self {
            numerator: numerator / a,
            denominator: denominator / a,
        })
    }

    pub(super) fn mul(self, other: Self) -> Option<Self> {
        let left = Self::new(self.numerator, other.denominator)?;
        let right = Self::new(other.numerator, self.denominator)?;
        Some(Self {
            numerator: left.numerator.checked_mul(right.numerator)?,
            denominator: left.denominator.checked_mul(right.denominator)?,
        })
    }

    pub(super) fn div(self, other: Self) -> Option<Self> {
        let left = Self::new(self.numerator, other.numerator)?;
        let right = Self::new(other.denominator, self.denominator)?;
        Some(Self {
            numerator: left.numerator.checked_mul(right.numerator)?,
            denominator: left.denominator.checked_mul(right.denominator)?,
        })
    }

    pub(super) fn pow(self, exponent: i8) -> Option<Self> {
        let mut result = Self::ONE;
        for _ in 0..i16::from(exponent).unsigned_abs() {
            result = result.mul(self)?;
        }
        if exponent < 0 {
            Self::ONE.div(result)
        } else {
            Some(result)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Unit {
    pub(super) dimension: Dimension,
    pub(super) scale: Rational,
    pub(super) pi_power: i8,
    pub(super) pixel: bool,
}

impl Unit {
    pub(super) fn builtin_name(self) -> Option<&'static str> {
        const NAMES: &[&str] = &[
            "one", "m", "mm", "um", "kg", "g", "s", "ms", "us", "ns", "A", "K", "mol", "cd", "rad",
            "urad", "deg", "Hz", "N", "Nm", "px",
        ];
        NAMES
            .iter()
            .copied()
            .find(|name| Self::builtin(name) == Some(self))
    }

    pub(super) fn parse(name: &str) -> Option<Self> {
        fn inner(name: &str, depth: usize) -> Option<Unit> {
            if depth > 16 {
                return None;
            }
            if let Some(unit) = Unit::builtin(name) {
                return Some(unit);
            }
            let (operation, rest) = name.split_once('<')?;
            let content = rest.strip_suffix('>')?;
            let mut nesting = 0usize;
            let comma = content.char_indices().find_map(|(index, ch)| {
                match ch {
                    '<' => nesting += 1,
                    '>' => nesting = nesting.saturating_sub(1),
                    ',' if nesting == 0 => return Some(index),
                    _ => {}
                }
                None
            })?;
            let left = inner(&content[..comma], depth + 1)?;
            let right = &content[comma + 1..];
            match operation {
                "Mul" => left.mul(inner(right, depth + 1)?),
                "Div" => left.div(inner(right, depth + 1)?),
                "Pow" => left.pow(right.parse().ok()?),
                _ => None,
            }
        }
        inner(name, 0)
    }

    pub(super) fn builtin(name: &str) -> Option<Self> {
        if name == "px" {
            return Some(Self {
                dimension: Dimension([0; 8]),
                scale: Rational::ONE,
                pi_power: 0,
                pixel: true,
            });
        }
        let (powers, numerator, denominator, pi_power) = match name {
            "one" => ([0, 0, 0, 0, 0, 0, 0, 0], 1, 1, 0),
            "m" => ([1, 0, 0, 0, 0, 0, 0, 0], 1, 1, 0),
            "mm" => ([1, 0, 0, 0, 0, 0, 0, 0], 1, 1_000, 0),
            "um" => ([1, 0, 0, 0, 0, 0, 0, 0], 1, 1_000_000, 0),
            "kg" => ([0, 1, 0, 0, 0, 0, 0, 0], 1, 1, 0),
            "g" => ([0, 1, 0, 0, 0, 0, 0, 0], 1, 1_000, 0),
            "s" => ([0, 0, 1, 0, 0, 0, 0, 0], 1, 1, 0),
            "ms" => ([0, 0, 1, 0, 0, 0, 0, 0], 1, 1_000, 0),
            "us" => ([0, 0, 1, 0, 0, 0, 0, 0], 1, 1_000_000, 0),
            "ns" => ([0, 0, 1, 0, 0, 0, 0, 0], 1, 1_000_000_000, 0),
            "A" => ([0, 0, 0, 1, 0, 0, 0, 0], 1, 1, 0),
            "K" => ([0, 0, 0, 0, 1, 0, 0, 0], 1, 1, 0),
            "mol" => ([0, 0, 0, 0, 0, 1, 0, 0], 1, 1, 0),
            "cd" => ([0, 0, 0, 0, 0, 0, 1, 0], 1, 1, 0),
            "rad" => ([0, 0, 0, 0, 0, 0, 0, 1], 1, 1, 0),
            "urad" => ([0, 0, 0, 0, 0, 0, 0, 1], 1, 1_000_000, 0),
            "deg" => ([0, 0, 0, 0, 0, 0, 0, 1], 1, 180, 1),
            "Hz" => ([0, 0, -1, 0, 0, 0, 0, 0], 1, 1, 0),
            "N" => ([1, 1, -2, 0, 0, 0, 0, 0], 1, 1, 0),
            "Nm" => ([2, 1, -2, 0, 0, 0, 0, 0], 1, 1, 0),
            _ => return None,
        };
        Some(Self {
            dimension: Dimension(powers),
            scale: Rational::new(numerator, denominator)?,
            pi_power,
            pixel: false,
        })
    }

    pub(super) fn mul(self, other: Self) -> Option<Self> {
        if self.pixel || other.pixel {
            return None;
        }
        Some(Self {
            dimension: self.dimension.mul(other.dimension)?,
            scale: self.scale.mul(other.scale)?,
            pi_power: self.pi_power.checked_add(other.pi_power)?,
            pixel: false,
        })
    }

    pub(super) fn div(self, other: Self) -> Option<Self> {
        if self.pixel || other.pixel {
            return None;
        }
        Some(Self {
            dimension: self.dimension.div(other.dimension)?,
            scale: self.scale.div(other.scale)?,
            pi_power: self.pi_power.checked_sub(other.pi_power)?,
            pixel: false,
        })
    }

    pub(super) fn pow(self, exponent: i8) -> Option<Self> {
        if self.pixel {
            return None;
        }
        Some(Self {
            dimension: self.dimension.pow(exponent)?,
            scale: self.scale.pow(exponent)?,
            pi_power: self.pi_power.checked_mul(exponent)?,
            pixel: false,
        })
    }

    pub(super) fn conversion_to(self, target: Self) -> Option<Rational> {
        (self.dimension == target.dimension
            && self.pi_power == target.pi_power
            && self.pixel == target.pixel)
            .then(|| self.scale.div(target.scale))
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_si_dimension_and_scale_algebra() {
        let get = |name| Unit::builtin(name).unwrap();
        assert_eq!(get("one").dimension, Dimension([0; 8]));
        assert_eq!(get("N").mul(get("m")), Some(get("Nm")));
        assert_eq!(
            get("m").div(get("s")).unwrap().dimension,
            Dimension([1, 0, -1, 0, 0, 0, 0, 0])
        );
        assert_eq!(get("s").pow(-1), Some(get("Hz")));
        assert_eq!(get("mm").conversion_to(get("m")), Rational::new(1, 1_000));
        assert_eq!(get("m").conversion_to(get("mm")), Rational::new(1_000, 1));
        assert_eq!(
            get("urad").conversion_to(get("rad")),
            Rational::new(1, 1_000_000)
        );
        assert_eq!(get("rad").conversion_to(get("m")), None);
        assert_eq!(get("m").pow(2).unwrap().div(get("m")), Some(get("m")));
        assert_eq!(get("deg").dimension, get("rad").dimension);
        assert_eq!(get("deg").conversion_to(get("rad")), None);
        assert_eq!(Unit::parse("Mul<N,m>"), Some(get("Nm")));
        assert_eq!(Unit::parse("Div<one,s>"), Some(get("Hz")));
        assert_eq!(
            Unit::parse("Pow<mm,2>").unwrap().dimension,
            Dimension([2, 0, 0, 0, 0, 0, 0, 0])
        );
        assert!(
            Dimension([127, 0, 0, 0, 0, 0, 0, 0])
                .mul(get("m").dimension)
                .is_none()
        );
        assert!(
            Rational::new(i128::MAX, 1)
                .unwrap()
                .mul(Rational::new(2, 1).unwrap())
                .is_none()
        );
        assert_eq!(
            Rational::new(i128::MAX, 2)
                .unwrap()
                .mul(Rational::new(2, i128::MAX).unwrap()),
            Some(Rational::ONE)
        );
    }
}
