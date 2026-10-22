//! Invar source generation for v0.2 reduction tests: exact f32 literals and
//! a `main` that returns 0 when every reduction equals its expected bits, or
//! the 1-based index of the first mismatch.
#![allow(dead_code)]

use num_bigint::BigInt;

/// An Invar expression whose value has exactly these binary32 bits (any NaN
/// for a NaN word).
pub fn literal(word: u32) -> String {
    let negative = word >> 31 == 1;
    let magnitude = word & 0x7fff_ffff;
    let text = if magnitude > 0x7f80_0000 {
        return "(0.0 / 0.0)".into();
    } else if magnitude == 0x7f80_0000 {
        "(1.0 / 0.0)".to_owned()
    } else {
        decimal(magnitude)
    };
    if negative { format!("(-{text})") } else { text }
}

/// The exact decimal spelling of a finite nonnegative binary32 value.
fn decimal(magnitude: u32) -> String {
    let exponent = (magnitude >> 23) as i64;
    let fraction = u64::from(magnitude & 0x7f_ffff);
    let (significand, power) = if exponent == 0 {
        (fraction, -149)
    } else {
        (fraction | 1 << 23, exponent - 150)
    };
    if power >= 0 {
        return format!("{}.0", BigInt::from(significand) << power as usize);
    }
    let places = (-power) as usize;
    let digits = (BigInt::from(significand) * BigInt::from(5).pow(places as u32)).to_string();
    let digits = format!("{digits:0>width$}", width = places + 1);
    let (whole, part) = digits.split_at(digits.len() - places);
    let part = part.trim_end_matches('0');
    format!("{whole}.{}", if part.is_empty() { "0" } else { part })
}

/// A Bool expression that holds iff `value` has the expected bits.
pub fn same(value: &str, expected: u32) -> String {
    let magnitude = expected & 0x7fff_ffff;
    if magnitude > 0x7f80_0000 {
        return format!("{value} != {value}");
    }
    if magnitude == 0 {
        let sign = if expected >> 31 == 1 { "<" } else { ">" };
        return format!("{value} == 0.0 and 1.0 / {value} {sign} 0.0");
    }
    format!("{value} == {}", literal(expected))
}

/// `cases` are (reduction call over `a{i}`/`b{i}`, a, b, expected bits).
pub fn program(cases: &[(String, Vec<u32>, Vec<u32>, u32)]) -> String {
    let mut source = String::from("fn main() -> i32 {\n");
    for (index, (call, a, b, expected)) in cases.iter().enumerate() {
        for (name, values) in [("a", a), ("b", b)] {
            let items = values.iter().map(|&w| literal(w)).collect::<Vec<_>>();
            source.push_str(&format!(
                "    let {name}{index}: Array<f32> = [{}];\n",
                items.join(", ")
            ));
        }
        let call = call
            .replace("A", &format!("a{index}"))
            .replace("B", &format!("b{index}"));
        source.push_str(&format!("    let r{index} = {call};\n"));
        source.push_str(&format!(
            "    if not ({}) {{ return {}; }}\n",
            same(&format!("r{index}"), *expected),
            index + 1
        ));
    }
    source.push_str("    return 0;\n}\n");
    source
}

#[cfg(test)]
mod tests {
    #[test]
    fn decimal_spellings_are_exact() {
        assert_eq!(super::literal(0x3f80_0000), "1.0");
        assert_eq!(super::literal(0xbf00_0000), "(-0.5)");
        assert_eq!(super::literal(0x4b80_0000), "16777216.0");
        assert_eq!(
            super::literal(1),
            "0.00000000000000000000000000000000000000000000140129846432481707092372958328991613128026194187651577175706828388979108268586060148663818836212158203125"
        );
    }
}
