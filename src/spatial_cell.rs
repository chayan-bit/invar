//! Exact binary32 cell mapping shared by source and standalone native hosts.
//! Bit encodings are validated before integer significand/exponent division.

fn parts(bits: u32) -> (u32, i32) {
    let exponent = (bits >> 23) & 255;
    if exponent == 0 {
        (bits & 0x7fffff, -149)
    } else {
        ((bits & 0x7fffff) | (1 << 23), exponent as i32 - 150)
    }
}

/// Exact floor(coord / cell), refusing non-finite coordinates, invalid cell
/// widths, and results outside i32. Callers filter excluded points first.
pub fn index(coord: u32, cell: u32) -> Option<i32> {
    if !(1..0x7f800000).contains(&cell) || coord & 0x7fffffff >= 0x7f800000 {
        return None;
    }
    let (numerator, a) = parts(coord);
    if numerator == 0 {
        return Some(0);
    }
    let (denominator, b) = parts(cell);
    let shift = a - b;
    let negative = coord >> 31 != 0;
    let (numerator, denominator) = if shift >= 0 {
        // The smallest significand ratio shifted by 64 already exceeds i32.
        if shift >= 64 {
            return None;
        }
        (u128::from(numerator) << shift, u128::from(denominator))
    } else {
        if shift <= -64 {
            return Some(if negative { -1 } else { 0 });
        }
        (u128::from(numerator), u128::from(denominator) << -shift)
    };
    let magnitude = numerator / denominator + u128::from(negative && numerator % denominator != 0);
    let limit = if negative {
        1u128 << 31
    } else {
        i32::MAX as u128
    };
    if magnitude > limit {
        return None;
    }
    Some(if negative {
        -(magnitude as i64)
    } else {
        magnitude as i64
    } as i32)
}
