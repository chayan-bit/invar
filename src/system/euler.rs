//! Roll, pitch and yaw of a unit quaternion for generated URDF origins.
//!
//! The angles come from the correctly rounded `asin` and `atan2` of invar-math-cr-v1 (CORE-MATH),
//! never the host libm, whose `asin` differs by one ulp between glibc and the Windows CRT; the
//! generated tree is therefore identical on every platform.

use invar_math::{Function, f64_bits};

fn cr(function: Function, y: f64, x: f64) -> f64 {
    f64::from_bits(f64_bits(function, y.to_bits(), x.to_bits()))
}

/// `(roll, pitch, yaw)` of the unit quaternion `(w, x, y, z)`.
pub(super) fn rpy(w: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let roll = cr(
        Function::Atan2,
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    );
    let pitch = cr(
        Function::Asin,
        (2.0 * (w * y - z * x)).clamp(-1.0, 1.0),
        0.0,
    );
    let yaw = cr(
        Function::Atan2,
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    );
    (roll, pitch, yaw)
}

#[cfg(test)]
mod tests {
    use super::rpy;

    #[test]
    fn euler_pitch_near_a_quarter_turn_is_the_correctly_rounded_asin() {
        // asin(1 - 2^-52), the UR5e shoulder pitch: CORE-MATH and glibc give ...724, the Windows CRT ...722.
        let (_, pitch, _) = rpy(0.4999999999999999, 0.0, 1.0, 0.0);
        assert_eq!(pitch, 1.5707963057214724);
    }
}
