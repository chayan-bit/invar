use super::*;
use sha2::{Digest, Sha256};

#[test]
fn vendored_sources_match_the_recorded_upstream_checksums() {
    let sums = include_str!("../core-math/SHA256SUMS");
    let mut checked = 0;
    for f in Function::ALL {
        for binary64 in [false, true] {
            let (text, headers) = source(f, binary64);
            let main = if binary64 {
                format!("./binary64/{0}/{0}.c", f.name())
            } else {
                format!("./binary32/{0}/{0}f.c", f.name())
            };
            let dir = if binary64 { "binary64" } else { "binary32" };
            let files = std::iter::once((main, text)).chain(
                headers
                    .iter()
                    .map(|(name, text)| (format!("./{dir}/{}/{name}", f.name()), *text)),
            );
            for (path, text) in files {
                let digest = Sha256::digest(text.as_bytes());
                let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
                assert!(
                    sums.lines().any(|line| line == format!("{hex}  {path}")),
                    "{path} drifted from SHA256SUMS"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(
        checked,
        sums.lines().count(),
        "every pinned file is embedded"
    );
}

#[test]
fn spot_values_are_the_correctly_rounded_results() {
    use Function as F;
    for (f, x, y, expected) in [
        (F::Sin, 1.0f64, 0.0f64, 0x3fea_ed54_8f09_0cee_u64),
        (F::Cos, 1.0, 0.0, 0x3fe1_4a28_0fb5_068c),
        (F::Exp, 1.0, 0.0, 0x4005_bf0a_8b14_5769),
        (F::Log, 2.0, 0.0, 0x3fe6_2e42_fefa_39ef),
        (F::Pow, 2.0, 0.5, 0x3ff6_a09e_667f_3bcd),
        (F::Atan2, 1.0, 1.0, 0x3fe9_21fb_5444_2d18),
        (F::Cbrt, 27.0, 0.0, 0x4008_0000_0000_0000),
        (F::Hypot, 3.0, 4.0, 0x4014_0000_0000_0000),
    ] {
        assert_eq!(
            f64_bits(f, x.to_bits(), y.to_bits()),
            expected,
            "{f:?}({x}, {y})"
        );
    }
    assert_eq!(f32_bits(F::Sin, 1.0f32.to_bits(), 0), 0x3f57_6aa4);
    assert_eq!(f32_bits(F::Exp, 1.0f32.to_bits(), 0), 0x402d_f854);
    assert_eq!(
        f32_bits(F::Atan2, 1.0f32.to_bits(), (-1.0f32).to_bits()),
        0x4016_cbe4
    );
    assert!(f64::from_bits(f64_bits(F::Log, (-1.0f64).to_bits(), 0)).is_nan());
    assert!(!ARCHIVE.is_empty());
}

#[test]
fn core_math_rounds_to_nearest_under_a_directed_host_mode() {
    // Without the bracket CORE-MATH honours the ambient mode: exp(1) and
    // pow(3, 0.5) round up to ...76a and ...cab under FE_UPWARD.
    set_upward_rounding(true);
    assert!(upward_rounding());
    let exp = f64_bits(Function::Exp, 1.0f64.to_bits(), 0);
    let pow = f64_bits(Function::Pow, 3.0f64.to_bits(), 0.5f64.to_bits());
    let restored = upward_rounding();
    set_upward_rounding(false);
    assert_eq!(exp, 0x4005_bf0a_8b14_5769);
    assert_eq!(pow, 0x3ffb_b67a_e858_4caa);
    assert!(restored, "the caller's directed mode is restored");
}
