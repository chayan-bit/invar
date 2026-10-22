//! Lowercase hexadecimal rendering shared by the digest and receipt writers.

use sha2::{Digest, Sha256};

pub use invar_device::hex;

/// Lowercase hex of the SHA-256 digest of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_two_lowercase_digits_per_byte() {
        assert_eq!(hex(&[]), "");
        assert_eq!(hex(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
    }

    #[test]
    fn metal_aot_keeps_its_public_hex_path() {
        assert_eq!(crate::metal_aot::hex(&[0x0a, 0xff]), "0aff");
    }

    #[test]
    fn sha256_hex_matches_the_known_digest_of_abc() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
