//! SHA-256 of a file, shared by the planner tests.
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

pub fn hash(path: &Path) -> String {
    Sha256::digest(fs::read(path).unwrap())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
