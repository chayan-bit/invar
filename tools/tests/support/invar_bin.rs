//! Binaries of the `invar` package. `CARGO_BIN_EXE_*` only names a package's own binaries, so a
//! tools test finds them next to its own test executable (target/<profile>/deps/<test>).
//! Building both packages (`cargo test -p invar -p invar-tools`) builds them.
#![allow(dead_code)]
use std::path::PathBuf;

pub fn invar_bin(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let path = exe
        .parent()
        .and_then(|deps| deps.parent())
        .unwrap()
        .join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
    assert!(
        path.is_file(),
        "{} is missing: build the invar package binaries first (cargo build -p invar --bin {name})",
        path.display()
    );
    path
}
