//! Embed every `targets/*.json` file (ENG-06, #61), so a new target is a new
//! file and needs no Rust change. `target::Registry::builtin` parses them.

use std::path::Path;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let dir = Path::new(&manifest).join("../../targets");
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|entry| entry.expect("readable targets directory").path())
        .filter(|path| path.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    let mut out = String::from("pub(super) const FILES: &[(&str, &str)] = &[\n");
    for path in &files {
        println!("cargo:rerun-if-changed={}", path.display());
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("UTF-8 target file name");
        let full = path.canonicalize().expect("target file path");
        out.push_str(&format!(
            "    ({stem:?}, include_str!({:?})),\n",
            full.display().to_string()
        ));
    }
    out.push_str("];\n");
    let target =
        Path::new(&std::env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join("targets.rs");
    std::fs::write(&target, out).unwrap_or_else(|e| panic!("{}: {e}", target.display()));
}
