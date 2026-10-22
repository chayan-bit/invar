use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn collect(path: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for item in fs::read_dir(path)? {
        let path = item?.path();
        if path.is_dir() {
            collect(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-env-changed=INVAR_SOURCE_REVISION");
    let git = |args: &[&str]| {
        let result = Command::new("git").args(args).output().ok()?;
        result
            .status
            .success()
            .then(|| String::from_utf8(result.stdout).ok())
            .flatten()
            .map(|text| text.trim().to_owned())
    };
    let revision = std::env::var("INVAR_SOURCE_REVISION")
        .ok()
        .or_else(|| {
            // Re-run when HEAD or its branch moves, so a commit that touches no
            // hashed input still refreshes the embedded revision. A path that
            // does not exist would re-run every build, so only existing ones.
            let mut watched = vec![git(&["rev-parse", "--git-path", "HEAD"])?];
            if let Some(branch) = git(&["symbolic-ref", "-q", "HEAD"]) {
                watched.extend(git(&["rev-parse", "--git-path", &branch]));
                watched.extend(git(&["rev-parse", "--git-path", "packed-refs"]));
            }
            for path in watched.iter().filter(|path| Path::new(path).exists()) {
                println!("cargo:rerun-if-changed={path}");
            }
            git(&["rev-parse", "HEAD"])
        })
        .unwrap_or_default();
    let revision = revision.trim();
    if revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        println!("cargo:rustc-env=INVAR_SOURCE_REVISION={revision}");
    }

    let mut files = vec![
        PathBuf::from("Cargo.toml"),
        PathBuf::from("Cargo.lock"),
        PathBuf::from("build.rs"),
        PathBuf::from("crates/invar-device/src/f32-exact.inc"),
        // Embedded in standalone native hosts as well as the device crate.
        PathBuf::from("crates/invar-device/src/tc/reference.rs"),
    ];
    collect(Path::new("src"), &mut files)?;
    for path in [
        "crates/invar-foreign/src",
        "crates/invar-cuda/src",
        "crates/invar-metal/src",
        "crates/invar-device/src",
    ] {
        collect(Path::new(path), &mut files)?;
    }
    files.sort();
    let mut hash = Sha256::new();
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path.to_string_lossy();
        let bytes = fs::read(&path)?;
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    let rustc = std::env::var_os("RUSTC").ok_or("Cargo did not supply rustc")?;
    let version = Command::new(rustc).arg("-vV").output()?;
    if !version.status.success() {
        return Err("rustc build identity probe failed".into());
    }
    hash.update(version.stdout);
    for name in [
        "TARGET",
        "PROFILE",
        "OPT_LEVEL",
        "DEBUG",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_FEATURE_NATIVE",
        "CARGO_FEATURE_CUDA",
        "CARGO_FEATURE_METAL",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
        hash.update(name.as_bytes());
        hash.update(std::env::var(name).unwrap_or_default().as_bytes());
    }
    let digest = hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    println!("cargo:rustc-env=INVAR_BUILD_DIGEST={digest}");
    Ok(())
}
