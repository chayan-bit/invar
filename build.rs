use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn collect(path: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    collect_if(path, files, &|path| {
        path.extension().is_some_and(|ext| ext == "rs")
    })
}
fn collect_if(
    path: &Path,
    files: &mut Vec<PathBuf>,
    keep: &dyn Fn(&Path) -> bool,
) -> std::io::Result<()> {
    for item in fs::read_dir(path)? {
        let path = item?.path();
        if path.is_dir() {
            collect_if(&path, files, keep)?;
        } else if keep(&path) {
            files.push(path);
        }
    }
    Ok(())
}
/// Length-framed SHA-256 of `files` (sorted), each registered for re-runs.
fn hash_files(mut files: Vec<PathBuf>) -> Result<Sha256, Box<dyn std::error::Error>> {
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
    Ok(hash)
}
fn hex(hash: Sha256) -> String {
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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
    let mut hash = hash_files(files)?;
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
    println!("cargo:rustc-env=INVAR_BUILD_DIGEST={}", hex(hash));
    // The compiler half of the NVRTC compile cache key (src/nvrtc_cache.rs):
    // every file that decides how `invar-nvrtc` turns a source into a CUBIN.
    let mut compiler: Vec<_> = [
        "Cargo.lock",
        "src/bin/invar-nvrtc.rs",
        "src/nvrtc_cache.rs",
        "crates/invar-cuda/Cargo.toml",
        "crates/invar-device/Cargo.toml",
        "crates/invar-device/build.rs",
    ]
    .map(PathBuf::from)
    .into();
    for path in ["crates/invar-cuda/src", "crates/invar-device/src"] {
        collect_if(Path::new(path), &mut compiler, &|_| true)?;
    }
    println!(
        "cargo:rustc-env=INVAR_NVRTC_COMPILER_DIGEST={}",
        hex(hash_files(compiler)?)
    );
    // Windows gives a main thread 1 MiB of stack, Linux and macOS 8 MiB; give
    // the binaries the same 8 MiB so deep programs behave alike on every host.
    if std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "windows") {
        if std::env::var("CARGO_CFG_TARGET_ENV").is_ok_and(|env| env == "msvc") {
            println!("cargo:rustc-link-arg-bins=/STACK:8388608");
        } else {
            println!("cargo:rustc-link-arg-bins=-Wl,--stack,8388608");
        }
    }
    Ok(())
}
