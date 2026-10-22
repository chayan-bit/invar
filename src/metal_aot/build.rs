//! Build time: run Apple's offline Metal compiler with the pinned options and
//! write a container. Run time: read a container file privately and bounded.
//! Only the build half spawns a compiler; it never runs during a dispatch.
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use sha2::{Digest, Sha256};

use super::*;

/// Wall-clock bound for one compiler invocation.
const TOOL_DEADLINE: Duration = Duration::from_secs(120);
const MAX_TOOL_OUTPUT: u64 = 64 * 1024;
pub const INSTALL_HINT: &str = "xcodebuild -downloadComponent MetalToolchain";

/// The pinned toolchain's identity, or a typed refusal naming what is missing.
pub fn toolchain(scratch: &Path) -> Result<Toolchain, Error> {
    let first_line = |args: &[&str], name: &str| {
        let out = tool(args, scratch, name)?;
        out.lines()
            .next()
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| Error::MissingToolchain(format!("{name}: no version output")))
    };
    let metal = first_line(&["-sdk", "macosx", "metal", "--version"], "metal")?;
    let metallib = first_line(&["-sdk", "macosx", "metallib", "--version"], "metallib")?;
    let sdk = first_line(&["--sdk", "macosx", "--show-sdk-version"], "sdk")?;
    let build = first_line(&["--sdk", "macosx", "--show-sdk-build-version"], "sdk")?;
    Ok(Toolchain {
        metal,
        metallib,
        sdk: format!("{sdk} ({build})"),
    })
}

/// Compile every kernel ahead and return a container for `route`/`entry`.
pub fn build(
    route: Route,
    entry: &str,
    source: &[u8],
    kernels: &[(String, Vec<String>)],
) -> Result<Vec<u8>, Error> {
    let scratch = Scratch::new()?;
    let toolchain = toolchain(&scratch.0)?;
    let mut compiled = Vec::with_capacity(kernels.len());
    for (index, (msl, functions)) in kernels.iter().enumerate() {
        let bytes = compile(msl, &scratch.0, index)?;
        metallib(&bytes).map_err(|e| Error::Compile(format!("metallib output:{e:?}")))?;
        compiled.push((
            Library {
                msl_sha256: Sha256::digest(msl.as_bytes()).into(),
                functions: functions.clone(),
                bytes: bytes.len(),
                sha256: Sha256::digest(&bytes).into(),
            },
            bytes,
        ));
    }
    if compiled.is_empty() || compiled.len() > MAX_LIBRARIES {
        return Err(Error::Size);
    }
    compiled.sort_by_key(|a| a.0.msl_sha256);
    compiled.dedup_by(|a, b| a.0.msl_sha256 == b.0.msl_sha256);
    let (libraries, images): (Vec<_>, Vec<_>) = compiled.into_iter().unzip();
    let header = Header {
        route,
        entry: entry.into(),
        source_sha256: Sha256::digest(source).into(),
        toolchain,
        invar: env!("CARGO_PKG_VERSION").into(),
        libraries,
    };
    encode(&header, &images)
}

/// `metal -c` then `metallib`, both with the pinned options, in `scratch`.
fn compile(msl: &str, scratch: &Path, index: usize) -> Result<Vec<u8>, Error> {
    let input = scratch.join(format!("k{index}.metal"));
    let air = scratch.join(format!("k{index}.air"));
    let lib = scratch.join(format!("k{index}.metallib"));
    private_write(&input, msl.as_bytes())?;
    let mut args = vec!["-sdk".to_owned(), "macosx".into(), "metal".into()];
    args.extend(options());
    args.extend(["-c".into(), path(&input)?, "-o".into(), path(&air)?]);
    let args = args.iter().map(String::as_str).collect::<Vec<_>>();
    tool(&args, scratch, "metal")?;
    tool(
        &[
            "-sdk",
            "macosx",
            "metallib",
            &path(&air)?,
            "-o",
            &path(&lib)?,
        ],
        scratch,
        "metallib",
    )?;
    // The compiler's output lives in the private scratch directory.
    read_capped(&lib, MAX_LIBRARY_BYTES)
}

/// Run `xcrun ARGS` with a deadline; stdout on success, a typed error
/// otherwise. A missing Metal Toolchain is reported as such.
fn tool(args: &[&str], scratch: &Path, name: &str) -> Result<String, Error> {
    let out = scratch.join(format!("{name}.out"));
    let err = scratch.join(format!("{name}.err"));
    let mut child = Command::new("xcrun")
        .args(args)
        .stdin(Stdio::null())
        .stdout(fs::File::create(&out).map_err(io)?)
        .stderr(fs::File::create(&err).map_err(io)?)
        .spawn()
        .map_err(|e| Error::MissingToolchain(format!("xcrun: {e}; install Xcode")))?;
    let begun = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().map_err(io)? {
            break status;
        }
        if begun.elapsed() > TOOL_DEADLINE {
            let _ = child.kill();
            let _ = child.wait();
            return Err(Error::Compile(format!(
                "{name}: exceeded {TOOL_DEADLINE:?}"
            )));
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let text = |p: &Path| {
        let mut s = String::new();
        let _ = fs::File::open(p).map(|f| f.take(MAX_TOOL_OUTPUT).read_to_string(&mut s));
        s
    };
    let (stdout, stderr) = (text(&out), text(&err));
    if stderr.contains("missing Metal Toolchain") || stdout.contains("missing Metal Toolchain") {
        return Err(Error::MissingToolchain(format!(
            "Apple's Metal Toolchain component is not installed; install it with `{INSTALL_HINT}`"
        )));
    }
    if !status.success() {
        return Err(Error::Compile(format!(
            "{name}: {status}: {}",
            stderr.trim()
        )));
    }
    Ok(stdout)
}

/// A private (0700) directory removed when dropped.
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Result<Self, Error> {
        let unique = format!(
            "invar-metal-build-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        );
        let dir = std::env::temp_dir().join(unique);
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder.create(&dir).map_err(io)?;
        Ok(Self(dir))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

// The archive cache uses these exact file checks and publication rules too.
#[path = "../../crates/invar-metal/src/cache_file.rs"]
mod cache_file;

fn file_error(error: cache_file::Error) -> Error {
    match error {
        cache_file::Error::Io(e) => io(e),
        cache_file::Error::Permissions => Error::Permissions,
        cache_file::Error::Size => Error::Size,
    }
}

/// Publish a container through a private, synced sibling and atomic rename.
pub fn write_atomic(output: &Path, bytes: &[u8]) -> Result<(), Error> {
    cache_file::write_atomic(output, bytes).map_err(file_error)
}

fn private_write(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options.open(path).map_err(io)?;
    file.write_all(bytes).map_err(io)?;
    file.sync_all().map_err(io)
}

/// Read a bounded, owner-writable regular file through one guarded descriptor.
pub fn read(path: &Path) -> Result<Image, Error> {
    decode(&read_capped(path, MAX_CONTAINER_BYTES)?)
}

fn read_capped(path: &Path, cap: usize) -> Result<Vec<u8>, Error> {
    cache_file::read_capped(path, cap).map_err(file_error)
}

fn path(p: &Path) -> Result<String, Error> {
    p.to_str()
        .map(str::to_owned)
        .ok_or_else(|| Error::Io("non-UTF-8 scratch path".into()))
}

fn io(e: std::io::Error) -> Error {
    Error::Io(e.to_string())
}
