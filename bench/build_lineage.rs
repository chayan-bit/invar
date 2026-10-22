//! Controlled, local build observations for M5-01. Integration is deliberately
//! separate from the existing CLI and its caller-supplied revision fields.
//!
//! The caller authorizes a reviewed build executable, nonsecret argv and version
//! probes. No ambient environment or raw subprocess output enters the receipt.
//! All tracked regular files are hashed before/after; expected_inputs additionally
//! pins reviewed manifests, generators and other inputs. Dependencies, ignored
//! files, the OS and tools remain trusted: this is not a sandbox, a reproducible
//! build proof, a signature, or cryptographic attestation of source derivation.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

// 256 MiB: hashed tool and output executables include the ~151 MB all-features Debug `invar`.
const MAX_FILE: u64 = 256 * 1024 * 1024;
const MAX_INPUT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_FILES: usize = 8192;
const MAX_RECEIPT: u64 = 4 * 1024 * 1024;
const MAX_OUTPUT: usize = 1024 * 1024;
const MAX_DEADLINE: Duration = Duration::from_secs(600);
static NEXT_RECEIPT: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Io,
    InvalidRequest,
    UnsupportedPlatform,
    InvalidPath,
    DirtyTree,
    StaleRevision,
    InputMismatch,
    SourceChanged,
    ToolChanged,
    BinaryChanged,
    ReceiptChanged,
    AlreadyExists,
    SizeLimit,
    OutputLimit,
    Timeout,
    ProcessFailed,
}

type Result<T> = std::result::Result<T, Error>;

/// The executable is resolved to an absolute regular file. Arguments must be
/// reviewed as nonsecret by the caller; the helper never executes a shell itself.
#[derive(Clone)]
pub struct Invocation {
    pub executable: PathBuf,
    pub args: Vec<String>,
}

pub struct BuildRequest {
    pub repository: PathBuf,
    pub expected_revision: String,
    /// Nonempty, repository-relative paths and SHA-256 values frozen by a caller.
    /// These must be tracked; all other tracked files are also snapshotted.
    pub expected_inputs: BTreeMap<String, String>,
    pub build: Invocation,
    /// Actual version queries (e.g. rustc -vV), not caller version strings.
    /// Both tool bytes and bounded version-output fingerprints are captured.
    pub toolchain: Vec<Invocation>,
    /// Only the documented nonsecret allowlist is accepted; no env is inherited.
    pub environment: BTreeMap<String, String>,
    /// Fresh paths under existing ignored directories inside the repository.
    pub output: PathBuf,
    pub receipt: PathBuf,
    pub deadline: Duration,
    pub output_limit: usize,
}

pub struct Receipt {
    /// Pin outside the receipt when passing it to a later report consumer.
    pub sha256: String,
    pub document: Value,
}

fn io<T>(value: std::io::Result<T>) -> Result<T> {
    value.map_err(|_| Error::Io)
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn hash_one(path: &Path) -> Result<String> {
    let mut budget = MAX_FILE;
    hash_file(path, &mut budget)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn hash_file(path: &Path, budget: &mut u64) -> Result<String> {
    let metadata = io(fs::symlink_metadata(path))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(Error::InvalidPath);
    }
    if metadata.len() > MAX_FILE || metadata.len() > *budget {
        return Err(Error::SizeLimit);
    }
    let mut file = io(File::open(path))?;
    let mut hasher = Sha256::new();
    let mut bytes = [0; 16384];
    let mut length = 0_u64;
    loop {
        let n = io(file.read(&mut bytes))?;
        if n == 0 {
            break;
        }
        length += n as u64;
        if length > MAX_FILE || length > *budget {
            return Err(Error::SizeLimit);
        }
        hasher.update(&bytes[..n]);
    }
    *budget -= length;
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Require canonical relative components and reject symlinks, including parents.
fn local_path(root: &Path, relative: &Path, fresh: bool) -> Result<PathBuf> {
    if relative.as_os_str().is_empty()
        || relative
            .to_str()
            .is_none_or(|s| s.split('/').any(|c| c.is_empty() || c == "." || c == ".."))
        || relative
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        || relative.components().any(|c| c.as_os_str() == ".git")
    {
        return Err(Error::InvalidPath);
    }
    let mut full = root.to_path_buf();
    let count = relative.components().count();
    for (index, component) in relative.components().enumerate() {
        full.push(component);
        match fs::symlink_metadata(&full) {
            Ok(meta) if meta.file_type().is_symlink() => return Err(Error::InvalidPath),
            Ok(_) if fresh && index + 1 == count => return Err(Error::AlreadyExists),
            Ok(meta) if index + 1 < count && !meta.is_dir() => {
                return Err(Error::InvalidPath);
            }
            Ok(_) => {}
            Err(error)
                if fresh && index + 1 == count && error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(Error::InvalidPath),
        }
    }
    Ok(full)
}

#[cfg(unix)]
fn terminate_group(child: &mut std::process::Child) {
    // Safe Rust has no killpg API. The OS kill utility targets only our newly
    // created process group, including ordinary descendants that retain pipes.
    if let Ok(mut kill) = Command::new("/bin/kill")
        .args(["-KILL", "--", &format!("-{}", child.id())])
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        let started = Instant::now();
        while matches!(kill.try_wait(), Ok(None)) && started.elapsed() < Duration::from_secs(1) {
            thread::sleep(Duration::from_millis(2));
        }
        let _ = kill.kill();
        let _ = kill.wait();
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Concurrent cap+1 readers avoid pipe deadlock and bound allocated output.
/// The same deadline covers parent exit AND pipe closure by its descendants.
#[cfg(unix)]
fn run(mut command: Command, until: Instant, cap: usize) -> Result<Output> {
    use std::os::unix::process::CommandExt;
    if Instant::now() >= until {
        return Err(Error::Timeout);
    }
    if !Path::new("/bin/kill").is_file() {
        return Err(Error::UnsupportedPlatform);
    }
    command
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = io(command.spawn())?;
    let stdout = child.stdout.take().ok_or(Error::Io)?;
    let stderr = child.stderr.take().ok_or(Error::Io)?;
    let read = move |pipe: Box<dyn Read + Send>| {
        let mut bytes = Vec::new();
        pipe.take(cap as u64 + 1)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    };
    let mut readers = [
        Some(thread::spawn(move || read(Box::new(stdout)))),
        Some(thread::spawn(move || read(Box::new(stderr)))),
    ];
    let mut buffers: [Option<Vec<u8>>; 2] = [None, None];
    let mut status = None;
    let result = (|| loop {
        for (reader, buffer) in readers.iter_mut().zip(&mut buffers) {
            if reader.as_ref().is_some_and(|h| h.is_finished()) {
                let handle = reader.take().ok_or(Error::Io)?;
                let bytes = io(handle.join().map_err(|_| Error::Io)?)?;
                if bytes.len() > cap {
                    return Err(Error::OutputLimit);
                }
                *buffer = Some(bytes);
            }
        }
        if Instant::now() >= until {
            return Err(Error::Timeout);
        }
        if status.is_none() {
            status = io(child.try_wait())?;
        }
        if let (Some(status), [None, None]) = (status, &readers) {
            let [stdout, stderr] = &mut buffers;
            return Ok(Output {
                status,
                stdout: stdout.take().ok_or(Error::Io)?,
                stderr: stderr.take().ok_or(Error::Io)?,
            });
        }
        thread::sleep(Duration::from_millis(2));
    })();
    // Also terminate leftover ordinary children on a successful parent exit.
    terminate_group(&mut child);
    // Readers blocked by an escaped process are detached, never joined here.
    // Builds must not daemonize/escape the process group (see trust boundary).
    result
}

#[cfg(not(unix))]
fn run(_command: Command, _until: Instant, _cap: usize) -> Result<Output> {
    Err(Error::UnsupportedPlatform)
}

fn git(root: &Path, args: &[&str], until: Instant) -> Result<Vec<u8>> {
    let mut command = Command::new("git");
    command
        .current_dir(root)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_NO_REPLACE_OBJECTS", "1")
        .args([
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
        ])
        .args(args);
    let output = run(command, until, MAX_OUTPUT)?;
    if !output.status.success() {
        return Err(Error::ProcessFailed);
    }
    Ok(output.stdout)
}

fn head(root: &Path, until: Instant) -> Result<String> {
    let bytes = git(root, &["rev-parse", "--verify", "HEAD"], until)?;
    let value = std::str::from_utf8(&bytes)
        .map_err(|_| Error::InvalidRequest)?
        .trim();
    if ![40, 64].contains(&value.len()) || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::InvalidRequest);
    }
    Ok(value.into())
}

// Git status can trust cached stat data (even with no skip-worktree flags).
// Hash actual bytes using Git's object format, then compare to the observed HEAD
// tree, so a same-size edit with restored mtime cannot appear to be a clean build.
fn verify_head_bytes(root: &Path, inputs: &BTreeMap<String, String>, until: Instant) -> Result<()> {
    let raw = git(root, &["ls-tree", "-r", "-z", "--full-tree", "HEAD"], until)?;
    let mut tree = BTreeMap::new();
    for item in raw.split(|b| *b == 0).filter(|p| !p.is_empty()) {
        let entry = std::str::from_utf8(item).map_err(|_| Error::InvalidPath)?;
        let (metadata, path) = entry.split_once('\t').ok_or(Error::InvalidPath)?;
        let parts: Vec<_> = metadata.split(' ').collect();
        if parts.len() != 3 || parts[1] != "blob" || !["100644", "100755"].contains(&parts[0]) {
            return Err(Error::InvalidPath);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let executable = io(fs::metadata(local_path(root, Path::new(path), false)?))?
                .permissions()
                .mode()
                & 0o111
                != 0;
            if executable != (parts[0] == "100755") {
                return Err(Error::DirtyTree);
            }
        }
        tree.insert(path, parts[2]);
    }
    if !tree.keys().copied().eq(inputs.keys().map(String::as_str)) {
        return Err(Error::DirtyTree);
    }
    let paths: Vec<_> = inputs.keys().map(String::as_str).collect();
    for chunk in paths.chunks(128) {
        let mut args = vec!["hash-object", "--no-filters", "--"];
        args.extend_from_slice(chunk);
        let raw = git(root, &args, until)?;
        let text = std::str::from_utf8(&raw).map_err(|_| Error::InvalidPath)?;
        let hashes: Vec<_> = text.lines().collect();
        if hashes.len() != chunk.len()
            || hashes
                .iter()
                .zip(chunk)
                .any(|(hash, path)| tree.get(path) != Some(hash))
        {
            return Err(Error::DirtyTree);
        }
    }
    Ok(())
}

fn snapshot(root: &Path, revision: &str, until: Instant) -> Result<BTreeMap<String, String>> {
    if head(root, until)? != revision {
        return Err(Error::StaleRevision);
    }
    if !git(
        root,
        &[
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        until,
    )?
    .is_empty()
    {
        return Err(Error::DirtyTree);
    }
    let raw = git(root, &["ls-files", "-v", "-z"], until)?;
    let mut inputs = BTreeMap::new();
    let mut budget = MAX_INPUT_BYTES;
    for item in raw.split(|b| *b == 0).filter(|p| !p.is_empty()) {
        // Reject assume-unchanged and skip-worktree entries that hide dirtiness.
        if !item.starts_with(b"H ") {
            return Err(Error::DirtyTree);
        }
        let relative = std::str::from_utf8(&item[2..]).map_err(|_| Error::InvalidPath)?;
        if inputs.len() >= MAX_FILES || relative.len() > 1024 {
            return Err(Error::SizeLimit);
        }
        let path = local_path(root, Path::new(relative), false)?;
        inputs.insert(relative.into(), hash_file(&path, &mut budget)?);
        if Instant::now() >= until {
            return Err(Error::Timeout);
        }
    }
    if inputs.is_empty() {
        return Err(Error::InvalidRequest);
    }
    verify_head_bytes(root, &inputs, until)?;
    Ok(inputs)
}

fn ignored_destination(root: &Path, relative: &Path, until: Instant) -> Result<PathBuf> {
    let path = local_path(root, relative, true)?;
    let parent = relative
        .parent()
        .and_then(Path::to_str)
        .filter(|p| !p.is_empty())
        .ok_or(Error::InvalidPath)?;
    // `check-ignore` exits 1 when the parent is not ignored: that alone is an
    // invalid destination. A deadline or output-limit failure stays itself.
    git(root, &["check-ignore", "--no-index", "--", parent], until).map_err(
        |error| match error {
            Error::ProcessFailed => Error::InvalidPath,
            other => other,
        },
    )?;
    Ok(path)
}

fn environment(values: &BTreeMap<String, String>) -> Result<()> {
    for (key, value) in values {
        let valid = match key.as_str() {
            "CARGO_BUILD_JOBS" => value.parse::<u16>().is_ok_and(|n| (1..=64).contains(&n)),
            "SOURCE_DATE_EPOCH" => value.parse::<u64>().is_ok(),
            "RUSTUP_TOOLCHAIN" => {
                !value.is_empty()
                    && value.len() <= 128
                    && value
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
            }
            "LANG" | "LC_ALL" => matches!(value.as_str(), "C" | "C.UTF-8" | "en_US.UTF-8"),
            "TZ" => value == "UTC",
            // Paths are explicit nonsecret configuration; never read ambient env.
            "PATH" | "CARGO_HOME" | "RUSTUP_HOME" => {
                !value.is_empty()
                    && value.len() <= 4096
                    && value
                        .split(if key == "PATH" { ':' } else { '\0' })
                        .all(|p| Path::new(p).is_absolute() && !p.contains(['\n', '\r', '\0']))
            }
            _ => false,
        };
        if !valid {
            return Err(Error::InvalidRequest);
        }
    }
    Ok(())
}

fn tool(invocation: &Invocation) -> Result<Value> {
    if !invocation.executable.is_absolute()
        || invocation.args.len() > 128
        || invocation
            .args
            .iter()
            .any(|a| a.len() > 4096 || a.contains(['\0', '\n', '\r']))
    {
        return Err(Error::InvalidRequest);
    }
    let executable = io(fs::canonicalize(&invocation.executable))?;
    let name = executable.to_str().ok_or(Error::InvalidPath)?;
    let hash = hash_one(&executable)?;
    Ok(json!({"executable": name, "argv": invocation.args, "executable_sha256": hash}))
}

fn command(root: &Path, spec: &Value, env: &BTreeMap<String, String>) -> Result<Command> {
    let executable = spec["executable"].as_str().ok_or(Error::InvalidRequest)?;
    let mut command = Command::new(executable);
    command.current_dir(root).env_clear().envs(env);
    for arg in spec["argv"].as_array().ok_or(Error::InvalidRequest)? {
        command.arg(arg.as_str().ok_or(Error::InvalidRequest)?);
    }
    Ok(command)
}

fn fingerprint(output: &Output) -> Value {
    json!({"stdout_bytes": output.stdout.len(), "stdout_sha256": digest(&output.stdout),
        "stderr_bytes": output.stderr.len(), "stderr_sha256": digest(&output.stderr),
        "exit_code": output.status.code()})
}

fn check_tools(
    build: &Invocation,
    expected: &Value,
    probes: &[Invocation],
    records: &[Value],
) -> Result<()> {
    if tool(build)? != *expected
        || probes
            .iter()
            .zip(records)
            .any(|(probe, record)| tool(probe).as_ref() != Ok(&record["command"]))
    {
        return Err(Error::ToolChanged);
    }
    Ok(())
}

/// Build from a clean observed revision, checking reviewed input hashes before
/// execution and rechecking every tracked input, HEAD and tools afterward. A
/// preexisting output is rejected even if the requested build would overwrite it.
/// Only a successful bounded run can publish a new receipt. Failures may leave
/// build outputs; choose new paths or explicitly inspect/remove them before retry.
pub fn build(request: &BuildRequest) -> Result<Receipt> {
    if request.deadline.is_zero()
        || request.deadline > MAX_DEADLINE
        || request.output_limit == 0
        || request.output_limit > MAX_OUTPUT
        || request.toolchain.is_empty()
        || request.toolchain.len() > 8
        || request.expected_inputs.is_empty()
        || request.expected_inputs.len() > MAX_FILES
        || request.expected_inputs.values().any(|h| !valid_digest(h))
        || request.output == request.receipt
    {
        return Err(Error::InvalidRequest);
    }
    environment(&request.environment)?;
    let until = Instant::now() + request.deadline;
    let root = io(fs::canonicalize(&request.repository))?;
    let top = git(&root, &["rev-parse", "--show-toplevel"], until)?;
    if std::str::from_utf8(&top)
        .map_err(|_| Error::InvalidPath)?
        .trim()
        != root.to_str().ok_or(Error::InvalidPath)?
    {
        return Err(Error::InvalidPath);
    }
    let before = snapshot(&root, &request.expected_revision, until)?;
    for (path, hash) in &request.expected_inputs {
        if before.get(path) != Some(hash) {
            return Err(Error::InputMismatch);
        }
    }
    let output_path = ignored_destination(&root, &request.output, until)?;
    let receipt_path = ignored_destination(&root, &request.receipt, until)?;
    let build = tool(&request.build)?;
    let mut probes = Vec::new();
    for invocation in &request.toolchain {
        let spec = tool(invocation)?;
        let version = run(
            command(&root, &spec, &request.environment)?,
            until,
            request.output_limit,
        )?;
        if !version.status.success() {
            return Err(Error::ProcessFailed);
        }
        probes.push(json!({"command": spec, "version_probe": fingerprint(&version)}));
    }
    local_path(&root, &request.output, true)?;
    let result = run(
        command(&root, &build, &request.environment)?,
        until,
        request.output_limit,
    )?;
    if !result.status.success() {
        return Err(Error::ProcessFailed);
    }
    let after =
        snapshot(&root, &request.expected_revision, until).map_err(|error| match error {
            Error::DirtyTree | Error::StaleRevision => Error::SourceChanged,
            other => other,
        })?;
    if before != after {
        return Err(Error::SourceChanged);
    }
    check_tools(&request.build, &build, &request.toolchain, &probes)?;
    let output_sha = hash_one(&local_path(&root, &request.output, false)?)?;
    let document = json!({"schema_version": 1, "kind": "controlled_build_receipt",
        "provenance": "local_observed_build_not_trusted_attestation", "worktree_at_build": "clean",
        "source_revision": request.expected_revision, "tracked_inputs_sha256": before,
        "expected_inputs_sha256": request.expected_inputs, "working_directory": ".",
        "build_command": build, "toolchain": probes, "environment": request.environment,
        "environment_policy": "cleared_then_explicit_nonsecret_allowlist",
        "subprocess_output_policy": "fingerprints_only_no_raw_logs",
        "deadline_ms": request.deadline.as_millis(), "per_stream_output_limit": request.output_limit,
        "build_result": fingerprint(&result), "output_path": request.output,
        "output_sha256": output_sha, "platform": std::env::consts::OS,
        "architecture": std::env::consts::ARCH});
    let mut bytes = serde_json::to_vec_pretty(&document).map_err(|_| Error::InvalidRequest)?;
    bytes.push(b'\n');
    if bytes.len() as u64 > MAX_RECEIPT {
        return Err(Error::SizeLimit);
    }
    if Instant::now() >= until {
        return Err(Error::Timeout);
    }
    local_path(&root, &request.receipt, true)?;
    // Fully write a temporary inode, then publish without replacing any pathname.
    // hard_link is atomic create-if-absent on a local filesystem, unlike rename.
    let temp = receipt_path.with_file_name(format!(
        ".lineage-{}-{}",
        std::process::id(),
        NEXT_RECEIPT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut created = false;
    let publication = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|_| Error::AlreadyExists)?;
        created = true;
        io(file.write_all(&bytes))?;
        io(file.sync_all())?;
        let mut permissions = io(file.metadata())?.permissions();
        permissions.set_readonly(true);
        io(file.set_permissions(permissions))?;
        if hash_one(&output_path)? != document["output_sha256"] {
            return Err(Error::BinaryChanged);
        }
        fs::hard_link(&temp, &receipt_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                Error::AlreadyExists
            } else {
                Error::Io
            }
        })
    })();
    // Never remove a preexisting temp file on a name collision.
    if created {
        let _ = fs::remove_file(&temp);
    }
    publication?;
    Ok(Receipt {
        sha256: digest(&bytes),
        document,
    })
}

/// Verify against a separately pinned receipt digest and expected current HEAD.
/// A digest supplied alongside an untrusted replacement receipt proves nothing.
/// This checks current inputs/tools/output bytes; report execution must still
/// avoid races between verification and running the binary.
pub fn verify(
    repository: &Path,
    receipt: &Path,
    pinned_sha256: &str,
    revision: &str,
) -> Result<Value> {
    if !valid_digest(pinned_sha256) {
        return Err(Error::InvalidRequest);
    }
    let until = Instant::now() + Duration::from_secs(30);
    let root = io(fs::canonicalize(repository))?;
    let path = local_path(&root, receipt, false)?;
    let metadata = io(fs::symlink_metadata(&path))?;
    // Reject static FIFO/device/directory substitutions before an open that could
    // block. Concurrent pathname replacement still requires isolated ownership.
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(Error::InvalidPath);
    }
    if metadata.len() > MAX_RECEIPT {
        return Err(Error::SizeLimit);
    }
    let mut bytes = Vec::new();
    io(File::open(path))?
        .take(MAX_RECEIPT + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| Error::Io)?;
    if bytes.len() as u64 > MAX_RECEIPT {
        return Err(Error::SizeLimit);
    }
    if digest(&bytes) != pinned_sha256 {
        return Err(Error::ReceiptChanged);
    }
    let record: Value = serde_json::from_slice(&bytes).map_err(|_| Error::InvalidRequest)?;
    if record["schema_version"] != 1
        || record["kind"] != "controlled_build_receipt"
        || record["provenance"] != "local_observed_build_not_trusted_attestation"
        || record["source_revision"] != revision
    {
        return Err(Error::StaleRevision);
    }
    if serde_json::to_value(snapshot(&root, revision, until)?).map_err(|_| Error::Io)?
        != record["tracked_inputs_sha256"]
    {
        return Err(Error::InputMismatch);
    }
    let output = record["output_path"]
        .as_str()
        .ok_or(Error::InvalidRequest)?;
    if hash_one(&local_path(&root, Path::new(output), false)?)? != record["output_sha256"] {
        return Err(Error::BinaryChanged);
    }
    for spec in std::iter::once(&record["build_command"]).chain(
        record["toolchain"]
            .as_array()
            .ok_or(Error::InvalidRequest)?
            .iter()
            .map(|v| &v["command"]),
    ) {
        let path = spec["executable"].as_str().ok_or(Error::InvalidRequest)?;
        if hash_one(Path::new(path))? != spec["executable_sha256"] {
            return Err(Error::ToolChanged);
        }
    }
    Ok(record)
}

#[cfg(test)]
mod classification_tests {
    use super::*;

    #[test]
    fn issue_m5_01_build_lineage_expired_deadline_is_not_an_invalid_path() {
        // A valid fresh destination whose ignore check runs past the deadline.
        let root = std::env::temp_dir().join(format!(
            "invar-lineage-deadline-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(root.join("artifacts")).unwrap();
        let expired = Instant::now();
        let result = ignored_destination(&root, Path::new("artifacts/binary"), expired);
        fs::remove_dir_all(&root).unwrap();
        assert_eq!(result.err(), Some(Error::Timeout));
    }
}
