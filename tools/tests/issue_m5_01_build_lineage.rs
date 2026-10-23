#![forbid(unsafe_code)]
#![cfg(unix)]

#[allow(dead_code)]
#[path = "../../test-support/child_deadline.rs"]
mod child_deadline;

#[path = "../bench/build_lineage.rs"]
mod build_lineage;

use build_lineage::{BuildRequest, Error, Invocation, build, verify};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        RwLock, RwLockReadGuard, RwLockWriteGuard,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
/// One deadline covers the whole build, including eight git calls before the
/// command under test starts. Tests with sub-second deadlines take this lock
/// exclusively so that other tests in this binary (git storms, output floods,
/// a real rustc build) cannot consume that budget; every other test shares it.
/// In parallel Debug runs that interference turned a pending Timeout into the
/// wrong error. External load cannot be locked out, so a case that expects a
/// non-timeout error gets a deadline far beyond what that load can consume, and
/// any expiry is classified as Timeout (see `ignored_destination`).
static DEADLINES: RwLock<()> = RwLock::new(());
fn shared() -> RwLockReadGuard<'static, ()> {
    DEADLINES.read().unwrap_or_else(|p| p.into_inner())
}
/// nextest runs each test in its own process: list every caller in .config/nextest.toml.
fn exclusive() -> RwLockWriteGuard<'static, ()> {
    DEADLINES.write().unwrap_or_else(|p| p.into_inner())
}
/// Preferred C linker drivers for the sealed rustc build, in order. The sealed
/// PATH is fixed, and rustc's implicit default (`cc`) may be a dangling host
/// alternative, so the linker is selected and passed explicitly.
const LINKERS: [&str; 2] = ["/usr/bin/cc", "/usr/bin/gcc"];

/// First candidate that canonicalizes to an executable regular file. Returns
/// every candidate's rejection reason when none qualifies; callers must fail
/// rather than skip. Host files and symlinks are only inspected.
fn select_linker(candidates: &[&Path]) -> Result<PathBuf, String> {
    let mut reasons = Vec::new();
    for candidate in candidates {
        let reason = match fs::canonicalize(candidate) {
            Err(e) => format!("does not resolve ({e})"),
            Ok(target) => match fs::metadata(&target) {
                Err(e) => format!("{} unreadable ({e})", target.display()),
                Ok(m) if !m.is_file() => format!("{} is not a regular file", target.display()),
                Ok(m) if m.permissions().mode() & 0o111 == 0 => {
                    format!("{} is not executable", target.display())
                }
                Ok(_) => return Ok(candidate.to_path_buf()),
            },
        };
        reasons.push(format!("{}: {reason}", candidate.display()));
    }
    Err(reasons.join("; "))
}
const PAYLOAD: &[u8] = b"controlled CPU build input\n";

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

impl Fixture {
    fn git(&self, args: &[&str]) -> String {
        let output = Command::new("git")
            .current_dir(&self.0)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_TERMINAL_PROMPT", "0")
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success(), "fixture git failed");
        String::from_utf8(output.stdout).unwrap().trim().into()
    }

    fn new(script: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "invar-build-lineage-{}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let fixture = Self(dir);
        fs::write(fixture.0.join("input.txt"), PAYLOAD).unwrap();
        fs::write(fixture.0.join("build.sh"), script).unwrap();
        fs::write(fixture.0.join(".gitignore"), "artifacts/\n").unwrap();
        fs::create_dir(fixture.0.join("artifacts")).unwrap();
        fixture.git(&["init", "-q"]);
        fixture.git(&["add", "input.txt", "build.sh", ".gitignore"]);
        fixture.git(&[
            "-c",
            "user.name=Lineage Fixture",
            "-c",
            "user.email=lineage@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "CPU fixture",
        ]);
        fixture
    }

    fn request(&self) -> BuildRequest {
        BuildRequest {
            repository: self.0.clone(),
            expected_revision: self.git(&["rev-parse", "HEAD"]),
            expected_inputs: BTreeMap::from([("input.txt".into(), sha(PAYLOAD))]),
            build: Invocation {
                executable: "/bin/sh".into(),
                args: vec!["build.sh".into()],
            },
            toolchain: vec![Invocation {
                executable: "/bin/sh".into(),
                args: vec!["-c".into(), "printf 'fixture tool 1.0\\n'".into()],
            }],
            environment: BTreeMap::from([
                ("PATH".into(), "/usr/bin:/bin".into()),
                ("CARGO_BUILD_JOBS".into(), "1".into()),
            ]),
            output: "artifacts/binary".into(),
            receipt: "artifacts/receipt.json".into(),
            deadline: child_deadline::wall(Duration::from_secs(3)),
            output_limit: 4096,
        }
    }
}

fn failure(fixture: &Fixture, request: &BuildRequest, expected: Error) {
    assert_eq!(build(request).err(), Some(expected));
    assert!(
        !fixture.0.join(&request.receipt).exists(),
        "failure published a receipt"
    );
}

#[test]
fn issue_m5_01_build_lineage_clean_build_and_pinned_receipt() {
    let _deadlines = shared();
    let fixture = Fixture::new(
        "cp input.txt artifacts/binary\nprintf 'build output'\nprintf 'diagnostic' >&2\n",
    );
    let request = fixture.request();
    let receipt = build(&request).unwrap();
    assert_eq!(
        receipt.document["source_revision"],
        request.expected_revision
    );
    assert_eq!(receipt.document["output_sha256"], sha(PAYLOAD));
    assert_eq!(
        receipt.document["build_result"]["stdout_sha256"],
        sha(b"build output")
    );
    assert_eq!(
        receipt.document["build_result"]["stderr_sha256"],
        sha(b"diagnostic")
    );
    assert_eq!(
        receipt.document["build_command"]["argv"],
        serde_json::json!(["build.sh"])
    );
    assert_eq!(
        receipt.document["toolchain"][0]["version_probe"]["stdout_sha256"],
        sha(b"fixture tool 1.0\n")
    );
    assert_eq!(
        receipt.document["environment"],
        serde_json::json!({"PATH":"/usr/bin:/bin", "CARGO_BUILD_JOBS":"1"})
    );
    assert_eq!(
        receipt.document["tracked_inputs_sha256"]
            .as_object()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(
        verify(
            &fixture.0,
            &request.receipt,
            &receipt.sha256,
            &request.expected_revision
        )
        .unwrap(),
        receipt.document
    );
    let bytes = fs::read(fixture.0.join(&request.receipt)).unwrap();
    assert_eq!(sha(&bytes), receipt.sha256);
    assert!(
        fs::metadata(fixture.0.join(&request.receipt))
            .unwrap()
            .permissions()
            .readonly()
    );
    assert!(!String::from_utf8(bytes).unwrap().contains("diagnostic"));
    assert_eq!(fixture.git(&["status", "--porcelain"]), "");
}

#[test]
fn issue_m5_01_build_lineage_dirty_untracked_and_hidden_index_flags_fail() {
    let _deadlines = shared();
    for mode in ["dirty", "untracked", "assume", "skip"] {
        let fixture = Fixture::new("cp input.txt artifacts/binary\n");
        let request = fixture.request();
        match mode {
            "dirty" => fs::write(fixture.0.join("input.txt"), "changed").unwrap(),
            "untracked" => fs::write(fixture.0.join("untracked"), "new").unwrap(),
            "assume" => {
                fixture.git(&["update-index", "--assume-unchanged", "input.txt"]);
            }
            "skip" => {
                fixture.git(&["update-index", "--skip-worktree", "input.txt"]);
            }
            _ => unreachable!(),
        }
        failure(&fixture, &request, Error::DirtyTree);
    }
}

#[test]
fn issue_m5_01_build_lineage_stale_revision_and_wrong_or_missing_input_fail() {
    let _deadlines = shared();
    let fixture = Fixture::new("cp input.txt artifacts/binary\n");
    let mut request = fixture.request();
    request.expected_revision = "0".repeat(40);
    failure(&fixture, &request, Error::StaleRevision);
    request.expected_revision = fixture.git(&["rev-parse", "HEAD"]);
    request
        .expected_inputs
        .insert("input.txt".into(), sha(b"stale"));
    failure(&fixture, &request, Error::InputMismatch);
    request.expected_inputs = BTreeMap::from([("missing.txt".into(), sha(b""))]);
    failure(&fixture, &request, Error::InputMismatch);
}

#[test]
fn issue_m5_01_build_lineage_source_or_head_changed_during_build_fail() {
    let _deadlines = shared();
    for script in [
        "printf changed >> input.txt\ncp input.txt artifacts/binary\n",
        "cp input.txt artifacts/binary\nprintf changed > .gitignore\n",
        "git -c user.name=Fixture -c user.email=fixture@example.invalid -c commit.gpgsign=false commit --allow-empty -qm changed\ncp input.txt artifacts/binary\n",
    ] {
        let fixture = Fixture::new(script);
        failure(&fixture, &fixture.request(), Error::SourceChanged);
    }
}

#[test]
fn issue_m5_01_build_lineage_binary_and_receipt_tampering_fail() {
    let _deadlines = shared();
    let fixture = Fixture::new("cp input.txt artifacts/binary\n");
    let request = fixture.request();
    let receipt = build(&request).unwrap();
    fs::write(fixture.0.join(&request.output), b"different binary").unwrap();
    assert_eq!(
        verify(
            &fixture.0,
            &request.receipt,
            &receipt.sha256,
            &request.expected_revision
        )
        .err(),
        Some(Error::BinaryChanged)
    );
    fs::write(fixture.0.join(&request.output), PAYLOAD).unwrap();
    assert_eq!(
        verify(
            &fixture.0,
            &request.receipt,
            &sha(b"wrong pin"),
            &request.expected_revision
        )
        .err(),
        Some(Error::ReceiptChanged)
    );
    assert_eq!(
        verify(
            &fixture.0,
            &request.receipt,
            &receipt.sha256,
            &"0".repeat(40)
        )
        .err(),
        Some(Error::StaleRevision)
    );
    use std::os::unix::fs::PermissionsExt;
    let receipt_path = fixture.0.join(&request.receipt);
    fs::set_permissions(&receipt_path, fs::Permissions::from_mode(0o644)).unwrap();
    let mut replacement = receipt.document.clone();
    replacement["output_sha256"] = serde_json::json!(sha(b"forged binary"));
    fs::write(&receipt_path, serde_json::to_vec(&replacement).unwrap()).unwrap();
    assert_eq!(
        verify(
            &fixture.0,
            &request.receipt,
            &receipt.sha256,
            &request.expected_revision
        )
        .err(),
        Some(Error::ReceiptChanged)
    );
    // Restore the originally pinned bytes to isolate the following source test.
    let mut original = serde_json::to_vec_pretty(&receipt.document).unwrap();
    original.push(b'\n');
    fs::write(&receipt_path, original).unwrap();
    fs::write(fixture.0.join("input.txt"), b"new source").unwrap();
    assert_eq!(
        verify(
            &fixture.0,
            &request.receipt,
            &receipt.sha256,
            &request.expected_revision
        )
        .err(),
        Some(Error::DirtyTree)
    );
}

#[test]
fn issue_m5_01_build_lineage_stale_output_and_receipt_overwrite_fail() {
    let _deadlines = shared();
    let fixture = Fixture::new("cp input.txt artifacts/binary\n");
    let request = fixture.request();
    fs::write(fixture.0.join(&request.output), b"stale").unwrap();
    failure(&fixture, &request, Error::AlreadyExists);
    fs::remove_file(fixture.0.join(&request.output)).unwrap();
    let receipt = build(&request).unwrap();
    let before = fs::read(fixture.0.join(&request.receipt)).unwrap();
    fs::remove_file(fixture.0.join(&request.output)).unwrap();
    assert_eq!(build(&request).err(), Some(Error::AlreadyExists));
    assert_eq!(fs::read(fixture.0.join(&request.receipt)).unwrap(), before);
    assert_eq!(sha(&before), receipt.sha256);
}

#[test]
fn issue_m5_01_build_lineage_path_traversal_unignored_and_symlink_fail() {
    let _deadlines = shared();
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new("cp input.txt artifacts/binary\n");
    for path in [
        "../escape",
        "/tmp/escape",
        "artifacts/../escape",
        "artifacts/./binary",
        "artifacts//binary",
        "new-output",
        ".git/output",
    ] {
        let mut request = fixture.request();
        request.output = path.into();
        failure(&fixture, &request, Error::InvalidPath);
    }
    symlink("/tmp", fixture.0.join("artifacts/link")).unwrap();
    let mut request = fixture.request();
    request.receipt = "artifacts/link/receipt.json".into();
    failure(&fixture, &request, Error::InvalidPath);
    request.receipt = "artifacts/receipt.json".into();
    symlink("input.txt", fixture.0.join("linked-input")).unwrap();
    fixture.git(&["add", "linked-input"]);
    fixture.git(&[
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "symlink",
    ]);
    request.expected_revision = fixture.git(&["rev-parse", "HEAD"]);
    failure(&fixture, &request, Error::InvalidPath);
}

#[test]
fn issue_m5_01_build_lineage_nonzero_missing_output_and_output_symlink_fail() {
    let _deadlines = shared();
    for (script, expected) in [
        ("exit 7\n", Error::ProcessFailed),
        ("true\n", Error::InvalidPath),
        ("ln -s ../input.txt artifacts/binary\n", Error::InvalidPath),
    ] {
        let fixture = Fixture::new(script);
        failure(&fixture, &fixture.request(), expected);
    }
}

#[test]
fn issue_m5_01_build_lineage_concurrent_capped_streams_and_timeout_fail() {
    let _deadlines = exclusive();
    let fixture = Fixture::new("yes stdout &\nyes stderr >&2 &\nwait\n");
    let mut request = fixture.request();
    request.output_limit = 256;
    // The cap, not the deadline, must end this run: a deadline far beyond the
    // 2 s bound keeps a loaded host's slow git preflight from racing it.
    // The product rejects requests above its sealed 600 s maximum.
    request.deadline = child_deadline::wall(Duration::from_secs(60)).min(Duration::from_secs(600));
    let started = Instant::now();
    failure(&fixture, &request, Error::OutputLimit);
    assert!(started.elapsed() < child_deadline::wall(Duration::from_secs(2)));
    let fixture = Fixture::new("sleep 10\n");
    let mut request = fixture.request();
    request.deadline = Duration::from_millis(200);
    let started = Instant::now();
    failure(&fixture, &request, Error::Timeout);
    eprintln!("hung build child: typed Error::Timeout; owned group killed and reaped");
    assert!(started.elapsed() < child_deadline::wall(Duration::from_secs(2)));
}

#[test]
fn issue_m5_01_build_lineage_descendant_pipe_timeout_kills_process_group() {
    let _deadlines = exclusive();
    // The parent exits successfully; its background descendant keeps pipes open.
    // After the helper returns, that descendant must not perform a delayed write.
    let fixture = Fixture::new(
        "(sleep 1; printf orphan > artifacts/orphan) &\ncp input.txt artifacts/binary\nexit 0\n",
    );
    let mut request = fixture.request();
    request.deadline = Duration::from_millis(250);
    failure(&fixture, &request, Error::Timeout);
    std::thread::sleep(Duration::from_millis(1100));
    assert!(!fixture.0.join("artifacts/orphan").exists());
}

#[test]
fn issue_m5_01_build_lineage_nonsecret_environment_and_raw_output_policy() {
    let _deadlines = shared();
    let fixture = Fixture::new(
        "[ -z \"$HOME\" ] && [ -z \"$AWS_SECRET_ACCESS_KEY\" ] || exit 9\nprintf 'fixture-sensitive-output'\ncp input.txt artifacts/binary\n",
    );
    let request = fixture.request();
    let receipt = build(&request).unwrap();
    let raw = serde_json::to_string(&receipt.document).unwrap();
    assert!(!raw.contains("fixture-sensitive-output"));
    for key in [
        "AWS_SECRET_ACCESS_KEY",
        "CARGO_REGISTRIES_CRATES_IO_TOKEN",
        "RUSTFLAGS",
        "HOME",
    ] {
        let fixture = Fixture::new("cp input.txt artifacts/binary\n");
        let mut request = fixture.request();
        request
            .environment
            .insert(key.into(), "fixture-value".into());
        failure(&fixture, &request, Error::InvalidRequest);
    }
}

#[test]
fn issue_m5_01_build_lineage_invalid_limits_and_empty_contract_fail() {
    let _deadlines = shared();
    let fixture = Fixture::new("cp input.txt artifacts/binary\n");
    let mut request = fixture.request();
    request.deadline = Duration::ZERO;
    failure(&fixture, &request, Error::InvalidRequest);
    request.deadline = child_deadline::wall(Duration::from_secs(3));
    request.output_limit = 2 * 1024 * 1024;
    failure(&fixture, &request, Error::InvalidRequest);
    request.output_limit = 4096;
    request.toolchain.clear();
    failure(&fixture, &request, Error::InvalidRequest);
    request.toolchain = fixture.request().toolchain;
    request.expected_inputs.clear();
    failure(&fixture, &request, Error::InvalidRequest);
}

#[test]
fn issue_m5_01_build_lineage_tool_binary_change_is_detected() {
    let _deadlines = shared();
    use std::os::unix::fs::PermissionsExt;
    let fixture = Fixture::new("printf changed > artifacts/tool\ncp input.txt artifacts/binary\n");
    let tool = fixture.0.join("artifacts/tool");
    fs::write(&tool, "#!/bin/sh\nprintf 'tool version 1'\n").unwrap();
    fs::set_permissions(&tool, fs::Permissions::from_mode(0o755)).unwrap();
    let mut request = fixture.request();
    request.toolchain = vec![Invocation {
        executable: tool,
        args: vec![],
    }];
    failure(&fixture, &request, Error::ToolChanged);
}

#[test]
fn issue_m5_01_build_lineage_receipt_publication_race_preserves_existing_file() {
    let _deadlines = shared();
    let fixture = Fixture::new(
        "cp input.txt artifacts/binary\nprintf 'existing receipt' > artifacts/receipt.json\n",
    );
    let request = fixture.request();
    assert_eq!(build(&request).err(), Some(Error::AlreadyExists));
    assert_eq!(
        fs::read(fixture.0.join(&request.receipt)).unwrap(),
        b"existing receipt"
    );
    assert!(
        !fs::read_dir(fixture.0.join("artifacts"))
            .unwrap()
            .any(|entry| entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".lineage-"))
    );
}

#[test]
fn issue_m5_01_build_lineage_binary_digest_matches_external_sha256sum() {
    let _deadlines = shared();
    let fixture = Fixture::new("cp input.txt artifacts/binary\n");
    let receipt = build(&fixture.request()).unwrap();
    let output = Command::new("sha256sum")
        .arg(fixture.0.join("artifacts/binary"))
        .output()
        .unwrap();
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        receipt.document["output_sha256"],
        text.split_whitespace().next().unwrap()
    );
    assert_eq!(
        sha(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn issue_m5_01_build_lineage_git_stat_cache_cannot_hide_source_change() {
    let _deadlines = shared();
    let fixture = Fixture::new("cp input.txt artifacts/binary\n");
    let request = fixture.request();
    fixture.git(&["config", "core.trustctime", "false"]);
    fixture.git(&["config", "core.checkstat", "minimal"]);
    let path = fixture.0.join("input.txt");
    fs::File::open(&path)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(SystemTime::now() - Duration::from_secs(60)))
        .unwrap();
    fixture.git(&["update-index", "--refresh"]);
    fixture.git(&["status", "--porcelain"]);
    let metadata = fs::metadata(&path).unwrap();
    let mut changed = PAYLOAD.to_vec();
    changed[0] = b'X';
    fs::write(&path, changed).unwrap();
    fs::File::open(&path)
        .unwrap()
        .set_times(
            fs::FileTimes::new()
                .set_accessed(metadata.accessed().unwrap())
                .set_modified(metadata.modified().unwrap()),
        )
        .unwrap();
    assert_eq!(
        fixture.git(&["status", "--porcelain"]),
        "",
        "fixture must exercise a falsely clean Git stat cache"
    );
    failure(&fixture, &request, Error::DirtyTree);
}

#[test]
fn issue_m5_01_build_lineage_ignored_filemode_changes_are_rejected() {
    let _deadlines = shared();
    use std::os::unix::fs::PermissionsExt;
    let fixture = Fixture::new("cp input.txt artifacts/binary\n");
    let request = fixture.request();
    fixture.git(&["config", "core.filemode", "false"]);
    fs::set_permissions(
        fixture.0.join("build.sh"),
        fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    assert_eq!(fixture.git(&["status", "--porcelain"]), "");
    failure(&fixture, &request, Error::DirtyTree);
}

#[test]
fn issue_m5_01_build_lineage_build_cannot_redirect_receipt_parent() {
    let _deadlines = shared();
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new(
        "rmdir artifacts/receipts\nln -s ../outside artifacts/receipts\ncp input.txt artifacts/binary\n",
    );
    let outside = fixture.0.join("outside");
    // Keep the alternative destination ignored too, so the failure exercises
    // post-build path validation rather than the untracked-file cleanliness gate.
    fs::write(fixture.0.join(".gitignore"), "artifacts/\noutside/\n").unwrap();
    fixture.git(&["add", ".gitignore"]);
    fixture.git(&[
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "ignored path fixture",
    ]);
    fs::create_dir(&outside).unwrap();
    fs::create_dir(fixture.0.join("artifacts/receipts")).unwrap();
    let mut request = fixture.request();
    request.receipt = "artifacts/receipts/receipt.json".into();
    failure(&fixture, &request, Error::InvalidPath);
    assert!(!outside.join("receipt.json").exists());
    // Verify an already-existing output symlink is also a path error, not stale
    // output acceptance even when its target has the expected payload hash.
    fs::remove_file(fixture.0.join("artifacts/binary")).unwrap();
    symlink("../input.txt", fixture.0.join("artifacts/binary")).unwrap();
    request.receipt = "artifacts/receipt.json".into();
    failure(&fixture, &request, Error::InvalidPath);
}

#[test]
fn issue_m5_01_build_lineage_oversized_output_is_rejected_without_reading_it() {
    let _deadlines = shared();
    let fixture = Fixture::new("truncate -s 268435457 artifacts/binary\n");
    failure(&fixture, &fixture.request(), Error::SizeLimit);
}

#[test]
fn issue_m5_01_build_lineage_version_probe_failures_prevent_build() {
    let _deadlines = exclusive();
    // Only the hanging probe needs a short deadline; the others get one no
    // loaded host's git preflight can exhaust, so the deadline cannot race them.
    for (args, expected, deadline) in [
        (
            vec!["-c".into(), "exit 8".into()],
            Error::ProcessFailed,
            60_000,
        ),
        (
            vec!["-c".into(), "yes probe".into()],
            Error::OutputLimit,
            60_000,
        ),
        (vec!["-c".into(), "sleep 10".into()], Error::Timeout, 250),
    ] {
        let fixture = Fixture::new("cp input.txt artifacts/binary\n");
        let mut request = fixture.request();
        request.toolchain[0].args = args;
        request.deadline = if expected == Error::Timeout {
            Duration::from_millis(deadline)
        } else {
            child_deadline::wall(Duration::from_millis(deadline)).min(Duration::from_secs(600))
        };
        failure(&fixture, &request, expected);
        assert!(!fixture.0.join(&request.output).exists());
    }
}

#[test]
fn issue_m5_01_build_lineage_real_cpu_rustc_build() {
    let _deadlines = shared();
    // Locate the provisioned compiler, then bypass ambient rustup selection for
    // the sealed build and version probe. No tool installation or network use.
    let sysroot = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .unwrap();
    assert!(sysroot.status.success());
    let rustc = PathBuf::from(String::from_utf8(sysroot.stdout).unwrap().trim()).join("bin/rustc");
    // Explicit, recorded linker selection: never skip when none qualifies.
    let linker = select_linker(&LINKERS.map(Path::new))
        .unwrap_or_else(|reasons| panic!("no executable C linker for the sealed build: {reasons}"));
    // Pass the canonical driver itself, so the argv names exactly the file
    // whose bytes the toolchain probe hashes (no symlink between them). Only
    // the driver is bound: its helpers (collect2, ld, as), crt objects and
    // libraries are not separately hashed.
    let canonical = fs::canonicalize(&linker).unwrap();
    let linker_arg = format!("linker={}", canonical.display());
    let fixture = Fixture::new("true\n");
    let source = b"fn main() { println!(\"42\"); }\n";
    fs::write(fixture.0.join("program.rs"), source).unwrap();
    fixture.git(&["add", "program.rs"]);
    fixture.git(&[
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "CPU compiler source",
    ]);
    let mut request = fixture.request();
    request
        .expected_inputs
        .insert("program.rs".into(), sha(source));
    request.build = Invocation {
        executable: rustc.clone(),
        args: vec![
            "--edition=2024".into(),
            "-C".into(),
            linker_arg.clone(),
            "program.rs".into(),
            "-o".into(),
            "artifacts/binary".into(),
        ],
    };
    // The linker's canonical path, executable hash and version-output
    // fingerprint are bound by the existing hashed toolchain probe records,
    // and rechecked after the build like every other probe.
    request.toolchain = vec![
        Invocation {
            executable: rustc,
            args: vec!["-vV".into()],
        },
        Invocation {
            executable: canonical.clone(),
            args: vec!["--version".into()],
        },
    ];
    request.deadline = child_deadline::wall(Duration::from_secs(5));
    let receipt = build(&request).unwrap();
    assert_eq!(
        verify(
            &fixture.0,
            &request.receipt,
            &receipt.sha256,
            &request.expected_revision
        )
        .unwrap(),
        receipt.document
    );
    let argv = &receipt.document["build_command"]["argv"];
    assert_eq!(argv[1], "-C");
    assert_eq!(argv[2], linker_arg.as_str());
    eprintln!(
        "sealed build linker: {} -> {}",
        linker.display(),
        canonical.display()
    );
    let probe = &receipt.document["toolchain"][1];
    assert_eq!(probe["command"]["executable"], canonical.to_str().unwrap());
    // The linker named in the build argv is exactly the hashed probe executable.
    assert_eq!(
        argv[2],
        format!(
            "linker={}",
            probe["command"]["executable"].as_str().unwrap()
        )
        .as_str()
    );
    assert_eq!(probe["command"]["argv"][0], "--version");
    assert_eq!(
        probe["command"]["executable_sha256"],
        sha(&fs::read(&canonical).unwrap()).as_str()
    );
    assert_eq!(probe["version_probe"]["exit_code"], 0);
    assert_ne!(probe["version_probe"]["stdout_bytes"], 0);
    let run = child_deadline::command(5)
        .arg(fixture.0.join(&request.output))
        .output()
        .unwrap();
    assert!(run.status.success());
    assert_eq!(run.stdout, b"42\n");
    assert!(run.stderr.is_empty());
}

#[test]
fn issue_m5_01_build_lineage_linker_probe_change_after_build_fails_verify() {
    let _deadlines = shared();
    // A fake linker driver, probed and hashed exactly like the real one, is
    // rewritten after a successful build; generic receipt verification must
    // then refuse. This binds the driver only, not any helper it may invoke.
    let fixture = Fixture::new("cp input.txt artifacts/binary\n");
    let fake = fixture.0.join("artifacts/fake-cc");
    fs::write(&fake, "#!/bin/sh\nprintf 'fake cc 1.0\\n'\n").unwrap();
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();
    let canonical = fs::canonicalize(&fake).unwrap();
    let mut request = fixture.request();
    request.toolchain.push(Invocation {
        executable: canonical.clone(),
        args: vec!["--version".into()],
    });
    let receipt = build(&request).unwrap();
    let probe = &receipt.document["toolchain"][1];
    assert_eq!(probe["command"]["executable"], canonical.to_str().unwrap());
    assert_eq!(
        probe["command"]["executable_sha256"],
        sha(&fs::read(&canonical).unwrap()).as_str()
    );
    let pinned = || {
        verify(
            &fixture.0,
            &request.receipt,
            &receipt.sha256,
            &request.expected_revision,
        )
    };
    assert_eq!(pinned().unwrap(), receipt.document);
    // Same version output, different driver bytes.
    fs::write(&fake, "#!/bin/sh\nprintf 'fake cc 1.0\\n' # changed\n").unwrap();
    assert_eq!(pinned().err(), Some(Error::ToolChanged));
}

#[test]
fn issue_m5_01_build_lineage_linker_selection_fails_rather_than_skips() {
    let _deadlines = shared();
    let dir = std::env::temp_dir().join(format!(
        "invar-lineage-linker-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir).unwrap();
    let dangling = dir.join("cc");
    std::os::unix::fs::symlink(dir.join("missing-alternative"), &dangling).unwrap();
    let plain = dir.join("plain");
    fs::write(&plain, "not a linker").unwrap();
    fs::set_permissions(&plain, fs::Permissions::from_mode(0o644)).unwrap();
    let directory = dir.join("directory");
    fs::create_dir(&directory).unwrap();
    let tool = dir.join("gcc");
    fs::write(&tool, "#!/bin/sh\nexit 0\n").unwrap();
    fs::set_permissions(&tool, fs::Permissions::from_mode(0o755)).unwrap();
    // Unusable earlier candidates are skipped over in order; the executable wins.
    assert_eq!(
        select_linker(&[&dangling, &plain, &directory, &tool]).unwrap(),
        tool
    );
    assert_eq!(select_linker(&[&tool, &dangling]).unwrap(), tool);
    // No usable candidate is an explicit error naming every candidate.
    let reasons = select_linker(&[&dangling, &plain, &directory]).unwrap_err();
    assert!(reasons.contains("does not resolve"), "{reasons}");
    assert!(reasons.contains("is not executable"), "{reasons}");
    assert!(reasons.contains("is not a regular file"), "{reasons}");
    assert!(select_linker(&[]).is_err());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn issue_m5_01_build_lineage_fifo_receipt_fails_before_open() {
    let _deadlines = shared();
    const CHILD_FLAG: &str = "INVAR_LINEAGE_FIFO_REPRO";
    if let Some(repository) = std::env::var_os(CHILD_FLAG) {
        let pin = std::env::var("INVAR_LINEAGE_FIFO_PIN").unwrap();
        let revision = std::env::var("INVAR_LINEAGE_FIFO_REVISION").unwrap();
        assert_eq!(
            verify(
                &PathBuf::from(repository),
                std::path::Path::new("artifacts/receipt.json"),
                &pin,
                &revision
            )
            .err(),
            Some(Error::InvalidPath)
        );
        return;
    }
    let fixture = Fixture::new("cp input.txt artifacts/binary\n");
    let request = fixture.request();
    let receipt = build(&request).unwrap();
    let path = fixture.0.join(&request.receipt);
    fs::remove_file(&path).unwrap();
    let create = child_deadline::command(2)
        .args(["mkfifo"])
        .arg(&path)
        .status()
        .unwrap();
    assert!(create.success());
    // No writer is opened. The external limit contains the vulnerable open()
    // without leaking blocked threads into the test suite or waiting 30 seconds.
    let child = child_deadline::command(2)
        .arg(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "issue_m5_01_build_lineage_fifo_receipt_fails_before_open",
            "--nocapture",
        ])
        .env(CHILD_FLAG, &fixture.0)
        .env("INVAR_LINEAGE_FIFO_PIN", &receipt.sha256)
        .env("INVAR_LINEAGE_FIFO_REVISION", &request.expected_revision)
        .output()
        .unwrap();
    assert!(
        child.status.success(),
        "FIFO receipt verification failed or blocked: exit {:?}; {}",
        child.status.code(),
        String::from_utf8_lossy(&child.stdout)
    );
}
