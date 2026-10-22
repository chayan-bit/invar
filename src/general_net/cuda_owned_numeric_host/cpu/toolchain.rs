//! Compile-stage processes and toolchain identity for the subprocess CPU group
//! model (review F1/F2). Every compile-stage process (compiler probes, the
//! compile, loader `--list` queries) runs through a fixed `/bin/sh` script that
//! applies `ulimit -v` (address space) and `ulimit -f` (file size) before
//! `exec`; the limits are argv, never interpolated. Identity binds the shell,
//! the compiler driver, `cc1plus`, `collect2`, `as` and `ld` with every shared
//! library their ELF interpreter resolves, plus the model executable's own
//! runtime libraries (none for the static product profile). The kernel and the
//! vDSO are outside the identity.
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

/// HARD: per-process address-space cap (inherited rlimit) for every
/// compile-stage process.
pub(crate) const COMPILE_ADDRESS_SPACE_BYTES: u64 = 1024 * 1024 * 1024;
/// HARD: per-file size cap (inherited rlimit) for every file a compile-stage
/// process writes.
pub(crate) const COMPILE_FILE_BYTES: u64 = 64 * 1024 * 1024;
/// MODELLED, NOT ENFORCED: the gcc driver runs its pipeline sequentially (no
/// `-pipe`), so at most the driver, `collect2` and `ld` are alive together;
/// the aggregate address space is therefore modelled as 3 x the hard cap.
pub(crate) const COMPILE_MAX_PROCESSES: u64 = 3;
/// POST-COMPILE CHECK, NOT A PEAK CAP: the scratch directory's total bytes
/// after the compile stage; the peak is bounded only by the hard per-file cap
/// times the files alive together (modelled).
pub(crate) const COMPILE_DISK_BYTES: u64 = 256 * 1024 * 1024;
pub(crate) const MAX_PROGRAM_BYTES: u64 = 512 * 1024 * 1024;
pub(crate) const MAX_LIBRARIES: usize = 64;
pub(crate) const MAX_PATH_BYTES: usize = 4096;
pub(crate) const MAX_LOG_BYTES: u64 = 64 * 1024;
const GUARD_SCRIPT: &str = "ulimit -v \"$1\" && ulimit -f \"$2\" && shift 2 && exec \"$@\"";
const SHELL: &str = "/bin/sh";

/// One hashed toolchain program and the shared libraries its interpreter maps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Program {
    pub role: &'static str,
    pub path: PathBuf,
    pub sha256: [u8; 32],
    pub libraries: Vec<(PathBuf, [u8; 32])>,
}

pub(crate) fn file_sha256(path: &Path, cap: u64) -> Result<[u8; 32], String> {
    let meta = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.len() == 0 || meta.len() > cap {
        return Err(format!("{}: file size or type", path.display()));
    }
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut chunk = [0u8; 64 * 1024];
    let mut seen = 0u64;
    loop {
        let n = file.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        seen += n as u64;
        if seen > cap {
            return Err("file grew".into());
        }
        hash.update(&chunk[..n]);
    }
    if seen != meta.len() {
        return Err("file changed".into());
    }
    Ok(hash.finalize().into())
}

/// Resolve a bare program name through `path_env`, or canonicalize a path.
pub(crate) fn resolve(program: &Path, path_env: &std::ffi::OsStr) -> Result<PathBuf, String> {
    if program.components().count() > 1 || program.is_absolute() {
        return fs::canonicalize(program).map_err(|e| e.to_string());
    }
    for dir in std::env::split_paths(path_env) {
        let candidate = dir.join(program);
        if candidate.is_file() {
            return fs::canonicalize(candidate).map_err(|e| e.to_string());
        }
    }
    Err(format!("{} not found", program.display()))
}

/// The ELF interpreter (`PT_INTERP`) of a 64-bit little-endian ELF file, or
/// `None` for a static executable. Anything else fails closed.
pub(crate) fn elf_interp(path: &Path) -> Result<Option<PathBuf>, String> {
    let bytes = {
        let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
        let mut head = Vec::new();
        (&mut file)
            .take(64 * 1024)
            .read_to_end(&mut head)
            .map_err(|e| e.to_string())?;
        head
    };
    let u16_at = |at: usize| -> Option<u64> {
        Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?) as u64)
    };
    let u32_at = |at: usize| -> Option<u64> {
        Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?) as u64)
    };
    let u64_at = |at: usize| -> Option<u64> {
        Some(u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?))
    };
    let malformed = || format!("{}: not a 64-bit little-endian ELF", path.display());
    if bytes.get(..6) != Some(&[0x7f, b'E', b'L', b'F', 2, 1][..]) {
        return Err(malformed());
    }
    let phoff = u64_at(0x20).ok_or_else(malformed)? as usize;
    let phentsize = u16_at(0x36).ok_or_else(malformed)? as usize;
    let phnum = u16_at(0x38).ok_or_else(malformed)? as usize;
    if phentsize < 0x38 || phnum == 0 || phnum > 64 {
        return Err(malformed());
    }
    for i in 0..phnum {
        let at = phoff
            .checked_add(i.checked_mul(phentsize).ok_or_else(malformed)?)
            .ok_or_else(malformed)?;
        if u32_at(at).ok_or_else(malformed)? == 3 {
            let offset = u64_at(at + 0x08).ok_or_else(malformed)? as usize;
            let size = u64_at(at + 0x20).ok_or_else(malformed)? as usize;
            if size == 0 || size > MAX_PATH_BYTES {
                return Err(malformed());
            }
            let raw = bytes
                .get(offset..offset.checked_add(size).ok_or_else(malformed)?)
                .ok_or_else(malformed)?;
            let text = raw.strip_suffix(&[0]).unwrap_or(raw);
            let text = std::str::from_utf8(text).map_err(|_| malformed())?;
            return Ok(Some(PathBuf::from(text)));
        }
    }
    Ok(None)
}

/// Limits for one guarded compile-stage process.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Guard {
    pub address_space_bytes: u64,
    pub file_bytes: u64,
    pub deadline: Duration,
}

/// Run `program args` under the guard script in its own process group, with a
/// cleared environment plus `env`; stdout/stderr go to files in `dir` and are
/// read back capped. A timed-out group is killed and reaped.
pub(crate) fn guarded(
    program: &Path,
    args: &[&std::ffi::OsStr],
    env: &[(&str, &std::ffi::OsStr)],
    dir: &Path,
    name: &str,
    guard: Guard,
) -> Result<(bool, Vec<u8>, Vec<u8>), String> {
    let out = dir.join(format!("{name}.stdout"));
    let err = dir.join(format!("{name}.stderr"));
    let mut command = Command::new(SHELL);
    command
        .env_clear()
        .arg("-c")
        .arg(GUARD_SCRIPT)
        .arg("sh")
        .arg((guard.address_space_bytes / 1024).to_string())
        .arg(guard.file_bytes.div_ceil(512).to_string())
        .arg(program)
        .args(args);
    for (k, v) in env {
        command.env(k, v);
    }
    let mut child = command
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(fs::File::create(&out).map_err(|e| e.to_string())?)
        .stderr(fs::File::create(&err).map_err(|e| e.to_string())?)
        .process_group(0)
        .spawn()
        .map_err(|e| e.to_string())?;
    let begun = Instant::now();
    let status = loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => break status,
            None if begun.elapsed() >= guard.deadline => {
                invar_foreign::numeric::terminate_compiler_group(child.id());
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{name} deadline"));
            }
            None => std::thread::sleep(Duration::from_millis(5)),
        }
    };
    let read = |p: &Path| -> Result<Vec<u8>, String> {
        let mut bytes = Vec::new();
        fs::File::open(p)
            .map_err(|e| e.to_string())?
            .take(MAX_LOG_BYTES)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    };
    Ok((status.success(), read(&out)?, read(&err)?))
}

/// Shared libraries the ELF interpreter maps for `program` (empty if static),
/// each canonicalized and hashed, including the interpreter itself.
pub(crate) fn libraries(
    program: &Path,
    dir: &Path,
    name: &str,
    guard: Guard,
) -> Result<Vec<(PathBuf, [u8; 32])>, String> {
    let Some(interp) = elf_interp(program)? else {
        return Ok(Vec::new());
    };
    let interp = fs::canonicalize(&interp).map_err(|e| e.to_string())?;
    let (ok, out, err) = guarded(
        &interp,
        &["--list".as_ref(), program.as_os_str()],
        &[],
        dir,
        name,
        guard,
    )?;
    if !ok {
        return Err(format!(
            "{}: loader listing failed: {}",
            program.display(),
            String::from_utf8_lossy(&err)
        ));
    }
    let text = String::from_utf8(out).map_err(|_| "loader listing is not UTF-8")?;
    let mut found = vec![interp];
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if line.contains("not found") {
            return Err(format!("{}: {line}", program.display()));
        }
        // `name => /path (0x..)`, `/path (0x..)`, or the vDSO `name (0x..)`.
        let target = line.split_once("=>").map_or(line, |(_, rest)| rest.trim());
        let path = target.split(" (").next().unwrap_or("").trim();
        if path.starts_with('/') {
            found.push(fs::canonicalize(path).map_err(|e| e.to_string())?);
        } else if !line.starts_with("linux-vdso") {
            return Err(format!(
                "{}: unparsed loader line {line}",
                program.display()
            ));
        }
    }
    found.sort();
    found.dedup();
    if found.len() > MAX_LIBRARIES || found.iter().any(|p| p.as_os_str().len() > MAX_PATH_BYTES) {
        return Err("too many or too long library paths".into());
    }
    found
        .into_iter()
        .map(|p| file_sha256(&p, MAX_PROGRAM_BYTES).map(|h| (p, h)))
        .collect()
}

/// Hash the shell, the driver and the programs the driver names, with their
/// libraries.
pub(crate) fn toolchain(
    driver: &Path,
    path_env: &std::ffi::OsStr,
    dir: &Path,
    guard: Guard,
) -> Result<Vec<Program>, String> {
    let mut programs = Vec::new();
    let mut add = |role: &'static str, path: PathBuf| -> Result<(), String> {
        if path.as_os_str().len() > MAX_PATH_BYTES {
            return Err("program path too long".into());
        }
        let sha256 = file_sha256(&path, MAX_PROGRAM_BYTES)?;
        let libraries = libraries(&path, dir, &format!("libraries-{role}"), guard)?;
        programs.push(Program {
            role,
            path,
            sha256,
            libraries,
        });
        Ok(())
    };
    add("shell", fs::canonicalize(SHELL).map_err(|e| e.to_string())?)?;
    add("driver", driver.to_path_buf())?;
    for role in ["cc1plus", "collect2", "as", "ld"] {
        let arg = format!("-print-prog-name={role}");
        let (ok, out, _) = guarded(
            driver,
            &[arg.as_ref()],
            &[("PATH", path_env)],
            dir,
            &format!("prog-name-{role}"),
            guard,
        )?;
        let named = String::from_utf8_lossy(&out).trim().to_owned();
        if !ok || named.is_empty() {
            return Err(format!("cannot resolve {role}"));
        }
        add(role, resolve(Path::new(&named), path_env)?)?;
    }
    Ok(programs)
}

/// Total bytes of the regular files directly in `dir`.
pub(crate) fn dir_bytes(dir: &Path) -> Result<u64, String> {
    let mut total = 0u64;
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let meta = entry
            .map_err(|e| e.to_string())?
            .metadata()
            .map_err(|e| e.to_string())?;
        if meta.is_file() {
            total = total.checked_add(meta.len()).ok_or("disk total overflow")?;
        }
    }
    Ok(total)
}

/// Re-hash every recorded program and library after the compile and refuse
/// any change (review R3). This narrows, but does not close, the window: a
/// file changed and restored between the two checks is not detected.
pub(crate) fn verify(programs: &[Program]) -> Result<(), String> {
    for p in programs {
        if file_sha256(&p.path, MAX_PROGRAM_BYTES)? != p.sha256 {
            return Err(format!(
                "toolchain changed during compile: {}",
                p.path.display()
            ));
        }
        for (lib, h) in &p.libraries {
            if file_sha256(lib, MAX_PROGRAM_BYTES)? != *h {
                return Err(format!(
                    "toolchain changed during compile: {}",
                    lib.display()
                ));
            }
        }
    }
    Ok(())
}
