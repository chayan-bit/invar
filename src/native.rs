//! Native LLVM 21 lowering for verified MIR. All source failures cross the
//! versioned Rust host ABI and terminate the generated process without unwinding.

use std::{
    collections::HashMap,
    fmt::Write as _,
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use inkwell::{
    OptimizationLevel,
    context::Context,
    memory_buffer::MemoryBuffer,
    targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine},
};

use crate::{
    front::{BinaryOp, Span, Type, UnaryOp},
    mir::{self, Function, Operation, Terminator, ValueId},
};

pub const ABI_VERSION: u32 = 1;
const MAX_IR_BYTES: usize = 8 * 1024 * 1024;
const PROCESS_DEADLINE: Duration = Duration::from_secs(30);
const MAX_OUTPUT_BYTES: usize = 1_000_000;
static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
type EmbeddedBlobs = Vec<(Vec<u8>, bool)>;
type NativeIr = (String, Vec<String>, EmbeddedBlobs, Vec<Option<ForeignSpec>>);

#[derive(Clone, Debug)]
pub struct ForeignSpec {
    pub symbol: String,
    pub result_tag: u32,
    pub args: Vec<(u32, u32, usize)>,
}

fn foreign_tag(ty: &Type) -> Result<u32, String> {
    match ty {
        Type::I32 => Ok(invar_foreign::I32),
        Type::U32 => Ok(invar_foreign::U32),
        Type::F64 => Ok(invar_foreign::F64),
        Type::Named(name) if name == "Bytes" => Ok(invar_foreign::BYTES),
        Type::Named(name) if name == "String" => Ok(invar_foreign::STRING),
        _ => Err(error("verified foreign type is unsupported")),
    }
}

#[derive(Debug)]
pub struct Object {
    pub triple: String,
    pub data_layout: String,
    /// The emitted IR as LLVM parsed it, before any pass (what the audit checked).
    pub ir: String,
    /// The IR after [`optimize::PIPELINE`], when it ran.
    pub optimized_ir: Option<String>,
    pub bytes: Vec<u8>,
    pub names: Vec<String>,
    pub blobs: EmbeddedBlobs,
    pub foreign: Vec<Option<ForeignSpec>>,
    /// Set by [`emit_mir_debug`]: DWARF to keep and the host's value layouts (DX-20).
    pub debug: Option<String>,
    /// CPU, features and pipeline the code was generated for (#101).
    pub machine: Machine,
    /// The facts behind every promise in `ir`, indexed by `!invar.fact` node.
    pub facts: Vec<Fact>,
}

fn error(message: impl Into<String>) -> String {
    message.into()
}

fn span(span: Span) -> Result<(u32, u32), String> {
    Ok((
        u32::try_from(span.start).map_err(|_| error("native source span exceeds u32"))?,
        u32::try_from(span.end).map_err(|_| error("native source span exceeds u32"))?,
    ))
}

struct Ir<'a> {
    text: String,
    temp: usize,
    block: usize,
    locals: HashMap<&'a str, usize>,
    names: &'a HashMap<(usize, String), usize>,
    functions: &'a HashMap<&'a str, usize>,
    value_types: HashMap<ValueId, Type>,
    function_id: usize,
    program: &'a mir::Program,
    next_blob: usize,
    exporting: bool,
    /// Checked frame helpers trap at their caller's site, carried as hidden
    /// `%site_start`/`%site_end` parameters (the outermost helper call wins).
    helper: bool,
    /// Debug builds: `, !dbg !N` appended to every instruction line.
    location: String,
    debug_bytes: usize,
}

include!("native/ir.rs");
include!("native/floating.rs");
mod debug;
pub(crate) mod export;
mod optimize;
pub mod region;
pub use debug::{Source as DebugSource, canonical_ir, run_program_debug};
pub use optimize::{
    Build, Cpu, Fact, IntervalOp, Machine, Operand, PIPELINE, Promise, audit, compile_ir,
    fact_nodes, resolve,
};

include!("native/emit.rs");

/// Lower a verified MIR program to a portable, optimized host object
/// ([`Build::PORTABLE`]). LLVM verifies the resulting IR.
pub fn emit_mir(program: &mir::Program) -> Result<Object, String> {
    emit_mir_for(program, &Build::PORTABLE)
}

/// Lower like [`emit_mir`] for the CPU and pipeline `build` names.
pub fn emit_mir_for(program: &mir::Program, build: &Build) -> Result<Object, String> {
    optimize::object(llvm_ir(program)?, Vec::new(), build)
}

/// Lower unoptimized ([`Build::UNOPTIMIZED`]), plus DWARF 5 mapping the code
/// back to `source` (DX-20). The code is the same as `emit_mir_for(program,
/// &Build::UNOPTIMIZED)`'s; only debug metadata is added.
pub fn emit_mir_debug(program: &mir::Program, source: &DebugSource) -> Result<Object, String> {
    let mut object = object_from_ir(llvm_ir_debug(program, None, Some(source))?)?;
    object.debug = Some(debug::host_types(program));
    Ok(object)
}

/// The unoptimized generic path: debug and export objects (see [`Build::UNOPTIMIZED`]).
fn object_from_ir(data: NativeIr) -> Result<Object, String> {
    optimize::object(data, Vec::new(), &Build::UNOPTIMIZED)
}

pub fn unique_directory() -> Result<PathBuf, String> {
    unique_directory_in(&std::env::temp_dir())
}

fn unique_directory_in(parent: &Path) -> Result<PathBuf, String> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let directory = parent.join(format!(
        "invar-native-{}-{nonce}-{}",
        std::process::id(),
        NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
    ));
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(&directory)
        .map_err(|error| error.to_string())?;
    Ok(directory)
}

fn read_capped(pipe: impl Read) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    pipe.take(MAX_OUTPUT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    Ok(bytes)
}

type Reader = Option<thread::JoinHandle<std::io::Result<Vec<u8>>>>;

/// Join a finished reader into `slot`; never blocks on a running reader.
fn collect(reader: &mut Reader, slot: &mut Option<Vec<u8>>) -> Result<(), String> {
    if reader.as_ref().is_some_and(|handle| handle.is_finished())
        && let Some(handle) = reader.take()
    {
        let bytes = handle
            .join()
            .map_err(|_| error("native output reader failed"))?
            .map_err(|error| error.to_string())?;
        *slot = Some(bytes);
    }
    Ok(())
}

/// Run `command`, draining stdout and stderr concurrently so a chatty child
/// cannot block on a full pipe. Each stream is capped at `MAX_OUTPUT_BYTES`
/// and the whole run at `PROCESS_DEADLINE`; either limit kills and reaps it.
fn wait_bounded(mut command: Command) -> Result<Output, String> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let started = Instant::now();
    // A concurrent process launch can briefly retain a newly linked executable's
    // writable descriptor. This error means exec never started the program, so
    // retrying it cannot repeat source execution or consume source fuel.
    let mut child = loop {
        match command.spawn() {
            Ok(child) => break child,
            Err(error)
                if error.kind() == std::io::ErrorKind::ExecutableFileBusy
                    && started.elapsed() < Duration::from_millis(250) =>
            {
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => return Err(error.to_string()),
        }
    };
    let mut readers: [Reader; 2] = [
        child
            .stdout
            .take()
            .map(|pipe| thread::spawn(move || read_capped(pipe))),
        child
            .stderr
            .take()
            .map(|pipe| thread::spawn(move || read_capped(pipe))),
    ];
    let mut outputs: [Option<Vec<u8>>; 2] = [None, None];
    let mut status = None;
    loop {
        for (reader, slot) in readers.iter_mut().zip(outputs.iter_mut()) {
            if let Err(message) = collect(reader, slot) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(message);
            }
        }
        let failure = if outputs
            .iter()
            .flatten()
            .any(|bytes| bytes.len() > MAX_OUTPUT_BYTES)
        {
            Some("native output limit exceeded")
        } else if started.elapsed() >= PROCESS_DEADLINE {
            Some("native subprocess deadline exceeded")
        } else {
            None
        };
        if let Some(message) = failure {
            // Reader threads are detached; they end when the pipes close.
            let _ = child.kill();
            let _ = child.wait();
            return Err(error(message));
        }
        if status.is_none() {
            status = child.try_wait().map_err(|error| error.to_string())?;
        }
        if let (Some(status), [None, None]) = (status, &readers) {
            let [stdout, stderr] = outputs;
            return Ok(Output {
                status,
                stdout: stdout.unwrap_or_default(),
                stderr: stderr.unwrap_or_default(),
            });
        }
        thread::sleep(Duration::from_millis(10));
    }
}

/// Link the versioned host ABI and object into a standalone native executable.
pub fn link(object: &Object, output: &Path) -> Result<(), String> {
    link_with_library(object, output, None)
}

fn dependency_rlib(name: &str) -> Result<(PathBuf, PathBuf), String> {
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let parent = executable
        .parent()
        .ok_or_else(|| error("native executable directory missing"))?;
    let deps = if parent.file_name().is_some_and(|name| name == "deps") {
        parent.to_path_buf()
    } else {
        parent.join("deps")
    };
    let prefix = format!("lib{name}-");
    let mut matches = fs::read_dir(&deps)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&prefix) && name.ends_with(".rlib"))
        })
        .collect::<Vec<_>>();
    matches.sort();
    let [path] = matches.as_slice() else {
        return Err(error(format!(
            "native host requires exactly one {name} rlib beside this executable"
        )));
    };
    Ok((deps, path.clone()))
}

pub fn link_with_library(
    object: &Object,
    output: &Path,
    library: Option<&Path>,
) -> Result<(), String> {
    if output.exists() {
        return Err(error("native output already exists"));
    }
    let has_foreign = object.foreign.iter().any(Option::is_some);
    let library = match (has_foreign, library) {
        (true, Some(path)) if path.is_absolute() && path.is_file() => path
            .to_str()
            .ok_or_else(|| error("C library path must be UTF-8"))?,
        (true, _) => {
            return Err(error(
                "foreign native build requires an absolute --c-library file path",
            ));
        }
        (false, Some(_)) => return Err(error("--c-library requires a foreign declaration")),
        (false, None) => "",
    };
    // The linker child writes the executable beside `output`, so `publish` can
    // hard-link it there: same filesystem by construction.
    let output = std::path::absolute(output).map_err(|error| error.to_string())?;
    let parent = output
        .parent()
        .ok_or_else(|| error("native output needs a parent directory"))?;
    let directory = unique_directory_in(parent)?;
    let result = (|| {
        let object_path = directory.join("program.o");
        let host_path = directory.join("host.rs");
        let executable = directory.join("program");
        fs::write(&object_path, &object.bytes).map_err(|error| error.to_string())?;
        // invar-math-cr-v1: the vendored CORE-MATH archive resolves `cr_*` calls.
        let core_math_path = directory.join("libinvar_coremath.a");
        fs::write(&core_math_path, invar_math::ARCHIVE).map_err(|error| error.to_string())?;
        let names = object
            .names
            .iter()
            .map(|name| format!("{name:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        let blobs = object
            .blobs
            .iter()
            .enumerate()
            .map(|(id, (bytes, text))| {
                fs::write(directory.join(format!("blob-{id}.bin")), bytes)
                    .map_err(|error| error.to_string())?;
                Ok(format!("(include_bytes!(\"blob-{id}.bin\"), {text})"))
            })
            .collect::<Result<Vec<_>, String>>()?
            .join(", ");
        let foreign = object
            .foreign
            .iter()
            .map(|spec| match spec {
                Some(spec) => format!(
                    "Some(({:?}, {}, &[{}]))",
                    spec.symbol,
                    spec.result_tag,
                    spec.args
                        .iter()
                        .map(|(tag, mode, name)| format!("({tag}, {mode}, {name})"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                None => "None".to_owned(),
            })
            .collect::<Vec<_>>()
            .join(", ");
        let foreign_source = crate::native_foreign_source::standalone();
        let sort_source = format!("mod invar_sort_cpu {{ {} }}\n", include_str!("sort_cpu.rs"));
        let host = format!(
            "{}\n#[allow(dead_code)] mod invar_tc {{ {} }}\n{sort_source}\nconst MAX_ARRAY_LEN: usize = {max_array};\nconst MAX_LIVE_ELEMENTS: usize = {max_live};\nconst NAMES: &[&str] = &[{names}];\nconst BLOBS: &[(&[u8], bool)] = &[{blobs}];\nconst C_LIBRARY: &str = {library:?};\nconst FOREIGN: &[Option<(&str, u32, &[(u32,u32,usize)])>] = &[{foreign}];\n#[cfg(invar_has_foreign)] mod invar_foreign {{ {} }}\n#[allow(dead_code)] mod fsum {{ {} }}\n{}\n{}",
            include_str!("angle.rs"),
            include_str!("../crates/invar-device/src/tc/reference.rs"),
            foreign_source,
            include_str!("fsum.rs"),
            include_str!("native_host.rs"),
            debug::host_source(object.debug.as_deref()),
            max_array = crate::array_limits::MAX_ARRAY_LEN,
            max_live = crate::array_limits::MAX_LIVE_ELEMENTS,
        );
        let has_spatial = object.ir.contains("call i64 @invar_spatial_v1(");
        let host = if has_spatial {
            // The standalone host has no module files: inline numeric_contract's
            // file submodules (f64 math, ENG-03) where the file declares them.
            let numeric = [
                (
                    "mod binary64;",
                    include_str!("numeric_contract/binary64.rs"),
                ),
                ("pub mod fold;", include_str!("numeric_contract/fold.rs")),
                ("mod ieee;", include_str!("numeric_contract/ieee.rs")),
                (
                    "pub mod storage;",
                    include_str!("numeric_contract/storage.rs"),
                ),
            ]
            .into_iter()
            .fold(
                include_str!("numeric_contract.rs").to_owned(),
                |text, (declaration, body)| {
                    let module = declaration.trim_end_matches(';');
                    text.replacen(declaration, &format!("{module} {{ {body} }}"), 1)
                },
            );
            format!(
                "{host}\nmod numeric_contract {{ {} }}\nmod fsum_reference {{ {} }}\nuse invar_sort_cpu as sort_cpu;\nmod spatial_cell {{ {} }}\nmod invar_spatial {{ {} }}\n{}",
                numeric,
                // numeric_contract::fold reaches crate::fsum_reference.
                include_str!("fsum_reference.rs"),
                include_str!("spatial_cell.rs"),
                include_str!("native_spatial.rs"),
                include_str!("native_spatial_host.rs")
            )
        } else {
            host
        };
        fs::write(&host_path, host).map_err(|error| error.to_string())?;
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let mut command = Command::new(rustc);
        command
            .arg("--edition=2024")
            .arg("-C")
            .arg("opt-level=2")
            .arg("-C")
            .arg("panic=abort")
            .arg(&host_path)
            .arg("-C")
            .arg(format!("link-arg={}", object_path.display()))
            .arg("-C")
            .arg(format!("link-arg={}", core_math_path.display()))
            // CORE-MATH reads the rounding mode and raises flags through fenv (libm).
            .arg("-C")
            .arg("link-arg=-lm")
            .args(debug::host_flags(object.debug.is_some()))
            .arg("-o")
            .arg(&executable);
        if has_foreign {
            let (deps, rlib) = dependency_rlib("libloading")?;
            command
                .arg("--cfg")
                .arg("invar_has_foreign")
                .arg("-L")
                .arg(format!("dependency={}", deps.display()))
                .arg("--extern")
                .arg(format!("libloading={}", rlib.display()));
        }
        if has_spatial {
            command.arg("--cfg").arg("invar_has_spatial");
            for name in ["num_bigint", "num_rational", "serde_json"] {
                let (deps, rlib) = dependency_rlib(name)?;
                command
                    .arg("-L")
                    .arg(format!("dependency={}", deps.display()))
                    .arg("--extern")
                    .arg(format!("{name}={}", rlib.display()));
            }
        }
        let linked = wait_bounded(command)?;
        if !linked.status.success() {
            return Err(format!(
                "native host link failed: {}",
                String::from_utf8_lossy(&linked.stderr)
            ));
        }
        // Symbols go first, so a failed publication never leaves an
        // executable without them; they are retracted if `output` fails.
        if object.debug.is_some() {
            debug::bundle(&executable, &output)?;
            debug::publish_symbols(&executable, &output)?;
        }
        let published = publish(&executable, &output).map_err(|failure| match failure.kind() {
            std::io::ErrorKind::AlreadyExists => error("native output already exists"),
            _ => format!("native output publication failed: {failure}"),
        });
        if published.is_err() && object.debug.is_some() {
            debug::retract_symbols(&output);
        }
        published
    })();
    let _ = fs::remove_dir_all(directory);
    result
}

/// Publish a linker-written executable at `output` without this process ever
/// holding a writable descriptor to it. Such a descriptor leaks into any child
/// another host thread forks meanwhile, and exec of `output` then fails with
/// ETXTBSY until that child execs or exits. The hard link is atomic and fails
/// with `AlreadyExists` instead of replacing an existing `output`.
// ponytail: no copy fallback for filesystems without hard links (FAT, some
// network mounts); add a child-process writer if a user needs one.
fn publish(linked: &Path, output: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // chmod by path: no descriptor is opened.
        fs::set_permissions(linked, fs::Permissions::from_mode(0o755))?;
    }
    fs::hard_link(linked, output)
}

/// Execute a native binary within the bounded local subprocess deadline.
pub fn run(executable: &Path, fuel: u64) -> Result<Output, String> {
    if fuel == 0 {
        return Err(error("native fuel must be positive"));
    }
    let mut command = Command::new(executable);
    command.arg(fuel.to_string());
    wait_bounded(command)
}

/// Compile and execute without leaving temporary artifacts behind.
pub fn run_program(program: &mir::Program, fuel: u64) -> Result<(i32, serde_json::Value), String> {
    run_program_with_library(program, fuel, None)
}

pub fn run_program_with_library(
    program: &mir::Program,
    fuel: u64,
    library: Option<&Path>,
) -> Result<(i32, serde_json::Value), String> {
    // The object runs here and now, so it is tuned for the live CPU.
    run_object(&emit_mir_for(program, &Build::HOST)?, fuel, library)
}

fn run_object(
    object: &Object,
    fuel: u64,
    library: Option<&Path>,
) -> Result<(i32, serde_json::Value), String> {
    let directory = unique_directory()?;
    let result = (|| {
        let executable = directory.join("program");
        link_with_library(object, &executable, library)?;
        let output = run(&executable, fuel)?;
        let document: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|_| error("native host returned invalid JSON"))?;
        let status = output
            .status
            .code()
            .ok_or_else(|| error("native host terminated by signal"))?;
        let expected = match document.get("status").and_then(serde_json::Value::as_str) {
            Some("ok") => 0,
            Some("tool_error") => 2,
            Some("trap") => 3,
            Some("evaluation_limit") => 4,
            _ => return Err(error("native host returned unknown status")),
        };
        if status != expected {
            return Err(error("native host status and document disagree"));
        }
        Ok((status, document))
    })();
    let _ = fs::remove_dir_all(directory);
    result
}

#[cfg(test)]
mod tests {
    use super::emit_mir;
    use crate::{check, front, mir};

    #[test]
    fn native_temporary_directories_are_unique_across_parallel_calls() {
        let paths = std::thread::scope(|scope| {
            (0..32)
                .map(|_| scope.spawn(super::unique_directory))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|thread| thread.join().unwrap().unwrap())
                .collect::<Vec<_>>()
        });
        let unique = paths.iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), paths.len());
        for path in paths {
            std::fs::remove_dir(path).unwrap();
        }
    }

    #[test]
    fn native_cpu_trap_abi_smoke() {
        let source = b"fn main() -> i32 { return 42; }";
        let parsed = front::parse(source).unwrap();
        let checked = check::check(parsed, true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let object = emit_mir(&program).unwrap();
        assert!(object.ir.contains("define i64 @invar_entry_v1"));
        assert!(!object.bytes.is_empty());
        let directory = super::unique_directory().unwrap();
        let executable = directory.join("program");
        super::link(&object, &executable).unwrap();
        let output = super::run(&executable, 100).unwrap();
        assert_eq!(output.status.code(), Some(0));
        assert!(String::from_utf8_lossy(&output.stdout).contains("\"value\":42"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn shell(script: &str) -> std::process::Command {
        let mut command = std::process::Command::new("sh");
        command.arg("-c").arg(script);
        command
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn wait_bounded_handles_transient_executable_writer() {
        use std::os::unix::fs::PermissionsExt;

        let directory = super::unique_directory().unwrap();
        let executable = directory.join("held-writable");
        std::fs::write(&executable, b"#!/bin/sh\nprintf launched-once").unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        let writer = std::fs::OpenOptions::new()
            .write(true)
            .open(&executable)
            .unwrap();
        let error = std::process::Command::new(&executable)
            .output()
            .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::ExecutableFileBusy);
        std::thread::scope(|scope| {
            scope.spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                drop(writer);
            });
            let output = super::wait_bounded(std::process::Command::new(&executable)).unwrap();
            assert!(output.status.success());
            assert_eq!(output.stdout, b"launched-once");
        });
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn wait_bounded_reports_persistent_executable_writer() {
        use std::os::unix::fs::PermissionsExt;

        let directory = super::unique_directory().unwrap();
        let executable = directory.join("held-writable");
        std::fs::write(&executable, b"#!/bin/sh\nprintf must-not-run").unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        let writer = std::fs::OpenOptions::new()
            .write(true)
            .open(&executable)
            .unwrap();
        let started = std::time::Instant::now();
        let error = super::wait_bounded(std::process::Command::new(&executable)).unwrap_err();
        assert!(error.contains("Text file busy"), "{error}");
        assert!(started.elapsed() < crate::child_deadline::wall(std::time::Duration::from_secs(2)));
        drop(writer);
        std::fs::remove_dir_all(directory).unwrap();
    }

    /// Callers of `link` exec the result directly, without `wait_bounded`'s
    /// retry. Sixteen threads publish and exec while their own spawns race the
    /// other threads' publications; a writable descriptor held by this process
    /// during publication leaks into those children and fails exec with ETXTBSY.
    #[cfg(target_os = "linux")]
    #[test]
    fn published_executables_survive_concurrent_spawns() {
        let directory = super::unique_directory().unwrap();
        // The 1 MiB comment widens any descriptor window without changing behavior.
        let writer = "{ printf '#!/bin/sh\\nprintf published-once\\n#'; \
                      head -c 1048576 /dev/zero | tr '\\0' x; echo; } > \"$1\"";
        std::thread::scope(|scope| {
            let workers = (0..16)
                .map(|worker| {
                    let directory = &directory;
                    scope.spawn(move || {
                        for iteration in 0..8 {
                            let linked = directory.join(format!("linked-{worker}-{iteration}"));
                            let output = directory.join(format!("output-{worker}-{iteration}"));
                            // Production's linker is a child too; this process never writes it.
                            let status = std::process::Command::new("sh")
                                .args(["-c", writer, "writer"])
                                .arg(&linked)
                                .status()
                                .unwrap();
                            assert!(status.success());
                            super::publish(&linked, &output).unwrap();
                            let ran = std::process::Command::new(&output).output().unwrap();
                            assert!(ran.status.success(), "{ran:?}");
                            assert_eq!(ran.stdout, b"published-once");
                            assert!(ran.stderr.is_empty());
                            std::fs::remove_file(linked).unwrap();
                            std::fs::remove_file(output).unwrap();
                        }
                    })
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker.join().unwrap();
            }
        });
        std::fs::remove_dir(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn publish_refuses_existing_output_and_keeps_it() {
        let directory = super::unique_directory().unwrap();
        let linked = directory.join("linked");
        let output = directory.join("output");
        std::fs::write(&linked, b"new").unwrap();
        std::fs::write(&output, b"existing").unwrap();
        let error = super::publish(&linked, &output).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&output).unwrap(), b"existing");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn wait_bounded_drains_output_larger_than_a_pipe_buffer() {
        let started = std::time::Instant::now();
        let output = super::wait_bounded(shell(
            "head -c 300000 /dev/zero; head -c 200000 /dev/zero >&2",
        ))
        .unwrap();
        assert!(output.status.success());
        assert_eq!(output.stdout.len(), 300_000);
        assert_eq!(output.stderr.len(), 200_000);
        assert!(
            started.elapsed() < crate::child_deadline::wall(std::time::Duration::from_secs(10))
        );
    }

    #[test]
    fn wait_bounded_reports_output_limit_promptly() {
        for script in [
            "head -c 2000000 /dev/zero",
            "head -c 2000000 /dev/zero >&2",
            "while :; do head -c 65536 /dev/zero; done",
        ] {
            let started = std::time::Instant::now();
            let error = super::wait_bounded(shell(script)).unwrap_err();
            assert_eq!(error, "native output limit exceeded", "{script}");
            assert!(
                started.elapsed() < crate::child_deadline::wall(std::time::Duration::from_secs(10))
            );
        }
    }
}
