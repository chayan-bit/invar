//! `invar build --target native`, and `--debug` for native builds and runs
//! (#305): DWARF 5 in the executable, native-serial only.

use std::{
    ffi::OsString,
    sync::atomic::{AtomicBool, Ordering},
};

use invar::eval::EvalError;

use super::report_eval;

static DEBUG: AtomicBool = AtomicBool::new(false);

/// Whether this invocation asked for `--debug`.
pub(crate) fn debug() -> bool {
    DEBUG.load(Ordering::Relaxed)
}

/// `invar build PATH ... --debug` and `invar run PATH ... --debug`: drop the
/// flag, record it, and run the command again. Debug builds target native
/// code only; a run without `--engine` becomes native-serial and says so.
pub(crate) fn debug_command(args: &[OsString]) -> Option<i32> {
    if args.len() >= 3 && args[1] == "debug" {
        return Some(session(&args[2], &args[3..]));
    }
    if args.len() < 3 || !(args[1] == "build" || args[1] == "run") {
        return None;
    }
    let at = args[3..].iter().position(|arg| arg == "--debug")? + 3;
    let mut rest = args.to_vec();
    rest.remove(at);
    let options = |name: &str| {
        rest[3..]
            .chunks(2)
            .find(|pair| pair[0] == name)
            .and_then(|pair| pair.get(1).cloned())
    };
    let json = options("--format").is_some_and(|format| format == "json");
    if rest[1] == "build" {
        if options("--target").is_none_or(|target| target != "native") {
            return Some(refuse(
                &args[2],
                json,
                "--debug builds only --target native",
            ));
        }
    } else {
        match options("--engine") {
            None => {
                eprintln!("note: --debug implies --engine native-serial");
                rest.extend(["--engine".into(), "native-serial".into()]);
            }
            Some(engine) if engine == "native-serial" => {}
            Some(_) => {
                return Some(refuse(
                    &args[2],
                    json,
                    "--debug supports only --engine native-serial; cpu-parallel and device engines are not debuggable",
                ));
            }
        }
    }
    DEBUG.store(true, Ordering::Relaxed);
    Some(super::run(rest))
}

/// `invar debug PATH [--gdb|--lldb] [--fuel N] [--c-library ABSOLUTE_PATH] [-- DEBUGGER_ARGS]`:
/// a `--debug` build in a private directory, run under gdb or lldb with the
/// Invar helpers (`share/invar`) loaded. lldb is the default on macOS, gdb
/// elsewhere. The exit code is the debugger's.
#[cfg(feature = "native")]
fn session(path: &OsString, args: &[OsString]) -> i32 {
    let (options, extra) = match args.iter().position(|arg| arg == "--") {
        Some(at) => (&args[..at], &args[at + 1..]),
        None => (args, &[][..]),
    };
    let mut gdb = !cfg!(target_os = "macos");
    let (mut fuel, mut library) = (None, None);
    let mut options = options.iter();
    while let Some(option) = options.next() {
        match option.to_str() {
            Some("--gdb") => gdb = true,
            Some("--lldb") => gdb = false,
            Some("--fuel") => match options.next().and_then(|fuel| fuel.to_str()) {
                Some(value) if value.parse::<u64>().is_ok_and(|fuel| fuel > 0) => {
                    fuel = Some(value)
                }
                _ => return super::usage(),
            },
            Some("--c-library") => match options.next() {
                Some(value) => library = Some(value),
                None => return super::usage(),
            },
            _ => return super::usage(),
        }
    }
    let (debugger, name, helper) = if gdb {
        (
            "gdb",
            "invar_gdb.py",
            include_str!("../share/invar/invar_gdb.py"),
        )
    } else {
        (
            "lldb",
            "invar_lldb.py",
            include_str!("../share/invar/invar_lldb.py"),
        )
    };
    if std::process::Command::new(debugger)
        .arg("--version")
        .output()
        .is_err()
    {
        return refuse(path, false, &format!("{debugger} was not found on PATH"));
    }
    // Owner-only: the debugger loads the helper written here.
    let directory = match invar::native::unique_directory() {
        Ok(directory) => directory,
        Err(error) => return report_eval(build_error(error), path, false),
    };
    let program = directory.join("program");
    let helper_path = directory.join(name);
    DEBUG.store(true, Ordering::Relaxed);
    let code = match std::fs::write(&helper_path, helper) {
        Err(error) => report_eval(build_error(error.to_string()), path, false),
        Ok(()) => match build(path, &program.clone().into_os_string(), library) {
            0 => {
                let mut command = std::process::Command::new(debugger);
                if gdb {
                    command
                        .arg("-q")
                        .arg("-x")
                        .arg(&helper_path)
                        .args(extra)
                        .arg("--args");
                } else {
                    // Double-quoted for lldb's command line, where a space
                    // would split the path.
                    let mut quoted = String::new();
                    for c in helper_path.display().to_string().chars() {
                        if matches!(c, '\\' | '"') {
                            quoted.push('\\');
                        }
                        quoted.push(c);
                    }
                    let import = format!("command script import \"{quoted}\"");
                    command.arg("-o").arg(import).args(extra).arg("--");
                }
                command.arg(&program).args(fuel);
                match command.status() {
                    Ok(status) => status.code().unwrap_or(1),
                    Err(error) => {
                        refuse(path, false, &format!("{debugger} failed to start: {error}"))
                    }
                }
            }
            code => code,
        },
    };
    let _ = std::fs::remove_dir_all(&directory);
    code
}

#[cfg(not(feature = "native"))]
fn session(path: &OsString, _: &[OsString]) -> i32 {
    build(path, path, None)
}

fn refuse(path: &OsString, json: bool, message: &str) -> i32 {
    report_eval(
        EvalError::Tool {
            code: "DEBUG_UNSUPPORTED",
            message: message.into(),
        },
        path,
        json,
    )
}

#[cfg(feature = "native")]
pub(crate) fn build(path: &OsString, output: &OsString, c_library: Option<&OsString>) -> i32 {
    let (source, checked) = match super::load_checked(path, true) {
        Ok(loaded) => loaded,
        Err(error) => return super::report_front(error, path, false),
    };
    let result = invar::mir::lower(&checked)
        .map_err(|error| format!("{}: {}", error.code, error.message))
        .and_then(|program| {
            if !debug() {
                return invar::native::emit_mir(&program);
            }
            let absolute = std::path::absolute(path).map_err(|error| error.to_string())?;
            invar::native::emit_mir_debug(
                &program,
                &invar::native::DebugSource {
                    path: &absolute,
                    text: &source,
                },
            )
        })
        .and_then(|object| {
            invar::native::link_with_library(
                &object,
                std::path::Path::new(output),
                c_library.map(std::path::Path::new),
            )
        });
    match result {
        Ok(()) => {
            println!("Built {}", output.to_string_lossy());
            if debug() {
                println!("Debug info: DWARF 5 (native-serial)");
            }
            0
        }
        Err(message) => report_eval(build_error(message), path, false),
    }
}

#[cfg(feature = "native")]
pub(crate) fn build_error(message: String) -> EvalError {
    EvalError::Tool {
        code: "NATIVE_BUILD",
        message,
    }
}

#[cfg(not(feature = "native"))]
pub(crate) fn build(path: &OsString, _: &OsString, _: Option<&OsString>) -> i32 {
    report_eval(
        EvalError::Tool {
            code: "NATIVE_UNAVAILABLE",
            message: "native build requires the native Cargo feature and LLVM 21".into(),
        },
        path,
        false,
    )
}
