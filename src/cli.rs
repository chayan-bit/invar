//! Command-line helpers shared by the commands in `main.rs`, and the options
//! taken out of the argument list before its positional parsing sees them.

use std::ffi::OsString;

pub(crate) fn usage() -> i32 {
    eprintln!("{}", invar::help());
    2
}

pub(crate) fn source_directory(path: &OsString) -> &std::path::Path {
    std::path::Path::new(path)
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."))
}

/// `--numeric-contract trig-f32-v1` on `run`, `explain` and `build` (#210 C4):
/// f32 `sin` and `cos` run `invar-trig-f32-v1` for this process. Returns the
/// arguments without the option, or the exit code of a refused value.
pub(crate) fn numeric_contract(mut args: Vec<OsString>) -> Result<Vec<OsString>, i32> {
    let command = args.get(1).and_then(|arg| arg.to_str());
    if !matches!(command, Some("run" | "explain" | "build")) {
        return Ok(args);
    }
    while let Some(at) = args.iter().position(|arg| arg == "--numeric-contract") {
        match args.get(at + 1).and_then(|value| value.to_str()) {
            Some("trig-f32-v1") => invar::math::select_trig_f32_v1(),
            value => {
                eprintln!(
                    "error[NUMERIC_CONTRACT]: --numeric-contract takes trig-f32-v1, not {}",
                    value.unwrap_or("nothing")
                );
                return Err(2);
            }
        }
        args.drain(at..at + 2);
    }
    Ok(args)
}
