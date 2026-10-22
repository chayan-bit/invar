//! `invar-bench --domain-report SUITE|all` and `--write-domain-proposal [PATH]`.
use super::{manifest, report};
use std::{ffi::OsString, fs, path::PathBuf, thread};

const PROPOSAL: &str = "bench/manifest-v3-proposal.json";
/// The reference evaluator recurses; give it the stack a test thread lacks.
const STACK: usize = 64 * 1024 * 1024;

pub fn run(args: Vec<OsString>) -> Result<(), String> {
    let args: Vec<String> = args
        .into_iter()
        .map(|a| a.into_string().map_err(|_| "non-UTF-8 argument".to_owned()))
        .collect::<Result<_, _>>()?;
    let text = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["--domain-report", suite] => {
            let suite = (*suite).to_owned();
            thread::Builder::new()
                .stack_size(STACK)
                .spawn(move || report::report(&suite))
                .map_err(|e| e.to_string())?
                .join()
                .map_err(|_| "report thread panicked".to_owned())??
        }
        ["--write-domain-proposal"] => return write(PathBuf::from(PROPOSAL)),
        ["--write-domain-proposal", path] => return write(PathBuf::from(path)),
        _ => {
            return Err(
                "usage: invar-bench --domain-report <ml|games|robotics|all> \
                 | --write-domain-proposal [PATH]"
                    .into(),
            );
        }
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&text).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn write(path: PathBuf) -> Result<(), String> {
    let text = serde_json::to_string_pretty(&manifest::build()?).map_err(|e| e.to_string())?;
    fs::write(&path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
}
