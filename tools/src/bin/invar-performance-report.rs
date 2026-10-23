#[path = "../../bench/acceptance.rs"]
mod acceptance;
#[path = "../../bench/hash.rs"]
mod hash;

use serde_json::Value;
use std::{
    env, fs,
    io::{Read, Write},
    path::PathBuf,
    process::ExitCode,
};

fn read(path: &PathBuf) -> Result<Vec<u8>, String> {
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() > 32 * 1024 * 1024 {
        return Err("input exceeds 32 MiB".into());
    }
    let mut raw = Vec::new();
    file.take(32 * 1024 * 1024 + 1)
        .read_to_end(&mut raw)
        .map_err(|e| e.to_string())?;
    if raw.len() > 32 * 1024 * 1024 {
        return Err("input exceeds 32 MiB".into());
    }
    Ok(raw)
}
fn run() -> Result<bool, String> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() != 10 {
        return Err("usage: invar-performance-report --manifest FILE --plan FILE --results FILE --report NEW.json --markdown NEW.md".into());
    }
    let mut paths = std::collections::BTreeMap::new();
    for pair in args.chunks_exact(2) {
        let key = pair[0].to_str().ok_or("invalid option")?;
        if ![
            "--manifest",
            "--plan",
            "--results",
            "--report",
            "--markdown",
        ]
        .contains(&key)
            || paths.insert(key, PathBuf::from(&pair[1])).is_some()
        {
            return Err("unknown/duplicate option".into());
        }
    }
    let get = |key| paths.get(key).ok_or_else(|| format!("missing {key}"));
    let manifest = read(get("--manifest")?)?;
    let plan: Value = serde_json::from_slice(&read(get("--plan")?)?).map_err(|e| e.to_string())?;
    let bundle: Value =
        serde_json::from_slice(&read(get("--results")?)?).map_err(|e| e.to_string())?;
    let report = acceptance::report(&manifest, &plan, &bundle)?;
    let json = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
    let markdown = acceptance::markdown(&report);
    // Reserve both outputs before writing either; never overwrite a prior report.
    let mut json_file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(get("--report")?)
        .map_err(|e| e.to_string())?;
    let mut md_file = match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(get("--markdown")?)
    {
        Ok(file) => file,
        Err(error) => {
            fs::remove_file(get("--report")?).map_err(|e| e.to_string())?;
            return Err(error.to_string());
        }
    };
    json_file
        .write_all(&json)
        .and_then(|()| json_file.sync_all())
        .map_err(|e| e.to_string())?;
    md_file
        .write_all(markdown.as_bytes())
        .and_then(|()| md_file.sync_all())
        .map_err(|e| e.to_string())?;
    println!(
        "{}: {} / {} passed; {}",
        report["status"], report["passed"], report["denominator"], report["counts"]
    );
    Ok(report["status"] == "passed")
}
fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(error) => {
            eprintln!("M5_REPORT_REFUSED: {error}");
            ExitCode::from(2)
        }
    }
}
