//! Thin calibration CLI; all measurements and profile validation live in lib.
use invar::placement::{self, calibration};
use serde_json::json;
use std::{ffi::OsString, path::PathBuf};
fn failure(error: placement::Error, message: &str) -> i32 {
    println!(
        "{}",
        invar::eval::EvalError::Tool {
            code: error.code(),
            message: message.into()
        }
        .to_diagnostics_json("", b"")
    );
    2
}
pub fn command(args: &[OsString]) -> i32 {
    let mut out = None;
    let mut family = false;
    let mut host = None;
    let mut options = args.chunks_exact(2);
    if !options.remainder().is_empty() {
        return failure(
            placement::Error::Invalid,
            "calibrate options require values",
        );
    }
    for pair in &mut options {
        if pair[0] == "--out" && out.is_none() {
            out = Some(PathBuf::from(&pair[1]));
        } else if pair[0] == "--family" && pair[1] == placement::MAP_FAMILY && !family {
            family = true;
        } else if pair[0] == "--host-class" && host.is_none() {
            host = pair[1].to_str().map(str::to_owned);
            if host.is_none() {
                return failure(placement::Error::Invalid, "host class must be UTF-8");
            }
        } else {
            return failure(
                placement::Error::Invalid,
                "unknown or duplicate calibrate option/family",
            );
        }
    }
    let host =
        host.unwrap_or_else(|| format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH));
    let path = out.unwrap_or_else(|| PathBuf::from(format!("calibration/{host}.json")));
    if path.to_str().is_none() {
        return failure(
            placement::Error::Invalid,
            "profile output path must be UTF-8",
        );
    }
    if std::fs::symlink_metadata(&path).is_ok() {
        return failure(placement::Error::Io, "profile path already exists");
    }
    // Archive builds supply the exact verified checkout revision explicitly.
    // Never infer source identity from an arbitrary runtime working directory.
    let revision = option_env!("INVAR_SOURCE_REVISION").unwrap_or("");
    if revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        return failure(
            placement::Error::Invalid,
            "build has no verified source revision; set INVAR_SOURCE_REVISION when building an archive",
        );
    }
    let document = match calibration::calibrate(&host, revision) {
        Ok(v) => v,
        Err(e) => return failure(e, "calibration profile rejected"),
    };
    if let Err(e) = calibration::publish(&path, &document) {
        return failure(
            e,
            "cannot publish profile without overwriting; create its parent directory first",
        );
    }
    println!(
        "{}",
        json!({"status":"ok","profile":path,"provisional":true,
        "numeric_contract_version":placement::MAP_NUMERIC_CONTRACT,
        "measurements":document["measurements"],"refusals":document["refusals"],
        "coverage":document["coverage"],"elapsed_ms":document["elapsed_ms"]})
    );
    0
}
