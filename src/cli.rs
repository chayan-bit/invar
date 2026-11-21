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

/// Take `--max-c LEVEL`, `--max-n LEVEL` and `--region-report` out of a `run`,
/// `check` or `explain` command line (`docs/spec/invar-v0.2-axes.md`). A cap
/// refuses a function declared above it and never raises an undeclared one; the
/// report (with `check` only) prints each function's axes and ends the command.
pub(crate) fn region_options(args: Vec<OsString>) -> Result<Vec<OsString>, i32> {
    use invar::axes::{AXES, Axis, C_AXIS, N_AXIS};
    const CAPS: [(&str, &Axis, &[&str]); 2] = [
        ("--max-c", &C_AXIS, &["c0", "c1", "c2"]),
        ("--max-n", &N_AXIS, &["n0", "n1", "n2", "n3"]),
    ];
    let (mut kept, mut caps, mut report) = (Vec::new(), Vec::new(), false);
    let mut rest = args.into_iter();
    while let Some(arg) = rest.next() {
        if let Some((_, axis, allowed)) = CAPS.iter().find(|(flag, ..)| arg == *flag) {
            let value = rest.next().and_then(|value| value.into_string().ok());
            match value.filter(|value| allowed.contains(&value.as_str())) {
                Some(cap)
                    if !caps
                        .iter()
                        .any(|(known, _): &(&Axis, String)| known.key == axis.key) =>
                {
                    caps.push((*axis, cap));
                }
                _ => return Err(usage()),
            }
        } else if arg == "--region-report" && !report {
            report = true;
        } else {
            kept.push(arg);
        }
    }
    if caps.is_empty() && !report {
        return Ok(kept);
    }
    let command = kept.get(1).and_then(|command| command.to_str());
    if kept.len() < 3
        || !matches!(command, Some("run" | "check" | "explain"))
        || (report && command != Some("check"))
    {
        return Err(usage());
    }
    let (path, json) = (
        &kept[2],
        kept.windows(2)
            .any(|pair| pair[0] == "--format" && pair[1] == "json"),
    );
    let caps: Vec<_> = caps
        .iter()
        .map(|(axis, cap)| (*axis, cap.as_str()))
        .collect();
    let program = crate::read_source(path)
        .and_then(|source| invar::front::parse(&source))
        .and_then(|program| invar::axes::check_caps(&program, &caps).map(|()| program));
    let program = program.map_err(|error| crate::report_front(error, path, json))?;
    if !report {
        return Ok(kept);
    }
    let rows: Vec<_> = program
        .functions
        .iter()
        .map(|function| {
            let name = function
                .module
                .as_ref()
                .map_or(function.name.clone(), |module| {
                    format!("{module}::{}", function.name)
                });
            let mut row =
                serde_json::json!({"function": name, "contract": function.axes.contract()});
            for axis in AXES {
                row[axis.key] = function.axes.level(axis).into();
            }
            row
        })
        .collect();
    if json {
        println!("{}", serde_json::json!({"regions": rows}));
    } else {
        println!(
            "{:<24} {:<4} {:<18} {:<4} {:<4} t",
            "function", "c", "contract", "s", "n"
        );
        for row in &rows {
            let text = |key: &str| row[key].as_str().unwrap_or("-").to_owned();
            println!(
                "{:<24} {:<4} {:<18} {:<4} {:<4} {}",
                text("function"),
                text("c"),
                text("contract"),
                text("s"),
                text("n"),
                text("t")
            );
        }
    }
    Err(0)
}
