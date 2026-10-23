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
