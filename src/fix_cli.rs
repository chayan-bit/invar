//! `invar check PATH --fix [--write]`: apply validated fixes in rounds, print
//! the unified diff, and write the file only with `--write` and only when it
//! is byte-for-byte what was checked.

use std::ffi::OsString;

use sha2::{Digest, Sha256};

pub fn run(path: &OsString, flags: &[OsString]) -> i32 {
    let write = match flags {
        [] => false,
        [flag] if flag == "--write" => true,
        _ => return crate::usage(),
    };
    let source = match crate::read_source(path) {
        Ok(source) => source,
        Err(error) => return crate::report_front(error, path, false),
    };
    let name = path.to_string_lossy();
    if invar::front::contains_system_declaration(&source) {
        eprintln!("{name}: --fix does not apply to system files");
        return 2;
    }
    let outcome = invar::fixes::fix(&source);
    let (Ok(old), Ok(new)) = (
        std::str::from_utf8(&source),
        std::str::from_utf8(&outcome.fixed),
    ) else {
        eprintln!("{name}: --fix needs UTF-8 source");
        return 2;
    };
    print!("{}", invar::fixes::unified_diff(&name, old, new));
    if !outcome.applied.is_empty() {
        eprintln!("{name}: applied {}", outcome.applied.join("; "));
    }
    if write
        && outcome.fixed != source
        && let Err(message) =
            invar::fixes::write_if_unchanged(path, &Sha256::digest(&source), &outcome.fixed)
    {
        eprintln!("{name}: {message}");
        return 2;
    }
    match outcome.remaining {
        None => 0,
        Some(error) => crate::report_front(error, path, false),
    }
}
