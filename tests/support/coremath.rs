//! Upstream CORE-MATH as an independent oracle: builds
//! `crates/invar-math/oracle/driver.c` against a fresh upstream clone named by
//! `INVAR_COREMATH_UPSTREAM` and streams its `(x, y, result)` records.
#![allow(dead_code)]

use invar::math::MathFn;
use std::{
    io::{BufReader, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

pub const SEED: &str = "20261018";

pub fn upstream() -> PathBuf {
    PathBuf::from(std::env::var_os("INVAR_COREMATH_UPSTREAM").expect("set INVAR_COREMATH_UPSTREAM"))
}

/// The library functions CORE-MATH implements.
pub fn transcendental() -> impl Iterator<Item = MathFn> {
    MathFn::ALL
        .into_iter()
        .filter(|f| invar::math::core_math(*f).is_some())
}

fn source(root: &Path, f: MathFn, binary64: bool) -> PathBuf {
    let name = f.name();
    if binary64 {
        root.join(format!("src/binary64/{name}/{name}.c"))
    } else {
        root.join(format!("src/binary32/{name}/{name}f.c"))
    }
}

/// Build the driver against upstream; a mutation `(file, from, to)` replaces
/// one unique substring in a private copy of that file (and its headers).
pub fn build(directory: &Path, mutation: Option<(&str, &str, &str)>) -> PathBuf {
    let root = upstream();
    let mut sources = Vec::new();
    for binary64 in [false, true] {
        sources.extend(transcendental().map(|f| source(&root, f, binary64)));
    }
    if let Some((relative, from, to)) = mutation {
        let original = root.join("src").join(relative);
        let copy = directory.join("mutant").join(relative);
        std::fs::create_dir_all(copy.parent().unwrap()).unwrap();
        for entry in std::fs::read_dir(original.parent().unwrap()).unwrap() {
            let entry = entry.unwrap().path();
            if entry.extension().is_some_and(|e| e == "h") {
                std::fs::copy(
                    &entry,
                    copy.parent().unwrap().join(entry.file_name().unwrap()),
                )
                .unwrap();
            }
        }
        let text = std::fs::read_to_string(&original).unwrap();
        assert_eq!(
            text.matches(from).count(),
            1,
            "mutation site must be unique"
        );
        std::fs::write(&copy, text.replace(from, to)).unwrap();
        sources.retain(|path| path != &original);
        sources.push(copy);
    }
    let driver = directory.join(if mutation.is_some() {
        "mutant-driver"
    } else {
        "driver"
    });
    let status = Command::new("cc")
        .args(["-O3", "-march=native", "-std=gnu11", "-D_GNU_SOURCE"])
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/crates/invar-math/oracle/driver.c"
        ))
        .args(&sources)
        .arg("-o")
        .arg(&driver)
        .arg("-lm")
        .status()
        .unwrap();
    assert!(status.success());
    driver
}

pub fn worst_cases(f: MathFn, binary64: bool) -> Option<PathBuf> {
    let path = source(&upstream(), f, binary64).with_extension("wc");
    path.exists().then_some(path)
}

/// Stream the driver's records for one function and type in chunks of at
/// most `chunk`: worst cases, specials, `random` samples, then (binary32
/// univariate) every `stride`-th encoding.
pub fn records(
    driver: &Path,
    f: MathFn,
    binary64: bool,
    random: u64,
    stride: u64,
    chunk: usize,
    mut each: impl FnMut(&[[u64; 3]]),
) {
    let wc = worst_cases(f, binary64).map_or("-".into(), |p| p.display().to_string());
    let mut child = Command::new(driver)
        .args([
            f.name(),
            if binary64 { "64" } else { "32" },
            &wc,
            &random.to_string(),
            SEED,
            &stride.to_string(),
        ])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut reader = BufReader::with_capacity(1 << 20, child.stdout.take().unwrap());
    let mut batch = Vec::with_capacity(chunk);
    let mut record = [0u8; 24];
    while reader.read_exact(&mut record).is_ok() {
        let word = |i: usize| u64::from_le_bytes(record[i * 8..i * 8 + 8].try_into().unwrap());
        batch.push([word(0), word(1), word(2)]);
        if batch.len() == chunk {
            each(&batch);
            batch.clear();
        }
    }
    if !batch.is_empty() {
        each(&batch);
    }
    assert!(child.wait().unwrap().success());
}

/// Whether two encodings agree, with NaN as one class.
pub fn same(binary64: bool, got: u64, want: u64) -> bool {
    let nan = |bits: u64| {
        if binary64 {
            f64::from_bits(bits).is_nan()
        } else {
            f32::from_bits(bits as u32).is_nan()
        }
    };
    got == want || nan(got) && nan(want)
}

pub fn tempdir(name: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!("invar-math-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    directory
}
