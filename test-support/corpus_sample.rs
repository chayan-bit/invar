//! Source corpus walk and the deterministic sample of the exhaustive corpus
//! sweeps that pull-request CI runs. `INVAR_FULL_CORPUS=1` (set by `just test`
//! and by CI on pushes to main) runs every program instead.
#![allow(dead_code)]

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

/// One program path in `STRIDE` is kept, chosen by a stable hash of the path.
pub const STRIDE: usize = 8;

/// Every `.inv` file under `dir`, in sorted path order.
pub fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "inv") {
            out.push(path);
        }
    }
}

/// True when `INVAR_FULL_CORPUS=1` asks for the full sweep.
pub fn full() -> bool {
    std::env::var_os("INVAR_FULL_CORPUS").is_some_and(|value| value == "1")
}

/// A count floor written for the full corpus, scaled to the sample.
pub fn floor(full_floor: usize) -> usize {
    if full() {
        full_floor
    } else {
        full_floor / STRIDE
    }
}

/// 64-bit FNV-1a over the `/`-separated path: the same on every host,
/// toolchain and run.
fn hash(path: &str) -> u64 {
    path.bytes()
        .map(|b| if b == b'\\' { b'/' } else { b })
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
}

/// A program's category: its first three directories, such as
/// `tests/fixtures/s1_03` or `examples/cells/reach_place`.
pub fn category(path: &str) -> String {
    let path = path.replace('\\', "/");
    let dir = path.rsplit_once('/').map_or("", |(dir, _)| dir);
    dir.split('/').take(3).collect::<Vec<_>>().join("/")
}

/// The program paths to run, from `(path, category)` pairs. Under
/// `INVAR_FULL_CORPUS=1` that is every path; otherwise the paths whose hash
/// falls in bucket 0 of `STRIDE`, plus the lowest-hash path of each category
/// the hash missed, so every category keeps at least one program.
pub fn paths(programs: impl IntoIterator<Item = (String, String)>) -> BTreeSet<String> {
    let programs: Vec<(String, String)> = programs.into_iter().collect();
    if full() {
        return programs.into_iter().map(|(path, _)| path).collect();
    }
    let mut kept: BTreeSet<String> = programs
        .iter()
        .filter(|(path, _)| hash(path).is_multiple_of(STRIDE as u64))
        .map(|(path, _)| path.clone())
        .collect();
    let mut least = BTreeMap::<&str, &str>::new();
    for (path, category) in &programs {
        let first = least.entry(category).or_insert(path);
        if hash(path) < hash(first) {
            *first = path.as_str();
        }
    }
    for (category, path) in least {
        if !programs
            .iter()
            .any(|(p, c)| c == category && kept.contains(p))
        {
            kept.insert(path.to_owned());
        }
    }
    let categories: BTreeSet<&str> = programs.iter().map(|(_, c)| c.as_str()).collect();
    let covered: BTreeSet<&str> = programs
        .iter()
        .filter(|(path, _)| kept.contains(path))
        .map(|(_, c)| c.as_str())
        .collect();
    assert_eq!(covered, categories, "the corpus sample lost a category");
    let total: BTreeSet<&str> = programs.iter().map(|(path, _)| path.as_str()).collect();
    println!(
        "corpus sample: {} of {} programs, every one of {} categories; INVAR_FULL_CORPUS=1 runs all",
        kept.len(),
        total.len(),
        categories.len()
    );
    kept
}

/// `items` whose path is in the sample, each path's `category` being its
/// directory; every item unless `sampled`.
pub fn keep<T>(items: Vec<T>, sampled: bool, path: impl Fn(&T) -> &String) -> Vec<T> {
    if !sampled {
        return items;
    }
    let kept = paths(
        items
            .iter()
            .map(|item| (path(item).clone(), category(path(item)))),
    );
    items
        .into_iter()
        .filter(|item| kept.contains(path(item)))
        .collect()
}

/// The whole corpus, as a 0-based `(index, count)` part.
pub const WHOLE: (usize, usize) = (0, 1);

/// The corpus part this process runs, from `INVAR_CORPUS_PART=k/n` (1-based,
/// so CI job names read `1/2`, `2/2`); the whole corpus when the variable is
/// unset or empty. Anything else fails the run.
pub fn corpus_part() -> (usize, usize) {
    match std::env::var("INVAR_CORPUS_PART") {
        Ok(value) => parse_part(&value),
        Err(std::env::VarError::NotPresent) => WHOLE,
        Err(error) => panic!("INVAR_CORPUS_PART: {error}"),
    }
}

/// `k/n` (1-based) as a 0-based part; empty is `WHOLE`; anything else panics.
pub fn parse_part(value: &str) -> (usize, usize) {
    if value.is_empty() {
        return WHOLE;
    }
    value
        .split_once('/')
        .and_then(|(k, n)| Some((k.parse::<usize>().ok()?, n.parse::<usize>().ok()?)))
        .filter(|&(k, n)| (1..=n).contains(&k))
        .map(|(k, n)| (k - 1, n))
        .unwrap_or_else(|| panic!("INVAR_CORPUS_PART must be k/n with 1 <= k <= n, got {value:?}"))
}

/// Part `(index, count)` of `items` (in their sorted path order): the items at
/// positions `index` mod `count`. Neighbouring programs share a directory and a
/// similar cost, so dealing them out in turn keeps the parts' run times close.
pub fn part<T>(items: Vec<T>, (index, count): (usize, usize)) -> Vec<T> {
    items
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % count == index)
        .map(|(_, item)| item)
        .collect()
}
