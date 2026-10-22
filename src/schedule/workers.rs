//! Bounded scoped worker execution with chunk validation and fault injection.

use std::{
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

use crate::laws::{self, Affine, Law};

/// Bound scoped thread creation even on very large hosts or explicit requests.
pub const MAX_WORKERS: usize = 64;
const PARALLEL_THRESHOLD: usize = 64;

/// Automatic CPU width, respecting OS affinity/quota and preferring Apple's
/// performance cores. INVAR_CPU_WORKERS overrides the preference for testing or
/// measurement, within the OS limit and the hard bound.
pub fn auto_workers() -> Result<usize, Error> {
    let available = thread::available_parallelism().map_or(1, |count| count.get());
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let performance = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.perflevel0.physicalcpu"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|text| text.trim().parse::<usize>().ok());
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let performance = None;
    let override_count = match std::env::var("INVAR_CPU_WORKERS") {
        Ok(text) => Some(text),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err(Error::InvalidWorkers),
    };
    select_workers(available, performance, override_count.as_deref())
}

fn select_workers(
    available: usize,
    performance: Option<usize>,
    override_count: Option<&str>,
) -> Result<usize, Error> {
    let requested = if let Some(text) = override_count {
        let count = text.parse().map_err(|_| Error::InvalidWorkers)?;
        if !(1..=MAX_WORKERS).contains(&count) {
            return Err(Error::InvalidWorkers);
        }
        count
    } else {
        performance.filter(|count| *count > 0).unwrap_or(available)
    };
    Ok(requested.min(available.max(1)).clamp(1, MAX_WORKERS))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    None,
    Spawn,
    Join,
    Panic,
    Allocation,
    Overlap,
    Timeout,
    Cancel,
}

#[derive(Clone, Debug)]
pub struct Hooks {
    pub fault: Fault,
    /// Worker threads that began running.
    pub started: Arc<AtomicUsize>,
    /// Worker threads whose join completed (successfully or not).
    pub joined: Arc<AtomicUsize>,
    /// With a fault injected: every live `Array<u32>` at the failing loop head,
    /// read before the region's workers ran and again after they failed.
    pub live_arrays: Arc<Mutex<Option<LiveArrays>>>,
    /// Test seam (#263): forge the extracted normal form before it is checked.
    pub forge: Option<fn(&mut crate::general_net::normal_form::NormalForm)>,
}

/// Owner name and elements of each live `Array<u32>`, sorted by name.
pub type Snapshot = Vec<(String, Vec<u32>)>;

/// Runtime array state around a failed parallel region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiveArrays {
    pub before: Snapshot,
    pub after: Snapshot,
}

/// Panic payload of an injected worker that finished its chunk and then failed,
/// so its join reports an error.
struct JoinFailure;

impl Default for Hooks {
    fn default() -> Self {
        Self {
            fault: Fault::None,
            started: Arc::new(AtomicUsize::new(0)),
            joined: Arc::new(AtomicUsize::new(0)),
            live_arrays: Arc::new(Mutex::new(None)),
            forge: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidWorkers,
    ResourceLimit,
    Allocation,
    Spawn,
    Join,
    WorkerPanic,
    Overlap,
    Timeout,
    Cancelled,
    Candidate,
}

impl Error {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidWorkers => "CPU_WORKERS",
            Self::ResourceLimit => "RESOURCE_LIMIT",
            Self::Allocation => "RESOURCE_LIMIT",
            Self::Spawn => "CPU_SPAWN",
            Self::Join => "CPU_JOIN",
            Self::WorkerPanic => "CPU_WORKER_PANIC",
            Self::Overlap => "CPU_CHUNK_OVERLAP",
            Self::Timeout => "CPU_TIMEOUT",
            Self::Cancelled => "CPU_CANCELLED",
            Self::Candidate => "CPU_CANDIDATE",
        }
    }
}

pub(crate) type Scheduled<T> = (T, Vec<Range<usize>>, usize);

fn chunks(
    len: usize,
    workers: usize,
    fault: Fault,
    limit: usize,
) -> Result<Vec<Range<usize>>, Error> {
    if len > limit {
        return Err(Error::ResourceLimit);
    }
    if len == 0 {
        return Ok(Vec::new());
    }
    let count = if len < PARALLEL_THRESHOLD {
        1
    } else {
        workers.min(len)
    };
    let width = len.div_ceil(count);
    let mut ranges = (0..len)
        .step_by(width)
        .map(|start| start..(start + width).min(len))
        .collect::<Vec<_>>();
    if fault == Fault::Overlap && ranges.len() > 1 {
        ranges[1].start = ranges[0].start;
    }
    let mut end = 0;
    for range in &ranges {
        if range.start != end || range.end <= range.start || range.end > len {
            return Err(Error::Overlap);
        }
        end = range.end;
    }
    if end != len {
        return Err(Error::Overlap);
    }
    Ok(ranges)
}

/// Run one injected-fault-aware worker body over `range`.
fn worker_body<T>(
    fault: Fault,
    index: usize,
    range: Range<usize>,
    work: &(impl Fn(Range<usize>) -> Result<T, Error> + Sync),
) -> Result<T, Error> {
    match fault {
        Fault::Panic if index == 0 => {
            // Fill part of the worker's private buffer, then fail mid-chunk.
            let middle = range.start + range.len() / 2;
            let partial = work(range.start..middle);
            drop(partial);
            panic!("injected CPU worker panic");
        }
        Fault::Join if index == 0 => {
            let complete = work(range);
            drop(complete);
            std::panic::panic_any(JoinFailure);
        }
        _ => work(range),
    }
}

/// Split `0..len` into ordered chunks and run them on scoped worker threads.
/// Results are returned in chunk order only after every started worker joined;
/// workers write private buffers, never live state.
pub(crate) fn parallel_chunks<T: Send>(
    len: usize,
    workers: usize,
    hooks: &Hooks,
    work: impl Fn(Range<usize>) -> Result<T, Error> + Sync,
) -> Result<Scheduled<Vec<T>>, Error> {
    parallel_chunks_with_limit(len, workers, hooks, laws::MAX_ELEMENTS, work)
}

/// Raw host-buffer callers supply their independently verified admission cap.
/// Source executors retain the default source-array limit above.
pub(crate) fn parallel_chunks_with_limit<T: Send>(
    len: usize,
    workers: usize,
    hooks: &Hooks,
    limit: usize,
    work: impl Fn(Range<usize>) -> Result<T, Error> + Sync,
) -> Result<Scheduled<Vec<T>>, Error> {
    if !(1..=MAX_WORKERS).contains(&workers) {
        return Err(Error::InvalidWorkers);
    }
    // D6: timeout and cancellation have no source trigger; they are injected before work.
    match hooks.fault {
        Fault::Timeout => return Err(Error::Timeout),
        Fault::Cancel => return Err(Error::Cancelled),
        Fault::Allocation => return Err(Error::Allocation),
        _ => {}
    }
    let ranges = chunks(len, workers, hooks.fault, limit)?;
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(ranges.len())
        .map_err(|_| Error::Allocation)?;
    if ranges.len() <= 1 {
        for range in &ranges {
            outputs.push(work(range.clone())?);
        }
        return Ok((outputs, ranges, 0));
    }
    let fault = hooks.fault;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        thread::scope(|scope| {
            let mut handles = Vec::new();
            handles
                .try_reserve_exact(ranges.len())
                .map_err(|_| Error::Allocation)?;
            let mut first_error = None;
            for (index, range) in ranges.iter().cloned().enumerate() {
                // An injected spawn failure strikes after earlier workers are running.
                if fault == Fault::Spawn && index + 1 == ranges.len() {
                    first_error = Some(Error::Spawn);
                    break;
                }
                let started = Arc::clone(&hooks.started);
                let work_ref = &work;
                let spawned = thread::Builder::new().spawn_scoped(scope, move || {
                    started.fetch_add(1, Ordering::SeqCst);
                    worker_body(fault, index, range, work_ref)
                });
                match spawned {
                    Ok(handle) => handles.push(handle),
                    Err(_) => {
                        first_error = Some(Error::Spawn);
                        break;
                    }
                }
            }
            for handle in handles {
                let joined = handle.join();
                hooks.joined.fetch_add(1, Ordering::SeqCst);
                let error = match joined {
                    Ok(Ok(value)) => {
                        outputs.push(value);
                        continue;
                    }
                    Ok(Err(error)) => error,
                    Err(payload) if payload.is::<JoinFailure>() => Error::Join,
                    Err(_) => Error::WorkerPanic,
                };
                first_error.get_or_insert(error);
            }
            first_error.map_or(Ok(()), Err)
        })
    }));
    match outcome {
        Ok(Ok(())) => Ok((outputs, ranges.clone(), ranges.len())),
        Ok(Err(error)) => Err(error),
        Err(_) => Err(Error::WorkerPanic),
    }
}

pub(super) fn reduction_output(
    law: Law,
    input: &[u32],
    workers: usize,
    hooks: &Hooks,
) -> Result<Scheduled<u32>, Error> {
    let (parts, ranges, joined) = parallel_chunks(input.len(), workers, hooks, |range| {
        Ok(input[range]
            .iter()
            .copied()
            .fold(law.identity(), |a, b| law.combine(a, b)))
    })?;
    Ok((
        parts
            .into_iter()
            .fold(law.identity(), |a, b| law.combine(a, b)),
        ranges,
        joined,
    ))
}

pub(super) fn pair_reduction_output(
    a: &[u32],
    b: &[u32],
    workers: usize,
    hooks: &Hooks,
) -> Result<Scheduled<Affine>, Error> {
    if a.len() != b.len() {
        return Err(Error::Candidate);
    }
    let (parts, ranges, joined) = parallel_chunks(a.len(), workers, hooks, |range| {
        Ok(a[range.clone()]
            .iter()
            .zip(&b[range])
            .fold(Affine::IDENTITY, |acc, (&a, &b)| acc.then(Affine { a, b })))
    })?;
    Ok((
        parts.into_iter().fold(Affine::IDENTITY, Affine::then),
        ranges,
        joined,
    ))
}

pub(super) fn scan_output(
    values: &[Affine],
    seed: u32,
    workers: usize,
    hooks: &Hooks,
) -> Result<Scheduled<Vec<u32>>, Error> {
    let (parts, ranges, joined) = parallel_chunks(values.len(), workers, hooks, |range| {
        Ok(laws::scan_affine(&values[range])
            .map_err(|_| Error::ResourceLimit)?
            .prefixes)
    })?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(values.len())
        .map_err(|_| Error::Allocation)?;
    let mut carry = Affine::IDENTITY;
    for part in parts {
        for prefix in &part {
            result.push(carry.then(*prefix).apply(seed));
        }
        if let Some(last) = part.last() {
            carry = carry.then(*last);
        }
    }
    Ok((result, ranges, joined))
}

#[cfg(test)]
mod ai_01_fault_tests {
    use super::*;

    #[test]
    fn cpu_worker_selection_respects_hardware_preference_and_limits() {
        for (available, performance, override_count, expected) in [
            (16, None, None, 16),
            (10, Some(4), None, 4),
            (2, Some(4), None, 2),
            (10, Some(0), None, 10),
            (256, None, None, MAX_WORKERS),
            (0, None, None, 1),
            (10, Some(4), Some("10"), 10),
            (16, None, Some("2"), 2),
            (2, None, Some("16"), 2),
        ] {
            assert_eq!(
                select_workers(available, performance, override_count),
                Ok(expected)
            );
        }
        for text in ["0", "65", "", "-1", "garbage"] {
            assert_eq!(
                select_workers(16, None, Some(text)),
                Err(Error::InvalidWorkers)
            );
        }
    }

    fn injected(fault: Fault) -> Hooks {
        Hooks {
            fault,
            ..Hooks::default()
        }
    }

    #[test]
    fn issue_ai_01_machine_diagnostics_scheduler_faults() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../schema/invar-diagnostics-v1.schema.json"
        ))
        .unwrap();
        for (fault, code) in [
            (Fault::Spawn, "CPU_SPAWN"),
            (Fault::Join, "CPU_JOIN"),
            (Fault::Panic, "CPU_WORKER_PANIC"),
            (Fault::Overlap, "CPU_CHUNK_OVERLAP"),
            (Fault::Timeout, "CPU_TIMEOUT"),
            (Fault::Cancel, "CPU_CANCELLED"),
        ] {
            let error = parallel_chunks(128, 2, &injected(fault), |range| {
                Ok::<_, Error>(range.collect::<Vec<_>>())
            })
            .unwrap_err();
            assert_eq!(error.code(), code);
            let document = crate::diagnostics::document(
                "tool_error",
                vec![crate::diagnostics::diagnostic(
                    error.code(),
                    "injected scheduler fault",
                    "fixture.inv",
                    b"",
                    None,
                )],
            );
            assert!(jsonschema::is_valid(&schema, &document), "{document}");
        }
    }
}
