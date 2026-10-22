//! Scratch names unique within and across test processes. The clock can tick
//! in microseconds (macOS), so nanos alone repeat for concurrent callers.
use std::{
    process,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

/// `<pid>-<counter>-<nanos>`: the counter separates callers in one process,
/// the pid and nanos separate processes.
pub fn unique() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "{}-{}-{nanos}",
        process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}
