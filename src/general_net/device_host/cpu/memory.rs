//! Memory admission of CPU device model compiles. A starved host (another
//! tenant holding most of its RAM) must slow the corpus down, not have the
//! kernel kill `cc1plus` mid-compile.
use std::{
    fs,
    path::Path,
    sync::{Condvar, Mutex, PoisonError},
    time::{Duration, Instant},
};

/// Memory reserved for each running compile. Over the 475 compiles of
/// `issue_gnet_03_device_corpus_cpu_model_agrees_with_oracle` on a Linux test machine
/// (g++, -O1, UBSan) peak RSS is 458 MiB at the median, 706 MiB at p90 and
/// 833 MiB at most.
const COMPILE_RESERVE_BYTES: u64 = 1 << 30;

/// How long a compile waits when none of this process is running. After it
/// the compile starts anyway, as before this admission: the reading may be
/// pessimistic, and a starved host then fails no worse than it did. It is the
/// compile wall deadline, so a waiting corpus still ends in bounded time.
const FIRST_WAIT: Duration = Duration::from_secs(300);

/// Compiles running in this process, and the waiters for a free one.
static RUNNING: (Mutex<usize>, Condvar) = (Mutex::new(0), Condvar::new());

/// A running compile; dropping it lets the next waiter re-check memory.
pub(super) struct Slot;

impl Drop for Slot {
    fn drop(&mut self) {
        let (running, freed) = &RUNNING;
        *running.lock().unwrap_or_else(PoisonError::into_inner) -= 1;
        freed.notify_all();
    }
}

/// Wait until memory admits one more compile. Memory other processes free is
/// seen on the next poll; a compile of this process ending wakes waiters at once.
pub(super) fn slot() -> Slot {
    let (running, freed) = &RUNNING;
    let start = Instant::now();
    let mut active = running.lock().unwrap_or_else(PoisonError::into_inner);
    while !admits(*active, available()) && (*active > 0 || start.elapsed() < FIRST_WAIT) {
        active = freed
            .wait_timeout(active, Duration::from_millis(500))
            .unwrap_or_else(PoisonError::into_inner)
            .0;
    }
    *active += 1;
    Slot
}

/// A compile needs a reserve for itself and every compile already running.
/// Without a memory reading (no `/proc`) compiles are not bounded.
fn admits(active: usize, available: Option<u64>) -> bool {
    let needed = (active as u64 + 1).saturating_mul(COMPILE_RESERVE_BYTES);
    available.is_none_or(|bytes| bytes >= needed)
}

/// Bytes this process may still allocate: the host's `MemAvailable`, lowered
/// by the room left under each enclosing cgroup v2 `memory.max`. A build job's
/// cgroup charges its page cache, so reclaimable inactive file pages count as
/// room there.
fn available() -> Option<u64> {
    let host = mem_available(&fs::read_to_string("/proc/meminfo").ok()?)?;
    let cgroup = fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    let path = cgroup.lines().find_map(|line| line.strip_prefix("0::"));
    let rooms = path.into_iter().flat_map(|path| {
        Path::new("/sys/fs/cgroup")
            .join(path.trim_start_matches('/'))
            .ancestors()
            .take_while(|dir| dir.starts_with("/sys/fs/cgroup"))
            .filter_map(|dir| {
                let read = |name: &str| fs::read_to_string(dir.join(name)).ok();
                let stat = read("memory.stat").unwrap_or_default();
                cgroup_room(&read("memory.max")?, &read("memory.current")?, &stat)
            })
            .collect::<Vec<_>>()
    });
    Some(rooms.fold(host, u64::min))
}

/// `MemAvailable` from `/proc/meminfo`, in bytes.
fn mem_available(meminfo: &str) -> Option<u64> {
    let line = meminfo
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))?;
    let kib = line.trim().strip_suffix("kB")?.trim().parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

/// Room left under one cgroup's `memory.max`, counting its `inactive_file`
/// pages as reclaimable; `max` means no limit there.
fn cgroup_room(max: &str, current: &str, stat: &str) -> Option<u64> {
    let max = max.trim().parse::<u64>().ok()?;
    let current = current.trim().parse::<u64>().ok()?;
    let reclaimable = stat
        .lines()
        .find_map(|line| line.strip_prefix("inactive_file "))
        .and_then(|bytes| bytes.trim().parse::<u64>().ok())
        .unwrap_or(0);
    Some(max.saturating_sub(current.saturating_sub(reclaimable)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn every_compile_needs_a_reserve_for_itself_and_each_running_one() {
        assert!(
            !admits(0, Some(0)),
            "a known-low reading holds the first compile too"
        );
        assert!(!admits(0, Some(COMPILE_RESERVE_BYTES - 1)));
        assert!(admits(0, Some(COMPILE_RESERVE_BYTES)));
        assert!(!admits(1, Some(2 * COMPILE_RESERVE_BYTES - 1)));
        assert!(admits(1, Some(2 * COMPILE_RESERVE_BYTES)));
        assert!(!admits(4, Some(5 * COMPILE_RESERVE_BYTES - 1)));
        assert!(
            admits(0, None) && admits(7, None),
            "no reading leaves compiles unbounded"
        );
    }

    #[test]
    fn readings_parse_meminfo_and_cgroup_limits() {
        let meminfo = "MemTotal:       131072000 kB\nMemAvailable:    5242880 kB\n";
        assert_eq!(mem_available(meminfo), Some(5 * GIB));
        assert_eq!(mem_available("MemTotal: 1 kB\n"), None);
        assert_eq!(cgroup_room("max\n", "123\n", ""), None);
        assert_eq!(
            cgroup_room("8589934592\n", "6442450944\n", ""),
            Some(2 * GIB)
        );
        assert_eq!(cgroup_room("100\n", "200\n", ""), Some(0));
        let stat = "anon 1073741824\nactive_file 5\ninactive_file 4294967296\n";
        assert_eq!(
            cgroup_room("8589934592\n", "8589934592\n", stat),
            Some(4 * GIB),
            "inactive page cache is reclaimable room"
        );
    }
}
