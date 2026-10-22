//! The bounded NVRTC helper child shared by every CUDA compile path.
//!
//! The helper leads its own process group. The source write and both output
//! readers run on worker threads, so nothing blocks past the deadline: not a
//! helper that never reads its input, not one that never exits, and not a
//! descendant that keeps the pipes open after the helper exits. Exit is
//! observed without reaping; on every return path the whole group is killed
//! while the leader is still unreaped, then the leader is reaped and the
//! workers are joined. A worker can outlive that only when a process that left
//! the group (a new session) holds its pipe; it is abandoned at the deadline,
//! and the compile is then a typed timeout.
use super::{CompiledCubin, Error, MAX_ERROR_BYTES};
use std::{
    io::{self, Read, Write},
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc,
    thread::JoinHandle,
    time::{Duration, Instant},
};

/// How long after a kill at or past the deadline the workers get to end, so a
/// timed-out or refused compile still joins them before returning.
const JOIN_GRACE: Duration = Duration::from_secs(1);

/// Keep at most `cap` bytes of a pipe and discard the rest to EOF, so the
/// writer never sees a closed pipe (EPIPE) while it is still running.
pub(super) fn drain_capped(mut pipe: impl Read, cap: usize) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    (&mut pipe).take(cap as u64).read_to_end(&mut bytes)?;
    io::copy(&mut pipe, &mut io::sink())?;
    Ok(bytes)
}

/// One worker's result. The slot index is also the worker's index.
enum Io {
    Wrote(io::Result<()>),
    Image(io::Result<Vec<u8>>),
    Log(io::Result<Vec<u8>>),
}

#[derive(Default)]
struct Streams {
    wrote: Option<io::Result<()>>,
    image: Option<io::Result<Vec<u8>>>,
    log: Option<io::Result<Vec<u8>>>,
}
impl Streams {
    fn put(&mut self, io: Io) {
        match io {
            Io::Wrote(r) => self.wrote = Some(r),
            Io::Image(r) => self.image = Some(r),
            Io::Log(r) => self.log = Some(r),
        }
    }
    fn arrived(&self) -> [bool; 3] {
        [
            self.wrote.is_some(),
            self.image.is_some(),
            self.log.is_some(),
        ]
    }
}

/// The helper as its group's leader. `reap` (also run on drop) kills the whole
/// group while the leader is unreaped, so the group id cannot have been
/// reused, then reaps the leader exactly once.
struct Group(Child, Option<io::Result<ExitStatus>>);
impl Group {
    fn reap(&mut self) -> &io::Result<ExitStatus> {
        self.1.get_or_insert_with(|| {
            #[cfg(unix)]
            if let Ok(pgid) = libc::pid_t::try_from(self.0.id()) {
                // SAFETY: plain syscall on the process group our unreaped
                // child leads.
                unsafe { libc::killpg(pgid, libc::SIGKILL) };
            }
            let _ = self.0.kill();
            self.0.wait()
        })
    }
}
impl Drop for Group {
    fn drop(&mut self) {
        self.reap();
    }
}

/// Whether the helper has exited, observed without reaping it.
#[cfg(unix)]
fn exited(child: &mut Child) -> io::Result<bool> {
    let pid = libc::id_t::from(child.id());
    // SAFETY: an all-zero siginfo_t is a valid value; waitid writes only it.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    // SAFETY: `pid` is our unreaped child; WNOWAIT leaves it waitable and
    // WNOHANG returns at once. `info` is a valid, writable siginfo_t.
    let rc = unsafe {
        libc::waitid(
            libc::P_PID,
            pid,
            &mut info,
            libc::WEXITED | libc::WNOWAIT | libc::WNOHANG,
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: waitid filled `info`, or left it zeroed when nothing exited.
    Ok(unsafe { info.si_pid() } != 0)
}
/// Without process groups, exit is observed by reaping.
#[cfg(not(unix))]
fn exited(child: &mut Child) -> io::Result<bool> {
    child.try_wait().map(|s| s.is_some())
}

fn compile_error(error: impl ToString) -> Error {
    Error::Compile(error.to_string())
}

/// Run the helper `command` on `source` within `deadline`, keeping at most
/// `image_cap` image bytes and a bounded log prefix. GNET permits truncated
/// logs; strict floating helpers reject overflow. The command's stdio and
/// process group are set here.
pub(super) fn compile(
    command: &mut Command,
    source: &str,
    image_cap: usize,
    deadline: Duration,
    allow_truncated_log: bool,
) -> Result<CompiledCubin, Error> {
    let started = Instant::now();
    let mut input = Vec::new();
    input
        .try_reserve_exact(source.len())
        .map_err(|_| Error::OutOfMemory)?;
    input.extend_from_slice(source.as_bytes());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(command, 0);
    let mut child = Group(
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(compile_error)?,
        None,
    );
    let (send, recv) = mpsc::channel();
    let mut workers: [Option<JoinHandle<()>>; 3] = [None, None, None];
    let mut streams = Streams::default();
    let watched = start(&mut child.0, input, image_cap, send, &mut workers).and_then(|()| {
        watch(
            &mut child.0,
            &recv,
            &mut streams,
            image_cap,
            started,
            deadline,
        )
    });
    let status = child.reap().as_ref().map(|s| *s).map_err(compile_error);
    // The group is dead, so every pipe it held is closed and each worker ends
    // at once; only a pipe held outside the group can outlast this wait.
    let join_by = (started + deadline).max(Instant::now() + JOIN_GRACE);
    while streams.arrived().contains(&false) {
        let Some(left) = join_by.checked_duration_since(Instant::now()) else {
            break;
        };
        match recv.recv_timeout(left) {
            Ok(io) => streams.put(io),
            Err(_) => break,
        }
    }
    for (worker, arrived) in workers.into_iter().zip(streams.arrived()) {
        // A worker that has not reported is blocked on a pipe held outside the
        // group; it is abandoned and ends when that pipe closes.
        if let (Some(worker), true) = (worker, arrived) {
            let _ = worker.join();
        }
    }
    watched?;
    let status = status?;
    let (Some(wrote), Some(image), Some(log)) = (streams.wrote, streams.image, streams.log) else {
        return Err(Error::Timeout);
    };
    let (image, log) = (image.map_err(compile_error)?, log.map_err(compile_error)?);
    if image.len() > image_cap {
        return Err(Error::Limit);
    }
    if !status.success() {
        return Err(Error::Compile(format!(
            "compiler exit {status}: {}",
            String::from_utf8_lossy(&log)
        )));
    }
    if let Err(error) = wrote {
        return Err(Error::Compile(format!("compiler stdin: {error}")));
    }
    if image.is_empty() || (!allow_truncated_log && log.len() > MAX_ERROR_BYTES) {
        return Err(Error::Limit);
    }
    Ok(CompiledCubin {
        image,
        log: String::from_utf8_lossy(&log).trim_end().into(),
    })
}

/// Start the source writer and both readers. The image reader stops one byte
/// past its cap, which `watch` refuses at once; the log reader drains to EOF.
fn start(
    child: &mut Child,
    input: Vec<u8>,
    image_cap: usize,
    send: mpsc::Sender<Io>,
    workers: &mut [Option<JoinHandle<()>>; 3],
) -> Result<(), Error> {
    let (Some(mut stdin), Some(stdout), Some(stderr)) =
        (child.stdin.take(), child.stdout.take(), child.stderr.take())
    else {
        return Err(Error::Compile("compiler pipes unavailable".into()));
    };
    let jobs: [Box<dyn FnOnce() -> Io + Send>; 3] = [
        Box::new(move || Io::Wrote(stdin.write_all(&input))),
        Box::new(move || {
            let mut bytes = Vec::new();
            Io::Image(
                stdout
                    .take(image_cap as u64 + 1)
                    .read_to_end(&mut bytes)
                    .map(|_| bytes),
            )
        }),
        Box::new(move || Io::Log(drain_capped(stderr, MAX_ERROR_BYTES + 1))),
    ];
    for (slot, job) in workers.iter_mut().zip(jobs) {
        let send = send.clone();
        let worker = std::thread::Builder::new().spawn(move || {
            let _ = send.send(job());
        });
        *slot = Some(worker.map_err(compile_error)?);
    }
    Ok(())
}

/// Poll until the helper exits, its image overflows, or the deadline passes.
/// Never reaps the helper.
fn watch(
    child: &mut Child,
    recv: &mpsc::Receiver<Io>,
    streams: &mut Streams,
    image_cap: usize,
    started: Instant,
    deadline: Duration,
) -> Result<(), Error> {
    loop {
        while let Ok(io) = recv.try_recv() {
            streams.put(io);
        }
        if matches!(&streams.image, Some(Ok(image)) if image.len() > image_cap) {
            return Err(Error::Limit);
        }
        if exited(child).map_err(compile_error)? {
            return Ok(());
        }
        if started.elapsed() >= deadline {
            return Err(Error::Timeout);
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[cfg(all(test, unix))]
pub(super) mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Checked-in stand-in helpers; never written at test time, so a
    /// concurrent fork cannot make their exec fail with ETXTBSY.
    pub(in crate::cuda) fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/compiler-child")
            .join(name)
    }

    /// Whether `pid` still exists, polled for up to 2 s while its new parent
    /// reaps it.
    pub(in crate::cuda) fn lingers(pid: libc::pid_t) -> bool {
        let begun = Instant::now();
        // SAFETY: signal 0 only checks that `pid` exists.
        while unsafe { libc::kill(pid, 0) } == 0 {
            if begun.elapsed() > Duration::from_secs(2) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }

    /// The descendant pid a fixture printed as `descendant=<pid>`.
    pub(in crate::cuda) fn descendant(text: &str) -> libc::pid_t {
        text.rsplit("descendant=")
            .next()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or_else(|| panic!("no descendant pid in {text:?}"))
    }

    fn run(
        name: &str,
        pidfile: Option<&PathBuf>,
        deadline: Duration,
    ) -> Result<CompiledCubin, Error> {
        let mut command = Command::new(fixture(name));
        if let Some(pidfile) = pidfile {
            command.env("INVAR_COMPILER_PIDFILE", pidfile);
        }
        compile(
            &mut command,
            "extern \"C\" __global__ void k() {}",
            1024,
            deadline,
            false,
        )
    }

    #[test]
    fn compiler_child_kills_the_lingering_descendant_and_keeps_the_success() {
        let begun = Instant::now();
        let compiled = run("lingering.sh", None, Duration::from_secs(20)).unwrap();
        assert!(
            begun.elapsed() < Duration::from_secs(5),
            "{:?}",
            begun.elapsed()
        );
        assert_eq!(compiled.image, b"mock-cubin");
        assert!(!lingers(descendant(&compiled.log)), "{}", compiled.log);
    }

    #[test]
    fn compiler_child_lingering_descendant_keeps_an_oversized_log_a_limit() {
        let begun = Instant::now();
        let result = run("lingering-log.sh", None, Duration::from_secs(20));
        assert!(
            matches!(result, Err(Error::Limit)),
            "{:?}",
            result.as_ref().err()
        );
        assert!(
            begun.elapsed() < Duration::from_secs(5),
            "{:?}",
            begun.elapsed()
        );
    }

    #[test]
    fn compiler_child_that_never_exits_times_out_and_leaves_no_descendant() {
        let pidfile =
            std::env::temp_dir().join(format!("invar-compiler-child-{}.pid", std::process::id()));
        let deadline = Duration::from_millis(500);
        let begun = Instant::now();
        let result = run("never-exits.sh", Some(&pidfile), deadline);
        let took = begun.elapsed();
        assert!(
            matches!(result, Err(Error::Timeout)),
            "{:?}",
            result.as_ref().err()
        );
        assert!(
            took >= deadline && took < deadline + Duration::from_secs(3),
            "{took:?}"
        );
        let pid = std::fs::read_to_string(&pidfile).unwrap();
        let _ = std::fs::remove_file(&pidfile);
        assert!(!lingers(descendant(&pid)), "{pid}");
    }

    /// A descendant in its own session survives the group kill and holds the
    /// pipes; the compile still ends at the deadline with a typed timeout.
    #[cfg(target_os = "linux")]
    #[test]
    fn compiler_child_escaped_descendant_ends_at_the_deadline() {
        let deadline = Duration::from_secs(1);
        let begun = Instant::now();
        let result = run("escaped.sh", None, deadline);
        let took = begun.elapsed();
        assert!(
            matches!(result, Err(Error::Timeout)),
            "{:?}",
            result.as_ref().err()
        );
        assert!(
            took >= deadline && took < deadline + Duration::from_secs(3),
            "{took:?}"
        );
    }

    /// Many concurrent helpers, each leaving a descendant on its pipes: every
    /// compile succeeds well inside the deadline and no descendant survives.
    #[test]
    fn compiler_child_parallel_lingering_stress() {
        let runs: Vec<_> = (0..64)
            .map(|_| std::thread::spawn(|| run("lingering.sh", None, Duration::from_secs(20))))
            .collect();
        let begun = Instant::now();
        for result in runs {
            let compiled = result.join().unwrap().unwrap();
            assert_eq!(compiled.image, b"mock-cubin");
            assert!(!lingers(descendant(&compiled.log)), "{}", compiled.log);
        }
        assert!(
            begun.elapsed() < Duration::from_secs(15),
            "{:?}",
            begun.elapsed()
        );
    }
}
