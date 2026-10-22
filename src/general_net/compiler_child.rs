//! The bounded C++ compiler child shared by the native numeric hosts
//! (GNET-08 to GNET-11).
//!
//! The compiler runs in its own process group. Its exit is observed without
//! reaping it, and on every return path the whole group is killed before the
//! leader is reaped, so descendants never outlive the call and the group id
//! cannot be reused under the kill. Each stream is captured to a log file that
//! exists before the child starts; before returning, the capture workers are
//! given `LOG_GRACE` after the kill to drain what the pipes already hold, so a
//! caller reading the logs sees every byte the compiler wrote.
use invar_foreign::numeric;
use std::{
    fs,
    io::{Read, Write},
    path::Path,
    process::{Command, Stdio},
    sync::mpsc,
    time::{Duration, Instant},
};

/// Bytes kept per compiler stream, in memory and in its log file.
pub(super) const MAX_LOG: usize = 64 * 1024;
/// How long, after the group kill, the capture workers may take to drain the
/// pipes into the logs. Only a descendant that left the group can hold a pipe
/// open this long; its log is then incomplete and the call still returns.
const LOG_GRACE: Duration = Duration::from_secs(1);

/// The compiler process. `reap` (or dropping it) kills the whole group while
/// the leader is still unreaped, then reaps the leader exactly once.
struct Child(
    std::process::Child,
    Option<std::io::Result<std::process::ExitStatus>>,
);
impl Child {
    fn reap(&mut self) -> &std::io::Result<std::process::ExitStatus> {
        self.1.get_or_insert_with(|| {
            #[cfg(unix)]
            numeric::terminate_compiler_group(self.0.id());
            let _ = self.0.kill();
            self.0.wait()
        })
    }
}
impl Drop for Child {
    fn drop(&mut self) {
        self.reap();
    }
}

/// Read at most `MAX_LOG + 1` bytes, writing the first `MAX_LOG` to `log` as
/// they arrive.
fn capture(mut input: impl Read, mut log: fs::File) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(MAX_LOG + 1)
        .map_err(std::io::Error::other)?;
    let mut chunk = [0; 4096];
    loop {
        let room = MAX_LOG + 1 - bytes.len();
        if room == 0 {
            break;
        }
        let n = input.read(&mut chunk[..room.min(4096)])?;
        if n == 0 {
            break;
        }
        let kept = (MAX_LOG - bytes.len().min(MAX_LOG)).min(n);
        log.write_all(&chunk[..kept])?;
        bytes.extend_from_slice(&chunk[..n]);
    }
    Ok(bytes)
}

/// Run `command` to completion within `deadline`, keeping its streams in
/// `prefix.stdout` and `prefix.stderr`. Returns success and both streams.
pub(super) fn bounded(
    command: &mut Command,
    prefix: &Path,
    deadline: Duration,
) -> Result<(bool, Vec<u8>, Vec<u8>), String> {
    let begun = Instant::now();
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(command, 0);
    let logs = [
        fs::File::create(prefix.with_extension("stdout")).map_err(|e| e.to_string())?,
        fs::File::create(prefix.with_extension("stderr")).map_err(|e| e.to_string())?,
    ];
    let mut child = Child(
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| e.to_string())?,
        None,
    );
    let out = child.0.stdout.take().ok_or("missing compiler stdout")?;
    let err = child.0.stderr.take().ok_or("missing compiler stderr")?;
    let (send, recv) = mpsc::channel();
    let [out_log, err_log] = logs;
    let pipes: [(Box<dyn Read + Send>, fs::File); 2] =
        [(Box::new(out), out_log), (Box::new(err), err_log)];
    for (kind, (pipe, log)) in pipes.into_iter().enumerate() {
        let send = send.clone();
        std::thread::Builder::new()
            .spawn(move || {
                let _ = send.send((kind, capture(pipe, log)));
            })
            .map_err(|e| e.to_string())?;
    }
    let mut streams = [None, None];
    let outcome = watch(&mut child, &recv, &mut streams, begun, deadline);
    // Every path: kill the group while the leader is unreaped, then reap it.
    let status = child
        .reap()
        .as_ref()
        .map(|s| s.success())
        .map_err(|e| e.to_string());
    // The logs must hold what the pipes delivered before the caller reads them.
    let grace = Instant::now();
    while streams.iter().any(Option::is_none) {
        let Some(left) = LOG_GRACE.checked_sub(grace.elapsed()) else {
            break;
        };
        match recv.recv_timeout(left) {
            Ok((kind, bytes)) => streams[kind] = Some(bytes.map_err(|e| e.to_string())),
            Err(_) => break,
        }
    }
    outcome?;
    let success = status?;
    let [Some(stdout), Some(stderr)] = streams else {
        return Err("missing compiler stream".into());
    };
    Ok((success, stdout?, stderr?))
}

type Captured = Result<Vec<u8>, String>;

/// Poll until the compiler has exited and both streams ended, a stream fails or
/// overflows, or the deadline passes. Never reaps the compiler.
fn watch(
    child: &mut Child,
    recv: &mpsc::Receiver<(usize, std::io::Result<Vec<u8>>)>,
    streams: &mut [Option<Captured>; 2],
    begun: Instant,
    deadline: Duration,
) -> Result<(), String> {
    loop {
        while let Ok((kind, bytes)) = recv.try_recv() {
            let stream = streams[kind].insert(bytes.map_err(|e| e.to_string()));
            match stream {
                Err(e) => return Err(e.clone()),
                Ok(bytes) if bytes.len() > MAX_LOG => return Err("compiler output limit".into()),
                Ok(_) => {}
            }
        }
        let exited = numeric::compiler_exited(&mut child.0).map_err(|e| e.to_string())?;
        if exited && streams.iter().all(Option::is_some) {
            return Ok(());
        }
        if begun.elapsed() >= deadline {
            return Err("compiler deadline".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// The deadline path used to return while the capture workers were still
    /// creating or filling the logs, so a caller's read could fail (ENOENT) or
    /// miss bytes. Many concurrent compilers make that window likely: each
    /// prints, marks that the bytes are in the pipe, and leaves a descendant
    /// holding the pipes past a 100 ms deadline.
    #[test]
    fn deadline_logs_are_complete_on_return_under_parallel_stress() {
        const RUNS: usize = 200;
        const PARALLEL: usize = 16;
        const TEXT: &[u8] = b"partial compiler log";
        let root = std::env::temp_dir().join(format!(
            "gnet-compiler-child-stress-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        // Keep all 200 calls, without testing descriptor exhaustion instead
        // of capture completion on hosts with a 1024-descriptor soft limit.
        let mut printed = 0;
        for first in (0..RUNS).step_by(PARALLEL) {
            printed += std::thread::scope(|scope| {
                let runs = (first..(first + PARALLEL).min(RUNS))
                .map(|i| {
                    let dir = root.join(i.to_string());
                    scope.spawn(move || {
                        fs::create_dir(&dir).unwrap();
                        let result = bounded(
                            Command::new("sh")
                                .args([
                                    "-c",
                                    "printf 'partial compiler log' >&2; : > \"$1\"; sleep 10 & exit 0",
                                    "sh",
                                ])
                                .arg(dir.join("printed")),
                            &dir.join("deadline"),
                            Duration::from_millis(100),
                        );
                        let error = result.map(|_| ()).unwrap_err();
                        assert!(error.contains("deadline"), "{i}: {error}");
                        let log = fs::read(dir.join("deadline.stderr"))
                            .unwrap_or_else(|e| panic!("{i}: stderr log: {e}"));
                        let marked = dir.join("printed").exists();
                        if marked {
                            assert_eq!(log, TEXT, "{i}");
                        } else {
                            assert!(TEXT.starts_with(&log), "{i}: {log:?}");
                        }
                        marked
                    })
                })
                .collect::<Vec<_>>();
                runs.into_iter()
                    .map(|run| run.join().unwrap())
                    .filter(|&marked| marked)
                    .count()
            });
        }
        // Not vacuous: some compilers printed before their deadline.
        assert!(printed > 0, "no compiler printed before its deadline");
        fs::remove_dir_all(&root).unwrap();
    }
}
