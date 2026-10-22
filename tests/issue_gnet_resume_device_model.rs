//! CPU execution model of the emitted ABI3 CUDA/Metal per-root continuation
//! kernel `invar_gnet_resume_v3`. The exact emitted device text is compiled with
//! a minimal C++ shim and launched lane by lane in host-fenced quantum epochs.
//! Every launched lane, including rounded-up lanes past the root count, has
//! real backing storage: those tails and a red zone hold sentinels that are
//! checked after every epoch, and the model runs under ASan and UBSan.
//! This checks the device lowering of ordinary-source owned continuations; it
//! is never a GPU, driver, memory-model or performance claim.
#[allow(dead_code)]
#[path = "../test-support/child_deadline.rs"]
mod child_deadline;

use invar::{
    check, eval, front,
    general_net::{self, Net, compiled, oracle},
};
use std::{
    collections::BTreeSet,
    ffi::OsString,
    fmt::Write as _,
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

fn lowered(source: &str) -> Net {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
    invar::ownership::check(&checked).unwrap();
    Net::lower(&checked, general_net::Limits::default()).unwrap()
}
fn scratch(name: &str) -> PathBuf {
    let root = std::env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = root.join(format!("gnet-resume-device-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}
/// Hoist includes out of a namespace-wrapped body; drop Metal's library header.
fn split(source: &str) -> (Vec<&str>, String) {
    let mut includes = Vec::new();
    let mut body = String::new();
    for line in source.lines() {
        if line.starts_with("#include") {
            if !line.contains("metal_stdlib") {
                includes.push(line);
            }
        } else if line.trim() != "using namespace metal;" {
            body.push_str(line);
            body.push('\n');
        }
    }
    (includes, body)
}
fn macros(source: &str) -> Vec<String> {
    source
        .lines()
        .filter_map(|l| l.strip_prefix("#define "))
        .map(|rest| rest.split([' ', '(']).next().unwrap().to_owned())
        .collect()
}
const CUDA_SHIM: &str = r#"
#define __device__
#define __global__
struct Dim { unsigned x = 1, y = 1, z = 1; };
Dim gridDim, blockDim{32, 1, 1}, threadIdx{0, 0, 0}, blockIdx{0, 0, 0};
"#;
const METAL_SHIM: &str = r#"
#define device
#define thread
#define threadgroup
#define constant
#define kernel
using uint = unsigned; using ulong = unsigned long;
"#;
/// Exit codes of the generated model.
const TASK: i32 = 10;
const HEAP: i32 = 13;
const TRACE: i32 = 12;
const TAIL: i32 = 15;
const ASAN: i32 = 86;
const UBSAN: i32 = 87;
/// Wall and CPU budgets for each compiler invocation and model run. The largest
/// sanitized model run takes a few seconds; a hang or runaway is a failure.
const WALL: Duration = Duration::from_secs(180);
const CPU_SECONDS: u64 = 180;

/// Classified sanitizer report, independent of the process exit code.
#[derive(Debug, PartialEq)]
enum Sanitizer {
    None,
    /// AddressSanitizer error kind, e.g. `heap-buffer-overflow` or `SEGV`.
    Address(String),
    /// UndefinedBehaviorSanitizer runtime error message.
    Undefined(String),
}
fn sanitizer(stderr: &str) -> Sanitizer {
    for line in stderr.lines() {
        if let Some(rest) = line.split("ERROR: AddressSanitizer: ").nth(1) {
            return Sanitizer::Address(rest.split(' ').next().unwrap_or("").to_owned());
        }
        if let Some(rest) = line.split("runtime error: ").nth(1) {
            return Sanitizer::Undefined(rest.to_owned());
        }
    }
    Sanitizer::None
}
#[derive(Debug)]
struct Outcome {
    code: Option<i32>,
    signal: Option<i32>,
    timed_out: bool,
    stdout: String,
    stderr: String,
    sanitizer: Sanitizer,
}
impl Outcome {
    fn summary(&self) -> String {
        format!(
            "code={:?} signal={:?} timed_out={} sanitizer={:?} last={:?}",
            self.code,
            self.signal,
            self.timed_out,
            self.sanitizer,
            self.stdout.lines().last().unwrap_or("")
        )
    }
}
/// Run a subprocess with a wall deadline and (on Unix) a CPU-seconds limit.
/// On Unix the child leads its own process group, and expiry kills the whole
/// group, so compiler descendants cannot outlive the deadline. Elsewhere only
/// the wall deadline and a direct kill apply.
fn bounded(
    program: &Path,
    args: &[OsString],
    env: &[(&str, String)],
    wall: Duration,
    cpu_seconds: u64,
) -> Outcome {
    #[cfg(unix)]
    let mut command = {
        use std::os::unix::process::CommandExt;
        let mut c = Command::new("sh");
        c.arg("-c")
            .arg(format!("ulimit -t {cpu_seconds} && exec \"$0\" \"$@\""))
            .arg(program)
            .args(args);
        c.process_group(0);
        c
    };
    #[cfg(not(unix))]
    let mut command = {
        let _ = cpu_seconds;
        let mut c = Command::new(program);
        c.args(args);
        c
    };
    command
        .envs(env.iter().map(|(k, v)| (*k, v.as_str())))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().unwrap();
    let drain = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = pipe.read_to_end(&mut bytes);
            bytes
        })
    };
    let out = drain(Box::new(child.stdout.take().unwrap()));
    let err = drain(Box::new(child.stderr.take().unwrap()));
    let begun = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if begun.elapsed() >= wall {
            timed_out = true;
            #[cfg(unix)]
            invar_foreign::numeric::terminate_compiler_group(child.id());
            let _ = child.kill();
            break child.wait().unwrap();
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let stdout = String::from_utf8_lossy(&out.join().unwrap()).into_owned();
    let stderr = String::from_utf8_lossy(&err.join().unwrap()).into_owned();
    #[cfg(unix)]
    let signal = std::os::unix::process::ExitStatusExt::signal(&status);
    #[cfg(not(unix))]
    let signal = None;
    Outcome {
        code: status.code(),
        signal,
        timed_out,
        sanitizer: sanitizer(&stderr),
        stdout,
        stderr,
    }
}

/// Lane-by-lane model of one fenced epoch per quantum on the device artifact,
/// against the CPU artifact's own `gnet_resume`, compared after every epoch.
fn harness(
    cpu: &compiled::Artifact,
    dev: &compiled::Artifact,
    roots: &[u32],
    l: oracle::Limits,
) -> String {
    let layout = cpu.owned_layout().unwrap();
    let (heap, clean) = (layout.heap_words(), layout.cleanup_words());
    assert_eq!(dev.owned_layout().unwrap().heap_words(), heap);
    assert_eq!(dev.owned_layout().unwrap().cleanup_words(), clean);
    let params = cpu.params().len();
    assert!(params <= 1);
    let (cpu_includes, cpu_body) = split(cpu.source());
    let (dev_includes, dev_body) = split(dev.source());
    let mut text = String::new();
    for line in cpu_includes.iter().chain(&dev_includes) {
        text.push_str(line);
        text.push('\n');
    }
    text.push_str("#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <vector>\n");
    write!(text, "namespace cpu {{\n{cpu_body}\n}}\n").unwrap();
    for name in macros(cpu.source()) {
        writeln!(text, "#undef {name}").unwrap();
    }
    let metal = dev.target() == compiled::Target::Metal;
    text.push_str(if metal { METAL_SHIM } else { CUDA_SHIM });
    write!(text, "namespace dev {{\n{dev_body}\n}}\n").unwrap();
    if metal {
        text.push_str(
            "#undef device\n#undef thread\n#undef threadgroup\n#undef constant\n#undef kernel\n",
        );
    }
    let inputs = roots
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let launch = if metal {
        "for (unsigned lane = 0; lane < LANES; ++lane) dev::invar_gnet_resume_v3(din.data(), dout.data(), dtrace.data(), config, dheap.data(), dclean.data(), dtask.data(), lane);"
    } else {
        "gridDim.x = LANES / 32; for (unsigned b = 0; b < gridDim.x; ++b) for (unsigned t = 0; t < 32; ++t) { blockIdx.x = b; threadIdx.x = t; dev::invar_gnet_resume_v3(din.data(), dout.data(), dtrace.data(), config, dheap.data(), dclean.data(), dtask.data()); }"
    };
    write!(
        text,
        r#"
static_assert(sizeof(cpu::GnetTask) == sizeof(dev::GnetTask), "task layout");
// Every launched lane owns real storage. Lanes past R start from a fresh zero
// task and zero input; their output/trace/heap/cleanup slices and a red zone
// beyond the last lane hold a sentinel. Any write there is a lane-guard,
// stride or offset fault and is reported before the per-root comparison.
int main(int argc, char** argv) {{
    if (argc != 2) return 2;
    const unsigned R = {r}, P = {params}, PP = P ? P : 1, LANES = ((R + 31) / 32) * 32, RED = 64;
    const unsigned long long FUEL = {fuel}, TRANSITIONS = {transitions}, CAP = {cap}, QUANTUM = std::strtoull(argv[1], nullptr, 10);
    const unsigned long long H = {heap}, C = {clean}, T = CAP * 7;
    const unsigned long long S = 0x5a5a5a5a5a5a5a5aULL;
    const long long roots[] = {{{inputs}}};
    std::vector<cpu::gnet_w> cin(R * PP), cout(R * 10), cheap(R * H);
    std::vector<cpu::gnet_u> ctrace(R * T + 1), cclean(R * C + 1);
    std::vector<cpu::GnetTask> ctask(R);
    std::vector<dev::gnet_w> din(LANES * PP + RED, 0), dout(LANES * 10 + RED, (dev::gnet_w)S), dheap(LANES * H + RED, (dev::gnet_w)S);
    std::vector<dev::gnet_u> dtrace(LANES * T + RED, S), dclean(LANES * C + RED, S);
    std::vector<dev::GnetTask> dtask(LANES + 1);
    for (unsigned long long i = 0; i < R * 10; ++i) dout[i] = 0;
    for (unsigned long long i = 0; i < R * H; ++i) dheap[i] = 0;
    for (unsigned long long i = 0; i < R * T; ++i) dtrace[i] = 0;
    for (unsigned long long i = 0; i < R * C; ++i) dclean[i] = 0;
    for (unsigned i = 0; i < R && P; ++i) {{ cin[i] = roots[i]; din[i] = roots[i]; }}
    for (unsigned long long i = LANES * PP; i < din.size(); ++i) din[i] = (dev::gnet_w)S;
    const std::vector<dev::gnet_w> din0 = din;
    std::vector<unsigned char> zero_task(sizeof(dev::GnetTask) * (LANES + 1 - R), 0);
    dev::gnet_u config[5] = {{R, FUEL, TRANSITIONS, CAP, QUANTUM}};
    auto tails = [&]() -> const char* {{
        if (din != din0) return "input";
        if (std::memcmp(dtask.data() + R, zero_task.data(), zero_task.size())) return "task";
        for (unsigned long long i = R * 10; i < dout.size(); ++i) if ((unsigned long long)dout[i] != S) return "output";
        for (unsigned long long i = R * H; i < dheap.size(); ++i) if ((unsigned long long)dheap[i] != S) return "heap";
        for (unsigned long long i = R * T; i < dtrace.size(); ++i) if (dtrace[i] != S) return "trace";
        for (unsigned long long i = R * C; i < dclean.size(); ++i) if (dclean[i] != S) return "cleanup";
        return nullptr;
    }};
    unsigned epochs = 0;
    for (;;) {{
        unsigned pending = 0;
        for (unsigned i = 0; i < R; ++i) pending += dtask[i].done ? 0 : 1;
        std::printf("P %u\n", pending);
        if (!pending) break;
        if (++epochs > 200000) return 3;
        {launch}
        if (const char* where = tails()) {{ std::printf("tail %s write epoch %u\n", where, epochs); return 15; }}
        for (unsigned i = 0; i < R; ++i)
            if (!ctask[i].done)
                cpu::gnet_resume(cin.data() + i * P, cout.data() + i * 10, ctrace.data() + i * T, FUEL, TRANSITIONS, CAP, cheap.data() + i * H, cclean.data() + i * C, &ctask[i], QUANTUM);
        if (std::memcmp(ctask.data(), dtask.data(), R * sizeof(cpu::GnetTask))) {{ std::printf("task mismatch epoch %u\n", epochs); return 10; }}
        if (std::memcmp(cout.data(), dout.data(), R * 10 * 8)) {{ std::printf("output mismatch epoch %u\n", epochs); return 11; }}
        if (std::memcmp(ctrace.data(), dtrace.data(), R * T * 8)) {{ std::printf("trace mismatch epoch %u\n", epochs); return 12; }}
        if (std::memcmp(cheap.data(), dheap.data(), R * H * 8)) {{ std::printf("heap mismatch epoch %u\n", epochs); return 13; }}
        if (std::memcmp(cclean.data(), dclean.data(), R * C * 8)) {{ std::printf("cleanup mismatch epoch %u\n", epochs); return 14; }}
    }}
    std::printf("E %u\n", epochs);
    for (unsigned i = 0; i < R; ++i) {{
        std::printf("O %u", i); for (unsigned j = 0; j < 10; ++j) std::printf(" %lld", (long long)dout[i * 10 + j]); std::printf("\n");
        unsigned long long used = (unsigned long long)dout[i * 10 + 7];
        std::printf("T %u", i); for (unsigned long long j = 0; j < used * 7; ++j) std::printf(" %llu", (unsigned long long)dtrace[i * T + j]); std::printf("\n");
    }}
    return 0;
}}
"#,
        r = roots.len(),
        fuel = l.fuel,
        transitions = l.transitions,
        cap = l.trace_events,
    )
    .unwrap();
    text
}

struct Run {
    epochs: u64,
    pending: Vec<u64>,
    outputs: Vec<Vec<i64>>,
    traces: Vec<Vec<u64>>,
}
/// A generated model compiled once under ASan+UBSan; runs take the quantum.
struct Model {
    dir: PathBuf,
    exe: PathBuf,
}
impl Drop for Model {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}
fn build(name: &str, text: &str) -> Model {
    let dir = scratch(name);
    let cpp = dir.join("model.cpp");
    let exe = dir.join("model");
    fs::write(&cpp, text).unwrap();
    let compiler = PathBuf::from(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()));
    let args: Vec<OsString> = [
        "-std=c++17",
        "-O1",
        "-g",
        "-fno-omit-frame-pointer",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all",
    ]
    .iter()
    .map(OsString::from)
    .chain([
        cpp.into_os_string(),
        "-o".into(),
        exe.clone().into_os_string(),
    ])
    .collect();
    let o = bounded(
        &compiler,
        &args,
        &[],
        child_deadline::wall(WALL),
        CPU_SECONDS,
    );
    assert!(
        o.code == Some(0) && !o.timed_out,
        "{name}: compiler {} {}",
        o.summary(),
        o.stderr
    );
    Model { dir, exe }
}
impl Model {
    /// Bounded sanitized run with the quantum as its only argument.
    fn run(&self, quantum: u64) -> Outcome {
        bounded(
            &self.exe,
            &[quantum.to_string().into()],
            &[
                ("ASAN_OPTIONS", format!("detect_leaks=0:exitcode={ASAN}")),
                (
                    "UBSAN_OPTIONS",
                    format!("halt_on_error=1:print_stacktrace=1:exitcode={UBSAN}"),
                ),
            ],
            child_deadline::wall(WALL),
            CPU_SECONDS,
        )
    }
    fn execute(&self, label: &str, quantum: u64) -> Run {
        let o = self.run(quantum);
        assert!(
            o.code == Some(0) && !o.timed_out && o.sanitizer == Sanitizer::None,
            "{label}: {} {}",
            o.summary(),
            o.stderr
        );
        let stdout = o.stdout;
        let mut run = Run {
            epochs: 0,
            pending: Vec::new(),
            outputs: Vec::new(),
            traces: Vec::new(),
        };
        for line in stdout.lines() {
            let mut words = line.split(' ');
            match words.next() {
                Some("P") => run.pending.push(words.next().unwrap().parse().unwrap()),
                Some("E") => run.epochs = words.next().unwrap().parse().unwrap(),
                Some("O") => run
                    .outputs
                    .push(words.skip(1).map(|w| w.parse().unwrap()).collect()),
                Some("T") => run
                    .traces
                    .push(words.skip(1).map(|w| w.parse().unwrap()).collect()),
                _ => panic!("unexpected model output {line}"),
            }
        }
        run
    }
}

/// Independent net-oracle header and trace for one root, in ABI3 word order.
fn expected(
    net: &Net,
    entry: &str,
    args: &[eval::Value],
    l: oracle::Limits,
) -> (Vec<i64>, Vec<u64>) {
    let o = oracle::observe_call(net, &net.verify().unwrap(), entry, args, l).unwrap();
    let (status, result, start, end) = match &o.semantic.result {
        Ok(eval::Value::I32(n)) => (0, i64::from(*n), 0, 0),
        Err(eval::EvalError::Trap { category, span }) => {
            let status = match category {
                eval::TrapCategory::IntegerOverflow => 1,
                eval::TrapCategory::DivisionByZero => 2,
                eval::TrapCategory::IndexOutOfBounds => 9,
                other => panic!("{other:?}"),
            };
            (status, 0, span.start as i64, span.end as i64)
        }
        Err(eval::EvalError::EvaluationLimit { .. }) => (3, 0, 0, 0),
        Err(eval::EvalError::Tool { code, .. }) => match (o.failure, *code) {
            (Some(oracle::ToolFailure::TransitionLimit), _) => (5, 0, 0, 0),
            (Some(oracle::ToolFailure::TraceLimit), _) => (6, 0, 0, 0),
            (None, "CALL_DEPTH_LIMIT") => (4, 0, 0, 0),
            (None, "ARRAY_SIZE_LIMIT") => (10, 0, 0, 0),
            other => panic!("{other:?}"),
        },
        x => panic!("{x:?}"),
    };
    let header = vec![
        status,
        result,
        start,
        end,
        o.semantic.steps_used as i64,
        o.transitions as i64,
        o.max_frames as i64,
        o.events.len() as i64,
        o.semantic.cleanup_trace.len() as i64,
    ];
    let trace = o
        .events
        .iter()
        .flat_map(|e| {
            [
                u64::from(e.kind == oracle::EventKind::Complete),
                e.frame,
                e.activation,
                e.token.block.function as u64,
                e.token.block.block as u64,
                e.token.position as u64,
                e.rule as u64,
            ]
        })
        .collect();
    (header, trace)
}

struct Case {
    name: &'static str,
    source: &'static str,
    entry: &'static str,
    roots: &'static [u32],
    trace_events: usize,
    transitions: u64,
    storage: compiled::OwnedLimits,
    /// Roots whose backend storage limit (status 11-13) has no oracle analogue;
    /// they are checked word-for-word against the CPU artifact every epoch.
    backend: &'static [(u32, i64)],
}
impl Case {
    fn limits(&self) -> oracle::Limits {
        oracle::Limits {
            fuel: 1000,
            transitions: self.transitions,
            trace_events: self.trace_events,
            ..oracle::Limits::default()
        }
    }
}
const STORAGE: compiled::OwnedLimits = compiled::OwnedLimits {
    elements: 64,
    allocations: 16,
    cleanup_events: 32,
};
const CASES: &[Case] = &[
    Case {
        name: "fib",
        source: "fn fib(n: u32) -> i32 { if n < 2u32 { return 1; } return fib(n - 1u32) + fib(n - 2u32); } fn main() -> i32 { return fib(5u32); }",
        entry: "fib",
        roots: &[0, 1, 2, 3, 5, 7, 9, 4, 6, 8],
        trace_events: 2000,
        transitions: 10_000,
        storage: STORAGE,
        backend: &[],
    },
    Case {
        name: "loans",
        source: "fn mutate(xs: mut Array<i32>, n: u32) -> Unit { for i in 0u32..n { let tmp = filled_i32(2u32, 3); xs[0u32] = xs[0u32] + tmp[1u32]; } } fn run(n: u32) -> i32 { var xs: Array<i32> = filled_i32(n + 1u32, 1); mutate(xs, n); return xs[0u32]; } fn main() -> i32 { return run(2u32); }",
        entry: "run",
        roots: &[0, 1, 2, 5, 3, 8],
        trace_events: 2000,
        transitions: 10_000,
        storage: STORAGE,
        backend: &[],
    },
    Case {
        name: "moves",
        source: "fn consume(xs: take Array<i32>, n: u32) -> i32 { if n == 0u32 { return xs[0u32]; } return consume(move xs, n - 1u32); } fn run(n: u32) -> i32 { let xs: Array<i32> = [7]; return consume(move xs, n); } fn main() -> i32 { return run(3u32); }",
        entry: "run",
        roots: &[0, 3, 1, 6, 2],
        trace_events: 2000,
        transitions: 10_000,
        storage: STORAGE,
        backend: &[],
    },
    Case {
        name: "mixed",
        source: "fn run(n: u32) -> i32 { if n == 0u32 { while true {} } if n == 1u32 { return 17; } if n == 4u32 { return 1 / 0; } let xs: Array<i32> = [1]; return xs[n]; } fn main() -> i32 { return run(1u32); }",
        entry: "run",
        roots: &[0, 1, 2, 4, 3, 1],
        trace_events: 10_000,
        transitions: 10_000,
        storage: STORAGE,
        backend: &[],
    },
    Case {
        name: "limits",
        source: "fn deep(n: u32) -> i32 { return deep(n + 1u32); } fn run(n: u32) -> i32 { if n == 0u32 { let x: i32 = 2147483647; return x + 1; } if n == 1u32 { return deep(0u32); } if n == 2u32 { let big: Array<i32> = filled_i32(16777217u32, 0); return big[0u32]; } return 5; } fn main() -> i32 { return run(3u32); }",
        entry: "run",
        roots: &[0, 1, 2, 3],
        trace_events: 2000,
        transitions: 10_000,
        storage: STORAGE,
        backend: &[],
    },
    Case {
        name: "transitions",
        source: "fn fib(n: u32) -> i32 { if n < 2u32 { return 1; } return fib(n - 1u32) + fib(n - 2u32); } fn main() -> i32 { return fib(5u32); }",
        entry: "fib",
        roots: &[1, 6, 9],
        trace_events: 10_000,
        transitions: 150,
        storage: STORAGE,
        backend: &[],
    },
    Case {
        name: "storage",
        source: "fn run(n: u32) -> i32 { if n == 0u32 { let wide: Array<i32> = filled_i32(9u32, 0); return wide[0u32]; } if n == 1u32 { for i in 0u32..3u32 { let t: Array<i32> = filled_i32(1u32, 0); } return 1; } if n == 2u32 { let a: Array<i32> = [1]; let b: Array<i32> = [2]; let c: Array<i32> = [3]; let d: Array<i32> = [4]; let e: Array<i32> = [5]; return 1; } return 2; } fn main() -> i32 { return run(3u32); }",
        entry: "run",
        roots: &[0, 1, 2, 3],
        trace_events: 2000,
        transitions: 10_000,
        storage: compiled::OwnedLimits {
            elements: 8,
            allocations: 4,
            cleanup_events: 2,
        },
        backend: &[(0, 11), (1, 13), (2, 12)],
    },
    Case {
        // More than 32 roots: two CUDA blocks, the second partly past R, so the
        // block offset in the device lane index is exercised.
        name: "blocks",
        source: "fn fib(n: u32) -> i32 { if n < 2u32 { return 1; } return fib(n - 1u32) + fib(n - 2u32); } fn main() -> i32 { return fib(5u32); }",
        entry: "fib",
        roots: &[
            0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0,
            1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 6,
        ],
        trace_events: 2000,
        transitions: 10_000,
        storage: STORAGE,
        backend: &[],
    },
];
fn emit(net: &Net, case: &Case, target: compiled::Target) -> compiled::Artifact {
    let proof = net.verify().unwrap();
    compiled::emit_resumable(
        net,
        &proof,
        case.entry,
        target,
        compiled::Limits::default(),
        case.storage,
    )
    .unwrap()
}

#[test]
fn issue_gnet_resume_device_model_matches_cpu_every_epoch_and_oracle() {
    let mut observed = Vec::new();
    let mut statuses = BTreeSet::new();
    for case in CASES {
        let l = case.limits();
        let net = lowered(case.source);
        let cpu = emit(&net, case, compiled::Target::Cpu);
        let expected: Vec<_> = case
            .roots
            .iter()
            .map(|n| {
                if case.backend.iter().any(|(root, _)| root == n) {
                    None
                } else {
                    Some(expected(&net, case.entry, &[eval::Value::U32(*n)], l))
                }
            })
            .collect();
        for target in [compiled::Target::Cuda, compiled::Target::Metal] {
            let dev = emit(&net, case, target);
            assert_eq!(dev.target(), target);
            assert!(dev.source().contains("invar_gnet_resume_v3"));
            let model = build(
                &format!("{}-{target:?}", case.name),
                &harness(&cpu, &dev, case.roots, l),
            );
            for quantum in [1, 3, 17] {
                let label = format!("{}-{target:?}-{quantum}", case.name);
                let run = model.execute(&label, quantum);
                assert_eq!(run.outputs.len(), case.roots.len(), "{label}");
                for (lane, root) in case.roots.iter().enumerate() {
                    let output = &run.outputs[lane];
                    match &expected[lane] {
                        Some((header, trace)) => {
                            assert_eq!(&output[..9], &header[..], "{label} root {lane}");
                            assert_eq!(&run.traces[lane], trace, "{label} root {lane}");
                        }
                        None => {
                            let (_, status) = case.backend.iter().find(|(r, _)| r == root).unwrap();
                            assert_eq!(output[0], *status, "{label} root {lane}");
                        }
                    }
                    statuses.insert(output[0]);
                }
                // Epochs advance every unfinished root by one quantum: pending
                // roots never increase, and the final epoch leaves none.
                assert!(run.pending.windows(2).all(|w| w[1] <= w[0]), "{label}");
                assert_eq!(run.pending.first(), Some(&(case.roots.len() as u64)));
                assert_eq!(run.pending.last(), Some(&0));
                assert_eq!(run.pending.len() as u64, run.epochs + 1);
                observed.push((label, run.epochs));
            }
        }
    }
    // Success; overflow, division and index traps; fuel, call-depth,
    // transition, trace and source array-size limits; backend element,
    // allocation-slot and cleanup storage limits. Status 7 (invalid internal
    // state) has no ordinary-source producer; status 8 has its own test.
    assert_eq!(statuses, [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13].into());
    eprintln!("ABI3 device-model runs (label, epochs): {observed:?}");
}

#[test]
fn issue_gnet_resume_device_model_invalid_quantum_finishes_with_status_8() {
    let case = &CASES[0];
    let net = lowered(case.source);
    let cpu = emit(&net, case, compiled::Target::Cpu);
    for target in [compiled::Target::Cuda, compiled::Target::Metal] {
        let dev = emit(&net, case, target);
        let model = build(
            &format!("status8-{target:?}"),
            &harness(&cpu, &dev, case.roots, case.limits()),
        );
        for quantum in [0, 1_000_001] {
            let label = format!("status8-{target:?}-{quantum}");
            let run = model.execute(&label, quantum);
            assert_eq!(run.epochs, 1, "{label}");
            assert_eq!(run.pending, [case.roots.len() as u64, 0], "{label}");
            assert!(
                run.outputs.iter().all(|o| o[0] == 8 && o[7] == 0),
                "{label}"
            );
        }
    }
}

/// Apply a text substitution to the device namespace only; the CPU reference
/// side of the generated model is left untouched.
fn mutate_device(text: &str, from: &str, to: &str) -> String {
    let start = text.find("namespace dev {").unwrap();
    let (head, tail) = text.split_at(start);
    assert!(tail.contains(from), "missing device text {from}");
    format!("{head}{}", tail.replacen(from, to, 1))
}

#[test]
fn issue_gnet_resume_device_model_detects_device_lowering_mutations() {
    // Each mutation must fail deterministically with its specific exit code.
    let mutations: [(&str, usize, &str, &str, i32); 7] = [
        (
            "late-yield",
            0,
            "if (slice_steps==quantum) { GNET_YIELD; }",
            "if (slice_steps==quantum+1) { GNET_YIELD; }",
            TASK,
        ),
        ("no-lane-guard", 0, "if (lane>=config[0]) return;", "", TAIL),
        (
            "swapped-config",
            0,
            "config[1],config[2],config[3]",
            "config[2],config[1],config[3]",
            TASK,
        ),
        (
            "dropped-trace-offset",
            0,
            "(config[3]?trace+lane*config[3]*7:trace)",
            "trace",
            TRACE,
        ),
        // Filled below with case-specific heap strides. The far stride writes
        // just past the device heap allocation, so only ASan can (and must) stop it.
        ("short-heap-stride", 1, "", "", HEAP),
        ("far-heap-stride", 0, "", "", ASAN),
        // CUDA only: Metal takes the lane from thread_position_in_grid.
        (
            "no-block-offset",
            7,
            "(gnet_u)blockIdx.x * blockDim.x + threadIdx.x",
            "(gnet_u)threadIdx.x",
            TASK,
        ),
    ];
    for (name, case_index, from, to, code) in mutations {
        let case = &CASES[case_index];
        let net = lowered(case.source);
        let cpu = emit(&net, case, compiled::Target::Cpu);
        let heap = cpu.owned_layout().unwrap().heap_words();
        let (from, to) = match name {
            "short-heap-stride" => (
                format!("heap+lane*{heap}"),
                format!("heap+lane*{}", heap - 4),
            ),
            // Lane 1's first heap write (the live flag at offset 3 on task
            // start) lands exactly one element past the device heap vector of
            // LANES*H words plus the 64-word red zone.
            "far-heap-stride" => (
                format!("heap+lane*{heap}"),
                format!(
                    "heap+lane*{}",
                    case.roots.len().div_ceil(32) * 32 * heap + 64 - 3
                ),
            ),
            _ => (from.to_owned(), to.to_owned()),
        };
        let targets: &[_] = if name == "no-block-offset" {
            &[compiled::Target::Cuda]
        } else {
            &[compiled::Target::Cuda, compiled::Target::Metal]
        };
        for &target in targets {
            let dev = emit(&net, case, target);
            let text = harness(&cpu, &dev, case.roots, case.limits());
            let o = build(
                &format!("{name}-{target:?}"),
                &mutate_device(&text, &from, &to),
            )
            .run(3);
            assert!(!o.timed_out, "{name} {target:?}: {}", o.summary());
            assert_eq!(o.code, Some(code), "{name} {target:?}: {}", o.summary());
            // A sanitizer exit must carry the expected classified report; a
            // comparison exit must carry none.
            let want = if code == ASAN {
                Sanitizer::Address("heap-buffer-overflow".into())
            } else {
                Sanitizer::None
            };
            assert_eq!(o.sanitizer, want, "{name} {target:?}: {}", o.summary());
            eprintln!(
                "mutation {name} {target:?}: exit {code}, sanitizer {:?}, {}",
                o.sanitizer,
                o.stdout.lines().last().unwrap_or("")
            );
        }
    }
}

#[test]
fn issue_gnet_resume_device_model_numeric_owners_have_no_device_route() {
    // Forced device coverage must not silently fall back to the CPU ABI11 path.
    let net = lowered(
        "fn run(x: f32) -> Array<f32> { let a: Array<f32> = [x, -0.0]; return a; } fn main() -> i32 { return 0; }",
    );
    let proof = net.verify().unwrap();
    assert!(matches!(
        compiled::emit_resumable_owned_numeric(
            &net,
            &proof,
            "run",
            compiled::Target::Metal,
            compiled::Limits::default(),
            STORAGE
        ),
        Err(compiled::Error::NumericTarget)
    ));
    // Direct CUDA emission is the distinct ABI12 profile: the ABI3 device model
    // here never executes it, provenance admits it only as a compile receipt
    // without execution authority and no ABI12 runtime path exists
    // (issue_gnet_12 tests), so no silent fallback.
    assert_eq!(
        compiled::emit_resumable_owned_numeric(
            &net,
            &proof,
            "run",
            compiled::Target::Cuda,
            compiled::Limits::default(),
            STORAGE
        )
        .unwrap()
        .abi_version(),
        12
    );
    assert!(
        compiled::emit_resumable_owned_numeric(
            &net,
            &proof,
            "run",
            compiled::Target::Cpu,
            compiled::Limits::default(),
            STORAGE
        )
        .is_ok()
    );
}

#[cfg(unix)]
#[test]
fn issue_gnet_resume_device_model_bounded_runner_limits_and_classifies() {
    // Wall deadline: the whole process group, including a backgrounded
    // descendant, is killed and the outcome is reported as timed out.
    let sh = Path::new("sh");
    let o = bounded(
        sh,
        &["-c".into(), "sleep 30 & echo $!; wait".into()],
        &[],
        Duration::from_millis(500),
        60,
    );
    assert!(
        o.timed_out && o.code.is_none() && o.signal == Some(9),
        "{o:?}"
    );
    let descendant = o.stdout.trim().to_owned();
    assert!(!descendant.is_empty(), "{o:?}");
    #[cfg(target_os = "linux")]
    {
        let stat = PathBuf::from(format!("/proc/{descendant}/stat"));
        let begun = Instant::now();
        loop {
            match fs::read_to_string(&stat) {
                Err(_) => break,
                Ok(s) if s.rsplit(") ").next().is_some_and(|t| t.starts_with('Z')) => break,
                Ok(_) => {
                    assert!(
                        begun.elapsed() < child_deadline::wall(Duration::from_secs(5)),
                        "descendant survived"
                    );
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }
    // CPU limit: a busy loop is stopped by SIGXCPU/SIGKILL long before the
    // generous wall deadline.
    let o = bounded(
        sh,
        &["-c".into(), "while :; do :; done".into()],
        &[],
        child_deadline::wall(WALL),
        1,
    );
    assert!(!o.timed_out && o.code.is_none(), "{o:?}");
    assert!(matches!(o.signal, Some(24) | Some(9)), "{o:?}");
    // Report classification is independent of exit codes.
    assert_eq!(
        sanitizer("==1==ERROR: AddressSanitizer: SEGV on unknown address 0x0"),
        Sanitizer::Address("SEGV".into())
    );
    assert_eq!(
        sanitizer("model.cpp:3:5: runtime error: signed integer overflow"),
        Sanitizer::Undefined("signed integer overflow".into())
    );
    assert_eq!(sanitizer("task mismatch epoch 1"), Sanitizer::None);
    // An actual UBSan abort is classified as undefined behaviour with its own
    // exit code, distinct from ASan's.
    let model = build(
        "ubsan-probe",
        "#include <climits>\n#include <cstdlib>\nint main(int argc, char** argv) { int x = std::atoi(argv[1]) + INT_MAX; return x == 7; }\n",
    );
    let o = model.run(1);
    assert_eq!(o.code, Some(UBSAN), "{}", o.summary());
    assert!(
        matches!(&o.sanitizer, Sanitizer::Undefined(m) if m.contains("overflow")),
        "{}",
        o.summary()
    );
}
