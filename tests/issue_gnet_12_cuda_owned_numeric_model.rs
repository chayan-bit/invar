//! CPU execution model of emitted ABI12 (`cuda-owned-numeric-continuation-v1`)
//! CUDA text. The exact emitted kernel `invar_gnet_owned_numeric_resume_v1` is
//! compiled with a minimal C++ shim and launched over every lane of full
//! 32-thread blocks in host-fenced quantum epochs. After every epoch each lane's
//! task bytes, output, trace, heap and cleanup must equal the ABI11 CPU
//! artifact's own `gnet_resume`; tails and a red zone must keep sentinels.
//! Terminal values are checked against the independent net oracle. This is
//! never NVRTC compilation, device execution, host support or a performance
//! claim; ABI12 has no runtime loader in stage 1.
#![cfg(unix)]
#[allow(dead_code)]
#[path = "../test-support/child_deadline.rs"]
mod child_deadline;

use invar::{
    check, eval, front,
    general_net::{self, Net, compiled, oracle},
    numeric_contract::F32Bits,
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

const KERNEL: &str = "invar_gnet_owned_numeric_resume_v1";
const WALL: Duration = Duration::from_secs(180);
const TASK: i32 = 10;
const HEAP: i32 = 13;
const TAIL: i32 = 15;
const ASAN: i32 = 86;
const UBSAN: i32 = 87;

fn lowered(source: &str) -> Net {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
    invar::ownership::check(&checked).unwrap();
    Net::lower(&checked, general_net::Limits::default()).unwrap()
}

#[derive(Debug)]
struct Outcome {
    code: Option<i32>,
    timed_out: bool,
    stdout: String,
    stderr: String,
}
impl Outcome {
    fn sanitizer(&self) -> Option<&str> {
        if self.stderr.contains("ERROR: AddressSanitizer") {
            Some("asan")
        } else if self.stderr.contains("runtime error:") {
            Some("ubsan")
        } else {
            None
        }
    }
    fn last(&self) -> &str {
        self.stdout.lines().last().unwrap_or("")
    }
}
/// Own process group, wall deadline kills the group, CPU limit via `sh`.
fn bounded(program: &Path, args: &[OsString], env: &[(&str, String)]) -> Outcome {
    let wall = child_deadline::wall(WALL);
    use std::os::unix::process::CommandExt;
    let mut command = Command::new("sh");
    command
        .arg("-c")
        .arg("ulimit -t 180 && exec \"$0\" \"$@\"")
        .arg(program)
        .args(args)
        .envs(env.iter().map(|(k, v)| (*k, v.as_str())))
        .process_group(0)
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
            invar_foreign::numeric::terminate_compiler_group(child.id());
            let _ = child.kill();
            break child.wait().unwrap();
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    Outcome {
        code: status.code(),
        timed_out,
        stdout: String::from_utf8_lossy(&out.join().unwrap()).into_owned(),
        stderr: String::from_utf8_lossy(&err.join().unwrap()).into_owned(),
    }
}

fn split(source: &str) -> (Vec<&str>, String) {
    let mut includes = Vec::new();
    let mut body = String::new();
    for line in source.lines() {
        if line.starts_with("#include") {
            includes.push(line);
        } else {
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

/// Lane-by-lane epoch model of the ABI12 CUDA text against the ABI11 CPU text.
fn harness(
    cpu: &compiled::Artifact,
    dev: &compiled::Artifact,
    roots: &[Vec<u32>],
    l: oracle::Limits,
) -> String {
    assert_eq!(cpu.abi_version(), 11);
    assert_eq!(dev.abi_version(), 12);
    let layout = cpu.owned_layout().unwrap();
    let (heap, clean) = (layout.heap_words(), layout.cleanup_words());
    assert_eq!(dev.owned_layout().unwrap().heap_words(), heap);
    assert_eq!(dev.owned_layout().unwrap().cleanup_words(), clean);
    let params = cpu.params().len();
    assert!(roots.iter().all(|r| r.len() == params));
    let (cpu_includes, cpu_body) = split(cpu.source());
    let (dev_includes, dev_body) = split(dev.source());
    assert!(dev_includes.is_empty(), "ABI12 text must be header-free");
    let mut text = String::new();
    for line in &cpu_includes {
        text.push_str(line);
        text.push('\n');
    }
    text.push_str("#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <vector>\n");
    write!(text, "namespace cpu {{\n{cpu_body}\n}}\n").unwrap();
    for name in macros(cpu.source()) {
        writeln!(text, "#undef {name}").unwrap();
    }
    text.push_str(CUDA_SHIM);
    write!(text, "namespace dev {{\n{dev_body}\n}}\n").unwrap();
    let inputs = roots
        .iter()
        .flatten()
        .map(|w| w.to_string())
        .collect::<Vec<_>>()
        .join(",");
    write!(
        text,
        r#"
static_assert(sizeof(cpu::GnetTask) == sizeof(dev::GnetTask), "ABI12 task bytes equal ABI11");
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
    for (unsigned long long i = 0; i < (unsigned long long)R * P; ++i) {{ cin[i] = roots[i]; din[i] = roots[i]; }}
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
        gridDim.x = LANES / 32;
        for (unsigned b = 0; b < gridDim.x; ++b) for (unsigned t = 0; t < 32; ++t) {{ blockIdx.x = b; threadIdx.x = t; dev::{KERNEL}(din.data(), dout.data(), dtrace.data(), config, dheap.data(), dclean.data(), dtask.data()); }}
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
    let root = std::env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = root.join(format!("gnet12-model-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
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
    let o = bounded(&compiler, &args, &[]);
    assert!(
        o.code == Some(0) && !o.timed_out,
        "{name}: compiler {:?} {}",
        o.code,
        o.stderr
    );
    Model { dir, exe }
}
struct Run {
    epochs: u64,
    pending: Vec<u64>,
    outputs: Vec<Vec<i64>>,
    traces: Vec<Vec<u64>>,
}
impl Model {
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
        )
    }
    fn execute(&self, label: &str, quantum: u64) -> Run {
        let o = self.run(quantum);
        assert!(
            o.code == Some(0) && !o.timed_out && o.sanitizer().is_none(),
            "{label}: {:?} {} {}",
            o.code,
            o.last(),
            o.stderr
        );
        let mut run = Run {
            epochs: 0,
            pending: Vec::new(),
            outputs: Vec::new(),
            traces: Vec::new(),
        };
        for line in o.stdout.lines() {
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

/// Independent oracle header (ABI10 word order) and trace for one root.
fn expected(
    net: &Net,
    entry: &str,
    args: &[eval::Value],
    l: oracle::Limits,
) -> (Vec<i64>, Vec<u64>) {
    let o = oracle::observe_call(net, &net.verify().unwrap(), entry, args, l).unwrap();
    let (status, result, start, end) = match &o.semantic.result {
        Ok(eval::Value::F32(x)) => (0, i64::from(x.to_bits()), 0, 0),
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

const SPECIAL: [u32; 20] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x0080_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f80_0001,
    0x4b80_0000,
    0xcb80_0000,
    0x7f7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0x7fa1_2345,
    0xffc0_0001,
    0x3400_0000,
    0x3eaa_aaab,
    0x4040_0000,
];

struct Case {
    name: &'static str,
    source: &'static str,
    entry: &'static str,
    roots: Vec<Vec<u32>>,
    trace_events: usize,
    transitions: u64,
    storage: compiled::OwnedLimits,
    /// Roots (by first word) whose backend storage status has no oracle analogue.
    backend: &'static [(u32, i64)],
}
const STORAGE: compiled::OwnedLimits = compiled::OwnedLimits {
    elements: 64,
    allocations: 16,
    cleanup_events: 32,
};
const SPECIAL_SOURCE: &str = "fn mix(xs: mut Array<f32>, n: u32) -> Unit { if n > 0u32 { xs[0u32] = fma(xs[0u32], xs[1u32], xs[2u32]); mix(xs, n - 1u32); } } fn run(a: f32, b: f32, c: f32, n: u32) -> f32 { var xs: Array<f32> = [a, b, c]; mix(xs, n % 4u32); let t: f32 = xs[0u32]; if t < b { return t / b; } if t == t { return t - c + f32_from_u32(n) - f32_from_i32(0 - 3); } return xs[n]; } fn main() -> i32 { return 0; }";
const LIMITS_SOURCE: &str = "fn deep(x: f32) -> f32 { return deep(x + 1.0); } fn run(a: f32, n: u32) -> f32 { if n == 0u32 { return deep(a); } if n == 1u32 { let big: Array<f32> = filled_f32(16777217u32, a); return big[0u32]; } if n == 2u32 { var k: i32 = 2147483647; k = k + 1; return a; } if n == 3u32 { let z: u32 = n - 3u32; let q: u32 = 7u32 / z; return a; } if n == 4u32 { while true {} } var s: f32 = a; for i in 0u32..n { s = fma(s, a, a); } return s; } fn main() -> i32 { return 0; }";
const STORAGE_SOURCE: &str = "fn run(a: f32, n: u32) -> f32 { if n == 0u32 { let wide: Array<f32> = filled_f32(9u32, a); return wide[0u32]; } if n == 1u32 { for i in 0u32..3u32 { let t: Array<f32> = filled_f32(1u32, a); } return a; } if n == 2u32 { let p: Array<f32> = [a]; let q: Array<f32> = [a]; let r: Array<f32> = [a]; let s: Array<f32> = [a]; let u: Array<f32> = [a]; return a; } return a; } fn main() -> i32 { return 0; }";

fn special_roots() -> Vec<Vec<u32>> {
    // 40 roots (two CUDA blocks): special-bit triples with n covering every
    // recursion depth, NaN-payload copies (n < 3) and index traps (n >= 3).
    (0..40u32)
        .map(|i| {
            let s = |k: u32| SPECIAL[((i * 7 + k * 3) % SPECIAL.len() as u32) as usize];
            vec![s(0), s(1), s(2), i % 6]
        })
        .collect()
}
fn cases() -> Vec<Case> {
    let one = 0x3f80_0000u32;
    vec![
        Case {
            name: "special",
            source: SPECIAL_SOURCE,
            entry: "run",
            roots: special_roots(),
            trace_events: 2000,
            transitions: 10_000,
            storage: STORAGE,
            backend: &[],
        },
        Case {
            name: "limits",
            source: LIMITS_SOURCE,
            entry: "run",
            roots: (0..6u32)
                .map(|n| vec![one, n])
                .chain([vec![0x7fa1_2345, 9], vec![0x8000_0001, 7]])
                .collect(),
            trace_events: 10_000,
            transitions: 10_000,
            storage: STORAGE,
            backend: &[],
        },
        Case {
            name: "transitions",
            source: LIMITS_SOURCE,
            entry: "run",
            roots: vec![vec![one, 5], vec![one, 50]],
            trace_events: 10_000,
            transitions: 60,
            storage: STORAGE,
            backend: &[],
        },
        Case {
            name: "trace",
            source: LIMITS_SOURCE,
            entry: "run",
            roots: vec![vec![one, 5], vec![one, 50]],
            trace_events: 30,
            transitions: 10_000,
            storage: STORAGE,
            backend: &[],
        },
        Case {
            name: "storage",
            source: STORAGE_SOURCE,
            entry: "run",
            roots: (0..4u32).map(|n| vec![one, n]).collect(),
            trace_events: 2000,
            transitions: 10_000,
            storage: compiled::OwnedLimits {
                elements: 8,
                allocations: 4,
                cleanup_events: 2,
            },
            backend: &[(0, 11), (1, 13), (2, 12)],
        },
    ]
}
impl Case {
    fn limits(&self) -> oracle::Limits {
        oracle::Limits {
            fuel: 1000,
            transitions: self.transitions,
            trace_events: self.trace_events,
            array_elements: self.storage.elements,
        }
    }
    fn artifacts(&self, net: &Net) -> (compiled::Artifact, compiled::Artifact) {
        let proof = net.verify().unwrap();
        let emit = |target| {
            compiled::emit_resumable_owned_numeric(
                net,
                &proof,
                self.entry,
                target,
                compiled::Limits::default(),
                self.storage,
            )
            .unwrap()
        };
        (emit(compiled::Target::Cpu), emit(compiled::Target::Cuda))
    }
    fn args(&self, root: &[u32], net: &Net) -> Vec<eval::Value> {
        let proof = net.verify().unwrap();
        let mir = net.compiled_mir(&proof).unwrap();
        let f = mir.functions.iter().find(|f| f.name == self.entry).unwrap();
        f.params
            .iter()
            .zip(root)
            .map(|(p, w)| match p.ty {
                front::Type::F32 => eval::Value::F32(F32Bits::from_bits(*w)),
                front::Type::U32 => eval::Value::U32(*w),
                ref other => panic!("{other:?}"),
            })
            .collect()
    }
}

#[test]
fn issue_gnet_12_model_matches_abi11_every_epoch_and_oracle() {
    let mut statuses = BTreeSet::new();
    let mut observed = Vec::new();
    for case in cases() {
        let l = case.limits();
        let net = lowered(case.source);
        let (cpu, dev) = case.artifacts(&net);
        let want: Vec<_> = case
            .roots
            .iter()
            .map(|root| {
                if case.backend.iter().any(|(n, _)| *n == root[root.len() - 1]) {
                    None
                } else {
                    Some(expected(&net, case.entry, &case.args(root, &net), l))
                }
            })
            .collect();
        let model = build(case.name, &harness(&cpu, &dev, &case.roots, l));
        for quantum in [1, 3, 17] {
            let label = format!("{}-{quantum}", case.name);
            let run = model.execute(&label, quantum);
            assert_eq!(run.outputs.len(), case.roots.len(), "{label}");
            for (lane, root) in case.roots.iter().enumerate() {
                let output = &run.outputs[lane];
                match &want[lane] {
                    Some((header, trace)) => {
                        assert_eq!(&output[..9], &header[..], "{label} root {lane} {root:x?}");
                        assert_eq!(&run.traces[lane], trace, "{label} root {lane}");
                    }
                    None => {
                        let n = root[root.len() - 1];
                        let (_, status) = case.backend.iter().find(|(r, _)| *r == n).unwrap();
                        assert_eq!(output[0], *status, "{label} root {lane}");
                    }
                }
                statuses.insert(output[0]);
            }
            assert!(run.pending.windows(2).all(|w| w[1] <= w[0]), "{label}");
            assert_eq!(run.pending.first(), Some(&(case.roots.len() as u64)));
            assert_eq!(run.pending.last(), Some(&0));
            assert_eq!(run.pending.len() as u64, run.epochs + 1);
            observed.push((label, run.epochs));
        }
    }
    // Success; overflow, division and index traps; fuel, call-depth,
    // transition, trace and source array-size limits; backend element,
    // allocation-slot and cleanup limits. Status 7 has no ordinary-source
    // producer; status 8 has its own test.
    assert_eq!(statuses, [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13].into());
    eprintln!("ABI12 model runs (label, epochs): {observed:?}");
}

#[test]
fn issue_gnet_12_model_special_results_are_exact_bits() {
    // Terminal f32 results include NaN-payload copies (moves are bit-exact)
    // and canonical NaN from arithmetic, both read back from the device model.
    let case = &cases()[0];
    let l = case.limits();
    let net = lowered(case.source);
    let (cpu, dev) = case.artifacts(&net);
    let run = build("special-bits", &harness(&cpu, &dev, &case.roots, l)).execute("bits", 5);
    let (mut payload_copies, mut canonical, mut traps) = (0, 0, 0);
    for (lane, root) in case.roots.iter().enumerate() {
        let o = &run.outputs[lane];
        match o[0] {
            0 => {
                let bits = o[1] as u32;
                if bits == 0x7fc0_0000 {
                    canonical += 1;
                } else if F32Bits::from_bits(bits).is_nan() {
                    assert!(
                        root[..3].contains(&bits),
                        "non-canonical NaN must be a copy"
                    );
                    payload_copies += 1;
                }
            }
            9 => traps += 1,
            other => panic!("unexpected status {other} for {root:x?}"),
        }
    }
    eprintln!(
        "special bits: payload copies {payload_copies}, canonical NaN {canonical}, index traps {traps}"
    );
    assert!(payload_copies > 0 && canonical > 0 && traps > 0);
}

#[test]
fn issue_gnet_12_model_invalid_quantum_finishes_with_status_8() {
    let case = &cases()[0];
    let net = lowered(case.source);
    let (cpu, dev) = case.artifacts(&net);
    let model = build("status8", &harness(&cpu, &dev, &case.roots, case.limits()));
    for quantum in [0, 1_000_001] {
        let run = model.execute(&format!("status8-{quantum}"), quantum);
        assert_eq!(run.epochs, 1);
        assert_eq!(run.pending, [case.roots.len() as u64, 0]);
        assert!(run.outputs.iter().all(|o| o[0] == 8 && o[7] == 0));
    }
}

/// Replace every occurrence in the device namespace only (the yield check is
/// emitted once per continuation state); the CPU reference is untouched.
fn mutate_device(text: &str, from: &str, to: &str) -> String {
    let start = text.find("namespace dev {").unwrap();
    let (head, tail) = text.split_at(start);
    assert!(tail.contains(from), "missing device text {from}");
    format!("{head}{}", tail.replace(from, to))
}

#[test]
fn issue_gnet_12_model_detects_device_mutations() {
    let case = &cases()[0];
    let net = lowered(case.source);
    let (cpu, dev) = case.artifacts(&net);
    let heap = cpu.owned_layout().unwrap().heap_words();
    let text = harness(&cpu, &dev, &case.roots, case.limits());
    // A hardware-float substitute for the integer software adder: it differs
    // from strict-f32-v1 at least on NaN payload propagation.
    let float_add = "GNET_FN gnet_u gnet_f32_add(gnet_u a,gnet_u b) { float x,y; unsigned int ua=(unsigned int)a,ub=(unsigned int)b; __builtin_memcpy(&x,&ua,4); __builtin_memcpy(&y,&ub,4); float z=x+y; unsigned int r; __builtin_memcpy(&r,&z,4); return r; }\nGNET_FN gnet_u gnet_f32_add_integer(gnet_u a,gnet_u b) {";
    let mutations: [(&str, String, String, i32); 7] = [
        (
            "no-lane-guard",
            "if (lane>=config[0]) return;".into(),
            String::new(),
            TAIL,
        ),
        (
            "no-block-offset",
            "(gnet_u)blockIdx.x * blockDim.x + threadIdx.x".into(),
            "(gnet_u)threadIdx.x".into(),
            TASK,
        ),
        (
            "swapped-config",
            "config[1],config[2],config[3]".into(),
            "config[2],config[1],config[3]".into(),
            TASK,
        ),
        (
            "short-heap-stride",
            format!("heap+lane*{heap}"),
            format!("heap+lane*{}", heap - 4),
            HEAP,
        ),
        (
            "late-yield",
            "if (slice_steps==quantum) { GNET_YIELD; }".into(),
            "if (slice_steps==quantum+1) { GNET_YIELD; }".into(),
            TASK,
        ),
        (
            "tie-rounding",
            "(gnet_f32_below(a,shift-1)||(q&1))".into(),
            "true".into(),
            TASK,
        ),
        (
            "hardware-float-add",
            "GNET_FN gnet_u gnet_f32_add(gnet_u a,gnet_u b) {".into(),
            float_add.into(),
            TASK,
        ),
    ];
    for (name, from, to, code) in mutations {
        let o = build(name, &mutate_device(&text, &from, &to)).run(3);
        assert!(!o.timed_out, "{name}");
        assert_eq!(o.sanitizer(), None, "{name}: {}", o.stderr);
        assert_eq!(o.code, Some(code), "{name}: {}", o.last());
        eprintln!("mutation {name}: exit {code}, {}", o.last());
    }
}
