use crate::child_deadline;

use invar::{
    check, eval, front,
    general_net::{self, oracle},
    mir_eval,
    numeric_contract::F32Bits,
    stencil_source::{Contract, Error},
};
use invar_device::{
    Capabilities, Provenance,
    stencil::{self, Boundary},
};
use std::{
    ffi::OsString,
    fs,
    io::Read,
    path::Path,
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

/// Wall and CPU budgets for each `c++` invocation and generated program run.
/// Generated programs also limit themselves internally (alarm/RLIMIT_CPU).
const COMPILE_WALL: Duration = Duration::from_secs(180);
const RUN_WALL: Duration = Duration::from_secs(60);
#[cfg(unix)]
const CPU_SECONDS: u64 = 120;

/// Run a subprocess with a wall deadline and, on Unix, a CPU-seconds limit.
/// The child leads its own process group; on expiry the whole group is
/// killed, so compiler descendants (cc1plus, as, ld) cannot outlive it.
/// Returns the output and whether the wall deadline fired.
fn bounded(program: &Path, args: &[OsString], wall: Duration) -> (Output, bool) {
    #[cfg(unix)]
    let mut command = {
        use std::os::unix::process::CommandExt;
        let mut c = Command::new("sh");
        c.arg("-c")
            .arg(format!("ulimit -t {CPU_SECONDS} && exec \"$0\" \"$@\""))
            .arg(program)
            .args(args);
        c.process_group(0);
        c
    };
    #[cfg(not(unix))]
    let mut command = {
        let mut c = Command::new(program);
        c.args(args);
        c
    };
    command
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
    let output = Output {
        status,
        stdout: out.join().unwrap(),
        stderr: err.join().unwrap(),
    };
    (output, timed_out)
}
/// Bounded `c++` compile of `source` into `exe`; panics with diagnostics on failure.
fn compile(options: &[&str], extra: &[&str], source: &Path, exe: &Path) {
    let args: Vec<OsString> = options
        .iter()
        .chain(extra)
        .map(OsString::from)
        .chain([source.into(), "-o".into(), exe.into()])
        .collect();
    let (built, timed_out) = bounded(Path::new("c++"), &args, child_deadline::wall(COMPILE_WALL));
    assert!(
        built.status.success() && !timed_out,
        "c++ status={:?} timed_out={timed_out} {}",
        built.status,
        String::from_utf8_lossy(&built.stderr)
    );
}
/// Bounded run of a generated program; a wall timeout is always a failure.
fn run_program(exe: &Path) -> Output {
    let wall = child_deadline::wall(RUN_WALL);
    let (output, timed_out) = bounded(exe, &[], wall);
    assert!(!timed_out, "{} exceeded {wall:?}", exe.display());
    output
}
fn checked(s: &str) -> check::CheckedProgram {
    check::check(front::parse(s.as_bytes()).unwrap(), false).unwrap()
}
fn function(boundary: Boundary) -> String {
    let edge = |v: &str| {
        if boundary == Boundary::Clamp {
            format!("else {{ {v}=a[r*cols+c]; }}")
        } else {
            String::new()
        }
    };
    format!(
        "fn step(a:read Array<f32>,rows:u32,cols:u32)->Array<f32>{{var b=filled_f32(rows*cols,0.0);for r in 0u32..rows{{for c in 0u32..cols{{var n=0.0; if r>0u32{{n=a[(r-1u32)*cols+c];}} {ne} var s=0.0; if r+1u32<rows{{s=a[(r+1u32)*cols+c];}} {se} var w=0.0; if c>0u32{{w=a[r*cols+(c-1u32)];}} {we} var e=0.0; if c+1u32<cols{{e=a[r*cols+(c+1u32)];}} {ee} b[r*cols+c]=(((a[r*cols+c]+n)+s)+w)+e;}}}}return b;}}",
        ne = edge("n"),
        se = edge("s"),
        we = edge("w"),
        ee = edge("e")
    )
}
fn expected(input: &[u32], dims: [u32; 2], boundary: Boundary) -> Vec<u32> {
    let [rows, cols] = dims;
    let load = |y: i64, x: i64| {
        if boundary == Boundary::ConstantZero
            && (y < 0 || x < 0 || y >= rows as i64 || x >= cols as i64)
        {
            F32Bits::POS_ZERO
        } else {
            F32Bits::from_bits(
                input[(y.clamp(0, rows as i64 - 1) * cols as i64 + x.clamp(0, cols as i64 - 1))
                    as usize],
            )
        }
    };
    let mut out = Vec::new();
    for y in 0..rows as i64 {
        for x in 0..cols as i64 {
            out.push(
                load(y, x)
                    .add(load(y - 1, x))
                    .add(load(y + 1, x))
                    .add(load(y, x - 1))
                    .add(load(y, x + 1))
                    .to_bits(),
            );
        }
    }
    out
}
#[test]
fn issue_m4_02_transpose_stencil_schedules_source_binding_and_mutations() {
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        let s = function(boundary);
        let p = checked(&s);
        let c = Contract::from_checked(&p, "step").unwrap();
        assert_eq!(c.boundary(), boundary);
        assert_eq!(c.input(), "a");
        assert_eq!(c.output(), "b");
        let e = c
            .emit_cuda([3, 5], 20, Capabilities::cuda_12_9_sm89())
            .unwrap();
        e.verify_descriptor(e.plan().ir()).unwrap();
        for change in 0..5 {
            let mut wrong = e.plan().ir().clone();
            match change {
                0 => wrong.function_contract[0] ^= 1,
                1 => {
                    wrong.boundary = if boundary == Boundary::Clamp {
                        Boundary::ConstantZero
                    } else {
                        Boundary::Clamp
                    }
                }
                2 => wrong.dimensions = [5, 3],
                3 => wrong.numeric_policy = "relaxed".into(),
                4 => wrong.graph_revision += 1,
                _ => unreachable!(),
            };
            assert_eq!(e.verify_descriptor(&wrong), Err(Error::DescriptorMismatch));
        }
        let mut forged = e.plan().ir().clone();
        forged.function_contract = [7; 32];
        let origin = Provenance {
            source_identity: forged.source_identity,
            graph_revision: forged.graph_revision,
        };
        let physical = stencil::verify(forged, Capabilities::cuda_12_9_sm89(), origin).unwrap();
        assert_eq!(
            e.verify_descriptor(physical.ir()),
            Err(Error::DescriptorMismatch)
        );
        for changed in [
            s.replace("+n)+s", "+s)+n"),
            s.replace("r>0u32", "r>1u32"),
            s.replace("r+1u32<rows", "r<rows"),
            s.replace("c-1u32", "c+1u32"),
            s.replace("var n=0.0", "var n=1.0"),
            s.replace("+w)+e", "+(w+e))"),
            s.replace("a:read", "a:mut"),
            s.replace("return b;", "return filled_f32(0u32,0.0);"),
            s.replace("b[r*cols+c]=", "b[c*rows+r]="),
            s.replace("var n=0.0;", "let ignored=len(a); var n=0.0;"),
            s.replace(
                "var b=filled_f32(rows*cols,0.0);",
                "var b=filled_f32(rows+cols,0.0);",
            ),
        ] {
            let p = checked(&changed);
            assert!(
                matches!(Contract::from_checked(&p, "step"), Err(Error::Unrecognized)),
                "{changed}"
            );
        }
        let mut stale = checked(&s);
        stale.expression_types.clear();
        assert_eq!(
            Contract::from_checked(&stale, "step").unwrap_err(),
            Error::CheckedMismatch
        );
        for (dims, len) in [([2, 3], 5), ([u32::MAX, 2], 0), ([1001, 1000], 1_000_000)] {
            assert_eq!(
                c.emit_cuda(dims, len, Capabilities::cuda_12_9_sm89())
                    .unwrap_err(),
                Error::Bounds
            );
        }
        let renamed = s
            .replace("step", "smooth")
            .replace("rows", "height")
            .replace("cols", "width")
            .replace(" r ", " y ")
            .replace("r*", "y*")
            .replace("r>", "y>")
            .replace("r+", "y+")
            .replace("r-", "y-");
        assert_eq!(
            Contract::from_checked(&checked(&renamed), "smooth")
                .unwrap()
                .function(),
            "smooth"
        );
    }
}
#[test]
fn issue_m4_02_transpose_stencil_schedules_ordinary_source_and_repeated_slices() {
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        for dims in [[0, 5], [5, 0], [1, 1], [1, 3], [3, 1], [3, 5]] {
            for steps in [1, 3] {
                let count = (dims[0] * dims[1]) as usize;
                let input: Vec<_> = (0..count).map(|i| ((i as i32 % 7) - 3) as f32).collect();
                let mut want: Vec<_> = input.iter().map(|x| x.to_bits()).collect();
                for _ in 0..steps {
                    want = expected(&want, dims, boundary);
                }
                let list = |bits: &[u32]| {
                    bits.iter()
                        .map(|&v| format!("{}", f32::from_bits(v)) + ".0")
                        .collect::<Vec<_>>()
                        .join(",")
                };
                let a = list(&input.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
                let b = list(&want);
                let mut calls = format!("let slice0=step(a,{}u32,{}u32);", dims[0], dims[1]);
                for i in 1..steps {
                    calls.push_str(&format!(
                        "let slice{i}=step(slice{},{}u32,{}u32);",
                        i - 1,
                        dims[0],
                        dims[1]
                    ));
                }
                let s = format!(
                    "{} fn main()->i32{{let a:Array<f32> = [{a}]; {calls} let want:Array<f32> = [{b}]; for i in 0u32..len(want){{if slice{}[i]!=want[i]{{return 0;}}}}return 7;}}",
                    function(boundary),
                    steps - 1
                );
                let p = checked(&s);
                let c = Contract::from_checked(&p, "step").unwrap();
                let m = c.verified_mir().unwrap();
                let reference = eval::observe(&p, 1_000_000);
                let serial = mir_eval::observe(&m, 1_000_000);
                assert_eq!(reference.result, Ok(eval::Value::I32(7)));
                assert_eq!(serial.result, reference.result);
                assert_eq!(serial.steps_used, reference.steps_used);
                assert_eq!(serial.cleanup_trace, reference.cleanup_trace);
                let net = general_net::Net::encode(&m, general_net::Limits::default()).unwrap();
                let proof = net.verify().unwrap();
                let observed = oracle::observe(
                    &net,
                    &proof,
                    oracle::Limits {
                        fuel: 1_000_000,
                        ..oracle::Limits::default()
                    },
                )
                .unwrap();
                assert_eq!(observed.failure, None);
                assert_eq!(observed.semantic.result, reference.result);
                assert_eq!(observed.semantic.steps_used, reference.steps_used);
                assert_eq!(observed.semantic.cleanup_trace, reference.cleanup_trace);
                println!(
                    "ordinary_source boundary={} dims={dims:?} steps={steps} fuel={} cleanup={}",
                    boundary.name(),
                    reference.steps_used,
                    reference.cleanup_trace.len()
                );
            }
        }
    }
}
#[test]
fn issue_m4_02_transpose_stencil_schedules_fallback_failure_prefixes() {
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        for len in [1, 6] {
            let s = format!(
                "{} fn main()->i32{{let a=filled_f32({len}u32,1.0);let b=step(a,2u32,3u32);return 7;}}",
                function(boundary)
            );
            let p = checked(&s);
            let c = Contract::from_checked(&p, "step").unwrap();
            let m = c.verified_mir().unwrap();
            let done = eval::observe(&p, 10000);
            if len == 1 {
                assert!(done.result.is_err());
                assert!(
                    c.emit_cuda([2, 3], 1, Capabilities::cuda_12_9_sm89())
                        .is_err()
                );
            }
            for fuel in 1..=done.steps_used + 1 {
                let a = eval::observe(&p, fuel);
                let b = mir_eval::observe(&m, fuel);
                assert_eq!(a.result, b.result);
                assert_eq!(a.steps_used, b.steps_used);
                assert_eq!(a.cleanup_trace, b.cleanup_trace);
            }
        }
    }
}
#[test]
fn issue_m4_02_transpose_stencil_schedules_emitted_cpu_bits_guards_and_failures() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        for dims in [
            [0, 33],
            [33, 0],
            [1, 1],
            [1, 33],
            [33, 1],
            [3, 5],
            [31, 33],
            [127, 257],
        ] {
            let count = (dims[0] * dims[1]) as usize;
            let special = [
                0x80000000, 0, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fa12345, 0x4b800000,
                0xcb800000, 0x3f800000,
            ];
            let input: Vec<_> = (0..count)
                .map(|i| {
                    if i < special.len() {
                        special[i]
                    } else {
                        ((i % 19) as f32 - 9.0).to_bits()
                    }
                })
                .collect();
            let want = expected(&input, dims, boundary);
            let c = Contract::from_checked(&checked(&function(boundary)), "step").unwrap();
            let p = c
                .emit_cuda(dims, count as u64, Capabilities::cuda_12_9_sm89())
                .unwrap();
            let cpu = stencil::emit_cpu(p.plan()).unwrap();
            let dir = std::env::temp_dir().join(format!(
                "invar-stencil-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&dir).unwrap();
            let words = input
                .iter()
                .map(|v| format!("{v}u"))
                .collect::<Vec<_>>()
                .join(",");
            let driver = format!(
                r#"
#include <vector>
#include <iostream>
#include <unistd.h>
int main() {{ std::vector<std::uint32_t> words={{{words}}}; std::vector<float> input(words.size()+8), output({count}+8); for(unsigned i=0;i<words.size();++i) std::memcpy(&input[i],&words[i],4); const std::uint32_t guard=0xdeadbeef; for(auto& v:output)std::memcpy(&v,&guard,4); auto before=input;
if (invar_stencil_f32_v1(input.data(),output.data()))return 2;
if (std::memcmp(input.data(),before.data(),input.size()*4))return 3;
for(unsigned i={count};i<output.size();++i){{std::uint32_t v;std::memcpy(&v,&output[i],4);if(v!=guard)return 4;}}
if ({count}>0) {{ auto saved=output; if(invar_stencil_f32_v1(input.data(),input.data())!=2)return 5; if(invar_stencil_f32_v1(nullptr,output.data())!=2)return 6; if(std::fesetround(FE_DOWNWARD))return 7; if(invar_stencil_f32_v1(input.data(),output.data())!=1)return 8; if(std::memcmp(output.data(),saved.data(),output.size()*4))return 9; if(std::fesetround(FE_TONEAREST))return 10; }}
for(unsigned i=0;i<{count};++i){{std::uint32_t v;std::memcpy(&v,&output[i],4);std::cout<<v<<' ';}}return 0;}}
"#
            );
            fs::write(dir.join("step.cpp"), format!("{cpu}\n{driver}")).unwrap();
            compile(
                p.plan().cpu_options(),
                &["-O2", "-fsanitize=undefined", "-fno-sanitize-recover=all"],
                &dir.join("step.cpp"),
                &dir.join("step"),
            );
            let run = run_program(&dir.join("step"));
            assert!(
                run.status.success(),
                "status={:?} stderr={}",
                run.status,
                String::from_utf8_lossy(&run.stderr)
            );
            let actual: Vec<u32> = String::from_utf8(run.stdout)
                .unwrap()
                .split_whitespace()
                .map(|n| n.parse().unwrap())
                .collect();
            assert_eq!(actual, want);
            println!(
                "cpu_actual boundary={} dims={dims:?} words={count} cuda_hash={} cpu_hash={} guard=8",
                boundary.name(),
                invar_device::source_hash(p.source()),
                invar_device::source_hash(&cpu)
            );
            fs::remove_dir_all(dir).unwrap();
        }
    }
}

#[test]
fn issue_m4_02_transpose_stencil_schedules_hand_derived_fp32_goldens() {
    for (word, zero, clamp) in [
        (0x80000000, 0, 0x80000000),
        (1, 1, 5),
        (0x7fa12345, 0x7fc00000, 0x7fc00000),
        (0x7f800000, 0x7f800000, 0x7f800000),
    ] {
        assert_eq!(
            expected(&[word], [1, 1], Boundary::ConstantZero),
            vec![zero]
        );
        assert_eq!(expected(&[word], [1, 1], Boundary::Clamp), vec![clamp]);
    }
    // At the center: ((2^24 + -2^24) + 1) + 0 + 0 = 1. Reordering
    // -2^24 + 1 first rounds away that 1 and would incorrectly produce 0.
    let mut input = vec![0; 9];
    input[4] = 0x4b800000;
    input[1] = 0xcb800000;
    input[7] = 0x3f800000;
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        assert_eq!(expected(&input, [3, 3], boundary)[4], 0x3f800000);
    }
}

fn array_words(result: &Result<eval::Value, eval::EvalError>) -> Vec<u32> {
    match result {
        Ok(eval::Value::ArrayF32(values)) => values.borrow().iter().map(|v| v.to_bits()).collect(),
        other => panic!("expected a successful FP32 owner, got {other:?}"),
    }
}

fn observe_special_source(
    boundary: Boundary,
    wrapper: &str,
    args: &[eval::Value],
    input: &[u32],
    dims: [u32; 2],
) {
    let source = format!("{} {wrapper}", function(boundary));
    let checked = checked(&source);
    let contract = Contract::from_checked(&checked, "step").unwrap();
    let net = general_net::Net::encode(
        &contract.verified_mir().unwrap(),
        general_net::Limits::default(),
    )
    .unwrap();
    let proof = net.verify().unwrap();
    let full = eval::named::observe_owned_numeric_named(&checked, "probe", args, 10_000);
    let want = expected(input, dims, boundary);
    assert_eq!(array_words(&full.result), want);
    for fuel in [
        1,
        full.steps_used / 2,
        full.steps_used - 1,
        full.steps_used,
        full.steps_used + 1,
    ] {
        let source = eval::named::observe_owned_numeric_named(&checked, "probe", args, fuel);
        let graph = oracle::observe_call(
            &net,
            &proof,
            "probe",
            args,
            oracle::Limits {
                fuel,
                transitions: 100_000,
                trace_events: 100_000,
                array_elements: 100,
            },
        )
        .unwrap();
        assert_eq!(graph.failure, None);
        // Prefixes below the full run must stop on the fuel limit, not succeed early.
        match &source.result {
            Ok(_) => assert!(fuel >= full.steps_used, "early success at fuel {fuel}"),
            Err(eval::EvalError::EvaluationLimit { fuel_limit, .. }) => {
                assert!(fuel < full.steps_used && *fuel_limit == fuel)
            }
            Err(other) => panic!("unexpected source failure at fuel {fuel}: {other:?}"),
        }
        match (&source.result, &graph.semantic.result) {
            (Ok(_), Ok(_)) => {
                assert_eq!(array_words(&source.result), want);
                assert_eq!(array_words(&graph.semantic.result), want);
            }
            (Err(a), Err(b)) => assert_eq!(a, b),
            other => panic!("source/graph outcome mismatch at fuel {fuel}: {other:?}"),
        }
        assert_eq!(source.steps_used, graph.semantic.steps_used);
        assert_eq!(source.cleanup_trace, graph.semantic.cleanup_trace);
    }
    println!(
        "special_source boundary={} dims={dims:?} input={input:08x?} output={want:08x?} fuel={} cleanup={} source={} graph={} function_contract={:02x?}",
        boundary.name(),
        full.steps_used,
        full.cleanup_trace.len(),
        contract
            .emit_cuda(dims, input.len() as u64, Capabilities::cuda_12_9_sm89())
            .unwrap()
            .plan()
            .ir()
            .source_identity,
        net.graph.revision,
        contract.function_contract(),
    );
}

#[test]
fn issue_m4_02_transpose_stencil_schedules_source_special_value_bits() {
    // Exact external scalar bits enter an ordinary source wrapper, which builds
    // its own array and calls the recognized operation. This existing private
    // observer does not add a public bitcast or bypass source array ownership.
    let singleton = "fn probe(x:f32)->Array<f32>{let a:Array<f32> = [x];return step(a,1u32,1u32);}";
    let float = |bits| eval::Value::F32(F32Bits::from_bits(bits));
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        for (word, zero, clamp) in [
            (0x00000000, 0x00000000, 0x00000000),
            (0x80000000, 0x00000000, 0x80000000),
            (0x00000001, 0x00000001, 0x00000005),
            (0x80000001, 0x80000001, 0x80000005),
            (0x7f800000, 0x7f800000, 0x7f800000),
            (0xff800000, 0xff800000, 0xff800000),
            (0x7fa12345, 0x7fc00000, 0x7fc00000),
            (0xffc12345, 0x7fc00000, 0x7fc00000),
            (0x7f7fffff, 0x7f7fffff, 0x7f800000),
        ] {
            assert_eq!(
                expected(&[word], [1, 1], boundary),
                vec![if boundary == Boundary::Clamp {
                    clamp
                } else {
                    zero
                }]
            );
            observe_special_source(boundary, singleton, &[float(word)], &[word], [1, 1]);
        }
        let cross = "fn probe(c:f32,n:f32,s:f32,w:f32,e:f32)->Array<f32>{let a:Array<f32> = [0.0,n,0.0,w,c,e,0.0,s,0.0];return step(a,3u32,3u32);}";
        for (values, center) in [
            ([0x4b800000, 0xcb800000, 0x3f800000, 0, 0], 0x3f800000),
            ([0x4b800000, 0x3f800000, 0xcb800000, 0, 0], 0),
            ([0x7f800000, 0xff800000, 0, 0, 0], 0x7fc00000),
            ([0x7fa12345, 1, 0, 0, 0], 0x7fc00000),
            ([0x80000000; 5], 0x80000000),
            ([1, 1, 1, 1, 1], 5),
            ([1, 0x80000001, 1, 0, 0], 1),
        ] {
            let [c, n, s, w, e] = values;
            let input = [0, n, 0, w, c, e, 0, s, 0];
            assert_eq!(expected(&input, [3, 3], boundary)[4], center);
            observe_special_source(boundary, cross, &values.map(float), &input, [3, 3]);
        }
        // These values also originate entirely in source expressions, including
        // signed zero, infinities, canonical NaN and the smallest subnormal.
        let literal = "fn probe()->Array<f32>{let a:Array<f32> = [-0.0,1.0/0.0,0.0/0.0,-1.0/0.0,0.00000000000000000000000000000000000000000000140129846432481707092372958328991613128026194187651577175706828388979108268586060148663818836212158203125,1.0,16777216.0,-16777216.0];return step(a,2u32,4u32);}";
        observe_special_source(
            boundary,
            literal,
            &[],
            &[
                0x80000000, 0x7f800000, 0x7fc00000, 0xff800000, 1, 0x3f800000, 0x4b800000,
                0xcb800000,
            ],
            [2, 4],
        );
    }
}

// This harness compiles the emitted text as C++ for CPU modeling. It is neither
// NVRTC compilation nor GPU execution, and cannot prove GPU memory behavior.
// Qualifier/intrinsic mappings leave the kernel text itself byte-for-byte intact.
const SANITIZED_MODEL: &[&str] = &[
    "-O2",
    "-pthread",
    "-fsanitize=address,undefined",
    "-fno-sanitize-recover=all",
];
// Modeling limits of this shim (also recorded in the hardening evidence):
// - `__shared__` becomes a function-local `static`: zero-initialized once and
//   persistent across tiles, blocks and grid launches. A kernel that read a
//   shared cell not written in the current phase would see zero or a stale
//   value instead of undefined device memory, and neither ASan nor UBSan
//   reports uninitialized reads. The model cannot detect such a defect.
// - The 256 lanes are real pthreads joined by mutex/condition-variable
//   barriers, but no ThreadSanitizer run is made: there is no data-race
//   guarantee, and pthread ordering is not the CUDA memory model.
const CUDA_CPU_SHIM: &str = r#"
#include <cfenv>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <pthread.h>
#include <unistd.h>
#include <sys/resource.h>
#include <vector>
#include <iostream>
#define __device__
#define __global__
#define __shared__ static
struct Dim { unsigned int x=0,y=0,z=0; };
static thread_local Dim threadIdx,blockIdx;
static Dim blockDim{16,16,1},gridDim;
struct Barrier {
  std::mutex mutex;
  std::condition_variable changed;
  unsigned int arrived=0,generation=0;
  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    const auto observed=generation;
    if (++arrived==256) { arrived=0; ++generation; changed.notify_all(); }
    else changed.wait(lock,[&]{return generation!=observed;});
  }
};
static Barrier phase;
static std::atomic<unsigned long long> kernel_barriers{0};
static void __syncthreads() { ++kernel_barriers; phase.wait(); }
static unsigned int __float_as_uint(float x) { unsigned int v; std::memcpy(&v,&x,4); return v; }
static float __uint_as_float(unsigned int x) { float v; std::memcpy(&v,&x,4); return v; }
static float __fadd_rn(float a,float b) { volatile float v=a+b; return v; }
"#;

#[test]
#[cfg_attr(
    not(unix),
    ignore = "the CPU barrier model limits itself with setrlimit (sys/resource.h)"
)]
fn issue_m4_02_transpose_stencil_schedules_exact_cuda_text_cpu_barrier_model() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    // Each process has at most 256 workers, 128KiB each (32MiB total stack),
    // <=1155 cells and a 20s wall/15s CPU limit. Processes execute sequentially.
    // grid=1 tests repeated tile reuse; grid=3 tests masked/skipped block work.
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        for dims in [[0, 17], [1, 1], [1, 17], [17, 1], [17, 19], [33, 35]] {
            let count = (dims[0] * dims[1]) as usize;
            let special = [
                0x80000000, 0, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fa12345, 0x4b800000,
                0xcb800000, 0x3f800000,
            ];
            let input: Vec<_> = (0..count)
                .map(|i| {
                    if i < special.len() {
                        special[i]
                    } else {
                        ((i % 13) as f32 - 6.0).to_bits()
                    }
                })
                .collect();
            let want = expected(&input, dims, boundary);
            let contract = Contract::from_checked(&checked(&function(boundary)), "step").unwrap();
            let emitted = contract
                .emit_cuda(dims, count as u64, Capabilities::cuda_12_9_sm89())
                .unwrap();
            let dir = std::env::temp_dir().join(format!(
                "invar-stencil-cpu-model-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&dir).unwrap();
            let words = input
                .iter()
                .map(|v| format!("{v}u"))
                .collect::<Vec<_>>()
                .join(",");
            let tiles = u64::from(dims[0]).div_ceil(16) * u64::from(dims[1]).div_ceil(16);
            let driver = format!(
                r#"
static const float* model_input;
static float* model_output;
static void* worker(void* argument) {{
  const unsigned int lane=*static_cast<const unsigned int*>(argument);
  threadIdx=Dim{{lane%16,lane/16,0}};
  for(unsigned int b=0;b<gridDim.x;++b) {{
    blockIdx=Dim{{b,0,0}};
    invar_stencil_f32_v1(model_input,model_output);
    phase.wait(); // Sequential block ownership, outside the emitted kernel.
  }}
  return nullptr;
}}
int main() {{
  const rlimit cpu{{15,15}}; if(setrlimit(RLIMIT_CPU,&cpu))return 10;
  if(std::fesetround(FE_TONEAREST))return 11;
  std::vector<std::uint32_t> words={{{words}}};
  std::vector<float> input(words.size()),output({count}+8);
  for(unsigned int i=0;i<words.size();++i)std::memcpy(&input[i],&words[i],4);
  const auto before=input;
  const std::uint32_t guard=0xdeadbeef;
  for(unsigned int g:{{1u,3u}}) {{
    for(auto& v:output)std::memcpy(&v,&guard,4);
    gridDim=Dim{{g,1,1}}; model_input=input.data();model_output=output.data();kernel_barriers=0;
    pthread_attr_t attr;
    if(pthread_attr_init(&attr)||pthread_attr_setstacksize(&attr,128*1024))return 12;
    pthread_t threads[256]; unsigned int lanes[256];
    for(unsigned int lane=0;lane<256;++lane){{lanes[lane]=lane;if(pthread_create(&threads[lane],&attr,worker,&lanes[lane]))std::_Exit(13);}}
    pthread_attr_destroy(&attr);
    for(auto& thread:threads)if(pthread_join(thread,nullptr))return 14;
    if(kernel_barriers!={tiles}ull*2*256)return 15;
    if (!input.empty() && std::memcmp(input.data(),before.data(),input.size()*4))return 16;
    for(unsigned int i={count};i<output.size();++i){{std::uint32_t v;std::memcpy(&v,&output[i],4);if(v!=guard)return 17;}}
    for(unsigned int i=0;i<{count};++i){{std::uint32_t v;std::memcpy(&v,&output[i],4);std::cout<<v<<' ';}}
    std::cout<<'\n';
  }}
  return 0;
}}
"#
            );
            let translation = format!("{CUDA_CPU_SHIM}\n{}\n{driver}", emitted.source());
            assert_eq!(translation.matches(emitted.source()).count(), 1);
            fs::write(dir.join("model.cpp"), translation).unwrap();
            compile(
                emitted.plan().cpu_options(),
                SANITIZED_MODEL,
                &dir.join("model.cpp"),
                &dir.join("model"),
            );
            let run = run_program(&dir.join("model"));
            assert!(
                run.status.success(),
                "CPU model boundary={boundary:?} dims={dims:?} status={:?} stderr={}",
                run.status,
                String::from_utf8_lossy(&run.stderr)
            );
            let stdout = String::from_utf8(run.stdout).unwrap();
            let lines: Vec<_> = stdout.lines().collect();
            assert_eq!(lines.len(), 2);
            for line in lines {
                let actual: Vec<u32> = line
                    .split_whitespace()
                    .map(|x| x.parse().unwrap())
                    .collect();
                assert_eq!(
                    actual, want,
                    "CPU model boundary={boundary:?} dims={dims:?}"
                );
            }
            println!(
                "cpu_model_exact_cuda_text boundary={} dims={dims:?} logical_threads=256 stack_bytes=33554432 grids=1,3 uniform_kernel_barriers_per_grid={} cuda_source_hash={} guards=8 actual_gpu=false",
                boundary.name(),
                tiles * 2 * 256,
                invar_device::source_hash(emitted.source())
            );
            if boundary == Boundary::ConstantZero && dims == [1, 1] {
                // This deliberate mask mutation must be caught even if an
                // out-of-domain halo value is never used by an output lane.
                let mutant = emitted.source().replace("y<1ll && x<1ll", "y<2ll && x<2ll");
                assert_ne!(mutant, emitted.source());
                fs::write(
                    dir.join("overread.cpp"),
                    format!("{CUDA_CPU_SHIM}\n{mutant}\n{driver}"),
                )
                .unwrap();
                compile(
                    emitted.plan().cpu_options(),
                    SANITIZED_MODEL,
                    &dir.join("overread.cpp"),
                    &dir.join("overread"),
                );
                let fault = run_program(&dir.join("overread"));
                let diagnostic = String::from_utf8_lossy(&fault.stderr);
                assert!(!fault.status.success());
                assert!(diagnostic.contains("heap-buffer-overflow"), "{diagnostic}");
                println!(
                    "cpu_model_negative_mutated_halo_mask rejected=heap-buffer-overflow actual_gpu=false"
                );
            }
            fs::remove_dir_all(dir).unwrap();
        }
    }
}

#[cfg(unix)]
#[test]
fn issue_m4_02_transpose_stencil_schedules_bounded_subprocess_cleanup() {
    // The wall deadline kills the whole process group: a shell and the sleep
    // it backgrounds are both gone afterwards.
    let (out, timed_out) = bounded(
        Path::new("sh"),
        &["-c".into(), "sleep 30 & echo $!; wait".into()],
        Duration::from_millis(500),
    );
    assert!(timed_out && !out.status.success(), "{out:?}");
    let descendant = String::from_utf8(out.stdout).unwrap().trim().to_owned();
    assert!(!descendant.is_empty());
    #[cfg(target_os = "linux")]
    {
        let stat = format!("/proc/{descendant}/stat");
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
    // A compiler failure is reported, not hidden, and does not time out.
    let dir = std::env::temp_dir().join(format!("invar-stencil-bounded-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("bad.cpp"), "int main( {").unwrap();
    let (built, timed_out) = bounded(
        Path::new("c++"),
        &[
            dir.join("bad.cpp").into(),
            "-o".into(),
            dir.join("bad").into(),
        ],
        child_deadline::wall(COMPILE_WALL),
    );
    assert!(!built.status.success() && !timed_out);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn issue_m4_02_stencil_cap_2_28_source_plan() {
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        let c = Contract::from_checked(&checked(&function(boundary)), "step").unwrap();
        for shape in [[16384, 16384], [1, 1 << 28], [1 << 28, 1]] {
            let p = c
                .emit_cuda(shape, 1 << 28, Capabilities::cuda_12_9_sm89())
                .unwrap();
            assert_eq!(p.plan().ir().input.elements, 1 << 28);
        }
        assert_eq!(
            c.emit_cuda(
                [1, (1 << 28) + 1],
                (1 << 28) + 1,
                Capabilities::cuda_12_9_sm89()
            )
            .unwrap_err(),
            Error::Bounds
        );
    }
}
