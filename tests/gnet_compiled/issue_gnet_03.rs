#![forbid(unsafe_code)]

use crate::{child_deadline, gnet_words};
use general_net::{Net, compiled, oracle};
use invar::{check, eval, front, general_net};
use std::{fs, path::PathBuf, process::Command};

const FIB: &str = "fn fib(n: u32) -> i32 { if n < 2u32 { return 1; } return fib(n - 1u32) + fib(n - 2u32); } fn main() -> i32 { return fib(8u32); }";
const DRIVER: &str = r#"
#include <iostream>
#include <vector>
#include <string>
int main(int argc, char** argv) {
    if (argc<4) return 2;
    gnet_u fuel=std::stoull(argv[1]), transitions=std::stoull(argv[2]), cap=std::stoull(argv[3]);
    if (cap>1000000) return 2;
    std::vector<gnet_w> input;
    for (int i=4;i<argc;++i) input.push_back(std::stoll(argv[i]));
    gnet_w output[8]={}; std::vector<gnet_u> trace(cap*7);
    gnet_run(input.data(),output,trace.data(),fuel,transitions,cap);
    for (auto word:output) std::cout<<word<<' ';
    for (gnet_u i=0;i<(gnet_u)output[7]*7;++i) std::cout<<trace[i]<<' ';
    std::cout<<'\n';
}
"#;

fn lowered(source: &str) -> (check::CheckedProgram, Net) {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let net = Net::lower(&checked, general_net::Limits::default()).unwrap();
    (checked, net)
}

struct Cpu {
    dir: PathBuf,
}
impl Cpu {
    fn new(artifact: &compiled::Artifact) -> Self {
        let dir = child_deadline::scratch("invar-gnet");
        fs::write(
            dir.join("source.cpp"),
            format!("{}\n{DRIVER}", artifact.source()),
        )
        .unwrap();
        // UBSan checks emitted integer arithmetic, including signed MIN/-1 and
        // the u32 product that cannot fit a signed 64-bit intermediate.
        child_deadline::cxx(
            child_deadline::CXX17_UBSAN_O1,
            &dir.join("source.cpp"),
            &dir.join("run"),
        );
        Self { dir }
    }
    fn run(&self, args: &[eval::Value], limits: oracle::Limits) -> Vec<i64> {
        let mut command = Command::new(self.dir.join("run"));
        command.args([
            limits.fuel.to_string(),
            limits.transitions.to_string(),
            limits.trace_events.to_string(),
        ]);
        for arg in args {
            command.arg(match arg {
                eval::Value::I32(n) => n.to_string(),
                eval::Value::U32(n) => n.to_string(),
                eval::Value::Bool(n) => usize::from(*n).to_string(),
                _ => panic!("scalar args"),
            });
        }
        let output = command.output().unwrap();
        child_deadline::assert_success(&output);
        String::from_utf8(output.stdout)
            .unwrap()
            .split_whitespace()
            .map(|n| n.parse().unwrap())
            .collect()
    }
}
impl Drop for Cpu {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap();
    }
}

fn expected(observation: &oracle::Observation) -> Vec<i64> {
    let (status, start, end) = gnet_words::status(observation);
    let mut words = vec![
        status,
        gnet_words::result(observation),
        start,
        end,
        observation.semantic.steps_used as i64,
        observation.transitions as i64,
        observation.max_frames as i64,
        observation.events.len() as i64,
    ];
    for event in &observation.events {
        words.extend(gnet_words::event_i64(event));
    }
    words
}

fn compare(
    cpu: &Cpu,
    net: &Net,
    entry: &str,
    args: &[eval::Value],
    limits: oracle::Limits,
) -> oracle::Observation {
    let observation =
        oracle::observe_call(net, &net.verify().unwrap(), entry, args, limits).unwrap();
    assert_eq!(
        cpu.run(args, limits),
        expected(&observation),
        "compiled control differs for {entry} {args:?} {limits:?}"
    );
    observation
}

fn compiled_cpu(net: &Net, entry: &str) -> Cpu {
    Cpu::new(
        &compiled::emit(
            net,
            &net.verify().unwrap(),
            entry,
            compiled::Target::Cpu,
            compiled::Limits::default(),
        )
        .unwrap(),
    )
}

#[test]
fn issue_gnet_03_compiled_recursive_runtime_inputs_match_all_observations() {
    let (_, net) = lowered(FIB);
    let cpu = compiled_cpu(&net, "fib");
    for (n, result) in [1, 1, 2, 3, 5, 8, 13, 21, 34].into_iter().enumerate() {
        let observation = compare(
            &cpu,
            &net,
            "fib",
            &[eval::Value::U32(n as u32)],
            oracle::Limits::default(),
        );
        assert_eq!(observation.semantic.result, Ok(eval::Value::I32(result)));
    }
    compare(
        &cpu,
        &net,
        "fib",
        &[eval::Value::I32(-1)],
        oracle::Limits::default(),
    );
    compare(
        &cpu,
        &net,
        "fib",
        &[eval::Value::U32(1)],
        oracle::Limits {
            fuel: 0,
            ..oracle::Limits::default()
        },
    );
}

#[test]
fn issue_gnet_03_mutual_calls_and_mixed_runtime_parameters_are_compiled() {
    let (_, net) = lowered(
        "fn even(n: u32) -> Bool { if n == 0u32 { return true; } return odd(n - 1u32); } fn odd(n: u32) -> Bool { if n == 0u32 { return false; } return even(n - 1u32); } fn choose(n: u32, x: i32, flag: Bool) -> i32 { if even(n) and flag { return x; } return -x; } fn main() -> i32 { return choose(2u32, 7, true); }",
    );
    let cpu = compiled_cpu(&net, "choose");
    for n in 0..8 {
        for flag in [false, true] {
            for x in [-9, 0, 14] {
                compare(
                    &cpu,
                    &net,
                    "choose",
                    &[
                        eval::Value::U32(n),
                        eval::Value::I32(x),
                        eval::Value::Bool(flag),
                    ],
                    oracle::Limits::default(),
                );
            }
        }
    }
    compare(
        &cpu,
        &net,
        "choose",
        &[
            eval::Value::U32(2),
            eval::Value::I32(1),
            eval::Value::U32(2),
        ],
        oracle::Limits::default(),
    );
}

#[test]
fn issue_gnet_03_compiled_source_loops_phi_short_circuit_and_integer_edges() {
    for source in [
        "fn main() -> i32 { var n = 0; for i in 0u32..5u32 { if i < 3u32 { n = n + 4; } else { n = n - 2; } } while n < 12 { n = n + 1; } return n; }",
        "fn bad() -> Bool { return 1 / 0 == 0; } fn main() -> i32 { if false and bad() { return 0; } if true or bad() { return 7; } return 0; }",
        "fn main() -> i32 { return (-2147483648i32) % (-1); }",
        "fn main() -> i32 { return (-2147483648i32) / (-1); }",
        "fn main() -> i32 { let n = 4294967295u32 * 4294967295u32; return 0; }",
        "fn main() -> i32 { let n = 0u32 - 1u32; return 0; }",
        "fn main() -> i32 { let n = wrapping_mul(4294967295u32,4294967295u32); if n == 1u32 { return 1; } return 0; }",
        "fn f() -> Unit { return; } fn main() -> i32 { f(); for i in 4294967295u32..4294967295u32 { return 1/0; } return 9; }",
        "fn first() -> i32 { return 2147483647 + 1; } fn second() -> i32 { return 1/0; } fn main() -> i32 { return first() + second(); }",
    ] {
        let (checked, net) = lowered(source);
        let cpu = compiled_cpu(&net, "main");
        let observation = compare(&cpu, &net, "main", &[], oracle::Limits::default());
        let reference = eval::observe(&checked, 100_000);
        assert_eq!(observation.semantic.result, reference.result, "{source}");
        assert_eq!(
            observation.semantic.steps_used, reference.steps_used,
            "{source}"
        );
    }
}

#[test]
fn issue_gnet_03_compiled_fuel_tool_prefixes_and_call_depth_are_distinct() {
    let (_, net) = lowered(FIB);
    let cpu = compiled_cpu(&net, "fib");
    let args = [eval::Value::U32(3)];
    let complete = compare(&cpu, &net, "fib", &args, oracle::Limits::default());
    for fuel in 1..=complete.semantic.steps_used {
        compare(
            &cpu,
            &net,
            "fib",
            &args,
            oracle::Limits {
                fuel,
                ..oracle::Limits::default()
            },
        );
    }
    for transitions in [0, 1, 4, 17, complete.transitions - 1] {
        compare(
            &cpu,
            &net,
            "fib",
            &args,
            oracle::Limits {
                transitions,
                ..oracle::Limits::default()
            },
        );
    }
    for trace_events in [0, 1, 4, 17, complete.events.len() - 1] {
        compare(
            &cpu,
            &net,
            "fib",
            &args,
            oracle::Limits {
                trace_events,
                ..oracle::Limits::default()
            },
        );
    }
    let (_, net) = lowered(
        "fn f(n: u32) -> i32 { if n == 0u32 { return 7; } return f(n - 1u32); } fn main() -> i32 { return f(1u32); }",
    );
    let cpu = compiled_cpu(&net, "f");
    for n in [0, 62, 63, 64, 70] {
        compare(
            &cpu,
            &net,
            "f",
            &[eval::Value::U32(n)],
            oracle::Limits::default(),
        );
    }
    let (_, net) = lowered(
        "fn first() -> i32 { while true {} return 0; } fn later() -> i32 { return 1/0; } fn main() -> i32 { return first() + later(); }",
    );
    let cpu = compiled_cpu(&net, "main");
    compare(
        &cpu,
        &net,
        "main",
        &[],
        oracle::Limits {
            fuel: 50,
            ..oracle::Limits::default()
        },
    );
}

#[test]
fn issue_gnet_03_checked_admission_and_program_specific_device_sources() {
    let (_, net) = lowered(FIB);
    let proof = net.verify().unwrap();
    for target in [compiled::Target::Cuda, compiled::Target::Metal] {
        let artifact =
            compiled::emit(&net, &proof, "fib", target, compiled::Limits::default()).unwrap();
        assert!(artifact.source().contains(compiled::KERNEL_NAME));
        assert!(artifact.source().contains("switch(fr.pc)"));
        assert!(!artifact.source().contains("opcode"));
        assert!(!artifact.source().contains("goto "));
        assert_eq!(artifact.params(), vec![front::Type::U32]);
        assert!(artifact.frame_bytes() <= 128 * 1024);
    }
    for limits in [
        compiled::Limits {
            values: 0,
            ..compiled::Limits::default()
        },
        compiled::Limits {
            states: 0,
            ..compiled::Limits::default()
        },
        compiled::Limits {
            source_bytes: 100,
            ..compiled::Limits::default()
        },
        compiled::Limits {
            frame_bytes: 1,
            ..compiled::Limits::default()
        },
    ] {
        assert_eq!(
            compiled::emit(&net, &proof, "fib", compiled::Target::Cpu, limits).unwrap_err(),
            compiled::Error::Limit
        );
    }
    let mut changed = net.clone();
    changed.agents[0].next.position += 1;
    assert!(matches!(
        compiled::emit(
            &changed,
            &proof,
            "fib",
            compiled::Target::Cpu,
            compiled::Limits::default()
        ),
        Err(compiled::Error::Boundary(_))
    ));
    let (_, net) = lowered(
        "fn unused() -> i32 { let xs: Array<i32> = [1]; return xs[0u32]; } fn main() -> i32 { return 7; }",
    );
    assert_eq!(
        compiled::emit(
            &net,
            &net.verify().unwrap(),
            "main",
            compiled::Target::Cpu,
            compiled::Limits::default()
        )
        .unwrap_err(),
        compiled::Error::UnsupportedType
    );
}

#[test]
fn issue_gnet_03_recursive_device_bundle_is_derived_from_checked_public_source() {
    let (_, net) = lowered(FIB);
    let proof = net.verify().unwrap();
    let cuda = compiled::emit(
        &net,
        &proof,
        "fib",
        compiled::Target::Cuda,
        compiled::Limits::default(),
    )
    .unwrap();
    let metal = compiled::emit(
        &net,
        &proof,
        "fib",
        compiled::Target::Metal,
        compiled::Limits::default(),
    )
    .unwrap();
    let cpu = compiled::emit(
        &net,
        &proof,
        "fib",
        compiled::Target::Cpu,
        compiled::Limits::default(),
    )
    .unwrap();
    let limits = oracle::Limits {
        fuel: 10_000,
        transitions: 100_000,
        trace_events: 10_000,
        ..oracle::Limits::default()
    };
    let expected: Vec<_> = (0..9)
        .map(|n| {
            expected(
                &oracle::observe_call(&net, &proof, "fib", &[eval::Value::U32(n)], limits).unwrap(),
            )
        })
        .collect();
    assert!(expected.iter().all(|words| words[0] == 0 && words[7] > 0));
    if let Some(dir) = std::env::var_os("GNET_BUNDLE_OUT") {
        let dir = PathBuf::from(dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("recursive.inv"), FIB).unwrap();
        fs::write(dir.join("recursive.cu"), cuda.source()).unwrap();
        fs::write(dir.join("recursive.metal"), metal.source()).unwrap();
        fs::write(
            dir.join("recursive.cpp"),
            format!("{}\n{DRIVER}", cpu.source()),
        )
        .unwrap();
        fs::write(dir.join("expected.json"),serde_json::to_vec_pretty(&serde_json::json!({"contract":1,"kernel":compiled::KERNEL_NAME,"count":9,"input":[0,1,2,3,4,5,6,7,8],"config":[9,limits.fuel,limits.transitions,limits.trace_events],"output_words":8,"event_words":7,"max_frames":64,"frame_bytes":cuda.frame_bytes(),"expected":expected})).unwrap()).unwrap();
    }
}
