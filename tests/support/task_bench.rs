//! P4b bench support: the handwritten B-form baseline source, sample files
//! and the D5 statistics (median of paired ratios, one-sided hierarchical
//! paired percentile bootstrap U95, as in the P4a spike).
#![allow(dead_code)]

use std::{fs, path::PathBuf, time::Instant};

use invar::{
    eval::Value,
    general_net::task::{self, Plan, Profile, Program, Schedule, host::Outcome},
};
use serde_json::{Value as Json, json};

use super::model::{Admitted, C1_INPUTS, Rng, c1, observe};

pub const PROGRAMS: usize = 12;
pub const TUNING: [usize; 6] = [0, 2, 3, 4, 7, 8];
pub const WARMUP: usize = 10;
pub const PAIRS: usize = 30;
pub const SESSIONS: usize = 3;
pub const RESAMPLES: usize = 10_000;

/// D5 residual tier: median and U95 limits (resident, host).
pub const RESIDENT: (f64, f64) = (3.0, 3.15);
pub const HOST: (f64, f64) = (2.0, 2.10);

pub fn out_dir() -> PathBuf {
    let dir = std::env::var_os("P4B_OUT")
        .map(PathBuf::from)
        .expect("set P4B_OUT to the bench output directory");
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Radix bits of program `index` (2 for three-way forks, 1 for binary).
pub fn radix_bits(index: usize) -> u32 {
    if matches!(index, 4 | 7 | 9) { 1 } else { 2 }
}

/// Replay depths the pilot may choose for program `index`.
pub fn depths(index: usize) -> Vec<u32> {
    if radix_bits(index) == 1 {
        (2..=16).step_by(2).collect()
    } else {
        (1..=8).collect()
    }
}

pub fn groups(index: usize, depth: u32) -> i32 {
    let lanes = 1u64 << (depth * radix_bits(index));
    lanes.div_ceil(256) as i32
}

const SOURCE: &str = include_str!("../../bench/task-region-v01/c1_handwritten.inc");

pub fn metal_baseline(index: usize) -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n#define PROGRAM {index}\n\
         #define HW_FN inline\n#define HW_TH thread\n#define HW_I64 long\n#define HW_SHARED threadgroup\n\
         #define HW_BARRIER threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device)\n#define HW_IDS\n\
         #define HW_ARGS device const int* params [[buffer(0)]],device O* part [[buffer(1)]],device uint* stops [[buffer(2)]],device P* paths [[buffer(3)]],device O* out [[buffer(4)]],uint TID [[thread_index_in_threadgroup]],uint GROUP [[threadgroup_position_in_grid]]\n\
         #define HW_KERNEL_B kernel void hw_b(HW_ARGS)\n#define HW_KERNEL_TOP kernel void hw_top(HW_ARGS)\n{SOURCE}"
    )
}

pub fn cuda_baseline(index: usize) -> String {
    format!(
        "typedef unsigned int uint;\n#define PROGRAM {index}\n\
         #define HW_FN __device__ inline\n#define HW_TH\n#define HW_I64 long long\n#define HW_SHARED __shared__\n\
         #define HW_BARRIER __syncthreads()\n#define HW_IDS uint TID=threadIdx.x;uint GROUP=blockIdx.x;(void)GROUP;\n\
         #define HW_ARGS const int* params,O* part,uint* stops,P* paths,O* out\n\
         #define HW_KERNEL_B extern \"C\" __global__ void hw_b(HW_ARGS)\n#define HW_KERNEL_TOP extern \"C\" __global__ void hw_top(HW_ARGS)\n{SOURCE}"
    )
}

/// One timed pair of a session.
#[derive(Clone, Debug)]
pub struct Pair {
    pub program: usize,
    pub input: usize,
    pub t_resident: f64,
    pub t_host: f64,
    pub hw_resident: f64,
    pub hw_host: f64,
}

pub fn pair_json(p: &Pair, session: usize) -> Json {
    json!({
        "session": session, "program": p.program, "input": p.input,
        "t_resident_s": p.t_resident, "t_host_s": p.t_host,
        "hw_resident_s": p.hw_resident, "hw_host_s": p.hw_host,
    })
}

pub fn read_pairs(path: &PathBuf) -> Vec<(usize, Pair)> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| {
            let v: Json = serde_json::from_str(line).unwrap();
            let f = |k: &str| v[k].as_f64().unwrap();
            let u = |k: &str| v[k].as_u64().unwrap() as usize;
            (
                u("session"),
                Pair {
                    program: u("program"),
                    input: u("input"),
                    t_resident: f("t_resident_s"),
                    t_host: f("t_host_s"),
                    hw_resident: f("hw_resident_s"),
                    hw_host: f("hw_host_s"),
                },
            )
        })
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Median of all paired ratios and its one-sided U95: resample sessions,
/// then pairs within each chosen session.
pub fn ratio_stats(sessions: &[Vec<f64>], seed: u64) -> (f64, f64) {
    let mut all: Vec<f64> = sessions.iter().flatten().copied().collect();
    let estimate = median(&mut all);
    let mut rng = Rng(seed);
    let mut boot = Vec::with_capacity(RESAMPLES);
    for _ in 0..RESAMPLES {
        let mut sample = Vec::with_capacity(all.len());
        for _ in 0..sessions.len() {
            let s = &sessions[rng.below(sessions.len() as u64) as usize];
            for _ in 0..s.len() {
                sample.push(s[rng.below(s.len() as u64) as usize]);
            }
        }
        boot.push(median(&mut sample));
    }
    boot.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (estimate, boot[(RESAMPLES * 95) / 100])
}

pub fn verdict(stats: (f64, f64), limits: (f64, f64)) -> &'static str {
    if stats.0 <= limits.0 && stats.1 <= limits.1 {
        "PASS"
    } else {
        "MISS"
    }
}

/// A prepared T call and a prepared handwritten call (`[a, b, depth, groups]`
/// to `([value, trap], resident seconds)`).
pub type TCall = Box<dyn FnMut(&[Value]) -> Result<Outcome, String>>;
pub type HwCall = Box<dyn FnMut([i32; 4]) -> ([i32; 2], f64)>;

pub struct Backend<'a> {
    pub name: &'static str,
    pub profile: Profile,
    pub t: &'a mut dyn FnMut(&Program, &Plan) -> TCall,
    pub hw: &'a mut dyn FnMut(usize) -> HwCall,
}

fn expected(admitted: &Admitted, input: [i32; 2]) -> i32 {
    match observe(admitted, &input) {
        Some(Ok(Value::I32(v))) => v,
        other => panic!("C1 input {input:?} is not a value: {other:?}"),
    }
}

/// One handwritten call: (resident s, host s), checked against `want`.
fn hw_call(hw: &mut HwCall, index: usize, input: [i32; 2], depth: u32, want: i32) -> (f64, f64) {
    let begun = Instant::now();
    let (out, resident) = hw([input[0], input[1], depth as i32, groups(index, depth)]);
    let host = begun.elapsed().as_secs_f64();
    assert_eq!(
        out,
        [want, 0],
        "handwritten C1-{index:02} {input:?} D={depth}"
    );
    (resident, host)
}

/// One T call: (resident s, host s), checked against `want`.
fn t_call(t: &mut TCall, index: usize, input: [i32; 2], want: i32) -> (f64, f64) {
    let args = [Value::I32(input[0]), Value::I32(input[1])];
    let begun = Instant::now();
    let outcome = t(&args).expect("task call");
    let host = begun.elapsed().as_secs_f64();
    assert_eq!(
        outcome.result,
        Ok(Value::I32(want)),
        "T C1-{index:02} {input:?}"
    );
    (outcome.gpu_seconds, host)
}

fn median_of(mut v: Vec<f64>) -> f64 {
    median(&mut v)
}

/// Expansion widths the pilot calibrates over: below the resident lane count
/// (subtrees become serial tasks early) up to well past it.
pub const WIDTHS: [u32; 5] = [64, 256, 1024, 4096, 16384];

/// Cascade grid sizes the pilot calibrates over, as multiples of the
/// profile's lanes (the staged schedule has no persistent grid).
pub fn lane_choices(profile: &Profile) -> Vec<u32> {
    match profile.schedule {
        Schedule::Staged => vec![profile.lanes],
        Schedule::Cascade => vec![profile.lanes / 2, profile.lanes, profile.lanes * 2],
    }
}

/// T's plan for `program` with the frozen expansion width and lanes.
fn frozen_plan(program: &Program, profile: &Profile, width: u32, lanes: u32) -> Plan {
    task::plan(
        program,
        &Profile {
            width,
            lanes,
            ..*profile
        },
    )
}

/// Freeze the handwritten depth per program and input and T's expansion
/// width (tuning split only).
pub fn pilot(backend: Backend) {
    let mut depths_json = Vec::new();
    for (index, inputs) in C1_INPUTS.iter().enumerate() {
        let admitted = c1(index);
        let mut hw = (backend.hw)(index);
        let mut chosen = Vec::new();
        for &input in inputs {
            let want = expected(&admitted, input);
            let best = depths(index)
                .into_iter()
                .map(|d| {
                    let times: Vec<f64> = (0..5)
                        .map(|_| hw_call(&mut hw, index, input, d, want).0)
                        .collect();
                    (median_of(times), d)
                })
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap();
            chosen.push(best.1);
        }
        eprintln!("C1-{index:02} handwritten depths {chosen:?}");
        depths_json.push(json!(chosen));
    }
    let mut scores = Vec::new();
    for lanes in lane_choices(&backend.profile) {
        for width in WIDTHS {
            let (mut log_sum, mut n) = (0.0, 0.0);
            for &index in &TUNING {
                let admitted = c1(index);
                let plan = frozen_plan(&admitted.program, &backend.profile, width, lanes);
                let mut t = (backend.t)(&admitted.program, &plan);
                for input in C1_INPUTS[index] {
                    let want = expected(&admitted, input);
                    let times: Vec<f64> = (0..5)
                        .map(|_| t_call(&mut t, index, input, want).0)
                        .collect();
                    log_sum += median_of(times).ln();
                    n += 1.0;
                }
            }
            let mean = (log_sum / n).exp();
            eprintln!(
                "T lanes {lanes} width {width}: geometric mean resident {:.1} us",
                mean * 1e6
            );
            scores.push((mean, width, lanes));
        }
    }
    let &(_, width, lanes) = scores
        .iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap();
    let pilot = json!({
        "handwritten_depths": depths_json, "t_width": width, "t_lanes": lanes,
        "t_schedule": format!("{:?}", backend.profile.schedule),
    });
    fs::write(
        out_dir().join(format!("pilot-{}.json", backend.name)),
        serde_json::to_string_pretty(&pilot).unwrap(),
    )
    .unwrap();
}

/// One scored session: warmup and interleaved pairs per program and input.
pub fn session(backend: Backend, session: usize) {
    let dir = out_dir();
    let pilot: Json = serde_json::from_str(
        &fs::read_to_string(dir.join(format!("pilot-{}.json", backend.name))).unwrap(),
    )
    .unwrap();
    let width = pilot["t_width"].as_u64().unwrap() as u32;
    let lanes = pilot["t_lanes"].as_u64().unwrap() as u32;
    let mut rng = Rng(0x5e55_1010 + session as u64);
    let mut lines = String::new();
    for index in 0..PROGRAMS {
        let admitted = c1(index);
        let plan = frozen_plan(&admitted.program, &backend.profile, width, lanes);
        let mut t = (backend.t)(&admitted.program, &plan);
        let mut hw = (backend.hw)(index);
        for (k, input) in C1_INPUTS[index].into_iter().enumerate() {
            let depth = pilot["handwritten_depths"][index][k].as_u64().unwrap() as u32;
            let want = expected(&admitted, input);
            for i in 0..WARMUP + PAIRS {
                let (tt, hh) = if rng.below(2) == 0 {
                    let tt = t_call(&mut t, index, input, want);
                    (tt, hw_call(&mut hw, index, input, depth, want))
                } else {
                    let hh = hw_call(&mut hw, index, input, depth, want);
                    (t_call(&mut t, index, input, want), hh)
                };
                if i >= WARMUP {
                    let pair = Pair {
                        program: index,
                        input: k,
                        t_resident: tt.0,
                        t_host: tt.1,
                        hw_resident: hh.0,
                        hw_host: hh.1,
                    };
                    lines.push_str(&pair_json(&pair, session).to_string());
                    lines.push('\n');
                }
            }
        }
        eprintln!("{} session {session}: C1-{index:02} done", backend.name);
    }
    fs::write(
        dir.join(format!("session-{}-{session}.jsonl", backend.name)),
        lines,
    )
    .unwrap();
}

/// Per-cell medians and U95 against the D5 residual limits.
pub fn report(name: &str) {
    let dir = out_dir();
    let mut pairs = Vec::new();
    for s in 0..SESSIONS {
        pairs.extend(read_pairs(&dir.join(format!("session-{name}-{s}.jsonl"))));
    }
    let mut cells = Vec::new();
    let mut table = String::from(
        "| Program | Input | T/HW resident | T/HW host | T resident us | HW resident us |\n|---|---|---|---|---|---|\n",
    );
    let (mut resident_pass, mut host_pass, mut total) = (0, 0, 0);
    for (index, inputs) in C1_INPUTS.iter().enumerate() {
        for (k, &input) in inputs.iter().enumerate() {
            let cell: Vec<&(usize, Pair)> = pairs
                .iter()
                .filter(|(_, p)| p.program == index && p.input == k)
                .collect();
            let by_session = |f: &dyn Fn(&Pair) -> f64| -> Vec<Vec<f64>> {
                (0..SESSIONS)
                    .map(|s| {
                        cell.iter()
                            .filter(|(t, _)| *t == s)
                            .map(|(_, p)| f(p))
                            .collect()
                    })
                    .collect()
            };
            let seed = 0xb00f_0000 + (index * 8 + k) as u64;
            let resident = ratio_stats(&by_session(&|p| p.t_resident / p.hw_resident), seed);
            let host = ratio_stats(&by_session(&|p| p.t_host / p.hw_host), seed + 1);
            let med =
                |f: &dyn Fn(&Pair) -> f64| median_of(cell.iter().map(|(_, p)| f(p)).collect());
            let (rv, hv) = (verdict(resident, RESIDENT), verdict(host, HOST));
            total += 1;
            resident_pass += usize::from(rv == "PASS");
            host_pass += usize::from(hv == "PASS");
            table.push_str(&format!(
                "| C1-{index:02} | ({},{}) | {:.3} [{:.3}] {rv} | {:.3} [{:.3}] {hv} | {:.1} | {:.1} |\n",
                input[0],
                input[1],
                resident.0,
                resident.1,
                host.0,
                host.1,
                med(&|p| p.t_resident) * 1e6,
                med(&|p| p.hw_resident) * 1e6
            ));
            cells.push(json!({
                "program": index, "input": input, "resident": [resident.0, resident.1, rv],
                "host": [host.0, host.1, hv],
            }));
        }
    }
    table.push_str(&format!(
        "\n{name}: resident {resident_pass}/{total} PASS; host {host_pass}/{total} PASS.\n"
    ));
    eprintln!("{table}");
    fs::write(dir.join(format!("report-{name}.md")), &table).unwrap();
    fs::write(
        dir.join(format!("report-{name}.json")),
        serde_json::to_string_pretty(&json!(cells)).unwrap(),
    )
    .unwrap();
}
