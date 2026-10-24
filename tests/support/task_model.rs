//! P4b helpers: compile a task-region program, build its CPU model (the
//! emitted scheduler, run kernel and state machines compiled as C++ under
//! ASan and UBSan), and compare its observations with the CPU net oracle.
#![allow(dead_code)]

use std::{
    fs,
    io::Read,
    path::PathBuf,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use invar::{
    eval::{EvalError, Value},
    general_net::{
        Net, Proof,
        compiled::Target,
        normal_form,
        oracle,
        task::{self, Mutation, Plan, Profile, Program, Schedule},
    },
};

#[path = "normal_form.rs"]
pub mod normal_form_support;

pub struct Admitted {
    pub net: Net,
    pub proof: Proof,
    pub program: Program,
}

pub fn admit(source: &str) -> Result<Admitted, task::Refusal> {
    let (net, proof) = normal_form_support::compile(source);
    let form = normal_form::extract(&net, &proof).expect("extract");
    let program = task::admit(&net, &proof, &form, "run")?;
    Ok(Admitted {
        net,
        proof,
        program,
    })
}

/// Lane step budget of the test plans: a lane past it reports the
/// `RESOURCE_LIMIT` tool failure, which (like the oracle's fuel limit) skips
/// the case.
pub const TEST_BUDGET: u32 = 1 << 20;

pub fn cpu_plan(program: &Program) -> Plan {
    Plan {
        budget: TEST_BUDGET,
        ..task::plan(program, &Profile::CPU_MODEL)
    }
}

/// The CPU model of the cascade schedule.
pub fn cascade_plan(program: &Program) -> Plan {
    Plan {
        budget: TEST_BUDGET,
        ..task::plan(program, &Profile::CPU_MODEL_CASCADE)
    }
}

/// A plan whose arena fills after a few expansions (arena-full fallbacks).
/// The cascade stops forking once `width` nodes exist, so its tight plan
/// keeps the width past the arena: forks fill the arena and fall back.
pub fn tight(plan: Plan) -> Plan {
    let width = match plan.schedule {
        Schedule::Staged => 4,
        Schedule::Cascade => 64,
    };
    Plan {
        width,
        arena: 10,
        ..plan
    }
}

/// The four CPU-model plans: both schedules, wide and tight.
pub fn cpu_plans(program: &Program) -> [(&'static str, Plan); 4] {
    [
        ("wide", cpu_plan(program)),
        ("tight", tight(cpu_plan(program))),
        ("cascade", cascade_plan(program)),
        ("cascade-tight", tight(cascade_plan(program))),
    ]
}

/// Whether an observation is the lane step budget's tool failure.
pub fn budget_exhausted(result: &Result<Value, EvalError>) -> bool {
    matches!(
        result,
        Err(EvalError::Tool {
            code: "RESOURCE_LIMIT",
            ..
        })
    )
}

pub fn oracle_limits() -> oracle::Limits {
    oracle::Limits {
        fuel: 50_000_000,
        transitions: 500_000_000,
        trace_events: 1 << 24,
        array_elements: 1 << 20,
    }
}

/// Run a test body on a 64 MiB thread: the reference evaluator overflows a
/// 2 MiB Debug test thread before its 64-frame limit (see `gnet_agree.rs`).
pub fn with_stack<T: Send>(body: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn_scoped(scope, body)
            .unwrap()
            .join()
            .unwrap()
    })
}

/// The oracle's observation of `run(args)`; `None` when it hit one of its own
/// limits (fuel, or a tool `RESOURCE_LIMIT` such as the trace budget), so the
/// case has no reference observation.
pub fn observe(admitted: &Admitted, args: &[i32]) -> Option<Result<Value, EvalError>> {
    let values: Vec<Value> = args.iter().map(|x| Value::I32(*x)).collect();
    let observed = oracle::observe_call(
        &admitted.net,
        &admitted.proof,
        "run",
        &values,
        oracle_limits(),
    )
    .expect("oracle");
    match observed.semantic.result {
        Err(EvalError::EvaluationLimit { .. }) => None,
        ref other if budget_exhausted(other) => None,
        other => Some(other),
    }
}

/// Observation equality: values and traps exactly, tool failures by code.
pub fn same(a: &Result<Value, EvalError>, b: &Result<Value, EvalError>) -> bool {
    match (a, b) {
        (Err(EvalError::Tool { code: x, .. }), Err(EvalError::Tool { code: y, .. })) => x == y,
        _ => a == b,
    }
}

/// One CPU-model line: root observation words and scheduler counters.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub done: u32,
    pub words: [u32; 4],
    pub rounds: u32,
    pub tasks: u32,
    pub passes: u32,
    pub arena: u32,
}

pub struct Model {
    dir: PathBuf,
    exe: PathBuf,
}

impl Drop for Model {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

const WALL: Duration = Duration::from_secs(240);

fn scratch(name: &str) -> PathBuf {
    let root = std::env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = root.join(format!("gnet-task-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run `program args` with a wall deadline; return (success, stdout, stderr).
fn bounded(program: &PathBuf, args: &[PathBuf]) -> (bool, String, String) {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let drain = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut text = String::new();
            let _ = pipe.read_to_string(&mut text);
            text
        })
    };
    let out = drain(Box::new(child.stdout.take().unwrap()));
    let err = drain(Box::new(child.stderr.take().unwrap()));
    let begun = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if begun.elapsed() >= WALL {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let (out, err) = (out.join().unwrap(), err.join().unwrap());
    (status.is_some_and(|s| s.success()), out, err)
}

/// Emit the CPU model of `admitted` under `plan` with `mutation` and build it.
pub fn build(name: &str, admitted: &Admitted, plan: &Plan, mutation: Mutation) -> Model {
    let source =
        task::emit(&admitted.program, Target::Cpu, plan, mutation).expect("emit cpu model");
    let dir = scratch(name);
    let cpp = dir.join("model.cpp");
    let exe = dir.join("model");
    fs::write(&cpp, &source.text).unwrap();
    let compiler = PathBuf::from(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()));
    let args: Vec<PathBuf> = [
        "-std=c++17",
        "-O1",
        "-fno-omit-frame-pointer",
        "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all",
        "-w",
    ]
    .iter()
    .map(PathBuf::from)
    .chain([cpp, "-o".into(), exe.clone()])
    .collect();
    let (ok, _, err) = bounded(&compiler, &args);
    assert!(ok, "{name}: compiler failed\n{err}");
    Model { dir, exe }
}

impl Model {
    /// Run every case (entry arguments) through the model.
    pub fn run(&self, cases: &[Vec<i32>]) -> Vec<Line> {
        let input = self.dir.join("cases.txt");
        let text: String = cases
            .iter()
            .map(|case| {
                let words: Vec<String> = case.iter().map(i32::to_string).collect();
                words.join(" ") + "\n"
            })
            .collect();
        fs::write(&input, text).unwrap();
        let (ok, out, err) = bounded(&self.exe, &[input]);
        assert!(ok, "model failed\n{err}");
        let lines: Vec<Line> = out
            .lines()
            .map(|line| {
                let w: Vec<u32> = line
                    .split_whitespace()
                    .map(|x| x.parse().unwrap())
                    .collect();
                Line {
                    done: w[0],
                    words: [w[1], w[2], w[3], w[4]],
                    rounds: w[5],
                    tasks: w[6],
                    passes: w[7],
                    arena: w[8],
                }
            })
            .collect();
        assert_eq!(lines.len(), cases.len(), "model output\n{out}\n{err}");
        lines
    }
}

/// Tally of one differential run.
#[derive(Default, Debug)]
pub struct Tally {
    pub cases: u64,
    pub values: u64,
    pub traps: u64,
    pub depth_limits: u64,
    pub skipped: u64,
    pub budget: u64,
    pub fallbacks: u64,
    pub mismatches: Vec<String>,
}

impl Tally {
    pub fn add(&mut self, other: Tally) {
        self.cases += other.cases;
        self.values += other.values;
        self.traps += other.traps;
        self.depth_limits += other.depth_limits;
        self.skipped += other.skipped;
        self.budget += other.budget;
        self.fallbacks += other.fallbacks;
        self.mismatches.extend(other.mismatches);
    }
}

/// The oracle's observation of every case (`None`: the oracle hit a limit).
pub type Expected = Vec<Option<Result<Value, EvalError>>>;

pub fn expected(admitted: &Admitted, cases: &[Vec<i32>]) -> Expected {
    cases.iter().map(|case| observe(admitted, case)).collect()
}

/// Compare the model with the oracle on every case.
pub fn differential(
    name: &str,
    admitted: &Admitted,
    plan: &Plan,
    mutation: Mutation,
    cases: &[Vec<i32>],
) -> Tally {
    differential_against(
        name,
        admitted,
        plan,
        mutation,
        cases,
        &expected(admitted, cases),
    )
}

/// [`differential`] against precomputed oracle observations.
pub fn differential_against(
    name: &str,
    admitted: &Admitted,
    plan: &Plan,
    mutation: Mutation,
    cases: &[Vec<i32>],
    expected: &Expected,
) -> Tally {
    let model = build(name, admitted, plan, mutation);
    let lines = model.run(cases);
    let mut tally = Tally::default();
    for ((case, line), expected) in cases.iter().zip(lines).zip(expected) {
        let Some(expected) = expected else {
            tally.skipped += 1;
            continue;
        };
        let ours = task::decode(&admitted.program, line.words);
        if budget_exhausted(&ours) && line.done == 1 {
            tally.budget += 1;
            continue;
        }
        tally.cases += 1;
        match expected {
            Ok(_) => tally.values += 1,
            Err(EvalError::Trap { .. }) => tally.traps += 1,
            Err(_) => tally.depth_limits += 1,
        }
        if line.arena > plan.arena {
            tally.fallbacks += 1;
        }
        if line.done != 1 || !same(&ours, expected) {
            tally.mismatches.push(format!(
                "{name} {case:?}: model {ours:?} (done {}), oracle {expected:?}",
                line.done
            ));
        }
    }
    tally
}

/// GNET-05 v2 manifest inputs of the C1 family (index 4 is the large one).
pub const C1_INPUTS: [[[i32; 2]; 5]; 12] = [
    [[1, 187], [4, 654], [6, 361], [6, 104], [14, 361]],
    [[1, 510], [3, 928], [5, 586], [5, 837], [15, 928]],
    [[0, 278], [3, 611], [6, 614], [6, 49], [13, 611]],
    [[3, 0], [27, 3], [185, 6], [328, 8], [256, 24]],
    [[0, 68], [3, 213], [5, 158], [6, 206], [9, 206]],
    [[0, 126], [3, 876], [6, 454], [5, 32], [14, 32]],
    [[1, 474], [4, 705], [5, 419], [6, 344], [11, 344]],
    [[3, 1], [41, 2], [287, 6], [383, 9], [2048, 24]],
    [[1, 0], [52, 2], [322, 6], [302, 7], [192, 24]],
    [[1, 0], [52, 4], [275, 5], [317, 8], [256, 24]],
    [[2, 0], [40, 3], [303, 6], [380, 8], [1536, 24]],
    [[1, 0], [22, 2], [298, 6], [206, 7], [3072, 24]],
];

/// Two leaves that each run an `m`-iteration serial loop under one fork: with a
/// small cascade poll limit, idle workers wait on them far longer than the
/// limit. The comparison join keeps the region residual, not flat.
pub const LONG_LEAVES: &str = "fn t(n: i32, s: i32, m: i32) -> i32 {
    if n <= 0 { var x = s; var i = 0; while i < m { x = (x * 3 + 1) % 10007; i = i + 1; } return x; }
    let a = t(n - 1, (s * 7 + 3) % 1009, m);
    let b = t(n - 1, (s * 5 + 11) % 1009, m);
    if a > b { return (a * 2 + b) % 10007; }
    return (b * b % 10007 + a) % 10007;
}
fn run(a: i32, b: i32, m: i32) -> i32 { return t(a, b % 1009, m); }
fn main() -> i32 { return run(1, 5, 10); }
";

/// Leaf iterations for the device tests: past the net oracle's limits, so the
/// device result is checked against [`long_leaves_expected`].
pub const LONG_STEPS: i32 = 20_000_000;

/// `run(n, s, m)` of [`LONG_LEAVES`] for non-negative `s` (no overflow: every
/// intermediate stays below 10007^2).
pub fn long_leaves_expected(n: i32, s: i32, m: i32) -> i32 {
    fn t(n: i32, s: i32, m: i32) -> i32 {
        if n <= 0 {
            let mut x = s;
            for _ in 0..m {
                x = (x * 3 + 1) % 10007;
            }
            return x;
        }
        let a = t(n - 1, (s * 7 + 3) % 1009, m);
        let b = t(n - 1, (s * 5 + 11) % 1009, m);
        if a > b {
            (a * 2 + b) % 10007
        } else {
            (b * b % 10007 + a) % 10007
        }
    }
    t(n, s % 1009, m)
}

pub fn c1_source(index: usize) -> String {
    // The package root, or the repository root when an `invar-tools` bench includes this file.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .ancestors()
        .find(|dir| dir.join("tests/net_task_fixtures").is_dir())
        .unwrap();
    let path = root.join(format!(
        "tests/net_task_fixtures/c1/gnet05-forkjoin-{index:02}.inv"
    ));
    fs::read_to_string(path).unwrap()
}

pub fn c1(index: usize) -> Admitted {
    admit(&c1_source(index)).unwrap_or_else(|r| panic!("C1-{index:02}: {r:?}"))
}

/// SplitMix64 (deterministic, no new dependency).
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
    pub fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len() as u64) as usize]
    }
    /// Mostly small values, sometimes extreme (overflow), sometimes zero.
    pub fn i32(&mut self) -> i32 {
        match self.below(10) {
            0 => [i32::MIN, i32::MAX, -1, 0, 1][self.below(5) as usize],
            1 | 2 => self.range(-(1 << 30), 1 << 30) as i32,
            _ => self.range(-50, 50) as i32,
        }
    }
}

/// Manifest inputs plus random ones around them, with extreme values that
/// trap in the entry or reach the depth limit.
pub fn c1_cases(index: usize, rng: &mut Rng, extra: usize) -> Vec<Vec<i32>> {
    let inputs = C1_INPUTS[index];
    let mut cases: Vec<Vec<i32>> = inputs.iter().map(|x| x.to_vec()).collect();
    let (max_a, max_b) = inputs[..4]
        .iter()
        .fold((0, 0), |(a, b), x| (a.max(x[0]), b.max(x[1])));
    for _ in 0..extra {
        let a = match rng.below(12) {
            0 => rng.i32(),
            _ => rng.range(-3, i64::from(max_a)) as i32,
        };
        let b = match rng.below(12) {
            0 => rng.i32(),
            _ => rng.range(-3, i64::from(max_b)) as i32,
        };
        cases.push(vec![a, b]);
    }
    cases
}

/// A random fork-join program over `t(n, s)`: nested forks of two or three
/// siblings (recursive and plain), trap-heavy sibling arguments, leaves and
/// combines, an optional continuation call, and an optional deep chain that
/// reaches the call depth limit.
pub fn generated(rng: &mut Rng) -> String {
    let arg = [
        "(s * 7 + 3) % 1009",
        "s / (s % 7)",
        "s + 2147483000",
        "s * 3 - n",
        "(s % 1009) * 2000000",
        "s - n * 5",
        "s % (n + 4)",
        "-s",
    ];
    let leaf = [
        "return x / (y + 3);",
        "return (x % 101) * (x % 101) - y;",
        "if x % 5 == 0 { return x * 100000; } return x + y;",
        "var a = x % 50; var i = 0; while i < x % 4 { a = a * 3 + i; i = i + 1; } return a;",
        "return x - y * 7;",
    ];
    let combine = [
        "return a0 * a1 + a2;",
        "return (a0 - a1) * 3 + a2;",
        "return a0 / (a1 % 5) + a2;",
        "if a0 > a1 { return a0 * 2 - a1 + a2; } return a1 - a0 - a2;",
        "return (a0 % 10007) * 31 + (a1 % 10007) - a2;",
        "return a0 - a1 * 2 + a2 * 3;",
    ];
    let deep = rng.below(4) == 0;
    let second = if deep {
        "0".to_owned()
    } else {
        format!("n - {}", rng.range(1, 2))
    };
    // A deep chain keeps every other call shallow (linear, not exponential).
    let other = if deep { "0" } else { "n - 2" };
    let third = match rng.below(3) {
        0 => "let a2 = 0;".to_owned(),
        1 => format!("let a2 = leaf({}, n);", rng.pick(&arg)),
        _ => format!("let a2 = t({other}, {});", rng.pick(&arg)),
    };
    let continuation = if rng.below(2) == 0 {
        format!(
            "if (a0 + a1) % 3 == 0 {{ let e = t({other}, {}); return e - a0 + a2; }}",
            rng.pick(&arg)
        )
    } else {
        String::new()
    };
    let modulus = if deep { 80 } else { rng.range(5, 8) };
    format!(
        "fn leaf(x: i32, y: i32) -> i32 {{ {leaf} }}\n\
         fn t(n: i32, s: i32) -> i32 {{\n\
         if n <= 0 {{ return leaf(s, n); }}\n\
         let a0 = t(n - 1, {a0});\n\
         let a1 = t({second}, {a1});\n\
         {third}\n\
         {continuation}\n\
         {combine}\n\
         }}\n\
         fn run(a: i32, b: i32) -> i32 {{ return t(a % {modulus}, b); }}\n",
        leaf = rng.pick(&leaf),
        a0 = rng.pick(&arg),
        a1 = rng.pick(&arg),
        combine = rng.pick(&combine),
    )
}

pub fn generated_cases(rng: &mut Rng, count: usize) -> Vec<Vec<i32>> {
    (0..count).map(|_| vec![rng.i32(), rng.i32()]).collect()
}
