#![forbid(unsafe_code)]
//! P4a: the path-sum lowering against the CPU net oracle (observation contract
//! v2: the value or the first source-order trap). Every case runs the lowered
//! kernel's guard, launch and publication; a fast answer must equal the
//! oracle bit for bit, a trapping or tool-limited input must never get one,
//! and each named mutation must be caught. The CPU model re-checks every
//! narrow operator, so an unsound licence aborts instead of agreeing.

#[path = "support/normal_form.rs"]
mod support;

use std::collections::BTreeMap;

use invar::{
    eval::{EvalError, Value},
    general_net::{
        Net, Proof,
        law_collective::Representation,
        normal_form::{
            Annotation,
            certificate::{self, Rejection},
        },
        oracle,
        path_sum::{
            self, CostModel, Launcher, Lowering, Mutation, Outcome, Plan, Scope, Shape,
            device::CpuModel,
        },
        regions,
    },
};

/// SplitMix64: a deterministic generator (no new dependency).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
}

const GATE: [&str; 4] = ["divide-00", "divide-02", "compose-00", "compose-02"];

const L9_PATHSUM: &str = "fn leaf(x: i32) -> i32 { if x % 2 == 0 { return (x / 2 + 16) % 10007; } return (x * 3 + 1) % 10007; } fn dc(lo: i32, hi: i32, d: i32) -> i32 { if hi - lo <= 5 { return leaf(lo * 4 + hi); } if d <= 0 { return leaf(hi - lo + 7); } let m1 = lo + 1 + ((hi - lo - 2) * 3) / 9; let m2 = m1 + 1 + ((hi - m1 - 2) * 3) / 6; let a = dc(lo, m1, d - 1); let b = dc(m1, m2, d - 1); let c = dc(m2, hi, d - 1); return (a * 5 + b + c * 2) % 100003; } fn run(lo: i32, a: i32, b: i32) -> i32 { return dc(lo, lo + a, b); }";
const L9_TAIL: &str = "fn w(x: i32) -> i32 { return (x * 7 + 3) % 1009; } fn t(n: i32, s: i32) -> i32 { if n <= 0 { return w(s); } let sel = (s * 29 + n * 6) % 4; if sel == 0 { let v0 = t(n - 1, (s * 14 + 60) % 1009); let v1 = t(n - 1, (s * 12 + 66) % 1009); return (v0 * 7 + v1 + s) % 10007; } if sel == 1 { return (w(s) + n) % 10007; } return t(n - 1, (s * 4 + 22) % 1009); }";
const L10_RANGE: &str = "fn leaf(x: i32) -> i32 { if x % 2 == 0 { return x * x - 7; } return x * x * 3 + 1; } fn sum(lo: i32, hi: i32) -> i32 { if hi - lo <= 1 { return leaf(lo); } let m = lo + (hi - lo) / 2; let a = sum(lo, m); let b = sum(m, hi); return a + b; }";
/// Splits may leave the first child empty (geometry fallback); the combine
/// constant depends on the node (owner-lane constants).
const GEOMETRY: &str = "fn leaf(x: i32) -> i32 { return (x * 7 + 3) % 101; } fn dc(lo: i32, hi: i32, d: i32) -> i32 { if hi - lo <= 2 { return leaf(lo * 3 + hi); } if d <= 0 { return leaf(hi - lo); } let m = lo + (hi - lo) / 2 + d % 3 - 1; let a = dc(lo, m, d - 1); let b = dc(m, hi, d - 1); return (a * 3 + b * 5 + lo % 7 + 7) % 1009; } fn run(lo: i32, a: i32, b: i32) -> i32 { return dc(lo, lo + a, b); }";

type Args = fn(&mut Rng) -> Vec<i64>;

struct Program {
    name: String,
    source: String,
    entry: &'static str,
    args: Args,
    cases: usize,
}

fn big(r: &mut Rng) -> i64 {
    match r.below(4) {
        0 => r.range(i64::from(i32::MAX) - 600, i64::from(i32::MAX)),
        1 => r.range(i64::from(i32::MIN), i64::from(i32::MIN) + 600),
        2 => r.range(1 << 28, 1 << 30),
        _ => -r.range(1 << 28, 1 << 30),
    }
}

/// `run(a, b)` of the divide and compose fixtures: every scored size, the
/// licence edges, prologue and leaf overflow traps, and raw roots.
fn divide_args(r: &mut Rng) -> Vec<i64> {
    let a = match r.below(16) {
        0 => r.range(-6, 0),
        1..=5 => r.range(0, 64),
        6..=8 => r.range(64, 700),
        9 => r.range(2900, 3100),
        10 => [0, 32, 256, 3000, 20000][r.below(5) as usize],
        11 => r.range(19_990, 20_010),
        12 | 13 => big(r),
        _ => r.range(700, 2900),
    };
    let b = match r.below(8) {
        0 => r.range(-3, 0),
        1 => big(r),
        2 => r.range(14, 31),
        _ => r.range(0, 14),
    };
    // Keep oracle cost bounded: deep and wide roots only at modest sizes.
    let b = if a > 3100 && b > 3 { r.range(-1, 3) } else { b };
    vec![a, b]
}

/// `run(a, b)` of the branch and frontier fixtures (`a` is a depth there).
fn corpus_args(r: &mut Rng) -> Vec<i64> {
    let a = match r.below(8) {
        0 => r.range(-4, 0),
        1 => big(r),
        _ => r.range(0, 16),
    };
    let b = match r.below(6) {
        0 => r.range(-4, 0),
        1 => big(r),
        _ => r.range(0, 2000),
    };
    vec![a, b]
}

fn pathsum_args(r: &mut Rng) -> Vec<i64> {
    let lo = match r.below(10) {
        0 => r.range(1 << 28, 1 << 30),
        1 => r.range(-300, -1),
        2 => big(r),
        _ => r.range(0, 400),
    };
    let a = match r.below(6) {
        0 => r.range(0, 5),
        1 => big(r),
        _ => r.range(0, 250),
    };
    vec![lo, a, r.range(-1, 6)]
}

fn tail_args(r: &mut Rng) -> Vec<i64> {
    let s = match r.below(10) {
        0 => r.range(1 << 28, 1 << 30),
        1 => r.range(-2000, -1),
        2 => big(r),
        _ => r.range(0, 1008),
    };
    vec![r.range(-1, 12), s]
}

fn range_args(r: &mut Rng) -> Vec<i64> {
    let lo = match r.below(6) {
        0 => r.range(40_000, 50_000),
        1 => r.range(-50_000, -40_000),
        2 => big(r),
        _ => r.range(-200, 200),
    };
    vec![
        lo,
        lo.saturating_add(r.range(-2, 300)).min(i64::from(i32::MAX)),
    ]
}

fn geometry_args(r: &mut Rng) -> Vec<i64> {
    let lo = match r.below(8) {
        0 => r.range(-400, -1),
        1 => big(r),
        _ => r.range(0, 400),
    };
    vec![lo, r.range(-2, 120), r.range(-1, 9)]
}

fn fixture(name: &str) -> String {
    let path = support::gnet05().join(format!("programs/gnet05-{name}.inv"));
    std::fs::read_to_string(path).expect("fixture")
}

fn programs() -> Vec<Program> {
    let mut out: Vec<Program> = GATE
        .iter()
        .map(|name| Program {
            name: (*name).to_owned(),
            source: fixture(name),
            entry: "run",
            args: divide_args,
            cases: 700,
        })
        .collect();
    for family in ["divide", "compose", "branch", "frontier"] {
        for i in 0..12 {
            let name = format!("{family}-{i:02}");
            if GATE.contains(&name.as_str()) {
                continue;
            }
            let divide = matches!(family, "divide" | "compose");
            out.push(Program {
                source: fixture(&name),
                name,
                entry: "run",
                args: if divide { divide_args } else { corpus_args },
                cases: 120,
            });
        }
    }
    for (name, source, entry, args) in [
        ("l9-pathsum", L9_PATHSUM, "run", pathsum_args as Args),
        ("l9-tail", L9_TAIL, "t", tail_args),
        ("l10-range", L10_RANGE, "sum", range_args),
        ("geometry", GEOMETRY, "run", geometry_args),
    ] {
        out.push(Program {
            name: name.to_owned(),
            source: source.to_owned(),
            entry,
            args,
            cases: 600,
        });
    }
    out
}

fn limits() -> oracle::Limits {
    oracle::Limits {
        fuel: 50_000_000,
        transitions: 500_000_000,
        trace_events: 1 << 24,
        array_elements: 1 << 20,
    }
}

/// The oracle's observation, or `None` on a tool limit (no observation).
fn observe(
    net: &Net,
    proof: &Proof,
    entry: &str,
    args: &[i64],
) -> Option<Result<Value, EvalError>> {
    let values: Vec<Value> = args.iter().map(|a| Value::I32(*a as i32)).collect();
    let observed = oracle::observe_call(net, proof, entry, &values, limits()).expect("oracle");
    match observed.semantic.result {
        Err(EvalError::Tool { .. } | EvalError::EvaluationLimit { .. }) => None,
        result => Some(result),
    }
}

#[derive(Default, Debug)]
pub struct Tally {
    pub cases: u64,
    pub traps: u64,
    pub fast: u64,
    pub fallback: BTreeMap<&'static str, u64>,
    pub mismatches: Vec<String>,
}

/// One case: the lowering's outcome against the oracle.
fn case(
    net: &Net,
    proof: &Proof,
    lowering: &Lowering,
    launcher: &mut dyn Launcher,
    model: &CostModel,
    args: &[i64],
    tally: &mut Tally,
) {
    let Some(expected) = observe(net, proof, &lowering.entry.name, args) else {
        return;
    };
    tally.cases += 1;
    tally.traps += u64::from(matches!(expected, Err(EvalError::Trap { .. })));
    // Resident scope: the cost model never refuses, so every kernel runs.
    match lowering.execute(args, model, Scope::Resident, launcher) {
        Ok(Outcome::Fast(v)) => {
            tally.fast += 1;
            if expected != Ok(Value::I32(v)) {
                tally
                    .mismatches
                    .push(format!("{args:?}: fast {v}, oracle {expected:?}"));
            }
        }
        // The exact path is the oracle itself: agreement by construction.
        Ok(Outcome::Fallback(reason)) => *tally.fallback.entry(reason).or_default() += 1,
        Err(e) => tally
            .mismatches
            .push(format!("{args:?}: launcher error {e}")),
    }
}

/// Differential run of every program that lowers; returns per-program tallies
/// and refusals.
pub fn differential(
    open: &mut dyn FnMut(&Lowering) -> Box<dyn Launcher>,
    model: &CostModel,
    scale: usize,
    mutation: Mutation,
    only: Option<&[&str]>,
) -> (BTreeMap<String, Tally>, BTreeMap<String, String>) {
    let mut tallies = BTreeMap::new();
    let mut refused = BTreeMap::new();
    for (index, program) in programs().iter().enumerate() {
        if only.is_some_and(|names| !names.contains(&program.name.as_str())) {
            continue;
        }
        let (net, proof) = support::compile(&program.source);
        let lowering = match path_sum::lower_with(&net, &proof, program.entry, mutation) {
            Ok(lowering) => lowering,
            Err(refusal) => {
                refused.insert(program.name.clone(), format!("{refusal:?}"));
                continue;
            }
        };
        let mut launcher = open(&lowering);
        let mut tally = Tally::default();
        let mut rng = Rng(0x5eed_0000 + index as u64);
        if GATE.contains(&program.name.as_str()) {
            for row in include_str!("gnet_p4a/spike-oracle.tsv").lines() {
                let col: Vec<&str> = row.split('\t').collect();
                if col[0] == program.name {
                    let args = [col[1].parse().expect("a"), col[2].parse().expect("b")];
                    case(
                        &net,
                        &proof,
                        &lowering,
                        launcher.as_mut(),
                        model,
                        &args,
                        &mut tally,
                    );
                }
            }
        }
        for _ in 0..program.cases * scale / 100 {
            let args = (program.args)(&mut rng);
            case(
                &net,
                &proof,
                &lowering,
                launcher.as_mut(),
                model,
                &args,
                &mut tally,
            );
        }
        tallies.insert(program.name.clone(), tally);
    }
    (tallies, refused)
}

fn report(label: &str, tallies: &BTreeMap<String, Tally>, refused: &BTreeMap<String, String>) {
    for (name, t) in tallies {
        eprintln!(
            "{label} {name}: cases {} traps {} fast {} fallback {:?} mismatches {}",
            t.cases,
            t.traps,
            t.fast,
            t.fallback,
            t.mismatches.len()
        );
        for m in t.mismatches.iter().take(5) {
            eprintln!("  {m}");
        }
    }
    for (name, why) in refused {
        eprintln!("{label} {name}: refused {why}");
    }
    let total: u64 = tallies.values().map(|t| t.cases).sum();
    let fast: u64 = tallies.values().map(|t| t.fast).sum();
    let traps: u64 = tallies.values().map(|t| t.traps).sum();
    eprintln!("{label} total: cases {total} traps {traps} fast {fast}");
}

fn cpu(lowering: &Lowering) -> Box<dyn Launcher> {
    Box::new(CpuModel::compile(lowering).expect("CPU model"))
}

#[test]
fn cpu_model_agrees_with_the_net_oracle() {
    let (tallies, refused) = differential(&mut cpu, &CostModel::METAL, 100, Mutation::None, None);
    report("cpu", &tallies, &refused);
    for name in GATE
        .iter()
        .chain(&["l9-pathsum", "l9-tail", "l10-range", "geometry"])
    {
        let t = tallies
            .get(*name)
            .unwrap_or_else(|| panic!("{name} refused: {refused:?}"));
        assert!(t.fast > t.cases / 3, "{name}: too few fast cases {t:?}");
        assert!(
            t.traps > 0 || name.starts_with("l10"),
            "{name}: no trapping case"
        );
    }
    let bad: Vec<_> = tallies
        .iter()
        .filter(|(_, t)| !t.mismatches.is_empty())
        .map(|(n, t)| (n, &t.mismatches[..t.mismatches.len().min(3)]))
        .collect();
    assert!(bad.is_empty(), "mismatches: {bad:?}");
    let total: u64 = tallies.values().map(|t| t.cases).sum();
    assert!(total >= 6000, "only {total} cases");
}

/// Device runs share the CPU model's assertions: zero mismatches, and the gate
/// fixtures take the fast path.
#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
fn device_agrees(
    label: &str,
    open: &mut dyn FnMut(&Lowering) -> Box<dyn Launcher>,
    model: &CostModel,
) {
    let (tallies, refused) = differential(open, model, 100, Mutation::None, None);
    report(label, &tallies, &refused);
    for name in GATE {
        let t = tallies
            .get(name)
            .unwrap_or_else(|| panic!("{name} refused"));
        assert!(t.fast > t.cases / 3, "{name}: too few fast cases {t:?}");
    }
    let bad: usize = tallies.values().map(|t| t.mismatches.len()).sum();
    assert_eq!(bad, 0, "{label}: mismatching cases");
    // The device-only reduction rewrite (the flag vote) has its own mutation.
    let names: &[&str] = &["l9-pathsum", "geometry"];
    let (tallies, _) = differential(open, model, 40, Mutation::DropLaneFlags, Some(names));
    let caught: usize = tallies.values().map(|t| t.mismatches.len()).sum();
    eprintln!("{label} mutation DropLaneFlags: {caught} mismatching cases");
    assert!(caught > 0, "{label}: DropLaneFlags was not caught");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "needs a Metal device"]
fn metal_kernels_agree_with_the_net_oracle() {
    let mut device = invar_metal::MetalDevice::open_runtime_source().expect("Metal device");
    let mut open = |l: &Lowering| -> Box<dyn Launcher> {
        Box::new(path_sum::device::Metal::new(&mut device, l).expect("Metal kernels"))
    };
    device_agrees("metal", &mut open, &CostModel::METAL);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA device"]
fn cuda_kernels_agree_with_the_net_oracle() {
    let mut device = invar_cuda::cuda::CudaDevice::open().expect("CUDA device");
    let mut open = |l: &Lowering| -> Box<dyn Launcher> {
        Box::new(path_sum::device::Cuda::new(&mut device, l).expect("CUDA kernels"))
    };
    device_agrees("cuda", &mut open, &CostModel::CUDA);
}

/// Review finding: kernels may write `stops[blockIdx.x]` (the P4a profile's
/// handwritten gate kernels do), so the stop buffer must cover every group the
/// launcher admits, not only one 256-thread reduce width.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA device"]
fn cuda_pathsum_stops_cover_every_admitted_group() {
    const SOURCE: &str = r#"
struct Args { unsigned int words[16]; };
extern "C" __global__ void mark(Args a, unsigned int* part, unsigned int* out, unsigned int* stops) {
    if (threadIdx.x == 0u) { part[blockIdx.x] = 0u; stops[blockIdx.x] = blockIdx.x + 1u; }
}
extern "C" __global__ void last(Args a, unsigned int* part, unsigned int* out, unsigned int* stops) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) { out[0] = stops[a.words[0] - 1u]; }
}
"#;
    // Past the 256-word stop buffer the launcher used to allocate.
    let groups = 300usize;
    let mut device = invar_cuda::cuda::CudaDevice::open().expect("CUDA device");
    let mut kernels = device
        .pathsum(SOURCE, &["mark", "last"], groups, 4)
        .expect("path-sum kernels");
    assert!(kernels.stop_capacity() >= groups);
    kernels
        .write_args(&(groups as u32).to_le_bytes())
        .expect("args");
    kernels
        .dispatch(&[(0, groups as u32), (1, 1)])
        .expect("dispatch");
    assert_eq!(kernels.read_out(1).expect("out"), vec![groups as u32]);
}

/// Review finding: a refused stage must not leave earlier stages in flight.
/// Every stage is validated before the first launch, so `seven` never runs.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA device"]
fn cuda_pathsum_refuses_a_bad_stage_before_any_launch() {
    const SOURCE: &str = r#"
struct Args { unsigned int words[16]; };
extern "C" __global__ void seven(Args a, unsigned int* part, unsigned int* out, unsigned int* stops) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) { out[0] = 7u; }
}
"#;
    let mut device = invar_cuda::cuda::CudaDevice::open().expect("CUDA device");
    let mut kernels = device
        .pathsum(SOURCE, &["seven"], 1, 4)
        .expect("path-sum kernels");
    assert!(matches!(
        kernels.dispatch(&[(0, 1), (1, 1)]),
        Err(invar_cuda::Error::Limit)
    ));
    assert_eq!(kernels.read_out(1).expect("out"), vec![0]);
    kernels.dispatch(&[(0, 1)]).expect("dispatch");
    assert_eq!(kernels.read_out(1).expect("out"), vec![7]);
}

/// Review finding: a shorter argument write must not keep tail words of an
/// earlier, longer one in the persistent Metal argument block.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "needs a Metal device"]
fn metal_pathsum_argument_writes_clear_the_whole_block() {
    const SOURCE: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
struct Args { uint words[16]; };\n\
kernel void second(constant Args& a [[buffer(0)]], device uint* part [[buffer(1)]], device uint* out [[buffer(2)]], device uint* stops [[buffer(3)]], uint tid [[thread_position_in_grid]]) {\n\
    if (tid == 0u) { out[0] = a.words[1]; }\n\
}\n";
    let mut device = invar_metal::MetalDevice::open_runtime_source().expect("Metal device");
    let mut kernels = device
        .pathsum(SOURCE, &["second"], 1, 4)
        .expect("path-sum kernels");
    let words = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    kernels.write_args(&words(&[1, 2])).expect("args");
    kernels.dispatch(&[(0, 1)]).expect("dispatch");
    assert_eq!(kernels.read_out(1).expect("out"), vec![2]);
    kernels.write_args(&words(&[1])).expect("args");
    kernels.dispatch(&[(0, 1)]).expect("dispatch");
    assert_eq!(kernels.read_out(1).expect("out"), vec![0]);
}

/// Programs whose generators exercise each mutation.
fn mutations() -> [(Mutation, &'static [&'static str]); 7] {
    [
        (Mutation::DropGeometry, &["geometry"]),
        (Mutation::AssumeGeometry, &["geometry"]),
        (Mutation::SkipRaw, &["l9-pathsum", "divide-00"]),
        (Mutation::DropTableGuard, &["l9-pathsum"]),
        (Mutation::WrongWeight, &["divide-00", "compose-02"]),
        (Mutation::ConstantEveryLane, &["geometry"]),
        (Mutation::WidenLicence, &["l9-pathsum", "divide-02"]),
    ]
}

#[test]
fn every_named_mutation_is_caught() {
    for (mutation, names) in mutations() {
        let (tallies, refused) =
            differential(&mut cpu, &CostModel::METAL, 40, mutation, Some(names));
        assert!(refused.is_empty(), "{mutation:?}: {refused:?}");
        let caught: usize = tallies.values().map(|t| t.mismatches.len()).sum();
        eprintln!("mutation {mutation:?}: {caught} mismatching cases");
        assert!(caught > 0, "{mutation:?} was not caught");
    }
}

#[test]
fn annotations_rederive_and_the_base_certificate_checks() {
    for name in GATE {
        let (net, proof) = support::compile(&fixture(name));
        let lowering = path_sum::lower(&net, &proof, "run").expect("lower");
        let partition = regions::partition(&net, &proof).expect("partition");
        let annotations: Vec<Annotation> = lowering
            .certificate
            .annotations
            .iter()
            .map(|a| a.annotation.clone())
            .collect();
        assert!(
            annotations
                .iter()
                .any(|a| matches!(a, Annotation::Narrow { bits: 32, .. }))
        );
        assert_eq!(
            path_sum::rederive(&partition, &lowering.certificate),
            Ok(annotations),
            "{name}"
        );
        let mut base = lowering.certificate.clone();
        base.annotations.clear();
        assert_eq!(certificate::check(&net, &proof, &base), Ok(()), "{name}");
        // Without the named analysis P1 cannot verify the range annotations;
        // with it, the certificate checks, and a widened licence is rejected.
        assert_eq!(
            certificate::check(&net, &proof, &lowering.certificate),
            Err(Rejection::Unverified)
        );
        let analysis = path_sum::RangeAnalysis(&lowering.certificate);
        assert_eq!(
            certificate::check_with_analyses(&partition, &lowering.certificate, &[&analysis]),
            Ok(()),
            "{name}"
        );
        // The reduction is consumed through P8: its checked L9 representation
        // carries the certificate's modulus, and P8 emits the ordered residue
        // reduction for both targets. The kernel keeps its fused group
        // reduction, which also carries the flag and raw words P8's one-word
        // element cannot.
        let collective = lowering.collective.as_ref().expect("P8 representation");
        assert!(matches!(
            collective.representation(),
            Representation::ResidueSum { .. }
        ));
        for cuda in [false, true] {
            let emission = lowering
                .p8_reduce(cuda)
                .expect("L9 has a P8 representation")
                .expect("P8 reduce");
            assert_eq!((emission.entry, emission.words), ("invar_law_reduce_v1", 1));
        }
        // A widened licence no longer re-derives.
        let mut wide = lowering.certificate.clone();
        for a in &mut wide.annotations {
            if let Annotation::Narrow { license, .. } = &mut a.annotation {
                for g in license.iter_mut() {
                    g.high = i64::from(i32::MAX);
                }
            }
        }
        assert_ne!(
            path_sum::rederive(&partition, &wide),
            path_sum::rederive(&partition, &lowering.certificate)
        );
        let analysis = path_sum::RangeAnalysis(&wide);
        assert!(matches!(
            certificate::check_with_analyses(&partition, &wide, &[&analysis]),
            Err(r) if r != Rejection::Unverified
        ));
    }
}

/// Review finding: a prologue product of two `u32` values can pass `i64::MAX`.
/// The checked entry evaluation must send that call to the exact fallback
/// instead of overflowing on the host.
#[test]
fn entry_prologue_overflow_takes_the_entry_fallback() {
    let source = fixture("divide-00")
        .replace(
            "fn run(a: i32, b: i32) -> i32 { return dc(45, 45 + a, b); }",
            "fn run(a: i32, b: i32, c: u32) -> i32 { let w = c * c; return dc(45, 45 + a, b); }",
        )
        .replace("return run(63, 2);", "return run(63, 2, 3u32);");
    assert!(source.contains("let w = c * c;"), "fixture shape changed");
    let (net, proof) = support::compile(&source);
    let lowering = path_sum::lower(&net, &proof, "run").expect("lower");
    let plan = |c: i64| lowering.plan(&[32, 3, c], &CostModel::METAL);
    assert!(matches!(plan(3), Plan::Launch(_)), "{:?}", plan(3));
    assert_eq!(plan(i64::from(u32::MAX)), Plan::Fallback("entry"));
}

#[test]
fn the_shape_model_fuses_small_roots_and_splits_large_ones() {
    for model in [CostModel::METAL, CostModel::CUDA] {
        assert_eq!(model.shape(1), Shape::Root);
        assert_eq!(model.shape(32), Shape::Fused);
        assert_eq!(model.shape(256), Shape::Fused);
        assert_eq!(model.shape(3000), Shape::Parallel);
        assert_eq!(model.shape(20_000), Shape::Parallel);
    }
    let (net, proof) = support::compile(&fixture("divide-00"));
    let lowering = path_sum::lower(&net, &proof, "run").expect("lower");
    let plan = |a: i64, b: i64| lowering.plan(&[a, b], &CostModel::METAL);
    let shape = |a, b| match plan(a, b) {
        Plan::Launch(launch) => Some(launch.shape),
        Plan::Fallback(_) => None,
    };
    // The P4a profile's scored inputs all lie inside the proven licence.
    assert_eq!(shape(0, 0), Some(Shape::Root));
    assert_eq!(shape(32, 3), Some(Shape::Fused));
    assert_eq!(shape(256, 8), Some(Shape::Fused));
    assert_eq!(shape(20_000, 30), Some(Shape::Parallel));
    assert_eq!(plan(i64::from(i32::MAX), 0), Plan::Fallback("entry"));
    assert_eq!(
        plan(i64::from(i32::MAX) - 100, 0),
        Plan::Fallback("licence")
    );
    eprintln!(
        "divide-00 licence {:?} max depth {}",
        lowering.licence, lowering.max_depth
    );
}

#[test]
fn run_publishes_the_oracle_observation_on_fallback() {
    let (net, proof) = support::compile(&fixture("divide-02"));
    let lowering = path_sum::lower(&net, &proof, "run").expect("lower");
    let mut model = CpuModel::compile(&lowering).expect("CPU model");
    for args in [
        [32, 3],
        [i64::from(i32::MAX), 0],
        [-100, 3],
        [0, i64::from(i32::MIN)],
    ] {
        let (result, _) = lowering
            .run(&net, &proof, &args, &CostModel::METAL, &mut model, limits())
            .expect("run");
        assert_eq!(
            Some(result),
            observe(&net, &proof, "run", &args),
            "{args:?}"
        );
    }
    // A refused host call publishes the oracle observation without a launch.
    let (result, outcome) = lowering
        .run(
            &net,
            &proof,
            &[256, 8],
            &CostModel::CUDA,
            &mut model,
            limits(),
        )
        .expect("run");
    assert_eq!(outcome, Outcome::Fallback("cost"));
    assert_eq!(Some(result), observe(&net, &proof, "run", &[256, 8]));
}

/// `(device, program, a, b, scope)` of every D5 cell that missed in the
/// committed evidence runs (median above 1.00 or U95 above 1.05), kept as
/// test fixtures because the full records are not retained.
fn measured_misses() -> Vec<(&'static str, String, i64, i64, String)> {
    let runs = [
        ("cuda", "cuda-gate-idle.txt"),
        ("metal", "metal-gate-head.txt"),
        ("metal", "metal-gate-exclusive.txt"),
    ];
    let mut misses = Vec::new();
    for (device, file) in runs {
        let path = format!(
            "{}/tests/fixtures/archived-evidence/p4a-path-sum/{file}",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).expect("evidence");
        let rows: Vec<Vec<&str>> = text
            .lines()
            .map(|l| l.split('\t').collect::<Vec<_>>())
            .filter(|c| c.len() == 10 && matches!(c[9], "pass" | "miss"))
            .collect();
        assert!(
            !rows.is_empty() && rows.len().is_multiple_of(32),
            "{file}: whole runs of 32 cells"
        );
        for c in rows.iter().filter(|c| c[9] == "miss") {
            let cell = (
                device,
                c[0].to_owned(),
                c[1].parse().unwrap(),
                c[2].parse().unwrap(),
                c[3].to_owned(),
            );
            if !misses.contains(&cell) {
                misses.push(cell);
            }
        }
    }
    misses
}

#[test]
fn the_cost_model_refuses_every_measured_host_miss() {
    let misses = measured_misses();
    let mut refused = Vec::new();
    for name in GATE {
        let (net, proof) = support::compile(&fixture(name));
        let lowering = path_sum::lower(&net, &proof, "run").expect("lower");
        let large = if name.starts_with("divide") {
            [20_000, 30]
        } else {
            [3000, 14]
        };
        for args in [[0, 0], [32, 3], [256, 8], large] {
            for (device, model) in [("cuda", CostModel::CUDA), ("metal", CostModel::METAL)] {
                let plan = lowering.plan(&args, &model);
                assert!(matches!(plan, Plan::Launch(_)), "{name} {args:?}");
                assert_eq!(lowering.route(&args, &model, Scope::Resident), plan);
                let host = lowering.route(&args, &model, Scope::Host);
                let miss = (device, name.to_owned(), args[0], args[1], "host".to_owned());
                if host == Plan::Fallback("cost") {
                    refused.push(format!("{device} {name} {} {}", args[0], args[1]));
                } else {
                    assert_eq!(host, plan);
                    assert!(!misses.contains(&miss), "{miss:?} missed but launches");
                }
            }
        }
    }
    // The model refuses by shape and lanes, not by program: on Metal it also
    // refuses the 256-lane host cells that passed by small margins.
    let mut expected: Vec<String> = Vec::new();
    for name in GATE {
        let large = if name.starts_with("divide") {
            "20000 30"
        } else {
            "3000 14"
        };
        for args in ["0 0", "32 3", "256 8", large] {
            expected.push(format!("cuda {name} {args}"));
        }
        expected.push(format!("metal {name} 256 8"));
    }
    refused.sort();
    expected.sort();
    assert_eq!(refused, expected);
    // Resident misses stay launched: only a U95 miss from one outlier
    // session (median 0.980) is on record.
    let resident: Vec<_> = misses.iter().filter(|m| m.4 == "resident").collect();
    assert_eq!(
        resident,
        [&(
            "metal",
            "compose-02".to_owned(),
            32,
            3,
            "resident".to_owned()
        )]
    );
    eprintln!("refused host cells: {refused:?}");
}

#[test]
fn the_partition_is_proven_statically_only_where_it_holds() {
    let mut cases: Vec<(String, String, &str, bool)> = GATE
        .iter()
        .map(|name| ((*name).to_owned(), fixture(name), "run", true))
        .collect();
    cases.push(("l9-pathsum".into(), L9_PATHSUM.into(), "run", true));
    cases.push(("l10-range".into(), L10_RANGE.into(), "sum", true));
    // `m` may land on or outside `lo`: children can be empty.
    cases.push(("geometry".into(), GEOMETRY.into(), "run", false));
    for (name, source, entry, proven) in cases {
        let (net, proof) = support::compile(&source);
        let lowering = path_sum::lower(&net, &proof, entry).expect("lower");
        assert_eq!(lowering.geometry, proven, "{name}");
        assert_eq!(
            lowering.source().contains("PROVEN(!("),
            proven,
            "{name}: geometry checks follow the proof"
        );
    }
}

#[test]
fn the_licence_covers_every_scored_gate_input() {
    for name in GATE {
        let (net, proof) = support::compile(&fixture(name));
        let lowering = path_sum::lower(&net, &proof, "run").expect("lower");
        eprintln!(
            "{name}: licence {:?}, max depth {}, {:?}",
            lowering.licence, lowering.max_depth, lowering.enumerator
        );
        let large = if name.starts_with("divide") {
            [20_000, 30]
        } else {
            [3000, 14]
        };
        for args in [[0, 0], [32, 3], [256, 8], large] {
            assert!(
                matches!(lowering.plan(&args, &CostModel::METAL), Plan::Launch(_)),
                "{name} {args:?} outside the licence"
            );
        }
    }
}
