//! P4a gate: the P4a profile protocol with COMPILER-generated path-sum kernels.
//!
//! `invar-pathsum-gate sample <metal|cuda> <fixtures> <handwritten> <oracle.tsv> <session>`
//! prints `sample,...` CSV rows: per scored input, scope (resident GPU
//! interval, host-to-host wall) and 40 interleaved pairs in random order, the
//! first 10 discarded. Variant 0 is the compiler's guarded A' launch; the
//! other variant is the frozen P4a profile handwritten baseline (2 = C fused,
//! 3 = C two-stage, 4 = C checked serial). Every sample is checked against the
//! frozen oracle value before it is recorded.
//!
//! `invar-pathsum-gate stats <dir>` reads `session-*.csv` and prints the D5
//! verdict per cell: median of the 90 paired ratios and its hierarchical
//! bootstrap upper 95% bound (sessions, then pairs; 10,000 resamples).
//!
//! `invar-pathsum-gate breakdown <fixtures>` prints, per scored input, the
//! host-only cost of the guarded call that the baseline does not pay (`plan`,
//! `arg_bytes` and `finish`), in nanoseconds per call.

use std::{collections::BTreeMap, env, fs, time::Instant};

use invar::general_net::path_sum::{self, CostModel, Lowering, Plan, device};

/// `(program, a, b, frozen baseline variant)` from the P4a profile's `plan.tsv`.
const PLAN: [(&str, i32, i32, usize); 16] = [
    ("divide-00", 0, 0, 4),
    ("divide-00", 32, 3, 2),
    ("divide-00", 256, 8, 2),
    ("divide-00", 20000, 30, 3),
    ("divide-02", 0, 0, 4),
    ("divide-02", 32, 3, 2),
    ("divide-02", 256, 8, 2),
    ("divide-02", 20000, 30, 3),
    ("compose-00", 0, 0, 2),
    ("compose-00", 32, 3, 2),
    ("compose-00", 256, 8, 2),
    ("compose-00", 3000, 14, 3),
    ("compose-02", 0, 0, 4),
    ("compose-02", 32, 3, 2),
    ("compose-02", 256, 8, 2),
    ("compose-02", 3000, 14, 3),
];
const WARMUPS: usize = 10;
const PAIRS: usize = 30;

/// One device's path-sum launcher (`invar_metal::PathSum` or the CUDA one).
trait Kernels {
    fn write_args(&mut self, args: &[u8]) -> Result<(), String>;
    /// Device interval in milliseconds.
    fn dispatch(&mut self, stages: &[(usize, u32)]) -> Result<f64, String>;
    fn read_out(&mut self, words: usize) -> Result<Vec<u32>, String>;
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl Kernels for invar_metal::PathSum {
    fn write_args(&mut self, args: &[u8]) -> Result<(), String> {
        invar_metal::PathSum::write_args(self, args).map_err(|e| format!("{e:?}"))
    }
    fn dispatch(&mut self, stages: &[(usize, u32)]) -> Result<f64, String> {
        let stages: Vec<(usize, usize)> = stages.iter().map(|(k, g)| (*k, *g as usize)).collect();
        invar_metal::PathSum::dispatch(self, &stages).map_err(|e| format!("{e:?}"))
    }
    fn read_out(&mut self, words: usize) -> Result<Vec<u32>, String> {
        invar_metal::PathSum::read_out(self, words).map_err(|e| format!("{e:?}"))
    }
}

#[cfg(feature = "cuda")]
impl Kernels for invar_cuda::cuda::PathSum {
    fn write_args(&mut self, args: &[u8]) -> Result<(), String> {
        invar_cuda::cuda::PathSum::write_args(self, args).map_err(|e| format!("{e:?}"))
    }
    fn dispatch(&mut self, stages: &[(usize, u32)]) -> Result<f64, String> {
        invar_cuda::cuda::PathSum::dispatch(self, stages).map_err(|e| format!("{e:?}"))
    }
    fn read_out(&mut self, words: usize) -> Result<Vec<u32>, String> {
        invar_cuda::cuda::PathSum::read_out(self, words).map_err(|e| format!("{e:?}"))
    }
}

/// Handwritten P4a profile entry points used by the baselines, in launcher order.
const HANDWRITTEN: [&str; 4] = ["C1", "C2", "reduce", "C0"];

/// Open `(compiler, handwritten)` kernels on `target`.
type Open = dyn FnMut(&str, &[&str], usize, usize) -> Result<Box<dyn Kernels>, String>;

fn opener(target: &str) -> Result<Box<Open>, String> {
    match target {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "metal" => {
            let mut device =
                invar_metal::MetalDevice::open_runtime_source().map_err(|e| format!("{e:?}"))?;
            eprintln!("device={}", device.name());
            Ok(Box::new(
                move |source: &str, names: &[&str], groups, bytes| {
                    device
                        .pathsum(source, names, groups, bytes)
                        .map(|k| Box::new(k) as Box<dyn Kernels>)
                        .map_err(|e| format!("{e:?}"))
                },
            ))
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            let mut device = invar_cuda::cuda::CudaDevice::open().map_err(|e| format!("{e:?}"))?;
            eprintln!("device={} caps={:?}", device.name(), device.capabilities());
            Ok(Box::new(
                move |source: &str, names: &[&str], groups, bytes| {
                    device
                        .pathsum(source, names, groups, bytes)
                        .map(|k| Box::new(k) as Box<dyn Kernels>)
                        .map_err(|e| format!("{e:?}"))
                },
            ))
        }
        _ => Err(format!("target {target} is not built in")),
    }
}

/// NVRTC has no host headers: the handwritten CUDA files include three that
/// the kernels do not use (they define `uint`, `max` and `min` themselves).
fn handwritten_source(target: &str, dir: &str, program: &str) -> Result<String, String> {
    let ext = if target == "metal" { "metal" } else { "cu" };
    let text = fs::read_to_string(format!("{dir}/{program}.{ext}")).map_err(|e| e.to_string())?;
    Ok(if target == "cuda" {
        text.lines()
            .filter(|l| !l.starts_with("#include"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        text
    })
}

fn compile(source: &str) -> Result<(invar::general_net::Net, invar::general_net::Proof), String> {
    let parsed = invar::front::parse(source.as_bytes()).map_err(|e| format!("{e:?}"))?;
    let checked = invar::check::check(parsed, false).map_err(|e| format!("{e:?}"))?;
    invar::ownership::check(&checked).map_err(|e| format!("{e:?}"))?;
    let net = invar::general_net::Net::lower(&checked, invar::general_net::Limits::default())
        .map_err(|e| format!("{e:?}"))?;
    let proof = net.verify().map_err(|e| format!("{e:?}"))?;
    Ok((net, proof))
}

struct Sample {
    gpu: f64,
    wall: f64,
    value: i32,
}

/// One guarded A' call from the host: plan (entry, licence, shape), launch,
/// publish. Resident samples reuse the args written before timing.
fn compiler_sample(
    lowering: &Lowering,
    model: &CostModel,
    kernels: &mut dyn Kernels,
    args: [i64; 2],
    host: bool,
) -> Result<Sample, String> {
    let started = Instant::now();
    let Plan::Launch(launch) = lowering.plan(&args, model) else {
        return Err(format!("{args:?} left the licence"));
    };
    if host {
        kernels.write_args(&launch.arg_bytes())?;
    }
    let gpu = kernels.dispatch(&device::stages(&launch))?;
    let out = kernels.read_out(lowering.out_words())?;
    let value = lowering.finish(&out).map_err(|r| format!("fallback {r}"))?;
    Ok(Sample {
        gpu,
        wall: started.elapsed().as_secs_f64() * 1000.0,
        value,
    })
}

fn handwritten_args(a: i32, b: i32) -> Vec<u8> {
    let groups = (a.max(1) as u32).div_ceil(256).min(256);
    [a as u32, b as u32, 0, groups]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

fn handwritten_stages(variant: usize, a: i32) -> Vec<(usize, u32)> {
    let groups = (a.max(1) as u32).div_ceil(256).min(256);
    match variant {
        2 => vec![(0, 1)],
        3 => vec![(1, groups), (2, 1)],
        _ => vec![(3, 1)],
    }
}

fn handwritten_sample(
    kernels: &mut dyn Kernels,
    variant: usize,
    a: i32,
    b: i32,
    host: bool,
) -> Result<Sample, String> {
    let started = Instant::now();
    if host {
        kernels.write_args(&handwritten_args(a, b))?;
    }
    let gpu = kernels.dispatch(&handwritten_stages(variant, a))?;
    // `struct O { int value; uint trap, span, end, key; }`
    let out = kernels.read_out(5)?;
    if out[1] != 0 {
        return Err(format!("handwritten trap {out:?}"));
    }
    Ok(Sample {
        gpu,
        wall: started.elapsed().as_secs_f64() * 1000.0,
        value: out[0] as i32,
    })
}

fn sample(
    target: &str,
    fixtures: &str,
    handwritten: &str,
    oracle: &str,
    session: u64,
) -> Result<(), String> {
    let model = if target == "cuda" {
        CostModel::CUDA
    } else {
        CostModel::METAL
    };
    let expected: BTreeMap<(String, i32, i32), i32> = fs::read_to_string(oracle)
        .map_err(|e| e.to_string())?
        .lines()
        .filter_map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            (c.len() >= 8 && c[4] == "0").then(|| {
                Some((
                    (c[0].to_owned(), c[1].parse().ok()?, c[2].parse().ok()?),
                    c[3].parse().ok()?,
                ))
            })?
        })
        .collect();
    let mut open = opener(target)?;
    let mut seed = 0x0019_2821_bb12_u64 ^ session;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    println!("mode,session,program,a,b,scope,pair,variant,comparison,gpu_ms,wall_ms,value");
    for program in ["divide-00", "divide-02", "compose-00", "compose-02"] {
        let source = fs::read_to_string(format!("{fixtures}/gnet05-{program}.inv"))
            .map_err(|e| e.to_string())?;
        let (net, proof) = compile(&source)?;
        let lowering = path_sum::lower(&net, &proof, "run").map_err(|e| format!("{e:?}"))?;
        eprintln!(
            "{program}: licence {:?}, max depth {}, source {} bytes",
            lowering.licence,
            lowering.max_depth,
            lowering.source().len()
        );
        let mut ours = open(
            lowering.source(),
            &device::KERNELS,
            device::max_groups(),
            lowering.out_words() * 4,
        )?;
        let mut theirs = open(
            &handwritten_source(target, handwritten, program)?,
            &HANDWRITTEN,
            256,
            20,
        )?;
        for &(_, a, b, c) in PLAN.iter().filter(|row| row.0 == program) {
            let want = *expected
                .get(&(program.to_owned(), a, b))
                .ok_or("missing oracle row")?;
            let Plan::Launch(launch) = lowering.plan(&[a.into(), b.into()], &model) else {
                return Err(format!("{program} {a} {b}: outside the licence"));
            };
            eprintln!(
                "{program} {a} {b}: A' shape {:?}, baseline {c}",
                launch.shape
            );
            for scope in ["resident", "host"] {
                let host = scope == "host";
                for i in 0..WARMUPS + PAIRS {
                    let reverse = rng() & 1 != 0;
                    for variant in if reverse { [c, 0] } else { [0, c] } {
                        let s = if variant == 0 {
                            ours.write_args(&launch.arg_bytes())?;
                            compiler_sample(
                                &lowering,
                                &model,
                                ours.as_mut(),
                                [a.into(), b.into()],
                                host,
                            )?
                        } else {
                            theirs.write_args(&handwritten_args(a, b))?;
                            handwritten_sample(theirs.as_mut(), variant, a, b, host)?
                        };
                        if s.value != want {
                            return Err(format!(
                                "{program} {a} {b} variant {variant}: {} != {want}",
                                s.value
                            ));
                        }
                        if i >= WARMUPS {
                            println!(
                                "sample,{session},{program},{a},{b},{scope},{},{variant},0,{:.9},{:.9},{}",
                                i - WARMUPS,
                                s.gpu,
                                s.wall,
                                s.value
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Host-only cost per call: median over 7 batches of 100,000 calls.
fn breakdown(fixtures: &str) -> Result<(), String> {
    const CALLS: u32 = 100_000;
    println!("program\ta\tb\thost_ns");
    for program in ["divide-00", "divide-02", "compose-00", "compose-02"] {
        let source = fs::read_to_string(format!("{fixtures}/gnet05-{program}.inv"))
            .map_err(|e| e.to_string())?;
        let (net, proof) = compile(&source)?;
        let lowering = path_sum::lower(&net, &proof, "run").map_err(|e| format!("{e:?}"))?;
        let out = vec![0u32; lowering.out_words()];
        for &(_, a, b, _) in PLAN.iter().filter(|row| row.0 == program) {
            let args = [i64::from(a), i64::from(b)];
            let mut batches = Vec::new();
            for _ in 0..7 {
                let started = Instant::now();
                for _ in 0..CALLS {
                    let plan = lowering.plan(std::hint::black_box(&args), &CostModel::METAL);
                    let Plan::Launch(launch) = plan else {
                        return Err(format!("{program} {a} {b}: outside the licence"));
                    };
                    std::hint::black_box(launch.arg_bytes());
                    std::hint::black_box(lowering.finish(std::hint::black_box(&out)))?;
                }
                batches.push(started.elapsed().as_secs_f64() * 1e9 / f64::from(CALLS));
            }
            println!("{program}\t{a}\t{b}\t{:.1}", median(&mut batches));
        }
    }
    Ok(())
}

fn median(x: &mut [f64]) -> f64 {
    x.sort_by(f64::total_cmp);
    (x[(x.len() - 1) / 2] + x[x.len() / 2]) / 2.
}

type Cell = (String, i32, i32, String);
/// `(session, pair) -> (compiler ms, baseline ms)` per cell.
type Pairs = BTreeMap<(usize, usize), (f64, f64)>;

/// D5 flat verdicts from `session-*.csv` (the P4a profile's `stats.rs` statistic).
fn stats(dir: &str) -> Result<(), String> {
    let mut cells: BTreeMap<Cell, Pairs> = BTreeMap::new();
    let mut sessions_seen = std::collections::BTreeSet::new();
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("session-") || !name.ends_with(".csv") {
            continue;
        }
        for line in fs::read_to_string(entry.path())
            .map_err(|e| e.to_string())?
            .lines()
        {
            let c: Vec<&str> = line.split(',').collect();
            if c[0] != "sample" {
                continue;
            }
            let parse = |i: usize| c[i].parse::<f64>().map_err(|e| e.to_string());
            let session: usize = c[1].parse().map_err(|_| "session")?;
            sessions_seen.insert(session);
            let pair: usize = c[6].parse().map_err(|_| "pair")?;
            let variant: usize = c[7].parse().map_err(|_| "variant")?;
            let time = if c[5] == "resident" {
                parse(9)?
            } else {
                parse(10)?
            };
            if !(time.is_finite() && time > 0.0) {
                return Err(format!("bad time {line}"));
            }
            let key = (
                c[2].to_owned(),
                c[3].parse().map_err(|_| "a")?,
                c[4].parse().map_err(|_| "b")?,
                c[5].to_owned(),
            );
            let row = cells
                .entry(key)
                .or_default()
                .entry((session, pair))
                .or_insert((0., 0.));
            if variant == 0 {
                row.0 = time;
            } else {
                row.1 = time;
            }
        }
    }
    let sessions: Vec<usize> = sessions_seen.into_iter().collect();
    println!("program\ta\tb\tscope\tpoint\tupper95\tgen_ms\tC_ms\tsessions\tgate");
    let mut seed = 0x0031_1bc1_9e81_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as usize
    };
    let (mut pass, mut total) = (0, 0);
    for ((program, a, b, scope), pairs) in cells {
        let mut per: Vec<Vec<f64>> = vec![Vec::new(); sessions.len()];
        let (mut gs, mut cs) = (Vec::new(), Vec::new());
        for ((s, _), (g, c)) in &pairs {
            if *g <= 0.0 || *c <= 0.0 {
                return Err(format!("{program} {a} {b} {scope}: unpaired sample"));
            }
            let i = sessions.iter().position(|x| x == s).ok_or("session")?;
            per[i].push(g / c);
            gs.push(*g);
            cs.push(*c);
        }
        let mut all: Vec<f64> = per.iter().flatten().copied().collect();
        let point = median(&mut all);
        let mut boot = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            let mut ratios = Vec::with_capacity(all.len());
            for _ in 0..per.len() {
                let s = &per[next() % per.len()];
                for _ in 0..s.len() {
                    ratios.push(s[next() % s.len()]);
                }
            }
            boot.push(median(&mut ratios));
        }
        boot.sort_by(f64::total_cmp);
        let upper = boot[boot.len() * 95 / 100 - 1];
        let passed = point <= 1.0 && upper <= 1.05;
        let gate = if passed { "pass" } else { "miss" };
        total += 1;
        pass += usize::from(passed);
        let points: Vec<String> = per
            .iter_mut()
            .map(|s| format!("{:.6}", median(s)))
            .collect();
        println!(
            "{program}\t{a}\t{b}\t{scope}\t{point:.6}\t{upper:.6}\t{:.9}\t{:.9}\t{}\t{gate}",
            median(&mut gs),
            median(&mut cs),
            points.join(" ")
        );
    }
    println!("D5 flat: {pass}/{total} cells pass (median <= 1.00, U95 <= 1.05)");
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("sample") if args.len() == 7 => {
            let session = args[6].parse().unwrap_or(1);
            sample(&args[2], &args[3], &args[4], &args[5], session)
        }
        Some("stats") if args.len() == 3 => stats(&args[2]),
        Some("breakdown") if args.len() == 3 => breakdown(&args[2]),
        Some("source") if args.len() == 4 => fs::read_to_string(&args[2])
            .map_err(|e| e.to_string())
            .and_then(|text| compile(&text))
            .and_then(|(net, proof)| {
                path_sum::lower(&net, &proof, &args[3]).map_err(|r| format!("{r:?}"))
            })
            .map(|lowering| print!("{}", lowering.source())),
        _ => Err("usage: invar-pathsum-gate sample <metal|cuda> <fixtures> <handwritten> <oracle.tsv> <session> | stats <dir> | breakdown <fixtures> | source <program.inv> <entry>".into()),
    };
    if let Err(e) = result {
        eprintln!("invar-pathsum-gate: {e}");
        std::process::exit(1);
    }
}
