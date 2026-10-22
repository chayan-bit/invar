//! GNET-05 robotics corpus: host-array variants and device admission census.
//!
//! The array programs (`bench/general-net-v01/gnet05/robotics/array`) take
//! their bulk inputs as `read Array<f32>` host buffers and write the output
//! into a `mut Array<f32>`. Their files, held-out guard and reference/oracle
//! outputs are checked by default through the ABI13 host-array entry; the
//! device census needs a GPU and stays ignored.
//! Regenerate with `INVAR_GNET05_WRITE=1` (validation digests need Release).
#![forbid(unsafe_code)]

#[path = "support/gnet_agree.rs"]
mod agree;
#[path = "support/gnet_device.rs"]
mod device;
#[path = "../bench/general-net-v01/gnet05/generate.rs"]
mod generate;
#[path = "support/gnet_corpus_guard.rs"]
mod guard;
#[path = "support/gnet_par.rs"]
mod par;
#[path = "support/gnet_parallel.rs"]
mod parallel;
#[path = "../bench/general-net-v01/gnet05/robotics/mod.rs"]
mod robotics;

use generate::Split;
#[cfg(any(feature = "metal", feature = "cuda"))]
use invar::general_net::{compiled, device_host};
use invar::{
    check,
    eval::{
        self,
        host_read::{HostArgument, HostMutArray, HostReadArray},
    },
    front,
    general_net::{self, Net, oracle},
    numeric_contract::F32Bits,
};
use robotics::array::{self, Arg};
use robotics::corpus::{self, Program};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    fs,
    path::{Path, PathBuf},
    rc::Rc,
};

/// Reference fuel for the scored inputs (never reached by any of them).
const FUEL: u64 = 1 << 40;

fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}
fn root() -> PathBuf {
    repo().join("bench/general-net-v01/gnet05/robotics/array")
}
fn write() -> bool {
    std::env::var_os("INVAR_GNET05_WRITE").is_some()
}
fn arm_of<'a>(
    p: &Program,
    arms: &'a (robotics::arm::Arm, robotics::arm::Arm),
) -> Option<&'a robotics::arm::Arm> {
    match p.case.arm {
        "panda" => Some(&arms.0),
        "ur5e" => Some(&arms.1),
        _ => None,
    }
}

#[test]
fn issue_gnet_05_robotics_array_files_are_exactly_the_generator_output() {
    let programs = array::generate(repo());
    assert_eq!(programs, array::generate(repo()), "deterministic");
    let manifest =
        serde_json::to_string_pretty(&array::manifest(repo(), &programs)).unwrap() + "\n";
    if write() {
        let dir = root().join("programs");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for p in &programs {
            fs::write(dir.join(format!("{}.inv", p.id)), &p.source).unwrap();
        }
        fs::write(root().join("manifest.json"), &manifest).unwrap();
    }
    assert_eq!(
        fs::read_to_string(root().join("manifest.json")).unwrap(),
        manifest
    );
    assert_eq!(
        fs::read_dir(root().join("programs")).unwrap().count(),
        programs.len()
    );
    let seed = corpus::generate(repo());
    let mut identities = std::collections::BTreeSet::new();
    for (p, s) in programs.iter().zip(&seed) {
        let on_disk = fs::read_to_string(root().join(format!("programs/{}.inv", p.id))).unwrap();
        assert_eq!(on_disk, p.source, "{}", p.id);
        assert_eq!(
            (p.split, &p.oracle_inputs),
            (s.split, &s.oracle_inputs),
            "{}",
            p.id
        );
        assert!(identities.insert(generate::normalized_sha256(p.source.as_bytes())));
        assert!(identities.insert(generate::normalized_sha256(s.source.as_bytes())));
        checked(p);
        assert!(
            !p.source.contains("stream(") && !p.source.contains("seed"),
            "{}",
            p.id
        );
    }
}

#[test]
fn issue_gnet_05_robotics_array_held_out_programs_never_reach_tuning() {
    let programs = array::generate(repo());
    let held = array::held_out(&programs);
    let manifest: Json =
        serde_json::from_slice(&fs::read(root().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        manifest["held_out"]["normalized_identity_sha256"],
        generate::hex(&held.digest())
    );
    let seed_held = corpus::held_out(&corpus::generate(repo()));
    assert_eq!(
        corpus::tuning_programs(&programs, &held).len(),
        programs.len() - held.len()
    );
    for p in &programs {
        let admitted = held.admit_for_tuning(p.source.as_bytes()).is_ok();
        assert_eq!(admitted, p.split == Split::Tuning, "{}", p.id);
        assert!(seed_held.admit_for_tuning(p.source.as_bytes()).is_ok());
        if p.split == Split::HeldOut {
            for disguised in [guard::renamed(&p.source), guard::recommented(&p.source)] {
                assert!(
                    held.admit_for_tuning(disguised.as_bytes()).is_err(),
                    "{}: a disguised held-out program reached tuning",
                    p.id
                );
            }
        }
    }
}

fn checked(p: &Program) -> check::CheckedProgram {
    check::check(front::parse(p.source.as_bytes()).unwrap(), true)
        .unwrap_or_else(|e| panic!("{}: ordinary source must check: {e:?}", p.id))
}

fn bits(v: &[f32]) -> Vec<F32Bits> {
    v.iter().map(|x| F32Bits::from_bits(x.to_bits())).collect()
}

/// Entry values for the oracle and the device (fresh owners per call).
fn values(args: &[Arg]) -> Vec<eval::Value> {
    args.iter()
        .map(|a| match a {
            Arg::Read(v) => eval::Value::ArrayF32(Rc::new(RefCell::new(bits(v)))),
            Arg::Out(n) => eval::Value::ArrayF32(Rc::new(RefCell::new(bits(&vec![0.0; *n])))),
            Arg::U32(x) => eval::Value::U32(*x),
        })
        .collect()
}

/// The output buffer after a call.
fn out_of(values: &[eval::Value]) -> Vec<f32> {
    match values.last() {
        Some(eval::Value::ArrayF32(v)) => v
            .borrow()
            .iter()
            .map(|x| f32::from_bits(x.to_bits()))
            .collect(),
        other => panic!("output buffer {other:?}"),
    }
}

/// Reference evaluator on host buffers: (observation, output buffer).
fn reference(c: &check::CheckedProgram, args: &[Arg], fuel: u64) -> (eval::Observation, Vec<f32>) {
    let reads: Vec<Vec<F32Bits>> = args
        .iter()
        .map(|a| match a {
            Arg::Read(v) => bits(v),
            _ => Vec::new(),
        })
        .collect();
    let Some(Arg::Out(n)) = args.last() else {
        panic!("output last")
    };
    let mut out = vec![F32Bits::from_bits(0); *n];
    let mut host: Vec<HostArgument> = Vec::new();
    for (a, r) in args.iter().zip(&reads) {
        match a {
            Arg::Read(_) => host.push(HostArgument::Read(HostReadArray::F32(r))),
            Arg::U32(x) => host.push(HostArgument::U32(*x)),
            Arg::Out(_) => {}
        }
    }
    host.push(HostArgument::Mut(HostMutArray::F32(&mut out)));
    let o = eval::host_read::observe_host_call(c, "run", &mut host, fuel);
    drop(host);
    (o, out.iter().map(|x| f32::from_bits(x.to_bits())).collect())
}

fn digest(out: &[f32]) -> String {
    let mut h = Sha256::new();
    for x in out {
        h.update(x.to_bits().to_le_bytes());
    }
    generate::hex(&h.finalize())
}

/// The seed program's frozen output (words, digest) for one input.
fn seed_output(seed: &Json, id: &str, kind: &str, input: &[u32]) -> (u64, String) {
    let row = seed["programs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == id)
        .unwrap_or_else(|| panic!("{id} recorded"));
    let r = row[kind]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["input"] == json!(input))
        .unwrap_or_else(|| panic!("{id} {input:?} recorded"));
    (
        r["words"].as_u64().unwrap(),
        r["sha256"].as_str().unwrap().to_owned(),
    )
}

/// Every array program's full output equals its seed program's frozen
/// output: oracle inputs through the reference and the CPU net oracle (bit
/// for bit, steps and cleanup included), scored inputs (Release) through
/// the reference. The point cloud's host buffer beyond the seed program's
/// returned length stays zero.
#[test]
fn issue_gnet_05_robotics_array_outputs_equal_seed_outputs() {
    array_outputs();
}

fn array_outputs() {
    let programs = array::generate(repo());
    let arms = corpus::arms(repo());
    let seed: Json = serde_json::from_slice(
        &fs::read(repo().join("bench/general-net-v01/gnet05/robotics/validation.json")).unwrap(),
    )
    .unwrap();
    let release = !cfg!(debug_assertions);
    let rows = par::map(&programs, |p| {
        let c = checked(p);
        let net = Net::lower(&c, general_net::Limits::default())
            .unwrap_or_else(|e| panic!("{}: net lowering {e:?}", p.id));
        let proof = net.verify().unwrap();
        let seed_id = p.id.replacen("gnet05ra-", "gnet05r-", 1);
        let check = |kind: &str, input: &Vec<u32>, with_oracle: bool| {
            let context = format!("{} {input:?}", p.id);
            let args = array::expand(&p.case, arm_of(p, &arms), input);
            let (o, out) = reference(&c, &args, FUEL);
            assert!(o.result.is_ok(), "{context}: {:?}", o.result);
            if with_oracle {
                let v = values(&args);
                let graph = oracle::observe_call(
                    &net,
                    &proof,
                    "run",
                    &v,
                    oracle::Limits {
                        fuel: FUEL,
                        transitions: u64::MAX,
                        trace_events: usize::MAX,
                        array_elements: 1 << 22,
                    },
                )
                .unwrap_or_else(|e| panic!("{context}: oracle {e:?}"));
                assert_eq!(graph.failure, None, "{context}");
                agree::same(&graph.semantic, &o, &context);
                assert_eq!(
                    digest(&out_of(&v)),
                    digest(&out),
                    "{context}: oracle buffer"
                );
            }
            let (words, sha) = seed_output(&seed, &seed_id, kind, input);
            let words = words as usize;
            assert_eq!(digest(&out[..words]), sha, "{context}: seed output");
            assert!(
                out[words..].iter().all(|x| x.to_bits() == 0),
                "{context}: tail"
            );
            json!({"input": input, "steps": o.steps_used, "result": format!("{:?}", o.result),
                "buffer_words": out.len(), "seed_words": words, "sha256": digest(&out)})
        };
        let oracle_rows: Vec<Json> = p
            .oracle_inputs
            .iter()
            .map(|i| check("oracle_inputs", i, true))
            .collect();
        let scored_rows: Vec<Json> = if release {
            p.scored_inputs
                .iter()
                .map(|i| check("scored_inputs", i, false))
                .collect()
        } else {
            Vec::new()
        };
        println!("gnet05ra {} ok", p.id);
        json!({"id": p.id, "split": p.split.label(), "oracle_inputs": oracle_rows,
            "scored_inputs": scored_rows})
    });
    let report = json!({"schema": "invar-gnet05-robotics-array-validation-v1",
        "requires": array::REQUIRES, "programs": rows});
    let path = root().join("validation.json");
    if write() && release {
        fs::write(&path, serde_json::to_string_pretty(&report).unwrap() + "\n").unwrap();
    }
    let recorded: Json = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    for (r, n) in recorded["programs"]
        .as_array()
        .unwrap()
        .iter()
        .zip(report["programs"].as_array().unwrap())
    {
        assert_eq!(r["oracle_inputs"], n["oracle_inputs"], "{}", n["id"]);
        if release {
            assert_eq!(r["scored_inputs"], n["scored_inputs"], "{}", n["id"]);
        }
    }
}

#[cfg(any(feature = "metal", feature = "cuda"))]
/// Admission configurations of the census: the corpus device limits, and
/// wide compiler limits with and without speculation slots. The owned
/// limits stay small: the programs allocate no owned arrays, and every
/// speculation slot reserves heap words for the owned limit, so a large
/// one exceeds the slots' scratch budget.
fn admission_configs() -> Vec<(&'static str, device_host::Limits)> {
    let wide = |slots| {
        let mut l = parallel::limits();
        l.compiler = compiled::Limits {
            values: 1 << 16,
            places: 4096,
            states: 1 << 20,
            frame_bytes: 1 << 26,
            source_bytes: 1 << 26,
        };
        l.owned = compiled::OwnedLimits {
            elements: 65_536,
            allocations: 64,
            cleanup_events: 64,
        };
        l.private.slots = slots;
        l.buffer_bytes = 1 << 30;
        l
    };
    vec![
        ("default", parallel::limits()),
        ("wide", wide(parallel::SLOTS)),
        ("wide-serial", wide(0)),
    ]
}

/// One census row: a program under one admission configuration, and the
/// device run it selects, if any.
#[cfg(any(feature = "metal", feature = "cuda"))]
struct Entry {
    row: Json,
    admission: std::time::Duration,
    run: Option<Run>,
}

/// An admitted plan the census runs: its first oracle input, wide and as a
/// capacity-0 serial control.
#[cfg(any(feature = "metal", feature = "cuda"))]
struct Run {
    context: String,
    plan: device_host::Plan,
    args: Vec<eval::Value>,
    wide: usize,
}

/// Ahead-of-time compile outcome per device source (SHA-256 of the source):
/// compile seconds, or the typed refusal class recorded in the row.
#[cfg(any(feature = "metal", feature = "cuda"))]
type Ahead = std::collections::HashMap<[u8; 32], Result<f64, String>>;

/// Device admission of every seed and array program under each admission
/// configuration, with the typed refusal class. `INVAR_GNET05_CENSUS_RUN`
/// (comma-separated) limits the device runs to admitted programs whose
/// `config/id` contains one of its entries (Metal pipeline compilation of
/// the large kernels takes minutes each).
#[cfg(any(feature = "metal", feature = "cuda"))]
fn census_admission(target: compiled::Target) -> Vec<Entry> {
    let arms = corpus::arms(repo());
    let seed = corpus::generate(repo());
    let arrays = array::generate(repo());
    let run_filter = std::env::var("INVAR_GNET05_CENSUS_RUN").unwrap_or_default();
    let mut entries = Vec::new();
    for (kind, p) in seed
        .iter()
        .map(|p| ("seed", p))
        .chain(arrays.iter().map(|p| ("array", p)))
    {
        for (name, limits) in admission_configs() {
            let started = std::time::Instant::now();
            let plan = device_host::Plan::from_source(p.source.as_bytes(), "run", target, limits);
            let mut row = json!({"id": p.id, "kind": kind, "config": name});
            let run = match plan {
                Err(e) => {
                    row["admission"] = json!(format!("refused {}", device_host::refusal_class(&e)));
                    None
                }
                Ok(plan) => {
                    row["admission"] = json!("admitted");
                    let selected = run_filter
                        .split(',')
                        .any(|f| format!("{name}/{}", p.id).contains(f));
                    selected.then(|| {
                        let input = &p.oracle_inputs[0];
                        let args = if kind == "seed" {
                            input.iter().map(|&x| eval::Value::U32(x)).collect()
                        } else {
                            values(&array::expand(&p.case, arm_of(p, &arms), input))
                        };
                        Run {
                            context: format!("{} {name}", p.id),
                            plan,
                            args,
                            wide: limits.private.slots.min(parallel::SLOTS),
                        }
                    })
                }
            };
            entries.push(Entry {
                row,
                admission: started.elapsed(),
                run,
            });
        }
    }
    entries
}

/// Every selected admitted program runs its first oracle input on the
/// device, wide and as a capacity-0 serial control, exact against the CPU
/// net oracle. One backend serves both capacities of a (program, config)
/// unless a run poisons it. A source whose ahead-of-time compile was refused
/// records that refusal and runs nothing. Rows are appended to
/// `INVAR_GNET05_CENSUS` as they complete, when set.
#[cfg(any(feature = "metal", feature = "cuda"))]
fn census_run<B: device_host::Backend>(
    label: &str,
    entries: Vec<Entry>,
    ahead: &Ahead,
    mut backend: impl FnMut() -> B,
) {
    let mut out = std::env::var_os("INVAR_GNET05_CENSUS").map(|p| fs::File::create(p).unwrap());
    let mut executed = 0;
    for Entry {
        mut row,
        admission,
        run,
    } in entries
    {
        let started = std::time::Instant::now();
        if let Some(run) = run {
            let key: [u8; 32] = Sha256::digest(run.plan.artifact().source()).into();
            let compile = ahead.get(&key);
            match compile {
                Some(Ok(seconds)) => row["compile"] = json!({"seconds": seconds}),
                Some(Err(class)) => row["compile"] = json!(format!("refused {class}")),
                None => {}
            }
            if !matches!(compile, Some(Err(_))) {
                executed += run_capacities(label, &run, &mut row, &mut backend);
            }
        }
        row["seconds"] = json!((admission + started.elapsed()).as_secs_f64());
        println!("gnet05ra census {row}");
        if let Some(f) = out.as_mut() {
            use std::io::Write;
            writeln!(f, "{row}").unwrap();
        }
    }
    println!("gnet05ra census {label} executed={executed}");
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn run_capacities<B: device_host::Backend>(
    label: &str,
    run: &Run,
    row: &mut Json,
    backend: &mut impl FnMut() -> B,
) -> usize {
    let Run {
        context,
        plan,
        args,
        wide,
    } = run;
    let mut executed = 0;
    let mut device = backend();
    let capacities = if *wide > 0 { vec![*wide, 0] } else { vec![0] };
    for capacity in capacities {
        if device.poisoned() {
            device = backend();
        }
        let context = format!("{label} {context} capacity {capacity}");
        // The device configuration caps fuel, transitions and
        // trace events at 1,000,000 each.
        let cfg = |events| compiled::DeviceConfig {
            transitions: DEVICE_CAP,
            ..device::config(DEVICE_CAP, events, 16, 64, capacity)
        };
        let sized = device::reference(plan, &values_copy(args), cfg(DEVICE_CAP as usize));
        let cfg = cfg(sized.events.len().max(1));
        let expected_args = values_copy(args);
        let expected = device::reference(plan, &expected_args, cfg);
        let run_args = values_copy(args);
        // Includes the backend's pipeline compilation unless it was
        // compiled ahead (see the row's `compile`).
        let device_started = std::time::Instant::now();
        let result = device::drive(plan, &mut device, &run_args, cfg);
        let device_seconds = device_started.elapsed().as_secs_f64();
        match result {
            Ok((got, epochs)) => {
                device::agree(&context, plan, &got, &expected);
                assert_eq!(run_args, expected_args, "{context}: published buffers");
                let f = device::Frontier::of(&got);
                row[format!("capacity_{capacity}")] = json!({"status": format!("{:?}", got.status),
                    "steps": expected.semantic.steps_used, "spawned": f.spawned,
                    "committed": f.committed, "epochs": epochs, "exact": true,
                    "device_seconds": device_seconds});
                executed += 1;
            }
            Err(e) => {
                row[format!("capacity_{capacity}")] =
                    json!(format!("device refused {}", device_host::refusal_class(&e)));
            }
        }
    }
    executed
}

/// Device configuration cap on fuel, transitions and trace events
/// (`compiled::DeviceLayout::buffer_bytes`).
#[cfg(any(feature = "metal", feature = "cuda"))]
const DEVICE_CAP: u64 = 1_000_000;

#[cfg(any(feature = "metal", feature = "cuda"))]
fn values_copy(v: &[eval::Value]) -> Vec<eval::Value> {
    v.iter()
        .map(|x| match x {
            eval::Value::ArrayF32(a) => {
                eval::Value::ArrayF32(Rc::new(RefCell::new(a.borrow().clone())))
            }
            eval::Value::U32(x) => eval::Value::U32(*x),
            other => panic!("census argument {other:?}"),
        })
        .collect()
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires an Apple Metal device; array programs need ABI13"]
fn issue_gnet_05_robotics_device_census_on_metal() {
    use invar::general_net::device_host::metal::Metal;
    let entries = census_admission(compiled::Target::Metal);
    census_run("metal", entries, &Ahead::new(), || {
        Metal::open_runtime_source().expect("Metal device")
    });
}

/// Memory one `invar-nvrtc` child may need for a census program: measured
/// peaks were about 1.5 GiB (an H100 host, NVRTC 12.4). Sixteen children in a
/// 12 GiB cgroup thrashed until every compile hit its 300 s deadline.
#[cfg(feature = "cuda")]
const NVRTC_CHILD_BYTES: u64 = 2 << 30;

/// Project policy: a test job uses at most 12 GiB of RAM, whatever its cgroup allows.
#[cfg(feature = "cuda")]
const JOB_MEMORY_CEILING: u64 = 12 << 30;

/// Concurrent `invar-nvrtc` children for the census's ahead-of-time compiles:
/// `INVAR_GNET05_COMPILE_JOBS`, else the available parallelism, never more
/// than the job's memory holds at `NVRTC_CHILD_BYTES` each. That memory is
/// the cgroup's limit capped at `JOB_MEMORY_CEILING`, so at most 6 children;
/// their margin over the measured peak (3 GiB at 6) is the parent's reserve.
#[cfg(feature = "cuda")]
fn compile_jobs() -> usize {
    let cgroup_max = fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|c| {
            c.lines()
                .find_map(|l| l.strip_prefix("0::").map(str::to_owned))
        })
        .and_then(|path| fs::read_to_string(format!("/sys/fs/cgroup{path}/memory.max")).ok())
        .and_then(|max| max.trim().parse::<u64>().ok());
    pool_size(
        std::env::var("INVAR_GNET05_COMPILE_JOBS")
            .ok()
            .and_then(|n| n.parse().ok()),
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        cgroup_max,
    )
}

/// `compile_jobs` without the environment: `requested` children, else
/// `cpus`, clamped to what `cgroup_max` (unlimited if `None`) capped at
/// `JOB_MEMORY_CEILING` holds.
#[cfg(feature = "cuda")]
fn pool_size(requested: Option<usize>, cpus: usize, cgroup_max: Option<u64>) -> usize {
    let memory = cgroup_max.map_or(JOB_MEMORY_CEILING, |max| max.min(JOB_MEMORY_CEILING));
    let ceiling = ((memory / NVRTC_CHILD_BYTES) as usize).max(1);
    requested.unwrap_or(cpus).clamp(1, ceiling)
}

/// The NVRTC version and target a cubin is valid for; an ahead cache
/// directory is bound to exactly this.
#[cfg(feature = "cuda")]
fn cubin_identity(caps: invar_device::Capabilities) -> String {
    let (major, minor) = invar_cuda::cuda::nvrtc_version().expect("NVRTC");
    format!("nvrtc {major}.{minor} {caps:?}")
}

/// Each distinct device source the census runs, once.
#[cfg(feature = "cuda")]
fn census_sources(entries: &[Entry]) -> Vec<&str> {
    let mut sources: Vec<&str> = entries
        .iter()
        .filter_map(|e| e.run.as_ref())
        .map(|run| run.plan.artifact().source())
        .collect();
    sources.sort_unstable();
    sources.dedup();
    sources
}

/// One ahead-of-time compile: the cubin or its typed refusal, and seconds.
#[cfg(feature = "cuda")]
type Compiled = (
    Result<Vec<u8>, invar::general_net::device_host::cuda_nvrtc::CompileRefusal>,
    f64,
);

/// Compile `sources` in a bounded pool of `invar-nvrtc` children.
#[cfg(feature = "cuda")]
fn compile_pool(caps: invar_device::Capabilities, sources: &[&str]) -> Vec<Compiled> {
    use invar::general_net::device_host::cuda_nvrtc;
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let jobs = compile_jobs();
    let started = std::time::Instant::now();
    let compiled = device::bounded_pool(sources, jobs, |source| {
        let started = std::time::Instant::now();
        let result = cuda_nvrtc::compile_ahead(compiler, caps, source);
        (result, started.elapsed().as_secs_f64())
    });
    println!(
        "gnet05ra census cuda ahead sources={} jobs={jobs} seconds={:.1}",
        sources.len(),
        started.elapsed().as_secs_f64()
    );
    compiled
}

#[cfg(feature = "cuda")]
fn source_key(source: &str) -> [u8; 32] {
    Sha256::digest(source).into()
}

#[cfg(feature = "cuda")]
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `INVAR_GNET05_CUDA_CAPS` (`MAJOR.MINOR/SM`, for example `12.4/90`): the
/// target the ahead compile builds for without opening a device.
#[cfg(feature = "cuda")]
fn caps_from_env() -> invar_device::Capabilities {
    let caps =
        std::env::var("INVAR_GNET05_CUDA_CAPS").expect("INVAR_GNET05_CUDA_CAPS=MAJOR.MINOR/SM");
    let parse = || -> Option<invar_device::Capabilities> {
        let (version, sm) = caps.split_once('/')?;
        let (major, minor) = version.split_once('.')?;
        Some(invar_device::Capabilities::Cuda {
            major: major.parse().ok()?,
            minor: minor.parse().ok()?,
            sm: sm.parse().ok()?,
        })
    };
    parse().unwrap_or_else(|| panic!("INVAR_GNET05_CUDA_CAPS={caps}: expected MAJOR.MINOR/SM"))
}

/// The CUDA census's ahead-of-time compile, without a GPU: run it on a CPU
/// host (no GPU slot held) with NVRTC on the library path. It writes
/// `identity`, one `<sha256>.cubin` per compiled source and `ahead.jsonl`
/// (seconds or the typed refusal per source) to `INVAR_GNET05_CUBIN_DIR`;
/// `issue_gnet_05_robotics_device_census_on_cuda` then only executes.
/// Without `INVAR_GNET05_CUBIN_DIR` (an `--include-ignored` GPU CI run) there
/// is nothing to compile ahead: the device census compiles what it lacks.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires NVRTC, INVAR_GNET05_CUBIN_DIR and INVAR_GNET05_CUDA_CAPS"]
fn issue_gnet_05_robotics_census_compile_ahead_for_cuda() {
    let Some(dir) = std::env::var_os("INVAR_GNET05_CUBIN_DIR").map(PathBuf::from) else {
        println!("gnet05ra census cuda ahead: INVAR_GNET05_CUBIN_DIR unset, nothing to compile");
        return;
    };
    let caps = caps_from_env();
    let entries = census_admission(compiled::Target::Cuda);
    let sources = census_sources(&entries);
    fs::create_dir_all(&dir).unwrap();
    let mut manifest = String::new();
    for (source, (result, seconds)) in sources.iter().zip(compile_pool(caps, &sources)) {
        let key = hex(&source_key(source));
        let row = match result {
            Ok(image) => {
                fs::write(dir.join(format!("{key}.cubin")), image).unwrap();
                json!({"source_sha256": key, "seconds": seconds})
            }
            Err(refusal) => {
                json!({"source_sha256": key, "seconds": seconds, "refused": refusal.class()})
            }
        };
        manifest.push_str(&format!("{row}\n"));
    }
    fs::write(dir.join("ahead.jsonl"), manifest).unwrap();
    fs::write(dir.join("identity"), cubin_identity(caps)).unwrap();
}

/// Ahead results for the census: read from `INVAR_GNET05_CUBIN_DIR` when it
/// is set (it must match this device's identity); any source it lacks is
/// compiled here in the bounded pool.
#[cfg(feature = "cuda")]
fn ahead_for(
    caps: invar_device::Capabilities,
    sources: &[&str],
) -> (invar::general_net::device_host::cuda_nvrtc::Cubins, Ahead) {
    use invar::general_net::device_host::cuda_nvrtc::Cubins;
    let mut cubins = Cubins::new(caps);
    let mut ahead = Ahead::new();
    let mut missing = Vec::new();
    let dir = std::env::var_os("INVAR_GNET05_CUBIN_DIR").map(PathBuf::from);
    let mut stored = std::collections::HashMap::new();
    if let Some(dir) = &dir {
        let identity = fs::read_to_string(dir.join("identity")).expect("ahead identity");
        assert_eq!(
            identity,
            cubin_identity(caps),
            "ahead cubins are for another NVRTC or device"
        );
        for line in fs::read_to_string(dir.join("ahead.jsonl")).unwrap().lines() {
            let row: Json = serde_json::from_str(line).unwrap();
            stored.insert(row["source_sha256"].as_str().unwrap().to_owned(), row);
        }
    }
    for source in sources {
        let key = source_key(source);
        match (&dir, stored.get(&hex(&key))) {
            (Some(dir), Some(row)) => {
                let seconds = row["seconds"].as_f64().unwrap();
                match row["refused"].as_str() {
                    Some(class) => {
                        ahead.insert(key, Err(class.to_owned()));
                    }
                    None => {
                        let image = fs::read(dir.join(format!("{}.cubin", hex(&key)))).unwrap();
                        cubins.insert(source, image);
                        ahead.insert(key, Ok(seconds));
                    }
                }
            }
            _ => missing.push(*source),
        }
    }
    if !missing.is_empty() {
        for (source, (result, seconds)) in missing.iter().zip(compile_pool(caps, &missing)) {
            match result {
                Ok(image) => {
                    cubins.insert(source, image);
                    ahead.insert(source_key(source), Ok(seconds));
                }
                Err(refusal) => {
                    ahead.insert(source_key(source), Err(refusal.class().to_owned()));
                }
            }
        }
    }
    (cubins, ahead)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device and NVRTC; array programs need ABI13"]
fn issue_gnet_05_robotics_device_census_on_cuda() {
    use invar::general_net::device_host::cuda_nvrtc::CudaNvrtc;
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let entries = census_admission(compiled::Target::Cuda);
    let caps = CudaNvrtc::open(compiler)
        .expect("CUDA device")
        .capabilities();
    // Each distinct device source compiles once (ideally ahead, outside the
    // GPU slot); both capacities of every config load that cubin.
    let (cubins, ahead) = ahead_for(caps, &census_sources(&entries));
    let cubins = std::sync::Arc::new(cubins);
    census_run("cuda", entries, &ahead, || {
        CudaNvrtc::open(compiler)
            .expect("CUDA device")
            .with_cubins(cubins.clone())
            .expect("cubins for this device")
    });
}

#[cfg(feature = "cuda")]
#[test]
fn compile_pool_never_exceeds_the_job_memory_ceiling() {
    const GIB: u64 = 1 << 30;
    // A large or unlimited cgroup still holds only 12 GiB of children.
    assert_eq!(pool_size(None, 64, Some(20 * GIB)), 6);
    assert_eq!(pool_size(None, 64, None), 6);
    // An explicit request is clamped the same way.
    assert_eq!(pool_size(Some(16), 64, Some(64 * GIB)), 6);
    assert_eq!(pool_size(Some(0), 64, None), 1);
    // Smaller cgroups and fewer CPUs bound it further, never below one.
    assert_eq!(pool_size(None, 64, Some(8 * GIB)), 4);
    assert_eq!(pool_size(None, 3, Some(20 * GIB)), 3);
    assert_eq!(pool_size(Some(2), 64, Some(20 * GIB)), 2);
    assert_eq!(pool_size(None, 64, Some(GIB)), 1);
}

#[test]
fn bounded_pool_keeps_input_order_and_its_worker_bound() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let (live, peak) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let items: Vec<u32> = (0..40).collect();
    let got = device::bounded_pool(&items, 3, |&x| {
        let now = live.fetch_add(1, Ordering::SeqCst) + 1;
        peak.fetch_max(now, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(2));
        live.fetch_sub(1, Ordering::SeqCst);
        x * 2
    });
    assert_eq!(got, items.iter().map(|x| x * 2).collect::<Vec<_>>());
    assert!((1..=3).contains(&peak.load(Ordering::SeqCst)));
    assert!(device::bounded_pool(&[] as &[u32], 4, |&x| x).is_empty());
}
