//! GNET-05 robotics corpus: host-array variants and device admission census.
//!
//! The array programs (`bench/general-net-v01/gnet05/robotics/array`) take
//! their bulk inputs as `read Array<f32>` host buffers and write the output
//! into a `mut Array<f32>`. Their files, held-out guard and reference/oracle
//! outputs are checked by default through the ABI13 host-array entry; the
//! device census needs a GPU and stays ignored.
//! Regenerate with `INVAR_GNET05_WRITE=1` (validation digests need Release).
#![forbid(unsafe_code)]

use crate::agree;
use crate::device;
use crate::generate;
use crate::guard;
use crate::par;
#[cfg(any(feature = "metal", feature = "cuda"))]
use crate::parallel;
use crate::robotics;

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

#[cfg(any(feature = "metal", feature = "cuda"))]
mod census;

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
fn robotics_array_files_are_exactly_the_generator_output() {
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
fn robotics_array_held_out_programs_never_reach_tuning() {
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
fn robotics_array_outputs_equal_seed_outputs() {
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
