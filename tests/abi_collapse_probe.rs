//! TEMPORARY evidence probe (deleted before merge): runs the whole GNET corpus
//! through the loaded CPU ABI8, ABI9, ABI10 and ABI11 hosts and records every
//! admission and every observation (Debug form is bit-exact: F32Bits prints its
//! encoding) for the full fuel and each fuel prefix.
use invar::{
    eval,
    general_net::{
        self, compiled, numeric_host, numeric_owned_host, numeric_owned_resume_host,
        numeric_resume_host, oracle,
    },
    numeric_contract::F32Bits,
};
use serde_json::{Value as Json, json};
use std::{collections::BTreeMap, fs, path::Path};

const ROOT: &str = "bench/general-net-v01";
const MAX_PREFIXES: u64 = 400;

fn args(row: &Json) -> Vec<eval::Value> {
    row.as_array()
        .map(|a| a.as_slice())
        .unwrap_or(&[])
        .iter()
        .map(|v| match v["type"].as_str().unwrap() {
            "f32" => eval::Value::F32(F32Bits::from_bits(v["bits"].as_u64().unwrap() as u32)),
            "i32" => eval::Value::I32(v["value"].as_i64().unwrap() as i32),
            "u32" => eval::Value::U32(v["value"].as_u64().unwrap() as u32),
            "bool" => eval::Value::Bool(v["value"].as_bool().unwrap()),
            t => panic!("argument type {t}"),
        })
        .collect()
}

fn storage() -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: 128,
        allocations: 128,
        cleanup_events: 128,
    }
}

fn err<E: std::fmt::Debug>(e: E) -> String {
    let s = format!("ERR {e:?}");
    // Drop evidence paths so equal failures compare equal across ABIs.
    s.split("evidence")
        .next()
        .unwrap()
        .chars()
        .take(300)
        .collect()
}

enum Host {
    A8(numeric_host::Plan),
    A9(numeric_resume_host::Plan),
    A10(numeric_owned_host::Plan),
    A11(numeric_owned_resume_host::Plan),
}

impl Host {
    fn build(abi: u32, source: &[u8], entry: &str) -> Result<Self, String> {
        let c = Path::new("c++");
        let l = Default::default();
        match abi {
            8 => numeric_host::Plan::from_source(source, entry, l, c)
                .map(Self::A8)
                .map_err(err),
            9 => numeric_resume_host::Plan::from_source(source, entry, l, c)
                .map(Self::A9)
                .map_err(err),
            10 => numeric_owned_host::Plan::from_source(source, entry, l, storage(), c)
                .map(Self::A10)
                .map_err(err),
            _ => numeric_owned_resume_host::Plan::from_source(source, entry, l, storage(), c)
                .map(Self::A11)
                .map_err(err),
        }
    }
    fn run(&mut self, xs: &[eval::Value], l: oracle::Limits, quantum: u64) -> String {
        match self {
            Self::A8(p) => p.run(xs, l).map(|o| format!("{o:?}")).unwrap_or_else(err),
            Self::A10(p) => p.run(xs, l).map(|o| format!("{o:?}")).unwrap_or_else(err),
            Self::A9(p) => {
                let mut run = match p.start(xs, l) {
                    Ok(r) => r,
                    Err(e) => return err(e),
                };
                loop {
                    match run.resume(quantum) {
                        Ok(numeric_resume_host::Progress::Pending(_)) => {}
                        Ok(numeric_resume_host::Progress::Complete(o)) => return format!("{o:?}"),
                        Err(e) => return err(e),
                    }
                }
            }
            Self::A11(p) => {
                let mut run = match p.start(xs, l) {
                    Ok(r) => r,
                    Err(e) => return err(e),
                };
                loop {
                    match run.resume(quantum) {
                        Ok(numeric_owned_resume_host::Progress::Pending(_)) => {}
                        Ok(numeric_owned_resume_host::Progress::Complete(o)) => {
                            return format!("{o:?}");
                        }
                        Err(e) => return err(e),
                    }
                }
            }
        }
    }
}

struct Row {
    id: String,
    path: String,
    entry: String,
    args: Vec<eval::Value>,
    limits: oracle::Limits,
}

fn rows() -> Vec<Row> {
    let mut out = Vec::new();
    for doc in [
        "expanded-manifest.json",
        "runtime-vectors.json",
        "child-vectors.json",
        "numeric-vectors.json",
    ] {
        let d: Json =
            serde_json::from_slice(&fs::read(Path::new(ROOT).join(doc)).unwrap()).unwrap();
        let ordinary = doc == "expanded-manifest.json";
        for case in d["cases"].as_array().unwrap() {
            out.push(Row {
                id: format!("{doc}:{}", case["id"].as_str().unwrap()),
                path: format!("{ROOT}/{}", case["source"].as_str().unwrap()),
                entry: if ordinary {
                    "main".into()
                } else {
                    case["entry"].as_str().unwrap().into()
                },
                args: if ordinary {
                    vec![]
                } else {
                    args(&case["arguments"])
                },
                limits: oracle::Limits {
                    fuel: case["fuel"].as_u64().unwrap(),
                    transitions: case["limits"]["transitions"].as_u64().unwrap_or(100_000),
                    trace_events: case["limits"]["trace_events"].as_u64().unwrap_or(100_000)
                        as usize,
                    array_elements: case["limits"]["array_elements"].as_u64().unwrap_or(128)
                        as usize,
                },
            });
        }
    }
    let mut fixtures: Vec<_> = fs::read_dir("tests/issue_gnet_10_f32_arrays_fixtures")
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "inv"))
        .collect();
    fixtures.sort();
    for p in fixtures {
        out.push(Row {
            id: format!("fixture:{}", p.display()),
            path: p.display().to_string(),
            entry: "main".into(),
            args: vec![],
            limits: oracle::Limits {
                fuel: 1000,
                transitions: 100_000,
                trace_events: 10_000,
                array_elements: 128,
            },
        });
    }
    out
}

fn oracle_string(source: &[u8], entry: &str, xs: &[eval::Value], l: oracle::Limits) -> String {
    let Ok(c) = invar::check::check(
        match invar::front::parse(source) {
            Ok(p) => p,
            Err(e) => return err(e),
        },
        true,
    ) else {
        return "ERR check".into();
    };
    if invar::ownership::check(&c).is_err() {
        return "ERR ownership".into();
    }
    let Ok(net) = general_net::Net::lower(&c, general_net::Limits::default()) else {
        return "ERR lower".into();
    };
    let Ok(proof) = net.verify() else {
        return "ERR verify".into();
    };
    oracle::observe_call(&net, &proof, entry, xs, l)
        .map(|o| format!("{o:?}"))
        .unwrap_or_else(err)
}

#[test]
fn abi_collapse_probe() {
    let rows = rows();
    let mut plans: BTreeMap<(String, String, u32), Result<Host, String>> = BTreeMap::new();
    let mut report = Vec::new();
    let mut summary = json!({"rows": rows.len(), "admitted": {}, "older_admitted_abi11_refused": [], "observation_mismatches": 0, "runs": 0, "abi11_vs_oracle_mismatches": 0});
    let mut admitted = BTreeMap::<u32, usize>::new();
    let mut mismatches = 0usize;
    let mut oracle_mismatches = 0usize;
    let mut runs = 0usize;
    let mut refused = Vec::new();
    for row in &rows {
        let source = fs::read(&row.path).unwrap();
        let mut admission = BTreeMap::new();
        for abi in 8..=11 {
            let key = (row.path.clone(), row.entry.clone(), abi);
            let plan = plans
                .entry(key)
                .or_insert_with(|| Host::build(abi, &source, &row.entry));
            admission.insert(abi, plan.as_ref().err().cloned());
        }
        for (abi, e) in &admission {
            if e.is_none() {
                *admitted.entry(*abi).or_default() += 1;
            }
        }
        if admission[&11].is_some() && (8..=10).any(|a| admission[&a].is_none()) {
            refused.push(json!({"row": row.id, "admission": format!("{admission:?}")}));
        }
        // Full fuel (with several quanta) plus every fuel prefix up to steps + 1.
        let full = row.limits;
        let reference = oracle_string(&source, &row.entry, &row.args, full);
        let steps = reference
            .split("steps_used: ")
            .nth(1)
            .and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let mut fuels = vec![(full, 1_000_000u64), (full, 1), (full, 7)];
        for fuel in 1..=(steps + 1).min(MAX_PREFIXES).min(full.fuel) {
            fuels.push((oracle::Limits { fuel, ..full }, 1_000_000));
        }
        let mut diffs = Vec::new();
        for (l, quantum) in fuels {
            let mut seen = BTreeMap::new();
            for abi in 8..=11 {
                let key = (row.path.clone(), row.entry.clone(), abi);
                if let Some(Ok(host)) = plans.get_mut(&key) {
                    // ABI8/10 ignore the quantum; run them once per fuel.
                    if quantum != 1_000_000 && matches!(abi, 8 | 10) {
                        continue;
                    }
                    seen.insert(abi, host.run(&row.args, l, quantum));
                    runs += 1;
                }
            }
            let Some(latest) = seen.get(&11).cloned() else {
                continue;
            };
            if l.fuel == full.fuel
                && quantum == 1_000_000
                && latest != oracle_string(&source, &row.entry, &row.args, l)
            {
                oracle_mismatches += 1;
                diffs.push(json!({"fuel": l.fuel, "quantum": quantum, "kind": "abi11_vs_oracle", "abi11": latest, "oracle": reference}));
            }
            for (abi, obs) in &seen {
                if *abi != 11 && *obs != latest {
                    mismatches += 1;
                    if diffs.len() < 6 {
                        diffs.push(json!({"fuel": l.fuel, "quantum": quantum, "abi": abi, "older": obs, "abi11": latest}));
                    }
                }
            }
        }
        report.push(json!({"row": row.id, "entry": row.entry, "admission": admission.iter().map(|(k,v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_,_>>(), "prefixes": steps, "diffs": diffs}));
    }
    summary["admitted"] = json!(admitted);
    summary["older_admitted_abi11_refused"] = json!(refused);
    summary["observation_mismatches"] = json!(mismatches);
    summary["abi11_vs_oracle_mismatches"] = json!(oracle_mismatches);
    summary["runs"] = json!(runs);
    let dir = std::env::var("ABI_PROBE_OUT").unwrap_or_else(|_| "target/abi-probe".into());
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        format!("{dir}/summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
    fs::write(
        format!("{dir}/rows.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
}
