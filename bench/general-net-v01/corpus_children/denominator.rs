//! Independent source groups run concurrently; receipts retain manifest order.
//! Every row is checked against its source and oracle expectations; only the
//! sampled sources (`corpus_sample`) also compile and run the CPU backends.
use super::*;
use std::collections::BTreeSet;

type Cache = BTreeMap<(String, String, u8, usize), Harness>;

pub(super) fn run() {
    let ordinary = base::document("expanded-manifest.json");
    let runtime = base::document("runtime-vectors.json");
    let added = base::document("child-vectors.json");
    let queues = queues([&ordinary, &runtime, &added]);
    let keep = sample(&queues);
    let mut results = std::thread::scope(|scope| {
        let workers = queues
            .into_iter()
            .map(|queue| {
                let (ordinary, keep) = (&ordinary, &keep);
                scope.spawn(move || {
                    let mut cache = Cache::new();
                    queue
                        .into_iter()
                        .map(|(index, kind, case)| {
                            let (row, comparisons) = row(kind, case, ordinary, keep, &mut cache);
                            (index, row, comparisons)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    results.sort_by_key(|(index, _, _)| *index);
    let comparisons: usize = results.iter().map(|(_, _, count)| count).sum();
    let rows = results
        .into_iter()
        .map(|(_, row, _)| row)
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 158);
    eprintln!(
        "158 complete child corpus rows, {comparisons} actual CPU/model comparisons; old119 rows retained"
    );
    report(
        "children-full",
        rows,
        json!({"retained_ordinary":52,"retained_runtime":67,"new_runtime_and_main":39,"matched_cpu_model_observations":comparisons}),
    );
}

/// The sampled source paths; a category is the case kind and source directory.
fn sample(queues: &[Vec<(usize, &str, &Json)>]) -> BTreeSet<String> {
    corpus_sample::paths(queues.iter().flatten().map(|(_, kind, case)| {
        let path = case["source"].as_str().unwrap();
        (
            path.to_owned(),
            format!("{kind}:{}", corpus_sample::category(path)),
        )
    }))
}

fn queues(documents: [&Json; 3]) -> [Vec<(usize, &str, &Json)>; 4] {
    let mut ordinal = 0;
    let mut groups = BTreeMap::<&str, Vec<_>>::new();
    for (kind, document) in [
        ("ordinary", documents[0]),
        ("runtime", documents[1]),
        ("child", documents[2]),
    ] {
        for case in document["cases"].as_array().unwrap() {
            groups
                .entry(case["source"].as_str().unwrap())
                .or_default()
                .push((ordinal, kind, case));
            ordinal += 1;
        }
    }
    let mut queues = [vec![], vec![], vec![], vec![]];
    for (index, group) in groups.into_values().enumerate() {
        queues[index % 4].extend(group);
    }
    queues
}

fn source_bytes(case: &Json, ordinary: &Json) -> Vec<u8> {
    let path = case["source"].as_str().unwrap();
    let bytes = fs::read(base::root().join(path)).unwrap();
    let hash = case["source_sha256"]
        .as_str()
        .or_else(|| {
            ordinary["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["source"].as_str() == Some(path))
                .and_then(|row| row["source_sha256"].as_str())
        })
        .unwrap();
    assert_eq!(
        base::digest(&bytes),
        hash,
        "{} source bytes",
        case["id"].as_str().unwrap()
    );
    bytes
}

struct Case<'a> {
    kind: &'a str,
    case: &'a Json,
    net: Net,
    inputs: Vec<eval::Value>,
    entry: &'a str,
    limits: oracle::Limits,
    expected: oracle::Observation,
    source: eval::Observation,
}

fn observe<'a>(kind: &'a str, case: &'a Json, checked: check::CheckedProgram) -> Case<'a> {
    let entry = if kind == "ordinary" {
        "main"
    } else {
        case["entry"].as_str().unwrap()
    };
    let inputs = if kind == "ordinary" {
        vec![]
    } else {
        args(&case["arguments"])
    };
    let limits = limits(case);
    let source = if kind == "ordinary" {
        eval::observe(&checked, limits.fuel)
    } else {
        eval::named::observe_named(&checked, entry, &inputs, limits.fuel)
    };
    let expectation = if kind == "ordinary" {
        "expected_reference_run"
    } else {
        "expected_source"
    };
    let id = case["id"].as_str().unwrap();
    base::expectation(&source, &case[expectation], id);
    let net = Net::lower(&checked, general_net::Limits::default()).unwrap();
    let expected =
        oracle::observe_call(&net, &net.verify().unwrap(), entry, &inputs, limits).unwrap();
    match expected.failure {
        None => base::equal(&expected.semantic, &source, id),
        Some(oracle::ToolFailure::Unsupported) => {}
        Some(failure) => assert_eq!(case["expected_oracle_failure"], format!("{failure:?}")),
    }
    Case {
        kind,
        case,
        net,
        inputs,
        entry,
        limits,
        expected,
        source,
    }
}

fn row(
    kind: &str,
    case: &Json,
    ordinary: &Json,
    keep: &BTreeSet<String>,
    cache: &mut Cache,
) -> (Json, usize) {
    let bytes = source_bytes(case, ordinary);
    let checked = match checked(&bytes) {
        Ok(c) => c,
        Err(e) => {
            assert_eq!(case["expected_check"]["diagnostic_codes"][0], e.code);
            let rejection = json!({"status":"source_rejected","reason":e.code});
            return (
                json!({"id":case["id"],"kind":kind,"source":rejection,"oracle":rejection,"abi4_cpu":rejection,"abi5_cuda_cpu_model":rejection,"abi5_metal_cpu_model":rejection}),
                0,
            );
        }
    };
    let context = observe(kind, case, checked);
    let mut row = json!({"id":case["id"],"kind":kind,"source_sha256":base::digest(&bytes),"entry":context.entry,"arguments":case["arguments"],"source":base::recorded(&context.source),"oracle":observed(&context.expected)});
    let mut comparisons = 0;
    for (target, key, tag) in [
        (compiled::Target::Cpu, "abi4_cpu", 4),
        (compiled::Target::Cuda, "abi5_cuda_cpu_model", 5),
        (compiled::Target::Metal, "abi5_metal_cpu_model", 6),
    ] {
        let owned = if context.expected.failure == Some(oracle::ToolFailure::Allocation) {
            compiled::OwnedLimits {
                elements: context.limits.array_elements,
                ..storage()
            }
        } else {
            storage()
        };
        match emit(&context.net, context.entry, target, private(), owned) {
            Err(error) => row[key] = rejected(error),
            Ok(_) if !keep.contains(case["source"].as_str().unwrap()) => {
                row[key] = json!({"status":"sampled_out"});
            }
            Ok(artifact) => {
                let harness = cache
                    .entry((
                        case["source"].as_str().unwrap().to_owned(),
                        context.entry.to_owned(),
                        tag,
                        owned.elements,
                    ))
                    .or_insert_with(|| Harness::new(artifact));
                let runs = runs(&context, harness);
                comparisons += runs.len();
                row[key] = json!({"status":"executed","runs":runs});
            }
        }
    }
    (row, comparisons)
}

fn runs(context: &Case<'_>, harness: &Harness) -> Vec<Json> {
    let mut runs = Vec::new();
    for (quanta, capacity) in [(vec![1], 2), (vec![3, 17], 2), (vec![17], 0)] {
        let run = harness.execute(&context.inputs, context.limits, &quanta, capacity, "");
        let mut receipt = harness.compare(&context.net, &context.expected, &run);
        receipt["quanta"] = json!(quanta);
        receipt["capacity"] = json!(capacity);
        runs.push(receipt);
        if context.kind == "child"
            && context.case["id"]
                .as_str()
                .unwrap()
                .starts_with("ordered_0_")
        {
            let program = context
                .net
                .compiled_mir(&context.net.verify().unwrap())
                .unwrap();
            let later = program
                .functions
                .iter()
                .position(|f| f.name == "later")
                .unwrap();
            assert!(
                context
                    .expected
                    .events
                    .iter()
                    .all(|e| e.token.block.function != later)
            );
            assert_eq!(
                run.stats[1], 0,
                "no private child commits before divergent oldest call"
            );
        }
    }
    runs
}
