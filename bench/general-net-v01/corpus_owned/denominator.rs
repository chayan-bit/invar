//! Per-source caches stay local to workers; report rows retain manifest order.
use super::*;

type Cache = BTreeMap<(String, String, bool, usize), Cpu>;

pub(super) fn run() {
    let manifest = document("expanded-manifest.json");
    let vectors = document("runtime-vectors.json");
    let config = document("owned-vectors.json");
    let storage = owned_limits(&config["storage"]);
    let schedules = schedules(&config);
    let mut results = std::thread::scope(|scope| {
        let workers = queues([&manifest, &vectors])
            .into_iter()
            .map(|queue| {
                let schedules = &schedules;
                scope.spawn(move || {
                    let mut cache = Cache::new();
                    queue
                        .into_iter()
                        .map(|(index, kind, case)| {
                            let (row, matched) = row(kind, case, storage, schedules, &mut cache);
                            (index, row, matched)
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
    let matched: usize = results.iter().map(|(_, _, count)| count).sum();
    let rows = results
        .into_iter()
        .map(|(_, row, _)| row)
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 119);
    eprintln!(
        "119 full-denominator ABI2/3 rows; {matched} real CPU observations including runtime changing quanta"
    );
    report(
        "owned-full",
        rows,
        json!({"ordinary_denominator":52,"runtime_denominator":67,"matched_cpu_observations":matched,"quantum_schedules":schedules}),
    );
}

fn queues(documents: [&Json; 2]) -> [Vec<(usize, &str, &Json)>; 4] {
    let mut ordinal = 0;
    let mut groups = BTreeMap::<&str, Vec<_>>::new();
    for (kind, document) in [("ordinary", documents[0]), ("runtime", documents[1])] {
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

struct Case<'a> {
    id: &'a str,
    case: &'a Json,
    net: Net,
    inputs: Vec<eval::Value>,
    entry: &'a str,
    limits: oracle::Limits,
    expected: oracle::Observation,
    source: eval::Observation,
}

fn case_limits(kind: &str, case: &Json) -> oracle::Limits {
    if kind == "ordinary" {
        oracle::Limits {
            fuel: case["fuel"].as_u64().unwrap(),
            trace_events: 100000,
            ..oracle::Limits::default()
        }
    } else {
        limits(case)
    }
}

fn observe_case<'a>(kind: &str, case: &'a Json, checked: check::CheckedProgram) -> Case<'a> {
    let id = case["id"].as_str().unwrap();
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
    let limits = case_limits(kind, case);
    let source = if kind == "ordinary" {
        eval::observe(&checked, limits.fuel)
    } else {
        eval::named::observe_named(&checked, entry, &inputs, limits.fuel)
    };
    expectation(
        &source,
        if kind == "ordinary" {
            &case["expected_reference_run"]
        } else {
            &case["expected_source"]
        },
        id,
    );
    let net = Net::lower(&checked, general_net::Limits::default()).unwrap();
    let expected = observe(&net, entry, &inputs, limits);
    match expected.failure {
        None => equal(&expected.semantic, &source, id),
        Some(oracle::ToolFailure::Unsupported) => {}
        Some(failure) => assert_eq!(
            case["expected_oracle_failure"],
            format!("{failure:?}"),
            "{id}"
        ),
    }
    Case {
        id,
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
    storage: compiled::OwnedLimits,
    schedules: &[Vec<u64>],
    cache: &mut Cache,
) -> (Json, usize) {
    let id = case["id"].as_str().unwrap();
    let bytes = fs::read(root().join(case["source"].as_str().unwrap())).unwrap();
    let checked = match checked(&bytes) {
        Ok(c) => c,
        Err(error) => {
            assert_eq!(case["expected_check"]["diagnostic_codes"][0], error.code);
            let rejection = json!({"status":"source_rejected","reason":error.code});
            return (
                json!({"id":id,"kind":kind,"source":rejection,"oracle":rejection,"owned_cpu":rejection,"resumable_cpu":rejection}),
                0,
            );
        }
    };
    let context = observe_case(kind, case, checked);
    // The pre-existing allocation vector retains its lower private backend cap.
    // No source cap is replaced by this budget.
    let storage = if context.expected.failure == Some(oracle::ToolFailure::Allocation) {
        compiled::OwnedLimits {
            elements: context.limits.array_elements,
            ..storage
        }
    } else {
        storage
    };
    let (owned, first) = backend(&context, false, storage, schedules, cache);
    let (resumable, second) = backend(&context, true, storage, schedules, cache);
    source_order(&context);
    (
        json!({"id":id,"kind":kind,"source_sha256":digest(&bytes),"entry":context.entry,"arguments":case["arguments"],"source":recorded(&context.source),"oracle":observed(&context.expected),"storage":{"elements":storage.elements,"allocations":storage.allocations,"cleanup_events":storage.cleanup_events},"owned_cpu":owned,"resumable_cpu":resumable}),
        first + second,
    )
}

fn backend(
    context: &Case<'_>,
    resumable: bool,
    storage: compiled::OwnedLimits,
    schedules: &[Vec<u64>],
    cache: &mut Cache,
) -> (Json, usize) {
    let artifact = match emit(&context.net, context.entry, resumable, storage) {
        Ok(artifact) => artifact,
        Err(error) => return (compiled_rejection(error), 0),
    };
    let key = (
        context.case["source"].as_str().unwrap().to_owned(),
        context.entry.to_owned(),
        resumable,
        storage.elements,
    );
    let cpu = cache.entry(key).or_insert_with(|| Cpu::new(&artifact));
    let schedules = if resumable {
        schedules.to_vec()
    } else {
        vec![vec![1]]
    };
    let runs = schedules
        .into_iter()
        .map(|quanta| {
            let run = cpu.execute(&context.inputs, context.limits, &quanta);
            let mut receipt = cpu.compare(&context.net, &context.expected, &run);
            receipt["quanta"] = json!(quanta);
            receipt
        })
        .collect::<Vec<_>>();
    let count = runs.len();
    (json!({"status":"executed","runs":runs}), count)
}

fn source_order(context: &Case<'_>) {
    if context.id == "ordered_earlier_divergence" {
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
                .all(|e| e.token.block.function != later),
            "older divergent call never demands later trap"
        );
    }
}
