//! M5-02 qualification: CPU program observations and explicit device family rows.
use invar::{
    admission, check, eval, front, general_net, mir, mir_eval, mir_net, ownership, schedule,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

pub mod corpus;
// Optional hardware bodies and the shared read-only family corpora are not
// executed by a CPU-only binary; their boundary checks remain callable.
#[allow(dead_code)]
pub mod device;
use corpus::{Case, Expected};

pub const SUITE: &str = "issue_m5_02_full_pipeline_qualification";
const ENGINES: [&str; 14] = [
    "reference",
    "mir",
    "interaction-extracted",
    "net-extracted",
    "native-serial",
    "native-net",
    "cpu-parallel-1",
    "cpu-parallel-2",
    "cpu-parallel-4",
    "net-oracle",
    "cuda-generated",
    "cuda-library-assisted",
    "metal-generated",
    "metal-library-assisted",
];

#[derive(Debug)]
pub struct Failure {
    pub code: &'static str,
    pub detail: String,
}

impl Failure {
    fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

fn row(engine: &str, status: &str, diagnostic: Option<&str>, observation: Value) -> Value {
    json!({"engine":engine,"status":status,"diagnostic":diagnostic,"observation":observation})
}

fn semantic(observed: &eval::Observation) -> Value {
    let mut document = match &observed.result {
        Ok(eval::Value::I32(value)) => json!({"status":"ok","result":{"type":"i32","value":value}}),
        Ok(value) => json!({"status":"unexpected_value","value":format!("{value:?}")}),
        Err(error) => {
            let mut document = error.to_json("");
            // Messages are descriptive; codes and spans define semantic identity.
            if let eval::EvalError::Tool { code, .. } = error {
                document["tool_code"] = json!(code);
            }
            document
        }
    };
    document["steps_used"] = json!(observed.steps_used);
    document["cleanup_trace"] = json!(
        observed
            .cleanup_trace
            .iter()
            .map(|x| json!({"name":x.name,"length":x.length}))
            .collect::<Vec<_>>()
    );
    identity(&document)
}

fn identity(document: &Value) -> Value {
    json!({"status":document["status"],"result":document["result"],"trap":document["trap"],"fuel_limit":document["fuel_limit"],"steps_used":document["steps_used"],"tool_code":document["tool_code"],"cleanup_trace":document["cleanup_trace"]})
}

/// Shared comparison boundary also used by deliberate wrong-result mutations.
pub fn agree(expected: &Value, actual: &Value) -> Result<(), Failure> {
    if identity(expected) != identity(actual) {
        return Err(Failure::new(
            "M5_SEMANTIC_DISAGREEMENT",
            format!(
                "expected {}, actual {}",
                identity(expected),
                identity(actual)
            ),
        ));
    }
    Ok(())
}

fn expected(case: &Case, observed: &eval::Observation) -> bool {
    match (&case.expected, &observed.result) {
        (Expected::Value(a), Ok(eval::Value::I32(b))) => a == b,
        (
            Expected::Trap(code, span),
            Err(eval::EvalError::Trap {
                category,
                span: actual,
            }),
        ) => {
            *code
                == match category {
                    eval::TrapCategory::IntegerOverflow => "INTEGER_OVERFLOW",
                    eval::TrapCategory::DivisionByZero => "DIVISION_BY_ZERO",
                    eval::TrapCategory::IndexOutOfBounds => "INDEX_OUT_OF_BOUNDS",
                    eval::TrapCategory::UnitConversionInexact => "UNIT_CONVERSION_INEXACT",
                }
                && span == actual
        }
        (
            Expected::Limit,
            Err(eval::EvalError::EvaluationLimit {
                fuel_limit,
                steps_used,
            }),
        ) => *fuel_limit == case.fuel && *steps_used == case.fuel,
        (Expected::Tool(a), Err(eval::EvalError::Tool { code: b, .. })) => a == b,
        _ => false,
    }
}

fn front_check(source: &[u8]) -> Result<check::CheckedProgram, front::FrontError> {
    let checked = check::check(front::parse(source)?, true)?;
    ownership::check(&checked)?;
    Ok(checked)
}

/// Limits for the harness's own pipeline stages. `run_case` uses the defaults;
/// tests inject tighter ones to reach the stage-failure codes.
pub struct Bounds {
    pub interaction: mir_net::Limits,
    pub net: general_net::Limits,
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            interaction: INTERACTION_LIMITS,
            net: general_net::Limits::default(),
        }
    }
}

const INTERACTION_LIMITS: mir_net::Limits = mir_net::Limits {
    functions: 128,
    blocks: 4096,
    instructions: 100_000,
    wires: 400_000,
};

pub fn run_case(case: &Case) -> Result<Value, Failure> {
    run_case_with(case, &Bounds::default())
}

pub fn run_case_with(case: &Case, bounds: &Bounds) -> Result<Value, Failure> {
    let checked = match front_check(&case.source) {
        Ok(checked) => checked,
        Err(error) => {
            if case.expected != Expected::Refusal(error.code) {
                return Err(Failure::new(
                    "M5_UNEXPECTED_REFUSAL",
                    format!("{}: {error:?}", case.id),
                ));
            }
            let diagnostic = error.to_json(&case.id);
            let rows = ENGINES
                .iter()
                .map(|engine| row(engine, "blocked", Some(error.code), Value::Null))
                .collect::<Vec<_>>();
            return Ok(
                json!({"id":case.id,"class":case.class,"source_sha256":digest(&case.source),"source_bytes":case.source.len(),"fuel":case.fuel,"frontend":"refused","diagnostic":diagnostic,"rows":rows}),
            );
        }
    };
    if matches!(case.expected, Expected::Refusal(_)) {
        return Err(Failure::new("M5_MISSING_REFUSAL", &case.id));
    }
    let program = mir::lower(&checked)
        .map_err(|e| Failure::new(e.code, format!("{}: {}", case.id, e.message)))?;
    mir::verify(&program).map_err(|e| Failure::new(e.code, e.message))?;
    let graph = mir_net::Net::encode(&program, bounds.interaction)
        .map_err(|e| Failure::new("M5_INTERACTION", format!("{e:?}")))?;
    let extracted = graph
        .extract(
            &graph
                .verify()
                .map_err(|e| Failure::new("M5_INTERACTION", format!("{e:?}")))?,
        )
        .map_err(|e| Failure::new("M5_INTERACTION", format!("{e:?}")))?;
    let net = general_net::Net::lower(&checked, bounds.net)
        .map_err(|e| Failure::new("M5_NET", format!("{e:?}")))?;
    let proof = net
        .verify()
        .map_err(|e| Failure::new("M5_NET", format!("{e:?}")))?;
    let compiled = net
        .compiled_mir(&proof)
        .map_err(|e| Failure::new("M5_NET", format!("{e:?}")))?;
    let reference = eval::observe(&checked, case.fuel);
    if !expected(case, &reference) {
        return Err(Failure::new(
            "M5_INDEPENDENT_EXPECTATION",
            format!(
                "{}: expected {:?}, actual {:?}",
                case.id, case.expected, reference
            ),
        ));
    }
    let expected = semantic(&reference);
    let mut rows = vec![row("reference", "qualified", None, expected.clone())];
    for (engine, p) in [
        ("mir", &program),
        ("interaction-extracted", &extracted),
        ("net-extracted", &compiled),
    ] {
        let actual = semantic(&mir_eval::observe(p, case.fuel));
        agree(&expected, &actual)
            .map_err(|e| Failure::new(e.code, format!("{} {engine}: {}", case.id, e.detail)))?;
        rows.push(row(engine, "qualified", None, actual));
    }
    for (engine, p) in [("native-serial", &program), ("native-net", &compiled)] {
        #[cfg(feature = "native")]
        {
            let (_, actual) = invar::native::run_program(p, case.fuel)
                .map_err(|e| Failure::new("M5_NATIVE", e))?;
            agree(&expected, &actual)
                .map_err(|e| Failure::new(e.code, format!("{} {engine}: {}", case.id, e.detail)))?;
            rows.push(row(engine, "qualified", None, identity(&actual)));
        }
        #[cfg(not(feature = "native"))]
        {
            let _ = p;
            rows.push(row(
                engine,
                "unavailable",
                Some("NATIVE_FEATURE_REQUIRED"),
                Value::Null,
            ));
        }
    }
    for workers in [1, 2, 4] {
        let run = schedule::run_source(&case.source, &compiled, case.fuel, workers)
            .map_err(|e| Failure::new(e.code(), &case.id))?;
        let actual = semantic(&run.observation);
        agree(&expected, &actual).map_err(|e| {
            Failure::new(
                e.code,
                format!("{} cpu-parallel-{workers}: {}", case.id, e.detail),
            )
        })?;
        let mut item = row(
            &format!("cpu-parallel-{workers}"),
            "qualified",
            None,
            actual,
        );
        item["dispatch"] = json!({"plan":format!("{:?}",run.plan),"workers_used":run.workers_used,"published_regions":run.executed.len(),"fallback_reason":run.fallback_reason});
        rows.push(item);
    }
    let oracle = net_oracle(&net, &proof, case.fuel)?;
    if oracle.failure == Some(general_net::oracle::ToolFailure::Unsupported) {
        rows.push(row(
            "net-oracle",
            "unavailable",
            Some("UNSUPPORTED_FEATURE"),
            semantic(&oracle.semantic),
        ));
    } else {
        let actual = semantic(&oracle.semantic);
        agree(&expected, &actual)
            .map_err(|e| Failure::new(e.code, format!("{} net-oracle: {}", case.id, e.detail)))?;
        rows.push(row("net-oracle", "qualified", None, actual));
    }
    for engine in [
        "cuda-generated",
        "cuda-library-assisted",
        "metal-generated",
        "metal-library-assisted",
    ] {
        rows.push(row(
            engine,
            "unavailable",
            Some("M5_DEVICE_NOT_REQUESTED"),
            Value::Null,
        ));
    }
    Ok(
        json!({"id":case.id,"class":case.class,"source_sha256":digest(&case.source),"source_bytes":case.source.len(),"mir_sha256":program.contract_digest().iter().map(|b| format!("{b:02x}")).collect::<String>(),"fuel":case.fuel,"frontend":"accepted","lowerings":["checked-ownership","mir-verified","interaction-verified-extracted","executable-net-verified-extracted","admission-dispatch"],"rows":rows}),
    )
}

/// The net oracle runs only a verified candidate; a stale proof is refused.
pub fn net_oracle(
    net: &general_net::Net,
    proof: &general_net::Proof,
    fuel: u64,
) -> Result<general_net::oracle::Observation, Failure> {
    general_net::oracle::observe(
        net,
        proof,
        general_net::oracle::Limits {
            fuel,
            ..Default::default()
        },
    )
    .map_err(|e| Failure::new("M5_NET_ORACLE", format!("{e:?}")))
}

pub fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub fn run(seed: u64) -> Result<Value, Failure> {
    let cases = corpus::corpus(seed);
    if cases.is_empty()
        || cases.len() > 256
        || cases
            .iter()
            .any(|c| c.source.len() > 65536 || c.fuel == 0 || c.fuel > 100000)
    {
        return Err(Failure::new("M5_CORPUS_LIMIT", "corpus bound exceeded"));
    }
    let mut programs = Vec::new();
    for case in &cases {
        programs.push(run_case(case)?);
    }
    let mut engines = BTreeMap::<String, Value>::new();
    let mut classes = BTreeMap::<String, usize>::new();
    for program in &programs {
        *classes
            .entry(program["class"].as_str().unwrap().into())
            .or_default() += 1;
        for row in program["rows"].as_array().unwrap() {
            let count = engines
                .entry(row["engine"].as_str().unwrap().into())
                .or_insert_with(
                    || json!({"denominator":0,"qualified":0,"unavailable":0,"blocked":0}),
                );
            let status = row["status"].as_str().unwrap();
            count["denominator"] = json!(count["denominator"].as_u64().unwrap() + 1);
            count[status] = json!(count[status].as_u64().unwrap() + 1);
        }
    }
    let corpus_bytes = cases
        .iter()
        .flat_map(|case| {
            let mut bytes = (case.source.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(&case.source);
            bytes.extend_from_slice(&case.fuel.to_le_bytes());
            bytes
        })
        .collect::<Vec<_>>();
    Ok(
        json!({"schema":"invar-m5-qualification-v1","schema_version":1,"suite":SUITE,"status":"cpu_semantics_passed_device_qualification_pending","issue_ids":[34],"seed":seed,"corpus_sha256":digest(&corpus_bytes),"program_count":programs.len(),"class_counts":classes,"engines":engines,"programs":programs,"gpu_performance_result":false,"pending":["device adapters and exact preflight","all frozen manifest families","M4-06 / M5-01 freeze / hardware performance"],"bounds":{"programs":256,"source_bytes_per_program":65536,"semantic_fuel":100000,"oracle_transitions":1000000,"deadline":"external timeout 3h"}}),
    )
}

pub fn require_gpu(report: &Value) -> Result<(), Failure> {
    if !report["device_qualification"].is_null() {
        return device::require(report);
    }
    // Future adapters replace these rows with qualified observations; a fallback
    // or mock cannot turn an unavailable device into a qualifying target.
    let programs = report["programs"]
        .as_array()
        .filter(|p| !p.is_empty())
        .ok_or_else(|| Failure::new("M5_REPORT_INVALID", "nonzero programs required"))?;
    if !programs.iter().any(|p| p["frontend"] == "accepted") {
        return Err(Failure::new(
            "M5_REPORT_INVALID",
            "no device-eligible cases",
        ));
    }
    for program in programs {
        if program["frontend"] == "refused" {
            continue;
        }
        for engine in [
            "cuda-generated",
            "cuda-library-assisted",
            "metal-generated",
            "metal-library-assisted",
        ] {
            if !program["rows"].as_array().is_some_and(|rows| {
                rows.iter()
                    .any(|r| r["engine"] == engine && r["status"] == "qualified")
            }) {
                return Err(Failure::new(
                    "M5_GPU_UNAVAILABLE",
                    format!("{} {engine}", program["id"]),
                ));
            }
        }
    }
    Ok(())
}

/// Publish only a completed successful qualification; preserve existing reports.
pub fn publish(path: &Path, report: &Value) -> Result<(), Failure> {
    use std::io::Write;
    if report["status"] != "cpu_semantics_passed_device_qualification_pending"
        && report["status"] != "bounded_device_corpus_passed"
    {
        return Err(Failure::new(
            "M5_REPORT_INVALID",
            "failed qualification cannot publish",
        ));
    }
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|e| Failure::new("M5_REPORT_INVALID", e.to_string()))?;
    if bytes.len() > 4 * 1024 * 1024 {
        return Err(Failure::new("M5_REPORT_LIMIT", "report exceeds 4 MiB"));
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let temporary = parent.join(format!(".m5-qualification-{}.partial", std::process::id()));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| Failure::new("M5_REPORT_IO", e.to_string()))?;
    let result = (|| {
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| Failure::new("M5_REPORT_IO", e.to_string()))?;
        fs::hard_link(&temporary, path).map_err(|e| Failure::new("M5_REPORT_IO", e.to_string()))
    })();
    let _ = fs::remove_file(&temporary);
    result
}

/// Every refusal carries a code, a source span when it is a source error, and
/// a span that lies inside the refused input.
pub fn refusal_is_typed(
    code: &str,
    kind: front::ErrorKind,
    span: Option<front::Span>,
    source_len: usize,
) -> Result<(), Failure> {
    if code.is_empty() || (kind == front::ErrorKind::Source && span.is_none()) {
        return Err(Failure::new(
            "M5_UNTYPED_REFUSAL",
            format!("{code:?} {kind:?} {span:?}"),
        ));
    }
    if let Some(span) = span
        && (span.start > span.end || span.end > source_len)
    {
        return Err(Failure::new(
            "M5_INVALID_DIAGNOSTIC_SPAN",
            format!("{code:?} {span:?} in {source_len} bytes"),
        ));
    }
    Ok(())
}

/// Bounded parser/checker mutations. Always preserve the seed and concrete input.
pub fn fuzz_boundary(seed: u64) -> Result<Value, Failure> {
    let template = b"fn main()->i32 { let x:i32 = 7; return x+2; }";
    let mut state = seed;
    let mut refused = BTreeMap::<String, usize>::new();
    let mut accepted = 0;
    let mut inputs = Vec::new();
    for i in 0..128 {
        let mut source = template.to_vec();
        let at = corpus::next(&mut state) as usize % source.len();
        match i % 4 {
            0 => {
                source[at] = corpus::next(&mut state) as u8;
            }
            1 => {
                source.truncate(at);
            }
            2 => {
                source.splice(at..at, [b'(', b'/', b'*']);
            }
            _ => {}
        }
        match front_check(&source) {
            Ok(_) => accepted += 1,
            Err(error) => {
                refusal_is_typed(error.code, error.kind, error.span, source.len())?;
                *refused.entry(error.code.into()).or_default() += 1;
            }
        }
        inputs.push(json!({"id":i,"bytes":source}));
    }
    Ok(
        json!({"seed":seed,"denominator":128,"accepted":accepted,"refused":refused,"input_sha256":digest(&serde_json::to_vec(&inputs).unwrap()),"max_input_bytes":256}),
    )
}

pub fn mutations() -> Result<Value, Failure> {
    mutations_with(INTERACTION_LIMITS)
}

pub fn mutations_with(limits: mir_net::Limits) -> Result<Value, Failure> {
    let source = b"fn main()->i32 { let a: Array<i32> = [1]; return a[0u32]+2; }";
    let checked = front_check(source).map_err(|e| Failure::new(e.code, e.message))?;
    let program = mir::lower(&checked).map_err(|e| Failure::new(e.code, e.message))?;
    let graph = mir_net::Net::encode(&program, limits)
        .map_err(|e| Failure::new("M5_MUTATION_SETUP", format!("{e:?}")))?;
    let proof = graph
        .verify()
        .map_err(|e| Failure::new("M5_MUTATION_SETUP", format!("{e:?}")))?;
    let mut results = Vec::new();
    for kind in [
        "schema",
        "missing-wire",
        "duplicate-wire",
        "wrong-type",
        "stale-proof",
    ] {
        let mut changed = graph.clone();
        match kind {
            "schema" => changed.schema_version += 1,
            "missing-wire" => {
                changed.wires.pop();
            }
            "duplicate-wire" => changed.wires.push(changed.wires[0].clone()),
            "wrong-type" => changed.wires[0].ty = front::Type::Bool,
            _ => changed.revision += 1,
        }
        let error = if kind == "stale-proof" {
            changed.extract(&proof).err()
        } else {
            changed.verify().err()
        };
        let error = error.ok_or_else(|| Failure::new("M5_MUTATION_SURVIVED", kind))?;
        results
            .push(json!({"mutation":kind,"status":"rejected","diagnostic":format!("{error:?}")}));
    }
    let subtract_source = b"fn main()->i32 { return 10-3; }";
    let subtract = mir::lower(&front_check(subtract_source).unwrap()).unwrap();
    let mut rewired = mir_net::Net::encode(&subtract, limits).unwrap();
    // This subtraction has two operand wires followed by the return wire.
    let [left, right, ..] = rewired.wires.as_mut_slice() else {
        return Err(Failure::new(
            "M5_MUTATION_SETUP",
            "subtraction ports missing",
        ));
    };
    std::mem::swap(&mut left.producer, &mut right.producer);
    let changed = rewired.extract(&rewired.verify().unwrap()).unwrap();
    let before = semantic(&mir_eval::observe(&subtract, 1000));
    let after = semantic(&mir_eval::observe(&changed, 1000));
    let error = agree(&before, &after)
        .err()
        .ok_or_else(|| Failure::new("M5_MUTATION_SURVIVED", "lawful-wrong-wire"))?;
    results.push(json!({"mutation":"lawful-wrong-wire","status":"detected","diagnostic":error.code,"before":before,"after":after}));
    let report = admission::analyze(source, &program)
        .map_err(|e| Failure::new(e.code, e.message))?
        .to_json();
    for key in ["schema_version", "source_identity", "regions"] {
        let mut changed = report.clone();
        changed[key] = json!("corrupt");
        let error = admission::validate_serialized(&changed, source, &program)
            .err()
            .ok_or_else(|| Failure::new("M5_MUTATION_SURVIVED", key))?;
        results.push(
            json!({"mutation":format!("cache-{key}"),"status":"rejected","diagnostic":error.code}),
        );
    }
    let mut invalid = program.clone();
    invalid.schema_version += 1;
    let error = mir::verify(&invalid)
        .err()
        .ok_or_else(|| Failure::new("M5_MUTATION_SURVIVED", "mir-schema"))?;
    results.push(json!({"mutation":"mir-schema","status":"rejected","diagnostic":error.code}));
    let reference = semantic(&eval::observe(&checked, 1000));
    for key in ["result", "trap", "steps_used", "cleanup_trace"] {
        let mut changed = reference.clone();
        changed[key] = json!("corrupt");
        let error = agree(&reference, &changed).unwrap_err();
        results.push(json!({"mutation":format!("observation-{key}"),"status":"detected","diagnostic":error.code}));
    }
    Ok(json!({"denominator":results.len(),"detected":results.len(),"survived":0,"rows":results}))
}
