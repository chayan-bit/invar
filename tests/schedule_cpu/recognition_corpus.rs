//! Generated recognition corpus for the M2-05 CPU scheduler fallback.
use super::issue_m2_05::lower;
use crate::corpus_sample;
use invar::{eval, schedule};

/// Where a generated program's array comes from.
#[derive(Clone, Copy)]
enum Input {
    Literal,
    Filled,
    Parameter,
}

/// Wrap `body` (which reads array `xs`) in a program whose `xs` comes from `input`.
fn corpus_program(input: Input, len: usize, body: &str, helpers: &str) -> String {
    let literal = (0..len)
        .map(|i| format!("{}u32", (i as u32).wrapping_mul(2_654_435_761) ^ 0x5bd1))
        .collect::<Vec<_>>()
        .join(",");
    match input {
        Input::Literal => format!(
            "{helpers} fn main() -> i32 {{ let pad: u32 = 3u32; let xs: Array<u32> = [{literal}]; {body} }}"
        ),
        Input::Filled => format!(
            "{helpers} fn main() -> i32 {{ let xs: Array<u32> = filled_u32({len}u32, 4294967293u32); let pad: u32 = 3u32; {body} }}"
        ),
        Input::Parameter => format!(
            "{helpers} fn work(xs: read Array<u32>) -> i32 {{ let pad: u32 = 3u32; {body} }} \
             fn main() -> i32 {{ let values: Array<u32> = [{literal}]; return work(values); }}"
        ),
    }
}

const CHECKSUM: &str = "var h: u32 = 1u32; for k in 0u32 .. len(OUT) { h = wrapping_add(wrapping_mul(h, 31u32), OUT[k]); } var tail: u32 = pad; tail = tail + 1u32; if h == tail { return 1; } return 0;";

fn checksum(out: &str) -> String {
    CHECKSUM.replace("OUT", out)
}

/// Positive families: every variant must plan as expected and agree with the reference.
fn corpus_positives(variant: usize) -> Vec<(String, Vec<schedule::PlanKind>)> {
    use schedule::PlanKind::{FusedMap, Map, Reduction, Scan};
    let c = [7u32, 4_294_967_295, 65_537][variant % 3];
    let (ys, i, s) = if variant.is_multiple_of(2) {
        ("ys", "i", "s")
    } else {
        ("outcome", "row", "carry")
    };
    let map_ops = [
        format!("wrapping_add(xs[{i}], {c}u32)"),
        format!("wrapping_add({c}u32, xs[{i}])"),
        format!("wrapping_mul(xs[{i}], {c}u32)"),
        format!("wrapping_mul({c}u32, xs[{i}])"),
    ];
    let map = format!(
        "var {ys}: Array<u32> = filled_u32(len(xs), 0u32); for {i} in 0u32 .. len(xs) {{ {ys}[{i}] = {}; }} {}",
        map_ops[variant % 4],
        checksum(ys)
    );
    let consumer = [
        "wrapping_mul(mid[j], 5u32)",
        "wrapping_add(mid[j], xs[j])",
        "wrapping_mul(xs[j], mid[j])",
        "wrapping_add(9u32, mid[j])",
    ][variant % 4];
    let fused = format!(
        "var first: Array<u32> = filled_u32(len(xs), 0u32); for {i} in 0u32 .. len(xs) {{ first[{i}] = {}; }} \
         let mid: Array<u32> = move first; var second: Array<u32> = filled_u32(len(mid), 0u32); \
         for j in 0u32 .. len(mid) {{ second[j] = {consumer}; }} {}",
        map_ops[(variant + 1) % 4],
        checksum("second")
    );
    let reduction_op = [
        format!("wrapping_add({s}, xs[{i}])"),
        format!("wrapping_add(xs[{i}], {s})"),
        format!("wrapping_mul({s}, xs[{i}])"),
        format!("wrapping_mul(xs[{i}], {s})"),
    ][variant % 4]
        .clone();
    let reduction = format!(
        "var {s}: u32 = {c}u32; for {i} in 0u32 .. len(xs) {{ {s} = {reduction_op}; }} if {s} == pad {{ return 1; }} return 0;"
    );
    let update = [
        format!("{s} = wrapping_add(wrapping_mul({c}u32, {s}), xs[{i}]);"),
        format!("{s} = wrapping_add(xs[{i}], wrapping_mul({s}, {c}u32));"),
        format!("{s} = wrapping_add(wrapping_mul({s}, xs[{i}]), {c}u32);"),
        format!("{s} = wrapping_add({c}u32, wrapping_mul(xs[{i}], {s}));"),
    ][variant % 4]
        .clone();
    let publish = format!("{ys}[{i}] = {s};");
    let body = if variant.is_multiple_of(2) {
        format!("{update} {publish}")
    } else {
        format!("{publish} {update}")
    };
    let scan = format!(
        "var {s}: u32 = pad; var {ys}: Array<u32> = filled_u32(len(xs), 0u32); for {i} in 0u32 .. len(xs) {{ {body} }} {}",
        checksum(ys)
    );
    vec![
        (map, vec![Map, Scan]),
        (fused, vec![FusedMap, FusedMap, Scan]),
        (reduction, vec![Reduction]),
        (scan, vec![Scan, Scan]),
    ]
}

/// Near-matches stay serial with a stable first rejected obligation.
fn corpus_negatives() -> Vec<(String, &'static str, &'static str)> {
    let map = |element: &str| {
        format!(
            "var ys: Array<u32> = filled_u32(len(xs), 0u32); for i in 0u32 .. len(xs) {{ ys[i] = {element}; }} return 0;"
        )
    };
    let callback = "fn step(x: u32) -> u32 { return x + 4294967295u32; }";
    vec![
        (map("xs[i] + 1u32"), "", "TOTALITY"),
        (map("wrapping_add(ys[i - 1u32], 1u32)"), "", "DEPENDENCE"),
        (map("wrapping_add(ys[i], 1u32)"), "", "ALIAS"),
        (map("step(xs[i])"), callback, "UNKNOWN_EFFECT"),
        (
            "var s: u32 = 3u32; for i in 0u32 .. len(xs) { s = wrapping_add(wrapping_mul(s, s), xs[i]); } return 0;".into(),
            "",
            "STATE_DEPENDENCE",
        ),
        (
            "var s: u32 = 3u32; for i in 0u32 .. len(xs) { s = s + xs[i]; } return 0;".into(),
            "",
            "NUMERIC_POLICY",
        ),
    ]
}

fn assert_corpus_case(source: &str, expected: &[schedule::PlanKind], reason: Option<&str>) {
    // Opt-in: save the corpus so it can be replayed through the real CLI engines.
    if let Some(dir) = std::env::var_os("INVAR_CORPUS_DIR") {
        let name = format!("{:016x}.inv", fnv1a(source.as_bytes()));
        std::fs::write(std::path::Path::new(&dir).join(name), source).unwrap();
    }
    let (checked, program) = lower(source);
    let report = invar::admission::analyze(source.as_bytes(), &program).unwrap();
    let kinds = report
        .regions()
        .iter()
        .map(|region| region.plan_kind())
        .collect::<Vec<_>>();
    assert_eq!(kinds, expected, "{source}");
    if let Some(reason) = reason {
        assert_eq!(
            report.regions()[0].first_rejected_obligation(),
            Some(reason),
            "{source}"
        );
    }
    let reference = eval::observe(&checked, 1_000_000);
    for workers in 1..=schedule::MAX_WORKERS {
        let run = schedule::run_source(source.as_bytes(), &program, 1_000_000, workers).unwrap();
        assert_eq!(
            run.observation.result, reference.result,
            "{workers} {source}"
        );
        assert_eq!(run.observation.steps_used, reference.steps_used, "{source}");
        assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
    }
}

/// Generated variants guard against recognizers keyed on names, operand order,
/// constants, surrounding statements, input provenance or sizes.
/// Pull-request CI runs the deterministic sample of `corpus_sample`, which keeps every
/// family; `INVAR_FULL_CORPUS=1` runs every variant.
#[test]
fn issue_m2_05_cpu_scheduler_fallback_recognition_corpus() {
    const FAMILIES: [&str; 4] = ["map", "fused", "reduction", "scan"];
    let mut cases = Vec::new();
    for len in [0usize, 1, 63, 64, 65, 257] {
        for (index, input) in [Input::Literal, Input::Filled, Input::Parameter]
            .into_iter()
            .enumerate()
        {
            for variant in [index + len, index + len + 1] {
                for (family, (body, expected)) in corpus_positives(variant).into_iter().enumerate()
                {
                    let id = format!("m2/{}/{index}-{len}-{variant}", FAMILIES[family]);
                    let source = corpus_program(input, len, &body, "");
                    cases.push((id, source, expected, None));
                }
            }
            for (body, helpers, reason) in corpus_negatives() {
                let id = format!("m2/{reason}/{index}-{len}");
                let source = corpus_program(input, len, &body, helpers);
                cases.push((id, source, vec![schedule::PlanKind::Serial], Some(reason)));
            }
        }
    }
    assert_eq!(cases.len(), 6 * 3 * (2 * 4 + 6));
    let kept = corpus_sample::paths(
        cases
            .iter()
            .map(|case| (case.0.clone(), corpus_sample::category(&case.0))),
    );
    for (id, source, expected, reason) in cases {
        if kept.contains(&id) {
            assert_corpus_case(&source, &expected, reason);
        }
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}
