//! ABI12 follow-up P1: a tight net-limit precedence fixture. With the net
//! function limit one below a source's exact function count, every target and
//! owned-numeric route reports `Net(Limit)`, before any target guard or
//! emission; at the exact count the ordinary outcome returns. Size limits and
//! source errors still precede the net limit. Outcomes equal a golden from
//! pre-ABI12 `259d2fa`, except the named valid CUDA continuations admitted as
//! ABI12 compile receipts in stage 2b. No receipt authority is involved.
use invar::general_net::{compiled, provenance};
use serde_json::{Value, json};

const VALID_OWNED: &str = "fn run(a: f32, n: u32) -> f32 { var xs: Array<f32> = filled_f32(n + 1u32, a); xs[0u32] = xs[0u32] * a + a; return xs[0u32]; } fn main() -> i32 { return 0; }";
const VALID_SCALAR: &str = "fn run(n: u32) -> i32 { if n < 2u32 { return 1; } return run(n - 1u32) + run(n - 2u32); } fn main() -> i32 { return run(5u32); }";
const AGGREGATE: &str = "fn make(ok: Bool) -> Result<i32,u32> { if ok { return Result::Ok(7); } return Result::Err(9u32); } fn run(ok: Bool) -> i32 { let x = make(ok); return match move x { Result::Ok(a) => a, Result::Err(e) => 9 }; } fn main() -> i32 { return run(true); }";

/// (case, source, exact function count written in the source text).
const SOURCES: [(&str, &str, usize); 3] = [
    ("valid_owned_numeric", VALID_OWNED, 2),
    ("valid_scalar", VALID_SCALAR, 2),
    ("valid_aggregate", AGGREGATE, 3),
];

const TARGETS: [compiled::Target; 3] = [
    compiled::Target::Cpu,
    compiled::Target::Cuda,
    compiled::Target::Metal,
];

fn limits(functions: usize) -> provenance::Limits {
    let mut l = provenance::Limits::default();
    l.net.functions = functions;
    l
}

fn storage() -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: 64,
        allocations: 16,
        cleanup_events: 32,
    }
}

fn record(result: Result<provenance::ProvenancedArtifact, provenance::Error>) -> Value {
    match result {
        Ok(a) => json!({
            "abi": a.artifact().abi_version(),
            "target": format!("{:?}", a.artifact().target()),
        }),
        Err(e) => json!({ "error": format!("{e:?}") }),
    }
}

fn both_routes(
    map: &mut serde_json::Map<String, Value>,
    case: &str,
    source: &[u8],
    entry: &str,
    functions: usize,
) {
    for target in TARGETS {
        let l = limits(functions);
        map.insert(
            format!("{case}/functions={functions}/compile_resumable_owned_numeric/{target:?}"),
            record(provenance::compile_resumable_owned_numeric(
                source,
                entry,
                target,
                l,
                storage(),
            )),
        );
        map.insert(
            format!("{case}/functions={functions}/compile_owned_numeric/{target:?}"),
            record(provenance::compile_owned_numeric(
                source,
                entry,
                target,
                l,
                storage(),
            )),
        );
    }
}

fn outcomes() -> serde_json::Map<String, Value> {
    let mut map = serde_json::Map::new();
    for (case, source, count) in SOURCES {
        both_routes(&mut map, case, source.as_bytes(), "run", count - 1);
        both_routes(&mut map, case, source.as_bytes(), "run", count);
        // A missing entry is a compiler error: the net limit still precedes it.
        both_routes(
            &mut map,
            &format!("{case}_missing_entry"),
            source.as_bytes(),
            "nope",
            count - 1,
        );
    }
    // Size limits and source errors precede even a zero net function limit.
    both_routes(
        &mut map,
        "oversized_source",
        &vec![b' '; 1024 * 1024 + 1],
        "run",
        0,
    );
    both_routes(&mut map, "empty_entry", VALID_OWNED.as_bytes(), "", 0);
    both_routes(&mut map, "unparsable", b"fn (", "run", 0);
    both_routes(
        &mut map,
        "type_error",
        b"fn run() -> i32 { return true; } fn main() -> i32 { return 0; }",
        "run",
        0,
    );
    map
}

const GOLDEN: &str = include_str!("fixtures/gnet12/pre-abi12-net-limit-precedence-golden.json");

/// Stage-2b named outcome updates: valid sources at their exact function
/// count whose CUDA continuation is now an ABI12 compile receipt.
const ADMITTED_CUDA: [&str; 2] = [
    "valid_owned_numeric/functions=2/compile_resumable_owned_numeric/Cuda",
    "valid_scalar/functions=2/compile_resumable_owned_numeric/Cuda",
];

#[test]
fn issue_gnet_12_net_limit_precedes_target_guard_and_equals_pre_abi12() {
    for (_, source, count) in SOURCES {
        assert_eq!(source.matches("fn ").count(), count);
    }
    let current = outcomes();
    if let Some(path) = std::env::var_os("INVAR_GNET12_WRITE_NET_PRECEDENCE") {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&Value::Object(current)).unwrap() + "\n",
        )
        .unwrap();
        return;
    }
    let golden: serde_json::Map<String, Value> = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(
        golden.keys().collect::<Vec<_>>(),
        current.keys().collect::<Vec<_>>()
    );
    for (key, want) in &golden {
        if ADMITTED_CUDA.contains(&key.as_str()) {
            assert_eq!(want["error"], "Compiler(NumericTarget)", "{key}");
            assert_eq!(golden[&key.replace("/Cuda", "/Cpu")]["abi"], 11, "{key}");
            assert_eq!(current[key]["abi"], 12, "{key}");
            assert_eq!(current[key]["target"], "Cuda", "{key}");
            continue;
        }
        assert_eq!(&current[key], want, "{key} differs from pre-ABI12");
    }
    // Independent of the golden: one below the exact count, every target and
    // route (including the Metal and CUDA guards) reports the net limit.
    let net = json!({"error": "Net(Limit)"});
    let mut tight = 0;
    for (key, value) in &current {
        let below = SOURCES
            .iter()
            .any(|(case, _, count)| key.starts_with(&format!("{case}/functions={}/", count - 1)))
            || key.contains("_missing_entry/");
        if below {
            assert_eq!(value, &net, "{key}");
            tight += 1;
        }
    }
    assert_eq!(tight, 3 * 2 * 3 * 2);
    for (prefix, error) in [
        ("oversized_source/", "Limit"),
        ("empty_entry/", "Limit"),
        ("unparsable/", "Source("),
        ("type_error/", "Source("),
    ] {
        for (key, value) in current.iter().filter(|(k, _)| k.starts_with(prefix)) {
            assert!(
                value["error"]
                    .as_str()
                    .is_some_and(|e| e.starts_with(error)),
                "{key}: {value}"
            );
        }
    }
    eprintln!(
        "net-limit precedence: {} outcomes equal to pre-ABI12 ({tight} tight Net(Limit))",
        golden.len()
    );
}
