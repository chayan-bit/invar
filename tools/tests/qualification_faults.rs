//! Fault and boundary controls for the M5-02 qualification harness (`qualification/`). The harness is
//! compiled into this test to inject its faults; it expects these names at the crate root. The
//! diagnostic registry (`tests/issue_ai_01_machine_diagnostics`) cites these tests by name.
#![forbid(unsafe_code)]
use invar::front::Span;
#[allow(unused_imports)]
use invar::{check, front, mir, schedule};
pub use invar::{general_net, hex, layout_contract, numeric_contract, tensor_cache};
use serde_json::json;

#[path = "../qualification/mod.rs"]
#[allow(dead_code)]
mod qualification;

fn qualification_case(
    source: &[u8],
    fuel: u64,
    expected: qualification::corpus::Expected,
) -> qualification::corpus::Case {
    qualification::corpus::Case {
        id: "fixture".to_owned(),
        class: "fixture",
        source: source.to_vec(),
        fuel,
        expected,
    }
}

fn qualification_net(source: &[u8]) -> general_net::Net {
    let checked = check::check(front::parse(source).unwrap(), true).unwrap();
    general_net::Net::lower(&checked, Default::default()).unwrap()
}

#[test]
fn issue_ai_01_machine_diagnostics_qualification_pipeline_stage_faults() {
    use qualification::{Bounds, corpus::Expected};
    let source = b"fn main()->i32 { return 7; }";
    let case = qualification_case(source, 1000, Expected::Value(7));
    qualification::run_case(&case).unwrap();
    // A stage bound tighter than the program fails that stage, never the row.
    let interaction = Bounds {
        interaction: invar::mir_net::Limits {
            functions: 0,
            blocks: 0,
            instructions: 0,
            wires: 0,
        },
        ..Bounds::default()
    };
    let failure = qualification::run_case_with(&case, &interaction).unwrap_err();
    assert_eq!(failure.code, "M5_INTERACTION");
    let net = Bounds {
        net: general_net::Limits {
            blocks: 0,
            ..Default::default()
        },
        ..Bounds::default()
    };
    assert_eq!(
        qualification::run_case_with(&case, &net).unwrap_err().code,
        "M5_NET"
    );
    // The oracle refuses a proof that belongs to a different net.
    let proof = qualification_net(source).verify().unwrap();
    let other = qualification_net(b"fn main()->i32 { return 9; }");
    assert_eq!(
        qualification::net_oracle(&other, &proof, 1000)
            .unwrap_err()
            .code,
        "M5_NET_ORACLE"
    );
    // A refusal the corpus did not expect, and an observation two engines
    // disagree on, are qualification failures.
    let refused = qualification_case(b"fn main()->i32 { return ; }", 1000, Expected::Value(7));
    assert_eq!(
        qualification::run_case(&refused).unwrap_err().code,
        "M5_UNEXPECTED_REFUSAL"
    );
    let disagreement = qualification::agree(&json!({"result": 7}), &json!({"result": 9}));
    assert_eq!(disagreement.unwrap_err().code, "M5_SEMANTIC_DISAGREEMENT");
}

#[test]
fn issue_ai_01_machine_diagnostics_qualification_refusal_shape_faults() {
    use front::ErrorKind::{Source, Tool};
    let span = |start, end| Some(Span { start, end });
    for (code, kind, at) in [("", Tool, None), ("X", Source, None)] {
        let failure = qualification::refusal_is_typed(code, kind, at, 8).unwrap_err();
        assert_eq!(failure.code, "M5_UNTYPED_REFUSAL");
    }
    for at in [span(5, 3), span(0, 9)] {
        let failure = qualification::refusal_is_typed("X", Source, at, 8).unwrap_err();
        assert_eq!(failure.code, "M5_INVALID_DIAGNOSTIC_SPAN");
    }
    // The boundaries themselves are lawful: a tool error needs no span, and a
    // span may end exactly at the end of the input.
    qualification::refusal_is_typed("X", Tool, None, 8).unwrap();
    qualification::refusal_is_typed("X", Source, span(8, 8), 8).unwrap();
}

#[test]
fn issue_ai_01_machine_diagnostics_qualification_mutation_setup_fault() {
    qualification::mutations().unwrap();
    let none = invar::mir_net::Limits {
        functions: 0,
        blocks: 0,
        instructions: 0,
        wires: 0,
    };
    assert_eq!(
        qualification::mutations_with(none).unwrap_err().code,
        "M5_MUTATION_SETUP"
    );
}

#[cfg(feature = "native")]
#[test]
fn issue_ai_01_machine_diagnostics_qualification_native_fault() {
    use qualification::corpus::Expected;
    // Every engine before the native one accepts a zero fuel budget as the
    // typed INVALID_FUEL tool error; the native host alone refuses to run.
    let case = qualification_case(
        b"fn main()->i32 { return 7; }",
        0,
        Expected::Tool("INVALID_FUEL"),
    );
    assert_eq!(
        qualification::run_case(&case).unwrap_err().code,
        "M5_NATIVE"
    );
}

#[test]
fn issue_ai_01_machine_diagnostics_qualification_report_boundaries() {
    let directory = std::env::temp_dir().join(format!("invar-ai01-report-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let status = "cpu_semantics_passed_device_qualification_pending";
    let report = json!({"status": status});
    let oversized = json!({"status": status, "padding": "x".repeat(4 * 1024 * 1024)});
    let path = directory.join("report.json");
    let failure = qualification::publish(&path, &oversized).unwrap_err();
    assert_eq!(failure.code, "M5_REPORT_LIMIT");
    let missing = directory.join("absent").join("report.json");
    let failure = qualification::publish(&missing, &report).unwrap_err();
    assert_eq!(failure.code, "M5_REPORT_IO");
    // An existing report is never replaced, and no partial file remains.
    qualification::publish(&path, &report).unwrap();
    let before = std::fs::read(&path).unwrap();
    let failure = qualification::publish(&path, &report).unwrap_err();
    assert_eq!(failure.code, "M5_REPORT_IO");
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn issue_ai_01_machine_diagnostics_qualification_device_rows_not_requested() {
    use qualification::corpus::Expected;
    let case = qualification_case(b"fn main()->i32 { return 7; }", 1000, Expected::Value(7));
    let program = qualification::run_case(&case).unwrap();
    let rows = program["rows"].as_array().unwrap();
    let device = rows
        .iter()
        .filter(|row| {
            row["status"] == "unavailable" && row["diagnostic"] == "M5_DEVICE_NOT_REQUESTED"
        })
        .count();
    assert_eq!(device, 4);
}

#[test]
fn issue_ai_01_machine_diagnostics_qualification_native_rows_follow_the_feature() {
    use qualification::corpus::Expected;
    let case = qualification_case(b"fn main()->i32 { return 7; }", 1000, Expected::Value(7));
    let program = qualification::run_case(&case).unwrap();
    for engine in ["native-serial", "native-net"] {
        let row = program["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["engine"] == engine)
            .unwrap();
        if cfg!(feature = "native") {
            assert_eq!(row["status"], "qualified", "{engine}");
        } else {
            assert_eq!(row["status"], "unavailable", "{engine}");
            assert_eq!(row["diagnostic"], "NATIVE_FEATURE_REQUIRED", "{engine}");
        }
    }
}
