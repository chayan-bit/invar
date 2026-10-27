//! Shared by the GNET-04 CPU (ABI4) and device-group model (ABI5) speculation
//! tests: the source net, limits, driver invocation and the oracle comparison
//! of the 9-word header, event trace and cleanup trace.
use invar::general_net::{self, Net, compiled, oracle};
use invar::{check, eval, front};
use std::process::Command;

pub fn lowered(source: &str) -> Net {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    Net::lower(&checked, general_net::Limits::default()).unwrap()
}
pub fn storage() -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: 64,
        allocations: 128,
        cleanup_events: 128,
    }
}
pub fn spec() -> compiled::SpeculationLimits {
    compiled::SpeculationLimits {
        quantum: 64,
        ..compiled::SpeculationLimits::default()
    }
}
pub fn limits() -> oracle::Limits {
    oracle::Limits {
        fuel: 10000,
        transitions: 100000,
        trace_events: 10000,
        ..oracle::Limits::default()
    }
}

/// Run a compiled driver (`fuel trans cap quantum capacity args...`) and parse its stdout words.
pub fn run(
    mut command: Command,
    mode: &str,
    args: &[eval::Value],
    limits: oracle::Limits,
    quantum: u64,
    capacity: usize,
) -> Vec<i64> {
    let words: Vec<_> = args
        .iter()
        .map(|x| match x {
            eval::Value::I32(n) => n.to_string(),
            eval::Value::U32(n) => n.to_string(),
            eval::Value::Bool(n) => usize::from(*n).to_string(),
            _ => panic!("scalar input"),
        })
        .collect();
    let output = command
        .env("GNET_TEST_MODE", mode)
        .args([
            limits.fuel.to_string(),
            limits.transitions.to_string(),
            limits.trace_events.to_string(),
            quantum.to_string(),
            capacity.to_string(),
        ])
        .args(words)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect()
}

/// Assert the header, events and cleanup trace in `words` equal the oracle;
/// returns the offset just past the cleanup trace.
pub fn compare(
    net: &Net,
    entry: &str,
    args: &[eval::Value],
    limits: oracle::Limits,
    words: &[i64],
) -> usize {
    let reference = oracle::observe_call(net, &net.verify().unwrap(), entry, args, limits).unwrap();
    let (status, result, start, end) = match &reference.semantic.result {
        Ok(eval::Value::I32(n)) => (0, i64::from(*n), 0, 0),
        Ok(eval::Value::U32(n)) => (0, i64::from(*n), 0, 0),
        Ok(eval::Value::Bool(n)) => (0, i64::from(*n), 0, 0),
        Ok(eval::Value::Unit) => (0, 0, 0, 0),
        Err(eval::EvalError::Trap { category, span }) => (
            match category {
                eval::TrapCategory::IntegerOverflow => 1,
                eval::TrapCategory::DivisionByZero => 2,
                eval::TrapCategory::IndexOutOfBounds => 9,
                _ => panic!("trap"),
            },
            0,
            span.start as i64,
            span.end as i64,
        ),
        Err(eval::EvalError::EvaluationLimit { .. }) => (3, 0, 0, 0),
        Err(eval::EvalError::Tool {
            code: "CALL_DEPTH_LIMIT",
            ..
        }) => (4, 0, 0, 0),
        Err(eval::EvalError::Tool {
            code: "ARRAY_SIZE_LIMIT",
            ..
        }) => (10, 0, 0, 0),
        Err(_) => (
            match reference.failure {
                Some(oracle::ToolFailure::TransitionLimit) => 5,
                Some(oracle::ToolFailure::TraceLimit) => 6,
                Some(oracle::ToolFailure::Allocation) => 11,
                _ => panic!("{reference:?}"),
            },
            0,
            0,
            0,
        ),
        x => panic!("{x:?}"),
    };
    assert_eq!(
        &words[..9],
        &[
            status,
            result,
            start,
            end,
            reference.semantic.steps_used as i64,
            reference.transitions as i64,
            reference.max_frames as i64,
            reference.events.len() as i64,
            reference.semantic.cleanup_trace.len() as i64
        ],
        "{entry} {args:?} {limits:?}"
    );
    let mut offset = 21;
    for event in &reference.events {
        assert_eq!(
            &words[offset..offset + 7],
            &[
                i64::from(event.kind == oracle::EventKind::Complete),
                event.frame as i64,
                event.activation as i64,
                event.token.block.function as i64,
                event.token.block.block as i64,
                event.token.position as i64,
                event.rule as i64
            ],
            "event at {offset} for {entry} {args:?}"
        );
        offset += 7;
    }
    let mir = net.compiled_mir(&net.verify().unwrap()).unwrap();
    for event in &reference.semantic.cleanup_trace {
        assert_eq!(
            mir.functions[words[offset] as usize].declaration_order[words[offset + 1] as usize],
            event.name
        );
        assert_eq!(words[offset + 2], event.length as i64);
        offset += 4;
    }
    offset
}
