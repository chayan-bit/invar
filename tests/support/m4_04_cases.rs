//! Shared M4-04 device cases: ordinary-source functions, adversarial inputs
//! (no segments, all-empty, empty interior, uniform, skewed, one dominant,
//! ragged, block boundaries, 0/1/50/99/100% selectivity, maximum sizes) and
//! the source-derived expectation each device output must equal bit for bit.
//!
//! The expectation is computed from a source `probe()` that builds the same
//! inputs and calls the function: by the reference evaluator, and, for every
//! case small enough for its event trace, also by the bounded CPU net oracle
//! (which must agree exactly). Large cases use the reference evaluator only.

use invar::{
    check, compaction_source,
    eval::{self, Value},
    front,
    general_net::{self, oracle},
    segmented_device, segmented_source,
};
use invar_device::{
    Capabilities, Provenance,
    segmented::{self as seg, Ir},
};

pub const A: u32 = 2_654_435_761;
pub const B: u32 = 40_503;
/// The oracle stores every logical event; beyond this the reference evaluator
/// alone (which GNET-02 proves agrees with the oracle) sets the expectation.
const ORACLE_ELEMENTS: usize = 20_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Map,
    Sum,
    Product,
    Compact(&'static str),
}

pub struct Case {
    pub name: String,
    pub kind: Kind,
    pub input: Vec<u32>,
    /// Map/reduce offsets; empty for compaction.
    pub offsets: Vec<u32>,
    pub cutoff: u32,
    /// Offsets are `t * stride` for `t < S` and `N` at `S` (large cases).
    pub stride: Option<u32>,
}

impl Case {
    pub fn large(&self) -> bool {
        self.input.len() > ORACLE_ELEMENTS
    }
}

pub fn word(i: u32) -> u32 {
    i.wrapping_mul(A).wrapping_add(B)
}

pub fn function(kind: Kind) -> String {
    match kind {
        Kind::Map => "fn run(xs: read Array<u32>, offsets: read Array<u32>, count: u32) -> Array<u32> { var out = filled_u32(len(xs), 0u32); for s in 0u32..count { for i in offsets[s]..offsets[s + 1u32] { out[i] = wrapping_add(wrapping_mul(wrapping_add(xs[i], 7u32), 2654435761u32), s); } } return move out; }".to_owned(),
        Kind::Sum | Kind::Product => format!(
            "fn run(xs: read Array<u32>, offsets: read Array<u32>, count: u32) -> Array<u32> {{ var out = filled_u32(count, 0u32); for s in 0u32..count {{ var acc = {}u32; for i in offsets[s]..offsets[s + 1u32] {{ acc = {}(acc, xs[i]); }} out[s] = acc; }} return move out; }}",
            u32::from(kind == Kind::Product),
            if kind == Kind::Product { "wrapping_mul" } else { "wrapping_add" }
        ),
        Kind::Compact(op) => format!(
            "fn run(xs: read Array<u32>, count: u32, cutoff: u32) -> Array<u32> {{ var selected = 0u32; for i in 0u32..count {{ if xs[i] {op} cutoff {{ selected = selected + 1u32; }} }} var out = filled_u32(selected, 0u32); var pos = 0u32; for j in 0u32..count {{ if xs[j] {op} cutoff {{ out[pos] = xs[j]; pos = pos + 1u32; }} }} return move out; }}"
        ),
    }
}

fn segmented(name: &str, kind: Kind, lengths: &[u32]) -> Case {
    let mut offsets = vec![0u32];
    for length in lengths {
        offsets.push(offsets.last().unwrap() + length);
    }
    let n = *offsets.last().unwrap();
    Case {
        name: format!("{kind:?}/{name}"),
        kind,
        input: (0..n).map(word).collect(),
        offsets,
        cutoff: 0,
        stride: None,
    }
}

fn uniform_large(name: &str, kind: Kind, n: u32, segments: u32, stride: u32) -> Case {
    let mut offsets: Vec<u32> = (0..segments).map(|t| t * stride).collect();
    offsets.push(n);
    Case {
        name: format!("{kind:?}/{name}"),
        kind,
        input: (0..n).map(word).collect(),
        offsets,
        cutoff: 0,
        stride: Some(stride),
    }
}

fn compaction(name: &str, op: &'static str, n: u32, cutoff: u32) -> Case {
    Case {
        name: format!("Compact({op})/{name}"),
        kind: Kind::Compact(op),
        input: (0..n).map(word).collect(),
        offsets: Vec::new(),
        cutoff,
        stride: None,
    }
}

/// Deterministic segment lengths; no RNG dependency.
fn lengths(seed: u32, count: usize, max: u32) -> Vec<u32> {
    (0..count as u32)
        .map(|i| word(i ^ seed) % (max + 1))
        .collect()
}

pub fn cases(include_large: bool) -> Vec<Case> {
    let mut all = Vec::new();
    for kind in [Kind::Map, Kind::Sum, Kind::Product] {
        let mut skewed: Vec<u32> = (0..11).map(|e| 1 << e).collect();
        skewed.extend([0, 0, 3]);
        for (name, shape) in [
            ("no-segments", Vec::new()),
            ("all-empty", vec![0; 6]),
            ("empty-interior", vec![3, 0, 0, 5, 0, 1]),
            ("uniform", vec![37; 64]),
            ("skewed", skewed),
            ("one-dominant", vec![1, 5000, 1, 0, 2]),
            ("ragged", lengths(0x5eed, 200, 40)),
            ("block-boundaries", vec![255, 256, 257, 0, 513, 1, 511]),
        ] {
            all.push(segmented(name, kind, &shape));
        }
    }
    let ops = ["<", "<=", "==", "!=", ">=", ">"];
    for (index, op) in ops.into_iter().enumerate() {
        for n in [0, 1, 255, 256, 257] {
            all.push(compaction(&format!("n{n}"), op, n, word(index as u32)));
        }
    }
    for (name, op, cutoff) in [
        ("select-0pct", "<", 0),
        ("select-1pct", "<", u32::MAX / 100),
        ("select-50pct", "<", u32::MAX / 2),
        ("select-99pct", "<", u32::MAX / 100 * 99),
        ("select-100pct", ">=", 0),
    ] {
        all.push(compaction(name, op, 10_000, cutoff));
    }
    if include_large {
        // Maximum sizes under the source limits: 1,000,000 elements per array
        // and 2,000,000 live elements (inputs, offsets and output together).
        all.push(uniform_large("max-elements", Kind::Map, 999_000, 999, 1000));
        all.push(uniform_large(
            "max-segments",
            Kind::Sum,
            1_000_000,
            499_999,
            2,
        ));
        all.push(uniform_large(
            "max-segments",
            Kind::Product,
            1_000_000,
            499_999,
            2,
        ));
        all.push(compaction("max-100pct", ">=", 1_000_000, 0));
        all.push(compaction("max-50pct", "<", 1_000_000, u32::MAX / 2));
    }
    all
}

fn literals(values: &[u32]) -> String {
    values
        .iter()
        .map(|v| format!("{v}u32"))
        .collect::<Vec<_>>()
        .join(",")
}

/// The function plus a `probe()` that builds this case's inputs and calls it.
pub fn probe_source(case: &Case) -> String {
    let n = case.input.len();
    let build = format!(
        "var xs: Array<u32> = filled_u32({n}u32, 0u32); for g in 0u32..{n}u32 {{ xs[g] = wrapping_add(wrapping_mul(g, {A}u32), {B}u32); }}"
    );
    let call = match (case.kind, case.stride) {
        (Kind::Compact(_), _) => format!("return run(xs, {n}u32, {}u32);", case.cutoff),
        (_, None) => format!(
            "let offsets: Array<u32> = [{}]; return run(xs, offsets, {}u32);",
            literals(&case.offsets),
            case.offsets.len() - 1
        ),
        (_, Some(stride)) => {
            let segments = case.offsets.len() - 1;
            format!(
                "var offsets: Array<u32> = filled_u32({}u32, {n}u32); for t in 0u32..{segments}u32 {{ offsets[t] = t * {stride}u32; }} return run(xs, offsets, {segments}u32);",
                segments + 1
            )
        }
    };
    format!(
        "{} fn probe() -> Array<u32> {{ {build} {call} }}",
        function(case.kind)
    )
}

fn array(result: Result<Value, eval::EvalError>, context: &str) -> Vec<u32> {
    match result {
        Ok(Value::ArrayU32(values)) => values.borrow().clone(),
        other => panic!("{context}: {other:?}"),
    }
}

/// The source result for this case, from the reference evaluator and (below
/// the trace bound) the CPU net oracle, which must agree bit for bit.
pub fn expected(case: &Case) -> Vec<u32> {
    let source = probe_source(case);
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
    let fuel = 200_000_000;
    let reference = array(
        eval::named::observe_owned_numeric_named(&checked, "probe", &[], fuel).result,
        &case.name,
    );
    if !case.large() {
        let net = general_net::Net::lower(&checked, general_net::Limits::default()).unwrap();
        let observed = oracle::observe_call(
            &net,
            &net.verify().unwrap(),
            "probe",
            &[],
            oracle::Limits {
                fuel,
                transitions: 100_000_000,
                trace_events: 20_000_000,
                array_elements: 2_000_000,
            },
        )
        .unwrap();
        assert_eq!(observed.failure, None, "{}", case.name);
        assert_eq!(
            array(observed.semantic.result, &case.name),
            reference,
            "{}",
            case.name
        );
    }
    reference
}

/// Independent sequential family laws for sizes beyond evaluator limits.
/// No device IR operations, emitted code, or reference evaluator are used.
pub fn sequential_expected(case: &Case) -> Vec<u32> {
    match case.kind {
        Kind::Map => case
            .offsets
            .windows(2)
            .enumerate()
            .flat_map(|(segment, pair)| {
                case.input[pair[0] as usize..pair[1] as usize]
                    .iter()
                    .map(move |v| {
                        v.wrapping_add(7)
                            .wrapping_mul(A)
                            .wrapping_add(segment as u32)
                    })
            })
            .collect(),
        Kind::Sum | Kind::Product => case
            .offsets
            .windows(2)
            .map(|pair| {
                case.input[pair[0] as usize..pair[1] as usize].iter().fold(
                    u32::from(case.kind == Kind::Product),
                    |acc, &v| match case.kind {
                        Kind::Sum => acc.wrapping_add(v),
                        _ => acc.wrapping_mul(v),
                    },
                )
            })
            .collect(),
        Kind::Compact(op) => case
            .input
            .iter()
            .copied()
            .filter(|&v| match op {
                "<" => v < case.cutoff,
                "<=" => v <= case.cutoff,
                "==" => v == case.cutoff,
                "!=" => v != case.cutoff,
                ">=" => v >= case.cutoff,
                ">" => v > case.cutoff,
                _ => panic!("unsupported test predicate {op}"),
            })
            .collect(),
    }
}

/// Verified device IR for this case on `caps`, lowered from the source-bound
/// contract of the exact function bytes.
pub fn plan(case: &Case, caps: Capabilities) -> seg::Verified {
    let function = function(case.kind);
    let (ir, origin): (Ir, Provenance) = match case.kind {
        Kind::Map => segmented_device::MapContract::from_source(function.as_bytes(), "run")
            .unwrap()
            .ir(&case.input, &case.offsets)
            .unwrap(),
        Kind::Sum | Kind::Product => {
            let contract =
                segmented_source::Contract::from_source(function.as_bytes(), "run").unwrap();
            segmented_device::reduce_ir(&contract, &case.input, &case.offsets).unwrap()
        }
        Kind::Compact(_) => {
            let contract =
                compaction_source::Contract::from_source(function.as_bytes(), "run").unwrap();
            segmented_device::compact_ir(&contract, case.input.len() as u32, case.cutoff).unwrap()
        }
    };
    seg::verify(ir, caps, origin).unwrap()
}
