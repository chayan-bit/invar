//! GNET-05 seeded generator for the general-net acceptance corpus.
//!
//! Every program is ordinary Invar source with a runtime entry
//! `run(a, b) -> i32` (and a `main` calling it on one recorded input); the
//! inputs, not the source, choose recursion depth, range, graph size and
//! which branches run. Positive inputs finish within 100,000 reference steps
//! and far below the 64-frame call-depth limit. Families:
//! irregular recursive divide-and-conquer, input-dependent branch trees,
//! dynamically discovered graph frontiers (recursive discovery and a serial
//! array BFS), serial/negative controls, cross-family compositions, and
//! residual fork-join (recursion whose combine is not linear in the child
//! results, or whose control reads them).
//! Programs whose work grows with the input also carry a ladder of larger
//! candidate inputs; `calibrate` records the rung with the most reference
//! steps within `LARGE_STEPS` as one more input.
//! Every input declares its reference and oracle step budget: 100,000
//! (`DEFAULT_BUDGET`, the fuel every GNET path runs with) except the deep
//! inputs (`DEEP_INPUTS`), whose budgets are capped at `MAX_BUDGET`.
//! Structure is drawn from a SplitMix64 stream per program; the stream seed,
//! the generator version and the split are recorded in `manifest.json`.
//! Within each family a seeded shuffle assigns half the programs to the
//! held-out split; held-out programs must never reach a tuning path
//! (`HeldOut::admit_for_tuning`).
#![allow(dead_code)]

use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write;

pub const GENERATOR: &str = "gnet05-gen-v3";
pub const ROOT_SEED: u64 = 0x474e_4554_3035_7631;
pub const FAMILIES: [&str; 6] = [
    "divide", "branch", "frontier", "serial", "compose", "forkjoin",
];
pub const PER_FAMILY: usize = 12;
pub const HELD_OUT_PER_FAMILY: usize = 6;
/// Step budget of a calibrated large input: 90% of the 100,000-step fuel the
/// reference, the oracle and the device configuration all run with.
pub const LARGE_STEPS: u64 = 90_000;
/// Declared reference and oracle step budget of every recorded input that
/// is not a deep input.
pub const DEFAULT_BUDGET: u64 = 100_000;
/// Corpus-level cap on a declared per-input step budget.
pub const MAX_BUDGET: u64 = 10_000_000;
/// Deep inputs: per program, the ladder rung `deep_select.py` selected with
/// the #133 net-laws prototype (the first rung whose law-directed average
/// parallelism W/S reaches about 1,000, least work across chains), and its
/// declared step budget: the reference steps rounded up to a whole million.
pub const DEEP_INPUTS: &[(&str, (i64, i64), u64)] = &[
    ("gnet05-divide-00", (49152, 24), 2_000_000),
    ("gnet05-divide-01", (24576, 24), 2_000_000),
    ("gnet05-divide-02", (32768, 24), 2_000_000),
    ("gnet05-divide-03", (49152, 24), 2_000_000),
    ("gnet05-divide-04", (24576, 24), 1_000_000),
    ("gnet05-divide-05", (65536, 24), 1_000_000),
    ("gnet05-divide-06", (24576, 24), 1_000_000),
    ("gnet05-divide-07", (24576, 24), 2_000_000),
    ("gnet05-divide-08", (4096, 24), 2_000_000),
    ("gnet05-divide-09", (12288, 24), 2_000_000),
    ("gnet05-divide-10", (24576, 24), 2_000_000),
    ("gnet05-divide-11", (6144, 24), 3_000_000),
    ("gnet05-compose-00", (2048, 16), 1_000_000),
    ("gnet05-compose-01", (12288, 16), 2_000_000),
    ("gnet05-compose-02", (12288, 16), 1_000_000),
    ("gnet05-compose-03", (65536, 16), 1_000_000),
    ("gnet05-compose-06", (16384, 16), 1_000_000),
    ("gnet05-compose-10", (1536, 16), 3_000_000),
    ("gnet05-forkjoin-00", (19, 654), 1_000_000),
    ("gnet05-forkjoin-01", (18, 837), 1_000_000),
    ("gnet05-forkjoin-02", (21, 49), 3_000_000),
    ("gnet05-forkjoin-03", (12288, 24), 4_000_000),
    ("gnet05-forkjoin-04", (15, 206), 5_000_000),
    ("gnet05-forkjoin-05", (26, 32), 7_000_000),
    ("gnet05-forkjoin-06", (14, 419), 1_000_000),
    ("gnet05-forkjoin-07", (49152, 24), 2_000_000),
    ("gnet05-forkjoin-08", (8192, 24), 7_000_000),
    ("gnet05-forkjoin-09", (12288, 24), 5_000_000),
    ("gnet05-forkjoin-10", (24576, 24), 2_000_000),
    ("gnet05-forkjoin-11", (65536, 24), 2_000_000),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Tuning,
    HeldOut,
}
impl Split {
    pub fn label(self) -> &'static str {
        match self {
            Split::Tuning => "tuning",
            Split::HeldOut => "held_out",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub id: String,
    pub family: &'static str,
    pub ordinal: usize,
    pub seed: u64,
    pub split: Split,
    pub source: String,
    /// `i32` or `u32`: the type of both `run` parameters.
    pub param: &'static str,
    /// Runtime inputs `(a, b)`; the first is always a tiny control.
    pub inputs: Vec<(i64, i64)>,
    /// Structural choices drawn for this program.
    pub shape: Vec<(&'static str, i64)>,
    /// Inputs that must trap or exceed a limit (negative cases).
    pub negative_inputs: Vec<usize>,
    /// Candidate large inputs as ascending chains (empty when work cannot
    /// grow with the input).
    pub ladder: Vec<Vec<(i64, i64)>>,
    /// Index in `inputs` of the calibrated large input.
    pub large_input: Option<usize>,
    /// Declared reference and oracle step budget per input.
    pub budgets: Vec<u64>,
    /// Index in `inputs` of the deep input.
    pub deep_input: Option<usize>,
}

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in `lo..=hi`.
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo + 1) as u64) as i64
    }
}

struct Gen<'a> {
    r: &'a mut Rng,
    shape: Vec<(&'static str, i64)>,
}
impl Gen<'_> {
    fn draw(&mut self, name: &'static str, lo: i64, hi: i64) -> i64 {
        let v = self.r.range(lo, hi);
        self.shape.push((name, v));
        v
    }
    fn free(&mut self, lo: i64, hi: i64) -> i64 {
        self.r.range(lo, hi)
    }
}

/// A small i32 leaf with a data-dependent loop or branch; results < 10007.
fn leaf(g: &mut Gen, name: &str) -> String {
    match g.draw("leaf_kind", 0, 2) {
        0 => {
            let (l, m) = (g.free(3, 17), g.free(2, 13));
            format!(
                "fn {name}(x: i32) -> i32 {{\n    var s = x % 101;\n    var i = 0;\n    while i < x % {l} {{\n        s = (s * {m} + i) % 10007;\n        i = i + 1;\n    }}\n    return s;\n}}\n"
            )
        }
        1 => {
            let k = g.free(1, 50);
            format!(
                "fn {name}(x: i32) -> i32 {{\n    if x % 2 == 0 {{ return (x / 2 + {k}) % 10007; }}\n    return (x * 3 + 1) % 10007;\n}}\n"
            )
        }
        _ => {
            let cap = g.free(20, 60);
            format!(
                "fn {name}(x: i32) -> i32 {{\n    var v = x % 997 + 1;\n    var k = 0;\n    while v != 1 {{\n        if k == {cap} {{ return k; }}\n        if v % 2 == 0 {{ v = v / 2; }} else {{ v = v * 3 + 1; }}\n        k = k + 1;\n    }}\n    return k;\n}}\n"
            )
        }
    }
}

/// Irregular divide and conquer over `[lo, hi)` with a data-dependent split
/// point, arity 2 or 3, uneven depth budgets and a pluggable leaf call.
/// `nonlinear` replaces the linear combine (see `nonlinear_combine`).
fn divide_fn(g: &mut Gen, name: &str, leaf_call: &str, nonlinear: Option<i64>) -> String {
    let arity = g.draw("arity", 2, 3);
    let t = g.draw("threshold", 2, 6);
    let q = g.draw("split_den", 5, 9);
    let rr = g.draw("split_skew", 1, 3);
    let p = g.free(1, q - rr);
    let (c1, c2, w, w2) = (g.free(1, 9), g.free(0, 9), g.free(1, 7), g.free(1, 5));
    let uneven = g.draw("uneven_depth", 1, 2);
    let mut s = String::new();
    writeln!(s, "fn {name}(lo: i32, hi: i32, d: i32) -> i32 {{").unwrap();
    writeln!(
        s,
        "    if hi - lo <= {t} {{ return {}; }}",
        call(leaf_call, &format!("lo * {c1} + hi"))
    )
    .unwrap();
    writeln!(
        s,
        "    if d <= 0 {{ return {}; }}",
        call(leaf_call, &format!("hi - lo + {c2}"))
    )
    .unwrap();
    writeln!(
        s,
        "    let m1 = lo + 1 + ((hi - lo - 2) * ({p} + (lo % {rr} + {rr}) % {rr})) / {q};"
    )
    .unwrap();
    let children: &[&str] = if arity == 2 {
        writeln!(s, "    let a = {name}(lo, m1, d - 1);").unwrap();
        writeln!(s, "    let b = {name}(m1, hi, d - {uneven});").unwrap();
        if nonlinear.is_none() {
            writeln!(s, "    return (a * {w} + b) % 100003;").unwrap();
        }
        &["a", "b"]
    } else {
        let (pp, qq) = (g.free(1, 4), g.free(5, 8));
        writeln!(s, "    let m2 = m1 + 1 + ((hi - m1 - 2) * {pp}) / {qq};").unwrap();
        writeln!(s, "    let a = {name}(lo, m1, d - 1);").unwrap();
        writeln!(s, "    let b = {name}(m1, m2, d - 1);").unwrap();
        writeln!(s, "    let c = {name}(m2, hi, d - {uneven});").unwrap();
        if nonlinear.is_none() {
            writeln!(s, "    return (a * {w} + b + c * {w2}) % 100003;").unwrap();
        }
        &["a", "b", "c"]
    };
    if let Some(kind) = nonlinear {
        let extra = format!("{name}(lo, m1, d - 2)");
        s.push_str(&nonlinear_combine(g, kind, children, &extra, "    ", ""));
    }
    s.push_str("}\n");
    s
}

/// `leaf_call` is either a function name or a template with `$` for the argument.
fn call(leaf_call: &str, arg: &str) -> String {
    if leaf_call.contains('$') {
        leaf_call.replace('$', arg)
    } else {
        format!("{leaf_call}({arg})")
    }
}

/// A combine of child results `xs` (each below 10007) that no linear law
/// covers, so the recursion keeps its fork-join form:
/// 0: degree-two product of children;
/// 1: the children's order selects between two non-linear combines;
/// 2: the children's values decide whether one more recursive call (`extra`)
///    runs, and its result multiplies a child.
/// Products stay below 2^31: every operand is reduced modulo 10007 first.
fn nonlinear_combine(
    g: &mut Gen,
    kind: i64,
    xs: &[&str],
    extra: &str,
    indent: &str,
    tag: &str,
) -> String {
    let (x0, x1) = (xs[0], xs[1]);
    let rest = xs[1..].join(" + ");
    let w = g.free(2, 9);
    match kind {
        0 => format!("{indent}return ({x0} * {x1} + {x0} * {w} + {rest}) % 10007;\n"),
        1 => format!(
            "{indent}if {x0} > {x1} {{ return ({x0} * {w} + {rest}) % 10007; }}\n{indent}return ({x1} * {x1} % 10007 + {x0} + {rest}) % 10007;\n"
        ),
        _ => {
            let k = g.free(2, 5);
            format!(
                "{indent}if ({x0} + {x1}) % {k} == 0 {{\n{indent}    let extra{tag} = {extra};\n{indent}    return (extra{tag} * {x0} + {rest}) % 10007;\n{indent}}}\n{indent}return ({x0} * {w} + {rest}) % 10007;\n"
            )
        }
    }
}

/// Input-dependent branch tree: each node selects a serial, fan-out or leaf
/// shape from its runtime state.
fn branch_fn(g: &mut Gen, name: &str, leaf_name: &str, nonlinear: Option<i64>) -> String {
    let k = g.draw("branches", 2, 4);
    let (c, e) = (g.free(2, 31), g.free(1, 7));
    let mut kinds: Vec<i64> = (0..k).map(|_| g.free(0, 3)).collect();
    // Every tree has at least one fan-out and one non-fan node, so which
    // shape runs depends on the input.
    kinds[0] = g.draw("fan_width", 2, 3) - 1;
    let last = kinds.len() - 1;
    if kinds[last] == 1 || kinds[last] == 2 {
        kinds[last] = 0;
    }
    let mut s = String::new();
    writeln!(s, "fn {name}(n: i32, s: i32) -> i32 {{").unwrap();
    writeln!(s, "    if n <= 0 {{ return {leaf_name}(s); }}").unwrap();
    writeln!(s, "    let sel = (s * {c} + n * {e}) % {k};").unwrap();
    for (j, kind) in kinds.iter().enumerate() {
        let body = branch_body(g, name, leaf_name, *kind, j, nonlinear);
        if j == last {
            s.push_str(
                &body
                    .replace("\n        ", "\n    ")
                    .replacen("        ", "    ", 1),
            );
        } else {
            writeln!(s, "    if sel == {j} {{").unwrap();
            s.push_str(&body);
            s.push_str("    }\n");
        }
    }
    s.push_str("}\n");
    s
}

fn next_state(g: &mut Gen) -> String {
    format!("(s * {} + {}) % 1009", g.free(2, 29), g.free(0, 97))
}

fn branch_body(
    g: &mut Gen,
    name: &str,
    leaf_name: &str,
    kind: i64,
    j: usize,
    nonlinear: Option<i64>,
) -> String {
    match kind {
        // serial node
        0 => format!("        return {name}(n - 1, {});\n", next_state(g)),
        // fan-out of 2 or 3 independent calls with uneven depth
        1 | 2 => {
            let width = kind + 1;
            let mut body = String::new();
            let names: Vec<String> = (0..width).map(|i| format!("v{j}_{i}")).collect();
            for v in &names {
                let d = g.free(1, 2);
                writeln!(
                    body,
                    "        let {v} = {name}(n - {d}, {});",
                    next_state(g)
                )
                .unwrap();
            }
            let m = g.free(1, 9);
            let Some(nl) = nonlinear else {
                writeln!(
                    body,
                    "        return ({} * {m} + {}) % 10007;",
                    names[0],
                    names[1..].join(" + ")
                )
                .unwrap();
                return body;
            };
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            let extra = format!("{name}(n - 2, {})", next_state(g));
            body.push_str(&nonlinear_combine(
                g,
                nl,
                &refs,
                &extra,
                "        ",
                &j.to_string(),
            ));
            body
        }
        // leaf node
        _ => format!("        return ({leaf_name}(s) + n) % 10007;\n"),
    }
}

/// Recursive discovery: a vertex's out-degree and successors are computed
/// from its runtime value in an implicit graph of runtime size `n`.
fn visit_fn(g: &mut Gen, name: &str, work_call: &str) -> String {
    let dmod = g.draw("max_degree", 3, 5);
    let c = g.free(2, 19);
    let mut s = String::new();
    writeln!(s, "fn {name}(v: i32, n: i32, d: i32) -> i32 {{").unwrap();
    writeln!(s, "    if d <= 0 {{ return {}; }}", call(work_call, "v")).unwrap();
    writeln!(s, "    let deg = (v * {c} + d) % {dmod};").unwrap();
    writeln!(s, "    if deg == 0 {{ return {}; }}", call(work_call, "v")).unwrap();
    for deg in 1..dmod.min(4) {
        let mut names = Vec::new();
        let mut body = String::new();
        for i in 0..deg {
            let v = format!("w{deg}_{i}");
            let (a, b) = (g.free(2, 41), g.free(1, 97));
            writeln!(
                body,
                "        let {v} = {name}((v * {a} + {b}) % n, n, d - 1);"
            )
            .unwrap();
            names.push(v);
        }
        if deg + 1 == dmod.min(4) {
            s.push_str(&body.replace("        ", "    "));
            writeln!(s, "    return ({} + v) % 10007;", names.join(" + ")).unwrap();
        } else {
            writeln!(s, "    if deg == {deg} {{").unwrap();
            s.push_str(&body);
            writeln!(s, "        return ({} + v) % 10007;", names.join(" + ")).unwrap();
            s.push_str("    }\n");
        }
    }
    s.push_str("}\n");
    s
}

/// Serial frontier: a level-synchronous BFS over owned arrays in an implicit
/// graph whose degree depends on the vertex. No independent call sites.
fn bfs_program(g: &mut Gen) -> String {
    let (a, b, c) = (g.free(2, 31), g.free(1, 13), g.free(0, 11));
    let dmax = g.draw("max_degree", 2, 4);
    let n0 = g.draw("min_vertices", 4, 9);
    let rounds = g.draw("round_mod", 3, 7);
    format!(
        "fn succ(v: u32, k: u32, n: u32) -> u32 {{ return (v * {a}u32 + k * {b}u32 + {c}u32) % n; }}
fn bfs(n: u32, start: u32, rounds: u32) -> i32 {{
    var seen: Array<u32> = filled_u32(n, 0u32);
    var frontier: Array<u32> = filled_u32(n, 0u32);
    frontier[start % n] = 1u32;
    var count: i32 = 0;
    var round: u32 = 0u32;
    while round < rounds {{
        var next: Array<u32> = filled_u32(n, 0u32);
        for u in 0u32 .. n {{
            if frontier[u] == 1u32 and seen[u] == 0u32 {{
                seen[u] = 1u32;
                count = count + 1;
                let degree = u % {dmax}u32 + 1u32;
                for k in 0u32 .. degree {{
                    if seen[succ(u, k, n)] == 0u32 {{ next[succ(u, k, n)] = 1u32; }}
                }}
            }}
        }}
        for i in 0u32 .. n {{ frontier[i] = next[i]; }}
        round = round + 1u32;
    }}
    return count;
}}
fn run(a: u32, b: u32) -> i32 {{ return bfs(a + {n0}u32, b, 1u32 + a % {rounds}u32); }}
"
    )
}

/// Source, parameter type, inputs, drawn shape, negative input indices and
/// the large-input ladder.
type Drawn = (
    String,
    &'static str,
    Vec<(i64, i64)>,
    Vec<(&'static str, i64)>,
    Vec<usize>,
    Vec<Vec<(i64, i64)>>,
);

/// Sizes 64, 90, 128, 180, ... (ratio about the square root of two).
fn sizes(base: i64, rungs: i64) -> impl Iterator<Item = i64> {
    (0..rungs).map(move |k| [base, base * 3 / 2][(k % 2) as usize] << (k / 2))
}

/// Range-size ladder for a divide and conquer `run(a, b)`: `a` grows, `b` is
/// a depth budget that keeps recursion far below the call-depth limit.
fn range_ladder(depth: i64) -> Vec<Vec<(i64, i64)>> {
    vec![sizes(64, 21).map(|a| (a, depth)).collect()]
}

/// Depth ladders for a branch tree `run(n, s)`, one per recorded state `s`
/// (the state picks the tree's shape); `n <= 48` keeps every call path below
/// the 64-frame limit.
fn depth_ladder(inputs: &[(i64, i64)]) -> Vec<Vec<(i64, i64)>> {
    inputs[1..]
        .iter()
        .map(|&(_, s)| (6..=48).map(|n| (n, s)).collect())
        .collect()
}

fn program(family: &'static str, seed: u64) -> Drawn {
    let mut r = Rng::new(seed);
    let mut g = Gen {
        r: &mut r,
        shape: Vec::new(),
    };
    let mut negative = Vec::new();
    let mut ladder = Vec::new();
    let (source, param, inputs) = match family {
        "divide" => {
            let off = g.free(0, 50);
            let src = format!(
                "{}{}fn run(a: i32, b: i32) -> i32 {{ return dc({off}, {off} + a, b); }}\n",
                leaf(&mut g, "leaf"),
                divide_fn(&mut g, "dc", "leaf", None)
            );
            let inputs = vec![
                (g.free(0, 3), g.free(0, 1)),
                (g.free(20, 80), g.free(2, 4)),
                (g.free(150, 400), g.free(5, 6)),
                (g.free(200, 400), 1),
            ];
            ladder = range_ladder(24);
            (src, "i32", inputs)
        }
        "branch" => {
            let src = format!(
                "{}{}fn run(a: i32, b: i32) -> i32 {{ return t(a, b % 1009); }}\n",
                leaf(&mut g, "w"),
                branch_fn(&mut g, "t", "w", None)
            );
            let inputs = vec![
                (g.free(0, 1), g.free(0, 1008)),
                (g.free(3, 4), g.free(0, 1008)),
                (g.free(5, 6), g.free(0, 1008)),
                (g.free(5, 6), g.free(0, 1008)),
            ];
            ladder = depth_ladder(&inputs);
            (src, "i32", inputs)
        }
        "frontier" if g.draw("serial_bfs", 0, 2) == 0 => {
            let src = bfs_program(&mut g);
            let inputs = vec![
                (0, g.free(0, 3)),
                (g.free(5, 20), g.free(0, 30)),
                (g.free(30, 60), g.free(0, 90)),
                (g.free(30, 60), g.free(0, 90)),
            ];
            // `a` one below a multiple of the round modulus runs every round.
            let rounds = g.shape.iter().find(|(k, _)| *k == "round_mod").unwrap().1;
            ladder = vec![
                sizes(4, 25)
                    .map(|k| (rounds * k - 1, inputs[3].1))
                    .collect(),
            ];
            (src, "u32", inputs)
        }
        "frontier" => {
            let n0 = g.draw("min_vertices", 5, 40);
            let dmax = g.draw("max_depth", 4, 6);
            let src = format!(
                "{}{}fn run(a: i32, b: i32) -> i32 {{\n    let n = a + {n0};\n    return visit(b % n, n, 1 + (a + b) % {dmax});\n}}\n",
                leaf(&mut g, "work"),
                visit_fn(&mut g, "visit", "work")
            );
            let inputs = vec![
                (0, 0),
                (g.free(0, 50), g.free(0, 100)),
                (g.free(50, 500), g.free(0, 1000)),
                (g.free(50, 500), g.free(0, 1000)),
            ];
            (src, "i32", inputs)
        }
        "serial" => {
            let trap = g.draw("trap", 0, 1) == 1;
            let deep = g.draw("deep_recursion", 0, 1) == 1;
            let (m, k1, k2) = (g.free(2, 13), g.free(0, 99), g.free(0, 99));
            let tmod = g.free(3, 9);
            let z = g.free(0, tmod - 1);
            let mut src = leaf(&mut g, "f");
            writeln!(
                src,
                "fn g(x: i32, k: i32) -> i32 {{ return (f(x) * {m} + k) % 10007; }}"
            )
            .unwrap();
            if deep {
                src.push_str("fn down(n: i32) -> i32 {\n    if n <= 0 { return 0; }\n    return 1 + down(n - 1);\n}\n");
            }
            src.push_str("fn run(a: i32, b: i32) -> i32 {\n    let s1 = f(a);\n");
            writeln!(src, "    let s2 = g(s1 + b, {k1});").unwrap();
            writeln!(src, "    let s3 = f(s2 + {k2});").unwrap();
            let mut tail = "s3".to_string();
            if trap {
                writeln!(src, "    let q = s3 / (b % {tmod} - {z});").unwrap();
                tail.push_str(" + q");
            }
            if deep {
                // Positive inputs (b < 10) stay far below the evaluator's
                // call-depth limit; the negative input (b >= 2000) exceeds it.
                writeln!(src, "    let e = down(s3 % 3 + b);").unwrap();
                tail.push_str(" + e");
            }
            writeln!(src, "    return ({tail}) % 10007;\n}}").unwrap();
            // b chosen so the divisor is nonzero unless the input is negative.
            let safe = |g: &mut Gen, lo: i64, hi: i64| loop {
                let b = g.free(lo, hi);
                if b % tmod != z {
                    break b;
                }
            };
            let mut inputs = vec![
                (g.free(0, 2), safe(&mut g, 0, 2)),
                (g.free(10, 500), safe(&mut g, 0, 9)),
            ];
            if trap {
                inputs.push((g.free(0, 500), z + tmod * g.free(0, 3)));
                negative.push(inputs.len() - 1);
            } else {
                inputs.push((g.free(0, 500), safe(&mut g, 0, 9)));
            }
            if deep {
                // Deep enough to exceed the evaluator's call-depth limit.
                inputs.push((g.free(0, 500), safe(&mut g, 2000, 4000)));
                negative.push(inputs.len() - 1);
            } else {
                inputs.push((g.free(0, 500), safe(&mut g, 10, 40)));
            }
            (src, "i32", inputs)
        }
        "compose" => {
            if g.draw("compose_kind", 0, 1) == 0 {
                // divide and conquer whose leaves are input-dependent branch trees
                let off = g.free(0, 50);
                let src = format!(
                    "{}{}{}fn run(a: i32, b: i32) -> i32 {{ return dc({off}, {off} + a, b); }}\n",
                    leaf(&mut g, "w"),
                    branch_fn(&mut g, "t", "w", None),
                    divide_fn(&mut g, "dc", "t(($) % 2, ($) % 1009)", None)
                );
                let inputs = vec![
                    (g.free(0, 3), g.free(0, 1)),
                    (g.free(20, 60), g.free(1, 2)),
                    (g.free(60, 150), g.free(2, 3)),
                    (g.free(60, 150), 1),
                ];
                ladder = range_ladder(16);
                (src, "i32", inputs)
            } else {
                // discovered frontier whose vertex work is a divide and conquer
                let n0 = g.draw("min_vertices", 5, 40);
                let dmax = g.draw("max_depth", 3, 4);
                let src = format!(
                    "{}{}{}fn run(a: i32, b: i32) -> i32 {{\n    let n = a + {n0};\n    return visit(b % n, n, 1 + (a + b) % {dmax});\n}}\n",
                    leaf(&mut g, "leaf"),
                    divide_fn(&mut g, "dc", "leaf", None),
                    visit_fn(&mut g, "visit", "dc($ % 17, $ % 17 + 9, 2)")
                );
                let inputs = vec![
                    (0, 0),
                    (g.free(0, 50), g.free(0, 100)),
                    (g.free(50, 500), g.free(0, 1000)),
                    (g.free(50, 500), g.free(0, 1000)),
                ];
                (src, "i32", inputs)
            }
        }
        "forkjoin" => {
            let combine = g.draw("nonlinear_combine", 0, 2);
            if g.draw("forkjoin_tree", 0, 1) == 0 {
                let off = g.free(0, 50);
                let src = format!(
                    "{}{}fn run(a: i32, b: i32) -> i32 {{ return dc({off}, {off} + a, b); }}\n",
                    leaf(&mut g, "leaf"),
                    divide_fn(&mut g, "dc", "leaf", Some(combine))
                );
                let inputs = vec![
                    (g.free(0, 3), g.free(0, 1)),
                    (g.free(20, 80), g.free(2, 4)),
                    (g.free(150, 400), g.free(5, 6)),
                    (g.free(200, 400), g.free(7, 9)),
                ];
                ladder = range_ladder(24);
                (src, "i32", inputs)
            } else {
                let src = format!(
                    "{}{}fn run(a: i32, b: i32) -> i32 {{ return t(a, b % 1009); }}\n",
                    leaf(&mut g, "w"),
                    branch_fn(&mut g, "t", "w", Some(combine))
                );
                let inputs = vec![
                    (g.free(0, 1), g.free(0, 1008)),
                    (g.free(3, 4), g.free(0, 1008)),
                    (g.free(5, 6), g.free(0, 1008)),
                    (g.free(5, 6), g.free(0, 1008)),
                ];
                ladder = depth_ladder(&inputs);
                (src, "i32", inputs)
            }
        }
        _ => unreachable!("unknown family {family}"),
    };
    (source, param, inputs, g.shape, negative, ladder)
}

/// The per-program stream seed: root seed, family index and ordinal mixed by
/// one SplitMix64 step, so programs are independent of generation order.
pub fn program_seed(family_index: usize, ordinal: usize) -> u64 {
    Rng::new(ROOT_SEED ^ ((family_index as u64) << 32) ^ ordinal as u64).next()
}

/// Seeded Fisher-Yates over ordinals; the first `HELD_OUT_PER_FAMILY` are held out.
pub fn held_out_ordinals(family_index: usize) -> BTreeSet<usize> {
    let mut r = Rng::new(ROOT_SEED.rotate_left(17) ^ family_index as u64);
    let mut order: Vec<usize> = (0..PER_FAMILY).collect();
    for i in (1..order.len()).rev() {
        let j = (r.next() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order.into_iter().take(HELD_OUT_PER_FAMILY).collect()
}

pub fn generate() -> Vec<Program> {
    let mut out = Vec::new();
    for (fi, family) in FAMILIES.iter().enumerate() {
        let held = held_out_ordinals(fi);
        for ordinal in 0..PER_FAMILY {
            let seed = program_seed(fi, ordinal);
            let (mut source, param, inputs, shape, negative_inputs, ladder) = program(family, seed);
            // A `main` makes each program runnable as is; it calls `run` on
            // the second recorded input.
            let suffix = if param == "u32" { "u32" } else { "" };
            let (a, b) = inputs[1];
            writeln!(
                source,
                "fn main() -> i32 {{ return run({a}{suffix}, {b}{suffix}); }}"
            )
            .unwrap();
            out.push(Program {
                id: format!("gnet05-{family}-{ordinal:02}"),
                family,
                ordinal,
                seed,
                split: if held.contains(&ordinal) {
                    Split::HeldOut
                } else {
                    Split::Tuning
                },
                source,
                param,
                budgets: vec![DEFAULT_BUDGET; inputs.len()],
                inputs,
                shape,
                negative_inputs,
                ladder,
                large_input: None,
                deep_input: None,
            });
        }
    }
    out
}

/// Records, per program, the ladder rung with the most reference steps among
/// those whose run returns a value within `budget` steps (`steps` gives the
/// step count of a run that returned a value, `None` for a trap or a fuel
/// stop) as one more input; ties keep the earlier rung. Each chain ascends,
/// so its walk stops at the first rung that does not fit.
pub fn calibrate(
    programs: Vec<Program>,
    budget: u64,
    mut steps: impl FnMut(&Program, (i64, i64)) -> Option<u64>,
) -> Vec<Program> {
    programs
        .into_iter()
        .map(|p| {
            let mut best: Option<(u64, (i64, i64))> = None;
            for chain in &p.ladder {
                for &rung in chain {
                    match steps(&p, rung) {
                        Some(n) if n <= budget => {
                            if best.is_none_or(|(m, _)| n > m) {
                                best = Some((n, rung));
                            }
                        }
                        _ => break,
                    }
                }
            }
            let Some((_, rung)) = best else { return p };
            let mut inputs = p.inputs.clone();
            inputs.push(rung);
            let mut budgets = p.budgets.clone();
            budgets.push(DEFAULT_BUDGET);
            Program {
                large_input: Some(p.inputs.len()),
                inputs,
                budgets,
                ..p
            }
        })
        .collect()
}

/// Appends each program's deep input from `DEEP_INPUTS` with its declared
/// budget, after the calibrated large input.
pub fn deepen(programs: Vec<Program>) -> Vec<Program> {
    programs
        .into_iter()
        .map(|p| {
            let Some(&(_, input, budget)) = DEEP_INPUTS.iter().find(|d| d.0 == p.id) else {
                return p;
            };
            let mut inputs = p.inputs.clone();
            inputs.push(input);
            let mut budgets = p.budgets.clone();
            budgets.push(budget);
            Program {
                deep_input: Some(p.inputs.len()),
                inputs,
                budgets,
                ..p
            }
        })
        .collect()
}

const KEYWORDS: &[&str] = &[
    "fn",
    "let",
    "var",
    "if",
    "else",
    "while",
    "for",
    "in",
    "return",
    "true",
    "false",
    "and",
    "or",
    "not",
    "i32",
    "u32",
    "bool",
    "Array",
    "read",
    "filled_i32",
    "filled_u32",
    "wrapping_add",
    "wrapping_mul",
];

/// Identity of a program up to formatting, comments and consistent renaming
/// of its identifiers: identifiers are numbered by first appearance.
pub fn normalized_sha256(source: &[u8]) -> [u8; 32] {
    let text = String::from_utf8_lossy(source);
    let bytes = text.as_bytes();
    let mut names: Vec<&str> = Vec::new();
    let mut out = String::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if c.is_ascii_whitespace() {
            i += 1;
        } else if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let word = &text[start..i];
            if KEYWORDS.contains(&word) {
                out.push_str(word);
            } else {
                let k = names.iter().position(|n| *n == word).unwrap_or_else(|| {
                    names.push(word);
                    names.len() - 1
                });
                write!(out, "${k}").unwrap();
            }
            out.push(' ');
        } else {
            out.push(c as char);
            i += 1;
            if c.is_ascii_digit() {
                while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
                    out.push(bytes[i] as char);
                    i += 1;
                }
            }
            out.push(' ');
        }
    }
    Sha256::digest(out.as_bytes()).into()
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The held-out set as normalized identities. Tuning paths call
/// `admit_for_tuning` and must refuse anything it refuses.
pub struct HeldOut {
    identities: BTreeSet<[u8; 32]>,
}
impl HeldOut {
    pub fn of(programs: &[Program]) -> Self {
        Self::of_sources(
            programs
                .iter()
                .filter(|p| p.split == Split::HeldOut)
                .map(|p| p.source.as_str()),
        )
    }
    /// The same guard over any held-out sources (the robotics corpus).
    pub fn of_sources<'a>(held_out: impl IntoIterator<Item = &'a str>) -> Self {
        Self {
            identities: held_out
                .into_iter()
                .map(|s| normalized_sha256(s.as_bytes()))
                .collect(),
        }
    }
    pub fn admit_for_tuning(&self, source: &[u8]) -> Result<(), String> {
        let id = normalized_sha256(source);
        if self.identities.contains(&id) {
            return Err(format!("held-out program {}", hex(&id)));
        }
        Ok(())
    }
    /// SHA-256 over the sorted held-out identities, pinned in the manifest.
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        for id in &self.identities {
            h.update(id);
        }
        h.finalize().into()
    }
    pub fn len(&self) -> usize {
        self.identities.len()
    }
}

/// The only corpus view a tuning path may consume.
pub fn tuning_programs<'a>(programs: &'a [Program], held: &HeldOut) -> Vec<&'a Program> {
    programs
        .iter()
        .filter(|p| p.split == Split::Tuning)
        .filter(|p| held.admit_for_tuning(p.source.as_bytes()).is_ok())
        .collect()
}

pub fn manifest(programs: &[Program]) -> serde_json::Value {
    let held = HeldOut::of(programs);
    serde_json::json!({
        "schema": "invar-gnet05-corpus-v1",
        "generator": {"name": GENERATOR, "root_seed": format!("{ROOT_SEED:#018x}"), "rng": "splitmix64",
            "per_family": PER_FAMILY, "held_out_per_family": HELD_OUT_PER_FAMILY,
            "large_steps": LARGE_STEPS, "default_budget": DEFAULT_BUDGET, "max_budget": MAX_BUDGET},
        "families": FAMILIES,
        "held_out": {"count": held.len(), "normalized_identity_sha256": hex(&held.digest())},
        "entry": "run",
        "programs": programs.iter().map(|p| serde_json::json!({
            "id": p.id,
            "family": p.family,
            "ordinal": p.ordinal,
            "seed": format!("{:#018x}", p.seed),
            "split": p.split.label(),
            "file": format!("programs/{}.inv", p.id),
            "source_sha256": hex(&Sha256::digest(p.source.as_bytes())),
            "normalized_sha256": hex(&normalized_sha256(p.source.as_bytes())),
            "param": p.param,
            "inputs": p.inputs,
            "negative_inputs": p.negative_inputs,
            "ladder": p.ladder,
            "large_input": p.large_input,
            "budgets": p.budgets,
            "deep_input": p.deep_input,
            "shape": p.shape.iter().map(|(k, v)| serde_json::json!([k, v])).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}
