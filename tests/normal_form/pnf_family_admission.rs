#![forbid(unsafe_code)]
//! Independent source/MIR recognizers are test oracles only. Agreement is a
//! strict gate: every recognizer match needs an admission or a named native
//! refusal derived by production analysis. INVAR_P3_REPORT exports both; no
//! case-name, source-hash or expected-failure allowlist participates.

use crate::contraction_source;
use crate::frontier_source;
use crate::support;

use invar::{
    admission, check, eval, front, general_net,
    general_net::{
        normal_form::{self, Node, Rep, Ring, certificate, exec},
        regions::{
            self,
            family::{self, Family},
        },
    },
    mir, numeric_contract,
    plan::Execution,
};
use serde_json::{Value as Json, json};
use std::{
    cell::RefCell,
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    rc::Rc,
};

mod cases;

type Key = (usize, usize, usize, Family);
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn checked(source: &str) -> check::CheckedProgram {
    let c = check::check(front::parse(source.as_bytes()).expect("parse"), false).expect("check");
    invar::ownership::check(&c).expect("ownership");
    c
}
fn key(function: usize, span: front::Span, family: Family) -> Key {
    (function, span.start, span.end, family)
}

fn legacy(source: &str, checked: &check::CheckedProgram, program: &mir::Program) -> BTreeSet<Key> {
    let mut out = BTreeSet::new();
    for region in admission::analyze(source.as_bytes(), program)
        .expect("MIR admission")
        .regions()
    {
        let f = program
            .functions
            .iter()
            .position(|f| f.name == region.function())
            .expect("function");
        let family = match region.execution() {
            Execution::Map(_) => Some(Family::Map),
            Execution::FusedProducer(_) => {
                out.insert(key(f, region.span(), Family::FusedMap));
                Some(Family::Map)
            }
            Execution::DagProducer(_) => {
                out.insert(key(f, region.span(), Family::Dag));
                Some(Family::Map)
            }
            Execution::FusedConsumer { .. } | Execution::DagConsumer { .. } => Some(Family::Map),
            Execution::Reduction(_) | Execution::PairReduction(_) => Some(Family::Reduce),
            Execution::Scan(_) => Some(Family::Scan),
            Execution::Serial => None,
        };
        if let Some(family) = family {
            out.insert(key(f, region.span(), family));
        }
    }
    for (i, f) in program.functions.iter().enumerate() {
        // Every non-core recognizer below requires a read array as its first
        // parameter. Avoid repeatedly lowering entire robotics libraries for
        // scalar helpers which cannot satisfy that necessary condition.
        if !f.params.first().is_some_and(|p| {
            p.mode == Some(front::Mode::Read)
                && matches!(p.ty, front::Type::ArrayU32 | front::Type::ArrayF32)
        }) || !matches!(
            f.result,
            front::Type::ArrayU32 | front::Type::ArrayF32 | front::Type::I32 | front::Type::U32
        ) {
            continue;
        }
        let mut add = |span, family| {
            out.insert(key(i, span, family));
        };
        if let Ok(c) = invar::transpose_source::Contract::from_checked(checked, &f.name) {
            add(c.loop_span(), Family::Transpose);
        }
        if let Ok(c) = invar::stencil_source::Contract::from_checked(checked, &f.name) {
            add(c.loop_span(), Family::Stencil);
        }
        if let Ok(c) = invar::tensor::Contraction::from_checked(checked, &f.name) {
            add(c.contract().loop_span(), Family::Tensor);
        }
        if let Ok(c) = contraction_source::Contract::from_source(source.as_bytes(), &f.name) {
            add(c.loop_span(), Family::Tensor);
        }
        if let Ok(c) = invar::segmented_source::Contract::from_source(source.as_bytes(), &f.name) {
            add(c.loop_span(), Family::SegmentedReduce);
        }
        if let Ok(c) = invar::compaction_source::Contract::from_source(source.as_bytes(), &f.name) {
            for s in c.loop_spans() {
                add(s, Family::Compaction);
            }
        }
        if let Ok(c) =
            invar::indexed_source::Contract::from_checked(source.as_bytes(), checked, &f.name)
        {
            add(
                c.loop_span(),
                match c.family() {
                    invar::indexed_source::Family::Gather => Family::Gather,
                    invar::indexed_source::Family::InjectiveScatter => Family::Scatter,
                },
            );
        }
        if let Ok(c) =
            invar::scatter_proof::Contract::from_checked(source.as_bytes(), checked, &f.name)
        {
            add(c.certificate().loop_span(), Family::AffineScatter);
        }
        if let Ok(c) =
            invar::histogram_source::Contract::from_checked(source.as_bytes(), checked, &f.name)
        {
            add(c.loop_span(), Family::Histogram);
        }
        if let Ok(c) = frontier_source::Contract::from_checked(source.as_bytes(), checked, &f.name)
        {
            add(c.loop_span(), Family::Frontier);
        }
        if let Ok(c) = invar::summary_source::Contract::from_checked(checked, &f.name) {
            add(c.loop_span(), Family::Summary);
        }
    }
    out
}

fn compare(id: &str, source: &str) -> (Vec<Json>, usize, Vec<Json>) {
    let checked = checked(source);
    let net = general_net::Net::lower(&checked, general_net::Limits::default()).expect("lower");
    let proof = net.verify().expect("verify");
    let mir = net.compiled_mir(&proof).expect("MIR");
    let form = normal_form::extract(&net, &proof).expect("PNF");
    let legacy = legacy(source, &checked, &mir);
    let (mut ours, refusals, error) = match family::analyze(&net, &proof, &form) {
        Ok(r) => (
            r.admissions
                .iter()
                .map(|a| key(a.region.function, a.span, a.family))
                .collect::<BTreeSet<_>>(),
            r.refusals,
            None,
        ),
        Err(e) => (BTreeSet::new(), Vec::new(), Some(format!("{e:?}"))),
    };
    // This base has no segmented_device::MapContract. An unavailable oracle is
    // not a negative admission; record that coverage gap in the report below.
    ours.retain(|key| key.3 != Family::SegmentedMap);
    let mut differences = Vec::new();
    if let Some(error) = error {
        differences.push(json!({"id":id,"error":error,"source":source}));
    }
    let mut accounted = ours.clone();
    let mut refused = Vec::new();
    for refusal in refusals {
        let a = &refusal.candidate;
        let k = key(a.region.function, a.span, a.family);
        if ours.contains(&k) {
            differences.push(json!({"id":id,"error":"both admitted and refused","source":source}));
        }
        assert_eq!(refusal.reason, family::RefusalReason::MissingFiniteStateLaw);
        assert_eq!(a.family, Family::Summary);
        accounted.insert(k);
        refused.push(json!({"id":id,"function":mir.functions[a.region.function].name,"span":[a.span.start,a.span.end],"family":format!("{:?}",a.family),"reason":format!("{:?}",refusal.reason),"pnf":false,"recognizer":legacy.contains(&k),"source":source}));
    }
    for &(f, start, end, kind) in accounted.symmetric_difference(&legacy) {
        let loop_form = form.functions[f].loops.iter().find(|l| l.region.head.is_some_and(|h| matches!(&mir.functions[f].blocks[h].end,Some(mir::Terminator::Branch { span,.. }) if span.start == start && span.end == end)));
        differences.push(json!({"id":id,"function":mir.functions[f].name,"span":[start,end],"family":format!("{kind:?}"),"pnf":ours.contains(&(f,start,end,kind)),"recognizer":legacy.contains(&(f,start,end,kind)),"node":format!("{:?}",loop_form.map(|l| &l.node)),"source":source}));
    }
    (
        differences,
        form.functions.iter().map(|f| f.loops.len()).sum(),
        refused,
    )
}

fn files(path: &Path, out: &mut Vec<PathBuf>) {
    let mut paths = fs::read_dir(path)
        .expect("directory")
        .map(|e| e.expect("entry").path())
        .collect::<Vec<_>>();
    paths.sort();
    for p in paths {
        if p.is_dir() {
            files(&p, out);
        } else if p.extension().is_some_and(|e| e == "inv") {
            out.push(p);
        }
    }
}

const MAP: &str = "fn run(a: read Array<u32>) -> Array<u32> { var out = filled_u32(len(a),0u32); for i in 0u32..len(a) { out[i] = wrapping_add(a[i],3u32); } return out; }";
const REDUCE: &str = "fn run(a: read Array<u32>, n: u32, seed: u32) -> u32 { var s=seed; for i in 0u32..n { s=wrapping_add(s,a[i]); } return s; }";
const SCAN: &str = "fn run(a: read Array<u32>, n: u32, seed: u32) -> u32 { var s=seed; for i in 0u32..n { s=wrapping_add(wrapping_mul(s,3u32),a[i]); } return s; }";
const GATHER: &str = "fn run(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out=filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i]=a[ix[i]]; } return out; }";
const SCATTER: &str = "fn run(a: read Array<u32>, ix: read Array<u32>, n: u32) -> Array<u32> { var out=filled_u32(n,0u32); for i in 0u32..len(ix) { out[ix[i]]=a[i]; } return out; }";
const HISTOGRAM: &str = "fn run(a: read Array<u32>, ix: read Array<u32>, n: u32) -> Array<u32> { var out=filled_u32(n,0u32); for i in 0u32..len(ix) { out[ix[i]]=wrapping_add(out[ix[i]],a[i]); } return out; }";
const AFFINE: &str = "fn run(a: read Array<u32>) -> Array<u32> { var out=filled_u32(7u32,0u32); for i in 0u32..3u32 { out[2u32*i+1u32]=a[i]; } return out; }";
const TRANSPOSE: &str = "fn run(a: read Array<f32>, rows: u32, cols: u32) -> Array<f32> { var out=filled_f32(rows*cols,0.0); for i in 0u32..rows { for j in 0u32..cols { out[j*rows+i]=a[i*cols+j]; } } return out; }";
const TENSOR: &str = "fn run(a: read Array<f32>, b: read Array<f32>, rows: u32, inner: u32, cols: u32) -> Array<f32> { var out=filled_f32(rows*cols,0.0); for i in 0u32..rows { for j in 0u32..cols { var s=0.0; for k in 0u32..inner { s=s+a[i*inner+k]*b[k*cols+j]; } out[i*cols+j]=s; } } return out; }";
const DIRECT: &str = "fn run(a: read Array<f32>, b: read Array<f32>, rows: u32, inner: u32, cols: u32) -> Array<f32> { var out=filled_f32(rows*cols,0.0); for i in 0u32..rows { for j in 0u32..cols { for k in 0u32..inner { out[i*cols+j]=out[i*cols+j]+a[i*inner+k]*b[k*cols+j]; } } } return out; }";
const SEGMENTED: &str = "fn run(a: read Array<u32>, offsets: read Array<u32>, count: u32) -> Array<u32> { var out=filled_u32(count,0u32); for s in 0u32..count { var acc=0u32; for i in offsets[s]..offsets[s+1u32] { acc=wrapping_add(acc,a[i]); } out[s]=acc; } return out; }";
const SEGMENTED_MAP: &str = "fn run(a: read Array<u32>, offsets: read Array<u32>, count: u32) -> Array<u32> { var out=filled_u32(len(a),0u32); for s in 0u32..count { for i in offsets[s]..offsets[s+1u32] { out[i]=wrapping_add(a[i],s); } } return out; }";
const TABLE: &str = "fn run(coeff: read Array<u32>, seed: u32, count: u32) -> u32 { var s=seed; for i in 0u32..count { s=coeff[i*4u32+s]; } return s; }";
const COMPACT: &str = "fn run(a: read Array<u32>, n: u32, cutoff: u32) -> Array<u32> { var size=0u32; for i in 0u32..n { if a[i]>cutoff { size=size+1u32; } } var out=filled_u32(size,0u32); var pos=0u32; for j in 0u32..n { if a[j]>cutoff { out[pos]=a[j]; pos=pos+1u32; } } return out; }";
const FUSED: &str = "fn run(a: read Array<u32>) -> Array<u32> { var t=filled_u32(len(a),0u32); for i in 0u32..len(a) { t[i]=wrapping_add(a[i],1u32); } let b=move t; var out=filled_u32(len(b),0u32); for j in 0u32..len(b) { out[j]=wrapping_mul(b[j],3u32); } return out; }";

fn matrix(d: usize, prefix: bool) -> String {
    let mut s = "fn run(coeff: read Array<u32>, seed: read Array<u32>, count: u32) -> Array<u32> {"
        .to_owned();
    for r in 0..d {
        s.push_str(&format!("var x{r}=seed[{r}u32];"));
    }
    if prefix {
        s.push_str(&format!("var out=filled_u32(count*{d}u32,0u32);"));
    }
    s.push_str("for i in 0u32..count {");
    for r in 0..d {
        let mut dot = "0u32".to_owned();
        for c in 0..d {
            dot = format!(
                "wrapping_add({dot},wrapping_mul(coeff[i*{}u32+{}u32],x{c}))",
                d * d,
                r * d + c
            );
        }
        s.push_str(&format!("let y{r}={dot};"));
    }
    for r in 0..d {
        s.push_str(&format!("x{r}=y{r};"));
    }
    if prefix {
        for r in 0..d {
            s.push_str(&format!("out[i*{d}u32+{r}u32]=x{r};"));
        }
    }
    s.push_str("} return ");
    if prefix {
        s.push_str("out;");
    } else {
        s.push_str(&format!(
            "[{}];",
            (0..d)
                .map(|r| format!("x{r}"))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    s.push('}');
    s
}

fn stencil(clamp: bool) -> String {
    let mut s="fn run(a: read Array<f32>, rows: u32, cols: u32) -> Array<f32> { var out=filled_f32(rows*cols,0.0); for i in 0u32..rows { for j in 0u32..cols {".to_owned();
    for (n, cond, ix) in [
        ("north", "i>0u32", "(i-1u32)*cols+j"),
        ("south", "i+1u32<rows", "(i+1u32)*cols+j"),
        ("west", "j>0u32", "i*cols+(j-1u32)"),
        ("east", "j+1u32<cols", "i*cols+(j+1u32)"),
    ] {
        s.push_str(&format!("var {n}=0.0; if {cond} {{ {n}=a[{ix}]; }}"));
        if clamp {
            s.push_str(&format!("else {{ {n}=a[i*cols+j]; }}"));
        }
    }
    s.push_str("out[i*cols+j]=(((a[i*cols+j]+north)+south)+west)+east; } } return out; }");
    s
}

fn focused() -> Vec<(String, String)> {
    let mut cases = vec![
        ("map", MAP),
        ("reduce", REDUCE),
        ("scan", SCAN),
        ("gather", GATHER),
        ("scatter", SCATTER),
        ("histogram", HISTOGRAM),
        ("affine", AFFINE),
        ("transpose", TRANSPOSE),
        ("tensor", TENSOR),
        ("direct", DIRECT),
        ("segmented", SEGMENTED),
        ("segmented-map", SEGMENTED_MAP),
        ("table", TABLE),
        ("compact", COMPACT),
        ("fused", FUSED),
    ]
    .into_iter()
    .map(|(id, s)| (format!("family/{id}"), s.to_owned()))
    .collect::<Vec<_>>();
    for d in 1..=4 {
        for prefix in [false, true] {
            cases.push((format!("family/matrix-{d}-{prefix}"), matrix(d, prefix)));
        }
    }
    for clamp in [false, true] {
        cases.push((format!("family/stencil-{clamp}"), stencil(clamp)));
    }
    // The frontier observer's resource budget is a live refusal, not a static
    // source admission bound. Zero-trip and larger shapes still match.
    for n in [0, 5, 17] {
        cases.push((
            format!("family/frontier-{n}"),
            include_str!("../../bench/general-net-v01/current/frontier_graph_a.inv")
                .replace("5u32", &format!("{n}u32")),
        ));
    }
    // Review finding: A0's ordered affine-pair reduction is a Reduce
    // admission, and a pair over two coefficient arrays is refused by both.
    let pair = include_str!("../../examples/pair-reduction.inv");
    cases.push(("examples/pair-reduction".into(), pair.into()));
    cases.push((
        "family/pair-reduction-swapped".into(),
        pair.replace(
            "        a = wrapping_mul(aa[i], a);\n        b = wrapping_add(wrapping_mul(aa[i], b), bb[i]);",
            "        b = wrapping_add(bb[i], wrapping_mul(b, aa[i]));\n        a = wrapping_mul(a, aa[i]);",
        ),
    ));
    cases.push((
        "family/pair-reduction-two-coefficients".into(),
        pair.replace("wrapping_mul(aa[i], b)", "wrapping_mul(bb[i], b)"),
    ));
    cases.push((
        "family/fused-literal".into(),
        FUSED.replace("len(a)", "4u32").replace("len(b)", "4u32"),
    ));
    cases.push(("family/dag-three".into(), "fn run(a: read Array<u32>) -> Array<u32> { var x=filled_u32(4u32,0u32); for i in 0u32..4u32 { x[i]=wrapping_add(a[i],1u32); } let b=move x; var y=filled_u32(4u32,0u32); for j in 0u32..4u32 { y[j]=wrapping_mul(b[j],2u32); } let c=move y; var out=filled_u32(4u32,0u32); for k in 0u32..4u32 { out[k]=wrapping_add(c[k],3u32); } return out; }".into()));
    for exclusive in [false, true] {
        let store = "out[i]=s;";
        cases.push((format!("family/scan-prefix-{exclusive}"),format!("fn run(a: read Array<u32>, n: u32, seed: u32) -> Array<u32> {{ var s=seed; var out=filled_u32(n,0u32); for i in 0u32..n {{ {} s=wrapping_add(wrapping_mul(s,3u32),a[i]); {} }} return out; }}", if exclusive { store } else { "" },if exclusive { "" } else { store })));
    }
    for (id, source) in [
        (
            "map-checked",
            MAP.replace("wrapping_add(a[i],3u32)", "a[i]+3u32"),
        ),
        ("map-offset", MAP.replace("out[i]", "out[i+1u32]")),
        (
            "map-constant-first",
            MAP.replace("wrapping_add(a[i],3u32)", "wrapping_add(3u32,a[i])"),
        ),
        (
            "reduce-commuted",
            REDUCE.replace("wrapping_add(s,a[i])", "wrapping_add(a[i],s)"),
        ),
        (
            "reduce-checked",
            REDUCE.replace("wrapping_add(s,a[i])", "s+a[i]"),
        ),
        (
            "scan-commuted",
            SCAN.replace(
                "wrapping_add(wrapping_mul(s,3u32),a[i])",
                "wrapping_add(a[i],wrapping_mul(3u32,s))",
            ),
        ),
        (
            "scan-nonlinear",
            SCAN.replace("wrapping_mul(s,3u32)", "wrapping_mul(s,s)"),
        ),
        (
            "scan-mixed-checked",
            SCAN.replace("wrapping_mul(s,3u32)", "s*3u32"),
        ),
        (
            "transpose-wrong-layout",
            TRANSPOSE.replace("out[j*rows+i]", "out[i*cols+j]"),
        ),
        (
            "transpose-u32",
            TRANSPOSE.replace("f32", "u32").replace("0.0", "0u32"),
        ),
        (
            "tensor-wrong-product",
            TENSOR.replace("a[i*inner+k]*b[k*cols+j]", "b[k*cols+j]*a[i*inner+k]"),
        ),
        (
            "segmented-product",
            SEGMENTED
                .replace("acc=0u32", "acc=1u32")
                .replace("wrapping_add", "wrapping_mul"),
        ),
        (
            "segmented-wrong-seed",
            SEGMENTED.replace("acc=0u32", "acc=1u32"),
        ),
        (
            "histogram-commuted",
            HISTOGRAM.replace(
                "wrapping_add(out[ix[i]],a[i])",
                "wrapping_add(a[i],out[ix[i]])",
            ),
        ),
        (
            "histogram-checked",
            HISTOGRAM.replace("wrapping_add(out[ix[i]],a[i])", "out[ix[i]]+a[i]"),
        ),
        (
            "affine-singleton",
            AFFINE
                .replace("0u32..3u32", "0u32..1u32")
                .replace("2u32*i", "0u32*i"),
        ),
        (
            "affine-empty",
            AFFINE
                .replace("0u32..3u32", "2u32..1u32")
                .replace("2u32*i", "0u32*i"),
        ),
        ("affine-collision", AFFINE.replace("2u32*i", "0u32*i")),
        ("affine-out-of-bounds", AFFINE.replace("7u32", "2u32")),
        (
            "compact-mismatched-predicate",
            COMPACT.replace("a[j]>cutoff", "a[j]>=cutoff"),
        ),
        ("table-too-wide", TABLE.replace("4u32", "17u32")),
    ] {
        cases.push((format!("near-miss/{id}"), source));
    }
    for (id, s) in cases.clone() {
        cases.push((
            format!("{id}-renamed"),
            s.replace("run(", "renamed(").replace("out", "answer"),
        ));
    }
    cases
}

#[test]
fn corpus_agreement_with_existing_recognizers() {
    let mut corpus = focused();
    let mut paths = Vec::new();
    files(&support::gnet05().join("programs"), &mut paths);
    files(&support::gnet05().join("robotics/programs"), &mut paths);
    // 72 generated GNET-05 programs and 48 robotics programs.
    assert_eq!(paths.len(), 120);
    for p in paths {
        corpus.push((
            p.strip_prefix(root()).unwrap().display().to_string(),
            fs::read_to_string(p).unwrap(),
        ));
    }
    let proto: Json = serde_json::from_str(
        &fs::read_to_string(root().join("tests/net_normal_form_fixtures/prototype.json")).unwrap(),
    )
    .unwrap();
    for row in proto["m501"].as_array().unwrap() {
        corpus.push((
            format!("m501/{}", row["id"].as_str().unwrap()),
            row["source"].as_str().unwrap().to_owned(),
        ));
    }
    let mut differences = Vec::new();
    let mut loops = 0;
    let mut refusals = Vec::new();
    for (id, source) in &corpus {
        let (diff, n, refused) = compare(id, source);
        differences.extend(diff);
        loops += n;
        refusals.extend(refused);
    }
    let report = json!({"comparison":"admission-or-explicit-native-refusal","programs":corpus.len(),"loops":loops,"family_predicates":Family::ALL.len(),"unavailable_comparisons":["segmented_device::MapContract", "device u32 stencil", "explicit Tensor Core source contracts"],"disagreements":differences,"native_refusals":refusals});
    if let Ok(path) = std::env::var("INVAR_P3_REPORT") {
        fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    }
    println!(
        "programs={} loops={loops} disagreements={} native_refusals={}",
        corpus.len(),
        differences.len(),
        refusals.len()
    );
    for d in &differences {
        println!(
            "{} {} {} PNF={} recognizer={} {}",
            d["id"], d["function"], d["family"], d["pnf"], d["recognizer"], d["error"]
        );
    }
    assert!(
        differences.is_empty(),
        "{} PNF/recognizer disagreements; see INVAR_P3_REPORT for complete programs",
        differences.len()
    );
}

fn report(source: &str) -> family::Report {
    let (net, proof) = support::compile(source);
    let form = normal_form::extract(&net, &proof).unwrap();
    family::analyze(&net, &proof, &form).unwrap()
}
