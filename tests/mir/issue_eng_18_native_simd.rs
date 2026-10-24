#![cfg(feature = "native")]
//! ENG-18 (#101): the O3 pipeline, CPU target selection and the
//! proof-carrying promise rule of the native backend.

use invar::{
    check::check,
    front::parse,
    mir,
    native::{self, Build, Cpu, Fact, IntervalOp, Operand, Promise},
};

#[path = "../support/scratch.rs"]
mod scratch;

fn lower(source: &str) -> mir::Program {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    mir::lower(&checked).unwrap()
}

/// Exit status and stdout of `program` built as `build` and run with `fuel`.
fn run(program: &mir::Program, build: &Build, fuel: u64) -> (Option<i32>, String) {
    let object = native::emit_mir_for(program, build).unwrap();
    let directory = std::env::temp_dir().join(format!("invar-eng-18-{}", scratch::unique()));
    std::fs::create_dir(&directory).unwrap();
    let executable = directory.join("program");
    native::link(&object, &executable).unwrap();
    let output = native::run(&executable, fuel).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    (
        output.status.code(),
        String::from_utf8(output.stdout).unwrap(),
    )
}

const CORPUS: &[(&str, u64)] = &[
    (include_str!("../../examples/sum.inv"), 10_000),
    (include_str!("../../examples/ownership.inv"), 10_000),
    (include_str!("../../examples/fk_panda.inv"), 1_000_000),
    ("fn main() -> i32 { return 2147483647i32 + 1i32; }", 10_000),
    (
        "fn main() -> i32 { var x: i32 = 0; for i in 0u32 .. 1000u32 { x = x + 3; } return x; }",
        100_000,
    ),
    (
        "fn main() -> i32 { var x: i32 = 1; while true { x = x * 3; } return x; }",
        100_000,
    ),
    ("fn main() -> i32 { while true {} return 0; }", 64),
    (
        "fn main() -> i32 { let xs: Array<i32> = filled_i32(64u32, 5); var s: i32 = 0; for i in 0u32 .. 65u32 { s = s + xs[i]; } return s; }",
        10_000,
    ),
    (
        "fn main() -> i32 { var xs = filled_f64(32u32, 0.1); var s: f64 = 0.0; for i in 0u32 .. 32u32 { s = s + xs[i] * 3.0; } if s == 9.600000000000003 { return 1; } else { return 0; } }",
        10_000,
    ),
    (
        "fn main() -> i32 { var s: u32 = 7u32; for i in 0u32 .. 300u32 { s = wrapping_mul(s, 2654435761u32); s = wrapping_add(s, i); } return 0; }",
        10_000,
    ),
];

/// Optimization and target features never change an observation: status
/// and the whole result document (values, traps, spans, fuel, cleanup)
/// match the unoptimized generic build, which issue_m1_02 ties to the
/// reference evaluator.
#[test]
fn issue_eng_18_optimized_builds_observe_identically() {
    let builds = [Build::UNOPTIMIZED, Build::PORTABLE, Build::HOST];
    for (source, fuel) in CORPUS {
        let program = lower(source);
        let baseline = run(&program, &builds[0], *fuel);
        for build in &builds[1..] {
            assert_eq!(run(&program, build, *fuel), baseline, "{build:?}: {source}");
        }
    }
}

#[test]
fn issue_eng_18_objects_record_machine_and_pipeline() {
    let program = lower(include_str!("../../examples/sum.inv"));
    let portable = native::emit_mir(&program).unwrap();
    assert_eq!(portable.machine.cpu, "generic");
    assert_eq!(portable.machine.features, "");
    assert_eq!(portable.machine.pipeline, Some(native::PIPELINE));
    assert!(portable.optimized_ir.is_some());
    // The MIR lowering gives LLVM no promise at all.
    assert!(portable.facts.is_empty());
    let unoptimized = native::emit_mir_for(&program, &Build::UNOPTIMIZED).unwrap();
    assert_eq!(unoptimized.machine.pipeline, None);
    assert!(unoptimized.optimized_ir.is_none());
    assert_eq!(
        unoptimized.ir, portable.ir,
        "the audited IR does not depend on the build"
    );
    let host = native::emit_mir_for(&program, &Build::HOST).unwrap();
    eprintln!("host machine: {:?}", host.machine);
    // A known live CPU names its description; LLVM may report its features
    // (x86_64) or let the CPU name imply them (Apple silicon).
    let described = match std::env::consts::ARCH {
        "x86_64" if host.machine.cpu == "znver4" => Some("cpu-x86_64-zen4"),
        "aarch64" if host.machine.cpu == "apple-m4" => Some("cpu-aarch64-apple-m4"),
        _ => None,
    };
    if described.is_some() {
        assert_eq!(
            host.machine.target.as_deref(),
            described,
            "{:?}",
            host.machine
        );
    }
    if host.machine.cpu == "generic" {
        assert_eq!(host.machine.features, "");
    }
}

fn triple_arch() -> &'static str {
    std::env::consts::ARCH
}

#[test]
fn issue_eng_18_described_targets_select_cpu_and_features() {
    let (name, other) = match triple_arch() {
        "x86_64" => ("cpu-x86_64-zen4", "cpu-aarch64-apple-m4"),
        "aarch64" => ("cpu-aarch64-apple-m4", "cpu-x86_64-zen4"),
        arch => panic!("no described CPU for {arch}"),
    };
    let triple = format!("{}-unknown-linux-gnu", triple_arch());
    let machine = native::resolve(&Cpu::Described(name.into()), &triple).unwrap();
    assert_eq!(machine.target.as_deref(), Some(name));
    if triple_arch() == "x86_64" {
        assert_eq!(machine.cpu, "znver4");
        for feature in ["+avx2", "+fma", "+avx512f", "+avx512bw", "+avx512vl"] {
            assert!(
                machine.features.split(',').any(|f| f == feature),
                "{machine:?}"
            );
        }
    } else {
        assert_eq!(machine.cpu, "apple-m4");
        for feature in ["+neon", "+dotprod", "+fullfp16"] {
            assert!(
                machine.features.split(',').any(|f| f == feature),
                "{machine:?}"
            );
        }
    }
    let wrong = native::resolve(&Cpu::Described(other.into()), &triple).unwrap_err();
    assert!(wrong.contains("host triple"), "{wrong}");
    let unknown = native::resolve(&Cpu::Described("cpu-x86_64-none".into()), &triple).unwrap_err();
    assert!(unknown.contains("unknown native CPU target"), "{unknown}");
    let generic = native::resolve(&Cpu::Generic, &triple).unwrap();
    assert_eq!(
        (generic.cpu.as_str(), generic.features.as_str()),
        ("generic", "")
    );
}

/// `out[i] = in[i] * 3 + 1` over `n` u32 elements: the shape a lowered map
/// region takes. `promised` decides whether the pointers and accesses carry
/// promises (each tied to fact `!100`, `!101` or `!102`).
fn map_kernel(promised: bool) -> String {
    let flag = |word: &str| {
        if promised {
            format!("{word} ")
        } else {
            String::new()
        }
    };
    let tag = |id: u32| {
        if promised {
            format!(", !invar.fact !{id}")
        } else {
            String::new()
        }
    };
    let (noalias, inbounds, nuw) = (flag("noalias"), flag("inbounds"), flag("nuw"));
    let (bounds, step, out_bounds) = (tag(101), tag(102), tag(103));
    let disjoint = if promised { " !invar.fact !100" } else { "" };
    let nodes = if promised {
        native::fact_nodes(&facts(), 100)
    } else {
        String::new()
    };
    format!(
        "define void @map(ptr {noalias}%in, ptr {noalias}%out, i64 %n){disjoint} {{
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %done, label %body
body:
  %i = phi i64 [ 0, %entry ], [ %next, %body ]
  %src = getelementptr {inbounds}i32, ptr %in, i64 %i{bounds}
  %x = load i32, ptr %src
  %y = mul i32 %x, 3
  %z = add i32 %y, 1
  %dst = getelementptr {inbounds}i32, ptr %out, i64 %i{out_bounds}
  store i32 %z, ptr %dst
  %next = add {nuw}i64 %i, 1{step}
  %more = icmp ult i64 %next, %n
  br i1 %more, label %body, label %done
done:
  ret void
}}
{nodes}"
    )
}

fn bounds(base: &str) -> Fact {
    let elements = invar::array_limits::MAX_ARRAY_LEN as u64;
    Fact::InBounds {
        base: base.into(),
        index: operand("%i", 0..=i128::from(elements) - 1),
        elements,
        guard: "caller: n <= len(xs) and n <= len(out)".into(),
    }
}

fn operand(value: &str, range: std::ops::RangeInclusive<i128>) -> Operand {
    Operand {
        value: value.into(),
        range,
    }
}

fn step(lhs: std::ops::RangeInclusive<i128>) -> Fact {
    Fact::Interval {
        op: IntervalOp::Add,
        bits: 64,
        lhs: operand("%i", lhs),
        rhs: operand("1", 1..=1),
    }
}

fn facts() -> Vec<Fact> {
    vec![
        Fact::Disjoint {
            params: vec!["%in".into(), "%out".into()],
            owners: vec!["xs".into(), "out".into()],
        },
        bounds("%in"),
        step(0..=invar::array_limits::MAX_ARRAY_LEN as i128),
        bounds("%out"),
    ]
}

#[test]
fn issue_eng_18_promise_audit_maps_every_promise_to_a_fact() {
    let ir = map_kernel(true);
    let promised = native::audit(&ir, &facts()).unwrap();
    let count = |kind| promised.iter().filter(|(p, _)| *p == kind).count();
    assert_eq!(count(Promise::NoAlias), 2);
    assert_eq!(count(Promise::InBounds), 2);
    assert_eq!(count(Promise::Nuw), 1);
    let (object, _) = native::compile_ir(&ir, facts(), &Build::PORTABLE).unwrap();
    assert_eq!(object.facts, facts());
}

#[test]
fn issue_eng_18_planted_unproven_promises_fail() {
    let ir = map_kernel(true);
    // A promise with no fact attached.
    let planted = ir.replace("%y = mul i32 %x, 3", "%y = mul nsw i32 %x, 3");
    let error = native::audit(&planted, &facts()).unwrap_err();
    assert!(error.contains("without a fact"), "{error}");
    let error = native::compile_ir(&planted, facts(), &Build::PORTABLE).unwrap_err();
    assert!(error.contains("without a fact"), "{error}");
    // A fact that does not prove the promise: 0..=2^64-1 plus 1 overflows u64.
    let mut wrong = facts();
    wrong[2] = step(0..=u64::MAX as i128);
    let error = native::audit(&ir, &wrong).unwrap_err();
    assert!(error.contains("does not prove Nuw"), "{error}");
    // Corners that overflow i128 itself are refused, never wrapped.
    let huge = ir.replace("%next = add nuw i64 %i, 1", "%next = mul nuw i64 %i, %i");
    let mut overflowing = facts();
    overflowing[2] = Fact::Interval {
        op: IntervalOp::Mul,
        bits: 64,
        lhs: operand("%i", 0..=u64::MAX as i128),
        rhs: operand("%i", 0..=u64::MAX as i128),
    };
    let error = native::audit(&huge, &overflowing).unwrap_err();
    assert!(error.contains("does not prove Nuw"), "{error}");
    // Facts are bound to their site: another opcode, width or operand, a
    // constant outside its range, another GEP base or another noalias
    // parameter is refused even when the fact itself is consistent.
    for (from, to) in [
        ("%next = add nuw i64 %i, 1", "%next = mul nuw i64 %i, 1"),
        ("%next = add nuw i64 %i, 1", "%next = add nuw i64 %n, 1"),
        ("%next = add nuw i64 %i, 1", "%next = add nuw i64 %i, 2"),
        (
            "ptr %in, i64 %i, !invar.fact !101",
            "ptr %out, i64 %i, !invar.fact !101",
        ),
        (
            "ptr %out, i64 %i, !invar.fact !103",
            "ptr %in, i64 %i, !invar.fact !103",
        ),
        (
            "ptr noalias %out, i64 %n",
            "ptr noalias %out, ptr noalias %other, i64 %n",
        ),
        // A return-position noalias names no parameter the fact covers.
        ("define void @map(", "define noalias ptr @map("),
        // The bounds fact names the access's only index, never another one.
        (
            "ptr %in, i64 %i, !invar.fact !101",
            "ptr %in, i64 4294967296, !invar.fact !101",
        ),
        (
            "ptr %out, i64 %i, !invar.fact !103",
            "ptr %out, i64 %n, !invar.fact !103",
        ),
    ] {
        let moved = ir.replace(from, to);
        assert_ne!(moved, ir, "{from}");
        let error = native::audit(&moved, &facts()).unwrap_err();
        assert!(
            error.contains("covers")
                || error.contains("does not name")
                || error.contains("outside"),
            "{from}: {error}"
        );
    }
    let mut narrow = facts();
    narrow[2] = Fact::Interval {
        op: IntervalOp::Add,
        bits: 8,
        lhs: operand("%i", 0..=1),
        rhs: operand("1", 1..=1),
    };
    let error = native::audit(&ir, &narrow).unwrap_err();
    assert!(error.contains("does not name"), "{error}");
    // An index range that reaches past the element bound proves nothing.
    let mut past = facts();
    if let Fact::InBounds {
        index, elements, ..
    } = &mut past[1]
    {
        index.range = 0..=i128::from(*elements);
    }
    let error = native::audit(&ir, &past).unwrap_err();
    assert!(error.contains("outside"), "{error}");
    // A fact of the wrong form, and a fact index nobody recorded.
    let error = native::audit(&ir, &facts()[..3]).unwrap_err();
    assert!(error.contains("unrecorded fact"), "{error}");
    let mut swapped = facts();
    swapped.swap(0, 1);
    assert!(native::audit(&ir, &swapped).is_err());
    // Aliasing owners never prove noalias.
    let mut same = facts();
    same[0] = Fact::Disjoint {
        params: vec!["%in".into(), "%out".into()],
        owners: vec!["xs".into(), "xs".into()],
    };
    assert!(
        native::audit(&ir, &same)
            .unwrap_err()
            .contains("distinct owners")
    );
    // No fact form covers fast-math or termination promises; strict never
    // takes contract or reassoc.
    for (from, to) in [
        (
            "%y = mul i32 %x, 3",
            "%y = mul i32 %x, 3\n  %f = fadd contract float 1.0, 2.0",
        ),
        (
            "%y = mul i32 %x, 3",
            "%y = mul i32 %x, 3\n  %f = fadd reassoc float 1.0, 2.0",
        ),
        (
            "%y = mul i32 %x, 3",
            "%y = mul i32 %x, 3\n  %f = fadd fast float 1.0, 2.0",
        ),
        ("%z = add i32 %y, 1", "%z = or disjoint i32 %y, 1"),
        (
            "%y = mul i32 %x, 3",
            "%y = mul i32 %x, 3\n  %e = udiv exact i32 %x, 3",
        ),
        (
            "define void @map(",
            "define void @map(ptr nonnull %unused, ",
        ),
        (
            "%x = load i32, ptr %src",
            "%x = load i32, ptr %src, !range !100",
        ),
        (
            "%x = load i32, ptr %src",
            "%x = load i32, ptr %src, !noundef !100",
        ),
        (
            "%x = load i32, ptr %src",
            "%x = load i32, ptr %src, !nonnull !100",
        ),
        (
            "br i1 %more, label %body, label %done",
            "br i1 %more, label %body, label %done, !llvm.loop !100",
        ),
    ] {
        let planted = ir.replace(from, to);
        assert_ne!(planted, ir);
        let error = native::audit(&planted, &facts()).unwrap_err();
        assert!(error.contains("unproven promise"), "{error}");
    }
    // Names, strings and comments that spell a promise are not promises.
    let named = map_kernel(false)
        .replace("%z", "%nsw")
        .replace("ret void", "ret void ; nuw fast");
    assert!(native::audit(&named, &[]).unwrap().is_empty());
}

/// Vector instructions in the optimized IR and the target assembly: ymm or
/// zmm registers on x86_64, NEON q-form arrangements on aarch64.
fn assert_vectorized(object: &native::Object, assembly: &str) {
    let optimized = object.optimized_ir.as_deref().unwrap();
    assert!(
        optimized.contains("x i32>"),
        "no vector type after {}: {optimized}",
        native::PIPELINE
    );
    let vector_registers = match triple_arch() {
        "x86_64" => assembly.contains("%ymm") || assembly.contains("%zmm"),
        _ => assembly.contains(".4s"),
    };
    assert!(vector_registers, "{:?}\n{assembly}", object.machine);
}

#[test]
fn issue_eng_18_promised_map_vectorizes_for_the_live_cpu() {
    let (object, assembly) = native::compile_ir(&map_kernel(true), facts(), &Build::HOST).unwrap();
    eprintln!("machine: {:?}", object.machine);
    if object.machine.cpu == "generic" {
        eprintln!("unknown host CPU: vectorization evidence not available here");
        return;
    }
    assert_vectorized(&object, &assembly);
}

#[test]
fn issue_eng_18_described_target_vectorizes_the_map() {
    let name = match triple_arch() {
        "x86_64" => "cpu-x86_64-zen4",
        _ => "cpu-aarch64-apple-m4",
    };
    let build = Build {
        cpu: Cpu::Described(name.into()),
        optimize: true,
    };
    let (object, assembly) = native::compile_ir(&map_kernel(true), facts(), &build).unwrap();
    assert_vectorized(&object, &assembly);
    // Without the unoptimized path's middle end, nothing is vectorized.
    let (plain, _) = native::compile_ir(&map_kernel(true), facts(), &Build::UNOPTIMIZED).unwrap();
    assert!(plain.optimized_ir.is_none());
    assert!(!plain.ir.contains("x i32>"));
}
