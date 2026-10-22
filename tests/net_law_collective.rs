#![forbid(unsafe_code)]

use invar::general_net::{
    law_collective::{Error, LawRepr, Layout, Representation, cuda, msl},
    normal_form::{self, Rep},
};

const SUM: &str = "fn f(xs: read Array<u32>) -> u32 { var s = 0u32; for i in 0u32..len(xs) { s = wrapping_add(s, xs[i]); } return s; }";

#[test]
fn p8_certificate_boundary_and_word_layout() {
    let (net, proof) = ls::compile(SUM);
    let form = normal_form::extract(&net, &proof).unwrap();
    normal_form::certificate::check_all(&net, &proof, &form).unwrap();
    let certificate = form
        .certificates()
        .find(|c| matches!(c.rep, Some(Rep::Sealed(_))))
        .unwrap();
    let repr = LawRepr::from_certificate(&net, &proof, certificate).unwrap();
    assert_eq!(repr.word_count(), 1);
    assert!(matches!(
        repr.representation(),
        Representation::Carried(Rep::Sealed(_))
    ));
    let mut forged = certificate.clone();
    forged.fingerprint[0] ^= 1;
    assert!(matches!(
        LawRepr::from_certificate(&net, &proof, &forged),
        Err(Error::Certificate(_))
    ));
    for emit in [
        msl::emit_reduce,
        msl::emit_scan,
        cuda::emit_reduce,
        cuda::emit_scan,
    ] {
        assert_eq!(
            emit(
                &repr,
                Layout {
                    lanes: 3,
                    items_per_lane: 1
                }
            ),
            Err(Error::Layout)
        );
    }
}

#[path = "support/law_collective.rs"]
mod law_support;
use invar::eval::{self, EvalError};
use law_support as ls;

use ls::oracle_result;

#[test]
fn p8_l10_l11_refusal_selects_executable_native_fallback() {
    use invar::general_net::{compiled, normal_form::LawId, oracle};
    use std::{fs, process::Command};
    // The L10 tree succeeds where a checked left fold of its leaves traps:
    // (MAX + 0) + (1 + -1). L11 instead must trap at its second same-target
    // writer even though the final mathematical sum is representable.
    let cases = [
        (
            LawId::L10RangeSum,
            "fn f(lo:u32,hi:u32)->i32 {if hi-lo<=1u32 {if lo==0u32{return 2147483647;} if lo==2u32{return 1;} if lo==3u32{return -1;} return 0;} let mid=lo+(hi-lo)/2u32; let a=f(lo,mid); let b=f(mid,hi); return a+b;} fn main()->i32{return f(0u32,4u32);}",
            false,
        ),
        (
            LawId::L10RangeSum,
            "fn f(lo:u32,hi:u32)->i32 {if hi-lo<=1u32 {if lo==0u32{return 2147483647;} if lo==1u32{return 1;} if lo==2u32{return -1;} return 0;} let mid=lo+(hi-lo)/2u32; let a=f(lo,mid); let b=f(mid,hi); return a+b;} fn main()->i32{return f(0u32,4u32);}",
            true,
        ),
        (
            LawId::L11ScatterCombine,
            "fn f(h:mut Array<i32>,ix:read Array<u32>,w:read Array<i32>,n:u32,m:u32)->Unit {for i in 0u32..n {h[ix[i]%m]=h[ix[i]%m]+w[i];}} fn main()->i32 {var h:Array<i32> = [0]; let ix:Array<u32> = [0u32,0u32,0u32]; let w:Array<i32> = [2147483647,1,-1]; f(h,ix,w,3u32,1u32); return h[0u32];}",
            true,
        ),
        (
            LawId::L11ScatterCombine,
            "fn f(h:mut Array<u32>,ix:read Array<u32>,w:read Array<u32>,n:u32,m:u32)->Unit {for i in 0u32..n {h[ix[i]%m]=wrapping_add(h[ix[i]%m],w[i]);}} fn main()->i32 {var h:Array<u32> = [9u32]; let ix:Array<u32> = [0u32,0u32,0u32]; let w:Array<u32> = [4294967295u32,1u32,3u32]; f(h,ix,w,3u32,1u32); if h[0u32]==12u32{return 12;} return 0;}",
            false,
        ),
    ];
    for (i, (law, source, traps)) in cases.into_iter().enumerate() {
        let (net, proof) = ls::compile(source);
        let form = normal_form::extract(&net, &proof).unwrap();
        normal_form::certificate::check_all(&net, &proof, &form).unwrap();
        let cert = form.certificates().find(|c| c.law == law).unwrap();
        let native = match LawRepr::from_certificate(&net, &proof, cert) {
            Err(Error::Unsupported(reason)) => {
                assert!(reason.starts_with(if law == LawId::L10RangeSum {
                    "L10 native fallback:"
                } else {
                    "L11 native fallback:"
                }));
                compiled::emit_owned(
                    &net,
                    &proof,
                    "main",
                    compiled::Target::Cpu,
                    compiled::Limits::default(),
                    compiled::OwnedLimits {
                        elements: 64,
                        allocations: 16,
                        cleanup_events: 32,
                    },
                )
                .unwrap()
            }
            other => panic!("must select native fallback before a collective launch: {other:?}"),
        };
        let reference =
            oracle::observe_call(&net, &proof, "main", &[], oracle::Limits::default()).unwrap();
        assert_eq!(reference.semantic.result.is_err(), traps);
        let layout = native.owned_layout().unwrap();
        let dir = std::env::temp_dir().join(format!("p8-native-{}-{i}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let driver = format!(
            r#"
#include <iostream>
#include <vector>
int main() {{gnet_w input[1]={{}}, out[10]={{}}; std::vector<gnet_w> heap({heap}); std::vector<gnet_u> cleanup({clean}),trace(700000);
gnet_run(input,out,trace.data(),100000,1000000,100000,heap.data(),cleanup.data());
for(auto x:out) std::cout<<x<<' ';}}
"#,
            heap = layout.heap_words(),
            clean = layout.cleanup_words()
        );
        fs::write(
            dir.join("source.cpp"),
            format!("{}\n{driver}", native.source()),
        )
        .unwrap();
        let build = Command::new("c++")
            .current_dir(&dir)
            .args(["-std=c++17", "-O1", "source.cpp", "-o", "run"])
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
        let run = Command::new(dir.join("run")).output().unwrap();
        assert!(run.status.success());
        let out: Vec<i64> = String::from_utf8(run.stdout)
            .unwrap()
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        match reference.semantic.result {
            Ok(eval::Value::I32(value)) => {
                assert_eq!(out[0], 0);
                assert_eq!(out[1], i64::from(value));
            }
            Err(EvalError::Trap {
                category: eval::TrapCategory::IntegerOverflow,
                span,
            }) => {
                assert_eq!(out[0], 1);
                assert_eq!((out[2], out[3]), (span.start as i64, span.end as i64));
            }
            other => panic!("unexpected oracle {other:?}"),
        }
        fs::remove_dir_all(dir).unwrap();
    }
    println!(
        "P8 L10/L11: 4 explicit native fallback executions match oracle values/first trap spans"
    );
}

#[test]
fn p8_emission_layout_and_resource_boundaries() {
    let r = ls::repr(ls::SUM, |r| matches!(r, Rep::LiftedSum(_)));
    for emit in [
        msl::emit_reduce,
        msl::emit_scan,
        cuda::emit_reduce,
        cuda::emit_scan,
    ] {
        let e = emit(&r, Layout::default()).unwrap();
        assert!(e.source.contains("p8_then"));
        assert_eq!(e.words, 6);
        assert_eq!(e.tile_elements, 1024);
        assert_eq!(e.launch(0).unwrap().groups, 1);
        assert_eq!(e.launch(1025).unwrap().groups, 2);
        assert_eq!(e.launch((1 << 26) + 1), Err(Error::Limit));
        assert_eq!(
            emit(
                &r,
                Layout {
                    lanes: 1024,
                    items_per_lane: 16
                }
            )
            .unwrap()
            .threadgroup_bytes,
            24608
        );
    }
    let e = cuda::emit_scan(&r, Layout::default()).unwrap();
    assert!(e.source.contains("invar_law_locate_v1"));
    assert!(e.launch(1 << 26).is_ok());
}

#[test]
#[ignore = "CPU host only: compile emitted algebra with existing C++ compiler"]
fn p8_emitted_lifted_sum_and_product_match_net_oracle_6000_cases() {
    let mut rng = ls::Rng(0x5048_5343_414e);
    for (source, is_product) in [(ls::SUM, false), (ls::PRODUCT, true)] {
        let repr = ls::repr(source, |r| {
            if is_product {
                matches!(r, Rep::SaturatedProduct(_))
            } else {
                matches!(r, Rep::LiftedSum(_))
            }
        });
        let (net, proof) = ls::compile(source);
        let mut cases = Vec::new();
        let mut raw = Vec::new();
        for case in 0..3000 {
            let n = if case < 4 {
                case
            } else {
                (rng.next() % 24) as usize
            };
            let seed = rng.int();
            let xs: Vec<i32> = (0..n).map(|_| rng.int()).collect();
            let elements = xs
                .iter()
                .map(|&x| {
                    if is_product {
                        ls::product(x)
                    } else {
                        ls::lifted(x)
                    }
                })
                .collect();
            cases.push((ls::seed(repr.word_count(), i64::from(seed)), elements));
            raw.push((seed, xs));
        }
        let cpu = ls::CpuAlgebra::new(&repr, None);
        let output = cpu.run(&cases);
        let mut traps = 0;
        assert_eq!(output.len(), 3000);
        for ((seed, xs), row) in raw.iter().zip(&output) {
            let args = [
                ls::array_i32(xs),
                eval::Value::U32(xs.len() as u32),
                eval::Value::I32(*seed),
            ];
            let expected = oracle_result(&net, &proof, &args);
            let mut state = *seed;
            let mut first = xs.len();
            for (i, x) in xs.iter().enumerate() {
                match if is_product {
                    state.checked_mul(*x)
                } else {
                    state.checked_add(*x)
                } {
                    Some(s) => state = s,
                    None => {
                        first = i;
                        break;
                    }
                }
            }
            let w = repr.word_count() as usize;
            match expected {
                Ok(eval::Value::I32(value)) => {
                    assert_eq!(ls::read_wide(row), i64::from(value));
                    assert_eq!(first, xs.len());
                }
                Err(EvalError::Trap { .. }) => {
                    traps += 1;
                    assert!(first < xs.len());
                    assert_eq!(row[w], 0);
                    assert!((row[w + 1] as usize) <= first);
                }
                other => panic!("unexpected oracle {other:?}"),
            }
            if !is_product {
                assert_eq!(row[w + 1] as usize, first);
            }
        }
        assert!(traps > 500);
        println!(
            "P8 {}: 3000 oracle cases, {traps} traps",
            if is_product { "product" } else { "sum" }
        );
    }
}

#[test]
#[ignore = "CPU host only: compile emitted algebra with existing C++ compiler"]
fn p8_emitted_affine_residue_and_strict_f32_select_match_net_oracle() {
    let mut rng = ls::Rng(0x5048_4146_4649);
    for residue in [false, true] {
        let source = if residue { ls::RESIDUE } else { ls::AFFINE };
        let repr = ls::repr(source, |r| matches!(r, Rep::Affine(_)));
        let (net, proof) = ls::compile(source);
        let mut cases = Vec::new();
        let mut args = Vec::new();
        for _ in 0..1000 {
            let n = (rng.next() % 20) as usize;
            let seed = rng.next() % 20000;
            let a: Vec<u32> = (0..n).map(|_| rng.next() % 4).collect();
            let b: Vec<u32> = (0..n).map(|_| rng.next() % 10).collect();
            cases.push((
                vec![seed, 0],
                a.iter().zip(&b).map(|(&a, &b)| vec![a, b]).collect(),
            ));
            args.push(if residue {
                vec![
                    ls::array_i32(&a.iter().map(|&x| x as i32).collect::<Vec<_>>()),
                    ls::array_i32(&b.iter().map(|&x| x as i32).collect::<Vec<_>>()),
                    eval::Value::U32(n as u32),
                    eval::Value::I32(seed as i32),
                ]
            } else {
                vec![
                    ls::array_u32(&a),
                    ls::array_u32(&b),
                    eval::Value::U32(n as u32),
                    eval::Value::U32(seed),
                ]
            });
        }
        let cpu = ls::CpuAlgebra::new(&repr, None);
        for (row, args) in cpu.run(&cases).iter().zip(args) {
            let expected = oracle_result(&net, &proof, &args).unwrap();
            let value = match expected {
                eval::Value::I32(x) => x as u32,
                eval::Value::U32(x) => x,
                _ => panic!("scalar"),
            };
            assert_eq!(row[0], value);
        }
    }
    let repr = ls::repr(ls::SELECT, |r| matches!(r, Rep::Select { .. }));
    let (net, proof) = ls::compile(ls::SELECT);
    let values = [
        0u32, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f800001,
        0x3f800000, 0xbf800000,
    ];
    let mut cases = Vec::new();
    let mut inputs = Vec::new();
    for _ in 0..1000 {
        let n = (rng.next() % 24) as usize;
        let seed = values[rng.next() as usize % values.len()];
        let xs: Vec<u32> = (0..n)
            .map(|_| values[rng.next() as usize % values.len()])
            .collect();
        cases.push((
            vec![1, seed, u32::MAX],
            xs.iter()
                .enumerate()
                .map(|(i, &x)| vec![1, x, i as u32])
                .collect(),
        ));
        inputs.push((seed, xs));
    }
    let cpu = ls::CpuAlgebra::new(&repr, None);
    for (row, (seed, xs)) in cpu.run(&cases).iter().zip(inputs) {
        use invar::numeric_contract::F32Bits;
        let args = [
            eval::Value::ArrayF32(std::rc::Rc::new(std::cell::RefCell::new(
                xs.iter().map(|&x| F32Bits::from_bits(x)).collect(),
            ))),
            eval::Value::U32(xs.len() as u32),
            eval::Value::F32(F32Bits::from_bits(seed)),
        ];
        assert_eq!(
            oracle_result(&net, &proof, &args).unwrap(),
            eval::Value::U32(row[2])
        );
        let mut best = seed;
        for x in xs {
            if f32::from_bits(x) < f32::from_bits(best) {
                best = x;
            }
        }
        assert_eq!(
            row[1], best,
            "NaN seed, payloads and signed-zero ties preserve bits"
        );
    }
    println!("P8 affine/residue/select: 3000 oracle cases");
}

#[test]
#[ignore = "CPU host only: compile emitted algebra with existing C++ compiler"]
fn p8_named_mutations_are_killed() {
    let sum = ls::repr(ls::SUM, |r| matches!(r, Rep::LiftedSum(_)));
    let cases = vec![(
        ls::seed(6, 0),
        vec![
            ls::lifted(i32::MAX),
            ls::lifted(1),
            ls::lifted(-1),
            ls::lifted(1),
        ],
    )];
    let good = ls::CpuAlgebra::new(&sum, None).run(&cases);
    for (name, from, to) in [
        (
            "drop-prefix-high",
            "p8_smax(p8_signed(p8_get(a,4u)),p8_signed(at+p8_get(b,4u)))",
            "0l",
        ),
        ("drop-guard", "return ((low)", "return true || ((low)"),
        ("last-trap", "first=i;break;", "first=i;"),
    ] {
        let bad = ls::CpuAlgebra::new(&sum, Some((from, to))).run(&cases);
        assert_ne!(good, bad, "survived {name}");
    }
    let affine = ls::repr(ls::AFFINE, |r| matches!(r, Rep::Affine(_)));
    let cases = vec![(vec![7, 0], vec![vec![2, 3], vec![5, 11]])];
    let good = ls::CpuAlgebra::new(&affine, None).run(&cases);
    let bad = ls::CpuAlgebra::new(&affine, Some(("r.w[0u]=uint(1ul);", "r.w[0u]=uint(0ul);")))
        .run(&cases);
    assert_ne!(good, bad, "wrong-affine-identity survived");
    println!("P8 killed: drop-prefix-high, drop-guard, last-trap, wrong-affine-identity");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "physical Metal qualification"]
fn p8_metal_ordered_tiles_ragged_lengths_and_trap_rare_path() {
    let mut device =
        invar_metal::MetalDevice::open_runtime_source().expect("physical Metal required");
    let affine = ls::repr(ls::AFFINE, |r| matches!(r, Rep::Affine(_)));
    let sum = ls::repr(ls::SUM, |r| matches!(r, Rep::LiftedSum(_)));
    let select = ls::repr(ls::SELECT, |r| matches!(r, Rep::Select { .. }));
    let lengths = [0u32, 1, 31, 32, 33, 127, 128, 129, 1023, 1024, 1025, 4097];
    let layout = Layout {
        lanes: 32,
        items_per_lane: 4,
    };
    let mut checked = 0;
    for layout in ls::HARDWARE_LAYOUTS {
        for repr in [&affine, &sum, &select] {
            let source = if repr == &affine {
                ls::AFFINE
            } else if repr == &sum {
                ls::SUM
            } else {
                ls::SELECT
            };
            let (net, proof) = ls::compile(source);
            for n in lengths {
                let w = repr.word_count() as usize;
                let mut input = Vec::new();
                let mut expected = Vec::new();
                let mut a = 1u32;
                let mut b = 0u32;
                let mut total = 0i64;
                let mut best = f32::INFINITY.to_bits();
                let mut at = u32::MAX;
                for i in 0..n {
                    match repr.representation() {
                        Representation::Carried(Rep::Affine(_)) => {
                            expected.extend([a, b]);
                            let ai = (i % 5) * 2 + 1;
                            let bi = i.wrapping_mul(13);
                            input.extend([ai, bi]);
                            a = ai.wrapping_mul(a);
                            b = ai.wrapping_mul(b).wrapping_add(bi);
                        }
                        Representation::Carried(Rep::LiftedSum(_)) => {
                            expected
                                .extend([ls::wide(total), ls::wide(0), ls::wide(total)].concat());
                            let x = (i % 7) as i32;
                            input.extend(ls::lifted(x));
                            total += i64::from(x);
                        }
                        Representation::Carried(Rep::Select { .. }) => {
                            expected.extend([1, best, at]);
                            let x = match i % 7 {
                                0 => 0x7fc01234,
                                1 => 0x80000000,
                                2 => 0,
                                3 => 1,
                                4 => 0x80000001,
                                _ => (i as f32).to_bits(),
                            };
                            input.extend([1, x, i]);
                            if f32::from_bits(x) < f32::from_bits(best) {
                                best = x;
                                at = i;
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                let e = msl::emit_scan(repr, layout).unwrap();
                let plan = e.launch(n).unwrap();
                let input = if input.is_empty() { vec![0; w] } else { input };
                let out = vec![0xcdcdcdcd; plan.output_words.max(w)];
                let table = vec![0; plan.table_words.max(1)];
                let (result, _) = device
                    .run_law_source(
                        &e.source,
                        e.entry,
                        [&input, &out, &[n], &table, &table, &[1]],
                        plan.groups,
                        layout.lanes,
                    )
                    .unwrap();
                if repr == &select {
                    // Invalid identities and all-NaN prefixes carry no candidate. Apply once
                    // to the source seed; do not compare irrelevant invalid payload words.
                    for (i, row) in result[1][..n as usize * w].chunks_exact(w).enumerate() {
                        let gold = &expected[i * w..(i + 1) * w];
                        let value = if row[0] == 0 {
                            f32::INFINITY.to_bits()
                        } else {
                            row[1]
                        };
                        let index = if row[0] == 0 { u32::MAX } else { row[2] };
                        assert_eq!((value, index), (gold[1], gold[2]), "select n={n} i={i}");
                    }
                } else {
                    assert_eq!(&result[1][..n as usize * w], expected, "scan n={n}");
                }
                if n >= 129 {
                    let adversarial = e
                        .source
                        .replace("#define P8_REVERSE 0u", "#define P8_REVERSE 1u")
                        .replace("#define P8_USE_PREFIXES 1u", "#define P8_USE_PREFIXES 0u")
                        .replace("#define P8_POLLS 128u", "#define P8_POLLS 0u");
                    let (fallback, _) = device
                        .run_law_source(
                            &adversarial,
                            e.entry,
                            [&input, &out, &[n], &table, &table, &[1]],
                            plan.groups,
                            layout.lanes,
                        )
                        .unwrap();
                    // The complete word stream is schedule-invariant for these inputs.
                    assert_eq!(fallback[1], result[1], "reverse/fallback n={n}");
                }
                // The last published prefix is an inclusive total, held in packed atomics.
                // Repeated block reduction exercises a different group tree on the same data.
                let mut partial = input.clone();
                let mut length = n;
                loop {
                    let e = msl::emit_reduce(repr, layout).unwrap();
                    let p = e.launch(length).unwrap();
                    let output = vec![0; p.output_words];
                    let (r, _) = device
                        .run_law_source(
                            &e.source,
                            e.entry,
                            [&partial, &output, &[length], &[0], &[0], &[1]],
                            p.groups,
                            layout.lanes,
                        )
                        .unwrap();
                    partial = r[1].clone();
                    length = p.groups;
                    if length == 1 {
                        break;
                    }
                }
                match repr.representation() {
                    Representation::Carried(Rep::Affine(_)) => assert_eq!(&partial[..2], &[a, b]),
                    Representation::Carried(Rep::LiftedSum(_)) => {
                        assert_eq!(ls::read_wide(&partial), total)
                    }
                    Representation::Carried(Rep::Select { .. }) => {
                        let index = if partial[0] == 0 {
                            u32::MAX
                        } else {
                            partial[2]
                        };
                        assert_eq!(index, at);
                    }
                    _ => unreachable!(),
                }
                let args = if repr == &affine {
                    let a: Vec<u32> = input
                        .chunks_exact(w)
                        .take(n as usize)
                        .map(|x| x[0])
                        .collect();
                    let b: Vec<u32> = input
                        .chunks_exact(w)
                        .take(n as usize)
                        .map(|x| x[1])
                        .collect();
                    vec![
                        ls::array_u32(&a),
                        ls::array_u32(&b),
                        eval::Value::U32(n),
                        eval::Value::U32(0),
                    ]
                } else if repr == &sum {
                    let xs: Vec<i32> = input
                        .chunks_exact(w)
                        .take(n as usize)
                        .map(|x| ls::read_wide(x) as i32)
                        .collect();
                    vec![ls::array_i32(&xs), eval::Value::U32(n), eval::Value::I32(0)]
                } else {
                    use invar::numeric_contract::F32Bits;
                    let xs = input
                        .chunks_exact(w)
                        .take(n as usize)
                        .map(|x| F32Bits::from_bits(x[1]))
                        .collect();
                    vec![
                        eval::Value::ArrayF32(std::rc::Rc::new(std::cell::RefCell::new(xs))),
                        eval::Value::U32(n),
                        eval::Value::F32(F32Bits::from_bits(f32::INFINITY.to_bits())),
                    ]
                };
                let actual = if repr == &affine {
                    eval::Value::U32(partial[1])
                } else if repr == &sum {
                    eval::Value::I32(ls::read_wide(&partial) as i32)
                } else {
                    eval::Value::U32(if partial[0] == 0 {
                        u32::MAX
                    } else {
                        partial[2]
                    })
                };
                assert_eq!(
                    oracle_result(&net, &proof, &args).unwrap(),
                    actual,
                    "Metal vs net oracle n={n}"
                );
                checked += 1;
            }
        }
    }
    // Each fast-path mutation must change the D5-layout output.
    let (input, expected) = ls::affine_case(4097);
    for (name, from, to) in ls::FAST_PATH_MUTATIONS {
        let scan = name != "reduce-order";
        let emit = if scan {
            msl::emit_scan
        } else {
            msl::emit_reduce
        };
        let e = emit(&affine, Layout::default()).unwrap();
        let plan = e.launch(4097).unwrap();
        let mutant = e.source.replace(from, to);
        assert_ne!(mutant, e.source, "{name} target missing");
        let table = vec![0; plan.table_words.max(1)];
        let out = vec![0; plan.output_words];
        let mut run = |source: &str| {
            device
                .run_law_source(
                    source,
                    e.entry,
                    [&input, &out, &[4097], &table, &table, &[1]],
                    plan.groups,
                    256,
                )
                .unwrap()
                .0[1]
                .clone()
        };
        let good = run(&e.source);
        if scan {
            assert_eq!(good, expected, "{name} baseline");
        }
        assert_ne!(run(&mutant), good, "{name} survived");
    }
    // A reduction total can recover in range after an earlier overflowing prefix.
    let xs = [i32::MAX, 1, -1, 0];
    let input: Vec<u32> = xs.into_iter().flat_map(ls::lifted).collect();
    let e = msl::emit_scan(&sum, layout).unwrap();
    let plan = e.launch(4).unwrap();
    let out = vec![0; 24];
    let table = vec![0; plan.table_words];
    let (r, _) = device
        .run_law_source(
            &e.source,
            e.entry,
            [&input, &out, &[4], &table, &table, &[1]],
            plan.groups,
            layout.lanes,
        )
        .unwrap();
    let (located, _) = device
        .run_law_source(
            &e.source,
            "invar_law_locate_v1",
            [&input, &r[1], &[4], &ls::seed(6, 0), &[4], &[1]],
            1,
            32,
        )
        .unwrap();
    assert_eq!(
        located[4][0], 1,
        "first source-order violation, not recovered final total or last violation"
    );
    let last_index_source = e.source.replace(
        "atomic_fetch_min_explicit(earliest,i,memory_order_relaxed)",
        "atomic_fetch_max_explicit(earliest,i,memory_order_relaxed)",
    );
    assert_ne!(last_index_source, e.source, "GPU mutation target missing");
    let (last_index, _) = device
        .run_law_source(
            &last_index_source,
            "invar_law_locate_v1",
            [&input, &r[1], &[4], &ls::seed(6, 0), &[0], &[1]],
            1,
            32,
        )
        .unwrap();
    assert_eq!(last_index[4][0], 3);
    assert_ne!(
        last_index[4][0], located[4][0],
        "last-source-index GPU mutation survived"
    );
    // A failed guard performs no output or look-back publication.
    let (failed, _) = device
        .run_law_source(
            &e.source,
            e.entry,
            [&input, &[77; 24], &[4], &table, &table, &[0]],
            plan.groups,
            32,
        )
        .unwrap();
    assert_eq!(failed[1], vec![77; 24]);
    assert_eq!(failed[3], table);
    assert_eq!(failed[4], table);
    println!(
        "P8 Metal: {checked} net-oracle scan+reduce cases over 3 layouts, 4 fast-path mutations, trap-first-index, last-index mutation and guard-no-publication passed"
    );
}

#[test]
#[ignore = "CPU host only: compile emitted algebra with existing C++ compiler"]
fn p8_checked_affine_matrix_and_tropical_prefixes_replay_exactly() {
    use invar::general_net::normal_form::Ring;
    const MATRIX: &str = "fn f(cs: read Array<i32>, n: u32, a0: i32, b0: i32) -> i32 {var a=a0; var b=b0; for i in 0u32..n {let na=a+b*cs[i]; let nb=a-b; a=na; b=nb;} return a;}";
    const CHECKED: &str = "fn f(xs: read Array<i32>, ys: read Array<i32>, n: u32, seed: i32) -> i32 {var s=seed; for i in 0u32..n {s=s*xs[i]+ys[i];} return s;}";
    let mut rng = ls::Rng(0x4d4154524958);
    for (kind, source) in [(0, CHECKED), (1, MATRIX), (2, ls::TROPICAL)] {
        let repr = ls::repr(source, |r| {
            matches!(
                (kind, r),
                (0, Rep::Affine(Ring::Checked64(_)))
                    | (1, Rep::Matrix { .. })
                    | (2, Rep::Tropical { .. })
            )
        });
        assert_eq!(
            repr.trap_recovery(),
            invar::general_net::law_collective::TrapRecovery::ReplayEveryLane
        );
        let w = repr.word_count() as usize;
        let mut cases = Vec::new();
        let mut expected = Vec::new();
        let mut prefix_gold = Vec::new();
        let mut traps = 0;
        let (net, proof) = ls::compile(source);
        for _ in 0..1000 {
            let n = (rng.next() % 20) as usize;
            let s0 = rng.int();
            let b0 = rng.int();
            let xs: Vec<i32> = (0..n).map(|_| rng.int()).collect();
            let ys: Vec<i32> = (0..n).map(|_| rng.int()).collect();
            let mut s = s0;
            let mut b = b0;
            let mut gold = Vec::new();
            let mut failed = false;
            for (&x, &y) in xs.iter().zip(&ys) {
                if failed {
                    gold.push(None);
                    continue;
                }
                gold.push(Some((s, b)));
                let step = match kind {
                    0 => s
                        .checked_mul(x)
                        .and_then(|v| v.checked_add(y))
                        .map(|v| (v, b)),
                    1 => b
                        .checked_mul(x)
                        .and_then(|v| s.checked_add(v))
                        .zip(s.checked_sub(b)),
                    _ => s.checked_add(x).map(|v| (v.max(0), b)),
                };
                if let Some((ns, nb)) = step {
                    s = ns;
                    b = nb;
                } else {
                    failed = true;
                }
            }
            let args = match kind {
                0 => vec![
                    ls::array_i32(&xs),
                    ls::array_i32(&ys),
                    eval::Value::U32(n as u32),
                    eval::Value::I32(s0),
                ],
                1 => vec![
                    ls::array_i32(&xs),
                    eval::Value::U32(n as u32),
                    eval::Value::I32(s0),
                    eval::Value::I32(b0),
                ],
                _ => vec![
                    ls::array_i32(&xs),
                    eval::Value::U32(n as u32),
                    eval::Value::I32(s0),
                ],
            };
            let result = oracle_result(&net, &proof, &args);
            if failed {
                assert!(matches!(result, Err(EvalError::Trap { .. })));
                traps += 1;
            } else {
                assert_eq!(result.unwrap(), eval::Value::I32(s));
            }
            expected.push((failed, s));
            prefix_gold.push(gold);
            let mut seed = ls::seed(w as u32, i64::from(s0));
            if kind == 1 {
                seed[2..4].copy_from_slice(&ls::wide(i64::from(b0)));
            }
            let elements = xs
                .iter()
                .zip(&ys)
                .map(|(&x, &y)| match kind {
                    0 => [ls::wide(i64::from(x)), ls::wide(i64::from(y))].concat(),
                    1 => [1i64, i64::from(x), 0, 1, -1, 0, 0, 0, 1]
                        .into_iter()
                        .flat_map(ls::wide)
                        .collect(),
                    _ => [ls::wide(i64::from(x)), ls::wide(0), vec![1, 1]].concat(),
                })
                .collect();
            cases.push((seed, elements));
        }
        let cpu = ls::CpuAlgebra::new(&repr, None);
        let output = cpu.run(&cases);
        let clean: Vec<bool> = expected.iter().map(|(failed, _)| !*failed).collect();
        if kind == 1 || kind == 2 {
            let (name, from, to) = if kind == 1 {
                (
                    "wrong-matrix-identity",
                    "r=p8_put(r,0u,1ul);",
                    "r=p8_put(r,0u,0ul);",
                )
            } else {
                ("drop-tropical-shift", "r.w[4]=1u;", "r.w[4]=0u;")
            };
            let bad = ls::CpuAlgebra::new(&repr, Some((from, to))).run(&cases);
            assert!(
                output
                    .iter()
                    .zip(&bad)
                    .zip(&clean)
                    .any(|((good, bad), clean)| *clean && good[..2] != bad[..2]),
                "survived {name}"
            );
            println!("P8 killed {name} on an oracle-success case");
        }
        for (((row, (_, elements)), (failed, value)), gold) in
            output.iter().zip(&cases).zip(expected).zip(prefix_gold)
        {
            if !failed {
                assert_eq!(ls::read_wide(row), i64::from(value));
            }
            let start = w + 2 + elements.len() * w;
            for (i, state) in gold.into_iter().enumerate() {
                if let Some((s, b)) = state {
                    let applied = &row[start + i * w..start + (i + 1) * w];
                    assert_eq!(
                        ls::read_wide(applied),
                        i64::from(s),
                        "kind={kind}, prefix={i}"
                    );
                    if kind == 1 {
                        assert_eq!(ls::read_wide(&applied[2..]), i64::from(b));
                    }
                }
            }
        }
        println!(
            "P8 checked kind={kind}: 1000 oracle cases, {traps} traps, exact prefix seeds before first trap"
        );
        assert!(traps > 100);
    }
}

#[test]
#[ignore = "CPU host only: compile emitted algebra with existing C++ compiler"]
fn p8_l9_path_weighted_residue_reduce_and_nan_mutations() {
    use invar::general_net::normal_form::LawId;
    const SOURCE: &str = "fn f(n: i32, v: i32) -> i32 {if n<=0 {return v;} let a=f(n-1,v); let b=f(n-1,v+1); return (a*2+b*3+1)%10007;}";
    let (net, proof) = ls::compile(SOURCE);
    let form = normal_form::extract(&net, &proof).unwrap();
    normal_form::certificate::check_all(&net, &proof, &form).unwrap();
    let cert = form
        .certificates()
        .find(|c| c.law == LawId::L9PathSum)
        .unwrap();
    let repr = LawRepr::from_certificate(&net, &proof, cert).unwrap();
    assert_eq!(
        repr.representation(),
        &Representation::ResidueSum { modulus: 10007 }
    );
    assert_eq!(repr.guards().len(), 3);
    fn leaves(n: i32, v: i32, weight: u32, out: &mut Vec<Vec<u32>>) {
        if n == 0 {
            out.push(vec![(v as u32 * weight) % 10007]);
        } else {
            out.push(vec![weight]);
            leaves(n - 1, v, (weight * 2) % 10007, out);
            leaves(n - 1, v + 1, (weight * 3) % 10007, out);
        }
    }
    let mut cases = Vec::new();
    let mut values = Vec::new();
    for j in 0..300 {
        let n = j % 8;
        let v = j % 101;
        let mut xs = Vec::new();
        leaves(n, v, 1, &mut xs);
        cases.push((vec![0], xs));
        let value =
            oracle_result(&net, &proof, &[eval::Value::I32(n), eval::Value::I32(v)]).unwrap();
        values.push(value);
    }
    let cpu = ls::CpuAlgebra::new(&repr, None);
    let gold = cpu.run(&cases);
    for (row, value) in gold.iter().zip(values) {
        assert_eq!(eval::Value::I32(row[0] as i32), value);
    }
    let bad = ls::CpuAlgebra::new(&repr, Some(("%10007ul", "%10009ul"))).run(&cases);
    assert!(
        gold.iter().zip(&bad).any(|(g, b)| g[0] != b[0]),
        "wrong-residue-modulus survived"
    );
    let select = ls::repr(ls::SELECT, |r| matches!(r, Rep::Select { .. }));
    let cases = vec![(vec![1, 0x7f800000, u32::MAX], vec![vec![1, 0xffc01234, 0]])];
    let good = ls::CpuAlgebra::new(&select, None).run(&cases);
    let bad = ls::CpuAlgebra::new(&select, Some(("if(p8_nan(b.w[1])) b.w[0]=0u;", ""))).run(&cases);
    assert_ne!(good[0][1], bad[0][1], "drop-NaN-lift survived");
    let product = ls::repr(ls::PRODUCT, |r| matches!(r, Rep::SaturatedProduct(_)));
    let cases = vec![(
        ls::seed(4, 1),
        vec![ls::product(i32::MAX), ls::product(2), ls::product(0)],
    )];
    let good = ls::CpuAlgebra::new(&product, None).run(&cases);
    let bad = ls::CpuAlgebra::new(
        &product,
        Some((
            "p8_max(p8_get(a,2u),p8_satmul(p8_abs(x),p8_get(b,2u)))",
            "1ul",
        )),
    )
    .run(&cases);
    assert_eq!(good[0][4], 0);
    assert_eq!(bad[0][4], 1, "drop-product-prefix-bound survived");
    println!(
        "P8 L9: 300 oracle cases; killed wrong-residue-modulus, drop-NaN-lift, drop-product-prefix-bound"
    );
}

#[test]
#[ignore = "CPU host only: compile emitted algebra with existing C++ compiler"]
fn p8_sealed_laws_match_oracle_and_identity_mutations() {
    let mut rng = ls::Rng(0x5048_5345_414c);
    for product in [false, true] {
        let op = if product {
            "wrapping_mul"
        } else {
            "wrapping_add"
        };
        let source = format!(
            "fn f(xs: read Array<u32>, n: u32, seed: u32) -> u32 {{var s=seed;for i in 0u32..n {{s={op}(s,xs[i]);}}return s;}}"
        );
        let repr = ls::repr(&source, |r| matches!(r, Rep::Sealed(_)));
        let (net, proof) = ls::compile(&source);
        let mut cases = Vec::new();
        let mut expected = Vec::new();
        for j in 0..100 {
            let n = j % 20;
            let seed = rng.next();
            let xs: Vec<u32> = (0..n).map(|_| rng.next()).collect();
            cases.push((vec![seed], xs.iter().map(|&x| vec![x]).collect()));
            expected.push(
                oracle_result(
                    &net,
                    &proof,
                    &[
                        ls::array_u32(&xs),
                        eval::Value::U32(n),
                        eval::Value::U32(seed),
                    ],
                )
                .unwrap(),
            );
        }
        let good = ls::CpuAlgebra::new(&repr, None).run(&cases);
        for (row, expected) in good.iter().zip(expected) {
            assert_eq!(eval::Value::U32(row[0]), expected);
        }
        let (from, to) = if product {
            ("r.w[0]=1u;", "r.w[0]=0u;")
        } else {
            ("r.w[0]=0u;", "r.w[0]=1u;")
        };
        let bad = ls::CpuAlgebra::new(&repr, Some((from, to))).run(&cases);
        assert!(
            good.iter().zip(&bad).any(|(g, b)| g[0] != b[0]),
            "wrong-sealed-identity survived product={product}"
        );
    }
    println!("P8 sealed: 200 oracle cases, wrong-sum-identity and wrong-product-identity killed");
}

#[test]
fn p8_residue_recurrence_replays_checked_arithmetic_before_remainder() {
    use invar::general_net::law_collective::TrapRecovery;
    let repr = ls::repr(ls::RESIDUE, |r| matches!(r, Rep::Affine(_)));
    let (net, proof) = ls::compile(ls::RESIDUE);
    // Every residue guard holds, but seed*coefficient overflows before `% P`.
    let args = [
        ls::array_i32(&[i32::MAX]),
        ls::array_i32(&[1]),
        eval::Value::U32(1),
        eval::Value::I32(2),
    ];
    assert!(matches!(
        oracle_result(&net, &proof, &args),
        Err(EvalError::Trap { .. })
    ));
    assert_eq!(repr.trap_recovery(), TrapRecovery::ReplayEveryLane);
}

#[test]
fn p8_normalized_fold_keeps_hidden_checked_intermediates() {
    use invar::general_net::law_collective::TrapRecovery;
    const SOURCE: &str = "fn f(xs: read Array<i32>, n: u32, seed: i32) -> i32 {var s=seed; for i in 0u32..n {s=s*2-s+xs[i];} return s;}";
    let repr = ls::repr(SOURCE, |r| matches!(r, Rep::LiftedSum(_)));
    let (net, proof) = ls::compile(SOURCE);
    // Algebraically an addition by zero, but the original multiply traps.
    let args = [
        ls::array_i32(&[0]),
        eval::Value::U32(1),
        eval::Value::I32(1 << 30),
    ];
    assert!(matches!(
        oracle_result(&net, &proof, &args),
        Err(EvalError::Trap { .. })
    ));
    assert_eq!(repr.trap_recovery(), TrapRecovery::ReplayEveryLane);
    assert_eq!(
        ls::repr(ls::SUM, |r| matches!(r, Rep::LiftedSum(_))).trap_recovery(),
        TrapRecovery::PrefixBounds
    );
}
