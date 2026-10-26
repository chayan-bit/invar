//! Physical Metal qualification of the law-collective emitters; the CPU tests stay in `tests/law_collective/net_law_collective.rs`.
#[path = "../support/law_collective.rs"]
mod law_support;
use invar::eval;
use invar::general_net::{
    law_collective::{Layout, Representation, msl},
    normal_form::Rep,
};
use law_support as ls;
use ls::oracle_result;

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
