#![forbid(unsafe_code)]
#[path = "../support/law_collective.rs"]
mod support;
use invar::{
    eval,
    general_net::{
        law_collective::{Layout, cuda},
        normal_form::Rep,
    },
};
use invar_cuda::cuda::CudaDevice;
use std::path::Path;

#[test]
#[ignore = "physical CUDA and NVRTC; run under the host's GPU lock"]
fn p8_cuda_scan_reduce_ragged_and_ordered() {
    let compiler = Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let mut device = CudaDevice::open().expect("physical CUDA and compatible NVRTC required");
    println!("P8 CUDA: {} {:?}", device.name(), device.capabilities());
    let lengths = [0u32, 1, 31, 32, 33, 127, 128, 129, 1023, 1024, 1025, 4097];
    let mut checked = 0;
    for layout in support::HARDWARE_LAYOUTS {
        for affine in [false, true] {
            let source = if affine {
                support::AFFINE
            } else {
                support::SUM
            };
            let repr = support::repr(source, |r| {
                if affine {
                    matches!(r, Rep::Affine(_))
                } else {
                    matches!(r, Rep::LiftedSum(_))
                }
            });
            let (net, proof) = support::compile(source);
            let w = repr.word_count() as usize;
            for n in lengths {
                let mut a = 1u32;
                let mut b = 0u32;
                let mut total = 0i64;
                let mut input = Vec::new();
                let mut expected = Vec::new();
                for i in 0..n {
                    if affine {
                        expected.extend([a, b]);
                        let ai = (i % 5) * 2 + 1;
                        let bi = i.wrapping_mul(13);
                        input.extend([ai, bi]);
                        a = ai.wrapping_mul(a);
                        b = ai.wrapping_mul(b).wrapping_add(bi);
                    } else {
                        expected.extend(
                            [support::wide(total), support::wide(0), support::wide(total)].concat(),
                        );
                        let x = (i % 7) as i32;
                        input.extend(support::lifted(x));
                        total += i64::from(x);
                    }
                }
                let args = if affine {
                    let a: Vec<u32> = input.chunks_exact(w).map(|x| x[0]).collect();
                    let b: Vec<u32> = input.chunks_exact(w).map(|x| x[1]).collect();
                    vec![
                        support::array_u32(&a),
                        support::array_u32(&b),
                        eval::Value::U32(n),
                        eval::Value::U32(0),
                    ]
                } else {
                    let xs: Vec<i32> = input
                        .chunks_exact(w)
                        .map(|x| support::read_wide(x) as i32)
                        .collect();
                    vec![
                        support::array_i32(&xs),
                        eval::Value::U32(n),
                        eval::Value::I32(0),
                    ]
                };
                let value = if affine {
                    eval::Value::U32(b)
                } else {
                    eval::Value::I32(total as i32)
                };
                assert_eq!(support::oracle_result(&net, &proof, &args).unwrap(), value);
                let e = cuda::emit_scan(&repr, layout).unwrap();
                let plan = e.launch(n).unwrap();
                let input = if input.is_empty() { vec![0; w] } else { input };
                let output = vec![0; plan.output_words.max(w)];
                let table = vec![0; plan.table_words.max(1)];
                let result = device
                    .run_law_source(
                        &e.source,
                        e.entry,
                        [&input, &output, &[n], &table, &table, &[1]],
                        (plan.groups, layout.lanes),
                        compiler,
                    )
                    .unwrap();
                assert_eq!(
                    &result[1][..n as usize * w],
                    expected,
                    "CUDA scan affine={affine} n={n}"
                );
                if n >= 129 {
                    let reversed = e
                        .source
                        .replace("#define P8_REVERSE 0u", "#define P8_REVERSE 1u")
                        .replace("#define P8_USE_PREFIXES 1u", "#define P8_USE_PREFIXES 0u")
                        .replace("#define P8_POLLS 128u", "#define P8_POLLS 0u");
                    let fallback = device
                        .run_law_source(
                            &reversed,
                            e.entry,
                            [&input, &output, &[n], &table, &table, &[1]],
                            (plan.groups, layout.lanes),
                            compiler,
                        )
                        .unwrap();
                    assert_eq!(fallback[1], result[1], "CUDA forced fallback");
                }
                let mut partial = input.clone();
                let mut count = n;
                loop {
                    let e = cuda::emit_reduce(&repr, layout).unwrap();
                    let p = e.launch(count).unwrap();
                    let out = vec![0; p.output_words];
                    let r = device
                        .run_law_source(
                            &e.source,
                            e.entry,
                            [&partial, &out, &[count], &[0], &[0], &[1]],
                            (p.groups, layout.lanes),
                            compiler,
                        )
                        .unwrap();
                    partial = r[1].clone();
                    count = p.groups;
                    if count == 1 {
                        break;
                    }
                }
                if affine {
                    assert_eq!(&partial[..2], &[a, b]);
                } else {
                    assert_eq!(support::read_wide(&partial), total);
                }
                checked += 1;
            }
        }
    }
    // Each fast-path mutation must change the D5-layout output.
    let affine = support::repr(support::AFFINE, |r| matches!(r, Rep::Affine(_)));
    let (input, expected) = support::affine_case(4097);
    for (name, from, to) in support::FAST_PATH_MUTATIONS {
        let scan = name != "reduce-order";
        let emit = if scan {
            cuda::emit_scan
        } else {
            cuda::emit_reduce
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
                    (plan.groups, 256),
                    compiler,
                )
                .unwrap()[1]
                .clone()
        };
        let good = run(&e.source);
        if scan {
            assert_eq!(good, expected, "{name} baseline");
        }
        assert_ne!(run(&mutant), good, "{name} survived");
    }
    let layout = Layout {
        lanes: 32,
        items_per_lane: 4,
    };
    let sum = support::repr(support::SUM, |r| matches!(r, Rep::LiftedSum(_)));
    let input: Vec<u32> = [i32::MAX, 1, -1, 0]
        .into_iter()
        .flat_map(support::lifted)
        .collect();
    let e = cuda::emit_scan(&sum, layout).unwrap();
    let p = e.launch(4).unwrap();
    let out = vec![0; 24];
    let table = vec![0; p.table_words];
    let seed = support::seed(6, 0);
    let r = device
        .run_law_source(
            &e.source,
            e.entry,
            [&input, &out, &[4], &table, &table, &[1]],
            (p.groups, 32),
            compiler,
        )
        .unwrap();
    let located = device
        .run_law_source(
            &e.source,
            "invar_law_locate_v1",
            [&input, &r[1], &[4], &seed, &[4], &[1]],
            (1, 32),
            compiler,
        )
        .unwrap();
    assert_eq!(located[4][0], 1);
    let mutant = e
        .source
        .replace("atomicMin(earliest,i)", "atomicMax(earliest,i)");
    assert_ne!(mutant, e.source);
    let last = device
        .run_law_source(
            &mutant,
            "invar_law_locate_v1",
            [&input, &r[1], &[4], &seed, &[0], &[1]],
            (1, 32),
            compiler,
        )
        .unwrap();
    assert_eq!(last[4][0], 3);
    assert_ne!(last[4][0], located[4][0]);
    let failed = device
        .run_law_source(
            &e.source,
            e.entry,
            [&input, &[77; 24], &[4], &table, &table, &[0]],
            (p.groups, 32),
            compiler,
        )
        .unwrap();
    assert_eq!(failed[1], vec![77; 24]);
    assert_eq!(failed[3], table);
    assert_eq!(failed[4], table);
    println!(
        "P8 CUDA: {checked} net-oracle scan+reduce cases over 3 layouts, reverse fallback, 4 fast-path mutations, trap-first-index, last-index mutation and guard-no-publication passed"
    );
}
