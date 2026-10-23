//! Physical Metal law-collective pilot, sessions, warm path and calibration; the CPU statistics stay in `tests/net_law_collective_bench.rs`.
#[path = "../support/law_bench.rs"]
mod bench;
#[path = "../support/law_collective.rs"]
mod support;
#[test]
#[ignore = "physical Metal baseline correctness pilot"]
fn p8_d5_metal_pilot() {
    let mut device = invar_metal::MetalDevice::open_runtime_source().unwrap();
    for kind in 0..4 {
        let sources = bench::sources(kind, true);
        let (input, expected) = bench::data(kind, 1025);
        device
            .benchmark_law_pair(
                [&sources[0], &sources[1]],
                if kind >= 2 {
                    "invar_law_scan_v1"
                } else {
                    "invar_law_reduce_v1"
                },
                &invar_metal::HostPages::copy_from(&input).unwrap(),
                &expected,
                bench::repr(kind).word_count(),
                1025,
                kind >= 2,
                None,
            )
            .unwrap();
    }
}

#[test]
#[ignore = "physical Metal, foreground and exclusive, pre-registered independent sessions"]
fn p8_d5_metal_session() {
    let mut device = invar_metal::MetalDevice::open_runtime_source().unwrap();
    println!("P8 D5 Metal {} {:?}", device.name(), device.capabilities());
    bench::session(
        true,
        |sources, entry, input, expected, words, n, scan, native| {
            // The host-resident input lives in caller-owned pages (untimed);
            // the generated copy is freed so the host holds one input copy.
            let host = invar_metal::HostPages::copy_from(&input).map_err(|e| format!("{e:?}"))?;
            drop(input);
            device
                .benchmark_law_pair(sources, entry, &host, expected, words, n, scan, native)
                .map_err(|e| format!("{e:?}"))
        },
    );
}

#[test]
#[ignore = "physical Metal: the no-copy warm path is exact and checked"]
fn p8_metal_zero_copy_warm_is_exact_and_mutation_is_killed() {
    let mut device = invar_metal::MetalDevice::open_runtime_source().unwrap();
    for kind in 0..4 {
        let sources = bench::sources(kind, true);
        let kernels = sources
            .iter()
            .map(|source| (source.clone(), invar::metal_aot::msl_entry_points(source)))
            .collect::<Vec<_>>();
        let bytes = invar::metal_aot::build::build(
            invar::metal_aot::Route::Run,
            "p8-zero-copy",
            b"P8 zero-copy qualification",
            &kernels,
        )
        .unwrap();
        invar::metal_aot::decode(&bytes)
            .unwrap()
            .install(&mut device)
            .unwrap();
        let (input, expected) = bench::data(kind, 1 << 14);
        let host = invar_metal::HostPages::copy_from(&input).unwrap();
        assert_eq!(&*host, &input[..]);
        let mut run = |mutation| {
            device.benchmark_law_pair_with(
                [&sources[0], &sources[1]],
                if kind >= 2 {
                    "invar_law_scan_v1"
                } else {
                    "invar_law_reduce_v1"
                },
                &host,
                &expected,
                [bench::repr(kind).word_count(), 1 << 14],
                kind >= 2,
                None,
                mutation,
            )
        };
        assert_eq!(run(invar_metal::WarmMutation::None).unwrap().len(), 30);
        // Wrapping other pages must be caught: the warm call reads the wrap.
        match run(invar_metal::WarmMutation::ZeroedHost) {
            Err(invar_metal::Error::Device(m)) if m.contains("differs from oracle") => {}
            other => panic!(
                "{} zero-copy mutation survived: {other:?}",
                bench::name(kind)
            ),
        }
    }
    assert_eq!(device.cache_report().source_compiles, 0);
    println!("P8 Metal zero-copy warm path: 4 laws exact, ZeroedHost mutation killed for each");
}

#[test]
#[ignore = "physical Metal, foreground and exclusive, records the cost-model calibration"]
fn p8_calibrate_metal() {
    let mut device = invar_metal::MetalDevice::open_runtime_source().unwrap();
    let name = device.name().to_owned();
    bench::calibrate(true, &name, |words| {
        device
            .calibrate_law_transfer(words)
            .map_err(|e| format!("{e:?}"))
    });
}
