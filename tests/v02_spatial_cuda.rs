#![cfg(feature = "cuda")]
#[path = "support/spatial_oracle.rs"]
#[allow(dead_code)]
mod oracle;
use invar::numeric_contract::F32Bits as F;
use invar_cuda::cuda::{CollectiveFault, CudaDevice};
use invar_device::{sorting, spatial};
use std::path::Path;
fn compiler() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}
fn f(v: f32) -> F {
    F::from_bits(v.to_bits())
}
fn cloud(p: &[[F; 3]]) -> [Vec<F>; 3] {
    std::array::from_fn(|a| p.iter().map(|p| p[a]).collect())
}
fn view(p: &[Vec<F>; 3]) -> oracle::Cloud<'_> {
    oracle::Cloud::new(&p[0], &p[1], &p[2]).unwrap()
}
fn pack(p: &[[F; 3]]) -> Vec<u32> {
    (0..3)
        .flat_map(|a| p.iter().map(move |p| p[a].to_bits()))
        .collect()
}
struct Grid {
    points: Vec<u32>,
    ids: Vec<u32>,
    ranges: Vec<u32>,
}
fn build(device: &mut CudaDevice, points: &[[F; 3]], cell: F) -> Grid {
    let n = points.len();
    let packed = pack(points);
    let mapped = device
        .run_spatial(
            &spatial::Plan::mapping(n as u32, cell.to_bits()).unwrap(),
            [&packed, &[], &[], &[]],
            compiler(),
        )
        .unwrap();
    assert_eq!(mapped.counts.len(), n);
    assert_eq!(mapped.indices.len(), 3 * n);
    let mut ids = Vec::new();
    let mut cell_cache = std::collections::BTreeMap::new();
    for (i, p) in points.iter().enumerate() {
        if p.iter().any(|v| v.is_nan() || v.is_infinite()) {
            assert_eq!(mapped.counts[i], 1);
            continue;
        }
        let expected = *cell_cache.entry(p.map(F::to_bits)).or_insert_with(|| {
            let v = p.map(|v| oracle::cell_index(v, cell).ok());
            Some([v[0]?, v[1]?, v[2]?])
        });
        let Some(expected) = expected else {
            assert_eq!(mapped.counts[i], 2);
            continue;
        };
        assert_eq!(mapped.counts[i], 0);
        for (a, &expected) in expected.iter().enumerate() {
            assert_eq!(
                mapped.indices[a * n + i],
                expected as u32,
                "point {i} axis {a}"
            );
        }
        ids.push(i as u32);
    }
    let mut expected = ids.clone();
    expected.sort_by_key(|&i| {
        std::array::from_fn::<_, 3, _>(|a| mapped.indices[a * n + i as usize] as i32)
    });
    for axis in (0..3).rev() {
        let keys = ids
            .iter()
            .map(|&i| mapped.indices[axis * n + i as usize])
            .collect::<Vec<_>>();
        let result = device
            .run_sort(
                &sorting::Plan::new(ids.len() as u32, sorting::Kind::I32).unwrap(),
                &keys,
                &ids,
                compiler(),
            )
            .unwrap();
        ids = result.payload;
    }
    assert_eq!(ids, expected);
    let m = ids.len();
    let keys = (0..3)
        .flat_map(|a| {
            ids.iter().map({
                let mapped = &mapped;
                move |&i| mapped.indices[a * n + i as usize]
            })
        })
        .collect::<Vec<_>>();
    let boundaries = device
        .run_spatial(
            &spatial::Plan::boundaries(m as u32).unwrap(),
            [&keys, &[], &[], &[]],
            compiler(),
        )
        .unwrap();
    let mut ranges = Vec::new();
    for i in 0..m {
        let start = i == 0 || (0..3).any(|a| keys[a * m + i] != keys[a * m + i - 1]);
        assert_eq!(boundaries.counts[i], u32::from(start));
        if start {
            if i != 0 {
                ranges.push(i as u32);
            }
            ranges.push(i as u32);
        }
    }
    if m != 0 {
        ranges.push(m as u32);
    }
    Grid {
        points: (0..3)
            .flat_map(|a| ids.iter().map(move |&i| points[i as usize][a].to_bits()))
            .collect(),
        ids,
        ranges,
    }
}
fn query(
    device: &mut CudaDevice,
    g: &Grid,
    q: &[[F; 3]],
    r: Option<F>,
    k: u32,
) -> invar_cuda::cuda::SpatialResult {
    let plan = spatial::Plan::query(
        g.ids.len() as u32,
        q.len() as u32,
        (g.ranges.len() / 2) as u32,
        k,
        r.map(F::to_bits),
    )
    .unwrap();
    device
        .run_spatial(&plan, [&g.points, &pack(q), &g.ids, &g.ranges], compiler())
        .unwrap()
}
#[test]
#[ignore = "physical CUDA"]
fn spatial_cuda_raw_strict_oracle() {
    let mut device = CudaDevice::open().expect("physical CUDA required");
    eprintln!("CUDA {} {:?}", device.name(), device.capabilities());
    let points = [
        [f(1.0), f(0.0), f(0.0)],
        [f(-1.0), f(0.0), f(0.0)],
        [F::NEG_ZERO; 3],
        [F::NAN; 3],
        [F::INFINITY; 3],
        [f(f32::MAX), f(0.0), F::NAN],
        [F::from_bits(0x3f800001), F::from_bits(0x39800000), f(0.0)],
        [f(4096.0), f(1.0), f(1.0)],
        [F::from_bits(0x1a800000), f(0.0), f(0.0)],
        [F::from_bits(0x17800000), f(0.0), f(0.0)],
    ];
    let query_pattern = [[f(0.0); 3], [F::NAN; 3], [F::INFINITY; 3], [f(1.0); 3]];
    let queries = (0..129).map(|i| query_pattern[i % 4]).collect::<Vec<_>>();
    let p = cloud(&points);
    let q = cloud(&queries);
    let grid = build(&mut device, &points, f(1.0));
    for radius in [
        f(0.0),
        F::NEG_ZERO,
        f(1.0),
        f(2.0),
        f(-1.0),
        F::NAN,
        F::INFINITY,
        F::NEG_INFINITY,
    ] {
        for k in [0, 1, 4, 12] {
            let (counts, idx) = oracle::radius_neighbors(view(&p), view(&q), radius, k).unwrap();
            let actual = query(&mut device, &grid, &queries, Some(radius), k);
            assert_eq!(actual.counts, counts);
            assert_eq!(actual.indices, idx);
        }
    }
    for k in [0, 1, 4, 12] {
        let (idx, d2) = oracle::knn(view(&p), view(&q), k).unwrap();
        let actual = query(&mut device, &grid, &queries, None, k);
        assert_eq!(actual.indices, idx);
        assert_eq!(
            actual.distances,
            d2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    let largest = f(f32::MAX);
    let extreme_points = [[largest; 3], [largest.negate(); 3]];
    let extreme_queries = [[f(0.0); 3]];
    let extremes = build(&mut device, &extreme_points, largest);
    let (idx, d2) = oracle::knn(
        view(&cloud(&extreme_points)),
        view(&cloud(&extreme_queries)),
        3,
    )
    .unwrap();
    let actual = query(&mut device, &extremes, &extreme_queries, None, 3);
    assert_eq!(actual.indices, idx);
    assert_eq!(
        actual.distances,
        d2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let actual = query(
        &mut device,
        &extremes,
        &extreme_queries,
        Some(F::INFINITY),
        1,
    );
    assert_eq!(actual.counts, [2]);
    assert_eq!(actual.indices, [0]);
    let tiny = build(
        &mut device,
        &[[F::from_bits(0x17800000), f(0.0), f(0.0)]],
        F::from_bits(0x0d800000),
    );
    assert_eq!(
        query(&mut device, &tiny, &[[f(0.0); 3]], Some(f(0.0)), 1).indices,
        [0]
    );
    let mut state = 0x5350415449414cu64;
    for _ in 0..3 {
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            f(((state >> 32) as i32 % 200) as f32 * 0.25)
        };
        let mut points = (0..32)
            .map(|_| [next(), next(), next()])
            .collect::<Vec<_>>();
        points.extend_from_within(..4);
        let mut queries = points[..4].to_vec();
        queries.extend((0..4).map(|_| [next(), next(), next()]));
        let p = cloud(&points);
        let q = cloud(&queries);
        let grid = build(&mut device, &points, f(1.0));
        for radius in [f(2.5), f(100.0)] {
            let expected = oracle::radius_neighbors(view(&p), view(&q), radius, 7).unwrap();
            let actual = query(&mut device, &grid, &queries, Some(radius), 7);
            assert_eq!((actual.counts, actual.indices), expected);
        }
        let (idx, d2) = oracle::knn(view(&p), view(&q), 7).unwrap();
        let actual = query(&mut device, &grid, &queries, None, 7);
        assert_eq!(actual.indices, idx);
        assert_eq!(
            actual.distances,
            d2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
#[test]
#[ignore = "physical CUDA"]
fn spatial_cuda_exact_cells_rational_and_errors() {
    let mut device = CudaDevice::open().unwrap();
    let mut seed = 0x43454c4cu32;
    for _ in 0..64 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let coord = F::from_bits(seed);
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let cell = F::from_bits(seed & 0x7fffffff);
        if cell.is_zero() || cell.is_nan() || cell.is_infinite() {
            continue;
        }
        build(&mut device, &[[coord, f(0.0), f(0.0)]], cell);
    }
    for (coordinate, cell) in [
        (f(-0.1), f(1.0)),
        (f(-2147483648.0), f(1.0)),
        (f(2147483648.0), f(1.0)),
        (F::from_bits(1), F::from_bits(1)),
    ] {
        build(&mut device, &[[coordinate, f(0.0), f(0.0)]], cell);
    }
}
fn c13_periodic_cloud(sizes: &[usize]) {
    let mut device = CudaDevice::open().unwrap();
    let period = (0..256)
        .map(|i| [f((i % 16) as f32 - 8.0), f((i / 16) as f32 - 8.0), f(0.0)])
        .collect::<Vec<_>>();
    let q = [[f(0.0); 3], [f(3.0), f(-2.0), f(0.0)]];
    let q_axes = cloud(&q);
    // Exact periodic decomposition reuses the frozen <=1024-point oracle. The
    // helper limits remain unchanged; each full block is the same 256 points.
    for &n in sizes {
        let points = (0..n).map(|i| period[i % 256]).collect::<Vec<_>>();
        let grid = build(&mut device, &points, f(1.0));
        // Compose independent <=256-point oracle blocks, offset original ids,
        // sum true counts, and merge each block's retained candidates globally.
        // This preserves ties across arbitrary numbers of repeated blocks.
        for radius in [None, Some(f(2.0))] {
            let mut candidates = vec![Vec::new(); q.len()];
            let mut counts = vec![0; q.len()];
            for (block_len, repetitions, start) in [
                (256, n / 256, 0),
                (n % 256, usize::from(n % 256 != 0), n / 256),
            ] {
                if repetitions == 0 {
                    continue;
                }
                let block = cloud(&period[..block_len]);
                let (block_counts, indices) = if let Some(r) = radius {
                    oracle::radius_neighbors(view(&block), view(&q_axes), r, 7).unwrap()
                } else {
                    let (idx, _) = oracle::knn(view(&block), view(&q_axes), 7).unwrap();
                    (vec![block_len as u32; q.len()], idx)
                };
                for row in 0..q.len() {
                    counts[row] += block_counts[row] * repetitions as u32;
                    for repetition in 0..repetitions {
                        for &id in &indices[row * 7..(row + 1) * 7] {
                            if id != u32::MAX {
                                candidates[row].push((
                                    oracle::squared_distance(period[id as usize], q[row]).to_bits(),
                                    ((start + repetition) * 256) as u32 + id,
                                ));
                            }
                        }
                    }
                }
            }
            let mut idx = Vec::new();
            let mut d2 = Vec::new();
            for row in &mut candidates {
                row.sort_unstable();
                for slot in 0..7 {
                    let (distance, id) = row.get(slot).copied().unwrap_or((0x7f800000, u32::MAX));
                    idx.push(id);
                    d2.push(distance);
                }
            }
            let actual = query(&mut device, &grid, &q, radius, 7);
            assert_eq!(actual.indices, idx, "n={n} radius={radius:?}");
            if radius.is_some() {
                assert_eq!(actual.counts, counts, "n={n}");
            } else {
                assert_eq!(actual.distances, d2, "n={n}");
            }
        }
        eprintln!("C13 n={n} map/sort/boundaries/radius/knn exact; no performance admission");
    }
}
#[test]
#[ignore = "physical CUDA; C13 raw harness through 2^18 (the per-push set)"]
fn spatial_cuda_c13_periodic_cloud_through_2_18() {
    c13_periodic_cloud(&[0, 1, 31, 129, 4097, 65536, 1 << 18]);
}
#[test]
#[ignore = "physical CUDA; C13 raw harness through 2^22, nightly long list (#248)"]
fn spatial_cuda_c13_large_periodic_cloud() {
    c13_periodic_cloud(&[0, 1, 31, 129, 4097, 65536, 1 << 18, 1 << 20, 1 << 22]);
}
#[test]
#[ignore = "physical CUDA"]
fn spatial_cuda_source_and_failure_publication() {
    let source = std::fs::read("demo/spatial-checker.inv").unwrap();
    let checked = invar::check::check(invar::front::parse(&source).unwrap(), true).unwrap();
    let mir = invar::mir::lower(&checked).unwrap();
    let reference = invar::eval::observe(&checked, 100000);
    let mut device = CudaDevice::open().unwrap();
    let run = invar::schedule::cuda::run_source_with_device(
        &source,
        &mir,
        100000,
        &mut device,
        compiler(),
    )
    .unwrap();
    assert_eq!(run.observation.result, reference.result);
    assert_eq!(run.observation.steps_used, reference.steps_used);
    assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
    assert!(
        run.executed
            .iter()
            .any(|r| r.implementation == Some("cuda-sorted-grid-strict-f32"))
    );
    let snapshot = r#"fn pass(g: take VoxelGrid)->VoxelGrid { return g; }
        fn main()->i32 { var xs:Array<f32> = [0.0]; let g=voxel_grid_build(xs,xs,xs,1.0);
        xs[0u32]=100.0; let g2=pass(move g); let q:Array<f32> = [0.0];
        let near=radius_neighbors(g2,q,q,q,0.0,1u32); let counts=move near.counts;
        if counts[0u32]==1u32 { return 7; } return 0; }"#;
    let checked =
        invar::check::check(invar::front::parse(snapshot.as_bytes()).unwrap(), true).unwrap();
    let mir = invar::mir::lower(&checked).unwrap();
    let expected = invar::eval::observe(&checked, 100000);
    let actual = invar::schedule::cuda::run_source_with_device(
        snapshot.as_bytes(),
        &mir,
        100000,
        &mut device,
        compiler(),
    )
    .unwrap();
    assert_eq!(actual.observation.result, expected.result);
    assert_eq!(actual.observation.steps_used, expected.steps_used);
    assert_eq!(actual.observation.cleanup_trace, expected.cleanup_trace);
    let prefix = "let xs: Array<f32> = [0.0]; let q: Array<f32> = [0.0,1.0];";
    for body in [
        format!("{prefix} let g=voxel_grid_build(xs,q,xs,0.0);"),
        format!("{prefix} let g=voxel_grid_build(xs,xs,xs,0.0);"),
        "let huge:Array<f32> = [2147483648.0]; let g=voxel_grid_build(huge,huge,huge,1.0);".into(),
        format!(
            "{prefix} let g=voxel_grid_build(xs,xs,xs,1.0); let r=knn(g,q,xs,q,4294967295u32);"
        ),
        format!("{prefix} let g=voxel_grid_build(xs,xs,xs,1.0); let r=knn(g,q,q,q,4294967295u32);"),
        format!("{prefix} let g=voxel_grid_build(xs,xs,xs,1.0); let r=knn(g,q,xs,q,1u32/0u32);"),
        format!(
            "{prefix} let g=voxel_grid_build(xs,xs,xs,1.0); let r=radius_neighbors(g,q,q,q,(0.0/0.0),4294967295u32);"
        ),
        format!(
            "{prefix} let g=voxel_grid_build(xs,xs,xs,1.0); let r=knn(g,q,q,q,{}u32);",
            invar::array_limits::MAX_ARRAY_LEN / 2 + 1
        ),
    ] {
        let source = format!("fn main()->i32 {{ {body} return 0; }}");
        let checked =
            invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let mir = invar::mir::lower(&checked).unwrap();
        let expected = invar::eval::observe(&checked, 100000);
        assert!(expected.result.is_err());
        let actual = invar::schedule::cuda::run_source_with_device(
            source.as_bytes(),
            &mir,
            100000,
            &mut device,
            compiler(),
        )
        .unwrap();
        assert_eq!(actual.observation.result, expected.result, "{source}");
        assert_eq!(actual.observation.steps_used, expected.steps_used);
        assert_eq!(actual.observation.cleanup_trace, expected.cleanup_trace);
    }
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "run",
            "demo/spatial-checker.inv",
            "--engine",
            "cuda",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["result"]["value"], 3);
    let p = spatial::Plan::mapping(1, 1.0f32.to_bits()).unwrap();
    for fault in [
        CollectiveFault::OutOfMemory,
        CollectiveFault::LaunchFailure,
        CollectiveFault::PartialCompletion,
    ] {
        let mut device = CudaDevice::open().unwrap();
        assert!(
            device
                .run_spatial_with_fault(&p, [&[0; 3], &[], &[], &[]], compiler(), fault)
                .is_err()
        );
        if fault == CollectiveFault::PartialCompletion {
            assert!(device.is_poisoned());
        }
    }
}

#[test]
fn spatial_cuda_unmeasured_auto_keeps_mixed_intrinsics_on_cpu() {
    let demo = std::fs::read_to_string("demo/spatial-checker.inv").unwrap();
    let source = demo.replace(
        "fn main() -> i32 {",
        "fn main() -> i32 { var keys:Array<u32> = [2u32,1u32]; sort(keys);",
    );
    assert_ne!(source, demo);
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = invar::mir::lower(&checked).unwrap();
    let expected = invar::eval::observe(&checked, 100000);
    // No CUDA context or compiler is needed for an unmeasured family. This
    // catches dispatch through the sort-only provider before any device work.
    let actual = invar::schedule::cuda::run_auto_source(
        source.as_bytes(),
        &program,
        100000,
        Path::new("/unavailable/spatial-nvrtc"),
    )
    .unwrap();
    assert_eq!(actual.observation.result, expected.result);
    assert_eq!(actual.observation.steps_used, expected.steps_used);
    assert_eq!(actual.observation.cleanup_trace, expected.cleanup_trace);
    assert_eq!(actual.cache_hits, 0);
    assert!(
        actual
            .choices
            .iter()
            .any(|c| c.reason == invar::spatial::placement::UNMEASURED)
    );
    assert!(actual.choices.iter().all(|c| c.backend == "cpu-parallel"));
}
