#![forbid(unsafe_code)]
use invar::sorting::{KeyKind, argsort_words, sort_words};

fn canonical(bits: u32) -> u32 {
    if f32::from_bits(bits).is_nan() {
        0x7fc00000
    } else {
        bits
    }
}
fn expected(input: &[u32], kind: KeyKind) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..input.len() as u32).collect();
    indices.sort_by(|&a, &b| {
        let (a, b) = (input[a as usize], input[b as usize]);
        match kind {
            KeyKind::U32 => a.cmp(&b),
            KeyKind::I32 => (a as i32).cmp(&(b as i32)),
            KeyKind::F32 => f32::from_bits(canonical(a)).total_cmp(&f32::from_bits(canonical(b))),
        }
    });
    indices
}
#[test]
fn radix_differential_raw_bits_stability_workers_and_boundaries() {
    for workers in [1, 2, 4] {
        let mut keys = vec![7u32; 137];
        let ids: Vec<u32> = (0..137).collect();
        let mut payload = ids.clone();
        sort_words(&mut keys, Some(&mut payload), KeyKind::U32, None, workers).unwrap();
        assert_eq!(keys, vec![7; 137]);
        assert_eq!(payload, ids, "all-equal keys must preserve payload order");
    }
    for kind in [KeyKind::U32, KeyKind::I32, KeyKind::F32] {
        for length in [0, 1, 3, 63, 64, 65, 255, 256, 257, 4096] {
            for seed in [1u32, 0xdeadbeef, 0x12345678] {
                let mut state = seed;
                let edges = [
                    0,
                    0x80000000,
                    0x7f800000,
                    0xff800000,
                    0x7fc00001,
                    0xff800001,
                    1,
                    u32::MAX,
                ];
                let input: Vec<u32> = (0..length)
                    .map(|i| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        if i % 3 == 0 {
                            edges[i % edges.len()]
                        } else {
                            state
                        }
                    })
                    .collect();
                let order = expected(&input, kind);
                for workers in [1, 2, 4] {
                    assert_eq!(argsort_words(&input, kind, workers).unwrap(), order);
                    let mut keys = input.clone();
                    let mut payload: Vec<u32> = (0..length as u32).collect();
                    sort_words(&mut keys, Some(&mut payload), kind, None, workers).unwrap();
                    assert_eq!(
                        payload, order,
                        "{kind:?} length={length} seed={seed} workers={workers}"
                    );
                    assert_eq!(
                        keys,
                        order
                            .iter()
                            .map(|&i| if kind == KeyKind::F32 {
                                canonical(input[i as usize])
                            } else {
                                input[i as usize]
                            })
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}
#[test]
fn radix_offsets_and_failure_atomicity() {
    for workers in [1, 2, 4] {
        let mut keys = [9, 3, 1, 8];
        sort_words(
            &mut keys,
            None,
            KeyKind::U32,
            Some(&[0, 1, 1, 3, 4]),
            workers,
        )
        .unwrap();
        assert_eq!(keys, [9, 1, 3, 8]);
        for offsets in [&[0][..], &[0, 0][..]] {
            sort_words(&mut [], None, KeyKind::U32, Some(offsets), workers).unwrap();
        }
        for offsets in [
            &[][..],
            &[0][..],
            &[2][..],
            &[1, 3][..],
            &[0, 3][..],
            &[0, 2, 4, 3][..],
            &[0, 2, 5][..],
        ] {
            let mut keys = [0xffc00001, 0, 0x80000000, 0x7fc00001];
            let mut payload = [4, 3, 2, 1];
            let before = (keys, payload);
            assert!(
                sort_words(
                    &mut keys,
                    Some(&mut payload),
                    KeyKind::F32,
                    Some(offsets),
                    workers
                )
                .is_err()
            );
            assert_eq!((keys, payload), before);
        }
        let mut keys = [3, 1, 2];
        let mut payload = [9, 8];
        assert!(sort_words(&mut keys, Some(&mut payload), KeyKind::U32, None, workers).is_err());
        assert_eq!(keys, [3, 1, 2]);
        assert_eq!(payload, [9, 8]);
    }
    assert!(argsort_words(&[1], KeyKind::U32, 0).is_err());
    assert!(argsort_words(&[1], KeyKind::U32, invar::schedule::MAX_WORKERS + 1).is_err());
}

#[test]
fn radix_host_buffer_limit_is_distinct_from_source_array_limit() {
    let mut keys = vec![7u32; invar::array_limits::MAX_ARRAY_LEN + 1];
    let expected: Vec<u32> = (0..keys.len() as u32).collect();
    let mut ids = expected.clone();
    sort_words(&mut keys, Some(&mut ids), KeyKind::U32, None, 4).unwrap();
    assert_eq!(ids, expected);
    assert!(keys.iter().all(|&k| k == 7));
    let source = format!(
        "fn main() -> i32 {{ var xs = filled_u32({}u32,0u32); sort(xs); return 7; }}",
        invar::array_limits::MAX_ARRAY_LEN + 1
    );
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    assert!(matches!(
        invar::eval::observe(&checked, 100_000).result,
        Err(invar::eval::EvalError::Tool {
            code: "ARRAY_SIZE_LIMIT",
            ..
        })
    ));
}
