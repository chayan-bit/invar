use super::*;
use libloading::{Library, Symbol};
use std::{
    path::Path,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

#[test]
fn record_layout_tags_and_bounds_match_the_contract() {
    assert_eq!(std::mem::size_of::<HostArrayRecordV1>(), 24);
    assert_eq!(std::mem::align_of::<HostArrayRecordV1>(), 8);
    assert_eq!(
        [
            std::mem::offset_of!(HostArrayRecordV1, element),
            std::mem::offset_of!(HostArrayRecordV1, mode),
            std::mem::offset_of!(HostArrayRecordV1, data),
            std::mem::offset_of!(HostArrayRecordV1, len),
        ],
        [0, 4, 8, 16]
    );
    assert_eq!(
        (ELEMENT_I32, ELEMENT_U32, MODE_READ, MODE_MUT),
        (1, 2, 1, 3)
    );
    // No public inv_arg_v1 mode decodes as a private mut.
    for public in [crate::Mode::Value, crate::Mode::Read, crate::Mode::Take] {
        assert_ne!(public as u32, MODE_MUT);
    }
    assert_eq!(
        (
            MAX_ARRAYS,
            MAX_ELEMENTS,
            MAX_LIVE_ELEMENTS,
            MAX_SNAPSHOT_BYTES
        ),
        (8, 1_000_000, 2_000_000, 8_000_000)
    );
    // Lengths are representable both ways on the supported 64-bit hosts.
    assert_eq!(MAX_ELEMENTS as u64 as usize, MAX_ELEMENTS);
}

#[test]
fn records_are_built_only_from_the_borrowed_slices() {
    let shared = [1, -2, 3];
    let mut out = [7u32, 8, 9, 10];
    let mut empty: [i32; 0] = [];
    let other = [5u32];
    let (shared_at, out_at, other_at) = (
        shared.as_ptr().addr(),
        out.as_ptr().addr(),
        other.as_ptr().addr(),
    );
    let mut call = HostArrayCall::prepare(vec![
        HostArray::ReadI32(&shared),
        HostArray::MutU32(&mut out),
        HostArray::MutI32(&mut empty),
        HostArray::ReadI32(&shared),
        HostArray::ReadU32(&other),
    ])
    .unwrap();
    let records = call.records();
    let view: Vec<(u32, u32, usize, u64)> = records
        .iter()
        .map(|r| (r.element(), r.mode(), r.address(), r.len()))
        .collect();
    assert_eq!(
        view,
        vec![
            (ELEMENT_I32, MODE_READ, shared_at, 3),
            (ELEMENT_U32, MODE_MUT, out_at, 4),
            (ELEMENT_I32, MODE_MUT, 0, 0),
            (ELEMENT_I32, MODE_READ, shared_at, 3),
            (ELEMENT_U32, MODE_READ, other_at, 1),
        ]
    );
    // Rebuilding records is idempotent and allocation-free (within capacity).
    let capacity = call.records.capacity();
    let again: Vec<(u32, u32, usize, u64)> = call
        .records()
        .iter()
        .map(|r| (r.element(), r.mode(), r.address(), r.len()))
        .collect();
    assert_eq!(again, view);
    assert_eq!(call.records.capacity(), capacity);
    let records = call.records();
    assert_eq!((records.len(), records.is_empty()), (5, false));
    assert_eq!(records.get(1).map(HostArrayRecordV1::mode), Some(MODE_MUT));
    assert!(records.get(5).is_none());
    assert_eq!(
        records.as_ptr(),
        std::ptr::from_ref(records.get(0).unwrap())
    );
    call.commit();
    assert_eq!(out, [7, 8, 9, 10]);
}

fn extent(element: u32, mutable: bool, address: usize, length: usize) -> Extent {
    Extent {
        element,
        mutable,
        address,
        length,
    }
}

#[test]
fn validation_refuses_in_contract_order_on_integer_metadata() {
    let base = 1 << 24;
    let read = |a, n| extent(ELEMENT_I32, false, a, n);
    let write = |a, n| extent(ELEMENT_I32, true, a, n);
    assert_eq!(
        validate(&vec![read(base, 1); 9]),
        Err(HostArrayError::Count)
    );
    assert_eq!(
        validate(&[read(base, 1_000_001)]),
        Err(HostArrayError::Length)
    );
    assert_eq!(
        validate(&[read(usize::MAX - 3, 1)]),
        Err(HostArrayError::Extent)
    );
    // Any overlap with a mut extent, or across element types, is refused.
    assert_eq!(
        validate(&[write(base, 10), read(base + 4, 1)]),
        Err(HostArrayError::Alias)
    );
    assert_eq!(
        validate(&[write(base, 10), write(base, 10)]),
        Err(HostArrayError::Alias)
    );
    assert_eq!(
        validate(&[read(base, 10), extent(ELEMENT_U32, false, base, 10)]),
        Err(HostArrayError::Alias)
    );
    // Read aliases are admitted and counted once; touching extents add.
    assert_eq!(
        validate(&[read(base, 10), read(base, 10), read(base + 4, 2)]),
        Ok(0)
    );
    assert_eq!(validate(&[write(base, 10), read(base + 40, 10)]), Ok(40));
    assert_eq!(
        validate(&[write(base, 0), write(base, 0), read(base, 3)]),
        Ok(0)
    );
    // The live union is checked; disjoint mut snapshots then fit exactly.
    assert_eq!(
        validate(&[
            read(base, 1_000_000),
            read(base + 4_000_000, 1_000_000),
            read(base + 8_000_000, 1)
        ]),
        Err(HostArrayError::Live)
    );
    let eight: Vec<Extent> = (0..8)
        .map(|i| write(base + i * 1_000_000, 250_000))
        .collect();
    assert_eq!(validate(&eight), Ok(MAX_SNAPSHOT_BYTES));
}

#[test]
fn every_reservation_failure_refuses_before_any_write() {
    for fail_at in 0..4 {
        let mut a = [1, 2, 3];
        let mut b = [4u32, 5];
        let reads = [9];
        let prepared = HostArrayCall::prepare_with(
            vec![
                HostArray::MutI32(&mut a),
                HostArray::ReadI32(&reads),
                HostArray::MutU32(&mut b),
            ],
            Some(fail_at),
        );
        assert_eq!(
            prepared.err(),
            Some(HostArrayError::Snapshot),
            "reservation {fail_at}"
        );
        assert_eq!((a, b), ([1, 2, 3], [4, 5]));
    }
}

/// Simulated callee writes through the borrowed slices (pure Rust).
fn scribble(call: &mut HostArrayCall<'_>, partial: usize) {
    for array in call.arrays_mut() {
        match array {
            HostArray::MutI32(v) => v.iter_mut().take(partial).for_each(|x| *x = i32::MIN),
            HostArray::MutU32(v) => v.iter_mut().take(partial).for_each(|x| *x = 0x7fa0_0001),
            _ => {}
        }
    }
}

const BITS: [u32; 4] = [0x7f80_0001, 0x8000_0000, 0xffc1_2345, 0];

#[test]
fn rollback_restores_every_mut_array_on_every_non_commit_path() {
    // Read arrays interleaved before and between the mut arrays check that
    // snapshots stay paired with their own slices.
    for path in ["restore", "finish_err", "drop", "panic"] {
        for partial in [1, 4] {
            let reads = [3, 4];
            let mut a = [-1, i32::MAX, 0, 7];
            let mut b = BITS;
            let mut c = [11u32];
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut call = HostArrayCall::prepare(vec![
                    HostArray::ReadI32(&reads),
                    HostArray::MutI32(&mut a),
                    HostArray::ReadI32(&reads),
                    HostArray::MutU32(&mut b),
                    HostArray::MutU32(&mut c),
                ])
                .unwrap();
                scribble(&mut call, partial);
                match path {
                    "restore" => call.restore(),
                    "finish_err" => {
                        let typed: Result<(), &str> = call.finish(Err("trap"));
                        assert_eq!(typed, Err("trap"));
                    }
                    "drop" => drop(call),
                    _ => panic!("injected panic before commit"),
                }
            }));
            assert_eq!(outcome.is_err(), path == "panic", "{path}");
            assert_eq!(a, [-1, i32::MAX, 0, 7], "{path} {partial}");
            assert_eq!(b, BITS, "{path} {partial}: exact bits");
            assert_eq!(c, [11], "{path} {partial}");
        }
    }
}

#[test]
fn commit_keeps_every_write_only_after_success() {
    let mut a = [1, 2];
    let mut b = BITS;
    let mut call =
        HostArrayCall::prepare(vec![HostArray::MutI32(&mut a), HostArray::MutU32(&mut b)]).unwrap();
    scribble(&mut call, 2);
    let kept: Result<u32, ()> = call.finish(Ok(5));
    assert_eq!(kept, Ok(5));
    assert_eq!(a, [i32::MIN, i32::MIN]);
    assert_eq!(b, [0x7fa0_0001, 0x7fa0_0001, 0xffc1_2345, 0]);
}

// ---------------------------------------------------------------------------
// C-FIXTURE EVIDENCE (not compiled-source support): an isolated C library
// checks the layout from C and writes at the real caller addresses.

static NEXT: AtomicU64 = AtomicU64::new(0);

fn fixture() -> (std::path::PathBuf, Library) {
    let directory = std::env::temp_dir().join(format!(
        "invar-host-array-fixture-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&directory).unwrap();
    let library = directory.join(if cfg!(target_os = "macos") {
        "libhost_array.dylib"
    } else {
        "libhost_array.so"
    });
    let status = Command::new("cc")
        .arg(if cfg!(target_os = "macos") {
            "-dynamiclib"
        } else {
            "-shared"
        })
        .args(["-fPIC", "-std=c11", "-Wall", "-Wextra", "-Werror"])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/host_array_v1.c"))
        .arg("-o")
        .arg(&library)
        .status()
        .unwrap();
    assert!(status.success());
    // SAFETY: the library was just compiled from the checked fixture source.
    let loaded = unsafe { Library::new(&library) }.unwrap();
    (directory, loaded)
}

type CallFn =
    unsafe extern "C" fn(*const HostArrayRecordV1, u32, u64, u32, *mut u64, *mut i64) -> i32;

#[test]
fn c_fixture_layout_and_in_place_writes_with_rollback() {
    let (directory, library) = fixture();
    {
        // SAFETY: the fixture defines these exact signatures.
        let layout: Symbol<'_, unsafe extern "C" fn(*mut u64)> =
            unsafe { library.get(b"ha1_fixture_layout_v1\0") }.unwrap();
        let call_fn: Symbol<'_, CallFn> = unsafe { library.get(b"ha1_fixture_call_v1\0") }.unwrap();
        let mut seen_layout = [0u64; 6];
        // SAFETY: `seen_layout` holds the six u64 the fixture writes.
        unsafe { layout(seen_layout.as_mut_ptr()) };
        assert_eq!(
            seen_layout,
            [24, 8, 0, 4, 8, 16],
            "C layout equals Rust layout"
        );

        let run = |a: &mut [i32],
                   b: &mut [u32],
                   reads: &[i32],
                   writes: u64,
                   fail: u32,
                   commit_ok: bool| {
            let expected = [
                reads.as_ptr().addr() as u64,
                a.as_ptr().addr() as u64,
                b.as_ptr().addr() as u64,
            ];
            let mut call = HostArrayCall::prepare(vec![
                HostArray::ReadI32(reads),
                HostArray::MutI32(a),
                HostArray::MutU32(b),
            ])
            .unwrap();
            let records = call.records();
            let mut seen = [0u64; 3];
            let mut sum = 0i64;
            // SAFETY: the records come from `call`, which keeps every slice
            // borrowed for this synchronous call; `seen` has one slot per
            // record and `sum` is a valid i64. The fixture writes only
            // within mut records' lengths.
            let status = unsafe {
                call_fn(
                    records.as_ptr(),
                    records.len() as u32,
                    writes,
                    fail,
                    seen.as_mut_ptr(),
                    &mut sum,
                )
            };
            assert_eq!(seen, expected, "the fixture received the caller addresses");
            assert_eq!(sum, reads.iter().map(|x| i64::from(*x)).sum::<i64>());
            let outcome: Result<i32, i32> = if status == 0 && commit_ok {
                Ok(status)
            } else {
                Err(status)
            };
            call.finish(outcome)
        };

        // Normal return: the writes the fixture made in place are kept.
        let reads = [10, -3];
        let mut a = [1, 2, 3];
        let mut b = BITS;
        assert_eq!(run(&mut a, &mut b, &reads, 3, 0, true), Ok(0));
        assert_eq!(
            a.map(|x| x as u32),
            [0x7fa0_0001 ^ 1, 0x7fa0_0001 ^ 8, 0x7fa0_0001 ^ 15]
        );
        assert_eq!(b, [0xffc1_2345 ^ 2, 0xffc1_2345 ^ 9, 0xffc1_2345 ^ 16, 0]);

        // Non-normal status after partial writes: restored bit-exactly.
        let mut a = [-1, i32::MIN, 5];
        let mut b = BITS;
        assert_eq!(run(&mut a, &mut b, &reads, 2, 1, true), Err(1));
        assert_eq!((a, b), ([-1, i32::MIN, 5], BITS));

        // Normal status but host-result preparation failed: no commit.
        let mut a = [4, 4, 4];
        let mut b = BITS;
        assert_eq!(run(&mut a, &mut b, &reads, 3, 0, false), Err(0));
        assert_eq!((a, b), ([4, 4, 4], BITS));

        // Panic after the fixture wrote, before commit: unwinding restores.
        let mut a = [6, 6, 6];
        let mut b = BITS;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut call =
                HostArrayCall::prepare(vec![HostArray::MutI32(&mut a), HostArray::MutU32(&mut b)])
                    .unwrap();
            let records = call.records();
            let (mut seen, mut sum) = ([0u64; 2], 0i64);
            // SAFETY: as above, for two mut records.
            let status = unsafe { call_fn(records.as_ptr(), 2, 3, 0, seen.as_mut_ptr(), &mut sum) };
            assert_eq!(status, 0);
            panic!("injected panic after foreign writes");
        }));
        assert!(outcome.is_err());
        assert_eq!((a, b), ([6, 6, 6], BITS));
    }
    drop(library);
    let _ = std::fs::remove_dir_all(directory);
}

// ---------------------------------------------------------------------------
// Snapshot pairing for every read/mut order (up to five arrays).

#[test]
fn rollback_pairs_snapshots_for_every_read_mut_order() {
    for arity in 1..=5usize {
        for pattern in 0u32..(1 << arity) {
            // Bit k set: argument k is mut; mut arrays alternate i32/u32.
            let mut ints: Vec<Vec<i32>> = (0..arity)
                .map(|k| vec![k as i32 * 10 - 3, i32::MIN + 1 + k as i32])
                .collect();
            let mut uints: Vec<Vec<u32>> = (0..arity)
                .map(|k| vec![0x7f80_0000 + k as u32, u32::MAX - k as u32, k as u32])
                .collect();
            let reads: Vec<Vec<i32>> = (0..arity).map(|k| vec![k as i32]).collect();
            // Every seed differs from its scribble value (i32::MIN for i32,
            // 0x7fa0_0001 for u32), so any missed write changes the result.
            assert!(ints.iter().flatten().all(|&x| x != i32::MIN));
            assert!(uints.iter().flatten().all(|&x| x != 0x7fa0_0001));
            let (before_i, before_u) = (ints.clone(), uints.clone());
            for commit in [false, true] {
                let mut iv = ints.iter_mut();
                let mut uv = uints.iter_mut();
                let arrays: Vec<HostArray<'_>> = (0..arity)
                    .map(|k| match (pattern >> k & 1 == 1, k % 2 == 0) {
                        (true, true) => HostArray::MutI32(iv.next().unwrap()),
                        (true, false) => HostArray::MutU32(uv.next().unwrap()),
                        (false, _) => HostArray::ReadI32(&reads[k]),
                    })
                    .collect();
                let mut call = HostArrayCall::prepare(arrays).unwrap();
                let mutable = call
                    .records()
                    .iter()
                    .filter(|r| r.mode() == MODE_MUT)
                    .count();
                assert_eq!(mutable, pattern.count_ones() as usize);
                scribble(&mut call, 3);
                if commit {
                    call.commit();
                } else {
                    call.restore();
                }
                if commit {
                    // Exactly the simulated writes, per array: the j-th mut
                    // i32 argument takes ints[j], the j-th mut u32 uints[j];
                    // `scribble(_, 3)` covers every element of both shapes.
                    let (mut expected_i, mut expected_u) = (before_i.clone(), before_u.clone());
                    let (mut next_i, mut next_u) = (0, 0);
                    for k in (0..arity).filter(|k| pattern >> k & 1 == 1) {
                        if k % 2 == 0 {
                            expected_i[next_i].iter_mut().for_each(|x| *x = i32::MIN);
                            next_i += 1;
                        } else {
                            expected_u[next_u].iter_mut().for_each(|x| *x = 0x7fa0_0001);
                            next_u += 1;
                        }
                    }
                    assert_eq!(ints, expected_i, "{arity} {pattern:b} commit i32");
                    assert_eq!(uints, expected_u, "{arity} {pattern:b} commit u32");
                    ints.clone_from(&before_i);
                    uints.clone_from(&before_u);
                } else {
                    assert_eq!(ints, before_i, "{arity} {pattern:b} restore i32");
                    assert_eq!(uints, before_u, "{arity} {pattern:b} restore u32");
                }
            }
        }
    }
}

#[test]
fn descriptor_tags_are_private_and_match_the_loader() {
    assert_eq!(
        crate::exports::ARRAY_TAGS,
        [
            descriptor_tag(ELEMENT_I32, MODE_READ),
            descriptor_tag(ELEMENT_I32, MODE_MUT),
            descriptor_tag(ELEMENT_U32, MODE_READ),
            descriptor_tag(ELEMENT_U32, MODE_MUT),
            descriptor_tag(ELEMENT_F32, MODE_READ),
            descriptor_tag(ELEMENT_F32, MODE_MUT),
        ]
    );
    for tag in crate::exports::ARRAY_TAGS {
        // Never a public inv_arg_v1 tag (1..=4).
        assert!(tag > crate::STRING);
        assert!(decode_descriptor_tag(tag).is_some());
    }
    for tag in [0, 1, 2, 3, 4, 0x110, 0x112, 0x132, 0x141, 0x211, 0x1111] {
        assert_eq!(decode_descriptor_tag(tag), None, "{tag:#x}");
    }
}

mod compile_checks;

#[test]
fn f32_record_and_exact_bit_rollback_on_drop_error_and_panic() {
    assert_eq!(ELEMENT_F32, 3);
    for mode in [MODE_READ, MODE_MUT] {
        assert_eq!(
            decode_descriptor_tag(descriptor_tag(ELEMENT_F32, mode)),
            Some((3, mode))
        );
    }
    assert_eq!(decode_descriptor_tag(descriptor_tag(3, 2)), None);
    assert_eq!(decode_descriptor_tag(descriptor_tag(4, 1)), None);
    for path in ["drop", "error", "panic", "commit"] {
        let reads = BITS.map(f32::from_bits);
        let mut xs = BITS.map(f32::from_bits);
        let address = xs.as_ptr().addr();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut call = HostArrayCall::prepare(vec![
                HostArray::ReadF32(&reads),
                HostArray::MutF32(&mut xs),
            ])
            .unwrap();
            assert_eq!(
                call.records()
                    .get(1)
                    .map(|r| (r.element(), r.mode(), r.address())),
                Some((3, 3, address))
            );
            if let HostArray::MutF32(v) = &mut call.arrays_mut()[1] {
                v.fill(1.0);
            }
            match path {
                "commit" => call.commit(),
                "error" => {
                    let result: Result<(), ()> = call.finish(Err(()));
                    assert!(result.is_err());
                }
                "panic" => panic!("injected f32 decode panic"),
                _ => drop(call),
            }
        }));
        assert_eq!(outcome.is_err(), path == "panic");
        assert_eq!(
            xs.map(f32::to_bits),
            if path == "commit" {
                [1.0f32.to_bits(); 4]
            } else {
                BITS
            }
        );
    }
    for fail_at in 0..3 {
        let mut xs = BITS.map(f32::from_bits);
        assert_eq!(
            HostArrayCall::prepare_with(vec![HostArray::MutF32(&mut xs)], Some(fail_at)).err(),
            Some(HostArrayError::Snapshot)
        );
        assert_eq!(xs.map(f32::to_bits), BITS);
    }
}
