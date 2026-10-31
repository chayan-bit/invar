use super::*;

fn at(element: Type, address: usize, length: usize) -> Extent {
    Extent {
        element,
        address,
        length,
        mutable: false,
    }
}
fn muts(element: Type, address: usize, length: usize) -> Extent {
    Extent {
        mutable: true,
        ..at(element, address, length)
    }
}
fn i32s(address: usize, length: usize) -> Extent {
    at(Type::ArrayI32, address, length)
}
fn ok(extents: &[Extent]) -> Plan {
    admit(extents, STAGING_LIMIT_BYTES).unwrap()
}
fn code(extents: &[Extent], staging: usize) -> &'static str {
    admit(extents, staging).unwrap_err().0
}

#[test]
fn exact_aliases_share_one_group_and_count_once() {
    let plan = ok(&[i32s(4096, 10), i32s(4096, 10), i32s(4096, 10)]);
    assert_eq!((plan.group, plan.live_elements), (vec![0, 0, 0], 10));
    // A same-address prefix is a partial overlap, not an exact alias.
    let plan = ok(&[i32s(4096, 10), i32s(4096, 4), i32s(4096, 4)]);
    assert_eq!((plan.group, plan.live_elements), (vec![0, 1, 1], 10));
    let plan = ok(&[i32s(1 << 20, 1_000_000), i32s(1 << 20, 1_000_000)]);
    assert_eq!(plan.live_elements, 1_000_000);
}

#[test]
fn partial_overlaps_count_the_union_and_stay_separate_owners() {
    let plan = ok(&[i32s(4096, 10), i32s(4096 + 20, 10)]);
    assert_eq!((plan.group, plan.live_elements), (vec![0, 1], 15));
    // Touching extents do not overlap; disjoint extents add.
    let plan = ok(&[i32s(4096, 10), i32s(4096 + 40, 10), i32s(1 << 20, 3)]);
    assert_eq!((plan.group, plan.live_elements), (vec![0, 1, 2], 23));
    // 1.6M union of two 1M slices is admitted; the per-argument sum is 2M.
    let plan = ok(&[
        i32s(1 << 20, 1_000_000),
        i32s((1 << 20) + 2_400_000, 1_000_000),
    ]);
    assert_eq!(plan.live_elements, 1_600_000);
    // Disjoint buffers over the live limit are refused.
    let base = 1 << 24;
    assert_eq!(
        code(
            &[
                i32s(base, super::super::MAX_LIVE_ELEMENTS / 2),
                i32s(
                    base + super::super::MAX_LIVE_ELEMENTS * 2,
                    super::super::MAX_LIVE_ELEMENTS / 2
                ),
                i32s(base + super::super::MAX_LIVE_ELEMENTS * 4, 1)
            ],
            STAGING_LIMIT_BYTES
        ),
        "ARRAY_SIZE_LIMIT"
    );
}

#[test]
fn empty_extents_never_alias() {
    let plan = ok(&[i32s(4096, 0), i32s(4096, 10), i32s(4096, 0)]);
    assert_eq!((plan.group, plan.live_elements), (vec![0, 1, 2], 10));
    // Not even across element types, including an empty extent strictly
    // inside another buffer (as `&xs[1..1]` would be).
    let plan = ok(&[i32s(4096, 0), at(Type::ArrayF32, 4096, 3)]);
    assert_eq!(plan.live_elements, 3);
    let plan = ok(&[at(Type::ArrayF32, 4096, 3), i32s(4100, 0)]);
    assert_eq!((plan.group, plan.live_elements), (vec![0, 1], 3));
}

#[test]
fn overlapping_extents_of_different_element_types_are_refused() {
    for other in [Type::ArrayU32, Type::ArrayF32] {
        for (address, length) in [(4096, 10), (4100, 1), (4096 + 36, 5)] {
            assert_eq!(
                code(
                    &[i32s(4096, 10), at(other.clone(), address, length)],
                    STAGING_LIMIT_BYTES
                ),
                "HOST_ARRAY_ALIAS",
                "{other:?} {address} {length}"
            );
        }
        // Disjoint or touching extents of different types are admitted.
        assert!(
            admit(
                &[i32s(4096, 10), at(other, 4096 + 40, 10)],
                STAGING_LIMIT_BYTES
            )
            .is_ok()
        );
    }
}

#[test]
fn extent_arithmetic_is_checked_and_never_dereferences() {
    // Synthetic metadata only; these addresses are never read.
    let top = usize::MAX - 3; // element aligned
    assert_eq!(
        code(&[i32s(top, 1)], STAGING_LIMIT_BYTES),
        "HOST_ARRAY_EXTENT"
    );
    assert_eq!(
        code(&[i32s(4096, usize::MAX / 2)], STAGING_LIMIT_BYTES),
        "HOST_ARRAY_EXTENT"
    );
    assert_eq!(
        code(&[i32s(4097, 1)], STAGING_LIMIT_BYTES),
        "HOST_ARRAY_EXTENT"
    );
    assert!(admit(&[i32s(top, 0)], STAGING_LIMIT_BYTES).is_ok());
    // Extents are checked before alias rules, alias rules before limits.
    assert_eq!(
        code(
            &[i32s(4096, 10), at(Type::ArrayF32, 4096, 10), i32s(top, 1)],
            0
        ),
        "HOST_ARRAY_EXTENT"
    );
    let base = 1 << 24;
    assert_eq!(
        code(
            &[
                i32s(base, 1_000_000),
                i32s(base + 4_000_000, 1_000_000),
                i32s(base + 8_000_000, 1),
                at(Type::ArrayF32, base, 1)
            ],
            0
        ),
        "HOST_ARRAY_ALIAS"
    );
    assert_eq!(
        code(&[i32s(base, super::super::MAX_LIVE_ELEMENTS + 1)], 0),
        "ARRAY_SIZE_LIMIT"
    );
}

#[test]
fn staging_copies_each_exact_alias_group_once_within_its_bound() {
    // One 40-byte copy serves both exact aliases.
    assert!(admit(&[i32s(4096, 10), i32s(4096, 10)], 40).is_ok());
    assert_eq!(
        code(&[i32s(4096, 10), i32s(4096, 10)], 39),
        "HOST_STAGING_LIMIT"
    );
    // A partial overlap stages each argument.
    assert_eq!(
        code(&[i32s(4096, 10), i32s(4100, 10)], 40),
        "HOST_STAGING_LIMIT"
    );
    assert!(admit(&[i32s(4096, 10), i32s(4100, 10)], 80).is_ok());
    // Copies (8 x 2^24 x 8) plus snapshots (2^28 live elements x 8), counted at the widest element (HA-1.4).
    assert_eq!(STAGING_LIMIT_BYTES, 3072 * 1024 * 1024);
}

#[test]
fn staging_refusal_is_a_zero_step_call() {
    let source =
        "fn lens(a: read Array<i32>, b: read Array<i32>) -> u32 { return len(a) + len(b); }";
    let program =
        crate::check::check(crate::front::parse(source.as_bytes()).unwrap(), false).unwrap();
    let xs = [1, 2, 3];
    let args = [
        HostReadArgument::Array(HostReadArray::I32(&xs)),
        HostReadArgument::Array(HostReadArray::I32(&xs)),
    ];
    let o = observe_bounded(&program, "lens", &args, 100, 11);
    assert_eq!(
        (o.result, o.steps_used, o.cleanup_trace.len()),
        (
            Err(EvalError::tool(
                "HOST_STAGING_LIMIT",
                "host staging exceeds its byte bound"
            )),
            0,
            0
        )
    );
    let o = observe_bounded(&program, "lens", &args, 100, 12);
    assert_eq!(o.result, Ok(Value::U32(6)));
}

#[test]
fn any_overlap_with_a_mut_extent_is_refused_and_empty_never_aliases() {
    for other in [
        i32s(4096, 10),
        muts(Type::ArrayI32, 4096, 10),
        i32s(4100, 2),
        muts(Type::ArrayI32, 4092, 2),
        at(Type::ArrayF32, 4096, 10),
        at(Type::ArrayU32, 4128, 5),
    ] {
        let pair = [muts(Type::ArrayI32, 4096, 10), other.clone()];
        assert_eq!(
            code(&pair, STAGING_LIMIT_BYTES),
            "HOST_ARRAY_ALIAS",
            "{other:?}"
        );
        let pair = [other.clone(), muts(Type::ArrayI32, 4096, 10)];
        assert_eq!(
            code(&pair, STAGING_LIMIT_BYTES),
            "HOST_ARRAY_ALIAS",
            "{other:?}"
        );
    }
    // Touching, disjoint and empty extents are admitted; each mut is its
    // own group and counts toward the union.
    let plan = ok(&[
        muts(Type::ArrayI32, 4096, 10),
        i32s(4136, 10),
        muts(Type::ArrayI32, 4100, 0),
        i32s(4096, 0),
    ]);
    assert_eq!((plan.group, plan.live_elements), (vec![0, 1, 2, 3], 20));
    // Extent checks precede the mut alias rule.
    assert_eq!(
        code(
            &[
                muts(Type::ArrayI32, 4096, 10),
                i32s(4096, 10),
                i32s(4097, 1)
            ],
            STAGING_LIMIT_BYTES
        ),
        "HOST_ARRAY_EXTENT"
    );
}

#[test]
fn mut_snapshots_fit_the_bound_because_mut_extents_are_disjoint() {
    // Eight disjoint mut buffers inside the 2M live union: at most
    // 8,000,000 snapshot bytes, on top of at most 32,000,000 copy bytes.
    let base = 1 << 24;
    let eight: Vec<Extent> = (0..8)
        .map(|i| muts(Type::ArrayI32, base + i * 1_000_000, 250_000))
        .collect();
    assert_eq!(ok(&eight).live_elements, 2_000_000);
    let mut nine = eight.clone();
    nine.push(muts(
        Type::ArrayI32,
        base + 9_000_000,
        super::super::MAX_LIVE_ELEMENTS,
    ));
    assert_eq!(code(&nine, STAGING_LIMIT_BYTES), "ARRAY_SIZE_LIMIT");
    assert_eq!(
        MAX_PARAMETERS * MAX_ARRAY_LEN * MAX_ELEMENT_BYTES
            + super::super::MAX_LIVE_ELEMENTS * MAX_ELEMENT_BYTES,
        STAGING_LIMIT_BYTES
    );
}

const MUT_SOURCE: &str = "fn bump(xs: mut Array<i32>, ys: read Array<i32>, k: i32) -> i32 { xs[0u32] = ys[0u32] + k; xs[1u32] = 0 - xs[1u32]; return xs[0u32]; }";

fn bump(xs: &mut [i32], ys: &[i32], k: i32, staging: usize, inject: Inject) -> Observation {
    let program =
        crate::check::check(crate::front::parse(MUT_SOURCE.as_bytes()).unwrap(), false).unwrap();
    let mut args = [
        HostArgument::Mut(HostMutArray::I32(xs)),
        HostArgument::Read(HostReadArray::I32(ys)),
        HostArgument::I32(k),
    ];
    observe_inner(
        &program,
        "bump",
        &mut args,
        1000,
        staging,
        Policy::HostCall,
        inject,
    )
}

#[test]
fn normal_return_commits_after_precheck() {
    let mut xs = [5, 6, 7];
    let o = bump(&mut xs, &[40], 2, STAGING_LIMIT_BYTES, Inject::None);
    assert_eq!(o.result, Ok(Value::I32(42)));
    assert_eq!(xs, [42, -6, 7]);
    assert!(o.cleanup_trace.is_empty());
}

#[test]
fn reservation_failures_and_bounds_refuse_at_zero_steps_without_writes() {
    let reference = bump(&mut [5, 6, 7], &[40], 2, STAGING_LIMIT_BYTES, Inject::None);
    for n in 0..2 {
        let mut xs = [5, 6, 7];
        let o = bump(
            &mut xs,
            &[40],
            2,
            STAGING_LIMIT_BYTES,
            Inject::ReserveFailAt(n),
        );
        assert_eq!(
            (o.result, o.steps_used, o.cleanup_trace.len(), xs),
            (
                Err(EvalError::tool(
                    "HOST_STAGING_LIMIT",
                    "host staging reservation failed"
                )),
                0,
                0,
                [5, 6, 7]
            ),
            "reservation {n}"
        );
    }
    // Copies need 16 bytes: 15 is refused in admission.
    let mut xs = [5, 6, 7];
    let o = bump(&mut xs, &[40], 2, 15, Inject::None);
    assert_eq!((o.steps_used, xs), (0, [5, 6, 7]));
    assert_eq!(
        o.result,
        Err(EvalError::tool(
            "HOST_STAGING_LIMIT",
            "host staging exceeds its byte bound"
        ))
    );
    assert_eq!(
        bump(&mut xs, &[40], 2, 16, Inject::None).steps_used,
        reference.steps_used
    );
}

#[test]
fn failed_precheck_writes_nothing_and_keeps_the_evaluated_steps() {
    let reference = bump(&mut [5, 6, 7], &[40], 2, STAGING_LIMIT_BYTES, Inject::None);
    for inject in [Inject::BorrowedAtCommit, Inject::ResizedAtCommit] {
        let mut xs = [5, 6, 7];
        let o = bump(&mut xs, &[40], 2, STAGING_LIMIT_BYTES, inject);
        assert_eq!(
            o.result,
            Err(EvalError::tool(
                "INTERNAL_HOST_COMMIT",
                "a mut owner failed its writeback precheck"
            ))
        );
        assert_eq!(
            (o.steps_used, &o.cleanup_trace),
            (reference.steps_used, &reference.cleanup_trace)
        );
        assert_eq!(xs, [5, 6, 7]);
    }
}

#[test]
fn caught_panics_leave_caller_buffers_unchanged() {
    for inject in [
        Inject::PanicAfterCopyIn,
        Inject::PanicAfterEval,
        Inject::PanicBeforeCommit,
    ] {
        let mut xs = [5, 6, 7];
        let ys = [40];
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bump(&mut xs, &ys, 2, STAGING_LIMIT_BYTES, inject)
        }));
        assert!(caught.is_err());
        assert_eq!((xs, ys), ([5, 6, 7], [40]));
    }
}

/// AI-01 invariant fault test for `INTERNAL_HOST_READ`: a read owner
/// changed during the call is reported, nothing is committed.
#[test]
fn machine_diagnostics_host_read_changed_read_argument_fault() {
    let mut xs = [5, 6, 7];
    let o = bump(
        &mut xs,
        &[40],
        2,
        STAGING_LIMIT_BYTES,
        Inject::ChangedReadAtEnd,
    );
    assert_eq!(
        o.result,
        Err(EvalError::tool(
            "INTERNAL_HOST_READ",
            "read argument changed during the call"
        ))
    );
    assert_eq!(xs, [5, 6, 7], "a non-normal outcome commits nothing");
    assert!(o.steps_used > 0);
}

/// AI-01 invariant fault test for `INTERNAL_HOST_COMMIT`: a failed
/// writeback precheck is reported and writes nothing.
#[test]
fn machine_diagnostics_host_commit_precheck_fault() {
    for inject in [Inject::BorrowedAtCommit, Inject::ResizedAtCommit] {
        let mut xs = [5, 6, 7];
        let o = bump(&mut xs, &[40], 2, STAGING_LIMIT_BYTES, inject);
        assert_eq!(
            o.result,
            Err(EvalError::tool(
                "INTERNAL_HOST_COMMIT",
                "a mut owner failed its writeback precheck"
            ))
        );
        assert_eq!(xs, [5, 6, 7]);
    }
}

// --- HA-1.4: f64 scalars, Array<f64> and Array<Bool> -----------------------

fn f64s(address: usize, length: usize) -> Extent {
    at(Type::ArrayF64, address, length)
}
fn bools(address: usize, length: usize) -> Extent {
    at(Type::ArrayBool, address, length)
}

#[test]
fn f64_and_bool_extents_use_their_own_element_size() {
    // Eight-byte elements: an overlap of five elements leaves a 15-element union.
    let plan = ok(&[f64s(4096, 10), f64s(4096 + 40, 10)]);
    assert_eq!((plan.group, plan.live_elements), (vec![0, 1], 15));
    // Exact aliases still share one owner.
    let plan = ok(&[f64s(4096, 10), f64s(4096, 10)]);
    assert_eq!((plan.group, plan.live_elements), (vec![0, 0], 10));
    // One-byte elements have no alignment and count one element per byte.
    let plan = ok(&[bools(4097, 10), bools(4097 + 5, 10)]);
    assert_eq!(plan.live_elements, 15);
    // Touching extents of different element sizes never merge into a four-byte count.
    assert_eq!(ok(&[i32s(4096, 8), f64s(4128, 2)]).live_elements, 10);
    assert_eq!(ok(&[bools(4096, 8), f64s(4104, 2)]).live_elements, 10);
}

#[test]
fn f64_extents_are_eight_byte_aligned_and_checked() {
    let limit = STAGING_LIMIT_BYTES;
    assert_eq!(code(&[f64s(4100, 1)], limit), "HOST_ARRAY_EXTENT");
    assert_eq!(
        code(&[f64s(usize::MAX - 15, 3)], limit),
        "HOST_ARRAY_EXTENT"
    );
    assert!(admit(&[bools(4097, 1)], limit).is_ok());
}

#[test]
fn f64_and_bool_overlaps_follow_the_same_alias_rules() {
    let limit = STAGING_LIMIT_BYTES;
    for other in [i32s(4100, 1), bools(4100, 1), at(Type::ArrayF32, 4100, 1)] {
        assert_eq!(code(&[f64s(4096, 2), other], limit), "HOST_ARRAY_ALIAS");
    }
    assert_eq!(
        code(&[muts(Type::ArrayF64, 4096, 2), f64s(4096, 2)], limit),
        "HOST_ARRAY_ALIAS"
    );
    assert_eq!(
        code(&[bools(4096, 2), muts(Type::ArrayBool, 4097, 1)], limit),
        "HOST_ARRAY_ALIAS"
    );
    // An empty extent never aliases, whatever its element type.
    assert!(admit(&[f64s(4096, 2), bools(4100, 0)], limit).is_ok());
}

#[test]
fn staging_counts_eight_bytes_per_f64_element() {
    // 3 f64 elements stage 24 bytes, 3 bool elements 3.
    assert!(admit(&[f64s(4096, 3)], 24).is_ok());
    assert_eq!(code(&[f64s(4096, 3)], 23), "HOST_STAGING_LIMIT");
    assert!(admit(&[bools(4096, 3)], 3).is_ok());
    assert_eq!(code(&[bools(4096, 3)], 2), "HOST_STAGING_LIMIT");
}

const V02_SOURCE: &str = "\
fn blend(xs: mut Array<f64>, ys: read Array<f64>, flags: mut Array<Bool>, k: f64) -> f64 { xs[0u32] = ys[0u32] * k; flags[0u32] = xs[0u32] > 1.0f64; return xs[0u32]; }
fn mirror(xs: mut Array<f64>, ys: read Array<f64>) -> u32 { for i in 0u32..len(ys) { xs[i] = ys[i]; } return len(xs); }
fn count(bs: read Array<Bool>) -> u32 { var n: u32 = 0u32; for i in 0u32..len(bs) { if bs[i] { n = n + 1u32; } } return n; }
fn trap_after_write(xs: mut Array<f64>, flags: mut Array<Bool>) -> u32 { xs[0u32] = 9.0f64; flags[0u32] = true; xs[len(xs)] = 1.0f64; return 0u32; }
fn halves() -> Array<f64> { return filled_f64(2u32, 0.5f64); }
";

fn v02(name: &str, args: &mut [HostArgument<'_>]) -> Observation {
    let program =
        crate::check::check(crate::front::parse(V02_SOURCE.as_bytes()).unwrap(), false).unwrap();
    observe_host_call(&program, name, args, 10_000)
}

#[test]
fn f64_scalar_and_arrays_commit_exact_bits_after_a_normal_return() {
    let mut xs = [1.0f64, 2.0];
    let mut flags = [false, false];
    let o = v02(
        "blend",
        &mut [
            HostArgument::Mut(HostMutArray::Float64(&mut xs)),
            HostArgument::Read(HostReadArray::Float64(&[3.0])),
            HostArgument::Mut(HostMutArray::Bool(&mut flags)),
            HostArgument::F64(F64Bits::from_bits(0.5f64.to_bits())),
        ],
    );
    assert_eq!(
        o.result,
        Ok(Value::F64(F64Bits::from_bits(1.5f64.to_bits())))
    );
    assert_eq!((xs, flags), ([1.5, 2.0], [true, false]));
}

#[test]
fn f64_arrays_keep_nan_payloads_and_signed_zero() {
    let source = [
        f64::from_bits(0x7ff8_0000_0000_0123),
        f64::from_bits(0xfff0_0000_0000_0001),
        -0.0,
        f64::from_bits(1),
    ];
    let mut target = [0.0f64; 4];
    let o = v02(
        "mirror",
        &mut [
            HostArgument::Mut(HostMutArray::Float64(&mut target)),
            HostArgument::Read(HostReadArray::Float64(&source)),
        ],
    );
    assert_eq!(o.result, Ok(Value::U32(4)));
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&target), bits(&source));
}

#[test]
fn a_trap_leaves_f64_and_bool_buffers_unchanged() {
    let mut xs = [f64::from_bits(0x7ff8_0000_0000_0123), 2.0];
    let mut flags = [false, true];
    let (before_xs, before_flags) = (xs.map(f64::to_bits), flags);
    let o = v02(
        "trap_after_write",
        &mut [
            HostArgument::Mut(HostMutArray::Float64(&mut xs)),
            HostArgument::Mut(HostMutArray::Bool(&mut flags)),
        ],
    );
    assert!(
        matches!(o.result, Err(EvalError::Trap { .. })),
        "{:?}",
        o.result
    );
    assert_eq!((xs.map(f64::to_bits), flags), (before_xs, before_flags));
}

#[test]
fn bool_read_arrays_and_an_array_f64_result() {
    let o = v02(
        "count",
        &mut [HostArgument::Read(HostReadArray::Bool(&[
            true, false, true, true,
        ]))],
    );
    assert_eq!(o.result, Ok(Value::U32(3)));
    let o = v02("count", &mut [HostArgument::Read(HostReadArray::Bool(&[]))]);
    assert_eq!(o.result, Ok(Value::U32(0)));
    let o = v02("halves", &mut []);
    let Ok(Value::ArrayF64(values)) = o.result else {
        panic!("expected Array<f64>, got {:?}", o.result);
    };
    assert_eq!(
        *values.borrow(),
        vec![F64Bits::from_bits(0.5f64.to_bits()); 2]
    );
}

#[test]
fn an_argument_of_the_wrong_element_type_is_refused_at_zero_steps() {
    let o = v02(
        "count",
        &mut [HostArgument::Read(HostReadArray::Float64(&[1.0]))],
    );
    assert_eq!((o.steps_used, o.result.is_err()), (0, true));
    let mut xs = [1.0f32];
    let o = v02(
        "mirror",
        &mut [
            HostArgument::Mut(HostMutArray::Float32(&mut xs)),
            HostArgument::Read(HostReadArray::Float64(&[1.0])),
        ],
    );
    assert_eq!((o.steps_used, o.result.is_err()), (0, true));
    let o = v02(
        "blend",
        &mut [
            HostArgument::Mut(HostMutArray::Float64(&mut [0.0])),
            HostArgument::Read(HostReadArray::Float64(&[1.0])),
            HostArgument::Mut(HostMutArray::Bool(&mut [false])),
            HostArgument::F32(F32Bits::from_bits(0)),
        ],
    );
    assert_eq!((o.steps_used, o.result.is_err()), (0, true));
}

#[test]
fn the_read_only_entry_admits_f64_and_bool_buffers() {
    let program =
        crate::check::check(crate::front::parse(V02_SOURCE.as_bytes()).unwrap(), false).unwrap();
    let o = observe_host_read_call(
        &program,
        "count",
        &[HostReadArgument::Array(HostReadArray::Bool(&[true, true]))],
        10_000,
    );
    assert_eq!(o.result, Ok(Value::U32(2)));
}
