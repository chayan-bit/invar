//! Bridge planning, failure-injection and fault tests.
use super::*;

fn f32a(address: usize, length: usize) -> PlanArray {
    PlanArray {
        dtype: Dtype::F32,
        address,
        length,
        mutable: false,
    }
}

#[test]
fn ha1_bridge_plan_preserves_exact_and_partial_f32_aliases() {
    let base = 1 << 26;
    // Exact alias: one union, both arguments at offset 0.
    let plan = plan_bridges(&[f32a(base, 999_999), f32a(base, 999_999)]).unwrap();
    assert_eq!(
        plan.routes,
        vec![
            Route::Union {
                union: 0,
                offset: 0
            };
            2
        ]
    );
    assert_eq!(
        (
            plan.unions.clone(),
            plan.bridge_bytes,
            plan.core_copy_bytes,
            plan.live_elements
        ),
        (vec![999_999], 3_999_996, 3_999_996, 999_999)
    );
    // Partial overlap: one union of 1.4M, offsets 0 and 400,000; the core
    // still copies each argument (two groups) and counts the union once.
    let plan = plan_bridges(&[f32a(base, 1_000_000), f32a(base + 1_600_000, 1_000_000)]).unwrap();
    assert_eq!(
        plan.routes,
        vec![
            Route::Union {
                union: 0,
                offset: 0
            },
            Route::Union {
                union: 0,
                offset: 400_000
            }
        ]
    );
    assert_eq!(
        (
            plan.unions,
            plan.bridge_bytes,
            plan.core_copy_bytes,
            plan.live_elements
        ),
        (vec![1_400_000], 5_600_000, 8_000_000, 1_400_000)
    );
    // Touching extents are separate unions; empty f32 reads need no bridge.
    let plan = plan_bridges(&[f32a(base, 10), f32a(base + 40, 10), f32a(base + 8, 0)]).unwrap();
    assert_eq!(
        plan.routes,
        vec![
            Route::Union {
                union: 0,
                offset: 0
            },
            Route::Union {
                union: 1,
                offset: 0
            },
            Route::EmptyF32
        ]
    );
    assert_eq!(plan.unions, vec![10, 10]);
}

#[test]
fn ha1_bridge_plan_accounts_every_temporary_within_forty_megabytes() {
    let base = 1 << 26;
    // Worst admissible case: eight partially overlapping 1M f32 reads
    // filling exactly the 2,000,000-element live union: one union bridge
    // of 8,000,000 bytes plus eight core copies of 4,000,000 bytes is
    // exactly the 40,000,000-byte bound, and is admitted.
    let full: Vec<PlanArray> = (0..8)
        .map(|i| f32a(base + i * 1_000_000 / 7 * 4, 1_000_000))
        .collect();
    let plan = plan_bridges(&full).unwrap();
    assert_eq!(
        (
            plan.live_elements,
            plan.unions.clone(),
            plan.bridge_bytes,
            plan.core_copy_bytes
        ),
        (2_000_000, vec![2_000_000], 8_000_000, 32_000_000)
    );
    assert_eq!(
        plan.bridge_bytes + plan.core_copy_bytes,
        TEMPORARY_LIMIT_BYTES
    );
    // The combined check is real: a ninth argument (beyond the arity the
    // binding admits) would exceed 40,000,000 and is refused.
    let mut nine = full.clone();
    nine.push(f32a(base + 4, 1_000_000));
    assert_eq!(plan_bridges(&nine), Err(PlanError::Staging));
    // Live union over 2M, cross-type and mut overlaps are refused first.
    assert_eq!(
        plan_bridges(&[
            f32a(base, 1_000_000),
            f32a(base + 4_000_000, 1_000_000),
            f32a(base + 8_000_000, 1)
        ]),
        Err(PlanError::Live)
    );
    let int = PlanArray {
        dtype: Dtype::I32,
        ..f32a(base + 4, 1)
    };
    assert_eq!(plan_bridges(&[f32a(base, 10), int]), Err(PlanError::Alias));
    let mutable = PlanArray {
        mutable: true,
        ..f32a(base + 4, 1)
    };
    assert_eq!(
        plan_bridges(&[f32a(base, 10), mutable]),
        Err(PlanError::Alias)
    );
    // Mut bridges of every dtype are counted; each is its own core group.
    let mi = PlanArray {
        dtype: Dtype::I32,
        mutable: true,
        ..f32a(base + 1_000, 5)
    };
    let mf = PlanArray {
        mutable: true,
        ..f32a(base + 2_000, 3)
    };
    let plan = plan_bridges(&[f32a(base, 10), mi, mf, f32a(base, 10)]).unwrap();
    assert_eq!(plan.routes[1..3], [Route::Mut, Route::Mut]);
    assert_eq!(
        (plan.bridge_bytes, plan.core_copy_bytes),
        (40 + 20 + 12, 40 + 20 + 12)
    );
}

/// HA-1.4: f64 extents are eight bytes and bool extents one, in every count and bound.
#[test]
fn ha1_4_bridge_plan_counts_f64_and_bool_by_element_size() {
    let base = 1 << 26;
    let f64a = |address, length| PlanArray {
        dtype: Dtype::F64,
        address,
        length,
        mutable: false,
    };
    let bool_a = |address, length| PlanArray {
        dtype: Dtype::Bool,
        address,
        length,
        mutable: false,
    };
    // Two disjoint 1M f64 reads fill the 2M live bound: direct routes, no bridge, 16 MB of core copies.
    let plan = plan_bridges(&[f64a(base, 1_000_000), f64a(base + 8_000_000, 1_000_000)]).unwrap();
    assert_eq!(
        (plan.live_elements, plan.bridge_bytes, plan.core_copy_bytes),
        (2_000_000, 0, 16_000_000)
    );
    assert_eq!(plan.routes, [Route::Direct, Route::Direct]);
    assert_eq!(
        plan_bridges(&[
            f64a(base, 1_000_000),
            f64a(base + 8_000_000, 1_000_000),
            f64a(base + 16_000_000, 1)
        ]),
        Err(PlanError::Live)
    );
    // Two 1M mut f64 arrays bridge 16 MB: the bound doubles with an 8-byte element, and an all-f32 call keeps 8 MB.
    let mut_f64 = |address| PlanArray {
        mutable: true,
        ..f64a(address, 1_000_000)
    };
    let plan = plan_bridges(&[mut_f64(base), mut_f64(base + 8_000_000)]).unwrap();
    assert_eq!(
        (plan.bridge_bytes, plan.core_copy_bytes),
        (16_000_000, 16_000_000)
    );
    let mut_f32 = |address| PlanArray {
        mutable: true,
        ..f32a(address, 1_000_000)
    };
    assert_eq!(
        plan_bridges(&[mut_f32(base), mut_f32(base + 4_000_000)])
            .unwrap()
            .bridge_bytes,
        BRIDGE_LIMIT_BYTES
    );
    // Touching extents of different element sizes count separately; overlapping ones are refused.
    let plan = plan_bridges(&[bool_a(base, 8), f64a(base + 8, 2)]).unwrap();
    assert_eq!(plan.live_elements, 10);
    assert_eq!(
        plan_bridges(&[bool_a(base, 16), f64a(base + 8, 1)]),
        Err(PlanError::Alias)
    );
    assert_eq!(
        plan_bridges(&[
            f64a(base, 4),
            PlanArray {
                mutable: true,
                ..f64a(base + 8, 1)
            }
        ]),
        Err(PlanError::Alias)
    );
}

/// The binding refuses before any bridge exists at the HA-1 protocol bounds
/// (frozen in docs/host-array-arguments-ha1.md): they are not the language
/// limits, which were raised separately, and must change only with the protocol.
#[test]
fn ha1_bridge_limits_are_the_protocol_bounds() {
    use invar_foreign::host_array;
    assert_eq!(LIVE_ELEMENTS, 2_000_000);
    assert_eq!(LIVE_ELEMENTS, host_array::MAX_LIVE_ELEMENTS);
    assert_eq!(BRIDGE_LIMIT_BYTES, host_array::MAX_SNAPSHOT_BYTES);
    assert_eq!(TEMPORARY_LIMIT_BYTES, 40_000_000);
    assert_eq!(
        TEMPORARY_LIMIT_BYTES,
        host_array::MAX_ARRAYS * host_array::MAX_ELEMENTS * ELEMENT_BYTES
            + host_array::MAX_SNAPSHOT_BYTES
    );
}

const BRIDGE_SOURCE: &str = "fn mix(a: mut Array<f32>, b: mut Array<f32>, c: mut Array<i32>, r: read Array<f32>, s: read Array<f32>, k: i32) -> Result<i32, u32> { a[0u32] = r[0u32]; a[1u32] = 0.0 * (0.0 - 1.0); b[0u32] = s[1u32]; c[0u32] = k; if k < 0 { return Result::Err(7u32); } return Result::Ok(c[1u32] / k); }\nfn near_f(a: read Array<f32>, b: read Array<f32>) -> u32 { let t = filled_u32(1u32, 0u32); return len(a) + len(b) + len(t); }\nfn near_i(a: read Array<i32>, b: read Array<i32>) -> u32 { let t = filled_u32(1u32, 0u32); return len(a) + len(b) + len(t); }\nfn three_f(a: read Array<f32>, b: read Array<f32>, c: read Array<f32>) -> u32 { return len(a) + len(b) + len(c); }\nfn three_i(a: read Array<i32>, b: read Array<i32>, c: read Array<i32>) -> u32 { return len(a) + len(b) + len(c); }";

fn owned_f32<'py>(py: Python<'py>, bits: &[u32]) -> Bound<'py, numpy::PyArray1<f32>> {
    let array = numpy::PyArray1::<f32>::zeros(py, bits.len(), false);
    for (slot, b) in array
        .readwrite()
        .as_slice_mut()
        .unwrap()
        .iter_mut()
        .zip(bits)
    {
        *slot = f32::from_bits(*b);
    }
    array
}

fn bits(array: &Bound<'_, numpy::PyArray1<f32>>) -> Vec<u32> {
    array
        .readonly()
        .as_slice()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

#[test]
fn ha1_bridge_failures_leave_every_caller_array_unchanged() {
    Python::attach(|py| {
        let module = compile_source(
            py,
            PyString::new(py, BRIDGE_SOURCE).as_any(),
            "bridge.inv",
            "reference",
            None,
        )
        .unwrap();
        let a = owned_f32(py, &[0x3f80_0000, 0x4000_0000]);
        let b = owned_f32(py, &[0x7f80_0001, 0x8000_0000]);
        let c = owned(py, &[5, 6]);
        let r = owned_f32(py, &[0x7fa0_0001, 0xffc1_2345]);
        let call = |k: i32, inject: BindingInject| {
            let args = PyTuple::new(
                py,
                [
                    a.as_any().clone(),
                    b.as_any().clone(),
                    c.as_any().clone(),
                    r.as_any().clone(),
                    r.as_any().clone(),
                    k.into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            module.reference_call(py, "mix", &args, Fuel(1000), inject)
        };
        let before = (
            bits(&a),
            bits(&b),
            c.readonly().as_slice().unwrap().to_vec(),
            bits(&r),
        );
        let unchanged = |label: &str| {
            assert_eq!(
                (
                    bits(&a),
                    bits(&b),
                    c.readonly().as_slice().unwrap().to_vec(),
                    bits(&r)
                ),
                before,
                "{label}"
            );
            for array in [&a, &b] {
                assert!(array.try_readwrite().is_ok(), "{label}: guard released");
            }
        };
        // Bridges: one f32 union (r twice) + three mut bridges.
        for n in 0..4 {
            let error = call(2, BindingInject::BridgeReserveFailAt(n)).unwrap_err();
            assert!(error.is_instance_of::<ResourceError>(py));
            assert_eq!(
                code_and_steps(py, &error),
                ("PY_STAGING_LIMIT".into(), 0),
                "reservation {n}"
            );
            unchanged(&format!("reservation {n}"));
        }
        // The core returned normally and committed into the bridges, but
        // building the Python result failed: no caller array is written.
        let error = call(2, BindingInject::ResultBuildFail).unwrap_err();
        assert!(error.is_instance_of::<pyo3::exceptions::PyMemoryError>(py));
        unchanged("result construction failure");
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            call(2, BindingInject::PanicBeforeWriteback)
        }));
        assert!(panicked.is_err());
        unchanged("panic before writeback");
        // A trap after writes leaves everything unchanged.
        let error = call(0, BindingInject::None).unwrap_err();
        assert!(error.is_instance_of::<TrapError>(py));
        unchanged("trap");
        // Normal return: every mut buffer committed with exact bits.
        let document = call(2, BindingInject::None).unwrap();
        let result = document.bind(py).get_item("result").unwrap();
        assert_eq!(
            result
                .get_item("variant")
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "Ok"
        );
        assert_eq!(bits(&a), vec![0x7fa0_0001, 0x8000_0000]);
        assert_eq!(bits(&b), vec![0xffc1_2345, 0x8000_0000]);
        assert_eq!(c.readonly().as_slice().unwrap(), &[2, 6]);
        assert_eq!(bits(&r), before.3);
        // Result::Err is a normal, committing return.
        let document = call(-4, BindingInject::None).unwrap();
        let result = document.bind(py).get_item("result").unwrap();
        assert_eq!(
            result
                .get_item("variant")
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "Err"
        );
        assert_eq!(c.readonly().as_slice().unwrap(), &[-4, 6]);
    });
}

#[test]
fn ha1_f32_live_union_matches_i32_and_exact_aliases_count_once() {
    Python::attach(|py| {
        let module = compile_source(
            py,
            PyString::new(py, BRIDGE_SOURCE).as_any(),
            "bridge.inv",
            "reference",
            None,
        )
        .unwrap();
        let call = |name: &str, args: &[&Bound<'_, PyAny>]| {
            let args = PyTuple::new(py, args.iter().map(|a| (*a).clone())).unwrap();
            module.reference_call(py, name, &args, Fuel(1000), BindingInject::None)
        };
        let result = |name: &str, args: &[&Bound<'_, PyAny>]| {
            call(name, args)
                .unwrap()
                .bind(py)
                .get_item("result")
                .unwrap()
                .extract::<u32>()
                .unwrap()
        };
        let f = numpy::PyArray1::<f32>::zeros(py, 1_000_000, false);
        let g = numpy::PyArray1::<f32>::zeros(py, 1_000_000, false);
        let h = numpy::PyArray1::<f32>::zeros(py, 1_000_000, false);
        let i = numpy::PyArray1::<i32>::zeros(py, 1_000_000, false);
        let j = numpy::PyArray1::<i32>::zeros(py, 1_000_000, false);
        let k = numpy::PyArray1::<i32>::zeros(py, 1_000_000, false);
        // The callee temporary is the core's allocation, bounded by the
        // language limit (raised separately), not by the host-array bound.
        for (name, x, y) in [
            ("near_f", f.as_any(), g.as_any()),
            ("near_i", i.as_any(), j.as_any()),
        ] {
            assert_eq!(result(name, &[x, y]), 2_000_001);
        }
        // Exact alias: counted once toward the 2M host live union.
        assert_eq!(
            result("three_f", &[f.as_any(), f.as_any(), g.as_any()]),
            3_000_000
        );
        assert_eq!(
            result("three_i", &[i.as_any(), i.as_any(), j.as_any()]),
            3_000_000
        );
        // Three distinct arrays: 3M live exceeds the host bound before
        // execution, identically for f32 and i32.
        let fe = call("three_f", &[f.as_any(), g.as_any(), h.as_any()]).unwrap_err();
        let ie = call("three_i", &[i.as_any(), j.as_any(), k.as_any()]).unwrap_err();
        assert_eq!(code_and_steps(py, &fe), code_and_steps(py, &ie));
        assert_eq!(code_and_steps(py, &fe), ("ARRAY_SIZE_LIMIT".to_string(), 0));
    });
}

#[test]
fn issue_ai_01_machine_diagnostics_python_invalid_checked_signature_fault_is_typed() {
    Python::attach(|py| {
        for invalid_parameter in [false, true] {
            let source = PyBytes::new(py, b"fn main() -> i32 { return 1; }");
            let mut module =
                compile_source(py, source.as_any(), "<fault>", "reference", None).unwrap();
            // Fault injection only: valid checked L1-04 cannot contain these signatures.
            module
                .checked
                .program
                .foreign_functions
                .push(invar::front::ForeignFunction {
                    name: "invalid_import".to_owned(),
                    params: if invalid_parameter {
                        vec![invar::front::Param {
                            name: "value".to_owned(),
                            mode: None,
                            ty: Type::Bool,
                            span: invar::front::Span { start: 0, end: 0 },
                        }]
                    } else {
                        vec![]
                    },
                    result: Type::Unit,
                    oracle: "invalid".to_owned(),
                    span: invar::front::Span { start: 0, end: 0 },
                });
            let args = if invalid_parameter {
                PyTuple::new(py, [1]).unwrap()
            } else {
                PyTuple::empty(py)
            };
            // This existing text file must never reach the library loader.
            let library = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
            let error = module
                .call_foreign(py, "invalid_import", library.to_str().unwrap(), &args)
                .unwrap_err();
            assert!(error.is_instance_of::<EngineError>(py));
            let code: String = error
                .value(py)
                .getattr("diagnostics")
                .unwrap()
                .get_item("diagnostics")
                .unwrap()
                .get_item(0)
                .unwrap()
                .get_item("code")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(code, "PY_ABI_TYPE_UNAVAILABLE");
            let document: Value = serde_json::from_str(
                &error
                    .value(py)
                    .getattr("diagnostic_json")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
            )
            .unwrap();
            let schema: Value = serde_json::from_str(include_str!(
                "../../../../schema/invar-diagnostics-v3.schema.json"
            ))
            .unwrap();
            assert!(jsonschema::is_valid(&schema, &document));
            assert_eq!(document["diagnostics"][0]["suggested_fixes"], json!([]));
        }
    });
}
