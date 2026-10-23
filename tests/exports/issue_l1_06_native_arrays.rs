//! HA-1.2 array parameters of selected exports: admission, the copying
//! reference oracle, and (with `native`) zero-copy native-serial execution
//! at the caller's addresses with rollback on every non-normal outcome.
use invar::{
    eval::{
        EvalError, TrapCategory,
        host_read::{HostArgument, HostMutArray, HostReadArray},
    },
    exports::{
        ArrayMode, CompiledExports, ExportEngine, ExportError, ScalarType, ScalarValue,
        compile_exports,
    },
};

const SOURCE: &str = "\
fn sum(xs: read Array<i32>) -> i32 { var acc: i32 = 0; for i in 0u32..len(xs) { acc = acc + xs[i]; } return acc; }
fn scale(xs: read Array<u32>, out: mut Array<u32>, k: u32) -> u32 { for i in 0u32..len(xs) { out[i] = xs[i] * k; } return len(out); }
fn poke(xs: mut Array<i32>, i: u32, v: i32) -> i32 { xs[0u32] = v; xs[i] = v + 1; return xs[0u32]; }
fn count(xs: read Array<i32>) -> u32 { return len(xs); }
fn first(xs: read Array<i32>) -> i32 { return xs[0u32]; }
fn pair(a: read Array<i32>, b: read Array<i32>) -> i32 { return a[0u32] + b[len(b) - 1u32]; }
fn twice(x: i32) -> i32 { return x * 2; }
fn spin(xs: mut Array<u32>) -> u32 { for i in 0u32..len(xs) { xs[i] = xs[i] + 1u32; } var n: u32 = 0u32; for j in 0u32..1000u32 { n = wrapping_add(n, j); } return n; }";
const NAMES: [&str; 8] = [
    "sum", "scale", "poke", "count", "first", "pair", "twice", "spin",
];
const PATH: &str = "arrays.inv";

fn engines() -> Vec<ExportEngine> {
    let mut engines = vec![ExportEngine::Reference];
    if cfg!(feature = "native") {
        engines.push(ExportEngine::NativeSerial);
    }
    engines
}

fn artifact(engine: ExportEngine) -> CompiledExports {
    compile_exports(SOURCE.as_bytes(), PATH, &NAMES, engine).unwrap()
}

fn tool_code(error: &ExportError) -> &'static str {
    match error {
        ExportError::Boundary { code, .. }
        | ExportError::Evaluation(EvalError::Tool { code, .. }) => code,
        other => panic!("expected a typed tool refusal, got {other:?}"),
    }
}

/// One call's observable outcome: result or error, and every mut array after.
type Observed = (Result<ScalarValue, String>, Vec<i32>, Vec<u32>);

fn observe(
    artifact: &mut CompiledExports,
    name: &str,
    fuel: u64,
    xs: &[i32],
    us: &[u32],
    scalars: (u32, i32),
) -> Observed {
    let mut signed = xs.to_vec();
    let mut unsigned = us.to_vec();
    let input_u = us.to_vec();
    let mut args: Vec<HostArgument<'_>> = match name {
        "sum" | "count" | "first" => vec![HostArgument::Read(HostReadArray::I32(&signed))],
        "pair" => vec![
            HostArgument::Read(HostReadArray::I32(&signed)),
            HostArgument::Read(HostReadArray::I32(&signed)),
        ],
        "scale" => vec![
            HostArgument::Read(HostReadArray::U32(&input_u)),
            HostArgument::Mut(HostMutArray::U32(&mut unsigned)),
            HostArgument::U32(scalars.0),
        ],
        "poke" => vec![
            HostArgument::Mut(HostMutArray::I32(&mut signed)),
            HostArgument::U32(scalars.0),
            HostArgument::I32(scalars.1),
        ],
        "spin" => vec![HostArgument::Mut(HostMutArray::U32(&mut unsigned))],
        _ => vec![HostArgument::I32(scalars.1)],
    };
    let outcome = artifact
        .invoke_arrays(name, &mut args, fuel)
        .map(|outcome| outcome.value)
        .map_err(|error| format!("{error:?}"));
    drop(args);
    (outcome, signed, unsigned)
}

#[test]
fn issue_l1_06_native_arrays_descriptors_and_admission() {
    for engine in engines() {
        let artifact = artifact(engine);
        let scale = &artifact.signatures()[1];
        let modes = scale
            .params
            .iter()
            .map(|p| (p.ty, p.array))
            .collect::<Vec<_>>();
        assert_eq!(
            modes,
            vec![
                (ScalarType::U32, Some(ArrayMode::Read)),
                (ScalarType::U32, Some(ArrayMode::Mut)),
                (ScalarType::U32, None),
            ]
        );
        for refused in [
            "fn f(xs: take Array<i32>) -> i32 { return 0; }",
            "fn f(xs: read Array<i32>) -> Array<i32> { let ys = filled_i32(1u32, 0); return ys; }",
            "fn f(xs: read Array<i32>) -> i32 { let ys = filled_i32(1u32, 0); return ys[0u32]; }",
            "fn g(xs: read Array<i32>) -> i32 { return xs[0u32]; } fn f(xs: read Array<i32>) -> i32 { return g(xs); }",
            "fn f(xs: read Array<i32>) -> i32 { let ys = xs; return ys[0u32]; }",
            "fn f(xs: read Array<i32>) -> f32 { return 1.0; }",
        ] {
            let error = compile_exports(refused.as_bytes(), PATH, &["f"], engine)
                .err()
                .unwrap_or_else(|| panic!("admitted {refused}"));
            assert!(
                matches!(
                    error,
                    ExportError::Boundary {
                        code: "NATIVE_ABI",
                        ..
                    } | ExportError::Source(_)
                ),
                "{refused}: {error:?}"
            );
        }
        // The scalar entry point never accepts an array descriptor.
        let mut artifact = artifact;
        assert_eq!(
            tool_code(
                &artifact
                    .invoke("sum", &[ScalarValue::I32(1)], 100)
                    .unwrap_err()
            ),
            "NATIVE_ABI"
        );
        // Wrong kinds and arity are typed refusals before any call.
        let xs = [1i32];
        let mut wrong = [HostArgument::Read(HostReadArray::U32(&[1u32]))];
        assert_eq!(
            tool_code(&artifact.invoke_arrays("sum", &mut wrong, 100).unwrap_err()),
            "NATIVE_ABI"
        );
        let mut extra = [
            HostArgument::Read(HostReadArray::I32(&xs)),
            HostArgument::I32(1),
        ];
        assert_eq!(
            tool_code(&artifact.invoke_arrays("sum", &mut extra, 100).unwrap_err()),
            "NATIVE_ABI"
        );
        let mut ys = [1i32];
        let mut mut_for_read = [HostArgument::Mut(HostMutArray::I32(&mut ys))];
        assert_eq!(
            tool_code(
                &artifact
                    .invoke_arrays("sum", &mut mut_for_read, 100)
                    .unwrap_err()
            ),
            "NATIVE_ABI"
        );
        let mut ok = [HostArgument::Read(HostReadArray::I32(&xs))];
        assert_eq!(
            tool_code(&artifact.invoke_arrays("missing", &mut ok, 100).unwrap_err()),
            "NATIVE_ABI"
        );
        assert_eq!(
            tool_code(&artifact.invoke_arrays("sum", &mut ok, 0).unwrap_err()),
            "INVALID_FUEL"
        );
    }
}

/// Name, i32 array, u32 array and the (u32, i32) scalar pair of one call.
type Case = (&'static str, Vec<i32>, Vec<u32>, (u32, i32));

#[test]
fn issue_l1_06_native_arrays_match_the_reference_oracle_on_every_fuel_prefix() {
    let cases: Vec<Case> = vec![
        ("sum", vec![], vec![], (0, 0)),
        ("sum", vec![1, -2, 30, i32::MAX], vec![], (0, 0)),
        ("sum", vec![5, 6, 7], vec![], (0, 0)),
        ("scale", vec![], vec![1, 2, 3], (7, 0)),
        ("scale", vec![], vec![1, u32::MAX], (2, 0)),
        ("scale", vec![], vec![], (2, 0)),
        ("poke", vec![10, 20, 30], vec![], (2, -4)),
        ("poke", vec![10, 20, 30], vec![], (3, -4)),
        ("poke", vec![10], vec![], (0, i32::MAX)),
        ("poke", vec![], vec![], (0, 1)),
        ("count", vec![], vec![], (0, 0)),
        ("count", vec![4, 5], vec![], (0, 0)),
        ("first", vec![], vec![], (0, 0)),
        ("first", vec![-9], vec![], (0, 0)),
        ("pair", vec![3, 4, 5], vec![], (0, 0)),
        ("twice", vec![], vec![], (0, 21)),
        ("spin", vec![], vec![0, u32::MAX - 1, 5], (0, 0)),
    ];
    let mut reference = artifact(ExportEngine::Reference);
    #[cfg(feature = "native")]
    let mut native = artifact(ExportEngine::NativeSerial);
    #[cfg_attr(not(feature = "native"), allow(unused_mut))]
    let mut compared = 0;
    for (name, xs, us, scalars) in &cases {
        let full = observe(&mut reference, name, 1_000_000, xs, us, *scalars);
        if full.0.is_err() {
            assert_eq!(
                (&full.1, &full.2),
                (xs, us),
                "{name}: failure must not publish"
            );
        }
        let mut prefixes = vec![1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 1_000_000];
        prefixes.extend(1..40);
        for fuel in prefixes {
            let expected = observe(&mut reference, name, fuel, xs, us, *scalars);
            if expected.0.is_err() {
                assert_eq!((&expected.1, &expected.2), (xs, us), "{name} fuel {fuel}");
            }
            #[cfg(feature = "native")]
            {
                let actual = observe(&mut native, name, fuel, xs, us, *scalars);
                assert_eq!(actual, expected, "{name} fuel {fuel}");
                compared += 1;
            }
        }
    }
    eprintln!(
        "HA-1.2 parity: {} cases, {compared} native/reference comparisons",
        cases.len()
    );
    assert!(!cfg!(feature = "native") || compared > 0);
}

#[test]
fn issue_l1_06_native_arrays_traps_carry_the_original_span() {
    for engine in engines() {
        let mut artifact = artifact(engine);
        let mut xs = vec![1i32, 2, 3];
        let mut args = [
            HostArgument::Mut(HostMutArray::I32(&mut xs)),
            HostArgument::U32(3),
            HostArgument::I32(9),
        ];
        let error = artifact.invoke_arrays("poke", &mut args, 1000).unwrap_err();
        let ExportError::Evaluation(EvalError::Trap { category, span }) = error else {
            panic!("{engine:?}: expected a trap, got {error:?}");
        };
        assert_eq!(category, TrapCategory::IndexOutOfBounds);
        assert_eq!(&SOURCE[span.start..span.end], "xs[i]");
        // The write to xs[0] before the trap was rolled back (native) or never
        // published (reference): the caller's bits are unchanged.
        assert_eq!(xs, [1, 2, 3], "{engine:?}");
        // The artifact stays callable after a trap.
        let ys = [4i32, 5];
        let mut args = [HostArgument::Read(HostReadArray::I32(&ys))];
        assert_eq!(
            artifact
                .invoke_arrays("sum", &mut args, 1000)
                .unwrap()
                .value,
            ScalarValue::I32(9)
        );
    }
}

#[test]
fn issue_l1_06_native_arrays_size_limits_refuse_before_any_step() {
    for engine in engines() {
        let mut artifact = artifact(engine);
        // The private HA-1 native host protocol keeps its documented tool bound.
        let max = match engine {
            ExportEngine::Reference => invar::array_limits::MAX_ARRAY_LEN,
            ExportEngine::NativeSerial => invar_foreign::host_array::MAX_ELEMENTS,
        };
        let mut big = vec![7u32; max + 1];
        let small = vec![1u32; 1];
        let mut args = [
            HostArgument::Read(HostReadArray::U32(&small)),
            HostArgument::Mut(HostMutArray::U32(&mut big)),
            HostArgument::U32(2),
        ];
        let error = artifact
            .invoke_arrays("scale", &mut args, 1000)
            .unwrap_err();
        assert_eq!(tool_code(&error), "ARRAY_SIZE_LIMIT", "{engine:?}");
        assert!(big.iter().all(|value| *value == 7));
        // The exact current cap is admitted without traversing every element.
        let at_cap = vec![1i32; max];
        let mut args = [HostArgument::Read(HostReadArray::I32(&at_cap))];
        assert_eq!(
            artifact
                .invoke_arrays("count", &mut args, 100)
                .unwrap()
                .value,
            ScalarValue::U32(u32::try_from(max).unwrap()),
            "{engine:?}"
        );
        // A large admitted array round-trips.
        let input = (0..1_000_000u32).collect::<Vec<_>>();
        let mut output = vec![0u32; 1_000_000];
        let mut args = [
            HostArgument::Read(HostReadArray::U32(&input)),
            HostArgument::Mut(HostMutArray::U32(&mut output)),
            HostArgument::U32(3),
        ];
        let outcome = artifact
            .invoke_arrays("scale", &mut args, 10_000_000)
            .unwrap();
        assert_eq!(outcome.value, ScalarValue::U32(1_000_000));
        assert!(
            output
                .iter()
                .zip(&input)
                .all(|(o, i)| *o == i.wrapping_mul(3))
        );
    }
}

#[cfg(feature = "native")]
#[test]
fn issue_l1_06_native_arrays_zero_copy_same_address_and_rollback() {
    let mut native = artifact(ExportEngine::NativeSerial);
    let input = vec![1u32, 2, 3, 4];
    let mut output = vec![0u32; 4];
    let (input_address, output_address) = (input.as_ptr().addr(), output.as_ptr().addr());
    let mut args = [
        HostArgument::Read(HostReadArray::U32(&input)),
        HostArgument::Mut(HostMutArray::U32(&mut output)),
        HostArgument::U32(10),
    ];
    use invar::exports::ArrayCallOutcome;
    let ArrayCallOutcome {
        value,
        zero_copy,
        record_addresses,
    } = native.invoke_arrays("scale", &mut args, 1000).unwrap();
    // Same buffer on both sides: the private records handed to native code
    // carry the caller's own addresses, and the results are in that buffer.
    assert_eq!(value, ScalarValue::U32(4));
    assert!(zero_copy);
    assert_eq!(record_addresses, vec![input_address, output_address]);
    assert_eq!(output, [10, 20, 30, 40]);
    assert_eq!(output.as_ptr().addr(), output_address);
    // Empty arrays: the record carries NULL (address 0); nothing is touched.
    let empty: Vec<i32> = Vec::new();
    let mut args = [HostArgument::Read(HostReadArray::I32(&empty))];
    let outcome = native.invoke_arrays("count", &mut args, 100).unwrap();
    assert_eq!(
        (outcome.value, outcome.record_addresses),
        (ScalarValue::U32(0), vec![0])
    );
    // Exact read/read aliases are admitted and see one buffer.
    let shared = vec![3i32, 4, 5];
    let mut args = [
        HostArgument::Read(HostReadArray::I32(&shared)),
        HostArgument::Read(HostReadArray::I32(&shared)),
    ];
    let outcome = native.invoke_arrays("pair", &mut args, 100).unwrap();
    assert_eq!(outcome.value, ScalarValue::I32(8));
    assert_eq!(outcome.record_addresses, vec![shared.as_ptr().addr(); 2]);
    // Fuel exhaustion after in-place writes restores the exact pre-call bits;
    // the reference copies and agrees.
    let mut reference = artifact(ExportEngine::Reference);
    for artifact in [&mut native, &mut reference] {
        let mut xs = vec![0u32, 7, u32::MAX - 1];
        let mut args = [HostArgument::Mut(HostMutArray::U32(&mut xs))];
        let error = artifact.invoke_arrays("spin", &mut args, 40).unwrap_err();
        assert!(matches!(
            error,
            ExportError::Evaluation(EvalError::EvaluationLimit {
                fuel_limit: 40,
                steps_used: 40
            })
        ));
        assert_eq!(xs, [0, 7, u32::MAX - 1]);
    }
}

#[test]
fn issue_l1_06_native_arrays_reference_is_disclosed_as_copying() {
    let mut reference = artifact(ExportEngine::Reference);
    let xs = [1i32, 2];
    let mut args = [HostArgument::Read(HostReadArray::I32(&xs))];
    let outcome = reference.invoke_arrays("sum", &mut args, 100).unwrap();
    assert_eq!(outcome.value, ScalarValue::I32(3));
    assert!(!outcome.zero_copy);
    assert!(outcome.record_addresses.is_empty());
}
