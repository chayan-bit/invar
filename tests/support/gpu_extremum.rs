//! #207 on a device engine: `min_of`, `max_of`, `argmin` and `argmax` have no
//! device lowering. A program whose only candidate is such a call is refused
//! with `<BACKEND>_UNAVAILABLE`; in a program with a device region the call
//! runs serially on the host, and value, trap, fuel and cleanup equal the
//! reference evaluator. Shared by the CUDA and Metal binaries; the including
//! module supplies `BACKEND` and `run_device` (`Ok` with the observation and
//! the number of published device regions, or `Err` with the error code).
use super::{BACKEND, run_device};
use invar::{check, eval, front, mir};

const FUEL: u64 = 50_000_000;

/// Floats as `main` returns them: NaN -1, -0 -2, +0 -3, else 4x the value.
const CLASS_F32: &str = "fn class(r: f32) -> i32 { if r != r { return -1; } \
    if r == 0.0 { if 1.0 / r < 0.0 { return -2; } return -3; } return i32_from_f32(r * 4.0); }";
const CLASS_U32: &str = "fn class(r: u32) -> i32 { return i32_from_f32(f32_from_u32(r)); }";

/// `n` elements `i % 13 + 3` from a device map, then `call`.
fn with_region(n: u32, call: &str, class: &str) -> String {
    format!(
        "{class} fn main() -> i32 {{ let nan: f32 = 0.0 / 0.0; \
         var xs: Array<u32> = filled_u32({n}u32, 0u32); \
         for i in 0u32 .. {n}u32 {{ xs[i] = i % 13u32; }} \
         let base: Array<u32> = move xs; \
         var a: Array<u32> = filled_u32(len(base), 0u32); \
         for j in 0u32 .. len(base) {{ a[j] = wrapping_add(base[j], 3u32); }} \
         let ys: Array<u32> = move a; \
         let fs: Array<f32> = [2.0, nan, -0.0, 0.0, -0.0, 2.0, nan]; \
         let nans: Array<f32> = [nan, nan]; \
         let none: Array<f32> = filled_f32(0u32, 0.0); \
         let r = {call}; return class(r); }}"
    )
}

fn observe(source: &str) -> (eval::Observation, mir::Program) {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    (eval::observe(&checked, FUEL), program)
}

#[test]
#[ignore = "requires a GPU (CUDA with NVRTC, or Apple Metal)"]
fn issue_207_extremum_calls_agree_with_the_reference_beside_device_regions() {
    // (call, class, result for n >= 13, result for n = 1); None is the empty trap.
    let cases: [(&str, &str, Option<i32>, Option<i32>); 11] = [
        ("min_of(ys)", CLASS_U32, Some(3), Some(3)),
        ("max_of(ys)", CLASS_U32, Some(15), Some(3)),
        ("argmin(ys)", CLASS_U32, Some(0), Some(0)),
        ("argmax(ys)", CLASS_U32, Some(12), Some(0)),
        ("min_of(fs)", CLASS_F32, Some(-2), Some(-2)),
        ("max_of(fs)", CLASS_F32, Some(8), Some(8)),
        ("argmin(fs)", CLASS_U32, Some(2), Some(2)),
        ("argmax(fs)", CLASS_U32, Some(0), Some(0)),
        ("min_of(nans)", CLASS_F32, Some(-1), Some(-1)),
        ("argmax(nans)", CLASS_U32, Some(0), Some(0)),
        ("argmin(none)", CLASS_U32, None, None),
    ];
    let mut runs = 0;
    for n in [1u32, 4093, 100_003] {
        for (call, class, large, single) in cases {
            let source = with_region(n, call, class);
            let (reference, program) = observe(&source);
            let expected = if n == 1 { single } else { large };
            match expected {
                Some(value) => assert_eq!(
                    reference.result,
                    Ok(eval::Value::I32(value)),
                    "{call} n={n}"
                ),
                None => assert!(
                    matches!(&reference.result, Err(eval::EvalError::Trap { .. })),
                    "{call} n={n}: {:?}",
                    reference.result
                ),
            }
            let (observation, published) = run_device(source.as_bytes(), &program, FUEL)
                .unwrap_or_else(|code| panic!("{BACKEND} {call} n={n}: {code}"));
            assert!(
                published > 0,
                "{BACKEND} {call} n={n}: no device region ran"
            );
            assert_eq!(
                observation.result, reference.result,
                "{BACKEND} {call} n={n}"
            );
            assert_eq!(
                observation.steps_used, reference.steps_used,
                "{BACKEND} {call} n={n}"
            );
            assert_eq!(
                observation.cleanup_trace, reference.cleanup_trace,
                "{BACKEND} {call} n={n}"
            );
            runs += 1;
        }
    }
    assert_eq!(runs, 33);
}

#[test]
#[ignore = "requires a GPU (CUDA with NVRTC, or Apple Metal)"]
fn issue_207_extremum_only_program_is_refused_on_the_device() {
    for call in ["min_of(xs)", "max_of(xs)", "argmin(xs)", "argmax(xs)"] {
        let class = if call.starts_with("arg") {
            CLASS_U32
        } else {
            CLASS_F32
        };
        let source = format!(
            "{class} fn main() -> i32 {{ let xs: Array<f32> = [1.0, -0.0, 0.0]; let r = {call}; return class(r); }}"
        );
        let (_, program) = observe(&source);
        let code = run_device(source.as_bytes(), &program, FUEL).err();
        let expected = format!("{}_UNAVAILABLE", BACKEND.to_uppercase());
        assert_eq!(code.as_deref(), Some(expected.as_str()), "{BACKEND} {call}");
    }
}
