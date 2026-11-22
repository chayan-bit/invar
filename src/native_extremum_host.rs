// Part of the standalone native host (`src/native_host.rs`), appended by
// `native::build`: the #207 extremum intrinsics over host-owned arrays.

/// `min_of`/`max_of`/`argmin`/`argmax` (#207) over a host-owned array;
/// `elem` is 1 `i32`, 2 `u32`, 3 `f32` words, 4 `f64` encodings. An empty
/// input traps INDEX_OUT_OF_BOUNDS at the call span.
#[unsafe(no_mangle)]
pub extern "C" fn invar_extremum_v1(kind: u32, elem: u32, a: u64, span_start: u32, span_end: u32) -> i64 {
    use extremum::{F32Word, F64Word, Kind, Outcome, reference, reference_iter};
    let result = RUNTIME.with(|cell| {
        let runtime = cell.borrow();
        let kind = Kind::from_code(kind).ok_or(1u32)?;
        let index = |i: u32| i64::from(i);
        Ok(match (elem, array(&runtime, a)) {
            (1, Some(Array::I32(v))) => reference(kind, v).map(|o| match o {
                Outcome::Index(i) => index(i),
                Outcome::Value(x) => i64::from(x),
            }),
            (2, Some(Array::U32(v))) => reference(kind, v).map(|o| match o {
                Outcome::Index(i) => index(i),
                Outcome::Value(x) => i64::from(x),
            }),
            (3, Some(Array::U32(v))) => {
                reference_iter(kind, v.iter().map(|&w| F32Word(w))).map(|o| match o {
                    Outcome::Index(i) => index(i),
                    Outcome::Value(x) => i64::from(x.0),
                })
            }
            (4, Some(Array::F64(v))) => {
                reference_iter(kind, v.iter().map(|&w| F64Word(w as u64))).map(|o| match o {
                    Outcome::Index(i) => index(i),
                    Outcome::Value(x) => x.0 as i64,
                })
            }
            _ => return Err(1u32),
        })
    });
    match result {
        Ok(Some(value)) => value,
        Ok(None) => trap(3, span_start, span_end),
        Err(code) => tool(code),
    }
}
