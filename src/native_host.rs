// Compiled as a separate Rust host by `invar build`. Its only unsafe operation
// is calling the fixed, LLVM-verified C ABI entry point linked from program.o.
use std::{cell::RefCell, process};

unsafe extern "C" {
    fn invar_entry_v1() -> i64;
}

// Allocation limits are injected by native::build alongside NAMES and BLOBS.
const MAX_CLEANUP_EVENTS: usize = 512;
const MAX_LIVE_VIEWS: usize = 4096;

enum Array {
    I32(Vec<i32>),
    U32(Vec<u32>),
    /// binary64 encodings (kind 4); no views.
    F64(Vec<i64>),
    Bytes(Vec<u8>),
    Text(String),
    Object(Vec<Option<(i64, bool)>>),
    #[cfg(invar_has_spatial)]
    Grid(invar_spatial::Grid),
    View { owner: u64, start: u32, len: u32, stride: u32, mutable: bool },
}

impl Array {
    fn len(&self) -> usize {
        match self {
            Self::I32(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::Bytes(values) => values.len(),
            Self::Text(value) => value.len(),
            Self::Object(values) => values.len(),
            #[cfg(invar_has_spatial)]
            Self::Grid(grid) => grid.storage_elements(),
            Self::View { len, .. } => *len as usize,
        }
    }
}

struct Runtime {
    fuel: u64,
    steps: u64,
    depth: usize,
    arrays: Vec<Option<Array>>,
    live_elements: usize,
    live_views: usize,
    cleanup: Vec<(usize, usize)>,
}

thread_local! {
    static RUNTIME: RefCell<Runtime> = const { RefCell::new(Runtime {
        fuel: 0,
        steps: 0,
        depth: 0,
        arrays: Vec::new(),
        live_elements: 0,
        live_views: 0,
        cleanup: Vec::new(),
    }) };
}

fn cleanup_json(runtime: &Runtime) -> String {
    runtime
        .cleanup
        .iter()
        .map(|(name, length)| format!("{{\"name\":{:?},\"length\":{length}}}", NAMES[*name]))
        .collect::<Vec<_>>()
        .join(",")
}

fn report(status: &str, extra: &str, code: i32, schema_version: u8) -> ! {
    RUNTIME.with(|cell| {
        if let Ok(runtime) = cell.try_borrow() {
            println!(
                "{{\"schema_version\":{},\"status\":\"{status}\",\"diagnostics\":[],{extra},\"steps_used\":{},\"cleanup_trace\":[{}]}}",
                schema_version,
                runtime.steps,
                cleanup_json(&runtime)
            );
        } else {
            println!("{{\"schema_version\":1,\"status\":\"tool_error\",\"diagnostics\":[],\"tool_code\":\"NATIVE_ABI\",\"steps_used\":0,\"cleanup_trace\":[]}}");
        }
    });
    process::exit(code)
}

fn tool(code: u32) -> ! {
    let name = match code {
        1 => "NATIVE_ABI",
        2 => "ARRAY_SIZE_LIMIT",
        3 => "RESOURCE_LIMIT",
        4 => "MIR_FOR",
        5 => "CALL_DEPTH_LIMIT",
        6 => "NATIVE_TRACE_LIMIT",
        7 => "TC_INPUT_RANGE",
        _ => "NATIVE_TOOL",
    };
    report("tool_error", &format!("\"tool_code\":\"{name}\""), 2, 1)
}

fn trap(category: u32, start: u32, end: u32) -> ! {
    invar_debug_trap_v1(category, start, end);
    let category = match category {
        1 => "INTEGER_OVERFLOW",
        2 => "DIVISION_BY_ZERO",
        3 => "INDEX_OUT_OF_BOUNDS",
        4 => "UNIT_CONVERSION_INEXACT",
        _ => tool(1),
    };
    report(
        "trap",
        &format!(
            "\"trap\":{{\"category\":\"{category}\",\"span\":{{\"start\":{start},\"end\":{end}}}}}"
        ),
        3,
        if category == "UNIT_CONVERSION_INEXACT" { 2 } else { 1 },
    )
}

fn array(runtime: &Runtime, handle: u64) -> Option<&Array> {
    runtime
        .arrays
        .get(handle.wrapping_sub(1) as usize)
        .and_then(Option::as_ref)
}

fn array_mut(runtime: &mut Runtime, handle: u64) -> Option<&mut Array> {
    runtime
        .arrays
        .get_mut(handle.wrapping_sub(1) as usize)
        .and_then(Option::as_mut)
}

fn physical(runtime: &Runtime, handle: u64, index: u64) -> Result<Option<(u64, usize)>, u32> {
    let source = array(runtime, handle).ok_or(1u32)?;
    let (owner, offset) = match source {
        Array::View { owner, start, len, stride, .. } => {
            if index >= u64::from(*len) { return Ok(None); }
            let offset = u64::from(*start).checked_add(index.checked_mul(u64::from(*stride)).ok_or(1u32)?).ok_or(1u32)?;
            (*owner, offset)
        }
        _ => (handle, index),
    };
    let offset = usize::try_from(offset).map_err(|_| 1u32)?;
    Ok((offset < array(runtime, owner).ok_or(1u32)?.len()).then_some((owner, offset)))
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_view_new_v1(handle: u64, start: u64, end: u64, stride: u64, mutable: u32, span_start: u32, span_end: u32) -> u64 {
    let result = RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        let source = array(&runtime, handle).ok_or(1u32)?;
        let length = source.len() as u64;
        if mutable > 1 { return Err(1); }
        if start > end || end > length || stride == 0 || stride > u32::MAX as u64 { return Ok(None); }
        let (owner, base, base_stride, allowed) = match source {
            Array::View { owner, start, stride, mutable, .. } => (*owner, u64::from(*start), u64::from(*stride), *mutable),
            Array::I32(_) | Array::U32(_) => (handle, 0, 1, true),
            Array::Object(_) | Array::F64(_) | Array::Bytes(_) | Array::Text(_) => return Err(1),
            #[cfg(invar_has_spatial)]
            Array::Grid(_) => return Err(1),
        };
        if mutable == 1 && !allowed { return Err(1); }
        let Some(step) = base_stride.checked_mul(stride).filter(|step| *step <= u32::MAX as u64) else { return Ok(None); };
        let Some(origin) = start.checked_mul(base_stride).and_then(|offset| base.checked_add(offset)).filter(|origin| *origin <= u32::MAX as u64) else { return Ok(None); };
        let len = (end - start) / stride + u64::from((end - start) % stride != 0);
        if len > u32::MAX as u64 { return Ok(None); }
        if len > 0 {
            let Some(last) = (len - 1).checked_mul(step).and_then(|offset| origin.checked_add(offset)) else { return Ok(None); };
            if last >= array(&runtime, owner).ok_or(1u32)?.len() as u64 { return Ok(None); }
        }
        if runtime.live_views >= MAX_LIVE_VIEWS || runtime.arrays.try_reserve(1).is_err() { return Err(3); }
        runtime.arrays.push(Some(Array::View { owner, start: origin as u32, len: len as u32, stride: step as u32, mutable: mutable == 1 }));
        runtime.live_views += 1;
        Ok(Some(runtime.arrays.len() as u64))
    });
    match result { Ok(Some(handle)) => handle, Ok(None) => trap(3, span_start, span_end), Err(code) => tool(code) }
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_view_drop_v1(handle: u64) {
    if handle == 0 { return; }
    let valid = RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        let Some(slot) = runtime.arrays.get_mut(handle.wrapping_sub(1) as usize) else { return false; };
        if !matches!(slot, Some(Array::View { .. })) { return false; }
        *slot = None;
        runtime.live_views -= 1;
        true
    });
    if !valid { tool(1); }
}

fn reserve_array(runtime: &mut Runtime, len: usize) -> Result<(), u32> {
    if len > MAX_ARRAY_LEN
        || runtime
            .live_elements
            .checked_add(len)
            .is_none_or(|n| n > MAX_LIVE_ELEMENTS)
    {
        return Err(2);
    }
    if runtime.arrays.try_reserve(1).is_err() {
        return Err(3);
    }
    runtime.live_elements += len;
    Ok(())
}

fn new_array(kind: u32, len: usize, fill: i64) -> u64 {
    let result = RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        reserve_array(&mut runtime, len)?;
        let array = match kind {
            1 => {
                let mut values = Vec::new();
                if values.try_reserve_exact(len).is_err() {
                    return Err(3);
                }
                values.resize(len, fill as i32);
                Array::I32(values)
            }
            2 => {
                let mut values = Vec::new();
                if values.try_reserve_exact(len).is_err() {
                    return Err(3);
                }
                values.resize(len, fill as u32);
                Array::U32(values)
            }
            3 => {
                let mut values = Vec::new();
                if values.try_reserve_exact(len).is_err() { return Err(3); }
                values.resize_with(len, || None);
                Array::Object(values)
            }
            4 => {
                let mut values = Vec::new();
                if values.try_reserve_exact(len).is_err() { return Err(3); }
                values.resize(len, fill);
                Array::F64(values)
            }
            _ => return Err(1),
        };
        runtime.arrays.push(Some(array));
        Ok(runtime.arrays.len() as u64)
    });
    result.unwrap_or_else(|code| tool(code))
}

fn copy_blob(bytes: &[u8]) -> Result<Vec<u8>, u32> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(bytes.len()).map_err(|_| 3u32)?;
    copy.extend_from_slice(bytes);
    Ok(copy)
}

fn insert_blob(value: Array) -> u64 {
    let length = value.len();
    let result = RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        if length > 1_048_576 || runtime.live_elements.checked_add(length).is_none_or(|used| used > MAX_LIVE_ELEMENTS) { return Err(2); }
        runtime.arrays.try_reserve(1).map_err(|_| 3u32)?;
        runtime.live_elements += length;
        runtime.arrays.push(Some(value));
        Ok(runtime.arrays.len() as u64)
    });
    result.unwrap_or_else(|code| tool(code))
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_tick_v1() {
    RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        if runtime.steps == runtime.fuel {
            let fuel = runtime.fuel;
            let steps = runtime.steps;
            drop(runtime);
            report(
                "evaluation_limit",
                &format!("\"fuel_limit\":{fuel},\"steps_used_at_limit\":{steps}"),
                4,
                1,
            );
        }
        runtime.steps += 1;
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_enter_v1() {
    RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        if runtime.depth == 64 {
            drop(runtime);
            tool(5);
        }
        runtime.depth += 1;
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_leave_v1() {
    RUNTIME.with(|cell| cell.borrow_mut().depth -= 1);
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_trap_v1(category: u32, start: u32, end: u32) {
    trap(category, start, end)
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_angle_convert_v1(value: i64, direction: u32, signed: u32, start: u32, end: u32) -> i64 {
    if direction > 1 || signed > 1 { tool(1); }
    match convert(value, signed != 0, direction != 0) {
        Ok(result) => result,
        Err(AngleError::Overflow) => trap(1, start, end),
        Err(AngleError::Ambiguous) => tool(1),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_tool_v1(code: u32) {
    tool(code)
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_array_new_v1(kind: u32, len: u32) -> u64 {
    new_array(kind, len as usize, 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_blob_new_v1(id: u32) -> u64 {
    let Some((bytes, text)) = BLOBS.get(id as usize) else { tool(1); };
    if bytes.len() > 1_048_576 { tool(1); }
    let bytes = copy_blob(bytes).unwrap_or_else(|code| tool(code));
    let value = if *text {
        let Ok(text) = String::from_utf8(bytes) else { tool(1); };
        Array::Text(text)
    } else {
        Array::Bytes(bytes)
    };
    insert_blob(value)
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_text_v1(kind: u32, handle: u64, start: u64, end: u64, span_start: u32, span_end: u32) -> u64 {
    let prepared = RUNTIME.with(|cell| {
        let runtime = cell.borrow();
        let source = array(&runtime, handle).ok_or(1u32)?;
        let range = || {
            let start = usize::try_from(start).map_err(|_| 3u32)?;
            let end = usize::try_from(end).map_err(|_| 3u32)?;
            if start > end || end > source.len() { return Err(3u32); }
            Ok(start..end)
        };
        match (kind, source) {
            (1, Array::Bytes(bytes)) => Ok(Some(Array::Bytes(copy_blob(bytes.get(range()?).ok_or(3u32)?)?))),
            (2, Array::Text(text)) => Ok(text.get(range()?).map(|part| copy_blob(part.as_bytes()).and_then(|bytes| String::from_utf8(bytes).map(Array::Text).map_err(|_| 1u32))).transpose()?),
            (3, Array::Bytes(bytes)) => Ok(std::str::from_utf8(bytes).ok().map(|text| copy_blob(text.as_bytes()).and_then(|bytes| String::from_utf8(bytes).map(Array::Text).map_err(|_| 1u32))).transpose()?),
            (4, Array::Text(text)) => Ok(Some(Array::Bytes(copy_blob(text.as_bytes())?))),
            _ => Err(1u32),
        }
    });
    let prepared = match prepared { Ok(value) => value, Err(3) => trap(3, span_start, span_end), Err(code) => tool(code) };
    if matches!(kind, 1 | 4) {
        return insert_blob(prepared.unwrap_or_else(|| tool(1)));
    }
    let (tag, payload) = match prepared {
        Some(value) => (0, insert_blob(value)),
        None => (1, new_array(3, 2, 0)),
    };
    let result = new_array(3, 2, 0);
    invar_object_set_v1(result, 0, tag, 0);
    invar_object_set_v1(result, 1, payload as i64, 1);
    result
}

/// `sum_tree`/`dot_tree`/`sum_exact`/`dot_exact` over host-owned f32 words;
/// a dot length mismatch traps at the call span.
#[unsafe(no_mangle)]
pub extern "C" fn invar_fsum_v1(kind: u32, a: u64, b: u64, span_start: u32, span_end: u32) -> u64 {
    let result = RUNTIME.with(|cell| {
        let runtime = cell.borrow();
        let (Some(Array::U32(a)), Some(Array::U32(b))) = (array(&runtime, a), array(&runtime, b)) else {
            return Err(1u32);
        };
        let kind = fsum::Kind::from_code(kind).ok_or(1u32)?;
        if kind.is_dot() && a.len() != b.len() {
            return Err(3u32);
        }
        Ok(match kind {
            fsum::Kind::SumTree => fsum::sum_tree(a),
            fsum::Kind::DotTree => fsum::dot_tree(a, b),
            fsum::Kind::SumExact => fsum::sum_exact(a),
            fsum::Kind::DotExact => fsum::dot_exact(a, b),
        })
    });
    match result {
        Ok(bits) => u64::from(bits),
        Err(3) => trap(3, span_start, span_end),
        Err(code) => tool(code),
    }
}

#[cfg(invar_has_foreign)]
#[unsafe(no_mangle)]
pub extern "C" fn invar_foreign_call_v1(id: u32, a0: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64, a6: i64, a7: i64) -> u64 {
    use invar_foreign::{Arg, Mode, Output};
    let Some(Some((symbol, expected, specs))) = FOREIGN.get(id as usize) else { tool(1); };
    let raw = [a0, a1, a2, a3, a4, a5, a6, a7];
    enum Owned { I32(i32), U32(u32), F64(u64), Bytes(Vec<u8>, Mode), Text(String, Mode) }
    let prepared = RUNTIME.with(|cell| {
        let runtime = cell.borrow();
        specs.iter().enumerate().map(|(index, (tag, mode, _))| {
            let value = raw[index];
            let mode = match mode { 0 => Mode::Value, 1 => Mode::Read, 2 => Mode::Take, _ => return Err(1u32) };
            match (*tag, mode) {
                (1, Mode::Value) => i32::try_from(value).map(Owned::I32).map_err(|_| 1u32),
                (2, Mode::Value) => u32::try_from(value).map(Owned::U32).map_err(|_| 1u32),
                (5, Mode::Value) => Ok(Owned::F64(u64::from_ne_bytes(value.to_ne_bytes()))),
                (3, Mode::Read | Mode::Take) => match array(&runtime, value as u64) {
                    Some(Array::Bytes(bytes)) => Ok(Owned::Bytes(copy_blob(bytes)?, mode)),
                    _ => Err(1),
                },
                (4, Mode::Read | Mode::Take) => match array(&runtime, value as u64) {
                    Some(Array::Text(text)) => String::from_utf8(copy_blob(text.as_bytes())?).map(|text| Owned::Text(text, mode)).map_err(|_| 1),
                    _ => Err(1),
                },
                _ => Err(1),
            }
        }).collect::<Result<Vec<_>, u32>>()
    }).unwrap_or_else(|code| tool(code));
    let args = prepared.iter().map(|item| match item {
        Owned::I32(value) => Arg::I32(*value),
        Owned::U32(value) => Arg::U32(*value),
        Owned::F64(bits) => Arg::F64(*bits),
        Owned::Bytes(value, mode) => Arg::Bytes(value, *mode),
        Owned::Text(value, mode) => Arg::String(value, *mode),
    }).collect::<Vec<_>>();
    let outcome = invar_foreign::call(std::path::Path::new(C_LIBRARY), symbol, &args, *expected);
    for (index, (_, mode, name)) in specs.iter().enumerate() {
        if *mode == 2 { invar_array_drop_v1(raw[index] as u64, *name as u32); }
    }
    let (tag, payload, owned) = match outcome {
        Ok(Output::I32(value)) => (0, i64::from(value), 0),
        Ok(Output::U32(value)) => (0, i64::from(value), 0),
        Ok(Output::Bytes(value)) => (0, insert_blob(Array::Bytes(value)) as i64, 1),
        Ok(Output::String(value)) => (0, insert_blob(Array::Text(value)) as i64, 1),
        Err(_) => (1, new_array(3, 2, 0) as i64, 1),
    };
    let result = new_array(3, 2, 0);
    invar_object_set_v1(result, 0, tag, 0);
    invar_object_set_v1(result, 1, payload, owned);
    result
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_array_fill_v1(kind: u32, len: u64, fill: i64) -> u64 {
    let len = usize::try_from(len).unwrap_or_else(|_| tool(2));
    new_array(kind, len, fill)
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_array_set_v1(handle: u64, index: u32, value: i64) {
    let valid = RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        match array_mut(&mut runtime, handle) {
            Some(Array::I32(values)) => values
                .get_mut(index as usize)
                .map(|slot| *slot = value as i32)
                .is_some(),
            Some(Array::U32(values)) => values
                .get_mut(index as usize)
                .map(|slot| *slot = value as u32)
                .is_some(),
            Some(Array::F64(values)) => values
                .get_mut(index as usize)
                .map(|slot| *slot = value)
                .is_some(),
            Some(Array::Object(_) | Array::Bytes(_) | Array::Text(_)) => false,
            Some(Array::View { .. }) => false,
            #[cfg(invar_has_spatial)]
            Some(Array::Grid(_)) => false,
            None => false,
        }
    });
    if !valid {
        tool(1);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_object_set_v1(handle: u64, index: u32, value: i64, owned: u32) {
    if owned > 1 { tool(1); }
    let valid = RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        match array_mut(&mut runtime, handle) {
            Some(Array::Object(values)) => values.get_mut(index as usize)
                .is_some_and(|slot| {
                    if slot.is_some() { false } else { *slot = Some((value, owned == 1)); true }
                }),
            _ => false,
        }
    });
    if !valid { tool(1); }
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_array_index_v1(handle: u64, index: u64, start: u32, end: u32) -> i64 {
    let result = RUNTIME.with(|cell| {
        let runtime = cell.borrow();
        let Some((owner, index)) = physical(&runtime, handle, index)? else { return Ok(None); };
        match array(&runtime, owner) {
            Some(Array::I32(values)) => Ok(values.get(index).map(|value| i64::from(*value))),
            Some(Array::U32(values)) => Ok(values.get(index).map(|value| i64::from(*value))),
            Some(Array::F64(values)) => Ok(values.get(index).copied()),
            Some(Array::Bytes(values)) => Ok(values.get(index).map(|value| i64::from(*value))),
            Some(Array::Object(values)) => Ok(values.get(index).and_then(Option::as_ref).map(|(value, _)| *value)),
            Some(Array::Text(_) | Array::View { .. }) | None => Err(1),
            #[cfg(invar_has_spatial)]
            Some(Array::Grid(_)) => Err(1),
        }
    });
    match result {
        Ok(Some(value)) => value,
        Ok(None) => trap(3, start, end),
        Err(code) => tool(code),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_object_take_v1(handle: u64, index: u32) -> i64 {
    let value = RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        match array_mut(&mut runtime, handle) {
            Some(Array::Object(values)) => values.get_mut(index as usize).and_then(Option::take).map(|(value, _)| value),
            _ => None,
        }
    });
    value.unwrap_or_else(|| tool(1))
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_variant_payload_v1(handle: u64, tag: u32) -> i64 {
    let value = RUNTIME.with(|cell| {
        let runtime = cell.borrow();
        match array(&runtime, handle) {
            Some(Array::Object(values)) if values.len() == 2
                && values[0].as_ref().is_some_and(|(value, _)| *value == i64::from(tag)) =>
                values[1].as_ref().map(|(value, _)| *value),
            _ => None,
        }
    });
    value.unwrap_or_else(|| tool(1))
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_variant_take_payload_v1(handle: u64, tag: u32) -> i64 {
    let value = RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        match array_mut(&mut runtime, handle) {
            Some(Array::Object(values)) if values.len() == 2
                && values[0].as_ref().is_some_and(|(value, _)| *value == i64::from(tag)) =>
                values[1].take().map(|(value, _)| value),
            _ => None,
        }
    });
    value.unwrap_or_else(|| tool(1))
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_array_store_v1(handle: u64, index: u64, value: i64, start: u32, end: u32) {
    let result = RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        if matches!(array(&runtime, handle), Some(Array::View { mutable: false, .. })) { return Err(1); }
        let Some((owner, index)) = physical(&runtime, handle, index)? else { return Ok(false); };
        match array_mut(&mut runtime, owner) {
            Some(Array::I32(values)) => Ok(values
                .get_mut(index)
                .map(|slot| *slot = value as i32)
                .is_some()),
            Some(Array::U32(values)) => Ok(values
                .get_mut(index)
                .map(|slot| *slot = value as u32)
                .is_some()),
            Some(Array::F64(values)) => Ok(values
                .get_mut(index)
                .map(|slot| *slot = value)
                .is_some()),
            Some(Array::Object(_) | Array::Bytes(_) | Array::Text(_) | Array::View { .. }) | None => Err(1),
            #[cfg(invar_has_spatial)]
            Some(Array::Grid(_)) => Err(1),
        }
    });
    match result {
        Ok(true) => {}
        Ok(false) => trap(3, start, end),
        Err(code) => tool(code),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_array_len_v1(handle: u64) -> u64 {
    RUNTIME
        .with(|cell| array(&cell.borrow(), handle).map(Array::len))
        .map(|length| length as u64)
        .unwrap_or_else(|| tool(1))
}

fn drop_owned(runtime: &mut Runtime, handle: u64) -> Result<usize, u32> {
    let mut pending = vec![handle];
    let mut total = 0usize;
    while let Some(handle) = pending.pop() {
        let object = runtime.arrays.get_mut(handle.wrapping_sub(1) as usize)
            .and_then(Option::take).ok_or(1u32)?;
        if matches!(object, Array::View { .. }) { return Err(1); }
        total = total.checked_add(object.len()).ok_or(3u32)?;
        runtime.live_elements = runtime.live_elements.checked_sub(object.len()).ok_or(1u32)?;
        if let Array::Object(fields) = object {
            let owned = fields.iter().filter(|field| field.as_ref().is_some_and(|(_, owned)| *owned)).count();
            pending.try_reserve(owned).map_err(|_| 3u32)?;
            for field in fields.into_iter().rev() {
                if let Some((value, true)) = field {
                    pending.push(u64::try_from(value).map_err(|_| 1u32)?);
                }
            }
        }
    }
    Ok(total)
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_array_drop_v1(handle: u64, name: u32) {
    if handle == 0 {
        return;
    }
    let result = RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        if name as usize >= NAMES.len() {
            return Err(1);
        }
        if runtime.cleanup.len() == MAX_CLEANUP_EVENTS {
            return Err(6);
        }
        let length = drop_owned(&mut runtime, handle)?;
        runtime.cleanup.push((name as usize, length));
        Ok(())
    });
    result.unwrap_or_else(|code| tool(code));
}

fn main() {
    let fuel = std::env::args()
        .nth(1)
        .map(|arg| arg.parse::<u64>().ok())
        .unwrap_or(Some(1_000_000));
    let Some(fuel) = fuel.filter(|fuel| *fuel > 0) else {
        eprintln!("expected positive fuel argument");
        process::exit(2);
    };
    RUNTIME.with(|cell| cell.borrow_mut().fuel = fuel);
    std::hint::black_box(invar_debug_render_v1 as extern "C" fn(u64, u32) -> *const u8);
    // SAFETY: `invar_entry_v1` is emitted with this exact C signature and
    // LLVM-verified before linking. Every callback has a fixed ABI version.
    let value = unsafe { invar_entry_v1() } as i32;
    report(
        "ok",
        &format!("\"result\":{{\"type\":\"i32\",\"value\":{value}}}"),
        0,
        1,
    );
}

// Standalone sort ABI: word representation is selected from verified MIR,
// never inferred from U32 storage (which also carries f32 and Bool arrays).
#[unsafe(no_mangle)]
pub extern "C" fn invar_sort_v1(op: u32, kind: u32, keys: u64, other: u64, start: u32, end: u32) -> u64 {
    use invar_sort_cpu::{Error,KeyKind};
    fn snapshot(runtime: &Runtime, handle: u64) -> Result<Vec<u32>,Error> {
        let source=array(runtime,handle).ok_or(Error::Internal)?;
        let mut out=invar_sort_cpu::reserve(source.len())?;
        match source {
            Array::I32(v)=>out.extend(v.iter().map(|&v|v as u32)),
            Array::U32(v)=>out.extend_from_slice(v),
            _=>return Err(Error::Internal),
        }
        Ok(out)
    }
    fn fail(error: Error,start:u32,end:u32)->! {
        match error {
            Error::LengthMismatch|Error::Offset{..}=>trap(3,start,end),
            Error::Allocation=>tool(3),Error::Limit=>tool(2),_=>tool(1),
        }
    }
    let kind=match kind {0=>KeyKind::U32,1=>KeyKind::I32,2=>KeyKind::F32,_=>tool(1)};
    if op>3 || op==1 && keys==other || op==3 && keys==other { tool(1); }
    let prepared=RUNTIME.with(|cell| {
        let runtime=cell.try_borrow().map_err(|_|Error::Internal)?;
        let expected=match kind {KeyKind::I32=>matches!(array(&runtime,keys),Some(Array::I32(_))),_=>matches!(array(&runtime,keys),Some(Array::U32(_)))};
        if !expected {return Err(Error::Internal);}
        let keys=snapshot(&runtime,keys)?;
        let other=if op==1 || op==3 {Some(snapshot(&runtime,other)?)} else {None};
        Ok((keys,other))
    });
    let (mut key_words,mut other_words)=prepared.unwrap_or_else(|e|fail(e,start,end));
    if op==2 {
        let indices=invar_sort_cpu::order_with(&key_words,kind,None,invar_sort_cpu::serial_pairs).unwrap_or_else(|e|fail(e,start,end));
        let handle=new_array(2,indices.len(),0);
        RUNTIME.with(|cell| {
            let mut runtime=cell.borrow_mut();
            if let Some(Array::U32(v))=array_mut(&mut runtime,handle) { *v=indices; } else { drop(runtime);tool(1); }
        });
        return handle;
    }
    let (offsets,mut payload)=if op==3 {(other_words.take(),None)} else {(None,other_words.take())};
    invar_sort_cpu::sort_with(&mut key_words,payload.as_deref_mut(),kind,offsets.as_deref(),invar_sort_cpu::serial_pairs).unwrap_or_else(|e|fail(e,start,end));
    fn write(runtime:&mut Runtime,handle:u64,words:Vec<u32>)->Result<(),Error> {
        match array_mut(runtime,handle).ok_or(Error::Internal)? {
            Array::I32(v) if v.len()==words.len()=>{for (v,word) in v.iter_mut().zip(words) {*v=word as i32;}},
            Array::U32(v) if v.len()==words.len()=>*v=words,
            _=>return Err(Error::Internal),
        }
        Ok(())
    }
    let result=RUNTIME.with(|cell| {
        let mut runtime=cell.try_borrow_mut().map_err(|_|Error::Internal)?;
        write(&mut runtime,keys,key_words)?;
        if let Some(words)=payload {write(&mut runtime,other,words)?;}
        Ok(())
    });
    result.unwrap_or_else(|e|fail(e,start,end));
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_tc_step_v1(a: u64, b: u64, c: u64) -> u64 {
    let a = invar_tc::f16_round(a as u32).unwrap_or_else(|| tool(7));
    let b = invar_tc::f16_round(b as u32).unwrap_or_else(|| tool(7));
    u64::from((f32::from_bits(a) * f32::from_bits(b) + f32::from_bits(c as u32)).to_bits())
}

#[unsafe(no_mangle)]
pub extern "C" fn invar_tc_reference_v1(a: u64, b: u64, rows: u64, inner: u64, cols: u64, output: u64) {
    if output == a || output == b { tool(1); }
    if rows == 0 || inner == 0 || cols == 0 { return; }
    let result = RUNTIME.with(|cell| {
        let mut runtime = cell.borrow_mut();
        let dims = [rows, inner, cols].map(|n| u32::try_from(n).map_err(|_| 1u32));
        let [rows, inner, cols] = dims;
        let dims = [rows?, inner?, cols?];
        let Some(Array::U32(a)) = array(&runtime, a) else { return Err(1); };
        let Some(Array::U32(b)) = array(&runtime, b) else { return Err(1); };
        let extent = |n| usize::try_from(n).map_err(|_| 1u32);
        let a = a.get(..extent(u64::from(dims[0])*u64::from(dims[1]))?).ok_or(1u32)?;
        let b = b.get(..extent(u64::from(dims[1])*u64::from(dims[2]))?).ok_or(1u32)?;
        let values = invar_tc::reference(dims, a, b).map_err(|_| 7u32)?;
        let Some(Some(Array::U32(out))) = output.checked_sub(1).and_then(|n| usize::try_from(n).ok()).and_then(|n| runtime.arrays.get_mut(n)) else { return Err(1); };
        if out.len() != values.len() { return Err(1); }
        *out = values;
        Ok(())
    });
    if let Err(code) = result { tool(code); }
}
