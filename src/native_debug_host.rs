
// Debugger hooks for `--debug` builds (#305), appended to the native host with
// a `DEBUG_TYPES` table of value layouts (`native::debug::host_source`).
// share/invar/invar_gdb.py and invar_lldb.py break at `invar_debug_trap_v1`
// and call `invar_debug_render_v1` in the live process.

const RENDER_ELEMENTS: usize = 32;
const RENDER_TEXT: usize = 256;
const RENDER_DEPTH: usize = 3;

/// How a 64-bit slot or array element reads; `Type` indexes `DEBUG_TYPES`.
#[allow(dead_code)]
#[derive(Clone, Copy)]
enum DebugKind { I32, U32, F32, F64, Bool, Unit, Raw, Type(usize) }

#[allow(dead_code)]
enum DebugLayout {
    Array(DebugKind),
    Record(&'static [(&'static str, DebugKind)]),
    Variant(&'static [(&'static str, Option<DebugKind>)]),
}

thread_local! {
    static RENDERED: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Entered by every trap before the report exits the process, so a
/// breakpoint here sees the category and the source byte span.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn invar_debug_trap_v1(category: u32, start: u32, end: u32) {
    std::hint::black_box((category, start, end));
}

/// NUL-terminated text, valid until the next hook call on this thread.
fn rendered(text: &str) -> *const u8 {
    RENDERED.with(|cell| {
        let mut buffer = cell.borrow_mut();
        buffer.clear();
        buffer.extend(text.bytes().filter(|byte| *byte != 0));
        buffer.push(0);
        buffer.as_ptr()
    })
}

/// FNV-1a of a layout name. The helpers name a value's type by this hash so
/// that rendering it is one call: gdb 12 crashes on a second inferior call
/// made while it prints a frame.
fn name_hash(name: &str) -> u32 {
    name.bytes().fold(0x811c_9dc5, |hash, byte| (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193))
}

/// The host value behind `handle` as text. When `type_hash` matches exactly
/// one layout's [`name_hash`] the text starts with the type name and decodes
/// every field; otherwise it shows the stored words.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn invar_debug_render_v1(handle: u64, type_hash: u32) -> *const u8 {
    let mut layouts = (0..DEBUG_TYPES.len()).filter(|index| name_hash(DEBUG_TYPES[*index].0) == type_hash);
    let layout = match (layouts.next(), layouts.next()) {
        (Some(index), None) => Some(index),
        _ => None,
    };
    let text = RUNTIME.with(|cell| match cell.try_borrow() {
        Ok(runtime) => match layout {
            Some(index) => render_typed(&runtime, handle, index, RENDER_DEPTH),
            None => render(&runtime, handle, RENDER_DEPTH),
        },
        Err(_) => "<runtime busy>".to_owned(),
    });
    rendered(&text)
}

fn render_list<T>(values: impl ExactSizeIterator<Item = T>, show: impl FnMut(T) -> String) -> String {
    let len = values.len();
    let mut items: Vec<String> = values.take(RENDER_ELEMENTS).map(show).collect();
    if len > RENDER_ELEMENTS {
        items.push(format!("... {} more", len - RENDER_ELEMENTS));
    }
    items.join(", ")
}

/// A slot or element read as `kind`; nested handles render to `depth`.
fn render_word(runtime: &Runtime, word: i64, kind: DebugKind, depth: usize) -> String {
    match kind {
        DebugKind::I32 => (word as i32).to_string(),
        DebugKind::U32 => (word as u32).to_string(),
        DebugKind::F32 => format!("{:?}", f32::from_bits(word as u32)),
        DebugKind::F64 => format!("{:?}", f64::from_bits(word as u64)),
        DebugKind::Bool => (word != 0).to_string(),
        DebugKind::Unit => "()".to_owned(),
        DebugKind::Raw => word.to_string(),
        DebugKind::Type(index) if depth > 0 => render_typed(runtime, word as u64, index, depth - 1),
        DebugKind::Type(_) => format!("<handle {word}>"),
    }
}

fn render_typed(runtime: &Runtime, handle: u64, index: usize, depth: usize) -> String {
    let (name, layout) = &DEBUG_TYPES[index];
    let slot = |slot: &Option<(i64, bool)>, kind: DebugKind| match slot {
        Some((word, _)) => render_word(runtime, *word, kind, depth),
        None => "<moved>".to_owned(),
    };
    match (layout, array(runtime, handle)) {
        (_, None) => format!("{name} <no value at handle {handle}>"),
        (DebugLayout::Array(kind), Some(Array::I32(values))) => {
            format!("{name} [{}]", render_list(values.iter(), |value| render_word(runtime, i64::from(*value), *kind, depth)))
        }
        (DebugLayout::Array(kind), Some(Array::U32(values))) => {
            format!("{name} [{}]", render_list(values.iter(), |value| render_word(runtime, i64::from(*value), *kind, depth)))
        }
        (DebugLayout::Array(kind), Some(Array::F64(values))) => {
            format!("{name} [{}]", render_list(values.iter(), |value| render_word(runtime, *value, *kind, depth)))
        }
        (DebugLayout::Record(fields), Some(Array::Object(slots))) if fields.len() == slots.len() => {
            let shown = fields.iter().zip(slots).map(|((field, kind), value)| format!("{field}: {}", slot(value, *kind)));
            format!("{name} {{ {} }}", shown.collect::<Vec<_>>().join(", "))
        }
        // A variant is (tag, payload).
        (DebugLayout::Variant(arms), Some(Array::Object(slots))) => {
            let arm = slots.first().and_then(|tag| tag.as_ref()).and_then(|(tag, _)| arms.get(usize::try_from(*tag).ok()?));
            match (arm, slots.get(1)) {
                (Some((arm, Some(kind))), Some(payload)) => format!("{name}::{arm}({})", slot(payload, *kind)),
                (Some((arm, _)), _) => format!("{name}::{arm}"),
                (None, _) => format!("{name} {}", render(runtime, handle, depth)),
            }
        }
        _ => format!("{name} {}", render(runtime, handle, depth)),
    }
}

/// The stored words behind `handle`, for types without a layout.
fn render(runtime: &Runtime, handle: u64, depth: usize) -> String {
    let Some(value) = array(runtime, handle) else {
        return format!("<no value at handle {handle}>");
    };
    match value {
        Array::I32(values) => format!("[{}]", render_list(values.iter(), |value| value.to_string())),
        Array::U32(values) => format!("[{}]", render_list(values.iter(), |value| value.to_string())),
        Array::F64(values) => format!("[{}]", render_list(values.iter(), |value| format!("{:?}", f64::from_bits(*value as u64)))),
        Array::Bytes(values) => format!("bytes [{}]", render_list(values.iter(), |value| value.to_string())),
        Array::Text(value) => {
            let shown: String = value.chars().take(RENDER_TEXT).collect();
            let more = if shown.len() < value.len() { "..." } else { "" };
            format!("{shown:?}{more}")
        }
        // An owned slot is a nested handle.
        Array::Object(fields) => format!("{{{}}}", render_list(fields.iter(), |field| match field {
            None => "<moved>".to_owned(),
            Some((value, true)) if depth > 0 => render(runtime, *value as u64, depth - 1),
            Some((value, _)) => value.to_string(),
        })),
        #[cfg(invar_has_spatial)]
        Array::Grid(grid) => format!("grid of {} elements", grid.storage_elements()),
        Array::View { len, .. } => format!("view [{}]", render_list(0..*len, |index| {
            match physical(runtime, handle, u64::from(index)) {
                Ok(Some((owner, offset))) => match array(runtime, owner) {
                    Some(Array::I32(values)) => values[offset].to_string(),
                    Some(Array::U32(values)) => values[offset].to_string(),
                    _ => "?".to_owned(),
                },
                _ => "?".to_owned(),
            }
        })),
    }
}
