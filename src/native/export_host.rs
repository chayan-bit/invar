// Compiled as a compiler-owned cdylib. inv_private_namespace is replaced before rustc.
// The only unsafe operations are checked C pointer decoding, fixed generated
// entry calls, and element access through validated private host-array records
// (HA-1 section 12). Library lifetime and caller pointer validity are boundary contracts.
//
// The image is `no_std` and holds no thread-local storage (#320): `std` itself
// links thread-local statics (panic count, current thread, output capture). On
// macOS any such static gives the image a thread-local variable descriptor, which
// dyld can never unload and which costs one of the process's 512 pthread keys, so
// about 500 loaded export libraries abort the interpreter. Without it `dlclose`
// unloads the image. Allocation goes to the C allocator; a panic aborts.
#![cfg_attr(not(test), no_std)]
extern crate alloc;
use alloc::vec::Vec;
use core::{
    cell::RefCell,
    ptr,
    sync::atomic::{AtomicBool, Ordering},
};

#[cfg(not(test))]
mod runtime_support {
    use core::alloc::{GlobalAlloc, Layout};
    #[link(name = "c")]
    unsafe extern "C" {
        fn malloc(size: usize) -> *mut u8;
        fn calloc(count: usize, size: usize) -> *mut u8;
        fn realloc(pointer: *mut u8, size: usize) -> *mut u8;
        fn free(pointer: *mut u8);
        fn abort() -> !;
    }
    /// The C allocator returns memory aligned for any fundamental type (16 bytes
    /// on every supported target); a stricter request fails like exhaustion.
    const MAX_ALIGN: usize = 16;
    struct CAllocator;
    // SAFETY: malloc, calloc, realloc and free uphold the GlobalAlloc contract
    // for every layout whose alignment is at most MAX_ALIGN; others return null.
    unsafe impl GlobalAlloc for CAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if layout.align() > MAX_ALIGN { return core::ptr::null_mut(); }
            unsafe { malloc(layout.size()) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            if layout.align() > MAX_ALIGN { return core::ptr::null_mut(); }
            unsafe { calloc(1, layout.size()) }
        }
        unsafe fn dealloc(&self, pointer: *mut u8, _: Layout) { unsafe { free(pointer) } }
        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            if layout.align() > MAX_ALIGN { return core::ptr::null_mut(); }
            unsafe { realloc(pointer, size) }
        }
    }
    #[global_allocator]
    static ALLOCATOR: CAllocator = CAllocator;
    /// The precompiled `alloc` references the unwinding personality. With
    /// `panic=abort` nothing unwinds, so an empty symbol satisfies the linker.
    #[unsafe(no_mangle)]
    pub extern "C" fn rust_eh_personality() {}
    #[panic_handler]
    fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
        // SAFETY: abort takes no arguments and never returns.
        unsafe { abort() }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Arg { tag: u32, mode: u32, scalar: i64, data: *const u8, len: u64 }
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Output {
    abi_version: u32, tag: u32, scalar: i64, data: *mut u8, len: u64,
    release: Option<unsafe extern "C" fn(*mut u8, u64)>,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Failure {
    abi_version: u32, kind: u32, reason: u32, span_start: u32,
    span_end: u32, reserved: u32, fuel_limit: u64, steps_used: u64,
}
/// Private host-array record (HA-1 section 12); never part of the public C ABI.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HostArray { element: u32, mode: u32, data: *mut u8, len: u64 }
/// One validated array of the active transaction, addressed by its 1-based handle.
#[derive(Clone, Copy)]
struct Entry { data: *mut u8, len: u64, element: u32, mutable: bool }
const MAX_ELEMENTS: u64 = 1_000_000;
const _: () = {
    assert!(core::mem::size_of::<HostArray>() == 24);
    assert!(core::mem::align_of::<HostArray>() == 8);
    assert!(core::mem::size_of::<Arg>() == 32);
    assert!(core::mem::size_of::<Output>() == 40);
    assert!(core::mem::size_of::<Failure>() == 40);
    assert!(core::mem::align_of::<Failure>() == 8);
};
struct Runtime {
    active: bool, called: bool, depth: usize, failure: Failure,
    frames: Vec<Vec<i64>>,
    arrays: Vec<Entry>,
}
/// Process-global runtime of this library image, never a `thread_local!`.
/// On macOS any thread-local makes dyld give the image a thread-local variable
/// descriptor: the image can then never be unloaded and it consumes one of the
/// process's 512 pthread keys, so about 500 loaded export libraries abort the
/// interpreter (#320). Loaders are thread-confined, so one transaction at a time
/// per image is the contract; `begin` on a second thread's loader reports busy.
/// The lock only serializes each entry's own short access.
struct Global { lock: AtomicBool, cell: RefCell<Runtime> }
// SAFETY: every access to `cell` goes through `with`, which holds `lock`.
unsafe impl Sync for Global {}
impl Global {
    fn with<R>(&self, access: impl FnOnce(&RefCell<Runtime>) -> R) -> R {
        while self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            core::hint::spin_loop();
        }
        let result = access(&self.cell);
        self.lock.store(false, Ordering::Release);
        result
    }
}
static RUNTIME: Global = Global { lock: AtomicBool::new(false), cell: RefCell::new(Runtime {
    active: false, called: false, depth: 0,
    failure: Failure { abi_version: 1, kind: 0, reason: 0, span_start: 0,
        span_end: 0, reserved: 0, fuel_limit: 0, steps_used: 0 },
    frames: Vec::new(),
    arrays: Vec::new(),
}) };
fn latch(runtime: &mut Runtime, kind: u32, reason: u32, start: u32, end: u32) {
    if runtime.failure.kind == 0 {
        runtime.failure.kind = kind;
        runtime.failure.reason = reason;
        runtime.failure.span_start = start;
        runtime.failure.span_end = end;
    }
}

#[unsafe(export_name = "inv_private_namespaceexport_begin_v1")]
pub extern "C" fn begin(fuel: u64) -> i32 {
    RUNTIME.with(|cell| {
        let Ok(mut runtime) = cell.try_borrow_mut() else { return 1; };
        if runtime.active || fuel == 0 { return 1; }
        runtime.frames.clear();
        runtime.arrays.clear();
        // Frame allocation failures occur inside the active invocation and
        // return an explicit tool record. Begin itself only rejects zero/busy.
        runtime.active = true;
        runtime.called = false;
        runtime.depth = 0;
        runtime.failure = Failure { abi_version: 1, fuel_limit: fuel, ..Failure::default() };
        0
    })
}

#[unsafe(export_name = "inv_private_namespaceexport_end_v1")]
pub unsafe extern "C" fn end(out: *mut Failure) -> i32 {
    if out.is_null() || !(out as usize).is_multiple_of(core::mem::align_of::<Failure>()) { return 1; }
    RUNTIME.with(|cell| {
        let Ok(mut runtime) = cell.try_borrow_mut() else { return 1; };
        if !runtime.active { return 1; }
        if !runtime.called { latch(&mut runtime, 4, 1, 0, 0); }
        // SAFETY: the reviewed loader supplies aligned writable Failure storage;
        // arbitrary foreign callers must uphold the same C pointer contract.
        unsafe { ptr::write(out, runtime.failure); }
        runtime.frames.clear();
        runtime.arrays.clear();
        runtime.depth = 0;
        runtime.active = false;
        runtime.called = false;
        0
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn inv_abi_version_v1() -> u32 { 1 }
#[unsafe(export_name = "inv_private_namespaceexport_identity_v1")]
pub extern "C" fn identity() -> *const u8 { INPUT_IDENTITY.as_ptr() }

#[unsafe(export_name = "inv_private_namespacetick_v1")]
pub extern "C" fn tick() -> i32 {
    RUNTIME.with(|cell| {
        let Ok(mut runtime) = cell.try_borrow_mut() else { return 1; };
        if !runtime.active || !runtime.called { return 1; }
        if runtime.failure.kind != 0 { return 1; }
        if runtime.failure.steps_used == runtime.failure.fuel_limit {
            latch(&mut runtime, 3, 0, 0, 0);
            return 1;
        }
        runtime.failure.steps_used += 1;
        0
    })
}
#[unsafe(export_name = "inv_private_namespaceenter_v1")]
pub extern "C" fn enter() -> i32 {
    RUNTIME.with(|cell| {
        let Ok(mut runtime) = cell.try_borrow_mut() else { return 1; };
        if !runtime.active || runtime.failure.kind != 0 { return 1; }
        if runtime.depth >= 64 { latch(&mut runtime, 2, 5, 0, 0); return 1; }
        runtime.depth += 1;
        0
    })
}
#[unsafe(export_name = "inv_private_namespaceframe_new_v1")]
pub extern "C" fn frame_new(slots: u32) -> *mut i64 {
    RUNTIME.with(|cell| {
        let Ok(mut runtime) = cell.try_borrow_mut() else { return ptr::null_mut(); };
        if !runtime.active || runtime.failure.kind != 0 { return ptr::null_mut(); }
        if slots > 256 || runtime.frames.len() >= 64 || runtime.frames.len() + 1 != runtime.depth {
            latch(&mut runtime, 2, 3, 0, 0); return ptr::null_mut();
        }
        // Moving Vec headers does not move their word allocations. Reserve
        // fallibly before push, so the bounded inventory cannot implicitly grow.
        if runtime.frames.try_reserve(1).is_err() {
            latch(&mut runtime, 2, 3, 0, 0); return ptr::null_mut();
        }
        let mut frame = Vec::new();
        if frame.try_reserve_exact(slots as usize).is_err() {
            latch(&mut runtime, 2, 3, 0, 0); return ptr::null_mut();
        }
        frame.resize(slots as usize, 0);
        let pointer = frame.as_mut_ptr();
        runtime.frames.push(frame);
        pointer
    })
}
#[unsafe(export_name = "inv_private_namespaceleave_v1")]
pub extern "C" fn leave() {
    RUNTIME.with(|cell| {
        if let Ok(mut runtime) = cell.try_borrow_mut() {
            if runtime.depth == 0 || runtime.frames.pop().is_none() {
                latch(&mut runtime, 2, 1, 0, 0);
            } else { runtime.depth -= 1; }
        }
    });
}
#[unsafe(export_name = "inv_private_namespacetrap_v1")]
pub extern "C" fn trap(reason: u32, start: u32, end: u32) {
    RUNTIME.with(|cell| {
        if let Ok(mut runtime) = cell.try_borrow_mut() { latch(&mut runtime, 1, reason, start, end); }
    });
}
#[unsafe(export_name = "inv_private_namespacetool_v1")]
pub extern "C" fn tool(reason: u32) {
    RUNTIME.with(|cell| {
        if let Ok(mut runtime) = cell.try_borrow_mut() { latch(&mut runtime, 2, reason, 0, 0); }
    });
}

fn claim() -> bool {
    RUNTIME.with(|cell| {
        let Ok(mut runtime) = cell.try_borrow_mut() else { return false; };
        if !runtime.active { return false; }
        if runtime.called { latch(&mut runtime, 4, 1, 0, 0); return false; }
        runtime.called = true;
        true
    })
}
fn aligned<T>(pointer: *const T) -> bool { (pointer as usize).is_multiple_of(core::mem::align_of::<T>()) }
fn is_array_tag(tag: u32) -> bool { tag > 2 }

unsafe fn dispatch(ordinal: usize, args: *const Arg, argc: u32, out: *mut Output) -> i32 {
    if !claim() { return 1; }
    let Some((tags, _)) = SIGNATURES.get(ordinal) else { tool(1); return 1; };
    if tags.iter().any(|tag| is_array_tag(*tag)) || argc as usize != tags.len() || out.is_null()
        || !aligned(out) || (argc != 0 && (args.is_null() || !aligned(args)))
    { reject_abi(); return 1; }
    let mut values = [0i64; 8];
    for (n, tag) in tags.iter().enumerate() {
        // SAFETY: arity/alignment/null are checked; caller supplies readable
        // storage for argc Args. Copy decoding creates no escaping reference.
        let Some(value) = (unsafe { scalar_arg(args.add(n), *tag) }) else { reject_abi(); return 1; };
        values[n] = value;
    }
    // SAFETY: out was checked above; finish only writes it after full success.
    unsafe { finish(ordinal, &values, out) }
}

/// Decode one public scalar wire argument.
///
/// # Safety
/// `arg` must be non-null, aligned and readable.
unsafe fn scalar_arg(arg: *const Arg, tag: u32) -> Option<i64> {
    // SAFETY: guaranteed by the caller; the copy creates no escaping reference.
    let arg = unsafe { ptr::read(arg) };
    (arg.tag == tag && arg.mode == 0 && arg.data.is_null() && arg.len == 0
        && valid_scalar(tag, arg.scalar)).then_some(arg.scalar)
}

/// Private array entry (HA-1.2). Scalars use public wire records in order;
/// arrays use private records in order. No public inv_export_N exists for a
/// function with array parameters.
///
/// # Safety
/// `args` holds `argc` readable Args and `arrays` holds `count` readable
/// records whose `data` ranges stay valid, unaliased as declared, and
/// unretained for this synchronous call; `out` is writable.
#[unsafe(export_name = "inv_private_namespaceexport_call_arrays_v1")]
pub unsafe extern "C" fn call_arrays(
    ordinal: u32, args: *const Arg, argc: u32, arrays: *const HostArray, count: u32, out: *mut Output,
) -> i32 {
    if !claim() { return 1; }
    let ordinal = ordinal as usize;
    let Some((tags, _)) = SIGNATURES.get(ordinal) else { tool(1); return 1; };
    let array_count = tags.iter().filter(|tag| is_array_tag(**tag)).count();
    if array_count == 0 || count as usize != array_count || argc as usize != tags.len() - array_count
        || out.is_null() || !aligned(out)
        || (argc != 0 && (args.is_null() || !aligned(args)))
        || arrays.is_null() || !aligned(arrays)
    { reject_abi(); return 1; }
    let reserved = RUNTIME.with(|cell| {
        let Ok(mut runtime) = cell.try_borrow_mut() else { return false; };
        runtime.arrays.try_reserve_exact(array_count).is_ok()
    });
    if !reserved { tool(3); return 1; }
    let (mut scalar, mut array) = (0usize, 0usize);
    let mut values = [0i64; 8];
    for (n, tag) in tags.iter().enumerate() {
        if !is_array_tag(*tag) {
            // SAFETY: argc, null and alignment were checked against the signature.
            let Some(value) = (unsafe { scalar_arg(args.add(scalar), *tag) }) else { reject_abi(); return 1; };
            scalar += 1;
            values[n] = value;
            continue;
        }
        // SAFETY: count, null and alignment were checked; the copy creates no reference.
        let record = unsafe { ptr::read(arrays.add(array)) };
        array += 1;
        let Some(entry) = entry_for(*tag, record) else { reject_abi(); return 1; };
        let handle = RUNTIME.with(|cell| {
            let Ok(mut runtime) = cell.try_borrow_mut() else { return 0; };
            runtime.arrays.push(entry);
            runtime.arrays.len() as i64
        });
        if handle == 0 { reject_abi(); return 1; }
        values[n] = handle;
    }
    // SAFETY: out was checked above; finish only writes it after full success.
    unsafe { finish(ordinal, &values, out) }
}

/// Validate a private record against its signature tag (integer metadata only).
fn entry_for(tag: u32, record: HostArray) -> Option<Entry> {
    let (element, mode) = ((tag >> 4) & 0xf, tag & 0xf);
    let shaped = tag & !0xff == 0x100 && matches!(element, 1 | 2 | 3) && matches!(mode, 1 | 3);
    let address = record.data as usize;
    let valid = shaped && record.element == element && record.mode == mode
        && record.len <= MAX_ELEMENTS && record.data.is_null() == (record.len == 0)
        && address.is_multiple_of(4)
        && address.checked_add(record.len as usize * 4).is_some();
    valid.then_some(Entry { data: record.data, len: record.len, element, mutable: mode == 3 })
}

/// Call the generated entry and publish the result after full success.
///
/// # Safety
/// `out` is non-null, aligned and writable.
unsafe fn finish(ordinal: usize, values: &[i64; 8], out: *mut Output) -> i32 {
    let Some((_, result_tag)) = SIGNATURES.get(ordinal) else { tool(1); return 1; };
    let mut value = 0i64;
    // SAFETY: generated exact fixed signatures receive validated values and
    // writable scalar storage; LLVM returns status before publishing on failure.
    let status = unsafe { call_entry(ordinal, values, &mut value) };
    if status != 0 { return 1; }
    let clean = RUNTIME.with(|cell| {
        let Ok(runtime) = cell.try_borrow() else { return false; };
        runtime.failure.kind == 0 && runtime.depth == 0 && runtime.frames.is_empty()
    });
    if !clean || !valid_scalar(*result_tag, value) { tool(1); return 1; }
    let result = Output { abi_version: 1, tag: *result_tag, scalar: value,
        data: ptr::null_mut(), len: 0, release: None };
    // SAFETY: the caller's out storage is valid/aligned and remains live for
    // this synchronous call. This is the only publication, after full success.
    unsafe { ptr::write(out, result); }
    0
}
fn valid_scalar(tag: u32, value: i64) -> bool {
    match tag { 1 => i32::try_from(value).is_ok(), 2 => u32::try_from(value).is_ok(), _ => false }
}
fn reject_abi() {
    RUNTIME.with(|cell| {
        if let Ok(mut runtime) = cell.try_borrow_mut() { latch(&mut runtime, 4, 1, 0, 0); }
    });
}

/// The validated entry for `handle` in the active, called, non-failed transaction.
fn lookup(handle: i64) -> Option<Entry> {
    RUNTIME.with(|cell| {
        let runtime = cell.try_borrow().ok()?;
        if !runtime.active || !runtime.called || runtime.failure.kind != 0 { return None; }
        let index = usize::try_from(handle).ok()?.checked_sub(1)?;
        runtime.arrays.get(index).copied()
    })
}
fn element_index(entry: Entry, index: i64, start: u32, end: u32) -> Option<usize> {
    match u64::try_from(index) {
        Ok(index) if index < entry.len => Some(index as usize),
        _ => { trap(3, start, end); None }
    }
}
#[unsafe(export_name = "inv_private_namespacearray_len_v1")]
pub unsafe extern "C" fn array_len(handle: i64, out: *mut i64) -> i32 {
    let Some(entry) = lookup(handle) else { tool(1); return 1; };
    if out.is_null() || !aligned(out) { tool(1); return 1; }
    // SAFETY: generated code passes its own aligned i64 frame slot.
    unsafe { ptr::write(out, entry.len as i64) };
    0
}
#[unsafe(export_name = "inv_private_namespacearray_index_v1")]
pub unsafe extern "C" fn array_index(handle: i64, index: i64, start: u32, end: u32, out: *mut i64) -> i32 {
    let Some(entry) = lookup(handle) else { tool(1); return 1; };
    if out.is_null() || !aligned(out) { tool(1); return 1; }
    let Some(index) = element_index(entry, index, start, end) else { return 1; };
    // SAFETY: the record was validated (non-null, 4-aligned, len elements in
    // the caller's live borrowed slice) and index < len; read at the caller's address.
    let value = unsafe {
        match entry.element {
            1 => i64::from(ptr::read(entry.data.cast::<i32>().add(index))),
            // Binary32 is a zero-extended bit carrier, never a numeric conversion.
            _ => i64::from(ptr::read(entry.data.cast::<u32>().add(index))),
        }
    };
    // SAFETY: generated code passes its own aligned i64 frame slot.
    unsafe { ptr::write(out, value) };
    0
}
#[unsafe(export_name = "inv_private_namespacearray_store_v1")]
pub unsafe extern "C" fn array_store(handle: i64, index: i64, value: i64, start: u32, end: u32) -> i32 {
    let Some(entry) = lookup(handle) else { tool(1); return 1; };
    if !entry.mutable || !valid_scalar(if entry.element == 3 { 2 } else { entry.element }, value) { tool(1); return 1; }
    let Some(index) = element_index(entry, index, start, end) else { return 1; };
    // SAFETY: a mut record was validated as above; the caller's exclusive
    // borrow covers len elements and index < len; write at the caller's address.
    unsafe {
        match entry.element {
            1 => ptr::write(entry.data.cast::<i32>().add(index), value as i32),
            _ => ptr::write(entry.data.cast::<u32>().add(index), value as u32),
        }
    }
    0
}
