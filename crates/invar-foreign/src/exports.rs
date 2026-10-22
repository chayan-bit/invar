//! Persistent, thread-confined loading of compiler-owned scalar export artifacts.
//!
//! The caller supplies the compiler's artifact specification and a fresh private
//! artifact path. As with the legacy foreign loader, loading executes library
//! constructors: the selected library must be trusted to implement the frozen C
//! signatures and keep its identity bytes readable for its loaded lifetime.

use std::{fs::File, io::Read, marker::PhantomData, path::Path, rc::Rc};

use libloading::Library;
use sha2::{Digest, Sha256};

use super::{ABI_VERSION, Arg, Error, I32, InvArg, InvResult, Output, U32};

type VersionFn = unsafe extern "C" fn() -> u32;
type IdentityFn = unsafe extern "C" fn() -> *const u8;
type BeginFn = unsafe extern "C" fn(u64) -> i32;
type EndFn = unsafe extern "C" fn(*mut FailureRecord) -> i32;
type CallFn = unsafe extern "C" fn(*const InvArg, u32, *mut InvResult) -> i32;
/// Private HA-1.2 array entry: ordinal, public scalar wire records, private
/// array records (HA-1 section 12 layout), result.
type CallArraysFn = unsafe extern "C" fn(
    u32,
    *const InvArg,
    u32,
    *const std::ffi::c_void,
    u32,
    *mut InvResult,
) -> i32;

/// Private HA-1.2 descriptor tags of `read`/`mut` `Array<i32|u32>` parameters
/// (`0x100 | element << 4 | mode`). They never appear in an `inv_arg_v1`.
pub(crate) const ARRAY_TAGS: [u32; 6] = [0x111, 0x113, 0x121, 0x123, 0x131, 0x133];

fn is_array_tag(tag: u32) -> bool {
    ARRAY_TAGS.contains(&tag)
}

/// Private compiler-host record. Its layout does not change the public C ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FailureRecord {
    pub abi_version: u32,
    pub kind: u32,
    pub reason: u32,
    pub span_start: u32,
    pub span_end: u32,
    pub reserved: u32,
    pub fuel_limit: u64,
    pub steps_used: u64,
}

const _: () = {
    assert!(std::mem::size_of::<FailureRecord>() == 40);
    assert!(std::mem::align_of::<FailureRecord>() == 8);
    assert!(std::mem::offset_of!(FailureRecord, abi_version) == 0);
    assert!(std::mem::offset_of!(FailureRecord, kind) == 4);
    assert!(std::mem::offset_of!(FailureRecord, reason) == 8);
    assert!(std::mem::offset_of!(FailureRecord, span_start) == 12);
    assert!(std::mem::offset_of!(FailureRecord, span_end) == 16);
    assert!(std::mem::offset_of!(FailureRecord, reserved) == 20);
    assert!(std::mem::offset_of!(FailureRecord, fuel_limit) == 24);
    assert!(std::mem::offset_of!(FailureRecord, steps_used) == 32);
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Descriptor {
    pub artifact_id: [u8; 32],
    pub symbol: String,
    pub parameter_tags: Vec<u32>,
    pub result_tag: u32,
}

#[derive(Clone, Debug)]
pub struct ArtifactSpec {
    pub identity: [u8; 32],
    pub input_identity: [u8; 32],
    pub library_sha256: [u8; 32],
    pub descriptors: Vec<Descriptor>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Invocation {
    pub output: Option<Output>,
    pub failure: FailureRecord,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ExportLoadError {
    Boundary(Error),
    Protocol,
    Identity,
    Busy,
    SymbolOrdinal { position: usize, ordinal: u32 },
}

/// Owns the library throughout every begin/call/end transaction.
///
/// The explicit `Rc` marker prevents both sending and sharing this TLS-backed
/// loader across threads. Function pointers never escape this owner.
///
/// ```compile_fail
/// use invar_foreign::exports::LoadedExports;
/// fn require_send<T: Send>() {}
/// require_send::<LoadedExports>();
/// ```
///
/// ```compile_fail
/// use invar_foreign::exports::LoadedExports;
/// fn require_sync<T: Sync>() {}
/// require_sync::<LoadedExports>();
/// ```
pub struct LoadedExports {
    _library: Library,
    spec: ArtifactSpec,
    /// `None` for a descriptor with array parameters: it has no public symbol.
    functions: Vec<Option<CallFn>>,
    call_arrays: Option<CallArraysFn>,
    begin: BeginFn,
    end: EndFn,
    _thread: PhantomData<Rc<()>>,
}

fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, ExportLoadError> {
    // SAFETY: The compiler-owned library implements the frozen signature for
    // each public/control symbol; callers retain the library for pointer use.
    unsafe { library.get::<T>(name) }
        .map(|function| *function)
        .map_err(|_| ExportLoadError::Boundary(Error::Symbol))
}

fn control_name(input_identity: &[u8; 32], suffix: &str) -> Vec<u8> {
    let mut name = String::from("invar_");
    for byte in input_identity {
        use std::fmt::Write;
        // Writing to a String cannot fail.
        let _ = write!(name, "{byte:02x}");
    }
    name.push('_');
    name.push_str(suffix);
    let mut bytes = name.into_bytes();
    bytes.push(0);
    bytes
}

fn symbol_ordinal(name: &str) -> Result<u32, ExportLoadError> {
    let layout = || ExportLoadError::Boundary(Error::Layout);
    let index = name.strip_prefix("inv_export_").ok_or_else(layout)?;
    if index.is_empty()
        || !index.bytes().all(|byte| byte.is_ascii_digit())
        || (index != "0" && index.starts_with('0'))
    {
        return Err(layout());
    }
    index.parse().map_err(|_| layout())
}

fn validate_spec(spec: &ArtifactSpec) -> Result<(), ExportLoadError> {
    if spec.descriptors.is_empty() {
        return Err(ExportLoadError::Boundary(Error::Layout));
    }
    for (index, descriptor) in spec.descriptors.iter().enumerate() {
        if descriptor.artifact_id != spec.identity {
            return Err(ExportLoadError::Identity);
        }
        let ordinal = symbol_ordinal(&descriptor.symbol)?;
        if descriptor.parameter_tags.len() > 8
            || !descriptor
                .parameter_tags
                .iter()
                .all(|tag| matches!(*tag, I32 | U32) || is_array_tag(*tag))
            || !matches!(descriptor.result_tag, I32 | U32)
            || spec.descriptors[..index]
                .iter()
                .any(|prior| prior.symbol == descriptor.symbol)
        {
            return Err(ExportLoadError::Boundary(Error::Layout));
        }
        if usize::try_from(ordinal).ok() != Some(index) {
            return Err(ExportLoadError::SymbolOrdinal {
                position: index,
                ordinal,
            });
        }
    }
    Ok(())
}

fn file_digest(path: &Path) -> Result<[u8; 32], ExportLoadError> {
    let mut file = File::open(path).map_err(|_| ExportLoadError::Boundary(Error::Library))?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 16_384];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|_| ExportLoadError::Boundary(Error::Library))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(digest.finalize().into())
}

impl LoadedExports {
    /// Verify and load an explicit compiler-owned absolute regular-file path.
    /// The file must remain unchanged throughout loading. Later pathname changes
    /// have no effect on calls through the retained library handle.
    pub fn load(path: &Path, spec: &ArtifactSpec) -> Result<Self, ExportLoadError> {
        validate_spec(spec)?;
        if !path.is_absolute()
            || !std::fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_file())
        {
            return Err(ExportLoadError::Boundary(Error::Library));
        }
        if file_digest(path)? != spec.library_sha256 {
            return Err(ExportLoadError::Identity);
        }
        // SAFETY: The explicit compiler-owned artifact is trusted to implement
        // the frozen ABI, and its content digest was checked before loading.
        let library =
            unsafe { Library::new(path) }.map_err(|_| ExportLoadError::Boundary(Error::Library))?;
        let version: VersionFn = symbol(&library, b"inv_abi_version_v1\0")?;
        // SAFETY: The loaded artifact promises the version symbol's C signature.
        if unsafe { version() } != ABI_VERSION {
            return Err(ExportLoadError::Boundary(Error::Version));
        }
        let identity: IdentityFn = symbol(
            &library,
            &control_name(&spec.input_identity, "export_identity_v1"),
        )?;
        // SAFETY: The private identity function returns either null or a pointer
        // to 32 immutable bytes retained by this library for its loaded lifetime.
        let identity_pointer = unsafe { identity() };
        if identity_pointer.is_null() {
            return Err(ExportLoadError::Identity);
        }
        // SAFETY: The trusted identity protocol guarantees those 32 readable
        // bytes; `library` is still owned here.
        let actual_identity = unsafe { std::slice::from_raw_parts(identity_pointer, 32) };
        if actual_identity != spec.input_identity {
            return Err(ExportLoadError::Identity);
        }
        let begin = symbol(
            &library,
            &control_name(&spec.input_identity, "export_begin_v1"),
        )?;
        let end = symbol(
            &library,
            &control_name(&spec.input_identity, "export_end_v1"),
        )?;
        let mut functions = Vec::new();
        functions
            .try_reserve_exact(spec.descriptors.len())
            .map_err(|_| ExportLoadError::Boundary(Error::Allocation))?;
        let mut arrays = false;
        for descriptor in &spec.descriptors {
            if descriptor
                .parameter_tags
                .iter()
                .any(|tag| is_array_tag(*tag))
            {
                arrays = true;
                functions.push(None);
                continue;
            }
            let mut name = descriptor.symbol.as_bytes().to_vec();
            name.push(0);
            functions.push(Some(symbol(&library, &name)?));
        }
        let call_arrays = if arrays {
            Some(symbol(
                &library,
                &control_name(&spec.input_identity, "export_call_arrays_v1"),
            )?)
        } else {
            None
        };
        Ok(Self {
            _library: library,
            spec: spec.clone(),
            functions,
            call_arrays,
            begin,
            end,
            _thread: PhantomData,
        })
    }

    /// Invoke exactly one admitted scalar export, completing its private
    /// transaction before decoding output or returning an error.
    pub fn invoke(
        &mut self,
        descriptor: &Descriptor,
        args: &[Arg<'_>],
        fuel: u64,
    ) -> Result<Invocation, ExportLoadError> {
        let index = self.index(descriptor)?;
        let function = self.functions[index].ok_or(ExportLoadError::Boundary(Error::Layout))?;
        if fuel == 0 || args.len() != descriptor.parameter_tags.len() {
            return Err(ExportLoadError::Boundary(Error::Layout));
        }
        let wire = wire_scalars(args, &descriptor.parameter_tags)?;
        self.transact(descriptor, fuel, |result| {
            // SAFETY: Full inventory and scalar arguments were validated before
            // begin. Argument storage and result stay live for this call, and the
            // retained library may neither retain them nor unwind through C.
            unsafe { function(wire.as_ptr(), wire.len() as u32, result) }
        })
    }

    /// Invoke one admitted array export through the private entry (HA-1.2).
    /// Only the consuming host-array guard calls this, with its own records.
    ///
    /// # Safety
    /// `records` points to `count` private host-array records that were built
    /// from live borrowed slices, match the descriptor's array tags in order,
    /// and stay valid and exclusively borrowed for this synchronous call.
    pub(crate) unsafe fn invoke_arrays(
        &mut self,
        descriptor: &Descriptor,
        scalars: &[Arg<'_>],
        records: *const std::ffi::c_void,
        count: usize,
        fuel: u64,
    ) -> Result<Invocation, ExportLoadError> {
        self.index(descriptor)?;
        let call = self
            .call_arrays
            .ok_or(ExportLoadError::Boundary(Error::Layout))?;
        let arrays = descriptor
            .parameter_tags
            .iter()
            .filter(|tag| is_array_tag(**tag))
            .count();
        let scalar_tags = descriptor
            .parameter_tags
            .iter()
            .copied()
            .filter(|tag| !is_array_tag(*tag))
            .collect::<Vec<_>>();
        if fuel == 0
            || arrays == 0
            || count != arrays
            || scalars.len() != scalar_tags.len()
            || records.is_null()
        {
            return Err(ExportLoadError::Boundary(Error::Layout));
        }
        let wire = wire_scalars(scalars, &scalar_tags)?;
        let ordinal = symbol_ordinal(&descriptor.symbol)?;
        self.transact(descriptor, fuel, |result| {
            // SAFETY: the caller guarantees the records; scalar wire storage
            // and the result are live for this call, and the library may not
            // retain any pointer or unwind through C (HA-1 section 12).
            unsafe {
                call(
                    ordinal,
                    wire.as_ptr(),
                    wire.len() as u32,
                    records,
                    count as u32,
                    result,
                )
            }
        })
    }

    fn index(&self, descriptor: &Descriptor) -> Result<usize, ExportLoadError> {
        self.spec
            .descriptors
            .iter()
            .position(|known| known == descriptor)
            .ok_or(ExportLoadError::Identity)
    }

    /// One begin/call/end transaction; the call receives the result storage.
    fn transact(
        &mut self,
        descriptor: &Descriptor,
        fuel: u64,
        call: impl FnOnce(*mut InvResult) -> i32,
    ) -> Result<Invocation, ExportLoadError> {
        // SAFETY: This thread-confined owner retains the fixed-signature begin
        // pointer; validated positive fuel cannot invalidate an existing call.
        match unsafe { (self.begin)(fuel) } {
            0 => {}
            1 => return Err(ExportLoadError::Busy),
            _ => return Err(ExportLoadError::Protocol),
        }
        let transaction = EndGuard {
            end: self.end,
            active: true,
            _library: &self._library,
            _thread: PhantomData,
        };
        let mut result = InvResult {
            abi_version: ABI_VERSION,
            tag: 0,
            scalar: 0,
            data: std::ptr::null_mut(),
            len: 0,
            release: None,
        };
        let status = call(&mut result);
        let failure = transaction.finish()?;
        validate_failure(&failure, status, fuel)?;
        let output = if status == 0 {
            Some(decode_scalar(descriptor.result_tag, &result)?)
        } else {
            // Failed exports must leave every initialized public output field
            // untouched. The scalar protocol has no release callback to call.
            if result.abi_version != ABI_VERSION
                || result.tag != 0
                || result.scalar != 0
                || !result.data.is_null()
                || result.len != 0
                || result.release.is_some()
            {
                return Err(ExportLoadError::Protocol);
            }
            None
        };
        Ok(Invocation { output, failure })
    }
}

/// Public scalar wire records for `args`, typed by `tags` (scalar tags only).
fn wire_scalars(args: &[Arg<'_>], tags: &[u32]) -> Result<Vec<InvArg>, ExportLoadError> {
    let mut wire = Vec::new();
    wire.try_reserve_exact(args.len())
        .map_err(|_| ExportLoadError::Boundary(Error::Allocation))?;
    for (arg, tag) in args.iter().zip(tags) {
        let scalar = match (arg, *tag) {
            (Arg::I32(value), I32) => i64::from(*value),
            (Arg::U32(value), U32) => i64::from(*value),
            _ => return Err(ExportLoadError::Boundary(Error::Layout)),
        };
        wire.push(InvArg {
            tag: *tag,
            mode: 0,
            scalar,
            data: std::ptr::null(),
            len: 0,
        });
    }
    Ok(wire)
}

struct EndGuard<'library> {
    end: EndFn,
    active: bool,
    _library: &'library Library,
    _thread: PhantomData<Rc<()>>,
}

impl EndGuard<'_> {
    fn finish(mut self) -> Result<FailureRecord, ExportLoadError> {
        let mut failure = FailureRecord::default();
        // Disable Drop before invoking end: even a malformed end result is
        // completed exactly once and never reused as an active transaction.
        self.active = false;
        // SAFETY: This guard is created only after successful begin; the owner
        // retains the library, and end receives a valid initialized record.
        if unsafe { (self.end)(&mut failure) } != 0 {
            return Err(ExportLoadError::Protocol);
        }
        Ok(failure)
    }
}

impl Drop for EndGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            let mut failure = FailureRecord::default();
            self.active = false;
            // SAFETY: Successful begin created this same-thread transaction;
            // the library is retained and the record is writable. Drop handles
            // a Rust-side early exit without retrying or exposing a record.
            let _ = unsafe { (self.end)(&mut failure) };
        }
    }
}

fn validate_failure(record: &FailureRecord, status: i32, fuel: u64) -> Result<(), ExportLoadError> {
    if record.abi_version != ABI_VERSION
        || record.reserved != 0
        || record.fuel_limit != fuel
        || record.steps_used > fuel
        || !matches!(status, 0 | 1)
        || (status == 0) != (record.kind == 0)
        || (record.kind != 1 && (record.span_start != 0 || record.span_end != 0))
    {
        return Err(ExportLoadError::Protocol);
    }
    let valid = match record.kind {
        0 => record.reason == 0,
        // 3 = index out of bounds (HA-1.2 array exports).
        1 => matches!(record.reason, 1..=3) && record.span_start < record.span_end,
        2 => matches!(record.reason, 1 | 3 | 4 | 5),
        3 => record.reason == 0 && record.steps_used == fuel && fuel > 0,
        4 => record.reason == 1,
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(ExportLoadError::Protocol)
    }
}

fn decode_scalar(tag: u32, result: &InvResult) -> Result<Output, ExportLoadError> {
    if result.abi_version != ABI_VERSION
        || result.tag != tag
        || !result.data.is_null()
        || result.len != 0
        || result.release.is_some()
    {
        return Err(ExportLoadError::Protocol);
    }
    match tag {
        I32 => i32::try_from(result.scalar)
            .map(Output::I32)
            .map_err(|_| ExportLoadError::Protocol),
        U32 => u32::try_from(result.scalar)
            .map(Output::U32)
            .map_err(|_| ExportLoadError::Protocol),
        _ => Err(ExportLoadError::Protocol),
    }
}

#[cfg(test)]
mod tests;
