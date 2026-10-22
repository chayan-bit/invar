//! The sole reviewed unsafe boundary for the narrow Invar C call ABI.
//! A selected C library must uphold pointer validity for its declared buffers and release callback.

use std::path::Path;

use libloading::{Library, Symbol};

pub mod component;
pub mod device;
pub mod exports;
pub mod host_array;
pub mod numeric;
pub mod numeric_owned;
pub mod numeric_owned_resume;
pub mod numeric_resume;

pub const ABI_VERSION: u32 = 1;
pub const MAX_BYTES: usize = 1_048_576;

pub const I32: u32 = 1;
pub const U32: u32 = 2;
pub const BYTES: u32 = 3;
pub const STRING: u32 = 4;
/// A value argument carrying binary64 encoding bits in `scalar`.
pub const F64: u32 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Mode {
    Value = 0,
    Read = 1,
    Take = 2,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct InvArg {
    pub tag: u32,
    pub mode: u32,
    pub scalar: i64,
    pub data: *const u8,
    pub len: u64,
}

#[repr(C)]
pub struct InvResult {
    pub abi_version: u32,
    pub tag: u32,
    pub scalar: i64,
    pub data: *mut u8,
    pub len: u64,
    pub release: Option<unsafe extern "C" fn(*mut u8, u64)>,
}

const _: () = {
    assert!(std::mem::size_of::<InvArg>() == 32);
    assert!(std::mem::align_of::<InvArg>() == 8);
    assert!(std::mem::offset_of!(InvArg, tag) == 0);
    assert!(std::mem::offset_of!(InvArg, mode) == 4);
    assert!(std::mem::offset_of!(InvArg, scalar) == 8);
    assert!(std::mem::offset_of!(InvArg, data) == 16);
    assert!(std::mem::offset_of!(InvArg, len) == 24);
    assert!(std::mem::size_of::<InvResult>() == 40);
    assert!(std::mem::align_of::<InvResult>() == 8);
    assert!(std::mem::offset_of!(InvResult, abi_version) == 0);
    assert!(std::mem::offset_of!(InvResult, tag) == 4);
    assert!(std::mem::offset_of!(InvResult, scalar) == 8);
    assert!(std::mem::offset_of!(InvResult, data) == 16);
    assert!(std::mem::offset_of!(InvResult, len) == 24);
    assert!(std::mem::offset_of!(InvResult, release) == 32);
};

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Library,
    Version,
    Symbol,
    Status(i32),
    Layout,
    Utf8,
    Limit,
    Allocation,
}

pub enum Arg<'a> {
    I32(i32),
    U32(u32),
    /// binary64 encoding.
    F64(u64),
    Bytes(&'a [u8], Mode),
    String(&'a str, Mode),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Output {
    I32(i32),
    U32(u32),
    Bytes(Vec<u8>),
    String(String),
}

impl Arg<'_> {
    fn wire(&self) -> Result<InvArg, Error> {
        if matches!(
            self,
            Self::Bytes(_, Mode::Value) | Self::String(_, Mode::Value)
        ) {
            return Err(Error::Layout);
        }
        let (tag, mode, scalar, data, len) = match self {
            Self::I32(value) => (
                I32,
                Mode::Value as u32,
                i64::from(*value),
                std::ptr::null(),
                0,
            ),
            Self::U32(value) => (
                U32,
                Mode::Value as u32,
                i64::from(*value),
                std::ptr::null(),
                0,
            ),
            Self::F64(bits) => (
                F64,
                Mode::Value as u32,
                i64::from_ne_bytes(bits.to_ne_bytes()),
                std::ptr::null(),
                0,
            ),
            Self::Bytes(value, mode) => (BYTES, *mode as u32, 0, value.as_ptr(), value.len()),
            Self::String(value, mode) => (STRING, *mode as u32, 0, value.as_ptr(), value.len()),
        };
        if len > MAX_BYTES {
            return Err(Error::Limit);
        }
        Ok(InvArg {
            tag,
            mode,
            scalar,
            data,
            len: len as u64,
        })
    }
}

/// Load only the explicit path, verify the ABI and symbol, call once, copy validated output,
/// and invoke its C-owned release callback exactly once while the library remains loaded.
pub fn call(path: &Path, symbol: &str, args: &[Arg<'_>], expected: u32) -> Result<Output, Error> {
    if !path.is_absolute() || !path.is_file() {
        return Err(Error::Library);
    }
    if args.len() > 8
        || !matches!(expected, I32 | U32 | BYTES | STRING)
        || symbol.is_empty()
        || !symbol
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(Error::Layout);
    }
    let wire = args.iter().map(Arg::wire).collect::<Result<Vec<_>, _>>()?;
    // SAFETY: Loading the explicitly selected library runs its constructors; the caller opted in.
    let library = unsafe { Library::new(path) }.map_err(|_| Error::Library)?;
    // SAFETY: The declared version symbol has the fixed C signature required by the header.
    let version: Symbol<'_, unsafe extern "C" fn() -> u32> =
        unsafe { library.get(b"inv_abi_version_v1\0") }.map_err(|_| Error::Version)?;
    // SAFETY: The selected C library promises this symbol's C signature; the version is checked first.
    if unsafe { version() } != ABI_VERSION {
        return Err(Error::Version);
    }
    let mut name = symbol.as_bytes().to_vec();
    name.push(0);
    // SAFETY: The versioned shim contract assigns this fixed signature to declared symbols.
    let function: Symbol<'_, unsafe extern "C" fn(*const InvArg, u32, *mut InvResult) -> i32> =
        unsafe { library.get(name.as_slice()) }.map_err(|_| Error::Symbol)?;
    let mut result = InvResult {
        abi_version: ABI_VERSION,
        tag: 0,
        scalar: 0,
        data: std::ptr::null_mut(),
        len: 0,
        release: None,
    };
    // SAFETY: Arguments refer to live Rust buffers for the call; the C shim may read but never retain or free them.
    let status = unsafe { function(wire.as_ptr(), wire.len() as u32, &mut result) };
    let decoded = decode(status, expected, &result);
    if !result.data.is_null()
        && let Some(release) = result.release
    {
        // SAFETY: The versioned C shim owns this returned pointer and promises this callback accepts it once.
        unsafe {
            release(result.data, result.len);
        }
    }
    decoded
}

fn decode(status: i32, expected: u32, result: &InvResult) -> Result<Output, Error> {
    if status != 0 {
        return Err(Error::Status(status));
    }
    if result.abi_version != ABI_VERSION || result.tag != expected {
        return Err(Error::Layout);
    }
    if matches!(expected, I32 | U32) {
        if !result.data.is_null() || result.len != 0 || result.release.is_some() {
            return Err(Error::Layout);
        }
        return if expected == I32 {
            i32::try_from(result.scalar)
                .map(Output::I32)
                .map_err(|_| Error::Layout)
        } else {
            u32::try_from(result.scalar)
                .map(Output::U32)
                .map_err(|_| Error::Layout)
        };
    }
    let len = usize::try_from(result.len).map_err(|_| Error::Limit)?;
    if len > MAX_BYTES {
        return Err(Error::Limit);
    }
    if (result.data.is_null() && len != 0) || result.data.is_null() == result.release.is_some() {
        return Err(Error::Layout);
    }
    let mut copy = Vec::new();
    copy.try_reserve_exact(len).map_err(|_| Error::Allocation)?;
    if len > 0 {
        // SAFETY: The versioned C shim promises the pointer is valid for `len` bytes until release.
        let bytes = unsafe { std::slice::from_raw_parts(result.data.cast_const(), len) };
        copy.extend_from_slice(bytes);
    }
    if expected == BYTES {
        Ok(Output::Bytes(copy))
    } else {
        String::from_utf8(copy)
            .map(Output::String)
            .map_err(|_| Error::Utf8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        process::Command,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn issue_l1_04_bytes_strings_cabi_c_owned_and_invar_owned_transfers() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let directory = std::env::temp_dir().join(format!(
            "invar-foreign-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let library_path = directory.join(if cfg!(target_os = "macos") {
            "libfixture.dylib"
        } else {
            "libfixture.so"
        });
        let mut compiler = Command::new("cc");
        compiler
            .arg(if cfg!(target_os = "macos") {
                "-dynamiclib"
            } else {
                "-shared"
            })
            .arg("-fPIC")
            .arg("-std=c11")
            .arg("-Wall")
            .arg("-Werror")
            .arg("-I")
            .arg(root.join("include"))
            .arg(root.join("tests/fixtures/l1_04/foreign.c"))
            .arg("-o")
            .arg(&library_path);
        assert!(compiler.status().unwrap().success());
        {
            // SAFETY: The test fixture is compiled from the checked repository C source above.
            let library = unsafe { Library::new(&library_path) }.unwrap();
            // SAFETY: The fixture exports this exact test-only counter signature.
            let releases: Symbol<'_, unsafe extern "C" fn() -> u32> =
                unsafe { library.get(b"inv_release_count_v1\0") }.unwrap();
            let count = || {
                // SAFETY: `library` stays loaded and the fixture's counter function takes no arguments.
                unsafe { releases() }
            };
            assert_eq!(
                call(&library_path, "c_add", &[Arg::I32(3), Arg::I32(4)], I32),
                Ok(Output::I32(7))
            );
            assert_eq!(
                call(
                    &library_path,
                    "c_copy_read",
                    &[Arg::Bytes(b"abc", Mode::Read)],
                    U32
                ),
                Ok(Output::U32(3))
            );
            assert_eq!(
                call(
                    &library_path,
                    "c_copy_take",
                    &[Arg::Bytes(b"abc", Mode::Take)],
                    BYTES
                ),
                Ok(Output::Bytes(b"abc".to_vec()))
            );
            assert_eq!(count(), 1);
            assert_eq!(
                call(&library_path, "c_text", &[], STRING),
                Ok(Output::String("é".to_owned()))
            );
            assert_eq!(count(), 2);
            assert_eq!(
                call(&library_path, "c_status", &[], BYTES),
                Err(Error::Status(42))
            );
            assert_eq!(count(), 3);
            assert_eq!(
                call(&library_path, "c_bad_utf8", &[], STRING),
                Err(Error::Utf8)
            );
            assert_eq!(count(), 4);
            assert_eq!(
                call(&library_path, "c_overflow_len", &[], BYTES),
                Err(Error::Limit)
            );
            assert_eq!(count(), 5);
            assert_eq!(
                call(&library_path, "c_wrong_tag", &[], I32),
                Err(Error::Layout)
            );
            assert_eq!(
                call(&library_path, "c_borrowed_return", &[], BYTES),
                Err(Error::Layout)
            );
            assert_eq!(
                call(&library_path, "c_null_data", &[], BYTES),
                Err(Error::Layout)
            );
            assert_eq!(count(), 5);
            assert_eq!(call(&library_path, "missing", &[], I32), Err(Error::Symbol));
        }
        let wrong_version = directory.join(if cfg!(target_os = "macos") {
            "libwrong.dylib"
        } else {
            "libwrong.so"
        });
        let status = Command::new("cc")
            .arg(if cfg!(target_os = "macos") {
                "-dynamiclib"
            } else {
                "-shared"
            })
            .arg("-fPIC")
            .arg("-std=c11")
            .arg("-DINV_BAD_VERSION")
            .arg("-I")
            .arg(root.join("include"))
            .arg(root.join("tests/fixtures/l1_04/foreign.c"))
            .arg("-o")
            .arg(&wrong_version)
            .status()
            .unwrap();
        assert!(status.success());
        assert_eq!(
            call(&wrong_version, "c_add", &[Arg::I32(3), Arg::I32(4)], I32),
            Err(Error::Version)
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
