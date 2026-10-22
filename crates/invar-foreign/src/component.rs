//! Reviewed unsafe boundary for the generated component-v1 lifecycle adapter.
//! The selected foreign library promises valid C pointers and the generated header's signatures.

use std::ffi::{c_char, c_void};
use std::path::Path;

use libloading::Library;

pub mod planner;

type VersionFn = unsafe extern "C" fn() -> *const c_char;
type InitFn = unsafe extern "C" fn(*mut *mut c_void, *const u8, u32) -> i32;
type ConfigureFn = unsafe extern "C" fn(*mut c_void, *const c_void) -> i32;
type StepFn = unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void) -> i32;
type ShutdownFn = unsafe extern "C" fn(*mut *mut c_void) -> i32;

const MAX_PATH_BYTES: usize = 4096;
const MAX_VERSION_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Loaded,
    Initialized,
    Configured,
    Shutdown,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ComponentError {
    Library,
    Symbol,
    Version,
    Input,
    Lifecycle,
    Status(i32),
    Output,
    Protocol,
}

#[repr(C)]
struct FkConfig {
    model_xml_path: *const u8,
    model_xml_path_len: u32,
    model_kind: u32,
}

#[repr(C)]
struct Planar2Config {
    proximal_m: f64,
    distal_m: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub translation_m: [f64; 3],
    pub quaternion_xyzw: [f64; 4],
}

const _: () = {
    assert!(std::mem::size_of::<FkConfig>() == 16);
    assert!(std::mem::align_of::<FkConfig>() == 8);
    assert!(std::mem::offset_of!(FkConfig, model_xml_path_len) == 8);
    assert!(std::mem::offset_of!(FkConfig, model_kind) == 12);
    assert!(std::mem::size_of::<Planar2Config>() == 16);
    assert!(std::mem::align_of::<Planar2Config>() == 8);
    assert!(std::mem::size_of::<Pose>() == 56);
    assert!(std::mem::align_of::<Pose>() == 8);
    assert!(std::mem::offset_of!(Pose, quaternion_xyzw) == 24);
};

pub struct Component {
    _library: Library,
    init_fn: InitFn,
    configure_fn: ConfigureFn,
    step_fn: StepFn,
    shutdown_fn: ShutdownFn,
    context: *mut c_void,
    state: State,
    joint_count: usize,
    version: String,
}

fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, ComponentError> {
    // SAFETY: The selected library promises that this generated ABI symbol has the requested C signature.
    unsafe { library.get::<T>(name) }
        .map(|function| *function)
        .map_err(|_| ComponentError::Symbol)
}

fn name(prefix: &str, suffix: &str) -> Vec<u8> {
    format!("{prefix}_{suffix}\0").into_bytes()
}

fn path_bytes(path: &Path) -> Result<Vec<u8>, ComponentError> {
    let text = path.to_str().ok_or(ComponentError::Input)?;
    if !path.is_absolute()
        || !path.is_file()
        || text.len() > MAX_PATH_BYTES
        || text.as_bytes().contains(&0)
    {
        return Err(ComponentError::Input);
    }
    Ok(text.as_bytes().to_vec())
}

impl Component {
    /// Load an explicitly selected component-v1 adapter and verify its five symbols and version.
    pub fn open(path: &Path, prefix: &str, joint_count: usize) -> Result<Self, ComponentError> {
        Self::open_expected(path, prefix, joint_count, "invar-component-v1/")
    }

    fn open_expected(
        path: &Path,
        prefix: &str,
        joint_count: usize,
        expected_version: &str,
    ) -> Result<Self, ComponentError> {
        if !path.is_absolute()
            || !path.is_file()
            || !matches!(joint_count, 2 | 6 | 7)
            || prefix.is_empty()
            || prefix.len() > 64
            || !prefix
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
            || !prefix
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(ComponentError::Input);
        }
        // SAFETY: Loading this exact user-selected path runs its constructors; the caller opted into that foreign library.
        let library = unsafe { Library::new(path) }.map_err(|_| ComponentError::Library)?;
        let version_fn: VersionFn = symbol(&library, &name(prefix, "version"))?;
        let init_fn: InitFn = symbol(&library, &name(prefix, "init"))?;
        let configure_fn: ConfigureFn = symbol(&library, &name(prefix, "configure"))?;
        let step_fn: StepFn = symbol(&library, &name(prefix, "step"))?;
        let shutdown_fn: ShutdownFn = symbol(&library, &name(prefix, "shutdown"))?;
        // SAFETY: The generated ABI promises a pointer to a stable NUL-terminated version string.
        let version_ptr = unsafe { version_fn() };
        if version_ptr.is_null() {
            return Err(ComponentError::Version);
        }
        let mut bytes = Vec::new();
        for index in 0..MAX_VERSION_BYTES {
            // SAFETY: The selected C adapter promises readable bytes through a NUL terminator within this bound.
            let byte = unsafe { *version_ptr.add(index) } as u8;
            if byte == 0 {
                break;
            }
            bytes.push(byte);
        }
        if bytes.len() == MAX_VERSION_BYTES {
            return Err(ComponentError::Version);
        }
        let version = String::from_utf8(bytes).map_err(|_| ComponentError::Version)?;
        if !version.starts_with(expected_version) || version.len() <= expected_version.len() {
            return Err(ComponentError::Version);
        }
        Ok(Self {
            _library: library,
            init_fn,
            configure_fn,
            step_fn,
            shutdown_fn,
            context: std::ptr::null_mut(),
            state: State::Loaded,
            joint_count,
            version,
        })
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn state(&self) -> State {
        self.state
    }

    /// `child` is required for a separate-process adapter and omitted in-process.
    pub fn init(&mut self, child: Option<&Path>) -> Result<(), ComponentError> {
        if self.state != State::Loaded {
            return Err(ComponentError::Lifecycle);
        }
        let path = child.map(path_bytes).transpose()?;
        let (pointer, len) = path.as_ref().map_or((std::ptr::null(), 0), |bytes| {
            (bytes.as_ptr(), bytes.len() as u32)
        });
        // SAFETY: `self.context` starts null; the selected adapter owns any returned context and borrows path bytes only during this call.
        let status = unsafe { (self.init_fn)(&mut self.context, pointer, len) };
        if status != 0 {
            if !self.context.is_null() {
                self.state = State::Initialized;
                return Err(ComponentError::Protocol);
            }
            return Err(ComponentError::Status(status));
        }
        if self.context.is_null() {
            return Err(ComponentError::Protocol);
        }
        self.state = State::Initialized;
        Ok(())
    }

    pub fn configure_fk(&mut self, model: &Path, kind: u32) -> Result<(), ComponentError> {
        if self.state != State::Initialized {
            return Err(ComponentError::Lifecycle);
        }
        if !matches!(kind, 6 | 7) || self.joint_count != kind as usize {
            return Err(ComponentError::Input);
        }
        let bytes = path_bytes(model)?;
        let config = FkConfig {
            model_xml_path: bytes.as_ptr(),
            model_xml_path_len: bytes.len() as u32,
            model_kind: kind,
        };
        // SAFETY: The generated header fixes this config layout; path bytes stay live until the C call returns and may not be retained.
        let status = unsafe { (self.configure_fn)(self.context, (&raw const config).cast()) };
        if status != 0 {
            return Err(ComponentError::Status(status));
        }
        self.state = State::Configured;
        Ok(())
    }

    pub fn configure_planar2(
        &mut self,
        proximal_m: f64,
        distal_m: f64,
    ) -> Result<(), ComponentError> {
        if self.state != State::Initialized {
            return Err(ComponentError::Lifecycle);
        }
        if self.joint_count != 2
            || !proximal_m.is_finite()
            || !distal_m.is_finite()
            || proximal_m <= 0.0
            || distal_m <= 0.0
        {
            return Err(ComponentError::Input);
        }
        let config = Planar2Config {
            proximal_m,
            distal_m,
        };
        // SAFETY: The generated header fixes this two-f64 config layout and the C adapter borrows it only during the call.
        let status = unsafe { (self.configure_fn)(self.context, (&raw const config).cast()) };
        if status != 0 {
            return Err(ComponentError::Status(status));
        }
        self.state = State::Configured;
        Ok(())
    }

    pub fn step(&mut self, joint_rad: &[f64]) -> Result<Pose, ComponentError> {
        if self.state != State::Configured {
            return Err(ComponentError::Lifecycle);
        }
        if joint_rad.len() != self.joint_count || !joint_rad.iter().all(|value| value.is_finite()) {
            return Err(ComponentError::Input);
        }
        // The generated FK input struct has exactly one fixed joint-array field.
        let mut pose = Pose {
            translation_m: [f64::NAN; 3],
            quaternion_xyzw: [f64::NAN; 4],
        };
        // SAFETY: The joint slice and temporary output are live for this call; the generated C adapter may not retain either pointer.
        let status = unsafe {
            (self.step_fn)(
                self.context,
                joint_rad.as_ptr().cast(),
                (&raw mut pose).cast(),
            )
        };
        if status != 0 {
            return Err(ComponentError::Status(status));
        }
        let norm: f64 = pose.quaternion_xyzw.iter().map(|value| value * value).sum();
        if !pose
            .translation_m
            .iter()
            .chain(pose.quaternion_xyzw.iter())
            .all(|value| value.is_finite())
            || !norm.is_finite()
            || (norm.sqrt() - 1.0).abs() > 1e-9
        {
            return Err(ComponentError::Output);
        }
        Ok(pose)
    }

    pub fn shutdown(&mut self) -> Result<(), ComponentError> {
        if !matches!(self.state, State::Initialized | State::Configured) {
            return Err(ComponentError::Lifecycle);
        }
        // SAFETY: The selected adapter owns this context and its shutdown contract clears the pointer on success.
        let status = unsafe { (self.shutdown_fn)(&mut self.context) };
        if self.context.is_null() {
            self.state = State::Shutdown;
        }
        if status != 0 {
            return Err(ComponentError::Status(status));
        }
        if !self.context.is_null() {
            return Err(ComponentError::Protocol);
        }
        Ok(())
    }
}

impl Drop for Component {
    fn drop(&mut self) {
        if !self.context.is_null() && matches!(self.state, State::Initialized | State::Configured) {
            // SAFETY: The selected adapter owns this context and this is the final best-effort release before the library unloads.
            let _ = unsafe { (self.shutdown_fn)(&mut self.context) };
        }
    }
}
