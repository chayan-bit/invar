//! Compiled-ahead Metal libraries, the Metal counterpart of the CUDA image in
//! `general_net::device_host::cuda_aot`.
//!
//! `invar build --target metal` runs Apple's offline compiler once, at build
//! time, and writes this container. A production run reads it as untrusted
//! bytes, checks every field it relies on, matches it to the program it is
//! asked to run, and loads each library with `newLibraryWithData`. Nothing on
//! that path compiles MSL source.
//!
//! Layout (integers little-endian):
//!
//! | bytes          | field                                                 |
//! |----------------|-------------------------------------------------------|
//! | `0..8`         | magic `INVGMETL`                                      |
//! | `8..12`        | format version, `1`                                   |
//! | `12..16`       | header length `H`, `1..=MAX_HEADER_BYTES`             |
//! | `16..24`       | image length `I`, `1..=MAX_IMAGE_BYTES`               |
//! | `24..56`       | SHA-256 of bytes `0..24`, the header and the image    |
//! | `56..56+H`     | header: canonical JSON with an exact key set          |
//! | `56+H..56+H+I` | image: the header's `.metallib` files, in its order   |
//!
//! Each library is the Metal IR of one generated MSL source, keyed by that
//! source's SHA-256. The digests give integrity, not authenticity: anyone can
//! write a well-formed container, so every field is validated and bounded
//! before Metal sees a library, and Metal remains the final loader.
pub use crate::hex::hex;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

pub mod build;
mod kernels;
pub use kernels::{gnet_kernels, region_msl, run_kernels};

pub const MAGIC: [u8; 8] = *b"INVGMETL";
pub const FORMAT_VERSION: u32 = 1;
pub const SCHEMA: &str = "invar-metal-library-v1";
/// The ABI5 general-net entry point, as in the CUDA image.
pub const GNET_KERNEL: &str = "invar_gnet_children_v5";
pub const MAX_HEADER_BYTES: usize = 64 * 1024;
/// Matches `invar_metal::MAX_LIBRARIES`, the libraries one device may hold.
pub const MAX_LIBRARIES: usize = 64;
pub const MAX_LIBRARY_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_IMAGE_BYTES: usize = 64 * 1024 * 1024;
pub const PREFIX_BYTES: usize = 56;
/// Largest well-formed container; a reader may stop one byte past it.
pub const MAX_CONTAINER_BYTES: usize = PREFIX_BYTES + MAX_HEADER_BYTES + MAX_IMAGE_BYTES;
/// Strict numerics: the same MSL version and fast-math policy as the
/// developer source path. Contraction pragmas live in the MSL itself.
pub const LANGUAGE: &str = "metal3.1";
pub const MIN_OS: (u16, u16) = (14, 0);
const METALLIB_MACOS: u8 = 0x81;

/// The `kernel void NAME(` entry points generated MSL declares, sorted.
pub fn msl_entry_points(msl: &str) -> Vec<String> {
    let mut names = msl
        .split("kernel void ")
        .skip(1)
        .filter_map(|rest| rest.split_once('(').map(|(name, _)| name.trim().to_owned()))
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

/// The exact `metal` front-end options; `metallib` takes none.
pub fn options() -> [String; 3] {
    [
        format!("-std={LANGUAGE}"),
        "-fno-fast-math".into(),
        format!("-mmacosx-version-min={}.{}", MIN_OS.0, MIN_OS.1),
    ]
}

/// What the container was built for. Each route has its own kernel ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// `invar run --engine metal|auto`: every Metal region plus the probe.
    Run,
    /// `invar gnet-device ... metal`: the general-net kernel and any D1 probe.
    Gnet,
    /// `invar source-device PATH tensor ENTRY metal FUEL`: the D1 probe plus
    /// every strict contraction the source reaches.
    Tensor,
}

impl Route {
    fn name(self) -> &'static str {
        match self {
            Self::Run => "run-device-ir-v1",
            Self::Gnet => "gnet-abi5",
            Self::Tensor => "tensor-f32-v1",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Truncated,
    Magic,
    Version(u32),
    Size,
    Digest,
    Header(&'static str),
    Metallib(&'static str),
    /// The container was built from another program, route or entry: stale
    /// or for a different contract.
    Program(&'static str),
    /// Built with options or a target other than the pinned ones.
    Toolchain(&'static str),
    /// A library does not export exactly the functions it records.
    EntryPoint(String),
    /// The artifact file is not a private regular file.
    Permissions,
    Io(String),
    /// Metal refused a library; it is never recompiled from source instead.
    Rejected(String),
    /// Build time only: the offline compiler is not installed.
    MissingToolchain(String),
    /// Build time only: the offline compiler failed.
    Compile(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Library {
    pub msl_sha256: [u8; 32],
    pub functions: Vec<String>,
    pub bytes: usize,
    pub sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Toolchain {
    pub metal: String,
    pub metallib: String,
    pub sdk: String,
}

/// Everything the container records about its program, target and toolchain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub route: Route,
    pub entry: String,
    pub source_sha256: [u8; 32],
    pub toolchain: Toolchain,
    pub invar: String,
    pub libraries: Vec<Library>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    header: Header,
    container_sha256: [u8; 32],
    image: Vec<u8>,
}

impl Header {
    /// The container must be the build of exactly this program and route.
    pub fn verify(&self, route: Route, entry: &str, source_sha256: [u8; 32]) -> Result<(), Error> {
        if self.route != route {
            return Err(Error::Program("route"));
        }
        if self.entry != entry {
            return Err(Error::Program("entry"));
        }
        if self.source_sha256 != source_sha256 {
            return Err(Error::Program("source_sha256"));
        }
        Ok(())
    }

    /// The library built from exactly this MSL, if the container holds one.
    pub fn library(&self, msl: &str) -> Option<&Library> {
        let digest: [u8; 32] = Sha256::digest(msl.as_bytes()).into();
        self.libraries.iter().find(|l| l.msl_sha256 == digest)
    }

    /// Require the exact GNET library inventory before installing any library.
    pub fn verify_gnet(
        &self,
        artifact: &crate::general_net::compiled::Artifact,
    ) -> Result<(), Error> {
        let kernels = gnet_kernels(artifact);
        if self.libraries.len() != kernels.len() {
            return Err(Error::Program("libraries"));
        }
        for (msl, functions) in kernels {
            let library = self.library(&msl).ok_or(Error::Program("msl_sha256"))?;
            if library.functions != functions {
                return Err(Error::EntryPoint(format!("{:?}", library.functions)));
            }
        }
        Ok(())
    }

    fn json(&self) -> Value {
        json!({
            "schema": SCHEMA,
            "route": self.route.name(),
            "entry": self.entry,
            "program": {"source_sha256": hex(&self.source_sha256)},
            "target": {"platform": "macosx", "language": LANGUAGE,
                "min_os": format!("{}.{}", MIN_OS.0, MIN_OS.1)},
            "toolchain": {
                "compiler": "xcrun metal + metallib",
                "metal": self.toolchain.metal,
                "metallib": self.toolchain.metallib,
                "sdk": self.toolchain.sdk,
                "options": options(),
                "invar": self.invar,
            },
            "libraries": self.libraries.iter().map(|l| json!({
                "msl_sha256": hex(&l.msl_sha256),
                "functions": l.functions,
                "bytes": l.bytes,
                "sha256": hex(&l.sha256),
            })).collect::<Vec<_>>(),
        })
    }

    fn parse(bytes: &[u8]) -> Result<Self, Error> {
        let value: Value = serde_json::from_slice(bytes).map_err(|_| Error::Header("json"))?;
        // Canonical form only: no whitespace, no duplicate or reordered keys.
        if serde_json::to_vec(&value).map_err(|_| Error::Header("json"))? != bytes {
            return Err(Error::Header("canonical"));
        }
        let top = exact(
            &value,
            &[
                "schema",
                "route",
                "entry",
                "program",
                "target",
                "toolchain",
                "libraries",
            ],
        )?;
        if top["schema"] != SCHEMA {
            return Err(Error::Header("schema"));
        }
        let route = match top["route"].as_str() {
            Some(r) if r == Route::Run.name() => Route::Run,
            Some(r) if r == Route::Gnet.name() => Route::Gnet,
            Some(r) if r == Route::Tensor.name() => Route::Tensor,
            _ => return Err(Error::Header("route")),
        };
        let program = exact(&top["program"], &["source_sha256"])?;
        let target = exact(&top["target"], &["platform", "language", "min_os"])?;
        let toolchain = exact(
            &top["toolchain"],
            &["compiler", "metal", "metallib", "sdk", "options", "invar"],
        )?;
        if target["platform"] != "macosx"
            || target["language"] != LANGUAGE
            || target["min_os"] != format!("{}.{}", MIN_OS.0, MIN_OS.1)
        {
            return Err(Error::Toolchain("target"));
        }
        if toolchain["compiler"] != "xcrun metal + metallib" {
            return Err(Error::Toolchain("compiler"));
        }
        if toolchain["options"] != json!(options()) {
            return Err(Error::Toolchain("options"));
        }
        let list = top["libraries"]
            .as_array()
            .filter(|l| (1..=MAX_LIBRARIES).contains(&l.len()))
            .ok_or(Error::Header("libraries"))?;
        let libraries = list.iter().map(library).collect::<Result<Vec<_>, _>>()?;
        // Sorted by MSL digest and unique, so one MSL maps to one library.
        if libraries
            .windows(2)
            .any(|w| w[0].msl_sha256 >= w[1].msl_sha256)
        {
            return Err(Error::Header("library order"));
        }
        Ok(Self {
            route,
            entry: text(&top["entry"], 4096, "entry")?,
            source_sha256: digest(&program["source_sha256"])?,
            toolchain: Toolchain {
                metal: text(&toolchain["metal"], 256, "metal")?,
                metallib: text(&toolchain["metallib"], 256, "metallib")?,
                sdk: text(&toolchain["sdk"], 64, "sdk")?,
            },
            invar: text(&toolchain["invar"], 64, "invar")?,
            libraries,
        })
    }
}

fn library(value: &Value) -> Result<Library, Error> {
    let map = exact(value, &["msl_sha256", "functions", "bytes", "sha256"])?;
    let functions = map["functions"]
        .as_array()
        .filter(|f| (1..=8).contains(&f.len()))
        .ok_or(Error::Header("functions"))?
        .iter()
        .map(|f| text(f, 128, "functions"))
        .collect::<Result<Vec<_>, _>>()?;
    if functions
        .iter()
        .any(|f| !f.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'))
    {
        return Err(Error::Header("functions"));
    }
    Ok(Library {
        msl_sha256: digest(&map["msl_sha256"])?,
        functions,
        bytes: map["bytes"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| (1..=MAX_LIBRARY_BYTES).contains(n))
            .ok_or(Error::Header("bytes"))?,
        sha256: digest(&map["sha256"])?,
    })
}

impl Image {
    pub fn header(&self) -> &Header {
        &self.header
    }
    /// Each recorded library with its validated `.metallib` bytes.
    pub fn libraries(&self) -> impl Iterator<Item = (&Library, &[u8])> {
        let mut at = 0;
        self.header.libraries.iter().map(move |library| {
            let bytes = &self.image[at..at + library.bytes];
            at += library.bytes;
            (library, bytes)
        })
    }
    /// The receipt of a load: everything recorded plus how it was loaded.
    pub fn receipt(&self) -> Value {
        let mut receipt = self.header.json();
        receipt["format_version"] = json!(FORMAT_VERSION);
        receipt["container_sha256"] = json!(hex(&self.container_sha256));
        receipt["loaded_via"] = json!("newLibraryWithData");
        receipt["runtime_compiler"] = json!("none");
        receipt
    }

    /// Load every library into `device`, which then refuses any kernel whose
    /// exact MSL is not among them.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn install(&self, device: &mut invar_metal::MetalDevice) -> Result<(), Error> {
        for (library, bytes) in self.libraries() {
            let functions = library
                .functions
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            device
                .load_library_with_identity(
                    library.msl_sha256,
                    bytes,
                    &functions,
                    &self.header.json().to_string(),
                )
                .map_err(|e| match e {
                    invar_metal::Error::Library(invar_metal::LibraryError::EntryPoint(m)) => {
                        Error::EntryPoint(m)
                    }
                    other => Error::Rejected(format!("{other:?}")),
                })?;
        }
        Ok(())
    }
}

/// Write a container from libraries sorted and unique by MSL digest.
pub fn encode(header: &Header, libraries: &[Vec<u8>]) -> Result<Vec<u8>, Error> {
    let text = serde_json::to_vec(&header.json()).map_err(|_| Error::Header("json"))?;
    let image = libraries.concat();
    if text.len() > MAX_HEADER_BYTES
        || image.is_empty()
        || image.len() > MAX_IMAGE_BYTES
        || libraries.len() != header.libraries.len()
    {
        return Err(Error::Size);
    }
    let mut out = Vec::with_capacity(PREFIX_BYTES + text.len() + image.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(
        &u32::try_from(text.len())
            .map_err(|_| Error::Size)?
            .to_le_bytes(),
    );
    out.extend_from_slice(
        &u64::try_from(image.len())
            .map_err(|_| Error::Size)?
            .to_le_bytes(),
    );
    let digest = container_digest(&out, &text, &image);
    out.extend_from_slice(&digest);
    out.extend_from_slice(&text);
    out.extend_from_slice(&image);
    Ok(out)
}

/// Read a container from untrusted bytes. Sizes are checked before anything
/// is hashed or parsed; every library must be a macOS metallib of its size.
pub fn decode(bytes: &[u8]) -> Result<Image, Error> {
    let prefix = bytes.get(..PREFIX_BYTES).ok_or(Error::Truncated)?;
    if prefix[..8] != MAGIC {
        return Err(Error::Magic);
    }
    let version = u32::from_le_bytes(le(prefix, 8).ok_or(Error::Truncated)?);
    if version != FORMAT_VERSION {
        return Err(Error::Version(version));
    }
    let header_len = usize::try_from(u32::from_le_bytes(le(prefix, 12).ok_or(Error::Size)?))
        .map_err(|_| Error::Size)?;
    let image_len = usize::try_from(u64::from_le_bytes(le(prefix, 16).ok_or(Error::Size)?))
        .map_err(|_| Error::Size)?;
    if !(1..=MAX_HEADER_BYTES).contains(&header_len) || !(1..=MAX_IMAGE_BYTES).contains(&image_len)
    {
        return Err(Error::Size);
    }
    let total = PREFIX_BYTES + header_len + image_len;
    if bytes.len() < total {
        return Err(Error::Truncated);
    }
    if bytes.len() > total {
        return Err(Error::Size);
    }
    let text = &bytes[PREFIX_BYTES..PREFIX_BYTES + header_len];
    let image = &bytes[PREFIX_BYTES + header_len..];
    let container_sha256 = container_digest(&prefix[..24], text, image);
    if prefix[24..56] != container_sha256 {
        return Err(Error::Digest);
    }
    let header = Header::parse(text)?;
    if header.libraries.iter().map(|l| l.bytes).sum::<usize>() != image.len() {
        return Err(Error::Digest);
    }
    let mut at = 0;
    for library in &header.libraries {
        let bytes = &image[at..at + library.bytes];
        at += library.bytes;
        if library.sha256 != <[u8; 32]>::from(Sha256::digest(bytes)) {
            return Err(Error::Digest);
        }
        metallib(bytes)?;
    }
    Ok(Image {
        header,
        container_sha256,
        image: image.to_vec(),
    })
}

/// The metallib header fields the loader relies on: magic, a macOS target at
/// the pinned deployment version, and a recorded file size equal to ours.
pub fn metallib(bytes: &[u8]) -> Result<(), Error> {
    if bytes.get(..4) != Some(b"MTLB") {
        return Err(Error::Metallib("magic"));
    }
    if bytes.get(11) != Some(&METALLIB_MACOS) {
        return Err(Error::Metallib("platform"));
    }
    let os = (
        le(bytes, 12).map(u16::from_le_bytes),
        le(bytes, 14).map(u16::from_le_bytes),
    );
    if os != (Some(MIN_OS.0), Some(MIN_OS.1)) {
        return Err(Error::Metallib("deployment target"));
    }
    let size = le(bytes, 16).map(u64::from_le_bytes);
    if size != Some(bytes.len() as u64) {
        return Err(Error::Metallib("file size"));
    }
    Ok(())
}

fn container_digest(prefix: &[u8], header: &[u8], image: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(&prefix[..24]);
    hash.update(header);
    hash.update(image);
    hash.finalize().into()
}

fn le<const N: usize>(bytes: &[u8], at: usize) -> Option<[u8; N]> {
    at.checked_add(N)
        .and_then(|end| bytes.get(at..end))
        .and_then(|b| b.try_into().ok())
}

fn exact<'a>(value: &'a Value, keys: &[&str]) -> Result<&'a Map<String, Value>, Error> {
    let map = value.as_object().ok_or(Error::Header("object"))?;
    if map.len() != keys.len() || keys.iter().any(|k| !map.contains_key(*k)) {
        return Err(Error::Header("keys"));
    }
    Ok(map)
}

fn text(value: &Value, cap: usize, field: &'static str) -> Result<String, Error> {
    value
        .as_str()
        .filter(|s| !s.is_empty() && s.len() <= cap)
        .map(Into::into)
        .ok_or(Error::Header(field))
}

fn digest(value: &Value) -> Result<[u8; 32], Error> {
    let s = value.as_str().ok_or(Error::Header("digest"))?;
    if s.len() != 64 || !s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(Error::Header("digest"));
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte =
            u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| Error::Header("digest"))?;
    }
    Ok(out)
}
