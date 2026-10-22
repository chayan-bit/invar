//! Compiled-ahead CUDA image for the ordinary-source ABI5 route.
//!
//! `invar build --target cuda` runs NVRTC once, at build time, and writes this
//! container. A production run reads it as untrusted bytes, checks every field
//! it relies on, matches it to the program it is asked to run, and loads the
//! CUBIN through the Driver API alone. Nothing here compiles anything.
//!
//! Layout (integers little-endian):
//!
//! | bytes          | field                                              |
//! |----------------|----------------------------------------------------|
//! | `0..8`         | magic `INVGCUDA`                                   |
//! | `8..12`        | format version, `1`                                |
//! | `12..16`       | header length `H`, `1..=MAX_HEADER_BYTES`          |
//! | `16..24`       | image length `I`, `1..=MAX_IMAGE_BYTES`            |
//! | `24..56`       | SHA-256 of bytes `0..24`, the header and the image |
//! | `56..56+H`     | header: canonical JSON with an exact key set       |
//! | `56+H..56+H+I` | image: one CUBIN (CUDA ELF64); nothing follows it  |
//!
//! The digests give integrity, not authenticity: anyone can write a
//! well-formed container, so every field is validated and bounded before the
//! driver sees the image, and the driver remains the final loader.
use super::Plan;
use crate::general_net::compiled::Target;
use crate::hex::hex;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

pub const MAGIC: [u8; 8] = *b"INVGCUDA";
pub const FORMAT_VERSION: u32 = 1;
pub const SCHEMA: &str = "invar-gnet-cuda-image-v1";
pub const KERNEL: &str = "invar_gnet_children_v5";
pub const MAX_HEADER_BYTES: usize = 16 * 1024;
/// Matches the NVRTC output cap of the build-time compiler.
pub const MAX_IMAGE_BYTES: usize = 16 * 1024 * 1024;
pub const PREFIX_BYTES: usize = 56;
/// Largest well-formed container; a reader may stop one byte past it.
pub const MAX_CONTAINER_BYTES: usize = PREFIX_BYTES + MAX_HEADER_BYTES + MAX_IMAGE_BYTES;
const EM_CUDA: u16 = 190;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Truncated,
    Magic,
    Version(u32),
    Size,
    Digest,
    Header(&'static str),
    Elf(&'static str),
    /// The image was built from a different program than the one to run.
    Program(&'static str),
    /// The image's SASS targets a different architecture than this device.
    Sm {
        image: u32,
        device: u32,
    },
    /// The driver predates the toolchain that produced the image.
    Driver {
        required: u32,
        found: u32,
    },
    /// The driver refused the image; it is never recompiled instead.
    Rejected(String),
}

/// Everything the container records about its program, target and toolchain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub entry: String,
    pub abi: u32,
    pub source_sha256: [u8; 32],
    pub device_source_sha256: [u8; 32],
    pub sm: u32,
    pub nvrtc: (u32, u32),
    pub invar: String,
    pub image_bytes: usize,
    pub image_sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    header: Header,
    container_sha256: [u8; 32],
    cubin: Vec<u8>,
}

/// The exact NVRTC options of the build-time compiler for `sm`.
pub fn options(sm: u32) -> [String; 6] {
    [
        format!("--gpu-architecture=sm_{sm}"),
        "--std=c++17".into(),
        "--fmad=false".into(),
        "--ftz=false".into(),
        "--prec-div=true".into(),
        "--prec-sqrt=true".into(),
    ]
}

/// Lowest `cuDriverGetVersion` accepted for an image from NVRTC `major.minor`.
pub fn min_driver((major, minor): (u32, u32)) -> u32 {
    major
        .saturating_mul(1000)
        .saturating_add(minor.saturating_mul(10))
}

impl Header {
    /// The identity of `plan` with image fields left for the caller to fill.
    pub fn of(plan: &Plan, sm: u32, nvrtc: (u32, u32), cubin: &[u8]) -> Self {
        Self {
            entry: plan.entry().into(),
            abi: plan.artifact().abi_version(),
            source_sha256: plan.source_digest(),
            device_source_sha256: Sha256::digest(plan.artifact().source()).into(),
            sm,
            nvrtc,
            invar: env!("CARGO_PKG_VERSION").into(),
            image_bytes: cubin.len(),
            image_sha256: Sha256::digest(cubin).into(),
        }
    }

    /// The image must be the build of exactly this program: same entry, same
    /// ordinary source, same generated device source and ABI, CUDA target.
    pub fn verify(&self, plan: &Plan) -> Result<(), Error> {
        let artifact = plan.artifact();
        if artifact.target() != Target::Cuda {
            return Err(Error::Program("target"));
        }
        if self.abi != artifact.abi_version() {
            return Err(Error::Program("abi"));
        }
        if self.entry != plan.entry() {
            return Err(Error::Program("entry"));
        }
        if self.source_sha256 != plan.source_digest() {
            return Err(Error::Program("source_sha256"));
        }
        if self.device_source_sha256 != <[u8; 32]>::from(Sha256::digest(artifact.source())) {
            return Err(Error::Program("device_source_sha256"));
        }
        Ok(())
    }

    /// Exact architecture and a driver no older than the image's toolchain.
    pub fn admit_device(&self, sm: u32, driver: u32) -> Result<(), Error> {
        if self.sm != sm {
            return Err(Error::Sm {
                image: self.sm,
                device: sm,
            });
        }
        let required = min_driver(self.nvrtc);
        if driver < required {
            return Err(Error::Driver {
                required,
                found: driver,
            });
        }
        Ok(())
    }

    fn json(&self) -> Value {
        json!({
            "schema": SCHEMA,
            "abi": self.abi,
            "kernel": KERNEL,
            "entry": self.entry,
            "program": {
                "source_sha256": hex(&self.source_sha256),
                "device_source_sha256": hex(&self.device_source_sha256),
            },
            "target": {"sm": self.sm, "min_driver": min_driver(self.nvrtc)},
            "toolchain": {
                "compiler": "nvrtc",
                "nvrtc": format!("{}.{}", self.nvrtc.0, self.nvrtc.1),
                "options": options(self.sm),
                "invar": self.invar,
            },
            "image": {"format": "cubin", "bytes": self.image_bytes, "sha256": hex(&self.image_sha256)},
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
                "abi",
                "kernel",
                "entry",
                "program",
                "target",
                "toolchain",
                "image",
            ],
        )?;
        if top["schema"] != SCHEMA || top["kernel"] != KERNEL {
            return Err(Error::Header("schema"));
        }
        let entry = text(&top["entry"], 4096, "entry")?;
        let abi = small(&top["abi"], "abi")?;
        let program = exact(&top["program"], &["source_sha256", "device_source_sha256"])?;
        let target = exact(&top["target"], &["sm", "min_driver"])?;
        let toolchain = exact(
            &top["toolchain"],
            &["compiler", "nvrtc", "options", "invar"],
        )?;
        let image = exact(&top["image"], &["format", "bytes", "sha256"])?;
        let sm = small(&target["sm"], "sm")?;
        if !(50..=255).contains(&sm) {
            return Err(Error::Header("sm"));
        }
        let nvrtc = version(&toolchain["nvrtc"])?;
        if toolchain["compiler"] != "nvrtc"
            || toolchain["options"] != json!(options(sm))
            || target["min_driver"] != min_driver(nvrtc)
        {
            return Err(Error::Header("toolchain"));
        }
        if image["format"] != "cubin" {
            return Err(Error::Header("image"));
        }
        Ok(Self {
            entry,
            abi,
            source_sha256: digest(&program["source_sha256"])?,
            device_source_sha256: digest(&program["device_source_sha256"])?,
            sm,
            nvrtc,
            invar: text(&toolchain["invar"], 64, "invar")?,
            image_bytes: image["bytes"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or(Error::Header("image"))?,
            image_sha256: digest(&image["sha256"])?,
        })
    }
}

impl Image {
    pub fn header(&self) -> &Header {
        &self.header
    }
    pub fn cubin(&self) -> &[u8] {
        &self.cubin
    }
    /// The receipt of a load: everything recorded plus how it was loaded.
    pub fn receipt(&self) -> Value {
        let mut receipt = self.header.json();
        receipt["format_version"] = json!(FORMAT_VERSION);
        receipt["container_sha256"] = json!(hex(&self.container_sha256));
        receipt["loaded_via"] = json!("cuModuleLoadData");
        receipt["runtime_compiler"] = json!("none");
        receipt
    }
}

/// Write a container. The header's image fields must describe `cubin`.
pub fn encode(header: &Header, cubin: &[u8]) -> Result<Vec<u8>, Error> {
    let text = serde_json::to_vec(&header.json()).map_err(|_| Error::Header("json"))?;
    if text.len() > MAX_HEADER_BYTES || cubin.is_empty() || cubin.len() > MAX_IMAGE_BYTES {
        return Err(Error::Size);
    }
    let mut out = Vec::with_capacity(PREFIX_BYTES + text.len() + cubin.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(
        &u32::try_from(text.len())
            .map_err(|_| Error::Size)?
            .to_le_bytes(),
    );
    out.extend_from_slice(
        &u64::try_from(cubin.len())
            .map_err(|_| Error::Size)?
            .to_le_bytes(),
    );
    let digest = container_digest(&out, &text, cubin);
    out.extend_from_slice(&digest);
    out.extend_from_slice(&text);
    out.extend_from_slice(cubin);
    Ok(out)
}

/// Read a container from untrusted bytes. Sizes are checked before anything
/// is hashed or parsed; the CUBIN's ELF header and tables must lie in bounds.
pub fn decode(bytes: &[u8]) -> Result<Image, Error> {
    let prefix = bytes.get(..PREFIX_BYTES).ok_or(Error::Truncated)?;
    if prefix[..8] != MAGIC {
        return Err(Error::Magic);
    }
    let version = u32::from_le_bytes(le(prefix, 8)?);
    if version != FORMAT_VERSION {
        return Err(Error::Version(version));
    }
    let header_len =
        usize::try_from(u32::from_le_bytes(le(prefix, 12)?)).map_err(|_| Error::Size)?;
    let image_len =
        usize::try_from(u64::from_le_bytes(le(prefix, 16)?)).map_err(|_| Error::Size)?;
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
    let cubin = &bytes[PREFIX_BYTES + header_len..];
    let container_sha256 = container_digest(&prefix[..24], text, cubin);
    if prefix[24..56] != container_sha256 {
        return Err(Error::Digest);
    }
    let header = Header::parse(text)?;
    if header.image_bytes != cubin.len()
        || header.image_sha256 != <[u8; 32]>::from(Sha256::digest(cubin))
    {
        return Err(Error::Digest);
    }
    if cubin_sm(cubin)? != header.sm {
        return Err(Error::Elf("architecture differs from header"));
    }
    Ok(Image {
        header,
        container_sha256,
        cubin: cubin.to_vec(),
    })
}

/// Validate the CUDA ELF64 fields the loader relies on and return the SASS
/// architecture. `cuModuleLoadData` takes no length, so every table and
/// file-backed section or segment must lie inside the buffer.
pub fn cubin_sm(image: &[u8]) -> Result<u32, Error> {
    let u16_at = |at: usize| le(image, at).map(u16::from_le_bytes);
    let u32_at = |at: usize| le(image, at).map(u32::from_le_bytes);
    let u64_at = |at: usize| {
        le(image, at)
            .map(u64::from_le_bytes)
            .and_then(|n| usize::try_from(n).map_err(|_| Error::Elf("offset")))
    };
    let ident = image.get(..16).ok_or(Error::Elf("short"))?;
    if ident[..4] != *b"\x7fELF" || ident[4] != 2 || ident[5] != 1 || ident[6] != 1 {
        return Err(Error::Elf("not a little-endian ELF64"));
    }
    if u16_at(16)? != 2 || u16_at(18)? != EM_CUDA || u16_at(52)? != 64 {
        return Err(Error::Elf("not a CUDA executable"));
    }
    let flags = u32_at(48)?;
    // CUDA ELF ABI 7 keeps the SM in the low flag byte; ABI 8 (sm_100 and
    // later, NVRTC 12.8+) moved it to the second byte.
    let sm = match (ident[7], ident[8]) {
        (0x33, 7) => flags & 0xff,
        (0x41, 8) => (flags >> 8) & 0xff,
        _ => return Err(Error::Elf("unknown CUDA ELF ABI")),
    };
    let within = |offset: usize, size: usize| {
        offset
            .checked_add(size)
            .is_some_and(|end| end <= image.len())
    };
    let table = |offset: usize, count: u16, entry: u16, want: u16| {
        let count = usize::from(count);
        if count > 0 && (entry != want || !within(offset, count * usize::from(want))) {
            return Err(Error::Elf("table out of bounds"));
        }
        Ok(count)
    };
    let (phoff, shoff) = (u64_at(32)?, u64_at(40)?);
    let segments = table(phoff, u16_at(56)?, u16_at(54)?, 56)?;
    let sections = table(shoff, u16_at(60)?, u16_at(58)?, 64)?;
    if sections == 0 || usize::from(u16_at(62)?) >= sections {
        return Err(Error::Elf("section table"));
    }
    let mut kernel = false;
    for i in 0..sections {
        let at = shoff + i * 64;
        let kind = u32_at(at + 4)?;
        // SHT_NOBITS occupies no file bytes.
        if kind != 8 && !within(u64_at(at + 24)?, u64_at(at + 32)?) {
            return Err(Error::Elf("section out of bounds"));
        }
        // A kernel is a non-empty SHT_PROGBITS section with SHF_EXECINSTR.
        kernel |= kind == 1 && u64_at(at + 8)? & 4 != 0 && u64_at(at + 32)? > 0;
    }
    // The Windows driver faults inside cuModuleLoadData on an ELF with no
    // code instead of returning CUDA_ERROR_INVALID_IMAGE, so refuse it here.
    if !kernel {
        return Err(Error::Elf("no executable kernel section"));
    }
    for i in 0..segments {
        let at = phoff + i * 56;
        if !within(u64_at(at + 8)?, u64_at(at + 32)?) {
            return Err(Error::Elf("segment out of bounds"));
        }
    }
    Ok(sm)
}

/// Build time only: compile `plan`'s generated device source with the
/// bounded NVRTC helper for `sm` and write a container.
#[cfg(feature = "cuda")]
pub fn build(plan: &Plan, compiler: &std::path::Path, sm: u32) -> Result<Vec<u8>, String> {
    let artifact = plan.artifact();
    if artifact.target() != Target::Cuda || artifact.abi_version() != 5 {
        return Err("CUDA ABI5 device artifact required".into());
    }
    let nvrtc = invar_cuda::cuda::nvrtc_version().map_err(|e| format!("NVRTC:{e:?}"))?;
    let caps = invar_device::Capabilities::Cuda {
        major: nvrtc.0,
        minor: nvrtc.1,
        sm,
    };
    let cubin = invar_cuda::cuda::compile_gnet_cubin_bounded(artifact.source(), compiler, caps)
        .map_err(|e| format!("NVRTC compile:{e:?}"))?
        .image;
    if cubin_sm(&cubin).map_err(|e| format!("NVRTC output:{e:?}"))? != sm {
        return Err("NVRTC output targets another architecture".into());
    }
    encode(&Header::of(plan, sm, nvrtc, &cubin), &cubin).map_err(|e| format!("{e:?}"))
}

fn container_digest(prefix: &[u8], header: &[u8], image: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(&prefix[..24]);
    hash.update(header);
    hash.update(image);
    hash.finalize().into()
}

fn le<const N: usize>(bytes: &[u8], at: usize) -> Result<[u8; N], Error> {
    at.checked_add(N)
        .and_then(|end| bytes.get(at..end))
        .and_then(|b| b.try_into().ok())
        .ok_or(Error::Elf("short"))
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

fn small(value: &Value, field: &'static str) -> Result<u32, Error> {
    value
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or(Error::Header(field))
}

fn version(value: &Value) -> Result<(u32, u32), Error> {
    let (major, minor) = value
        .as_str()
        .and_then(|s| s.split_once('.'))
        .ok_or(Error::Header("nvrtc"))?;
    let part = |s: &str| {
        (!s.is_empty() && s.len() <= 3 && s.bytes().all(|b| b.is_ascii_digit()))
            .then(|| s.parse().ok())
            .flatten()
            .ok_or(Error::Header("nvrtc"))
    };
    Ok((part(major)?, part(minor)?))
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
