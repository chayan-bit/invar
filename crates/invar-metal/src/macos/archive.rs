//! Bounded persistent backend-code cache. Cache bytes are validated through
//! AOT's single-descriptor reader, then copied to a fresh private staging URL.
//! Metal never reopens the cache path that was validated.
use super::*;
use crate::cache_file;
use objc2_foundation::{NSArray, NSProcessInfo, NSURL};
use objc2_metal::{
    MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLComputePipelineDescriptor, MTLFunction,
    MTLPipelineOption,
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const MAGIC: &[u8; 8] = b"INVGMBA1";
const HEADER: usize = 8 + 32 + 32;
const MAX_BYTES: usize = 64 * 1024 * 1024;
const OPTIONS: &str = "invar-metal-archive-v1;msl=3.1;fast-math=false;pipeline=default";
static NEXT: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Permissions,
    Size,
    Header,
    Identity,
    Digest,
    Metal,
    PipelineMiss,
    Io,
    CacheFull,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum State {
    #[default]
    Disabled,
    MemoryHit,
    Miss,
    Hit,
    Rebuilt(Refusal),
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub last: State,
    pub key: [u8; 32],
    pub hits: u64,
    pub misses: u64,
    pub rebuilds: u64,
    pub writes: u64,
    /// A refused cache write does not discard a successfully built pipeline.
    pub write_refusal: Option<Refusal>,
    /// Serialized backend archive bytes, not a claim about resident code size.
    pub bytes: usize,
    /// Time in the pipeline creation API, excluding cache file admission/save.
    pub pipeline_creation_ns: u64,
}

pub(super) struct Cache {
    root: PathBuf,
    pub report: Report,
}

fn file_error(e: cache_file::Error) -> Refusal {
    match e {
        cache_file::Error::Permissions => Refusal::Permissions,
        cache_file::Error::Size => Refusal::Size,
        cache_file::Error::Io(_) => Refusal::Io,
    }
}

fn identity(fields: &[&[u8]]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for field in fields {
        hash.update((field.len() as u64).to_le_bytes());
        hash.update(field);
    }
    hash.finalize().into()
}

fn decode<'a>(bytes: &'a [u8], key: &[u8; 32]) -> Result<&'a [u8], Refusal> {
    if bytes.len() <= HEADER || bytes.len() > MAX_BYTES {
        return Err(Refusal::Size);
    }
    if &bytes[..8] != MAGIC {
        return Err(Refusal::Header);
    }
    if &bytes[8..40] != key {
        return Err(Refusal::Identity);
    }
    let payload = &bytes[HEADER..];
    if bytes[40..HEADER] != Sha256::digest(payload)[..] {
        return Err(Refusal::Digest);
    }
    // The payload is Apple's opaque, OS-specific format. Its bounded bytes
    // are admitted by newBinaryArchiveWithDescriptor:error:, not guessed here.
    Ok(payload)
}

fn encode(key: &[u8; 32], payload: &[u8]) -> Result<Vec<u8>, Refusal> {
    if payload.len() > MAX_BYTES - HEADER {
        return Err(Refusal::Size);
    }
    let mut bytes = Vec::with_capacity(HEADER + payload.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(key);
    bytes.extend_from_slice(&Sha256::digest(payload));
    bytes.extend_from_slice(payload);
    decode(&bytes, key)?;
    Ok(bytes)
}

/// Reject symlinks and shared directories. Temporary transactions live inside
/// this private cache directory and are never named by an external caller.
fn private_root(root: &Path) -> Result<(), Refusal> {
    let mut builder = fs::DirBuilder::new();
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder
        .recursive(true)
        .create(root)
        .map_err(|_| Refusal::Io)?;
    let meta = fs::symlink_metadata(root).map_err(|_| Refusal::Io)?;
    if !meta.is_dir() || std::os::unix::fs::PermissionsExt::mode(&meta.permissions()) & 0o077 != 0 {
        return Err(Refusal::Permissions);
    }
    Ok(())
}
struct Scratch(PathBuf);
impl Scratch {
    fn new(root: &Path) -> Result<Self, Refusal> {
        private_root(root)?;
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = root.join(format!(".build-{}-{id}", std::process::id()));
        let mut builder = fs::DirBuilder::new();
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder.create(&dir).map_err(|_| Refusal::Io)?;
        Ok(Self(dir))
    }
    fn url(&self) -> Result<Retained<NSURL>, Refusal> {
        let path = self.0.join("pipeline.metallib");
        let path = path.to_str().ok_or(Refusal::Io)?;
        Ok(NSURL::fileURLWithPath(&NSString::from_str(path)))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

impl Cache {
    pub(super) fn new(root: PathBuf) -> Self {
        Self {
            root,
            report: Report::default(),
        }
    }
    fn path(&self, key: &[u8; 32]) -> PathBuf {
        self.root
            .join(format!("{}.invgmba", super::library::hex(key)))
    }
    pub(super) fn pipeline(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        function: &ProtocolObject<dyn MTLFunction>,
        source: &[u8; 32],
        compiler: &[u8; 32],
        name: &str,
        bound: &super::library::Bound,
    ) -> Result<PipelineRef, Error> {
        let os = NSProcessInfo::processInfo()
            .operatingSystemVersionString()
            .to_string();
        let registry = device.registryID().to_le_bytes();
        let device_name = device.name().to_string();
        let key = identity(&[
            source,
            compiler,
            name.as_bytes(),
            &registry,
            device_name.as_bytes(),
            os.as_bytes(),
            OPTIONS.as_bytes(),
        ]);
        self.report.key = key;
        self.report.write_refusal = None;
        self.report.bytes = 0;
        self.report.pipeline_creation_ns = 0;
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(function));
        let cached = (|| {
            private_root(&self.root)?;
            let bytes = match cache_file::read_capped(&self.path(&key), MAX_BYTES) {
                Err(cache_file::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(None);
                }
                other => other.map_err(file_error)?,
            };
            let payload = decode(&bytes, &key)?;
            let scratch = Scratch::new(&self.root)?;
            cache_file::write_atomic(&scratch.0.join("pipeline.metallib"), payload)
                .map_err(file_error)?;
            let archive_descriptor = MTLBinaryArchiveDescriptor::new();
            let url = scratch.url()?;
            archive_descriptor.setUrl(Some(&url));
            let archive = device
                .newBinaryArchiveWithDescriptor_error(&archive_descriptor)
                .map_err(|_| Refusal::Metal)?;
            descriptor.setBinaryArchives(Some(&NSArray::from_retained_slice(&[archive])));
            let begun = Instant::now();
            let pipeline = device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &descriptor,
                    MTLPipelineOption::FailOnBinaryArchiveMiss,
                    None,
                )
                .map_err(|_| Refusal::PipelineMiss)?;
            self.report.pipeline_creation_ns =
                u64::try_from(begun.elapsed().as_nanos()).unwrap_or(u64::MAX);
            Ok(Some((pipeline, payload.len())))
        })();
        match cached {
            Ok(Some((pipeline, bytes))) => {
                self.report.hits += 1;
                self.report.last = State::Hit;
                self.report.bytes = bytes;
                return Ok(pipeline);
            }
            Ok(None) => {
                self.report.misses += 1;
                self.report.last = State::Miss;
            }
            Err(reason) => {
                self.report.misses += 1;
                self.report.rebuilds += 1;
                self.report.last = State::Rebuilt(reason);
            }
        }
        descriptor.setBinaryArchives(None);
        // Rebuild only this cache entry, through the already-admitted library.
        // Corruption never causes runtime MSL compilation on a production device.
        let begun = Instant::now();
        let pipeline = bound.descriptor(device, &descriptor, MTLPipelineOption::empty())?;
        self.report.pipeline_creation_ns =
            u64::try_from(begun.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let save = (|| {
            let scratch = Scratch::new(&self.root)?;
            let archive = device
                .newBinaryArchiveWithDescriptor_error(&MTLBinaryArchiveDescriptor::new())
                .map_err(|_| Refusal::Metal)?;
            archive
                .addComputePipelineFunctionsWithDescriptor_error(&descriptor)
                .map_err(|_| Refusal::Metal)?;
            let url = scratch.url()?;
            archive
                .serializeToURL_error(&url)
                .map_err(|_| Refusal::Metal)?;
            let bytes =
                cache_file::read_capped(&scratch.0.join("pipeline.metallib"), MAX_BYTES - HEADER)
                    .map_err(file_error)?;
            let container = encode(&key, &bytes)?;
            // ponytail: no eviction. Refuse new identities at 128 files; racing
            // writers can overshoot by one bounded entry each. Existing keys rebuild.
            if fs::symlink_metadata(self.path(&key)).is_err() {
                let mut files = 0;
                for entry in fs::read_dir(&self.root).map_err(|_| Refusal::Io)? {
                    let entry = entry.map_err(|_| Refusal::Io)?;
                    if entry.path().extension().is_some_and(|e| e == "invgmba") {
                        files += 1;
                        if files == 128 {
                            return Err(Refusal::CacheFull);
                        }
                    }
                }
            }
            cache_file::write_atomic(&self.path(&key), &container).map_err(file_error)?;
            self.report.bytes = bytes.len();
            self.report.writes += 1;
            Ok(())
        })();
        if let Err(reason) = save {
            self.report.write_refusal = Some(reason);
        }
        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_identity_uses_every_field_and_length_boundaries() {
        let fields: Vec<Vec<u8>> = (0..7).map(|i| vec![i; 3]).collect();
        let key = identity(&fields.iter().map(Vec::as_slice).collect::<Vec<_>>());
        for i in 0..fields.len() {
            let mut changed = fields.clone();
            changed[i].push(1);
            assert_ne!(
                key,
                identity(&changed.iter().map(Vec::as_slice).collect::<Vec<_>>())
            );
        }
        assert_ne!(identity(&[b"ab", b"c"]), identity(&[b"a", b"bc"]));
    }
    #[test]
    fn corruption_identity_and_truncation_are_typed_refusals() {
        let key = [3; 32];
        let mut payload = vec![0; 32];
        payload[..4].copy_from_slice(&[0xcf, 0xfa, 0xed, 0xfe]);
        let bytes = encode(&key, &payload).unwrap();
        assert_eq!(decode(&bytes, &key), Ok(payload.as_slice()));
        for len in 0..bytes.len() {
            assert!(decode(&bytes[..len], &key).is_err());
        }
        assert_eq!(decode(&bytes, &[4; 32]), Err(Refusal::Identity));
        let mut bad = bytes.clone();
        bad[HEADER + 5] ^= 1;
        assert_eq!(decode(&bad, &key), Err(Refusal::Digest));
        bad = bytes.clone();
        bad[0] ^= 1;
        assert_eq!(decode(&bad, &key), Err(Refusal::Header));
        assert_eq!(decode(&[0; 80], &key), Err(Refusal::Header));
    }
}
