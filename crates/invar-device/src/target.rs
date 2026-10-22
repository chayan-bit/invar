//! ENG-06 target descriptions (#283): the hardware facts a schedule may use,
//! as versioned JSON data under `targets/` instead of constants in emitters.
//!
//! A file names one device class (`cuda-sm90`, `metal-apple9`,
//! `cpu-x86_64-zen4`) and states its group, memory, copy and matrix atom facts
//! with the public document they come from. Parsing is strict: an unknown or
//! duplicate field, a wrong type, a value out of its range or a name that does
//! not match the device is refused with the field that failed. The files are
//! embedded at build time (`build.rs` lists the directory), so a new target is
//! a new file and no Rust change. These facts confer no execution authority:
//! schedules are still verified and the runtimes still check live limits.

use std::collections::BTreeSet;
use std::io::Read;
use std::sync::OnceLock;

use serde::Deserialize;

use crate::Capabilities;

/// Schema identity every target file must carry.
pub const SCHEMA: &str = "invar-target-v1";
/// Largest target file read or parsed; every list and string in it is
/// bounded by this. The largest built-in file is under 4 KiB.
pub const MAX_FILE_BYTES: usize = 64 * 1024;
/// Most files one registry holds.
pub const MAX_FILES: usize = 256;

mod builtin {
    include!(concat!(env!("OUT_DIR"), "/targets.rs"));
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Not JSON, a wrong type, or an unknown, duplicate or missing field.
    Parse {
        file: String,
        detail: String,
    },
    /// A value outside its admitted range or form.
    Range {
        file: String,
        field: String,
    },
    /// The file name, the `name` field and the device facts disagree.
    Name {
        file: String,
    },
    /// Two files describe the same target.
    Duplicate {
        name: String,
    },
    /// No target file describes the live device; device execution through
    /// target-driven schedules is refused.
    Unknown {
        device: String,
    },
    Io {
        path: String,
        detail: String,
    },
    /// A file over `MAX_FILE_BYTES` or a set over `MAX_FILES`, refused
    /// before it is read in full or parsed.
    Limit {
        what: String,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse { file, detail } => write!(f, "target {file}: {detail}"),
            Self::Range { file, field } => write!(f, "target {file}: {field} is out of range"),
            Self::Name { file } => write!(f, "target {file}: name does not match its device"),
            Self::Duplicate { name } => write!(f, "target {name} is described twice"),
            Self::Unknown { device } => write!(f, "no target description for {device}"),
            Self::Io { path, detail } => write!(f, "target directory {path}: {detail}"),
            Self::Limit { what } => write!(f, "target limit exceeded: {what}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Cuda,
    Metal,
    Cpu,
}

/// Which live device a file describes. CUDA files set `sm`, Metal files
/// `apple_family`, CPU files `arch` and `model`; nothing else.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Device {
    pub backend: Backend,
    #[serde(default)]
    pub sm: Option<u32>,
    #[serde(default)]
    pub apple_family: Option<u32>,
    #[serde(default)]
    pub arch: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

/// Memory levels a schedule may stage through. `shared` is CUDA shared memory
/// and Metal threadgroup memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Global,
    Shared,
    Registers,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryLevel {
    pub level: Level,
    pub alignment_bytes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CopyKind {
    /// Ordinary loads and stores, scalar or vector.
    Plain,
    /// `cp.async`: global to shared without registers, waited on by group.
    CpAsync,
    /// Hopper bulk tensor copy (TMA) of a box of up to `max_box_extent` per axis.
    Tma,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CopyAtom {
    pub name: String,
    pub kind: CopyKind,
    pub from: Level,
    pub to: Level,
    /// Bytes one thread moves per instruction (TMA: the transfer granule).
    pub bytes: u32,
    pub alignment_bytes: u32,
    #[serde(default)]
    pub max_box_extent: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Element {
    F32,
    Tf32,
    F16,
    Bf16,
    E4m3,
    E5m2,
}

/// One matrix instruction: `threads` lanes together compute an `m x n`
/// accumulator tile over `k`. `contract` names the numeric contract the atom
/// implements; `reference` names the function that defines its result, or is
/// null while no contract implementation exists (such an atom is data only).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MmaAtom {
    pub name: String,
    pub instruction: String,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub threads: u32,
    pub a_type: Element,
    pub b_type: Element,
    pub acc_type: Element,
    pub contract: String,
    pub reference: Option<String>,
}

/// Register reallocation between producer and consumer warpgroups
/// (`setmaxnreg`): per-thread counts from `min` to `max` in steps of `step`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarpSpecialization {
    pub min_registers: u32,
    pub max_registers: u32,
    pub register_step: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gpu {
    pub subgroup_width: u32,
    pub max_threads_per_group: u32,
    /// Statically declared shared memory a group may use.
    pub shared_bytes_per_group: u32,
    /// Dynamic shared memory a group may use after an explicit opt-in.
    pub shared_bytes_per_group_optin: u32,
    /// Null where the vendor does not publish it.
    pub registers_per_thread: Option<u32>,
    pub memory_levels: Vec<MemoryLevel>,
    pub copy_atoms: Vec<CopyAtom>,
    pub mma_atoms: Vec<MmaAtom>,
    #[serde(default)]
    pub warp_specialization: Option<WarpSpecialization>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cache {
    pub level: u32,
    pub bytes: u64,
    pub line_bytes: u32,
    pub shared_by_cores: u32,
    /// `performance` or `efficiency` on hybrid parts; null when uniform.
    pub core_class: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cpu {
    pub simd_width_bits: u32,
    pub isa: Vec<String>,
    pub caches: Vec<Cache>,
    /// Null when the node count depends on the host and is read at run time.
    pub numa_nodes: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub schema: String,
    pub name: String,
    /// The public documents the facts come from.
    pub source: String,
    pub device: Device,
    #[serde(default)]
    pub gpu: Option<Gpu>,
    #[serde(default)]
    pub cpu: Option<Cpu>,
}

fn pow2(value: u64, lo: u64, hi: u64) -> bool {
    value.is_power_of_two() && (lo..=hi).contains(&value)
}

/// Lowercase words joined by single hyphens, optionally ending in `-vN`.
fn ident(text: &str) -> bool {
    !text.is_empty()
        && text.split('-').all(|word| {
            !word.is_empty()
                && word
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        })
}

impl Target {
    /// Strictly parse one file. `file` is the name without `.json`; it must
    /// equal the target's name.
    pub fn parse(file: &str, text: &str) -> Result<Self, Error> {
        if text.len() > MAX_FILE_BYTES {
            return Err(Error::Limit {
                what: format!("{file} is over {MAX_FILE_BYTES} bytes"),
            });
        }
        let target: Self = serde_json::from_str(text).map_err(|e| Error::Parse {
            file: file.to_owned(),
            detail: e.to_string(),
        })?;
        target.check(file)?;
        Ok(target)
    }

    fn check(&self, file: &str) -> Result<(), Error> {
        let range = |ok: bool, field: &str| {
            if ok {
                Ok(())
            } else {
                Err(Error::Range {
                    file: file.to_owned(),
                    field: field.to_owned(),
                })
            }
        };
        range(self.schema == SCHEMA, "schema")?;
        range(!self.source.trim().is_empty(), "source")?;
        let d = &self.device;
        let expected = match (d.backend, d.sm, d.apple_family, &d.arch, &d.model) {
            (Backend::Cuda, Some(sm), None, None, None) => {
                range((50..=199).contains(&sm), "device.sm")?;
                format!("cuda-sm{sm}")
            }
            (Backend::Metal, None, Some(family), None, None) => {
                range((1..=99).contains(&family), "device.apple_family")?;
                format!("metal-apple{family}")
            }
            (Backend::Cpu, None, None, Some(arch), Some(model)) => {
                range(ident(arch) && !arch.contains('-'), "device.arch")?;
                range(ident(model), "device.model")?;
                format!("cpu-{arch}-{model}")
            }
            _ => return range(false, "device"),
        };
        if self.name != expected || file != expected {
            return Err(Error::Name {
                file: file.to_owned(),
            });
        }
        match (d.backend, &self.gpu, &self.cpu) {
            (Backend::Cuda | Backend::Metal, Some(gpu), None) => gpu.check(&range),
            (Backend::Cpu, None, Some(cpu)) => cpu.check(&range),
            _ => range(false, "gpu/cpu section"),
        }
    }

    /// The GPU facts; `None` for a CPU target.
    pub fn gpu(&self) -> Option<&Gpu> {
        self.gpu.as_ref()
    }

    pub fn cpu(&self) -> Option<&Cpu> {
        self.cpu.as_ref()
    }
}

type Check<'a> = dyn Fn(bool, &str) -> Result<(), Error> + 'a;

impl Gpu {
    fn check(&self, range: &Check<'_>) -> Result<(), Error> {
        range(
            pow2(self.subgroup_width.into(), 4, 128),
            "gpu.subgroup_width",
        )?;
        range(
            (1..=4096).contains(&self.max_threads_per_group)
                && self
                    .max_threads_per_group
                    .is_multiple_of(self.subgroup_width),
            "gpu.max_threads_per_group",
        )?;
        range(
            self.shared_bytes_per_group_optin <= 1 << 20,
            "gpu.shared_bytes_per_group_optin",
        )?;
        range(
            (1..=self.shared_bytes_per_group_optin).contains(&self.shared_bytes_per_group),
            "gpu.shared_bytes_per_group",
        )?;
        range(
            self.registers_per_thread
                .is_none_or(|r| (16..=1024).contains(&r)),
            "gpu.registers_per_thread",
        )?;
        let levels: BTreeSet<Level> = self.memory_levels.iter().map(|m| m.level).collect();
        range(
            levels.len() == 3 && self.memory_levels.len() == 3,
            "gpu.memory_levels",
        )?;
        for m in &self.memory_levels {
            range(
                pow2(m.alignment_bytes.into(), 1, 256),
                "gpu.memory_levels.alignment_bytes",
            )?;
        }
        let mut names = BTreeSet::new();
        for c in &self.copy_atoms {
            range(
                ident(&c.name) && names.insert(&c.name),
                "gpu.copy_atoms.name",
            )?;
            let shape = match c.kind {
                CopyKind::Plain => c.from != c.to && c.max_box_extent.is_none(),
                CopyKind::CpAsync => {
                    (c.from, c.to) == (Level::Global, Level::Shared) && c.max_box_extent.is_none()
                }
                CopyKind::Tma => {
                    (c.from, c.to) == (Level::Global, Level::Shared)
                        && c.max_box_extent.is_some_and(|e| (1..=256).contains(&e))
                }
            };
            range(shape, "gpu.copy_atoms.kind")?;
            range(pow2(c.bytes.into(), 1, 128), "gpu.copy_atoms.bytes")?;
            range(
                pow2(c.alignment_bytes.into(), 1, 128),
                "gpu.copy_atoms.alignment_bytes",
            )?;
        }
        for a in &self.mma_atoms {
            range(
                ident(&a.name) && names.insert(&a.name),
                "gpu.mma_atoms.name",
            )?;
            range(
                !a.instruction.trim().is_empty(),
                "gpu.mma_atoms.instruction",
            )?;
            for (value, field) in [(a.m, "m"), (a.n, "n"), (a.k, "k")] {
                range(
                    (1..=256).contains(&value),
                    &format!("gpu.mma_atoms.{field}"),
                )?;
            }
            range(
                a.threads.is_multiple_of(self.subgroup_width)
                    && (1..=self.max_threads_per_group).contains(&a.threads),
                "gpu.mma_atoms.threads",
            )?;
            range(
                ident(&a.contract) && a.contract.rsplit('-').next().is_some_and(versioned),
                "gpu.mma_atoms.contract",
            )?;
            range(
                a.reference.as_deref().is_none_or(|r| {
                    !r.is_empty()
                        && r.split("::").all(|p| {
                            !p.is_empty() && p.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
                        })
                }),
                "gpu.mma_atoms.reference",
            )?;
        }
        if let Some(w) = &self.warp_specialization {
            range(
                pow2(w.register_step.into(), 1, 64)
                    && (w.register_step..=w.max_registers).contains(&w.min_registers)
                    && w.max_registers <= 1024
                    && w.min_registers.is_multiple_of(w.register_step)
                    && w.max_registers.is_multiple_of(w.register_step),
                "gpu.warp_specialization",
            )?;
        }
        Ok(())
    }
}

/// `v` followed by a decimal version.
fn versioned(word: &str) -> bool {
    word.strip_prefix('v')
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

impl Cpu {
    fn check(&self, range: &Check<'_>) -> Result<(), Error> {
        range(
            pow2(self.simd_width_bits.into(), 64, 2048),
            "cpu.simd_width_bits",
        )?;
        let mut isa = BTreeSet::new();
        range(
            !self.isa.is_empty() && self.isa.iter().all(|i| ident(i) && isa.insert(i)),
            "cpu.isa",
        )?;
        range(!self.caches.is_empty(), "cpu.caches")?;
        for c in &self.caches {
            range((1..=4).contains(&c.level), "cpu.caches.level")?;
            range(pow2(c.line_bytes.into(), 16, 256), "cpu.caches.line_bytes")?;
            range(c.bytes >= u64::from(c.line_bytes), "cpu.caches.bytes")?;
            range(
                (1..=1024).contains(&c.shared_by_cores),
                "cpu.caches.shared_by_cores",
            )?;
            range(
                c.core_class
                    .as_deref()
                    .is_none_or(|k| matches!(k, "performance" | "efficiency")),
                "cpu.caches.core_class",
            )?;
        }
        range(
            self.numa_nodes.is_none_or(|n| (1..=64).contains(&n)),
            "cpu.numa_nodes",
        )
    }
}

/// Every known target, sorted by name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Registry {
    targets: Vec<Target>,
}

impl Registry {
    /// Parse `(file name without .json, text)` pairs. Every file must parse;
    /// one bad file refuses the whole set.
    pub fn parse<'a>(files: impl IntoIterator<Item = (&'a str, &'a str)>) -> Result<Self, Error> {
        let mut targets = Vec::new();
        for (file, text) in files {
            if targets.len() == MAX_FILES {
                return Err(Error::Limit {
                    what: format!("more than {MAX_FILES} files"),
                });
            }
            targets.push(Target::parse(file, text)?);
        }
        targets.sort_by(|a, b| a.name.cmp(&b.name));
        if let Some(pair) = targets.windows(2).find(|p| p[0].name == p[1].name) {
            return Err(Error::Duplicate {
                name: pair[0].name.clone(),
            });
        }
        Ok(Self { targets })
    }

    /// The files under `targets/` at build time.
    pub fn builtin() -> Result<&'static Self, Error> {
        static BUILTIN: OnceLock<Result<Registry, Error>> = OnceLock::new();
        BUILTIN
            .get_or_init(|| Self::parse(builtin::FILES.iter().copied()))
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Every `*.json` file of a directory, for tools and tests that add a
    /// target without rebuilding.
    pub fn from_dir(dir: &std::path::Path) -> Result<Self, Error> {
        let io = |e: std::io::Error| Error::Io {
            path: dir.display().to_string(),
            detail: e.to_string(),
        };
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(dir).map_err(io)? {
            let path = entry.map_err(io)?.path();
            if path.extension().is_some_and(|e| e == "json") {
                if paths.len() == MAX_FILES {
                    return Err(Error::Limit {
                        what: format!("more than {MAX_FILES} files in {}", dir.display()),
                    });
                }
                paths.push(path);
            }
        }
        let mut files = Vec::new();
        for path in paths {
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            // Read at most one byte past the limit, so an oversized file is
            // refused without being loaded.
            let mut text = String::new();
            std::fs::File::open(&path)
                .and_then(|f| f.take(MAX_FILE_BYTES as u64 + 1).read_to_string(&mut text))
                .map_err(io)?;
            if text.len() > MAX_FILE_BYTES {
                return Err(Error::Limit {
                    what: format!("{} is over {MAX_FILE_BYTES} bytes", path.display()),
                });
            }
            files.push((stem.to_owned(), text));
        }
        Self::parse(files.iter().map(|(f, t)| (f.as_str(), t.as_str())))
    }

    pub fn targets(&self) -> &[Target] {
        &self.targets
    }

    pub fn get(&self, name: &str) -> Option<&Target> {
        self.targets.iter().find(|t| t.name == name)
    }

    /// The target describing a live GPU, or `Error::Unknown`.
    pub fn select(&self, caps: Capabilities) -> Result<&Target, Error> {
        let name = match caps {
            Capabilities::Cuda { sm, .. } => format!("cuda-sm{sm}"),
            Capabilities::Metal { apple_family, .. } => format!("metal-apple{apple_family}"),
        };
        self.get(&name).ok_or(Error::Unknown { device: name })
    }

    /// The target describing a CPU model, or `Error::Unknown`.
    pub fn select_cpu(&self, arch: &str, model: &str) -> Result<&Target, Error> {
        let name = format!("cpu-{arch}-{model}");
        self.get(&name).ok_or(Error::Unknown { device: name })
    }
}
