//! Versioned measured placement hints. Decisions describe costs, never grant
//! execution/publication authority; verified adapters retain those checks.
use crate::schedule::tuning::{Location, Residency};
pub use invar_cuda::pages::PageState;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs::File, io::Read, path::Path};

pub const MAX_PROFILE_BYTES: usize = 4 << 20;
fn fingerprint(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
pub mod calibration;

pub const VERSION: u64 = 1;
pub const PROBE: &str = "placement-v1";
pub const MAP_FAMILY: &str = "placement.wrapping_add1.u32";
pub const MAP_NUMERIC_CONTRACT: &str = "wrapping-u32-v1";

/// M5-01 measurement protocol. A profile that declares `statistics` must
/// satisfy all of it; one without it can never claim a calibrated family.
pub const WARMUPS: usize = 10;
pub const SAMPLES: usize = 30;
pub const FIRST_CALL: &str = "first_call";
/// The frozen-table families (`docs/performance-acceptance.md`).
pub const FAMILIES: [&str; 10] = [
    "map",
    "reduction",
    "scan",
    "compaction",
    "transpose",
    "stencil",
    "matrix",
    "segmented",
    "gather_scatter",
    "frontier",
];
pub const CENTER: &str = "median of 30 (mean of order statistics 15 and 16, 1-based)";
pub const INTERVAL: &str = "two-sided exact binomial order-statistic interval for the median: order statistics 10 and 21 (1-based) of 30, coverage 95.72% for independent continuous samples; ties conservative";
pub const SPREAD: &str = "q1 and q3 are order statistics 8 and 23 (1-based) of 30";
pub const POLICY: &str =
    "measurements only; GNET-04 chooses the cost model, margins and thresholds";
const ENGINES: [Engine; 3] = [Engine::Cpu, Engine::Cuda, Engine::Metal];

pub fn statistics() -> Value {
    json!({"warmups":WARMUPS,"samples":SAMPLES,"center":CENTER,"interval":INTERVAL,
        "spread":SPREAD,"policy":POLICY})
}
/// Order statistics over integer nanoseconds, so a stored summary is exactly
/// what its samples produce. One sample is a cold, compile-inclusive first call.
pub fn summarize(samples: &[u64]) -> Result<Value, Error> {
    if !(samples.len() == 1 || samples.len() == SAMPLES) || samples.contains(&0) {
        return Err(Error::Invalid);
    }
    let mut s = samples.to_vec();
    s.sort_unstable();
    let at = |k: usize| s[k.min(s.len() - 1)];
    let median = if s.len() == 1 {
        s[0] as f64
    } else {
        (s[14] as f64 + s[15] as f64) / 2.0
    };
    Ok(
        json!({"n":s.len(),"median_ns":median,"ci95_ns":[at(9),at(20)],"q1_ns":at(7),"q3_ns":at(22),
        "min_ns":s[0],"max_ns":s[s.len()-1]}),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Invalid,
    Identity,
    Io,
    FreezeRequired,
    Digest,
}
impl Error {
    pub fn code(self) -> &'static str {
        match self {
            Self::Invalid | Self::Identity | Self::Digest => "TUNING_HINT_INVALID",
            Self::Io => "CALIBRATION_IO",
            Self::FreezeRequired => "TUNING_FREEZE_REQUIRED",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Engine {
    Cpu,
    Cuda,
    Metal,
}
impl Engine {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
    fn parse(v: &Value) -> Result<Self, Error> {
        match v.as_str() {
            Some("cpu") => Ok(Self::Cpu),
            Some("cuda") => Ok(Self::Cuda),
            Some("metal") => Ok(Self::Metal),
            _ => Err(Error::Invalid),
        }
    }
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str, Error> {
    v[key]
        .as_str()
        .filter(|s| !s.trim().is_empty() && s.chars().take(257).count() <= 256)
        .ok_or(Error::Invalid)
}
fn figure(v: &Value, key: &str) -> Result<f64, Error> {
    v[key]
        .as_f64()
        .filter(|n| n.is_finite() && *n >= 0.0)
        .ok_or(Error::Invalid)
}
fn optional(v: &Value, key: &str) -> Result<Option<f64>, Error> {
    if v.get(key) == Some(&Value::Null) {
        Ok(None)
    } else {
        figure(v, key).map(Some)
    }
}
fn identity(v: &Value) -> Result<(), Error> {
    for key in ["host_class", "cpu", "device", "driver", "toolchain"] {
        text(v, key)?;
    }
    Ok(())
}
fn family_valid(family: &str) -> bool {
    let parts = family.split('.').collect::<Vec<_>>();
    parts.len() == 3
        && parts.iter().all(|p| {
            p.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
                && p.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        })
}
fn region_valid(v: &Value) -> bool {
    v.is_null()
        || (v["source_identity"].as_u64().is_some_and(|n| n != 0)
            && v["graph_revision"].as_u64().is_some()
            && v["region_identity"].as_u64().is_some())
}
/// The profile family ID each frozen-table family is probed with.
const PROBES: [(&str, &str); 10] = [
    (MAP_FAMILY, "map"),
    ("placement.wrapping_sum.u32", "reduction"),
    ("placement.affine_scan.u32", "scan"),
    ("placement.select_below_half.u32", "compaction"),
    ("placement.transpose.u32", "transpose"),
    ("placement.five_point_stencil.u32", "stencil"),
    ("placement.square_matmul.u32", "matrix"),
    ("placement.segmented_sum16.u32", "segmented"),
    ("placement.permuted_gather.u32", "gather_scatter"),
    ("placement.dense_bfs16.u32", "frontier"),
];
/// Device probes of the families beyond `map` time a whole forced-device
/// source program host to host; they price no link or launch separately.
pub const SOURCE_DEVICE_BOUNDARY: &str = "source-forced-device-v1";
/// The boundary each engine's probe of a table family times.
fn probe_boundary(engine: Engine, table: &str) -> &'static str {
    match (engine, table) {
        (Engine::Cpu, _) => "llvm-host-array-v1",
        (_, "map") => "host-map-v1",
        _ => SOURCE_DEVICE_BOUNDARY,
    }
}
/// Per-size samples of one row; returns the table family its probe measures.
/// The claim is derived from the row's own probe identity, never trusted,
/// and the row needs one cold `first_call` plus 30 samples for every size.
fn points(row: &Value, engine: Engine, sizes: &[u64]) -> Result<&'static str, Error> {
    let family = PROBES
        .iter()
        .find(|(id, _)| row["family"] == *id)
        .map(|(_, table)| *table)
        .filter(|table| {
            row["table_family"] == *table
                && row["numeric"] == "u32-wrapping"
                && row["numeric_contract_version"] == MAP_NUMERIC_CONTRACT
                && row["boundary"] == probe_boundary(engine, table)
                && row["region"].is_null()
        })
        .ok_or(Error::Invalid)?;
    let boundary = probe_boundary(engine, family);
    let points = row["points"]
        .as_array()
        .filter(|p| !p.is_empty() && p.len() <= 17)
        .ok_or(Error::Invalid)?;
    let mut keys = BTreeSet::new();
    for p in points {
        let elements = match (text(p, "boundary")?, &p["elements"]) {
            (FIRST_CALL, Value::Null) => None,
            (b, n) if b == boundary => Some(
                n.as_u64()
                    .filter(|n| sizes.contains(n))
                    .ok_or(Error::Invalid)?,
            ),
            _ => return Err(Error::Invalid),
        };
        let samples = p["samples_ns"]
            .as_array()
            .ok_or(Error::Invalid)?
            .iter()
            .map(|n| n.as_u64().ok_or(Error::Invalid))
            .collect::<Result<Vec<_>, _>>()?;
        if elements.is_none() != (samples.len() == 1)
            || summarize(&samples)? != p["summary"]
            || !keys.insert(elements)
        {
            return Err(Error::Invalid);
        }
    }
    if !keys.contains(&None) || sizes.iter().any(|n| !keys.contains(&Some(*n))) {
        return Err(Error::Invalid);
    }
    Ok(family)
}
/// Each (engine, table family) is measured or pending, exactly once.
fn coverage(v: &Value, measured: &BTreeSet<(Engine, &str)>) -> Result<(), Error> {
    if v["statistics"] != statistics() || v["coverage"]["families"] != json!(FAMILIES) {
        return Err(Error::Invalid);
    }
    let pending = v["coverage"]["pending"]
        .as_array()
        .filter(|p| p.len() <= ENGINES.len() * FAMILIES.len())
        .ok_or(Error::Invalid)?;
    let mut keys = BTreeSet::new();
    for p in pending {
        text(p, "reason")?;
        let family = text(p, "family")?;
        if !FAMILIES.contains(&family) || !keys.insert((Engine::parse(&p["engine"])?, family)) {
            return Err(Error::Invalid);
        }
    }
    for engine in ENGINES {
        for family in FAMILIES {
            if measured.contains(&(engine, family)) == keys.contains(&(engine, family)) {
                return Err(Error::Invalid);
            }
        }
    }
    Ok(())
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct Coefficients {
    fixed: f64,
    work: f64,
    bandwidth: Option<f64>,
    latency: f64,
    launch: f64,
    page_in: Option<f64>,
}
impl Coefficients {
    /// Rows a device prices across its link must carry the link bandwidth;
    /// whole-program source-device rows price no link.
    fn parse(v: &Value, engine: Engine, boundary: &str) -> Result<Self, Error> {
        let linked = engine != Engine::Cpu && boundary != SOURCE_DEVICE_BOUNDARY;
        let c = Self {
            fixed: figure(v, "fixed_ms")?,
            work: figure(v, "per_work_ms")?,
            bandwidth: optional(v, "link_bytes_per_ms")?,
            latency: figure(v, "link_latency_ms")?,
            launch: figure(v, "launch_floor_ms")?,
            page_in: optional(v, "page_in_ms_per_byte")?,
        };
        if c.bandwidth.is_some_and(|n| n == 0.0) || (linked && c.bandwidth.is_none()) {
            return Err(Error::Invalid);
        }
        Ok(c)
    }
    fn best(passes: &[Self]) -> Self {
        let min = |f: fn(&Self) -> f64| passes.iter().map(f).fold(f64::INFINITY, f64::min);
        let opt = |f: fn(&Self) -> Option<f64>, max: bool| {
            passes
                .iter()
                .map(f)
                .collect::<Option<Vec<_>>>()
                .and_then(|v| v.into_iter().reduce(if max { f64::max } else { f64::min }))
        };
        Self {
            fixed: min(|c| c.fixed),
            work: min(|c| c.work),
            bandwidth: opt(|c| c.bandwidth, true),
            latency: min(|c| c.latency),
            launch: min(|c| c.launch),
            page_in: opt(|c| c.page_in, false),
        }
    }
}
#[derive(Debug)]
struct Measurement {
    family: String,
    numeric: String,
    numeric_contract_version: String,
    engine: Engine,
    boundary: String,
    region: Value,
    min: u64,
    max: u64,
    cost: Coefficients,
}
#[derive(Debug)]
pub struct Profile {
    id: String,
    document: Value,
    measurements: Vec<Measurement>,
    calibrated: BTreeSet<(Engine, &'static str)>,
    pinned: bool,
}
impl Profile {
    /// Exact live identity match is mandatory, including driver and toolchain.
    /// Frozen profiles are refused until M5-01 supplies their trusted manifest.
    pub fn parse(bytes: &[u8], expected: &Value) -> Result<Self, Error> {
        if bytes.len() > MAX_PROFILE_BYTES {
            return Err(Error::Invalid);
        }
        let v: Value = serde_json::from_slice(bytes).map_err(|_| Error::Invalid)?;
        if v["schema_version"] != VERSION
            || v["probe_version"] != PROBE
            || v["units"] != "ms,bytes,work"
        {
            return Err(Error::Invalid);
        }
        identity(&v["identity"])?;
        identity(expected)?;
        if &v["identity"] != expected {
            return Err(Error::Identity);
        }
        match v["provisional"].as_bool() {
            Some(true) => {}
            Some(false) => return Err(Error::FreezeRequired),
            None => return Err(Error::Invalid),
        }
        let rev = text(&v, "source_revision")?;
        if rev.len() != 40 || !rev.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(Error::Invalid);
        }
        for when in ["before", "after"] {
            let o = &v["observations"][when];
            optional(o, "load")?;
            if o.get("swap_bytes") != Some(&Value::Null) && o["swap_bytes"].as_u64().is_none() {
                return Err(Error::Invalid);
            }
        }
        let refused = v["refusals"]
            .as_array()
            .filter(|a| a.len() <= 128)
            .ok_or(Error::Invalid)?;
        for refusal in refused {
            text(refusal, "engine")?;
            text(refusal, "reason")?;
        }
        let rows = v["measurements"]
            .as_array()
            .filter(|a| a.len() <= 128)
            .ok_or(Error::Invalid)?;
        let protocol = v.get("statistics").is_some();
        if !protocol && v.get("coverage").is_some() {
            return Err(Error::Invalid);
        }
        let mut measurements = Vec::new();
        let mut calibrated = BTreeSet::new();
        let mut keys = BTreeSet::new();
        for row in rows {
            let family = text(row, "family")?.to_owned();
            if !family_valid(&family) {
                return Err(Error::Invalid);
            }
            let numeric = text(row, "numeric")?.to_owned();
            let numeric_contract_version = text(row, "numeric_contract_version")?.to_owned();
            let engine = Engine::parse(&row["engine"])?;
            let boundary = text(row, "boundary")?.to_owned();
            let region = row
                .get("region")
                .filter(|r| region_valid(r))
                .ok_or(Error::Invalid)?
                .clone();
            let key = (
                family.clone(),
                numeric.clone(),
                numeric_contract_version.clone(),
                engine,
                boundary.clone(),
                region.to_string(),
            );
            if !keys.insert(key) {
                return Err(Error::Invalid);
            }
            let sizes = row["sizes"]
                .as_array()
                .filter(|a| a.len() >= 2 && a.len() <= 16)
                .ok_or(Error::Invalid)?;
            let sizes = sizes
                .iter()
                .map(|n| n.as_u64().filter(|n| *n <= 1 << 28).ok_or(Error::Invalid))
                .collect::<Result<Vec<_>, _>>()?;
            if sizes.windows(2).any(|p| p[0] >= p[1]) {
                return Err(Error::Invalid);
            }
            let passes = row["passes"]
                .as_array()
                .filter(|p| p.len() == 5)
                .ok_or(Error::Invalid)?;
            let passes = passes
                .iter()
                .map(|p| Coefficients::parse(p, engine, &boundary))
                .collect::<Result<Vec<_>, _>>()?;
            let cost = Coefficients::parse(&row["best"], engine, &boundary)?;
            if cost != Coefficients::best(&passes) {
                return Err(Error::Invalid);
            }
            if protocol {
                calibrated.insert((engine, points(row, engine, &sizes)?));
            } else if row.get("points").is_some() || row.get("table_family").is_some() {
                return Err(Error::Invalid);
            }
            measurements.push(Measurement {
                family,
                numeric,
                numeric_contract_version,
                engine,
                boundary,
                region,
                min: sizes[0],
                max: sizes[sizes.len() - 1],
                cost,
            });
        }
        if protocol {
            coverage(&v, &calibrated)?;
        }
        let id = fingerprint(bytes);
        Ok(Self {
            id,
            document: v,
            measurements,
            calibrated,
            pinned: false,
        })
    }
    /// `parse` after checking the size and then the exact bytes against a
    /// committed SHA-256, before any JSON is read.
    pub fn parse_pinned(bytes: &[u8], expected: &Value, sha256: &str) -> Result<Self, Error> {
        if bytes.len() > MAX_PROFILE_BYTES {
            return Err(Error::Invalid);
        }
        if fingerprint(bytes) != sha256 {
            return Err(Error::Digest);
        }
        Self::parse(bytes, expected).map(|p| Self { pinned: true, ..p })
    }
    /// True only for a family measured under the declared M5-01 protocol in
    /// a profile loaded by `parse_pinned`. Recomputation proves each summary,
    /// not that `passes` and `best` came from those samples, so only a
    /// committed digest of the exact bytes can vouch for a calibration.
    pub fn calibrated(&self, engine: Engine, family: &str) -> bool {
        self.pinned && self.calibrated.contains(&(engine, family))
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn document(&self) -> &Value {
        &self.document
    }
    /// A bounded read also covers files whose size changes after metadata.
    pub fn load(path: &Path, expected: &Value) -> Result<Self, Error> {
        let mut bytes = Vec::new();
        File::open(path)
            .map_err(|_| Error::Io)?
            .take((MAX_PROFILE_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|_| Error::Io)?;
        Self::parse(&bytes, expected)
    }
    /// Only absence of the implicit default is a CPU fallback. An explicit
    /// path, permissions error, or malformed default remains a typed error.
    pub fn load_default(
        path: &Path,
        explicit: bool,
        expected: &Value,
    ) -> Result<Option<Self>, Error> {
        match File::open(path) {
            Err(e) if !explicit && e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(_) => Err(Error::Io),
            Ok(mut file) => {
                let mut bytes = Vec::new();
                (&mut file)
                    .take((MAX_PROFILE_BYTES + 1) as u64)
                    .read_to_end(&mut bytes)
                    .map_err(|_| Error::Io)?;
                Self::parse(&bytes, expected).map(Some)
            }
        }
    }
    fn measurement(&self, f: &Facts<'_>, engine: Engine) -> Option<&Measurement> {
        self.measurements
            .iter()
            .filter(|m| {
                m.engine == engine
                    && m.family == f.family
                    && m.numeric == f.numeric
                    && m.numeric_contract_version == f.numeric_contract_version
                    && m.boundary
                        == if engine == Engine::Cpu {
                            f.cpu_boundary
                        } else {
                            f.device_boundary
                        }
                    && f.work as u64 >= m.min
                    && f.work as u64 <= m.max
                    && (m.region.is_null() || f.region == Some(&m.region))
            })
            .max_by_key(|m| !m.region.is_null())
    }
}
/// Facts must come from a freshly verified plan and the actual adapter API.
/// This descriptive API does not turn a caller's legality Boolean into proof.
pub struct Facts<'a> {
    pub family: &'a str,
    pub numeric: &'a str,
    pub numeric_contract_version: &'a str,
    pub cpu_boundary: &'a str,
    pub device_boundary: &'a str,
    pub region: Option<&'a Value>,
    pub work: usize,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub launches: usize,
    pub temporary_bytes: usize,
    pub capacity_bytes: usize,
    pub residency: Residency,
    pub pages: PageState,
    pub cuda_legal: bool,
    pub metal_legal: bool,
}
#[derive(Clone, Debug)]
pub struct Decision {
    pub engine: Engine,
    pub receipt: Value,
}
fn finite(n: f64) -> Option<f64> {
    (n.is_finite() && n >= 0.0).then_some(n)
}
fn link(c: Coefficients, bytes: usize, transfers: usize) -> Option<f64> {
    if bytes == 0 {
        Some(0.0)
    } else {
        finite(bytes as f64 / c.bandwidth? + c.latency * transfers as f64)
    }
}
fn page_cost(c: Coefficients, f: &Facts<'_>) -> Option<f64> {
    if f.input_bytes == 0 {
        return Some(0.0);
    }
    match f.pages {
        PageState::Known { resident, total } if total > 0 && resident <= total => {
            if resident == total {
                Some(0.0)
            } else {
                finite(c.page_in? * f.input_bytes as f64 * (total - resident) as f64 / total as f64)
            }
        }
        _ => None,
    }
}
fn cpu_transfer(c: Coefficients, f: &Facts<'_>) -> Option<f64> {
    let input = if f.residency.inputs == Location::Device {
        f.input_bytes
    } else {
        0
    };
    let output = if f.residency.output == Location::Device {
        f.output_bytes
    } else {
        0
    };
    link(
        c,
        input.checked_add(output)?,
        usize::from(input != 0) + usize::from(output != 0),
    )
}
fn cpu_pages(c: Coefficients, f: &Facts<'_>) -> Option<f64> {
    if f.residency.inputs == Location::Host {
        page_cost(c, f)
    } else {
        Some(0.0)
    }
}
fn cpu_cost(c: Coefficients, f: &Facts<'_>) -> Option<f64> {
    finite(c.fixed + c.work * f.work as f64 + cpu_transfer(c, f)? + cpu_pages(c, f)?)
}
fn device_pages(c: Coefficients, f: &Facts<'_>) -> Option<f64> {
    if f.residency.inputs == Location::Host {
        page_cost(c, f)
    } else {
        Some(0.0)
    }
}
fn device_cost(c: Coefficients, f: &Facts<'_>) -> Option<(f64, f64, f64, f64)> {
    let input = if f.residency.inputs == Location::Host {
        f.input_bytes
    } else {
        0
    };
    let output = if f.residency.output == Location::Host {
        f.output_bytes
    } else {
        0
    };
    if f.device_boundary == "host-map-v1" && f.residency != Residency::HOST {
        return None;
    }
    let page_in = device_pages(c, f)?;
    let transfer = link(
        c,
        input.checked_add(output)?,
        usize::from(input != 0) + usize::from(output != 0),
    )?;
    let compute = finite(c.fixed + c.work * f.work as f64)?;
    let launch = finite(c.launch * f.launches as f64)?;
    Some((
        compute,
        transfer + page_in,
        launch,
        finite(compute + transfer + launch + page_in)?,
    ))
}
/// Strict 5% win, then stable CUDA/Metal order. Missing/unknown costs retain CPU.
/// This never performs measurements, device work, or buffer reads.
pub fn select(profile: Option<&Profile>, trust_provisional: bool, f: &Facts<'_>) -> Decision {
    let pages = match f.pages {
        PageState::Unknown => Value::Null,
        PageState::Known { resident, total } => json!({"resident":resident,"total":total}),
    };
    let cpu_measurement = profile
        .filter(|_| trust_provisional)
        .and_then(|p| p.measurement(f, Engine::Cpu));
    let cpu = profile
        .filter(|_| trust_provisional)
        .and_then(|p| p.measurement(f, Engine::Cpu))
        .and_then(|m| cpu_cost(m.cost, f));
    let mut receipt = json!({"schema_version":VERSION,"profile_id":profile.map(Profile::id),"provisional":profile.map(|_|true),
        "profile_schema_version":profile.map(|_|VERSION),"probe_version":profile.map(|_|PROBE),
        "profile_identity":profile.map(|p|&p.document["identity"]),
        "input_bytes":f.input_bytes,"output_bytes":f.output_bytes,"temporary_bytes":f.temporary_bytes,
        "capacity_bytes":f.capacity_bytes,"launches":f.launches,
        "trust_provisional":trust_provisional,"family":f.family,"numeric":f.numeric,"region":f.region,
        "numeric_contract_version":f.numeric_contract_version,
        "cpu_boundary":f.cpu_boundary,"device_boundary":f.device_boundary,"work":f.work,
        "residency":{"inputs":format!("{:?}",f.residency.inputs),"output":format!("{:?}",f.residency.output)},
        "host_pages":pages,"cpu_ms":cpu,
        "cpu_cost":{"fixed_ms":cpu_measurement.map(|m|m.cost.fixed),"work_ms":cpu_measurement.and_then(|m|finite(m.cost.work * f.work as f64)),
            "page_in_ms":cpu_measurement.and_then(|m|cpu_pages(m.cost,f)),"transfer_ms":cpu_measurement.and_then(|m|cpu_transfer(m.cost,f)),"launch_ms":0.0,"total_ms":cpu},"margin":0.05,"variants":[],"engine":"cpu"});
    let mut chosen = Engine::Cpu;
    let mut best = f64::INFINITY;
    let mut variants = Vec::new();
    for (engine, legal) in [(Engine::Cuda, f.cuda_legal), (Engine::Metal, f.metal_legal)] {
        let capacity = f
            .input_bytes
            .checked_add(f.output_bytes)
            .and_then(|n| n.checked_add(f.temporary_bytes))
            .is_some_and(|n| n <= f.capacity_bytes);
        let measured = profile.and_then(|p| p.measurement(f, engine));
        let estimate = measured.and_then(|m| device_cost(m.cost, f));
        let reason = if !legal {
            "illegal"
        } else if !capacity {
            "capacity"
        } else if profile.is_none() {
            "profile-missing"
        } else if !trust_provisional {
            "provisional-untrusted"
        } else if cpu.is_none() {
            "cpu-cost-unknown"
        } else if estimate.is_none() {
            "device-cost-unknown"
        } else {
            "measured"
        };
        if f.work != 0
            && reason == "measured"
            && let (Some(c), Some((_, _, _, d))) = (cpu, estimate)
            && d < 0.95 * c
            && d < best
        {
            chosen = engine;
            best = d;
        }
        variants.push(json!({"engine":engine.name(),"legal":legal,"capacity_admitted":capacity,"reason":reason,"refusal_code":match reason {"measured"=>None,"illegal"=>Some("TUNING_ADMISSION"),"capacity"=>Some("TUNING_BUDGET"),"profile-missing"=>Some("TUNING_NO_CANDIDATES"),"provisional-untrusted"=>Some("TUNING_FREEZE_REQUIRED"),_=>Some("TUNING_COST_UNKNOWN")},
            "fixed_ms":measured.map(|m|m.cost.fixed),"work_ms":measured.and_then(|m|finite(m.cost.work * f.work as f64)),
            "page_in_ms":measured.and_then(|m|device_pages(m.cost,f)),"compute_ms":estimate.map(|e|e.0),"transfer_ms":estimate.and_then(|e|measured.and_then(|m|device_pages(m.cost,f)).map(|p|(e.1-p).max(0.0))),"launch_ms":estimate.map(|e|e.2),"total_ms":estimate.map(|e|e.3)}));
    }
    receipt["variants"] = json!(variants);
    receipt["engine"] = json!(chosen.name());
    Decision {
        engine: chosen,
        receipt,
    }
}

/// GNET-04 D1(a)/D2(a) legal engine set for the general net of `main`. The CPU
/// is always legal; a device is legal when its source plan admits the program,
/// otherwise its typed refusal class is recorded. No profile row measures
/// general-net dispatch yet, so the reason stays `uncalibrated` until M5-01
/// timing (#214) gives a measured policy. Admission never runs device work.
pub fn legality(source: &[u8]) -> Value {
    use crate::general_net::{compiled::Target, device_host};
    let device = |engine: Engine, target| match device_host::Plan::from_source(
        source,
        "main",
        target,
        Default::default(),
    ) {
        Ok(plan) => {
            let forks = plan.artifact().fork_plan();
            json!({"engine":engine.name(),"legal":true,"refusal":null,
                    "forks":forks.map_or(0, |f| f.candidates.len()),
                    "serialized":forks.into_iter().flat_map(|f| &f.serialized).map(|s| s.reason.label()).collect::<Vec<_>>()})
        }
        Err(e) => json!({"engine":engine.name(),"legal":false,
                "refusal":device_host::refusal_class(&e),"forks":0,"serialized":[]}),
    };
    // D2(a): only the LLVM native engine is compiled; never present the
    // reference evaluator as compiled.
    let native = cfg!(feature = "native");
    json!({"policy":"legal-set-cpu","reason":"uncalibrated","engines":[
        {"engine":"cpu","legal":true,"compiled":native,
            "reason":if native {Value::Null} else {json!("native-feature-unavailable")}},
        device(Engine::Cuda, Target::Cuda),
        device(Engine::Metal, Target::Metal)]})
}

/// GNET-04 D1(a)/D2(a) host route of a placement run. Before calibration the
/// CPU route runs: the whole program on LLVM native when the build has it and
/// no trusted profile exists. A trusted profile keeps #172's profiled host,
/// the MIR reference evaluator offering measured regions to other engines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Native,
    Reference(&'static str),
}
impl Route {
    pub fn select(native: bool, trusted_profile: bool) -> Self {
        match (native, trusted_profile) {
            (false, _) => Self::Reference("native-feature-unavailable"),
            (true, true) => Self::Reference("profiled-regions"),
            (true, false) => Self::Native,
        }
    }
    /// Never presents the reference evaluator as compiled.
    pub fn receipt(self) -> Value {
        match self {
            Self::Native => json!({"host":"native","reason":null}),
            Self::Reference(reason) => json!({"host":"reference","reason":reason}),
        }
    }
}
