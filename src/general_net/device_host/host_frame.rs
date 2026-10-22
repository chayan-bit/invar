//! P2 host frame. Native program-specific continuations own serial control;
//! certified regions dispatch through a separate physical backend. A complete
//! fence is required before decoding, merging, replay or host publication.
//!
//! v1 uses bounded private lane images. This correctness path deliberately
//! exposes its storage/copy ceiling rather than making a resident/D5 claim.
//! P5 can consume the same site/seed/capsule protocol in device epochs; P6 can
//! choose among these already-certified sites and the native frame.
use super::*;
use compiled::body_map::{self, Compiled, Shape, Site};
mod native;
use native::Native;

pub type Image = [Vec<u64>; 12];

/// A backend must retain all twelve images through confirmed completion. A
/// failed/unknown command is an error, never an invitation to CPU fallback.
/// Runtime-guard/resource fallbacks are selected only after a completed fence.
pub trait RegionBackend {
    fn execute(&mut self, program: &Compiled, image: &Image, groups: usize)
    -> Result<Image, Error>;
}

/// Explicit CPU execution model of the identical CUDA-spelled capsule.
/// This compiles the body; it does not invoke the oracle or interpret MIR.
#[derive(Default)]
pub struct CpuLanes {
    native: std::collections::BTreeMap<[usize; 12], Native>,
    identity: Option<[u8; 32]>,
}
impl RegionBackend for CpuLanes {
    fn execute(
        &mut self,
        program: &Compiled,
        image: &Image,
        _groups: usize,
    ) -> Result<Image, Error> {
        if program.lanes.target() != Target::Cuda {
            return Err(Error::Input);
        }
        let identity = Sha256::digest(program.lanes.source().as_bytes()).into();
        let bytes = std::array::from_fn(|i| image[i].len());
        if self.identity != Some(identity) {
            self.native.clear();
            self.identity = Some(identity);
        }
        if !self.native.contains_key(&bytes) {
            // Exactly two shapes cover the probe/reconstruct frontier passes.
            if self.native.len() == 2 {
                self.native.clear();
            }
            self.native
                .insert(bytes, Native::compile(&program.lanes, image, true)?);
        }
        self.native
            .get_mut(&bytes)
            .ok_or(Error::Malformed)?
            .run(image)
    }
}

pub struct Plan {
    entry: super::Plan,
    compiled: Compiled,
    native: Option<Native>,
    native_bytes: Option<[usize; 12]>,
}
#[derive(Debug)]
pub struct Report {
    pub observation: Observation,
    pub launches: u64,
    pub lanes: u64,
    pub replayed: u64,
    pub native_epochs: u64,
}
impl Plan {
    pub fn from_source(
        source: &[u8],
        entry: &str,
        target: Target,
        mut limits: Limits,
    ) -> Result<Self, Error> {
        // P2 never speculates independent calls; their continuation stays intact.
        limits.private.slots = 0;
        let entry = super::Plan::from_source(source, entry, target, limits)?;
        let form = super::super::normal_form::extract(entry.net(), entry.proof())
            .map_err(|e| Error::Backend(format!("PNF:{e:?}")))?;
        let compiled = body_map::emit(
            entry.net(),
            entry.proof(),
            &entry.entry,
            target,
            &form,
            limits.compiler,
            limits.owned,
        )
        .map_err(|e| Error::Backend(format!("P2 emission:{e:?}")))?;
        Ok(Self {
            entry,
            compiled,
            native: None,
            native_bytes: None,
        })
    }
    pub fn compiled(&self) -> &Compiled {
        &self.compiled
    }
    pub fn entry_plan(&self) -> &super::Plan {
        &self.entry
    }

    /// Synchronous invocation with exclusive host array leases. Runtime images
    /// are private and can only originate in this source-issued plan. Completed
    /// source state preserves the exact prefix, including a partially executed
    /// iteration recovered by native replay. Host mut arrays commit only on a
    /// normal return, matching the existing CPU net oracle entry contract.
    pub fn run<B: RegionBackend>(
        &mut self,
        backend: &mut B,
        args: &[Value],
        mut config: DeviceConfig,
    ) -> Result<Report, Error> {
        config.capacity = 0;
        config.rounds = 1;
        let artifact = &self.compiled.frame;
        let layout = artifact.device_layout().ok_or(Error::Malformed)?;
        let bytes = layout.buffer_bytes(config).map_err(Error::Compiler)?;
        let total = bytes
            .iter()
            .try_fold(0usize, |a, b| a.checked_add(*b))
            .ok_or(Error::Limit)?;
        if total
            .checked_mul(4)
            .is_none_or(|n| n > self.entry.limits.buffer_bytes.min(64 * 1024 * 1024))
        {
            return Err(Error::Limit);
        }
        let mut packet = marshal::encode(artifact, args)?;
        let mut image: Image = std::array::from_fn(|i| vec![0; bytes[i] / 8]);
        image[0][..packet.words.len()].copy_from_slice(&packet.words);
        image[3].copy_from_slice(&config.words());
        image[11][0] = 1;
        if self.native_bytes != Some(bytes) {
            self.native = Some(Native::compile(artifact, &image, false)?);
            self.native_bytes = Some(bytes);
        }
        let mut report = Report {
            observation: Observation {
                status: compiled::Status::InvalidState,
                result: None,
                span: None,
                steps: 0,
                transitions: 0,
                max_frames: 0,
                events: Vec::new(),
                cleanup: Vec::new(),
                private_state: [0; 14],
            },
            launches: 0,
            lanes: 0,
            replayed: 0,
            native_epochs: 0,
        };
        for _ in 0..self.entry.limits.submissions {
            image = self.native.as_mut().ok_or(Error::Malformed)?.run(&image)?;
            report.native_epochs += 1;
            let t = self.compiled.frame_words;
            if image[6][t + 1] == 1 {
                let task = &image[6];
                let h = &image[1];
                if task[t + 4] > config.fuel
                    || task[t + 5] > config.transitions
                    || task[t + 6] > config.trace_events as u64
                    || task[t + 2] > 64
                    || task[t + 3] > 64
                    || h[0] != task[t + 14]
                    || h[1] != task[t + 15]
                    || h[2] != task[t + 16]
                    || h[3] != task[t + 17]
                    || h[2] > h[3]
                    || h[3] > self.entry.source_bytes as u64
                {
                    return Err(Error::Malformed);
                }
                let status = match h[0] {
                    0 => compiled::Status::Success,
                    1 => compiled::Status::IntegerOverflow,
                    2 => compiled::Status::DivisionByZero,
                    3 => compiled::Status::FuelLimit,
                    4 => compiled::Status::CallDepthLimit,
                    5 => compiled::Status::TransitionLimit,
                    6 => compiled::Status::TraceLimit,
                    7 => compiled::Status::InvalidState,
                    8 => compiled::Status::InvalidInput,
                    9 => compiled::Status::IndexOutOfBounds,
                    10 => compiled::Status::ArraySizeLimit,
                    11 => compiled::Status::HeapLimit,
                    12 => compiled::Status::AllocationLimit,
                    13 => compiled::Status::CleanupLimit,
                    _ => return Err(Error::Malformed),
                };
                let result = if status == compiled::Status::Success {
                    let allocated = usize::try_from(task[t + 8]).map_err(|_| Error::Malformed)?;
                    Some(marshal::result(artifact, h[1], &image[4], allocated)?)
                } else {
                    None
                };
                let used = usize::try_from(task[t + 6]).map_err(|_| Error::Malformed)?;
                let cleanup = usize::try_from(task[t + 10]).map_err(|_| Error::Malformed)?;
                if used > image[2].len() / 7 || cleanup > image[5].len() / 4 {
                    return Err(Error::Malformed);
                }
                let observation = Observation {
                    status,
                    result,
                    span: (status != compiled::Status::Success).then_some(Span {
                        start: h[2] as usize,
                        end: h[3] as usize,
                    }),
                    steps: task[t + 4],
                    transitions: task[t + 5],
                    max_frames: task[t + 3] as usize,
                    events: image[2][..used * 7]
                        .chunks_exact(7)
                        .map(|e| e.try_into().expect("seven words"))
                        .collect(),
                    cleanup: image[5][..cleanup * 4]
                        .chunks_exact(4)
                        .map(|e| e.try_into().expect("four words"))
                        .collect(),
                    private_state: [0; 14],
                };
                if status == compiled::Status::Success {
                    marshal::publish(
                        &mut packet.leases,
                        artifact.owned_layout().ok_or(Error::Malformed)?,
                        &image[4],
                    )?;
                }
                report.observation = observation;
                return Ok(report);
            }
            let pending = image[11][1];
            if pending == 0 {
                continue;
            }
            let site = self
                .compiled
                .sites
                .get(usize::try_from(pending - 1).map_err(|_| Error::Malformed)?)
                .ok_or(Error::Malformed)?;
            let lo = image[11][3];
            let hi = image[11][4];
            if lo as i64 >= hi as i64 || hi as i64 > u32::MAX as i64 {
                return Err(Error::Malformed);
            }
            let Some(mut work) = self.capsule(&image, pending, lo, hi, config)? else {
                image[11][0] = 0;
                report.replayed += 1;
                continue;
            };
            let n = work[3][0] as usize;
            work = self.dispatch(backend, &work, n)?;
            report.launches += 1;
            report.lanes += n as u64;
            let resolved = if self.completed(&image, &work, n)? {
                if site.shape == Shape::Frontier {
                    if let Some(second) = self.frontier(&image, &work, site, n)? {
                        work = self.dispatch(backend, &second, n)?;
                        report.launches += 1;
                        self.completed(&image, &work, n)?
                    } else {
                        false
                    }
                } else {
                    true
                }
            } else {
                false
            };
            if resolved && self.merge(&mut image, &work, n, config)? {
                image[11][1] = 0;
            } else {
                // Only already checked successful lanes may commit before the
                // winning exit/trap. Frontier probes need reconstruction first.
                if !resolved && site.shape != Shape::Frontier {
                    let cut = work[11][9] as usize;
                    if cut > 0 {
                        let _ = self.merge(&mut image, &work, cut, config)?;
                    }
                }
                // Replay starts at the winning iteration (or at the untouched
                // checkpoint when the source resource budget cut came earlier).
                image[11][0] = 0;
                image[11][1] = 0;
                report.replayed += 1;
            }
        }
        Err(Error::Limit)
    }

    fn capsule(
        &self,
        root: &Image,
        site: u64,
        lo: u64,
        hi: u64,
        config: DeviceConfig,
    ) -> Result<Option<Image>, Error> {
        let owned = self.compiled.frame.owned_layout().ok_or(Error::Malformed)?;
        let trace = config.trace_events.min(2048);
        let task = root[6].len();
        let heap = owned.heap_words();
        let clean = owned.cleanup_words().max(1);
        let per = 10usize
            .checked_add(trace.checked_mul(7).ok_or(Error::Limit)?)
            .and_then(|n| n.checked_add(task))
            .and_then(|n| n.checked_add(heap))
            .and_then(|n| n.checked_add(clean))
            .and_then(|n| n.checked_add(14))
            .and_then(|n| n.checked_mul(8))
            .ok_or(Error::Limit)?;
        let base = (root[0].len() + task + heap + 16) * 8;
        // Bound simultaneous upload/readback/validation/merge copies, including
        // the CPU model's retained image and loader transfer staging.
        let budget = self.entry.limits.buffer_bytes.min(64 * 1024 * 1024) / 8;
        let n = budget.saturating_sub(base) / per;
        let n = n
            .min(
                usize::try_from((hi as i64).checked_sub(lo as i64).ok_or(Error::Malformed)?)
                    .map_err(|_| Error::Limit)?,
            )
            .min(1024);
        if n == 0 {
            return Ok(None);
        }
        let mut work: Image = std::array::from_fn(|_| vec![0]);
        work[0] = root[0].clone();
        work[1] = vec![0; n * 10];
        work[2] = vec![0; (n * trace * 7).max(1)];
        work[3] = vec![n as u64, 1_000_000, 1_000_000, trace as u64, site, lo];
        work[4] = vec![0; n * heap];
        work[5] = vec![0; n * clean];
        work[6] = vec![0; n * task];
        work[7] = root[6].clone();
        work[8] = root[4].clone();
        work[11] = vec![0; n * 14];
        Ok(Some(work))
    }

    fn dispatch<B: RegionBackend>(
        &self,
        backend: &mut B,
        input: &Image,
        n: usize,
    ) -> Result<Image, Error> {
        let output = backend.execute(
            &self.compiled,
            input,
            Compiled::groups(n).map_err(Error::Compiler)?,
        )?;
        if output.iter().zip(input).any(|(a, b)| a.len() != b.len())
            || [0, 3, 7, 8, 9, 10].iter().any(|&i| output[i] != input[i])
        {
            return Err(Error::Malformed);
        }
        Ok(output)
    }

    fn completed(&self, root: &Image, work: &Image, n: usize) -> Result<bool, Error> {
        let t = self.compiled.frame_words;
        let tw = t + 18;
        if work[6].len() != n * tw || work[11].len() != n * 14 {
            return Err(Error::Malformed);
        }
        let cut = work[11]
            .chunks_exact(14)
            .position(|c| c[6] != 1)
            .unwrap_or(n);
        if work[11][9] != cut as u64 {
            return Err(Error::Malformed);
        }
        for lane in 0..n {
            let task = &work[6][lane * tw..(lane + 1) * tw];
            let c = &work[11][lane * 14..(lane + 1) * 14];
            if c[0] != 2 || c[1] != work[3][4] || c[2] != root[6][t + 2] {
                return Err(Error::Malformed);
            }
            if c[6] != 1 || task[t + 1] != 0 || task[t + 14] != 0 {
                return Ok(false);
            }
            if c[8] != 1
                || task[t + 2] != root[6][t + 2]
                || task[t + 3] > 64
                || task[t + 4] > 1_000_000
                || task[t + 5] > 1_000_000
                || task[t + 6] > work[3][3]
                || task[t + 7] < root[6][t + 7]
                || task[t + 8] < root[6][t + 8]
                || task[t + 8] > self.entry.limits.owned.allocations as u64
                || task[t + 10] > self.entry.limits.owned.cleanup_events as u64
            {
                return Err(Error::Malformed);
            }
        }
        Ok(true)
    }

    fn frontier(
        &self,
        root: &Image,
        first: &Image,
        site: &Site,
        n: usize,
    ) -> Result<Option<Image>, Error> {
        let t = self.compiled.frame_words;
        let tw = t + 18;
        let stride = t / 64;
        let base = (root[6][t + 2] as usize - 1) * stride;
        let p = base + self.compiled.values;
        let owned = self.compiled.frame.owned_layout().ok_or(Error::Malformed)?;
        let heap = owned.heap_words();
        let pool = owned.allocations() * 4;
        let mut second = self
            .capsule(
                root,
                first[3][4],
                first[3][5],
                root[11][4],
                DeviceConfig {
                    fuel: root[3][0],
                    transitions: root[3][1],
                    trace_events: root[3][2] as usize,
                    quantum: root[3][3],
                    rounds: 1,
                    capacity: 0,
                },
            )?
            .ok_or(Error::Malformed)?;
        second[9] = vec![1];
        second[10] = vec![0; (n * site.flags.len()).max(1)];
        let mut prefixes: Vec<i64> = site
            .sums
            .iter()
            .map(|(slot, _)| root[6][p + slot] as i64)
            .collect();
        for lane in 0..n {
            for (j, (slot, unsigned)) in site.sums.iter().enumerate() {
                second[9].push(prefixes[j] as u64);
                let delta = (first[6][lane * tw + p + slot] as i64)
                    .checked_sub(root[6][p + slot] as i64)
                    .ok_or(Error::Malformed)?;
                let next = prefixes[j].checked_add(delta);
                let valid = next.is_some_and(|x| {
                    if *unsigned {
                        u32::try_from(x).is_ok()
                    } else {
                        i32::try_from(x).is_ok()
                    }
                });
                if !valid {
                    return Ok(None);
                }
                prefixes[j] = next.ok_or(Error::Malformed)?;
            }
            for (j, (slot, flag)) in site.flags.iter().enumerate() {
                let handle = root[6][p + slot];
                if handle == 0 || handle > owned.allocations() as u64 {
                    return Err(Error::Malformed);
                }
                let d = (handle as usize - 1) * 4;
                let index = (first[3][5] as usize)
                    .checked_add(lane)
                    .ok_or(Error::Malformed)?;
                let laneheap = &first[4][lane * heap..(lane + 1) * heap];
                if index >= laneheap[d + 1] as usize {
                    return Ok(None);
                }
                let at = pool
                    .checked_add(laneheap[d] as usize)
                    .and_then(|x| x.checked_add(index))
                    .ok_or(Error::Malformed)?;
                second[10][lane * site.flags.len() + j] =
                    u64::from(laneheap.get(at) == Some(&(*flag as u64)));
            }
        }
        Ok(Some(second))
    }

    fn merge(
        &self,
        root: &mut Image,
        work: &Image,
        n: usize,
        config: DeviceConfig,
    ) -> Result<bool, Error> {
        let t = self.compiled.frame_words;
        let tw = t + 18;
        let stride = t / 64;
        let depth = root[6][t + 2] as usize;
        if depth == 0 || depth > 64 {
            return Err(Error::Malformed);
        }
        let frame = (depth - 1) * stride;
        let frame_id = root[6][frame + stride - 2];
        let owned = self.compiled.frame.owned_layout().ok_or(Error::Malformed)?;
        let heap = owned.heap_words();
        let clean = owned.cleanup_words().max(1);
        let pool = owned.allocations() * 4;
        let initial = root[6].clone();
        let mut merged = root.clone();
        let mut steps = initial[t + 4];
        let mut transitions = initial[t + 5];
        let mut used = initial[t + 6];
        let mut next = initial[t + 7];
        let mut allocated = initial[t + 8];
        let mut cleanup = initial[t + 10];
        let mut activation = 0;
        let mut max_depth = initial[t + 3];
        for lane in 0..n {
            let task = &work[6][lane * tw..(lane + 1) * tw];
            steps = steps.checked_add(task[t + 4]).ok_or(Error::Malformed)?;
            transitions = transitions
                .checked_add(task[t + 5])
                .ok_or(Error::Malformed)?;
            let end = used.checked_add(task[t + 6]).ok_or(Error::Malformed)?;
            let cleanup_end = cleanup.checked_add(task[t + 10]).ok_or(Error::Malformed)?;
            let alloc_end = allocated
                .checked_add(task[t + 8] - initial[t + 8])
                .ok_or(Error::Malformed)?;
            if steps > config.fuel
                || transitions > config.transitions
                || end > config.trace_events as u64
                || cleanup_end > owned.cleanup_events() as u64
                || alloc_end > owned.allocations() as u64
            {
                return Ok(false);
            }
            let frame_delta = next - initial[t + 7];
            for (j, e) in work[2][lane * work[3][3] as usize * 7..][..task[t + 6] as usize * 7]
                .chunks_exact(7)
                .enumerate()
            {
                let mut event: [u64; 7] = e.try_into().expect("seven words");
                if event[1] == frame_id {
                    event[2] = event[2].checked_add(activation).ok_or(Error::Malformed)?;
                } else if event[1] >= initial[t + 7] {
                    event[1] = event[1].checked_add(frame_delta).ok_or(Error::Malformed)?;
                }
                merged[2][(used as usize + j) * 7..(used as usize + j + 1) * 7]
                    .copy_from_slice(&event);
            }
            for (j, e) in work[5][lane * clean..][..task[t + 10] as usize * 4]
                .chunks_exact(4)
                .enumerate()
            {
                let mut event: [u64; 4] = e.try_into().expect("four words");
                if event[3] >= initial[t + 7] {
                    event[3] = event[3].checked_add(frame_delta).ok_or(Error::Malformed)?;
                }
                merged[5][(cleanup as usize + j) * 4..(cleanup as usize + j + 1) * 4]
                    .copy_from_slice(&event);
            }
            let laneheap = &work[4][lane * heap..(lane + 1) * heap];
            for id in 0..initial[t + 8] as usize {
                let d = id * 4;
                if root[4][d + 3] & 1 == 0 {
                    continue;
                }
                let len = root[4][d + 1] as usize;
                if laneheap[d + 1] != root[4][d + 1]
                    || laneheap[d + 2] != root[4][d + 2]
                    || laneheap[d + 3] != root[4][d + 3]
                {
                    return Ok(false);
                }
                let old = pool
                    .checked_add(root[4][d] as usize)
                    .ok_or(Error::Malformed)?;
                let new = pool
                    .checked_add(laneheap[d] as usize)
                    .ok_or(Error::Malformed)?;
                let a = root[4]
                    .get(old..old.checked_add(len).ok_or(Error::Malformed)?)
                    .ok_or(Error::Malformed)?;
                let b = laneheap
                    .get(new..new.checked_add(len).ok_or(Error::Malformed)?)
                    .ok_or(Error::Malformed)?;
                for (j, (a, b)) in a.iter().zip(b).enumerate() {
                    if a != b {
                        merged[4][old + j] = *b;
                    }
                }
            }
            for (j, id) in (initial[t + 8] as usize..task[t + 8] as usize).enumerate() {
                if laneheap[id * 4 + 3] & 1 != 0 {
                    return Ok(false);
                }
                merged[4][(allocated as usize + j) * 4..(allocated as usize + j + 1) * 4]
                    .copy_from_slice(&laneheap[id * 4..id * 4 + 4]);
            }
            activation = activation
                .checked_add(
                    task[frame + stride - 1]
                        .checked_sub(initial[frame + stride - 1])
                        .ok_or(Error::Malformed)?,
                )
                .ok_or(Error::Malformed)?;
            next = next
                .checked_add(task[t + 7] - initial[t + 7])
                .ok_or(Error::Malformed)?;
            max_depth = max_depth.max(task[t + 3]);
            used = end;
            cleanup = cleanup_end;
            allocated = alloc_end;
        }
        merged[6][..t].copy_from_slice(&work[6][(n - 1) * tw..(n - 1) * tw + t]);
        merged[6][frame + stride - 1] = initial[frame + stride - 1]
            .checked_add(activation)
            .ok_or(Error::Malformed)?;
        merged[6][t + 3] = max_depth;
        merged[6][t + 4] = steps;
        merged[6][t + 5] = transitions;
        merged[6][t + 6] = used;
        merged[6][t + 7] = next;
        merged[6][t + 8] = allocated;
        merged[6][t + 10] = cleanup;
        *root = merged;
        Ok(true)
    }
}
