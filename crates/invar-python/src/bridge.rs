//! Host bridge planning and NumPy borrow guards for reference array calls.
use super::*;

/// The base element width of the HA-1 bounds below (`i32`, `u32`, `f32`). A call holding an 8-byte `f64` array
/// scales the byte bounds by its widest element over this base (HA-1.4); every other call keeps them as they were.
pub(crate) const ELEMENT_BYTES: usize = 4;
/// The HA-1 host-array protocol's live-element bound (HA-1.1 §5), checked on
/// the original extents before any bridge exists. It is the foreign crate's
/// own constant, so the binding cannot drift from the protocol. It is a
/// separate tool bound, not the language limit, which was raised separately.
pub(crate) const LIVE_ELEMENTS: usize = invar_foreign::host_array::MAX_LIVE_ELEMENTS;
/// Bridges cover pairwise disjoint parts of the live union.
pub(crate) const BRIDGE_LIMIT_BYTES: usize = LIVE_ELEMENTS * ELEMENT_BYTES;
/// HA-1.1 §6 host temporary bound for one reference call: bridges plus the
/// core's reference copies, at most every argument at the per-array bound
/// plus the live union.
pub(crate) const TEMPORARY_LIMIT_BYTES: usize = (invar_foreign::host_array::MAX_ARRAYS
    * invar_foreign::host_array::MAX_ELEMENTS
    + LIVE_ELEMENTS)
    * ELEMENT_BYTES;

/// Private fault-injection seam; `call_reference` passes `None`.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BindingInject {
    None,
    /// The `n`th bridge reservation (0-based, unions first) fails.
    BridgeReserveFailAt(usize),
    /// Python result construction fails after a normal core return.
    ResultBuildFail,
    /// Panic after every writeback precheck, before the first NumPy write.
    PanicBeforeWriteback,
}

/// Original metadata of one array argument, for planning.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlanArray {
    pub(crate) dtype: Dtype,
    pub(crate) address: usize,
    pub(crate) length: usize,
    pub(crate) mutable: bool,
}

/// How one array argument reaches the core.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    /// An i32, u32, f64 or Bool read: the borrowed NumPy slice itself.
    Direct,
    /// An f32 read: a slice of union bridge `union` at element `offset`.
    Union { union: usize, offset: usize },
    /// An empty f32 read.
    EmptyF32,
    /// Any mut array: its own bridge, written back after the result is built.
    Mut,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct BridgePlan {
    pub(crate) routes: Vec<Route>,
    /// Element length of each f32 read union bridge.
    pub(crate) unions: Vec<usize>,
    /// Bytes of all bridges (f32 unions and mut copies).
    pub(crate) bridge_bytes: usize,
    /// Upper bound of the core's reference copies: each exact-alias group of
    /// the arguments the core sees, once.
    pub(crate) core_copy_bytes: usize,
    /// Live union of the original extents, in elements.
    pub(crate) live_elements: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PlanError {
    Live,
    Alias,
    Staging,
}

/// Plan f32 union bridges and mut bridges on original metadata and account
/// every simultaneously retained host temporary. Integer metadata only.
pub(crate) fn plan_bridges(arrays: &[PlanArray]) -> Result<BridgePlan, PlanError> {
    let range = |a: &PlanArray| {
        a.length
            .checked_mul(a.dtype.size())
            .and_then(|bytes| a.address.checked_add(bytes))
            .map(|end| (a.address, end))
            .ok_or(PlanError::Staging)
    };
    let ranges = arrays.iter().map(range).collect::<Result<Vec<_>, _>>()?;
    let live = |i: usize| arrays[i].length != 0;
    // Overlap rules the core would apply to the original arrays; bridges are
    // separate allocations, so the core can no longer see these overlaps.
    for i in 0..arrays.len() {
        for j in 0..i {
            let (a, b) = (ranges[i], ranges[j]);
            if live(i)
                && live(j)
                && a.0 < b.1
                && b.0 < a.1
                && (arrays[i].mutable || arrays[j].mutable || arrays[i].dtype != arrays[j].dtype)
            {
                return Err(PlanError::Alias);
            }
        }
    }
    // Live union of the original extents, in elements (as the core counts it): overlapping extents share one
    // element size, and extents that only touch merge when their sizes agree.
    let mut spans: Vec<(usize, usize, usize)> = (0..arrays.len())
        .filter(|&i| live(i))
        .map(|i| (ranges[i].0, ranges[i].1, arrays[i].dtype.size()))
        .collect();
    spans.sort_unstable();
    let mut live_elements = 0usize;
    let mut current: Option<(usize, usize, usize)> = None;
    for (start, end, size) in spans {
        current = match current {
            Some((s, e, z)) if start < e || (start == e && size == z) => Some((s, e.max(end), z)),
            Some((s, e, z)) => {
                live_elements = live_elements
                    .checked_add((e - s) / z)
                    .ok_or(PlanError::Live)?;
                Some((start, end, size))
            }
            None => Some((start, end, size)),
        };
    }
    if let Some((s, e, z)) = current {
        live_elements = live_elements
            .checked_add((e - s) / z)
            .ok_or(PlanError::Live)?;
    }
    if live_elements > LIVE_ELEMENTS {
        return Err(PlanError::Live);
    }
    // f32 read unions: connected (strictly overlapping) original extents.
    let mut routes = vec![Route::Direct; arrays.len()];
    let mut members: Vec<usize> = (0..arrays.len())
        .filter(|&i| arrays[i].dtype == Dtype::F32 && !arrays[i].mutable)
        .collect();
    for &i in &members {
        if !live(i) {
            routes[i] = Route::EmptyF32;
        }
    }
    members.retain(|&i| live(i));
    members.sort_unstable_by_key(|&i| ranges[i]);
    let mut unions: Vec<(usize, usize)> = Vec::new();
    for i in members {
        let (start, end) = ranges[i];
        match unions.last_mut() {
            Some((_, union_end)) if start < *union_end => *union_end = (*union_end).max(end),
            _ => unions.push((start, end)),
        }
        let union = unions.len() - 1;
        routes[i] = Route::Union {
            union,
            offset: (start - unions[union].0) / ELEMENT_BYTES,
        };
    }
    let mut bridge_bytes = 0usize;
    for (start, end) in &unions {
        bridge_bytes = bridge_bytes
            .checked_add(end - start)
            .ok_or(PlanError::Staging)?;
    }
    for (i, array) in arrays.iter().enumerate() {
        if array.mutable {
            routes[i] = Route::Mut;
            bridge_bytes = bridge_bytes
                .checked_add(ranges[i].1 - ranges[i].0)
                .ok_or(PlanError::Staging)?;
        }
    }
    // The byte bounds are stated for 4-byte elements; an 8-byte array doubles them (HA-1.4).
    let widest = arrays.iter().map(|a| a.dtype.size()).max().unwrap_or(0);
    let scale = widest.div_ceil(ELEMENT_BYTES).max(1);
    if bridge_bytes > BRIDGE_LIMIT_BYTES * scale {
        return Err(PlanError::Staging);
    }
    // Core copies: the core sees direct slices at their original addresses,
    // union slices at the same relative layout and mut bridges as their own
    // groups, so its exact-alias groups are those of the original arrays.
    let mut core_copy_bytes = 0usize;
    for i in (0..arrays.len()).filter(|&i| live(i)) {
        let earlier_alias = !arrays[i].mutable
            && (0..i).any(|j| {
                live(j)
                    && !arrays[j].mutable
                    && arrays[j].dtype == arrays[i].dtype
                    && ranges[j] == ranges[i]
            });
        if !earlier_alias {
            core_copy_bytes = core_copy_bytes
                .checked_add(ranges[i].1 - ranges[i].0)
                .ok_or(PlanError::Staging)?;
        }
    }
    if bridge_bytes
        .checked_add(core_copy_bytes)
        .is_none_or(|total| total > TEMPORARY_LIMIT_BYTES * scale)
    {
        return Err(PlanError::Staging);
    }
    Ok(BridgePlan {
        routes,
        unions: unions
            .iter()
            .map(|(s, e)| (e - s) / ELEMENT_BYTES)
            .collect(),
        bridge_bytes,
        core_copy_bytes,
        live_elements,
    })
}

/// A numpy read guard (shared borrow plus strong reference).
pub(crate) enum ReadGuard<'py> {
    I32(PyReadonlyArrayDyn<'py, i32>),
    U32(PyReadonlyArrayDyn<'py, u32>),
    F32(PyReadonlyArrayDyn<'py, f32>),
    F64(PyReadonlyArrayDyn<'py, f64>),
    Bool(PyReadonlyArrayDyn<'py, bool>),
}

/// A numpy write guard (exclusive borrow plus strong reference).
pub(crate) enum WriteGuard<'py> {
    I32(PyReadwriteArrayDyn<'py, i32>),
    U32(PyReadwriteArrayDyn<'py, u32>),
    F32(PyReadwriteArrayDyn<'py, f32>),
    F64(PyReadwriteArrayDyn<'py, f64>),
    Bool(PyReadwriteArrayDyn<'py, bool>),
}

pub(crate) enum ReadSlice<'g> {
    I32(&'g [i32]),
    U32(&'g [u32]),
    F32(&'g [f32]),
    F64(&'g [f64]),
    Bool(&'g [bool]),
}

pub(crate) enum WriteSlice<'g> {
    I32(&'g mut [i32]),
    U32(&'g mut [u32]),
    F32(&'g mut [f32]),
    F64(&'g mut [f64]),
    Bool(&'g mut [bool]),
}

pub(crate) fn borrow_failure(error: BorrowError) -> Failure {
    match error {
        BorrowError::NotWriteable => Failure::Writability,
        _ => Failure::Alias,
    }
}

pub(crate) fn borrow_guard<'py>(
    value: &Bound<'py, PyAny>,
    dtype: Dtype,
    mutable: bool,
    readers: &mut Vec<ReadGuard<'py>>,
    writers: &mut Vec<WriteGuard<'py>>,
) -> Result<(), Failure> {
    macro_rules! take {
        ($t:ty, $v:ident) => {{
            let array = value.cast::<PyArrayDyn<$t>>().map_err(|_| Failure::Dtype)?;
            if mutable {
                writers.push(WriteGuard::$v(
                    array.try_readwrite().map_err(borrow_failure)?,
                ));
            } else {
                readers.push(ReadGuard::$v(array.try_readonly().map_err(borrow_failure)?));
            }
            Ok(())
        }};
    }
    match dtype {
        Dtype::I32 => take!(i32, I32),
        Dtype::U32 => take!(u32, U32),
        Dtype::F32 => take!(f32, F32),
        Dtype::F64 => take!(f64, F64),
        Dtype::Bool => take!(bool, Bool),
        Dtype::U8 => Err(Failure::Dtype),
    }
}

impl<'py> ReadGuard<'py> {
    pub(crate) fn slice(&self) -> Result<ReadSlice<'_>, Failure> {
        let contiguity = |_| Failure::Contiguity;
        Ok(match self {
            Self::I32(g) => ReadSlice::I32(g.as_slice().map_err(contiguity)?),
            Self::U32(g) => ReadSlice::U32(g.as_slice().map_err(contiguity)?),
            Self::F32(g) => ReadSlice::F32(g.as_slice().map_err(contiguity)?),
            Self::F64(g) => ReadSlice::F64(g.as_slice().map_err(contiguity)?),
            Self::Bool(g) => ReadSlice::Bool(g.as_slice().map_err(contiguity)?),
        })
    }
}

impl<'py> WriteGuard<'py> {
    pub(crate) fn slice(&mut self) -> Result<WriteSlice<'_>, Failure> {
        let contiguity = |_| Failure::Contiguity;
        Ok(match self {
            Self::I32(g) => WriteSlice::I32(g.as_slice_mut().map_err(contiguity)?),
            Self::U32(g) => WriteSlice::U32(g.as_slice_mut().map_err(contiguity)?),
            Self::F32(g) => WriteSlice::F32(g.as_slice_mut().map_err(contiguity)?),
            Self::F64(g) => WriteSlice::F64(g.as_slice_mut().map_err(contiguity)?),
            Self::Bool(g) => WriteSlice::Bool(g.as_slice_mut().map_err(contiguity)?),
        })
    }
}

impl ReadSlice<'_> {
    /// Address (as an integer, never dereferenced) and length.
    pub(crate) fn extent(&self) -> (usize, usize) {
        match self {
            Self::I32(v) => (v.as_ptr().addr(), v.len()),
            Self::U32(v) => (v.as_ptr().addr(), v.len()),
            Self::F32(v) => (v.as_ptr().addr(), v.len()),
            Self::F64(v) => (v.as_ptr().addr(), v.len()),
            Self::Bool(v) => (v.as_ptr().addr(), v.len()),
        }
    }
}

impl WriteSlice<'_> {
    pub(crate) fn extent(&self) -> (usize, usize) {
        match self {
            Self::I32(v) => (v.as_ptr().addr(), v.len()),
            Self::U32(v) => (v.as_ptr().addr(), v.len()),
            Self::F32(v) => (v.as_ptr().addr(), v.len()),
            Self::F64(v) => (v.as_ptr().addr(), v.len()),
            Self::Bool(v) => (v.as_ptr().addr(), v.len()),
        }
    }
}

/// A mut array's bridge: the core commits here, never into NumPy memory.
pub(crate) enum Bridge {
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<F32Bits>),
    /// Exact binary64 bits: a `Vec<f64>` copy never rounds or canonicalises.
    F64(Vec<f64>),
    Bool(Vec<bool>),
}

impl Bridge {
    /// A fallibly reserved copy of the caller's current contents.
    pub(crate) fn copy(slice: &WriteSlice<'_>) -> Option<Self> {
        fn reserve<T>(length: usize) -> Option<Vec<T>> {
            let mut out = Vec::new();
            out.try_reserve_exact(length).ok()?;
            Some(out)
        }
        Some(match slice {
            WriteSlice::I32(v) => {
                let mut out = reserve(v.len())?;
                out.extend_from_slice(v);
                Self::I32(out)
            }
            WriteSlice::U32(v) => {
                let mut out = reserve(v.len())?;
                out.extend_from_slice(v);
                Self::U32(out)
            }
            WriteSlice::F32(v) => {
                let mut out = reserve(v.len())?;
                out.extend(v.iter().map(|x| F32Bits::from_bits(x.to_bits())));
                Self::F32(out)
            }
            WriteSlice::F64(v) => {
                let mut out = reserve(v.len())?;
                out.extend_from_slice(v);
                Self::F64(out)
            }
            WriteSlice::Bool(v) => {
                let mut out = reserve(v.len())?;
                out.extend_from_slice(v);
                Self::Bool(out)
            }
        })
    }

    pub(crate) fn argument(&mut self) -> HostArgument<'_> {
        match self {
            Self::I32(v) => HostArgument::Mut(HostMutArray::I32(v)),
            Self::U32(v) => HostArgument::Mut(HostMutArray::U32(v)),
            Self::F32(v) => HostArgument::Mut(HostMutArray::F32(v)),
            Self::F64(v) => HostArgument::Mut(HostMutArray::Float64(v)),
            Self::Bool(v) => HostArgument::Mut(HostMutArray::Bool(v)),
        }
    }
}

/// One prechecked writeback: a NumPy slice and its bridge, same element type
/// and length.
pub(crate) enum Pair<'w, 'b> {
    I32(&'w mut [i32], &'b [i32]),
    U32(&'w mut [u32], &'b [u32]),
    F32(&'w mut [f32], &'b [F32Bits]),
    F64(&'w mut [f64], &'b [f64]),
    Bool(&'w mut [bool], &'b [bool]),
}

impl<'w, 'b> Pair<'w, 'b> {
    pub(crate) fn new(slice: &'w mut WriteSlice<'_>, bridge: &'b Bridge) -> Option<Self> {
        match (slice, bridge) {
            (WriteSlice::I32(d), Bridge::I32(s)) if d.len() == s.len() => Some(Self::I32(d, s)),
            (WriteSlice::U32(d), Bridge::U32(s)) if d.len() == s.len() => Some(Self::U32(d, s)),
            (WriteSlice::F32(d), Bridge::F32(s)) if d.len() == s.len() => Some(Self::F32(d, s)),
            (WriteSlice::F64(d), Bridge::F64(s)) if d.len() == s.len() => Some(Self::F64(d, s)),
            (WriteSlice::Bool(d), Bridge::Bool(s)) if d.len() == s.len() => Some(Self::Bool(d, s)),
            _ => None,
        }
    }

    /// Equal lengths are guaranteed by `new`, the only panic condition of
    /// `copy_from_slice`; the f32 loop writes exact bits.
    pub(crate) fn write(self) {
        match self {
            Self::I32(d, s) => d.copy_from_slice(s),
            Self::U32(d, s) => d.copy_from_slice(s),
            Self::F32(d, s) => {
                for (slot, bits) in d.iter_mut().zip(s) {
                    *slot = f32::from_bits(bits.to_bits());
                }
            }
            Self::F64(d, s) => d.copy_from_slice(s),
            Self::Bool(d, s) => d.copy_from_slice(s),
        }
    }
}

pub(crate) fn staging_limit(py: Python<'_>) -> PyErr {
    boundary_error(
        py,
        Class::Resource,
        "PY_STAGING_LIMIT",
        "host staging for a reference array call could not be reserved within its bound",
    )
}
