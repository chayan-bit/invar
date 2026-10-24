//! Private native host-array protocol v1 and its rollback guard (HA-1 §12,
//! `docs/host-array-arguments-ha1.md`). This is **not** part of the public
//! C ABI v1: `inv_arg_v1`, its tags and modes are unchanged, and nothing here
//! is exported to C callers. This first slice defines the record layout,
//! builds records only from live, validated Rust slices, and provides the
//! rollback guard. HA-1.2 adds the only native invocation,
//! [`HostArrayCall::invoke`], which consumes the guard.
//!
//! Supported host ABI: 64-bit pointers (`usize` = `u64`), C `uint32_t`,
//! `uint64_t` and object pointers with natural alignment, as on the Linux
//! and macOS targets this crate already requires (its public `InvArg` layout
//! asserts the same pointer width).

use std::marker::PhantomData;

/// Element tag: `int32_t`.
pub const ELEMENT_I32: u32 = 1;
/// Element tag: `uint32_t`.
pub const ELEMENT_U32: u32 = 2;
/// Element tag: IEEE binary32 (HA-1.3).
pub const ELEMENT_F32: u32 = 3;
/// Element tag: IEEE binary64 (HA-1.4).
pub const ELEMENT_F64: u32 = 4;
/// Element tag: one byte, exactly 0 or 1 (HA-1.4).
pub const ELEMENT_BOOL: u32 = 5;
/// Mode tag: the callee may only read the elements.
pub const MODE_READ: u32 = 1;
/// Mode tag: the callee may read and write the elements. Deliberately not 2,
/// the public `inv_arg_v1` take mode, so no public mode decodes as `mut`.
pub const MODE_MUT: u32 = 3;
/// Maximum array arguments per call (the export arity bound).
pub const MAX_ARRAYS: usize = 8;
/// Per-array element bound (language limit).
pub const MAX_ELEMENTS: usize = 1_000_000;
/// Live-element bound on the union of all extents (language limit).
pub const MAX_LIVE_ELEMENTS: usize = 2_000_000;
/// Rollback snapshot bound for a call whose widest element is at most four
/// bytes (HA-1.1 §6 `snapshot_bytes`); a call with an 8-byte array may
/// snapshot `MAX_LIVE_ELEMENTS * 8` bytes (HA-1.4).
pub const MAX_SNAPSHOT_BYTES: usize = 8_000_000;
const ELEMENT_BYTES: usize = 4;
const MAX_ELEMENT_BYTES: usize = 8;

/// Bytes per element of an element tag (alignment equals size).
const fn element_bytes(element: u32) -> usize {
    match element {
        ELEMENT_F64 => 8,
        ELEMENT_BOOL => 1,
        _ => 4,
    }
}

/// Private export-descriptor tag of an array parameter (HA-1.2). It is never
/// an `inv_arg_v1` tag (public tags are 1..=4): `0x100 | element << 4 | mode`.
pub const fn descriptor_tag(element: u32, mode: u32) -> u32 {
    0x100 | (element << 4) | mode
}

/// Decode a [`descriptor_tag`] into `(element, mode)`.
pub fn decode_descriptor_tag(tag: u32) -> Option<(u32, u32)> {
    let (element, mode) = ((tag >> 4) & 0xf, tag & 0xf);
    (tag & !0xff == 0x100
        && matches!(
            element,
            ELEMENT_I32 | ELEMENT_U32 | ELEMENT_F32 | ELEMENT_F64 | ELEMENT_BOOL
        )
        && matches!(mode, MODE_READ | MODE_MUT))
    .then_some((element, mode))
}

/// One private host-array record, C layout:
///
/// ```c
/// struct inv_private_host_array_v1 {
///     uint32_t element; /* offset 0: 1 = int32_t, 2 = uint32_t, 3 = float32,
///                          4 = float64, 5 = bool (one byte, 0 or 1) */
///     uint32_t mode;    /* offset 4: 1 = read, 3 = mut */
///     void    *data;    /* offset 8: first element; NULL iff len == 0 */
///     uint64_t len;     /* offset 16: element count, <= 1,000,000 */
/// };                    /* size 24, alignment 8 */
/// ```
///
/// Fields are private: a record can only be produced by
/// [`HostArrayCall::records`] from a validated, borrowed Rust slice. A record
/// is neither `Clone` nor `Copy`, so it exists only inside its guard and is
/// observable only through a [`HostArrayRecords`] view borrowed from that
/// live guard: no detached record can outlive the slices it describes.
///
/// No record can be built from an arbitrary address:
///
/// ```compile_fail,E0451
/// let record = invar_foreign::host_array::HostArrayRecordV1 {
///     element: 1, mode: 1, data: std::ptr::null_mut(), len: 0,
/// };
/// ```
///
/// and records cannot be copied out of the guard (the stale-record escape
/// demonstrated against the earlier `Copy` record):
///
/// ```compile_fail,E0599
/// use invar_foreign::host_array::{HostArray, HostArrayCall, HostArrayRecordV1};
/// let saved: Vec<HostArrayRecordV1>;
/// {
///     let mut xs = vec![1i32, 2];
///     let mut call = HostArrayCall::prepare(vec![HostArray::MutI32(&mut xs)]).unwrap();
///     saved = call.records().to_vec();
///     call.commit();
/// }
/// assert_eq!(saved.len(), 1);
/// ```
///
/// Stable rustdoc does not check the error codes above; the focused test
/// `compile_checks_pin_error_codes_and_positive_controls` does.
#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
pub struct HostArrayRecordV1 {
    element: u32,
    mode: u32,
    data: *mut u8,
    len: u64,
}

const _: () = {
    assert!(std::mem::size_of::<usize>() == 8);
    assert!(std::mem::size_of::<*mut u8>() == 8);
    assert!(std::mem::size_of::<HostArrayRecordV1>() == 24);
    assert!(std::mem::align_of::<HostArrayRecordV1>() == 8);
    assert!(std::mem::offset_of!(HostArrayRecordV1, element) == 0);
    assert!(std::mem::offset_of!(HostArrayRecordV1, mode) == 4);
    assert!(std::mem::offset_of!(HostArrayRecordV1, data) == 8);
    assert!(std::mem::offset_of!(HostArrayRecordV1, len) == 16);
    // Every admissible byte extent is representable: 1,000,000 * 8 bytes.
    assert!(MAX_ELEMENTS.checked_mul(MAX_ELEMENT_BYTES).is_some());
    // Snapshots cover pairwise-disjoint mut extents inside the live union.
    assert!(MAX_SNAPSHOT_BYTES == MAX_LIVE_ELEMENTS * ELEMENT_BYTES);
};

impl HostArrayRecordV1 {
    pub fn element(&self) -> u32 {
        self.element
    }
    pub fn mode(&self) -> u32 {
        self.mode
    }
    /// The address as an integer (never dereferenced by Rust code).
    pub fn address(&self) -> usize {
        self.data.addr()
    }
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// The records of one prepared call, borrowed from the live
/// [`HostArrayCall`] that built them (exclusively, for the view's lifetime).
///
/// The view is not `Clone` or `Copy`, exposes no slice, and is the only way
/// to observe records. Its raw pointer is crate-private: a future native
/// invocation must be a method of the guard that obtains this view itself,
/// never an API accepting detached records or record pointers.
#[must_use]
#[derive(Debug)]
pub struct HostArrayRecords<'c> {
    records: &'c [HostArrayRecordV1],
}

impl HostArrayRecords<'_> {
    pub fn len(&self) -> usize {
        self.records.len()
    }
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
    pub fn get(&self, index: usize) -> Option<&HostArrayRecordV1> {
        self.records.get(index)
    }
    /// Records in argument order. The iterator type is opaque, so it yields
    /// no slice of records.
    pub fn iter(&self) -> impl Iterator<Item = &HostArrayRecordV1> + '_ {
        self.records.iter()
    }
    /// First record, for the guard-bound [`HostArrayCall::invoke`] only.
    pub(crate) fn as_ptr(&self) -> *const HostArrayRecordV1 {
        self.records.as_ptr()
    }
}

/// One host array argument: a live borrowed slice and its mode.
#[derive(Debug)]
pub enum HostArray<'a> {
    ReadI32(&'a [i32]),
    ReadU32(&'a [u32]),
    MutI32(&'a mut [i32]),
    MutU32(&'a mut [u32]),
    ReadF32(&'a [f32]),
    MutF32(&'a mut [f32]),
    ReadF64(&'a [f64]),
    MutF64(&'a mut [f64]),
    ReadBool(&'a [bool]),
    MutBool(&'a mut [bool]),
}

impl HostArray<'_> {
    fn element(&self) -> u32 {
        match self {
            Self::ReadI32(_) | Self::MutI32(_) => ELEMENT_I32,
            Self::ReadU32(_) | Self::MutU32(_) => ELEMENT_U32,
            Self::ReadF32(_) | Self::MutF32(_) => ELEMENT_F32,
            Self::ReadF64(_) | Self::MutF64(_) => ELEMENT_F64,
            Self::ReadBool(_) | Self::MutBool(_) => ELEMENT_BOOL,
        }
    }
    fn mutable(&self) -> bool {
        matches!(
            self,
            Self::MutI32(_)
                | Self::MutU32(_)
                | Self::MutF32(_)
                | Self::MutF64(_)
                | Self::MutBool(_)
        )
    }
    fn address(&self) -> usize {
        match self {
            Self::ReadI32(v) => v.as_ptr().addr(),
            Self::ReadU32(v) => v.as_ptr().addr(),
            Self::MutI32(v) => v.as_ptr().addr(),
            Self::MutU32(v) => v.as_ptr().addr(),
            Self::ReadF32(v) => v.as_ptr().addr(),
            Self::MutF32(v) => v.as_ptr().addr(),
            Self::ReadF64(v) => v.as_ptr().addr(),
            Self::MutF64(v) => v.as_ptr().addr(),
            Self::ReadBool(v) => v.as_ptr().addr(),
            Self::MutBool(v) => v.as_ptr().addr(),
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::ReadI32(v) => v.len(),
            Self::ReadU32(v) => v.len(),
            Self::MutI32(v) => v.len(),
            Self::MutU32(v) => v.len(),
            Self::ReadF32(v) => v.len(),
            Self::MutF32(v) => v.len(),
            Self::ReadF64(v) => v.len(),
            Self::MutF64(v) => v.len(),
            Self::ReadBool(v) => v.len(),
            Self::MutBool(v) => v.len(),
        }
    }
}

/// Why a host-array call could not be prepared. Every refusal happens before
/// any record exists and before any caller element is touched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostArrayError {
    /// More than [`MAX_ARRAYS`] arrays.
    Count,
    /// An array longer than [`MAX_ELEMENTS`].
    Length,
    /// A byte extent that overflows the address space.
    Extent,
    /// Overlapping extents involving a mut array or two element types.
    Alias,
    /// The union of extents exceeds [`MAX_LIVE_ELEMENTS`].
    Live,
    /// Snapshots exceed [`MAX_SNAPSHOT_BYTES`] or could not be reserved.
    Snapshot,
}

/// Integer metadata of one array, for validation (never dereferenced).
#[derive(Clone, Copy, Debug)]
struct Extent {
    element: u32,
    mutable: bool,
    address: usize,
    length: usize,
}

/// Validate counts, lengths, extents, aliasing, the live union and the
/// snapshot total, in that order. Integer metadata only.
fn validate(extents: &[Extent]) -> Result<usize, HostArrayError> {
    if extents.len() > MAX_ARRAYS {
        return Err(HostArrayError::Count);
    }
    if extents.iter().any(|e| e.length > MAX_ELEMENTS) {
        return Err(HostArrayError::Length);
    }
    let ranges = extents
        .iter()
        .map(|e| {
            e.length
                .checked_mul(element_bytes(e.element))
                .and_then(|bytes| e.address.checked_add(bytes))
                .map(|end| (e.address, end))
                .ok_or(HostArrayError::Extent)
        })
        .collect::<Result<Vec<_>, _>>()?;
    for i in 0..extents.len() {
        for j in 0..i {
            let (a, b) = (ranges[i], ranges[j]);
            let overlap = a.0 < a.1 && b.0 < b.1 && a.0 < b.1 && b.0 < a.1;
            if overlap
                && (extents[i].mutable
                    || extents[j].mutable
                    || extents[i].element != extents[j].element)
            {
                return Err(HostArrayError::Alias);
            }
        }
    }
    // Live elements of the union (HA-1.4). Overlapping extents share one
    // element type, so a merged range has one element size; touching extents
    // of different sizes stay separate.
    let mut spans: Vec<(usize, usize, usize)> = ranges
        .iter()
        .zip(extents)
        .filter(|(r, _)| r.0 < r.1)
        .map(|(r, e)| (r.0, r.1, element_bytes(e.element)))
        .collect();
    spans.sort_unstable();
    let mut live = 0usize;
    let mut current: Option<(usize, usize, usize)> = None;
    for (start, end, size) in spans {
        current = match current {
            Some((s, e, z)) if start < e || (start == e && size == z) => Some((s, e.max(end), z)),
            Some((s, e, z)) => {
                live = live.checked_add((e - s) / z).ok_or(HostArrayError::Live)?;
                Some((start, end, size))
            }
            None => Some((start, end, size)),
        };
    }
    if let Some((s, e, z)) = current {
        live = live.checked_add((e - s) / z).ok_or(HostArrayError::Live)?;
    }
    if live > MAX_LIVE_ELEMENTS {
        return Err(HostArrayError::Live);
    }
    // 4-byte calls keep the 8 MB bound; a call holding an 8-byte array may
    // snapshot twice that, the same live-element bound in bytes.
    let widest = extents
        .iter()
        .map(|e| element_bytes(e.element))
        .max()
        .unwrap_or(ELEMENT_BYTES)
        .max(ELEMENT_BYTES);
    let snapshot_limit = MAX_SNAPSHOT_BYTES / ELEMENT_BYTES * widest;
    let snapshot_bytes = extents
        .iter()
        .zip(&ranges)
        .filter(|(e, _)| e.mutable)
        .try_fold(0usize, |total, (_, r)| total.checked_add(r.1 - r.0))
        .filter(|&total| total <= snapshot_limit)
        .ok_or(HostArrayError::Snapshot)?;
    Ok(snapshot_bytes)
}

/// Pre-call copy of one mut array's exact contents.
enum Snapshot {
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<u32>),
    F64(Vec<u64>),
    Bool(Vec<bool>),
}

/// A prepared host-array call: validated borrowed slices, snapshots of every
/// mut array reserved and taken before entry, and record capacity.
///
/// Rollback contract:
/// - [`commit`](Self::commit) keeps the callee's writes. Call it only after
///   the foreign call returned normally **and** every fallible host-result
///   preparation succeeded.
/// - Any other way out (`restore`, `finish(Err)`, an early return or a panic
///   unwinding through the owner) restores every mut array to its exact
///   pre-call bits. Restore allocates nothing and cannot panic.
/// - No guarantee across process abort or kill (HA-1 §8): no Rust code runs.
/// - Rollback is `Drop`-based, and leaking is safe in Rust: a guard passed to
///   `std::mem::forget` (or leaked another way) is never restored, so the
///   callee's writes stay without a commit. A native invocation must
///   therefore be a method of this guard that owns it for the whole call and
///   decides commit or restore itself; the guard must never be handed to
///   caller code between `prepare` and that decision.
/// - Today `prepare`, `records` and `commit` are public and the guard is held
///   by caller code, so the requirement above is **documentation only**. It
///   must be enforced when a native invocation lands: the invocation consumes
///   the guard (`fn invoke(self, ...)`) or constructs it internally, and no
///   API accepts a borrowed guard, a record view or record pointers.
///
/// The call is thread-confined:
///
/// ```compile_fail,E0277
/// fn send<T: Send>() {}
/// send::<invar_foreign::host_array::HostArrayCall<'static>>();
/// ```
///
/// a caller slice cannot be touched while records exist:
///
/// ```compile_fail,E0506
/// use invar_foreign::host_array::{HostArray, HostArrayCall};
/// let mut xs = [1i32];
/// let mut call = HostArrayCall::prepare(vec![HostArray::MutI32(&mut xs)]).unwrap();
/// let records = call.records();
/// xs[0] = 2;
/// assert_eq!(records.len(), 1);
/// ```
///
/// and the guard cannot be committed, restored or dropped while a record
/// view borrowed from it is live:
///
/// ```compile_fail,E0505
/// use invar_foreign::host_array::{HostArray, HostArrayCall};
/// let mut xs = [1i32];
/// let mut call = HostArrayCall::prepare(vec![HostArray::MutI32(&mut xs)]).unwrap();
/// let records = call.records();
/// call.commit();
/// assert_eq!(records.len(), 1);
/// ```
///
/// A future native invocation must be a method of this guard that builds the
/// [`HostArrayRecords`] view itself; records are not constructible,
/// cloneable or copyable, so no detached record can reach it.
pub struct HostArrayCall<'a> {
    arrays: Vec<HostArray<'a>>,
    snapshots: Vec<Snapshot>,
    records: Vec<HostArrayRecordV1>,
    armed: bool,
    _thread: PhantomData<std::rc::Rc<()>>,
}

impl<'a> HostArrayCall<'a> {
    /// Validate and take snapshots. Nothing is written to any caller slice.
    pub fn prepare(arrays: Vec<HostArray<'a>>) -> Result<Self, HostArrayError> {
        Self::prepare_with(arrays, None)
    }

    /// `fail_at`: private test seam making the `n`th reservation fail.
    fn prepare_with(
        arrays: Vec<HostArray<'a>>,
        fail_at: Option<usize>,
    ) -> Result<Self, HostArrayError> {
        let extents: Vec<Extent> = arrays
            .iter()
            .map(|a| Extent {
                element: a.element(),
                mutable: a.mutable(),
                address: a.address(),
                length: a.len(),
            })
            .collect();
        validate(&extents)?;
        // Injection seam: the `fail_at`th reservation (0-based) fails.
        let mut reservation = 0usize;
        let mut reserve = || -> Result<(), HostArrayError> {
            let failed = fail_at == Some(reservation);
            reservation += 1;
            if failed {
                return Err(HostArrayError::Snapshot);
            }
            Ok(())
        };
        let mut records = Vec::new();
        reserve()?;
        records
            .try_reserve_exact(arrays.len())
            .map_err(|_| HostArrayError::Snapshot)?;
        let mutable = arrays.iter().filter(|a| a.mutable()).count();
        let mut snapshots = Vec::new();
        reserve()?;
        snapshots
            .try_reserve_exact(mutable)
            .map_err(|_| HostArrayError::Snapshot)?;
        for array in &arrays {
            match array {
                HostArray::MutI32(v) => {
                    reserve()?;
                    let mut copy = Vec::new();
                    copy.try_reserve_exact(v.len())
                        .map_err(|_| HostArrayError::Snapshot)?;
                    copy.extend_from_slice(v);
                    snapshots.push(Snapshot::I32(copy));
                }
                HostArray::MutU32(v) => {
                    reserve()?;
                    let mut copy = Vec::new();
                    copy.try_reserve_exact(v.len())
                        .map_err(|_| HostArrayError::Snapshot)?;
                    copy.extend_from_slice(v);
                    snapshots.push(Snapshot::U32(copy));
                }
                HostArray::MutF32(v) => {
                    reserve()?;
                    let mut copy = Vec::new();
                    copy.try_reserve_exact(v.len())
                        .map_err(|_| HostArrayError::Snapshot)?;
                    copy.extend(v.iter().map(|value| value.to_bits()));
                    snapshots.push(Snapshot::F32(copy));
                }
                HostArray::MutF64(v) => {
                    reserve()?;
                    let mut copy = Vec::new();
                    copy.try_reserve_exact(v.len())
                        .map_err(|_| HostArrayError::Snapshot)?;
                    copy.extend(v.iter().map(|value| value.to_bits()));
                    snapshots.push(Snapshot::F64(copy));
                }
                HostArray::MutBool(v) => {
                    reserve()?;
                    let mut copy = Vec::new();
                    copy.try_reserve_exact(v.len())
                        .map_err(|_| HostArrayError::Snapshot)?;
                    copy.extend_from_slice(v);
                    snapshots.push(Snapshot::Bool(copy));
                }
                HostArray::ReadI32(_)
                | HostArray::ReadU32(_)
                | HostArray::ReadF32(_)
                | HostArray::ReadF64(_)
                | HostArray::ReadBool(_) => {}
            }
        }
        Ok(Self {
            arrays,
            snapshots,
            records,
            armed: true,
            _thread: PhantomData,
        })
    }

    /// Records for exactly one foreign call, in argument order, derived from
    /// reborrows of the validated slices. The returned view keeps `self`
    /// mutably borrowed, so Rust cannot touch the slices (nor commit or
    /// restore) until the call is over, and no record can be copied out of
    /// it. Pushes stay within the capacity reserved by `prepare`.
    pub fn records(&mut self) -> HostArrayRecords<'_> {
        self.records.clear();
        for array in &mut self.arrays {
            let (element, mode, data, len) = match array {
                HostArray::ReadI32(v) => (
                    ELEMENT_I32,
                    MODE_READ,
                    v.as_ptr().cast::<u8>().cast_mut(),
                    v.len(),
                ),
                HostArray::ReadU32(v) => (
                    ELEMENT_U32,
                    MODE_READ,
                    v.as_ptr().cast::<u8>().cast_mut(),
                    v.len(),
                ),
                HostArray::MutI32(v) => {
                    (ELEMENT_I32, MODE_MUT, v.as_mut_ptr().cast::<u8>(), v.len())
                }
                HostArray::MutU32(v) => {
                    (ELEMENT_U32, MODE_MUT, v.as_mut_ptr().cast::<u8>(), v.len())
                }
                HostArray::ReadF32(v) => (
                    ELEMENT_F32,
                    MODE_READ,
                    v.as_ptr().cast::<u8>().cast_mut(),
                    v.len(),
                ),
                HostArray::MutF32(v) => {
                    (ELEMENT_F32, MODE_MUT, v.as_mut_ptr().cast::<u8>(), v.len())
                }
                HostArray::ReadF64(v) => (
                    ELEMENT_F64,
                    MODE_READ,
                    v.as_ptr().cast::<u8>().cast_mut(),
                    v.len(),
                ),
                HostArray::MutF64(v) => {
                    (ELEMENT_F64, MODE_MUT, v.as_mut_ptr().cast::<u8>(), v.len())
                }
                HostArray::ReadBool(v) => (
                    ELEMENT_BOOL,
                    MODE_READ,
                    v.as_ptr().cast::<u8>().cast_mut(),
                    v.len(),
                ),
                HostArray::MutBool(v) => {
                    (ELEMENT_BOOL, MODE_MUT, v.as_mut_ptr().cast::<u8>(), v.len())
                }
            };
            self.records.push(HostArrayRecordV1 {
                element,
                mode,
                data: if len == 0 { std::ptr::null_mut() } else { data },
                // usize -> u64 is lossless on the supported 64-bit hosts.
                len: len as u64,
            });
        }
        HostArrayRecords {
            records: &self.records,
        }
    }

    /// Keep every write. Only after a normal return and successful
    /// fallible host-result preparation.
    pub fn commit(mut self) {
        self.armed = false;
    }

    /// Restore every mut array to its pre-call bits.
    pub fn restore(mut self) {
        self.restore_now();
        self.armed = false;
    }

    /// Commit on `Ok`, restore on `Err`, passing the outcome through.
    pub fn finish<T, E>(self, outcome: Result<T, E>) -> Result<T, E> {
        match outcome {
            Ok(value) => {
                self.commit();
                Ok(value)
            }
            Err(error) => {
                self.restore();
                Err(error)
            }
        }
    }

    /// Element-wise copy of each snapshot into its own slice. Snapshots were
    /// taken from these same slices, whose lengths cannot change; `zip` makes
    /// the loop panic-free regardless. No allocation.
    fn restore_now(&mut self) {
        // Snapshots exist only for mut arrays, in argument order.
        let mut snapshots = self.snapshots.iter();
        for array in &mut self.arrays {
            match array {
                HostArray::MutI32(target) => {
                    if let Some(Snapshot::I32(source)) = snapshots.next() {
                        for (slot, value) in target.iter_mut().zip(source) {
                            *slot = *value;
                        }
                    }
                }
                HostArray::MutU32(target) => {
                    if let Some(Snapshot::U32(source)) = snapshots.next() {
                        for (slot, value) in target.iter_mut().zip(source) {
                            *slot = *value;
                        }
                    }
                }
                HostArray::MutF32(target) => {
                    if let Some(Snapshot::F32(source)) = snapshots.next() {
                        for (slot, bits) in target.iter_mut().zip(source) {
                            *slot = f32::from_bits(*bits);
                        }
                    }
                }
                HostArray::MutF64(target) => {
                    if let Some(Snapshot::F64(source)) = snapshots.next() {
                        for (slot, bits) in target.iter_mut().zip(source) {
                            *slot = f64::from_bits(*bits);
                        }
                    }
                }
                HostArray::MutBool(target) => {
                    if let Some(Snapshot::Bool(source)) = snapshots.next() {
                        for (slot, value) in target.iter_mut().zip(source) {
                            *slot = *value;
                        }
                    }
                }
                HostArray::ReadI32(_)
                | HostArray::ReadU32(_)
                | HostArray::ReadF32(_)
                | HostArray::ReadF64(_)
                | HostArray::ReadBool(_) => {}
            }
        }
    }

    /// Test seam: safe access to the borrowed slices, standing in for a
    /// callee's writes in pure-Rust guard tests.
    #[cfg(test)]
    fn arrays_mut(&mut self) -> &mut [HostArray<'a>] {
        &mut self.arrays
    }
}

impl Drop for HostArrayCall<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.restore_now();
        }
    }
}

// The guard-consuming native invocation needs the real crate's loader; the
// isolated stand-in crate of the compile-check harness builds without it.
#[cfg(not(host_array_isolated))]
mod invoke;
#[cfg(not(host_array_isolated))]
pub use invoke::HostCallError;

#[cfg(test)]
mod tests;
