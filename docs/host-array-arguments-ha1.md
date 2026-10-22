# HA-1: host array arguments (revision HA-1.3)

P0b limit update (2026-10-15): the language per-array limit is now 2^24 and the live-element ceiling is 2^28, shared by source, MIR, native allocation and the net oracle. Historical 1,000,000/2,000,000 figures below describe the original HA-1 acceptance. GNET ABI13 adds a copying device entry with retained Rust leases and successful-return publication after confirmed completion; it does not establish the native/Python zero-copy requirements. The private HA-1 native host-array protocol retains its separate tool bounds: 1,000,000 elements per array, 2,000,000 live elements and 8,000,000 rollback bytes; larger host calls are typed refusals. These bounds do not cap native source-owned allocations. See [GNET entry ABI13](gnet-entry-v13.md).

**Status: frozen.** The maintainers approved the pending decisions on 2026-10-04
and delegated option choices to engineering judgement. HA-1 amends
L1-06 (`docs/issues/next/L1-06-python-binding.md`) at the host boundary; it does
not change the language, MIR, Value layout or any ABI.

**HA-1.1 (2026-10-05):**
- The read slice `ec0f7d8` was approved by the third review with no blockers.
- The maintainers allocated **A1** (an additive public Rust API in
  `invar::eval::host_read`: `HostArgument`, `HostMutArray`,
  `observe_host_call`) and **A2** (this revision) for stage 1 only: `mut` on
  the copying reference engine.
- HA-1.1 changes, relative to HA-1:
  - the staging bound (§6) becomes 40,000,000 bytes;
  - typed `take` and commit failures (§2, §7, §9);
  - `Result<scalar,scalar>` results at the host-call entry point (§7.1);
  - the explicit native status (§10).
- The 1,000,000-element per-array bound and the 2,000,000-element language
  live limit are unchanged.

**HA-1.2 (2026-10-13), decided 2026-10-13:**
- Option A: the native plan is approved as the HA-1.2 native array-parameter interface, and its remaining slices belong to the L1-06 workstream.
- Native-serial admits `read` and `mut` `Array<i32>`/`Array<u32>` parameters of selected exports, caller-provided outputs through `mut`, and `i32`/`u32` results.
- Zero-copy is claimed only for that native path and is proven by the same-address test (§10).
- `f32` arrays and returned arrays stay on the copying reference path, are reported as copying, and are never counted as zero-copy.
- The public C ABI v1 is unchanged, and §12 is unchanged: the private record keeps element values 1 (`i32`) and 2 (`u32`) only.
- Native `f32` arrays are an explicit open item, to follow M4-01.

**HA-1.3 (2026-10-14), decided 2026-10-14:**
- Extend the private section 12 record with element value 3 = `f32` (IEEE binary32), alongside 1 = `i32` and 2 = `u32`.
- Modes remain 1 = `read` and 3 = `mut`, with unchanged size, offsets, alignment, limits, alias rules and rollback obligations.
- Native-serial admits `read`/`mut Array<f32>` at caller addresses under `strict-f32-v1`, as defined by M4-01's `src/numeric_contract.rs` and `docs/m4-source.md`.
- Copies, stores and rollback preserve every bit, including NaN payloads and signed zero; arithmetic follows the existing policy (NaN payload/sign are unobservable arithmetic outcomes), with gradual underflow and no FTZ, reassociation or implicit contraction.
- The public C ABI v1 is unchanged; element 3 is private and is never a public scalar or array tag.
- Returned arrays remain on the explicitly copying reference path; `take` remains refused.

**HA-1.4 (2026-10-21), decided under #53; reference path only:**
- The admitted element types gain `f64` (IEEE binary64, 8 bytes) and `Bool` (NumPy `bool_`, 1 byte).
  Scalar `f64` is admitted at both host-call entry points.
- `Array<Bool>` maps onto NumPy `bool_`, one byte per element.
  A byte other than 0 or 1 is refused before any step (`PY_ARRAY_BOOL_VALUE`), on `read` and `mut` arrays alike.
  The binding proves this on a `uint8` view of the same memory before any `&[bool]` exists, because an invalid Rust `bool` is undefined behaviour.
  Results come back as new `bool_` arrays holding only 0 and 1.
- Extent arithmetic, alignment (§3), the live-element union (§5) and staging (§6) use each buffer's own element size instead of 4.
  An `f64` buffer must be 8-byte aligned, a `Bool` buffer has no alignment, and the union counts elements of each merged range's own size.
  Extents that only touch merge only when their element sizes agree.
  The alias rules (§4) are unchanged: any overlap of two element types is refused.
- The byte bounds of the binding and the rollback guard (8,000,000 bridge bytes, 40,000,000 temporary bytes, 8,000,000 snapshot bytes) are stated for 4-byte elements.
  A call that holds an 8-byte array doubles them, and every call without one keeps them as they were.
  The element and live-element counts (1,000,000 and 2,000,000) are unchanged.
- In Python an `f64` argument is an exact built-in `float`; an `int`, a `float` subclass (including `numpy.float64`) and anything else is `TYPE_MISMATCH`.
  An `f64` result is a `float`, which carries every bit.
  `Array<f64>` is a native-endian NumPy `float64` array validated exactly like the other numeric arrays, and its elements keep every bit, NaN payloads and signed zero included.
- Native-serial exports with these types are not part of HA-1.4 (`NATIVE_ABI`), and the §12 record is unchanged.

**Implementation status:**

| Area | Status |
|---|---|
| Read reference slice: `observe_host_read_call` | implemented (§3–§6, §10); behaviour unchanged by HA-1.1 |
| Stage 1, reference `mut`: `observe_host_call` | implemented (§7, §7.1, §9) |
| Rollback snapshots for native `mut` (§6, §7) | implemented (HA-1.2): `HostArrayCall::invoke` consumes the guard |
| Python/NumPy binding (§8) | implemented: `call_reference` (copying) and, for selected exports, `call` (HA-1.2) |
| Native engines with array arguments, `read` or `mut` (§10) | implemented for native-serial exports (HA-1.2 subset, §13); HA-1.3 adds `f32` arrays; returned arrays stay copying-reference only |
| Rust reference `take` (`db558b8`) | provisional, unexposed (§9) |

## Sources

| Source | What it says |
|---|---|
| L1-06:3 | "without copying arrays" |
| L1-06:13 | arrays "pass zero-copy through the buffer protocol"; validation before any call |
| L1-06:19 | a call never returns partial output |
| L1-06:20 | a failed validation raises before any Invar code runs |
| L1-06:21 | ownership transfer follows L1-04 rules |
| L1-06:22 | the GIL may be released only around validated, pinned calls |
| L1-06:23 | `unsafe` only in `crates/invar-python` |
| L1-06:31 | same buffer address on both sides |
| L1-06:37 | negative cases: aliasing, freed or resized buffers |
| `spec/bootstrap.md:49` | `read`/`mut`/`take` parameter modes |
| `spec/bootstrap.md:53` | read loans may coexist; a mutable loan excludes all other access |
| `spec/bootstrap.md:55–57` | loans last for the outer call; incompatible aliasing is rejected at the caller |
| `spec/bootstrap.md:81` | traps perform no language-level cleanup |
| `spec/bootstrap.md:117` | an `Err` is an ordinary value, distinct from traps, limits and tool failures |
| `spec/bootstrap.md:121` | `View`/`MutView` are non-owning views (relevant to §5.3) |
| `spec/bootstrap.md:143` | foreign boundary |
| `docs/native-toolchain-v1.md:59`; `src/eval.rs:20,23` | array length ≤ 1,000,000; live array elements ≤ 2,000,000 |
| `docs/scalar-exports-v1.md:39` | Python bindings retain the GIL throughout the call |
| `docs/rust-engineering.md:38` | fallible reservation where allocation failure is part of the contract |
| `crates/invar-python/src/validation.rs:74–75, 114, 120` | only NumPy-owned memory is admitted; half-open extents; empty buffers never alias; read/read overlap permitted |
| `crates/invar-python/src/lib.rs:909` | capabilities report `gil_released: false`, `external_array_owners: false` |

## Maintainer decisions

| # | Decision | Where |
|---|---|---|
| 1 | read/read aliases are allowed; the union of checked physical extents counts **once** toward the language live-element budget, not per argument; the 1,000,000 per-array bound stays | §4, §5 |
| 2 | native `mut` executes at the original address with a fallibly pre-reserved rollback snapshot; every non-normal outcome restores exact bytes before returning or raising; `Result::Err` is a normal value | §7 |
| 3 | rollback and reference-engine copies are explicit host staging outside the language allocation cap, with a separate bounded fallible byte budget; HA-1 is **not literally copy-free** | §6 |
| 4 | hold the GIL for array calls and pin exports; refuse every overlap involving `mut`; callers must prevent concurrent native access outside the GIL (the GIL cannot make that safe); no guarantees across process kill or abort | §4, §8 |
| 5 | NumPy-owned `take` is refused before execution; Rust reference `take` stays unexposed pending its own contract | §9 |
| 6 | the same-address proof applies to native engines only; reference engines are explicitly copying semantic oracles | §10 |
| — | no hidden fallback and no partial publication | §2, §7 |

## 1. Definitions

- **Host buffer:** a contiguous, element-aligned range supplied for one array
  parameter. Admitted element types are `i32`, `u32` and `f32`, 4 bytes each; HA-1.4 adds `f64` (8) and `Bool` (1).
- **Extent:** the half-open byte range `[base, base + len·size)`, `size` being the element size (4 unless HA-1.4 says otherwise).
  - `base` is the buffer address as an integer: `slice.as_ptr().addr()` in
    Rust, NumPy `data` metadata in the binding.
  - Extents are computed **without dereferencing** and without `unsafe` in the
    core.
- **Exact alias:** two non-empty extents with the same element type, base and
  length.
- **Partial overlap:** two non-empty extents of the same element type that
  intersect without being identical.
- **Cross-type overlap:** two non-empty extents of different element types
  that intersect.
- **Empty extents never alias**, including an empty extent strictly inside
  another buffer.
- **Engine classes:**
  - *reference*: the safe-Rust semantic oracle; it copies.
  - *native*: loaded generated code that executes at caller addresses.

## 2. Admission order

Every check completes before any source step. A refusal reports zero steps,
records no cleanup and leaves every host buffer's bytes unchanged. There is no
fallback to another engine and no partial result.

1. fuel, function, signature, `take`, argument count, exact types:
   - `observe_host_read_call` admits `read` arrays only; any `mut` or `take`
     parameter or `Result` result → `NATIVE_ABI` (unchanged);
   - `observe_host_call` admits `read` and `mut` arrays and scalar `Result`
     results (§7.1). A shape outside that → `NATIVE_ABI`. Then any `take`
     parameter → `HOST_TAKE_UNAVAILABLE`;
2. per-array bound (1,000,000 elements) → `ARRAY_SIZE_LIMIT`;
3. extent arithmetic (§3) → `HOST_ARRAY_EXTENT`;
4. alias rules (§4) → `HOST_ARRAY_ALIAS`;
5. live-element admission of the extent union (§5) → `ARRAY_SIZE_LIMIT`;
6. staging bound and fallible reservation (§6) → `HOST_STAGING_LIMIT`.

The binding (later slice) maps these codes as follows:
- `HOST_ARRAY_EXTENT` → `PY_BUFFER_LIMIT`;
- `HOST_ARRAY_ALIAS` → `PY_BUFFER_ALIAS`;
- `HOST_STAGING_LIMIT` → a new `PY_STAGING_LIMIT`, added with AI-01 schema
  coverage in that slice.

## 3. Extent arithmetic

- `end = base.checked_add(len.checked_mul(4)?)?`. Overflow, including an end
  equal to `usize::MAX + 1`, is refused.
- A base that is not a multiple of 4 is refused. It is impossible for safe
  Rust slices and NumPy-aligned arrays, so this check fails safe.
- Overlap tests compare integers only. No element reference is formed, and no
  extent is reinterpreted as another element type.

## 4. Alias rules

| Pair | Rule |
|---|---|
| read / read, exact alias | admitted; the reference engine shares **one** private owner, exactly like an in-source call passing one owner to both parameters (`spec/bootstrap.md:53`) |
| read / read, partial overlap (same type) | admitted; union accounting (§5); see §5.3 |
| any cross-type overlap | **refused** (`HOST_ARRAY_ALIAS`): one memory region would be two element types, which the Value model cannot represent and which would be a reinterpretation |
| any overlap involving `mut` (with `mut`, `read`, an output or `take`), including exact aliases | **refused** (`spec/bootstrap.md:53, 57`; L1-06:37) |

From NumPy only owning arrays are admitted, so a partial overlap is
impossible through Python and any alias is exact. Partial overlaps arise only
from the Rust slice API.

## 5. Live-element accounting

### 5.1 Rule
- **Host contribution:** the number of elements in the union of all
  non-empty extents (sorted and merged, union bytes ÷ 4).
- It is admitted against the 2,000,000 live-element budget before the first
  step, and stays live for the whole call (read loans are never released by
  the callee). The per-array bound still applies to each argument.
- Source allocations during the call add to the budget as usual.

### 5.2 Exact aliases: faithful in-source comparator
A host call passing buffer `X` to several `read` parameters must equal the
in-source call passing **one** owner to those parameters. "Equal" means:
- the same result;
- the same callee cleanup;
- the same step delta against a same-signature stub of the same outcome class.

Example: `grow2(X, X)` with `|X| = 1,000,000` returns `Ok(2000001)`. Before
HA-1 it was refused with `ARRAY_SIZE_LIMIT`.

### 5.3 Partial overlaps: plan without shared layout changes
- **No in-source counterpart.** The source language has no partially
  overlapping owners (`spec/bootstrap.md:49–57`). Union accounting is therefore
  an HA-1 *definition*, not an equivalence.
- **Reference engine (implemented):**
  - one private owner per argument, holding a copy of that argument's
    elements;
  - this is sound because `read` parameters cannot write, and the reference
    evaluator never observes owner identity (`src/eval.rs` has no identity
    comparison);
  - below the live limit, the call equals an in-source call with separate
    owners holding the same contents (result, cleanup, step delta);
  - at the limit, the union rule decides, and is tested directly (two 1M
    slices with a 1.6M union admit a further temporary).
- **Native engines (later):** execute at the caller's addresses; overlapping
  read-only pointers need no copy.
- **No `Value` or IR expansion is required or proposed.**
  - Shared-storage semantics are only needed if a slice ever has to *execute*
    over one storage region through two differently bounded owners. HA-1
    rules that out: every overlap involving `mut` is refused (§4).
  - If a later amendment admitted it (for example a non-copying reference
    oracle), the minimal interface would be a crate-private host-only window,
    `{ storage: Rc<RefCell<Vec<T>>>, offset, len }`, read through the existing
    checked index paths of `Value::Array*`, behind its own approved
    contract.
  - Mapping overlaps to language views (`spec/bootstrap.md:121`) is not
    possible without changing signatures, because a `read Array<T>` parameter
    is not a `View<T>`.

## 6. Host staging (outside the language budget)

- **What counts as staging:** reference-engine copies (§10) and `mut` rollback
  snapshots (§7). They are host staging, not Invar values, and never count
  toward the live-element budget.
- **Budget:** each call has a separate staging byte bound, checked before any
  step. Every staging buffer is reserved fallibly (`try_reserve_exact`), and
  the bound is enforced again where copies are reserved.
- **Failure:** exceeding the bound, or a failed reservation, is a typed
  `HOST_STAGING_LIMIT` refusal with zero steps and unchanged buffers.
- **Bound (HA-1.1):** `STAGING_LIMIT_BYTES = 40,000,000` bytes. It covers
  both terms below, for any engine.

  | Term | Content | Bound |
  |---|---|---|
  | `copy_bytes` | each exact-alias group once; a `mut` buffer is always its own group | ≤ `MAX_PARAMETERS × 1,000,000 × 4 = 32,000,000`. Partial `read` overlaps may copy more than the union, so the per-array limit, not the live limit, bounds this term |
  | `snapshot_bytes` | future native rollback snapshots: Σ bytes of `mut` buffers | ≤ `2,000,000 × 4 = 8,000,000`. Every overlap with `mut` is refused, so `mut` extents are pairwise disjoint and inside the extent union, already admitted against the 2,000,000-element live limit |

- **Engines:** the reference engine needs only `copy_bytes`; it never writes a
  caller buffer before commit, so it takes no snapshot. A native engine will
  need only `snapshot_bytes`. The bound is checked in admission and again at
  each reservation.
- **Python reference entry (HA-1.1 stage 2):** `call_reference` also holds
  host bridges, one exact-bits copy per connected union of `float32` read
  extents and one per `mut` array, and checks on the original metadata, before
  any reservation, that the bridges total at most 8,000,000 bytes (they are
  pairwise disjoint inside the admitted 2,000,000-element live union) and
  that bridges plus the core's reference copies (at most 32,000,000 bytes)
  total at most 40,000,000 bytes, refusing otherwise with a zero-step
  `PY_STAGING_LIMIT`, with the 1,000,000-element per-array and
  2,000,000-element live limits unchanged and with this bound applying only
  to the copying reference path, not to a future native engine, which still
  needs only `snapshot_bytes`.
- The read slice's earlier 32,000,000 is superseded; every call it admitted is
  still admitted.
- **Disclosure:** native execution at the caller's address satisfies L1-06:13
  and :31. The rollback snapshot (§7) copies `mut` contents, so HA-1 is **not
  literally copy-free** in the sense of L1-06:3. The maintainers accepted this.

## 7. `mut`

- **Admission:** no overlap of any kind with another argument (§4). A
  zero-step `HOST_ARRAY_ALIAS` refusal leaves every buffer byte-identical.
  Through Python, the buffer must also be writable and NumPy-owned (§8).
- **Native execution:** happens at the original address.
- **Snapshot:** before the first step, each `mut` buffer's exact bytes are
  copied into a fallibly pre-reserved snapshot within the staging budget (§6).
- **Outcomes:**
  - **Normal return**, including returning `Result::Err` (an ordinary value,
    `spec/bootstrap.md:117`): the buffer holds the final contents and the
    result is returned.
  - **Any non-normal outcome:** trap, fuel exhaustion, tool limit, internal
    error, or a Rust panic reaching the boundary. The buffer is restored to
    its exact pre-call bytes (including NaN payloads and `-0.0`) *before*
    returning or raising. No partial result object exists (L1-06:19).
- **Restore:** runs from a drop guard, is allocation-free and cannot fail: a
  byte copy into pinned, already-validated memory.
- **Reference engine (stage 1, implemented):**
  - **Copy-in:** copies in (§10) within the staging bound.
  - **Evaluation:** a `mut` argument binds like an in-source `mut` loan of a
    local owner. A `mut` parameter cannot be reassigned, so its length never
    changes.
  - **Commit:** only after a **normal** return, and only after the read
    post-check:
    1. *Precheck* every `mut` owner (same element type, no live borrow, length
       equal to the caller slice) and pair it with its caller slice. The
       pairs are pushed into capacity reserved before evaluation. A failed
       precheck writes nothing and yields `INTERNAL_HOST_COMMIT`, with the
       steps and cleanup of the completed evaluation.
    2. *Write back* each pair with `copy_from_slice`. Its only panic condition
       (unequal lengths) was excluded in step 1, and the loop allocates
       nothing and acquires no borrow, so it is infallible.
  - **Non-normal outcomes and refusals:** the caller buffers are never
    written: trap, fuel exhaustion, tool limit, internal error, any refusal,
    a failed reservation, and a panic caught by the caller before the
    write-back.
  - **Not claimed:** unwinding cannot be interrupted inside step 2 except by
    memory corruption, but this is not claimed as a hardware guarantee.
    Process abort or kill: no guarantee (§8).
  - **Cleanup:** writeback is a host operation; it never appears in the
    cleanup trace.

### 7.1 Results at the host-call boundary (HA-1.1)

- **Admitted results:** `observe_host_call` admits the read slice's scalar and
  array results, plus `Result<A,B>` with `A` and `B` each `i32`, `u32` or
  `Bool`, **not nested**. A nested `Result` → `NATIVE_ABI`.
- **Encoding:** the result is the evaluator's exact value, `Value::Variant`:
  tag 0 is `Ok`, tag 1 is `Err`, and the payload is the scalar.
- **Commit:** a returned `Result::Err` is an ordinary value
  (`spec/bootstrap.md:117`). It is a normal return and commits `mut`
  buffers.

## 8. GIL, pinning, concurrency, termination (binding; not implemented here)

- **GIL:** held for the whole of every call with array arguments
  (`docs/scalar-exports-v1.md:39`). Capabilities keep `gil_released: false`.
- **Pinning:** from validation through commit or restore, the binding holds a
  strong reference and a numpy (rust-numpy 0.29) dynamic borrow guard for every
  array, not a PEP 3118 buffer export; with the GIL held and no Python code
  running during the call, no array can be resized, reallocated or freed.
  Freed-or-resized cases (L1-06:37) cannot arise through Python during the call.
- **Caller precondition:** the GIL stops Python code only. It **cannot**
  prevent native threads that released the GIL, or external native code, from
  touching the memory. The caller must ensure that during the call:
  - no native code reads or writes a `mut` buffer;
  - no native code writes a `read` buffer.

  Violations are outside Invar's soundness claims, as for foreign code
  (`spec/bootstrap.md:143`).
- **Admitted memory:** only NumPy-owned, process-private memory
  (`validation.rs:74–75, 114`).
- **Termination:** no guarantee across process kill, abort, or allocator
  abort (`docs/rust-engineering.md:38`). Process-private memory does not
  outlive the process; widening admission (memmap, shared memory) would
  require re-deciding this.

## 9. `take`

- Python/NumPy-owned `take` arguments are refused with a typed error before
  any step. NumPy cannot give up its allocation, and L1-04 covers no such
  transfer.
- The Rust host-call entry point refuses any `take` parameter with
  `HOST_TAKE_UNAVAILABLE` at zero steps (HA-1.1).
- The Rust reference `take`/`mut` semantics in `db558b8` stay provisional and
  unexposed until a separate contract approves them.

## 10. Engine classes and the same-address proof

- **Native engines** must pass the L1-06:31 same-address test for every array
  argument.
- **Reference engines** are explicitly *copying semantic oracles*: they never
  run at the caller's address, are exempt from the same-address test, and are
  reported as copying in capabilities and test matrices. Their copies are
  staging (§6).
- Every native outcome, cleanup trace and step count must equal the reference
  oracle. No engine falls back to another silently.
- **Native status (HA-1.2):** native-serial exports accept the §13 subset at
  the caller's addresses and pass the same-address test. Outside it:
- **Native status (HA-1.1, historical):** no native engine accepted host array
  arguments, `read` or `mut`. The native paths refused them with typed errors:

  | Path | Refusal |
  |---|---|
  | export compilation (`src/exports.rs:205–206`) | `NATIVE_ABI`, "export parameter is not original i32/u32" |
  | owned-numeric host (`src/general_net/numeric_owned_host.rs:567–579`) | `Error::Input` for a non-scalar argument |
  | binding `call` | `PY_EXPORT_UNAVAILABLE` |
  | binding `call_foreign` | `PY_ABI_TYPE_UNAVAILABLE` |
  | public C ABI v1 | has no array tag and no `Mut` mode |

  The reference `read`/`mut` entry points do **not** imply native support. The
  no-copy, same-address native requirement (decisions 2 and 6) remains open
  until a native array-parameter interface is allocated and contracted.

## 11. Acceptance tests

| # | Test | Status |
|---|---|---|
| 1 | exact alias equals the single-owner in-source call: result, cleanup, outcome-matched step delta, and absolute stub steps; includes `grow2(X, X) == Ok(2000001)` | read slice |
| 2 | partial overlap counted by union (1.6M union admits a temporary); below the limit it equals separate in-source owners | read slice |
| 3 | cross-type overlap, extent overflow, misalignment and staging refusals: zero steps, typed codes, contract order; synthetic metadata only, no dereference | read slice |
| 4 | empty extents never alias, including one strictly inside another buffer | read slice |
| 5 | any overlap involving `mut` is refused with zero steps and unchanged bytes | stage 1: synthetic metadata (safe `&mut` borrows cannot overlap) plus every constructible refusal through `observe_host_call` |
| 6 | reference `mut`: commit on normal return including `Err`, equal to the in-source owner; unchanged bytes after traps, every fuel prefix, tool limits, failed reservations, failed prechecks and caught injected panics | stage 1 |
| 6n | native `mut`: restore after non-normal outcomes; snapshot reservation failure is a typed refusal | HA-1.3: integer suites plus `issue_l1_06_native_f32_arrays.rs`, `test_native_f32_arrays.py`, foreign guard and real native decode fault tests |
| 7 | native same-address `i32`/`u32`/`f32` read/mut arrays; reference disclosed as copying | HA-1.3: same files |
| 7r | returned arrays remain copying through `call_reference`; never counted as zero-copy | `test_reference_arrays.py`, `test_native_arrays.py` |
| 8 | NumPy `take` refused before any step; pin guard holds the export (a resize attempt raises) | binding |

## 12. Native host-array protocol v1 (private; slice 1)

**Status:**
- **Implemented:** the record definition, record construction and the
  rollback guard, in `crates/invar-foreign/src/host_array.rs` (private Rust
  module, not a C export).
- **Implemented by HA-1.2 (§13):** export admission, emission, loading and
  the guard-consuming call of native array code, with element values 1 and 2
  initially; HA-1.3 adds element 3 (`f32`).
- **Public C ABI v1 unchanged:** `inv_arg_v1`, its tags `1–4` and modes
  `0–2`, and `inv_result_v1`.

### 12.1 Record

```c
struct inv_private_host_array_v1 {   /* size 24, alignment 8 */
    uint32_t element;  /* offset 0:  1 = int32_t, 2 = uint32_t, 3 = float32 */
    uint32_t mode;     /* offset 4:  1 = read, 3 = mut */
    void    *data;     /* offset 8:  first element; NULL iff len == 0 */
    uint64_t len;      /* offset 16: element count, 0 ..= 1,000,000 */
};
```

- **Supported host ABI:** 64-bit object pointers and `usize`, natural
  alignment of `uint32_t`, `uint64_t` and pointers (the Linux and macOS
  targets already required by `inv_arg_v1`'s layout asserts).
  - `len` is an element count. `usize` ↔ `uint64_t` is lossless.
  - Byte length `len × 4` ≤ 4,000,000. `data` is 4-byte aligned for all three
    element types.
- **Mode values:** `mut` is **3**, never 2, so a public `inv_arg_v1` take
  mode can never decode as a private mode.
- **Construction:** a record exists only as the output of `records()` on a
  prepared call. It is derived from reborrows of live Rust slices that stay
  borrowed for the whole synchronous call, and its fields are private.
  Arbitrary or public addresses cannot form a record.
- **Guard binding (review3 fix):** a record is neither `Clone` nor `Copy`.
  `records()` returns a `HostArrayRecords` view that mutably borrows the live
  call guard and exposes no record slice; its raw pointer is crate-private.
  Records therefore cannot be detached, copied out or kept after commit,
  restore or drop. Any future native invocation must be a method of the
  guard that builds this view itself, never an API accepting detached
  records or record pointers.
- **Callee obligations:** the callee
  - must not write a `read` record;
  - must not access beyond `len`;
  - must not retain any pointer after returning;
  - must not call back into the host.
- **Validation order** (integer metadata only, before any record exists or
  any element is touched):
  1. count ≤ 8;
  2. each `len` ≤ 1,000,000;
  3. checked byte extents;
  4. any overlap involving `mut`, or across element types, is refused (§4);
  5. the extent union ≤ 2,000,000 elements (§5);
  6. snapshot bytes ≤ 8,000,000 (§6).

  Refusals are typed Rust errors (`Count`, `Length`, `Extent`, `Alias`,
  `Live`, `Snapshot`). AI-01 codes for them will be registered when they are
  first surfaced as diagnostics.

### 12.2 Rollback guard

- **Before entry:** every `mut` array's exact bits are copied into a
  fallibly reserved snapshot (total ≤ 8,000,000 bytes), and record capacity
  is reserved. A failed reservation is a refusal with no caller write.
- **Commit** happens only after the callee returned normally **and** every
  fallible host-result preparation succeeded.
- **Restore on every other path:** an explicit restore, a failed outcome
  (`finish(Err)`), dropping the guard, or unwinding from a Rust panic. Each
  `mut` array is restored to its pre-call bits. Restore is an element-wise
  copy into the same slices, cannot panic and allocates nothing.
- **Exclusions:** no guarantee across process abort, OS kill or allocator
  abort (§8). A native callee's `panic-abort` falls under the same
  exclusion.
- **Leaks:** rollback is `Drop`-based and leaking is safe Rust, so a guard
  passed to `std::mem::forget`, or otherwise leaked, is never restored; its
  `mut` arrays keep the callee's writes without a commit. A native invocation
  must therefore be a method of the guard that owns it for the whole call,
  builds the record view itself and decides commit or restore; the guard is
  never handed to caller code between `prepare` and that decision.
  - **Not yet enforced by types:** today `prepare`, `records()` and
    `commit()` are public and caller code holds the guard, so this is a
    documented requirement only. When the native invocation lands, it must
    consume the guard (`invoke(self, …)`) or construct it internally, and no
    API may accept a borrowed guard, a record view or record pointers.
- **Compile-check harness:**
  - **Compiler:** the pinned-code checks use `RUSTC`, else the `rustc` beside
    the running `cargo`, and first assert that its `release:` equals the
    `rust-toolchain.toml` channel. The crate's `rust-version` is only the
    MSRV and is not used as the pin. This is a **release pin, not exact
    compiler identity**: the full `rustc -vV` output (commit hash, host,
    LLVM) is printed to the test output as a receipt and not checked.
  - **Probes:** they cover an in-crate probe and an **isolated stand-in
    crate** built from `host_array.rs` alone and linked by `--extern`, in
    which the module's `pub(crate)` items are out of reach. The stand-in is
    not the real `invar_foreign` crate. A bounded static tripwire on the real
    crate's sources checks that the module is declared once, named nowhere
    else and not exposed by any glob re-export.
- **Evidence:** the C-fixture tests (an isolated C library writing at the
  real caller addresses) prove the guard mechanics and the C layout only.
  They are **not** compiled-source support.

## 13. Native array exports (HA-1.2 option A, extended by HA-1.3)

**Admission** (`src/exports/arrays.rs`), checked at export compile time for both engines, before any library is built:
- Selected functions may take `read`/`mut` `Array<i32>`/`Array<u32>`/`Array<f32>` parameters (at most 8 parameters in all) and return `i32`/`u32`.
- In those functions an array parameter is used only by element reads, `len`, and element stores to a `mut` parameter; an array value is never passed, bound, stored, returned or allocated.
- Reachable non-selected functions stay in the scalar fragment.
- Everything else is a zero-step `NATIVE_ABI` refusal: `take`, array results, array locals and allocation, views, and passing an array to a call.

**Private ABI:**
- Descriptor tags `0x100 | element << 4 | mode` (`0x111`, `0x113`, `0x121`, `0x123`, `0x131`, `0x133`) appear only in the private export descriptor, never in an `inv_arg_v1`.
- A function with array parameters has no public `inv_export_N` symbol.
  It is reached only through the namespaced private symbol `export_call_arrays_v1(ordinal, scalars, argc, records, count, out)` inside the existing `begin`/`end` transaction.
- The export host validates each §12 record against its tag (element, mode, `len <= 1,000,000`, NULL iff empty, 4-byte alignment, extent overflow) and registers it as a per-transaction handle.
- Generated code reaches elements only through `array_index_v1`, `array_len_v1` and `array_store_v1`.
  These return a status, bounds-check against `len` (trap reason 3, `INDEX_OUT_OF_BOUNDS`, with the source span), refuse stores to `read` arrays or out-of-range values, and read or write at the caller's address.
- The public C ABI v1 and the §12 record layout are unchanged; HA-1.3 adds element value 3.

**Call** (`CompiledExports::invoke_arrays`):
- Native-serial runs HA-1 admission (`HostArrayCall::prepare`), then `HostArrayCall::invoke`, which consumes the guard, builds the records itself, and commits only after a normal return whose result decoded; every other outcome restores the exact pre-call bits.
- The reference engine runs `observe_host_call` (copying).
- Native results, traps with spans, fuel exhaustion with step counts, and `mut` contents equal the reference on every tested fuel prefix.

**Binding:** `CheckedModule.call` admits these exports; `observe=True` reports the engine, whether the call copied, and the record addresses handed to native code.

**HA-1.3:** the same admission and call rules include `Array<f32>` using private descriptor tags `0x131` and `0x133` and the strict numeric policy above.
Returned arrays still use the copying `call_reference` path and are never reported as zero-copy.
