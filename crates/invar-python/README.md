# Invar Python boundary (L1-06)

This package compiles/checks source, reports its signatures, returns the existing
AI-01 explain document, and runs `main` with the selected CLI engine.
An explicit
`exports` list or tuple selects scalar functions for an owned export artifact.
`exports()` lists its exact descriptors and `call(name, *args, fuel=...)` invokes
that artifact. Reference artifacts use the named source evaluator; native-serial
artifacts use the compiler-produced C library and private returning failure
protocol. The native core must be built with `invar/native`.
The frozen core facade is required.
The package also calls explicit, checked **C imports** with `call_foreign()` using
the existing `invar-foreign` adapter.

```python
import invar_python as inv
module = inv.compile_source("fn main() -> i32 { return 6 * 7; }")
assert module.run_main() == 42
signatures = module.functions()
explain = module.explain()

module = inv.compile_source(
    "fn add(x: i32, y: i32) -> i32 { return x + y; }",
    exports=["add"], engine="reference", path="add.inv",
)
assert module.call("add", 20, 22) == 42
assert module.exports()[0]["artifact_id"] == module.artifact_id
```

`call_foreign(name, absolute_library_path, *args)` only accepts extern C imports
declared in checked source. Supported arguments are exact built-in `int` within
`i32`/`u32` range, immutable `bytes`, UTF-8 `str`, and rank-one C-contiguous
NumPy-owned `uint8` arrays for `read Bytes`. Byte arrays pass their original
pointer through the maintained NumPy read guard and existing C ABI. Results are
`int`, `bytes`, or `str`. C-owned byte/text results are validated, **copied**, and
released by the existing L1-04 adapter; no output zero-copy claim is made.
`take` imports reject before loading/calling C because ordinary Python objects
have no consumable Invar owner handle.

## Array exports (HA-1.2, zero-copy on native-serial)

Selected exports may take `read`/`mut` `Array<i32>`/`Array<u32>` parameters and
return `i32`/`u32`; `call()` passes rank-one, C-contiguous, aligned,
NumPy-owned `int32`/`uint32` arrays (writable for `mut`). Caller-provided
outputs are `mut` parameters.

```python
import numpy as np, invar_python as inv
m = inv.compile_source(
    "fn scale(xs: read Array<u32>, out: mut Array<u32>, k: u32) -> u32 "
    "{ for i in 0u32..len(xs) { out[i] = xs[i] * k; } return len(out); }",
    engine="native-serial", exports=["scale"])
xs, out = np.arange(4, dtype=np.uint32), np.zeros(4, np.uint32)
info = m.call("scale", xs, out, 10, observe=True)
assert info["copying"] is False and info["record_addresses"] == [xs.ctypes.data, out.ctypes.data]
```

- **native-serial** runs at the NumPy addresses. Every `mut` array is
  restored to its exact pre-call bits after a trap, a resource limit or any
  other failure; nothing partial is published.
- **reference** artifacts accept the same calls and copy (`copying: True`).
- Validation, the alias rules, the borrow guards and the GIL are as for
  `call_reference` below. Exact read/read aliases are admitted.
- HA-1.3 admits `read`/`mut Array<f32>` with NumPy `float32`, exact-bit rollback and strict-f32-v1 arithmetic.
  `take` and returned arrays are not exported (`NATIVE_ABI`).
  Returned arrays use the copying `call_reference`. The public C ABI v1 is unchanged: arrays use the private
  record of `docs/host-array-arguments-ha1.md` section 12 (section 13).

`call_reference(name, *args, fuel=...)` is the explicit **reference-engine** array
entry point (HA-1.1, `docs/host-array-arguments-ha1.md`). It never uses or falls
back to a native path. It also covers `f32` arrays and returned arrays.
- **Arguments:** exact built-in `int` for `i32`/`u32`, exact `bool` for `Bool`, exact
  built-in `float` for `f64`, and rank-one C-contiguous aligned NumPy-owned
  `int32`/`uint32`/`float32`/`float64`/`bool_` arrays for `read`/`mut Array<...>`
  (writable for `mut`). `f32` scalar parameters and `take` are refused
  (`PY_ABI_TYPE_UNAVAILABLE`, `PY_TAKE_UNAVAILABLE`).
  - An `int`, a `float` subclass (including `numpy.float64`) or anything else for an `f64`
    parameter is `TYPE_MISMATCH`; nothing is converted silently.
  - A `float64` array must be native-endian and 8-byte aligned.
    `Array<f64>` elements keep every bit, NaN payloads and signed zero included.
  - A `bool_` array is one byte per element and every byte must be 0 or 1.
    A `uint8` view can write another value, and that is refused before any step with
    `PY_ARRAY_BOOL_VALUE`, for `read` and `mut` arrays alike.
- **Before any step:** every argument is validated, any overlap involving a `mut`
  array is refused (`PY_BUFFER_ALIAS`), borrow guards are taken and metadata is
  rechecked. The GIL and the guards (strong references) are held for the whole
  call, and no Python code runs during evaluation.
- **Results:** a dict `{"engine": "reference", "copying": True, "result": ...,
  "steps": n, "cleanup": [{"name", "length"}, ...], "committed": [mut arg indices]}`.
  - `Result` values are `{"variant": "Ok"|"Err", "value": ...}`. A returned `Err` is
    a normal value and its `mut` writes are committed.
  - An `f32` result is `{"f32": float, "bits": int}`. An `f64` result is a Python
    `float`, which carries every bit. Array results are new NumPy arrays: `float64`
    for `Array<f64>`, `bool_` for `Array<Bool>`.
- **Failures:** traps and limits raise `TrapError`/`ResourceError` with the AI-01
  document (including the source span) and a `steps_used` attribute; every `mut`
  array is then unchanged.
- **Copying, not zero-copy:** the reference engine copies. The binding also uses
  host *bridges*, all of them exact bits and reserved fallibly:
  - `float32` read arrays: one bridge per connected union of their original
    extents. Argument slices keep the original offsets and lengths, so exact
    and partial aliases (and the live-element count) are unchanged.
  - Every `mut` array: one bridge. The core commits into it, and NumPy memory
    is written only after the Python result is fully built and every writeback
    is prechecked. The final copy is allocation-free.
- **Bounds:** before any borrow or reservation, the binding checks the live
  union (2,000,000 elements, as the core does), bridges ≤ 8,000,000 bytes, and
  bridges plus the core's copies ≤ 40,000,000 bytes. A call holding an 8-byte
  `float64` array doubles both byte bounds; the element counts do not change. A failed check or
  reservation is a zero-step `PY_STAGING_LIMIT` (or `ARRAY_SIZE_LIMIT`).

`validate_array(array, dtype, length=None, output=False)` checks dtype, rank,
length, contiguity, alignment, bounds, writability, and NumPy allocation ownership.
Supported metadata types are `uint8`, `int32`, `uint32`, `float32`, `float64` and `bool`
(a `bool` array also has its bytes checked).
`validate_buffers(input, output, dtype)` also checks equal shape and forbidden
input/output overlap. These are boundary-policy helpers, not numeric kernel calls
or retained pins. A returned address is only an observation at validation time.
Externally backed arrays, subclasses, and non-owning views reject conservatively.

Calls retain the GIL, strong input references, and NumPy read guards until the C
call and decoding finish. No Python callbacks run with borrowed slices live;
scalar objects with `__index__` hooks reject. Ordinary Python threads cannot resize
or drop the pinned allocation during the synchronous call. This does not
synchronize native NumPy operations already running on another thread or protect
against a C library violating its pointer contract. Callers must not concurrently
write a borrowed array through unchecked native code. See the maintained
[NumPy borrow contract](https://pyo3.github.io/rust-numpy/numpy/borrow/index.html).

Scalar export calls accept exact built-in `int` values in the descriptor's i32/u32
range, at most eight arguments, and direct i32/u32 results. Unknown names, wrong
arity, invalid integers, integer subclasses and callback-based values reject
before invocation. Functions outside the explicit selection raise `EngineError`.
The artifact retains original source/path and loader state across calls;
`artifact_id` and ordered descriptors identify it. Calls retain the GIL and the
module is restricted to its creating thread, matching the core loader's !Send/!Sync
contract. After a trap or resource error the same artifact can be called again.
The loader treats descriptor inventories as untrusted: canonical `inv_export_N`
symbols must fit `u32` and agree with their inventory positions, or loading returns
a typed error before opening the library. Native calls bind to symbols, including
the private array entry's symbol ordinal.
Selection metadata is bounded to 4096 names and one MiB of aggregate name text.

Errors produced by the binding are subclasses of `InvarError` with a structured
`diagnostics` attribute and matching `diagnostic_json` string. Compile/check/trap
documents reuse the core AI-01 implementation, including its v2 document for the
versioned unit-conversion trap. Python boundary reason codes start with `PY_` and
are registered with named boundary tests. Twenty codes describe external Python
or C-boundary failures; the corrupt checked-signature guard is an invariant with
a schema-valid Rust fault test. Existing compiler source-code fixture requirements
remain unchanged.
The `fuel` and requested `length` controls require exact built-in integers;
negative, nonrepresentable, Boolean, or callback-based values raise structured
boundary exceptions rather than leaking PyO3 conversion errors.

## Engine selection and execution receipts

`compile_source(source, engine=...)` accepts exactly the CLI names: `reference`, `native-serial`, `cpu-parallel`, `auto`, `cuda`, `metal`.
`native` is not an alias.
`run_main(fuel=1_000_000, workers=None, observe=False, artifact=None)` executes the selected engine and returns the reached `main` i32.
For `cpu-parallel`, `workers` is an exact built-in integer in `1..=64`, defaulting to one.
Other engines refuse a supplied worker count with `EngineError` / `CPU_WORKERS`.
`auto` uses the CLI's worker cap and device preference (Metal on a feature-enabled Mac, otherwise CUDA when enabled, otherwise CPU).
The selector lives in `invar::engine` and is shared by CLI run, CLI device explain and the binding.

```python
module = inv.compile_source(source, engine="auto")
receipt = module.run_main(observe=True)
assert receipt == module.receipt()
print(receipt["requested_engine"], receipt["backends"], receipt["result"])
explain = module.explain()
assert explain["device_execution"]["choices"] == receipt["choices"]
```

`receipt()` returns `None` before execution and after an execution failure.
A call refused for its arguments (for example `workers` on an engine other than `cpu-parallel`, or `artifact` on one other than `metal` and `auto`) runs nothing and keeps the previous receipt.
A successful receipt contains `requested_engine`, the actual published `backends`, and `result`.
Scheduled receipts also carry per-region `choices` and cache/hint counts.
An empty choice list reports `mir-serial`, never a GPU or parallel execution claim.
Python `explain()` remains the schema-valid static AI-01 report until execution.
After successful `auto`/`cuda`/`metal` execution it includes that run's frozen `invar-explain-device-v1` document; it does not rerun the program.
Receipts and explain output are bounded to one million bytes.

Missing features, missing devices, ineligible programs and false device guards raise `EngineError` carrying the original `CUDA_UNAVAILABLE`, `METAL_UNAVAILABLE` or other Invar code in `diagnostics` / `diagnostic_json`.
`CUDA_PENDING`, `METAL_PENDING` and toolchain mismatch propagate without CPU replay.
`auto` alone permits engine fallback under the existing scheduler's quiescence rules.
The CLI CPU scheduler has a documented serial guard path; Python refuses a wholly serial forced `cpu-parallel` result with `PY_ENGINE_UNAVAILABLE` rather than returning it silently.
Traps and fuel/resource exhaustion raise `TrapError` and `ResourceError`.
`capabilities()["engine_features"]` reports compiled features, not live device availability.

Build the binding with `--features native`, `--features cuda` or `--features metal` as needed.
`run_main(artifact=path)` takes a compiled-ahead Metal library, as the CLI's `--artifact` does, for `metal` and `auto` (#46).
Build it with `invar build PATH --target metal --output FILE`; `path` is a `str` or `os.PathLike` naming that file.
The binding passes it to the same `invar::engine::run_mir` parameter, so validation and refusals match the CLI.
A missing, unreadable or corrupt file raises `EngineError` / `DEVICE_TOOLCHAIN_MISMATCH` for `metal` and `auto`.
A library built from another program raises the same error for `metal` and `auto`, including where `auto` runs a sort or spatial program on the CPU alone.
In-memory bytes and bytes paths raise `ValidationError` / `TYPE_MISMATCH`.
Other engines refuse `artifact` with `EngineError` / `PY_ENGINE_UNAVAILABLE`.
A build without Metal on macOS refuses it with `EngineError` / `METAL_UNAVAILABLE` before any backend is chosen, as the CLI does.
Without `artifact`, forced `metal` raises `METAL_UNAVAILABLE` and `auto` stays on the CPU, unless the developer flag `INVAR_METAL_RUNTIME_SOURCE=1` allows runtime MSL compilation.

```python
module = inv.compile_source(source, engine="metal")
receipt = module.run_main(artifact="map.metallibc", observe=True)
```

`tests/test_metal_artifact.py` and the Metal case of `tests/test_engines.py` build their libraries with the CLI named by `INVAR_CLI` (built with `--features metal`).
Run them outside the Nix shell, where `xcrun metal` finds Apple's Metal toolchain; with `INVAR_REQUIRE_DEVICE=metal` a missing feature or CLI fails instead of skipping.

CUDA requires the matching `invar-nvrtc` helper; set `INVAR_NVRTC_HELPER` to its explicit executable path when Python is not installed alongside it.
Its NVRTC runtime must support the device's compute capability.

HA-1.3 host exports remain exactly `reference` (copying) and `native-serial` (same-address `read`/`mut` i32/u32/f32 arrays, with rollback snapshots).
An explicit `exports=[...]` selection with `cpu-parallel`, `auto`, `cuda` or `metal` raises `NATIVE_ABI` because those schedulers have no host-array entry point.
These engines support whole-program `run_main`, not a new NumPy device ABI.
`call_reference` remains explicitly copying even on a module with another selected engine, including f32 and returned arrays.
The public C ABI is unchanged, and NumPy `take` remains refused.
Native f32 device execution from the M4-01 strict-f32 contraction work is not a dependency of this change.

The crate has its own Cargo workspace and lockfile. `nix develop .#python`
(Linux x86_64/aarch64, macOS aarch64) provides Rust 1.95.0, maturin, Python
with numpy/pytest/jsonschema and LLVM 21 (`LLVM_SYS_211_PREFIX`). Inside it,
create a local venv with `python -m venv --system-site-packages .venv`, activate
it, then run:

```sh
maturin develop --manifest-path crates/invar-python/Cargo.toml --locked --offline
cargo test --locked --test issue_l1_06_python_binding issue_l1_06_python_binding
cargo test --locked --release --test issue_l1_06_python_binding issue_l1_06_python_binding
cargo clippy --manifest-path crates/invar-python/Cargo.toml --locked --offline --all-targets -- -D warnings
python -m pytest crates/invar-python/tests -q
# With a native-enabled core and the LLVM 21 toolchain, qualify both engines:
python -m pytest crates/invar-python/tests -q --scalar-engine reference --scalar-engine native-serial
# Record the L1-06 ABI corpus subset counts:
INVAR_CORPUS_SUMMARY=corpus.json python -m pytest crates/invar-python/tests -k abi_corpus --scalar-engine native-serial
```

On macOS aarch64 the release
extension with `--features native` builds with maturin inside `nix develop .#python`
and runs `demo/zero_copy.py` and `tests/test_native_arrays.py`. The Linux qualification ran outside Nix (no Nix on the Linux
host); the Linux x86_64 and aarch64 shells evaluate but were not built.

## L1-06 criteria after HA-1.3

| Criterion | Path and evidence |
|---|---|
| i32/u32/f32 read/mut arrays at the same address | Native-serial zero-copy; `test_native_arrays.py` and `test_native_f32_arrays.py` |
| Empty and 1,000,000-element arrays | Same native path, copying reference comparator; empty records carry NULL |
| Exact bits, NaN payloads, signed zero and rollback | Bit copies and integer snapshots; Rust/pytest compare original bytes after traps, every tested fuel prefix and tool errors |
| Strict f32 arithmetic, infinities and gradual underflow | Existing strict-f32-v1; native/reference agreement and LLVM flag checks; arithmetic NaN payloads remain unobservable under that policy |
| Returned arrays owned by Python | `call_reference`, copying; unchanged |
| NumPy take, invalid dtype/shape/contiguity/alignment, aliases | Typed refusal before execution; unchanged |

Rust callers use `HostReadArray::Float32`/`HostMutArray::Float32` for native `f32` slices.
The existing `F32` variants containing `F32Bits` remain reference storage and are refused natively rather than cast or copied into a claimed zero-copy call.

## Building a native wheel

The native wheel is private and unpublished.
Build it with the pinned Rust 1.95.0 and LLVM 21.1.8 environment and these workspace-local packaging tools:

```sh
python -m pip install maturin==1.9.6 patchelf==0.17.2.4 auditwheel==6.8.2
maturin build --manifest-path crates/invar-python/Cargo.toml --locked --offline \
  --release --features extension-module,native --jobs 2 --auditwheel skip \
  --out /tmp/invar-raw-wheel
auditwheel repair --plat manylinux_2_39_x86_64 \
  --wheel-dir /tmp/invar-repaired-wheel /tmp/invar-raw-wheel/*.whl
```

The separate repair step is necessary: maturin's built-in repair copies LLVM and its dependencies but keeps LLVM's original `$ORIGIN/../lib` RUNPATH, so the installed wheel fails to import its renamed libffi dependency.
Auditwheel gives LLVM and its dependent libraries the local `$ORIGIN` path.
Native compilation of export hosts still requires rustc on PATH; it is not a bundled runtime dependency.
The repaired wheel is qualified on Linux x86_64 with glibc 2.41 and CPython 3.12 only (all Python tests on both scalar engines in a fresh virtual environment without `PYTHONPATH`, `LLVM_SYS_211_PREFIX` or workspace library paths); macOS, ARM, older glibc and Nix are not qualified.
