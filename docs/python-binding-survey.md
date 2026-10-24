# Python binding dependency survey (L1-06, #14)

Measured 2026-10-21.
License and unsafe figures come from the locked crate sources; repository figures come from the GitHub API.
"Unsafe tokens" counts the `unsafe` keyword in each crate's `src/`, so it bounds the reviewed surface and is not a defect count.

## Chosen stack

| Component | Pinned | License | Maintenance | Unsafe tokens |
|---|---|---|---|---|
| PyO3 | `=0.29.2` (`crates/invar-python/Cargo.toml`) | MIT OR Apache-2.0 | `PyO3/pyo3` pushed 2026-10-20, 16.3k stars, latest release v0.30.1 | 1266 across 153 files |
| rust-numpy (`numpy`) | `=0.29.0` | BSD-2-Clause | `PyO3/rust-numpy` pushed 2026-10-20, 30 open issues, tracks PyO3 releases | 252 across 21 files |
| ndarray (via `numpy`) | 0.16.1 (lockfile) | MIT OR Apache-2.0 | rust-ndarray, widely used | 399 |
| maturin | `==1.9.6` (`crates/invar-python/pyproject.toml`) | MIT OR Apache-2.0 | `PyO3/maturin` pushed 2026-10-20, latest release v1.15.0 | build tool only, not linked |

The pins are exact on purpose: the dependency set is reproducible from `Cargo.lock` and the build backend version.
The next release line of each (PyO3 0.30, rust-numpy 0.30, maturin 1.15) exists; upgrading is a separate change.

## Alternatives

| Alternative | License | Why it lost |
|---|---|---|
| `ctypes` or `cffi` over the C ABI | cffi MIT; ctypes is stdlib | No zero-copy NumPy validation, dtype and contiguity checks, or typed exception classes without a hand-written layer; the issue forbids hand-rolled FFI. |
| `pybind11` | BSD-3-Clause | C++ toolchain and a second language in a Rust compiler repository; no Rust-side ownership model. |
| `nanobind` | BSD-3-Clause | Same: C++ glue, and it would still need a C ABI shim into the Rust core. |
| `cbindgen` plus a generated C header | MPL-2.0 | Generates the C side only; the Python side remains hand-written. |
| PyO3 without `numpy` | n/a | The binding would reimplement dtype, shape, strides and ownership checks over raw buffers, which is the unsafe surface the crate avoids. |

## Unsafe surface of this repository

- `crates/invar-python` is `#![forbid(unsafe_code)]` (`src/lib.rs:2`), including the bool-byte scan, which reads a NumPy `uint8` view through the safe `numpy` API.
- The core crate is `#![forbid(unsafe_code)]` (`src/lib.rs:1`, `src/main.rs:1`).
- Unsafe code lives in the reviewed crates only: `crates/invar-foreign` (the export loader and the private host-array record), `crates/invar-cuda`, `crates/invar-metal` and `crates/invar-math`, and in the generated export host `src/native/export_host.rs`, which is compiled as its own `no_std` library.
