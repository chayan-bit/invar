# Native Python wheel qualification

The historical selected i32/u32 export bridge at source `985fab1` has a
qualified CPython 3.12 wheel on its recorded Linux x86_64, glibc 2.39 host.
This is a runtime relocation check, not macOS, Nix, older-glibc, or full L1-06
acceptance. The wheel is private and has not been published.

The v0.09 consolidation updates PyO3 to 0.29.2 and rust-numpy to 0.29.0. Its
workspace-local Debug and Release installations each pass all 253 Python
cases with reference and native-serial engines, plus nine Rust tests per
profile. These checks do not renew the historical standalone-wheel relocation
qualification; a newly repaired wheel needs its own isolated installation and
loader-path checks before distribution.

Use the recorded Rust 1.95.0 and LLVM 21.1.8 build environment, with these pinned
workspace-local packaging tools:

```sh
python -m pip install maturin==1.9.6 patchelf==0.17.2.4 auditwheel==6.8.2
maturin build --manifest-path crates/invar-python/Cargo.toml --locked --offline \
  --release --features extension-module,native --jobs 2 --auditwheel skip \
  --out /tmp/invar-raw-wheel
auditwheel repair --plat manylinux_2_39_x86_64 \
  --wheel-dir /tmp/invar-repaired-wheel /tmp/invar-raw-wheel/*.whl
```

The separate repair step is necessary for this toolchain: maturin's built-in
repair copied LLVM and its dependencies but retained LLVM's original
`$ORIGIN/../lib` RUNPATH. The installed wheel then failed to import its renamed
libffi dependency. Auditwheel gives LLVM and its dependent libraries the local
`$ORIGIN` path. Both the failed import and the successful repaired-wheel check
were observed.

Install the repaired wheel and test dependencies into a fresh virtual
environment. The recorded run used NumPy 2.3.5, pytest 8.4.2 and jsonschema
4.25.1, then ran all 253 tests with both explicit scalar engines. It removed
`PYTHONPATH`, `LLVM_SYS_211_PREFIX`, and the workspace LLVM/native library paths.
The cloud interpreter itself requires its existing libpython directory in
`LD_LIBRARY_PATH`; that was the only retained loader path. Native compilation
still requires the recorded rustc on PATH to link newly generated export hosts.
The qualification does not make rustc a bundled runtime dependency.

`wheel-identity.json` records the wheel SHA-256, size, platform scope, bundled
library hashes and effective RPATHs. Raw build, repair, installation and pytest
logs identify each operation; the initial offline installation failure without
NumPy wheels is also preserved.

## Refreshed wheel after numeric owned-host implementation

The source-bound numeric owned host verification
recorded a newly built and auditwheel-repaired CPython 3.12 wheel using PyO3
0.29.2 and rust-numpy 0.29.0. Its SHA-256 is
`a4802fb1be9e59288c7597a17c206aef90c4e3f74a57ec44fa3c5920217079b4`
and size is 58,250,634 bytes. It is not published.

A new isolated virtual environment with NumPy 2.5.3, pytest 8.4.2 and jsonschema
4.25.1 passed **253 tests**, selecting both reference and native-serial scalar
engines. PYTHONPATH, LLVM_SYS_211_PREFIX and LIBRARY_PATH were absent; only the
cloud interpreter's libpython directory remained in LD_LIBRARY_PATH. The loader
resolved LLVM and its renamed dependencies from the installed wheel's own
library directory. Bundled hashes and RPATHs are recorded in the verification
JSON. Rustc remains an installed requirement for generated native export hosts.

Runtime verification here is **Linux x86_64, glibc 2.41, CPython 3.12.14**.
Auditwheel derived the manylinux_2_39_x86_64 compatibility tag; no execution on a
glibc 2.39 machine was performed for this wheel. This refresh does not extend
qualification to macOS, ARM, older glibc, Nix or the full public-array/L1-06
contract. The historical wheel and its original failed import evidence remain
unchanged.
