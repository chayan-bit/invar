# native

Optional LLVM 21 code generation for verified MIR (the `native` Cargo feature), the standalone Rust hosts, and DWARF debug info.

## Where to start

- `emit_mir`, `emit_mir_for` and `emit_mir_debug` in [../native.rs](../native.rs) produce an `Object`; `link`, `link_with_library` and `run` build and execute it.
- [emit.rs](emit.rs), [ir.rs](ir.rs) and [floating.rs](floating.rs) generate LLVM IR; [optimize.rs](optimize.rs) selects the CPU and pipeline; [region.rs](region.rs) emits native kernels for admitted map and reduce regions; [debug.rs](debug.rs) writes DWARF 5.
- [export.rs](export.rs) and [export_host.rs](export_host.rs) implement the scalar and array export boundary.
- The generated export host is `no_std` and holds no thread-local storage, so macOS can unload it; keep it that way, because one thread-local static or any `std` panic or output path brings the thread-local variable back.

## Position in the pipeline

After [mir](../mir/README.md); selected by `--engine native-serial` and by `invar build --target native`.

## Tests

- `cargo nextest run --locked --features native --test mir` (`native_cpu_trap_abi` trap ABI, `native_parallel_regions` and `native_simd_and_targets`) and `--test language`.
- `cargo nextest run --locked --test diagnostics` covers DWARF and the debugger helpers.

## Rules to keep

- No `nsw`, `nuw`, `noalias` or `inbounds` without a proved `!invar.fact`; fast-math flags and `llvm.assume` are always refused.

## Read next

- [native toolchain and trap ABI](../../docs/native-toolchain-v1.md)
- [debugging](../../docs/debugging.md)
