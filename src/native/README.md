# native

Optional LLVM CPU emission, object linking and execution.
The module root is ../native.rs; enable the native feature.
emit_mir builds Object; link and run manage native artifacts.
export.rs and export_host.rs implement the export boundary.
The generated export host is `no_std` and holds no thread-local storage, so macOS can unload it.
Keep it that way: one thread-local static, or any `std` panic or output path, brings the TLV back.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
