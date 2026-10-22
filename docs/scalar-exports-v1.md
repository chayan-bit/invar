# Selected scalar exports v1

The Rust `invar::exports` facade checks a complete source file and explicitly
selects original concrete i32/u32 functions, including library sources without
main. Selection preserves caller order and rejects duplicate names, inaccessible
module functions, erased nominal signatures, generic templates and unsupported
reachable MIR. A selected function has at most eight scalar value parameters.
Unselected functions still undergo normal source and ownership checking.

`compile_exports(source, path, names, ExportEngine::Reference)` retains checked
source, original bytes/path and descriptors. `NativeSerial` requires the `native`
Cargo feature and the LLVM 21 toolchain; its current shared-library target is
Linux. The builder also routes macOS through Mach-O two-level linkage and dylib
paths; actual macOS execution qualification remains pending. Missing
features/platforms are explicit errors. Neither engine silently
substitutes another engine.

Artifacts expose signatures and accept typed ScalarValue arguments through
`invoke` or a fully inventory-bound `invoke_descriptor`. Names, exact scalar
types, arity, positive fuel and descriptor identity validate before source work.
Original source errors/traps/fuel outcomes produce existing AI-01 diagnostics.
The facade never rereads a pathname to construct diagnostics.

Native exports retain the unchanged L1-04 call signature, tags and result layout.
A separate namespaced private begin/end record communicates failures. Generated
source functions return status/value pairs and propagate failures before later
source work. Source traps publish no output and return to the caller; the legacy
standalone native executable retains its original terminating protocol. Runtime
helpers/source functions are hidden or artifact-namespaced.

Each admitted function has at most 256 scalar local/value slots, 256 blocks and
2048 instructions. Native locals/values use fallibly allocated heap frames,
bounded to 64 frames and 128 KiB of scalar words. Invocation end reclaims host
frames after failure without recording source cleanup. Facade reference calls
use a bounded 16 MiB worker stack for the same source depth limit, independent of
the caller's Python/test-thread stack; only Copy scalar arguments/results cross
that boundary. Native loading and invocation remain confined to the calling
thread. Python bindings must retain the GIL throughout the synchronous call.

Identity binds source bytes/path, ordered export metadata, effective engine and
actual compiler build inputs. Native identity additionally binds effective target
layout/configuration, host rustc version and emitted library SHA-256. The loader
checks content, version, namespaced prelink identity and descriptor inventory,
then retains one library through all calls and transaction completion.

This scalar increment does not provide array/f32 exports, byte/text exports,
ownership transfer, other execution engines or macOS/Nix qualification. Full
L1-06 and v0.1 acceptance retain their original broader gates.

A private safe test seam retains the compiler-produced path/specification to
qualify unlink, atomic replacement and verified reload of actual Invar libraries.
Existing handles must preserve original behavior; the old specification must
reject changed bytes, and reload uses a fresh private path with the new matching
specification. Production does not expose the generated paths or C function
pointers through this seam.
