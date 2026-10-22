# Rust engineering contract

Status: project-wide implementation rules, adopted 2026-09-22 before compiler implementation. This changes the host implementation language, not Invar source syntax or arithmetic semantics.

## Scope and tools

Invar-owned compiler, reference interpreter, interaction engine, runtime, backend integration, and developer tools are implemented in Rust. Examples remain Invar source. Build manifests, documentation, and generated target code are not competing implementation languages. LLVM and GPU drivers may remain external native dependencies behind Rust interfaces; “implemented in Rust” does not claim those third-party systems are rewritten or memory-safe.

Use Rust 2024 and Cargo. M0-01 selects and tests an exact stable toolchain, records it in `rust-toolchain.toml`, declares the supported `rust-version` in `Cargo.toml`, and commits `Cargo.lock`. The edition, tested toolchain, and minimum supported Rust version are distinct facts. Do not guess a current version in these design documents or install tools during ordinary builds. Toolchain upgrades are reviewed changes with the same checks as implementation changes.

Start with one Cargo package containing a small library and a thin `invar` binary. Use modules with private implementation details; add crates only for an actual independent boundary, such as a later unsafe backend adapter. No workspace full of placeholder crates. The library owns semantics; the binary owns argument parsing, file IO, output, and process status.

## Types and ownership

- Model variants with enums and invalid states with restricted constructors, not strings, sentinel integers, or Boolean combinations. Small distinct IDs such as `ExprId` and `RegionId` are justified when mixing them would be a real bug.
- Prefer owned `Vec`, `String`, and explicit borrowed slices/`&str`. Pass ownership only when consuming it. Do not add `Clone`, `Copy`, or `Default` where their semantics are misleading.
- Keep trees simple. If an IR really needs sharing or cycles, use indexed storage and typed IDs rather than a pervasive `Rc<RefCell<Node>>` graph or self-referential Rust structs. Validate IDs at untrusted boundaries. Add generations only if slot reuse needs stale-handle detection.
- Make mutations local and explicit. Do not wrap the entire compiler in `Arc<Mutex<_>>`, introduce interior mutability to bypass a design problem, or clone whole trees to appease the borrow checker. Restructure ownership first; use interior mutability when its actual invariant warrants it.
- Prefer concrete functions/enums to a trait hierarchy for one implementation. Use standard traits when they express a real contract; generic parameters and trait objects need a present caller/use case.
- AST source identity and spans must survive checking and lowering. Do not expose every struct field as public to simplify a pass. Deterministic reports must sort unordered collections rather than rely on hash iteration order.

## Invar semantics are explicit

Rust's borrow checker protects the compiler implementation; it does not implement Invar's borrow checker. Invar moves, loans, initialization states, failure order, and cleanup need their own checked representation and tests.

Use `checked_*` and `wrapping_*` deliberately to implement Invar operators. Never inherit the host's debug/release overflow setting, indexing panic, or cast truncation as a source-language rule. Check host-size conversions and allocation-size arithmetic. In particular, Invar specifies `i32::MIN % -1 == 0`, so implement that special case before a Rust remainder operation; blindly forwarding a checked remainder failure would report the wrong Invar outcome.

Represent source traps, fuel exhaustion, rejected source, and tool errors explicitly. The interpreter may drop Rust-owned storage when returning a trap outcome; it must not count that housekeeping as Invar cleanup after a trap. Do not use Rust unwinding to simulate source-language control flow.

## Errors, trust, and unsafe code

Malformed source is expected input. Return typed errors with stable diagnostic codes and source spans. Use `Result` and `?`; keep structured errors until the output boundary instead of repeatedly converting them into strings. Avoid a universal error abstraction or dependency solely to save one small enum.

No `unwrap`, `expect`, indexing panic, `todo!`, or `unimplemented!` on user-controlled failure paths. Tests may use `expect`/assertions for test failure. An internal invariant assertion needs an actual established invariant and useful message; it is a compiler defect if reached, never a source trap. Do not use `catch_unwind` as a substitute for input validation.

M0 library and binary use `#![forbid(unsafe_code)]`. This constrains Invar-owned crate code; dependencies and the Rust standard library remain part of the trusted implementation and need separate review where relevant. Later FFI/device support lives in a small reviewed Rust boundary, with explicit ownership, validity, synchronization, error, and cleanup contracts. Every unsafe block/implementation documents why its preconditions hold; safe callers must not be able to violate them. No blanket unsafe `Send`/`Sync`, unchecked transmute, or raw pointer workaround for ordinary IR traversal.

Bound parsing depth, interpreter call depth, fuel, requested array size, and generated tests. Use fallible reservation where explicit allocation failure is part of the tool protocol. Do not promise that Rust can recover from every OS kill or global allocator abort; distinguish enforced-budget failures from uncatchable environmental failure. No Rust trait automatically proves Invar totality or optimization legality.

## Dependencies and performance

Use the standard library first, then a narrow reviewed crate when it removes a real correctness or maintenance burden. JSON is a good example: `serde`/`serde_json` can be preferable to a handwritten escaping/parser implementation. Check supported Rust version, licenses, maintenance, features, and transitive cost; commit resolution in the lockfile. Bootstrap needs no async runtime, ECS, parser framework, global service container, custom allocator, or plug-in system by default.

No network during compiler execution or routine tests. Dependencies/toolchains must be provisioned explicitly; `--locked` alone does not prohibit downloads. Offline verification uses cached dependencies with `--offline --locked`. Rust's own compiler using LLVM does not make Invar's interpreter native code generation.

Use clear loops or iterators according to readability and measured needs. Benchmark before introducing arenas, string interning, SIMD intrinsics, thread pools, pooling, or unchecked access. Do not add blanket `#[inline(always)]`, custom release flags, or unsafe code based on assumed speed. Preserve sequential baseline measurements.

## Writing and review

Use standard Rust naming, small cohesive modules, and comments explaining invariants or nonobvious semantic choices. Avoid comments that narrate each line. Document public interfaces with purpose, arguments, results, relevant errors, costs, and safety/panic conditions. Keep examples short and executable when implementation exists. Clippy suppressions are local, reasoned exceptions; never silence warnings across the project to pass CI.

Canonical gates after M0-01:

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo test --locked --release
cargo build --locked
```

Use ordinary `assert!`/`assert_eq!` in tests; required validation must never depend on `debug_assert!`. Add integration tests for the real CLI and focused unit tests for semantic boundaries. Do not generate one trivial test per method. Bounded Miri checks may supplement pure-core tests on a separately pinned compatible nightly toolchain; they are not a stable prerequisite or a soundness proof. Later unsafe/FFI code requires additional boundary-specific validation.

## Primary guidance

This is Invar's scoped policy, informed by the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/checklist.html), [Rust error-handling guidance](https://doc.rust-lang.org/book/ch09-03-to-panic-or-not-to-panic.html), [Cargo rust-version contract](https://doc.rust-lang.org/cargo/reference/rust-version.html), and [Rust 2024 project setup](https://doc.rust-lang.org/edition-guide/editions/creating-a-new-project.html). These are references, not a mandate to implement every suggested abstraction or trait.
