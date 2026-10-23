# Bootstrap language 0.1

## Status and precedence

This file is the detailed normative text of the implemented language; the compact summary with executable examples is [the v0.1 specification](../docs/spec/invar-v0.1.md). In case of disagreement: this file and [grammar.ebnf](grammar.ebnf) must be reconciled before implementation; then milestone issue acceptance applies; [architecture](../docs/architecture.md) describes later direction. The archived proposal and later-roadmap examples never expand the accepted bootstrap language.

A discovered contradiction requires a small documented clarification and regression case, not silent implementation-defined behavior. No backward-compatibility guarantee is made before the reference-core milestone is accepted.

## M0 scope

This section is the M0 baseline. The later L1, R1, S1 and M4 sections extend it, and where they lift an exclusion listed here the later section governs.

One UTF-8 source file, direct named functions, `Bool`, `i32`, `u32`, `Unit`, fixed-length owned `Array<i32>` and `Array<u32>`, local bindings, direct calls, `if`, `while`, counted `for`, and explicit `return`. No modules, imports, globals, closures, structs, enums, user generics, views, references as values, tasks, floats, FFI, exceptions, IO, or user-defined cleanup. Array type syntax is built-in, not user-generic support.

CLI execution starts at exactly one `fn main() -> i32`. Library-style test fixtures may have no main and use an internal harness; `check` accepts such files, `run` requires main. Missing main under `run`, or a declared main with the wrong signature under either command, is a source error (exit 1). Every non-Unit function must return on all conservatively analyzed paths. A Unit function may fall through. Source functions may recurse; recursion is not assumed total. A `return` terminates its path; an `if` terminates only when it has an `else` and both arms terminate. A block terminates when its reachable sequence reaches a terminating statement. Loops never establish definite return, even with a constant condition. Check every function and statement, including uncalled functions and unreachable code, for name/type/ownership validity; unreachable states do not contribute to continuing flow joins.

## Lexical rules

- UTF-8 must be valid. Identifiers are `[A-Za-z_][A-Za-z0-9_]*`; keywords are reserved. Non-ASCII text is allowed in comments only in this subset.
- Whitespace is insignificant. Comments use `//` through newline or nested `/* ... */`; unterminated comments are lexical errors. CRLF is accepted without changing byte-offset reporting.
- Decimal integer tokens consist of digits followed by optional `i32` or `u32`. No separators, bases or exponent; exact decimal and string literals arrive in the later sections. An identifier character immediately following a completed numeric token makes a malformed literal error; do not split `12oops` into valid tokens.
- Unsuffixed literals take an expected integer type from an annotation, parameter, return type, array element context, or the other arithmetic operand; otherwise default to `i32`. Conflicting concrete types are an error; there are no numeric promotions. Array literals require an expected `Array<T>` from an annotated binding, function return, call parameter, or whole-array assignment destination. Reject both `let xs = [];` and `let xs = [1, 2];`; no element-driven array-type inference exists in bootstrap.
- Parse literal magnitudes without native signed overflow. The directly negated spelling `-2147483648` is valid `i32`; positive `2147483648i32` is rejected. Unary minus on a nonliteral `i32` minimum traps. Unary minus on `u32` is rejected. A directly negated literal is range-checked as one mathematical negative value; a suffixed unsigned token stays unsigned and is rejected under minus.
- Source spans are zero-based UTF-8 byte intervals `[start,end)`. Human diagnostics may additionally display one-based lines and columns; JSON byte spans are authoritative.
- Reject out-of-range literals at compile time even in unreachable code. Later constant folding must preserve the distinction between static invalidity and a runtime arithmetic trap in an unselected branch.

## Values, binding, and mutability

Scalars are Copy. Arrays are non-Copy owners of fixed-length, fully initialized contiguous elements. The array length never changes. Bootstrap lengths and indices use `u32`; a supported length is in `[0, 2^32-1]` subject to available resources. Future `usize` support must be a separately specified extension.

`let` prevents rebinding and element mutation through that binding. `var` permits both when ownership allows it. A `mut` parameter grants mutable element access to its referent even though it is a borrower, not a mutable owner. A `read` parameter never grants mutation. Parameters themselves cannot be rebound; a `take` parameter owns its input but is immutable unless moved into a local `var`.

Names are function-local and block-scoped; no duplicate local or parameter names anywhere in a function, even disjoint blocks. Functions are resolved across the whole file, allowing forward calls and recursion. Duplicate function names, calls with wrong arity, unknown names, and redefining reserved intrinsics are errors. Local names may not hide a function or intrinsic name. No function values exist. Unit is legal only as a function return type in bootstrap; reject Unit parameters, bindings, comparisons, and array elements even where the general type grammar parses the spelling.

Declarations always initialize. Read of a moved value is rejected. Array equality and array arithmetic are unsupported; scalar equality requires equal types. Relational comparisons are for equal integer types. `not`, `and`, and `or` require Bool.

## Evaluation and control flow

Expressions and call arguments evaluate left to right. `and` and `or` short-circuit; only the selected `if` arm executes. Arithmetic and comparison operands evaluate left to right. All loop and condition expressions require the documented types; there is no truthiness conversion.

`for i in lo .. hi` evaluates both `u32` bounds once on entry. It iterates from `lo` inclusive to `hi` exclusive; `lo >= hi` is empty. The induction binding is immutable `u32`. Increment only after a completed body iteration; reaching `hi` ends the loop without an overflowing extra increment. `while` reevaluates its Bool condition each iteration and may diverge. `break` and `continue` are deferred, not silently implemented.

Assignment evaluates the RHS first. Then evaluate the destination index (if any), validate bounds and current access, replace the destination, and release the replaced owner if applicable. Index expressions are evaluated once. During destination-index evaluation, read access to that destination owner is permitted, but mutable borrowing or consumption of it is rejected. Recheck owner availability afterward, then validate/store. This prevents an index computation from invalidating the storage it is about to index. Scalar/index assignment requires exact type equality and a mutable destination. Whole-owner reassignment requires an available, unborrowed local `var` and a fresh or explicitly moved array. A moved binding may not be reinitialized in bootstrap; rejecting it avoids a second initialization-state protocol.

The owner referenced by an assignment destination must remain available after RHS evaluation. Thus self-moving assignment and an RHS that consumes the destination owner are rejected. Reads used to compute an ordinary scalar RHS end before the store, so `xs[i] = xs[i] + 1` is permitted with exclusive owner access.

`return` evaluates its value then exits all active scopes. `return;` is permitted only for Unit. Unit expressions may be used as expression statements or returned from a Unit function. They are not usable as integers or arrays.

## Ownership and call loans

Scalar parameters omit a mode and copy their values. Array parameters require exactly `read`, `mut`, or `take`. Modes on scalar parameters are errors. `read` accepts an available owner or a compatible existing read/mut parameter; `mut` accepts a mutable local owner or an existing mut parameter. `take` accepts an owned array, never a borrowed parameter.

An array owner moves in a binding, whole-owner assignment, or ordinary consuming call only through `move name`, or from a fresh expression such as an array literal or owned-returning call. Array `return name;` is an explicit consuming context and moves without an extra keyword; `return move name;` is also accepted for an owner. Return of a read/mut parameter is rejected. Bare owner-to-owner assignment or binding is rejected, never deep-copied. `move` on a scalar is rejected as unnecessary/unsupported in the bootstrap.

Borrowed arguments must be bare binding/parameter names in bootstrap. No borrowing temporaries, indexing into owners as borrow sources, reference-returning calls, or stored loans. Read loans may coexist. A mutable loan excludes all other access to that array. Owner movement/deallocation is prohibited while any call loan is live.

Argument processing acquires each loan immediately after that argument's evaluation and keeps it until the outer call completes. This matters for `f(xs, g(xs))`: if the first argument establishes a mutable loan, `g` cannot read `xs`. If it establishes a read loan, `g` may read but cannot mutate or consume it. There is no mutable-loan reservation or implicit two-phase borrowing. A mutable actual passed twice, or both borrowed and consumed by one call, is rejected. Mutation through the callee's unique parameter remains permitted; suspension of caller access does not suspend callee access.

Nested calls can temporarily reborrow an existing mut parameter as read or mut, suspending the parent access during the nested call and restoring it afterward. Reborrows cannot escape a call. Multiple parameters that would alias incompatibly are rejected at the caller, even if the callee body does not happen to use both.

At a join of continuing `if` paths, an owner is usable only if usable on every continuing path. A returning branch does not participate in that join. A move on only one continuing branch makes subsequent use illegal. Bootstrap rejects a move of a binding declared outside any enclosing loop from inside that loop, even if source bounds happen to be one; array values freshly declared each iteration may move. No general lifetime or range solver is required.

## Arrays and reserved intrinsics

Array literal elements evaluate in order and have the contextual element type. Empty literals need that same explicit context. Allocation failure is an environmental failure, not a recoverable program result. The reference model assumes sufficient resources; the tool reports actual exhaustion separately.

| Form | Contract |
| --- | --- |
| `len(xs)` | Read-borrows a named array for this call; returns its `u32` length. |
| `filled_i32(n, x)` | Evaluates `n: u32`, then `x: i32` once; returns an owned Array<i32> with `n` copies. |
| `filled_u32(n, x)` | Same for `u32`. |
| `wrapping_add(a, b)` | Both operands u32; returns addition modulo 2^32. |
| `wrapping_mul(a, b)` | Both operands u32; returns multiplication modulo 2^32. |
| `sort(xs)` | Call-scoped mutable loan of a named `Array<T>`; stable in-place order; returns Unit. |
| `sort_by_key(keys, values)` | Pair sort: call-scoped mutable loans of distinct named key/value arrays; stably reorder both by keys; no closure form; returns Unit. |
| `argsort(xs)` | Call-scoped read loan; returns owned `Array<u32>` of stable original indices without changing input bits. |
| `segmented_sort(xs, offsets)` | Mutable key loan and read `Array<u32>` offset loan; sort each adjacent half-open boundary pair; returns Unit. |

For sorting, `T` is exactly i32, u32, or f32. Payloads may have any supported
scalar element type: i32, u32, f32, or Bool (`Array<Bool>`). Unit is not an
array element. Floating keys use IEEE totalOrder after replacing each NaN
with `0x7fc00000`; signed zeros remain distinct. Payload bits are preserved.
Argsort compares canonicalized keys but preserves all original key bits.
Equal keys retain input order. No source loop is implicitly replaced by sort.

Sort-by-key length mismatch traps `INDEX_OUT_OF_BOUNDS`. Segmented offsets
follow M4-04: `segments + 1` entries, first 0, last len(xs), non-decreasing.
The number of segments is inferred as `len(offsets) - 1`. At least one offset
is required; `[0]` is valid only for an empty input (zero segments). Duplicate
boundaries describe empty segments. Validate every offset and both endpoints
before any key write or device dispatch. Empty offsets, missing endpoints,
descending or out-of-range entries trap `INDEX_OUT_OF_BOUNDS` at the call span.
This v0.2 C10 ruling supersedes the sort lane's earlier partial-boundary contract.
f64 sorting follows after f32 lands; i64 sorting follows B3, per C5.

These spellings avoid pretending a method/generic standard library already exists. They are bootstrap intrinsics; any later renaming needs a spec change and fixture migration. Indexing requires `u32`; negative signed indices do not convert. Bounds failure is a defined source trap. Reads copy scalar elements; stores require exclusive access. No raw pointers or uninitialized array builders exist in this subset.

## Numerical behavior

Ordinary `+`, `-`, `*`, and unary signed `-` check representability. Division by zero traps. Signed minimum divided by `-1` traps as overflow. Signed division otherwise truncates toward zero; remainder satisfies `a = (a / b) * b + (a % b)` mathematically and has the dividend's sign or zero. `MIN % -1` is explicitly `0`, without evaluating the overflowing signed division; zero divisor still traps. `u32` division is floor division; its remainder is in `[0, divisor)` and satisfies the usual quotient/remainder identity. Unsigned division or remainder by zero traps. `u32` subtraction below zero traps. No arithmetic changes between debug and release.

The source trap categories are `INTEGER_OVERFLOW`, `DIVISION_BY_ZERO`, and `INDEX_OUT_OF_BOUNDS`. Remainder with zero uses `DIVISION_BY_ZERO`. A trap records the originating operator/index source span, not a lowered instruction address. Loops need not encode iteration number in trap identity at bootstrap. If multiple demanded operations can trap, the first in reference evaluation order wins.

All normal returns release still-owned arrays in reverse declaration order for each exited scope. Moves suppress release of the old binding. Traps perform no language-level cleanup. Host-language memory management may reclaim interpreter storage but must not be represented as an Invar cleanup event after a trap.

## Floating point (M4-01, v0.2 f64math)

`f32` and `f64` are executable primitives with `Array<f32>` and `Array<f64>`, record fields, variant and `Result` payloads, generic instantiation, and `f64` value arguments of `extern c fn`.
`f32` follows `strict-f32-v1` and `f64` follows `strict-f64-v1`: IEEE 754 binary32 and binary64, round-to-nearest-even, gradual underflow, no contraction, and fusion only through an explicit `fma`.
NaNs form one class; a NaN produced by a library function or a float-to-float conversion is the canonical quiet NaN.
A decimal literal is digits, `.`, digits and an optional `f32` or `f64` suffix; an unsuffixed literal takes its type from context and defaults to `f32`.
There is no implicit conversion, no float remainder and no float view.

| Form | Contract |
| --- | --- |
| `filled_f32(n, x)`, `filled_f64(n, x)` | Owned array of `n` copies, like `filled_i32`. |
| `f32_from_i32`, `f32_from_u32`, `f64_from_i32`, `f64_from_u32`, `f64_from_f32` | Exact or rounded to nearest even. |
| `f32_from_f64(x)` | Rounds to nearest even. |
| `i32_from_f32`, `u32_from_f32`, `i32_from_f64`, `u32_from_f64` | Truncate toward zero; NaN or an out-of-range truncated value traps `INTEGER_OVERFLOW` at the call. |
| `fma(a, b, c)` | `a * b + c` with one rounding, for f32 or f64. |
| `invar-math-cr-v1` functions | `sqrt`, `cbrt`, `hypot`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(y, x)`, `sinh`, `cosh`, `tanh`, `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `pow`, `floor`, `ceil`, `trunc`, `round`, `abs`, `copysign`, `min`, `max`, overloaded on f32 or f64 and correctly rounded. |

`round` rounds half away from zero; `copysign` treats a NaN sign source as positive; `min` and `max` are IEEE 754-2019 minimumNumber and maximumNumber, so a NaN operand is ignored and `-0 < +0`.
All these names are reserved and cannot name a function, parameter or local.
Every engine that executes a type returns identical encodings; Metal has no binary64 and runs f64 programs on the CPU with explain reason `metal-no-f64`.

## Reference execution and tool protocol

The interpreter uses bounded fuel for automation. Every expression evaluation and statement dispatch consumes one unit before execution, including every reevaluation of a while condition; evaluating a literal, name, or intrinsic counts as an expression. A call consumes its call-expression step plus the steps of evaluated arguments/body; no extra step is charged just for entering a function. A block has no additional charge beyond its statements. For-loop entry charges its statement once; each iteration additionally charges one loop-control step before its body. Resource bookkeeping does not consume semantic fuel. Pin boundary tests for this convention; fuel is a tool limit and never an optimization legality fact.

With fuel `N`, exactly `N` charged steps may execute. Before step `N+1`, report `evaluation_limit` with `steps_used == N`. Static checking consumes no evaluator fuel.

Native and interpreter runs may consume unrelated amounts of work; tests compare semantics only for completed runs. Fuel exhaustion returns `evaluation_limit`, never “proved divergent.” Native divergence tests need an external deadline and report timed-out runs as inconclusive except where checking preservation of deliberately divergent fixtures.

The M0 commands are below; `invar --help` lists the full current command set, summarized in section 11.1 of the v0.1 specification.

```text
invar check path.inv [--format json]
invar run path.inv --engine reference [--fuel N] [--format json]
```

Default fuel is 1,000,000, overridable by positive decimal `N`. CLI invalid arguments report usage error. Default human output reports result/diagnostics without interpreting a returned main integer as a process exit code. `--format json` prints one JSON document to stdout (including diagnostic failures); no logs or banners are mixed into stdout. Source errors, traps, and tool failures remain distinct.

| Outcome | Process exit | JSON status |
| --- | --- | --- |
| check success | 0 | `ok` |
| main completed, regardless of its i32 value | 0 | `ok` |
| source rejected | 1 | `source_error` |
| CLI/file/encoding/resource/internal tool error | 2 | `tool_error` |
| source trap | 3 | `trap` |
| evaluator fuel exhausted | 4 | `evaluation_limit` |

For invalid source encoding, report tool error `INVALID_UTF8`; lexically invalid valid-UTF-8 text is a source error. JSON schema version 1 always has `schema_version`, `status`, and `diagnostics` (an array). Completed run additionally has `result: {type: "i32", value: <integer>}`. Trap additionally has `trap: {category, span: {start,end}}`. Diagnostics have `code`, `message`, `path`, `span: {start,end}`, and `notes` (an array). File-level errors use null span; normal source diagnostics must have a concrete span. Evaluation limits additionally report `fuel_limit` and `steps_used`. Extra documented fields may be added; required field meanings cannot silently change.

Minimum source diagnostics include `LEX_INVALID_TOKEN`, `LEX_INVALID_LITERAL`, `PARSE_EXPECTED_TOKEN`, `NAME_UNKNOWN`, `NAME_DUPLICATE`, `TYPE_MISMATCH`, `RETURN_MISSING`, `ENTRYPOINT_MISSING`, `ENTRYPOINT_SIGNATURE`, `OWN_USE_AFTER_MOVE`, `OWN_BORROW_CONFLICT`, `OWN_MOVE_REQUIRED`, `OWN_RETURN_BORROW`, and `UNSUPPORTED_FEATURE`. Do not hard-code diagnostics only for the supplied example files.

## Bounded L1-01 language extension

The L1-01 grammar adds lexical modules and imports, records, variants, and `Result`. Module names and imports resolve before function checking; public declarations may cross a module boundary, while private declarations remain in their module. Resolution is deterministic and rejects duplicate names, invalid imports, and dependency cycles. Record initialization evaluates declared fields in source order and requires every field exactly once. Record fields in this subset are `i32`, `u32`, `Bool`, or integer arrays. Reading a scalar field copies it; an array field requires `move owner.field`, which leaves that field unavailable and bars a later whole-owner move. Variants have bounded scalar payloads, require exhaustive matching, and evaluate only the selected arm. A borrowed match reads its payload; `match move owner` consumes the owner and grants ownership of its selected payload to the arm binder. Reusing a consumed owner is an error.

`Result<T,E>` is a built-in variant with `Ok(T)` and `Err(E)`. Each payload is `i32`, `u32`, `Bool`, or another `Result`; nesting is limited to four levels. `Result::Ok(value)` and `Result::Err(value)` require an expected `Result<T,E>` type and do not infer or convert the other payload type. Postfix `?` evaluates one `Result<T,E>` value. On `Ok`, it yields `T`; on `Err`, it returns `Err(E)` from the enclosing `Result<U,E>` function after normal owner cleanup. The error type must match exactly. An `Err` is an ordinary value, distinct from a source trap, an evaluation limit, and a tool failure. Constructors and propagation preserve source order and the usual expression/fuel accounting. No implicit conversion, clone, or error wrapping occurs.

## Bounded L1-02 borrowed views

`View<i32>`, `View<u32>`, `MutView<i32>`, and `MutView<u32>` are non-owning array views. `view(owner, start, end[, stride])` creates a read view; `view_mut(...)` creates a mutable view. The owner must be a bare array or view name. `start` and `end` are `u32` logical indices in the source, with a half-open range; `stride` is a positive `u32` and defaults to 1. The resulting length is `ceil((end-start)/stride)`, including zero for an empty range. For a nested view, physical access at index `i` uses the parent's physical origin plus `(start + i*stride)` times its stride. Range, stride, composition, and physical access are checked in `u32` arithmetic before access. Invalid values trap as `INDEX_OUT_OF_BOUNDS` at the view or index operation, consistently in reference, MIR, and native serial execution.

Views have lexical scope and no owner cleanup event. An owner cannot be moved, dropped, or mutated while a view of it is live. A mutable view suspends direct owner reads; nested reborrows suspend use of their parent until the child scope ends. Multiple mutable or read/mutable views of one owner require provably disjoint physical intervals; unknown bounds cannot establish disjointness. Two read views may overlap. The checker records the root owner, conservative physical interval, stride, mutability, parent, and return boundary; verified MIR repeats the root, alias, and return checks. At most 4096 view facts or simultaneously live native view handles are allowed.

A function may return `-> View<T> from xs` when `xs` is a `read` or `mut Array<T>` parameter, or `-> MutView<T> from xs` only when `xs: mut Array<T>`. Every returned view must derive from that one parameter, possibly through a nested view or a call with its own declared return source. The caller's owner remains borrowed until the returned view's scope ends. View values may be bound from constructors or view-returning calls, or returned directly; assigning or copying a view requires an explicit new reborrow. This subset has no view parameters, view fields, temporary-owner views, implicit materialization, or GPU strided-view lowering. An unsupported parallel view operation uses the checked serial path or reports unavailability; it cannot bypass these source and MIR checks.

## Bounded L1-03 generics and affine closures

A generic function declares one or two type parameters with the bound `Value`, for example `fn select<T: Value, U: Value>(x: T, y: U) -> T`. This is a function limit for L1-03, not a global type arity limit: later R1/S1 contracts add type, frame, unit and const parameters to records with three or more arguments. A `Value` argument is a complete concrete non-view value type supported by the current checker; an instantiated body is checked under its substituted types, so `Value` does not imply numeric operations or a trait solver. Parameter and result positions may use a bare type parameter; `Array<T>` is also permitted when `T` resolves to `i32` or `u32`. An ordinary call infers each type parameter only from exact argument types, never from the expected result; `f::<i32, u32>(...)` supplies all arguments explicitly. Missing, conflicting, or unsupported type arguments are source errors. Generic functions cannot be values, and higher-kinded or trait syntax is unsupported.

Each concrete instance is identified by the source function's qualified identity plus its ordered concrete type arguments. The same instance is built once and reused; no binary or instance survives a source change. The total lowered function limit remains 128, and expansion is additionally limited to 16 generic frames. Recursive generic instantiation, including a cycle that repeats the same concrete instance, is rejected before execution. Diagnostics retain original source spans. Monomorphized reference, MIR and native serial paths share the same checked concrete function contract; GPU closure/generic kernels are not introduced here.

A local affine closure is written `let c = closure [copy n: i32, read xs: Array<i32>, take ys: Array<i32>] (x: i32) -> i32 { ... };` and invoked as `c(arg)` at most once in that same lexical block. A closure has at most eight captures, eight explicit parameters, and 65,536 source bytes; its body obeys the ordinary function body and call-depth limits. Capture names and types are explicit and must match typed parameters or explicitly annotated active bindings. `copy` snapshots an integer or Bool at creation; `read` borrows an affine owner from creation until invocation or lexical scope exit; `take` moves an affine owner at creation and releases it through ordinary cleanup if never invoked. While a read closure is live, direct access to the captured owner is rejected; this subset also rejects read closure captures in a function that constructs views. The body can use only its captures, parameters and global functions; undeclared outer variables are source errors. A closure cannot escape, be assigned, passed, returned, invoked twice, or be generic. Closure calls appear in straight-line statements, outside control-flow conditions. Capture setup runs left to right before later source statements; invocation delivers stored and borrowed captures in declaration order, then evaluates explicit arguments left to right before entering the body. Borrow conflicts, double moves and hidden effects are source errors in every engine. Closure lowering makes captures and cleanup explicit in typed MIR and native serial execution. No dynamic dispatch or runtime closure object is part of this subset.

## Bounded L1-04 bytes and text

`Bytes` and `String` are distinct, immutable, owned values of at most 1,048,576 bytes each. A `b"..."` literal holds source UTF-8 bytes as written or arbitrary bytes from `\xNN` escapes. An ordinary `"..."` literal must be valid UTF-8. Both accept `\"`, `\\`, `\n`, `\r`, and `\t`; text does not accept `\xNN`. There is no implicit conversion or Unicode normalization. `len` reports byte length for both. `Bytes` indexing yields a `u32` byte and out-of-range access traps with `INDEX_OUT_OF_BOUNDS`; `String` has no indexing operation.

`bytes_slice(owner, start, end)` copies a half-open byte range into a new `Bytes`. `utf8_slice(owner, start, end)` copies a half-open byte range into `Result<String,Utf8Error>`; an in-range split of a UTF-8 code point returns `Err(Utf8Error)`. `utf8_decode(owner)` validates bytes and returns `Result<String,Utf8Error>`; `utf8_encode(owner)` copies UTF-8 bytes into `Bytes`. These four operations borrow a bare named immutable owner for the call. Their bounds are `u32`, out-of-range or reversed ranges trap with `INDEX_OUT_OF_BOUNDS`, and a failed decode or character boundary is an ordinary typed error value. Conversions never truncate. Result payloads may include `Bytes`, `String`, `Utf8Error`, or `ForeignError`; moving a Result transfers its owned payload and normal cleanup handles the unselected path. Aggregate live allocation remains bounded by the existing two-million-element runtime cap, counting each byte once.

The narrow foreign declaration is `extern c fn symbol(args) -> Result<T,ForeignError> oracle oracle_name;`, at top level only. At most eight arguments may be `i32`, `u32`, `f64` (tag `INV_TAG_F64_V1`, binary64 bits in `scalar`), `read Bytes`, `take Bytes`, `read String`, or `take String`; `T` is one of those four value types. The named ordinary Invar oracle must have the exact parameter modes/types and result type. `invar check` validates these declarations without loading a library. `invar run` and `invar build --target native` require an explicit absolute `--c-library PATH` when declarations are present; no source path, current-directory search or implicit fallback is allowed. Reference and MIR evaluation execute the independent Invar oracle. Native serial execution loads only the selected library, checks `inv_abi_version_v1() == 1` and the exact symbol, then calls the fixed shim in [`include/invar_foreign_v1.h`](../include/invar_foreign_v1.h). Building a native executable records the selected path without loading it; loading at runtime may execute that library's constructors.

The versioned C shim takes a pointer to `inv_arg_v1`, an argument count and an initialized `inv_result_v1`, and returns `int32_t` status. Tags, modes, 64-bit lengths and all offsets are frozen in the header and asserted in both C and Rust. Buffer arguments remain valid only during the call and C may neither retain nor free their pointers. `read` leaves the Invar owner live; `take` consumes and cleans that owner at the call boundary even when C reports an error. A successful C-owned buffer result must supply a release callback; Invar validates tag, version, null/length relation, maximum 1 MiB and UTF-8, copies the bytes, then calls release exactly once while the library remains loaded. Detectable invalid metadata, missing symbols and nonzero status become `Err(ForeignError::Failure())`; they do not trap or silently truncate. A dishonest pointer or release callback remains outside Invar soundness and cannot be dereferenced or freed safely by validation alone. No NUL-terminated signature, borrowed return, callback parameter, general FFI or GPU foreign call is admitted. This scalar/buffer dispatcher does not replace the separate typed lifecycle shim and f64 robotics layouts required by S1-02.

An S1-02 manifest may be a source file containing one `extern component` plus its frame declarations. It keeps explicit `@` rates on ports. `config c "...";` names a registered typed C layout, not an arbitrary C pointer. A `golden` clause may declare up to eight nonnegative exact-decimal `abs` tolerances in metres or radians; shim and golden paths are relative to the manifest directory. Repeatable `input "relative/path";` clauses declare additional source, header, build, model or child dependencies. Canonical resolution must remain under the explicit project root. A v3 system IR records the checked config layout, tolerance policy and declared input paths, leaving the frozen v1/v2 serialized forms unchanged. The initial v3 layouts are `inv_fk_config_v1` (borrowed UTF-8 model path during configure, path length, and model kind) and `inv_planar2_config_v1` (two f64 link lengths). Unsupported layouts and property text reject with `S1400`.

S1-06 adds `import component "relative/path/manifest.inv";` for a source-backed system. The path must be a bounded local `.inv` beneath the source directory, with no parent traversal or symlink escape. Compilation parses exactly one foreign manifest, verifies its transitive-input hash against a passing S1-02 evidence record, and inserts the checked component into the existing system IR component list. A changed or missing record fails with `S1405` before generation or process launch. Optional `for robot menagerie::name` selects one of several same-name component manifests by the product's robot. Exactly one candidate must match; duplicate or mixed selectors fail, and only the selected candidate requires current evidence. This permits separate fixed C layouts for Panda's seven and UR5e's six joints without padding or conversion. Manifest frame names used by ports must still be declared in the importing source; no network, library load, or automatic transform is implied. The serialized v3-v6 schemas are unchanged.

In an ordinary source component port, `std::robotics::JointPositions<f64,robot_dof>` resolves to a concrete seven- or six-joint type from the sole selected product robot before port type checking and IR serialization. It requires `menagerie::franka_emika_panda` or `menagerie::universal_robots_ur5e`; missing or unsupported selection is `S1001`. Foreign manifests continue to declare a fixed numeric count and matching C layout. The placeholder is not a general const expression or a type outside source component ports.

## Bounded M4-01 f32 source

`f32`, `Array<f32>`, bounded exact decimal literals, unary negation, `+ - * /` and IEEE comparisons are source features under numeric policy `strict-f32-v1`; the exact rules are in [the numeric contracts](../docs/numeric.md).
`%` is not defined on `f32`, an integer is never implicitly an `f32`, and `main` still returns `i32`.

## v0.2 deterministic float reductions

Four reserved intrinsics reduce float arrays: `sum_tree(xs)`, `dot_tree(a, b)`, `sum_exact(xs)` and `dot_exact(a, b)`.
Each operand is a bare named `Array<f32>` that the call read-borrows, exactly like `len`; the result has the element type, `f32`.
When `Array<f64>` exists the same spellings are overloaded by element type and return `f64`; mixing element types is a source error.
Evaluation charges one step for the call and one per operand, independent of the length.
A source `for` loop that folds floats is never reassociated; these intrinsics are the explicit opt-in to a parallel order.

The only trap is `INDEX_OUT_OF_BOUNDS` at the call span when the two dot operands differ in length, checked before any arithmetic.
No other input traps: NaN and infinity are values.
Every NaN result is the canonical quiet NaN (bits `0x7fc00000`).

`sum_tree` has one fixed shape that depends only on `n = len(xs)`:

- `n = 0` gives `+0`.
- Leaves are blocks of 8 consecutive elements; leaf `k` is the left-to-right fold of `xs[8k .. min(8k+8, n))` that starts from its first element (no `+0` seed).
  The leaf block is frozen at 8.
- With `m = ceil(n / 8)` leaves, node `(L, j)` exists when `j * 2^L < m`.
  Node `(0, j)` is leaf `j`; node `(L, j)` is node `(L-1, 2j)` plus node `(L-1, 2j+1)` when the latter exists, else node `(L-1, 2j)` unchanged.
  The result is node `(H, 0)` for the least `H` with `2^H >= m`.
- Every `+` is one IEEE 754 binary32 addition, rounded to nearest-even, with gradual underflow and no reassociation or contraction.

`dot_tree(a, b)` rounds each product `a[i] * b[i]` separately (no fused multiply-add) and sums the products with the `sum_tree` shape.
The tree result is the same on every engine and worker count but is generally not equal to a sequential fold.
Intermediate overflow follows IEEE arithmetic, so a tree sum may be infinite or NaN where the exact sum is finite.

`sum_exact` and `dot_exact` return the exact real sum of the elements (respectively of the exact products), rounded once to nearest-even.
They are order-independent, which is what lets them run in parallel.
Their special values are, in order:

- `n = 0` gives `+0`;
- a NaN element, a product of an infinity and a zero, or both `+inf` and `-inf` among the elements (respectively the products) give NaN;
- otherwise any infinity gives that infinity;
- an exact sum of zero gives `-0` only when every element (respectively every product) is `-0`, and `+0` otherwise;
- a finite exact sum whose rounding overflows gives the signed infinity, and one below half the least subnormal gives the signed zero.

`reference`, `native-serial`, `cpu-parallel` with any worker count, `cuda` and `metal` must return identical bits.
An engine may split the tree only at aligned nodes, so any worker count or device geometry gives the same shape.
`--engine metal` and `--engine cuda` run every reached nonempty call on the device; an empty call needs no work and keeps the MIR result.
`--engine auto` places each reached call by the calibrated cost model, like loop families, with the calibration family id `fsum.<op>.<type>` (for example `fsum.sum_tree.f32`).
CPU workers are chosen by a measured cost model: a call whose work does not repay starting its workers runs on one worker and is reported with backend `cpu-serial` and reason `measured-cpu-cost`; the fixed tree shape makes the bits identical either way.
`--engine auto` never places a call on Metal below the measured crossover recorded under its family id in `calibration/fsum-v1.json`.
Under a device engine or `auto`, every reached call leaves a receipt with its backend and a reason from the explain registry: `forced-device`, `measured-device-cost`, `measured-cpu-cost`, `device-unavailable`, `device-plan-ineligible`, `toolchain-unavailable`, or `metal-failed-before-publication`/`cuda-failed-before-publication` when the device result was discarded before publication.
A device whose binary32 arithmetic flushes subnormals must detect every operation whose result could differ (a subnormal input, or a zero result from operands that do not cancel exactly), discard its words, and let the CPU produce the call's result.
`invar explain` adds the region obligation `FLOAT_FOLD_HINT` (evidence `hint`) to a rejected loop whose body stores `acc + e` into the `f32` local `acc`, naming `sum_tree` and `sum_exact`; it never rewrites the loop.

## v0.2 spatial grid queries

`voxel_grid_build` returns opaque affine `VoxelGrid`; `radius_neighbors` returns built-in `Neighbors { counts: Array<u32>, idx: Array<u32> }`; `knn` returns built-in `Knn { idx: Array<u32>, d2: Array<f32> }`.
Query records bind normally, and `move result.counts`, `move result.idx`, and `move result.d2` transfer array fields with existing affine record rules; tuple binding is unsupported.
`grid_excluded(read grid) -> u32` reports points omitted because any coordinate is non-finite.
A non-finite query has no neighbours; kNN pads every missing slot with `u32::MAX` and positive infinity.
All spatial scalars and coordinates remain f32-only, with strict left-to-right squared distances and distance/index ordering.
Cell indices are exact mathematical `floor(coord / cell)` converted to i32, trapping `INTEGER_OVERFLOW` if out of range; a nonpositive or non-finite build cell also traps `INTEGER_OVERFLOW`.
Neighbour inclusion uses strict f32 `d2 <= radius * radius`; negative or NaN radii match nothing, positive infinity is allowed, and true counts are never clipped to the fixed output stride.
Coordinate-array length mismatch traps `INDEX_OUT_OF_BOUNDS`; u32 overflow of query count times output stride traps `INTEGER_OVERFLOW`; memory-budget exhaustion is the usual tool error.
The checker reserves the built-in types, permits `read` or `take` VoxelGrid parameters but never `mut`, and holds read loans through all scalar argument evaluations.
Reference execution implements the sorted-grid runtime and ruled traps. The owned grid snapshots finite points, uses the merged stable sort runtime, and traverses occupied cells exhaustively for exact queries. MIR, standalone native-serial and deterministic parallel CPU execution are accepted against the same independent oracle, including traps, moves, cleanup and worker-count independence. Physical device and performance acceptance remain separate checkpoints.
Large benchmark sizes use device harnesses or bench binaries until B2 lands, and calibration ids use `spatial.<op>.f32` with emitted explain reasons registered in AI-01.
No engine or performance acceptance follows from the test oracle or accepted source checking.

## S1-06 cell model notes

An attachable dynamic cell object may declare `grasp pose xyz(...) quat(...);`.
The pose is object-from-grasp: its translation and rotation are relative to the
object, independently of the object's initial/target poses and the robot TCP.
The clause requires an `object` with positive dynamic mass and collision geometry;
duplicate clauses, wrong translation units and zero quaternions fail source
checking. Explicit grasp sources serialize as additive `invar-system-v10`.
Unannotated sources retain their prior IR version and centred identity grasp.
MJCF places the object's weld site at the declared grasp. The checked task binding
derives the grasp tool goal as cube pose times object-from-grasp, with the existing
100 mm world-Z pregrasp offset. Place uses the observed attachment relation and
the unchanged final cube pose; acknowledgement and later observed weld state are
still required. No exact-home joint-goal planner interface is introduced.

The pinned-model frame adapter records `invar-tool-projection-v2`: source-base
from reference-base (`B`) and reference-tool from declared-tool (`T`). Its FK is
`B * FK * T`; a source planner goal becomes `inverse(B) * goal * inverse(T)` and
cube pose becomes `inverse(B) * cube`. Attachment becomes `T * tool_from_cube`.
Declared source frames and C/INVP/ROS layouts are unchanged. Planner runtime,
endpoint and evidence v3 bind this convention; v1/v2 records retain their previous
coordinate semantics and cannot be relabelled as v3 qualification.

## Implementation safety and extension gate

The interpreter is implemented in safe Rust. Its operations must avoid host panics, profile-dependent overflow, invalid division/remainder, invalid shifts, unbounded native recursion on hostile input, and unchecked host-size conversions. Use explicit checked/wrapping operations and typed outcomes; Rust ownership and Rust drop behavior do not substitute for Invar ownership and cleanup semantics. See [Rust engineering](../docs/rust-engineering.md). Enforce documented parser nesting/call-depth/array allocation limits as tool errors. Limit exhaustion does not change the abstract source semantics.

Before native support: freeze the supported target ABI and checked runtime trap mechanism. Before optimization: retain the source reference path, establish named rules and evidence invalidation, and test malformed candidate rejection. Before new syntax: update this contract, grammar, diagnostic cases, and semantic fixtures together. A convenience change that affects evaluation, ownership, trap, or cleanup behavior is a semantic change and must be reported.
