# M1-02 native CPU and trap ABI v1

Status: the verified MIR CPU backend is implemented on `main` (landed with the M1 native and interaction foundations) on the recorded macOS
arm64 host. The `native` Cargo feature uses LLVM 21.1.8 from locked nixpkgs
`e2587caef70cea85dd97d7daab492899902dbf5d`, Inkwell 0.8.0, and
`llvm-sys` 211.1.0. Rust/Cargo are 1.95.0; rustc's own LLVM is 22.1.2 and
does not replace the pinned backend LLVM. The emitted object target is
`arm64-apple-darwin27.0.0` with data layout
`e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32`.

## Interface

```sh
LLVM_SYS_211_PREFIX=/nix/store/c8hx4pqh3j466qfa93snkq2q1pbdm28q-llvm-21.1.8-dev \
  cargo build --locked --features native
target/debug/invar build examples/sum.inv --target native --output /tmp/invar-sum
/tmp/invar-sum 1000000
target/debug/invar run examples/sum.inv --engine native-serial --format json
```

`build` refuses an existing output path.
The linker writes into a private scratch directory beside the output, and the executable is published by chmod and an atomic hard link.
The host process never holds a writable descriptor to it, so a child forked by another host thread cannot make its exec fail with ETXTBSY.
The output filesystem must support hard links.
The output is a standalone Rust host
executable linked with LLVM's object using `rustc --edition=2024 -C
opt-level=2 -C panic=abort -C link-arg=program.o`. It prints one JSON result;
its optional positive positional argument is source fuel (default 1,000,000).
The CLI preserves the M0 result/status schema and adds internal step and
cleanup observations. Default Cargo builds omit LLVM and report
`NATIVE_UNAVAILABLE` for native requests.

The versioned ABI is `invar_entry_v1() -> i64` plus `invar_tick_v1`,
`invar_enter_v1`, `invar_leave_v1`, `invar_trap_v1`, `invar_tool_v1`, and
array allocation/access/store/length/drop callbacks. MIR blocks become LLVM
control flow, values and locals become stack slots, and calls become direct
native calls. Scalar arithmetic is widened to i64, checked against the exact
i32/u32 range, and branches to a terminating trap callback before an invalid
division. Wrapping u32 builtins truncate to 32 bits. The Rust host owns array
handles, bounded allocation, call depth, fuel, and cleanup observations.
The sole unsafe host operation is the fixed-signature call to the verified
linked entry point; callbacks are safe Rust and terminate on source failures.

Representative checked path in emitted IR:

```llvm
%sum = add i64 %left, %right
%low = icmp slt i64 %sum, -2147483648
%high = icmp sgt i64 %sum, 2147483647
%overflow = or i1 %low, %high
br i1 %overflow, label %trap, label %continue
trap:
  call void @invar_trap_v1(i32 1, i32 %start, i32 %end)
  unreachable
```

The snippet shows the lowering rule with symbolic names. The test inspects
the actual emitted IR for guards; the promise audit below refuses any
`nsw`, `nuw`, `noalias` or `inbounds` without a fact, and `mustprogress`
always. LLVM verifies each module before object
emission. The backend rejects malformed MIR, unsupported host/link failures,
and generated IR over 8 MiB. It limits functions to 128, instructions to
100,000, per-function blocks to 4,096 and locals to 8,192. The runtime limits
call depth to 64, array length to 1,000,000, live array elements to 2,000,000,
cleanup observations to 512, and each compile/execute subprocess to 30 s.
These are typed tool/resource outcomes, never source traps or divergence
proofs. A native resource limit can be reached before the source reference's
limit; such a run is inconclusive for value equivalence.

## Optimization, CPU targets and promises (ENG-18, #101)

Objects from `emit_mir` run LLVM's new pass manager pipeline `default<O3>` with the loop and SLP vectorizers, interleaving and unrolling on, then generate code at `OptimizationLevel::Aggressive`.
A `Build` names the CPU and whether the pipeline runs:

| Build | CPU | Pipeline | Used by |
| --- | --- | --- | --- |
| `Build::PORTABLE` | `generic`, no features | O3 | `emit_mir`, so `invar build` artifacts run on every CPU of the architecture |
| `Build::HOST` | the live CPU (LLVM's host name and features) | O3 | `run_program`, `--engine native-serial`: the object runs here and now |
| `Cpu::Described(name)` | a `targets/cpu-*.json` description: its model selects the LLVM CPU (`zen4` is `znver4`, `apple-m4` is `apple-m4`) and its `isa` list the features | O3 when asked | building for a described machine |
| `Build::UNOPTIMIZED` | `generic` | none | debug builds (DX-20 keeps variable locations) and export libraries |

An unknown live CPU (LLVM reports `generic`) uses the generic baseline.
A described target whose architecture differs from the host triple, or whose `isa` entry has no LLVM feature here, is refused.
`Object::machine` records the CPU, features, matched description and pipeline; `Object::ir` stays the audited IR before any pass and `Object::optimized_ir` holds the result.

No fast-math flag is emitted, and LLVM fuses only `llvm.fmuladd` (none is emitted), so float results are the unoptimized ones bit for bit.
Traps stay explicit branches to the trap callback before any use or store, so trap order, spans and fuel are unchanged.

Every promise the emitted IR gives LLVM carries `!invar.fact !N`, and node `!N = !{!"invar.fact", i64 ID}` names entry `ID` of `Object::facts`.
`native::audit` runs on the emitted text before LLVM parses it and refuses the module unless each fact proves its promise:

Each fact is bound to the site that carries it, so a stale or misplaced attachment is refused:

| Promise | Fact | Check |
| --- | --- | --- |
| `noalias` | `Disjoint { params, owners }` | on a `define` line, on parameters only (a return-value `noalias` is refused); every `noalias` parameter there is in `params`; at least two owners, all distinct (affine ownership), no more params than owners |
| `inbounds` | `InBounds { base, index, elements, guard }` | on a `getelementptr inbounds` whose base pointer is `base` and whose only index is `index.value`; `index.range` lies in `0..elements` (a constant index in its own range); a named runtime guard ran before the call; `elements` keeps byte offsets of 16-byte elements inside `isize` |
| `nsw`, `nuw` | `Interval { op, bits, lhs, rhs }` | on `%r = op [flags] i<bits> lhs, rhs` with exactly these operand names; a constant operand lies in its own range; operands and the exact result of `op` on the interval box (checked i128 arithmetic) fit the signed or unsigned `bits`-wide range |

Metadata attachments other than `!dbg` and `!invar.fact` (and debug records' `!DI...` nodes) are refused too, since `!nonnull`, `!range`, `!noundef` and loop access groups are promises.
Other poison or UB promises have no fact form and are always refused: `nusw`, `exact`, `disjoint`, `nneg`, `samesign`, `nonnull`, `noundef`, `dereferenceable`, `mustprogress`, `willreturn`, `llvm.assume` and every fast-math flag (`contract` and `reassoc` are never strict; the others wait for #100 contracts).
The MIR lowering itself emits no promise (arrays are host handles), so its objects carry no facts.

### Compiled cpu-parallel regions

`--engine cpu-parallel` runs admitted wrapping `u32` map regions (`Plan::LiveArray`: per-input operations, an optional zip, then stage operations) and lawful `u32` reductions (`WrappingSum`, `WrappingProduct`) through native kernels from `src/native/region.rs`.
Each kernel is emitted with `noalias` on the map output (a fresh buffer, distinct from every input owner), `inbounds` on every access and `nuw` on the index step, each tied to its fact; it is audited, optimized for the live CPU (`Build::HOST`), linked with `cc -shared`, checked by SHA-256 and loaded through `invar_foreign::region`.
That module is the only unsafe code involved: it checks every slice length and the kernel's arity before the call, and each worker passes the kernel its own chunk.
Chunks, their join order, the fixed combine order of reduction partials and the fuel rule are those of the interpreter, so observations are identical; wrapping `u32` add and multiply are associative modulo 2^32, so any vector regrouping of a reduction gives the left fold.
A fused-map capture, a missing or short slice, more than `u32::MAX` elements, or any compile, link or load failure selects the plan interpreter.
`INVAR_NATIVE_REGIONS=never|always` (default: regions of at least 2^23 elements, since building one kernel took about 40 ms on a Linux x86_64 test machine) and `native::region::set_policy` choose when kernels are used; kernels are compiled once per plan and process (at most 256).

Not yet measured: the R <= 1.10 (clang -O3) and R <= 1.15 (ISPC) timing gates; they need a quiet host under the M5-01 protocol.

## Acceptance evidence

On macOS 27 arm64, with the prefix above, the nonignored
`issue_m1_02_native_cpu_trap_abi` filter lists four tests. Filtered Debug
and Release tests, feature Clippy, full feature Debug tests, default fmt,
Clippy, Debug/Release tests, build, and offline tests pass. The fixed
source/MIR/native corpus covers 29 cases: values, exact trap categories and
byte spans, source-order failures, checked and wrapping boundaries, loops,
calls, recursion limit, arrays and loans, cleanup prefix, and fuel. The
real CLI `build` and `run` paths execute a linked object and reject an
existing output. The CI `Build (native)` job runs the native-feature suite
(63 tests across five binaries) and the `invar-foreign` tests on Linux
x86_64 with LLVM 21; that is a CI correctness run, not performance
qualification. No Windows native run or Miri result is claimed.

ENG-18 (#101), `tests/mir/issue_eng_18_native_simd.rs` with `--features native`: the corpus (examples `sum`, `ownership`, `fk_panda` in binary64 with CORE-MATH sin, cos and sqrt, plus overflow, fuel and bounds traps) prints byte-identical documents and exit statuses under the unoptimized, portable and host builds; the audit maps every planted promise to its fact and refuses planted unproven ones; and a promised u32 map kernel compiles to vector code (vector IR types plus ymm/zmm registers on x86_64, `.4s` NEON arrangements on aarch64) for the live CPU and the described target. `tests/mir/issue_eng_18_native_regions.rs` checks map kernels against `Plan::live_values` on edge ranges and aliased slots, reduce kernels against the left fold, vector code in the kernels, and identical cpu-parallel observations with kernels on and off at 1, 2 and 7 workers and on fuel prefixes, including a checked overflow that traps in source order.
