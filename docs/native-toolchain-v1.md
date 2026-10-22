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
the actual emitted IR for guards and rejects `nsw`, `nuw`, `noalias`,
`inbounds`, and `mustprogress`. LLVM verifies each module before object
emission. The backend rejects malformed MIR, unsupported host/link failures,
and generated IR over 8 MiB. It limits functions to 128, instructions to
100,000, per-function blocks to 4,096 and locals to 8,192. The runtime limits
call depth to 64, array length to 1,000,000, live array elements to 2,000,000,
cleanup observations to 512, and each compile/execute subprocess to 30 s.
These are typed tool/resource outcomes, never source traps or divergence
proofs. A native resource limit can be reached before the source reference's
limit; such a run is inconclusive for value equivalence.

## Acceptance evidence

On macOS 27 arm64, with the prefix above, the nonignored
`issue_m1_02_native_cpu_trap_abi` filter lists four tests. Filtered Debug
and Release tests, feature Clippy, full feature Debug tests, default fmt,
Clippy, Debug/Release tests, build, and offline tests pass. The fixed
source/MIR/native corpus covers 29 cases: values, exact trap categories and
byte spans, source-order failures, checked and wrapping boundaries, loops,
calls, recursion limit, arrays and loans, cleanup prefix, and fuel. The
real CLI `build` and `run` paths execute a linked object and reject an
existing output. No Linux/Windows native run or Miri result is claimed;
default Linux CI exercises the non-native path only.
