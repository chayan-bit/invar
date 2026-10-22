# CUDA target contract v1

The 2026-09-25 target change supersedes the earlier L4-only CUDA placement and
`sm_89` restriction. CUDA qualification may use the
selected CUDA GPU, with the actual model, driver, NVRTC version, and compute
capability recorded for each run.

Amended 2026-10-14: correctness-only CUDA test hosts are never performance gate hosts.
Performance gates are measured on an NVIDIA RTX 5070 Ti (sm_120, driver 580.178.04) and the Apple M4.

The verifier admits CUDA 12.4 or newer in the 12.x series and compute
capability 5.0 or newer. The `invar-math-cr-v1` CUDA kernels whose CORE-MATH
source declares an anonymous struct inside a union, such as binary64 `cos`,
`tan`, `atan2`, `log`, `log10` and `pow`, need NVRTC 12.9 or newer, because
NVRTC 12.4 rejects that construct; they are refused with a message naming
both versions before compilation ([v02-f64math](design/v02-f64math.md)). The runtime detects device zero and the loaded NVRTC
version, compiles for that device's explicit `sm` target, and rejects a plan
whose capabilities differ from the live device before submission. An absent
driver/compiler, unsupported target, or compile failure is a failure, never a
CPU result or hardware pass.

The M3 correctness checks, 64 Mi resident cases, copy-bandwidth fractions,
and handwritten-kernel ratio stay in force on the selected CUDA GPU. Record
each GPU's measurements separately; a result on one GPU cannot erase a failed
result on another.

Map DAGs, wrapping reductions and affine scans now admit up to 2^28 `u32`
elements through the device runtime API. Large standalone maps use the existing
DAG encoding; the frozen single-map encoding and ordinary source/reference
execution limits remain unchanged (1,000,000 elements per array, 2,000,000 live).
Admission is subject to transfer budgets and live device capacity, with typed
resource refusals.
This does not qualify M5 performance. Family verifier limits stay fixed. CUDA
floating, transpose, stencil and segmented admission now composes with the
shared 8 GiB transfer and owned-staging policy; stencil and transpose retain
their separate 5 GiB family caps. The runners perform a live free-memory
preflight before storage. CPU helper tests exercise injected capacity
shortfalls and checked-budget ordering; they do not execute a public GPU
runner. On the RTX 5070 Ti, the composed source passed one Debug 2^28
map/reduction/scan test and six Debug grid-family hardware tests, including
2^28 stencil and transpose. CUDA segmented, floating, Release and broader
capacity/performance qualification remain pending; see the linked evidence.
