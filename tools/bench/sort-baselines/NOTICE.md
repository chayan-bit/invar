# Standalone baseline lineage

`arrowmetal.metal` specializes the histogram, scan and stable payload scatter
from [ArrowMetal](https://github.com/singhpratech/ArrowMetal), revision
`1f2111540e68edfdf1adf87df8821ca9cd1958c0`,
`Sources/ArrowMetal/Kernels/SortSource.swift`, under Apache-2.0 (included).
Changes: specialize K=uint and digitBits=8; retain only the three dense-key
kernels, add the TG=256 prelude, use a standalone Rust host. The upstream
blockPlan is preserved. Four passes are run; pass-elision/nullable mapping
is excluded for the full-width dense u32 benchmark. This is an independent
benchmark baseline, not production Invar code. Pipeline compilation is
outside its timing; GPU-event times cover all twelve kernels. Allocation,
input/output copies and verification are excluded from GPU-event timing.

`cub.cu` is the foreign NVIDIA DeviceRadixSort::SortPairs baseline compiled
with nvcc 12.9.86 and official CCCL 12.9.27 headers. Header archive SHA-256:
`8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa`.
NVIDIA CCCL retains its own BSD-3-Clause notices in the downloaded headers.
No upstream host framework is vendored.
