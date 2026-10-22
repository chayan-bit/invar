# Shape and layout contract v1

This document is normative for M4-01 and freezes decisions D3 and D4.
Layout contract version 1 and device IR version 2 are distinct version fields.
Unknown versions are refused with typed version diagnostics rather than silently reinterpreted.

## Logical shapes and physical storage

A shape has rank at most 4 and each extent lies in `[0, 2^31 - 1]`.
Zero-length axes are legal and make the logical domain empty.
A rank-zero descriptor denotes one scalar element.
Every buffer's physical allocation and logical byte count are bounded by 2 GiB.
Element types are `U32`, `F32`, and `F16`, with widths 4, 4, and 2 bytes respectively.
Logical shape is independent of allocation identity and physical capacity.
Strides and offsets are non-negative integers measured in elements.
Row-major layout uses a final-axis stride of one and each preceding stride is the checked product of subsequent extents.
A strided layout records rank, shape, strides in elements, and offset explicitly.
A transpose permutes the logical shape and strides together without changing the allocation identity.
Internal rank and stride descriptors do not introduce public escaping borrowed views.

## Checked physical access

Each logical index must be below its corresponding extent.
The physical element index is `offset + sum(index[axis] * stride[axis])` using checked multiplication and addition.
The byte interval is exactly `[physical_index * element_width, (physical_index + 1) * element_width)`.
The allocation base has a proven power-of-two byte alignment at least equal to the element width.
Every nonempty logical access lies wholly within its allocation.
An empty domain may have a one-past origin but admits no element load or store.
Unknown strides, negative or excessive dimensions, rank mismatches, arithmetic overflow, invalid alignment, and out-of-bounds byte intervals are typed refusals.
A widened physical access is refused even when neighboring bytes happen to be allocated.
Read aliases require consistent allocation capacity and alignment metadata.
Write aliases or non-injective writable mappings require a separate proven law and are otherwise refused.
Metadata validation grants neither pointer lifetime nor ownership nor completion permission.

## IR and admission

Device IR v2 carries `ElemType`, `NumericPolicy { version, kind }`, the layout version, and a checked per-buffer strided layout.
An IR v1 plan remains readable as rank-1 contiguous u32 under the wrapping-u32 policy.
Unsupported versions receive a typed version refusal.
Rank-2 contractions are recognized from ordinary counted loops and checked flattened indices rather than privileged fixture names.
Typed ports, affine permissions, demand, effects, physical access, and source identity must verify before a region becomes device eligible.
Target-specific restrictions may refuse an otherwise valid strided descriptor and must preserve lawful CPU fallback.
MIR annotations, admission records, generated artifacts, and cache hints bind the numeric policy, logical and physical layout, and all applicable versions.
A change to any bound identity field invalidates a cache hint before use.
Tensor cache key format 2 hashes the device IR version immediately after the cache format version, invalidating the pre-v2 format 1 recipe.

## Semantic fixtures

The committed `issue_m4_01_layout_oracle` fixtures cover contiguous, transposed, bounded-stride, empty, misaligned, overflowing, and widened accesses.
The committed `issue_m4_01_numeric_layout_contracts` and `issue_m4_01_tensor_cache` fixtures cover ordinary matrix loops, physical bounds, source identity, version refusal, and cache invalidation.
IR construction alone does not establish source or engine acceptance.
