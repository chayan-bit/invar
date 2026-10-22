# v0.2 spatial structures and neighbourhood queries

Status: reference, native/parallel CPU, CUDA and Metal execution are qualified; spatial placement is partially qualified and independent performance gates remain pending.
The lane owns `feat/v02-spatial` and depends on `feat/v02-sort` before building the sorted grid or any device path.
The common brief and the spatial assignment are the work order.
Phase 2 starts only after phase 1 passes all five engines and its benchmark gates.

## Reuse survey, before implementation

The following primary sources were inspected on 2026-10-18.
Repository pages establish candidates and licenses, not a pinned revision, benchmark result, or maintenance guarantee.
No upstream source has been copied or ported in this checkpoint.
The CUDA port input is pinned to cuNSearch commit [`4dd5598824bb7dfceae86d89758a6be74e7a9ac4`](https://github.com/InteractiveComputerGraphics/cuNSearch/tree/4dd5598824bb7dfceae86d89758a6be74e7a9ac4), inspected on 2026-10-18.
Its MIT notice is copyright 2019-present cuNSearch contributors. The inspected kernel and host pipeline use cell counts, a prefix scan, point-index reordering, and per-query cell traversal.
Reuse that organization, while replacing upstream's fixed 27-cell traversal, rounded cell mapping, strict-radius comparison, zero-distance exclusion, and clipped neighbour count with Invar's exact cell mapping, inclusive strict-f32 comparison, self inclusion, true counts, and distance/index ordering. Those upstream semantics cannot be copied unchanged.
The pinned upstream is input provenance, not a delivered CUDA port or measured baseline.


| Candidate | License and source | Fit and maintenance evidence | Decision |
| --- | --- | --- | --- |
| cuNSearch | [MIT](https://github.com/InteractiveComputerGraphics/cuNSearch/blob/master/LICENSE) | [C++/CUDA fixed-radius grid](https://github.com/InteractiveComputerGraphics/cuNSearch), based on Hoetzlein; small repository, no release freshness verified | Preferred phase-1 kernel-port candidate; pin and inspect its actual kernels before porting, preserve notices |
| FRNN | [Repository](https://github.com/lxxue/FRNN); no top-level license found in the inspected listing | CUDA/PyTorch fixed-radius grid with reusable grid and sorted results; README lists CUDA 10.2/PyTorch 1.6 and unfinished collection of all neighbours | Do not copy without verified license; algorithm comparison only |
| nanoflann | [BSD-2-Clause](https://github.com/jlblancoc/nanoflann/blob/master/COPYING) | [Maintained C++ header library](https://github.com/jlblancoc/nanoflann), exact CPU kd-tree; public license includes 2026 updates | Required external CPU baseline, not the Invar grid or the exactness oracle |
| Open3D | [MIT](https://github.com/isl-org/Open3D/blob/main/LICENSE) | [Large active CPU/CUDA project](https://github.com/isl-org/Open3D) with point-cloud and nearest-neighbour infrastructure | CUDA baseline candidate; strict-f32/tie/true-count semantics must be checked independently |
| PCL | [BSD-3-Clause](https://github.com/PointCloudLibrary/pcl/blob/master/LICENSE.txt) | [CPU point-cloud library](https://github.com/PointCloudLibrary/pcl) with voxel and neighbourhood operations | Useful algorithm reference; avoid importing its broad C++ dependency tree into the Rust compiler |
| cuBQL | [Apache-2.0 repository](https://github.com/NVIDIA/cuBQL) | Header-only CUDA BVH build/query infrastructure, CUDA 12+ documented | Phase-2 traversal reference candidate, not a substitute for the decided Karras LBVH |
| ToruNiina/lbvh | [MIT](https://github.com/ToruNiina/lbvh/blob/master/LICENSE) | [Karras 2012 implementation](https://github.com/ToruNiina/lbvh), handles duplicate Morton codes with appended index; depends on Thrust | Preferred phase-2 port reference; pin and inspect before implementation |
| kiddo | [MIT or Apache-2.0](https://github.com/sdd-org/kiddo) | [crates.io](https://crates.io/crates/kiddo); actively documented Rust kd-tree, optional SIMD and Rayon | Does not implement the decided sorted uniform grid; no new dependency |
| kdtree | [MIT or Apache-2.0](https://github.com/mrhooray/kdtree-rs) | [crates.io](https://crates.io/crates/kdtree); smaller Rust kd-tree with user distance function | Does not implement the decided grid; no new dependency |

`Cargo.lock` contains none of `kiddo`, `kdtree`, or `rayon`, and no spatial-search dependency.
Reuse `numeric_contract::F32Bits` for the independent software oracle and the existing affine ownership checker when the source surface is implemented.
Do not use a kd-tree or a library's distance arithmetic as the semantics oracle.
Select cuNSearch's grid organization for the future port, Invar sort for key ordering, and a small independent all-pairs test oracle for correctness.
The test oracle deliberately has no grid, hash, Morton key, pruning, or backend dependency.

## Decided phase-1 contract

Points and queries use separate x, y, and z `Array<f32>` owners.
`voxel_grid_build(read xs, ys, zs, cell: f32) -> VoxelGrid` copies the borrowed input into a fresh opaque affine owner with sorted cell keys and a cell-start table.
The original points may be changed or released after the call; a grid cannot retain an untracked source loan.
Keys must distinguish full signed cell coordinates; hash collisions cannot merge distinct cells.
Stable original point indices must survive sorting and are the public query indices.

`radius_neighbors(read grid, read qx, qy, qz, radius: f32, max_k: u32) -> Neighbors` returns true counts and fixed-stride indices.
The index array has `query_count * max_k` slots.
Every row stores the nearest `min(counts[q], max_k)` neighbours ordered by squared distance, then original point index.
Counts are never clipped to the stride and exceeding the stride is not a trap.

`knn(read grid, read qx, qy, qz, k: u32) -> Knn` returns exact nearest neighbours, ordered by squared distance then original point index.
Both arrays have `query_count * k` slots.
Missing slots are `u32::MAX` and positive infinity.
A real neighbour with infinite squared distance precedes a missing slot.
Zero stride produces empty result arrays, while radius queries must still compute every true count.
Queries include the source point itself if it is a neighbour; there is no implicit self exclusion.

For each candidate, subtract in f32, round each product independently, then compute `((dx * dx) + (dy * dy)) + (dz * dz)` with a rounding after each addition.
No FMA, reassociation, f64 intermediate, flush-to-zero, approximate reciprocal, or approximate search is permitted.
Finite coordinates can still produce an infinite squared distance; this is an ordinary distance and not an overflow trap.

## Binding cross-lane rulings

The binding authority is section 4.4 of the robotics-language roadmap at `a41b6c7f` ([rulings](v02-robotics-language.md#44-rulings-2026-10-18)).
These replace the provisional checker/oracle conventions in the earlier checkpoint.
C1 replaces tuple query results with built-in affine records `Neighbors { counts: Array<u32>, idx: Array<u32> }` and `Knn { idx: Array<u32>, d2: Array<f32> }`.
C2 renames the opaque owner to `VoxelGrid`; source code cannot construct it, project its fields, or redefine it.
C5 keeps every spatial coordinate, cell width, radius, and distance f32-only.
C6 excludes a point with any non-finite coordinate and preserves the original index of every included point; `grid_excluded(read grid) -> u32` reports the excluded count.
A non-finite query returns no neighbours; kNN fills its slots with `u32::MAX` and positive infinity.
C7 requires exact `floor(coord / cell)` before conversion to i32, with `INTEGER_OVERFLOW` for an out-of-range index or a non-finite/nonpositive build cell width.
Excluded points never enter cell conversion, so their non-finite coordinates do not trap.
C8 requires inclusive `d2 <= radius * radius` using strict f32 arithmetic, with no matches for a negative or NaN radius; positive infinity is allowed and negative zero is not negative.
C9 requires `INDEX_OUT_OF_BOUNDS` for coordinate-array shape mismatch and `INTEGER_OVERFLOW` for u32 overflow of query count times output stride; memory-budget exhaustion remains a tool error.
The independent oracle uses existing `num-bigint` and `num-rational` dependencies to model exact cell division, never a rounded f32 quotient.
Its small-fixture work/memory limits remain helper limits, not new language traps.
Unused radius slots remain `u32::MAX` as the lane's deterministic padding convention.

C13 requires large benchmarks through device harnesses or bench binaries until B2 (#37) lands, rather than `invar run`.
C14 requires calibration ids `spatial.<op>.f32`, including `spatial.voxel_grid_build.f32`, `spatial.radius_neighbors.f32`, and `spatial.knn.f32`.
Any future explain reason must be registered in the AI-01 registry before it is emitted; this checker-only checkpoint emits no spatial placement reason and has no calibration record.
No benchmark or registry entry is fabricated before executable paths exist.

## Timing acceptance

Every timing/benchmark number records the host's 1-minute load beside it and counts only when that load is below 8; rerun overloaded measurements later.
Correctness evidence remains attributed to its original host and is not performance evidence.

## Checker and language integration

Shared intrinsic signatures and the built-in result-record fields live in `src/spatial.rs`.
The bounded tuple parser is removed; ordinary named record bindings, projections, and explicit array-field moves reuse the existing parser and affine checker.
Source record/variant declarations cannot redefine `VoxelGrid`, `Neighbors`, or `Knn`.
Grid parameters permit only `read` or `take`; query-result records use ordinary affine record rules.
All intrinsic reads acquire ordinary loans left to right and keep them until scalar argument evaluation finishes, so a nested call cannot consume or mutate an already borrowed owner.
Real-CLI tests cover signatures, opacity, result types, record moves, malformed tuple syntax, and held-loan conflicts.
Reference evaluation builds an owned sorted grid using the merged stable radix word runtime, with a full signed `(x,y,z)` cell key and occupied-cell ranges. Exact integer significand/exponent division implements C7 independently of the rational test oracle. Queries exhaustively visit occupied cells, preserving strict-f32 underflow and tie cases. The grid owns its point snapshot and participates in ordinary moves, loans, cleanup, and aggregate resource accounting.
MIR lowering remains explicitly unavailable until the next checkpoint; native/parallel/device routes, planner, and receipts still require integration. Exhaustive reference traversal is a correctness baseline, with no speed claim.

## Exactness traps for the grid port

A fixed 27-cell query is insufficient when radius exceeds the cell width.
Even radius zero can include distinct points whose squared differences round to zero.
For example, `dx = 2^-80` has squared distance zero in strict f32, while a cell width of `2^-100` separates the points by 1,048,576 cells.
Pruning requires conservative bounds for the actual rounded distance, including underflow, rounding, and overflow; a real-Euclidean bound alone is not sufficient.
Exact kNN cannot stop at the first occupied shell and cannot terminate on an equal lower bound without considering index ties.
For huge sparse extents, enumerate occupied cells or use another proven exhaustive traversal of the same grid rather than an unbounded empty-shell search.

### Conservative coordinate bound, not an implemented pruning path

For finite inputs and a finite nonnegative squared threshold `b`, any accepted rounded distance has each independently rounded squared component at most `b`, because nearest-even addition of nonnegative operands is monotone.
Consequently the exact square of each rounded subtraction is below `next_up(b)`.
Choose a representable power of two `B` whose mathematical square is at least `next_up(b)`; then the exact coordinate difference is bounded by `next_up(B)`, accounting for subtraction rounding as well as multiplication rounding.
For `b = 0`, `B = 2^-74` is conservative and includes the `2^-80` counterexample; a zero geometric search extent is incorrect.
An eventual implementation may round `q - next_up(B)` downward and `q + next_up(B)` upward before converting these bounds with the same monotone cell-key rule as construction.
If `next_up(b)` is infinite, or key/range arithmetic cannot be represented safely, scan occupied cells rather than wrapping or truncating a bound.
An exact kNN stop needs to cover every point with rounded distance at most the current worst retained distance, including equal-distance index improvements.
This is a proof obligation for the port, not evidence that a kernel or a key conversion already satisfies it; differential boundary/underflow tests remain required.

## Acceptance and sequence

1. Execute the independent oracle fixtures on the approved CPU host, including the FMA/reassociation, clipped-count, and tie-order mutations.
2. Apply the binding rulings, qualify the checker-only surface, and implement reference execution through the CLI.
3. Merge the pushed sort surface with a real merge, then build sorted cell keys and cell starts using that surface.
4. Implement serial native and parallel CPU paths, retaining all-pairs differential cases for random, clustered, duplicate, all-identical, empty, subnormal, boundary, and overflow inputs.
5. Implement CUDA and Metal ports with per-lowering differential and mutation checks, placement/explain decisions, and truthful receipts on all five engines.
6. Benchmark medians against nanoflann, Open3D or cuNSearch-class CUDA, and a handwritten Metal grid; retain raw samples and provenance, and refuse losing or unmeasured device sizes through the cost model.
7. Add the executable `examples/pointcloud_normals.inv` after kNN and f32 math are callable, record a real VHS demo, then qualify phase 1 before LBVH work.

Focused checker/oracle coverage and CPU tests are in the spatial test suites.
No timings, engine differential pass, or phase-1 acceptance is claimed.

## Executable CPU checkpoint

Reference is accepted at f08dfbb8 with full Debug/Release evidence. MIR, native-serial and parallel CPU now execute all four operations with matching strict values, C6–C9 traps, spans, fuel and cleanup. Parallel rows are independent, joined in source order before publication; worker counts 1/2/4 produce identical outputs. The standalone Rust native host shares exact cell/radix/numeric helpers and explicitly refuses Grid shared-library exports. No new dependency. Queries still exhaust occupied cells and have no performance admission. Physical devices, spatial placement, baselines and the executable VHS remain pending gates.


## Device, placement and real scan checkpoints

CUDA and Metal both pass physical oracle comparisons through 2^22 points, including strict f32 edge cases and restored device mutation runs.
Explain v3 supplies spatial family ids and registered reasons while preserving frozen v1/v2 schemas.
Placement checks separate accepted Metal checks from pending Linux/CUDA checks.
A real-scan run executes a PCL radius query on Metal and returns 264 total neighbors.
Metal runs a compiled-ahead library from `invar build PATH --target metal --output FILE`; a spatial program's library carries the sort kernels used by grid construction, the three spatial kernels and the strict f32 probe that gates the query kernel's normal window.
These later receipts supersede historical pending-device statements above.
Independent CPU/CUDA/Metal baseline comparisons, the unchanged 1.10x gate, and complete phase-1 acceptance remain pending.
Automatic spatial execution stays on CPU while unmeasured.
