# GNET-05 robotics corpus

Ordinary Invar source for the robotics workload categories of the general-net GPU track.
Every file here except the Rust modules is generator output.
Tests: `tests/gnet05_corpus/issue_gnet_05_robotics.rs` and `tests/gnet05_corpus/issue_gnet_05_robotics_array.rs`.
Handwritten baselines (comparison D): `tools/bench/baselines/{robotics-kernels.h,metal-robotics-handwritten.mm,cuda-robotics-handwritten.cu,robotics-baselines.py}`.
Calls per timing sample, calibrated per case to 5 ms of GPU time: `tools/bench/baselines/robotics-batch-metal.json` (Apple M4) and `tools/bench/baselines/robotics-batch-cuda.json` (RTX 5070 Ti).

## Files

| File | Content |
| --- | --- |
| `model.rs` | MJCF loader for the S1-05/S1-06 Panda and UR5e models (class defaults resolved, SHA-256 pinned) |
| `arm.rs` | f32 kinematic, inertial and collision tables of one arm |
| `emit.rs` | Invar source emitters: input streams, Cephes sine and cosine, Newton root, FK chain in three layouts, Jacobian, mass matrix, capsule distance |
| `families.rs` | The eight families and their entry points |
| `corpus.rs` | Seeded generator `gnet05-robotics-gen-v1`, held-out split, manifest, baseline tables |
| `check64.rs` | Independent binary64 kinematics and exact f32 replays of the input streams |
| `array.rs` | Generator `gnet05-robotics-array-gen-v1` of the host-array variants and their host-side input expansion |
| `programs/*.inv` | 48 programs, entry `run(seed, n[, m])` returning `Array<f32>` |
| `manifest.json` | Per-program provenance, math constants as f32 bits, model pins, limitations, pinned held-out digest |
| `validation.json` | Per input: reference steps, output words and SHA-256; independent-check maximum error and tolerance |
| `array/manifest.json`, `array/programs/*.inv` | 48 host-array variants, one per seed program (same split and inputs), with their signature and the stack they require |
| `array/validation.json` | Per input of each variant: reference result, steps, output words and the SHA-256 of the seed-program output prefix (Release run on the merged stack) |

Regenerate with `INVAR_GNET05_WRITE=1 cargo test --release --locked --test gnet05_corpus issue_gnet_05_robotics::` (about 23 minutes on a 32-core x86_64 host; only a Release run writes `validation.json`, because the scored inputs run there).
`tools/bench/baselines/robotics-tables.h` is written by the same run.

## Composition

Eight families, 6 programs each, 3 held out per family (24 tuning, 24 held out).
Each family draws two arms or transform modes and three layouts of the same math, and the test proves that the layouts compute identical bits.

| Family | Output per configuration | Arms and transforms | Oracle inputs | Scored inputs |
| --- | --- | --- | --- | --- |
| fk | 12 floats (rotation and translation) per body | Panda, UR5e; angles through sine and cosine | n = 1, 2 | n = 1, 32, 256, 4096 |
| fkpre | as fk | Panda, UR5e; rational precomputed-transform control | n = 1, 2 | n = 1, 32, 256, 4096 |
| jacobian | 6 x joints geometric Jacobian of the tool point | Panda, UR5e | n = 1, 2 | n = 1, 32, 256, 4096 |
| mass | joints x joints mass matrix (CRBA through Jacobians) | Panda, UR5e | n = 1, 2 | n = 1, 32, 256, 4096 |
| collision | signed distance of each UR5e primitive to each obstacle sphere | UR5e; trigonometric and rational | (1, 2), (2, 3) | (1, 16), (64, 16), (1024, 16) |
| rollout | tool-target cost and joint state per step of an integrated control sequence | Panda, UR5e | (1, 2), (2, 3) | n x horizon in {1, 32, 256} x {64, 256} |
| pointcloud | count, compacted points, neighbour counts | synthetic seeded cloud; fused, staged and scanned variants | (16, 4), (64, 8) | (1024, 64), (65536, 256) |
| pipeline | per-primitive clearance and ordered clearance cost | UR5e FK, then sensor cloud transform, crop and compaction | (1, 16), (2, 24) | (1, 256), (64, 256), (1024, 256) |

The FK layouts are a parent table walk, a fully unrolled chain and per-segment functions; the point-cloud variants are fused, staged and scanned.
Full angle-driven cases charge trigonometry in source (Cephes `sinf` and `cosf` with Cody-Waite reduction); `fkpre` and the rational collision and pipeline programs are the separate precomputed-transform controls.

## Math and oracles

The language is strict IEEE binary32 in source order with no contraction, so every output is a frozen bit pattern.
Each oracle input is checked three ways on the full output: the CPU net oracle equals the reference evaluator bit for bit, and an independent binary64 computation (`check64.rs`, kinematics from the MJCF model) agrees within a frozen tolerance.
Scored inputs record the reference output's SHA-256 and word count; every baseline must match them bit for bit.

| Quantity | Tolerance (absolute) | Largest observed error (families) |
| --- | --- | --- |
| frame | 2e-5 | 1.9e-7 (fk), 1.6e-7 (fkpre) |
| Jacobian | 2e-5 | 1.6e-7 |
| mass matrix | 1e-4 | 5.6e-7 |
| distance | 5e-5 | 2.9e-7 (collision), 7.5e-8 (pipeline) |
| rollout cost | 1e-4 | 3.4e-7 |
| point cloud | exact replay | 0 |
| binary64 model against the Pinocchio fixtures | 1e-9 | within tolerance (asserted) |

## Held-out discipline

`corpus::held_out` builds the same `HeldOut` guard as the dynamic corpus over the robotics held-out sources, and the digest is pinned in `manifest.json`.
A renamed or recommented held-out program is refused by `admit_for_tuning`, and `tuning_programs` is the only robotics view a tuning path may consume.
The test also proves that the compiler's device admission and the emitted CPU code ignore names, comments and source hashes, and that no compiler source names the corpus.

## Host-array variants

`array/` holds one variant per seed program, generated next to the seed programs, which stay byte-identical.
A variant takes its bulk inputs as host buffers and writes the seed program's output into a host buffer, returning a count:

| Family | Signature | Returns |
| --- | --- | --- |
| fk, fkpre, jacobian, mass | `a: read Array<f32>, out: mut Array<f32>` | configurations n |
| collision | `a, obs: read Array<f32>, out: mut Array<f32>` | configurations n |
| rollout | `q0, u, target: read Array<f32>, out: mut Array<f32>` | configurations n |
| pointcloud | `fr, raw: read Array<f32>, m: u32, out: mut Array<f32>` | kept point count |
| pipeline | `a, fr, raw: read Array<f32>, out: mut Array<f32>` | configurations n |

`array::expand` replays the seed program's streams on the host in exact binary32, so for the same `(seed, sizes)` input the variant's output equals the seed program's output bit for bit (the point cloud: the returned prefix, zero after).
The variants keep the seed programs' split, inputs and oracle discipline: on every oracle input the CPU net oracle equals the reference (returned value and the whole output buffer), and on every oracle and scored input the output prefix equals the seed program's frozen digest; `array/validation.json` freezes the results.
Their held-out guard is the robotics one over the variants' sources (pinned digest `9e48b08a...e1b4`), disjoint from the seed guard.
Host-array entry parameters landed on `feat/v01-gnet-f32-entry` (ABI13), so the manifest marks every variant `requires: abi13-host-arrays`: the files and held-out tests run on every stack, and the execution and device census tests are ignored here and run with `--ignored` on a stack that contains that branch.

## Limitations

- Panda collision geometry is meshes and fingertip pads; the vendor sphere sets under-cover the meshes and are not in the repository, so the collision and pipeline families use the UR5e only, on its nine exact capsule and cylinder primitives (no sphere envelopes).
- The seed programs expand their bulk inputs (angles, obstacles, controls, points) in source from a seed by a frozen 32-bit LCG; the expansion is charged to every compared path, the baselines included. The host-array variants take the same data as `read Array<f32>` buffers instead, and run only on a stack with host-array entry parameters.
- The point cloud is synthetic and seeded, not a public dataset.
- The rollout computes the cost of a given control sequence; it does not optimize it.
- Rational transforms are a control for trigonometry cost, not a model of real sensors.
- The mass matrix is computed through Jacobians (the CRBA identity) and compared within tolerance, not bit for bit, against binary64.
- Panda fingers are fixed at 0.04 m.

## Compiler admission today

On this branch's base (the seeded GNET-05 corpus) device admission refuses all 48 seed programs (`unsupported_type`: f32 arrays).
On the stack with f32 arrays, host-array entries and sine and cosine (the compiled GNET device path for `sin`, `cos` and `sqrt`, measured on a local merge):
- the seed programs are refused as `owner_entry_result`, because a device entry may not return an owned `Array<f32>` (at the corpus limits 42 of them are refused earlier, as `compiler_structural_budget`);
- the host-array variants are admitted for the 6 point-cloud programs at the corpus limits, and for 18 of 48 (12 of the 14 table-walk layouts of the other families, and every point cloud) at wide compiler limits, with or without speculation slots; the other 30 are refused as `compiler_structural_budget`.

The CPU owned-numeric compiler admits 19 of 48 seed programs with wide compiler limits; the other 29 are refused by the 1 MiB cap on emitted kernel text, which no configured limit lifts.
The variants have no CPU owned-numeric path (host-array parameters are device and oracle only).
The limits report in the evidence gives the file and line of each limit, the values each layout needs and what raising them would take.

## Prior art

- cuRobo v0.8.0 (Apache-2.0, commit `4ea77366`): batched FK with four lanes per configuration and the `A * Rot(axis, q)` association, sphere and capsule distance; the baselines follow its kernel layout.
- GRiD and Pinocchio: the CRBA mass matrix and the Panda and UR5e reference fixtures in `tests/fixtures`.
- Cephes: the single-precision sine and cosine polynomials and the Cody-Waite reduction constants.
