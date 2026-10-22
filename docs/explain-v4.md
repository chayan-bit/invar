# Explain JSON v4

Programs with checked spatial calls emit `invar-explain-v4` and `schema_version: 4`.
Sort-only programs emit [v3](explain-v3.md); other programs retain v1.
The v1, v2 and v3 schemas are unchanged.
A spatial program that also sorts lists its sort sites in `explicit_sorts` exactly as v3 does, including the C14 `calibration_id`.

`explicit_spatial` lists checked sites in MIR order, including unreached branches.
Each site records its function, operation, source span, `cpu-sorted-grid` default, `explicit-spatial-v1` rule, and registered `spatial-cost-unmeasured` reason.
The calibration identities are:

| Operation | Family |
| --- | --- |
| `voxel_grid_build` | `spatial.voxel_grid_build.f32` |
| `radius_neighbors` | `spatial.radius_neighbors.f32` |
| `knn` | `spatial.knn.f32` |
| `grid_excluded` | `spatial.grid_excluded.f32` |

The strict schema rejects unknown fields, operations, reasons, and mismatched operation/family pairs.
Checked sites are not execution or calibration samples.

With `--engine cuda|metal|auto`, `device_schema` is `invar-explain-device-v4`.
Reached spatial choices carry the same family identity.
Forced kernels retain their device backend, implementation, and `forced-device` reason.
Host metadata, empty operations without a launch, and explicit host sorts report `cpu-serial` with `spatial-host-work` instead of a device launch.
Unmeasured automatic spatial execution stays on CPU with no invented cost, even if the generic bandwidth model favors the GPU.
Independent benchmark qualification and the unchanged 1.10x gate remain prerequisites for automatic device placement.
