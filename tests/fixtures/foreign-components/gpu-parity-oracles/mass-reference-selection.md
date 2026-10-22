# Source-fidelity and compiled-runtime mass references

`mass_reference_selection.json` implements the previously approved distinction
for the eight historical Panda cases. It identifies exact files/fields rather
than recomputing or rounding their binary64 JSON strings. Source-fidelity uses
Pinocchio's source-inertia matrix with the independently recorded source-body
kinetic-energy reconstruction as witness. Compiled-runtime uses MuJoCo's matrix
with the separately recorded compiled-body reconstruction. Source armature is
included once in each; no fitted correction is applied.

The Rust lineage regression checks all 392 matrix entries per representation,
exact source-model/input identities, case order, finiteness, kinetic-energy
residuals and armature accounting. Its `<1e-10` witness check is the existing
attribution-script criterion, not a new scored GPU tolerance. Tests reject a
compiled matrix substituted for a source witness, wrong labels/states, NaNs,
erased failure status and double-counted armature even when the witness is also
altered to agree with the wrong matrix.

Original contract/reference/attribution files and the six failed Panda
cross-library comparisons remain byte-for-byte unchanged. This selector does
not turn those failures into cross-library passes. Benchmark acceptance status,
GPU floating-point contracts, models and numerical thresholds are unchanged.

The historical Panda selector itself performs no new library or GPU run. A
subsequent additive UR5e witness (not retained here)
recorded actual cloud x86_64 Pinocchio4.1.0/MuJoCo3.12.0 source and compiled
kinetic-energy reconstruction for all eight frozen inputs. Its source inertial
quaternion is explicitly normalized/rotated; raw matrices and fresh-to-frozen
residuals are retained. This separate evidence does not rewrite the historical
attribution or silently extend the Panda selector. The pinned ROS environment
and all original system/benchmark/GPU gates remain outstanding.
