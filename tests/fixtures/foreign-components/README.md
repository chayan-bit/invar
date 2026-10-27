# Foreign component fixtures

These are fixture APIs and independent FK oracles for S1-02/S1-06. `pinocchio_fk/abi_reference.h` is the hand-authored C reference ABI approved by the main integration task, **not** compiler-generated evidence. The main task owns the canonical S1-11 layouts, S1-02 generated header/binding, and its equality check against this reference before harness acceptance. The planar API and Pinocchio CLI remain private tests.

## Planar C fixture

`bash tests/fixtures/foreign-components/planar2/run.sh NEW_OUTPUT_DIR` builds the C11 shared library, regenerates and byte-compares seven Python `math` goldens (seed 20260927), then tests lifecycle order, joint count and input errors, unchanged output on failure, and FK values. It needs only a C compiler and Python 3.

## Pinocchio Panda and UR5e core

The committed `pinocchio_goldens.json` comes from `generate_pinocchio_goldens.py`, which uses Pinocchio 4.1.0's public Python binding in a separate process from the C++ implementation. It verifies Menagerie commit `c96a32d28fb5da84da38c1da4d749e7a13212855`, clean checkout, model/license hashes, finite joint limits, and seed 20260927. It records home plus three sampled valid configurations for each arm. Pose means tool in base, metres; quaternion `wxyz` is private to this oracle and is sign canonicalized. The Panda `link0` to `hand` and UR5e `base` to `attachment_site` frames are explicit in the JSON.

The pinned CPU container and model preparation are defined by the files under `examples/cells/env/` (`Containerfile`, `userspace/`). With those available on Linux, use a fresh output directory. Run the generator and comparison in **separate** bounded container processes; the example below assumes `IMAGE` is the locally built pinned image and `MODELS` is the pinned sparse checkout. Acquire the host CPU window first and check memory headroom. The commands keep each run to 10 GiB, no swap, four CPUs, and 20 minutes, with no network:

```sh
WORK=$(mktemp -d /tmp/invar-s1-foreign-fixtures-XXXXXX)
mkdir -p "$WORK/out" "$WORK/tests/fixtures" "$WORK/examples/cells/reach_place/components"
cp -R tests/fixtures/foreign-components "$WORK/tests/fixtures/"
cp -R examples/cells/reach_place/components/pinocchio_fk "$WORK/examples/cells/reach_place/components/"
IMAGE=invar-robotics-env:YOUR_PINNED_IMAGE
MODELS=/absolute/path/to/pinned/mujoco_menagerie
systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 \
  timeout --kill-after=10s 20m docker run --rm --network none --cpus 4 \
  --memory 10g --memory-swap 10g -v "$WORK:/work" -v "$MODELS:/models:ro" \
  "$IMAGE" bash -lc 'source /opt/ros/lyrical/setup.bash && python3 /work/tests/fixtures/foreign-components/generate_pinocchio_goldens.py /models /work/out/pinocchio_goldens.json'
cmp tests/fixtures/foreign-components/pinocchio_goldens.json "$WORK/out/pinocchio_goldens.json"
systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 \
  timeout --kill-after=10s 20m docker run --rm --network none --cpus 4 \
  --memory 10g --memory-swap 10g -v "$WORK:/work" -v "$MODELS:/models:ro" \
  "$IMAGE" bash -lc 'source /opt/ros/lyrical/setup.bash && cmake -S /work/tests/fixtures/foreign-components -B /work/out/build -DCMAKE_BUILD_TYPE=Release && cmake --build /work/out/build -j2 && python3 /work/tests/fixtures/foreign-components/test_pinocchio_exports.py /work/out/build/libpinocchio_fk_reference.so && /work/out/build/pinocchio_abi_layout && python3 /work/tests/fixtures/foreign-components/test_pinocchio_goldens.py /work/out/build/pinocchio_fk_cli /work/out/pinocchio_goldens.json /models && python3 /work/tests/fixtures/foreign-components/test_pinocchio_abi.py /work/out/build/libpinocchio_fk_reference.so /work/out/pinocchio_goldens.json /models && bash /work/tests/fixtures/foreign-components/planar2/run.sh /work/out/planar'
```

`CMakeLists.txt` requires exact Pinocchio 4.1.0. The C++ core and C ABI each compare eight poses to 1e-12 m and 1e-12 rad. The Linux export check allows only the five agreed C symbols. The C11 test checks layouts and symbol linkage; the ABI test checks lifecycle order, call-scoped path input, null/count/nonfinite/out-of-limit rejection, and unchanged output on failure. These tests do not replace the Invar-generated contract harness.

## Separate-process reference fixture

On Linux, `bash tests/fixtures/foreign-components/process/run.sh NEW_OUTPUT_DIR` builds the hand-authored C client and launches the deterministic Python child over a private Unix socket. It needs only a C11 compiler, Python 3 and `nm`; no robotics stack or GPU. The `INVP` v1 length-prefixed wire contract and C status mapping were approved by the main integration task. The direct test covers lifecycle, exact five-symbol exports, finite/count/path bounds, malformed and truncated frames, killed/hung child, unchanged pose on failure and exact child reaping. This is a joint-to-pose transport fixture, **not** a cuRobo planner, manifest binding or generated S1 harness.

## cuRobo planner launch reference

`bash tests/fixtures/foreign-components/curobo_planner/run.sh NEW_OUTPUT_DIR` compiles the hand-authored `invar_planner_v1` C client and tests it against an explicit deterministic fake child. It needs a C11 compiler, pthreads and Python 3; it does not import cuRobo or use a GPU. The production child is `examples/cells/reach_place/components/curobo_planner/planner_server.py` in the pinned cuRobo environment.

`curobo_planner/prepare_models.py --arm panda|ur5e` verifies pinned source bytes and emits one arm's private planner config, scene, and FK-verified URDFs from explicit inputs.
`curobo_planner/audit_panda_spheres.py` shows the bundled Franka spheres leave accepted-mesh vertices uncovered, so the Panda config instead uses `curobo_planner/panda_spheres.json`, fitted from the vendored Menagerie collision meshes by `fit_panda_spheres.py` (cuRobo 0.8.0, CUDA) and proven by `check_panda_spheres.py` (within 2 mm, link0 clear of the table); the real adapter accepts only that proven cover's digest.
