#!/usr/bin/env bash
# Build the pinned ROS 2 Lyrical workspace prepared by setup.sh inside the
# conda-forge environment: merged install, Release, tests off, bounded and
# nice'd so other users of the host keep priority.
# Usage: build.sh ROOT [extra colcon arguments...]
set -euo pipefail
root=${1:?usage: build.sh ROOT}
shift
export MAMBA_ROOT_PREFIX=$root/mamba
cd "$root/ws"
nice -n 15 timeout 5h "$root/bin/micromamba" run -p "$root/env" env \
  MAKEFLAGS=-j4 CMAKE_BUILD_PARALLEL_LEVEL=4 \
  colcon build --merge-install --install-base "$root/ws/install" \
    --parallel-workers 8 --event-handlers console_cohesion+ \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
      "-DPython3_EXECUTABLE=$root/env/bin/python" "-DCMAKE_PREFIX_PATH=$root/env" \
    "$@"
