#!/usr/bin/env bash
# Route 1 (no root, no containers): a project-local micromamba conda-forge
# environment plus a ROS 2 Lyrical source workspace pinned to the release
# tags of the rosdistro commit that matches the s104 image's 2026-09-18 pins.
# RoboStack has no Lyrical channel, so ROS is built from source (build.sh).
# Usage: setup.sh ROOT   (ROOT is created; it must not exist yet)
set -euo pipefail
root=${1:?usage: setup.sh ROOT}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ ! -e $root ]] || { echo "refusing to reuse $root" >&2; exit 1; }
[[ $(uname -m) == x86_64 ]] || { echo 'x86_64 host required' >&2; exit 1; }
mkdir -p "$root"/{bin,downloads,ws/src}

# 1. Static micromamba 2.9.0 from conda-forge, checked by SHA-256.
archive=$root/downloads/micromamba-2.9.0-0.tar.bz2
curl -fsSL --retry 3 -o "$archive" \
  https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.9.0-0.tar.bz2
echo "8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd  $archive" | sha256sum -c -
tar -xjf "$archive" -C "$root" bin/micromamba
export MAMBA_ROOT_PREFIX=$root/mamba
mm=$root/bin/micromamba

# 2. conda-forge environment, then its explicit lock (URLs and MD5 per package).
timeout 60m "$mm" create -y -p "$root/env" --override-channels -c conda-forge \
  --file "$here/env-spec.txt"
"$mm" env export -p "$root/env" --explicit --md5 > "$root/lock-linux-64.txt"

# 3. rosinstall_generator (not on conda-forge), hash-pinned, inside the env.
timeout 10m "$mm" run -p "$root/env" python -m pip install --no-deps --require-hashes \
  -r "$here/pip-tools.txt"

# MuJoCo Python and NumPy wheels exactly as in the image (hash-pinned).
timeout 10m "$mm" run -p "$root/env" python -m pip install --require-hashes \
  -r "$here/../mujoco-requirements.txt"

# 4. Workspace sources at the release tags of rosdistro 8bc9618e (2026-09-18).
# The upstream index names the live lyrical-cache.yaml.gz, which
# rosinstall_generator always prefers, so a pinned index URL alone resolves
# today's releases. Build the cache from the pinned commit instead and point a
# local copy of the index at it.
rosdistro_commit=8bc9618ed66325b4f0a63699369048427e104393
git init -q "$root/rosdistro"
git -C "$root/rosdistro" fetch -q --depth 1 https://github.com/ros/rosdistro.git "$rosdistro_commit"
git -C "$root/rosdistro" checkout -q "$rosdistro_commit"
mkdir -p "$root/rosdistro-cache"
(cd "$root/rosdistro-cache" && timeout 60m "$mm" run -p "$root/env" \
  rosdistro_build_cache "file://$root/rosdistro/index-v4.yaml" lyrical)
sed -i "s#http://repo.ros2.org/rosdistro_cache/lyrical-cache.yaml.gz#file://$root/rosdistro-cache/lyrical-cache.yaml.gz#" \
  "$root/rosdistro/index-v4.yaml"
grep -q "file://$root/rosdistro-cache/lyrical-cache.yaml.gz" "$root/rosdistro/index-v4.yaml"
export ROSDISTRO_INDEX_URL=file://$root/rosdistro/index-v4.yaml
mapfile -t packages < <(grep -v '^#' "$here/ros-packages.txt" | sed '/^$/d')
mapfile -t excluded < <(grep -v '^#' "$here/ros-excluded.txt" | sed '/^$/d')
"$mm" run -p "$root/env" rosinstall_generator --rosdistro lyrical --deps --format repos \
  --exclude "${excluded[@]}" -- "${packages[@]}" > "$root/ws/lyrical.repos"
# The s104 image pins these releases; a drifted cache must fail here, not in colcon.
for pin in mujoco_ros2_control/mujoco_ros2_control/0.1.2-1 ros2_control/hardware_interface/6.10.1-1; do
  grep -qx "    version: release/lyrical/${pin#*/}" "$root/ws/lyrical.repos" \
    || { echo "lyrical.repos does not pin $pin" >&2; exit 1; }
done
timeout 30m "$mm" run -p "$root/env" vcs import --shallow --retry 3 --input "$root/ws/lyrical.repos" "$root/ws/src"
timeout 10m "$mm" run -p "$root/env" vcs import --retry 3 --input "$here/extra.repos" "$root/ws/src"
"$mm" run -p "$root/env" vcs export --exact "$root/ws/src" > "$root/ws/lyrical-exact.repos"
# 5. Environment identity: one hash over the resolved inputs and this recipe.
(cd "$root" && sha256sum lock-linux-64.txt rosdistro-cache/lyrical-cache.yaml \
  ws/lyrical.repos ws/lyrical-exact.repos) > "$root/environment-identity.txt"
(cd "$here" && sha256sum setup.sh build.sh env-spec.txt pip-tools.txt ros-packages.txt \
  ros-excluded.txt extra.repos ../mujoco-requirements.txt) >> "$root/environment-identity.txt"
cat "$root/environment-identity.txt"
echo "route1 sha256:$(sha256sum < "$root/environment-identity.txt" | cut -d' ' -f1)"
