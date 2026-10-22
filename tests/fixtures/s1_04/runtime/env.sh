#!/usr/bin/env bash
set -euo pipefail

invar=${1:?invar binary}
run_dir=${2:?new evidence directory}
image=${3:?pinned ROS image}
expected_id=${4:?image ID}
fixture=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $(uname -s) == Linux && $(uname -m) == aarch64 ]] || { echo 'ARM64 Linux required' >&2; exit 1; }
[[ $run_dir == /* ]] || { echo 'absolute evidence directory required' >&2; exit 1; }
[[ $(docker image inspect --format '{{.Id}}' "$image") == "$expected_id" ]] || { echo 'ROS image ID differs' >&2; exit 1; }
mkdir "$run_dir"
printf 'host_arch=%s\nimage_tag=%s\nimage_id=%s\n' "$(uname -m)" "$image" "$expected_id" > "$run_dir/environment.txt"
container="invar-s1-04-env-$$"
trap 'docker stop -t 2 "$container" >/dev/null 2>&1 || true' EXIT

docker_run() {
  local limit=$1 command=$2
  timeout --kill-after=10s "$limit" docker run --rm --name "$container" \
    --network none --cpus 3 --memory 10g --memory-swap 10g --shm-size 256m --pids-limit 512 \
    -v "$workspace:/ws" -v "$fixture:/fixture:ro" -v "$run_dir:/evidence" -w /ws \
    "$image" bash -lc "$command"
}
record_generation() {
  python3 - "$workspace" "$1" >> "$run_dir/generation-manifests.txt" <<'PY'
import hashlib, json, pathlib, sys
root, label = pathlib.Path(sys.argv[1]), sys.argv[2]
files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
         for path in root.rglob('*') if path.is_file()}
payload = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
graph = json.loads((root / 'src/invar_cell_bringup/config/graph.json').read_text())
print(label, 'ir_sha256=' + graph['ir_sha256'], 'file_set_sha256=' + hashlib.sha256(payload).hexdigest(), 'files=' + str(len(files)))
PY
}

workspace="$run_dir/cell12"
"$invar" system gen ros2 "$fixture/../../s1_03/cell-12.inv" --out "$workspace" > "$run_dir/cell12-generation.json"
record_generation cell12
docker_run 1m 'source /opt/ros/lyrical/setup.bash; echo ROS_DISTRO=$ROS_DISTRO; dpkg-query -W ros-lyrical-rmw-zenoh-cpp ros-lyrical-ros2-control ros-lyrical-mujoco-ros2-control ros-lyrical-pinocchio ros-lyrical-generate-parameter-library; python3 -c "import mujoco; print(\"mujoco-python\", mujoco.__version__)"' >> "$run_dir/environment.txt"
docker_run 8m 'source /opt/ros/lyrical/setup.bash; colcon build --event-handlers console_direct+ --parallel-workers 3' > "$run_dir/cell12-build.log" 2>&1
! grep -E 'warning:|Warning C[0-9]+' "$run_dir/cell12-build.log"
python3 - "$workspace" > "$run_dir/hashes.log" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
hashes = sorted((root / 'src/invar_cell_interfaces/hashes').glob('*.json'))
assert hashes, 'no generated interface hashes'
for path in hashes:
    expected = json.loads(path.read_text())['hash_string']
    built = root / 'build/invar_cell_interfaces/rosidl_generator_type_description/invar_cell_interfaces/msg' / path.name
    actual = json.loads(built.read_text())['type_hashes'][0]['hash_string']
    assert actual == expected, (path.name, expected, actual)
    print(path.name, actual)
PY
docker_run 1m 'source /opt/ros/lyrical/setup.bash; source /ws/install/setup.bash; python3 -m py_compile /ws/src/invar_cell_bringup/launch/cell.launch.py; ros2 launch invar_cell_bringup cell.launch.py --show-args' > "$run_dir/cell12-launch-parse.log" 2>&1

workspace="$run_dir/runtime"
"$invar" system gen ros2 "$fixture/cell.inv" --out "$workspace" > "$run_dir/runtime-generation.json"
record_generation runtime
cp "$fixture/mock.urdf" "$workspace/src/invar_cell_bringup/config/ros2_control.urdf"
docker_run 8m 'source /opt/ros/lyrical/setup.bash; colcon build --event-handlers console_direct+ --parallel-workers 3; cc -shared -fPIC -I/ws/src/invar_planar2_node/include/invar_planar2_node -I/fixture /fixture/bridge.c /fixture/planar2.c -o /ws/libplanar2.so -lm' > "$run_dir/runtime-build.log" 2>&1
! grep -E 'warning:|Warning C[0-9]+' "$run_dir/runtime-build.log"
docker_run 1m 'source /opt/ros/lyrical/setup.bash; source /ws/install/setup.bash; python3 -m py_compile /ws/src/invar_cell_bringup/launch/cell.launch.py; ros2 launch invar_cell_bringup cell.launch.py --show-args' > "$run_dir/runtime-launch-parse.log" 2>&1
mkdir "$run_dir/smoke"
docker_run 2m 'bash /fixture/smoke.sh /ws /evidence/smoke planar2_library:=/ws/libplanar2.so planar2_proximal_m:=0.4 planar2_distal_m:=0.3' > "$run_dir/smoke-driver.log" 2>&1

workspace="$run_dir/range"
"$invar" system gen ros2 "$fixture/../range-valid.inv" --out "$workspace" > "$run_dir/range-generation.json"
record_generation range
docker_run 8m 'source /opt/ros/lyrical/setup.bash; colcon build --event-handlers console_direct+ --parallel-workers 3' > "$run_dir/range-build.log" 2>&1
! grep -E 'warning:|Warning C[0-9]+' "$run_dir/range-build.log"

workspace="$run_dir/separate"
"$invar" system gen ros2 "$fixture/cell-separate.inv" --out "$workspace" > "$run_dir/separate-generation.json"
record_generation separate
docker_run 8m 'source /opt/ros/lyrical/setup.bash; colcon build --event-handlers console_direct+ --parallel-workers 3' > "$run_dir/separate-build.log" 2>&1
! grep -E 'warning:|Warning C[0-9]+' "$run_dir/separate-build.log"
echo "S1-04 ROS environment acceptance passed: $run_dir"
