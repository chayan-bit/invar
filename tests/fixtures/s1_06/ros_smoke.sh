#!/usr/bin/env bash
set -euo pipefail

invar=${1:?invar binary}
run_dir=${2:?new absolute evidence directory}
image=${3:-}
expected_id=${4:-}
fixture=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$fixture/../../.." && pwd)
# shellcheck source=runner.sh
source "$fixture/runner.sh"
[[ $(uname -s) == Linux ]] || { echo 'Linux required' >&2; exit 1; }
[[ $run_dir == /* ]] || { echo 'absolute evidence directory required' >&2; exit 1; }
runner_check "$image" "$expected_id"
mkdir "$run_dir"
"$invar" system gen ros2 "$root/examples/cells/reach_place/cell-ur5e.inv" --out "$run_dir/workspace" > "$run_dir/generation.json"
"$invar" component generate "$root/examples/cells/reach_place/components/pinocchio_fk/manifest-ur5e.inv" --output "$run_dir/workspace/fk-header" > "$run_dir/fk-generation.json"
container="invar-s106-env-$$"
trap 'docker stop -t 2 "$container" >/dev/null 2>&1 || true' EXIT

docker_run() {
    run_env 2m "$container" "$run_dir/workspace:/work" "$fixture:/fixture:ro" "$root:/source:ro" -- "$1"
}
docker_run 'cmake -S /source/tests/fixtures/foreign-components -B /work/fk-build -DCMAKE_BUILD_TYPE=Release -DINVAR_GENERATED_UR5E_INCLUDE_DIR=/work/fk-header; cmake --build /work/fk-build --target pinocchio_fk_ur5e_generated --parallel 2' > "$run_dir/fk-build.log" 2>&1
docker_run 'colcon build --base-paths src --event-handlers console_direct+' > "$run_dir/build.log" 2>&1
! grep -E 'warning:|Warning C[0-9]+' "$run_dir/build.log"
private_zenoh_port "$run_dir/workspace" > "$run_dir/zenoh.txt"
docker_run 'source install/setup.bash; export RMW_IMPLEMENTATION=rmw_zenoh_cpp ZENOH_ROUTER_CHECK_ATTEMPTS=10 ZENOH_SESSION_CONFIG_URI=/work/install/invar_cell_bringup/share/invar_cell_bringup/config/rmw_zenoh/session.json5 ZENOH_ROUTER_CONFIG_URI=/work/install/invar_cell_bringup/share/invar_cell_bringup/config/rmw_zenoh/router.json5; setsid ros2 launch invar_cell_bringup cell.launch.py fk_library:=/work/fk-build/libpinocchio_fk_ur5e_generated.so fk_model:=/source/examples/cells/reach_place/assets/menagerie/universal_robots_ur5e/ur5e.xml > /work/launch.log 2>&1 & launch_pid=$!; sleep 4; ros2 bag record -s mcap -a -o /work/recording > /work/bag.log 2>&1 & bag_pid=$!; trap "kill -TERM $bag_pid 2>/dev/null || true; wait $bag_pid 2>/dev/null || true; kill -INT -- -$launch_pid 2>/dev/null || true; sleep 3; kill -KILL -- -$launch_pid 2>/dev/null || true" EXIT; sleep 2; timeout 35s python3 /fixture/ros_joint_smoke.py' > "$run_dir/motion.log" 2>&1
grep -q 'Configured and activated joint_state_broadcaster' "$run_dir/workspace/launch.log"
grep -q 'Configured and activated joint_trajectory_controller' "$run_dir/workspace/launch.log"
grep -q 'joint_motion_pass' "$run_dir/motion.log"
grep -q 'attachment_toggle_pass' "$run_dir/motion.log"
grep -q 'fk_pose_pass' "$run_dir/motion.log"
test -f "$run_dir/workspace/recording/metadata.yaml"
compgen -G "$run_dir/workspace/recording/*.mcap" >/dev/null
docker_run 'python3 /fixture/fk_mujoco_crosscheck.py /work/recording /work/src/invar_cell_bringup/config/mjcf/scene.xml /source/examples/cells/reach_place/assets/menagerie/universal_robots_ur5e/ur5e.xml /source/examples/cells/reach_place/components/pinocchio_fk/golden/panda_fk.json ur5e' > "$run_dir/fk-crosscheck.json" 2> "$run_dir/fk-crosscheck.err"
grep -q '"status": "pass"' "$run_dir/fk-crosscheck.json"
! docker inspect "$container" >/dev/null 2>&1
echo "S1-06 UR5e controller motion passed ($runner_mode): $run_dir"
