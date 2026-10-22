#!/usr/bin/env bash
# S1-04 runtime smoke on the S1-06 cell with mujoco_ros2_control, both arms:
# zero-warning build, both controllers active, /joint_states within 5 percent
# of the declared 100 Hz over 10 s, MCAP written and clean launch shutdown.
# Runs in the pinned image or the Route 1 user-space environment (runner.sh).
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
printf 'host_arch=%s\nrunner=%s\nimage_tag=%s\nimage_id=%s\nenv_root=%s\n' "$(uname -m)" "$runner_mode" \
    "$image" "$expected_id" "${INVAR_S1_06_ENV_ROOT:-}" > "$run_dir/environment.txt"
container="invar-s106-cell-smoke-$$"
trap 'docker stop -t 2 "$container" >/dev/null 2>&1 || true' EXIT

run() {
    local limit=$1 workspace=$2 command=$3
    run_env "$limit" "$container" "$workspace:/ws" "$root/tests/fixtures:/fixtures:ro" "$root:/source:ro" \
        "$run_dir:/evidence" -- "$command"
}

for arm in panda ur5e; do
    if [[ $arm == panda ]]; then
        source=cell.inv manifest=manifest.inv define=INVAR_GENERATED_INCLUDE_DIR target=pinocchio_fk_generated
        model=franka_emika_panda/panda.xml
    else
        source=cell-ur5e.inv manifest=manifest-ur5e.inv define=INVAR_GENERATED_UR5E_INCLUDE_DIR target=pinocchio_fk_ur5e_generated
        model=universal_robots_ur5e/ur5e.xml
    fi
    workspace="$run_dir/$arm"
    "$invar" system gen ros2 "$root/examples/cells/reach_place/$source" --out "$workspace" > "$run_dir/$arm-generation.json"
    "$invar" component generate "$root/examples/cells/reach_place/components/pinocchio_fk/$manifest" --output "$workspace/fk-header" > "$run_dir/$arm-fk-generation.json"
    run 8m "$workspace" "cmake -S /source/tests/fixtures/foreign-components -B /ws/fk-build -DCMAKE_BUILD_TYPE=Release -D$define=/ws/fk-header; cmake --build /ws/fk-build --target $target --parallel 3" > "$run_dir/$arm-fk-build.log" 2>&1
    run 10m "$workspace" 'colcon build --base-paths src --event-handlers console_direct+ --parallel-workers 3' > "$run_dir/$arm-build.log" 2>&1
    ! grep -E 'warning:|Warning C[0-9]+' "$run_dir/$arm-build.log"
    private_zenoh_port "$workspace" >> "$run_dir/environment.txt"
    run 2m "$workspace" "bash /fixtures/s1_04/runtime/smoke.sh /ws /evidence/$arm-smoke fk_library:=/ws/fk-build/lib$target.so fk_model:=/source/examples/cells/reach_place/assets/menagerie/$model" > "$run_dir/$arm-smoke-driver.log" 2>&1
    grep -q 'joint_state_broadcaster.*active' "$run_dir/$arm-smoke/controllers.log"
    grep -q 'joint_trajectory_controller.*active' "$run_dir/$arm-smoke/controllers.log"
    grep -h 'final_joint_state_hz=' "$run_dir/$arm-smoke-driver.log"
    test -s "$run_dir/$arm-smoke/mcap-sha256.txt"
    ! docker inspect "$container" >/dev/null 2>&1
done
echo "S1-06 cell MuJoCo smoke passed for panda and ur5e ($runner_mode): $run_dir"
