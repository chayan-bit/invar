#!/usr/bin/env bash
# Build-only gate. No planner child, simulation, network, pull, or package install.
set -euo pipefail
run_dir=${1:?new prepared evidence directory}
image_id=${2:?immutable local image ID}
[[ $(uname -s) == Linux ]] || { echo 'Linux required' >&2; exit 1; }
[[ $run_dir == /* && -d $run_dir ]] || { echo 'prepared absolute directory required' >&2; exit 1; }
[[ $image_id =~ ^sha256:[a-fA-F0-9]{64}$ ]] || { echo 'immutable image ID required' >&2; exit 1; }
[[ $(docker image inspect --format '{{.Id}}' "$image_id") == "$image_id" ]] || exit 1
docker image inspect "$image_id" > "$run_dir/image-inspect.json"
container="invar-planner-build-$$"
trap 'docker stop -t 2 "$container" >/dev/null 2>&1 || true' EXIT
for n in 6 7; do
    [[ -f $run_dir/$n/source-hashes.json ]] || exit 1
    timeout --kill-after=10s 8m docker run --rm --pull never --name "$container" \
        --network none --cpus 2 --memory 10g --memory-swap 10g --pids-limit 512 \
        --user "$(id -u):$(id -g)" -e HOME=/tmp -e MAKEFLAGS=-j2 \
        -v "$run_dir/$n:/work" -w /work "$image_id" bash -lc '
            set -euo pipefail
            source /opt/ros/lyrical/setup.bash
            test "$ROS_DISTRO" = lyrical
            dpkg-query -W ros-lyrical-rclcpp ros-lyrical-rclcpp-components ros-lyrical-rosidl-generator-type-description ros-lyrical-trajectory-msgs
            colcon build --packages-up-to invar_planner_node invar_controller_bridge \
                --event-handlers console_direct+ --parallel-workers 2 \
                --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-Werror
            test -x install/invar_planner_node/lib/invar_planner_node/planner_node
            test -x install/invar_controller_bridge/lib/invar_controller_bridge/controller_bridge
        ' > "$run_dir/$n/build.log" 2>&1
    ! docker inspect "$container" >/dev/null 2>&1
done
