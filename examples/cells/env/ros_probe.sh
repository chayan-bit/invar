#!/usr/bin/env bash
# Run inside the pinned ROS environment, preferably an isolated container network.
set -euo pipefail

out=${1:?usage: ros_probe.sh OUTPUT_DIR}
mkdir -p "$out"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-$((100 + RANDOM % 100))}

router_pid= bag_pid= pub_pid= echo_pid=
cleanup() {
  for pid in "$echo_pid" "$pub_pid" "$bag_pid" "$router_pid"; do
    if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

ros2 pkg prefix rmw_zenoh_cpp
ros2 pkg prefix rosbag2_storage_mcap
ros2 pkg prefix mujoco_ros2_control
ros2 pkg prefix mujoco_ros2_control_demos
python3 - <<'PY'
import ctypes
from pathlib import Path
from xml.etree import ElementTree
from ament_index_python.packages import get_package_prefix

prefix = Path(get_package_prefix("mujoco_ros2_control"))
xmls = (prefix / "share/mujoco_ros2_control").glob("*.xml")
assert any(
    "mujoco_ros2_control/MujocoSystemInterface" in
    [node.attrib.get("name") for node in ElementTree.parse(xml).iter("class")]
    for xml in xmls
), "MuJoCo hardware interface plugin declaration missing"
library = prefix / "lib/libmujoco_ros2_control.so"
assert library.is_file(), "MuJoCo hardware interface library missing"
ctypes.CDLL(str(library))
print(f"mujoco_ros2_control plugin library loaded: {library.name}")
PY

ros2 run rmw_zenoh_cpp rmw_zenohd >"$out/router.log" 2>&1 &
router_pid=$!
sleep 2
kill -0 "$router_pid"

ros2 bag record --storage mcap --output "$out/bag" --topics /invar_env_probe >"$out/record.log" 2>&1 &
bag_pid=$!
sleep 2
kill -0 "$bag_pid"

timeout 15s ros2 topic pub --rate 10 /invar_env_probe std_msgs/msg/String '{data: probe}' >"$out/publish.log" 2>&1 &
pub_pid=$!
timeout 15s ros2 topic echo --once /invar_env_probe std_msgs/msg/String >"$out/echo.log"
grep -q 'probe' "$out/echo.log"
sleep 2
kill -TERM "$pub_pid" "$bag_pid" 2>/dev/null || true
wait "$pub_pid" 2>/dev/null || true
wait "$bag_pid" 2>/dev/null || true
pub_pid= bag_pid=

ros2 bag info "$out/bag" >"$out/bag-info.log"
grep -Eq 'Storage id:[[:space:]]+mcap' "$out/bag-info.log"
grep -Eq 'Messages:[[:space:]]+[1-9]' "$out/bag-info.log"

timeout 15s ros2 topic echo --once /invar_env_probe std_msgs/msg/String >"$out/replayed.log" &
echo_pid=$!
sleep 1
timeout 15s ros2 bag play "$out/bag" >"$out/play.log" 2>&1
wait "$echo_pid"
echo_pid=
grep -q 'probe' "$out/replayed.log"

status=0
timeout --signal=INT --kill-after=5s 25s ros2 launch \
  mujoco_ros2_control_demos 01_basic_robot.launch.py headless:=true \
  >"$out/control-demo.log" 2>&1 || status=$?
[[ $status -eq 124 ]]
grep -q 'Activating MuJoCo hardware interface and starting Simulate threads' \
  "$out/control-demo.log"
echo 'zenoh pub/sub, MCAP write/read, and MuJoCo ros2_control hardware activation: ok'
