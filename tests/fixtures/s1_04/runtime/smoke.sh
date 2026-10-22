#!/usr/bin/env bash
set -eo pipefail
workspace=${1:?workspace}
evidence=${2:?evidence}
shift 2
# Remaining arguments are the generated launch file's arguments.
mkdir -p "$evidence"
# The caller may already have a ROS environment (Route 1 env.sh); else use the image install.
[[ -n ${ROS_DISTRO:-} ]] || source /opt/ros/lyrical/setup.bash
source "$workspace/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_SESSION_CONFIG_URI="$workspace/install/invar_cell_bringup/share/invar_cell_bringup/config/rmw_zenoh/session.json5"
export ZENOH_ROUTER_CONFIG_URI="$workspace/install/invar_cell_bringup/share/invar_cell_bringup/config/rmw_zenoh/router.json5"
setsid ros2 launch invar_cell_bringup cell.launch.py "$@" > "$evidence/launch.log" 2>&1 &
launch_pid=$!
group_running() {
  ps -eo pgid=,stat= | awk -v group="$launch_pid" '$1 == group && $2 !~ /^Z/ { found = 1 } END { exit !found }'
}
cleanup() {
  result=$?
  trap - EXIT
  set +e
  if [[ -n ${bag_pid:-} ]] && kill -0 "$bag_pid" 2>/dev/null; then
    kill -INT "$bag_pid" 2>/dev/null
    sleep 1
    kill -KILL "$bag_pid" 2>/dev/null
    wait "$bag_pid" 2>/dev/null
  fi
  kill -INT -- "-$launch_pid" 2>/dev/null
  for attempt in $(seq 1 20); do
    group_running || break
    sleep 0.25
  done
  escalated=0
  if group_running; then
    escalated=1
    # Record which launch-group members outlived the SIGINT grace period.
    ps -o pid=,stat=,etime=,args= --sid "$launch_pid" > "$evidence/shutdown-survivors.txt" 2>&1
    kill -TERM -- "-$launch_pid" 2>/dev/null
    for attempt in $(seq 1 8); do
      group_running || break
      sleep 0.25
    done
  fi
  if group_running; then
    escalated=1
    kill -KILL -- "-$launch_pid" 2>/dev/null
  fi
  wait "$launch_pid"
  launch_result=$?
  if (( result == 0 && (escalated != 0 || (launch_result != 0 && launch_result != 130)) )); then
    echo "ROS launch did not shut down cleanly: $launch_result" >&2
    result=1
  fi
  exit "$result"
}
trap cleanup EXIT
for attempt in $(seq 1 12); do
  if timeout 4s ros2 control list_controllers > "$evidence/controllers.log" 2>&1 \
    && grep -q 'joint_state_broadcaster.*active' "$evidence/controllers.log" \
    && grep -q 'joint_trajectory_controller.*active' "$evidence/controllers.log"; then
    break
  fi
  sleep 1
done
cat "$evidence/controllers.log"
grep -q 'joint_state_broadcaster.*active' "$evidence/controllers.log"
grep -q 'joint_trajectory_controller.*active' "$evidence/controllers.log"
timeout --signal=INT --kill-after=2s 14s ros2 bag record -s mcap -o "$evidence/mcap" --topics /joint_states > "$evidence/bag.log" 2>&1 &
bag_pid=$!
PYTHONUNBUFFERED=1 timeout --signal=INT --kill-after=2s 12s ros2 topic hz /joint_states --window 100 > "$evidence/hz.log" 2>&1 || status=$?
if [[ ${status:-0} != 124 ]]; then
  echo "ros2 topic hz exited unexpectedly: ${status:-0}" >&2
  exit 1
fi
cat "$evidence/hz.log"
python3 - "$evidence/hz.log" <<'PY'
import re, sys
text = open(sys.argv[1], encoding='utf-8').read()
rates = [float(rate) for rate in re.findall(r'average rate:\s*([0-9.]+)', text)]
if not rates:
    raise SystemExit('no joint-state rate samples')
rate = rates[-1]
print(f'final_joint_state_hz={rate:.3f}')
if not 95.0 <= rate <= 105.0:
    raise SystemExit('joint-state rate differs from 100 Hz by more than 5 percent')
PY
wait "$bag_pid" || bag_status=$?
bag_pid=
if [[ ${bag_status:-0} != 124 ]]; then
  echo "ros2 bag exited unexpectedly: ${bag_status:-0}" >&2
  exit 1
fi
find "$evidence/mcap" -name '*.mcap' -type f -size +0c -print -quit | grep -q .
sha256sum "$evidence"/mcap/*.mcap > "$evidence/mcap-sha256.txt"
