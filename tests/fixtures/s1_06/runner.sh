# Sourced by the S1-06 ROS fixture scripts. Runs a command either in the pinned
# container image (INVAR_S1_06_RUNNER=docker, default) or in the Route 1
# user-space environment on the host (INVAR_S1_06_RUNNER=userspace with
# INVAR_S1_06_ENV_ROOT set). Either way the ROS environment is sourced before
# the command runs. Commands name their inputs by container paths
# (for example /ws); in user-space mode each mount's container path is
# rewritten to its host path, and each generated workspace gets a private
# Zenoh router port so runs on a shared host cannot reach other routers.
runner_mode=${INVAR_S1_06_RUNNER:-docker}

runner_check() {
    local image=$1 expected_id=$2
    if [[ $runner_mode == docker ]]; then
        [[ $(uname -m) == aarch64 ]] || { echo 'ARM64 Linux required for the pinned image' >&2; return 1; }
        [[ $(docker image inspect --format '{{.Id}}' "$image") == "$expected_id" ]] \
            || { echo 'ROS image ID differs' >&2; return 1; }
    elif [[ $runner_mode == userspace ]]; then
        [[ -f ${INVAR_S1_06_ENV_ROOT:?set INVAR_S1_06_ENV_ROOT}/ws/install/setup.bash ]] \
            || { echo 'Route 1 environment is not built' >&2; return 1; }
    else
        echo "unknown INVAR_S1_06_RUNNER $runner_mode" >&2
        return 1
    fi
}

# run_env LIMIT CONTAINER_NAME HOST:CONTAINER[:ro]... -- COMMAND
# The first mount is the working directory.
run_env() {
    local limit=$1 name=$2
    shift 2
    local mounts=()
    while [[ $1 != -- ]]; do mounts+=("$1"); shift; done
    shift
    local command=$1
    if [[ $runner_mode == docker ]]; then
        local args=() workdir
        for mount in "${mounts[@]}"; do
            args+=(-v "$mount")
            [[ -n ${workdir:-} ]] || { workdir=${mount#*:}; workdir=${workdir%:ro}; }
        done
        timeout --kill-after=10s "$limit" docker run --rm --name "$name" \
            --network none --cpus 3 --memory 10g --memory-swap 10g --shm-size 256m --pids-limit 512 \
            "${args[@]}" -w "$workdir" "$image" bash -lc "source /opt/ros/lyrical/setup.bash && $command"
        return
    fi
    local host_workdir='' pairs=()
    for mount in "${mounts[@]}"; do
        local spec=${mount%:ro} host=${mount%%:*}
        pairs+=("${spec#*:}=$host")
        [[ -n $host_workdir ]] || host_workdir=$host
    done
    command=$(python3 - "$command" "${pairs[@]}" <<'PY'
import re, sys
command = sys.argv[1]
for pair in sys.argv[2:]:
    container, host = pair.split("=", 1)
    command = re.sub(r"(?<![\w./-])" + re.escape(container) + r"(?![\w.-])", host, command)
print(command)
PY
)
    local status=0
    (cd "$host_workdir" && INVAR_S1_06_WORKSPACE="$host_workdir" nice -n 10 timeout --kill-after=10s "$limit" bash -c \
        "source '$root/examples/cells/env/userspace/env.sh' '$INVAR_S1_06_ENV_ROOT' && $command") || status=$?
    # A shared account can have daemons from other lanes. Match our inherited
    # invocation marker, and signal through a pidfd so PID reuse cannot hit one.
    python3 - "$host_workdir" <<'PY' || return $?
import os, pathlib, signal, subprocess, sys

marker = os.fsencode("INVAR_S1_06_WORKSPACE=" + sys.argv[1])
pids = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "_ros2_daemon"],
                      stdout=subprocess.PIPE, check=False)
if pids.returncode not in (0, 1):
    raise SystemExit(pids.returncode)
for pid in pids.stdout.split():
    try:
        fd = os.pidfd_open(int(pid))
        try:
            if marker in pathlib.Path(f"/proc/{int(pid)}/environ").read_bytes().split(b"\0"):
                signal.pidfd_send_signal(fd, signal.SIGTERM)
        finally:
            os.close(fd)
    except (ProcessLookupError, FileNotFoundError, PermissionError):
        pass
PY
    return "$status"
}

# private_zenoh_port WORKSPACE: rewrite the run's installed Zenoh configs to an
# unused loopback port (user-space mode only; the container has its own network).
private_zenoh_port() {
    [[ $runner_mode == userspace ]] || return 0
    local workspace=$1 port
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    sed -i "s#tcp/127.0.0.1:7447#tcp/127.0.0.1:$port#" \
        "$workspace"/install/invar_cell_bringup/share/invar_cell_bringup/config/rmw_zenoh/*.json5
    echo "zenoh_router_port=$port"
}
