#!/usr/bin/env bash
# Linux-only regression: clean this invocation's daemon, preserve another lane's.
set -euo pipefail
umask 077
fixture=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $(uname -s) == Linux && -d /proc ]] || { echo 'Linux /proc required' >&2; exit 1; }
base=$(mktemp -d "${TMPDIR:?set a private TMPDIR}/s106-daemon-check.XXXXXX")
other= own=
cleanup() {
    # An unreaped own daemon can outlive a failed check, but never signal a
    # possibly reused orphan PID. Its sleep is bounded to 30 seconds.
    [[ -z $own ]] || timeout 35s tail --pid="$own" -f /dev/null 2>/dev/null || true
    [[ -z $other ]] || { kill "$other" 2>/dev/null || true; wait "$other" 2>/dev/null || true; }
    rm -rf -- "$base"
}
trap cleanup EXIT
root=$base/source
mkdir -p "$root/examples/cells/env/userspace" "$base/workspace" "$base/bin"
printf 'true\n' > "$root/examples/cells/env/userspace/env.sh"
# Restrict discovery to the fake PIDs. An old broad-pkill implementation must
# fail this check without being able to signal real daemons on the shared host.
printf '#!/bin/sh\ncat "$INVAR_TEST_OTHER_PID" "$INVAR_TEST_OWN_PID"\n' > "$base/bin/pgrep"
printf '#!/bin/sh\nexit 77\n' > "$base/bin/pkill"
chmod 700 "$base/bin/pgrep" "$base/bin/pkill"
export PATH=$base/bin:$PATH INVAR_TEST_OTHER_PID=$base/other.pid INVAR_TEST_OWN_PID=$base/workspace/daemon.pid
export INVAR_S1_06_RUNNER=userspace INVAR_S1_06_ENV_ROOT=$base/environment
# Only this test owns these fake daemon PIDs, including the unrelated marker.
INVAR_S1_06_WORKSPACE=$base/other bash -c 'exec -a _ros2_daemon sleep 60' &
other=$!
echo "$other" > "$base/other.pid"
source "$fixture/runner.sh"
run_env 5s unused "$base/workspace:/ws" -- \
    'bash -c "exec -a _ros2_daemon sleep 30" & echo $! > /ws/daemon.pid; sleep 0.1'
own=$(cat "$base/workspace/daemon.pid")
kill -0 "$other"
# A terminated, unreaped fake daemon may briefly be a zombie under PID 1.
for _ in {1..40}; do
    state=$(awk '{print $3}' "/proc/$own/stat" 2>/dev/null || true)
    [[ -z $state || $state == Z ]] && break
    sleep 0.05
done
[[ -z $state || $state == Z ]] || { echo 'own daemon survived cleanup' >&2; exit 1; }
own=
# Failure must still trigger cleanup and preserve the caller's exit status.
status=0
run_env 5s unused "$base/workspace:/ws" -- 'exit 23' || status=$?
[[ $status == 23 ]]
kill -0 "$other"
echo 'daemon cleanup: own invocation stopped; other lane preserved; failure status preserved'
