#!/usr/bin/env bash
# Build and run a git ref's CUDA suites on a remote Windows or Linux CUDA host over ssh.
# Usage: scripts/remote-cuda.sh <git-ref> [test-name-filters]
# Filters are libtest substrings separated by spaces. The suites are the root
# package's lib tests plus every test binary (tests/<t>.rs or tests/<t>/main.rs)
# with a file that names the cuda feature or CudaDevice, all --include-ignored.
# Env: REMOTE_CUDA_HOST (required; ssh alias or user@host), REMOTE_CUDA_DIR
# (invar-remote-cuda, under the remote home), REMOTE_CUDA_TIMEOUT_MIN (90),
# REMOTE_CUDA_WAIT_MIN (30, once for the runner lock and again for the GPU gate),
# REMOTE_CUDA_MIN_FREE_MIB (8192), REMOTE_CUDA_MAX_UTIL (20), REMOTE_CUDA_SHARED=1
# to run beside another tenant's GPU job (correctness only, timings invalid),
# REMOTE_CUDA_NO_SKIP=1 to run the known gaps below, REMOTE_CUDA_TESTS to replace
# the target list, REMOTE_CUDA_EXTRA_SKIPS (comma-separated) to add skips,
# REMOTE_CUDA_PROFILE (device-test) for the cargo profile the suites build with,
# REMOTE_CUDA_OS (Windows unless `uname -s` on the host prints Linux).
# A Linux host (scripts/remote-cuda-linux.sh) runs at low load by default:
# REMOTE_CUDA_LANES (2 suites at once), REMOTE_CUDA_JOBS (4 build jobs and test
# threads), REMOTE_CUDA_MEM_MAX (12G, no swap), REMOTE_CUDA_CPU_QUOTA (800%),
# REMOTE_CUDA_MAX_LOAD (8), REMOTE_CUDA_DEVICE_JOBS (2 GNET-05 programs), and
# Linux defaults TIMEOUT_MIN 240, WAIT_MIN 120, MAX_UTIL 5.
# Exit: 0 pass, 1 test failures (or no test ran, or a filter matched none), 2 build failure, 3 GPU or runner busy, 4 timeout,
# 5 ssh dropped mid-run (the run finished remotely; read its log for the result).
set -euo pipefail
ref=${1:?usage: scripts/remote-cuda.sh <git-ref> [test-name-filters]}
filter=${2:-}
host=${REMOTE_CUDA_HOST:?set REMOTE_CUDA_HOST to the ssh alias of the CUDA host}
dir=${REMOTE_CUDA_DIR:-invar-remote-cuda}
root=$(git rev-parse --show-toplevel)
here=$(cd "$(dirname "$0")" && pwd)
sha=$(git rev-parse -q --verify "$ref^{commit}" || { git fetch -q origin "$ref" && git rev-parse FETCH_HEAD; })
tests=$(for t in $(git grep -lE 'feature *= *"cuda"|CudaDevice' "$sha" -- tests |
    sed -nE -e "s|^$sha:tests/([^/]+)\\.rs$|\\1|p" -e "s|^$sha:tests/([^/]+)/.*|\\1|p" | sort -u); do
    # `if`, not `&&`: a last candidate that is no target (tests/support) must not fail the loop.
    if git cat-file -e "$sha:tests/$t.rs" 2>/dev/null || git cat-file -e "$sha:tests/$t/main.rs" 2>/dev/null; then echo "$t"; fi
done | paste -sd, -)
# Opt-in tests whose inputs the runner does not supply, and known Windows gaps,
# none of them a CUDA result (libtest --skip substrings):
skips=(
    issue_m3_05_gpu_full_m2_probe_corpus  # needs INVAR_M3_05_CORPUS_DIR
    benchmark_manifest_validation_and_rejections  # needs a git checkout
    source_packet_reverification_binds_source_key_dimensions_generation_and_input_bytes  # order-dependent on any OS
    p2_cuda_handwritten_diagnostic  # timing diagnostic; needs INVAR_NVRTC and a locked session
    robotics_baseline_outputs_match_reference  # needs INVAR_ROBOTICS_BASELINE_OUT
    task_region::pilot  # CUDA bench; needs P4B_OUT
    task_region::session  # CUDA bench; needs P4B_OUT and P4B_SESSION
    task_region::report  # CUDA bench report; needs P4B_OUT
    p8_calibrate_cuda  # opt-in; asserts INVAR_GPU_EXCLUSIVE=1
    p8_d5_cuda_session  # opt-in; asserts INVAR_GPU_EXCLUSIVE=1
    math_cr_v1_cuda_matches_upstream_core_math_bit_for_bit  # needs INVAR_COREMATH_UPSTREAM
    math_cr_v1_cuda_differential_catches_a_mutated_polynomial_coefficient  # needs INVAR_COREMATH_UPSTREAM
    math_matches_upstream_core_math_on_worst_cases_and_random_inputs  # host side; needs INVAR_COREMATH_UPSTREAM
    math_differential_catches_a_mutated_polynomial_coefficient  # host side; needs INVAR_COREMATH_UPSTREAM
    math_msl_host_build_matches_upstream_core_math  # needs INVAR_COREMATH_UPSTREAM
)
windows_skips=(
    system::substep_admission::     # S1405 manifest path checks assume Unix paths
    grid_cuda_completed_equality     # uses /bin/false as the failing compiler
    math_msl_prelude_compiles_warning_free_on_the_host  # host C++ build assumes a 64-bit long; Windows is LLP64
    math_msl_soft_binary64_matches_the_host_fpu  # host C++ build assumes a 64-bit long; Windows is LLP64
    math_msl_host_build_matches_the_cpu_library  # host C++ build assumes a 64-bit long; Windows is LLP64
    math_msl_host_build_catches_a_mutated_polynomial_coefficient  # host C++ build assumes a 64-bit long; Windows is LLP64
)
ssh_=(ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 "$host")
os=${REMOTE_CUDA_OS:-$("${ssh_[@]}" uname -s 2>/dev/null | tr -d '\r' || true)}
[ "$os" = Linux ] || skips+=("${windows_skips[@]}")
[ "${REMOTE_CUDA_NO_SKIP:-0}" = 1 ] && skips=()
[ -n "${REMOTE_CUDA_TESTS:-}" ] && tests=$REMOTE_CUDA_TESTS
[ -n "${REMOTE_CUDA_EXTRA_SKIPS:-}" ] && IFS=, read -ra extra <<<"$REMOTE_CUDA_EXTRA_SKIPS" && skips+=("${extra[@]}")
run="$(date -u +%Y%m%dT%H%M%SZ)-${sha:0:10}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Windows cannot create the tree's symlinks without privileges; ship them dereferenced.
mkdir "$tmp/src"
git archive "$sha" | tar -xf - -C "$tmp/src"
tar -czhf "$tmp/src.tgz" -C "$tmp/src" .
echo "remote-cuda: ref $ref = $sha, run $run, $(tr ',' '\n' <<<"$tests" | wc -l | tr -d ' ') test targets + lib on ${os:-Windows}"
mkdir -p "$root/target/remote-cuda"

if [ "$os" = Linux ]; then
    rd="$dir/runs/$run"
    q() { printf '%s=%q\n' "$1" "$2"; }
    {
        q BASE "$dir"; q TESTS "$tests"; q FILTER "$filter"; q SKIP "$(IFS=,; echo "${skips[*]}")"
        q REVISION "$sha"; q PROFILE "${REMOTE_CUDA_PROFILE:-device-test}"
        q TIMEOUT_MIN "${REMOTE_CUDA_TIMEOUT_MIN:-240}"; q WAIT_MIN "${REMOTE_CUDA_WAIT_MIN:-120}"
        q MIN_FREE_MIB "${REMOTE_CUDA_MIN_FREE_MIB:-8192}"; q MAX_UTIL "${REMOTE_CUDA_MAX_UTIL:-5}"
        q MAX_LOAD "${REMOTE_CUDA_MAX_LOAD:-8}"; q SHARED "${REMOTE_CUDA_SHARED:-0}"
        q LANES "${REMOTE_CUDA_LANES:-2}"; q JOBS "${REMOTE_CUDA_JOBS:-4}"; q DEVICE_JOBS "${REMOTE_CUDA_DEVICE_JOBS:-2}"
        q MEM_MAX "${REMOTE_CUDA_MEM_MAX:-12G}"; q CPU_QUOTA "${REMOTE_CUDA_CPU_QUOTA:-800%}"
    } >"$tmp/run.env"
    "${ssh_[@]}" "mkdir -p $rd"
    scp -q "$here/remote-cuda-linux.sh" "$tmp/src.tgz" "$tmp/run.env" "$host:$rd/"
    # Detached, so a dropped ssh session never stops the run; follow its output.
    pid=$("${ssh_[@]}" "cd $rd || exit; setsid nohup bash remote-cuda-linux.sh >out 2>&1 </dev/null & echo \$!")
    while :; do
        rc=0
        "${ssh_[@]}" "tail -n +1 -F --pid=$pid $rd/out 2>/dev/null" || rc=$?
        [ "$rc" = 255 ] || break
        echo "remote-cuda: ssh dropped; reconnecting to run $run"
        sleep 60
    done
    rc=$("${ssh_[@]}" "cat $rd/rc 2>/dev/null" || true)
    "${ssh_[@]}" "gzip -c $rd/cargo.log" 2>/dev/null | gunzip 2>/dev/null >"$root/target/remote-cuda/$run.log" &&
        echo "remote-cuda: full log target/remote-cuda/$run.log"
    exit "${rc:-5}"
fi

"${ssh_[@]}" "mkdir $dir\\runs\\$run"
scp -q "$here/remote-cuda.ps1" "$tmp/src.tgz" "$host:$dir/runs/$run/"
args=(-Run "$run" -Base "$dir" -Tests "$tests" -Revision "$sha"
    -Profile "${REMOTE_CUDA_PROFILE:-device-test}" -TimeoutMin "${REMOTE_CUDA_TIMEOUT_MIN:-90}" -WaitMin "${REMOTE_CUDA_WAIT_MIN:-30}"
    -MinFreeMiB "${REMOTE_CUDA_MIN_FREE_MIB:-8192}" -MaxUtil "${REMOTE_CUDA_MAX_UTIL:-20}")
# ssh joins argv with spaces, so quote the filter list for the remote command line.
[ -n "$filter" ] && args+=(-Filter "\"$filter\"")
[ ${#skips[@]} -gt 0 ] && args+=(-Skip "$(IFS=,; echo "${skips[*]}")")
[ -n "${REMOTE_CUDA_MINGW:-}" ] && args+=(-Mingw "$REMOTE_CUDA_MINGW")
[ "${REMOTE_CUDA_SHARED:-0}" = 1 ] && args+=(-Shared)
rc=0
"${ssh_[@]}" powershell -NoProfile -ExecutionPolicy Bypass -File "$dir\\runs\\$run\\remote-cuda.ps1" "${args[@]}" || rc=$?
if [ "$rc" = 255 ]; then
    # The ssh session dropped; the remote run keeps going. Wait for its lock to clear.
    echo "remote-cuda: ssh dropped; waiting for run $run to finish"
    until [ "$("${ssh_[@]}" "if exist $dir\\lock (echo HELD) else (echo FREE)" 2>/dev/null | tr -d '\r')" = FREE ]; do
        sleep 60
    done
    rc=5
fi
scp -q "$host:$dir/runs/$run/cargo.log" "$root/target/remote-cuda/$run.log" 2>/dev/null &&
    echo "remote-cuda: full log target/remote-cuda/$run.log"
exit "$rc"
