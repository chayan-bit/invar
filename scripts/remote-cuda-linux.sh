#!/usr/bin/env bash
# Remote half of scripts/remote-cuda.sh for a shared Linux CUDA host: build and
# run one ref's CUDA suites at the lowest priority the host offers.
# Everything lives in $HOME/<BASE>: .rustup/ and .cargo/ (per-user Rust),
# nvrtc/lib/ (NVIDIA's NVRTC redistributable), target/, src/ (the ref under
# test), nvrtc-cache/, runs/<id>/ and the lock file. The driver ships this
# script, src.tgz and run.env into runs/<id>/, starts it detached and reads
# runs/<id>/out; the exit code lands in runs/<id>/rc.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
. "$here/run.env"
# shellcheck disable=SC2153 # BASE comes from run.env
base=$HOME/$BASE
say() { echo "remote-cuda: $*"; }

# Low load: a user scope capping memory (no swap) and CPU, at nice 19 and the
# idle I/O class; every child inherits all four.
if [ -z "${REMOTE_CUDA_SCOPED:-}" ]; then
    export REMOTE_CUDA_SCOPED=1
    low=(nice -n 19 ionice -c3 bash "$here/${0##*/}")
    if systemd-run --user --scope -q true 2>/dev/null; then
        # OOMPolicy=continue: the cap kills only the process over it, which then fails its suite.
        exec systemd-run --user --scope -q -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 \
            -p CPUQuota="$CPU_QUOTA" -p OOMPolicy=continue "${low[@]}"
    fi
    # ulimit -v would break CUDA, which reserves far more address space than it uses.
    say "no systemd user scope: nice and ionice only, no memory or CPU cap"
    exec "${low[@]}"
fi
# Background jobs (sampler, suites) hold the lock fd: stop them, and their
# timeouts (which pass the signal to their process group), on any exit.
stop_jobs() { local p; for p in $(jobs -p); do pkill -TERM -P "$p"; kill "$p"; done 2>/dev/null; }
trap 'rc=$?; stop_jobs; rm -rf "$here/tmp" "$here/src.tgz"; echo "$rc" >"$here/rc"' EXIT
# A stopped run must never record a pass.
trap 'exit 143' TERM INT HUP
log=$here/cargo.log
src=$base/src
cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)

# One run at a time: the target dir and the GPU are shared. The kernel drops the
# lock when the last process holding it (this script or a child) exits.
# One wait budget covers the lock and the idle gate.
until_=$((SECONDS + WAIT_MIN * 60))
exec 9>"$base/lock"
flock -w $((WAIT_MIN * 60)) 9 || { say 'BUSY: another remote-cuda run holds the lock'; exit 3; }
# Keep the ten newest runs (ids sort by start time).
find "$base/runs" -mindepth 1 -maxdepth 1 -type d | sort | head -n -10 | xargs -r rm -rf

gpu() {
    IFS=, read -r g_name g_driver g_util g_used g_total < <(nvidia-smi \
        --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1)
    g_util=${g_util// /} g_used=${g_used// /} g_driver=${g_driver// /}
    g_free=$((${g_total// /} - g_used))
    load=$(cut -d' ' -f1 /proc/loadavg)
}
idle() {
    awk -v l="$load" -v m="$MAX_LOAD" 'BEGIN { exit !(l < m) }' && [ "$g_free" -ge "$MIN_FREE_MIB" ] &&
        { [ "$SHARED" = 1 ] || [ "$g_util" -le "$MAX_UTIL" ]; }
}
# Gate: three idle samples 5 s apart; otherwise check again every 5 minutes.
while :; do
    quiet=0
    for _ in 1 2 3; do
        gpu
        idle || break
        quiet=$((quiet + 1))
        [ "$quiet" = 3 ] || sleep 5
    done
    [ "$quiet" = 3 ] && break
    state="GPU util $g_util% (max $MAX_UTIL), free $g_free MiB (min $MIN_FREE_MIB), load $load (max $MAX_LOAD)"
    [ "$SECONDS" -ge "$until_" ] && { say "BUSY: $state"; exit 3; }
    say "waiting: $state"
    sleep 300
done
[ "$SHARED" = 1 ] && say "SHARED: GPU at $g_util% from another job; correctness only, timings are not valid"
say "gpu: $g_name, driver $g_driver, util $g_util%, free $g_free MiB; load $load"

rm -rf "$src" && mkdir -p "$src" "$here/tmp"
# -m: archive mtimes are the commit time, which can predate the last build and hide changes from cargo.
tar -xmzf "$here/src.tgz" -C "$src" || { say 'extract failed'; exit 2; }
# Component manifests resolve their inputs against the nearest ancestor holding
# .git (src/system/imports.rs), as in a checkout; the archive has none.
git init -q "$src" || { say 'git init failed'; exit 2; }
channel=$(sed -nE 's/^channel *= *"([^"]+)".*/\1/p' "$src/rust-toolchain.toml")
export RUSTUP_HOME=$base/.rustup CARGO_HOME=$base/.cargo RUSTUP_TOOLCHAIN=$channel
export PATH=$base/.cargo/bin:$PATH CCACHE_DISABLE=1 TMPDIR=$here/tmp
export CARGO_TARGET_DIR=$base/target CARGO_BUILD_JOBS=$JOBS RUST_TEST_THREADS=$JOBS
export INVAR_TEST_TIMEOUT_SCALE=${INVAR_TEST_TIMEOUT_SCALE:-8}
# build.rs embeds this revision; the extracted archive has no git checkout to read it from.
export INVAR_SOURCE_REVISION=$REVISION
case $PROFILE in dev | test) out=debug ;; release | bench) out=release ;; *) out=$PROFILE ;; esac
export INVAR_NVRTC_HELPER=$base/target/$out/invar-nvrtc
export INVAR_NVRTC_COMPILER=$INVAR_NVRTC_HELPER
# cudarc loads libcuda from the system and libnvrtc.so from here first.
export LD_LIBRARY_PATH=$base/nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# Verified compiles persist across runs (src/nvrtc_cache.rs, LRU-capped at 4 GiB).
export INVAR_NVRTC_CACHE_DIR=$base/nvrtc-cache
export INVAR_GNET05_DEVICE_JOBS=$DEVICE_JOBS
# The census sizes its NVRTC pool from the cgroup limit as if alone in it; the scope is shared.
export INVAR_GNET05_COMPILE_JOBS=$COMPILE_JOBS

# Peak CPU (cores, 5 s windows) and memory of this run's scope, and peak GPU
# memory of this run's own processes (binaries under $base).
sample() {
    local cpu=0 gmem=0 rss=0 u0 t0 u1 t1 g r
    u0=$(awk '/^usage_usec/ { print $2 }' "$cg/cpu.stat") t0=$(date +%s%N)
    while :; do
        sleep 5
        u1=$(awk '/^usage_usec/ { print $2 }' "$cg/cpu.stat") t1=$(date +%s%N)
        cpu=$(awk -v c="$cpu" -v du=$((u1 - u0)) -v dt=$(((t1 - t0) / 1000)) 'BEGIN { x = du / dt; print (x > c ? x : c) }')
        u0=$u1 t0=$t1
        g=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
            while IFS=', ' read -r pid mem; do
                case $(readlink "/proc/$pid/exe" 2>/dev/null) in "$base"/*) echo "$mem" ;; esac
            done | awk '{ s += $1 } END { print s + 0 }')
        # shellcheck disable=SC2016 # $2 is awk's
        r=$(xargs -r -I{} awk '/^VmRSS/ { print $2 }' /proc/{}/status <"$cg/cgroup.procs" 2>/dev/null | awk '{ s += $1 } END { print int(s / 1024) }')
        [ "$g" -gt "$gmem" ] && gmem=$g
        [ "$r" -gt "$rss" ] && rss=$r
        echo "$cpu $gmem $rss" >"$here/peak"
    done
}
sample &
sampler=$!

# One element per suite: the flag and its target, split again where used.
suites=(--lib)
IFS=, read -ra test_list <<<"$TESTS"
for t in "${test_list[@]}"; do [ -n "$t" ] && suites+=("--test $t"); done
read -ra filters <<<"$FILTER"
skips=()
IFS=, read -ra skip_list <<<"$SKIP"
for s in "${skip_list[@]}"; do [ -n "$s" ] && skips+=(--skip "$s"); done
common=(--locked --profile "$PROFILE" --features cuda --no-fail-fast)
deadline=$((SECONDS + TIMEOUT_MIN * 60))
left() { echo $((deadline > SECONDS ? deadline - SECONDS : 1)); }
cd "$src" || exit 2
begun=$SECONDS
# shellcheck disable=SC2068 # each suite is a flag and its value
timeout -k 30 "$(left)" cargo test "${common[@]}" --no-run ${suites[@]} >"$log" 2>&1
build_rc=$?
say "build: $((SECONDS - begun)) s"
timed_out=0
[ "$build_rc" = 124 ] && timed_out=1
# At most LANES suites at once: each opens its own CUDA context. GNU time, when
# present, records each suite's largest process.
rss_of=()
[ -x /usr/bin/time ] && rss_of=(/usr/bin/time -f %M -o)
if [ "$build_rc" = 0 ]; then
    for i in "${!suites[@]}"; do
        while [ "$(jobs -rp | grep -cv "^$sampler\$")" -ge "$LANES" ]; do wait -n; done
        (
            t=$SECONDS
            meter=()
            [ ${#rss_of[@]} -gt 0 ] && meter=("${rss_of[@]}" "$here/suite-$i.rss")
            # shellcheck disable=SC2086 # the suite is a flag and its value
            timeout -k 30 "$(left)" "${meter[@]}" \
                cargo test "${common[@]}" ${suites[$i]} -- --include-ignored "${skips[@]}" \
                "${filters[@]}" >"$here/suite-$i.log" 2>&1
            echo "$? $((SECONDS - t))" >"$here/suite-$i.rc"
        ) &
    done
    while [ "$(jobs -rp | grep -cv "^$sampler\$")" -gt 0 ]; do wait -n; done
    for i in "${!suites[@]}"; do cat "$here/suite-$i.log" >>"$log"; done
fi
kill "$sampler" 2>/dev/null
cd "$here" || exit 2
say "wall: $((SECONDS - begun)) s"
failed_lane=0
for i in "${!suites[@]}"; do
    [ -f "$here/suite-$i.rc" ] || continue
    read -r rc secs <"$here/suite-$i.rc"
    [ "$rc" = 124 ] && timed_out=1
    [ "$rc" = 0 ] || failed_lane=1
    rss=$(tail -1 "$here/suite-$i.rss" 2>/dev/null) && rss=", peak RSS $((rss / 1024)) MiB" || rss=
    say "suite ${suites[$i]}: $secs s, exit $rc$rss"
done
read -r cpu gmem rss <"$here/peak" 2>/dev/null || { cpu=0 gmem=0 rss=0; }
mem_peak=$(($(cat "$cg/memory.peak" 2>/dev/null || echo 0) / 1048576))
ooms=$(awk '/^oom_kill / { print $2 }' "$cg/memory.events" 2>/dev/null)
[ "${ooms:-0}" -gt 0 ] && say "MEMORY CAP: $ooms process(es) killed at MemoryMax=$MEM_MAX; their suites fail for memory, not CUDA"
say "load: peak CPU $cpu cores (5 s windows); peak RSS $rss MiB; scope memory peak $mem_peak MiB (page cache included); peak GPU memory of this run $gmem MiB"

awk '
    /^ +Running / { t = ($2 == "unittests") ? $3 : $2 }
    /^test result: / { sub(/\./, "", $3); printf "%-48s %-6s %6d passed %4d failed %5d ignored\n", t, $3, $4, $6, $8 }
' "$log" | tee "$here/rows"
grep -E '^test .* \.\.\. FAILED$' "$log" | sed 's/^/remote-cuda: /'
rows=$(wc -l <"$here/rows")
read -r passed failed ignored < <(awk '{ p += $3; f += $5; i += $7 } END { print p + 0, f + 0, i + 0 }' "$here/rows")
say "summary: $passed passed, $failed failed, $ignored ignored in $rows of ${#suites[@]} targets (lib included)"
[ -n "$SKIP" ] && say "skipped: $SKIP"
gpu
say "gpu after: util $g_util%, free $g_free MiB"
[ "$timed_out" = 1 ] && { say "TIMEOUT after $TIMEOUT_MIN min"; exit 4; }
[ "$rows" = 0 ] && { tail -40 "$log"; say 'BUILD FAILED'; exit 2; }
# A run that executed nothing, or a filter that matched no executed test, proves nothing.
ran=$(sed -nE 's/^test (\S+) \.\.\. (ok|FAILED)$/\1/p' "$log")
[ -z "$ran" ] && { say 'NO TESTS RAN: check the filters against this ref'; exit 1; }
for f in "${filters[@]}"; do
    grep -qF -- "$f" <<<"$ran" || { say "FILTER MATCHED NO TEST: $f"; exit 1; }
done
if [ "$failed_lane" = 1 ] || [ "$failed" -gt 0 ] || [ "$rows" -lt "${#suites[@]}" ]; then
    [ "$rows" -lt "${#suites[@]}" ] && say "a target ended without a result (killed, crashed or not built); see its suite line and the log"
    exit 1
fi
say PASS
