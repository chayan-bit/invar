# Remote half of scripts/remote-cuda.sh: build and run one ref's CUDA suites on a
# Windows CUDA host (per-user Rust gnullvm toolchain and llvm-mingw, no admin).
# Everything lives in %USERPROFILE%\<Base>: nvrtc\ (NVRTC DLLs), cargo-home\,
# target\, src\ (the ref under test), runs\<id>\ and the lock file.
# The pinned llvm-mingw (clang, lld and the mingw import libraries the gnullvm
# toolchain lacks) lives in %USERPROFILE%\<Mingw>.
param(
    [Parameter(Mandatory)][string]$Run,
    [Parameter(Mandatory)][string]$Base,
    [Parameter(Mandatory)][string]$Tests,
    [string]$Filter = '',
    [string]$Skip = '',
    [string]$Revision = '',
    [string]$Profile = 'device-test',
    [string]$Mingw = 'toolchains\llvm-mingw-20261006-ucrt-x86_64\bin',
    [int]$TimeoutMin = 90,
    [int]$WaitMin = 30,
    [int]$MinFreeMiB = 8192,
    [int]$MaxUtil = 20,
    [switch]$Shared
)
$ErrorActionPreference = 'Stop'
# Children inherit BelowNormal, so the build and tests never outrank other users.
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$base = Join-Path $env:USERPROFILE $Base
$runDir = Join-Path $base "runs\$Run"
$lock = Join-Path $base 'lock'
$src = Join-Path $base 'src'
$log = Join-Path $runDir 'cargo.log'
function Say($m) { Write-Output "remote-cuda: $m" }

function Gpu {
    $f = (& nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total `
            --format=csv,noheader,nounits | Select-Object -First 1).Split(',') | ForEach-Object { $_.Trim() }
    [pscustomobject]@{ Name = $f[0]; Driver = $f[1]; Util = [int]$f[2]; Free = [int]$f[4] - [int]$f[3] }
}

# One run at a time: the target dir and the GPU are shared. The lock is an open
# handle on $lock that no other process may share, so Windows releases it (and
# deletes the file) only when the holder and every child that inherited the
# handle have exited, however they exit. A live holder is never broken.
$lockUntil = (Get-Date).AddMinutes($WaitMin)
while ($true) {
    try {
        $held = [IO.FileStream]::new($lock, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::Inheritable, 1, [IO.FileOptions]::DeleteOnClose)
        break
    } catch {}
    if ((Get-Date) -gt $lockUntil) { Say 'BUSY: another remote-cuda run holds the lock'; exit 3 }
    Start-Sleep 30
}
try {
    # GPU gate: three quiet samples in a row, or memory headroom plus -Shared.
    $quiet = 0
    $gpuUntil = (Get-Date).AddMinutes($WaitMin)
    while ($true) {
        $g = Gpu
        if ($g.Util -le $MaxUtil -and $g.Free -ge $MinFreeMiB) { $quiet++ } else { $quiet = 0 }
        if ($quiet -ge 3) { break }
        if ((Get-Date) -gt $gpuUntil) {
            if ($Shared -and $g.Free -ge $MinFreeMiB) {
                Say "SHARED: GPU at $($g.Util)% from another job; correctness only, timings are not valid"; break
            }
            Say "BUSY: GPU util $($g.Util)% (max $MaxUtil), free $($g.Free) MiB (min $MinFreeMiB)"; exit 3
        }
        Start-Sleep 5
    }
    Say "gpu: $($g.Name), driver $($g.Driver), util $($g.Util)%, free $($g.Free) MiB"

    if (Test-Path $src) { Remove-Item $src -Recurse -Force }
    New-Item -ItemType Directory $src | Out-Null
    # -m: archive mtimes are the commit time, which can predate the last build and hide changes from cargo.
    tar -xmzf (Join-Path $runDir 'src.tgz') -C $src
    if ($LASTEXITCODE) { Say 'extract failed'; exit 2 }

    $channel = (Select-String -Path "$src\rust-toolchain.toml" -Pattern 'channel\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
    $env:RUSTUP_TOOLCHAIN = "$channel-x86_64-pc-windows-gnullvm"
    # The gnullvm toolchain ships no mingw import libraries (advapi32, ole32, ...),
    # so link and compile C with the pinned per-user llvm-mingw.
    $mingwBin = Join-Path $env:USERPROFILE $Mingw
    if (-not (Test-Path "$mingwBin\x86_64-w64-mingw32-clang.exe")) { Say "BUILD FAILED: no llvm-mingw at $mingwBin"; exit 2 }
    $env:RUSTFLAGS = '-Clinker=x86_64-w64-mingw32-clang'
    $env:CARGO_HOME = Join-Path $base 'cargo-home'
    $env:CARGO_TARGET_DIR = Join-Path $base 'target'
    $env:CARGO_BUILD_JOBS = '16'
    $env:RUST_TEST_THREADS = '8'
    if (-not $env:INVAR_TEST_TIMEOUT_SCALE) { $env:INVAR_TEST_TIMEOUT_SCALE = '8' }
    # build.rs embeds this revision; the extracted archive has no git checkout to read it from.
    if ($Revision) { $env:INVAR_SOURCE_REVISION = $Revision }
    $out = @{ dev = 'debug'; test = 'debug'; release = 'release'; bench = 'release' }[$Profile]
    if (-not $out) { $out = $Profile }
    $env:INVAR_NVRTC_HELPER = Join-Path $base "target\$out\invar-nvrtc.exe"
    $env:INVAR_NVRTC_COMPILER = $env:INVAR_NVRTC_HELPER
    # cudarc loads nvcuda.dll from System32 and the NVRTC DLL from PATH.
    # GNU timeout for test-support/child_deadline.rs comes from a per-user Portable Git, appended last.
    $gnuBin = Join-Path $env:USERPROFILE 'toolchains\PortableGit\usr\bin'
    if (-not (Test-Path "$gnuBin\timeout.exe")) { Say "BUILD FAILED: no GNU timeout at $gnuBin"; exit 2 }
    $env:PATH = "$(Join-Path $base 'nvrtc');$mingwBin;$env:PATH;$gnuBin"
    $env:TMP = $env:TEMP = Join-Path $runDir 'tmp'
    New-Item -ItemType Directory $env:TMP -Force | Out-Null

    $targets = ($Tests.Split(',') | Where-Object { $_ } | ForEach-Object { "--test $_" }) -join ' '
    $cargo = Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe'
    $skips = ($Skip.Split(',') | Where-Object { $_ } | ForEach-Object { "--skip $_" }) -join ' '
    $cmd = "`"$cargo`" test --locked --profile $Profile --features cuda --no-fail-fast --lib $targets -- --include-ignored $skips $Filter > `"$log`" 2>&1"
    $expected = $Tests.Split(',').Count + 1
    Push-Location $src
    $p = Start-Process cmd.exe -ArgumentList '/c', "`"$cmd`"" -NoNewWindow -PassThru
    $p.Handle | Out-Null  # keep the handle so ExitCode is readable after exit
    $timedOut = -not $p.WaitForExit($TimeoutMin * 60000)
    if ($timedOut) { & taskkill.exe /T /F /PID $p.Id | Out-Null }
    Pop-Location

    $text = Get-Content $log
    $rows = @(); $current = $null
    foreach ($line in $text) {
        if ($line -match '^\s+Running (?:unittests )?(\S+)') { $current = $Matches[1] }
        elseif ($line -match '^test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored') {
            $rows += [pscustomobject]@{ Target = $current; Passed = [int]$Matches[2]; Failed = [int]$Matches[3]; Ignored = [int]$Matches[4] }
        }
    }
    $rows | Format-Table -AutoSize | Out-String -Width 200 | Write-Output
    $text | Where-Object { $_ -match '^test .* \.\.\. FAILED$' } | ForEach-Object { Say $_ }
    $sum = { param($k) ($rows | Measure-Object -Property $k -Sum).Sum }
    Say ("summary: {0} passed, {1} failed, {2} ignored in {3} of {4} targets (lib included)" -f `
        (& $sum Passed), (& $sum Failed), (& $sum Ignored), $rows.Count, $expected)
    if ($Skip) { Say "skipped known Windows gaps: $Skip" }
    $g = Gpu
    Say "gpu after: $($g.Name), driver $($g.Driver), util $($g.Util)%, free $($g.Free) MiB"
    if ($timedOut) { Say "TIMEOUT after $TimeoutMin min; process tree killed"; exit 4 }
    if ($rows.Count -eq 0) { $text | Select-Object -Last 40 | Write-Output; Say 'BUILD FAILED'; exit 2 }
    # A run that executed nothing, or a filter that matched no executed test, proves nothing.
    $ran = @($text | Where-Object { $_ -match '^test (\S+) \.\.\. (ok|FAILED)$' } | ForEach-Object { ($_ -split ' ')[1] })
    if ($ran.Count -eq 0) { Say 'NO TESTS RAN: check the filters against this ref'; exit 1 }
    $unmatched = @($Filter.Split(' ') | Where-Object { $_ } | Where-Object { $f = $_; -not ($ran | Where-Object { $_.Contains($f) }) })
    if ($unmatched) { Say "FILTER MATCHED NO TEST: $($unmatched -join ', ')"; exit 1 }
    # A target without a result row also fails the run.
    if ($p.ExitCode -ne 0 -or (& $sum Failed) -gt 0 -or $rows.Count -lt $expected) {
        if ($rows.Count -lt $expected) { Say 'some targets did not run (build error?); see the log' }
        exit 1
    }
    Say 'PASS'; exit 0
} finally {
    $held.Dispose()
    Remove-Item (Join-Path $runDir 'tmp'), (Join-Path $runDir 'src.tgz') -Recurse -Force -ErrorAction SilentlyContinue
}
