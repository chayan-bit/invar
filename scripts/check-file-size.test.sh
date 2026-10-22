#!/bin/sh
set -eu
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
checker=${FILE_SIZE_CHECKER:-"$script_dir/check-file-size"}
: "${TMPDIR:?Set a private TMPDIR before running the self-test}"
exec python3 - "$checker" <<'PY'
import os
from pathlib import Path
import subprocess
import sys
import tempfile

checker = str(Path(sys.argv[1]).resolve())
env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
for key in list(env):
    if key.startswith("GIT_") and key not in ("GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL"):
        del env[key]


def git(*args):
    return subprocess.run(["git", *args], env=env, check=True,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def write(path, lines, newline_at_end=True):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("fn x() {}\n" * (lines - 1) + "fn x() {}" + ("\n" if newline_at_end else ""))


def allow(*entries):
    Path("scripts").mkdir(exist_ok=True)
    Path("scripts/file-size-allowlist.txt").write_text("# header\n\n" + "".join(f"{n} {p}\n" for n, p in entries))


def run(*args):
    return subprocess.run([checker, *args], env=env, capture_output=True, text=True)


def expect(name, code, needles=(), absent=(), args=()):
    result = run(*args)
    output = result.stdout + result.stderr
    if result.returncode != code or any(n not in output for n in needles) or any(a in output for a in absent):
        print(f"FAIL: {name} (exit {result.returncode}, expected {code})\n{output}")
        sys.exit(1)
    print(f"PASS: {name}")


with tempfile.TemporaryDirectory(prefix="file-size-test-", dir=os.environ["TMPDIR"]) as work:
    os.chdir(work)
    git("init", "-q")
    allow()
    write("src/small.rs", 10)
    write("src/edge.rs", 800)
    write("docs/big.md", 2000)
    write("tools/big.py", 2000)
    git("add", "-A")
    expect("empty allowlist, nothing over 800 lines", 0, ["0 violations"])

    write("src/big.rs", 801)
    git("add", "src/big.rs")
    expect("a new 801-line file is refused", 1, ["FAIL src/big.rs: 801 lines", "not on the allowlist"])
    write("src/big.rs", 801, newline_at_end=False)
    expect("an unterminated last line counts as wc -l does", 0, ["0 violations"])
    write("src/big.rs", 801)

    write("untracked.rs", 5000)
    expect("untracked files are not checked", 1, ["src/big.rs"], absent=["untracked.rs"])

    allow((801, "src/big.rs"))
    expect("an allowlisted file at its recorded size passes", 0, ["0 violations", "1 allowlisted"])
    write("src/big.rs", 802)
    expect("an allowlisted file may not grow", 1, ["FAIL src/big.rs: grew from 801 to 802"])
    write("src/big.rs", 801)

    allow((900, "src/big.rs"))
    expect("shrinking is allowed, with a note", 0, ["note src/big.rs: shrank from 900 to 801", "0 violations"])
    write("src/big.rs", 700)
    expect("falling under the limit passes, with a note to drop the entry", 0, ["now 700 lines; drop the entry"])
    allow((900, "src/gone.rs"))
    expect("an entry for an untracked path is only a note", 0, ["src/gone.rs: allowlisted but no longer tracked"])

    write("src/big.rs", 850)
    git("add", "src/big.rs")
    allow((900, "src/big.rs"))
    expect("--update lowers an entry", 0, ["1 entries"], args=["--update"])
    assert "850 src/big.rs\n" in Path("scripts/file-size-allowlist.txt").read_text()
    assert Path("scripts/file-size-allowlist.txt").read_text().startswith("# Tracked .rs files over 800 lines")
    write("src/big.rs", 700)
    expect("--update drops an entry that is back under the limit", 0, ["0 entries"], args=["--update"])
    assert "src/big.rs" not in Path("scripts/file-size-allowlist.txt").read_text()
    write("src/big.rs", 820)
    expect("--update never adds an entry", 1, ["not below its allowlisted size"], args=["--update"])
    allow((810, "src/big.rs"))
    expect("--update never raises an entry", 1, ["not below its allowlisted size"], args=["--update"])
    assert "810 src/big.rs\n" in Path("scripts/file-size-allowlist.txt").read_text()
    expect("--regenerate records the current tree", 0, ["regenerated with 1 entries"], args=["--regenerate"])
    assert "820 src/big.rs\n" in Path("scripts/file-size-allowlist.txt").read_text()
    expect("the regenerated allowlist passes", 0, ["0 violations"])

    write("src/with space.rs", 900)
    git("add", "src/with space.rs")
    allow((820, "src/big.rs"), (900, "src/with space.rs"))
    expect("a path with a space", 0, ["2 allowlisted"])

    # A tracked file that is not in the work tree (a sparse checkout without it) must not pass silently.
    Path("src/small.rs").unlink()
    expect("a tracked file missing from the work tree is unusable input", 2, ["src/small.rs is tracked but unreadable"])
    git("checkout", "--", "src/small.rs")

    for name, text, needle in [
        ("a malformed line", "abc src/big.rs\n", "expected `<lines> <path>`"),
        ("a missing path", "900\n", "expected `<lines> <path>`"),
        ("a duplicate entry", "900 src/big.rs\n901 src/big.rs\n", "duplicate entry"),
        ("an entry not over the limit", "800 src/edge.rs\n", "not over 800"),
    ]:
        Path("scripts/file-size-allowlist.txt").write_text(text)
        expect(name + " in the allowlist", 2, [needle])
    Path("scripts/file-size-allowlist.txt").unlink()
    expect("a missing allowlist", 2, ["scripts/file-size-allowlist.txt is missing"])
    expect("an unknown option", 2, ["usage:"], args=["--bogus"])

os.chdir(tempfile.gettempdir())
with tempfile.TemporaryDirectory(prefix="file-size-nogit-", dir=os.environ["TMPDIR"]) as work:
    os.chdir(work)
    expect("outside a work tree", 2, ["not inside a Git work tree"])
PY
