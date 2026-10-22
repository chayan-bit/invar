"""The Linux shim exposes exactly the agreed C symbols."""

import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "invar_pinocchio_v1_version",
    "invar_pinocchio_v1_init",
    "invar_pinocchio_v1_configure",
    "invar_pinocchio_v1_step",
    "invar_pinocchio_v1_shutdown",
}


def main(library: Path) -> None:
    output = subprocess.check_output(["nm", "-D", "--defined-only", str(library)], text=True)
    exported = {line.split()[-1] for line in output.splitlines() if line.split()}
    assert exported == EXPECTED, ("unexpected exports", sorted(exported ^ EXPECTED))
    print("Pinocchio reference C ABI: exactly five declared exports")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_pinocchio_exports.py SHARED_LIBRARY")
    main(Path(sys.argv[1]))
