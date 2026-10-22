"""Compare the C++ Pinocchio core with Python-binding goldens from a separate run."""

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cli: Path, model: Path, arm: dict, q: list[float]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(cli), str(model), arm["base_frame"], arm["tool_frame"], str(len(arm["joint_names"])),
         *arm["joint_names"], *(repr(value) for value in q)],
        text=True, capture_output=True, check=False, timeout=30,
    )


def main(cli: Path, goldens: Path, root: Path) -> None:
    record = json.loads(goldens.read_text())
    assert record["pinocchio_version"] == "4.1.0"
    assert record["seed"] == 20260927
    total = 0
    for arm in record["arms"]:
        model = root / arm["model_file"]
        assert sha256(model) == arm["model_sha256"]
        assert sha256(root / arm["license_file"]) == arm["license_sha256"]
        for vector in arm["vectors"]:
            result = run(cli, model, arm, vector["q_rad"])
            assert result.returncode == 0, result.stderr
            values = [float(value) for value in result.stdout.split()]
            assert len(values) == 7 and all(map(math.isfinite, values)), result.stdout
            for actual, expected in zip(values[:3], vector["translation_m"], strict=True):
                assert math.isclose(actual, expected, abs_tol=1e-12, rel_tol=0.0), (arm["arm"], actual, expected)
            actual_q, expected_q = values[3:], vector["quaternion_wxyz"]
            distance = min(
                math.dist(actual_q, expected_q),
                math.dist(actual_q, [-value for value in expected_q]),
            )
            angle_error = 2.0 * math.asin(min(1.0, distance / 2.0))
            assert angle_error <= 1e-12, (arm["arm"], angle_error)
            total += 1

        sample = arm["vectors"][0]["q_rad"]
        for bad_q in (sample[:-1], [math.nan, *sample[1:]], [math.inf, *sample[1:]], [1000.0, *sample[1:]]):
            result = run(cli, model, arm, bad_q)
            assert result.returncode != 0 and not result.stdout, (arm["arm"], result)
        wrong_frame = dict(arm, tool_frame="missing_tool_frame")
        result = run(cli, model, wrong_frame, sample)
        assert result.returncode != 0 and not result.stdout

    print(f"pinocchio C++ core: {total} independent Python goldens and invalid-input cases passed")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: test_pinocchio_goldens.py CLI GOLDENS_JSON MODEL_ROOT")
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
