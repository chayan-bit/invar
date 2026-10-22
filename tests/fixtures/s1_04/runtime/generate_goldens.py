"""Independent stdlib oracle for the private planar two-link C fixture."""

import json
import math
import random
import sys
from pathlib import Path


def main(output: Path) -> None:
    proximal, distal = 0.4, 0.3
    seed = 20260927
    rng = random.Random(seed)
    joints = [(0.0, 0.0), (math.pi / 2, 0.0), (math.pi / 2, -math.pi / 2)]
    joints.extend((rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)) for _ in range(4))
    vectors = []
    for q0, q1 in joints:
        yaw = q0 + q1
        vectors.append(
            {
                "q_rad": [q0, q1],
                "x_m": math.fsum((proximal * math.cos(q0), distal * math.cos(yaw))),
                "y_m": math.fsum((proximal * math.sin(q0), distal * math.sin(yaw))),
                "yaw_rad": yaw,
            }
        )
    record = {
        "source": "independent Python stdlib math oracle for ideal planar 2-link geometry",
        "seed": seed,
        "lengths_m": [proximal, distal],
        "vectors": vectors,
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: generate_goldens.py OUTPUT_JSON")
    main(Path(sys.argv[1]))
