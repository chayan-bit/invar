"""Generate FK goldens with Pinocchio's public Python binding, apart from the C++ shim."""

import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pinocchio as pin


REVISION = "c96a32d28fb5da84da38c1da4d749e7a13212855"
SEED = 20260927
ARMS = (
    {
        "name": "panda",
        "model": "franka_emika_panda/panda.xml",
        "license": "franka_emika_panda/LICENSE",
        "license_id": "Apache-2.0",
        "base_frame": "link0",
        "tool_frame": "hand",
        "joint_names": [f"joint{i}" for i in range(1, 8)],
        "home_rad": [0.0, -0.8, 0.0, -2.3, 0.0, 1.5, 0.8],
    },
    {
        "name": "ur5e",
        "model": "universal_robots_ur5e/ur5e.xml",
        "license": "universal_robots_ur5e/LICENSE",
        "license_id": "BSD-3-Clause",
        "base_frame": "base",
        "tool_frame": "attachment_site",
        "joint_names": [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        "home_rad": [0.0, -1.0, 1.0, -1.0, 0.5, 0.0],
    },
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_id(model: pin.Model, name: str) -> int:
    for index, frame in enumerate(model.frames):
        if frame.name == name:
            return index
    raise ValueError(f"frame {name!r} absent; known frames: {[f.name for f in model.frames]}")


def golden_for_arm(root: Path, spec: dict, rng: random.Random) -> dict:
    model_path = root / spec["model"]
    license_path = root / spec["license"]
    if not model_path.is_file() or not license_path.is_file():
        raise FileNotFoundError(model_path if not model_path.is_file() else license_path)
    model = pin.buildModelFromMJCF(str(model_path))
    data = model.createData()
    base_id = frame_id(model, spec["base_frame"])
    tool_id = frame_id(model, spec["tool_frame"])
    joint_ids = [model.getJointId(name) for name in spec["joint_names"]]
    if any(jid >= model.njoints or model.nqs[jid] != 1 for jid in joint_ids):
        raise ValueError(f"missing or non-scalar joint in {spec['name']}: {joint_ids}")
    indices = [model.idx_qs[jid] for jid in joint_ids]
    limits = [(float(model.lowerPositionLimit[i]), float(model.upperPositionLimit[i])) for i in indices]
    if any(not (math.isfinite(lo) and math.isfinite(hi) and lo < hi) for lo, hi in limits):
        raise ValueError(f"finite joint limits required for {spec['name']}: {limits}")

    vectors = [spec["home_rad"]]
    for _ in range(3):
        vectors.append([rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo)) for lo, hi in limits])
    output = []
    for arm_q in vectors:
        if any(not (lo <= value <= hi) for value, (lo, hi) in zip(arm_q, limits, strict=True)):
            raise ValueError(f"out-of-limit vector for {spec['name']}: {arm_q}")
        full_q = pin.neutral(model)
        for index, value in zip(indices, arm_q, strict=True):
            full_q[index] = value
        pin.forwardKinematics(model, data, full_q)
        pin.updateFramePlacements(model, data)
        tool_in_base = data.oMf[base_id].inverse() * data.oMf[tool_id]
        rotation = pin.Quaternion(tool_in_base.rotation)
        quaternion = [float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)]
        first_nonzero = next((value for value in quaternion if abs(value) > 1e-15), 1.0)
        if first_nonzero < 0.0:
            quaternion = [-value for value in quaternion]
        if not math.isclose(math.sqrt(sum(value * value for value in quaternion)), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Pinocchio returned a non-unit quaternion")
        output.append(
            {
                "q_rad": [float(value) for value in arm_q],
                "translation_m": [float(value) for value in tool_in_base.translation],
                "quaternion_wxyz": quaternion,
            }
        )
    return {
        "arm": spec["name"],
        "model_file": spec["model"],
        "model_sha256": sha256(model_path),
        "license_file": spec["license"],
        "license_id": spec["license_id"],
        "license_sha256": sha256(license_path),
        "base_frame": spec["base_frame"],
        "tool_frame": spec["tool_frame"],
        "joint_names": spec["joint_names"],
        "joint_limits_rad": [[lo, hi] for lo, hi in limits],
        "vectors": output,
    }


def main(root: Path, output: Path) -> None:
    root = root.resolve(strict=True)
    git = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    actual_revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if actual_revision != REVISION:
        raise ValueError(f"Menagerie revision {actual_revision} != {REVISION}")
    if subprocess.check_output([*git, "status", "--porcelain"]):
        raise ValueError("Menagerie checkout has local changes")
    if pin.__version__ != "4.1.0":
        raise ValueError(f"Pinocchio version {pin.__version__} != 4.1.0")
    rng = random.Random(SEED)
    record = {
        "oracle": "Pinocchio public Python binding, separate from C++ shim",
        "pinocchio_version": pin.__version__,
        "menagerie_revision": REVISION,
        "seed": SEED,
        "pose_convention": "tool-in-base; translation metres; private oracle quaternion wxyz, first nonzero component positive",
        "arms": [golden_for_arm(root, spec, rng) for spec in ARMS],
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"pinocchio={pin.__version__} revision={REVISION} vectors={[len(a['vectors']) for a in record['arms']]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: generate_pinocchio_goldens.py PINNED_MENAGERIE_ROOT OUTPUT_JSON")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
