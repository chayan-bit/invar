"""Bounded independent MuJoCo comparison, not ROS/controller acceptance."""
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


def run(original_path, projected_path):
    original = mujoco.MjModel.from_xml_path(str(original_path))
    projected = mujoco.MjModel.from_xml_path(str(projected_path))
    fields = []
    # Names and the name hash/address tables intentionally differ. Every other
    # exposed numeric model array must retain its exact shape, dtype and bytes.
    for name in sorted(dir(original)):
        if name.startswith("_") or name.startswith("name_") or name.startswith("names"):
            continue
        left, right = getattr(original, name), getattr(projected, name)
        if not isinstance(left, np.ndarray):
            continue
        if left.dtype.kind not in "biuf":
            continue
        if left.shape != right.shape or left.dtype != right.dtype or left.tobytes() != right.tobytes():
            raise ValueError(f"physical model array changed: {name}")
        fields.append(name)
    if not fields:
        raise ValueError("no physical model arrays checked")
    trajectories = []
    for seed in (0, 17, 29):
        left, right = mujoco.MjData(original), mujoco.MjData(projected)
        mujoco.mj_resetDataKeyframe(original, left, 0)
        mujoco.mj_resetDataKeyframe(projected, right, 0)
        random = np.random.default_rng(seed)
        controls = left.ctrl.copy()
        for step in range(1000):
            if step % 100 == 0:
                controls = np.clip(left.ctrl + random.uniform(-0.01, 0.01, original.nu),
                                   original.actuator_ctrlrange[:, 0], original.actuator_ctrlrange[:, 1])
            left.ctrl[:] = right.ctrl[:] = controls
            mujoco.mj_step(original, left)
            mujoco.mj_step(projected, right)
            for field in ("qpos", "qvel", "qacc", "act", "ctrl", "actuator_force", "qfrc_constraint"):
                a, b = getattr(left, field), getattr(right, field)
                if not np.isfinite(a).all() or a.tobytes() != b.tobytes():
                    raise ValueError(f"state differs or is nonfinite: seed={seed} step={step} field={field}")
            if not np.array_equal(left.warning.number, right.warning.number) or np.any(left.warning.number):
                raise ValueError(f"MuJoCo warning: seed={seed} step={step}")
        trajectories.append({"seed": seed, "steps": 1000, "state_comparison": "bit_exact"})
    return {
        "status": "passed", "scope": "CPU_MuJoCo_naming_equivalence_only",
        "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
        "original_scene_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "projected_scene_sha256": hashlib.sha256(projected_path.read_bytes()).hexdigest(),
        "unchanged_numeric_model_arrays": fields, "trajectories": trajectories,
        "ROS_controller_acceptance": "not_run", "S1_06_task_acceptance": "not_run",
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: tendon_projection_equivalence.py ORIGINAL_SCENE PROJECTED_SCENE")
    print(json.dumps(run(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
