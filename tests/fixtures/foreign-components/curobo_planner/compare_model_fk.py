#!/usr/bin/env python3
"""Compare pinned planner URDFs against independent Menagerie FK goldens."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_pinocchio_goldens import frame_id  # reuse accepted fixture helper


TOLERANCE_M = 1e-12
TOLERANCE_RAD = 1e-12
EXPECTED_PINOCCHIO = "4.1.0"
EXPECTED_XACRO = "2.1.1"
ROOT = Path(__file__).resolve().parent


def check_files(source_root, record):
    for entry in record["files"]:
        path = source_root / entry["path"]
        if (not path.is_file() or path.stat().st_size != entry["bytes"] or
                hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]):
            raise ValueError(f"source byte mismatch: {entry['path']}")


def derive_ur5e(source_root, xacro, output):
    import importlib.metadata
    if importlib.metadata.version("xacro") != EXPECTED_XACRO:
        raise ValueError("xacro version mismatch")
    source = source_root / "ur_e_description/urdf"
    with tempfile.TemporaryDirectory(prefix="invar-ur5e-xacro-") as scratch:
        work = Path(scratch)
        for name in ("ur5e_robot.urdf.xacro", "ur5e.urdf.xacro",
                     "ur.transmission.xacro", "ur.gazebo.xacro", "common.gazebo.xacro"):
            contents = (source / name).read_text()
            contents = contents.replace("$(find ur_e_description)/urdf/", str(work) + "/")
            (work / name).write_text(contents)
        result = subprocess.run([str(xacro), str(work / "ur5e_robot.urdf.xacro")],
                                capture_output=True, check=True, timeout=60)
        robot = ET.fromstring(result.stdout)
        if robot.get("name") != "ur5e":
            raise ValueError("unexpected generated robot")
        output.write_bytes(ET.tostring(robot, encoding="utf-8", xml_declaration=True))


def derive_panda(source, output):
    original = source.read_bytes()
    derived = original
    changes = (
        (b'rpy="-1.57079632679 0 0"', b'rpy="-1.5707963267948966 0 0"', 2),
        (b'rpy="1.57079632679 0 0"', b'rpy="1.5707963267948966 0 0"', 4),
        (b'rpy="0 0 -0.785398163397"', b'rpy="0 0 -0.78539812847762047"', 1),
    )
    for old, new, expected_count in changes:
        if derived.count(old) != expected_count or new in derived:
            raise ValueError("unexpected Panda source transform")
        derived = derived.replace(old, new)
    restored = derived
    for old, new, _ in changes:
        restored = restored.replace(new, old)
    if restored != original:
        raise ValueError("Panda derivation altered other bytes")
    output.write_bytes(derived)


def model_pose(model, q_values, names, base, tool):
    q = pin.neutral(model)
    for name, value in zip(names, q_values, strict=True):
        joint_id = model.getJointId(name)
        if joint_id >= model.njoints or model.nqs[joint_id] != 1:
            raise ValueError(f"missing/scalar joint: {name}")
        q[model.idx_qs[joint_id]] = value
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[frame_id(model, base)].inverse() * data.oMf[frame_id(model, tool)]


def delta(a, b):
    relative = pin.Quaternion(a.rotation.T @ b.rotation)
    return (float(np.linalg.norm(a.translation - b.translation)),
            float(2 * np.arctan2(np.linalg.norm([relative.x, relative.y, relative.z]),
                                 abs(relative.w))))


def golden_pose(vector):
    w, x, y, z = vector["quaternion_wxyz"]
    return pin.SE3(pin.Quaternion(w, x, y, z).toRotationMatrix(),
                   np.array(vector["translation_m"]))


def joint_limits(model, names):
    limits = []
    for name in names:
        jid = model.getJointId(name)
        if jid >= model.njoints:
            raise ValueError(f"missing joint: {name}")
        index = model.idx_qs[jid]
        limits.append((float(model.lowerPositionLimit[index]),
                       float(model.upperPositionLimit[index])))
    return limits


def compare(arm, planner_model, menagerie_model, golden, names, base, tool):
    all_errors = []
    for vector in golden["vectors"]:
        planner = model_pose(planner_model, vector["q_rad"], names, base, tool)
        all_errors.append(delta(golden_pose(vector), planner))
    planner_limits = joint_limits(planner_model, names)
    original_limits = golden["joint_limits_rad"]
    home = list(golden["vectors"][0]["q_rad"])
    for i, ((lo_a, hi_a), (lo_b, hi_b)) in enumerate(zip(
            planner_limits, original_limits, strict=True)):
        for value in (max(lo_a, lo_b), min(hi_a, hi_b)):
            probe = home.copy()
            probe[i] = value
            original = model_pose(menagerie_model, probe, golden["joint_names"],
                                  golden["base_frame"], golden["tool_frame"])
            planner = model_pose(planner_model, probe, names, base, tool)
            all_errors.append(delta(original, planner))
    max_m = max(value[0] for value in all_errors)
    max_rad = max(value[1] for value in all_errors)
    print(f"{arm}: poses={len(all_errors)} max_translation_m={max_m:.17g} "
          f"max_rotation_rad={max_rad:.17g} tolerance_m={TOLERANCE_M} "
          f"tolerance_rad={TOLERANCE_RAD}")
    print(f"{arm}: planner_limits={planner_limits} menagerie_limits={original_limits}")
    return max_m <= TOLERANCE_M and max_rad <= TOLERANCE_RAD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curobo-root", type=Path, required=True)
    parser.add_argument("--ur-root", type=Path, required=True)
    parser.add_argument("--menagerie-root", type=Path, required=True)
    parser.add_argument("--goldens", type=Path, required=True)
    parser.add_argument("--xacro", type=Path, required=True)
    parser.add_argument("--output-urdf", type=Path, required=True)
    parser.add_argument("--output-panda-urdf", type=Path, required=True)
    args = parser.parse_args()
    if pin.__version__ != EXPECTED_PINOCCHIO:
        raise ValueError("Pinocchio version mismatch")
    manifest = json.loads((ROOT / "model_sources.json").read_text())
    check_files(args.curobo_root, manifest["curobo_franka"])
    check_files(args.ur_root, manifest["ros_ur5e"])
    golden = json.loads(args.goldens.read_text())
    if golden["menagerie_revision"] != manifest["menagerie"]["revision"]:
        raise ValueError("Menagerie revision mismatch")
    for arm in golden["arms"]:
        entry = manifest["menagerie"]["models"][arm["arm"]]
        path = args.menagerie_root / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Menagerie model mismatch")
        license_path = args.menagerie_root / entry["license_path"]
        if (license_path.stat().st_size != entry["license_bytes"] or
                hashlib.sha256(license_path.read_bytes()).hexdigest() !=
                entry["license_sha256"]):
            raise ValueError("Menagerie license mismatch")
    args.output_urdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_panda_urdf.parent.mkdir(parents=True, exist_ok=True)
    derive_ur5e(args.ur_root, args.xacro, args.output_urdf)
    panda_source = (args.curobo_root /
        "curobo/content/assets/robot/franka_description/franka_panda.urdf")
    derive_panda(panda_source, args.output_panda_urdf)
    print("generated_ur5e_urdf_sha256=" + hashlib.sha256(args.output_urdf.read_bytes()).hexdigest())
    print("derived_panda_urdf_sha256=" + hashlib.sha256(args.output_panda_urdf.read_bytes()).hexdigest())
    records = {a["arm"]: a for a in golden["arms"]}
    panda_upstream = pin.buildModelFromUrdf(str(panda_source))
    panda = pin.buildModelFromUrdf(str(args.output_panda_urdf))
    ur5e = pin.buildModelFromUrdf(str(args.output_urdf))
    original_panda = pin.buildModelFromMJCF(str(args.menagerie_root / "franka_emika_panda/panda.xml"))
    original_ur5e = pin.buildModelFromMJCF(str(args.menagerie_root / "universal_robots_ur5e/ur5e.xml"))
    panda_names = [f"panda_joint{i}" for i in range(1, 8)]
    compare("panda_upstream", panda_upstream, original_panda, records["panda"],
            panda_names, "panda_link0", "panda_hand")
    passed = compare("panda_derived", panda, original_panda, records["panda"],
                     panda_names, "panda_link0", "panda_hand")
    passed &= compare("ur5e", ur5e, original_ur5e, records["ur5e"],
                      records["ur5e"]["joint_names"], "base_link", "tool0")
    if not passed:
        raise SystemExit("FK agreement failed")


if __name__ == "__main__":
    main()
