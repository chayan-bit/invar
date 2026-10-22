"""Compare Pinocchio and MuJoCo FK at joint states recorded in an S1-06 MCAP."""

import hashlib
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(bag, scene, model_path, golden_path, arm):
    records = json.loads(golden_path.read_text())
    spec = next(entry for entry in records["arms"] if entry["arm"] == arm)
    if digest(model_path) != spec["model_sha256"]:
        raise ValueError("model differs from independent Pinocchio oracle")
    files = list(bag.glob("*.mcap"))
    if len(files) != 1 or files[0].stat().st_size > 64 * 1024 * 1024:
        raise ValueError("expected one bounded MCAP")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    required = {
        "/joint_states",
        "/fk/pose",
        "/invar/tool_pose",
        "/invar/object_pose",
        "/invar/contact_geom_ids",
        "/invar/attachment_active",
        "/clock",
    }
    if not required <= types.keys():
        raise ValueError(f"missing recorded topics: {sorted(required - types.keys())}")
    counts = dict.fromkeys(sorted(required), 0)
    joint_type = get_message(types["/joint_states"])
    samples = []
    names = spec["joint_names"]
    while reader.has_next():
        topic, data = reader.read_next_ext()[:2]
        if topic in counts:
            counts[topic] += 1
        if topic != "/joint_states":
            continue
        message = deserialize_message(data, joint_type)
        if len(message.name) != len(message.position) or len(set(message.name)) != len(message.name):
            raise ValueError("invalid recorded JointState")
        positions = dict(zip(message.name, message.position, strict=True))
        q = [positions[name] for name in names]
        if not all(math.isfinite(value) for value in q):
            raise ValueError("nonfinite recorded joint")
        samples.append(q)
        if len(samples) > 200_000:
            raise ValueError("joint sample limit exceeded")
    if len(samples) < 16:
        raise ValueError("insufficient recorded joint samples")
    if not all(counts.values()):
        raise ValueError(f"empty recorded topics: {counts}")

    pin_model = pin.buildModelFromMJCF(str(model_path))
    pin_data = pin_model.createData()
    pin_indices = [pin_model.idx_qs[pin_model.getJointId(name)] for name in names]
    pin_base = pin_model.getFrameId(spec["base_frame"])
    pin_tool = pin_model.getFrameId(spec["tool_frame"])
    mj_model = mujoco.MjModel.from_xml_path(str(scene))
    mj_data = mujoco.MjData(mj_model)
    mj_indices = [mj_model.joint(name).qposadr for name in names]
    mj_base = mj_model.body(spec["base_frame"]).id
    mj_tool = mj_model.site("invar_tool").id

    errors = []
    for index in sorted({i * (len(samples) - 1) // 15 for i in range(16)}):
        q = samples[index]
        pin_q = pin.neutral(pin_model)
        for slot, value in zip(pin_indices, q, strict=True):
            pin_q[slot] = value
        pin.forwardKinematics(pin_model, pin_data, pin_q)
        pin.updateFramePlacements(pin_model, pin_data)
        tool_in_base = pin_data.oMf[pin_base].inverse() * pin_data.oMf[pin_tool]
        for slot, value in zip(mj_indices, q, strict=True):
            mj_data.qpos[slot] = value
        mujoco.mj_forward(mj_model, mj_data)
        predicted_world = (
            mj_data.xpos[mj_base]
            + mj_data.xmat[mj_base].reshape(3, 3) @ tool_in_base.translation
        )
        error = float(np.linalg.norm(predicted_world - mj_data.site_xpos[mj_tool]))
        if not math.isfinite(error):
            raise ValueError("nonfinite FK comparison")
        errors.append(error)
    maximum = max(errors)
    result = {
        "schema": "invar-s106-fk-mujoco-v1",
        "arm": arm,
        "recorded_joint_samples": len(samples),
        "recorded_topic_counts": counts,
        "compared_samples": len(errors),
        "max_translation_error_m": maximum,
        "threshold_m": 0.000001,
        "mcap_sha256": digest(files[0]),
        "scene_sha256": digest(scene),
        "model_sha256": digest(model_path),
        "pinocchio_version": pin.__version__,
        "mujoco_version": mujoco.__version__,
        "status": "pass" if maximum <= 0.000001 else "fail",
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit("usage: fk_mujoco_crosscheck.py BAG SCENE MODEL GOLDEN ARM")
    main(*(Path(value) for value in sys.argv[1:5]), sys.argv[5])
