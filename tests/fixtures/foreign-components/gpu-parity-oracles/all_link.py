#!/usr/bin/env python3
"""Full original-MJCF body-origin FK from independent Pinocchio and MuJoCo imports."""

import argparse
import json
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

from run import mj_data, pin_q, sha, values


def bodies_in_source(path):
    found = []

    def scan(node, parent):
        for body in node.findall("body"):
            name = body.attrib["name"]
            found.append([name, parent])
            scan(body, name)

    for world in ET.parse(path).getroot().findall("worldbody"):
        scan(world, None)
    return found


def pin_body_frame(model, name):
    matches = [i for i, frame in enumerate(model.frames) if frame.name == name]
    if len(matches) != 1 or str(model.frames[matches[0]].type) != "BODY":
        raise ValueError(f"missing, duplicate or wrong-type Pinocchio BODY frame: {name}")
    return matches[0]


def mj_body(model, name):
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if index < 0 or model.body(index).name != name:
        raise ValueError(f"MuJoCo BODY missing: {name}")
    return index


def pose_in_base(rotation_world_base, position_world_base, rotation_world_body, position_world_body):
    base_from_world = rotation_world_base.T
    return base_from_world @ (position_world_body - position_world_base), base_from_world @ rotation_world_body


def relative(parent, child):
    parent_t, parent_r = parent
    child_t, child_r = child
    return parent_r.T @ (child_t - parent_t), parent_r.T @ child_r


def compose(parent, local):
    parent_t, parent_r = parent
    local_t, local_r = local
    return parent_t + parent_r @ local_t, parent_r @ local_r


def rotation_from_source_quat(text):
    quat = np.fromstring(text, sep=" ")
    if quat.shape != (4,) or not np.all(np.isfinite(quat)) or np.linalg.norm(quat) == 0:
        raise ValueError("invalid source quaternion")
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def checked(a, b, limit):
    error = float(np.max(np.abs(a - b)))
    return {"passed": bool(np.isfinite(error) and error <= limit), "max_abs": repr(error)}


def rotation_check(rotation, limits):
    orth = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = abs(float(np.linalg.det(rotation)) - 1)
    return {"passed": bool(np.all(np.isfinite(rotation)) and
                           orth <= limits["rotation_orthogonality_abs_tolerance"] and
                           determinant <= limits["rotation_determinant_abs_tolerance"]),
            "max_orthogonality_abs": repr(orth), "determinant_abs_from_one": repr(determinant)}


def run(args):
    all_path, source_contract_path, reference_path = map(Path, (args.contract, args.source_contract, args.reference))
    contract = json.loads(all_path.read_text())
    original = json.loads(source_contract_path.read_text())
    reference = json.loads(reference_path.read_text())
    if contract["schema_version"] != 1 or contract["source_contract_sha256"] != sha(source_contract_path) or \
            contract["frozen_input_reference_sha256"] != sha(reference_path) or \
            pin.__version__ != original["libraries"]["pinocchio"] or \
            mujoco.__version__ != original["libraries"]["mujoco"] or \
            [m["name"] for m in contract["models"]] != [m["name"] for m in original["models"]]:
        raise ValueError("all-body contract/dependency mismatch")
    limits = contract["checks"]
    output = {"all_link_contract_sha256": sha(all_path),
              "source_contract_sha256": sha(source_contract_path),
              "frozen_input_reference_sha256": sha(reference_path),
              "pinocchio_version": pin.__version__, "mujoco_version": mujoco.__version__,
              "numpy_version": np.__version__, "python_version": sys.version.split()[0],
              "platform": platform.platform(), "environment_image_id": args.image_id,
              "models": [], "runtime_status": "pending"}
    failures = []
    checked_transforms = 0
    for model_spec in contract["models"]:
        name = model_spec["name"]
        source_spec = next(s for s in original["models"] if s["name"] == name)
        frozen = next(s for s in reference["models"] if s["name"] == name)
        source = Path(args.models_root) / source_spec["file"]
        ordered = model_spec["ordered_bodies"]
        if sha(source) != model_spec["source_model_sha256"] or \
                sha(source.parent / "LICENSE") != source_spec["license_sha256"] or \
                bodies_in_source(source) != ordered or \
                [c["name"] for c in frozen["cases"]] != contract["case_names"]:
            raise ValueError(f"source/body/input mismatch: {name}")
        p_model = pin.buildModelFromMJCF(str(source))
        m_model = mujoco.MjModel.from_xml_path(str(source))
        if m_model.nbody - 1 != len(ordered):
            raise ValueError(f"compiled body count mismatch: {name}")
        p_frames = {body: pin_body_frame(p_model, body) for body, _ in ordered}
        m_bodies = {body: mj_body(m_model, body) for body, _ in ordered}
        p_data = p_model.createData()
        base = model_spec["base_body"]
        if ordered[0] != [base, None]:
            raise ValueError(f"invalid base order: {name}")
        source_hand = next((b for b in ET.parse(source).getroot().iter("body")
                            if b.get("name") == "hand"), None) if name == "panda" else None
        hand_local = None
        if name == "panda":
            if source_hand is None or source_hand.get("pos") is None or source_hand.get("quat") is None:
                raise ValueError("Panda fixed hand source frame missing")
            hand_local = (np.fromstring(source_hand.get("pos"), sep=" "),
                          rotation_from_source_quat(source_hand.get("quat")))
        record = {"name": name, "source_model_sha256": sha(source),
                  "ordered_bodies": ordered, "cases": []}
        for case in frozen["cases"]:
            q = list(map(float, case["q_rad"]))
            pin.forwardKinematics(p_model, p_data, pin_q(p_model, source_spec, q))
            pin.updateFramePlacements(p_model, p_data)
            m_data = mj_data(m_model, source_spec, q)
            p_base = p_data.oMf[p_frames[base]]
            m_base_id = m_bodies[base]
            m_base_r = np.asarray(m_data.xmat[m_base_id]).reshape(3, 3)
            m_base_t = np.asarray(m_data.xpos[m_base_id])
            poses_p, poses_m = {}, {}
            for body, _ in ordered:
                p_frame = p_data.oMf[p_frames[body]]
                poses_p[body] = (np.asarray((p_base.inverse() * p_frame).translation).copy(),
                                 np.asarray((p_base.inverse() * p_frame).rotation).copy())
                body_id = m_bodies[body]
                poses_m[body] = pose_in_base(m_base_r, m_base_t,
                                             np.asarray(m_data.xmat[body_id]).reshape(3, 3),
                                             np.asarray(m_data.xpos[body_id]))
            case_output = {"name": case["name"], "q_rad": case["q_rad"], "bodies": []}
            for body, parent in ordered:
                p_t, p_r = poses_p[body]
                m_t, m_r = poses_m[body]
                checks = {
                    "translation": checked(p_t, m_t, limits["translation_element_abs_tolerance_m"]),
                    "rotation": checked(p_r, m_r, limits["rotation_element_abs_tolerance"]),
                    "pinocchio_rotation_valid": rotation_check(p_r, limits),
                    "mujoco_rotation_valid": rotation_check(m_r, limits)}
                body_output = {"name": body, "source_parent": parent,
                               "pinocchio_translation_m": [repr(float(x)) for x in p_t],
                               "pinocchio_rotation_row_major": values(p_r),
                               "mujoco_translation_m": [repr(float(x)) for x in m_t],
                               "mujoco_rotation_row_major": values(m_r)}
                if parent is not None:
                    p_local, m_local = relative(poses_p[parent], poses_p[body]), relative(poses_m[parent], poses_m[body])
                    checks["parent_relative_translation"] = checked(
                        p_local[0], m_local[0], limits["parent_relative_element_abs_tolerance"])
                    checks["parent_relative_rotation"] = checked(
                        p_local[1], m_local[1], limits["parent_relative_element_abs_tolerance"])
                    p_recomposed, m_recomposed = compose(poses_p[parent], p_local), compose(poses_m[parent], m_local)
                    checks["parent_recomposition"] = {"passed": all((
                        checked(p_recomposed[0], p_t, limits["parent_recomposition_element_abs_tolerance"])["passed"],
                        checked(p_recomposed[1], p_r, limits["parent_recomposition_element_abs_tolerance"])["passed"],
                        checked(m_recomposed[0], m_t, limits["parent_recomposition_element_abs_tolerance"])["passed"],
                        checked(m_recomposed[1], m_r, limits["parent_recomposition_element_abs_tolerance"])["passed"]))}
                    if body == "hand":
                        checks["source_fixed_hand_translation"] = checked(
                            p_local[0], hand_local[0], limits["panda_fixed_hand_source_relative_element_abs_tolerance"])
                        checks["source_fixed_hand_rotation"] = checked(
                            p_local[1], hand_local[1], limits["panda_fixed_hand_source_relative_element_abs_tolerance"])
                        checks["mujoco_fixed_hand_translation"] = checked(
                            m_local[0], hand_local[0], limits["panda_fixed_hand_source_relative_element_abs_tolerance"])
                        checks["mujoco_fixed_hand_rotation"] = checked(
                            m_local[1], hand_local[1], limits["panda_fixed_hand_source_relative_element_abs_tolerance"])
                    body_output["pinocchio_parent_translation_m"] = [repr(float(x)) for x in p_local[0]]
                    body_output["pinocchio_parent_rotation_row_major"] = values(p_local[1])
                    body_output["mujoco_parent_translation_m"] = [repr(float(x)) for x in m_local[0]]
                    body_output["mujoco_parent_rotation_row_major"] = values(m_local[1])
                else:
                    checks["base_identity"] = {"passed": all((
                        checked(p_t, np.zeros(3), limits["translation_element_abs_tolerance_m"])["passed"],
                        checked(m_t, np.zeros(3), limits["translation_element_abs_tolerance_m"])["passed"],
                        checked(p_r, np.eye(3), limits["rotation_element_abs_tolerance"])["passed"],
                        checked(m_r, np.eye(3), limits["rotation_element_abs_tolerance"])["passed"]))}
                body_output["comparison"] = checks
                if any(not check["passed"] for check in checks.values()):
                    failures.append(f"{name}/{case['name']}/{body}: {checks}")
                case_output["bodies"].append(body_output)
                checked_transforms += 1
            record["cases"].append(case_output)
        output["models"].append(record)
        print(f"{name}: {len(record['cases'])} cases x {len(ordered)} body frames", flush=True)
    output["checked_transforms"] = checked_transforms
    output["failures"] = failures
    output["runtime_status"] = "passed" if not failures else "failed"
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"checked {checked_transforms} transforms; failures {len(failures)}", flush=True)
    if failures:
        print("\n".join(failures), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", required=True)
    raise SystemExit(run(parser.parse_args()))
