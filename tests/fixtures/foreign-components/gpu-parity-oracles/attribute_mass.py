#!/usr/bin/env python3
"""Attribute the frozen Panda mass gap by summing per-body kinetic energy."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from run import mj_data


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(values):
    xx, yy, zz, xy, xz, yz = map(float, values.split())
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def source_bodies(path, model):
    bodies = []
    for body in ET.parse(path).getroot().iter("body"):
        inertial = body.find("inertial")
        if inertial is None:
            raise ValueError(f"body without explicit source inertia: {body.get('name')}")
        if "quat" in inertial:
            raise ValueError("source inertial quaternion needs an explicit frame transform")
        name = body.attrib["name"]
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"compiled body missing: {name}")
        if "fullinertia" in inertial.attrib:
            source_i = matrix(inertial.attrib["fullinertia"])
            kind = "fullinertia"
        else:
            source_i = np.diag(list(map(float, inertial.attrib["diaginertia"].split())))
            kind = "diaginertia"
        inertial_rot = np.empty(9)
        mujoco.mju_quat2Mat(inertial_rot, model.body_iquat[body_id])
        inertial_rot = inertial_rot.reshape(3, 3)
        compiled_i = inertial_rot @ np.diag(model.body_inertia[body_id]) @ inertial_rot.T
        source_mass = float(inertial.attrib["mass"])
        source_com = np.fromstring(inertial.attrib["pos"], sep=" ")
        if source_com.shape != (3,):
            raise ValueError(f"bad source COM: {name}")
        bodies.append((name, kind, body_id, source_mass, source_com, source_i, compiled_i,
                       inertial_rot))
    if len(bodies) != model.nbody - 1:
        raise ValueError("source/compiled body count mismatch")
    return bodies


def jacobian(model, data, point, body_id, dofs):
    linear = np.zeros((3, model.nv))
    angular = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, linear, angular, point, body_id)
    return linear[:, dofs], angular[:, dofs]


def numbers(matrix):
    return [[repr(float(x)) for x in row] for row in matrix]


def run(args):
    contract_path, reference_path = Path(args.contract), Path(args.reference)
    contract, reference = json.loads(contract_path.read_text()), json.loads(reference_path.read_text())
    spec = next(s for s in contract["models"] if s["name"] == "panda")
    prior = next(m for m in reference["models"] if m["name"] == "panda")
    source = Path(args.models_root) / spec["file"]
    if sha(source) != spec["sha256"] or sha(source.parent / "LICENSE") != spec["license_sha256"] or \
            reference["contract_sha256"] != sha(contract_path) or \
            reference["runtime_status"] != "failed":
        raise ValueError("frozen source/reference mismatch")
    model = mujoco.MjModel.from_xml_path(str(source))
    bodies = source_bodies(source, model)
    dofs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in spec["active_joints"]]
    n = len(dofs)
    armature = np.diag(list(map(float, spec["source_armature_diag_text"])))
    output = {"source_sha256": sha(source), "contract_sha256": sha(contract_path),
              "reference_sha256": sha(reference_path), "method": "sum mass*Jv_com.T*Jv_com + Jw.T*I_com_world*Jw per source body; compare source and compiled inertias without fitting",
              "body_differences": [], "cases": []}
    for name, kind, body_id, mass, com, source_i, compiled_i, inertial_rot in bodies:
        rotated_source = inertial_rot.T @ source_i @ inertial_rot
        row, col = max(((0, 1), (0, 2), (1, 2)),
                       key=lambda ij: abs(rotated_source[ij]))
        offdiag = abs(rotated_source[row, col])
        tau = (rotated_source[col, col] - rotated_source[row, row]) / (2 * rotated_source[row, col]) if offdiag else 0
        turn = np.copysign(1.0, tau) / (abs(tau) + np.sqrt(1 + tau * tau)) if offdiag else 0
        cosine_gap = 1 - 1 / np.sqrt(1 + turn * turn)
        output["body_differences"].append({
            "name": name, "source_inertia_kind": kind,
            "source_mass": repr(mass), "compiled_mass": repr(float(model.body_mass[body_id])),
            "source_com": [repr(float(x)) for x in com],
            "compiled_com": [repr(float(x)) for x in model.body_ipos[body_id]],
            "compiled_minus_source_inertia": numbers(compiled_i - source_i),
            "max_inertia_delta": repr(float(np.max(np.abs(compiled_i - source_i)))),
            "jacobi_max_offdiag_in_compiled_axes": repr(float(offdiag)),
            "jacobi_cosine_gap": repr(float(cosine_gap)),
            "jacobi_stop_condition": ("offdiag_below_1e-12" if offdiag < 1e-12 else
                                      "cosine_above_1_minus_1e-12" if cosine_gap < 1e-12 else
                                      "neither")})
    for case in prior["cases"]:
        q = list(map(float, case["q_rad"]))
        data = mj_data(model, spec, q)
        source_m, compiled_m = armature.copy(), armature.copy()
        inertia_delta_m = np.zeros((n, n))
        body_inertia_delta_max = {}
        for _, _, body_id, source_mass, source_com, source_i, compiled_i, _ in bodies:
            rot = np.asarray(data.xmat[body_id]).reshape(3, 3)
            origin = np.asarray(data.xpos[body_id])
            source_jv, source_jw = jacobian(model, data, origin + rot @ source_com, body_id, dofs)
            compiled_jv, compiled_jw = jacobian(model, data, origin + rot @ model.body_ipos[body_id], body_id, dofs)
            source_m += source_mass * source_jv.T @ source_jv + source_jw.T @ (rot @ source_i @ rot.T) @ source_jw
            compiled_m += model.body_mass[body_id] * compiled_jv.T @ compiled_jv + compiled_jw.T @ (rot @ compiled_i @ rot.T) @ compiled_jw
            contribution = source_jw.T @ (rot @ (compiled_i - source_i) @ rot.T) @ source_jw
            inertia_delta_m += contribution
            body_inertia_delta_max[model.body(body_id).name] = repr(float(np.max(np.abs(contribution))))
        pin_m = np.array(case["mass_pinocchio"], dtype=float)
        mj_m = np.array(case["mass_mujoco"], dtype=float)
        source_residual = source_m - pin_m
        compiled_residual = compiled_m - mj_m
        delta_residual = (compiled_m - source_m) - (mj_m - pin_m)
        inertia_delta_residual = inertia_delta_m - (mj_m - pin_m)
        output["cases"].append({
            "name": case["name"], "q_rad": case["q_rad"],
            "source_kinetic_mass": numbers(source_m),
            "compiled_kinetic_mass": numbers(compiled_m),
            "predicted_compiled_minus_source": numbers(compiled_m - source_m),
            "predicted_inertia_delta_only": numbers(inertia_delta_m),
            "observed_mujoco_minus_pinocchio": numbers(mj_m - pin_m),
            "body_inertia_delta_max": body_inertia_delta_max,
            "max_source_vs_pinocchio_residual": repr(float(np.max(np.abs(source_residual)))),
            "max_compiled_vs_mujoco_residual": repr(float(np.max(np.abs(compiled_residual)))),
            "max_delta_residual": repr(float(np.max(np.abs(delta_residual)))),
            "max_inertia_delta_residual": repr(float(np.max(np.abs(inertia_delta_residual))))})
    output["max_source_vs_pinocchio_residual"] = repr(max(float(c["max_source_vs_pinocchio_residual"]) for c in output["cases"]))
    output["max_compiled_vs_mujoco_residual"] = repr(max(float(c["max_compiled_vs_mujoco_residual"]) for c in output["cases"]))
    output["max_delta_residual"] = repr(max(float(c["max_delta_residual"]) for c in output["cases"]))
    output["max_inertia_delta_residual"] = repr(max(float(c["max_inertia_delta_residual"]) for c in output["cases"]))
    output["status"] = "attributed" if max(map(float, (output["max_source_vs_pinocchio_residual"],
                                                       output["max_compiled_vs_mujoco_residual"],
                                                       output["max_delta_residual"],
                                                       output["max_inertia_delta_residual"]))) < 1e-10 else "unresolved"
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("cases", len(output["cases"]), "status", output["status"],
          "source/pin", output["max_source_vs_pinocchio_residual"],
          "compiled/mujoco", output["max_compiled_vs_mujoco_residual"],
          "delta", output["max_delta_residual"],
          "inertia-only delta", output["max_inertia_delta_residual"])
    return 0 if output["status"] == "attributed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--output", required=True)
    raise SystemExit(run(parser.parse_args()))
