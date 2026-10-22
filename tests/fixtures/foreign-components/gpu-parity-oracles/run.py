#!/usr/bin/env python3
"""Independent Pinocchio/MuJoCo CPU oracles for the frozen robotics contract."""

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_id(model, name):
    for i, frame in enumerate(model.frames):
        if frame.name == name:
            return i
    raise ValueError(f"Pinocchio frame missing: {name}")


def pin_q(model, spec, arm_q):
    q = pin.neutral(model)
    for name, value in zip(spec["active_joints"], arm_q, strict=True):
        jid = model.getJointId(name)
        if jid >= model.njoints or model.nqs[jid] != 1:
            raise ValueError(f"Pinocchio active joint missing/non-scalar: {name}")
        q[model.idx_qs[jid]] = value
    for name, value in spec["fixed_inactive_joints"].items():
        jid = model.getJointId(name)
        if jid >= model.njoints or model.nqs[jid] != 1:
            raise ValueError(f"Pinocchio inactive joint missing/non-scalar: {name}")
        q[model.idx_qs[jid]] = float(value)
    return q


def pin_jac(model, data, spec, q):
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    tool = frame_id(model, spec["tool_name"])
    base = frame_id(model, spec["base_body"])
    world_jac = np.asarray(pin.getFrameJacobian(model, data, tool, pin.LOCAL_WORLD_ALIGNED))
    if world_jac.shape != (6, model.nv):
        raise ValueError(f"unexpected Pinocchio Jacobian shape {world_jac.shape}")
    base_from_world = np.asarray(data.oMf[base].rotation).T
    dofs = [model.idx_vs[model.getJointId(name)] for name in spec["active_joints"]]
    return np.vstack((base_from_world @ world_jac[:3, dofs],
                      base_from_world @ world_jac[3:, dofs]))


def mj_data(model, spec, arm_q):
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key < 0:
        raise ValueError("MuJoCo home keyframe missing")
    mujoco.mj_resetDataKeyframe(model, data, key)
    for name, value in zip(spec["active_joints"], arm_q, strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"MuJoCo active hinge missing: {name}")
        data.qpos[model.jnt_qposadr[jid]] = value
        data.qvel[model.jnt_dofadr[jid]] = 0
    for name, value in spec["fixed_inactive_joints"].items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_SLIDE:
            raise ValueError(f"MuJoCo inactive slide missing: {name}")
        data.qpos[model.jnt_qposadr[jid]] = float(value)
        data.qvel[model.jnt_dofadr[jid]] = 0
    mujoco.mj_forward(model, data)
    return data


def mj_tool(model, data, spec):
    if spec["tool_kind"] == "body_origin":
        tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec["tool_name"])
        if tool < 0:
            raise ValueError("MuJoCo tool body missing")
        return np.asarray(data.xpos[tool]).copy(), np.asarray(data.xmat[tool]).reshape(3, 3).copy()
    if spec["tool_kind"] == "site_origin":
        tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec["tool_name"])
        if tool < 0:
            raise ValueError("MuJoCo tool site missing")
        return np.asarray(data.site_xpos[tool]).copy(), np.asarray(data.site_xmat[tool]).reshape(3, 3).copy()
    raise ValueError("unknown tool kind")


def mj_jac(model, data, spec):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    if spec["tool_kind"] == "body_origin":
        tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec["tool_name"])
        if tool < 0:
            raise ValueError("MuJoCo tool body missing")
        mujoco.mj_jacBody(model, data, jacp, jacr, tool)
    else:
        tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec["tool_name"])
        if tool < 0:
            raise ValueError("MuJoCo tool site missing")
        mujoco.mj_jacSite(model, data, jacp, jacr, tool)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec["base_body"])
    if base < 0:
        raise ValueError("MuJoCo base body missing")
    base_from_world = np.asarray(data.xmat[base]).reshape(3, 3).T
    dofs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in spec["active_joints"]]
    return np.vstack((base_from_world @ jacp[:, dofs],
                      base_from_world @ jacr[:, dofs]))


def mj_fd_jac(model, spec, q, step):
    center = mj_data(model, spec, q)
    _, rotation = mj_tool(model, center, spec)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec["base_body"])
    base_from_world = np.asarray(center.xmat[base]).reshape(3, 3).T
    result = np.zeros((6, len(q)))
    for i in range(len(q)):
        lower, upper = q.copy(), q.copy()
        lower[i] -= step
        upper[i] += step
        p_minus, r_minus = mj_tool(model, mj_data(model, spec, lower), spec)
        p_plus, r_plus = mj_tool(model, mj_data(model, spec, upper), spec)
        rdot = (r_plus - r_minus) / (2 * step)
        skew = rdot @ rotation.T
        angular_world = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        result[:3, i] = base_from_world @ ((p_plus - p_minus) / (2 * step))
        result[3:, i] = base_from_world @ angular_world
    return result


def full_mass(model, data, spec):
    dense = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, dense)
    dofs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in spec["active_joints"]]
    return dense[np.ix_(dofs, dofs)], np.asarray(model.dof_armature)[dofs]


def pin_mass(model, data, spec, q):
    imported_armature = np.asarray(model.armature).copy()
    model.armature[:] = 0
    try:
        upper = np.asarray(pin.crba(model, data, q)).copy()
    finally:
        model.armature[:] = imported_armature
    body_mass = np.triu(upper) + np.triu(upper, 1).T
    dofs = [model.idx_vs[model.getJointId(name)] for name in spec["active_joints"]]
    return body_mass[np.ix_(dofs, dofs)], imported_armature[dofs]


def values(matrix):
    return [[repr(float(x)) for x in row] for row in np.asarray(matrix)]


def compare(a, b, absolute, relative):
    delta = np.abs(a - b)
    allowed = absolute + relative * np.maximum(np.abs(a), np.abs(b))
    return {"passed": bool(np.all(delta <= allowed)),
            "max_abs": repr(float(np.max(delta))),
            "max_excess": repr(float(np.max(delta - allowed)))}


def cases(spec, golden, jacobian, inputs_spec):
    limits = [tuple(map(float, entry.split())) for entry in spec["active_limits_rad_text"]]
    result = [("source_home", list(map(float, spec["home_active_rad_text"])))]
    result += [(f"prior_fk_{i}", list(vector["q_rad"])) for i, vector in enumerate(golden["vectors"])]
    fraction = inputs_spec["near_limit_fraction_of_span"]
    for invert, name in ((False, "near_limit_low_high"), (True, "near_limit_high_low")):
        result.append((name, [hi - fraction * (hi - lo) if (i % 2 == 1) != invert
                              else lo + fraction * (hi - lo)
                              for i, (lo, hi) in enumerate(limits)]))
    rng = random.Random(spec["candidate_seed"])
    candidates = [[rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))
                   for lo, hi in limits]
                  for _ in range(inputs_spec["least_singular_selection"]["candidate_count"])]
    scores = [float(np.linalg.svd(jacobian(q), compute_uv=False)[-1]) for q in candidates]
    chosen = min(range(len(scores)), key=lambda i: (scores[i], i))
    result.append(("least_singular_seeded", candidates[chosen]))
    if [name for name, _ in result] != inputs_spec["ordered_cases"] or any(len(q) != len(limits) or
                               any(not lo <= v <= hi for v, (lo, hi) in zip(q, limits, strict=True))
                               for _, q in result):
        raise ValueError("invalid input cases")
    return result, chosen


def run(args):
    contract_path, goldens_path = Path(args.contract), Path(args.goldens)
    contract = json.loads(contract_path.read_text())
    goldens = json.loads(goldens_path.read_text())
    if contract["schema_version"] != 1 or pin.__version__ != contract["libraries"]["pinocchio"] or \
            mujoco.__version__ != contract["libraries"]["mujoco"] or \
            sha(goldens_path) != contract["source"]["prior_fk_goldens_sha256"] or \
            sha(Path(args.source_facts)) != contract["source"]["source_facts_sha256"] or \
            goldens["menagerie_revision"] != contract["source"]["commit"]:
        raise ValueError("contract/dependency version mismatch")
    output = {"contract_sha256": sha(contract_path), "source_commit": contract["source"]["commit"],
              "input_generator": contract["inputs"], "pinocchio_version": pin.__version__,
              "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
              "environment_image_id": args.image_id, "platform": platform.platform(),
              "python_version": sys.version.split()[0], "models": [], "runtime_status": "pending"}
    failures = []
    for spec in contract["models"]:
        source = Path(args.models_root) / spec["file"]
        if sha(source) != spec["sha256"] or sha(source.parent / "LICENSE") != spec["license_sha256"]:
            raise ValueError("pinned source mismatch")
        golden = next(g for g in goldens["arms"] if g["arm"] == spec["name"])
        # A derived model (contract v2: principal Panda inertias) keeps the FK
        # goldens of the upstream model it was derived from; FK is unchanged.
        if golden["model_sha256"] != spec.get("derived_from_sha256", spec["sha256"]) or \
                golden["joint_names"] != spec["active_joints"]:
            raise ValueError("FK golden identity mismatch")
        p_model = pin.buildModelFromMJCF(str(source))
        m_model = mujoco.MjModel.from_xml_path(str(source))
        p_data = p_model.createData()
        def pj(q):
            return pin_jac(p_model, p_data, spec, pin_q(p_model, spec, q))
        inputs, selected = cases(spec, golden, pj, contract["inputs"])
        record = {"name": spec["name"], "source_model_sha256": spec["sha256"],
                  "active_joints": spec["active_joints"], "fixed_inactive_joints": spec["fixed_inactive_joints"],
                  "least_singular_candidate_index": selected, "cases": []}
        n = len(spec["active_joints"])
        source_armature = np.array([float(x) for x in spec["source_armature_diag_text"]])
        for name, q in inputs:
            p_q = pin_q(p_model, spec, q)
            p_j = pin_jac(p_model, p_data, spec, p_q)
            m_data = mj_data(m_model, spec, q)
            m_j = mj_jac(m_model, m_data, spec)
            f_j = mj_fd_jac(m_model, spec, q, contract["tool_jacobian"]["finite_difference_step_rad"])
            p_body, p_imported_armature = pin_mass(p_model, p_data, spec, p_q)
            m_raw, m_imported_armature = full_mass(m_model, m_data, spec)
            p_mass = p_body + np.diag(source_armature)
            m_mass = m_raw - np.diag(m_imported_armature) + np.diag(source_armature)
            jt = contract["tool_jacobian"]
            mt = contract["active_mass_matrix"]
            comparisons = {
                "analytic_jacobian": compare(p_j, m_j, jt["analytic_element_abs_tolerance"],
                                             jt["analytic_element_relative_tolerance"]),
                "pinocchio_vs_finite_difference": compare(p_j, f_j, jt["finite_difference_element_abs_tolerance"], 0),
                "mujoco_vs_finite_difference": compare(m_j, f_j, jt["finite_difference_element_abs_tolerance"], 0),
                "active_mass": compare(p_mass, m_mass, mt["analytic_element_abs_tolerance"],
                                       mt["analytic_element_relative_tolerance"]),
            }
            checks = [np.max(np.abs(x - x.T)) <= mt["symmetry_abs_tolerance"] and
                      np.min(np.linalg.eigvalsh(x)) > 0 for x in (p_mass, m_mass)]
            comparisons["mass_symmetry_positive_definite"] = {"passed": bool(all(checks))}
            warning_counts = [int(w.number) for w in m_data.warning]
            comparisons["mujoco_no_warnings"] = {"passed": not any(warning_counts)}
            if any(not item["passed"] for item in comparisons.values()):
                failures.append(f"{spec['name']}/{name}: {comparisons}")
            case = {"name": name, "q_rad": [repr(float(x)) for x in q],
                    "jacobian_pinocchio": values(p_j), "jacobian_mujoco": values(m_j),
                    "jacobian_finite_difference": values(f_j),
                    "mass_pinocchio_without_armature": values(p_body),
                    "mass_mujoco_raw": values(m_raw),
                    "mass_pinocchio": values(p_mass), "mass_mujoco": values(m_mass),
                    "pinocchio_imported_armature": [repr(float(x)) for x in p_imported_armature],
                    "mujoco_imported_armature": [repr(float(x)) for x in m_imported_armature],
                    "mujoco_warning_counts": warning_counts,
                    "jacobian_min_singular_value": repr(float(np.linalg.svd(p_j, compute_uv=False)[-1])),
                    "comparison": comparisons}
            if any(np.asarray(x).shape != shape or not np.all(np.isfinite(x)) for x, shape in
                   ((p_j, (6, n)), (m_j, (6, n)), (f_j, (6, n)),
                    (p_mass, (n, n)), (m_mass, (n, n)))):
                raise ValueError("nonfinite or wrong shape oracle output")
            record["cases"].append(case)
        output["models"].append(record)
        print(f"{spec['name']}: {len(inputs)} cases, least-singular candidate {selected}", flush=True)
    output["runtime_status"] = "passed" if not failures else "failed"
    output["failures"] = failures
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if failures:
        print("\n".join(failures), flush=True)
        return 1
    print("independent Jacobian, finite-difference and active mass comparisons passed", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--goldens", required=True)
    parser.add_argument("--source-facts", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", required=True)
    raise SystemExit(run(parser.parse_args()))
