"""Fresh pinned Pinocchio/MuJoCo CPU reproduction through generated C++ adapter."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import mujoco
import numpy as np
import pinocchio as pin


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose(transform):
    q = pin.Quaternion(transform.rotation)
    return list(transform.translation) + [q.x, q.y, q.z, q.w]


def transform(values):
    q = values[3:]
    return pin.SE3(pin.Quaternion(q[3], q[0], q[1], q[2]).matrix(), np.asarray(values[:3]))


def error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    same, opposite = np.linalg.norm(a[3:]-b[3:]), np.linalg.norm(a[3:]+b[3:])
    return float(np.linalg.norm(a[:3]-b[:3])), float(4*math.atan2(min(same, opposite), max(same, opposite)))


def frame_id(model, name):
    ids = [i for i, f in enumerate(model.frames) if f.name == name]
    if len(ids) != 1:
        raise ValueError(f"nonunique reference frame {name}")
    return ids[0]


def run(repo, roots):
    if (pin.__version__, mujoco.__version__, np.__version__) != ("4.1.0", "3.12.0", "2.5.3"):
        raise ValueError("requires pinned Pinocchio4.1.0 MuJoCo3.12.0 NumPy2.5.3")
    started = time.monotonic()
    golden_path = repo/"tests/fixtures/foreign-components/pinocchio_goldens.json"
    records = json.loads(golden_path.read_bytes())["arms"]
    source = repo/"tests/fixtures/foreign-components/tool_projection_probe.cpp"
    rows, bindings = [], {}
    for arm in records:
        name = arm["arm"]
        workspace = roots[name]
        header = workspace/"src/invar_fk_node/include/invar_fk_node/tool_projection.hpp"
        scene = workspace/"src/invar_cell_bringup/config/mjcf/scene.xml"
        projection_path = workspace/"src/invar_fk_node/tool-projection.json"
        projection = json.loads(projection_path.read_bytes())
        model_path = repo/"examples/cells/reach_place/assets/menagerie"/arm["model_file"]
        if sha(model_path) != projection["reference_model_sha256"]:
            raise ValueError("unbound reference model")
        offset = transform(projection["reference_from_cell"]["translation_m"] + projection["reference_from_cell"]["quaternion_xyzw"])
        base_projection = projection.get("source_base_from_reference_base", {"translation_m": [0.,0.,0.], "quaternion_xyzw": [0.,0.,0.,1.]})
        source_base = transform(base_projection["translation_m"] + base_projection["quaternion_xyzw"])
        model = pin.buildModelFromMJCF(str(model_path))
        data = model.createData()
        base, tool = frame_id(model, arm["base_frame"]), frame_id(model, arm["tool_frame"])
        mjmodel = mujoco.MjModel.from_xml_path(str(scene))
        mjbase = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_BODY, arm["base_frame"])
        site = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_SITE, "invar_tool")
        assert mjbase >= 0 and site >= 0
        bindings[name] = {"generated_header_sha256": sha(header), "generated_scene_sha256": sha(scene),
                          "projection_sha256": sha(projection_path), "model_sha256": sha(model_path)}
        with tempfile.TemporaryDirectory(prefix="invar-tool-projection-cpu-") as temporary:
            executable = Path(temporary)/"probe"
            subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                            "-I", str(header.parent), str(source), "-o", str(executable)], check=True, timeout=20)
            inputs, expected = [], []
            for index, vector in enumerate(arm["vectors"]):
                q = pin.neutral(model)
                mjdata = mujoco.MjData(mjmodel)
                for joint_name, value in zip(arm["joint_names"], vector["q_rad"], strict=True):
                    joint = model.getJointId(joint_name)
                    assert joint < model.njoints and model.nqs[joint] == 1
                    q[model.idx_qs[joint]] = value
                    mjjoint = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    assert mjjoint >= 0
                    mjdata.qpos[mjmodel.jnt_qposadr[mjjoint]] = value
                pin.forwardKinematics(model, data, q)
                pin.updateFramePlacements(model, data)
                native = data.oMf[base].inverse()*data.oMf[tool]
                cell = source_base*native*offset
                cube = pin.SE3(np.eye(3), np.array([.3, 0, .025]))
                cell_from_cube = cell.inverse()*cube
                reference_from_cube = native.inverse()*source_base.inverse()*cube
                mujoco.mj_forward(mjmodel, mjdata)
                rotation = mjdata.xmat[mjbase].reshape(3, 3)
                actual_cell = pin.SE3(rotation.T@mjdata.site_xmat[site].reshape(3, 3),
                                     rotation.T@(mjdata.site_xpos[site]-mjdata.xpos[mjbase]))
                actual_cell = source_base*actual_cell
                inputs.append(" ".join(format(float(x), ".17g") for x in pose(native)+pose(cell)+pose(cell_from_cube)+pose(cube)))
                expected.append((index, native, actual_cell, reference_from_cube, cell_from_cube, cube))
            output = subprocess.run([str(executable)], input="\n".join(inputs)+"\n", text=True,
                                    capture_output=True, check=True, timeout=5)
            lines = output.stdout.splitlines()
            assert len(lines) == len(expected) == 4
            for line, (index, native, actual_cell, reference_from_cube, cell_from_cube, cube) in zip(lines, expected, strict=True):
                values = [float(x) for x in line.split()]
                assert len(values) == 28 and all(math.isfinite(x) for x in values)
                fk_error = error(values[:7], pose(actual_cell))
                goal_error = error(values[7:14], pose(native))
                attachment_error = error(values[14:21], pose(reference_from_cube))
                cube_error = error(values[21:], pose(source_base.inverse()*cube))
                attachment_base_error = error(pose(transform(values[:7])*cell_from_cube), pose(cube))
                original = records[[a["arm"] for a in records].index(name)]["vectors"][index]
                legacy_pose = original["translation_m"] + original["quaternion_wxyz"][1:] + original["quaternion_wxyz"][:1]
                legacy_error = error(pose(native), legacy_pose)
                wrong = pose(native*offset.inverse())
                wrong_error = error(wrong, pose(actual_cell))
                # Original gates remain; the stronger 1e-12 checks are diagnostic
                # reproduction checks, never a relaxed acceptance threshold.
                assert fk_error[0] <= 1e-6 and fk_error[0] <= .002 and fk_error[1] <= math.pi/180
                assert max(fk_error+goal_error+attachment_error+cube_error+attachment_base_error+legacy_error) < 1e-12
                assert wrong_error[0] > .002 or wrong_error[1] > math.pi/180
                rows.append({"arm": name, "source_vector": index, "generated_cpp_cell_pose": values[:7],
                             "fresh_mujoco_cell_pose": pose(actual_cell), "fresh_pinocchio_reference_pose": pose(native),
                             "fk_error_m_rad": fk_error, "inverse_goal_error_m_rad": goal_error,
                             "cube_reference_error_m_rad": cube_error, "attachment_error_m_rad": attachment_error, "attachment_base_error_m_rad": attachment_base_error,
                             "fresh_reference_vs_legacy_error_m_rad": legacy_error, "wrong_direction_error_m_rad": wrong_error})
    return {"schema": "invar-tool-projection-cpu-reproduction-v2", "status": "pass_cpu_only",
            "scope": "generated private C++ adapter plus fresh pinned CPU FK; no live ROS/cuRobo/task qualification",
            "pinocchio": pin.__version__, "mujoco": mujoco.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform(), "bindings": bindings,
            "source_goldens_sha256": sha(golden_path), "probe_sha256": sha(source),
            "runner_sha256": sha(Path(__file__)), "rows": rows, "gpu_jobs": 0, "wall_seconds": time.monotonic()-started}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--panda", type=Path, required=True)
    parser.add_argument("--ur5e", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.repo.resolve(), {"panda": args.panda.resolve(), "ur5e": args.ur5e.resolve()})
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2)+"\n")
    for arm in ["panda", "ur5e"]:
        rows = [r for r in result["rows"] if r["arm"] == arm]
        print(json.dumps({"arm": arm, "states": len(rows), "fk_translation_max_m": max(r["fk_error_m_rad"][0] for r in rows),
                          "fk_orientation_max_rad": max(r["fk_error_m_rad"][1] for r in rows)}))
