"""Bounded CPU source-goal diagnostics; no planner, geometry fit or qualification."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pinocchio as pin
import scipy
from scipy.optimize import least_squares


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose(transform):
    q = pin.Quaternion(transform.rotation)
    return list(map(float, transform.translation)) + [float(q.x), float(q.y), float(q.z), float(q.w)]


def transform(position, rotation):
    return pin.SE3(np.asarray(rotation).reshape(3, 3).copy(), np.asarray(position).copy())


def error(a, b):
    return [float(np.linalg.norm(a.translation-b.translation)),
            float(np.linalg.norm(pin.log3(a.rotation.T @ b.rotation)))]


def frame(model, name):
    indices = [i for i, f in enumerate(model.frames) if f.name == name]
    if len(indices) != 1:
        raise ValueError("nonunique reference frame")
    return indices[0]


def run(repo, roots):
    if (pin.__version__, mujoco.__version__, np.__version__, scipy.__version__) != ("4.1.0", "3.12.0", "2.5.3", "1.17.0"):
        raise ValueError("requires exact recorded CPU numerical versions")
    start = time.monotonic()
    goldens = repo/"tests/fixtures/foreign-components/pinocchio_goldens.json"
    rows = []
    for arm in json.loads(goldens.read_bytes())["arms"]:
        name = arm["arm"]
        workspace = roots[name]
        scene = workspace/"src/invar_cell_bringup/config/mjcf/scene.xml"
        projection_path = workspace/"src/invar_fk_node/tool-projection.json"
        descriptor_path = workspace/"src/invar_cell_bringup/config/cell-snapshot-v1.json"
        projection = json.loads(projection_path.read_bytes())
        descriptor = json.loads(descriptor_path.read_bytes())
        reference = repo/"examples/cells/reach_place/assets/menagerie"/arm["model_file"]
        if sha(reference) != projection["reference_model_sha256"]:
            raise ValueError("unbound reference model")
        welds = [w.attrib for w in ET.parse(scene).iter("weld") if w.get("name") == "cube_attach"]
        if welds != [{"active": "false", "name": "cube_attach", "site1": "invar_tool", "site2": "cube_grasp"}]:
            raise ValueError("unsupported source weld")
        model = pin.buildModelFromMJCF(str(reference))
        data = model.createData()
        base, tool = frame(model, arm["base_frame"]), frame(model, arm["tool_frame"])
        offset = projection["reference_from_cell"]
        x, y, z, w = offset["quaternion_xyzw"]
        offset = pin.SE3(pin.Quaternion(w, x, y, z).matrix(), np.array(offset["translation_m"]))
        mj = mujoco.MjModel.from_xml_path(str(scene))
        state = mujoco.MjData(mj)
        key = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_KEY, "home")
        site = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "invar_tool")
        root = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, descriptor["robot_root"])
        table = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "table")
        cube = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "cube")
        target = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "place_target")
        anchor = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, projection["anchor_body"])
        if min(key, site, root, table, cube, target, anchor) < 0:
            raise ValueError("missing source identity")
        mujoco.mj_resetDataKeyframe(mj, state, key)
        mujoco.mj_forward(mj, state)
        full_q = pin.neutral(model)
        q_indices, mj_q_indices, home, bounds = [], [], [], []
        for joint in descriptor["robot_joints"]:
            pid = model.getJointId(joint["name"])
            mid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, joint["name"])
            if pid >= model.njoints or model.nqs[pid] != 1 or mid < 0:
                raise ValueError("joint identity mismatch")
            full_q[model.idx_qs[pid]] = state.qpos[mj.jnt_qposadr[mid]]
            if joint["name"] in descriptor["joint_names"]:
                q_indices.append(int(model.idx_qs[pid]))
                mj_q_indices.append(int(mj.jnt_qposadr[mid]))
                home.append(float(state.qpos[mj.jnt_qposadr[mid]]))
                bounds.append(joint["position_limits_si"])
        if descriptor["joint_names"] != arm["joint_names"]:
            raise ValueError("arm order mismatch")
        home = np.asarray(home)
        lo, hi = np.asarray(bounds).T

        def fk(q):
            full_q[q_indices] = q
            pin.forwardKinematics(model, data, full_q)
            pin.updateFramePlacements(model, data)
            return (data.oMf[base].inverse()*data.oMf[tool]*offset).copy()

        home_local = fk(home)
        base_world = transform(state.xpos[root], state.xmat[root])
        home_world = transform(state.site_xpos[site], state.site_xmat[site])
        key_cube = transform(state.xpos[cube], state.xmat[cube])
        w, x, y, z = mj.body_quat[cube]
        initial_cube = pin.SE3(pin.Quaternion(w, x, y, z).matrix(), mj.body_pos[cube].copy())
        place = transform(state.xpos[target], state.xmat[target])
        pregrasp = pin.SE3(initial_cube.rotation.copy(), initial_cube.translation+np.array([0., 0., .1]))
        robots = set()
        for body in range(1, mj.nbody):
            parent = body
            while parent:
                if parent == root:
                    robots.add(body)
                    break
                parent = int(mj.body_parentid[parent])

        def contacts(q):
            s = mujoco.MjData(mj)
            mujoco.mj_resetDataKeyframe(mj, s, key)
            s.qpos[mj_q_indices] = q
            mujoco.mj_forward(mj, s)
            forbidden = []
            for c in s.contact:
                a, b = int(c.geom1), int(c.geom2)
                ba, bb = int(mj.geom_bodyid[a]), int(mj.geom_bodyid[b])
                if (ba in robots and (bb in robots or bb == table)) or (bb in robots and ba == table):
                    forbidden.append({"geom_ids": [a, b], "body_names": [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, ba), mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, bb)], "distance_m": float(c.dist)})
            actual = transform(s.site_xpos[site], s.site_xmat[site])
            cross = error(base_world*fk(q), actual)
            if cross[0] > 1e-6 or cross[1] > 1e-12:
                raise ValueError("fresh FK/physics source-frame disagreement")
            return forbidden, cross

        # These are diagnostic placements of the unchanged terminal body's
        # existing compiled geoms. No claim that a joint state reaches them.
        static = []
        for goal_name, goal in [("grasp", initial_cube), ("place", place)]:
            s = mujoco.MjData(mj)
            mujoco.mj_resetDataKeyframe(mj, s, key)
            mujoco.mj_forward(mj, s)
            move = goal*home_world.inverse()
            distances = []
            for g in range(mj.ngeom):
                if int(mj.geom_bodyid[g]) != anchor or not (mj.geom_contype[g] or mj.geom_conaffinity[g]):
                    continue
                transformed = move*transform(s.geom_xpos[g], s.geom_xmat[g])
                s.geom_xpos[g] = transformed.translation
                s.geom_xmat[g] = transformed.rotation.reshape(-1)
                for t in range(mj.ngeom):
                    if int(mj.geom_bodyid[t]) == table and (mj.geom_contype[t] or mj.geom_conaffinity[t]):
                        d = mujoco.mj_geomDistance(mj, s, g, t, 2., np.zeros(6))
                        distances.append({"geom_ids": [g, t], "distance_m": float(d)})
            static.append({"goal": goal_name, "goal_world": pose(goal), "terminal_body": projection["anchor_body"], "source_geom_table_distances": distances, "qualification": "rigid_terminal_placement_only_not_a_reachable_joint_state"})

        seeds = [home, np.clip(np.asarray(arm["vectors"][0]["q_rad"]), lo+1e-10, hi-1e-10), (lo+hi)/2]
        solutions = []
        for goal_name, goal in [("pregrasp", pregrasp), ("grasp", initial_cube), ("place", place), ("home", home_world)]:
            local_goal = base_world.inverse()*goal
            for seed_index, seed in enumerate(seeds):
                if time.monotonic()-start > 45:
                    raise TimeoutError("bounded CPU feasibility budget exhausted")
                def residual(q):
                    r = pin.log6(local_goal.inverse()*fk(q)).vector.copy()
                    r[3:] *= .1
                    return r
                result = least_squares(residual, np.clip(seed, lo+1e-10, hi-1e-10), bounds=(lo, hi), max_nfev=150, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                actual = base_world*fk(result.x)
                e = error(actual, goal)
                admitted = e[0] <= .002 and e[1] <= math.pi/180
                forbidden, cross = contacts(result.x)
                solutions.append({"goal": goal_name, "seed": seed_index, "goal_world": pose(goal), "q_rad": result.x.tolist(), "nfev": result.nfev, "solver_status": result.status, "endpoint_error_m_rad": e, "within_existing_endpoint_gate": admitted, "within_source_joint_limits": bool(np.all(result.x >= lo) and np.all(result.x <= hi)), "home_max_joint_difference_rad": float(np.max(np.abs(result.x-home))), "forbidden_contacts": forbidden, "fresh_pinocchio_mujoco_error_m_rad": cross})
        home_contacts, home_cross = contacts(home)
        rows.append({"arm": name, "scene_sha256": sha(scene), "projection_sha256": sha(projection_path), "descriptor_sha256": sha(descriptor_path), "reference_model_sha256": sha(reference), "source_weld": welds[0], "physical_world_from_reference_base": pose(base_world), "configured_home_q_rad": home.tolist(), "home_world_from_tool": pose(home_world), "home_reference_base_from_tool": pose(home_local), "current_mount_only_runtime_same_state_error_m_rad": error(home_local, home_world), "declared_cube_pose_world": pose(initial_cube), "cube_pose_after_home_key_reset": pose(key_cube), "cube_home_reset_error_m_rad": error(initial_cube, key_cube), "home_forbidden_contacts": home_contacts, "home_fresh_fk_physics_error_m_rad": home_cross, "static_terminal_goal_checks": static, "bounded_ik_observations": solutions})
    return {"schema": "invar-source-task-feasibility-v1", "scope": "CPU diagnostics only; no cuRobo, trajectories, ROS or robotics qualification", "versions": {"pinocchio": pin.__version__, "mujoco": mujoco.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version()}, "bounds": {"internal_seconds": 45, "seeds_per_goal": 3, "goals_per_arm": 4, "max_nfev_per_seed": 150, "pregrasp_vertical_offset_m": .1}, "runner_sha256": sha(Path(__file__)), "goldens_sha256": sha(goldens), "rows": rows, "wall_seconds": time.monotonic()-start, "gpu_jobs": 0, "new_geometry_candidates": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ["repo", "panda", "ur5e", "output"]:
        parser.add_argument("--"+arg, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.repo.resolve(), {"panda": args.panda.resolve(), "ur5e": args.ur5e.resolve()})
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2)+"\n")
    for row in result["rows"]:
        print(json.dumps({"arm": row["arm"], "mount_only_error_m_rad": row["current_mount_only_runtime_same_state_error_m_rad"], "solves": len(row["bounded_ik_observations"]), "endpoint_gate_met": sum(s["within_existing_endpoint_gate"] for s in row["bounded_ik_observations"]), "static_min_distances_m": [min(d["distance_m"] for d in s["source_geom_table_distances"]) for s in row["static_terminal_goal_checks"]]}))
