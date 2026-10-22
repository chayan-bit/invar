"""Home initialization and bounded physics checks, not reach-and-place acceptance."""

import json
import math
from pathlib import Path
import sys
import warnings

import mujoco


def tree_nodes(node):
    yield node
    for child in node["children"]:
        yield from tree_nodes(child)


def quat(value):
    values = [float(x) for x in value]
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values]


def forbidden_body_pairs(model, data, robot_bodies, table):
    # mj_step advances qpos after computing contacts. Refresh position-derived
    # fields for the state being checked, including the final integrated state.
    mujoco.mj_fwdPosition(model, data)
    pairs = set()
    for contact in data.contact:
        a, b = (int(model.geom_bodyid[geom]) for geom in (contact.geom1, contact.geom2))
        if (a in robot_bodies and b in robot_bodies) or (a == table and b in robot_bodies) or (b == table and a in robot_bodies):
            pairs.add(tuple(sorted((model.body(a).name, model.body(b).name))))
    return pairs


def run(scene_path, document_path, mode):
    assert mode in ("physics", "strict-hold"), "unknown smoke mode"
    document = json.loads(Path(document_path).read_text())
    selected = document["products"][0]["robot"]
    robot = next(r for r in document["robots"] if selected.endswith("::" + r["name"]))
    cell = document["cells"][0]
    cube = next(obj for obj in cell["objects"] if obj["name"] == "cube")
    home = [node["attrs"] for node in tree_nodes(robot["source_tree"])
            if node["tag"] == "key" and node["attrs"].get("name") == "home"]
    assert len(home) == 1, "one source home key required"

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        key = model.key("home").id
        # Generated ros2_control selects this key, including controls/velocities.
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        assert model.site("invar_tool").id >= 0
        assert model.joint("cube_free").type == mujoco.mjtJoint.mjJNT_FREE
        assert float(model.body("cube").mass[0]) == float(cube["dynamic"]["mass"]["value"])
        weld = model.equality("cube_attach").id
        assert not model.eq_active0[weld] and not data.eq_active[weld]

        original_qpos = [float(x) for x in home[0]["qpos"].split()]
        assert list(data.qpos[:len(original_qpos)]) == original_qpos, "robot home changed"
        original_ctrl = [float(x) for x in home[0].get("ctrl", "").split()]
        # A source-declared actuator hold replaces that vendored home control.
        for actuator in robot["actuators"]:
            if "hold" in actuator:
                original_ctrl[model.actuator(actuator["name"]).id] = float(actuator["hold"]["value"])
        assert list(data.ctrl[:len(original_ctrl)]) == original_ctrl, "home controls changed"
        cube_qpos = int(model.joint("cube_free").qposadr[0])
        cube_pose = [float(x["value"]) for x in cube["pose"]["xyz"]] + quat(cube["pose"]["quat"])
        assert list(data.qpos[cube_qpos:cube_qpos + 7]) == cube_pose, "cube home pose changed"
        grasp = model.site("cube_grasp")
        assert model.body(int(grasp.bodyid[0])).name == "cube", "grasp is not cube relative"
        assert list(grasp.pos) == [float(x["value"]) for x in cube["grasp"]["xyz"]]
        expected_quat = quat(cube["grasp"]["quat"])
        assert min(max(abs(a - b) for a, b in zip(grasp.quat, expected_quat)),
                   max(abs(a + b) for a, b in zip(grasp.quat, expected_quat))) <= 1e-12

        robot_bodies = {model.body(link["name"]).id for link in robot["links"]}
        table = model.body("table").id
        joints = []
        observed = {}
        for joint in robot["joints"]:
            if joint["kind"] == "fixed":
                continue
            position_unit, velocity_unit = ("m", "m/s") if joint["kind"] == "prismatic" else ("rad", "rad/s")
            assert joint["position"]["lower"]["unit"] == position_unit
            assert joint["position"]["upper"]["unit"] == position_unit
            assert joint["velocity"]["unit"] == velocity_unit
            mj_joint = model.joint(joint["name"])
            lower = float(joint["position"]["lower"]["value"])
            upper = float(joint["position"]["upper"]["value"])
            velocity = float(joint["velocity"]["value"])
            assert list(mj_joint.range) == [lower, upper], "model/source joint limits disagree"
            joints.append((joint["name"], int(mj_joint.qposadr[0]), int(mj_joint.dofadr[0]), lower, upper, velocity))
            observed[joint["name"]] = {"min_position": math.inf, "max_position": -math.inf,
                                       "max_abs_velocity": 0.0, "position_limits": [lower, upper],
                                       "velocity_limit": velocity, "position_unit": position_unit}

        violations = []
        violation_count = 0
        forbidden_pairs = set()
        for step in range(1001):
            assert all(math.isfinite(x) for x in data.qpos)
            assert all(math.isfinite(x) for x in data.qvel)
            current_violations = []
            for name, q_slot, v_slot, lower, upper, velocity in joints:
                q, v = float(data.qpos[q_slot]), abs(float(data.qvel[v_slot]))
                row = observed[name]
                row["min_position"] = min(row["min_position"], q)
                row["max_position"] = max(row["max_position"], q)
                row["max_abs_velocity"] = max(row["max_abs_velocity"], v)
                if not lower <= q <= upper:
                    current_violations.append({"step": step, "joint": name, "kind": "position", "value": q})
                if v > velocity:
                    current_violations.append({"step": step, "joint": name, "kind": "velocity", "value": v})
            pairs = forbidden_body_pairs(model, data, robot_bodies, table)
            if step == 0:
                assert not current_violations, f"home violates source limits: {current_violations}"
                assert not pairs, f"home has forbidden contacts: {sorted(pairs)}"
            violation_count += len(current_violations)
            violations.extend(current_violations[:max(0, 32 - len(violations))])
            forbidden_pairs.update(pairs)
            if step < 1000:
                mujoco.mj_step(model, data)
        assert not emitted, [str(item.message) for item in emitted]
        assert all(item.number == 0 for item in data.warning)

    strict_pass = violation_count == 0 and not forbidden_pairs
    return {"arm": selected, "mode": mode, "mujoco_version": mujoco.__version__,
            "status": "pass" if mode == "physics" or strict_pass else "fail",
            "home_initialization": "pass", "steps": 1000, "simulated_seconds": float(data.time),
            "joint_observations": observed, "hold_limit_violation_count": violation_count,
            "first_hold_limit_violations": violations, "hold_forbidden_body_pairs": sorted(forbidden_pairs),
            "strict_hold_pass": strict_pass, "final_qpos": list(map(float, data.qpos)),
            "final_qvel": list(map(float, data.qvel)),
            "scope": "home/physics prerequisite; no planner, reach-and-place, ROS transport or MCAP acceptance"}


if __name__ == "__main__":
    result = run(*sys.argv[1:])
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    sys.exit(0 if result["status"] == "pass" else 1)
