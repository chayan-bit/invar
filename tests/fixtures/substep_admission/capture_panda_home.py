"""Capture the Panda full-cell home hold as sealed replay data for the substep policy.

Runs the binder's rendered model (`invar system gen mjcf` scene, whose SHA-256
must equal the policy's qualified `model_xml`) from the `home` key for 1,000
steps. At each step it tries gripper commands from the source-declared hold
(the home key's last control) downward, predicting each candidate on a copy of
the live data, and commits the first one the policy gate admits: joint position,
speed, control and force bounds from the source, and no robot contact with the
robot, table or environment. Output fields match `panda-home-diagnostic.json`.
`witness` is SHA-256 over qpos, qvel, act, qacc_warmstart, ctrl and time
(binary64 little-endian): a native digest, not a reconstruction of MjData.
Usage: capture_panda_home.py SCENE_XML EXPECTED_SHA256 OUTPUT_JSON
"""

import hashlib
import json
import sys

import mujoco
import numpy as np

STEPS = 1000
JOINTS = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]
SPEEDS = [2.175] * 4 + [2.61] * 3 + [0.05, 0.05]


def classes(model):
    names = {"table": "table", "cube": "object"}
    out = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        label = "environment"
        while body != 0:
            name = model.body(body).name
            if name == "link0":
                label = "robot"
                break
            if name in names:
                label = names[name]
                break
            body = int(model.body_parentid[body])
        out.append(label)
    return out


def snapshot(model, data, step, joints):
    witness = hashlib.sha256(b"".join(np.ascontiguousarray(a, dtype="<f8").tobytes() for a in (
        data.qpos, data.qvel, data.act, data.qacc_warmstart, data.ctrl, np.array([data.time])))).hexdigest()
    return {"step": step, "time": float(data.time),
            "q": [float(data.qpos[j.qposadr[0]]) for j in joints],
            "dq": [float(data.qvel[j.dofadr[0]]) for j in joints],
            "forces": [float(x) for x in data.actuator_force],
            "controls": [float(x) for x in data.ctrl],
            "contacts": [[int(c.geom1), int(c.geom2)] for c in data.contact],
            "witness": witness}


def admitted(model, state, joints, geom_class):
    for q, j in zip(state["q"], joints):
        if not j.range[0] <= q <= j.range[1]:
            return False
    if any(abs(v) > s for v, s in zip(state["dq"], SPEEDS)):
        return False
    for i, c in enumerate(state["controls"]):
        if not model.actuator_ctrlrange[i][0] <= c <= model.actuator_ctrlrange[i][1]:
            return False
    for i, f in enumerate(state["forces"]):
        if not model.actuator_forcerange[i][0] <= f <= model.actuator_forcerange[i][1]:
            return False
    forbidden = {"robot", "table", "environment"}
    return not any((geom_class[a] == "robot" and geom_class[b] in forbidden)
                   or (geom_class[b] == "robot" and geom_class[a] in forbidden)
                   for a, b in state["contacts"])


def main(scene, expected, output):
    if hashlib.sha256(open(scene, "rb").read()).hexdigest() != expected:
        raise SystemExit("scene differs from the qualified model_xml")
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    joints = [model.joint(name) for name in JOINTS]
    geom_class = classes(model)
    hold = int(data.ctrl[-1])
    if float(hold) != float(data.ctrl[-1]):
        raise SystemExit("home hold is not integral")
    initial = snapshot(model, data, 0, joints)
    candidates, executed = [], [initial]
    for step in range(1, STEPS + 1):
        for rank in range(hold + 1):
            trial = mujoco.MjData(model)
            mujoco.mj_copyData(trial, model, data)
            trial.ctrl[-1] = hold - rank
            mujoco.mj_step(model, trial)
            state = snapshot(model, trial, step, joints)
            candidates.append({"step": step, "control": hold - rank, "state": state})
            if admitted(model, state, joints, geom_class):
                data = trial
                executed.append(state)
                break
        else:
            raise SystemExit(f"no admissible gripper command at step {step}")
    document = {
        "scope": "Captured Panda full-cell home-held MuJoCo3.12.0 data. Sealed table lookup replay only; "
                 "no engine, moving-arm, ROS/JTC, attachment or host qualification. Native integration "
                 "digests are witnesses, not reconstructed MjData.",
        "model_xml_sha256": expected, "initial": initial, "candidates": candidates, "executed": executed}
    with open(output, "w") as handle:
        json.dump(document, handle, separators=(",", ":"))
    rejected = len(candidates) - STEPS
    selected = sorted({(c["step"], c["control"]) for c, ok in
                       ((c, executed[c["step"]] is c["state"]) for c in candidates) if ok and c["control"] != hold})
    print(json.dumps({"mujoco_version": mujoco.__version__, "hold": hold, "predictions": len(candidates),
                      "rejections": rejected, "non_hold_commands": selected,
                      "max_candidates_per_step": max(sum(1 for c in candidates if c["step"] == s)
                                                     for s in range(1, STEPS + 1)),
                      "final_time": repr(float(data.time))}))


if __name__ == "__main__":
    main(*sys.argv[1:])
