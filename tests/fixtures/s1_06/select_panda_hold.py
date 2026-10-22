"""Select the Panda gripper home hold command from the strict-hold criterion.

For each integral command c from 255 down, start from the generated scene's
`home` key, hold every control at its key value except `actuator8 = c`, step
10,000 times (20 s) and record every finger position against the declared
[0, 0.04] m range at all 10,001 states, with the strict test's exact check.
The selected hold is the largest c with zero violations; its margin is the
distance from the largest finger position after the initial state to 0.04 m.
Usage: select_panda_hold.py SCENE_XML [LOWEST_COMMAND]
"""

import json
import math
import sys

import mujoco

STEPS = 10_000
FINGERS = ("finger_joint1", "finger_joint2")


def run(scene, command):
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.ctrl[model.actuator("actuator8").id] = command
    joints = [model.joint(name) for name in FINGERS]
    violations, peak_after_start, low = 0, -math.inf, math.inf
    for step in range(STEPS + 1):
        for joint in joints:
            q = float(data.qpos[joint.qposadr[0]])
            lower, upper = (float(x) for x in joint.range)
            violations += not lower <= q <= upper
            low = min(low, q)
            if step > 0:
                peak_after_start = max(peak_after_start, q)
        if step < STEPS:
            mujoco.mj_step(model, data)
    return {"command": command, "violations": violations,
            "finger_max_after_start_m": repr(peak_after_start),
            "margin_to_upper_m": repr(0.04 - peak_after_start),
            "finger_min_m": repr(low),
            "steady_setpoint_m": repr(float(model.actuator_gainprm[model.actuator("actuator8").id][0]) * command / 100)}


def main(scene, lowest="245"):
    rows = [run(scene, c) for c in range(255, int(lowest) - 1, -1)]
    selected = next(row for row in rows if row["violations"] == 0)
    print(json.dumps({"mujoco_version": mujoco.__version__, "steps": STEPS, "rows": rows,
                      "selected": selected}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
