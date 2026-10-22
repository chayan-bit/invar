"""Print imported inertias for diagnosing a frozen Pinocchio/MuJoCo mismatch."""

import sys

import mujoco
import numpy as np
import pinocchio as pin

source = sys.argv[1]
model_p = pin.buildModelFromMJCF(source)
model_m = mujoco.MjModel.from_xml_path(source)
for name in [f"joint{i}" for i in range(1, 8)]:
    joint_p = model_p.getJointId(name)
    joint_m = mujoco.mj_name2id(model_m, mujoco.mjtObj.mjOBJ_JOINT, name)
    body_m = int(model_m.jnt_bodyid[joint_m])
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, model_m.body_iquat[body_m])
    rot = rot.reshape(3, 3)
    inertia_m = rot @ np.diag(model_m.body_inertia[body_m]) @ rot.T
    inertia_p = model_p.inertias[joint_p]
    print(name, model_m.body(body_m).name)
    print("mass", inertia_p.mass, model_m.body_mass[body_m])
    print("COM delta", np.asarray(inertia_p.lever) - model_m.body_ipos[body_m])
    print("I delta", np.asarray(inertia_p.inertia) - inertia_m)
    if name == "joint1":
        print("I pin", np.asarray(inertia_p.inertia))
        print("I mujoco", inertia_m)
    print("pin placement", model_p.jointPlacements[joint_p])
