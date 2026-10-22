"""A forbidden contact first appears at integrated state 1000."""

import json

import mujoco

from cell_physics_smoke import forbidden_body_pairs

model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="table"><geom type="sphere" size="0.1"/></body>
    <body name="arm" pos="2.199 0 0">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
""")
state = mujoco.MjData(model)
state.qvel[0] = -1
robot_bodies = {model.body("arm").id}
table = model.body("table").id
for step in range(1000):
    assert not forbidden_body_pairs(model, state, robot_bodies, table), step
    mujoco.mj_step(model, state)
assert state.ncon == 0, "fixture must expose stale post-step contacts"
assert forbidden_body_pairs(model, state, robot_bodies, table) == {("arm", "table")}
assert state.ncon > 0
assert all(item.number == 0 for item in state.warning)
print(json.dumps({"status": "pass", "steps": 1000,
                  "stale_final_contact_count": 0, "refreshed_final_contact_count": state.ncon}))
