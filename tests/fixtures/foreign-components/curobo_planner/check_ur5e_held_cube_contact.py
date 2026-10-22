#!/usr/bin/env python3
"""Held-cube contact with the UR5e wrist links (decision 2026-10-14).

The planner may ignore attached_object against wrist_2_link, like wrist_3_link, only
when the source cube's separation from the source link's collision geometry is
joint-invariant and never negative. The planner-model separation (the held-cube cover
spheres against the link's cover spheres) is reported alongside; it is not invariant,
because the cover's corner spheres turn with wrist_3. Both are measured over a seeded
joint sample plus a dense wrist_3 sweep; the result is reported, never assumed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[4]
                       / "examples/cells/reach_place/components/curobo_planner"))
from planner_server import compose, cube_cover  # noqa: E402
from compare_model_fk import model_pose  # noqa: E402
from prepare_models import JOINTS, collision_spheres, mount_exclusion  # noqa: E402

LINKS = ("wrist_2_link", "wrist_3_link")
SAMPLES = 2000
SWEEP = 1441  # wrist_3 over [-2pi, 2pi] in 0.5-degree steps
INVARIANT_M = 1e-9
# mj_geomDistance rounding: a flush contact reads within a few 1e-16 m of zero.
PENETRATION_TOLERANCE_M = 1e-12


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configurations(limits):
    rng = np.random.default_rng(20261014)
    sample = [[rng.uniform(lo, hi) for lo, hi in limits] for _ in range(SAMPLES)]
    sweep = [[0.0, -1.5708, 1.5708, -1.5708, -1.5708, a]
             for a in np.linspace(-2*math.pi, 2*math.pi, SWEEP)]
    return sample + sweep


def cover_sha256(spheres):
    """Digest of the wrist covers this check measured; prepare_models.py compares it."""
    text = json.dumps({link: spheres[link] for link in LINKS}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def planner_separation(urdf, spheres, tool_from_cube, edge, qs):
    model = pin.buildModelFromUrdf(str(urdf))
    # The held cover exactly as the planner child places it in the tool frame.
    cover = [(np.array(compose(tool_from_cube, (x, y, z, 0.0, 0.0, 0.0, 1.0))[:3]), r)
             for x, y, z, r in cube_cover(edge)]
    result = {}
    for link in LINKS:
        values = []
        for q in qs:
            link_from_tool = model_pose(model, q, JOINTS, link, "tool0")
            values.append(min(float(np.linalg.norm(link_from_tool * c - s["center"]))
                              - s["radius"] - r for c, r in cover for s in spheres[link]))
        result[link] = {"min_m": min(values), "max_m": max(values),
                        "joint_invariant": max(values) - min(values) <= INVARIANT_M}
    return result


def source_separation(scene, declared, tool_from_cube, edge, qs):
    spec = mujoco.MjSpec.from_file(str(scene))
    compiled = spec.compile()
    site = compiled.site("invar_tool")
    if compiled.body(int(site.bodyid[0])).name != "wrist_3_link":
        raise ValueError("invar_tool is not on wrist_3_link")
    pos, quat = declared
    if (np.abs(site.pos - pos).max() > 1e-12
            or min(np.abs(site.quat - quat).max(), np.abs(site.quat + quat).max()) > 1e-12):
        raise ValueError("generated scene tool site differs from the runtime's cell tool")
    w, qx, qy, qz = (float(v) for v in site.quat)  # MuJoCo wxyz; pin.Quaternion(w, x, y, z)
    tool = pin.SE3(pin.Quaternion(w, qx, qy, qz).toRotationMatrix(), site.pos.copy())
    x, y, z, qx, qy, qz, qw = tool_from_cube
    local = tool * pin.SE3(pin.Quaternion(qw, qx, qy, qz).toRotationMatrix(), np.array([x, y, z]))
    quat = pin.Quaternion(local.rotation)
    spec.body("wrist_3_link").add_geom(
        name="held_cube_probe", type=mujoco.mjtGeom.mjGEOM_BOX, size=[edge/2]*3,
        pos=local.translation, quat=[quat.w, quat.x, quat.y, quat.z],
        contype=0, conaffinity=0, group=5)
    model = spec.compile()
    data = mujoco.MjData(model)
    cube = model.geom("held_cube_probe").id
    # Frame guard: the probe centre is the compiled tool site composed with the grasp.
    mujoco.mj_forward(model, data)
    site_id = model.site("invar_tool").id
    expected = data.site_xpos[site_id] + data.site_xmat[site_id].reshape(3, 3) @ np.array(tool_from_cube[:3])
    if np.abs(data.geom_xpos[cube] - expected).max() > 1e-12:
        raise ValueError("held cube probe is not at the tool composed with the grasp")
    result = {}
    for link in LINKS:
        body = model.body(link).id
        geoms = [g for g in range(model.ngeom)
                 if model.geom_bodyid[g] == body and model.geom_group[g] == 3]
        if not geoms:
            raise ValueError(f"no source collision geometry: {link}")
        values = []
        for q in qs:
            for name, value in zip(JOINTS, q, strict=True):
                data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
            mujoco.mj_forward(model, data)
            values.append(min(float(mujoco.mj_geomDistance(model, data, cube, g, 1.0, None))
                              for g in geoms))
        result[link] = {"min_m": min(values), "max_m": max(values),
                        "joint_invariant": max(values) - min(values) <= INVARIANT_M,
                        "never_penetrates": min(values) >= -PENETRATION_TOLERANCE_M}
    return result


def cell_tool_site(runtime):
    """The runtime's declared cell tool as (pos, wxyz unit quaternion) in wrist_3_link."""
    pose = runtime["tool_projection"]["cell_tool"]
    if pose["link"] != "wrist_3_link":
        raise ValueError("UR5e cell tool is not on wrist_3_link")
    pos = np.array([float(v["value"]) for v in pose["pose"]["xyz"]])
    quat = np.array([float(v) for v in pose["pose"]["quat"]])
    return pos, quat / np.linalg.norm(quat)


def measure(urdf, menagerie_ur5e, scene, endpoint_path, runtime_path):
    if mujoco.__version__ != "3.12.0" or pin.__version__ != "4.1.0":
        raise ValueError("requires pinned MuJoCo 3.12.0 and Pinocchio 4.1.0")
    endpoint = json.loads(endpoint_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    reference = runtime["tool_projection"]["reference_from_cell"]
    if (reference["translation_m"] != [0.0, 0.0, 0.0]
            or abs(abs(reference["quaternion_xyzw"][3]) - 1) > 1e-12):
        raise ValueError("cell tool is not the planner tool0; cube transform would need mapping")
    cubes = {tuple(v["tool_from_cube"]) for v in endpoint["vectors"]}
    if len(cubes) != 1:
        raise ValueError("requests disagree on the grasp")
    tool_from_cube = list(cubes.pop())
    config = runtime["config"]
    edge = config["cube_edge_m"]
    qs = configurations(config["joint_limits_rad"])
    # The planner cover exactly as prepare_models.py builds it for this runtime.
    _, excluded = mount_exclusion(runtime_path, config["table_pose"], config["table_half_extents_m"])
    spheres, _ = collision_spheres(menagerie_ur5e, excluded)
    planner = planner_separation(urdf, spheres, tool_from_cube, edge, qs)
    source = source_separation(scene, cell_tool_site(runtime), tool_from_cube, edge, qs)
    wrist_2 = source["wrist_2_link"]
    ignore = wrist_2["joint_invariant"] and wrist_2["never_penetrates"]
    return {
        "schema": "invar-ur5e-held-cube-contact-v2",
        "inputs_sha256": {"urdf": sha(urdf), "menagerie_ur5e": sha(menagerie_ur5e),
                          "scene": sha(scene), "endpoint": sha(endpoint_path),
                          "runtime": sha(runtime_path)},
        "check_sha256": sha(Path(__file__)),
        "cover_sha256": cover_sha256(spheres),
        "penetration_tolerance_m": PENETRATION_TOLERANCE_M,
        "tool_from_cube": tool_from_cube, "held_cube_cover": cube_cover(edge),
        "configurations": {"seeded_uniform": SAMPLES, "wrist_3_sweep": SWEEP, "seed": 20261014},
        "planner_cover_separation": planner,
        "source_geometry_separation": source,
        "wrist_2_link_ignore_justified": ignore,
        "mujoco": mujoco.__version__, "pinocchio": pin.__version__,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("urdf", "menagerie-ur5e", "scene", "endpoint", "runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.urdf, args.menagerie_ur5e, args.scene, args.endpoint, args.runtime)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
