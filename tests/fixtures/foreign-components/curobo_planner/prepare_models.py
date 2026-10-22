#!/usr/bin/env python3
"""Build pinned S1-06 planner candidates from verified upstream model bytes.

--arm ur5e: capsule-derived covers with the cell's mount exclusion and the held-cube
contact proof. --arm panda: the pinned cuRobo fit of the Menagerie collision meshes
(fit_panda_spheres.py, proven by check_panda_spheres.py; decision 2026-10-14) mapped to
URDF link frames on the vendor franka.yml; link0's cover clears the table, so the Panda
cell needs no mount exclusion."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import yaml

from compare_model_fk import check_files, delta, derive_panda, derive_ur5e, model_pose

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "examples/cells/reach_place/components/curobo_planner"))
from planner_server import HELD_COVER_SPHERES  # noqa: E402
PANDA_SPHERES = HERE / "panda_spheres.json"
PANDA_SPHERES_SHA = "87ec3dbe5ba36e259abf3d79601214d9bbcc137d3884b39b2595d481358ae4a9"
PANDA_FIT_MODEL = HERE.parents[3] / "examples/cells/reach_place/assets/menagerie/franka_emika_panda/panda.xml"
FRANKA_ASSETS = "curobo/content/assets/robot/franka_description"
JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
LINKS = ("shoulder_link", "upper_arm_link", "forearm_link",
         "wrist_1_link", "wrist_2_link", "wrist_3_link")
EXPECTED_GEOMS = (1, 2, 2, 1, 2, 1)
MAX_STEP_M = 0.04
GEOMETRY_MARGIN_M = 1e-6
EXPECTED_PINOCCHIO = "4.1.0"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose(values):
    if (len(values) != 7 or not all(math.isfinite(x) for x in values) or
            abs(math.sqrt(sum(x*x for x in values[3:])) - 1) > 1e-12):
        raise ValueError("invalid xyzw pose")
    return [*values[:3], values[6], *values[3:6]]


def urdf_limits(path, names):
    joints = {j.get("name"): j for j in ET.parse(path).findall("./joint")}
    result = []
    for name in names:
        limit = joints[name].find("limit")
        values = tuple(float(limit.get(key)) for key in
                       ("lower", "upper", "velocity", "effort"))
        if not all(math.isfinite(x) for x in values) or not values[0] < values[1] or values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"invalid URDF limits: {name}")
        result.append(values)
    return result


def check_link_frames(menagerie_path, urdf_path, golden):
    menagerie = pin.buildModelFromMJCF(str(menagerie_path))
    planner = pin.buildModelFromUrdf(str(urdf_path))
    for q in ([0.0] * 6, *[v["q_rad"] for v in golden["vectors"]]):
        for link in LINKS:
            source = model_pose(menagerie, q, JOINTS, "base", link)
            target = model_pose(planner, q, JOINTS, "base_link", link)
            translation, rotation = delta(source, target)
            if translation > 1e-12 or rotation > 1e-12:
                raise ValueError(f"UR5e collision link frame mismatch: {link}")


def capsule_cover(center, axis, radius, half_length, exact_caps):
    """Spheres covering one capsule. Uniform: every sphere has the slab radius, so
    the caps are over-covered by up to hypot(r, step/2) - r. Exact caps (mount
    excluded links): the end spheres have the capsule radius and the slab spheres sit
    at slab midpoints, so the cover's extreme point along the axis is the capsule's."""
    intervals = math.ceil(2*half_length / MAX_STEP_M)
    step = 2*half_length / intervals
    slab_radius = math.hypot(radius, step / 2) + GEOMETRY_MARGIN_M
    # At each axial point its nearest center is at most step/2 away, so the slab
    # radius covers the cylinder; endpoint spheres cover the hemispherical caps.
    offsets = ([(-half_length + i*step, slab_radius) for i in range(intervals + 1)]
               if not exact_caps else
               [(-half_length, radius + GEOMETRY_MARGIN_M),
                (half_length, radius + GEOMETRY_MARGIN_M),
                *[(-half_length + (i + 0.5)*step, slab_radius) for i in range(intervals)]])
    return [{"center": [center[j] + offset*axis[j] for j in range(3)], "radius": r}
            for offset, r in offsets]


def collision_spheres(menagerie_path, exact_links=()):
    bodies = {body.get("name"): body for body in
              ET.parse(menagerie_path).findall(".//worldbody//body")}
    result, counts = {}, {}
    for link, expected in zip(LINKS, EXPECTED_GEOMS, strict=True):
        geoms = [g for g in bodies[link].findall("geom")
                 if g.get("class") in ("collision", "eef_collision")]
        if len(geoms) != expected:
            raise ValueError(f"unexpected Menagerie collision geoms: {link}")
        spheres = []
        for geom in geoms:
            radius, half_length = map(float, geom.get("size").split())
            center = list(map(float, geom.get("pos", "0 0 0").split()))
            quat = list(map(float, geom.get("quat", "1 0 0 0").split()))
            quat_norm = math.sqrt(sum(a*a for a in quat))
            if (not all(math.isfinite(x) for x in (radius, half_length, *center, *quat))
                    or radius <= 0 or half_length <= 0 or len(center) != 3
                    or len(quat) != 4 or quat_norm == 0):
                raise ValueError(f"invalid Menagerie collision geom: {link}")
            w, x, y, z = (v / quat_norm for v in quat)
            axis = (2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y))
            spheres.extend(capsule_cover(center, axis, radius, half_length,
                                         link in exact_links))
        result[link] = spheres
        counts[link] = len(spheres)
    return result, counts


def mount_clearance(urdf_path, spheres, links, table_pose_xyzw, half, configurations):
    """Lowest cover point of each excluded link above the table top, per link.
    cuRobo 0.8.0 has no per-link world-collision mask, so the planner can honor the
    cell's mount exclusion only when this clearance is non-negative: the pair then
    never reports a collision. Mounted links only rotate about the surface normal, so
    the clearance must be the same in every configuration; both are checked."""
    model = pin.buildModelFromUrdf(str(urdf_path))
    x, y, z, qx, qy, qz, qw = table_pose_xyzw
    table = pin.SE3(pin.Quaternion(qw, qx, qy, qz).toRotationMatrix(), np.array([x, y, z]))
    result = {}
    for link in links:
        values = []
        for q in configurations:
            pose = table.inverse() * model_pose(model, q, JOINTS, "base_link", link)
            values.append(min(float((pose * np.array(s["center"]))[2]) - s["radius"]
                              for s in spheres[link]) - half[2])
        if max(values) - min(values) > 1e-9:
            raise ValueError(f"mount exclusion link moves relative to the table: {link}")
        if min(values) < 0:
            raise ValueError(f"planner cannot honor mount exclusion: {link} cover "
                             f"penetrates the table by {-min(values):.6f} m")
        result[link] = min(values)
    return result


def mount_exclusion(runtime_path, table_pose_xyzw, half):
    config = json.loads(runtime_path.read_text())["config"]
    if (any(abs(a - b) > 1e-12 for a, b in zip(config["table_pose"], table_pose_xyzw, strict=True))
            or config["table_half_extents_m"] != list(half)):
        raise ValueError("table arguments differ from the runtime's cell-derived scene")
    links = config["mount_exclusion"]["links"]
    # The UR5e "base" body has no collision geometry and no planner spheres.
    if any(link not in (*LINKS, "base") for link in links):
        raise ValueError("mount exclusion names a link without a planner cover rule")
    return config["mount_exclusion"], [link for link in links if link in LINKS]


def held_cube_ignore(urdf_path, menagerie_ur5e, scene, endpoint, runtime, output):
    """Ignore links for attached_object beyond wrist_3_link. Granted only when
    check_ur5e_held_cube_contact.py, run here on this URDF, cover and generated cell
    scene, proves the cube's flush contact with wrist_2_link at the flange face is
    joint-invariant and never penetrates the source geometry (decision 2026-10-14)."""
    from check_ur5e_held_cube_contact import measure
    record = measure(urdf_path, menagerie_ur5e, scene, endpoint, runtime)
    output.write_text(json.dumps(record, indent=2) + "\n")
    if not record["wrist_2_link_ignore_justified"]:
        raise ValueError("held cube contact with wrist_2_link is not proven safe to ignore")
    return ["attached_object"], output


def panda_robot(curobo_root, urdf, menagerie_panda, golden, table, half):
    """Vendor franka.yml with the pinned proven cover in URDF link frames. Only the
    grasp-contact links (vendor grasp_contact_link_names) may ignore the held cube."""
    from check_panda_spheres import BODIES, check_frames, cover_sha256
    if sha(PANDA_SPHERES) != PANDA_SPHERES_SHA:
        raise ValueError("pinned Panda sphere fit digest mismatch")
    fit = json.loads(PANDA_SPHERES.read_text())
    if fit["source_sha256"]["panda.xml"] != sha(PANDA_FIT_MODEL):
        raise ValueError("Panda sphere fit source differs from the vendored model")
    body_from_link, _ = check_frames(PANDA_FIT_MODEL, urdf, golden)
    links = list(BODIES.values())
    spheres = {link: [{"center": (body_from_link[link].inverse()
                                  * np.array(s["center"], dtype=np.float64)).tolist(),
                       "radius": float(s["radius"])} for s in fit["links"][link]["spheres"]]
               for link in links}
    if list(table[3:]) != [0.0, 0.0, 0.0, 1.0] or max(delta(body_from_link["panda_link0"],
                                                           pin.SE3.Identity())) > 1e-12:
        raise ValueError("Panda table is not axis-aligned in the planner base frame")
    clearance = min(s["center"][2] - s["radius"] for s in spheres["panda_link0"]) - (table[2] + half[2])
    if clearance < 0:
        raise ValueError("Panda link0 cover penetrates the table")
    vendor = yaml.safe_load((curobo_root / "curobo/content/configs/robot/franka.yml")
                            .read_text())["robot_cfg"]["kinematics"]
    grasp = set(vendor["grasp_contact_link_names"]) - {"attached_object"}
    ignore = {link: [o for o in others if o in links or (o == "attached_object" and link in grasp)]
              for link, others in vendor["self_collision_ignore"].items() if link in links}
    home = [float(x) for x in ET.parse(menagerie_panda).find(
        ".//keyframe/key[@name='home']").get("qpos").split()]
    cspace = dict(vendor["cspace"]) | {"default_joint_position": home}
    attach = vendor["extra_links"]["attached_object"]
    if (cspace["joint_names"][:7] != [f"panda_joint{i}" for i in range(1, 8)] or len(home) != 9
            or {cspace["joint_names"][7]: home[7], cspace["joint_names"][8]: home[8]}
            != vendor["lock_joints"] or attach["parent_link_name"] != "panda_hand"
            or attach["fixed_transform"] != [0, 0, 0, 1, 0, 0, 0]):
        raise ValueError("unexpected vendor franka.yml")
    robot = {"robot_cfg": {"kinematics": {
        "format_version": 2.0,
        "base_link": "panda_link0",
        "tool_frames": ["panda_hand"],
        "urdf_path": str(urdf),
        "asset_root_path": str(curobo_root / FRANKA_ASSETS),
        "cspace": cspace,
        "lock_joints": vendor["lock_joints"],
        "collision_link_names": [*links, "attached_object"],
        "collision_spheres": spheres,
        "collision_sphere_buffer": 0.0,
        # The vendor buffers pad its hand-placed spheres; the proven cover needs none.
        "self_collision_buffer": {},
        "self_collision_ignore": ignore,
        "extra_collision_spheres": {"attached_object": HELD_COVER_SPHERES},
        "extra_links": {"attached_object": attach},
    }, "load_dynamics": False}}
    return robot, {"cover_sha256": cover_sha256(spheres), "link0_table_clearance_m": clearance,
                   "spheres_per_link": {link: len(spheres[link]) for link in links},
                   "fit": {"path": str(PANDA_SPHERES.relative_to(HERE.parents[3])),
                           "sha256": PANDA_SPHERES_SHA}}


def write_yaml(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("panda", "ur5e"), required=True)
    for name in ("curobo-root", "ur-root", "menagerie-root", "xacro", "goldens", "runtime",
                 "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("scene", "endpoint"):  # UR5e held-cube contact proof inputs
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--table-pose-xyzw", nargs=7, type=float, required=True)
    parser.add_argument("--table-half-extents", nargs=3, type=float, required=True)
    parser.add_argument("--cube-pose-xyzw", nargs=7, type=float, required=True)
    args = parser.parse_args()
    if args.arm == "ur5e" and not (args.scene and args.endpoint):
        raise ValueError("--arm ur5e requires --scene and --endpoint")
    if pin.__version__ != EXPECTED_PINOCCHIO:
        raise ValueError("Pinocchio version mismatch")
    roots = [args.curobo_root, args.ur_root, args.menagerie_root]
    if not all(p.is_absolute() and p.is_dir() for p in roots):
        raise ValueError("source roots must be existing absolute directories")
    if (not all(math.isfinite(x) and x > 0 for x in args.table_half_extents)
            or args.output_dir.exists() and
            (not args.output_dir.is_dir() or any(args.output_dir.iterdir()))):
        raise ValueError("invalid table or nonempty output directory")
    manifest = json.loads((HERE / "model_sources.json").read_text())
    check_files(args.curobo_root, manifest["curobo_franka"])
    check_files(args.ur_root, manifest["ros_ur5e"])
    goldens = json.loads(args.goldens.read_text())
    if goldens["menagerie_revision"] != manifest["menagerie"]["revision"]:
        raise ValueError("golden source revision mismatch")
    records = {record["arm"]: record for record in goldens["arms"]}
    if set(records) != {"panda", "ur5e"}:
        raise ValueError("golden arm mismatch")
    for arm, record in records.items():
        model = manifest["menagerie"]["models"][arm]
        for path, digest in ((model["path"], model["sha256"]),
                             (model["license_path"], model["license_sha256"])):
            if sha(args.menagerie_root / path) != digest:
                raise ValueError(f"Menagerie source mismatch: {path}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    panda_urdf, ur5e_urdf = output / "franka_panda.urdf", output / "ur5e.urdf"
    derive_panda(args.curobo_root /
                 "curobo/content/assets/robot/franka_description/franka_panda.urdf",
                 panda_urdf)
    derive_ur5e(args.ur_root, args.xacro, ur5e_urdf)
    if (sha(panda_urdf) != "eeecd06212a1ed160807243a1046ac0a1ce07d7e49370a4a17d181bed5a6f351"
            or sha(ur5e_urdf) != "5e28414e03835995fe9fed160a164718c17607e33af0b7031097bd730542aae2"):
        raise ValueError("derived URDF digest changed")
    panda_names = [f"panda_joint{i}" for i in range(1, 8)]
    panda_limits = urdf_limits(panda_urdf, panda_names)
    ur5e_limits = urdf_limits(ur5e_urdf, JOINTS)
    for actual, accepted in zip(panda_limits, records["panda"]["joint_limits_rad"], strict=True):
        if tuple(actual[:2]) != tuple(accepted):
            raise ValueError("Panda position limit mismatch")
    ur5e_policy = []
    for actual, accepted in zip(ur5e_limits, records["ur5e"]["joint_limits_rad"], strict=True):
        lo, hi = max(actual[0], accepted[0]), min(actual[1], accepted[1])
        if lo >= hi:
            raise ValueError("UR5e empty position limit intersection")
        ur5e_policy.append((lo, hi))
    if (ur5e_policy[2] != (-3.1415, 3.1415) or
            any(pair != (-2*math.pi, 2*math.pi) for i, pair in enumerate(ur5e_policy) if i != 2)):
        raise ValueError("unexpected UR5e source limits")
    exclusion, excluded = mount_exclusion(args.runtime, args.table_pose_xyzw,
                                          args.table_half_extents)
    scene = {"cuboid": {
        "table": {"dims": [2*x for x in args.table_half_extents],
                  "pose": pose(args.table_pose_xyzw)},
        "cube": {"dims": [0.05, 0.05, 0.05],
                 "pose": pose(args.cube_pose_xyzw)},
    }}
    common = {
        "panda_joint_names": panda_names,
        "panda_joint_limits_rad": [row[:2] for row in panda_limits],
        "panda_velocity_limits_rad_s": [row[2] for row in panda_limits],
        "panda_effort_limits_nm": [row[3] for row in panda_limits],
        "ur5e_joint_names": JOINTS,
        "ur5e_joint_limits_rad": ur5e_policy,
        "ur5e_velocity_limits_rad_s": [row[2] for row in ur5e_limits],
        "ur5e_effort_limits_nm": [row[3] for row in ur5e_limits],
    }
    if args.arm == "panda":
        if exclusion["links"]:
            raise ValueError("the Panda cover clears the table; the cell must not exclude links")
        menagerie_panda = args.menagerie_root / manifest["menagerie"]["models"]["panda"]["path"]
        panda, record = panda_robot(args.curobo_root, panda_urdf, menagerie_panda,
                                    records["panda"], args.table_pose_xyzw,
                                    args.table_half_extents)
        files = {"panda_config": output / "panda.yml", "scene": output / "scene.yml"}
        write_yaml(files["panda_config"], panda)
        write_yaml(files["scene"], scene)
        files["panda_urdf"] = panda_urdf
        print(json.dumps({"arm": "panda", "files": {name: {"path": str(path), "sha256": sha(path)}
                                                    for name, path in files.items()},
                          **common, "panda_collision_cover": record}, indent=2))
        return
    menagerie_ur5e = args.menagerie_root / manifest["menagerie"]["models"]["ur5e"]["path"]
    check_link_frames(menagerie_ur5e, ur5e_urdf, records["ur5e"])
    spheres, counts = collision_spheres(menagerie_ur5e, excluded)
    clearance = mount_clearance(
        ur5e_urdf, spheres, excluded, args.table_pose_xyzw, args.table_half_extents,
        [[0.0] * 6, *[v["q_rad"] for v in records["ur5e"]["vectors"]]])
    wrist_2_ignore, contact_record = held_cube_ignore(
        ur5e_urdf, menagerie_ur5e, args.scene, args.endpoint, args.runtime,
        output / "held-cube-contact.json")
    home = [float(x) for x in ET.parse(menagerie_ur5e).find(
        ".//keyframe/key[@name='home']").get("qpos").split()]
    if (len(home) != 6 or any(not lo <= q <= hi for q, (lo, hi) in
                              zip(home, ur5e_policy, strict=True))):
        raise ValueError("Menagerie UR5e home outside planner limits")

    ur5e = {"robot_cfg": {"kinematics": {
        "format_version": 2.0,
        "base_link": "base_link",
        "tool_frames": ["tool0"],
        "urdf_path": str(ur5e_urdf),
        "asset_root_path": str(args.ur_root),
        "cspace": {
            "joint_names": list(JOINTS),
            "default_joint_position": home,
            "cspace_distance_weight": [1] * 6,
            "null_space_weight": [1] * 6,
            "position_limit_clip": [0, 0, math.pi-3.1415, 0, 0, 0],
        },
        "collision_link_names": [*LINKS, "attached_object"],
        "collision_spheres": spheres,
        "collision_sphere_buffer": 0.0,
        "self_collision_buffer": {},
        "self_collision_ignore": {
            link: [LINKS[i+1]] for i, link in enumerate(LINKS[:-1])
        } | {"wrist_2_link": ["wrist_3_link", *wrist_2_ignore],
             "wrist_3_link": ["attached_object"]},
        "extra_collision_spheres": {"attached_object": HELD_COVER_SPHERES},
        "extra_links": {"attached_object": {
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_name": "attach_joint", "joint_type": "FIXED",
            "link_name": "attached_object", "parent_link_name": "tool0",
        }},
    }, "load_dynamics": False}}
    files = {"ur5e_config": output / "ur5e.yml", "scene": output / "scene.yml"}
    for name, value in (("ur5e_config", ur5e), ("scene", scene)):
        write_yaml(files[name], value)
    print(json.dumps({
        "arm": "ur5e",
        "files": {name: {"path": str(path), "sha256": sha(path)}
                  for name, path in {**files, "ur5e_urdf": ur5e_urdf}.items()},
        **common,
        "ur5e_collision_spheres_per_link": counts,
        "ur5e_held_cube_contact": {"path": str(contact_record), "sha256": sha(contact_record)},
        "ur5e_mount_exclusion": {**exclusion, "exact_cap_cover": excluded,
                                 "table_clearance_m": clearance},
    }, indent=2))


if __name__ == "__main__":
    main()
