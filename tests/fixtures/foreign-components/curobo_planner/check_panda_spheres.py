#!/usr/bin/env python3
"""Independent coverage proof for the fitted Panda planner spheres (decision 2026-10-14).

Uses only compiled MuJoCo 3.12.0 geometry (not the fitter's trimesh loading): every
group-3 collision geom of each body, as its mesh triangles and its convex hull (what
MuJoCo collides), plus box faces. Surfaces are sampled on a barycentric grid; since
f(p) = min_i(|p - c_i| - r_i) is 1-Lipschitz and every surface point lies within 2/3
of a sub-triangle edge of a sample, sup f <= max sampled f + that radius. The bound is
reported per link and must not exceed MARGIN_M. link0 stands on the table and its cover
must clear the table top, so its bound covers every surface point more than MARGIN_M
above the table (the contact slab is reported, not bounded). Pinocchio checks that each
Menagerie body frame is rigidly related to its URDF link frame (where planner spheres
are applied), and the link0 spheres must clear the cell table top.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
from scipy.spatial import ConvexHull
import yaml

from compare_model_fk import delta, model_pose

MODEL_SHA = "d0de66277ee0a56dcf9a2a997f7cd6be3a157fc32cb567b658cc116c79a0430b"
URDF_SHA = "eeecd06212a1ed160807243a1046ac0a1ce07d7e49370a4a17d181bed5a6f351"
MARGIN_M = 0.002
SPACING_M = 0.0005
CHUNK = 200_000
BODIES = {"link0": "panda_link0", "link1": "panda_link1", "link2": "panda_link2",
          "link3": "panda_link3", "link4": "panda_link4", "link5": "panda_link5",
          "link6": "panda_link6", "link7": "panda_link7", "hand": "panda_hand",
          "left_finger": "panda_leftfinger", "right_finger": "panda_rightfinger"}
ARM = [f"joint{i}" for i in range(1, 8)]
FINGERS = ("finger_joint1", "finger_joint2")
FINGER_OPEN_M = 0.04  # cuRobo franka.yml lock_joints


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frames(model_path, urdf_path, goldens):
    """Constant body_from_link transform per collision link (Menagerie body frame vs
    derived URDF link frame), checked at zero, home-like and every golden state."""
    source = pin.buildModelFromMJCF(str(model_path))
    planner = pin.buildModelFromUrdf(str(urdf_path))
    links = (*[f"panda_joint{i}" for i in range(1, 8)], "panda_finger_joint1", "panda_finger_joint2")
    result, variation = {}, 0.0
    for q in ([0.0] * 7, *[v["q_rad"] for v in goldens["vectors"]]):
        q = [*q, FINGER_OPEN_M, FINGER_OPEN_M]
        for body, link in BODIES.items():
            relative = (model_pose(source, q, (*ARM, *FINGERS), "link0", body).inverse()
                        * model_pose(planner, q, links, "panda_link0", link))
            first = result.setdefault(link, relative)
            variation = max(variation, *delta(first, relative))
    if variation > 1e-12:
        raise ValueError(f"Menagerie body and URDF link frames are not rigidly related: {variation}")
    return result, variation


def cover_sha256(spheres):
    """Canonical digest of a planner cover in URDF link frames; planner_server.py pins it."""
    rows = {link: [[*map(float, s["center"]), float(s["radius"])] for s in spheres[link]]
            for link in sorted(spheres)}
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def body_spheres(path, body_from_link):
    """Spheres per URDF link in Menagerie body frames, from a fit JSON (body frames)
    or a cuRobo robot yml (URDF link frames, whose cover digest is also returned)."""
    if path.suffix in (".yml", ".yaml"):
        rows = yaml.safe_load(path.read_text())["robot_cfg"]["kinematics"]["collision_spheres"]
        rows = {link: spheres for link, spheres in rows.items() if link in BODIES.values()}
        body = {link: [{"center": (body_from_link[link] * np.array(s["center"], dtype=np.float64)).tolist(),
                        "radius": s["radius"]} for s in spheres]
                for link, spheres in rows.items()}
        return body, None, cover_sha256(rows)
    fit = json.loads(path.read_text())
    return ({link: value["spheres"] for link, value in fit["links"].items()},
            fit["source_sha256"], None)


def overlaps(model, data, world):
    """Diagnostic: sphere-cover separation of every body pair at the current state."""
    excluded = {int(v) for v in model.exclude_signature}
    rows = []
    for (a, (ca, ra)), (b, (cb, rb)) in itertools.combinations(world.items(), 2):
        separation = float((np.linalg.norm(ca[:, None] - cb[None], axis=2) - ra[:, None] - rb[None]).min())
        ia, ib = model.body(a).id, model.body(b).id
        rows.append({"bodies": [a, b], "separation_m": separation,
                     "parent_child": int(model.body_parentid[ia]) == ib or int(model.body_parentid[ib]) == ia,
                     "mujoco_excluded": ((min(ia, ib) + 1) << 16) + max(ia, ib) + 1 in excluded})
    return [r for r in rows if r["separation_m"] < 0]


def triangles(model, data, geom):
    rotation, origin = data.geom_xmat[geom].reshape(3, 3), data.geom_xpos[geom]
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        v0, nv = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        f0, nf = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        vertices = model.mesh_vert[v0:v0 + nv].astype(np.float64)
        faces = model.mesh_face[f0:f0 + nf]
    elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        vertices = np.array(list(itertools.product(*[[-s, s] for s in model.geom_size[geom]])))
        faces = None
    else:
        raise ValueError("unexpected collision geom type")
    world = vertices @ rotation.T + origin
    hull = world[ConvexHull(world).simplices]
    return hull if faces is None else np.concatenate([world[faces], hull])


def samples(tri):
    edges = np.stack([np.linalg.norm(tri[:, i] - tri[:, (i + 1) % 3], axis=1)
                      for i in range(3)], axis=1).max(axis=1)
    counts = np.maximum(np.ceil(edges / SPACING_M).astype(int), 1)
    for n in np.unique(counts):
        group = tri[counts == n]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (i + j) <= n
        a, b = i[keep] / n, j[keep] / n
        e1, e2 = group[:, 1] - group[:, 0], group[:, 2] - group[:, 0]
        for start in range(0, len(group), max(1, CHUNK // len(a))):
            s = slice(start, start + max(1, CHUNK // len(a)))
            yield ((group[s, None, 0] + a[None, :, None] * e1[s, None]
                    + b[None, :, None] * e2[s, None]).reshape(-1, 3),
                   2.0 / 3.0 * float(edges[counts == n].max()) / n)


def coverage(model, data, body, centers, radii, band=None):
    """Gap bound over the body's collision surfaces; with `band`, only points more than
    `band` above the body's z = 0 plane (link0: the table top) count toward the bound,
    using every sample within its sampling radius of that region."""
    geoms = [g for g in range(model.ngeom)
             if model.geom_bodyid[g] == body and model.geom_group[g] == 3]
    rotation, origin = data.xmat[body].reshape(3, 3), data.xpos[body]
    worst = {"all": (-np.inf, None), "counted": (-np.inf, None)}
    radius, count = 0.0, 0
    for geom in geoms:
        for points, bound in samples(triangles(model, data, geom)):
            d = (np.linalg.norm(points[:, None] - centers[None], axis=2) - radii).min(axis=1)
            count += len(points); radius = max(radius, bound)
            height = (points - origin) @ rotation[:, 2]
            for key, mask in (("all", np.ones(len(d), bool)),
                              ("counted", np.ones(len(d), bool) if band is None else height > band - bound)):
                if mask.any() and d[mask].max() > worst[key][0]:
                    worst[key] = (float(d[mask].max()), points[mask][d[mask].argmax()])
    local = {key: (rotation.T @ (where - origin)).tolist() for key, (_, where) in worst.items()}
    return {"geoms": len(geoms), "samples": count, "sampling_radius_m": radius,
            "excluded_band_m": band,
            "max_sampled_gap_m": worst["counted"][0],
            "gap_bound_m": max(0.0, worst["counted"][0] + radius),
            "worst_point_body_frame_m": local["counted"],
            "all_surface_max_sampled_gap_m": worst["all"][0],
            "all_surface_worst_point_body_frame_m": local["all"]}


def main():
    parser = argparse.ArgumentParser()
    for name in ("model-dir", "spheres", "urdf", "goldens", "runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if mujoco.__version__ != "3.12.0" or pin.__version__ != "4.1.0":
        raise ValueError("requires pinned MuJoCo 3.12.0 and Pinocchio 4.1.0")
    model_path = args.model_dir / "panda.xml"
    if sha(model_path) != MODEL_SHA or sha(args.urdf) != URDF_SHA:
        raise ValueError("Panda model or URDF digest mismatch")
    goldens = next(a for a in json.loads(args.goldens.read_text())["arms"] if a["arm"] == "panda")
    body_from_link, variation = check_frames(model_path, args.urdf, goldens)
    spheres, sources, planner_cover = body_spheres(args.spheres, body_from_link)
    for name, digest in (sources or {}).items():
        if sha(args.model_dir / name) != digest:
            raise ValueError(f"fit input differs from the checked model: {name}")
    if set(spheres) != set(BODIES.values()):
        raise ValueError("sphere links differ from the Panda collision links")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if int((model.geom_group == 3).sum()) != 23:
        raise ValueError("compiled collision geometry count mismatch")
    data = mujoco.MjData(model)
    local = {}
    for body, link in BODIES.items():
        c = np.array([s["center"] for s in spheres[link]], dtype=np.float64).reshape(-1, 3)
        r = np.array([s["radius"] for s in spheres[link]], dtype=np.float64)
        if len(r) == 0 or not (np.isfinite(c).all() and np.isfinite(r).all() and (r > 0).all()):
            raise ValueError(f"invalid spheres: {link}")
        local[body] = (c, r)

    def pose(q_arm):
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        for name, value in zip(ARM, q_arm or [], strict=False):
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        for joint in FINGERS:
            data.qpos[model.jnt_qposadr[model.joint(joint).id]] = FINGER_OPEN_M
        mujoco.mj_forward(model, data)
        return {body: (c @ data.xmat[model.body(body).id].reshape(3, 3).T + data.xpos[model.body(body).id], r)
                for body, (c, r) in local.items()}

    world = pose(None)
    links = {}
    for body, link in BODIES.items():
        band = MARGIN_M if body == "link0" else None
        links[link] = coverage(model, data, model.body(body).id, *world[body], band) | {"spheres": len(local[body][1])}
        print(link, links[link], flush=True)
    self_overlaps = {"home": overlaps(model, data, world)}
    for index, vector in enumerate(goldens["vectors"]):
        self_overlaps[f"golden_{index}"] = overlaps(model, data, pose(vector["q_rad"]))
    config = json.loads(args.runtime.read_text())["config"]
    table, half = config["table_pose"], config["table_half_extents_m"]
    if table[3:] != [0.0, 0.0, 0.0, 1.0] or max(delta(body_from_link["panda_link0"], pin.SE3.Identity())) > 1e-12:
        raise ValueError("table or link0 frame is not the planner base frame")
    c, r = local["link0"]
    clearance = float((c[:, 2] - r).min()) - (table[2] + half[2])
    worst = max(v["gap_bound_m"] for v in links.values())
    result = {
        "schema": "invar-panda-sphere-coverage-v1",
        "status": "pass" if worst <= MARGIN_M and clearance >= 0 else "fail",
        "margin_m": MARGIN_M, "worst_gap_bound_m": worst,
        "link0_table_clearance_m": clearance,
        "body_from_link_rotation_deg": {link: float(np.degrees(np.linalg.norm(pin.log3(t.rotation))))
                                         for link, t in body_from_link.items()},
        "frame_variation_max": variation,
        "inputs_sha256": {"panda.xml": sha(model_path), "urdf": sha(args.urdf),
                          "spheres": sha(args.spheres), "goldens": sha(args.goldens),
                          "runtime": sha(args.runtime)},
        "check_sha256": sha(Path(__file__)),
        "planner_cover_sha256": planner_cover,
        "sampling_spacing_m": SPACING_M,
        "configuration": "home keyframe, fingers 0.04 m",
        "mujoco": mujoco.__version__, "pinocchio": pin.__version__,
        "links": links,
        "cover_overlaps_diagnostic": self_overlaps,
    }
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("links", "cover_overlaps_diagnostic")}, indent=2))
    for state, rows in self_overlaps.items():
        print(state, [(r["bodies"], round(r["separation_m"], 4), r["parent_child"], r["mujoco_excluded"]) for r in rows])
    if result["status"] != "pass":
        raise SystemExit("Panda sphere coverage fails the stated margin")


if __name__ == "__main__":
    main()
