#!/usr/bin/env python3
"""Fit Panda planner collision spheres from the vendored Menagerie collision geometry.

Decision 2026-10-14: cuRobo 0.8.0's MorphIt fitter places the spheres per collision
geom in its Menagerie body frame. link0 is fitted with cuRobo's clip plane at the table
top, so its spheres clear the table by CLEARANCE_M and no mount exclusion is needed; its
surface within TABLE_BAND_M of the table (the contact plate) is not a cover target.
MorphIt approximates volume, not a cover, so radii are then grown (by at most
GROW_LIMIT_M, and on link0 only while the sphere still clears the table) until every
sample of the mesh surface and of its convex hull (MuJoCo collides mesh hulls) is inside
the union with the sampling bound to spare: f(p) = min_i(|p - c_i| - r_i) is 1-Lipschitz
and every surface point lies within 2/3 of a sub-triangle edge of a sample. Targets that
growth cannot reach (MorphIt leaves convex corners, and link0's table-clamped spheres
cannot reach down its base corners) get greedy surface-centred spheres of
SURFACE_RADIUS_M, as cuRobo's SURFACE fit places them, lifted on link0 to clear the
table. Coverage is proven independently by check_panda_spheres.py.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh

import curobo
from curobo._src.types.device_cfg import DeviceCfg
from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh

MODEL_SHA = "93a8d8e137ae43a33e7499ff627c1f633fb27d94f58ea7545b97c8f65027db1d"
URDF_LINKS = {"link0": "panda_link0", "link1": "panda_link1", "link2": "panda_link2",
              "link3": "panda_link3", "link4": "panda_link4", "link5": "panda_link5",
              "link6": "panda_link6", "link7": "panda_link7", "hand": "panda_hand",
              "left_finger": "panda_leftfinger", "right_finger": "panda_rightfinger"}
CLEARANCE_M = 0.001      # link0 sphere bottoms above the table top (link0 z = 0)
CUROBO_CLIP_BUFFER_M = 0.02  # cuRobo 0.8.0 _apply_clip_plane hard buffer
GROW_SPACING_M = 0.001
TABLE_BAND_M = 0.002     # = check_panda_spheres.MARGIN_M; link0 table-contact slab
LINK0_SLACK_M = 0.001
GROW_LIMIT_M = 0.015     # largest radius growth of a fitted sphere
SURFACE_RADIUS_M = 0.01  # surface-centred patch spheres (as cuRobo's SURFACE fit)
MAX_PATCHES = 2000
SEED = 20261014
CHUNK = 200_000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collision_geoms(model_dir):
    """(body, geom index, trimesh in body frame, source files) for each collision geom."""
    source = model_dir / "panda.xml"
    if sha(source) != MODEL_SHA:
        raise ValueError("vendored Panda model digest mismatch")
    root = ET.parse(source).getroot()
    meshdir = model_dir / root.find("compiler").get("meshdir")
    meshes = {m.get("name", Path(m.get("file")).stem): m for m in root.findall("./asset/mesh")}
    defaults = {d.get("class"): d.find("geom").attrib for d in root.iter("default")
                if d.get("class") and d.find("geom") is not None}
    result, files = [], {"panda.xml": sha(source)}
    for body in root.iter("body"):
        for index, geom in enumerate(body.findall("geom")):
            cls = geom.get("class")
            if cls == "visual":
                continue
            if cls != "collision" and not (cls or "").startswith("fingertip_pad_collision_"):
                raise ValueError(f"unexpected geom class in {body.get('name')}")
            fields = dict(defaults["collision"]) | dict(defaults[cls]) | dict(geom.attrib)
            if any(k in fields for k in ("quat", "euler", "axisangle", "xyaxes", "zaxis", "fromto")):
                raise ValueError("geom orientation handling required")
            pos = np.array([float(v) for v in fields.get("pos", "0 0 0").split()])
            if fields["type"] == "mesh":
                mesh_elem = meshes[fields["mesh"]]
                if set(mesh_elem.attrib) - {"name", "file"}:
                    raise ValueError("mesh scale/refpos handling required")
                path = meshdir / mesh_elem.get("file")
                files[f"assets/{path.name}"] = sha(path)
                mesh = trimesh.load(path, force="mesh", process=False)
            elif fields["type"] == "box":
                size = np.array([float(v) for v in fields["size"].split()])
                mesh = trimesh.creation.box(extents=2 * size)
            else:
                raise ValueError("unsupported collision primitive")
            mesh = trimesh.Trimesh(mesh.vertices + pos, mesh.faces, process=False)
            result.append((body.get("name"), index, mesh))
    if len(result) != 23 or {b for b, _, _ in result} != set(URDF_LINKS):
        raise ValueError("unexpected Panda collision geometry")
    return result, files


def surface_samples(vertices, faces, spacing):
    """Barycentric grid on every triangle; returns points and the covering radius."""
    tri = vertices[faces]
    edges = np.stack([np.linalg.norm(tri[:, i] - tri[:, (i + 1) % 3], axis=1)
                      for i in range(3)], axis=1).max(axis=1)
    counts = np.maximum(np.ceil(edges / spacing).astype(int), 1)
    points, bound = [], 0.0
    for n in np.unique(counts):
        group = tri[counts == n]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (i + j) <= n
        a, b = i[keep] / n, j[keep] / n
        e1, e2 = group[:, 1] - group[:, 0], group[:, 2] - group[:, 0]
        points.append((group[:, None, 0] + a[None, :, None] * e1[:, None]
                       + b[None, :, None] * e2[:, None]).reshape(-1, 3))
        bound = max(bound, 2.0 / 3.0 * float(edges[counts == n].max()) / n)
    return np.concatenate(points), bound


def excess(points, centers, radii):
    """min_i(|p - c_i| - r_i) and the argmin sphere, chunked."""
    values, nearest = [], []
    for start in range(0, len(points), CHUNK):
        d = np.linalg.norm(points[start:start + CHUNK, None] - centers[None], axis=2) - radii
        values.append(d.min(axis=1)); nearest.append(d.argmin(axis=1))
    return np.concatenate(values), np.concatenate(nearest)


def grow(points, centers, radii, cap, delta):
    """Grow radii so every point is inside by delta; cap[i] bounds radius i. Each
    uncovered point picks the feasible sphere needing the least growth."""
    radii = radii.copy()
    value, _ = excess(points, centers, radii)
    todo = points[value > -delta]
    uncovered = []
    for start in range(0, len(todo), CHUNK):
        p = todo[start:start + CHUNK]
        need = np.linalg.norm(p[:, None] - centers[None], axis=2) + delta
        growth = np.where(need <= cap[None], need - radii[None], np.inf)
        best = growth.argmin(axis=1)
        ok = np.isfinite(growth[np.arange(len(p)), best])
        np.maximum.at(radii, best[ok], need[np.arange(len(p)), best][ok])
        uncovered.append(p[~ok])
    return radii, np.concatenate(uncovered) if uncovered else np.zeros((0, 3))


def host(value):
    """cuRobo returns numpy or (CUDA) torch values; bring them to float64 numpy."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def scalar(value):
    if isinstance(value, bool) or value is None or isinstance(value, (str, dict, list, tuple)):
        return None
    try:
        array = host(value)
    except (TypeError, ValueError):
        return None
    return float(array) if array.size == 1 else None


def fit_spheres(body, meshes, device):
    clip = ((0.0, 0.0, 1.0), CLEARANCE_M - CUROBO_CLIP_BUFFER_M) if body == "link0" else None
    centers, radii, metrics = [], [], []
    for index, mesh in meshes:
        torch.manual_seed(SEED); np.random.seed(SEED)
        fit = fit_spheres_to_mesh(mesh, fit_type=SphereFitType.MORPHIT,
                                  compute_metrics=True, clip_plane=clip,
                                  device_cfg=DeviceCfg(device=device))
        if fit.num_spheres < 1:
            raise ValueError(f"empty fit: {body}[{index}]")
        centers.append(host(fit.centers).reshape(-1, 3))
        radii.append(host(fit.radii).reshape(-1))
        m = fit.metrics
        metrics.append({"geom_index": index, "spheres": int(fit.num_spheres),
                        "curobo_metrics": {k: scalar(v) for k, v in vars(m).items()
                                           if scalar(v) is not None} if m else None})
    centers, radii = np.concatenate(centers), np.concatenate(radii)
    if body != "link0":
        return centers, radii, np.full(len(radii), np.inf), metrics, None
    # cuRobo's hard clamp runs before its float32 cast and floors radii at 1e-4 m;
    # re-apply the same rule (radius <= height above the plane) exactly in float64.
    cap = centers[:, 2] - CLEARANCE_M
    keep = cap > 0
    over = np.maximum(radii - cap, 0.0)
    clamp = {"removed": int((~keep).sum()), "clamped": int((over > 0).sum()),
             "max_clamp_m": float(over.max())}
    return centers[keep], np.minimum(radii, cap)[keep], cap[keep], metrics, clamp


def patch(targets, centers, radii, inside, lift):
    """Greedy surface patches for targets no fitted sphere can hold: a sphere of
    SURFACE_RADIUS_M centred on the worst remaining target (link0: lifted until its
    bottom clears the table by CLEARANCE_M), until every target is held."""
    value, _ = excess(targets, centers, radii)
    todo, added = targets[value > -inside], []
    while len(todo):
        if len(added) == MAX_PATCHES:
            raise ValueError("surface patching did not converge")
        worst = todo[excess(todo, centers, radii)[0].argmax()]
        center = worst + [0.0, 0.0, max(0.0, SURFACE_RADIUS_M + CLEARANCE_M - worst[2])] \
            if lift else worst
        centers = np.vstack([centers, center]); radii = np.append(radii, SURFACE_RADIUS_M)
        added.append(center)
        todo = todo[excess(todo, centers[-1:], radii[-1:])[0] > -inside]
    return centers, radii, len(added)


def fit_body(body, meshes, device):
    samples, bounds = [], []
    for _, mesh in meshes:
        for shape in (mesh, mesh.convex_hull):
            points, bound = surface_samples(np.asarray(shape.vertices), np.asarray(shape.faces),
                                            GROW_SPACING_M)
            samples.append(points); bounds.append(bound)
    samples, delta = np.concatenate(samples), max(bounds)
    link0 = body == "link0"
    # link0: the slab within TABLE_BAND_M of the table top is the table contact (the
    # cover must clear the table); every sample that can bound a point above it is a target.
    # Just above the band no sphere clearing the table can hold a point by the full
    # sampling radius, so link0 targets may stay up to LINK0_SLACK_M (< the margin) outside.
    targets = samples[samples[:, 2] > TABLE_BAND_M - delta] if link0 else samples
    inside = delta - LINK0_SLACK_M if link0 else delta
    centers, fitted, table_cap, metrics, clamp = fit_spheres(body, meshes, device)
    radii, uncovered = grow(targets, centers, fitted, np.minimum(table_cap, fitted + GROW_LIMIT_M),
                            inside)
    centers, radii, patches = patch(targets, centers, radii, inside, link0)
    worst, _ = excess(targets, centers, radii)
    everything, _ = excess(samples, centers, radii)
    grown = radii[:len(fitted)] - fitted
    return {
        "spheres": [{"center": c.tolist(), "radius": float(r)} for c, r in zip(centers, radii)],
        "fit": metrics, "fitted_spheres": int(len(fitted)),
        "samples": int(len(samples)), "target_samples": int(len(targets)),
        "sample_covering_radius_m": delta,
        "grown_spheres": int((grown > 0).sum()), "max_growth_m": float(grown.max()),
        "targets_beyond_growth": int(len(uncovered)), "surface_patches": patches,
        "worst_target_excess_m": float(worst.max()),
        "target_gap_bound_m": max(0.0, float(worst.max()) + delta),
        "all_surface_worst_excess_m": float(everything.max()),
        "min_bottom_above_table_m": float((centers[:, 2] - radii).min()) if link0 else None,
        "float64_clip_clamp": clamp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if curobo.__version__ != "0.8.0":
        raise ValueError("requires cuRobo 0.8.0")
    geoms, files = collision_geoms(args.model_dir)
    links = {}
    for body in URDF_LINKS:
        meshes = [(i, m) for b, i, m in geoms if b == body]
        links[URDF_LINKS[body]] = fit_body(body, meshes, args.device) | {"source_body": body}
        print(body, {k: v for k, v in links[URDF_LINKS[body]].items()
                     if k not in ("spheres", "fit")}, flush=True)
    result = {
        "schema": "invar-panda-collision-spheres-v1",
        "frame": "each Menagerie body frame (prepare_models.py maps to URDF link frames)",
        "source_sha256": files,
        "fitter": {"curobo": curobo.__version__, "torch": torch.__version__,
                   "trimesh": trimesh.__version__, "fit_type": "MORPHIT",
                   "defaults": "sphere_density 1.0, iterations 200", "seed": SEED,
                   "device": args.device},
        "link0_clip_plane": {"normal": [0, 0, 1], "offset_m": CLEARANCE_M - CUROBO_CLIP_BUFFER_M,
                             "curobo_buffer_m": CUROBO_CLIP_BUFFER_M, "clearance_m": CLEARANCE_M,
                             "table_band_m": TABLE_BAND_M, "target_slack_m": LINK0_SLACK_M},
        "grow_spacing_m": GROW_SPACING_M, "grow_limit_m": GROW_LIMIT_M,
        "surface_patch_radius_m": SURFACE_RADIUS_M,
        "fit_script_sha256": sha(Path(__file__)),
        "links": links,
    }
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
