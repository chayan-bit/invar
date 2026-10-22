#!/usr/bin/env python3
"""Show counterexamples to vendor-sphere coverage of accepted Panda meshes."""

import argparse
import hashlib
import math
from pathlib import Path
import struct

import pinocchio as pin
import yaml

from compare_model_fk import (EXPECTED_PINOCCHIO, ROOT, check_files, delta,
                              model_pose)

MESHES = {
    "link0": ("link0.stl", "dfc6d94330de8ddb005b311bfdba9f3b8e1aa7c256b71592ee7ff32cb9a9a5aa"),
    "link6": ("link6.stl", "20b768e99a0e0440b5754dcca108016434e57937cc356acd9c352ccd3cb27f77"),
    "hand": ("hand.stl", "94493e94f30fe940f2c8ca2f155c3bbe67bbff406d3edf5e261670d2f0f6e2ed"),
}
DERIVED_SHA256 = "eeecd06212a1ed160807243a1046ac0a1ce07d7e49370a4a17d181bed5a6f351"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stl_vertices(path):
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"short STL: {path}")
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * count or count == 0:
        raise ValueError(f"invalid binary STL: {path}")
    for i in range(count):
        values = struct.unpack_from("<9f", data, 84 + i*50 + 12)
        for j in range(0, 9, 3):
            point = values[j:j+3]
            if not all(math.isfinite(x) for x in point):
                raise ValueError(f"nonfinite STL vertex: {path}")
            yield point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curobo-root", type=Path, required=True)
    parser.add_argument("--menagerie-root", type=Path, required=True)
    parser.add_argument("--panda-urdf", type=Path, required=True)
    parser.add_argument("--goldens", type=Path, required=True)
    args = parser.parse_args()
    if pin.__version__ != EXPECTED_PINOCCHIO:
        raise ValueError("Pinocchio version mismatch")
    import json
    manifest = json.loads((ROOT / "model_sources.json").read_text())
    check_files(args.curobo_root, manifest["curobo_franka"])
    entry = manifest["menagerie"]["models"]["panda"]
    model_path = args.menagerie_root / entry["path"]
    if (sha(model_path) != entry["sha256"] or
            sha(args.menagerie_root / entry["license_path"]) != entry["license_sha256"] or
            sha(args.panda_urdf) != DERIVED_SHA256):
        raise ValueError("Panda model or license digest mismatch")
    golden = next(a for a in json.loads(args.goldens.read_text())["arms"]
                  if a["arm"] == "panda")
    source = pin.buildModelFromMJCF(str(model_path))
    planner = pin.buildModelFromUrdf(str(args.panda_urdf))
    names = [f"panda_joint{i}" for i in range(1, 8)]
    spheres = yaml.safe_load((args.curobo_root /
        "curobo/content/configs/robot/franka.yml").read_text())["robot_cfg"]["kinematics"]["collision_spheres"]
    failed = False
    for link, (mesh, digest) in MESHES.items():
        path = args.menagerie_root / "franka_emika_panda/assets" / mesh
        if sha(path) != digest:
            raise ValueError(f"Menagerie collision mesh digest mismatch: {mesh}")
        for vector in golden["vectors"]:
            a = model_pose(source, vector["q_rad"], golden["joint_names"], "link0", link)
            b = model_pose(planner, vector["q_rad"], names, "panda_link0", "panda_" + link)
            if max(delta(a, b)) > 1e-12:
                raise ValueError(f"Panda collision frame mismatch: {link}")
        radii = spheres["panda_" + link]
        points = []
        for vertex in stl_vertices(path):
            distance, index = min((math.dist(vertex, sphere["center"]) - sphere["radius"], i)
                                  for i, sphere in enumerate(radii))
            points.append((distance, index, vertex))
        uncovered = sum(distance > 1e-9 for distance, _, _ in points)
        worst, index, vertex = max(points)
        print(f"{link}: vertices={len(points)} uncovered={uncovered} "
              f"worst_outside_m={worst:.9g} vertex_m={vertex} "
              f"nearest_sphere={index}:{radii[index]}")
        failed |= uncovered > 0
    if failed:
        raise SystemExit("Panda vendor spheres under-cover accepted mesh vertices")


if __name__ == "__main__":
    main()
