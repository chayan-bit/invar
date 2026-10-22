#!/usr/bin/env python3
"""Offline, source-body-frame Panda coverage candidates; never emits planner config."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

MODEL_SHA = '96ad67da03710f17f798c9478fd9e9efdf24a3bf8359f05e456dd9fb158ea273'
LICENSE_SHA = 'a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9'
MARGIN = 1e-6  # Existing prepare_models.py geometry margin, not an acceptance change.
STEP = 0.04   # Existing UR5e preparation spacing; proposal only for Panda.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corners(lo, hi):
    return list(itertools.product(*zip(lo, hi)))


def vertices(path):
    raw = path.read_bytes()
    if path.suffix == '.stl':
        if len(raw) < 84:
            raise ValueError('short STL')
        count = struct.unpack_from('<I', raw, 80)[0]
        if count == 0 or len(raw) != 84 + 50*count:
            raise ValueError('invalid binary STL')
        points = [struct.unpack_from('<3f', raw, 84+50*i+12+12*j)
                  for i in range(count) for j in range(3)]
    elif path.suffix == '.obj':
        points = [tuple(map(float, line.split()[1:])) for line in raw.decode().splitlines()
                  if line.startswith('v ')]
    else:
        raise ValueError('unsupported mesh format')
    if not points or any(len(p) != 3 or not all(map(math.isfinite, p)) for p in points):
        raise ValueError('invalid mesh vertices')
    return points


def source_geometry(directory):
    source = directory / 'panda.xml'
    if sha(source) != MODEL_SHA or sha(directory/'LICENSE') != LICENSE_SHA:
        raise ValueError('pinned model/license mismatch')
    root = ET.parse(source).getroot()
    meshes = {m.get('name', Path(m.attrib['file']).stem): m for m in root.findall('./asset/mesh')}
    defaults = {d.attrib['class']: d.find('geom').attrib for d in root.findall('.//default')
                if 'class' in d.attrib and d.find('geom') is not None}
    geometry, hashes = [], {'panda.xml': sha(source), 'LICENSE': sha(directory/'LICENSE')}
    for body in root.findall('.//worldbody//body'):
        for index, geom in enumerate(body.findall('geom')):
            cls = geom.get('class')
            if cls == 'visual':
                continue
            if cls != 'collision' and not cls.startswith('fingertip_pad_collision_'):
                raise ValueError('unexpected source collision class')
            fields = dict(defaults['collision'])
            fields.update(defaults[cls]); fields.update(geom.attrib)
            if any(k in fields for k in ['quat', 'euler', 'axisangle', 'fromto']):
                raise ValueError('explicit geometry transform handling required')
            position = list(map(float, fields.get('pos', '0 0 0').split()))
            if len(position) != 3 or not all(map(math.isfinite, position)):
                raise ValueError('invalid source geometry position')
            if fields['type'] == 'mesh':
                mesh = meshes[fields['mesh']]
                if set(mesh.attrib) - {'name', 'file'}:
                    raise ValueError('explicit mesh scale/reference transform handling required')
                path = directory/'assets'/mesh.attrib['file']
                points = vertices(path)
                hashes['assets/'+mesh.attrib['file']] = sha(path)
            elif fields['type'] == 'box':
                size = list(map(float, fields['size'].split()))
                if len(size) != 3 or not all(math.isfinite(v) and v > 0 for v in size):
                    raise ValueError('invalid box size')
                points = corners([-v for v in size], size)
            else:
                raise ValueError('unsupported collision primitive')
            points = [tuple(v+p for v,p in zip(point, position)) for point in points]
            lo = [min(p[i] for p in points) for i in range(3)]
            hi = [max(p[i] for p in points) for i in range(3)]
            geometry.append({'body': body.attrib['name'], 'source_geom_index': index,
                             'kind': fields['type'], 'lo': lo, 'hi': hi, 'points': points})
    if len(geometry) != 23:
        raise ValueError('unexpected collision primitive count')
    return geometry, hashes


def cover_box(lo, hi, split):
    if len(lo) != 3 or len(hi) != 3 or any(not math.isfinite(a) or not math.isfinite(b) or a >= b for a,b in zip(lo,hi)):
        raise ValueError('invalid bounds')
    axis = max(range(3), key=lambda i: hi[i]-lo[i])
    count = max(1, math.ceil((hi[axis]-lo[axis])/STEP)) if split else 1
    cells = []
    for index in range(count):
        a, b = lo.copy(), hi.copy()
        a[axis] = lo[axis] + (hi[axis]-lo[axis])*index/count
        b[axis] = hi[axis] if index == count-1 else lo[axis] + (hi[axis]-lo[axis])*(index+1)/count
        center = [(x+y)/2 for x,y in zip(a,b)]
        radius = max(math.dist(center, p) for p in corners(a,b)) + MARGIN
        cells.append({'lo': a, 'hi': b, 'center': center, 'radius': radius})
    return axis, cells


def certify(lo, hi, axis, cells):
    # Each cell is convex, and a Euclidean ball is convex: containing every
    # corner contains the entire cell. Exact shared endpoints partition the box.
    if not cells or axis not in range(3):
        raise ValueError('missing coverage cells')
    end = lo[axis]
    clearance = math.inf
    for cell in cells:
        a,b,c,r = (cell[k] for k in ['lo','hi','center','radius'])
        if len(a)!=3 or len(b)!=3 or len(c)!=3 or not all(math.isfinite(x) for x in [*a,*b,*c,r]) or r <= 0:
            raise ValueError('invalid coverage cell')
        if a[axis] != end or b[axis] <= a[axis] or any(a[i]!=lo[i] or b[i]!=hi[i] for i in range(3) if i!=axis):
            raise ValueError('coverage partition gap')
        end = b[axis]
        distance = max(math.dist(c,p) for p in corners(a,b))
        if distance > r:
            raise ValueError('uncovered cell corner')
        clearance = min(clearance,r-distance)
    if end != hi[axis]:
        raise ValueError('incomplete coverage partition')
    return clearance


def candidate(geometry, split):
    records = []
    for geom in geometry:
        axis, cells = cover_box(geom['lo'],geom['hi'],split)
        clearance = certify(geom['lo'],geom['hi'],axis,cells)
        worst = max(min(math.dist(p,c['center'])-c['radius'] for c in cells) for p in geom['points'])
        if worst > 1e-9:  # Original vertex-audit rejection threshold unchanged.
            raise ValueError('source vertex outside candidate')
        records.append({k:v for k,v in geom.items() if k!='points'} | {
            'axis':axis,'cells':cells,'source_point_count':len(geom['points']),
            'worst_source_point_outside_m':worst,'minimum_cell_corner_clearance_m':clearance})
    radii = [c['radius'] for g in records for c in g['cells']]
    return {'sphere_count':len(radii),'max_radius_m':max(radii),
            'sum_sphere_volumes_m3_not_union':sum(4*math.pi*r**3/3 for r in radii),
            'max_source_point_outside_m':max(g['worst_source_point_outside_m'] for g in records),
            'minimum_cell_corner_clearance_m':min(g['minimum_cell_corner_clearance_m'] for g in records),
            'geometries':records}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--panda-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    geometry,hashes=source_geometry(args.panda_directory)
    report={'schema':'panda-source-aabb-coverage-proposal-v1',
            'status':'offline_candidates_only_not_production_approved',
            'frame':'each original Menagerie body local frame; no planner-frame mapping implied',
            'proof':'per-cell ball contains all box corners and hence entire convex cell; cells partition source AABB, containing mesh convex hull or box solid',
            'source_sha256':hashes,'geometry_margin_m':MARGIN,'maximum_axial_cell_length_m':STEP,
            'single_sphere':candidate(geometry,False),'axial_cells':candidate(geometry,True)}
    with args.output.open('x') as handle:
        handle.write(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({name:{k:v for k,v in report[name].items() if k!='geometries'} for name in ['single_sphere','axial_cells']},indent=2))


if __name__=='__main__':
    main()
