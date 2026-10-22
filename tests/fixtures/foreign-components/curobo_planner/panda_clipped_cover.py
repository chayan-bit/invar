#!/usr/bin/env python3
"""Bounded offline experiment: exact rational clipping of source outer hulls.

QHull supplies directions only. Exact support maxima over parsed source points
make every plane conservative; clipping and sphere enclosure use Fraction.
This is a diagnostic fixture, never a production planner configuration.
"""
import argparse
from fractions import Fraction as Q
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import scipy
from scipy.spatial import ConvexHull
from panda_coverage_candidate import MARGIN, source_geometry

DEPTH = 3
MAX_SPHERES = 256
MAX_VERTICES = 512
MAX_BITS = 4096
MAX_SECONDS = 90
SCHEMA = 'panda-exact-clipped-cell-experiment-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Q(0))


def vertices(faces):
    return sorted({v for face in faces for v in face})


def bound(faces, deadline):
    points = vertices(faces)
    if time.monotonic() > deadline:
        raise ValueError('exact clipping 90-second runtime bound exceeded')
    if len(points) > MAX_VERTICES:
        raise ValueError(f'exact clipping vertex bound exceeded: {len(points)} > {MAX_VERTICES}')
    if any(max(v.numerator.bit_length(), v.denominator.bit_length()) > MAX_BITS
           for point in points for v in point):
        raise ValueError('exact clipping rational bit bound exceeded')


def convex_face(points, axis):
    """Exact monotone-chain ordering; projection along a nonzero normal axis."""
    indices = [i for i in range(3) if i != axis]
    points = sorted(set(points), key=lambda p: tuple(p[i] for i in indices))
    def turn(a, b, c):
        i, j = indices
        return (b[i]-a[i])*(c[j]-a[j])-(b[j]-a[j])*(c[i]-a[i])
    chains = []
    for ordered in [points, list(reversed(points))]:
        chain = []
        for point in ordered:
            while len(chain) >= 2 and turn(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        chains.append(chain[:-1])
    return chains[0]+chains[1]


def box(lo, hi):
    points = list(itertools.product(*zip(lo, hi)))
    return [convex_face([p for p in points if p[axis] == value], axis)
            for axis in range(3) for value in [lo[axis], hi[axis]]]


def clip(faces, normal, support, deadline):
    """Closed convex polytope intersection with n.x <= support, exactly.

    Returns None for an empty/lower-dimensional intersection. Callers retain a
    full box for such partition cells; numerical infeasibility never drops volume.
    """
    bound(faces, deadline)
    values = {p: dot(normal, p)-support for p in vertices(faces)}
    if all(v <= 0 for v in values.values()):
        return faces
    if not any(v < 0 for v in values.values()):
        return None
    result, cap = [], set()
    for face in faces:
        polygon = []
        for a, b in zip(face, face[1:]+face[:1]):
            va, vb = values[a], values[b]
            if va <= 0:
                polygon.append(a)
                if va == 0:
                    cap.add(a)
            if (va < 0 < vb) or (vb < 0 < va):
                ratio = va/(va-vb)
                point = tuple(x+ratio*(y-x) for x, y in zip(a, b))
                if dot(normal, point) != support:
                    raise ValueError('nonexact cap intersection')
                polygon.append(point); cap.add(point)
        # Convex polygons may have redundant collinear vertices; retaining them
        # is harmless for enclosure, but discard zero-area face fragments.
        polygon = list(dict.fromkeys(polygon))
        if len(polygon) >= 3:
            result.append(polygon)
    axis = max(range(3), key=lambda i: abs(normal[i]))
    cap = convex_face(cap, axis)
    if len(cap) < 3:
        raise ValueError('clipping failed to close a full-dimensional polytope')
    result.append(cap)
    if any(dot(normal, p) > support for p in vertices(result)):
        raise ValueError('clip retained an outside vertex')
    bound(result, deadline)
    return result


def outer_hull(geom, deadline):
    points = sorted(set(tuple(Q(float(v)) for v in p) for p in geom['points']))
    lo, hi = [tuple(Q(float(v)) for v in geom[k]) for k in ['lo', 'hi']]
    faces = box(lo, hi)
    hull = ConvexHull(np.asarray(points, dtype=np.float64))
    normals = set()
    for triangle, equation in zip(hull.simplices, hull.equations, strict=True):
        a,b,c = [points[int(i)] for i in triangle]
        u,v = [tuple(y-x for x,y in zip(a,p)) for p in [b,c]]
        normal = (u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
        if not any(normal):
            raise ValueError('degenerate QHull triangle')
        # Orientation affects tightness only. Both directions remain source-
        # conservative because the offset is an exact maximum over all points.
        if sum(float(n)*float(e) for n,e in zip(normal,equation[:3])) < 0:
            normal = tuple(-n for n in normal)
        denominator = math.lcm(*(n.denominator for n in normal))
        integers = tuple(int(n*denominator) for n in normal)
        divisor = math.gcd(*integers)
        normals.add(tuple(Q(n//divisor) for n in integers))
    normals = sorted(normals)
    for normal in normals:
        # Exact support is independent of QHull's rounded plane offsets and
        # always contains every source point, hence their complete convex hull.
        support = max(dot(normal, p) for p in points)
        faces = clip(faces, normal, support, deadline)
        if faces is None:
            raise ValueError('source outer hull lost full dimensionality')
    return faces, lo, hi, len(normals)


def leaf_cells(lo, hi, depth=DEPTH):
    if depth == 0:
        yield lo, hi
        return
    axis = max(range(3), key=lambda i: hi[i]-lo[i])
    middle = (lo[axis]+hi[axis])/2
    lower, upper = list(hi), list(lo)
    lower[axis] = upper[axis] = middle
    yield from leaf_cells(lo, tuple(lower), depth-1)
    yield from leaf_cells(tuple(upper), hi, depth-1)


def cell_polytope(faces, lo, hi, deadline):
    clipped = faces
    for axis in range(3):
        for sign, value in [(-1, lo[axis]), (1, hi[axis])]:
            normal = tuple(Q(sign if i == axis else 0) for i in range(3))
            clipped = clip(clipped, normal, sign*value, deadline)
            if clipped is None:
                return box(lo, hi), True
    return clipped, False


def sphere(points):
    approximate = np.asarray(points, dtype=np.float64)
    seeds = [(approximate.min(axis=0)+approximate.max(axis=0))/2,
             approximate.mean(axis=0)]
    center = min(seeds, key=lambda c: np.linalg.norm(approximate-c, axis=1).max())
    exact_center = tuple(Q(float(v)) for v in center)
    squared = max(dot(tuple(v-c for v, c in zip(p, exact_center)),
                      tuple(v-c for v, c in zip(p, exact_center))) for p in points)
    radius = math.sqrt(float(squared))+MARGIN
    for _ in range(8):
        if Q(radius) > Q(MARGIN) and (Q(radius)-Q(MARGIN))**2 >= squared:
            return [*map(float, center), radius]
        radius = math.nextafter(radius, math.inf)
    raise ValueError('could not round sphere radius outward within bound')


def check_sphere(points, ball):
    if len(ball) != 4 or not all(math.isfinite(v) for v in ball) or ball[3] <= MARGIN:
        raise ValueError('invalid sphere')
    center = tuple(Q(float(v)) for v in ball[:3])
    radius = Q(float(ball[3]))-Q(MARGIN)
    squared = max(dot(tuple(v-c for v, c in zip(p, center)),
                      tuple(v-c for v, c in zip(p, center))) for p in points)
    if squared > radius**2:
        raise ValueError('sphere fails exact complete-cell vertex enclosure')
    return float(radius**2-squared)


def process(geometry, deadline, records=None, progress=None):
    output, stats = [], []
    if records is not None and len(records) != len(geometry):
        raise ValueError('missing geometry')
    for index, geom in enumerate(geometry):
        if progress is not None:
            progress({'geometry':index,'body':geom['body'],'source_geom_index':geom['source_geom_index']})
        faces, lo, hi, plane_count = outer_hull(geom, deadline)
        cells = list(leaf_cells(lo, hi)); balls = []; fallbacks = 0; minimum = math.inf
        if records is not None:
            record = records[index]
            if any(record[k] != geom[k] for k in ['body', 'source_geom_index', 'lo', 'hi']):
                raise ValueError('geometry identity mismatch')
            if len(record['spheres']) != len(cells):
                raise ValueError('missing partition cell sphere')
        for cell_index, (a, b) in enumerate(cells):
            poly, fallback = cell_polytope(faces, a, b, deadline)
            points = vertices(poly)
            ball = sphere(points) if records is None else record['spheres'][cell_index]
            minimum = min(minimum, check_sphere(points, ball))
            balls.append(ball); fallbacks += fallback
        output.append({k: geom[k] for k in ['body', 'source_geom_index', 'lo', 'hi']} |
                      {'spheres': balls})
        stats.append({'body': geom['body'], 'source_geom_index': geom['source_geom_index'],
                      'support_directions': plane_count, 'outer_hull_vertices': len(vertices(faces)),
                      'spheres': len(balls), 'empty_or_degenerate_box_fallbacks': fallbacks,
                      'minimum_exact_squared_slack_after_1um_margin_m2': minimum})
    if sum(len(r['spheres']) for r in output) > MAX_SPHERES:
        raise ValueError('aggregate sphere bound exceeded')
    return output, stats


def build(directory, progress=None):
    if (np.__version__, scipy.__version__) != ('2.5.3', '1.17.0'):
        raise ValueError('pinned numerical packages required')
    start = time.monotonic(); geometry, hashes = source_geometry(directory)
    records, stats = process(geometry, start+MAX_SECONDS, progress=progress)
    return {'schema': SCHEMA, 'status': 'offline_candidate_only', 'source_sha256': hashes,
            'source_parser_sha256': sha(Path(__file__).with_name('panda_coverage_candidate.py').read_bytes()),
            'producer_sha256': sha(Path(__file__).read_bytes()), 'partition_depth': DEPTH,
            'geometry_margin_m': MARGIN, 'numpy': np.__version__, 'scipy': scipy.__version__,
            'geometries': records, 'stats': stats}, time.monotonic()-start


def load_verified(path, directory):
    if path.stat().st_size > 2*1024*1024:
        raise ValueError('candidate byte bound exceeded')
    manifest = json.loads(path.read_bytes())
    geometry, hashes = source_geometry(directory)
    if (manifest['schema'], manifest['partition_depth'], manifest['geometry_margin_m']) != (SCHEMA, DEPTH, MARGIN):
        raise ValueError('candidate contract mismatch')
    if manifest['source_sha256'] != hashes:
        raise ValueError('source identity mismatch')
    if manifest['source_parser_sha256'] != sha(Path(__file__).with_name('panda_coverage_candidate.py').read_bytes()):
        raise ValueError('source parser identity mismatch')
    if manifest['producer_sha256'] != sha(Path(__file__).read_bytes()):
        raise ValueError('producer identity mismatch')
    records, stats = process(geometry, time.monotonic()+MAX_SECONDS, manifest['geometries'])
    if stats != manifest['stats']:
        raise ValueError('replayed certificate summary mismatch')
    return manifest, [(g, np.asarray(r['spheres'], dtype=np.float64))
                      for g, r in zip(geometry, records, strict=True)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--panda-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    def report(item):
        print(json.dumps(item | {'wall_seconds':time.monotonic()-started}),file=sys.stderr,flush=True)
    try:
        result, elapsed = build(args.panda_directory, progress=report)
    except Exception as error:
        report({'status':'bounded_preparation_failed','error':str(error),
                'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        raise
    encoded = (json.dumps(result, indent=2, sort_keys=True)+'\n').encode()
    if len(encoded) > 2*1024*1024:
        raise ValueError('candidate output byte bound exceeded')
    with args.output.open('xb') as handle:
        handle.write(encoded)
    print(json.dumps({'sha256': sha(encoded), 'bytes': len(encoded), 'wall_seconds': elapsed,
                      'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      'spheres': sum(len(g['spheres']) for g in result['geometries'])}))
