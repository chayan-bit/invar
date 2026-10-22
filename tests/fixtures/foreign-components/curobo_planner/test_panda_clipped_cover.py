"""Exact clipping, complete-solid enclosure and synthetic collision controls."""
from fractions import Fraction as Q
import itertools
import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
import mujoco
from panda_clipped_cover import (box, clip, vertices, sphere, check_sphere,
                                leaf_cells, cell_polytope, process, load_verified)
from verify_panda_hull_cover import first_overlap


class ClippedTests(unittest.TestCase):
    def cube(self):
        lo, hi = (Q(0),)*3, (Q(1),)*3
        return lo, hi, box(lo, hi)

    def test_exact_oblique_clip_contains_complete_tetrahedron(self):
        lo, hi, faces = self.cube()
        tetra = clip(faces, (Q(1),)*3, Q(1), time.monotonic()+5)
        expected = {(Q(0),)*3, (Q(1),Q(0),Q(0)),
                    (Q(0),Q(1),Q(0)), (Q(0),Q(0),Q(1))}
        self.assertEqual(set(vertices(tetra)), expected)
        interior = tuple(sum(p[i] for p in expected)/4 for i in range(3))
        volume = Q(0)
        for face in tetra:
            a = tuple(v-c for v,c in zip(face[0],interior))
            for j in range(1,len(face)-1):
                b,d = [tuple(v-c for v,c in zip(face[k],interior)) for k in [j,j+1]]
                cross = (b[1]*d[2]-b[2]*d[1],b[2]*d[0]-b[0]*d[2],b[0]*d[1]-b[1]*d[0])
                volume += abs(sum(x*y for x,y in zip(a,cross)))/6
        self.assertEqual(volume,Q(1,6))
        ball = sphere(vertices(tetra))
        check_sphere(vertices(tetra), ball)
        # The excluded box corner is not part of the source-solid requirement.
        self.assertGreater(np.linalg.norm(np.ones(3)-ball[:3]), ball[3])
        with self.assertRaisesRegex(ValueError, 'complete-cell'):
            check_sphere(vertices(tetra), [*ball[:3], ball[3]/2])

    def test_exact_thin_slice_and_closed_partition(self):
        lo, hi, faces = self.cube()
        width = Q(1, 2**60)
        thin = clip(faces, (Q(1),Q(0),Q(0)), width, time.monotonic()+5)
        self.assertEqual(max(p[0] for p in vertices(thin)), width)
        cells = list(leaf_cells(lo, hi))
        self.assertEqual(len(cells), 8)
        self.assertEqual(sum(np.prod([b-a for a,b in zip(l,h)]) for l,h in cells), Q(1))
        for a,b in cells:
            poly, fallback = cell_polytope(faces, a, b, time.monotonic()+5)
            self.assertFalse(fallback)
            self.assertEqual(set(vertices(poly)), set(itertools.product(*zip(a,b))))

    def test_empty_and_tangent_cells_retain_conservative_box(self):
        _, _, faces = self.cube()
        for lo,hi in [((Q(2),)*3,(Q(3),)*3),
                      ((Q(1),Q(0),Q(0)),(Q(2),Q(1),Q(1)))]:
            poly, fallback = cell_polytope(faces, lo, hi, time.monotonic()+5)
            self.assertTrue(fallback)
            self.assertEqual(set(vertices(poly)),set(itertools.product(*zip(lo,hi))))

    def test_missing_nan_and_shrunken_cells_rejected(self):
        geom = {'body':'cube','source_geom_index':0,'lo':[0.]*3,'hi':[1.]*3,
                'points':list(itertools.product([0.,1.],repeat=3))}
        records,_ = process([geom],time.monotonic()+5)
        for kind in ['missing','nan','shrink','reorder']:
            mutated = dict(records[0]); mutated['spheres']=[list(s) for s in mutated['spheres']]
            if kind=='missing':mutated['spheres'].pop()
            elif kind=='nan':mutated['spheres'][0][0]=float('nan')
            elif kind=='shrink':mutated['spheres'][0][3]/=2
            else:mutated['spheres'].reverse()
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                process([geom],time.monotonic()+5,[mutated])

    def test_penetration_clearance_and_false_positive_controls(self):
        result = collision_controls()
        self.assertEqual([r['sphere_overlap'] for r in result], [True,True,False])
        self.assertLess(result[0]['source_distance_m'],0)
        self.assertGreater(result[1]['source_distance_m'],0)
        self.assertGreater(result[2]['source_distance_m'],0)

    def test_real_candidate_rejects_identity_and_forged_radius(self):
        root=Path(__file__).resolve().parents[4]
        path=root/'tests/fixtures/archived-evidence/cloud-panda-clipped-coverage/candidate.json'
        original=json.loads(path.read_bytes())
        directory=root/'examples/cells/reach_place/assets/menagerie/franka_emika_panda'
        with tempfile.TemporaryDirectory() as temporary:
            corrupted=Path(temporary)/'forged.json'
            changed=json.loads(json.dumps(original))
            changed['source_sha256']['panda.xml']='0'*64
            corrupted.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'source identity'):
                load_verified(corrupted,directory)
            changed=json.loads(json.dumps(original))
            changed['geometries'][0]['spheres'][0][3]/=2
            corrupted.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'complete-cell'):
                load_verified(corrupted,directory)


def collision_controls():
    """Artificial cube pairs; exercises a real penetration and retained false positive.

    These controls are not robot poses and do not change source assets or ignores.
    """
    lo,hi = (Q(-1,2),)*3,(Q(1,2),)*3
    faces=box(lo,hi); balls=[]
    for a,b in leaf_cells(lo,hi):
        poly,_=cell_polytope(faces,a,b,time.monotonic()+5)
        ball=sphere(vertices(poly));check_sphere(vertices(poly),ball);balls.append(ball)
    balls=np.asarray(balls)
    rows=[]
    for name,shift in [('penetrating',[.99,.99,.99]),
                       ('separated_but_conservative_overlap',[1.01,0.,0.]),
                       ('far_clear',[3.,0.,0.])]:
        xml='<mujoco><worldbody><geom type="box" size=".5 .5 .5"/><geom type="box" size=".5 .5 .5" pos="'+ ' '.join(map(str,shift))+'"/></worldbody></mujoco>'
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        distance=float(mujoco.mj_geomDistance(model,data,0,1,10.,None))
        translated=balls.copy();translated[:,:3]+=shift
        gap=first_overlap(balls,translated)
        rows.append({'control':name,'translation_m':shift,'source_distance_m':distance,
                     'sphere_overlap':gap is not None,'first_sphere_overlap_witness_m':gap})
    return rows


if __name__=='__main__':unittest.main()
