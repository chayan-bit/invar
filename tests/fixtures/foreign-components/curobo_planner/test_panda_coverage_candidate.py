"""Offline coverage certificate regressions; no GPU or third-party packages."""
import copy
import math
from pathlib import Path
import unittest
import panda_coverage_candidate as candidate


class CoverageTests(unittest.TestCase):
    def test_solid_partition_and_interior_coverage(self):
        lo,hi=[-0.12,-0.02,-0.03],[0.18,0.02,0.03]
        axis,cells=candidate.cover_box(lo,hi,True)
        self.assertGreater(candidate.certify(lo,hi,axis,cells),0)
        # Endpoints alone are insufficient: these small endpoint balls leave a gap.
        endpoints=[{'center':lo,'radius':0.01},{'center':hi,'radius':0.01}]
        middle=[(a+b)/2 for a,b in zip(lo,hi)]
        self.assertTrue(all(math.dist(middle,s['center'])>s['radius'] for s in endpoints))
        for mutation in ['missing','shrink','shift','gap','nan']:
            altered=copy.deepcopy(cells)
            if mutation=='missing': altered.pop()
            elif mutation=='shrink': altered[0]['radius']*=0.9
            elif mutation=='shift': altered[0]['center'][1]+=0.1
            elif mutation=='gap': altered[0]['hi'][axis]-=0.001
            else: altered[0]['radius']=math.nan
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                candidate.certify(lo,hi,axis,altered)

    def test_all_source_primitives_deterministic_and_covered(self):
        root=Path(__file__).resolve().parents[4]
        geometry,hashes=candidate.source_geometry(root/'examples/cells/reach_place/assets/menagerie/franka_emika_panda')
        self.assertEqual(len(geometry),23)
        self.assertEqual(sum(g['kind']=='box' for g in geometry),10)
        self.assertEqual(len({g['body'] for g in geometry}),11)
        self.assertEqual(hashes['panda.xml'],candidate.MODEL_SHA)
        for split,count in [(False,23),(True,69)]:
            result=candidate.candidate(geometry,split)
            self.assertEqual(result,candidate.candidate(geometry,split))
            self.assertEqual(result['sphere_count'],count)
            self.assertLess(result['max_source_point_outside_m'],0)
            for g in result['geometries']:
                candidate.certify(g['lo'],g['hi'],g['axis'],g['cells'])


if __name__=='__main__':
    unittest.main()
