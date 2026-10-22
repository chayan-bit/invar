"""CPU certificate and broad-phase regressions for the offline proposal."""
import itertools
import json
from pathlib import Path
import tempfile
import zipfile
import time
import unittest
import numpy as np
from panda_hull_cover import generate,certify,load_verified,sha
from verify_panda_hull_cover import first_overlap


class HullTests(unittest.TestCase):
    def fixture(self):
        return {'lo':[-.02]*3,'hi':[.02]*3,
                'points':list(itertools.product([-.02,.02],repeat=3))}

    def test_complete_solid_certificate_rejects_holes_and_shrunken_spheres(self):
        geom=self.fixture();tree,raw,_=generate(geom,.01,time.monotonic()+10)
        self.assertGreater(certify(geom,.01,tree,raw)['minimum_cell_corner_clearance_m'],0)
        for kind in ['missing_node','missing_sphere','shrink','nan','false_discard','trailing']:
            t,r=tree,raw
            if kind=='missing_node':t=t[:-1]
            elif kind=='missing_sphere':r=r[:-32]
            elif kind in ['shrink','nan']:
                spheres=np.frombuffer(r,dtype='<f8').copy().reshape(-1,4)
                spheres[0,3]=0 if kind=='shrink' else np.nan;r=spheres.tobytes()
            elif kind=='false_discard':t=b'X';r=b''
            else:t+=b'X'
            with self.subTest(kind=kind),self.assertRaises(ValueError):certify(geom,.01,t,r)

    def test_generation_is_deterministic(self):
        first=generate(self.fixture(),.01,time.monotonic()+10)
        second=generate(self.fixture(),.01,time.monotonic()+10)
        self.assertEqual(first,second)

    def test_forged_archive_hash_cannot_hide_undercoverage(self):
        root=Path(__file__).resolve().parents[4]
        archive_path=root/'tests/fixtures/archived-evidence/cloud-panda-hull-coverage/10mm.zip'
        with zipfile.ZipFile(archive_path) as archive:
            files={n:archive.read(n) for n in archive.namelist()}
        raw=np.frombuffer(files['0.f64le'],dtype='<f8').copy().reshape(-1,4)
        raw[0,3]=0.;files['0.f64le']=raw.tobytes()
        manifest=json.loads(files['manifest.json'])
        manifest['files']['0.f64le']['sha256']=sha(files['0.f64le'])
        files['manifest.json']=json.dumps(manifest).encode()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'forged.zip'
            with zipfile.ZipFile(path,'w') as archive:
                for name,data in files.items():archive.writestr(name,data)
            with self.assertRaisesRegex(ValueError,'enclose its solid cell'):
                load_verified(path,root/'examples/cells/reach_place/assets/menagerie/franka_emika_panda')

    def test_pair_search_agrees_with_exhaustive_distances(self):
        rng=np.random.default_rng(72491)
        for offset in [0.,.1,1.,5.]:
            a=np.column_stack([rng.normal(size=(40,3))*.1,rng.uniform(.001,.03,40)])
            b=np.column_stack([rng.normal(size=(35,3))*.1+offset,rng.uniform(.001,.03,35)])
            brute=np.linalg.norm(a[:,None,:3]-b[None,:,:3],axis=2)-a[:,None,3]-b[None,:,3]
            self.assertEqual(first_overlap(a,b) is not None,bool(np.any(brute<0)))
        a=np.array([[0.,0.,0.,.5]])
        self.assertIsNone(first_overlap(a,np.array([[1.,0.,0.,.5]])))
        self.assertIsNotNone(first_overlap(a,np.array([[np.nextafter(1.,0.),0.,0.,.5]])))


if __name__=='__main__':unittest.main()
