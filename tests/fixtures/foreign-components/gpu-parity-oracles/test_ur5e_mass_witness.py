"""Pinned CPU witness checks; requires the same explicit foreign-library environment."""
from pathlib import Path
import unittest
import numpy as np
import ur5e_mass_witness as witness


class WitnessTests(unittest.TestCase):
    def test_source_quaternion_normalization_and_inertia_axes(self):
        # The UR5e wrist quaternion is non-unit wxyz=(1,0,0,1): +90 degrees about Z.
        expected = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        actual = witness.rotation('1 0 0 1')
        np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
        inertia = np.diag([0.000132134, 0.0000990863, 0.0000990863])
        np.testing.assert_allclose(actual @ inertia @ actual.T,
                                   np.diag([0.0000990863, 0.000132134, 0.0000990863]),
                                   atol=1e-18, rtol=0)
        for text in ['0 0 0 0', 'nan 0 0 1', '1 0 0', 'inf 0 0 1']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                witness.rotation(text)

    def test_residual_rejects_nonfinite_shape_and_detects_double_armature(self):
        identity = np.eye(6)
        self.assertEqual(witness.residual(identity, identity), 0.)
        self.assertGreater(witness.residual(identity + np.eye(6)*0.1, identity), witness.TOLERANCE)
        for matrix in [np.eye(7), np.full((6, 6), np.nan), np.full((6, 6), np.inf)]:
            with self.assertRaises(ValueError):
                witness.residual(identity, matrix)

    def test_source_parser_retains_wrist_orientation(self):
        root = Path(__file__).resolve().parents[4]
        source = root / 'examples/cells/reach_place/assets/menagerie/universal_robots_ur5e/ur5e.xml'
        model = witness.mujoco.MjModel.from_xml_path(str(source))
        bodies = witness.source_bodies(source, model)
        self.assertEqual(len(bodies), 7)
        wrist = next(b for b in bodies if model.body(b[0]).name == 'wrist_3_link')
        np.testing.assert_allclose(np.diag(wrist[3]), [0.0000990863, 0.000132134, 0.0000990863], atol=1e-18, rtol=0)


if __name__ == '__main__':
    unittest.main()
