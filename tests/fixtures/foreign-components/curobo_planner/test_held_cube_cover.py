"""The planner child's held-cube cover (decision D2): every face point is covered, and
the contact variant uncovers only the slab on the face resting toward the table.

Run: python3 -m unittest discover -s tests/fixtures/foreign-components/curobo_planner \
  -p test_held_cube_cover.py"""
import itertools
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]
                       / "examples/cells/reach_place/components/curobo_planner"))
import planner_server as child  # noqa: E402

EDGE = 0.05
HALF = EDGE / 2
STEPS = 41
TABLE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def face_points(axes=range(3)):
    grid = [-HALF + i * EDGE / (STEPS - 1) for i in range(STEPS)]
    for axis, sign in itertools.product(axes, (-1.0, 1.0)):
        for a, b in itertools.product(grid, grid):
            point = [a, b]
            point.insert(axis, sign * HALF)
            yield point


def covered(point, spheres):
    return any(math.dist(point, s[:3]) <= s[3] for s in spheres)


class HeldCubeCoverTests(unittest.TestCase):
    def test_full_cover_covers_every_face_with_bounded_bulge(self):
        spheres = child.cube_cover(EDGE)
        self.assertEqual(len(spheres), child.HELD_COVER_SPHERES)
        self.assertTrue(all(covered(p, spheres) for p in face_points()))
        bulge = max(max(abs(c) for c in s[:3]) + s[3] for s in spheres) - HALF
        self.assertLess(bulge, 0.0046)

    def test_contact_cover_clears_the_table_and_uncovers_only_a_5mm_slab(self):
        yawed = (0.4, 0.0, HALF, 0.0, 0.0, math.sin(0.3), math.cos(0.3))
        contact = child.contact_face(yawed, TABLE)
        self.assertEqual(contact, (2, -1.0))
        spheres = child.cube_cover(EDGE, contact)
        lowest = min(s[2] - s[3] for s in spheres)
        self.assertAlmostEqual(lowest, -HALF + child.CONTACT_CLEARANCE_M, places=12)
        side = [p for p in face_points(range(2)) if not covered(p, spheres)]
        self.assertTrue(side)
        self.assertLessEqual(max(p[2] for p in side), -HALF + 0.0051)
        top = [p for p in face_points([2]) if p[2] > 0]
        self.assertTrue(all(covered(p, spheres) for p in top))

    def test_contact_face_is_the_face_pointing_most_nearly_down(self):
        flipped = (0.4, 0.0, HALF, 1.0, 0.0, 0.0, 0.0)
        self.assertEqual(child.contact_face(flipped, TABLE), (2, 1.0))
        # Rolled 70 degrees about x: the cube's -y face now points most nearly down.
        half = math.radians(35)
        rolled = (0.0, 0.0, 0.0, math.sin(half), 0.0, 0.0, math.cos(half))
        self.assertEqual(child.contact_face(rolled, TABLE), (1, -1.0))


if __name__ == "__main__":
    unittest.main()
