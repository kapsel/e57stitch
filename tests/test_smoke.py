"""Fast smoke tests that do not depend on the sample scans.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from e57stitch.e57io import StreamSpec, write_e57_stream, _voxel_subsample
from e57stitch.cleanup import normalize_region, removal_mask
from e57stitch.features import estimate_gravity
from e57stitch.register import (
    cycle_consistency,
    invert,
    rotation_angle_deg,
    score_transform,
    snap_to_up,
    solve_global_poses,
)


class TestVoxel(unittest.TestCase):
    def test_one_point_per_cell(self):
        pts = np.array(
            [
                [0.01, 0.01, 0.01],
                [0.02, 0.02, 0.02],
                [0.06, 0.01, 0.01],
                [1.01, 0.01, 0.01],
            ],
            dtype=np.float32,
        )
        out, keep = _voxel_subsample(pts, 0.05)
        self.assertEqual(len(out), 3)
        self.assertEqual(len(keep), 3)
        np.testing.assert_allclose(out, pts[keep])

    def test_empty(self):
        out, keep = _voxel_subsample(np.empty((0, 3), np.float32), 0.1)
        self.assertEqual(len(out), 0)


class TestE57Write(unittest.TestCase):
    def test_roundtrip(self):
        import pye57

        rng = np.random.default_rng(3)
        n = 50_000
        xyz = rng.normal(0, 5, (n, 3)).astype(np.float32)
        rgb = rng.integers(0, 255, (n, 3), dtype=np.uint8)
        inten = rng.random(n).astype(np.float32)

        def chunks():
            for start in range(0, n, 7_777):
                stop = min(start + 7_777, n)
                yield {
                    "points": xyz[start:stop],
                    "colors": rgb[start:stop],
                    "intensity": inten[start:stop],
                }

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.e57")
            write_e57_stream(
                path,
                [StreamSpec("merged", chunks, has_colors=True, has_intensity=True)],
                chunk=7_777,
            )
            reader = pye57.E57(path)
            self.assertEqual(reader.scan_count, 1)
            self.assertEqual(reader.get_header(0).point_count, n)
            data = reader.read_scan_raw(0)
            got = np.stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]], axis=1)
            np.testing.assert_array_equal(got, xyz)
            got_rgb = np.stack([data["colorRed"], data["colorGreen"], data["colorBlue"]], axis=1)
            np.testing.assert_array_equal(got_rgb, rgb)
            np.testing.assert_allclose(np.asarray(data["intensity"]), inten)


class TestScoring(unittest.TestCase):
    def test_structure_shift_is_penalised(self):
        rng = np.random.default_rng(1)
        cloud = rng.normal(0, 2, (5000, 3)).astype(np.float32)
        # a mostly vertical "wall" plus a floor
        wall = np.column_stack(
            [rng.uniform(-3, 3, 4000), np.zeros(4000), rng.uniform(0, 4, 4000)]
        ).astype(np.float32)
        pts = np.vstack([cloud, wall])
        structure = wall.copy()

        same = score_transform(
            pts, pts, np.eye(4), 0.05, a_structure=structure, b_structure=structure
        )
        self.assertGreater(same["score"], 0.95)

        # shift perpendicular to the wall (along Y); a shift *along* a
        # featureless wall is deliberately not penalised by a point-overlap
        # metric - that is what edge clusters are for.
        shifted = np.eye(4)
        shifted[1, 3] = 1.0
        moved = score_transform(
            pts, pts, shifted, 0.05, a_structure=structure, b_structure=structure
        )
        self.assertLess(moved["score"], 0.2)


class TestGravity(unittest.TestCase):
    def _room(self):
        rng = np.random.default_rng(4)
        floor = np.column_stack([rng.uniform(-5, 5, 20000), rng.uniform(-5, 5, 20000), np.zeros(20000)])
        ceil = floor.copy()
        ceil[:, 2] = 3.0
        walls = []
        for x in (-5, 5):
            walls.append(np.column_stack([np.full(5000, x), rng.uniform(-5, 5, 5000), rng.uniform(0, 3, 5000)]))
        for y in (-5, 5):
            walls.append(np.column_stack([rng.uniform(-5, 5, 5000), np.full(5000, y), rng.uniform(0, 3, 5000)]))
        return np.vstack([floor, ceil, *walls]).astype(np.float32)

    def test_level_room(self):
        g = estimate_gravity(self._room())
        self.assertEqual(g["mode"], "level")
        self.assertAlmostEqual(abs(float(g["up"][2])), 1.0, places=2)

    def test_sideways_room(self):
        from scipy.spatial.transform import Rotation

        R = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        rotated = self._room() @ R.T
        g = estimate_gravity(rotated)
        self.assertEqual(g["mode"], "sideways")
        self.assertLess(abs(float(g["up"][2])), 0.2)
        # The detected axis should map back to the room's true vertical.
        self.assertGreater(abs(float((R @ g["up"])[2])), 0.9)


class TestCycleConsistency(unittest.TestCase):
    def _pair(self, a, b, T):
        return {"a": a, "b": b, "transform": np.asarray(T).tolist(), "accepted": True, "score": 0.5}

    def test_consistent_triangle(self):
        T_ab = np.eye(4)
        T_ab[:3, 3] = [1.0, 0.0, 0.0]
        T_bc = np.eye(4)
        T_bc[:3, 3] = [0.0, 2.0, 0.0]
        T_ac = T_bc @ T_ab
        pairs = [self._pair("a", "b", T_ab), self._pair("b", "c", T_bc), self._pair("a", "c", T_ac)]
        cycles = cycle_consistency(pairs)
        for key, value in cycles.items():
            self.assertLess(value["min_trans"], 1e-6, key)
            self.assertLess(value["min_rot"], 1e-6, key)

    def test_bad_link_is_exposed(self):
        good = np.eye(4)
        good[:3, 3] = [1.0, 0.0, 0.0]
        bad = np.eye(4)
        bad[:3, 3] = [9.0, 0.0, 0.0]
        pairs = [
            self._pair("a", "b", np.eye(4)),
            self._pair("b", "c", np.eye(4)),
            self._pair("a", "c", bad),
        ]
        cycles = cycle_consistency(pairs)
        self.assertGreater(cycles["a|c"]["min_trans"], 1.0)

    def test_snap_to_up(self):
        T = np.eye(4)
        T[:3, :3] = np.array(
            [[0.9, 0.0, 0.1], [0.0, 1.0, 0.0], [-0.1, 0.0, 0.9]], dtype=float
        )
        T[:3, :3] /= np.linalg.norm(T[:3, :3], axis=0, keepdims=True)
        up = np.array([0.0, 0.0, 1.0])
        snapped = snap_to_up(T, up, up, 1.0)
        mapped = snapped[:3, :3] @ up
        self.assertGreater(abs(float(mapped @ up)), 0.999)
        self.assertLess(rotation_angle_deg(snapped[:3, :3].T @ T[:3, :3]), 90.0)


class TestCleanup(unittest.TestCase):
    def test_box_and_sphere(self):
        pts = np.array(
            [[0, 0, 0], [1, 0, 0], [5, 5, 5], [0, 0, 2]], dtype=np.float64
        )
        box = normalize_region({"kind": "box", "center": [0, 0, 0], "size": [1.1, 1.1, 1.1]})
        mask = removal_mask(pts, [box])
        self.assertEqual(mask.tolist(), [True, True, False, False])
        sphere = normalize_region({"kind": "sphere", "center": [0, 0, 2], "radius": 0.5})
        mask = removal_mask(pts, [sphere])
        self.assertEqual(mask.tolist(), [False, False, False, True])

    def test_clip_box_drops_ceiling(self):
        # floor at z=0, wall point at 1.5, ceiling at 2.6; clip keeps z <= 2.45
        pts = np.array(
            [[0, 0, 0.0], [3, -2, 1.5], [1, 1, 2.6], [50, 50, 1.0]], dtype=np.float64
        )
        clip = normalize_region(
            {
                "clip": True,
                "op": "delete",  # forced to keep
                "scans": ["s1"],  # forced to all scans
                "center": [0, 0, -2.0],
                "size": [1e4, 1e4, 4.45],
            }
        )
        self.assertEqual((clip["op"], clip["kind"], clip["scans"]), ("keep", "box", None))
        mask = removal_mask(pts, [clip], scan="s2")
        self.assertEqual(mask.tolist(), [False, False, True, False])
        # combined with a delete box, both apply
        box = normalize_region({"center": [3, -2, 1.5], "size": [0.2, 0.2, 0.2]})
        mask = removal_mask(pts, [clip, box])
        self.assertEqual(mask.tolist(), [False, True, True, False])

    def test_scope_and_disable(self):
        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        region = normalize_region(
            {"kind": "box", "center": [0, 0, 0], "size": [2, 2, 2], "scans": ["s1"]}
        )
        self.assertTrue(removal_mask(pts, [region], scan="s1").all())
        self.assertFalse(removal_mask(pts, [region], scan="s2").any())
        region["enabled"] = False
        self.assertFalse(removal_mask(pts, [region], scan="s1").any())


class TestPoseGraph(unittest.TestCase):
    def _pair(self, a, b, T):
        return {"a": a, "b": b, "transform": np.asarray(T).tolist(), "accepted": True, "score": 0.5}

    def test_chain(self):
        T_ab = np.eye(4)
        T_ab[:3, 3] = [1.0, 0.0, 0.0]
        T_bc = np.eye(4)
        T_bc[:3, 3] = [0.0, 2.0, 0.0]
        T_ac = T_bc @ T_ab  # Open3D convention: T_ac = T_bc * T_ab
        pairs = [self._pair("a", "b", T_ab), self._pair("b", "c", T_bc), self._pair("a", "c", T_ac)]
        solution = solve_global_poses(["a", "b", "c"], pairs, reference="a", optimize=False)
        poses = solution["poses"]
        np.testing.assert_allclose(poses["a"], np.eye(4), atol=1e-9)
        np.testing.assert_allclose(poses["b"], invert(T_ab), atol=1e-6)
        np.testing.assert_allclose(poses["c"], invert(T_ac), atol=1e-6)
        self.assertEqual(solution["groups"][0]["members"], ["a", "b", "c"])

    def test_tree_from_leaf_reference(self):
        # Regression: when the reference is a leaf of the spanning tree, an
        # earlier directed BFS version left the rest of the group unposed.
        T_ab = np.eye(4); T_ab[0, 3] = 2.0
        T_bc = np.eye(4); T_bc[1, 3] = 3.0
        sol = solve_global_poses(
            ["a", "b", "c"],
            [self._pair("a", "b", T_ab), self._pair("b", "c", T_bc)],
            reference="c",
            optimize=False,
        )
        self.assertEqual(set(sol["poses"]), {"a", "b", "c"})
        self.assertEqual(sol["groups"][0]["edges"], 2)
        np.testing.assert_allclose(sol["poses"]["c"], np.eye(4), atol=1e-9)

    def test_disconnected_groups(self):
        T = np.eye(4)
        T[0, 3] = 3.0
        sol = solve_global_poses(
            ["a", "b", "x"], [self._pair("a", "b", T)], reference="a", optimize=False
        )
        self.assertEqual(len(sol["groups"]), 2)
        self.assertEqual(sol["groups"][0]["members"], ["a", "b"])
        self.assertTrue(sol["groups"][1]["unconnected"])
        self.assertIn("x", sol["unconnected"])


if __name__ == "__main__":
    unittest.main()
