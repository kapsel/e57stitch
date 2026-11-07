"""Tilt lock: 4-DoF ICP for IMU-levelled scans."""

from __future__ import annotations

import math
import unittest

import numpy as np

from e57stitch.register import estimate_normals, icp_level, level_transform, make_cloud, tilt_deg


def yaw_pose(deg: float, t) -> np.ndarray:
    a = math.radians(deg)
    T = np.eye(4)
    T[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    T[:3, 3] = t
    return T


def room(rng: np.random.Generator) -> np.ndarray:
    def plane(n, a, b, c, fix):
        u, v = rng.uniform(0, 1, (2, n))
        p = np.zeros((n, 3))
        i, j = [k for k in range(3) if k != fix]
        p[:, i], p[:, j], p[:, fix] = u * a, v * b, c
        return p

    return np.vstack([
        plane(3000, 8, 5, 0, 2), plane(3000, 8, 5, 3, 2), plane(2000, 5, 3, 0, 0), plane(2000, 5, 3, 8, 0),
        plane(2000, 8, 3, 0, 1), plane(2000, 8, 3, 5, 1), plane(1000, 1, 1, 1.0, 2) + [3, 2, 0],
    ])


class TiltLock(unittest.TestCase):
    def test_level_transform_keeps_yaw_and_translation(self):
        T = yaw_pose(30, [1, 2, 3])
        tilt = np.eye(4)
        tilt[:3, :3] = [[1, 0, 0], [0, math.cos(0.02), -math.sin(0.02)], [0, math.sin(0.02), math.cos(0.02)]]
        L = level_transform(T @ tilt)
        self.assertLess(tilt_deg(L), 1e-9)
        self.assertLess(np.abs(L - T).max(), 1e-3)

    def test_icp_level_recovers_pose_without_tilt(self):
        rng = np.random.default_rng(1)
        dst_pts = room(rng)
        truth = yaw_pose(7, [0.3, -0.2, 0.12])
        src = (dst_pts - truth[:3, 3]) @ truth[:3, :3] + rng.normal(0, 0.003, dst_pts.shape)
        dst = make_cloud(dst_pts)
        estimate_normals(dst, 0.1)
        T, fit, _ = icp_level(src, np.asarray(dst.points), np.asarray(dst.normals),
                              yaw_pose(5, [0.25, -0.1, 0.05]), 0.3, max_iteration=60)
        err = np.linalg.inv(truth) @ T
        self.assertLess(np.linalg.norm(err[:3, 3]), 0.005)
        self.assertLess(tilt_deg(T), 1e-9)
        self.assertGreater(fit, 0.9)


if __name__ == "__main__":
    unittest.main()


class SnapToModel(unittest.TestCase):
    def setUp(self):
        import tempfile
        rng = np.random.default_rng(2)
        self.tmp = tempfile.TemporaryDirectory()
        world = room(rng)
        self.truth = yaw_pose(20, [2.0, 1.5, 0.0])
        local = (world - self.truth[:3, 3]) @ self.truth[:3, :3]   # truth @ local = world
        self.a = f"{self.tmp.name}/a.npy"
        self.b = f"{self.tmp.name}/b.npy"
        np.save(self.a, local.astype(np.float32))
        np.save(self.b, (world + rng.normal(0, 0.003, world.shape)).astype(np.float32))

    def tearDown(self):
        self.tmp.cleanup()

    def snap(self, init):
        from e57stitch.register import snap_to_model
        return snap_to_model({"path": self.a, "targets": [(self.b, np.eye(4).tolist())],
                              "init": init.tolist(), "voxel": 0.05, "level": True})

    def test_recovers_pose_from_rough_placement(self):
        r = self.snap(yaw_pose(-3, [0.35, -0.25, 0]) @ self.truth)
        err = np.linalg.inv(self.truth) @ np.asarray(r["pose"])
        self.assertLess(np.linalg.norm(err[:3, 3]), 0.02)

    def test_never_leaves_the_trust_region(self):
        from e57stitch.register import SNAP_RADIUS
        init = yaw_pose(0, [1.6, 0, 0]) @ self.truth
        r = self.snap(init)
        if r.get("pose"):
            moved = np.linalg.inv(init) @ np.asarray(r["pose"])
            self.assertLess(np.linalg.norm(moved[:3, 3]), SNAP_RADIUS + 0.25)
