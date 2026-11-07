"""Pairwise and multi-view registration.

Pipeline per scan pair:

1. try ICP from the current (pose-derived) initial guess, if it already
   produces meaningful overlap;
2. otherwise run FPFH + Fast Global Registration (FGR) for a pose-free
   coarse alignment, refine with multi-scale point-to-plane ICP;
3. if FGR fails, fall back to FPFH + RANSAC and (optionally) a second FGR
   hypothesis;
4. score the result with a *structure-weighted* symmetric overlap.

Why structure-weighted?  Terrestrial scans almost always share a large flat
ground plane.  Alignments that slide along that plane can have a 70% "raw"
overlap while the walls/windows are metres apart.  Scoring only vertical
structure (|nz| < 0.6) separates good from bad alignments far more reliably.

All transforms follow the Open3D convention: ``T_ab`` maps points from scan
``a``'s local frame into scan ``b``'s local frame.
"""

from __future__ import annotations

import itertools
import math
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .features import (
    PreparedCloud,
    estimate_normals,
    make_cloud,
    prepare_registration,
    vertical_structure,
    voxel_down,
)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def _symmetric_overlap(
    a: np.ndarray, b: np.ndarray, transform: np.ndarray, tol: float, sample: int = 25_000, seed: int = 0
) -> dict[str, float]:
    if len(a) == 0 or len(b) == 0:
        return {"overlap": 0.0, "overlap_a": 0.0, "overlap_b": 0.0, "rmse": float("inf")}
    rng = np.random.default_rng(seed)
    sa = a if len(a) <= sample else a[rng.choice(len(a), sample, replace=False)]
    sb = b if len(b) <= sample else b[rng.choice(len(b), sample, replace=False)]
    sa = np.asarray(sa, dtype=np.float64)
    sb = np.asarray(sb, dtype=np.float64)

    ta = sa @ transform[:3, :3].T + transform[:3, 3]
    d_ab, _ = cKDTree(sb).query(ta, k=1, workers=KD_WORKERS)
    d_ba, _ = cKDTree(ta).query(sb, k=1, workers=KD_WORKERS)
    ia = d_ab < tol
    ib = d_ba < tol
    oa = float(ia.mean())
    ob = float(ib.mean())
    d = np.concatenate([d_ab[ia], d_ba[ib]])
    rmse = float(np.sqrt(np.mean(d * d))) if d.size else float("inf")
    return {"overlap": min(oa, ob), "overlap_a": oa, "overlap_b": ob, "rmse": rmse}


def score_transform(
    a: np.ndarray,
    b: np.ndarray,
    transform: np.ndarray,
    voxel: float,
    *,
    a_structure: np.ndarray | None = None,
    b_structure: np.ndarray | None = None,
    sample: int = 25_000,
    seed: int = 0,
) -> dict[str, float]:
    """Raw + structure-weighted overlap of ``a`` transformed into ``b``'s frame."""
    tol = max(float(voxel) * 2.0, 0.03)
    raw = _symmetric_overlap(a, b, transform, tol, sample=sample, seed=seed)
    struct = {"overlap": raw["overlap"], "rmse": raw["rmse"]}
    if a_structure is not None and b_structure is not None:
        if len(a_structure) > 300 and len(b_structure) > 300:
            struct = _symmetric_overlap(
                a_structure, b_structure, transform, max(tol, 0.15), sample=sample, seed=seed
            )
    score = float(struct["overlap"])
    if score == 0.0 and struct is raw:
        score = float(raw["overlap"])
    return {
        "overlap": float(raw["overlap"]),
        "overlap_a": float(raw["overlap_a"]),
        "overlap_b": float(raw["overlap_b"]),
        "rmse": float(raw["rmse"]),
        "structure_overlap": float(struct["overlap"]),
        "score": score,
    }


# --------------------------------------------------------------------------- #
# cached cloud levels (per worker process)
# --------------------------------------------------------------------------- #

_CLOUD_CACHE: dict[tuple[str, float, float], o3d.geometry.PointCloud] = {}
_CLOUD_ORDER: list[tuple[str, float, float]] = []
_CLOUD_MAX = 24
ICP_SOURCE_CAP = 10_000 # moving points per search ICP; the target and the evidence stay whole.  0 = all
ICP_SOURCE_SEED = 0  # which points the cap keeps; only varied to test stability


def _cloud_at(path: str, voxel: float, level: float) -> o3d.geometry.PointCloud:
    """Voxelised cloud (with normals) for one scan/level, cached per worker.

    Normals are only strictly needed for the ICP target, but sharing a single
    cloud for both roles halves the memory and lets the expensive normal
    estimation happen once per scan instead of once per pair.
    """
    key = (path, round(voxel, 6), round(level, 6))
    hit = _CLOUD_CACHE.get(key)
    if hit is not None:
        if key in _CLOUD_ORDER:
            _CLOUD_ORDER.remove(key)
        _CLOUD_ORDER.append(key)
        return hit
    points = np.load(path)
    pc = make_cloud(points).voxel_down_sample(float(level))
    # Cap the correspondence search: 5 cm x 1.5 M points is far more than ICP
    # needs and dominates runtime on large projects.
    if len(pc.points) > 200_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pc.points), 200_000, replace=False)
        pc = pc.select_by_index(idx)
    estimate_normals(pc, level)
    _CLOUD_CACHE[key] = pc
    _CLOUD_ORDER.append(key)
    while len(_CLOUD_ORDER) > _CLOUD_MAX:
        _CLOUD_CACHE.pop(_CLOUD_ORDER.pop(0), None)
    return pc


# --------------------------------------------------------------------------- #
# tilt lock: IMU-levelled scans only move in x, y, z and yaw
# --------------------------------------------------------------------------- #
#
# The scanner's IMU levels every scan far better than ICP can on a staircase
# or in a narrow corridor, where a free 6-DoF fit happily tilts a scan by
# 0.5-1.5° (8-25 cm at 10 m).  For levelled scans every ICP therefore solves
# only translation + rotation about the vertical axis.

LOCK_TILT = True
# Matching is fully level.  The final multi-view refine may still correct a
# scan's tilt by up to this much: the IMU is good to a few tenths of a degree,
# not exactly (benchmark B: free tilt fits 7.9 mm, zero tilt 9.4 mm), while a
# staircase can pull a free 6-DoF fit 1.5° off (seen on benchmark B, scan 023).
TILT_MAX_DEG = 0.5

# Engine of the final multi-view polish (6-DoF fits, tilt clamped after):
# "o3d" = Open3D point-to-plane + Tukey; "sg-gicp" / "sg-vgicp" = small_gicp
# (koide3) GICP / voxelised GICP, multithreaded preprocessing + solve.
# small_gicp is optional: without it the polish falls back to "o3d".
# "first+last" runs the final round with the second engine: small_gicp does the
# bulk of the move in seconds, Open3D sets the final accuracy (engine 13).
POLISH_ICP = "sg-gicp"
POLISH_THREADS = 0   # threads per fit; 0 = cores / workers
# thin each neighbour cloud once to the target spacing before the unions
POLISH_PRETHIN = True
# scans fitted at the same time, in processes (neither Open3D nor small_gicp
# releases the GIL).  Only scans with no link between them run together, so
# every fit still sees its neighbours' latest poses.  0 = cores / 4, 1 = in turn.
# In turn is the default: on a 20-thread server 5 workers took 267 s and 10
# took 321 s against 188 s in turn, same accuracy (engine 15).
POLISH_WORKERS = 1


def _small_gicp():
    try:
        import small_gicp
        return small_gicp
    except Exception:
        return None


def _levelled(prep: Any) -> bool:
    return LOCK_TILT and getattr(prep, "up_mode", "") == "level"


def tilt_deg(T: np.ndarray) -> float:
    """Angle between the transform's vertical axis and the true vertical."""
    return float(np.degrees(np.arccos(np.clip(np.asarray(T, dtype=np.float64)[2, 2], -1.0, 1.0))))


def level_transform(T: np.ndarray) -> np.ndarray:
    """Drop roll/pitch, keep yaw and translation."""
    T = np.asarray(T, dtype=np.float64)
    x = T[:3, 0]
    yaw = math.atan2(x[1], x[0])
    c, s_ = math.cos(yaw), math.sin(yaw)
    out = np.eye(4)
    out[:3, :3] = [[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]
    out[:3, 3] = T[:3, 3]
    return out


def clamp_tilt(T: np.ndarray, max_deg: float) -> np.ndarray:
    """Reduce the transform's tilt to at most ``max_deg``; yaw and translation kept."""
    T = np.asarray(T, dtype=np.float64)
    ang = tilt_deg(T)
    if ang <= max_deg:
        return T
    L = level_transform(T)
    if max_deg <= 0.0:
        return L
    rv = Rotation.from_matrix(L[:3, :3].T @ T[:3, :3]).as_rotvec() * (max_deg / ang)
    out = L.copy()
    out[:3, :3] = L[:3, :3] @ Rotation.from_rotvec(rv).as_matrix()
    return out


def icp_level(
    src: np.ndarray,
    dst: np.ndarray,
    dst_normals: np.ndarray,
    init: np.ndarray,
    max_dist: float,
    *,
    max_iteration: int = 40,
    loss_k: float | None = None,
    tree: cKDTree | None = None,
) -> tuple[np.ndarray, float, float]:
    """Point-to-plane ICP with 4 degrees of freedom (x, y, z, yaw).

    Gauss-Newton on n·(Rz(θ)(p-c)+c+t-q), with the yaw linearised about the
    matched points' centroid ``c``; optional Tukey weights (``loss_k``) so
    clutter that only one scan saw cannot pull the fit.
    Returns (transform, fitness, inlier rmse).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    nrm = np.asarray(dst_normals, dtype=np.float64)
    T = level_transform(init)
    if len(src) < 30 or len(dst) < 30 or len(nrm) != len(dst):
        return T, 0.0, float("inf")
    tree = tree if tree is not None else cKDTree(dst)
    fitness, rmse = 0.0, float("inf")
    for _ in range(int(max_iteration)):
        P = src @ T[:3, :3].T + T[:3, 3]
        d, j = tree.query(P, distance_upper_bound=float(max_dist), workers=KD_WORKERS)
        ok = np.isfinite(d)
        fitness = float(ok.mean())
        if ok.sum() < 30:
            break
        p, q, n = P[ok], dst[j[ok]], nrm[j[ok]]
        r = np.einsum("ij,ij->i", n, p - q)
        rmse = float(np.sqrt(np.mean(r * r)))
        c = p.mean(axis=0)
        pc = p - c
        J = np.column_stack([n[:, 0], n[:, 1], n[:, 2], n[:, 1] * pc[:, 0] - n[:, 0] * pc[:, 1]])
        if loss_k:
            u = np.clip(np.abs(r) / float(loss_k), 0.0, 1.0)
            w = (1.0 - u * u) ** 2
        else:
            w = np.ones_like(r)
        Jw = J * w[:, None]
        x = -np.linalg.solve(Jw.T @ J + np.eye(4) * 1e-9, Jw.T @ r)
        ct, st = math.cos(x[3]), math.sin(x[3])
        Rz = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        D = np.eye(4)
        D[:3, :3] = Rz
        D[:3, 3] = c - Rz @ c + x[:3]
        T = D @ T
        if np.linalg.norm(x[:3]) < 1e-5 and abs(x[3]) < 1e-6:
            break
    return T, fitness, rmse


class _Result:
    """Duck-typed stand-in for open3d's RegistrationResult."""

    def __init__(self, T: np.ndarray, fitness: float, rmse: float) -> None:
        self.transformation = T
        self.fitness = fitness
        self.inlier_rmse = rmse


def _icp_cached(
    path_a: str,
    path_b: str,
    voxel: float,
    level: float,
    init: np.ndarray,
    *,
    threshold_factor: float = 2.0,
    max_iteration: int = 60,
    lock: bool = False,
) -> Any:
    src = _cloud_at(path_a, voxel, level)
    dst = _cloud_at(path_b, voxel, level)
    if ICP_SOURCE_CAP and len(src.points) > ICP_SOURCE_CAP:
        src = src.select_by_index(np.random.default_rng(ICP_SOURCE_SEED).choice(len(src.points), ICP_SOURCE_CAP, replace=False))
    if lock:
        T, fit, rmse = icp_level(
            np.asarray(src.points), np.asarray(dst.points), np.asarray(dst.normals),
            np.asarray(init, dtype=np.float64), float(level) * threshold_factor,
            max_iteration=max_iteration,
        )
        return _Result(T, fit, rmse)
    return o3d.pipelines.registration.registration_icp(
        src,
        dst,
        float(level) * threshold_factor,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=max_iteration
        ),
    )


def _gicp_cached(
    path_a: str,
    path_b: str,
    voxel: float,
    level: float,
    init: np.ndarray,
    *,
    threshold_factor: float = 2.0,
    max_iteration: int = 30,
) -> o3d.pipelines.registration.RegistrationResult:
    """Generalized ICP: models the local surface covariance.

    Point-to-plane assumes the target surface is exactly planar; GICP is much
    better behaved on noisy TLS data, foliage and glass returns.
    """
    src = _cloud_at(path_a, voxel, level)
    dst = _cloud_at(path_b, voxel, level)
    return o3d.pipelines.registration.registration_generalized_icp(
        src,
        dst,
        float(level) * threshold_factor,
        init,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=max_iteration
        ),
    )


_COLORED_CACHE: dict[tuple[str, float], o3d.geometry.PointCloud | None] = {}
_COLORED_ORDER: list[tuple[str, float]] = []
_COLORED_MAX = 12


def _colored_cloud(reg_path: str, level: float) -> o3d.geometry.PointCloud | None:
    """Preview points + exposure-normalised colours, voxelised for coloured ICP."""
    from pathlib import Path

    key = (reg_path, round(float(level), 6))
    if key in _COLORED_CACHE:
        _COLORED_ORDER.remove(key)
        _COLORED_ORDER.append(key)
        return _COLORED_CACHE[key]
    preview = Path(reg_path).parent / "preview.npz"
    if not preview.exists():
        return None
    with np.load(preview, allow_pickle=False) as z:
        if "colors" not in z.files:
            return None
        points = np.asarray(z["points"], dtype=np.float64)
        colors = np.asarray(z["colors"], dtype=np.float64) / 255.0
    if len(points) < 500 or len(colors) != len(points):
        return None
    valid = colors.sum(axis=1) > 0.02
    if valid.sum() > 200:
        # Different stations have different exposure/white balance; remove the
        # per-scan mean and scale so colour ICP compares chroma, not brightness.
        c = colors[valid]
        mean = c.mean(axis=0)
        std = c.std(axis=0)
        colors = (colors - mean) / np.maximum(std, 0.05)
    pc = make_cloud(points)
    pc.colors = o3d.utility.Vector3dVector(colors)
    pc = pc.voxel_down_sample(float(level))
    if len(pc.points) > 150_000:
        idx = np.random.default_rng(4).choice(len(pc.points), 150_000, replace=False)
        pc = pc.select_by_index(idx)
    estimate_normals(pc, float(level))
    _COLORED_CACHE[key] = pc
    _COLORED_ORDER.append(key)
    while len(_COLORED_ORDER) > _COLORED_MAX:
        _COLORED_CACHE.pop(_COLORED_ORDER.pop(0), None)
    return pc


def _colored_icp_cached(
    path_a: str,
    path_b: str,
    voxel: float,
    level: float,
    init: np.ndarray,
    *,
    max_iteration: int = 30,
) -> o3d.pipelines.registration.RegistrationResult | None:
    src = _colored_cloud(path_a, level)
    dst = _colored_cloud(path_b, level)
    if src is None or dst is None or len(src.points) < 500 or len(dst.points) < 500:
        return None
    try:
        return o3d.pipelines.registration.registration_colored_icp(
            src,
            dst,
            float(level) * 2.0,
            init,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=max_iteration
            ),
        )
    except Exception:
        return None


def multiscale_icp_paths(
    path_a: str,
    path_b: str,
    voxel: float,
    init: np.ndarray,
    *,
    levels: Sequence[float] = (4.0, 2.0, 1.0),
) -> tuple[np.ndarray, o3d.pipelines.registration.RegistrationResult]:
    T = np.asarray(init, dtype=np.float64)
    result = None
    for factor in levels:
        level = max(float(voxel) * factor, float(voxel))
        result = _icp_cached(path_a, path_b, voxel, level, T, threshold_factor=2.5, max_iteration=30)
        T = result.transformation
    assert result is not None
    return T, result


def surface_accuracy(
    path_a: str,
    path_b: str,
    transform: np.ndarray,
    voxel: float,
    *,
    sample: int = 120_000,
) -> dict[str, float]:
    """Robust surface-to-surface deviation between two aligned scans.

    Uses point-to-plane distances (against the other scan's local normals) in
    both directions, so it measures how far the *surfaces* are off rather than
    how well the point samples happen to coincide.  Returns millimetres.
    """
    empty = {
        "mean_mm": None,
        "median_mm": None,
        "p95_mm": None,
        "rms_mm": None,
        "inlier_ratio": 0.0,
        "coverage_a": 0.0,
        "coverage_b": 0.0,
        "n": 0,
    }
    try:
        ca = _cloud_at(path_a, voxel, max(float(voxel), 0.02))
        cb = _cloud_at(path_b, voxel, max(float(voxel), 0.02))
        A = np.asarray(ca.points, dtype=np.float64)
        B = np.asarray(cb.points, dtype=np.float64)
        na = np.asarray(ca.normals, dtype=np.float64)
        nb = np.asarray(cb.normals, dtype=np.float64)
    except Exception:
        return empty
    if len(A) < 200 or len(B) < 200:
        return empty
    if len(A) > sample:
        sel = np.random.default_rng(2).choice(len(A), sample, replace=False)
        A, na = A[sel], na[sel]
    if len(B) > sample:
        sel = np.random.default_rng(3).choice(len(B), sample, replace=False)
        B, nb = B[sel], nb[sel]

    T = np.asarray(transform, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    Ti = invert(T)
    max_d = max(6.0 * float(voxel), 0.30)

    def one_way(src, src_normals, dst, dst_normals, M):
        moved = src @ M[:3, :3].T + M[:3, 3]
        d, idx = cKDTree(dst).query(moved, k=1, workers=KD_WORKERS)
        if len(idx) == 0:
            return np.empty(0)
        normal = dst_normals[idx]
        plane = np.abs(np.einsum("ij,ij->i", normal, moved - dst[idx]))
        nlen = np.linalg.norm(normal, axis=1)
        fallback = nlen < 1e-6
        if fallback.any():
            plane[fallback] = d[fallback]
        keep = d <= max_d
        plane = plane[keep]
        if len(plane) < 50:
            return np.empty(0)
        med = float(np.median(plane))
        mad = float(np.median(np.abs(plane - med)))
        if mad > 1e-9:
            plane = plane[plane <= med + 3.0 * 1.4826 * mad]
        return plane

    fwd = one_way(A, na, B, nb, T)
    rev = one_way(B, nb, A, na, Ti)
    allv = np.concatenate([fwd, rev]) if len(fwd) or len(rev) else np.empty(0)
    if len(allv) < 50:
        return empty
    return {
        "mean_mm": float(np.mean(allv) * 1000.0),
        "median_mm": float(np.median(allv) * 1000.0),
        "p95_mm": float(np.percentile(allv, 95) * 1000.0),
        "rms_mm": float(np.sqrt(np.mean(allv * allv)) * 1000.0),
        "inlier_ratio": float((len(fwd) + len(rev)) / max(len(A) + len(B), 1)),
        "coverage_a": float(len(fwd) / max(len(A), 1)),
        "coverage_b": float(len(rev) / max(len(B), 1)),
        "n": int(len(allv)),
    }


# --------------------------------------------------------------------------- #
# global (pose-free) coarse registration
# --------------------------------------------------------------------------- #


def global_register(
    a: PreparedCloud,
    b: PreparedCloud,
    *,
    attempt: int = 0,
    use_ransac_fallback: bool = True,
) -> tuple[np.ndarray, str, float]:
    """FPFH based coarse alignment. Returns (T_ab, method, coarse_fitness)."""
    coarse_voxel = max(a.voxel * 8.0, 0.15)
    mcd = float(np.clip(coarse_voxel * (3.0 + attempt * 1.5), 0.25, 3.0))

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        a.coarse,
        b.coarse,
        a.coarse_fpfh,
        b.coarse_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=mcd,
            iteration_number=64 + attempt * 48,
            maximum_tuple_count=1000,
            use_absolute_scale=bool(attempt % 2),
        ),
    )
    best = (result.transformation, "fgr", float(result.fitness))

    if use_ransac_fallback and attempt >= 1:
        try:
            o3d.utility.random.seed(attempt * 17 + 3)
        except Exception:
            pass
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            a.coarse,
            b.coarse,
            a.coarse_fpfh,
            b.coarse_fpfh,
            True,
            mcd,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(mcd),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(60_000, 0.995),
        )
        if ransac.fitness > best[2]:
            best = (ransac.transformation, "ransac", float(ransac.fitness))

    return best


# --------------------------------------------------------------------------- #
# gravity, structural hypotheses and robust correspondence selection
# --------------------------------------------------------------------------- #


def structure_band_scores(
    pa: PreparedCloud,
    pb: PreparedCloud,
    T: np.ndarray,
    voxel: float,
    *,
    bands: int = 3,
    min_points: int = 120,
    mode: str = "min",
) -> list[float | None]:
    """Vertical-overlap scores in several height bands.

    A floor-only alignment can still have a decent global structure score from
    a few pieces of furniture.  Splitting the vertical points by height makes
    it much harder for a floor fit to masquerade as a wall match.

    ``mode="min"`` (ranking) scores the worse-covered side; ``mode="max"``
    scores the better-covered side, i.e. "do the walls coincide at this
    height at all", which is what the flat-only veto needs when one scan (or
    a merged group cloud) is much larger than the other.
    """
    if pa.structure is None or pb.structure is None:
        return []
    if len(pa.structure) < min_points or len(pb.structure) < min_points:
        return []
    up = pb.up if pb.up is not None else np.array([0.0, 0.0, 1.0])
    T = np.asarray(T, dtype=np.float64)
    A = np.asarray(pa.structure, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
    B = np.asarray(pb.structure, dtype=np.float64)
    ha = A @ np.asarray(up, dtype=np.float64)
    hb = B @ np.asarray(up, dtype=np.float64)
    lo, hi = np.percentile(hb, [5.0, 95.0])
    if hi - lo < 0.3:
        return []
    edges = np.linspace(lo, hi, int(max(2, bands)) + 1)
    out: list[float | None] = []
    tol = max(float(voxel) * 2.0, 0.05)
    for i in range(len(edges) - 1):
        a_sel = (ha >= edges[i]) & (ha < edges[i + 1])
        b_sel = (hb >= edges[i]) & (hb < edges[i + 1])
        if int(a_sel.sum()) < min_points or int(b_sel.sum()) < min_points:
            out.append(None)
            continue
        ov = _symmetric_overlap(A[a_sel], B[b_sel], np.eye(4), tol, sample=10_000)
        if mode == "max":
            out.append(float(max(ov["overlap_a"], ov["overlap_b"])))
        else:
            out.append(float(ov["overlap"]))
    return out


# Large buildings: two stations 15-20 m apart can share only 10-25 % of their
# walls, so a *fraction* threshold rejects correct links.  Absolute shared wall
# area is independent of scan (or merged-model) size.
# Calibrated on benchmark A (5 cm): correct links shared 139-314 m² at
# 34-55 % cover (9-24 mm surface error); wrong/doubtful ones 50-100 m² at
# 9-31 % (76-114 mm).  Both conditions must hold.
MIN_WALL_AREA_M2 = 120.0
MIN_WALL_COVER = 0.30
# The area rule only *supplements* the wall bar: it may rescue a link with
# moderate wall overlap and a clear winner, never a near-tie.  (A12-A4
# passed on 127 m² alone with wall 0.17, margin 0.044 and 65 mm error.)
AREA_MIN_WALL_FRAC = 0.75   # x accept
AREA_MIN_MARGIN = 0.10
# Surface error is the cleanest separator seen so far: correct links measured
# 5-24 mm, wrong/doubtful ones 65-114 mm (5 cm matching voxel).
MAX_MEAN_MM = 40.0
# A link that is the *only* connection between two parts of a group cannot be
# cross-checked by cycles or the pose graph, so it must stand on its own.
BRIDGE_MAX_MM = 30.0
BRIDGE_MIN_MARGIN = 0.08


def shared_wall_area(pa: "PreparedCloud", pb: "PreparedCloud", T: np.ndarray, voxel: float) -> dict[str, float]:
    """Approximate wall area (m²) that coincides after applying ``T``.

    Structure points are voxelised at ``max(3*voxel, 0.15)``, so each matched
    point stands for roughly that cell's area of vertical surface.
    """
    if pa.structure is None or pb.structure is None or len(pa.structure) < 50 or len(pb.structure) < 50:
        return {"area": 0.0, "cover_a": 0.0, "cover_b": 0.0}
    cell = max(float(voxel) * 3.0, 0.15)
    tol = max(float(voxel) * 2.0, 0.15)
    ov = _symmetric_overlap(pa.structure, pb.structure, np.asarray(T, dtype=np.float64), tol)
    na = ov["overlap_a"] * len(pa.structure)
    nb = ov["overlap_b"] * len(pb.structure)
    return {
        "area": float(min(na, nb) * cell * cell),
        "cover_a": float(ov["overlap_a"]),
        "cover_b": float(ov["overlap_b"]),
    }


def combined_rank(score: float, wall: float, corners: int | None, ds: dict[str, Any] | None) -> float:
    """Hypothesis rank: overlap + wall bands (+ corners) + wall-direction balance.

    The balance term (geometric mean of long-wall and end-wall overlap) is
    what separates the right pose from a one-bay slide or a 180° turn in a
    long room, where long walls alone match either way.
    """
    if corners is not None:
        base = 0.45 * score + 0.35 * wall + 0.20 * min(1.0, corners / 6.0)
    else:
        base = 0.55 * score + 0.45 * wall
    if ds and ds.get("long") is not None and ds.get("end") is not None:
        bal = math.sqrt(max(float(ds["long"]), 0.0) * max(float(ds["end"]), 0.0))
        base = 0.65 * base + 0.35 * bal
    return float(base)


def link_verdict(res: dict[str, Any], accept: float) -> dict[str, Any]:
    """Acceptance rule for one pair result (shared by matcher and cache).

    Reads score, wall_score, wall_bands_cover (or wall_bands), wall_area,
    wall_cover, corner_matches and margin from ``res``.
    """
    accept = float(accept)
    score = float(res.get("score") or 0.0)
    wall = res.get("wall_score")
    bands = res.get("wall_bands_cover")
    if bands is None:
        bands = res.get("wall_bands") or []
    area = float(res.get("wall_area") or 0.0)
    cover = float(res.get("wall_cover") or 0.0)
    corners = int(res.get("corner_matches") or 0)
    margin = float(res.get("margin", 1.0) if res.get("margin") is not None else 1.0)
    mm = (res.get("accuracy") or {}).get("mean_mm")
    surface_bad = mm is not None and float(mm) > MAX_MEAN_MM
    lw, ew = res.get("long_wall"), res.get("end_wall")
    slid = bool(
        lw is not None and ew is not None
        and float(lw) >= accept * 0.75
        and float(ew) < END_RATIO_MIN * float(lw)
    )

    area_ok = bool(
        area >= MIN_WALL_AREA_M2
        and cover >= MIN_WALL_COVER
        and wall is not None
        and float(wall) >= accept * AREA_MIN_WALL_FRAC
        and margin >= AREA_MIN_MARGIN
    )
    weak_bands = sum(1 for b in bands if b is not None and b < accept * 0.5)
    # One weak band is usually just different vertical reach (one station sees
    # higher up the wall); a floor-only fit leaves at least two bands empty.
    flat_only = bool(
        wall is None
        or weak_bands >= 2
        or (float(wall) < accept * 1.05 and not area_ok)
    )
    wall_ok = bool((wall is not None and float(wall) >= accept * 1.2) or area_ok)
    corner_ok = corners >= 2
    strong = bool(wall_ok or (corner_ok and score >= accept * 1.15))
    margin_ok = bool(margin >= 0.035 or (wall_ok and corner_ok))
    ambiguous = bool(margin < 0.035 and not (wall_ok and corner_ok))
    # The end-wall ratio was tuned on one long hall and rejects correct poses
    # on stairs; where the physical constraint is measured it decides instead.
    if slid and res.get("weak_m2") is not None:
        slid = False
    accepted = bool(
        (score >= accept or area_ok) and not flat_only and strong and margin_ok and not surface_bad and not slid
    )
    if accepted:
        reason = None
    elif flat_only:
        reason = "flat-only / wall evidence too weak"
    elif ambiguous:
        reason = f"ambiguous: best-vs-second margin {margin:.3f}"
    elif not strong:
        reason = "no independent wall or corner verification"
    elif surface_bad:
        reason = f"surface error {float(mm):.0f} mm (> {MAX_MEAN_MM:.0f} mm)"
    elif slid:
        reason = f"end walls don't line up ({float(ew):.2f} vs long walls {float(lw):.2f}): slid along the room or turned"
    else:
        reason = "below acceptance threshold"
    return {
        "accepted": accepted,
        "flat_only": flat_only,
        "ambiguous": ambiguous,
        "wall_ok": wall_ok,
        "corner_ok": corner_ok,
        "area_ok": area_ok,
        "surface_bad": bool(surface_bad),
        "slid": slid,
        "rejection_reason": reason,
    }


def strong_link(res: dict[str, Any], accept: float) -> tuple[bool, str | None]:
    """Stricter test for links nothing else can cross-check (bridges).

    Must pass the plain score and wall bars (no area rescue), have a low
    surface error and a clear winner.  Missing evidence (old projects) is not
    held against a link.
    """
    accept = float(accept)
    score = float(res.get("score") or 0.0)
    wall = res.get("wall_score")
    mm = (res.get("accuracy") or {}).get("mean_mm")
    margin = res.get("margin")
    if score < accept:
        return False, f"score {score:.2f} < {accept:.2f}"
    if wall is not None and float(wall) < accept * 1.05:
        return False, f"wall overlap {float(wall):.2f} too low"
    if mm is not None and float(mm) > BRIDGE_MAX_MM:
        return False, f"surface error {float(mm):.0f} mm"
    if margin is not None and float(margin) < BRIDGE_MIN_MARGIN:
        return False, f"margin {float(margin):.3f} vs next-best pose"
    return True, None


def structure_score(pa: PreparedCloud, pb: PreparedCloud, T: np.ndarray, voxel: float) -> tuple[float | None, list[float | None]]:
    bands = structure_band_scores(pa, pb, T, voxel)
    valid = [b for b in bands if b is not None]
    if not valid:
        return None, bands
    # Require agreement in more than one band when several are observed.
    return float(np.mean(valid)), bands


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a /= max(float(np.linalg.norm(a)), 1e-12)
    b /= max(float(np.linalg.norm(b)), 1e-12)
    v = np.cross(a, b)
    c = float(np.clip(a @ b, -1.0, 1.0))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate about any axis perpendicular to a
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, helper)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _structure_icp(
    pa: PreparedCloud,
    pb: PreparedCloud,
    voxel: float,
    level: float,
    init: np.ndarray,
    *,
    max_iteration: int = 30,
) -> np.ndarray | None:
    """ICP using only non-horizontal structure (walls, columns, edges).

    The all-point ICP is dominated by floors and can stay in a floor-aligned
    local minimum.  This stage cannot discover a metres-away wall by itself,
    but it makes the final placement slide along walls instead of along the
    ground plane.
    """
    if pa.structure is None or pb.structure is None:
        return None
    if len(pa.structure) < 300 or len(pb.structure) < 300:
        return None
    try:
        src = make_cloud(pa.structure)
        dst = make_cloud(pb.structure)
        src = src.voxel_down_sample(max(float(level), 0.02))
        dst = dst.voxel_down_sample(max(float(level), 0.02))
        if len(src.points) < 200 or len(dst.points) < 200:
            return None
        estimate_normals(dst, max(float(level), 0.02))
        res = o3d.pipelines.registration.registration_icp(
            src,
            dst,
            max(float(level) * 3.0, 0.06),
            np.asarray(init, dtype=np.float64),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=max_iteration
            ),
        )
        if res.fitness <= 0.02:
            return None
        return np.asarray(res.transformation, dtype=np.float64)
    except Exception:
        return None


def snap_to_up(T: np.ndarray, up_a: np.ndarray, up_b: np.ndarray, sign: float = 1.0) -> np.ndarray:
    """Force the relative rotation to map scan A's gravity axis onto B's."""
    out = np.asarray(T, dtype=np.float64).copy()
    R = out[:3, :3]
    R_align = _rotation_between(R @ np.asarray(up_a), float(sign) * np.asarray(up_b))
    out[:3, :3] = R_align @ R
    return out


def _horizontal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.asarray(up, dtype=np.float64)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = helper - float(helper @ up) * up
    e1 /= max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(up, e1)
    return e1, e2


def _wall_angle(prep: PreparedCloud) -> tuple[float, float] | None:
    """Dominant horizontal wall direction, as an angle in the scan's up basis."""
    up = prep.up if prep.up is not None else np.array([0.0, 0.0, 1.0])
    normals = np.asarray(prep.coarse.normals, dtype=np.float64)
    if len(normals) < 50:
        return None
    m = np.abs(normals @ up) < 0.25
    if int(m.sum()) < 50:
        return None
    nn = normals[m]
    proj = nn - (nn @ up)[:, None] * up
    lengths = np.linalg.norm(proj, axis=1)
    keep = lengths > 0.2
    if int(keep.sum()) < 40:
        return None
    proj = proj[keep] / lengths[keep, None]
    e1, e2 = _horizontal_basis(up)
    angles = np.arctan2(proj @ e2, proj @ e1)
    hist, edges = np.histogram(angles, bins=72, range=(-math.pi, math.pi))
    k = int(np.argmax(hist))
    if hist[k] < max(10, 0.05 * len(angles)):
        return None
    theta = float((edges[k] + edges[k + 1]) * 0.5)
    return theta, float(hist[k] / max(len(angles), 1))


def _corner_xy(points: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e1, e2 = _horizontal_basis(up)
    pts = np.asarray(points, dtype=np.float64)
    return np.column_stack([pts @ e1, pts @ e2]), e1, e2


def corner_candidates(
    pa: PreparedCloud,
    pb: PreparedCloud,
    init: np.ndarray,
    *,
    max_corners: int = 18,
    tol: float = 0.45,
) -> list[np.ndarray]:
    """2-point RANSAC over vertical room corners / column lines.

    Corners are the one feature that directly constrains X/Y/yaw on a flat
    floor.  We only use the horizontal projection (gravity already gives
    roll/pitch) and solve a 2D rigid transform; the result is lifted back to 3D.
    """
    if pa.corners is None or pb.corners is None or pa.up is None or pb.up is None:
        return []
    A = np.asarray(pa.corners, dtype=np.float64)[:max_corners]
    B = np.asarray(pb.corners, dtype=np.float64)[:max_corners]
    if len(A) < 3 or len(B) < 3:
        return []
    A2, e1a, e2a = _corner_xy(A, pa.up)
    basis_a = np.column_stack([pa.up, e1a, e2a])
    out: list[np.ndarray] = []
    rng = np.random.default_rng(17)
    for flip in (1.0, -1.0):
        up_b = float(flip) * np.asarray(pb.up, dtype=np.float64)
        B2, e1b, e2b = _corner_xy(B, up_b)
        basis_b = np.column_stack([up_b, e1b, e2b])
        tree = cKDTree(B2)
        # Precompute pair vectors; max 18 corners => 153 pairs each.
        def pairs(coords: np.ndarray):
            return [
                (i, j, coords[j] - coords[i])
                for i in range(len(coords))
                for j in range(i + 1, len(coords))
            ]
        ap = pairs(A2)
        bp = pairs(B2)
        best = -1
        best_payload = None
        for i, j, da in ap:
            la = float(np.linalg.norm(da))
            if la < 0.35:
                continue
            ang_a = math.atan2(float(da[1]), float(da[0]))
            for k, l, db in bp:
                lb = float(np.linalg.norm(db))
                ratio = lb / max(la, 1e-9)
                if ratio < 0.7 or ratio > 1.4:
                    continue
                theta = math.atan2(float(db[1]), float(db[0])) - ang_a
                c, s = math.cos(theta), math.sin(theta)
                R2 = np.array([[c, -s], [s, c]])
                t2 = B2[k] - R2 @ A2[i]
                moved = A2 @ R2.T + t2
                dist, idx = tree.query(moved, k=1)
                inside = dist <= tol
                if int(inside.sum()) < 2:
                    continue
                unique_b = np.unique(idx[inside])
                score = int(inside.sum()) * 10 + int(len(unique_b))
                if score <= best:
                    continue
                best = score
                best_payload = (R2, t2, inside, idx)
        if best_payload is None:
            continue
        R2, t2, inside, idx = best_payload
        R23 = np.eye(3)
        R23[1:, 1:] = R2
        R3 = basis_b @ R23 @ basis_a.T
        ca = A[inside].mean(axis=0)
        cb = B[idx[inside]].mean(axis=0)
        t = cb - R3 @ ca
        # Vertical offset from a floor point is unreliable; the existing pose
        # or floor plane is a better estimate.
        t[2] = float(np.asarray(init)[2, 3])
        T = np.eye(4)
        T[:3, :3] = R3
        T[:3, 3] = t
        if not any(_pose_close(T, old, trans=0.4, rot_deg=8.0) for old in out):
            out.append(T)
    return out[:4]


def corner_match_score(pa: PreparedCloud, pb: PreparedCloud, T: np.ndarray, *, tol: float = 0.30) -> tuple[int, float]:
    """How many vertical corner lines line up (best-neighbour, horizontal)."""
    if pa.corners is None or pb.corners is None or pa.up is None or pb.up is None:
        return 0, 0.0
    A = np.asarray(pa.corners, dtype=np.float64)
    B = np.asarray(pb.corners, dtype=np.float64)
    if len(A) == 0 or len(B) == 0:
        return 0, 0.0
    T = np.asarray(T, dtype=np.float64)
    A3 = A @ T[:3, :3].T + T[:3, 3]
    A2, _, _ = _corner_xy(A3, pb.up)
    B2, _, _ = _corner_xy(B, pb.up)
    dist, _ = cKDTree(B2).query(A2, k=1)
    count = int((dist <= tol).sum())
    return count, float(count / max(1, min(len(A), len(B))))


def manhattan_hypotheses(pa: PreparedCloud, pb: PreparedCloud) -> list[np.ndarray]:
    """Try wall-aligned yaw hypotheses (Manhattan world) as a coarse seed."""
    if pa.up is None or pb.up is None:
        return []
    if pa.up_confidence < 0.35 or pb.up_confidence < 0.35:
        return []
    wa = _wall_angle(pa)
    wb = _wall_angle(pb)
    if wa is None or wb is None:
        return []
    theta_a, _ = wa
    theta_b, _ = wb
    ca = np.asarray(pa.structure if pa.structure is not None and len(pa.structure) > 200 else pa.points, dtype=np.float64)
    cb = np.asarray(pb.structure if pb.structure is not None and len(pb.structure) > 200 else pb.points, dtype=np.float64)
    ca = ca[np.random.default_rng(5).choice(len(ca), min(20_000, len(ca)), replace=False)]
    cb = cb[np.random.default_rng(6).choice(len(cb), min(20_000, len(cb)), replace=False)]
    ca_c = ca.mean(axis=0)
    cb_c = cb.mean(axis=0)
    e1a, e2a = _horizontal_basis(pa.up)
    basis_a = np.column_stack([pa.up, e1a, e2a])
    out: list[np.ndarray] = []
    for sign in (1.0, -1.0):
        up_b = float(sign) * np.asarray(pb.up, dtype=np.float64)
        e1b, e2b = _horizontal_basis(up_b)
        basis_b = np.column_stack([up_b, e1b, e2b])
        R0 = basis_b @ basis_a.T
        theta_b_signed = theta_b if sign > 0 else -theta_b
        for k in range(4):
            yaw = (theta_b_signed - theta_a) + k * (math.pi / 2.0)
            Rz = Rotation.from_rotvec(yaw * np.asarray(pb.up, dtype=np.float64)).as_matrix()
            R = Rz @ R0
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = cb_c - R @ ca_c
            out.append(T)
    return out


def _fpfh_correspondences(pa: PreparedCloud, pb: PreparedCloud, max_src: int = 3000):
    fa = np.asarray(pa.coarse_fpfh.data, dtype=np.float64).T
    fb = np.asarray(pb.coarse_fpfh.data, dtype=np.float64).T
    if len(fa) < 50 or len(fb) < 50:
        return None
    pa_pts = np.asarray(pa.coarse.points, dtype=np.float64)
    pb_pts = np.asarray(pb.coarse.points, dtype=np.float64)
    tree_b = cKDTree(fb)
    tree_a = cKDTree(fa)
    src = np.arange(len(fa))
    if len(src) > max_src:
        src = np.random.default_rng(7).choice(len(fa), max_src, replace=False)
    dist, dst = tree_b.query(fa[src], k=1, workers=KD_WORKERS)
    back, _ = tree_a.query(fb[dst], k=1, workers=KD_WORKERS)
    mutual = back == src
    src, dst = src[mutual], dst[mutual]
    if len(src) < 50:
        return None
    return pa_pts[src], pb_pts[dst]


def spectral_register(pa: PreparedCloud, pb: PreparedCloud) -> tuple[np.ndarray, float] | None:
    """SC2-PCR-style robust correspondence selection (no external dependency)."""
    corr = _fpfh_correspondences(pa, pb)
    if corr is None:
        return None
    A, B = corr
    n = len(A)
    if n > 600:
        idx = np.random.default_rng(8).choice(n, 600, replace=False)
        A, B = A[idx], B[idx]
        n = 600
    if n < 20:
        return None
    da = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
    db = np.linalg.norm(B[:, None, :] - B[None, :, :], axis=2)
    diff = np.abs(da - db)
    compat = np.exp(-((diff / 0.25) ** 2)) * (diff < 0.5)
    # Degenerate correspondences should not dominate the spectrum.
    compat[da < 0.15] *= 0.2
    try:
        _w, vecs = np.linalg.eigh(compat)
        score = vecs[:, -1]
    except np.linalg.LinAlgError:
        return None
    threshold = float(score.mean() + 0.5 * score.std())
    inliers = np.flatnonzero(score >= threshold)
    if len(inliers) < 8:
        inliers = np.argsort(score)[-max(8, n // 10):]
    Ac = A[inliers]
    Bc = B[inliers]
    ca, cb = Ac.mean(axis=0), Bc.mean(axis=0)
    H = (Ac - ca).T @ (Bc - cb)
    try:
        U, _S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cb - R @ ca
    return T, float(len(inliers) / n)


# --------------------------------------------------------------------------- #
# worker process
# --------------------------------------------------------------------------- #

_PREP_CACHE: dict[str, PreparedCloud] = {}
_PREP_ORDER: list[str] = []
_PREP_MAX = 8


def _load_prepared(path: str, voxel: float) -> PreparedCloud:
    key = f"{path}|{voxel}"
    if key in _PREP_CACHE:
        _PREP_ORDER.remove(key)
        _PREP_ORDER.append(key)
        return _PREP_CACHE[key]
    points = np.load(path)
    prep = prepare_registration(points, voxel)
    # The scan's orientation override lives in the bundle's meta.json.
    try:
        import json
        from pathlib import Path

        meta = json.loads((Path(path).parent / "meta.json").read_text())
        feature_file = Path(path).parent / "features.json"
        if feature_file.exists():
            try:
                features = json.loads(feature_file.read_text())
                corners = [
                    c["point"] for c in (features.get("corners") or []) if c.get("significant", True)
                ]
                if corners:
                    prep.corners = np.asarray(corners, dtype=np.float32)
            except Exception:
                pass
        orientation = str(meta.get("orientation") or "auto")
        if orientation == "level":
            prep.up = np.array([0.0, 0.0, 1.0])
            prep.up_confidence = 1.0
            prep.up_mode = "level"
            prep.tilt_deg = 0.0
            prep.structure = vertical_structure(points, voxel, prep.up)
        elif orientation == "free":
            prep.up_confidence = 0.0
            prep.up_mode = "unknown"
    except Exception:
        pass
    _PREP_CACHE[key] = prep
    _PREP_ORDER.append(key)
    while len(_PREP_ORDER) > _PREP_MAX:
        _PREP_CACHE.pop(_PREP_ORDER.pop(0), None)
    return prep


ENGINE_CACHE = 14    # cache key of stored stage results: bump only when an existing stage changes
ENGINE_VERSION = 21  # 21: after polish, each scan is fitted to every overlapping scan of its group, not only its links; 20: search ICP moves at most 10k points; 19: at most 20k; 18: polish with small_gicp only; 17: rival poses decided by mean evidence, floor 0.25, rescue stops when idle; 16: selection and route without capture order (evidence growth, self-check with restarts, piece rescue); 15: polish fits in turn again (parallel was slower, no accuracy gain); 14: polish fits unlinked scans in parallel processes; 13: polish sg-gicp+o3d with pre-thinned unions; 12: local (cropped) stage for unlinked route neighbours; 11: height search per floor-plan peak; 10: physical evidence; 9: tilt lock

# KD-tree query threads.  -1 (all cores) is right in the server process; pool
# workers set this to 1 (pipeline._worker_init), otherwise 8 workers each start
# a thread per core for every query and spend more time scheduling than working.
KD_WORKERS = -1


@dataclass
class PairJob:
    a: str
    b: str
    path_a: str
    path_b: str
    init: np.ndarray
    voxel: float
    accept_score: float = 0.15
    max_attempts: int = 2
    a_key: str = ""
    b_key: str = ""
    # scanner positions in each scan's registration frame (for the
    # see-through check); the E57 pose translation when poses are applied
    origin_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    source: str = ""


def _pose_close(a: np.ndarray, b: np.ndarray, trans: float = 0.15, rot_deg: float = 3.0) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if float(np.linalg.norm(a[:3, 3] - b[:3, 3])) > trans:
        return False
    return rotation_angle_deg(a[:3, :3].T @ b[:3, :3]) <= rot_deg


def register_pair_job(job: PairJob) -> dict[str, Any]:
    t0 = time.time()
    pa = _load_prepared(job.path_a, job.voxel)
    pb = _load_prepared(job.path_b, job.voxel)
    voxel = job.voxel
    candidates: list[tuple[float, np.ndarray, str, dict[str, float]]] = []

    def evaluate(T: np.ndarray) -> dict[str, float]:
        return score_transform(
            pa.points,
            pb.points,
            T,
            voxel,
            a_structure=pa.structure,
            b_structure=pb.structure,
        )

    def add(T: np.ndarray, method: str, s: dict[str, float] | None = None) -> dict[str, float]:
        T = np.asarray(T, dtype=np.float64)
        if s is None:
            s = evaluate(T)
        candidates.append((float(s["score"]), T, method, s))
        return s

    def refine(T: np.ndarray, current: dict[str, float]) -> tuple[np.ndarray, dict[str, float]]:
        """Adaptive multi-scale point-to-plane ICP."""
        levels = (2.0, 1.0) if current["score"] >= 0.15 else (4.0, 2.0, 1.0)
        best_T = np.asarray(T, dtype=np.float64)
        best = current
        for factor in levels:
            if factor <= 1.01 and best["score"] < max(0.04, job.accept_score * 0.4):
                break
            level = max(float(voxel) * factor, float(voxel))
            result = _icp_cached(job.path_a, job.path_b, voxel, level, best_T)
            candidate = evaluate(result.transformation)
            if candidate["score"] < best["score"] - 0.05:
                break
            best_T, best = result.transformation, candidate
        return best_T, best

    def add_refined(T: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, float]]:
        s0 = evaluate(T)
        T1, s1 = refine(T, s0)
        if s1["score"] >= s0["score"]:
            add(T1, f"{method}+icp", s1)
            return T1, s1
        add(T, method, s0)
        return np.asarray(T, dtype=np.float64), s0

    def best_candidate() -> tuple[float, np.ndarray, str, dict[str, float]]:
        return max(candidates, key=lambda c: c[0])

    # 1) current guess (from the existing poses), plus a gravity-snapped variant
    init = np.asarray(job.init, dtype=np.float64)
    s0 = add(init, "init")
    if s0["overlap"] >= 0.03:
        T1, s1 = refine(init, s0)
        if s1["score"] >= s0["score"]:
            add(T1, "icp", s1)
    if (
        pa.up is not None
        and pb.up is not None
        and pa.up_confidence >= 0.35
        and pb.up_confidence >= 0.35
    ):
        for sign in (1.0, -1.0):
            Ts = snap_to_up(init, pa.up, pb.up, sign)
            if not _pose_close(Ts, init, trans=0.01, rot_deg=0.5):
                ss = add(Ts, "gravity-snap")
                if ss["score"] >= job.accept_score * 0.6:
                    T2, s2 = refine(Ts, ss)
                    if s2["score"] >= ss["score"]:
                        add(T2, "gravity-snap+icp", s2)

    # 2) pose-free coarse alignment when the guess is not good enough
    attempt = 0
    while best_candidate()[0] < job.accept_score and attempt < max(1, job.max_attempts):
        T0, coarse_method, _ = global_register(pa, pb, attempt=attempt)
        add(T0, coarse_method)
        coarse_best: tuple[np.ndarray, dict[str, float], str] = (T0, evaluate(T0), coarse_method)
        # Gravity is a much stronger constraint than six free rotation DOF.
        if (
            pa.up is not None
            and pb.up is not None
            and pa.up_confidence >= 0.35
            and pb.up_confidence >= 0.35
        ):
            for sign in (1.0, -1.0):
                Ts = snap_to_up(T0, pa.up, pb.up, sign)
                ss = evaluate(Ts)
                add(Ts, f"{coarse_method}+gravity", ss)
                if ss["score"] > coarse_best[1]["score"]:
                    coarse_best = (Ts, ss, f"{coarse_method}+gravity")
        T1, s1 = refine(coarse_best[0], coarse_best[1])
        add(T1, f"{coarse_best[2]}+icp", s1)
        if best_candidate()[0] >= 0.4:
            break
        attempt += 1

    wall_cache: dict[tuple, tuple[float | None, list[float | None]]] = {}
    corner_cache: dict[tuple, tuple[int, float]] = {}

    def wall_info(T: np.ndarray) -> tuple[float | None, list[float | None]]:
        key = tuple(np.round(np.asarray(T, dtype=np.float64).ravel(), 4).tolist())
        if key not in wall_cache:
            wall_cache[key] = structure_score(pa, pb, T, voxel)
        return wall_cache[key]

    def corner_info(T: np.ndarray) -> tuple[int, float]:
        key = tuple(np.round(np.asarray(T, dtype=np.float64).ravel(), 4).tolist())
        if key not in corner_cache:
            corner_cache[key] = corner_match_score(pa, pb, T)
        return corner_cache[key]

    def has_corners() -> bool:
        return (
            pa.corners is not None
            and pb.corners is not None
            and len(pa.corners) >= 3
            and len(pb.corners) >= 3
        )

    dir_cache: dict[tuple, dict[str, Any] | None] = {}

    def dir_info(T: np.ndarray) -> dict[str, Any] | None:
        key = tuple(np.round(np.asarray(T, dtype=np.float64).ravel(), 4).tolist())
        if key not in dir_cache:
            try:
                dir_cache[key] = direction_scores(pa, pb, T, voxel)
            except Exception:
                dir_cache[key] = None
        return dir_cache[key]

    def rank_candidate(c: tuple[float, np.ndarray, str, dict[str, float]]) -> tuple[float, float | None]:
        sc, T, _method, _stats = c
        wall, _bands = wall_info(T)
        if wall is None:
            return sc, None
        # A floor-only fit can have a good global score; wall bands and
        # explicit corner lines should be the tie-breakers whenever available.
        corner = corner_info(T)[0] if has_corners() else None
        return combined_rank(sc, wall, corner, dir_info(T)), wall

    def best_ranked() -> tuple[float, np.ndarray, str, dict[str, float]]:
        return max(candidates, key=lambda c: rank_candidate(c)[0])

    def wall_weak() -> bool:
        _rank, wall = rank_candidate(best_candidate())
        return wall is not None and wall < job.accept_score * 0.75

    # 2c) Explicit vertical corner/column RANSAC.  Corners directly constrain
    #     X/Y/yaw on a flat floor; this is the seed floor-only ICP cannot find.
    if has_corners() and (corner_info(best_candidate()[1])[0] < 4 or wall_weak()):
        corner_seeds = corner_candidates(pa, pb, init)
        for T0 in corner_seeds[:3]:
            add_refined(T0, "corner-ransac")

    # 3) Manhattan-wall seed (excellent on repetitive architectural scans)
    if best_candidate()[0] < job.accept_score or wall_weak():
        hyps = manhattan_hypotheses(pa, pb)
        if hyps:
            scored = sorted(((evaluate(T)["score"], i, T) for i, T in enumerate(hyps)), reverse=True)
            for sc, _i, T in scored[:2]:
                add_refined(T, "manhattan")

    # 3b) Structure-only coarse seed when all-point alignment chose the floor.
    if wall_weak() and pa.structure is not None and pb.structure is not None:
        if len(pa.structure) >= 500 and len(pb.structure) >= 500:
            try:
                spa = prepare_registration(pa.structure, voxel)
                spb = prepare_registration(pb.structure, voxel)
                T0, _m, _f = global_register(spa, spb, attempt=0)
                add_refined(T0, "structure-fgr")
            except Exception:
                pass

    # 4) Spectral correspondence selection (SC2-PCR style, robust to outliers)
    if best_candidate()[0] < job.accept_score or wall_weak():
        spec = spectral_register(pa, pb)
        if spec is not None:
            T0, _frac = spec
            add_refined(T0, "spectral")

    # 4c) Long rooms: slide along the room (one bay) or turn 180° so the end
    #     walls line up.  ICP cannot climb out of either trap by itself.
    try:
        top = best_ranked()
        if top[0] >= job.accept_score * 0.6:
            for T0, label in axis_slide_seeds(pa, pb, top[1], k=3):
                add_refined(T0, label)
    except Exception:
        pass

    # 4b) Vertical-only ICP from the best few hypotheses.  This is what stops
    #     the final slide from being constrained by the floor.
    for _sc, T, method, _st in sorted(candidates, key=lambda c: rank_candidate(c)[0], reverse=True)[:3]:
        T_wall = _structure_icp(pa, pb, voxel, max(float(voxel), 0.02), T)
        if T_wall is not None:
            add(T_wall, f"{method}+wall-icp", evaluate(T_wall))

    b_sc, b_T, b_method, b_stats = best_ranked()

    # 5) GICP and coloured ICP refine the winner.  These do not change the
    #    hypothesis, only the accuracy of the surface alignment.
    if b_sc >= max(0.15, job.accept_score * 0.7):
        try:
            level = max(float(voxel), 0.02)
            g = _gicp_cached(job.path_a, job.path_b, voxel, level, b_T)
            gs = evaluate(g.transformation)
            add(g.transformation, "gicp", gs)
        except Exception:
            pass
        try:
            if b_stats.get("overlap", 0.0) >= 0.12:
                level = max(float(voxel), 0.02)
                c = _colored_icp_cached(job.path_a, job.path_b, voxel, level, b_T)
                if c is not None:
                    cs = evaluate(c.transformation)
                    if cs["score"] >= b_sc - 0.02:
                        add(c.transformation, "colored-icp", cs)
        except Exception:
            pass

    b_sc, b_T, b_method, b_stats = best_ranked()
    b_wall, b_bands = wall_info(b_T)
    b_corner_count, b_corner_frac = corner_info(b_T)

    # Keep the best few genuinely different poses for cycle repair / the UI.
    alternatives: list[dict[str, Any]] = []
    for sc, T, method, st in sorted(candidates, key=lambda c: rank_candidate(c)[0], reverse=True):
        if any(_pose_close(T, alt["transform"]) for alt in alternatives):
            continue
        wall, bands = wall_info(T)
        corner_count, corner_frac = corner_info(T)
        ds = dir_info(T) or {}
        alternatives.append(
            {
                "transform": np.asarray(T, dtype=np.float64).tolist(),
                "score": float(sc),
                "long_wall": ds.get("long"),
                "end_wall": ds.get("end"),
                "wall_score": None if wall is None else float(wall),
                "wall_bands": bands,
                "corner_matches": int(corner_count),
                "corner_frac": float(corner_frac),
                "method": method,
                "rmse": float(st.get("rmse", 0.0)),
                "overlap": float(st.get("overlap", 0.0)),
            }
        )
        if len(alternatives) >= 3:
            break

    def alternative_rank(alt: dict[str, Any]) -> float:
        score = float(alt.get("score", 0.0) or 0.0)
        wall = alt.get("wall_score")
        if wall is None:
            return score
        corner = int(alt.get("corner_matches", 0) or 0) if has_corners() else None
        return combined_rank(score, float(wall), corner, {"long": alt.get("long_wall"), "end": alt.get("end_wall")})

    ranked_alts = sorted(alternatives, key=alternative_rank, reverse=True)
    margin = (
        alternative_rank(ranked_alts[0]) - alternative_rank(ranked_alts[1])
        if len(ranked_alts) > 1
        else 1.0
    )
    b_cover_bands = structure_band_scores(pa, pb, b_T, voxel, mode="max")
    wa = shared_wall_area(pa, pb, b_T, voxel)
    b_dir = dir_info(b_T) or {}
    evidence = {
        "long_wall": b_dir.get("long"),
        "end_wall": b_dir.get("end"),
        "score": float(b_stats["score"]),
        "wall_score": None if b_wall is None else float(b_wall),
        "wall_bands": b_bands,
        "wall_bands_cover": b_cover_bands,
        "wall_area": wa["area"],
        "wall_cover": max(wa["cover_a"], wa["cover_b"]),
        "corner_matches": int(b_corner_count),
        "margin": float(margin),
    }
    accuracy = surface_accuracy(job.path_a, job.path_b, b_T, voxel)
    evidence["accuracy"] = accuracy
    verdict = link_verdict(evidence, job.accept_score)
    return {
        "a": job.a,
        "b": job.b,
        "transform": np.asarray(b_T, dtype=np.float64).tolist(),
        "overlap": float(b_stats["overlap"]),
        "overlap_a": float(b_stats["overlap_a"]),
        "overlap_b": float(b_stats["overlap_b"]),
        "rmse": float(b_stats["rmse"]),
        "structure_overlap": float(b_stats["structure_overlap"]),
        "score": float(b_stats["score"]),
        "wall_score": None if b_wall is None else float(b_wall),
        "wall_bands": b_bands,
        "corner_matches": int(b_corner_count),
        "corner_frac": float(b_corner_frac),
        "method": b_method,
        "seconds": round(time.time() - t0, 2),
        "margin": float(margin),
        "long_wall": b_dir.get("long"),
        "end_wall": b_dir.get("end"),
        "wall_bands_cover": b_cover_bands,
        "wall_area": wa["area"],
        "wall_cover": max(wa["cover_a"], wa["cover_b"]),
        **verdict,
        "accuracy": accuracy,
        "alternatives": alternatives,
        "up_used": {
            "a": None if pa.up is None else np.asarray(pa.up).round(3).tolist(),
            "b": None if pb.up is None else np.asarray(pb.up).round(3).tolist(),
            "a_mode": pa.up_mode,
            "b_mode": pb.up_mode,
        },
    }


# --------------------------------------------------------------------------- #
# wall directions: long rooms, bays and 180° symmetry
# --------------------------------------------------------------------------- #
#
# In a long rectangular room most vertical points sit on the two long walls.
# A pose slid by one bay, or turned 180°, keeps the long walls on top of each
# other and so scores almost as well as the right one; only the end walls
# (and cross walls / columns) disagree.  We therefore classify wall points by
# normal direction relative to the room's dominant wall direction and score
# the two classes separately.

# benchmark A: correct links 0.71-0.98, the 180°-turned / bay-slid ones 0.14-0.42.
END_RATIO_MIN = 0.60        # end-wall overlap / long-wall overlap below this = slid or turned
END_MIN_POINTS = 300


def _structure_normals(prep: PreparedCloud) -> np.ndarray | None:
    cached = getattr(prep, "_struct_normals", None)
    if cached is not None or prep.structure is None or len(prep.structure) < 50:
        return cached
    pc = make_cloud(np.asarray(prep.structure, dtype=np.float64))
    radius = max(float(prep.voxel) * 3.0, 0.15) * 2.0
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    n = np.asarray(pc.normals, dtype=np.float64)
    prep._struct_normals = n  # type: ignore[attr-defined]
    return n


def _horizontal_azimuth(n: np.ndarray, up: np.ndarray) -> np.ndarray:
    e1, e2 = _horizontal_basis(up)
    return np.arctan2(n @ e2, n @ e1)


def wall_axis(prep: PreparedCloud) -> tuple[np.ndarray, np.ndarray] | None:
    """(normal of the dominant walls, direction along them), horizontal, unit."""
    n = _structure_normals(prep)
    if n is None:
        return None
    up = prep.up if prep.up is not None else np.array([0.0, 0.0, 1.0])
    az = np.mod(_horizontal_azimuth(n, up), np.pi)
    hist, edges = np.histogram(az, bins=90, range=(0.0, np.pi))
    # circular smoothing so a peak at 0°/180° is not split
    sm = hist + np.roll(hist, 1) + np.roll(hist, -1)
    a = edges[int(np.argmax(sm))] + (edges[1] - edges[0]) / 2
    e1, e2 = _horizontal_basis(up)
    normal = np.cos(a) * e1 + np.sin(a) * e2
    along = np.cross(up / (np.linalg.norm(up) or 1.0), normal)
    return normal / np.linalg.norm(normal), along / (np.linalg.norm(along) or 1.0)


def direction_scores(pa: PreparedCloud, pb: PreparedCloud, T: np.ndarray, voxel: float) -> dict[str, Any] | None:
    """Overlap of long walls and of end (perpendicular) walls separately."""
    ax = wall_axis(pb)
    na, nb = _structure_normals(pa), _structure_normals(pb)
    if ax is None or na is None or nb is None:
        return None
    normal, _along = ax
    T = np.asarray(T, dtype=np.float64)
    A = np.asarray(pa.structure, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
    nA = na @ T[:3, :3].T
    B = np.asarray(pb.structure, dtype=np.float64)
    ca, cb = np.abs(nA @ normal), np.abs(nb @ normal)
    tol = max(float(voxel) * 2.0, 0.15)
    out: dict[str, Any] = {}
    for name, sa, sb in (("long", ca > 0.9, cb > 0.9), ("end", ca < 0.42, cb < 0.42)):
        out[f"n_{name}"] = int(min(sa.sum(), sb.sum()))
        if sa.sum() < END_MIN_POINTS or sb.sum() < END_MIN_POINTS:
            out[name] = None
            continue
        out[name] = float(_symmetric_overlap(A[sa], B[sb], np.eye(4), tol, sample=12_000)["overlap"])
    return out


def axis_slide_seeds(pa: PreparedCloud, pb: PreparedCloud, T: np.ndarray, *, k: int = 3) -> list[tuple[np.ndarray, str]]:
    """Candidate poses that slide ``T`` along the room so the end walls line up.

    End-wall points of both scans are projected onto the room axis and
    cross-correlated (as a 1-D histogram), which finds a one-bay slip that
    ICP cannot climb out of.  A 180° turn about the vertical is offered too.
    """
    ax = wall_axis(pb)
    na, nb = _structure_normals(pa), _structure_normals(pb)
    if ax is None or na is None or nb is None:
        return []
    normal, along = ax
    up = pb.up if pb.up is not None else np.array([0.0, 0.0, 1.0])
    seeds: list[tuple[np.ndarray, str]] = []

    def slides(Tc: np.ndarray, tag: str, kk: int) -> None:
        A = np.asarray(pa.structure, dtype=np.float64) @ Tc[:3, :3].T + Tc[:3, 3]
        nA = na @ Tc[:3, :3].T
        B = np.asarray(pb.structure, dtype=np.float64)
        ta = A[np.abs(nA @ along) > 0.9] @ along
        tb = B[np.abs(nb @ along) > 0.9] @ along
        if len(ta) < END_MIN_POINTS or len(tb) < END_MIN_POINTS:
            return
        step = 0.05
        lo = min(ta.min(), tb.min()) - 8.0
        hi = max(ta.max(), tb.max()) + 8.0
        bins = np.arange(lo, hi + step, step)
        ha, _ = np.histogram(ta, bins)
        hb, _ = np.histogram(tb, bins)
        cap_a = np.percentile(ha[ha > 0], 90) if (ha > 0).any() else 1
        cap_b = np.percentile(hb[hb > 0], 90) if (hb > 0).any() else 1
        ha = np.minimum(ha, cap_a).astype(np.float64)
        hb = np.minimum(hb, cap_b).astype(np.float64)
        ha = np.convolve(ha, [0.25, 0.5, 0.25], mode="same")
        hb = np.convolve(hb, [0.25, 0.5, 0.25], mode="same")
        maxs = int(8.0 / step)
        corr = np.array([float(np.dot(np.roll(ha, s), hb)) for s in range(-maxs, maxs + 1)])
        order = np.argsort(corr)[::-1]
        picked: list[float] = []
        for idx in order:
            d = (idx - maxs) * step
            if abs(d) < 0.25 or any(abs(d - p) < 0.5 for p in picked):
                continue
            if corr[idx] <= corr[maxs] * 1.05:  # not better than staying put
                break
            picked.append(d)
            M = np.eye(4)
            M[:3, 3] = d * along
            seeds.append((M @ Tc, f"{tag}slide{d:+.1f}m"))
            if len(picked) >= kk:
                break

    T = np.asarray(T, dtype=np.float64)
    slides(T, "", k)
    A = np.asarray(pa.structure, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
    flipped = _yaw_about(T, up, 180.0, A.mean(axis=0))
    seeds.append((flipped, "flip180"))
    slides(flipped, "flip180+", 1)
    return seeds


# --------------------------------------------------------------------------- #
# fast pass: levelled floor-plan (bird's-eye) matching
# --------------------------------------------------------------------------- #
#
# Terrestrial scans are levelled, so registration is really x, y, yaw (+ a
# floor offset).  Each scan's wall points become a top-down occupancy image;
# for every yaw (2° steps) an FFT cross-correlation finds the best x/y shift.
# That is an exhaustive *global* search in ~1 s per pair, and it also says
# when two scans do not overlap at all, which is where the slow matcher spent
# ~90 % of its time.  The best peaks are verified with one ICP ladder and the
# usual evidence + link_verdict, so fast links obey the same rules.

BEV_COARSE = 0.20       # m per cell for the yaw sweep
BEV_FINE = 0.10         # m per cell for the local refinement
BEV_YAW_STEP = 2.0      # degrees
# Best peak = fraction of the smaller floor plan that lands on the other's
# walls.  benchmark A: overlapping pairs 0.64-0.90, non-overlapping 0.24-0.31.
BEV_MIN = 0.40          # below: no overlap (no ICP at all)
BEV_GREY = 0.30         # [GREY, MIN): could be a small overlap -> full matcher
LEVEL_MIN_CONF = 0.35
Z_PEAKS = 3             # height offsets tried per floor-plan peak (engine 11)
Z_PEAK_REL = 0.10       # weaker height peaks (vs the best) are not tried
Z_SPAN = 9.0            # m searched up and down


def _bev(prep: PreparedCloud) -> dict[str, Any] | None:
    cached = getattr(prep, "_bev", None)
    if cached is not None:
        return cached
    if prep.structure is None or len(prep.structure) < 200:
        return None
    up = prep.up if prep.up is not None else np.array([0.0, 0.0, 1.0])
    R = _rotation_between(np.array(up, dtype=np.float64), np.array([0.0, 0.0, 1.0]))
    S = np.asarray(prep.structure, dtype=np.float64) @ R.T
    P = np.asarray(prep.points, dtype=np.float64)
    zf = float(np.percentile(P @ R[2], 2.0))  # floor height in the levelled frame
    xy = S[:, :2]
    # one point per fine cell keeps the rasters cheap
    keys = np.unique(np.floor(xy / BEV_FINE).astype(np.int64), axis=0)
    xy = (keys + 0.5) * BEV_FINE
    out = {"xy": xy, "R": R, "floor": zf, "c": xy.mean(axis=0)}
    prep._bev = out  # type: ignore[attr-defined]
    return out


def _raster(xy: np.ndarray, origin: np.ndarray, shape: tuple[int, int], cell: float) -> np.ndarray:
    ij = np.floor((xy - origin) / cell).astype(np.int64)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    g = np.zeros(shape, dtype=np.float32)
    g[ij[ok, 0], ij[ok, 1]] = 1.0
    return g


def _dilate(g: np.ndarray) -> np.ndarray:
    from scipy.ndimage import maximum_filter

    return maximum_filter(g, size=3)


def bev_search(pa: PreparedCloud, pb: PreparedCloud, *, k: int = 3) -> list[dict[str, Any]]:
    """Best floor-plan placements of ``pa`` on ``pb`` as 4x4 transforms (a→b)."""
    from scipy import fft as sfft

    A, B = _bev(pa), _bev(pb)
    if A is None or B is None:
        return []
    a0 = A["xy"] - A["c"]
    rA = float(np.max(np.linalg.norm(a0, axis=1))) + 1.0
    bmin, bmax = B["xy"].min(axis=0), B["xy"].max(axis=0)

    def sweep(cell: float, yaws: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
        origin = bmin - rA
        size = (bmax - bmin) + 2 * rA
        shape = tuple(sfft.next_fast_len(int(np.ceil(s / cell)) + 2) for s in size)
        Bg = _dilate(_raster(B["xy"], origin, shape, cell))
        nB = float(_raster(B["xy"], origin, shape, cell).sum())
        FB = sfft.rfft2(Bg)
        centre = origin + np.array(shape) * cell / 2.0
        found = []
        for yaw in yaws:
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            ar = a0 @ np.array([[c, -s], [s, c]]).T
            Ag = _raster(ar + centre, origin, shape, cell)
            nA = float(Ag.sum())
            if nA < 20:
                continue
            corr = sfft.irfft2(FB * np.conj(sfft.rfft2(Ag)), s=shape)
            idx = np.unravel_index(int(np.argmax(corr)), shape)
            shift = np.array([i if i <= n // 2 else i - n for i, n in zip(idx, shape)], dtype=np.float64)
            score = float(corr[idx]) / max(1.0, min(nA, nB))
            t = centre + shift * cell - (np.array([[c, -s], [s, c]]) @ A["c"])
            found.append((score, float(yaw), t))
        return found

    coarse = sweep(BEV_COARSE, np.arange(0.0, 360.0, BEV_YAW_STEP))
    coarse.sort(key=lambda x: x[0], reverse=True)
    picked: list[tuple[float, float, np.ndarray]] = []
    for sc, yaw, t in coarse:
        if any(abs(((yaw - y) + 180) % 360 - 180) < 8.0 and np.linalg.norm(t - tt) < 1.5 for _s, y, tt in picked):
            continue
        picked.append((sc, yaw, t))
        if len(picked) >= k:
            break
    peaks = []
    for sc, yaw, t in picked:
        fine = sweep(BEV_FINE, yaw + np.arange(-2.0, 2.01, 0.5))
        fs, fy, ft = max(fine, key=lambda x: x[0]) if fine else (sc, yaw, t)
        c, s = math.cos(math.radians(fy)), math.sin(math.radians(fy))
        L = np.eye(4)
        L[:2, :2] = [[c, -s], [s, c]]
        L[:2, 3] = ft
        L[2, 3] = B["floor"] - A["floor"]
        Ra, Rb = np.eye(4), np.eye(4)
        Ra[:3, :3], Rb[:3, :3] = A["R"], B["R"]
        peaks.append({"score": float(fs), "coarse": float(sc), "yaw": float(fy), "T": Rb.T @ L @ Ra})
    peaks.sort(key=lambda p: p["score"], reverse=True)
    return peaks


def z_offsets(pa: PreparedCloud, pb: PreparedCloud, T: np.ndarray, *, k: int = Z_PEAKS, rel: float = Z_PEAK_REL,
              cell: float = 0.25, dzbin: float = 0.05, span: float = Z_SPAN) -> list[tuple[float, float]]:
    """Height offsets (metres along b's up axis, added to ``T``) at which a's
    vertical profile matches b's, over the floor-plan cells both scans see.

    The floor-plan peak fixes x, y and yaw only.  Engine <= 10 took the
    height from the scans' lowest points, which in a stairwell is a
    different floor for each scan (the bottom of the stair eye), and ICP
    never leaves that floor.  Correlating the occupancy columns finds every
    floor the two scans could share; each becomes its own hypothesis.
    Returns [(relative strength, dz)], strongest first."""
    up = np.asarray(pb.up if pb.up is not None else (0.0, 0.0, 1.0), dtype=np.float64)
    R = _rotation_between(up, np.array([0.0, 0.0, 1.0]))
    A = (np.asarray(pa.points) @ T[:3, :3].T + T[:3, 3]) @ R.T
    B = np.asarray(pb.points) @ R.T
    A = A[:: max(1, len(A) // 200_000)]
    B = B[:: max(1, len(B) // 200_000)]
    ka = np.floor(A[:, :2] / cell).astype(np.int64)
    kb = np.floor(B[:, :2] / cell).astype(np.int64)
    ha = ka[:, 0] * 1_000_003 + ka[:, 1]
    hb = kb[:, 0] * 1_000_003 + kb[:, 1]
    common = np.intersect1d(ha, hb)
    if len(common) < 20:
        return []
    ia, ib = np.isin(ha, common), np.isin(hb, common)
    z0 = min(A[ia, 2].min(), B[ib, 2].min())
    nz = int(np.ceil((max(A[ia, 2].max(), B[ib, 2].max()) - z0) / dzbin)) + 1
    GA = np.zeros((len(common), nz), np.float32)
    GB = np.zeros((len(common), nz), np.float32)
    GA[np.searchsorted(common, ha[ia]), ((A[ia, 2] - z0) / dzbin).astype(np.int64)] = 1.0
    GB[np.searchsorted(common, hb[ib]), ((B[ib, 2] - z0) / dzbin).astype(np.int64)] = 1.0
    n = 1 << int(np.ceil(np.log2(2 * nz)))
    corr = np.fft.irfft((np.fft.rfft(GB, n) * np.conj(np.fft.rfft(GA, n))).sum(0), n)
    lags = np.arange(n)
    lags[lags > n // 2] -= n
    keep = np.abs(lags * dzbin) <= span
    order = np.argsort(lags[keep])
    lags, corr = lags[keep][order], corr[keep][order]
    from scipy.ndimage import maximum_filter1d

    loc = (corr == maximum_filter1d(corr, size=9)) & (corr > 0)
    out: list[tuple[float, float]] = []
    for c, dz in sorted(zip(corr[loc], lags[loc] * dzbin), reverse=True):
        if out and c < rel * out[0][0]:
            break
        if all(abs(dz - d) > 0.4 for _c, d in out):
            out.append((float(c), float(dz)))
        if len(out) >= k:
            break
    top = out[0][0] if out else 1.0
    return [(c / top, dz) for c, dz in out]


def _shift_up(T: np.ndarray, pb: PreparedCloud, dz: float) -> np.ndarray:
    up = np.asarray(pb.up if pb.up is not None else (0.0, 0.0, 1.0), dtype=np.float64)
    T = np.array(T, dtype=np.float64)
    T[:3, 3] += dz * up / max(1e-9, float(np.linalg.norm(up)))
    return T


def _bridges_of(nodes: Sequence[str], edges: list[tuple[str, str, str]]) -> set[str]:
    """Keys of edges whose removal disconnects their component (Tarjan)."""
    adj: dict[str, list[tuple[str, str]]] = {n: [] for n in nodes}
    for a, b, key in edges:
        if a in adj and b in adj:
            adj[a].append((b, key))
            adj[b].append((a, key))
    disc: dict[str, int] = {}
    low: dict[str, int] = {}
    out: set[str] = set()
    t = [0]

    def dfs(u: str, via: str | None) -> None:
        disc[u] = low[u] = t[0]
        t[0] += 1
        for v, key in adj[u]:
            if key == via:
                continue
            if v not in disc:
                dfs(v, key)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    out.add(key)
            else:
                low[u] = min(low[u], disc[v])

    for n in adj:
        if n not in disc:
            dfs(n, None)
    return out



# --------------------------------------------------------------------------- #
# hypotheses: see-through check, one evaluator, three workers
# --------------------------------------------------------------------------- #
#
# The pipeline never decides inside a pair job.  A job only produces
# *hypotheses* (candidate a→b transforms), each evaluated with the same
# evidence; which one (if any) becomes a link is decided globally from all
# pairs at once (pipeline.select_links).

RANGE_RES_DEG = 0.5
FREE_TOL = 0.25            # m (+1 % of range) a point may sit in front of the other scan's surface
# benchmark A: correct links 0.5-10 % (A5-A6: 9.9 %, open doors/glass), wrong
# ones 6.7-14 %: it overlaps, so this only guards against gross errors; the
# end/long wall ratio is what separates turned/slid poses.
FREE_MAX = 0.15            # fraction of points in the other scan's free space -> implausible
                           # (correct poses on both benchmarks: <= 0.13 except one at 0.18)
FREE_MIN_POINTS = 500


def _range_image(prep: PreparedCloud, origin: np.ndarray) -> np.ndarray:
    """Nearest range seen by the scanner per 0.5° direction bin."""
    key = tuple(np.round(np.asarray(origin, dtype=np.float64), 3).tolist())
    cache = getattr(prep, "_range_img", None)
    if cache is not None and cache[0] == key:
        return cache[1]
    res = math.radians(RANGE_RES_DEG)
    H, W = int(math.ceil(math.pi / res)) + 1, int(math.ceil(2 * math.pi / res)) + 1
    v = np.asarray(prep.points, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    r = np.linalg.norm(v, axis=1)
    ok = r > 0.3
    v, r = v[ok], r[ok]
    ia = ((np.arctan2(v[:, 1], v[:, 0]) + math.pi) / res).astype(np.int64)
    ie = ((np.arcsin(np.clip(v[:, 2] / r, -1, 1)) + math.pi / 2) / res).astype(np.int64)
    img = np.full((H, W), np.inf, dtype=np.float32)
    order = np.argsort(-r)  # write far first, near last: nearest wins
    img[ie[order], ia[order]] = r[order]
    prep._range_img = (key, img)  # type: ignore[attr-defined]
    return img


def free_space_fraction(query: np.ndarray, prep: PreparedCloud, origin: np.ndarray, sample: int = 30_000) -> tuple[float, int]:
    """Fraction of ``query`` points (in prep's frame) lying in space prep's
    scanner saw *through*: there, prep measured a surface further away."""
    if len(query) > sample:
        query = query[np.random.default_rng(0).choice(len(query), sample, replace=False)]
    res = math.radians(RANGE_RES_DEG)
    img = _range_image(prep, origin)
    v = np.asarray(query, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    r = np.linalg.norm(v, axis=1)
    ok = r > 0.5
    v, r = v[ok], r[ok]
    ia = ((np.arctan2(v[:, 1], v[:, 0]) + math.pi) / res).astype(np.int64)
    ie = ((np.arcsin(np.clip(v[:, 2] / r, -1, 1)) + math.pi / 2) / res).astype(np.int64)
    seen = img[ie, ia]
    valid = np.isfinite(seen)
    n = int(valid.sum())
    if n == 0:
        return 0.0, 0
    viol = valid & (r < seen - (FREE_TOL + 0.01 * r))
    return float(viol.sum()) / n, n


# A pose that is right only because the geometry repeats (stacked landings,
# identical bays) contradicts the other scan where the repetition ends, and
# that is diluted in the global see-through fraction.  Per 0.5 m height slice
# it is not: on the benchmarks 44 of 51 wrong poses had a slice >= 30 %, 3 of
# 54 correct ones did.  Used for whole-merge checks, not as a single-link gate.
SLICE_H = 0.5             # m
SLICE_MIN_POINTS = 300
CONTRA_SLICE = 0.45       # worst slice fraction ...
CONTRA_GLOBAL = 0.05      # ... and overall fraction for a pair to contradict


def slice_contradiction(pa: PreparedCloud, pb: PreparedCloud, T: np.ndarray,
                        origin_a: np.ndarray | None = None, origin_b: np.ndarray | None = None,
                        sample: int = 30_000) -> tuple[float, float]:
    """(global, worst-slice) fraction of points of one scan lying in space the
    other scanner saw through, both directions, slices of SLICE_H in the
    receiving scan's frame."""
    oa = np.zeros(3) if origin_a is None else np.asarray(origin_a, dtype=np.float64)
    ob = np.zeros(3) if origin_b is None else np.asarray(origin_b, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    res = math.radians(RANGE_RES_DEG)
    g_all, worst = 0.0, 0.0
    for src, dst, M, o in ((pa, pb, T, ob), (pb, pa, invert(T), oa)):
        P = np.asarray(src.points, dtype=np.float64)
        if len(P) > sample:
            P = P[np.random.default_rng(0).choice(len(P), sample, replace=False)]
        Q = P @ M[:3, :3].T + M[:3, 3]
        img = _range_image(dst, o)
        v = Q - o
        r = np.linalg.norm(v, axis=1)
        ok = r > 0.5
        v, r, z = v[ok], r[ok], Q[ok, 2]
        if len(r) == 0:
            continue
        ia = ((np.arctan2(v[:, 1], v[:, 0]) + math.pi) / res).astype(np.int64)
        ie = ((np.arcsin(np.clip(v[:, 2] / r, -1, 1)) + math.pi / 2) / res).astype(np.int64)
        seen = img[ie, ia]
        valid = np.isfinite(seen)
        if not valid.any():
            continue
        viol = valid & (r < seen - (FREE_TOL + 0.01 * r))
        g_all = max(g_all, float(viol.sum() / valid.sum()))
        zb = np.floor(z[valid] / SLICE_H).astype(np.int64)
        vb = viol[valid]
        if len(zb):
            zb -= zb.min()
            n = np.bincount(zb)
            k = np.bincount(zb, weights=vb.astype(np.float64))
            m = n >= SLICE_MIN_POINTS
            if m.any():
                worst = max(worst, float((k[m] / n[m]).max()))
    return g_all, worst


def _dir_name(v: Sequence[float] | None) -> str:
    if not v:
        return ""
    k = int(np.argmax(np.abs(np.asarray(v, dtype=np.float64))))
    return (" sideways", " sideways", " up/down", " round")[k]


def hypothesis_verdict(ev: dict[str, Any], accept: float) -> dict[str, Any]:
    """Is this single hypothesis *plausible*?  (Not: is it the link.)

    Same physical tests as before, minus the pair-local ambiguity margin
    (the global selection resolves ambiguity), plus the see-through test.
    """
    v = link_verdict({**ev, "margin": 1.0}, accept)
    fs = ev.get("see_through")
    if v["accepted"] and fs is not None and float(fs) > FREE_MAX:
        v = {**v, "accepted": False, "rejection_reason": f"sees through the other scan's walls ({float(fs) * 100:.0f}% of points)"}
    weak = ev.get("weak_m2")
    if v["accepted"] and weak is not None and float(weak) < WEAK_MIN_M2:
        v = {**v, "accepted": False,
             "rejection_reason": f"not pinned down: only {float(weak):.1f} m² of surface stops it sliding{_dir_name(ev.get('weak_dir'))}"}
    gain = ev.get("alt_gain")
    if v["accepted"] and gain is not None and float(gain) > -ALT_TIE:
        v = {**v, "accepted": False,
             "rejection_reason": f"ambiguous: a pose {float(ev.get('alt_shift_m') or 0):.2f} m / {float(ev.get('alt_turn_deg') or 0):.1f}° away fits as well"}
    return {"plausible": bool(v["accepted"]), "reason": v["rejection_reason"], "flat_only": v["flat_only"],
            "area_ok": v["area_ok"], "slid": v.get("slid", False), "surface_bad": v.get("surface_bad", False)}


# --------------------------------------------------------------------------- #
# physical evidence: is the pose pinned down, and is it the best one nearby?
# --------------------------------------------------------------------------- #

PHYS_SAMPLE = 40_000      # moving points used for the constraint / local search
# Chosen from every stored candidate of the two benchmark buildings:
WEAK_MIN_M2 = 0.5         # correct poses: >= 0.9 m² of surface resist the weakest move
ALT_TIE = 0.01            # another pose nearby within 1 % inliers: a tie
ALT_BETTER = 0.01         # another pose nearby fits >= 1 % better: take it
_PHYS_CACHE: dict[str, Any] = {}


def _phys_inputs(job: "PairJob") -> tuple[np.ndarray, Any, cKDTree]:
    voxel = float(job.voxel)
    ks, kd = f"src|{job.path_a}", f"dst|{job.path_b}|{voxel}"
    if ks not in _PHYS_CACHE:
        _PHYS_CACHE[ks] = _fit_sample(job.path_a)
    if kd not in _PHYS_CACHE:
        dst = _cloud_at(job.path_b, voxel, voxel)
        _PHYS_CACHE[kd] = (dst, cKDTree(np.asarray(dst.points)))
    if len(_PHYS_CACHE) > 24:
        for k in list(_PHYS_CACHE)[:8]:
            _PHYS_CACHE.pop(k, None)
    dst, tree = _PHYS_CACHE[kd]
    return _PHYS_CACHE[ks], dst, tree


def _fit_sample(path: str, n: int = PHYS_SAMPLE) -> np.ndarray:
    pts = np.load(path).astype(np.float64)
    if len(pts) > n:
        pts = pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]
    return pts


def pose_fit(src: np.ndarray, dst: Any, T: np.ndarray, voxel: float, tree: cKDTree | None = None) -> dict[str, float]:
    """Inlier fraction and residual of ``src`` (moving sample) on ``dst``
    (open3d cloud with normals) under ``T``: point-to-plane |r| < voxel/2
    within 2 voxels counts as on the surface."""
    D = np.asarray(dst.points)
    N = np.asarray(dst.normals)
    tree = tree if tree is not None else cKDTree(D)
    P = src @ T[:3, :3].T + T[:3, 3]
    d, j = tree.query(P, distance_upper_bound=2.0 * voxel, workers=KD_WORKERS)
    near = np.isfinite(d)
    if near.sum() < 50:
        return {"inliers": 0.0, "rms_mm": float("nan"), "near": float(near.mean())}
    r = np.abs(np.einsum("ij,ij->i", N[j[near]], P[near] - D[j[near]]))
    ok = r < 0.5 * voxel
    return {"inliers": float(ok.sum() / len(P)), "rms_mm": float(np.sqrt(np.mean(r[ok] ** 2)) * 1000 if ok.any() else float("nan")),
            "near": float(near.mean())}


def pose_constraint(src: np.ndarray, dst: Any, T: np.ndarray, voxel: float, tree: cKDTree | None = None) -> dict[str, Any]:
    """How firmly the overlapping surfaces pin the pose (x, y, z, yaw).

    The point-to-plane information matrix H = JᵀJ over the inliers, with the
    yaw column scaled by the overlap radius so all four columns are in
    metres.  Its smallest eigenvalue times voxel² is the area of surface (m²)
    that resists a move in the weakest direction: ~0 for a corridor without
    visible ends (free along the corridor), large when walls face every way.
    Nothing here depends on the kind of building.
    """
    D = np.asarray(dst.points)
    Nn = np.asarray(dst.normals)
    tree = tree if tree is not None else cKDTree(D)
    P = src @ T[:3, :3].T + T[:3, 3]
    d, j = tree.query(P, distance_upper_bound=1.5 * voxel, workers=KD_WORKERS)
    ok = np.isfinite(d)
    if ok.sum() < 100:
        return {"weak_m2": 0.0, "weak_dir": None, "axes_m2": [0.0, 0.0, 0.0, 0.0], "n": int(ok.sum())}
    p, n = P[ok], Nn[j[ok]]
    r = np.einsum("ij,ij->i", n, p - D[j[ok]])
    keep = np.abs(r) < 0.5 * voxel
    p, n = p[keep], n[keep]
    if len(p) < 100:
        return {"weak_m2": 0.0, "weak_dir": None, "axes_m2": [0.0, 0.0, 0.0, 0.0], "n": int(len(p))}
    c = p.mean(axis=0)
    pc = p - c
    rad = float(np.sqrt(np.mean(pc[:, 0] ** 2 + pc[:, 1] ** 2))) or 1.0
    J = np.column_stack([n[:, 0], n[:, 1], n[:, 2], (n[:, 1] * pc[:, 0] - n[:, 0] * pc[:, 1]) / rad])
    H = J.T @ J
    # sample -> full cloud: each sampled point stands for len(src)/len(P) points
    w, V = np.linalg.eigh(H)
    area = float(voxel) ** 2
    weak = V[:, 0]
    return {
        "weak_m2": float(w[0] * area),
        "weak_dir": [float(x) for x in weak],          # (x, y, z, yaw·r) unit vector
        "axes_m2": [float(H[i, i] * area) for i in range(4)],
        "n": int(len(p)),
        "radius_m": rad,
    }


LOCAL_SAMPLE = 4000       # moving points for the restarts (0 = all of PHYS_SAMPLE); engine 16: 37 % less search CPU, same plausible poses


def local_alternatives(src: np.ndarray, dst: Any, T: np.ndarray, voxel: float, weak_dir: Sequence[float] | None,
                       lock: bool = True, tree: cKDTree | None = None) -> dict[str, Any]:
    """Restart ICP around ``T``: one stair step up/down, half a floor, and
    along the weakest direction.  Returns the best distinct pose found and
    how its fit compares, so a pose stuck one step or one bay off is
    corrected, and a pose with an equally good neighbour is known to be
    ambiguous."""
    D = np.asarray(dst.points)
    Nn = np.asarray(dst.normals)
    tree = tree if tree is not None else cKDTree(D)
    base = pose_fit(src, dst, T, voxel, tree)
    offsets: list[np.ndarray] = []
    for dz in (-0.36, -0.18, 0.18, 0.36):
        offsets.append(np.array([0.0, 0.0, dz, 0.0]))
    if weak_dir is not None:
        v = np.asarray(weak_dir, dtype=np.float64)
        h = np.array([v[0], v[1], 0.0])
        if np.linalg.norm(h) > 0.3:
            h /= np.linalg.norm(h)
            for s_ in (-1.2, -0.6, -0.3, 0.3, 0.6, 1.2):
                offsets.append(np.array([h[0] * s_, h[1] * s_, 0.0, 0.0]))
        if abs(v[3]) > 0.3:
            for deg in (-4.0, 4.0):
                offsets.append(np.array([0.0, 0.0, 0.0, deg]))
    best = None
    fit_src = src
    if LOCAL_SAMPLE and len(src) > LOCAL_SAMPLE:
        fit_src = src[:: len(src) // LOCAL_SAMPLE]
    for o in offsets:
        T0 = np.asarray(T, dtype=np.float64).copy()
        if o[3]:
            T0 = yaw_about_center(T0, src, o[3])
        T0[:3, 3] += o[:3]
        T1 = T0
        for dist in (4.0 * voxel, 2.0 * voxel, 1.2 * voxel):
            if lock:
                T1, _f, _r = icp_level(fit_src, D, Nn, T1, dist, max_iteration=20, loss_k=dist, tree=tree)
            else:
                res = o3d.pipelines.registration.registration_icp(
                    make_cloud(fit_src), dst, dist, T1,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20))
                T1 = np.asarray(res.transformation)
        if _pose_close(T1, T, trans=0.05, rot_deg=0.5):
            continue
        f = pose_fit(src, dst, T1, voxel, tree)
        if best is None or f["inliers"] > best["fit"]["inliers"]:
            best = {"T": T1, "fit": f}
    out: dict[str, Any] = {"fit": base, "alt": None}
    if best is not None:
        d = invert(T) @ best["T"]
        out["alt"] = {
            "transform": best["T"].tolist(),
            "fit": best["fit"],
            "gain": float(best["fit"]["inliers"] - base["inliers"]),
            "shift_m": float(np.linalg.norm(d[:3, 3])),
            "dz_m": float(d[2, 3]),
            "turn_deg": float(rotation_angle_deg(d[:3, :3])),
        }
    return out


def yaw_about_center(T: np.ndarray, src: np.ndarray, deg: float) -> np.ndarray:
    c = (src @ T[:3, :3].T + T[:3, 3]).mean(axis=0)
    a = math.radians(deg)
    D = np.eye(4)
    D[:3, :3] = [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]
    D[:3, 3] = c - D[:3, :3] @ c
    return D @ T


LAZY_EVAL = True


def evaluate_hypothesis(pa: PreparedCloud, pb: PreparedCloud, job: PairJob, T: np.ndarray, source: str,
                        refit: bool = True, local: bool = True) -> dict[str, Any]:
    """Full evidence for one a→b transform (shared by every hypothesis source)."""
    voxel = float(job.voxel)
    T = np.asarray(T, dtype=np.float64)
    if refit and _levelled(pa) and _levelled(pb) and tilt_deg(T) > 0.005:
        # heavy matchers (FPFH, GICP, wall ICP) are 6-DoF: re-fit level
        try:
            T = np.asarray(_icp_cached(job.path_a, job.path_b, voxel, voxel, T, threshold_factor=2.5,
                                       max_iteration=40, lock=True).transformation)
        except Exception:
            T = level_transform(T)
    s = score_transform(pa.points, pb.points, T, voxel, a_structure=pa.structure, b_structure=pb.structure)
    wall, bands = structure_score(pa, pb, T, voxel)
    ds = direction_scores(pa, pb, T, voxel) or {}
    corners, corner_frac = corner_match_score(pa, pb, T)
    wa = shared_wall_area(pa, pb, T, voxel)
    ev: dict[str, Any] = {
        "transform": T.tolist(),
        "source": source,
        "score": float(s["score"]),
        "overlap": float(s["overlap"]),
        "overlap_a": float(s["overlap_a"]),
        "overlap_b": float(s["overlap_b"]),
        "rmse": float(s["rmse"]),
        "structure_overlap": float(s["structure_overlap"]),
        "wall_score": None if wall is None else float(wall),
        "wall_bands": bands,
        "wall_bands_cover": structure_band_scores(pa, pb, T, voxel, mode="max"),
        "wall_area": wa["area"],
        "wall_cover": max(wa["cover_a"], wa["cover_b"]),
        "corner_matches": int(corners),
        "corner_frac": float(corner_frac),
        "long_wall": ds.get("long"),
        "end_wall": ds.get("end"),
    }
    ev["rank"] = combined_rank(ev["score"], float(wall if wall is not None else ev["score"]),
                               int(corners) if pa.corners is not None and pb.corners is not None else None, ds)
    # (engine 16) a pose the wall evidence already rejects can only get worse; the
    # see-through, surface and restart checks (most of the search time) are
    # for poses that can still become links.  weak_m2 stands in for the
    # physical check that has not run yet, as it would clear the "slid" test.
    if LAZY_EVAL and not link_verdict({**ev, "margin": 1.0, "weak_m2": 1.0}, float(job.accept_score))["accepted"]:
        ev.update(hypothesis_verdict(ev, float(job.accept_score)))
        return ev
    # cheap gates first: skip the expensive checks for hopeless hypotheses
    if ev["score"] >= float(job.accept_score) * 0.5:
        A = np.asarray(pa.points, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
        Ti = invert(T)
        B = np.asarray(pb.points, dtype=np.float64) @ Ti[:3, :3].T + Ti[:3, 3]
        f_ab, n_ab = free_space_fraction(A, pb, job.origin_b)
        f_ba, n_ba = free_space_fraction(B, pa, job.origin_a)
        fs = [f for f, n in ((f_ab, n_ab), (f_ba, n_ba)) if n >= FREE_MIN_POINTS]
        ev["see_through"] = max(fs) if fs else None
        ev["accuracy"] = surface_accuracy(job.path_a, job.path_b, T, voxel)
        # physics: is the pose pinned down in every direction, and is it the
        # best fit in its neighbourhood (one stair step, one bay, a few °)?
        try:
            src, dst, tree = _phys_inputs(job)
            con = pose_constraint(src, dst, T, voxel, tree)
            ev["weak_m2"] = con["weak_m2"]
            ev["weak_dir"] = con["weak_dir"]
            if local:
                lock = _levelled(pa) and _levelled(pb)
                loc = local_alternatives(src, dst, T, voxel, con["weak_dir"], lock, tree)
                ev["fit_inliers"] = loc["fit"]["inliers"]
                alt = loc["alt"]
                if alt is not None:
                    ev["alt_gain"] = alt["gain"]
                    ev["alt_shift_m"] = alt["shift_m"]
                    ev["alt_dz_m"] = alt["dz_m"]
                    ev["alt_turn_deg"] = alt["turn_deg"]
                    if alt["gain"] >= ALT_BETTER:
                        # a better pose right next to this one: that is the
                        # real basin (ICP stopped one step / one bay short)
                        better = evaluate_hypothesis(pa, pb, job, np.asarray(alt["transform"]),
                                                     source + " +local", refit=False, local=False)
                        better["corrected_from"] = {"shift_m": alt["shift_m"], "dz_m": alt["dz_m"],
                                                    "turn_deg": alt["turn_deg"], "gain": alt["gain"]}
                        return better
        except Exception:
            import traceback as _tb
            _tb.print_exc()
    ev.update(hypothesis_verdict(ev, float(job.accept_score)))
    return ev


def _icp_ladder(job: PairJob, T: np.ndarray, factors: Sequence[float] = (4.0, 2.0, 1.0),
                lock: bool = False) -> np.ndarray:
    voxel = float(job.voxel)
    T = np.asarray(T, dtype=np.float64)
    for f in factors:
        try:
            # coarse levels only need to get into the basin
            T = _icp_cached(job.path_a, job.path_b, voxel, max(voxel * f, voxel), T,
                            threshold_factor=2.5, max_iteration=15 if f >= 2.0 else 40,
                            lock=lock).transformation
        except Exception:
            break
    return np.asarray(T, dtype=np.float64)


def _distinct(hyps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hyps = sorted(hyps, key=lambda h: h["rank"], reverse=True)
    out: list[dict[str, Any]] = []
    for h in hyps:
        if any(_pose_close(np.asarray(h["transform"]), np.asarray(o["transform"]), trans=0.15, rot_deg=1.5) for o in out):
            continue
        out.append(h)
    return out


def candidates_job(job: PairJob) -> dict[str, Any]:
    """Stage 1+2: floor-plan peaks → ICP → evaluated hypotheses (~2 s)."""
    return _candidates(job, _load_prepared(job.path_a, job.voxel), _load_prepared(job.path_b, job.voxel), "candidates")


LOCAL_R = 8.0   # m: local stage keeps points within this floor-plan radius of each scanner


def _crop_near(p: PreparedCloud, origin: np.ndarray | None, r: float) -> PreparedCloud:
    """A copy of ``p`` with only the points within ``r`` (floor plan) of its scanner."""
    import copy

    o = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)
    keep = lambda X: X[np.hypot(X[:, 0] - o[0], X[:, 1] - o[1]) < r]  # noqa: E731
    q = copy.copy(p)
    q.points = keep(np.asarray(p.points))
    q.structure = None if p.structure is None else keep(np.asarray(p.structure))
    if p.corners is not None:
        q.corners = keep(np.asarray(p.corners))
    q.__dict__.pop("_bev", None)
    return q


def local_job(job: PairJob) -> dict[str, Any]:
    """Engine 12 local stage: the candidate search on both scans cropped to
    LOCAL_R around their scanners.  Two neighbouring setups where one sees a
    big space (benchmark B, scan 033: 476k points) share only a few % of their points, so
    the full-cloud floor plan is dominated by look-alike placements 20 m away
    and the true pose scores 0.03 (not plausible).  Near the scanners the shared
    part dominates.  Run only for route neighbours the other matchers could
    not link (≈ 25 s a pair)."""
    pa = _crop_near(_load_prepared(job.path_a, job.voxel), getattr(job, "origin_a", None), LOCAL_R)
    pb = _crop_near(_load_prepared(job.path_b, job.voxel), getattr(job, "origin_b", None), LOCAL_R)
    out = _candidates(job, pa, pb, "local")
    for h in out["hyps"]:
        h["source"] = f"local {LOCAL_R:.0f} m: " + str(h.get("source", ""))
    return out


def _candidates(job: PairJob, pa: PreparedCloud, pb: PreparedCloud, stage: str) -> dict[str, Any]:
    t0 = time.time()
    out: dict[str, Any] = {"a": job.a, "b": job.b, "stage": stage, "hyps": []}
    level = min(pa.up_confidence, pb.up_confidence) >= LEVEL_MIN_CONF
    lock = _levelled(pa) and _levelled(pb)
    peaks = bev_search(pa, pb, k=5) if level else []
    out["bev_peak"] = peaks[0]["score"] if peaks else 0.0
    if not level:
        out["needs_heavy"] = "not levelled"
    elif not peaks or peaks[0]["score"] < BEV_GREY:
        out["no_overlap"] = True
    else:
        if peaks[0]["score"] < BEV_MIN:
            out["needs_heavy"] = "weak floor-plan overlap"
        # Cheap first: every peak gets coarse ICP (20 cm, 10 cm) and a cheap
        # rank; only the best two get the 5 cm ICP and the full evidence
        # (see-through, surface accuracy).  Most of the cost was full checks
        # of peaks that were never going to win.
        voxel = float(job.voxel)
        coarse = []
        for pk in peaks:
            if pk["score"] < max(BEV_GREY, peaks[0]["score"] * 0.6):
                continue
            # engine 11: every floor the two scans could share, not just the
            # one their lowest points suggest
            starts = [(1.0, 0.0)]
            if Z_PEAKS > 0:
                try:
                    starts = z_offsets(pa, pb, pk["T"]) or starts
                except Exception:
                    traceback.print_exc()
            for zrel, dz in starts:
                T = _icp_ladder(job, _shift_up(pk["T"], pb, dz), factors=(4.0, 2.0), lock=lock)
                s = score_transform(pa.points, pb.points, T, voxel, a_structure=pa.structure, b_structure=pb.structure)
                ds = direction_scores(pa, pb, T, voxel) or {}
                coarse.append((combined_rank(s["score"], s["structure_overlap"], None, ds), T, {**pk, "dz": dz, "zrel": zrel}))
        coarse.sort(key=lambda c: c[0], reverse=True)
        # the best two overall, plus the best of every other height: the
        # right floor often overlaps less than a wrong one
        pick: list[tuple[float, np.ndarray, dict[str, Any]]] = []
        for c in coarse:
            if len(pick) < 2 or all(abs(float(c[1][2, 3]) - float(p[1][2, 3])) > 0.4 for p in pick):
                if not any(_pose_close(c[1], p[1], trans=0.15, rot_deg=1.5) for p in pick):
                    pick.append(c)
            if len(pick) >= 2 + Z_PEAKS:
                break
        hyps = []
        for _r, T, pk in pick:
            T = _icp_ladder(job, T, factors=(1.0,), lock=lock)
            T_wall = _structure_icp(pa, pb, voxel, max(voxel, 0.02), T)
            if T_wall is not None:
                T = np.asarray(T_wall, dtype=np.float64)
            h = evaluate_hypothesis(pa, pb, job, T, f"floorplan (peak {pk['score']:.2f}, yaw {pk['yaw']:.0f}°, dz {pk.get('dz', 0.0):+.1f} m)")
            h["bev"] = pk["score"]
            hyps.append(h)
        out["hyps"] = _distinct(hyps)
    out["seconds"] = round(time.time() - t0, 2)
    return out


def heavy_job(job: PairJob) -> dict[str, Any]:
    """Stage 5: every heavy matcher (FPFH+FGR/RANSAC, Manhattan, spectral,
    corners, slide/flip, GICP, colour ICP), all their distinct poses evaluated."""
    t0 = time.time()
    pa = _load_prepared(job.path_a, job.voxel)
    pb = _load_prepared(job.path_b, job.voxel)
    r = register_pair_job(job)
    poses = [(np.asarray(r["transform"]), r.get("method", "heavy"))]
    for alt in r.get("alternatives") or []:
        if alt.get("transform"):
            poses.append((np.asarray(alt["transform"]), alt.get("method", "heavy")))
    hyps = [evaluate_hypothesis(pa, pb, job, T, f"heavy: {m}") for T, m in poses]
    return {"a": job.a, "b": job.b, "stage": "heavy", "hyps": _distinct(hyps), "seconds": round(time.time() - t0, 2)}


def evaluate_job(job: PairJob) -> dict[str, Any]:
    """Evaluate one given transform (``job.init``), e.g. from a merged-model match."""
    t0 = time.time()
    pa = _load_prepared(job.path_a, job.voxel)
    pb = _load_prepared(job.path_b, job.voxel)
    T = _icp_ladder(job, job.init, factors=(2.0, 1.0), lock=_levelled(pa) and _levelled(pb))
    h = evaluate_hypothesis(pa, pb, job, T, job.source or "evaluated")
    return {"a": job.a, "b": job.b, "stage": "evaluate", "hyps": [h], "seconds": round(time.time() - t0, 2)}


# --------------------------------------------------------------------------- #
# manual assist: refine from a user-placed rough pose
# --------------------------------------------------------------------------- #


def _yaw_about(T: np.ndarray, up: np.ndarray, deg: float, pivot: np.ndarray) -> np.ndarray:
    """Rotate pose ``T`` by ``deg`` about ``up`` through ``pivot`` (target frame)."""
    up = np.asarray(up, dtype=np.float64)
    up = up / (np.linalg.norm(up) or 1.0)
    a = np.deg2rad(deg)
    K = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = pivot - R @ pivot
    return M @ np.asarray(T, dtype=np.float64)


# Snap trusts the hand placement: it searches only this far from it.
SNAP_RADIUS = 0.60      # m
SNAP_YAW = 5.0          # degrees
SNAP_CELL = 0.10        # m, floor-plan cell of the local search


def _vertical_points(points: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points, normals, wall mask) of a cloud at ``voxel`` with estimated normals."""
    pc = make_cloud(points).voxel_down_sample(float(voxel))
    estimate_normals(pc, float(voxel) * 2.0)
    P = np.asarray(pc.points, dtype=np.float64)
    N = np.asarray(pc.normals, dtype=np.float64)
    if len(N) != len(P):
        N = np.tile([0.0, 0.0, 1.0], (len(P), 1))
    return P, N, np.abs(N[:, 2]) < 0.3


def snap_to_model(task: dict[str, Any]) -> dict[str, Any]:
    """Snap a hand-placed scan onto the merged cloud of its neighbours.

    ``task``: path (moving scan bundle), targets [(path, world pose)], init
    (hand-placed world pose), voxel, level (bool: tilt locked).

    1. merge the neighbours in world coordinates (one model, so a scan that
       overlaps three neighbours a little is still well constrained);
    2. exhaustive floor-plan search of the walls within SNAP_RADIUS / SNAP_YAW
       of the placement (every shift and turn, no basin to escape from);
    3. level (or 6-DoF) robust point-to-plane ICP from the best few peaks and
       the placement itself, with shrinking correspondence distance;
    4. results that left the trust region are discarded; the best wall fit
       wins, ties go to the one closest to the placement.
    """
    t0 = time.time()
    voxel = float(task["voxel"])
    init = np.asarray(task["init"], dtype=np.float64)
    lock = bool(task.get("level", True)) and LOCK_TILT
    if lock:
        init = clamp_tilt(init, TILT_MAX_DEG)
    A = np.load(task["path"]).astype(np.float64)
    Aw = A @ init[:3, :3].T + init[:3, 3]
    lo = Aw.min(axis=0) - (SNAP_RADIUS + 1.0)
    hi = Aw.max(axis=0) + (SNAP_RADIUS + 1.0)
    parts = []
    for path, pose in task["targets"]:
        B = np.load(path).astype(np.float64)
        P = np.asarray(pose, dtype=np.float64)
        Bw = B @ P[:3, :3].T + P[:3, 3]
        keep = np.all((Bw >= lo) & (Bw <= hi), axis=1)
        if keep.any():
            parts.append(Bw[keep])
    if not parts:
        return {"error": "the neighbours do not reach the placed scan"}
    M, Mn, Mwall = _vertical_points(np.concatenate(parts), voxel)
    A_pts, _An, Awall = _vertical_points(A, voxel)
    tree = cKDTree(M)

    # --- 2. local floor-plan search -------------------------------------
    origin = init[:3, 3]
    cell = SNAP_CELL
    mxy = M[Mwall, :2]
    grid_lo = mxy.min(axis=0) - cell * 2 if len(mxy) else np.zeros(2)
    shape = tuple((np.ceil((mxy.max(axis=0) - grid_lo) / cell).astype(int) + 3).tolist()) if len(mxy) else (1, 1)
    raster = np.zeros(shape, dtype=bool)
    if len(mxy):
        ij = np.floor((mxy - grid_lo) / cell).astype(int)
        raster[ij[:, 0], ij[:, 1]] = True
        from scipy.ndimage import binary_dilation
        raster = binary_dilation(raster)          # ±1 cell tolerance
    aw = A_pts[Awall] @ init[:3, :3].T + init[:3, 3]
    aw_xy = np.unique(np.floor(aw[:, :2] / (cell / 2)).astype(np.int64), axis=0) * (cell / 2) + cell / 4
    steps = int(round(SNAP_RADIUS / cell))
    shifts = np.arange(-steps, steps + 1)
    peaks: list[tuple[float, float, float, float]] = []   # (score, yaw, dx, dy)
    if len(aw_xy) >= 50:
        for yaw in np.arange(-SNAP_YAW, SNAP_YAW + 1e-9, 0.5):
            c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            rel = aw_xy - origin[:2]
            rot = np.column_stack([c * rel[:, 0] - s_ * rel[:, 1], s_ * rel[:, 0] + c * rel[:, 1]]) + origin[:2]
            base = np.floor((rot - grid_lo) / cell).astype(int)
            for di in shifts:
                ii = base[:, 0] + di
                oki = (ii >= 0) & (ii < shape[0])
                for dj in shifts:
                    jj = base[:, 1] + dj
                    ok = oki & (jj >= 0) & (jj < shape[1])
                    hits = int(raster[ii[ok], jj[ok]].sum())
                    peaks.append((hits / len(aw_xy), float(yaw), di * cell, dj * cell))
    peaks.sort(reverse=True)
    seeds: list[tuple[np.ndarray, str]] = [(init, "your placement")]
    for sc, yaw, dx, dy in peaks:
        if len(seeds) >= 4:
            break
        D = np.eye(4)
        c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        D[:2, :2] = [[c, -s_], [s_, c]]
        D[:2, 3] = origin[:2] - D[:2, :2] @ origin[:2] + [dx, dy]
        T = D @ init
        if any(_pose_close(T, S, trans=0.2, rot_deg=1.0) for S, _ in seeds):
            continue
        seeds.append((T, f"floor-plan peak {sc:.2f} ({dx:+.1f}, {dy:+.1f} m, {yaw:+.1f}°)"))

    # --- 3. ICP from every seed -----------------------------------------
    # a fixed random sample is plenty for 4 parameters and keeps big
    # outdoor scans (~1 M points at 5 cm) to a few seconds
    rng = np.random.default_rng(0)
    src = A_pts if len(A_pts) <= 60_000 else A_pts[rng.choice(len(A_pts), 60_000, replace=False)]
    coarse_src = src if len(src) <= 15_000 else src[rng.choice(len(src), 15_000, replace=False)]
    wall_src = A_pts[Awall]

    def icp(T: np.ndarray) -> np.ndarray:
        for dist, iters in ((0.30, 20), (0.15, 20), (max(0.06, voxel * 1.2), 30)):
            pts = coarse_src if dist > 0.2 else src
            if lock:
                T, _f, _r = icp_level(pts, M, Mn, T, dist, max_iteration=iters, loss_k=dist, tree=tree)
            else:
                res = o3d.pipelines.registration.registration_icp(
                    make_cloud(pts), _with_normals(M, Mn), dist, T,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(
                        o3d.pipelines.registration.TukeyLoss(k=dist)),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
                T = np.asarray(res.transformation)
        return T

    def fit(T: np.ndarray) -> dict[str, float]:
        tol = voxel * 1.5
        out = {}
        for name, pts in (("all", A_pts), ("walls", wall_src)):
            if len(pts) == 0:
                out[name] = 0.0
                continue
            P = pts @ T[:3, :3].T + T[:3, 3]
            d, _ = tree.query(P, distance_upper_bound=tol * 2, workers=KD_WORKERS)
            out[name] = float(np.mean(d < tol))
        return out

    results = []
    for T0, label in seeds:
        T = icp(T0)
        moved = invert(init) @ T
        shift = float(np.linalg.norm(moved[:3, 3]))
        turn = rotation_angle_deg(moved[:3, :3])
        if shift > SNAP_RADIUS + 0.2 or turn > SNAP_YAW + 1.0:
            continue          # left the trust region: not what the user meant
        f = fit(T)
        results.append({"T": T, "label": label, "shift": shift, "turn": turn, **f})
    if not results:
        return {"error": "no fit within %.0f cm / %.0f° of your placement" % (SNAP_RADIUS * 100, SNAP_YAW),
                "seconds": round(time.time() - t0, 2)}
    best_wall = max(r["walls"] for r in results)
    close = [r for r in results if r["walls"] >= best_wall - 0.01]
    best = min(close, key=lambda r: r["shift"] + 0.02 * r["turn"])
    return {
        "pose": best["T"].tolist(),
        "method": "model snap from " + best["label"],
        "shift_m": best["shift"],
        "turn_deg": best["turn"],
        "inliers": best["all"],
        "wall_inliers": best["walls"],
        "tried": [{k: (v if k != "T" else None) for k, v in r.items() if k != "T"} for r in results],
        "seconds": round(time.time() - t0, 2),
    }


def _with_normals(P: np.ndarray, N: np.ndarray) -> o3d.geometry.PointCloud:
    pc = make_cloud(P)
    pc.normals = o3d.utility.Vector3dVector(np.asarray(N, dtype=np.float64))
    return pc


def evaluate_fixed_job(job: PairJob) -> dict[str, Any]:
    """Full pair evidence for exactly ``job.init`` (no ICP), in the manual
    assist's result format."""
    pa = _load_prepared(job.path_a, job.voxel)
    pb = _load_prepared(job.path_b, job.voxel)
    ev = evaluate_hypothesis(pa, pb, job, np.asarray(job.init, dtype=np.float64), job.source or "manual", refit=False,
                             local=False)
    if ev.get("accuracy") is None:
        ev["accuracy"] = surface_accuracy(job.path_a, job.path_b, np.asarray(ev["transform"]), float(job.voxel))
    return {**ev, "a": job.a, "b": job.b, "accepted": bool(ev.get("plausible")),
            "rejection_reason": ev.get("reason"), "manual": True, "margin": 1.0, "method": job.source or "manual"}


def refine_from_guess(job: PairJob) -> dict[str, Any]:
    """Local registration around a hand-placed pose (no global search).

    The user's placement is trusted to be roughly right (±1 m, ±10°), so we
    never jump to a far-away hypothesis the way pose-free matching can.  A
    small yaw sweep plus a wide 8x→1x ICP schedule absorbs hand-placement
    error; walls and GICP then tighten the result.
    """
    t0 = time.time()
    pa = _load_prepared(job.path_a, job.voxel)
    pb = _load_prepared(job.path_b, job.voxel)
    voxel = float(job.voxel)
    init = np.asarray(job.init, dtype=np.float64)
    lock = _levelled(pa) and _levelled(pb)

    def evaluate(T: np.ndarray) -> dict[str, float]:
        return score_transform(pa.points, pb.points, T, voxel, a_structure=pa.structure, b_structure=pb.structure)

    def icp_ladder(T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=np.float64)
        for factor in (8.0, 4.0, 2.0, 1.0):
            level = max(voxel * factor, voxel)
            try:
                T = _icp_cached(job.path_a, job.path_b, voxel, level, T, threshold_factor=2.5, max_iteration=40,
                                lock=lock).transformation
            except Exception:
                break
        return T

    up = pb.up if pb.up is not None else np.array([0.0, 0.0, 1.0])
    pivot = (np.asarray(pa.points, dtype=np.float64).mean(axis=0) @ init[:3, :3].T) + init[:3, 3]
    seeds: list[tuple[np.ndarray, str]] = []
    base = init
    if pa.up is not None and pb.up is not None and pa.up_confidence >= 0.35 and pb.up_confidence >= 0.35:
        base = snap_to_up(init, pa.up, pb.up, 1.0)
    for deg in (0.0, -4.0, 4.0, -9.0, 9.0):
        seeds.append((_yaw_about(base, up, deg, pivot), f"manual{deg:+.0f}°"))

    candidates: list[tuple[float, np.ndarray, str, dict[str, float]]] = []
    for T0, label in seeds:
        T1 = icp_ladder(T0)
        s = evaluate(T1)
        wall, _bands = structure_score(pa, pb, T1, voxel)
        rank = 0.55 * s["score"] + 0.45 * (wall if wall is not None else s["score"])
        candidates.append((rank, T1, label, s))
    candidates.sort(key=lambda c: c[0], reverse=True)
    _rank, best_T, method, best_s = candidates[0]

    T_wall = _structure_icp(pa, pb, voxel, max(voxel, 0.02), best_T)
    if T_wall is not None:
        sw = evaluate(T_wall)
        if sw["score"] >= best_s["score"] - 0.01:
            best_T, best_s, method = T_wall, sw, method + "+wall-icp"
    try:
        g = _gicp_cached(job.path_a, job.path_b, voxel, max(voxel, 0.02), best_T)
        gs = evaluate(g.transformation)
        if gs["score"] >= best_s["score"] - 0.01:
            best_T, best_s, method = g.transformation, gs, method + "+gicp"
    except Exception:
        pass
    if lock and tilt_deg(best_T) > 0.005:
        # wall ICP and GICP are 6-DoF: bring the result back to level
        try:
            best_T = np.asarray(_icp_cached(job.path_a, job.path_b, voxel, voxel, best_T, threshold_factor=2.5,
                                            max_iteration=40, lock=True).transformation)
        except Exception:
            best_T = level_transform(best_T)
        best_s = evaluate(best_T)
        method += "+level"

    wall, bands = structure_score(pa, pb, best_T, voxel)
    cover_bands = structure_band_scores(pa, pb, best_T, voxel, mode="max")
    corners, corner_frac = corner_match_score(pa, pb, best_T)
    wa = shared_wall_area(pa, pb, best_T, voxel)
    evidence = {
        "score": float(best_s["score"]),
        "wall_score": None if wall is None else float(wall),
        "wall_bands": bands,
        "wall_bands_cover": cover_bands,
        "wall_area": wa["area"],
        "wall_cover": max(wa["cover_a"], wa["cover_b"]),
        "corner_matches": int(corners),
        "long_wall": (direction_scores(pa, pb, best_T, voxel) or {}).get("long"),
        "end_wall": (direction_scores(pa, pb, best_T, voxel) or {}).get("end"),
        # The user chose the basin, so a second hypothesis elsewhere is not
        # ambiguity.
        "margin": 1.0,
    }
    accuracy = surface_accuracy(job.path_a, job.path_b, best_T, voxel)
    evidence["accuracy"] = accuracy
    verdict = link_verdict(evidence, job.accept_score)
    moved = best_T @ invert(init)
    shift = float(np.linalg.norm(moved[:3, 3]))
    turn = rotation_angle_deg(moved[:3, :3])
    return {
        "a": job.a,
        "b": job.b,
        "transform": np.asarray(best_T, dtype=np.float64).tolist(),
        "overlap": float(best_s["overlap"]),
        "overlap_a": float(best_s["overlap_a"]),
        "overlap_b": float(best_s["overlap_b"]),
        "rmse": float(best_s["rmse"]),
        "structure_overlap": float(best_s["structure_overlap"]),
        "corner_frac": float(corner_frac),
        "method": "manual:" + method,
        "seconds": round(time.time() - t0, 2),
        "accuracy": accuracy,
        "snap_shift_m": shift,
        "snap_turn_deg": float(turn),
        "manual": True,
        **evidence,
        **verdict,
    }


# --------------------------------------------------------------------------- #
# pair proposal + global pose solving
# --------------------------------------------------------------------------- #


def propose_pairs(ids: Sequence[str], poses: dict[str, np.ndarray] | None = None) -> list[tuple[str, str]]:
    ids = list(ids)
    if len(ids) <= 20 or poses is None:
        return list(itertools.combinations(ids, 2))
    centers = {i: poses[i][:3, 3] for i in ids}
    scored = []
    for a, b in itertools.combinations(ids, 2):
        scored.append((float(np.linalg.norm(centers[a] - centers[b])), a, b))
    scored.sort()
    keep = min(len(scored), max(80, len(ids) * 6))
    return [(a, b) for _, a, b in scored[:keep]]


def invert(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def rotation_angle_deg(R: np.ndarray) -> float:
    v = (np.trace(np.asarray(R)[:3, :3]) - 1.0) / 2.0
    return float(math.degrees(math.acos(float(np.clip(v, -1.0, 1.0)))))


def cycle_consistency(pairs: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Check every accepted link against alternate two-hop paths.

    For triangle a-b-c the link T_ab should equal inv(T_bc) @ T_ac.  Repeated
    facades produce a self-consistent-looking pair that still disagrees with
    the rest of the graph; this residual exposes exactly those links.  The
    result is diagnostic (``max_trans`` / ``max_rot``), not an automatic
    rejection, so a human can inspect uncertain links.
    """
    edges: dict[tuple[str, str], np.ndarray] = {}
    adjacency: dict[str, set[str]] = {}
    original: dict[str, dict[str, Any]] = {}
    for p in pairs:
        if not p.get("accepted"):
            continue
        a, b = p.get("a"), p.get("b")
        if not a or not b or a == b:
            continue
        T = np.asarray(p["transform"], dtype=np.float64)
        edges[(a, b)] = T
        edges[(b, a)] = invert(T)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
        key = f"{a}|{b}" if a < b else f"{b}|{a}"
        original[key] = {"a": min(a, b), "b": max(a, b)}

    out: dict[str, dict[str, float]] = {}
    for key, ab in original.items():
        a, b = ab["a"], ab["b"]
        T_ab = edges.get((a, b))
        if T_ab is None:
            continue
        min_t = float("inf")
        min_r = float("inf")
        max_t = 0.0
        max_r = 0.0
        best_via = None
        worst_via = None
        n = 0
        for c in adjacency.get(a, set()) & adjacency.get(b, set()):
            if c == a or c == b:
                continue
            T_ac = edges.get((a, c))
            T_bc = edges.get((b, c))
            if T_ac is None or T_bc is None:
                continue
            alt = invert(T_bc) @ T_ac
            diff = invert(T_ab) @ alt
            t = float(np.linalg.norm(diff[:3, 3]))
            r = rotation_angle_deg(diff[:3, :3])
            if (t, r) < (min_t, min_r):
                min_t, min_r, best_via = t, r, c
            if (t, r) > (max_t, max_r):
                max_t, max_r, worst_via = t, r, c
            n += 1
        if n:
            out[key] = {
                "min_trans": float(min_t),
                "min_rot": float(min_r),
                "max_trans": float(max_t),
                "max_rot": float(max_r),
                "best_via": best_via,
                "worst_via": worst_via,
                "n": n,
            }
    return out


def solve_global_poses(
    ids: Sequence[str],
    pairs: Iterable[dict[str, Any]],
    reference: str | None = None,
    *,
    optimize: bool = True,
    fallback: dict[str, np.ndarray] | None = None,
    preserve_poses: bool = False,
) -> dict[str, Any]:
    """Split scans into connected overlap groups and solve each group.

    Scans that share no reliable overlap are deliberately *not* forced into one
    coordinate system.  Each group gets its own reference frame and its own
    pose-graph solution.
    """
    ids = list(ids)
    accepted = [p for p in pairs if p.get("accepted")]
    if not ids:
        return {"groups": [], "poses": {}, "group_of": {}, "reference": None, "unconnected": []}

    adjacency: dict[str, list[tuple[str, np.ndarray, float]]] = {i: [] for i in ids}
    for p in accepted:
        a, b = p["a"], p["b"]
        if a not in adjacency or b not in adjacency:
            continue
        T_ab = np.asarray(p["transform"], dtype=np.float64)
        # manual placements always win: they form the spanning tree first
        w = 1e6 if p.get("manual") else float(p.get("score", p.get("overlap", 0.0)) or 0.0)
        adjacency[a].append((b, T_ab, w))
        adjacency[b].append((a, invert(T_ab), w))

    # connected components over accepted pairs
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[ry] = rx
        return True

    for i in ids:
        for nxt, _, _ in adjacency[i]:
            union(i, nxt)

    comps: dict[str, list[str]] = {}
    for i in ids:
        comps.setdefault(find(i), []).append(i)
    # Always keep the largest connected component first: it is the natural
    # "main" group for export.  The user reference only decides the frame
    # inside its own group.
    ordered = sorted(comps.values(), key=len, reverse=True)

    groups = []
    all_poses: dict[str, np.ndarray] = {}
    group_of: dict[str, int] = {}
    for gid, members in enumerate(ordered):
        members = sorted(members)
        if len(members) == 1 or not any(adjacency[m] for m in members):
            for m in members:
                all_poses[m] = (
                    np.asarray(fallback[m], dtype=np.float64).copy()
                    if fallback and m in fallback
                    else np.eye(4)
                )
                group_of[m] = gid
            groups.append(
                {
                    "id": gid,
                    "members": members,
                    "reference": members[0],
                    "edges": 0,
                    "residuals": {},
                    "unconnected": True,
                    "outliers": [],
                }
            )
            continue

        deg = {m: sum(1 for n, _, _ in adjacency[m] if find(n) == find(m)) for m in members}
        ref = reference if reference in members else max(members, key=lambda m: deg[m])

        if preserve_poses:
            # The caller already has refined, mutually consistent poses (e.g.
            # incremental merged-model growth).  Only recompute connectivity;
            # do not throw the solution away by re-composing the pairwise
            # measurements (which may contain weak/false links).
            poses = {}
            for m in members:
                poses[m] = (
                    np.asarray(fallback[m], dtype=np.float64).copy()
                    if fallback and m in fallback
                    else np.eye(4)
                )
            component_pairs = [p for p in accepted if p["a"] in poses and p["b"] in poses]
            for m in members:
                all_poses[m] = poses[m]
                group_of[m] = gid
            groups.append(
                {
                    "id": gid,
                    "members": members,
                    "reference": ref,
                    "edges": len(component_pairs),
                    "residuals": _loop_residuals(component_pairs, poses),
                    "unconnected": False,
                    "outliers": [],
                    "preserved": True,
                }
            )
            continue

        # Kruskal maximum spanning tree (only edges inside this component)
        tree: dict[str, list[tuple[str, np.ndarray, float]]] = {m: [] for m in members}
        all_edges = [
            (m, nxt, T, w)
            for m in members
            for (nxt, T, w) in adjacency[m]
            if find(nxt) == find(members[0])
        ]
        all_edges.sort(key=lambda e: -e[3])
        local_parent = {m: m for m in members}

        def lfind(x: str) -> str:
            while local_parent[x] != x:
                local_parent[x] = local_parent[local_parent[x]]
                x = local_parent[x]
            return x

        for cur, nxt, T_cur_nxt, w in all_edges:
            a, b = lfind(cur), lfind(nxt)
            if a == b:
                continue
            local_parent[b] = a
            # Keep the tree undirected: the BFS may start at either end of an
            # edge, and a directed tree leaves half the group without a pose
            # when the reference happens to be a leaf.
            tree[cur].append((nxt, T_cur_nxt, w))
            tree[nxt].append((cur, invert(T_cur_nxt), w))

        poses: dict[str, np.ndarray] = {ref: np.eye(4)}
        queue = [ref]
        while queue:
            cur = queue.pop(0)
            for nxt, T_cur_nxt, _ in tree[cur]:
                if nxt in poses:
                    continue
                poses[nxt] = poses[cur] @ invert(T_cur_nxt)
                queue.append(nxt)

        component_pairs = [
            p for p in accepted if p["a"] in poses and p["b"] in poses
        ]
        outliers = []
        if optimize and len(poses) >= 3:
            poses, info = optimize_pose_graph(
                list(poses.keys()), component_pairs, poses, reference=ref
            )
            outliers = info.get("outliers", [])

        residuals = _loop_residuals(component_pairs, poses)
        for m in members:
            if m not in poses:
                poses[m] = (
                    np.asarray(fallback[m], dtype=np.float64).copy()
                    if fallback and m in fallback
                    else np.eye(4)
                )
        all_poses.update(poses)
        for m in members:
            group_of[m] = gid
        groups.append(
            {
                "id": gid,
                "members": members,
                "reference": ref,
                "edges": len(component_pairs),
                "residuals": residuals,
                "unconnected": False,
                "outliers": outliers,
            }
        )

    return {
        "groups": groups,
        "poses": all_poses,
        "group_of": group_of,
        "reference": groups[0]["reference"] if groups else None,
        "unconnected": [m for g in groups if g["unconnected"] for m in g["members"]],
    }


def _loop_residuals(pairs: Iterable[dict[str, Any]], poses: dict[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for p in pairs:
        if not p.get("accepted"):
            continue
        a, b = p["a"], p["b"]
        if a not in poses or b not in poses:
            continue
        T_meas = np.asarray(p["transform"], dtype=np.float64)
        T_pred = invert(poses[b]) @ poses[a]
        delta = invert(T_meas) @ T_pred
        out[f"{a}|{b}"] = {
            "rot_deg": rotation_angle_deg(delta),
            "trans": float(np.linalg.norm(delta[:3, 3])),
        }
    return out


# --------------------------------------------------------------------------- #
# pose graph optimisation
# --------------------------------------------------------------------------- #


MANUAL_SOLVE_WEIGHT = 1e4   # vs 0.2-0.9 for automatic links: manual always wins


def optimize_pose_graph(
    ids: Sequence[str],
    pairs: Iterable[dict[str, Any]],
    poses: dict[str, np.ndarray],
    reference: str | None = None,
    *,
    max_nfev: int = 120,
    min_score: float = 0.12,
    max_rot_deg: float = 5.0,
    max_trans: float = 0.6,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Robust least-squares pose-graph optimisation with outlier rejection."""
    from scipy.optimize import least_squares

    ids = [i for i in ids if i in poses]
    if len(ids) < 3:
        return poses, {"edges": 0, "outliers": [], "cost": 0.0}
    if reference is None or reference not in poses:
        reference = ids[0]
    variables = [i for i in ids if i != reference]
    index = {i: k for k, i in enumerate(variables)}

    edges: list[tuple[str, str, np.ndarray, float]] = []
    for p in pairs:
        if not p.get("accepted"):
            continue
        a, b = p["a"], p["b"]
        if a not in poses or b not in poses:
            continue
        manual = bool(p.get("manual"))
        w = MANUAL_SOLVE_WEIGHT if manual else float(p.get("weight", p.get("score", p.get("overlap", 0.0))) or 0.0)
        if w < min_score * 0.35:
            continue
        edges.append((a, b, np.asarray(p["transform"], dtype=np.float64), math.sqrt(w), manual))
    if not edges:
        return poses, {"edges": 0, "outliers": [], "cost": 0.0}

    ref_pose = np.asarray(poses[reference], dtype=np.float64)

    def unpack(x: np.ndarray) -> dict[str, np.ndarray]:
        out = {reference: ref_pose.copy()}
        for i in variables:
            k = index[i] * 6
            P = np.eye(4)
            P[:3, :3] = Rotation.from_rotvec(x[k : k + 3]).as_matrix()
            P[:3, 3] = x[k + 3 : k + 6]
            out[i] = P
        return out

    def residuals(x: np.ndarray) -> np.ndarray:
        P = unpack(x)
        res = []
        for a, b, T_meas, w, _man in edges:
            delta = invert(T_meas) @ (invert(P[b]) @ P[a])
            rot = Rotation.from_matrix(delta[:3, :3]).as_rotvec() * 10.0  # 10 m lever arm
            res.extend((rot * w).tolist())
            res.extend((delta[:3, 3] * w).tolist())
        return np.asarray(res, dtype=np.float64)

    x0 = []
    for i in variables:
        P = np.asarray(poses[i], dtype=np.float64)
        x0.extend(Rotation.from_matrix(P[:3, :3]).as_rotvec().tolist())
        x0.extend(P[:3, 3].tolist())
    x0 = np.asarray(x0, dtype=np.float64)

    result = least_squares(
        residuals,
        x0,
        method="trf",
        loss="cauchy",
        f_scale=0.12,
        max_nfev=max_nfev,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    refined = unpack(result.x)

    # outlier rejection: re-run once without grossly inconsistent edges
    outliers = []
    keep = []
    for e in edges:
        a, b, T_meas, w, man = e
        delta = invert(T_meas) @ (invert(refined[b]) @ refined[a])
        rot = rotation_angle_deg(delta)
        trans = float(np.linalg.norm(delta[:3, 3]))
        if (rot > max_rot_deg or trans > max_trans) and not man:   # manual links are never outliers
            outliers.append({"a": a, "b": b, "rot_deg": rot, "trans": trans})
        else:
            keep.append(e)
    if outliers and len(keep) >= 2:
        edges = keep
        result = least_squares(
            residuals,
            result.x,
            method="trf",
            loss="cauchy",
            f_scale=0.12,
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        refined = unpack(result.x)

    return refined, {"edges": len(edges), "outliers": outliers, "cost": float(result.cost)}


# --------------------------------------------------------------------------- #
# multi-view refinement
# --------------------------------------------------------------------------- #


_POLISH_CACHE: dict[Any, Any] = {}


def _polish_cached(key: Any, make: Callable[[], Any]) -> Any:
    if key not in _POLISH_CACHE:
        _POLISH_CACHE[key] = make()
    return _POLISH_CACHE[key]


def _polish_worker_init(threads: int) -> None:
    global KD_WORKERS
    KD_WORKERS = threads
    o3d.utility.set_max_threads(threads)


def _polish_colours(ids: Sequence[str], neighbours: dict[str, list[str]]) -> list[list[str]]:
    """Batches of scans with no link between them (greedy colouring, most
    connected scan first).  ``ids`` order breaks ties, so the result is stable."""
    batches: list[list[str]] = []
    for i in sorted(ids, key=lambda x: -len(neighbours[x])):
        for batch in batches:
            if not set(neighbours[i]) & set(batch):
                batch.append(i)
                break
        else:
            batches.append([i])
    return batches


def _polish_fit(job: dict[str, Any]) -> np.ndarray | None:
    """Fit one scan to the union of its neighbours; the new pose, or None when
    the fit is rejected.  Clouds stay cached in the calling process."""
    path, pose, voxel = job["path"], job["pose"], job["voxel"]
    engine, level, cap, threads = job["engine"], job["level"], job["cap"], job["threads"]

    def cloud(q: str) -> np.ndarray:
        return _polish_cached(("raw", q), lambda: np.load(q).astype(np.float32, copy=False))

    def thin(q: str) -> np.ndarray:
        if not job["prethin"]:
            return cloud(q)
        return _polish_cached(("thin", q, voxel),
                              lambda: voxel_down(cloud(q), voxel * 1.5).astype(np.float32, copy=False))

    parts = []
    for q, P in job["neighbours"]:
        parts.append((thin(q) @ P[:3, :3].T + P[:3, 3]).astype(np.float32, copy=False))
    target = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

    def accept(T: np.ndarray, fitness: float) -> np.ndarray | None:
        if fitness <= 0.05:
            return None
        return clamp_tilt(T, cap) if level else np.asarray(T)

    sg = _small_gicp() if engine.startswith("sg-") else None
    if sg is not None and not (level and cap <= 0.0):
        try:
            tg, tree = sg.preprocess_points(
                target.astype(np.float64), downsampling_resolution=voxel * 1.5,
                num_neighbors=10, num_threads=threads)
            if tg.size() < 100:
                return None
            src = _polish_cached(("sg", path, voxel), lambda: sg.preprocess_points(
                cloud(path).astype(np.float64), downsampling_resolution=voxel,
                num_neighbors=10, num_threads=threads)[0])
            kw = dict(max_correspondence_distance=voxel * 3.0, num_threads=threads, max_iterations=40)
            if engine == "sg-vgicp":
                vm = sg.GaussianVoxelMap(voxel * 3.0)
                vm.insert(tg)
                r = sg.align(vm, src, pose, **kw)
            else:
                r = sg.align(tg, src, tree, pose, registration_type="GICP", **kw)
            return accept(np.asarray(r.T_target_source), float(r.num_inliers) / max(1, src.size()))
        except Exception:
            pass  # falls through to Open3D
    target = voxel_down(target, voxel * 1.5)
    if len(target) < 100:
        return None
    try:
        tgt = make_cloud(target)
        estimate_normals(tgt, voxel * 1.5)
        source = _polish_cached(("o3d", path, voxel), lambda: make_cloud(cloud(path)).voxel_down_sample(voxel))
        if level and cap <= 0.0:
            T_new, fit, _ = icp_level(
                np.asarray(source.points), np.asarray(tgt.points), np.asarray(tgt.normals),
                pose, voxel * 3.0, max_iteration=40, loss_k=voxel * 2.0,
            )
            return accept(T_new, fit)
        res = o3d.pipelines.registration.registration_icp(
            source, tgt, voxel * 3.0, pose,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.TukeyLoss(k=voxel * 2.0)),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=40),
        )
        return accept(np.asarray(res.transformation), res.fitness)
    except Exception:
        return None


def refine_multiview(
    ids: Sequence[str],
    paths: dict[str, str],
    pairs: Iterable[dict[str, Any]],
    poses: dict[str, np.ndarray],
    voxel: float,
    *,
    reference: str | None = None,
    rounds: int = 2,
    min_score: float = 0.12,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    tol_mm: float = 0.3,
    tol_deg: float = 0.005,
    on_round: Callable[[int, float, float], None] | None = None,
    levelled: Iterable[str] = (),
    tilt_max_deg: float | None = None,
) -> dict[str, np.ndarray]:
    """Gauss-Seidel multi-view ICP against each scan's accepted neighbours.

    The reference scan stays fixed so the solution does not drift.  Each
    scan is fitted to the union of its neighbours with a Tukey-weighted
    point-to-plane ICP, so furniture, people and glass that only one scan
    saw cannot drag a pose.  ``rounds`` is an upper bound: refinement stops
    as soon as no scan moved more than ``tol_mm`` / ``tol_deg`` in a round.
    """
    ids = [i for i in ids if i in poses]
    if not ids:
        return poses
    accepted = [
        p
        for p in pairs
        if p.get("accepted")
        and (p.get("manual") or p.get("area_ok")
             or (p.get("score", p.get("overlap", 0)) or 0) >= min_score)
    ]
    neighbours: dict[str, list[str]] = {i: [] for i in ids}
    for p in accepted:
        a, b = p["a"], p["b"]
        if a in neighbours and b in neighbours and b not in neighbours[a]:
            neighbours[a].append(b)
            neighbours[b].append(a)

    if reference is None or reference not in poses:
        reference = max(ids, key=lambda i: len(neighbours[i]))
    poses = {k: np.asarray(v, dtype=np.float64).copy() for k, v in poses.items()}
    # levelled scans: tilt comes from the IMU.  With ``tilt_max_deg`` None the
    # ICP only moves x, y, z, yaw; otherwise it is free but the tilt is
    # clamped to that bound after every step.
    level_ids = {i for i in levelled if i in poses} if LOCK_TILT else set()
    cap = TILT_MAX_DEG if tilt_max_deg is None else float(tilt_max_deg)
    for i in level_ids:
        poses[i] = clamp_tilt(poses[i], cap)
    engine_first, _, engine_last = POLISH_ICP.partition("+")
    cores = os.cpu_count() or 4
    workers = max(1, cores // 4) if POLISH_WORKERS <= 0 else int(POLISH_WORKERS)
    fitted = [i for i in ids if i != reference and neighbours[i]]
    batches = _polish_colours(fitted, neighbours) if workers > 1 else [[i] for i in ids if i in fitted]
    workers = min(workers, max((len(b) for b in batches), default=1))
    threads = int(POLISH_THREADS) if POLISH_THREADS > 0 else max(1, cores // workers)
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_polish_worker_init, initargs=(threads,))

    total = rounds * max(1, len(fitted))
    done = 0
    prev_mm = float("inf")
    try:
        for rnd in range(rounds):
            if should_cancel is not None and should_cancel():
                return poses
            moved_mm, moved_deg = 0.0, 0.0
            engine = engine_last if engine_last and rnd == rounds - 1 else engine_first
            for batch in batches:
                if should_cancel is not None and should_cancel():
                    return poses
                jobs = [{
                    "path": paths[i], "pose": poses[i], "voxel": voxel, "engine": engine,
                    "prethin": bool(POLISH_PRETHIN), "level": i in level_ids, "cap": cap, "threads": threads,
                    "neighbours": [(paths[j], poses[j]) for j in neighbours[i]],
                } for i in batch]
                if pool is None:
                    results = [_polish_fit(j) for j in jobs]
                else:
                    try:
                        results = list(pool.map(_polish_fit, jobs))
                    except BrokenProcessPool:
                        pool = None
                        results = [_polish_fit(j) for j in jobs]
                for i, T_new in zip(batch, results):
                    if T_new is not None:
                        D = invert(poses[i]) @ T_new
                        moved_mm = max(moved_mm, float(np.linalg.norm(D[:3, 3])) * 1000.0)
                        c = float(np.clip((np.trace(D[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
                        moved_deg = max(moved_deg, float(np.degrees(np.arccos(c))))
                        poses[i] = T_new
                done += len(batch)
                if progress:
                    progress(done / total)
            if on_round:
                on_round(rnd + 1, moved_mm, moved_deg)
            converged = moved_mm < tol_mm and moved_deg < tol_deg
            # small moves that no longer shrink are a scan creeping along a weakly
            # constrained direction (a corridor), not a better fit: stop there
            creeping = moved_mm < 2.0 and moved_mm > 0.85 * prev_mm
            prev_mm = moved_mm
            if (converged or creeping) and not (engine_last and rnd < rounds - 1):
                if progress:
                    progress(1.0)
                break
    finally:
        _POLISH_CACHE.clear()  # the next polish may see re-prepared clouds at the same paths
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    return poses


# After polish: fit each scan to every overlapping scan of its group, not only
# its links.  benchmark B held-out overlap over 6 splits: 0.1175 -> 0.1249 on pairs not
# fitted to; 8 rounds 0.1262 at twice the time (engine 21).
CONSENSUS_FIT = True
CONSENSUS_DIST = (0.10, 0.06, 0.06, 0.04)
CONSENSUS_POINTS = 30_000
CONSENSUS_RADIUS = 25.0
CONSENSUS_LEVEL = 0.05


def consensus_fit(
    members: Sequence[str],
    paths: dict[str, str],
    poses: dict[str, np.ndarray],
    voxel: float,
    *,
    fixed: Iterable[str] = (),
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, np.ndarray]:
    """Fit each scan in turn to all group members within CONSENSUS_RADIUS.

    Four degrees of freedom only (x, y, z, yaw), so tilt stays as it is.
    ``fixed`` scans are targets but never move.  Returns the new poses.
    """
    W = {m: np.asarray(poses[m], dtype=np.float64).copy() for m in members}
    clouds = {m: _cloud_at(paths[m], voxel, max(voxel, CONSENSUS_LEVEL)) for m in members}
    pts = {m: np.asarray(c.points, dtype=np.float64) for m, c in clouds.items()}
    nrm = {m: np.asarray(c.normals, dtype=np.float64) for m, c in clouds.items()}
    rng = np.random.default_rng(0)
    src = {m: p[rng.choice(len(p), min(len(p), CONSENSUS_POINTS), replace=False)] for m, p in pts.items()}
    frozen = set(fixed)
    for dist in CONSENSUS_DIST:
        for i in members:
            if should_cancel is not None and should_cancel():
                return W
            if i in frozen:
                continue
            near = [j for j in members
                    if j != i and np.linalg.norm(W[j][:3, 3] - W[i][:3, 3]) < CONSENSUS_RADIUS]
            if not near:
                continue
            dst = np.vstack([pts[j] @ W[j][:3, :3].T + W[j][:3, 3] for j in near])
            dst_n = np.vstack([nrm[j] @ W[j][:3, :3].T for j in near])
            T, _, _ = icp_level(src[i] @ W[i][:3, :3].T + W[i][:3, 3], dst, dst_n, np.eye(4), dist,
                                max_iteration=12, loss_k=dist)
            W[i] = T @ W[i]
    return W


def pair_transform_from_poses(poses: dict[str, np.ndarray], a: str, b: str) -> np.ndarray:
    return invert(poses[b]) @ poses[a]
