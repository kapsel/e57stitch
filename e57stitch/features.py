"""Feature extraction used both for registration and for the UI overlays.

Two kinds of "features" are produced for every scan:

* **dominant planes** (RANSAC) - walls, floors, facades
* **edge / high-curvature clusters** (normal-disagreement + DBSCAN) - window
  frames, corners, beams, furniture edges ...

The actual coarse alignment is done with FPFH descriptors (see ``register.py``),
which are computed on top of the same voxelised cloud.  The plane/edge
clusters are used to (a) visualise what the matcher is locking onto and
(b) score how many distinct features two aligned scans share.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


FEATURES_VERSION = 6


def make_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))


def voxel_down(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0 or len(points) == 0:
        return np.asarray(points, dtype=np.float32)
    pc = make_cloud(points).voxel_down_sample(float(voxel))
    return np.asarray(pc.points, dtype=np.float32)


def estimate_normals(pc: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    """Estimate normals only.

    Do *not* run ``orient_normals_consistent_tangent_plane`` here: it is very
    expensive on multi-million point clouds and is unnecessary for
    point-to-plane ICP (the squared residual is invariant to normal sign) and
    for the feature overlays.
    """
    radius = max(float(voxel) * 4.0, 0.05)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40))
    return pc


# --------------------------------------------------------------------------- #
# noise cleanup (reflections, glass, sparse fragments)
# --------------------------------------------------------------------------- #


def clean_points(
    points: np.ndarray,
    voxel: float,
    mode: str = "light",
    *,
    return_mask: bool = False,
):
    """Remove sparse/isolated points that are typical of glass/reflections.

    light  : keep points with at least a few neighbours in a small radius
    strong : additionally drop small DBSCAN clusters (fragmented specks)
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 100 or str(mode).lower() in ("off", "none", "false", "0", ""):
        mask = np.ones(len(pts), dtype=bool)
        return (pts, mask) if return_mask else pts

    mode = str(mode).lower()
    radius = max(2.5 * float(voxel), 0.10)
    min_neighbours = 4 if mode != "strong" else 6
    tree = cKDTree(pts)
    counts = tree.query_ball_point(pts, radius, return_length=True, workers=-1)
    mask = counts - 1 >= min_neighbours

    if mode == "strong" and mask.sum() > 100:
        kept = pts[mask]
        pc = make_cloud(kept)
        labels = np.asarray(pc.cluster_dbscan(eps=max(3.0 * voxel, 0.15), min_points=10))
        sizes = {}
        for label in labels:
            if label >= 0:
                sizes[int(label)] = sizes.get(int(label), 0) + 1
        min_cluster = 40
        good = np.array([sizes.get(int(l), 0) >= min_cluster for l in labels], dtype=bool)
        new_mask = np.zeros(len(pts), dtype=bool)
        new_mask[np.flatnonzero(mask)] = good
        mask = new_mask

    if return_mask:
        return pts[mask], mask
    return pts[mask]


def transfer_mask(source_clean: np.ndarray, target: np.ndarray, radius: float) -> np.ndarray:
    """Keep target points that have a cleaned source point nearby."""
    if len(source_clean) == 0 or len(target) == 0:
        return np.zeros(len(target), dtype=bool)
    tree = cKDTree(np.asarray(source_clean, dtype=np.float64))
    dist, _ = tree.query(np.asarray(target, dtype=np.float64), k=1, workers=-1)
    return dist <= radius


# --------------------------------------------------------------------------- #
# coarse registration preparation
# --------------------------------------------------------------------------- #


@dataclass
class PreparedCloud:
    """Clouds + descriptors cached for one scan during pairwise matching."""

    points: np.ndarray
    voxel: float
    coarse: o3d.geometry.PointCloud
    coarse_fpfh: o3d.geometry.pipeline.Feature
    structure: np.ndarray | None = None  # vertical (non-ground) points, for scoring
    up: np.ndarray | None = None  # gravity axis in this scan's local frame
    corners: np.ndarray | None = None  # vertical corner/column line points (Nx3)
    up_confidence: float = 0.0
    up_mode: str = "unknown"  # level | tilted | sideways | unknown
    tilt_deg: float = 0.0

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)


def estimate_gravity(
    points: np.ndarray,
    *,
    sample: int = 200_000,
    bin_size: float = 0.08,
) -> dict[str, Any]:
    """Estimate which local axis is vertical (gravity), without an IMU.

    Terrestrial scanners are usually levelled, but an instrument can also be
    tilted, sideways or upside down.  We look for a pair of large, isotropic,
    parallel planes on opposite sides of the scanner (floor + ceiling).  A
    mild prior towards the scanner's own +Z keeps ordinary level scans stable
    even when a sloped roof happens to be the largest plane.

    Returns the up axis (its sign is arbitrary for sideways scans), a
    confidence and a coarse classification.  When confidence is low the caller
    should fall back to unconstrained 6-DOF registration.
    """
    from collections import Counter

    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 500:
        return {
            "up": np.array([0.0, 0.0, 1.0]),
            "confidence": 0.0,
            "mode": "unknown",
            "tilt_deg": 0.0,
        }
    if len(pts) > sample:
        pts = pts[np.random.default_rng(11).choice(len(pts), sample, replace=False)]

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    voxel = float(np.clip(diag / 120.0, 0.08, 0.35))
    pc = make_cloud(pts).voxel_down_sample(voxel)
    if len(pc.points) > 90_000:
        idx = np.random.default_rng(12).choice(len(pc.points), 90_000, replace=False)
        pc = pc.select_by_index(idx)
    estimate_normals(pc, voxel * 2.5)
    P = np.asarray(pc.points, dtype=np.float64)
    N = np.asarray(pc.normals, dtype=np.float64)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)

    # Fold the normal hemisphere (n and -n describe the same plane).
    F = N.copy()
    F[F[:, 2] < 0] *= -1.0
    keys = np.round(F / bin_size).astype(np.int16)
    counts = Counter(map(tuple, keys))
    candidates: list[np.ndarray] = []
    for key, _count in counts.most_common(120):
        d = np.asarray(key, dtype=np.float64)
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            continue
        d /= norm
        if all(abs(float(d @ e)) < 0.97 for e in candidates):
            candidates.append(d)
    for axis in (
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ):
        if all(abs(float(axis @ e)) < 0.97 for e in candidates):
            candidates.append(axis)
    if not candidates:
        return {
            "up": np.array([0.0, 0.0, 1.0]),
            "confidence": 0.0,
            "mode": "unknown",
            "tilt_deg": 0.0,
        }

    def patch_score(d: np.ndarray) -> float:
        # Points whose surface faces along d (horizontal surfaces for true up).
        m = np.abs(N @ d) > 0.85
        if int(m.sum()) < 250:
            return 0.0
        PP = P[m]
        T = PP @ d
        helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = helper - float(helper @ d) * d
        norm_u = float(np.linalg.norm(u))
        if norm_u < 1e-6:
            return 0.0
        u /= norm_u
        w = np.cross(d, u)
        U, V = PP @ u, PP @ w
        bins = np.floor((T - T.min()) / 0.3).astype(np.int64)
        patches: list[tuple[float, float]] = []
        for bi in np.unique(bins):
            sel = bins == bi
            if int(sel.sum()) < 250:
                continue
            du = float(U[sel].max() - U[sel].min())
            dv = float(V[sel].max() - V[sel].min())
            if du < 0.5 or dv < 0.5:
                continue
            iso = min(du, dv) / max(du, dv)
            area = du * dv * iso
            if area < 4.0:
                continue
            patches.append((float(np.median(T[sel])), area))
        if not patches:
            return 0.0
        patches.sort(key=lambda x: -x[1])
        best = 0.0
        for i in range(min(6, len(patches))):
            for j in range(i + 1, min(6, len(patches))):
                ta, aa = patches[i]
                tb, ab = patches[j]
                if abs(ta) > 0.05 and abs(tb) > 0.05 and ta * tb > 0.0:
                    continue  # must be on opposite sides of the scanner
                height = abs(ta - tb)
                if not (1.0 <= height <= 9.0):
                    continue
                hp = 0.5 + 0.5 * math.exp(-(((height - 2.7) / 2.0) ** 2))
                best = max(best, math.sqrt(aa * ab) * hp)
        return float(best)

    z_axis = np.array([0.0, 0.0, 1.0])
    scored: list[tuple[float, float, float, np.ndarray]] = []
    for d in candidates:
        raw = patch_score(d)
        ang = math.degrees(math.acos(float(np.clip(abs(d @ z_axis), -1.0, 1.0))))
        prior = 1.0 if ang < 12.0 else max(0.15, math.exp(-(((ang - 12.0) / 35.0) ** 2)))
        scored.append((raw * prior, raw, ang, d))
    scored.sort(key=lambda x: -x[0])
    _, top_raw, top_ang, top_dir = scored[0]
    z_raw, z_dir = 0.0, z_axis.copy()
    for _s, raw, ang, d in scored:
        if ang < 12.0:
            z_raw, z_dir = raw, d
            break

    if top_ang > 25.0 and top_raw > max(2.0 * z_raw, 40.0):
        up = top_dir
        mode = "sideways" if top_ang > 60.0 else "tilted"
        confidence = float(min(1.0, top_raw / max(z_raw, 1.0) / 6.0))
    else:
        up = z_dir
        mode = "level"
        confidence = float(min(1.0, 0.45 + z_raw / 200.0))
    if abs(float(up[2])) > 0.7:
        up = up if up[2] >= 0 else -up
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    tilt = math.degrees(math.acos(float(np.clip(abs(up @ z_axis), -1.0, 1.0))))
    return {
        "up": up,
        "confidence": confidence,
        "mode": mode,
        "tilt_deg": float(tilt),
    }


def vertical_structure(points: np.ndarray, voxel: float, up: np.ndarray) -> np.ndarray | None:
    """Points on non-horizontal surfaces, relative to the given gravity axis."""
    try:
        score_voxel = max(float(voxel) * 3.0, 0.15)
        spc = make_cloud(points).voxel_down_sample(score_voxel)
        estimate_normals(spc, score_voxel)
        normals = np.asarray(spc.normals)
        spoints = np.asarray(spc.points)
        vertical = np.abs(normals @ np.asarray(up, dtype=np.float64)) < 0.6
        if vertical.sum() >= 500:
            return np.ascontiguousarray(spoints[vertical], dtype=np.float32)
    except Exception:
        pass
    return None


def prepare_registration(points: np.ndarray, voxel: float, coarse_factor: float = 8.0) -> PreparedCloud:
    points = np.asarray(points, dtype=np.float32)
    coarse_voxel = max(float(voxel) * coarse_factor, 0.15)
    # Cap the descriptor cloud so FPFH stays fast on huge scenes.
    pc = make_cloud(points).voxel_down_sample(coarse_voxel)
    while len(pc.points) > 45000 and coarse_voxel < 5.0:
        coarse_voxel *= 1.4
        pc = make_cloud(points).voxel_down_sample(coarse_voxel)
    if len(pc.points) < 200:
        pc = make_cloud(points)
    estimate_normals(pc, coarse_voxel)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pc,
        o3d.geometry.KDTreeSearchParamHybrid(radius=coarse_voxel * 5.0, max_nn=100),
    )

    # Gravity axis (usually local +Z, but scanners can be tilted or sideways).
    grav = estimate_gravity(points)
    up = np.asarray(grav["up"], dtype=np.float64)

    # Vertical structure used by the pair score.  A large shared floor plane
    # otherwise lets badly shifted alignments look "good".  The filter follows
    # the detected gravity axis instead of assuming world Z.
    structure = vertical_structure(points, voxel, up)

    return PreparedCloud(
        points=points,
        voxel=float(voxel),
        coarse=pc,
        coarse_fpfh=fpfh,
        structure=structure,
        up=up,
        up_confidence=float(grav["confidence"]),
        up_mode=str(grav["mode"]),
        tilt_deg=float(grav["tilt_deg"]),
    )


# --------------------------------------------------------------------------- #
# plane / edge cluster detection
# --------------------------------------------------------------------------- #


@dataclass
class PlaneFeature:
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    inliers: int
    extent: np.ndarray
    points: np.ndarray  # sampled inlier points for the UI overlay
    area: float = 0.0
    significant: bool = False


@dataclass
class EdgeCluster:
    centroid: np.ndarray
    points: np.ndarray
    size: int
    extent: np.ndarray
    direction: np.ndarray
    length: float
    significant: bool = False


@dataclass
class CornerFeature:
    """A vertical room corner / column / strong edge line."""

    point: np.ndarray          # a point on the line, in scan-local coordinates
    support: int = 0
    height: float = 0.0
    significant: bool = False


@dataclass
class ScanFeatures:
    planes: list[PlaneFeature] = field(default_factory=list)
    edges: list[EdgeCluster] = field(default_factory=list)
    corners: list[CornerFeature] = field(default_factory=list)
    edge_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float32))
    plane_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float32))
    edge_point_ids: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float32))
    plane_ids: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))

    def summary(self) -> dict[str, Any]:
        return {
            "version": FEATURES_VERSION,
            "planes": [
                {
                    "normal": p.normal.tolist(),
                    "offset": float(p.offset),
                    "centroid": p.centroid.tolist(),
                    "inliers": int(p.inliers),
                    "extent": p.extent.tolist(),
                    "area": float(p.area),
                    "significant": bool(p.significant),
                }
                for p in self.planes
            ],
            "edges": [
                {
                    "centroid": e.centroid.tolist(),
                    "size": int(e.size),
                    "extent": e.extent.tolist(),
                    "direction": e.direction.tolist(),
                    "length": float(e.length),
                    "significant": bool(e.significant),
                }
                for e in self.edges[:80]
            ],
            "corners": [
                {
                    "point": c.point.tolist(),
                    "support": int(c.support),
                    "height": float(c.height),
                    "significant": bool(c.significant),
                }
                for c in self.corners
            ],
            "edge_count": len(self.edges),
            "edge_significant": sum(1 for e in self.edges if e.significant),
            "plane_count": len(self.planes),
            "plane_significant": sum(1 for p in self.planes if p.significant),
            "plane_area": float(sum(p.area for p in self.planes if p.significant)),
        }


def detect_features(
    points: np.ndarray,
    voxel: float,
    *,
    up: np.ndarray | None = None,
    max_planes: int = 8,
    max_edge_points: int = 25000,
) -> ScanFeatures:
    """Detect dominant planes and clusters of high-curvature/edge points."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 100:
        return ScanFeatures()

    work = voxel_down(pts, max(float(voxel), 0.02))
    if len(work) > 400_000:
        rng = np.random.default_rng(1)
        work = work[rng.choice(len(work), 400_000, replace=False)]

    pc = make_cloud(work)
    estimate_normals(pc, voxel)
    normals = np.asarray(pc.normals, dtype=np.float32)

    # --- edge score: how much a point's normal disagrees with its neighbours
    tree = cKDTree(work)
    k = min(12, len(work))
    _, idx = tree.query(work, k=k, workers=-1)
    neigh_normals = normals[idx]
    agree = np.abs(np.einsum("nkj,nj->nk", neigh_normals, normals)).mean(axis=1)
    score = 1.0 - agree
    thresh = np.quantile(score, 0.96)
    edge_mask = score >= max(thresh, 0.08)
    edge_pts = work[edge_mask]
    edge_ids = np.flatnonzero(edge_mask)

    edges: list[EdgeCluster] = []
    if len(edge_pts) > 20:
        epc = make_cloud(edge_pts)
        labels = np.asarray(
            epc.cluster_dbscan(eps=max(voxel * 6.0, 0.15), min_points=12, print_progress=False)
        )
        for label in np.unique(labels):
            if label < 0:
                continue
            sel = labels == label
            if sel.sum() < 12:
                continue
            cluster = edge_pts[sel]
            centroid = cluster.mean(axis=0)
            cov = np.cov((cluster - centroid).T)
            vals, vecs = np.linalg.eigh(cov)
            direction = vecs[:, int(np.argmax(vals))]
            extent = cluster.max(axis=0) - cluster.min(axis=0)
            length = float(np.sqrt(max(vals.max(), 0.0)) * 2.0)
            # Tiny blobs are not reliable evidence.  A feature has to be a
            # reasonably long/dense edge (window frame, beam, kerb, column ...).
            significant = (
                int(sel.sum()) >= 60
                and length >= 0.35
                and float(extent.max()) >= 0.40
            )
            edges.append(
                EdgeCluster(
                    centroid=centroid.astype(np.float32),
                    points=cluster.astype(np.float32),
                    size=int(sel.sum()),
                    extent=extent.astype(np.float32),
                    direction=direction.astype(np.float32),
                    length=length,
                    significant=bool(significant),
                )
            )
        edges.sort(key=lambda e: e.size, reverse=True)

    # --- dominant planes
    planes: list[PlaneFeature] = []
    remaining = pc
    for _ in range(max_planes):
        if len(remaining.points) < 500:
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=max(voxel * 2.0, 0.03), ransac_n=3, num_iterations=250
        )
        if len(inliers) < max(300, int(0.01 * len(remaining.points))):
            break
        inlier_pts = np.asarray(remaining.points)[inliers]
        normal = np.asarray(model[:3], dtype=np.float32)
        normal /= np.linalg.norm(normal) + 1e-12
        offset = float(model[3])
        centroid = inlier_pts.mean(axis=0)
        extent = inlier_pts.max(axis=0) - inlier_pts.min(axis=0)
        area = float(np.prod(np.sort(extent)[-2:]))
        significant = bool(
            len(inliers) >= max(400, int(0.004 * len(work))) and area >= 0.8
        )
        sample = inlier_pts
        if len(sample) > 4000:
            rng = np.random.default_rng(2)
            sample = sample[rng.choice(len(sample), 4000, replace=False)]
        planes.append(
            PlaneFeature(
                normal=normal,
                offset=offset,
                centroid=centroid.astype(np.float32),
                inliers=int(len(inliers)),
                extent=extent.astype(np.float32),
                points=sample.astype(np.float32),
                area=area,
                significant=significant,
            )
        )
        remaining = remaining.select_by_index(inliers, invert=True)

    # --- vertical room corners / vertical edges -----------------------------
    up_v = np.asarray(up if up is not None else [0.0, 0.0, 1.0], dtype=np.float64)
    up_v /= max(float(np.linalg.norm(up_v)), 1e-12)
    raw_corners: list[tuple[np.ndarray, int, float]] = []
    for e in edges:
        if not e.significant:
            continue
        direction = np.asarray(e.direction, dtype=np.float64)
        # Vertical columns/edges are ideal, but significant window frames and
        # wall junctions are useful horizontal anchors too.
        if abs(float(direction @ up_v)) > 0.65 or float(np.max(e.extent)) >= 0.5:
            raw_corners.append((np.asarray(e.centroid, dtype=np.float64), int(e.size), float(e.length)))
    vertical_planes = [
        p for p in planes if p.significant and abs(float(np.asarray(p.normal) @ up_v)) < 0.3
    ]
    for i in range(len(vertical_planes)):
        for j in range(i + 1, len(vertical_planes)):
            na = np.asarray(vertical_planes[i].normal, dtype=np.float64)
            nb = np.asarray(vertical_planes[j].normal, dtype=np.float64)
            if abs(float(na @ nb)) > 0.75:
                continue  # nearly parallel walls do not define a corner
            direction = np.cross(na, nb)
            nd = float(np.linalg.norm(direction))
            if nd < 1e-8:
                continue
            direction /= nd
            if abs(float(direction @ up_v)) < 0.75:
                continue
            if float(direction @ up_v) < 0:
                direction = -direction
            A = np.vstack([na, nb, direction])
            b = np.array(
                [
                    -float(vertical_planes[i].offset),
                    -float(vertical_planes[j].offset),
                    0.0,
                ],
                dtype=np.float64,
            )
            try:
                point = np.linalg.lstsq(A, b, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            # Support must be an actual line through the cloud, not a
            # theoretical plane intersection.
            shifted = work - point
            along = shifted @ direction
            perp = np.linalg.norm(shifted - along[:, None] * direction, axis=1)
            near = perp < max(0.15, voxel * 4.0)
            if int(near.sum()) < 30:
                continue
            hvals = (work[near] @ up_v)
            height = float(hvals.max() - hvals.min())
            if height < 0.5:
                continue
            raw_corners.append((point.astype(np.float64), int(near.sum()), height))
    corners: list[CornerFeature] = []
    for point, support, height in sorted(raw_corners, key=lambda x: -x[1]):
        point = np.asarray(point, dtype=np.float64)
        duplicate = False
        for existing in corners:
            delta = existing.point - point
            if float(np.linalg.norm(delta)) < 0.25:
                existing.support = max(existing.support, support)
                existing.height = max(existing.height, height)
                duplicate = True
                break
        if duplicate:
            continue
        if len(corners) >= 40:
            break
        corners.append(
            CornerFeature(
                point=point.astype(np.float32),
                support=int(support),
                height=float(height),
                significant=bool(support >= 60 and height >= 0.5),
            )
        )

    significant_edges = [e for e in edges if e.significant]
    edge_overlay = (
        np.concatenate([e.points for e in significant_edges], axis=0)
        if significant_edges
        else np.empty((0, 3), np.float32)
    )
    if len(edge_overlay) > max_edge_points:
        rng = np.random.default_rng(3)
        edge_overlay = edge_overlay[rng.choice(len(edge_overlay), max_edge_points, replace=False)]

    plane_overlay = (
        np.concatenate([p.points for p in planes], axis=0)
        if planes
        else np.empty((0, 3), np.float32)
    )
    plane_ids = (
        np.concatenate(
            [np.full(len(p.points), i, dtype=np.int32) for i, p in enumerate(planes)]
        )
        if planes
        else np.empty(0, dtype=np.int32)
    )

    return ScanFeatures(
        planes=planes,
        edges=edges,
        corners=corners,
        edge_points=np.ascontiguousarray(edge_overlay, dtype=np.float32),
        plane_points=np.ascontiguousarray(plane_overlay, dtype=np.float32),
        edge_point_ids=np.ascontiguousarray(edge_ids, dtype=np.int64),
        plane_ids=plane_ids,
    )


def match_features(
    a: ScanFeatures,
    b: ScanFeatures,
    transform: np.ndarray,
    *,
    tol: float = 0.6,
    up_a: np.ndarray | None = None,
    up_b: np.ndarray | None = None,
) -> dict[str, Any]:
    """Count edge clusters / planes of ``a`` that line up with ``b`` after ``transform``."""
    R = np.asarray(transform)[:3, :3]
    t = np.asarray(transform)[:3, 3]

    edge_matches: list[dict[str, Any]] = []
    ea = [e for e in a.edges if e.significant]
    eb = [e for e in b.edges if e.significant]
    if ea and eb:
        ca = np.array([e.centroid for e in ea], dtype=np.float64) @ R.T + t
        cb = np.array([e.centroid for e in eb], dtype=np.float64)
        tree = cKDTree(cb)
        dist, j = tree.query(ca, k=1)
        for i, (d, jj) in enumerate(zip(dist, j)):
            if d > tol:
                continue
            da = ea[i].direction @ R.T
            db = eb[jj].direction
            align = abs(float(np.dot(da, db)))
            ratio = ea[i].length / max(eb[jj].length, 1e-6)
            if align > 0.85 and 0.5 < ratio < 2.0:
                edge_matches.append(
                    {
                        "a": i,
                        "b": int(jj),
                        "dist": float(d),
                        "dist_mm": float(d) * 1000.0,
                        "align": align,
                        "length_ratio": ratio,
                        "a_centroid": ca[i].tolist(),
                        "b_centroid": cb[jj].tolist(),
                        "a_size": int(ea[i].size),
                        "b_size": int(eb[jj].size),
                        "a_length": float(ea[i].length),
                    }
                )

    plane_matches: list[dict[str, Any]] = []
    pa = [p for p in a.planes if p.significant]
    pb = [p for p in b.planes if p.significant]
    # A floor/ceiling plane is invariant under yaw and gives false confidence on
    # a flat site.  Only count near-vertical planes as wall evidence when gravity
    # is known; keep the all-plane diagnostic counts separately.
    if up_a is not None:
        up_a = np.asarray(up_a, dtype=np.float64)
        pa = [p for p in pa if abs(float(np.asarray(p.normal) @ up_a)) < 0.3]
    if up_b is not None:
        up_b = np.asarray(up_b, dtype=np.float64)
        pb = [p for p in pb if abs(float(np.asarray(p.normal) @ up_b)) < 0.3]
    if pa and pb:
        for i, plane_a in enumerate(pa):
            n_b = R @ plane_a.normal
            n_b = n_b / (np.linalg.norm(n_b) + 1e-12)
            offset_b = float(plane_a.offset - np.dot(n_b, t))
            c_a = R @ plane_a.centroid + t
            best = None
            for j, plane_b in enumerate(pb):
                align = abs(float(np.dot(n_b, plane_b.normal)))
                if align < 0.95:
                    continue
                plane_gap = abs(offset_b - float(plane_b.offset))
                if plane_gap > max(0.05, 1.5 * 0.05):
                    continue
                # in-plane centroid separation: two scans often see different
                # parts of a large wall, so only reject if the patch is far away
                dvec = c_a - plane_b.centroid
                dvec = dvec - np.dot(dvec, plane_b.normal) * plane_b.normal
                reach = max(2.5, 0.9 * (float(np.max(plane_a.extent)) + float(np.max(plane_b.extent))))
                if np.linalg.norm(dvec) > reach:
                    continue
                area = min(float(plane_a.area), float(plane_b.area))
                score = align - plane_gap * 10.0
                if best is None or score > best[0]:
                    best = (score, j, plane_gap, align, dvec, area)
            if best is not None:
                _, j, plane_gap, align, dvec, area = best
                plane_matches.append(
                    {
                        "a": i,
                        "b": int(j),
                        "dist": float(np.linalg.norm(dvec)),
                        "dist_mm": float(plane_gap) * 1000.0,
                        "centroid_dist_mm": float(np.linalg.norm(dvec)) * 1000.0,
                        "align": align,
                        "a_centroid": c_a.tolist(),
                        "b_centroid": pb[j].centroid.tolist(),
                        "a_area": float(plane_a.area),
                        "b_area": float(pb[j].area),
                        "a_extent": plane_a.extent.tolist(),
                        "b_extent": pb[j].extent.tolist(),
                        "area": area,
                    }
                )

    def _mean_mm(matches):
        if not matches:
            return None
        vals = sorted(m["dist_mm"] for m in matches)
        return {
            "mean_mm": float(np.mean(vals)),
            "median_mm": float(np.median(vals)),
            "p95_mm": float(np.percentile(vals, 95)),
            "n": len(vals),
        }

    edge_tight = [m for m in edge_matches if m["dist"] <= 0.10]
    plane_tight = [m for m in plane_matches if m["dist"] <= 0.10]

    return {
        "edges_matched": len(edge_matches),
        "edges_total": min(len(ea), len(eb)),
        "planes_matched": len(plane_matches),
        "planes_total": min(len(pa), len(pb)),
        "planes_area_matched": float(
            sum(min(m["a_area"], m["b_area"]) for m in plane_matches)
        ),
        "planes_area_total": float(
            min(
                sum(p.area for p in pa),
                sum(p.area for p in pb),
            )
        ),
        "edge_matches": edge_matches[:80],
        "plane_matches": plane_matches[:80],
        "edges_tight": len(edge_tight),
        "planes_tight": len(plane_tight),
        "edges_accuracy": _mean_mm(edge_tight) or _mean_mm(edge_matches),
        "planes_accuracy": _mean_mm(plane_tight) or _mean_mm(plane_matches),
    }
