"""Appearance check for a candidate pose, from the scans' own panoramas.

Repeated geometry (identical stair landings, bays, symmetric rooms) cannot
be told apart by shape, but its appearance differs: signs, posters, floor
numbers, dirt, light.  For a candidate a->b pose, a's panorama pixels (each a
measured 3D point) are re-observed in b's panorama; only points that b sees
at the same range count (occlusion-safe).  The normalised cross-correlation
of the grey values is robust to exposure differences between scans.

benchmark B measurements: accepted poses 0.57-0.70; a
geometrically perfect wrong floor 0.07; repeated identical landings 0.50-0.56
vs 0.66 for the right one.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

NCC_MIN = 0.20            # below: appearance does not match (wrong place)
NCC_MARGIN = 0.10         # this much better decides between equal geometric candidates
NCC_MIN_POINTS = 5000     # fewer co-visible points: no verdict
_CACHE: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}


def _load(folder: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    key = str(folder)
    if key not in _CACHE:
        from PIL import Image

        from .panorama import meta

        grey = np.asarray(Image.open(folder / "pano.jpg").convert("L"), dtype=np.float32)
        depth = np.load(folder / "depth.npy").astype(np.float32)
        _CACHE[key] = (grey, depth, meta(folder))
        while len(_CACHE) > 12:
            _CACHE.pop(next(iter(_CACHE)))
    return _CACHE[key]


def ncc(folder_a: Path, folder_b: Path, T_ab: np.ndarray, stride: int = 4, tol: float = 0.10) -> tuple[float | None, int]:
    """(NCC, co-visible points) of a's panorama re-observed by b under T_ab
    (a-local -> b-local).  None when too few points are co-visible."""
    ga, da, ma = _load(Path(folder_a))
    gb, db, mb = _load(Path(folder_b))
    T_ab = np.asarray(T_ab, dtype=np.float64)
    H, W = da.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    r = da[vs, us]
    ok = np.isfinite(r) & (r > 0.5) & (r < 25.0)
    vs, us, r = vs[ok], us[ok], r[ok]
    az = math.pi - (us + 0.5) / W * 2 * math.pi
    el = math.pi / 2 - (vs + 0.5) / H * math.pi
    pa = np.asarray(ma["origin"]) + r[:, None] * np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
    pb = pa @ T_ab[:3, :3].T + T_ab[:3, 3] - np.asarray(mb["origin"])
    rb = np.linalg.norm(pb, axis=1)
    azb = np.arctan2(pb[:, 1], pb[:, 0])
    elb = np.arcsin(np.clip(pb[:, 2] / np.maximum(rb, 1e-9), -1, 1))
    ub = ((math.pi - azb) / (2 * math.pi) * mb["w"]).astype(np.int64) % mb["w"]
    vb = np.clip(((math.pi / 2 - elb) / math.pi * mb["h"]).astype(np.int64), 0, mb["h"] - 1)
    seen = db[vb, ub]
    vis = np.isfinite(seen) & (np.abs(seen - rb) < tol + 0.01 * rb)
    n = int(vis.sum())
    if n < NCC_MIN_POINTS:
        return None, n
    x = ga[vs[vis], us[vis]]
    y = gb[vb[vis], ub[vis]]
    x = (x - x.mean()) / (x.std() + 1e-6)
    y = (y - y.mean()) / (y.std() + 1e-6)
    return float((x * y).mean()), n


FS_MAX = 0.30             # see-through share above this: the pose is physically impossible
FS_MIN_POINTS = 2000      # fewer comparable points: no verdict


def _see_through_one(folder_a: Path, folder_b: Path, T_ab: np.ndarray, stride: int) -> tuple[int, int]:
    ga, da, ma = _load(Path(folder_a))
    gb, db, mb = _load(Path(folder_b))
    H, W = da.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    r = da[vs, us]
    ok = np.isfinite(r) & (r > 0.5) & (r < 25.0)
    vs, us, r = vs[ok], us[ok], r[ok]
    az = math.pi - (us + 0.5) / W * 2 * math.pi
    el = math.pi / 2 - (vs + 0.5) / H * math.pi
    pa = np.asarray(ma["origin"]) + r[:, None] * np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
    pb = pa @ T_ab[:3, :3].T + T_ab[:3, 3] - np.asarray(mb["origin"])
    rb = np.linalg.norm(pb, axis=1)
    azb = np.arctan2(pb[:, 1], pb[:, 0])
    elb = np.arcsin(np.clip(pb[:, 2] / np.maximum(rb, 1e-9), -1, 1))
    ub = ((math.pi - azb) / (2 * math.pi) * mb["w"]).astype(np.int64) % mb["w"]
    vb = np.clip(((math.pi / 2 - elb) / math.pi * mb["h"]).astype(np.int64), 0, mb["h"] - 1)
    seen = db[vb, ub]
    f = np.isfinite(seen) & (seen > 0.5)
    tol = 0.10 + 0.01 * rb
    agree = int((f & (np.abs(seen - rb) < tol)).sum())
    viol = int((f & (rb < seen - 3 * tol)).sum())
    return agree, viol


def see_through(folder_a: Path, folder_b: Path, T_ab: np.ndarray, stride: int = 4) -> tuple[float | None, int]:
    """Free-space test of a candidate a->b pose, both ways, from the depth
    panoramas: a point one scan measured that lies clearly in front of the
    surface the other scan measured along the same ray was seen through - the
    pose is impossible.  Returns (worst see-through share, agreeing points);
    share None when too few points compare.

    benchmark B (engine 7): true pairs <= 0.20, half-storey-wrong
    poses 0.35-0.76.  A pose exactly one storey off on identical floors is NOT
    caught (0.01-0.06): nothing per pair can tell identical floors apart."""
    T_ab = np.asarray(T_ab, dtype=np.float64)
    a1, v1 = _see_through_one(folder_a, folder_b, T_ab, stride)
    a2, v2 = _see_through_one(folder_b, folder_a, np.linalg.inv(T_ab), stride)
    shares = [v / (v + a) for a, v in ((a1, v1), (a2, v2)) if a + v >= FS_MIN_POINTS]
    return (max(shares) if shares else None), a1 + a2
