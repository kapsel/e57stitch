"""Levelled 360° panoramas of a scan, where every pixel is a measured 3D point.

Built from the points themselves (not from camera images): each point is
projected onto a sphere around the scanner and the nearest point wins its
pixel.  That works for every E57, needs no camera calibration, and makes the
pixel <-> 3D mapping exact:

    u = (π − azimuth) / 2π · W        (azimuth = atan2(y, x) around the scanner)
    v = (π/2 − elevation) / π · H

Coordinates are the scan's levelled local frame (the same frame as its
registration bundle), so a picked point can be used directly with the scan's
transform.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .e57io import CACHE_DIR, ScanRef, read_full

PANO_W, PANO_H = 4096, 2048
PANO_VERSION = 1
PICK_RADIUS = 6  # px: a click on a hole or an edge snaps to the nearest measured pixel


def pano_dir(scan_key: str, apply_pose: bool) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in scan_key)
    return CACHE_DIR / "pano" / f"{safe}_{int(apply_pose)}_v{PANO_VERSION}"


def is_built(folder: Path) -> bool:
    return (folder / "meta.json").exists() and (folder / "pano.jpg").exists() and (folder / "depth.npy").exists()


def meta(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "meta.json").read_text())


def build(ref: ScanRef, apply_pose: bool, folder: Path) -> dict[str, Any]:
    """Render the panorama (pano.jpg), its depth (depth.npy) and meta.json."""
    from PIL import Image
    from scipy.ndimage import grey_dilation

    folder.mkdir(parents=True, exist_ok=True)
    data = read_full(ref, apply_pose=apply_pose, want_colors=True, want_intensity=True)
    xyz = np.asarray(data["points"], dtype=np.float64)
    origin = np.asarray(ref.pose, dtype=np.float64)[:3, 3] if (apply_pose and ref.has_pose) else np.zeros(3)
    d = xyz - origin
    r = np.linalg.norm(d, axis=1)
    ok = r > 0.05
    d, r = d[ok], r[ok]
    az = np.arctan2(d[:, 1], d[:, 0])
    el = np.arcsin(np.clip(d[:, 2] / r, -1.0, 1.0))
    u = np.floor((math.pi - az) / (2 * math.pi) * PANO_W).astype(np.int64) % PANO_W
    v = np.clip(np.floor((math.pi / 2 - el) / math.pi * PANO_H).astype(np.int64), 0, PANO_H - 1)

    cols = data.get("colors")
    has_colour = cols is not None and cols.ndim == 2 and cols.shape[1] == 3 and len(cols) == len(xyz)
    if has_colour:
        rgb = np.asarray(cols, dtype=np.uint8)[ok]
    else:
        inten = np.asarray(data.get("intensity") if data.get("intensity") is not None else [], dtype=np.float32)
        if len(inten) == len(xyz) and len(inten):
            inten = inten[ok]
            lo, hi = np.percentile(inten, [1, 99])
            g = np.clip((inten - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        else:
            g = np.clip(255 - r * 8, 40, 255).astype(np.uint8)
        rgb = np.repeat(g[:, None], 3, axis=1)

    # nearest point wins: write far-to-near so the last write is the nearest
    order = np.argsort(-r)
    img = np.zeros((PANO_H, PANO_W, 3), dtype=np.uint8)
    depth = np.full((PANO_H, PANO_W), np.nan, dtype=np.float32)
    img[v[order], u[order]] = rgb[order]
    depth[v[order], u[order]] = r[order].astype(np.float32)

    # display only: close the pin-holes between scan lines
    empty = ~np.isfinite(depth)
    for _ in range(2):
        filled = np.stack([grey_dilation(img[..., c], size=(3, 3)) for c in range(3)], axis=-1)
        img[empty] = filled[empty]
        empty = empty & ~(filled.max(axis=-1) > 0)

    Image.fromarray(img).save(folder / "pano.jpg", quality=88)
    np.save(folder / "depth.npy", depth.astype(np.float16))
    m = {"w": PANO_W, "h": PANO_H, "origin": origin.tolist(), "version": PANO_VERSION,
         "points": int(len(r)), "colour": bool(has_colour)}
    (folder / "meta.json").write_text(json.dumps(m))
    return m


def direction(u: float, v: float, w: int = PANO_W, h: int = PANO_H) -> np.ndarray:
    az = math.pi - (u / w) * 2 * math.pi
    el = math.pi / 2 - (v / h) * math.pi
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def pick(folder: Path, u: float, v: float, radius: int = PICK_RADIUS) -> dict[str, Any] | None:
    """3D point (scan-local) under pixel (u, v); snaps to the nearest measured
    pixel within ``radius``.  Among the pixels around the click the nearest
    surface wins, so a click on an object's edge does not land on the wall
    behind it."""
    m = meta(folder)
    depth = np.load(folder / "depth.npy", mmap_mode="r")
    w, h = int(m["w"]), int(m["h"])
    ui, vi = int(round(u)) % w, int(min(max(round(v), 0), h - 1))
    best = None
    for rad in (0, 1, 2, radius):
        v0, v1 = max(0, vi - rad), min(h, vi + rad + 1)
        us = [(ui + k) % w for k in range(-rad, rad + 1)]
        win = np.asarray(depth[v0:v1][:, us], dtype=np.float32)
        if not np.isfinite(win).any():
            continue
        ys, xs = np.nonzero(np.isfinite(win))
        dist = (ys + v0 - vi) ** 2 + np.array([min(abs(us[x] - ui), w - abs(us[x] - ui)) for x in xs]) ** 2
        # closest pixels to the click, then the nearest surface among them
        near = dist <= dist.min() + 2
        k = np.flatnonzero(near)[np.argmin(win[ys[near], xs[near]])]
        best = (us[xs[k]], ys[k] + v0, float(win[ys[k], xs[k]]))
        break
    if best is None:
        return None
    bu, bv, rng = best
    p = np.asarray(m["origin"], dtype=np.float64) + rng * direction(bu + 0.5, bv + 0.5, w, h)
    return {"local": p.tolist(), "u": bu + 0.5, "v": bv + 0.5, "range": rng}


def project(m: dict[str, Any], local: Sequence[float]) -> dict[str, float] | None:
    """Pixel of a scan-local 3D point (the inverse of ``pick``)."""
    d = np.asarray(local, dtype=np.float64) - np.asarray(m["origin"], dtype=np.float64)
    r = float(np.linalg.norm(d))
    if r < 1e-6:
        return None
    az = math.atan2(d[1], d[0])
    el = math.asin(max(-1.0, min(1.0, d[2] / r)))
    return {"u": ((math.pi - az) / (2 * math.pi) * m["w"]) % m["w"], "v": (math.pi / 2 - el) / math.pi * m["h"], "range": r}


def solve(world_pts: np.ndarray, local_pts: np.ndarray, level: bool = True) -> dict[str, Any]:
    """Pose T (local -> world) of the moving scan from marker pairs.

    Levelled scans: yaw + translation (2 pairs fix it).  Otherwise full
    rigid Kabsch (3 pairs).  Residual per pair in metres."""
    P = np.asarray(world_pts, dtype=np.float64)
    Q = np.asarray(local_pts, dtype=np.float64)
    need = 2 if level else 3
    if len(P) < need:
        return {"error": f"need at least {need} marker pairs"}
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    P0, Q0 = P - pc, Q - qc
    if level:
        s = float(np.sum(Q0[:, 0] * P0[:, 1] - Q0[:, 1] * P0[:, 0]))
        c = float(np.sum(Q0[:, 0] * P0[:, 0] + Q0[:, 1] * P0[:, 1]))
        a = math.atan2(s, c)
        R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    else:
        U, _S, Vt = np.linalg.svd(Q0.T @ P0)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
    t = pc - R @ qc
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    res = np.linalg.norm((Q @ R.T + t) - P, axis=1)
    return {"pose": T.tolist(), "residuals": res.tolist(), "rms": float(np.sqrt(np.mean(res ** 2)))}
