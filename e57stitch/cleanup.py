"""Non-destructive cleanup regions (boxes and spheres).

Regions are stored in registration-world coordinates and applied only at export
and preview time; removing a region brings the points back.  This is deliberate:
a manual cleanup mark should never silently alter the registration cloud or its
cached links.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np


def normalize_region(data: dict[str, Any]) -> dict[str, Any]:
    kind = str(data.get("kind") or "box").lower()
    if kind not in ("box", "sphere"):
        kind = "box"
    center = np.asarray(data.get("center") or [0, 0, 0], dtype=np.float64).reshape(3)
    region = {
        "id": str(data.get("id") or uuid.uuid4().hex[:10]),
        "kind": kind,
        "center": center.tolist(),
        "enabled": bool(data.get("enabled", True)),
        "op": "delete" if str(data.get("op") or "delete").lower() != "keep" else "keep",
        "scans": [str(s) for s in (data.get("scans") or [])] or None,
        "label": str(data.get("label") or "")[:80],
        "clip": bool(data.get("clip", False)),
    }
    if region["clip"]:
        # The clip box is the single project-wide "keep only this volume" box
        # (e.g. drop the ceiling): always an axis-aligned keep box on all scans.
        region["kind"] = "box"
        region["op"] = "keep"
        region["scans"] = None
        region["label"] = region["label"] or "clip box"
    if kind == "sphere":
        region["radius"] = float(data.get("radius") or 1.0)
    else:
        size = np.asarray(data.get("size") or [1, 1, 1], dtype=np.float64).reshape(3)
        region["size"] = np.abs(size).tolist()
    return region


def region_mask(points: np.ndarray, region: dict[str, Any]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    center = np.asarray(region["center"], dtype=np.float64)
    if region["kind"] == "sphere":
        r = float(region.get("radius") or 1.0)
        return np.sum((pts - center) ** 2, axis=1) <= r * r
    half = np.asarray(region.get("size") or [1, 1, 1], dtype=np.float64)
    return np.all(np.abs(pts - center) <= half, axis=1)


def removal_mask(
    points: np.ndarray, regions: list[dict[str, Any]] | None, *, scan: str | None = None
) -> np.ndarray:
    """True where points should be hidden/removed (delete regions only)."""
    if not regions or len(points) == 0:
        return np.zeros(len(points), dtype=bool)
    out = np.zeros(len(points), dtype=bool)
    keep_inside = np.zeros(len(points), dtype=bool)
    have_keep = False
    for region in regions:
        if not region.get("enabled", True):
            continue
        allowed = region.get("scans")
        if allowed and scan is not None and scan not in allowed:
            continue
        inside = region_mask(points, region)
        if region.get("op") == "keep":
            have_keep = True
            keep_inside |= inside
        else:
            out |= inside
    if have_keep:
        # keep regions define the only area that survives
        out |= ~keep_inside
    return out


def apply_cleanup(
    points: np.ndarray,
    colors: np.ndarray,
    intensity: np.ndarray,
    regions: list[dict[str, Any]] | None,
    *,
    scan: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = removal_mask(points, regions, scan=scan)
    if not mask.any():
        return points, colors, intensity
    keep = ~mask
    points = points[keep]
    if colors.size:
        colors = colors[keep]
    if intensity.size:
        intensity = intensity[keep]
    return points, colors, intensity
