"""E57 reading/writing and point-cloud export helpers.

The module deliberately keeps *one scan's* worth of full-resolution data in
memory at a time: the sample data sets contain ~9 million points per scan, so
everything else (previews, registration clouds) works on cached downsampled
copies.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pye57
import pye57.libe57 as libe57
from pye57.libe57 import (
    CompressedVectorNode,
    FloatNode,
    FloatPrecision,
    IntegerNode,
    SourceDestBuffer,
    StringNode,
    StructureNode,
    VectorNode,
    VectorSourceDestBuffer,
)

CACHE_DIR = Path(os.environ.get("E57STITCH_CACHE", Path.home() / ".cache" / "e57stitch"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #


@dataclass
class ScanRef:
    """A single scan inside an E57 file."""

    id: str
    file: str
    index: int
    name: str
    point_count: int
    fields: list[str]
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    has_pose: bool = False
    bounds: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pose"] = self.pose.tolist()
        return d

    @property
    def label(self) -> str:
        stem = Path(self.file).stem
        return f"{stem} #{self.index}" if self.index else stem


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def is_e57(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() == ".e57"


def probe_file(path: str, start_index: int = 0) -> list[ScanRef]:
    """Read only the headers of an E57 file (fast, no point data)."""
    path = str(Path(path).resolve())
    reader = pye57.E57(path)
    refs: list[ScanRef] = []
    for i in range(reader.scan_count):
        header = reader.get_header(i)
        pose = np.eye(4)
        has_pose = bool(_safe(lambda: header.has_pose(), False))
        if has_pose:
            w, x, y, z = (float(v) for v in header.rotation)
            rot = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ],
                dtype=np.float64,
            )
            pose[:3, :3] = rot
            pose[:3, 3] = np.asarray(header.translation, dtype=np.float64)

        bounds = None
        try:
            bounds = [
                [float(header.xMinimum), float(header.yMinimum), float(header.zMinimum)],
                [float(header.xMaximum), float(header.yMaximum), float(header.zMaximum)],
            ]
        except Exception:
            pass

        refs.append(
            ScanRef(
                id="",
                file=path,
                index=start_index + i,
                name=_safe(lambda: header.name, "") or Path(path).stem,
                point_count=int(header.point_count),
                fields=list(header.point_fields),
                pose=pose,
                has_pose=has_pose,
                bounds=bounds,
            )
        )
    return refs


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #


def _quat_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _cache_key(*parts: Any) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(repr(part).encode())
        h.update(b"\0")
    return h.hexdigest()[:24]


def _load_or_none(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None


def _save_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    # np.savez appends ".npz" when the name does not already end with it, so the
    # temporary file must keep the .npz suffix.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def read_scan(
    ref: ScanRef | str,
    index: int | None = None,
    *,
    voxel: float | None = None,
    max_points: int | None = None,
    apply_pose: bool = True,
    want_colors: bool = True,
    want_intensity: bool = True,
    cache: bool = True,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Read one scan and optionally downsample it.

    Exactly one of ``voxel`` / ``max_points`` may be given:

    * ``voxel``      -> deterministic voxel-grid downsample (for registration)
    * ``max_points`` -> random subsample (for previews, keeps colours intact)

    Returns a dict with ``points`` (float32 Nx3), ``colors`` (uint8 Nx3 or
    empty), ``intensity`` (float32 N or empty) and ``count``.
    """
    if isinstance(ref, str):
        ref = probe_file(ref)[0 if index is None else index]

    stat = os.stat(ref.file)
    key = _cache_key(
        Path(ref.file).resolve(),
        stat.st_size,
        stat.st_mtime_ns,
        ref.index,
        voxel,
        max_points,
        apply_pose,
        want_colors,
        want_intensity,
        seed,
    )
    cache_path = CACHE_DIR / f"{key}.npz"
    if cache:
        hit = _load_or_none(cache_path)
        if hit is not None:
            hit["count"] = np.array([len(hit["points"])], dtype=np.int64)
            return hit

    reader = pye57.E57(ref.file)
    raw = reader.read_scan_raw(ref.index, ignore_unsupported_fields=True)

    if "cartesianX" in raw:
        xyz = np.stack(
            [
                np.asarray(raw["cartesianX"], dtype=np.float64),
                np.asarray(raw["cartesianY"], dtype=np.float64),
                np.asarray(raw["cartesianZ"], dtype=np.float64),
            ],
            axis=1,
        )
    else:
        r = np.asarray(raw["sphericalRange"], dtype=np.float64)
        az = np.asarray(raw["sphericalAzimuth"], dtype=np.float64)
        el = np.asarray(raw["sphericalElevation"], dtype=np.float64)
        ce = np.cos(el)
        xyz = np.stack([r * ce * np.cos(az), r * ce * np.sin(az), r * np.sin(el)], axis=1)

    valid = np.isfinite(xyz).all(axis=1)
    if "cartesianInvalidState" in raw:
        valid &= np.asarray(raw["cartesianInvalidState"]) == 0
    valid &= np.abs(xyz).sum(axis=1) > 1e-9
    xyz = xyz[valid]

    n = len(xyz)
    colors: np.ndarray
    if want_colors and all(k in raw for k in ("colorRed", "colorGreen", "colorBlue")):
        colors = np.stack(
            [np.asarray(raw[k])[valid] for k in ("colorRed", "colorGreen", "colorBlue")], axis=1
        ).astype(np.uint8, copy=False)
    else:
        colors = np.empty((n, 0), dtype=np.uint8)

    intensity: np.ndarray
    if want_intensity and "intensity" in raw:
        intensity = np.asarray(raw["intensity"])[valid].astype(np.float32, copy=False)
    else:
        intensity = np.empty(0, dtype=np.float32)

    if apply_pose and ref.has_pose and not np.allclose(ref.pose, np.eye(4)):
        xyz = xyz @ ref.pose[:3, :3].T + ref.pose[:3, 3]

    xyz = xyz.astype(np.float32, copy=False)
    if voxel and voxel > 0:
        xyz, keep = _voxel_subsample(xyz, voxel)
        if colors.size:
            colors = colors[keep]
        if intensity.size:
            intensity = intensity[keep]
    elif max_points and n > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=int(max_points), replace=False)
        keep.sort()
        xyz = xyz[keep]
        if colors.size:
            colors = colors[keep]
        if intensity.size:
            intensity = intensity[keep]

    out = {
        "points": np.ascontiguousarray(xyz, dtype=np.float32),
        "colors": np.ascontiguousarray(colors, dtype=np.uint8),
        "intensity": np.ascontiguousarray(intensity, dtype=np.float32),
        "count": np.array([len(xyz)], dtype=np.int64),
    }
    if cache:
        _save_atomic(cache_path, {k: v for k, v in out.items() if k != "count"})
    return out


def _voxel_subsample(points: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Fast voxel-grid downsample returning one representative index per cell.

    Unlike an average, this keeps original points so colours/intensity can be
    carried through unchanged.
    """
    if len(points) == 0:
        return points, np.zeros(0, dtype=np.int64)
    keys = np.floor(points.astype(np.float64) / float(voxel)).astype(np.int64)
    # pack the three integer keys into one uint64 (coordinates are small enough
    # after subtracting the minimum, so 21 bits per axis is ample).
    keys -= keys.min(axis=0, keepdims=True)
    span = np.maximum(keys.max(axis=0, keepdims=True) + 1, 1)
    if float(span.max()) < 2**21:
        packed = (keys[:, 0] << 42) | (keys[:, 1] << 21) | keys[:, 2]
        _, first = np.unique(packed, return_index=True)
    else:  # pathological: fall back to a solid 64-bit hash per row
        packed = (
            keys[:, 0].astype(np.uint64) * np.uint64(73856093)
            ^ keys[:, 1].astype(np.uint64) * np.uint64(19349663)
            ^ keys[:, 2].astype(np.uint64) * np.uint64(83492791)
        )
        _, first = np.unique(packed, return_index=True)
    first.sort()
    return np.ascontiguousarray(points[first]), first.astype(np.int64)


def read_full(
    ref: ScanRef,
    *,
    apply_pose: bool = True,
    want_colors: bool = True,
    want_intensity: bool = True,
) -> dict[str, np.ndarray]:
    """Read one scan at full resolution (float32) without any downsampling."""
    reader = pye57.E57(ref.file)
    raw = reader.read_scan_raw(ref.index, ignore_unsupported_fields=True)

    if "cartesianX" in raw:
        xyz = np.stack(
            [
                np.asarray(raw["cartesianX"], dtype=np.float64),
                np.asarray(raw["cartesianY"], dtype=np.float64),
                np.asarray(raw["cartesianZ"], dtype=np.float64),
            ],
            axis=1,
        )
    else:
        r = np.asarray(raw["sphericalRange"], dtype=np.float64)
        az = np.asarray(raw["sphericalAzimuth"], dtype=np.float64)
        el = np.asarray(raw["sphericalElevation"], dtype=np.float64)
        ce = np.cos(el)
        xyz = np.stack([r * ce * np.cos(az), r * ce * np.sin(az), r * np.sin(el)], axis=1)

    valid = np.isfinite(xyz).all(axis=1)
    if "cartesianInvalidState" in raw:
        valid &= np.asarray(raw["cartesianInvalidState"]) == 0
    valid &= np.abs(xyz).sum(axis=1) > 1e-9

    xyz = xyz[valid]
    if apply_pose and ref.has_pose and not np.allclose(ref.pose, np.eye(4)):
        xyz = xyz @ ref.pose[:3, :3].T + ref.pose[:3, 3]

    colors = np.empty((len(xyz), 0), dtype=np.uint8)
    if want_colors and all(k in raw for k in ("colorRed", "colorGreen", "colorBlue")):
        colors = np.stack(
            [np.asarray(raw[k])[valid] for k in ("colorRed", "colorGreen", "colorBlue")], axis=1
        ).astype(np.uint8, copy=False)
    intensity = np.empty(0, dtype=np.float32)
    if want_intensity and "intensity" in raw:
        intensity = np.asarray(raw["intensity"])[valid].astype(np.float32, copy=False)

    return {
        "points": np.ascontiguousarray(xyz, dtype=np.float32),
        "colors": np.ascontiguousarray(colors, dtype=np.uint8),
        "intensity": np.ascontiguousarray(intensity, dtype=np.float32),
    }


def _density_cap_indices(points: np.ndarray, voxel: float, cap: int) -> np.ndarray:
    """Keep at most ``cap`` points per voxel; sparse voxels keep everything.

    This is a density-dependent thinning: over-sampled regions are reduced,
    sparse/accurate regions are untouched.  Returns sorted keep indices.
    """
    pts = np.asarray(points)
    n = len(pts)
    if n == 0 or cap <= 0 or voxel <= 0:
        return np.arange(n)
    keys = np.floor(pts.astype(np.float64) / float(voxel)).astype(np.int64)
    keys -= keys.min(axis=0, keepdims=True)
    span = np.maximum(keys.max(axis=0, keepdims=True) + 1, 1)
    if float(span.max()) < 2**21:
        packed = (keys[:, 0] << 42) | (keys[:, 1] << 21) | keys[:, 2]
    else:
        packed = (
            keys[:, 0].astype(np.uint64) * np.uint64(73856093)
            ^ keys[:, 1].astype(np.uint64) * np.uint64(19349663)
            ^ keys[:, 2].astype(np.uint64) * np.uint64(83492791)
        )
    order = np.argsort(packed, kind="stable")
    _, counts = np.unique(packed, return_counts=True)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    rank = np.arange(n) - np.repeat(starts, counts)
    keep = order[rank < int(cap)]
    keep.sort()
    return keep


def scan_chunks(
    ref: ScanRef,
    transform: np.ndarray | None = None,
    *,
    apply_pose: bool = True,
    chunk: int = 1_000_000,
    voxel: float | None = None,
    density_cap: int = 0,
    density_voxel: float = 0.02,
    noise_ref: str | None = None,
    noise_radius: float = 0.15,
    want_colors: bool = True,
    want_intensity: bool = True,
    cleanup: list[dict] | None = None,
    cleanup_scan: str | None = None,
    cancel: Any = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield full-resolution points in chunks, already transformed.

    ``voxel`` applies a deterministic per-scan voxel downsample before writing.
    """
    full = read_full(
        ref, apply_pose=apply_pose, want_colors=want_colors, want_intensity=want_intensity
    )
    points = full["points"]
    colors = full["colors"]
    intensity = full["intensity"]

    # transfer the load-time noise filter to the full-resolution cloud: a point
    # survives if there is a kept registration-voxel point nearby.
    if noise_ref:
        try:
            from scipy.spatial import cKDTree

            kept = np.load(noise_ref)
            if len(kept):
                tree = cKDTree(kept)
                dist, _ = tree.query(points, k=1, workers=-1)
                mask = dist <= float(noise_radius)
                points = points[mask]
                if colors.size:
                    colors = colors[mask]
                if intensity.size:
                    intensity = intensity[mask]
        except Exception:
            pass

    if density_cap and density_cap > 0 and density_voxel and density_voxel > 0:
        keep = _density_cap_indices(points, float(density_voxel), int(density_cap))
        points = points[keep]
        if colors.size:
            colors = colors[keep]
        if intensity.size:
            intensity = intensity[keep]
    elif voxel and voxel > 0:
        points, keep = _voxel_subsample(points, voxel)
        if colors.size:
            colors = colors[keep]
        if intensity.size:
            intensity = intensity[keep]

    T = np.eye(4) if transform is None else np.asarray(transform, dtype=np.float64)
    for start in range(0, len(points), chunk):
        if cancel is not None and cancel():
            raise RuntimeError("cancelled by user")
        stop = min(start + chunk, len(points))
        block = points[start:stop].astype(np.float64)
        world = block @ T[:3, :3].T + T[:3, 3]
        c = colors[start:stop] if colors.size else colors
        it = intensity[start:stop] if intensity.size else intensity
        if cleanup:
            from .cleanup import removal_mask

            mask = removal_mask(world, cleanup, scan=cleanup_scan)
            if mask.any():
                keep = ~mask
                world = world[keep]
                if c.size:
                    c = c[keep]
                if it.size:
                    it = it[keep]
                if len(world) == 0:
                    continue
        out = {"points": np.ascontiguousarray(world, dtype=np.float32)}
        if c.size:
            out["colors"] = np.array(c, copy=True)
        if it.size:
            out["intensity"] = np.array(it, copy=True)
        yield out


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #


@dataclass
class StreamSpec:
    """One data3D scan to be written from a stream of point chunks."""

    name: str
    chunks: Any  # callable() -> iterator of {"points", "colors"?, "intensity"?}
    has_colors: bool = False
    has_intensity: bool = False
    bounds: tuple[np.ndarray, np.ndarray] | None = None


def write_e57_stream(
    path: str,
    streams: Sequence[StreamSpec],
    *,
    chunk: int = 1_000_000,
    precision: str = "double",
    progress=None,
) -> None:
    """Write E57 scans from chunk iterators (constant memory)."""
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path + ".part"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    imf = libe57.ImageFile(tmp_path, "w")
    root = imf.root()
    root.set("formatName", StringNode(imf, "ASTM E57 3D Imaging Data File"))
    root.set("guid", StringNode(imf, "{%s}" % uuid.uuid4()))
    root.set("versionMajor", IntegerNode(imf, libe57.E57_FORMAT_MAJOR))
    root.set("versionMinor", IntegerNode(imf, libe57.E57_FORMAT_MINOR))
    root.set("e57LibraryVersion", StringNode(imf, "e57stitch"))
    root.set("coordinateMetadata", StringNode(imf, ""))
    cdt = StructureNode(imf)
    cdt.set("dateTimeValue", FloatNode(imf, 0.0))
    cdt.set("isAtomicClockReferenced", IntegerNode(imf, 0))
    root.set("creationDateTime", cdt)
    data3d = VectorNode(imf, True)
    root.set("data3D", data3d)

    total = 0
    for spec in streams:
        node = StructureNode(imf)
        data3d.append(node)
        node.set("guid", StringNode(imf, "{%s}" % uuid.uuid4()))
        node.set("name", StringNode(imf, str(spec.name or "scan")))

        if spec.bounds is not None:
            lo, hi = spec.bounds
            bounds = StructureNode(imf)
            for axis, key in enumerate(("x", "y", "z")):
                bounds.set(f"{key}Minimum", FloatNode(imf, float(lo[axis])))
                bounds.set(f"{key}Maximum", FloatNode(imf, float(hi[axis])))
            node.set("cartesianBounds", bounds)

        proto = StructureNode(imf)
        prec = FloatPrecision.E57_SINGLE if precision == "single" else FloatPrecision.E57_DOUBLE
        for axis in ("X", "Y", "Z"):
            proto.set(f"cartesian{axis}", FloatNode(imf, 0.0, prec))
        if spec.has_colors:
            for key in ("colorRed", "colorGreen", "colorBlue"):
                proto.set(key, IntegerNode(imf, 0, 0, 255))
        if spec.has_intensity:
            proto.set("intensity", FloatNode(imf, 0.0, FloatPrecision.E57_SINGLE))

        points_node = CompressedVectorNode(imf, proto, VectorNode(imf, True))
        node.set("points", points_node)

        # Allocate the destination buffers once; refill them in place for every
        # chunk and reuse a single CompressedVectorWriter.  Creating a new
        # writer per chunk would restart (and truncate) the compressed vector.
        x_buf = np.empty(chunk, dtype=np.float64)
        y_buf = np.empty(chunk, dtype=np.float64)
        z_buf = np.empty(chunk, dtype=np.float64)
        bufs = VectorSourceDestBuffer()
        bufs.append(SourceDestBuffer(imf, "cartesianX", x_buf, chunk, True, True))
        bufs.append(SourceDestBuffer(imf, "cartesianY", y_buf, chunk, True, True))
        bufs.append(SourceDestBuffer(imf, "cartesianZ", z_buf, chunk, True, True))
        r_buf = g_buf = b_buf = None
        if spec.has_colors:
            r_buf = np.empty(chunk, dtype=np.uint8)
            g_buf = np.empty(chunk, dtype=np.uint8)
            b_buf = np.empty(chunk, dtype=np.uint8)
            bufs.append(SourceDestBuffer(imf, "colorRed", r_buf, chunk, True, True))
            bufs.append(SourceDestBuffer(imf, "colorGreen", g_buf, chunk, True, True))
            bufs.append(SourceDestBuffer(imf, "colorBlue", b_buf, chunk, True, True))
        i_buf = None
        if spec.has_intensity:
            i_buf = np.empty(chunk, dtype=np.float32)
            bufs.append(SourceDestBuffer(imf, "intensity", i_buf, chunk, True, True))

        writer = points_node.writer(bufs)
        for chunk_data in spec.chunks():
            pts = np.asarray(chunk_data["points"], dtype=np.float64)
            n = len(pts)
            if n == 0:
                continue
            if n > chunk:
                raise ValueError(f"chunk of {n} points exceeds writer chunk size {chunk}")
            x_buf[:n] = pts[:, 0]
            y_buf[:n] = pts[:, 1]
            z_buf[:n] = pts[:, 2]
            if spec.has_colors:
                colors = chunk_data.get("colors")
                if colors is None or len(colors) != n:
                    r_buf[:n] = 200
                    g_buf[:n] = 200
                    b_buf[:n] = 200
                else:
                    rgb = np.asarray(colors, dtype=np.uint8)
                    r_buf[:n] = rgb[:, 0]
                    g_buf[:n] = rgb[:, 1]
                    b_buf[:n] = rgb[:, 2]
            if spec.has_intensity:
                inten = chunk_data.get("intensity")
                if inten is None or len(inten) != n:
                    i_buf[:n] = 0.0
                else:
                    i_buf[:n] = np.asarray(inten, dtype=np.float32)
            writer.write(n)
            total += n
            if progress:
                progress(total, None)
        writer.close()

    imf.close()
    os.replace(tmp_path, path)


def write_ply_stream(
    path: str,
    streams: Sequence[StreamSpec],
    *,
    progress=None,
) -> None:
    """Binary PLY with a patched vertex count (works without knowing N upfront)."""
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    final_path = path
    path = path + ".part"
    has_colors = any(s.has_colors for s in streams)
    has_intensity = any(getattr(s, "has_intensity", False) for s in streams)
    with open(path, "w+b") as f:
        header = [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 00000000000000000000",
            "property float x",
            "property float y",
            "property float z",
        ]
        if has_colors:
            header += ["property uchar red", "property uchar green", "property uchar blue"]
        if has_intensity:
            header += ["property float intensity"]
        header.append("end_header")
        header_bytes = ("\n".join(header) + "\n").encode()
        f.write(header_bytes)
        fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        if has_colors:
            fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        if has_intensity:
            fields += [("intensity", "<f4")]
        dtype = np.dtype(fields)
        total = 0
        for spec in streams:
            for chunk_data in spec.chunks():
                pts = np.asarray(chunk_data["points"], dtype=np.float32)
                n = len(pts)
                if n == 0:
                    continue
                out = np.empty(n, dtype=dtype)
                out["x"], out["y"], out["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
                if has_colors:
                    c = chunk_data.get("colors")
                    if c is None or len(c) != n:
                        c = np.full((n, 3), 200, dtype=np.uint8)
                    out["red"], out["green"], out["blue"] = c[:, 0], c[:, 1], c[:, 2]
                if has_intensity:
                    it = chunk_data.get("intensity")
                    if it is None or len(it) != n:
                        it = np.zeros(n, dtype=np.float32)
                    out["intensity"] = np.asarray(it, dtype=np.float32)
                f.write(out.tobytes())
                total += n
                if progress:
                    progress(total, None)
        # patch the fixed-width count
        marker = b"element vertex 00000000000000000000"
        pos = header_bytes.find(marker)
        f.seek(pos + len(b"element vertex "))
        f.write(f"{total:020d}".encode())
    os.replace(path, final_path)


def write_pts_stream(
    path: str,
    streams: Sequence[StreamSpec],
    *,
    chunk: int = 500_000,
    progress=None,
) -> None:
    """Leica .pts text file with a patched point count."""
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    final_path = path
    path = path + ".part"
    with open(path, "w+b") as f:
        f.write(b"00000000000000000000\n")
        total = 0
        for spec in streams:
            for chunk_data in spec.chunks():
                pts = np.asarray(chunk_data["points"], dtype=np.float64)
                n = len(pts)
                if n == 0:
                    continue
                c = chunk_data.get("colors")
                if c is None or len(c) != n:
                    c = np.full((n, 3), 200, dtype=np.int32)
                else:
                    c = np.asarray(c, dtype=np.int32)
                inten = chunk_data.get("intensity")
                if inten is None or len(inten) != n:
                    inten = np.zeros(n, dtype=np.float64)
                out = np.column_stack([pts, np.asarray(inten, dtype=np.float64), c])
                np.savetxt(f, out, fmt="%.4f %.4f %.4f %.4f %d %d %d")
                total += n
                if progress:
                    progress(total, None)
        f.seek(0)
        f.write(f"{total:020d}\n".encode())
    os.replace(path, final_path)


def write_xyz_stream(
    path: str,
    streams: Sequence[StreamSpec],
    *,
    progress=None,
) -> None:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    final_path = path
    path = path + ".part"
    has_intensity = any(getattr(s, "has_intensity", False) for s in streams)
    total = 0
    with open(path, "w") as f:
        for spec in streams:
            for chunk_data in spec.chunks():
                pts = np.asarray(chunk_data["points"], dtype=np.float64)
                n = len(pts)
                if n == 0:
                    continue
                c = chunk_data.get("colors")
                it = chunk_data.get("intensity")
                cols = [pts]
                fmt = "%.4f %.4f %.4f"
                if c is not None and len(c) == n:
                    cols.append(np.asarray(c, dtype=np.float64))
                    fmt += " %d %d %d"
                if has_intensity:
                    if it is None or len(it) != n:
                        it = np.zeros(n, dtype=np.float64)
                    cols.append(np.asarray(it, dtype=np.float64).reshape(-1, 1))
                    fmt += " %.4f"
                out = np.column_stack(cols)
                np.savetxt(f, out, fmt=fmt)
                total += n
                if progress:
                    progress(total, None)
    os.replace(path, final_path)


STREAM_WRITERS = {
    "e57": write_e57_stream,
    "ply": write_ply_stream,
    "pts": write_pts_stream,
    "xyz": write_xyz_stream,
}
