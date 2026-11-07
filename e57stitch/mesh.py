"""Mesh reconstruction + mesh writers for registered point clouds.

The registered cloud is a set of surfaces seen from many stations.  Poisson
reconstruction gives a watertight, smooth surface; ball pivoting keeps more
detail but leaves holes.  Both are run on a decimated copy so a 17 x 9 M point
project stays within memory, then colours and intensity are transferred back
from the original points.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .features import estimate_normals, make_cloud


def mesh_from_cloud(
    points: np.ndarray,
    *,
    colors: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    voxel: float = 0.02,
    method: str = "poisson",
    depth: int = 9,
    target_points: int = 700_000,
    target_triangles: int = 2_000_000,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Reconstruct a triangle mesh and transfer colour / intensity attributes."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2000:
        raise ValueError("not enough points to build a mesh")
    cols = None if colors is None or len(colors) != len(pts) else np.asarray(colors, dtype=np.uint8)
    inten = None if intensity is None or len(intensity) != len(pts) else np.asarray(intensity, dtype=np.float32)

    if progress:
        progress(0.02, "mesh: preparing cloud")
    # Decimate before normals/Poisson: quality is driven by surface coverage,
    # not by the 9 M raw points each station contributes.
    if len(pts) > target_points:
        idx = np.random.default_rng(21).choice(len(pts), target_points, replace=False)
        pts = pts[idx]
        if cols is not None:
            cols = cols[idx]
        if inten is not None:
            inten = inten[idx]

    pcd = make_cloud(pts)
    if progress:
        progress(0.08, f"mesh: normals for {len(pts):,} points")
    estimate_normals(pcd, max(float(voxel) * 3.0, 0.05))
    try:
        pcd.orient_normals_consistent_tangent_plane(24)
    except Exception:
        pcd.orient_normals_towards_camera_location(np.asarray(pcd.points).mean(axis=0))

    method = str(method or "poisson").lower()
    mesh = None
    if method in ("bpa", "ball", "ball-pivoting", "ball_pivoting"):
        if progress:
            progress(0.25, "mesh: ball pivoting")
        try:
            radii = o3d.utility.DoubleVector(
                [max(float(voxel) * 2.0, 0.02), max(float(voxel) * 4.0, 0.04), max(float(voxel) * 8.0, 0.08)]
            )
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
            if len(mesh.triangles) < 100:
                mesh = None
        except Exception:
            mesh = None
    if mesh is None:
        if progress:
            progress(0.25, f"mesh: Poisson depth {depth}")
        try:
            result = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=int(depth), scale=1.1, linear_fit=False, samples_per_node=1.0
            )
        except TypeError:
            result = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=int(depth), scale=1.1, linear_fit=False
            )
        mesh, densities = result
        dens = np.asarray(densities)
        if len(dens):
            # Poisson invents surfaces across holes and outside the observed
            # region.  Drop the low-density tail instead of exporting a shell.
            cut = float(np.quantile(dens, 0.02))
            keep = dens > cut
            if keep.sum() > 1000:
                mesh.remove_vertices_by_mask(~keep)

    if progress:
        progress(0.72, "mesh: cleaning")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > target_triangles:
        if progress:
            progress(0.78, "mesh: simplifying")
        mesh = mesh.simplify_quadric_decimation(int(target_triangles))
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if len(vertices) < 100 or len(faces) < 100:
        raise ValueError("reconstruction produced an empty mesh")

    out_cols = np.full((len(vertices), 3), 190, dtype=np.uint8)
    out_inten = np.zeros(len(vertices), dtype=np.float32)
    if cols is not None or inten is not None:
        if progress:
            progress(0.88, "mesh: transferring colours")
        tree = cKDTree(pts)
        _d, idx = tree.query(vertices, k=1, workers=-1)
        if cols is not None:
            out_cols = cols[idx].astype(np.uint8)
        if inten is not None:
            out_inten = inten[idx].astype(np.float32)
    if progress:
        progress(1.0, f"mesh: {len(vertices):,} vertices / {len(faces):,} triangles")
    return {
        "vertices": vertices.astype(np.float32),
        "faces": faces,
        "normals": normals,
        "colors": out_cols,
        "intensity": out_inten,
        "method": method,
        "depth": int(depth),
    }


def write_mesh_ply(
    path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    normals: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
) -> None:
    """Binary PLY with faces, vertex colours and an intensity scalar."""
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    V = np.asarray(vertices, dtype=np.float32)
    F = np.asarray(faces, dtype=np.int32)
    has_normals = normals is not None and len(normals) == len(V)
    has_colors = colors is not None and len(colors) == len(V)
    has_intensity = intensity is not None and len(intensity) == len(V)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(V)}"]
    header += ["property float x", "property float y", "property float z"]
    if has_normals:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
        header += ["property float nx", "property float ny", "property float nz"]
    if has_colors:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    if has_intensity:
        fields += [("intensity", "<f4")]
        header += ["property float intensity"]
    header += [
        f"element face {len(F)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    data = np.empty(len(V), dtype=np.dtype(fields))
    data["x"], data["y"], data["z"] = V[:, 0], V[:, 1], V[:, 2]
    if has_normals:
        N = np.asarray(normals, dtype=np.float32)
        data["nx"], data["ny"], data["nz"] = N[:, 0], N[:, 1], N[:, 2]
    if has_colors:
        C = np.asarray(colors, dtype=np.uint8)
        data["red"], data["green"], data["blue"] = C[:, 0], C[:, 1], C[:, 2]
    if has_intensity:
        data["intensity"] = np.asarray(intensity, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())
        face_record = np.empty(len(F), dtype=np.dtype([("n", "u1"), ("idx", "<i4", (3,))]))
        face_record["n"] = 3
        face_record["idx"] = F
        f.write(face_record.tobytes())


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(
    path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    normals: np.ndarray | None = None,
    colors: np.ndarray | None = None,
) -> None:
    """Minimal binary glTF 2.0 writer (positions, optional normals/colours)."""
    import json
    import struct

    V = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    F = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)
    N = np.asarray(normals, dtype=np.float32).reshape(-1, 3) if normals is not None and len(normals) == len(V) else None
    C = np.asarray(colors, dtype=np.uint8).reshape(-1, 3) if colors is not None and len(colors) == len(V) else None
    bin_parts: list[bytes] = []
    views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []

    def add_view(arr: np.ndarray, target: int) -> int:
        raw = _pad4(np.ascontiguousarray(arr).tobytes())
        offset = sum(len(p) for p in bin_parts)
        bin_parts.append(raw)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw) - ((4 - len(raw) % 4) % 4), "target": target})
        return len(views) - 1

    vi = add_view(V, 34962)
    accessors.append({
        "bufferView": vi, "componentType": 5126, "count": len(V), "type": "VEC3",
        "min": V.min(axis=0).tolist(), "max": V.max(axis=0).tolist(),
    })
    attributes = {"POSITION": 0}
    if N is not None:
        ni = add_view(N, 34962)
        attributes["NORMAL"] = len(accessors)
        accessors.append({"bufferView": ni, "componentType": 5126, "count": len(N), "type": "VEC3"})
    if C is not None:
        ci = add_view(C, 34962)
        attributes["COLOR_0"] = len(accessors)
        accessors.append({
            "bufferView": ci, "componentType": 5121, "normalized": True,
            "count": len(C), "type": "VEC3",
        })
    fi = add_view(F, 34963)
    accessors.append({"bufferView": fi, "componentType": 5125, "count": int(F.size), "type": "SCALAR"})
    index_accessor = len(accessors) - 1
    binary = b"".join(bin_parts)
    doc = {
        "asset": {"version": "2.0", "generator": "e57stitch"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": index_accessor, "mode": 4}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    js = _pad4(json.dumps(doc, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", len(binary), 0x004E4942))
        f.write(binary)


def write_mesh(path: str, mesh: dict[str, Any], *, intensity_ramp: bool = False) -> None:
    """Write .ply (native), .obj / .glb (Open3D / native GLB)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".glb":
        colors = np.asarray(mesh.get("colors"), dtype=np.uint8)
        if intensity_ramp and mesh.get("intensity") is not None:
            t = np.asarray(mesh["intensity"], dtype=np.float32)
            lo, hi = float(np.percentile(t, 2)), float(np.percentile(t, 98))
            u = np.clip((t - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            colors = np.stack(
                [
                    np.clip(1.5 - np.abs(4 * u - 3), 0, 1) * 255,
                    np.clip(1.5 - np.abs(4 * u - 2), 0, 1) * 255,
                    np.clip(1.5 - np.abs(4 * u - 1), 0, 1) * 255,
                ],
                axis=1,
            ).astype(np.uint8)
        write_glb(
            path,
            mesh["vertices"],
            mesh["faces"],
            normals=mesh.get("normals"),
            colors=colors,
        )
        return
    if suffix == ".ply":
        write_mesh_ply(
            path,
            mesh["vertices"],
            mesh["faces"],
            normals=mesh.get("normals"),
            colors=mesh.get("colors"),
            intensity=mesh.get("intensity"),
        )
        return
    tm = o3d.geometry.TriangleMesh()
    tm.vertices = o3d.utility.Vector3dVector(np.asarray(mesh["vertices"], dtype=np.float64))
    tm.triangles = o3d.utility.Vector3iVector(np.asarray(mesh["faces"], dtype=np.int32))
    colors = np.asarray(mesh.get("colors"), dtype=np.float32) / 255.0
    if intensity_ramp and mesh.get("intensity") is not None:
        # Bake intensity into a turbo-like ramp for formats without scalars.
        t = np.asarray(mesh["intensity"], dtype=np.float32)
        lo, hi = float(np.percentile(t, 2)), float(np.percentile(t, 98))
        u = np.clip((t - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4.0 * u - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * u - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * u - 1.0), 0.0, 1.0)
        colors = np.stack([r, g, b], axis=1)
    if colors.shape == (len(tm.vertices), 3):
        tm.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    ok = o3d.io.write_triangle_mesh(str(path), tm, write_vertex_colors=True)
    if not ok:
        raise ValueError(f"Open3D could not write {path}")


def load_mesh_for_view(path: str, max_triangles: int = 450_000) -> dict[str, Any]:
    """Read a PLY/OBJ/GLB mesh and return arrays suitable for the browser."""
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError("mesh file is empty or unsupported")
    if len(mesh.triangles) > max_triangles:
        mesh = mesh.simplify_quadric_decimation(int(max_triangles))
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.uint32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    if colors.shape != (len(vertices), 3) or not np.isfinite(colors).all():
        colors = np.full((len(vertices), 3), 0.72, dtype=np.float32)
    elif colors.size and float(colors.max()) <= 1.01:
        colors = colors
    elif colors.size:
        colors = colors / 255.0
    return {"vertices": vertices, "faces": faces, "colors": colors.astype(np.float32)}


def _mesh_job(payload: dict[str, Any]) -> dict[str, Any]:
    return mesh_from_cloud(
        payload["points"],
        colors=payload.get("colors"),
        intensity=payload.get("intensity"),
        voxel=float(payload.get("voxel") or 0.02),
        method=str(payload.get("method") or "poisson"),
        depth=int(payload.get("depth") or 9),
        target_points=int(payload.get("target_points") or 500_000),
        target_triangles=int(payload.get("target_triangles") or 2_000_000),
    )


def reconstruct_safely(
    points: np.ndarray,
    *,
    colors: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    voxel: float = 0.02,
    method: str = "poisson",
    depth: int = 9,
    progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run meshing in a detached child process.

    Open3D's Poisson implementation occasionally aborts the whole process on
    pathological geometry ("Failed to close loop" -> SIGSEGV).  A subprocess
    keeps the server alive; retry with a smaller cloud and lower depth.
    """
    import subprocess
    import sys
    import tempfile

    points = np.asarray(points, dtype=np.float32)
    attempts = [
        (int(depth), min(500_000, max(120_000, len(points)))),
        (max(6, int(depth) - 1), min(320_000, max(100_000, len(points)))),
        (max(5, int(depth) - 2), min(180_000, max(80_000, len(points)))),
    ]
    last_exc: Exception | None = None
    for attempt, (d, target) in enumerate(attempts, start=1):
        if progress:
            progress(0.12, f"mesh: {method} depth {d}, {target:,} points (attempt {attempt})")
        with tempfile.TemporaryDirectory(prefix="e57mesh-") as td:
            from pathlib import Path

            inp = Path(td) / "in.npz"
            out = Path(td) / "out.npz"
            payload = {"points": points}
            if colors is not None:
                payload["colors"] = np.asarray(colors, dtype=np.uint8)
            if intensity is not None:
                payload["intensity"] = np.asarray(intensity, dtype=np.float32)
            np.savez(inp, **payload)
            cmd = [
                sys.executable,
                "-m",
                "e57stitch.mesh_worker",
                str(inp),
                str(out),
                str(d),
                str(float(voxel)),
                str(method),
                str(target),
            ]
            # Open3D prints a lot of warnings; a pipe would eventually block
            # the child, so discard its output here.
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
            )
            deadline = time.time() + 1800
            while proc.poll() is None:
                if should_cancel is not None and should_cancel():
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise RuntimeError("mesh cancelled by user")
                if time.time() > deadline:
                    proc.kill()
                    proc.wait()
                    last_exc = RuntimeError("mesh worker timed out")
                    break
                time.sleep(0.5)
            if proc.returncode != 0 or not out.exists():
                last_exc = RuntimeError(f"mesh worker exited {proc.returncode}")
                if progress:
                    progress(0.2, f"mesh: reconstruction failed ({proc.returncode}), retrying")
                continue
            with np.load(out, allow_pickle=False) as z:
                result = {
                    "vertices": z["vertices"],
                    "faces": z["faces"],
                    "normals": z["normals"],
                    "colors": z["colors"],
                    "intensity": z["intensity"],
                    "method": str(z["method"]),
                    "depth": int(z["depth"]),
                }
            if progress:
                progress(0.95, "mesh: child process finished")
            return result
    raise RuntimeError(f"mesh reconstruction failed after retries: {last_exc}")
