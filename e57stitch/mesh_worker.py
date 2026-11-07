"""Standalone mesh reconstruction worker (isolates Open3D Poisson crashes)."""

from __future__ import annotations

import sys

import numpy as np

from .mesh import mesh_from_cloud


def main() -> None:
    inp, out = sys.argv[1], sys.argv[2]
    depth = int(sys.argv[3])
    voxel = float(sys.argv[4])
    method = str(sys.argv[5])
    target = int(sys.argv[6])
    with np.load(inp, allow_pickle=False) as z:
        points = z["points"]
        colors = z["colors"] if "colors" in z.files else None
        intensity = z["intensity"] if "intensity" in z.files else None
    result = mesh_from_cloud(
        points,
        colors=colors,
        intensity=intensity,
        voxel=voxel,
        method=method,
        depth=depth,
        target_points=target,
        target_triangles=2_000_000,
    )
    np.savez(
        out,
        vertices=result["vertices"],
        faces=result["faces"],
        normals=result["normals"],
        colors=result["colors"],
        intensity=result["intensity"],
        method=np.asarray(result["method"]),
        depth=np.asarray(result["depth"]),
    )


if __name__ == "__main__":
    main()
