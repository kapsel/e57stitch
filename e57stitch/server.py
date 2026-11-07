"""Local FastAPI server for the e57stitch UI."""

from __future__ import annotations

import json
import struct
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .projects import MANAGER


class _ActiveSession:
    """Resolve every attribute against the currently active project."""

    def __getattr__(self, name: str) -> Any:
        return getattr(MANAGER.current(), name)


SESSION = _ActiveSession()

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="e57stitch", version=__version__)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.middleware("http")
async def _no_stale_ui(request, call_next):
    # The UI has no build step / hashed filenames: make browsers revalidate
    # (ETag -> 304) so a new app.js is never paired with an old index.html.
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class PathsBody(BaseModel):
    paths: list[str]
    force: bool = False
    recursive: bool = True


class IdsBody(BaseModel):
    ids: list[str] | None = None
    force: bool = False


class RegisterBody(BaseModel):
    # stages 1-4: floor-plan candidates, global selection, refine (minutes)
    fresh: bool = False             # delete stored results for these scans first ("start clean")
    keep_improving: bool = True     # continue with stage 5 (heavy matchers) in the same job
    hours: float = 10.0             # time budget for stage 5
    voxel: float = 0.05             # matching resolution
    refine_voxel: float | None = 0.02
    pairs: list[list[str]] | None = None  # re-match just these pairs with every heavy matcher


class ImproveBody(BaseModel):
    hours: float = 10.0
    refine_voxel: float | None = 0.02


class PairBody(BaseModel):
    a: str
    b: str
    accepted: bool


class BlockBody(BaseModel):
    a: str
    b: str
    blocked: bool = True


class TransformBody(BaseModel):
    id: str
    matrix: list[list[float]]


class AssistBody(BaseModel):
    moving: list[str]
    anchor: str
    pose: list[list[float]] | None = None
    target_group: int | None = None
    force: bool = False
    markers: list[dict[str, Any]] | None = None
    kind: str | None = None          # recorded in the manual store (e.g. "assistant")


class RefBody(BaseModel):
    id: str


class RefineBody(BaseModel):
    voxel: float = 0.02
    rounds: int = 8


class SessionLoadBody(BaseModel):
    path: str


class ProjectCreateBody(BaseModel):
    name: str = "Untitled project"


class ProjectIdBody(BaseModel):
    id: str
    name: str | None = None


class IncludeBody(BaseModel):
    id: str
    included: bool = True


class PairAccuracyBody(BaseModel):
    a: str
    b: str


class RevealBody(BaseModel):
    path: str


class RecoverBody(BaseModel):
    from_scan: str | None = None


class OrientationBody(BaseModel):
    mode: str = "auto"


class CleanupBody(BaseModel):
    id: str | None = None
    kind: str = "box"
    center: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    size: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    radius: float = 1.0
    op: str = "delete"
    enabled: bool = True
    scans: list[str] | None = None
    label: str = ""
    clip: bool = False


class MeshBody(BaseModel):
    path: str
    ids: list[str] | None = None
    scope: str = "active"
    method: str = "poisson"
    voxel: float = 0.02
    depth: int = 9
    colors: bool = True
    intensity: bool = True
    intensity_ramp: bool = False


class SnapshotBody(BaseModel):
    label: str = "manual"


class ResetRegistrationBody(BaseModel):
    cleanups: bool = False
    vetoes: bool = False     # also forget rejected links
    cached: bool = False     # also drop stored match candidates (truly fresh)


class ParamsBody(BaseModel):
    voxel: float | None = None
    preview_points: int | None = None
    load_workers: int | None = None
    pair_workers: int | None = None
    accept_score: float | None = None
    use_pose: bool | None = None
    detect_features: bool | None = None
    cleanup: str | None = None
    max_attempts: int | None = None


class ExportBody(BaseModel):
    path: str
    ids: list[str] | None = None
    format: str = "e57"
    mode: str = "merged"
    scope: str = "active"
    voxel: float = 0.02
    precision: str = "double"
    colors: bool = True
    intensity: bool = True
    density_cap: int = 0
    density_voxel: float = 0.02
    filter_noise: bool = True
    name: str = "merged"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/snapshots")
def snapshots(limit: int = 20) -> list[dict[str, Any]]:
    project = MANAGER.active_project()
    base = (project.path.parent if project.path is not None else MANAGER.root / project.id) / "snapshots"
    out: list[dict[str, Any]] = []
    if base.exists():
        for path in sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            out.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "mtime": path.stat().st_mtime,
                    "bytes": path.stat().st_size,
                }
            )
            if len(out) >= max(1, int(limit)):
                break
    return out


@app.get("/api/runs")
def runs(limit: int = 12) -> list[dict[str, Any]]:
    from .e57io import CACHE_DIR

    base = CACHE_DIR / "runs"
    out: list[dict[str, Any]] = []
    if base.exists():
        for folder in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            session = folder / "session.json"
            if not session.exists():
                continue
            out.append(
                {
                    "id": folder.name,
                    "session": str(session),
                    "log": str(folder / "run.log"),
                    "mtime": folder.stat().st_mtime,
                    "bytes": session.stat().st_size,
                }
            )
            if len(out) >= max(1, int(limit)):
                break
    return out


@app.get("/api/health")
def health() -> dict[str, Any]:
    st = MANAGER.state()
    return {
        "ok": True,
        "version": __version__,
        "time": time.time(),
        "project": st.get("project", {}).get("name"),
        "scans": len(st.get("scans", [])),
        "loaded": sum(1 for s in st.get("scans", []) if s.get("loaded")),
        "pairs": len(st.get("pairs", [])),
        "accepted": sum(1 for p in st.get("pairs", []) if p.get("accepted")),
        "busy": st.get("busy"),
        "groups": len(st.get("groups", [])),
    }


@app.get("/api/state")
def state() -> dict[str, Any]:
    return MANAGER.state()


@app.get("/api/projects")
def list_projects() -> dict[str, Any]:
    return {"projects": MANAGER.list(), "active": MANAGER.active}


@app.post("/api/projects")
def create_project(body: ProjectCreateBody) -> dict[str, Any]:
    MANAGER.create(body.name)
    return MANAGER.state()


@app.post("/api/projects/activate")
def activate_project(body: ProjectIdBody) -> dict[str, Any]:
    try:
        return MANAGER.state_for(body.id)
    except KeyError:
        raise HTTPException(404, "unknown project")


@app.post("/api/projects/restore-backup")
def restore_backup() -> dict[str, Any]:
    try:
        return MANAGER.restore_backup(MANAGER.active)
    except FileNotFoundError:
        raise HTTPException(404, "no backup for this project yet")
    except KeyError:
        raise HTTPException(404, "no active project")


@app.post("/api/projects/rename")
def rename_project(body: ProjectIdBody) -> dict[str, Any]:
    if not body.name:
        raise HTTPException(400, "name required")
    MANAGER.rename(body.id, body.name)
    return MANAGER.state()


@app.delete("/api/projects/{pid}")
def delete_project(pid: str) -> dict[str, Any]:
    try:
        MANAGER.delete(pid)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return MANAGER.state()


@app.post("/api/reveal")
def reveal(body: RevealBody) -> dict[str, Any]:
    try:
        MANAGER.reveal(body.path)
    except FileNotFoundError:
        raise HTTPException(404, "path not found")
    return {"ok": True}


@app.post("/api/include")
def set_included(body: IncludeBody) -> dict[str, Any]:
    SESSION.set_included(body.id, body.included)
    return {"ok": True}


@app.post("/api/files")
def add_files(body: PathsBody) -> dict[str, Any]:
    if not body.paths:
        raise HTTPException(400, "no paths given")
    try:
        return MANAGER.ingest(
            MANAGER.active, body.paths, force=body.force, recursive=body.recursive
        )
    except KeyError:
        raise HTTPException(404, "no active project")


@app.get("/api/browse")
def browse(path: str | None = Query(default=None)) -> dict[str, Any]:
    if path:
        base = Path(path).expanduser().resolve()
    else:
        base = Path.cwd() / "scans"
        if not base.is_dir():
            base = Path.home() / "Desktop"
    if not base.exists() or not base.is_dir():
        base = Path.home()
    dirs = []
    files = []
    try:
        for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            try:
                if child.is_dir():
                    dirs.append({"name": child.name, "path": str(child)})
                elif child.suffix.lower() == ".e57":
                    files.append(
                        {
                            "name": child.name,
                            "path": str(child),
                            "size": child.stat().st_size,
                        }
                    )
            except OSError:
                continue
    except PermissionError:
        raise HTTPException(403, f"cannot read {base}")
    return {
        "path": str(base),
        "parent": str(base.parent) if base.parent != base else None,
        "dirs": dirs,
        "files": files,
    }


@app.post("/api/load")
def load(body: IdsBody) -> dict[str, str]:
    return {"job": SESSION.load(body.ids, force=body.force)}


def _require_unlocked() -> None:
    """A locked project (e.g. a benchmark reference) refuses every change to
    its placement until it is unlocked explicitly."""
    if SESSION.params.get("locked"):
        raise HTTPException(423, "this project is locked (benchmark reference) — unlock it first")


class LockBody(BaseModel):
    locked: bool


@app.post("/api/lock")
def set_lock(body: LockBody) -> dict[str, Any]:
    SESSION.params["locked"] = bool(body.locked)
    MANAGER.save()
    SESSION._notify()
    return {"locked": bool(body.locked)}


@app.post("/api/register")
def register(body: RegisterBody | None = None) -> dict[str, str]:
    _require_unlocked()
    opts = body or RegisterBody()
    if opts.pairs:
        a, b = opts.pairs[0]
        try:
            return {"job": SESSION.rematch_pair(a, b)}
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    try:
        MANAGER.snapshot(label="pre-register")
    except Exception:
        pass
    job = SESSION.register_scans(
        fresh=opts.fresh, keep_improving=opts.keep_improving, hours=opts.hours,
        voxel=opts.voxel, refine_voxel=opts.refine_voxel,
    )
    MANAGER.save()
    return {"job": job}


@app.post("/api/improve")
def improve(body: ImproveBody | None = None) -> dict[str, str]:
    _require_unlocked()
    opts = body or ImproveBody()
    try:
        MANAGER.snapshot(label="pre-improve")
    except Exception:
        pass
    return {"job": SESSION.improve(hours=opts.hours, refine_voxel=opts.refine_voxel)}


@app.post("/api/solve")
def solve() -> dict[str, str]:
    _require_unlocked()
    return {"job": SESSION.solve()}


@app.post("/api/cleanup/add")
def add_cleanup(body: CleanupBody) -> dict[str, Any]:
    region = SESSION.add_cleanup(body.model_dump())
    MANAGER.save()
    return region


@app.post("/api/cleanup/remove")
def remove_cleanup(body: CleanupBody) -> dict[str, str]:
    if not body.id:
        raise HTTPException(400, "region id required")
    SESSION.remove_cleanup(body.id)
    MANAGER.save()
    return {"removed": body.id}


@app.post("/api/cleanup/update")
def update_cleanup(body: CleanupBody) -> dict[str, Any]:
    if not body.id:
        raise HTTPException(400, "region id required")
    fields = body.model_dump(exclude_unset=True)
    cid = fields.pop("id")
    region = SESSION.update_cleanup(cid, **fields)
    MANAGER.save()
    return region


@app.post("/api/export-mesh")
def export_mesh(body: MeshBody) -> dict[str, str]:
    try:
        job = SESSION.export_mesh(body.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"job": job}


@app.post("/api/snapshot")
def snapshot(body: SnapshotBody | None = None) -> dict[str, str]:
    label = (body or SnapshotBody()).label
    try:
        path = MANAGER.snapshot(label=label)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return {"path": path}


@app.post("/api/measure")
def measure_accuracy() -> dict[str, str]:
    return {"job": SESSION.measure_accuracy()}


@app.post("/api/realign")
def realign() -> dict[str, str]:
    _require_unlocked()
    return {"job": SESSION.realign()}


@app.post("/api/refine")
def refine(body: RefineBody) -> dict[str, str]:
    _require_unlocked()
    return {"job": SESSION.refine(voxel=max(0.005, min(0.10, body.voxel)), rounds=max(1, min(20, body.rounds)))}


@app.post("/api/pairs/accept")
def accept_pair(body: PairBody) -> dict[str, Any]:
    _require_unlocked()
    SESSION.set_pair_accepted(body.a, body.b, body.accepted)
    return {"ok": True}


@app.post("/api/pairs/block")
def block_pair(body: BlockBody) -> dict[str, Any]:
    _require_unlocked()
    try:
        restored = SESSION.set_link_blocked(body.a, body.b, body.blocked)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    MANAGER.save()
    return {"ok": True, "restored": restored}


@app.post("/api/scans/{sid}/orientation")
def set_orientation(sid: str, body: OrientationBody) -> dict[str, Any]:
    _require_unlocked()
    try:
        state = SESSION.set_orientation(sid, body.mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    MANAGER.save()
    return state


@app.post("/api/scans/{sid}/recover")
def recover_scan(sid: str, body: RecoverBody | None = None) -> dict[str, Any]:
    try:
        return SESSION.recover_scan(sid, from_scan=(body.from_scan if body else None))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/assist/snap")
def assist_snap(body: AssistBody) -> dict[str, Any]:
    if body.pose is None or len(body.pose) != 4:
        raise HTTPException(400, "pose must be 4x4")
    try:
        return {"job": SESSION.assist_snap(body.moving, body.anchor, body.pose, body.target_group)}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


class PanoPrepareBody(BaseModel):
    ids: list[str]


class PanoPickBody(BaseModel):
    sid: str
    u: float
    v: float
    pose: list[list[float]] | None = None


class PanoSolveBody(BaseModel):
    moving: str
    target: str
    pairs: list[dict[str, Any]]


@app.post("/api/pano/prepare")
def pano_prepare(body: PanoPrepareBody) -> dict[str, Any]:
    try:
        return {"job": SESSION.pano_prepare(body.ids)}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/scans/{sid}/pano.jpg")
def pano_image(sid: str) -> FileResponse:
    try:
        return FileResponse(SESSION.pano_image(sid), media_type="image/jpeg",
                            headers={"Cache-Control": "max-age=86400"})
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/pano/pick")
def pano_pick(body: PanoPickBody) -> dict[str, Any]:
    try:
        return SESSION.pano_pick(body.sid, body.u, body.v, body.pose)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/pano/solve")
def pano_solve(body: PanoSolveBody) -> dict[str, Any]:
    try:
        return SESSION.pano_solve(body.moving, body.target, body.pairs)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


class MarkerSetBody(BaseModel):
    moving: str
    target: str
    pairs: list[dict[str, Any]] | None = None


@app.post("/api/pano/markers/save")
def pano_markers_save(body: MarkerSetBody) -> dict[str, Any]:
    SESSION.save_markers(body.moving, body.target, body.pairs or [])
    return {"ok": True}


@app.post("/api/pano/markers/load")
def pano_markers_load(body: MarkerSetBody) -> dict[str, Any]:
    return {"markers": SESSION.load_markers(body.moving, body.target)}


class ManualPairBody(BaseModel):
    a: str
    b: str


@app.post("/api/manual/delete")
def manual_delete(body: ManualPairBody) -> dict[str, Any]:
    _require_unlocked()
    try:
        MANAGER.snapshot(label="pre-delete-manual-link")
    except Exception:
        pass
    try:
        res = SESSION.delete_manual_link(body.a, body.b)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    MANAGER.save()
    return res


@app.post("/api/manual/wipe-auto")
def wipe_automatic_links() -> dict[str, Any]:
    _require_unlocked()
    try:
        MANAGER.snapshot(label="pre-wipe-automatic-links")
    except Exception:
        pass
    res = SESSION.wipe_automatic_links()
    MANAGER.save()
    return res


class PlaceExactBody(BaseModel):
    moving: list[str]
    anchor: str
    pose: list[list[float]]
    partners: list[str]
    kind: str = "assistant"
    note: str = ""


@app.post("/api/assist/place-exact")
def assist_place_exact(body: PlaceExactBody) -> dict[str, Any]:
    _require_unlocked()
    try:
        MANAGER.snapshot(label="pre-place-exact")
    except Exception:
        pass
    try:
        res = SESSION.place_exact(body.moving, body.anchor, body.pose, body.partners, body.kind, body.note)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    MANAGER.save()
    return res


@app.post("/api/assist/suggest")
def assist_suggest(body: AssistBody) -> dict[str, Any]:
    try:
        return {"job": SESSION.assist_suggest(body.moving, body.anchor, body.target_group)}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/assist/accept")
def assist_accept(body: AssistBody) -> dict[str, Any]:
    _require_unlocked()
    if body.pose is None or len(body.pose) != 4:
        raise HTTPException(400, "pose must be 4x4")
    try:
        MANAGER.snapshot(label="pre-manual-place")
    except Exception:
        pass
    try:
        res = SESSION.assist_accept(body.moving, body.anchor, body.pose, force=body.force, markers=body.markers, kind=body.kind)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    MANAGER.save()
    return res


@app.post("/api/assist/undo")
def assist_undo() -> dict[str, Any]:
    _require_unlocked()
    try:
        res = SESSION.assist_undo()
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    MANAGER.save()
    return res


@app.post("/api/pairs/accuracy")
def pair_accuracy(body: PairAccuracyBody) -> dict[str, Any]:
    try:
        return SESSION.pair_accuracy(body.a, body.b)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/transform")
def set_transform(body: TransformBody) -> dict[str, Any]:
    _require_unlocked()
    if len(body.matrix) != 4:
        raise HTTPException(400, "matrix must be 4x4")
    SESSION.set_transform(body.id, body.matrix)
    return {"ok": True}


@app.post("/api/reference")
def set_reference(body: RefBody) -> dict[str, str]:
    _require_unlocked()
    SESSION.set_reference(body.id)
    return {"job": SESSION.solve()}


@app.post("/api/session/load")
def load_session(body: SessionLoadBody) -> dict[str, Any]:
    _require_unlocked()
    """Restore transforms, pairs and groups from a saved session.json."""
    path = Path(body.path).expanduser()
    if not path.exists():
        raise HTTPException(404, f"no such file: {path}")
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise HTTPException(400, f"cannot read session: {exc}")
    state = SESSION.restore_session(data)
    MANAGER.save()
    return state


@app.post("/api/reset-registration")
def reset_registration(body: ResetRegistrationBody | None = None) -> dict[str, Any]:
    _require_unlocked()
    try:
        MANAGER.snapshot(label="pre-reset-registration")
    except Exception:
        pass
    body = body or ResetRegistrationBody()
    SESSION.reset_registration(cleanups=body.cleanups, vetoes=body.vetoes, cached=body.cached)
    MANAGER.save()
    return {"ok": True}


@app.post("/api/reset")
def reset() -> dict[str, Any]:
    _require_unlocked()
    try:
        MANAGER.snapshot(label="pre-reset")
    except Exception:
        pass
    SESSION.reset_alignment()
    return {"ok": True}


@app.post("/api/clear")
def clear() -> dict[str, Any]:
    _require_unlocked()
    try:
        MANAGER.snapshot(label="pre-clear")
    except Exception:
        pass
    pid = MANAGER.active
    SESSION.clear()
    if pid:
        try:
            from . import ingest as ingest_mod

            ingest_mod.release_project(pid)
        except Exception:
            traceback.print_exc()
    return {"ok": True}


@app.delete("/api/scans/{sid}")
def remove_scan(sid: str) -> dict[str, Any]:
    try:
        MANAGER.snapshot(label="pre-remove-scan")
    except Exception:
        pass
    pid = MANAGER.active
    sha = SESSION.scans.get(sid).file_hash if sid in SESSION.scans else ""
    SESSION.remove_scan(sid)
    if pid and sha and not any(s.file_hash == sha for s in SESSION.scans.values()):
        try:
            from . import ingest as ingest_mod

            ingest_mod.unregister(sha, pid)
        except Exception:
            traceback.print_exc()
    return {"ok": True}


@app.post("/api/params")
def set_params(body: ParamsBody) -> dict[str, Any]:
    changed_reload = False
    with SESSION._lock:
        for key in ("voxel", "preview_points", "use_pose", "detect_features", "cleanup"):
            value = getattr(body, key)
            if value is not None and SESSION.params.get(key) != value:
                SESSION.params[key] = value
                changed_reload = True
        for key in ("load_workers", "pair_workers", "accept_score", "max_attempts"):
            value = getattr(body, key)
            if value is not None:
                SESSION.params[key] = value
        if changed_reload:
            # Geometry-affecting settings need a rebuild of the in-memory clouds,
            # but they must NOT throw away the links/poses the user already has.
            # Links are keyed by a settings signature in SQLite; stale links are
            # simply not reused by the next registration.
            for s in SESSION.scans.values():
                s.loaded = False
                s.loading = False
        SESSION._notify()
    return {"params": SESSION.params, "reload_required": changed_reload}


@app.post("/api/export")
def export(body: ExportBody) -> dict[str, str]:
    return {"job": SESSION.export(body.model_dump())}


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    j = MANAGER.job(job_id)
    if j is None:
        raise HTTPException(404, "unknown job")
    return j


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict[str, Any]:
    job = MANAGER.job(job_id)
    if job is not None:
        for project in MANAGER.projects.values():
            if project.session is not None and job_id in project.session.jobs:
                project.session.cancel_job(job_id)
                break
    return {"ok": True}


# --------------------------------------------------------------------------- #
# binary endpoints
# --------------------------------------------------------------------------- #

PREVIEW_MAGIC = b"E57P"
FEATURES_MAGIC = b"E57F"


@app.get("/api/mesh.bin")
def mesh_binary(path: str = Query(..., min_length=1)) -> Response:
    from pathlib import Path

    from .mesh import load_mesh_for_view

    p = Path(path).expanduser()
    if not p.exists() or p.suffix.lower() not in (".ply", ".obj", ".glb", ".gltf"):
        raise HTTPException(404, "mesh file not found")
    try:
        data = load_mesh_for_view(str(p))
    except Exception as exc:
        raise HTTPException(400, f"cannot read mesh: {exc}")
    V = np.asarray(data["vertices"], dtype=np.float32)
    F = np.asarray(data["faces"], dtype=np.uint32)
    C = np.asarray(data["colors"], dtype=np.float32)
    C8 = np.clip(C * 255.0, 0, 255).astype(np.uint8)
    header = struct.pack("<4sIII", b"E57M", 1, len(V), len(F))
    return Response(
        content=header + V.tobytes() + C8.tobytes() + F.tobytes(),
        media_type="application/octet-stream",
    )


@app.get("/api/scans/{sid}/preview.bin")
def preview(sid: str) -> Response:
    s = SESSION.scans.get(sid)
    if s is None or s.preview is None:
        raise HTTPException(404, "scan not loaded")
    pts = np.ascontiguousarray(s.preview, dtype=np.float32)
    n = len(pts)
    has_colors = s.colors is not None and len(s.colors) == n and s.colors.size > 0
    has_intensity = s.intensity is not None and len(s.intensity) == n and s.intensity.size > 0
    flags = (1 if has_colors else 0) | (2 if has_intensity else 0)
    parts = [PREVIEW_MAGIC, struct.pack("<III", 1, n, flags), pts.tobytes()]
    if has_colors:
        parts.append(np.ascontiguousarray(s.colors, dtype=np.uint8).tobytes())
    if has_intensity:
        parts.append(np.ascontiguousarray(s.intensity, dtype=np.float32).tobytes())
    return Response(b"".join(parts), media_type="application/octet-stream")


@app.get("/api/scans/{sid}/features.bin")
def features(sid: str) -> Response:
    s = SESSION.scans.get(sid)
    if s is None:
        raise HTTPException(404, "unknown scan")
    edge = s.edge_points if s.edge_points is not None else np.empty((0, 3), np.float32)
    plane = s.plane_points if s.plane_points is not None else np.empty((0, 3), np.float32)
    edge = np.ascontiguousarray(edge, dtype=np.float32)
    plane = np.ascontiguousarray(plane, dtype=np.float32)
    parts = [
        FEATURES_MAGIC,
        struct.pack("<II", len(edge), len(plane)),
        edge.tobytes(),
        plane.tobytes(),
    ]
    return Response(b"".join(parts), media_type="application/octet-stream")




def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import threading
    import webbrowser

    import uvicorn

    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\n  e57stitch running at {url}\n")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("  WARNING: listening beyond this machine. e57stitch has no login, and its\n"
              "  file browser and export can read and write any path the server can.\n"
              "  Only do this on a network you trust.\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
