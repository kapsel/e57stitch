"""In-memory session: scans, previews, pair results, poses, jobs."""

from __future__ import annotations

import copy
import itertools
import json
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from . import e57io
from . import mesh as mesh_mod
from .db import DB
from .workers import process_pool
from .e57io import CACHE_DIR, ScanRef, StreamSpec
from .features import (
    FEATURES_VERSION,
    ScanFeatures,
    clean_points,
    match_features,
    transfer_mask,
    voxel_down,
)
from .register import (
    ENGINE_CACHE,
    LOCK_TILT,
    TILT_MAX_DEG,
    cycle_consistency,
    invert,
    clamp_tilt,
    _pose_close,
    evaluate_fixed_job,
    refine_multiview,
    snap_to_model,
    solve_global_poses,
)

PREVIEW_BASE = 500_000

PALETTE = [
    "#ff6b6b",
    "#4ecdc4",
    "#ffd93d",
    "#a78bfa",
    "#6bcB77",
    "#ff9f45",
    "#4d96ff",
    "#f78fb3",
    "#00d2d3",
    "#feca57",
    "#c56cf0",
    "#7bed9f",
    "#ff7f50",
    "#70a1ff",
    "#eccc68",
    "#9c88ff",
    "#ffa502",
    "#7f8fa6",
]

# --------------------------------------------------------------------------- #
# worker: load one scan once and derive every representation we need
# --------------------------------------------------------------------------- #


def _slice_preview(preview: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    n = len(preview["points"])
    count = int(count or 0)
    if count <= 0 or n <= count:
        return preview
    idx = np.linspace(0, n - 1, count).astype(np.int64)
    out = {"points": np.ascontiguousarray(preview["points"][idx], dtype=np.float32)}
    colors = preview.get("colors")
    out["colors"] = (
        np.ascontiguousarray(colors[idx], dtype=np.uint8)
        if colors is not None and len(colors) == n
        else colors if colors is not None else np.empty((0, 3), np.uint8)
    )
    inten = preview.get("intensity")
    out["intensity"] = (
        np.ascontiguousarray(inten[idx], dtype=np.float32)
        if inten is not None and len(inten) == n
        else inten if inten is not None else np.empty(0, np.float32)
    )
    return out


def is_unused_candidate(p: dict[str, Any]) -> bool:
    """A candidate the selection rejected: not a link, and not the user's call."""
    return not p.get("accepted", True) and not p.get("manual") and not p.get("user_rejected")


def _pair_worker_init() -> None:
    """Avoid 8 workers each spawning dozens of BLAS/OpenMP threads."""
    import os

    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def _orientation_gravity(reg_points: np.ndarray, orientation: str) -> dict[str, Any]:
    """Detected gravity, with the user's orientation override applied."""
    from .features import estimate_gravity

    grav = estimate_gravity(reg_points)
    if orientation == "level":
        return {
            "up": np.asarray([0.0, 0.0, 1.0]),
            "confidence": 1.0,
            "mode": "level",
            "tilt_deg": 0.0,
        }
    if orientation == "free":
        # Keep the detected axis for structure scoring, but disable all
        # gravity-constrained matching.
        grav = dict(grav)
        grav["confidence"] = 0.0
        grav["mode"] = "unknown"
    return grav


def prepare_scan_task(task: dict[str, Any]) -> dict[str, Any]:
    """Process-pool worker: read a scan and build preview + registration data.

    Everything is cached on disk, so re-opening a project is instant.
    """
    import numpy as _np

    from . import e57io as _io
    from .features import detect_features as _detect

    ref_dict = task["ref"]
    ref = ScanRef(
        id=task["sid"],
        file=ref_dict["file"],
        index=int(ref_dict["index"]),
        name=ref_dict["name"],
        point_count=int(ref_dict["point_count"]),
        fields=list(ref_dict["fields"]),
        pose=_np.asarray(ref_dict["pose"], dtype=_np.float64),
        has_pose=bool(ref_dict["has_pose"]),
        bounds=ref_dict.get("bounds"),
    )
    reg_voxel = float(task["reg_voxel"])
    preview_points = int(task["preview_points"])
    apply_pose = bool(task["apply_pose"])
    want_features = bool(task.get("features", True))
    cleanup = str(task.get("cleanup") or "light")
    orientation = str(task.get("orientation") or "auto")
    seed = int(task.get("seed", 0))
    # An E57 pose from the scanner carries its IMU levelling, which beats any
    # gravity guess from the points (a staircase scan read as "sideways").
    # "auto" therefore means level whenever that pose is applied; the user
    # can still force "free" for a scan whose IMU is not trustworthy.
    gravity_mode = "level" if orientation == "auto" and apply_pose and ref.has_pose else orientation

    stat = os.stat(ref.file)
    key = e57io._cache_key(
        Path(ref.file).resolve(),
        stat.st_size,
        stat.st_mtime_ns,
        ref.index,
        round(reg_voxel, 6),
        PREVIEW_BASE,
        apply_pose,
        want_features,
        cleanup,
        seed,
        "" if orientation == "auto" else orientation,
    )
    folder = CACHE_DIR / "scans" / key
    reg_path = str(folder / "reg.npy")
    t0 = time.time()
    from .db import DB as _DB
    from .features import FEATURES_VERSION as _FV

    scan_key = f"{task.get('file_hash') or ref.file}:{ref.index}"
    art_params = (
        f"{round(reg_voxel, 4)}|{cleanup}|{int(want_features)}|{int(apply_pose)}|"
        f"{PREVIEW_BASE}|{_FV}|{orientation}"
    )

    if (folder / "meta.json").exists() and (folder / "reg.npy").exists():
        meta = json.loads((folder / "meta.json").read_text())
        with _np.load(folder / "preview.npz", allow_pickle=False) as z:
            preview = {k: z[k] for k in z.files}
        overlay = {}
        if (folder / "overlay.npz").exists():
            with _np.load(folder / "overlay.npz", allow_pickle=False) as z:
                overlay = {k: z[k] for k in z.files}
        features_summary = {}
        if (folder / "features.json").exists():
            features_summary = json.loads((folder / "features.json").read_text())
        regrav = meta.get("orientation", "auto") != gravity_mode or "up" not in meta
        if regrav:
            old_up = _np.asarray(meta.get("up") or [0.0, 0.0, 1.0], dtype=float)
            reg_points_for_gravity = _np.load(reg_path)
            grav = _orientation_gravity(reg_points_for_gravity, gravity_mode)
            # features (corner lines) depend on the up axis
            regrav = float(_np.dot(old_up, _np.asarray(grav["up"], dtype=float))) < 0.999
            meta.update(
                {
                    "orientation": gravity_mode,
                    "up": _np.asarray(grav["up"], dtype=float).tolist(),
                    "up_confidence": float(grav["confidence"]),
                    "up_mode": str(grav["mode"]),
                    "tilt_deg": float(grav["tilt_deg"]),
                }
            )
            (folder / "meta.json").write_text(json.dumps(meta))
        if want_features and (regrav or int(features_summary.get("version") or 0) < _FV):
            # feature definitions changed (vertical corner lines): recompute
            # from the cached registration cloud, not the whole E57.

            reg_points = _np.load(reg_path)
            feats = _detect(reg_points, reg_voxel, up=_np.asarray(meta.get("up") or [0, 0, 1]))
            features_summary = feats.summary()
            overlay = {
                "edge_points": feats.edge_points,
                "plane_points": feats.plane_points,
                "plane_ids": feats.plane_ids,
            }
            _np.savez(folder / "overlay.npz", **overlay)
            (folder / "features.json").write_text(json.dumps(features_summary))
        _DB.put_artifact(scan_key, "bundle", art_params, str(folder), meta)
        _DB.put_noise(scan_key, art_params, int(meta.get("cleaned", 0)), float(meta.get("cleaned_pct", 0.0)))
        preview = _slice_preview(preview, preview_points)
        return {
            "sid": task["sid"],
            "reg_path": reg_path,
            "reg_count": int(meta.get("reg_count", 0)),
            "preview": preview,
            "overlay": overlay,
            "features": features_summary,
            "bbox": meta.get("bbox"),
            "cleaned": int(meta.get("cleaned", 0)),
            "cleaned_pct": float(meta.get("cleaned_pct", 0.0)),
            "up": meta.get("up"),
            "up_confidence": float(meta.get("up_confidence", 0.0)),
            "up_mode": str(meta.get("up_mode", "unknown")),
            "tilt_deg": float(meta.get("tilt_deg", 0.0)),
            "orientation": orientation,
            "cached": True,
            "seconds": round(time.time() - t0, 2),
        }

    full = _io.read_full(ref, apply_pose=apply_pose)
    points = full["points"]
    colors = full["colors"]
    intensity = full["intensity"]

    folder.mkdir(parents=True, exist_ok=True)
    reg_all, _keep = e57io._voxel_subsample(points, reg_voxel)
    reg_points = reg_all
    if cleanup not in ("off", "none", "false", "0", ""):
        reg_points = clean_points(reg_all, reg_voxel, cleanup)
    cleaned = int(len(reg_all) - len(reg_points))
    cleaned_pct = float(cleaned / max(len(reg_all), 1))
    _np.save(folder / "reg.npy", reg_points.astype(_np.float32, copy=False))

    n = len(points)
    base_points = PREVIEW_BASE
    if n > base_points:
        rng = _np.random.default_rng(seed)
        sel = rng.choice(n, base_points, replace=False)
        sel.sort()
    else:
        sel = _np.arange(n)
    preview = {
        "points": _np.ascontiguousarray(points[sel], dtype=_np.float32),
        "colors": _np.ascontiguousarray(colors[sel], dtype=_np.uint8)
        if colors.size
        else _np.empty((0, 3), _np.uint8),
        "intensity": _np.ascontiguousarray(intensity[sel], dtype=_np.float32)
        if intensity.size
        else _np.empty(0, _np.float32),
    }
    if cleanup not in ("off", "none", "false", "0", "") and len(reg_points):
        keep_mask = transfer_mask(reg_points, preview["points"], max(3.0 * reg_voxel, 0.15))
        if keep_mask.sum() >= 100:
            preview = {
                "points": _np.ascontiguousarray(preview["points"][keep_mask], dtype=_np.float32),
                "colors": _np.ascontiguousarray(preview["colors"][keep_mask], dtype=_np.uint8)
                if preview["colors"].size
                else preview["colors"],
                "intensity": _np.ascontiguousarray(preview["intensity"][keep_mask], dtype=_np.float32)
                if preview["intensity"].size
                else preview["intensity"],
            }
    _np.savez(folder / "preview.npz", **preview)

    grav = _orientation_gravity(reg_points, gravity_mode)
    features_summary: dict[str, Any] = {}
    overlay: dict[str, np.ndarray] = {}
    if want_features:
        feats = _detect(reg_points, reg_voxel, up=_np.asarray(grav["up"]))
        features_summary = feats.summary()
        overlay = {
            "edge_points": feats.edge_points,
            "plane_points": feats.plane_points,
            "plane_ids": feats.plane_ids,
        }
        _np.savez(folder / "overlay.npz", **overlay)
        (folder / "features.json").write_text(json.dumps(features_summary))

    bbox_src = reg_points if len(reg_points) else points
    bbox = [bbox_src.min(axis=0).tolist(), bbox_src.max(axis=0).tolist()]
    meta = {
        "reg_count": int(len(reg_points)),
        "preview_count": int(len(preview["points"])),
        "full_count": int(n),
        "cleaned": cleaned,
        "cleaned_pct": cleaned_pct,
        "bbox": bbox,
        "reg_voxel": reg_voxel,
        "cleanup": cleanup,
        "orientation": gravity_mode,
        "up": _np.asarray(grav["up"], dtype=float).tolist(),
        "up_confidence": float(grav["confidence"]),
        "up_mode": str(grav["mode"]),
        "tilt_deg": float(grav["tilt_deg"]),
    }
    (folder / "meta.json").write_text(json.dumps(meta))
    _DB.put_artifact(scan_key, "bundle", art_params, str(folder), meta)
    _DB.put_noise(scan_key, art_params, int(cleaned), float(cleaned_pct))
    preview = _slice_preview(preview, preview_points)
    return {
        "sid": task["sid"],
        "reg_path": reg_path,
        "reg_count": int(len(reg_points)),
        "preview": preview,
        "overlay": overlay,
        "features": features_summary,
        "bbox": bbox,
        "cleaned": cleaned,
        "cleaned_pct": cleaned_pct,
        "up": meta["up"],
        "up_confidence": meta["up_confidence"],
        "up_mode": meta["up_mode"],
        "tilt_deg": meta["tilt_deg"],
        "orientation": orientation,
        "cached": False,
        "seconds": round(time.time() - t0, 2),
    }


# --------------------------------------------------------------------------- #
# state objects
# --------------------------------------------------------------------------- #


@dataclass
class ScanState:
    sid: str
    ref: ScanRef
    color: str
    apply_pose: bool = True
    loaded: bool = False
    loading: bool = False
    included: bool = True
    visible: bool = True
    file_hash: str = ""
    cleaned: int = 0
    cleaned_pct: float = 0.0
    error: str | None = None
    reg_path: str | None = None
    reg_count: int = 0
    preview: np.ndarray | None = None
    colors: np.ndarray | None = None
    intensity: np.ndarray | None = None
    edge_points: np.ndarray | None = None
    plane_points: np.ndarray | None = None
    features: dict[str, Any] = field(default_factory=dict)
    bbox: list[list[float]] | None = None
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    display_offset: np.ndarray = field(default_factory=lambda: np.eye(4))
    group: int = -1
    orientation: str = "auto"  # auto | level | free
    up: list[float] | None = None
    up_confidence: float = 0.0
    up_mode: str = "unknown"
    tilt_deg: float = 0.0
    confidence: str = "unlinked"  # confirmed | single link | unlinked (from the last selection)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.sid,
            "label": self.ref.label,
            "file": self.ref.file,
            "index": self.ref.index,
            "name": self.ref.name,
            "point_count": self.ref.point_count,
            "fields": self.ref.fields,
            "has_pose": self.ref.has_pose,
            "color": self.color,
            "loaded": self.loaded,
            "loading": self.loading,
            "included": self.included,
            "visible": self.visible,
            "file_hash": self.file_hash,
            "cleaned": self.cleaned,
            "cleaned_pct": self.cleaned_pct,
            "error": self.error,
            "reg_count": self.reg_count,
            "preview_count": int(len(self.preview)) if self.preview is not None else 0,
            "bbox": self.bbox,
            "transform": self.transform.tolist(),
            # where the scanner stood, in the scan's own frame
            "origin": self.ref.pose[:3, 3].tolist() if self.apply_pose and self.ref.has_pose else [0.0, 0.0, 0.0],
            "display_offset": self.display_offset.tolist(),
            "group": self.group,
            "orientation": self.orientation,
            "up": self.up,
            "up_confidence": self.up_confidence,
            "up_mode": self.up_mode,
            "tilt_deg": self.tilt_deg,
            "confidence": self.confidence,
            "features": {
                "plane_count": self.features.get("plane_count", 0),
                "edge_count": self.features.get("edge_count", 0),
                "plane_significant": self.features.get("plane_significant", 0),
                "edge_significant": self.features.get("edge_significant", 0),
                "plane_area": self.features.get("plane_area", 0.0),
            },
        }


class Session:
    def __init__(self) -> None:
        self.scans: dict[str, ScanState] = {}
        self.pairs: dict[str, dict[str, Any]] = {}
        # pairs whose candidates the selection rejected: not links and not
        # listed, only the work list Improve links retries
        self.unresolved_pairs: list[list[str]] = []
        # poses the user rejected: {"a", "b", "transform"}; never re-chosen
        self.link_vetoes: list[dict[str, Any]] = []
        self.cleanups: list[dict[str, Any]] = []
        self.groups: list[dict[str, Any]] = []
        self.group_of: dict[str, int] = {}
        self.reference: str | None = None
        self.params: dict[str, Any] = {
            "voxel": 0.05,
            "preview_points": 120_000,
            "load_workers": max(2, min(4, (os.cpu_count() or 4) // 2)),
            "pair_workers": max(2, min(8, os.cpu_count() or 4)),
            "accept_score": 0.15,
            "use_pose": True,
            "detect_features": True,
            "cleanup": "light",
            "max_attempts": 2,
        }
        self.jobs: dict[str, dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="e57stitch-job")
        self._counter = 0
        self.last_result: dict[str, Any] | None = None
        self.baseline: dict[str, list] = {}
        self._busy_job: str | None = None
        self._pools: set[Any] = set()
        self._lock = threading.RLock()
        self._block_stash: dict[frozenset, dict[str, Any]] = {}  # rows set aside by a block, for an exact Undo
        self.on_change: Callable[[], None] | None = None
        # manual placements are read from (and written to) the permanent
        # manual store; the benchmark runner switches this off
        self.use_manual_store = True
        self.project_name = ""

    # ------------------------------------------------------------------ files

    def add_files(self, paths: list[str]) -> dict[str, Any]:
        added = []
        skipped = []
        for raw in paths:
            p = str(Path(raw).expanduser().resolve())
            if Path(p).is_dir():
                for child in sorted(Path(p).iterdir()):
                    if child.suffix.lower() == ".e57":
                        res = self._add_one(str(child))
                        (added if res else skipped).append(str(child))
                continue
            res = self._add_one(p)
            (added if res else skipped).append(p)
        self._notify()
        return {"added": added, "skipped": skipped, "scans": [s.to_dict() for s in self.scans.values()]}

    def _add_one(self, path: str) -> bool:
        existing = {(s.ref.file, s.ref.index) for s in self.scans.values()}
        try:
            refs = e57io.probe_file(path)
        except Exception:
            return False
        added_any = False
        for ref in refs:
            if (str(Path(ref.file).resolve()), ref.index) in existing:
                continue
            added_any = True
            sid = f"s{self._counter}"
            self._counter += 1
            ref.id = sid
            self.scans[sid] = ScanState(
                sid=sid,
                ref=ref,
                color=PALETTE[len(self.scans) % len(PALETTE)],
                apply_pose=bool(self.params.get("use_pose", True)),
            )
        return added_any

    def remove_scan(self, sid: str) -> None:
        self.scans.pop(sid, None)
        self.pairs = {k: v for k, v in self.pairs.items() if v["a"] != sid and v["b"] != sid}
        self.unresolved_pairs = [u for u in self.unresolved_pairs if sid not in u]
        if self.reference == sid:
            self.reference = None
        self._notify()

    def clear(self) -> None:
        self.baseline = {}
        self.scans.clear()
        self.pairs.clear()
        self.unresolved_pairs = []
        self.groups.clear()
        self.group_of.clear()
        self.cleanups.clear()
        self.reference = None
        self._counter = 0
        self.last_result = None
        self._notify()

    # ------------------------------------------------------------------- jobs

    def _new_job(self, kind: str, fn: Callable[[Callable], Any], lane: str = "main") -> str:
        """Queue a job.  ``lane="assist"`` runs beside the main job (manual
        placement must work while registration improves in the background)."""
        job_id = uuid.uuid4().hex[:10]
        self.jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "progress": 0.0,
            "message": "",
            "error": None,
            "result": None,
            "started": time.time(),
            "finished": None,
            "cancel": False,
        }

        def run() -> None:
            if self._cancelled(job_id):
                self._update(
                    job_id,
                    status="done",
                    progress=1.0,
                    result={"cancelled": True},
                    message="cancelled before start",
                    finished=time.time(),
                )
                self._notify()
                return
            if lane == "main":
                self._busy_job = job_id
            self._update(job_id, status="running", message="starting")
            try:
                result = fn(lambda p, msg=None, **kw: self._update(job_id, progress=p, message=msg, **kw))
                self._update(job_id, status="done", progress=1.0, result=result, message="done")
            except Exception as exc:  # pragma: no cover - surfaced in the UI
                if self._cancelled(job_id):
                    self._update(
                        job_id,
                        status="done",
                        progress=1.0,
                        result={"cancelled": True, "error": str(exc)},
                        message="cancelled",
                    )
                else:
                    traceback.print_exc()
                    self._update(job_id, status="error", error=str(exc), message=str(exc))
            finally:
                self._update(job_id, finished=time.time())
                if lane == "main":
                    self._busy_job = None
                self._notify()

        if lane == "assist":
            if getattr(self, "_assist_executor", None) is None:
                self._assist_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="e57stitch-assist")
            self._assist_executor.submit(run)
        else:
            self._executor.submit(run)
        return job_id

    def _notify(self) -> None:
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception:
                traceback.print_exc()

    def _update(self, job_id: str, **kw: Any) -> None:
        """Update a job.  Besides progress/message a job can report
        ``plan`` (list of stage names), ``stage`` (current stage) and
        ``done``/``total`` (items in that stage); stage changes are timed so
        the UI can estimate what is left."""
        job = self.jobs.get(job_id)
        if job is None:
            return
        now = time.time()
        stage = kw.get("stage")
        if stage is not None and stage != job.get("stage"):
            hist = job.setdefault("stages", [])
            if hist and hist[-1].get("finished") is None:
                hist[-1]["finished"] = now
            hist.append({"name": stage, "started": now, "finished": None})
            job["stage_started"] = now
            job.pop("done", None)
            job.pop("total", None)
        if kw.get("progress") is not None:
            kw["progress"] = max(float(job.get("progress") or 0.0), float(kw["progress"]))
        if kw.get("message") is None:
            kw.pop("message", None)
        job.update(kw)
        job["updated"] = now

    def _cancelled(self, job_id: str) -> bool:
        return bool(self.jobs.get(job_id, {}).get("cancel"))

    @staticmethod
    def _terminate_pool(pool: ProcessPoolExecutor) -> None:
        """Kill in-flight worker processes, then shut the pool down.

        ``shutdown(cancel_futures=True)`` only cancels queued work; a pair
        matcher that is already running at 2 cm can keep eight cores busy for
        minutes after the user pressed Stop.  Terminating the child processes
        makes Stop immediate and prevents orphan workers from blocking the next
        job.
        """
        try:
            for proc in list((getattr(pool, "_processes", None) or {}).values()):
                try:
                    proc.terminate()
                except Exception:
                    pass
        finally:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def cancel_job(self, job_id: str) -> None:
        if job_id not in self.jobs:
            return
        self.jobs[job_id]["cancel"] = True
        self.jobs[job_id]["status"] = "cancelling"
        self.jobs[job_id]["message"] = "stopping…"
        # Do not wait for 2 cm pair workers to finish their current scan.
        for pool in list(self._pools):
            self._terminate_pool(pool)
        self._notify()

    def job(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    # ------------------------------------------------------------------- load

    def load(self, ids: list[str] | None = None, *, force: bool = False) -> str:
        ids = ids or list(self.scans.keys())
        ids = [i for i in ids if i in self.scans]

        def job(progress: Callable) -> dict[str, Any]:
            return self._load_impl(progress, ids, force=force)

        return self._new_job("load", job)

    def _load_impl(
        self,
        progress: Callable,
        ids: list[str],
        *,
        force: bool = False,
        lo: float = 0.0,
        hi: float = 1.0,
    ) -> dict[str, Any]:
        """Load scans (used by the Load button and by the Go-deep runner)."""
        targets = [s for s in (self.scans[i] for i in ids) if force or not s.loaded]
        if not targets:
            return {"loaded": 0, "scans": len(ids)}
        for s in targets:
            s.loading = True
            s.error = None
        tasks = [
            {
                "sid": s.sid,
                "ref": {
                    "file": s.ref.file,
                    "index": s.ref.index,
                    "name": s.ref.name,
                    "point_count": s.ref.point_count,
                    "fields": s.ref.fields,
                    "pose": s.ref.pose.tolist(),
                    "has_pose": s.ref.has_pose,
                    "bounds": s.ref.bounds,
                },
                "reg_voxel": float(self.params["voxel"]),
                "preview_points": int(self.params["preview_points"]),
                "apply_pose": bool(s.apply_pose),
                "features": bool(self.params["detect_features"]),
                "cleanup": str(self.params.get("cleanup", "light")),
                "orientation": s.orientation,
                "file_hash": s.file_hash,
            }
            for s in targets
        ]
        total = len(tasks)
        done = 0
        workers = max(1, min(int(self.params["load_workers"]), total))
        pool = process_pool(workers)
        self._pools.add(pool)
        futures = {pool.submit(prepare_scan_task, t): t["sid"] for t in tasks}
        cancelled = False
        try:
            for fut in as_completed(futures):
                if self._cancelled(self._busy_job or ""):
                    cancelled = True
                    break
                sid = futures[fut]
                s = self.scans.get(sid)
                try:
                    res = fut.result()
                except Exception as exc:
                    if s:
                        s.loading = False
                        s.error = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                    done += 1
                    progress(lo + (hi - lo) * done / total, f"failed {sid}")
                    continue
                if s is not None:
                    with self._lock:
                        s.loaded = True
                        s.loading = False
                        s.reg_path = res["reg_path"]
                        s.reg_count = res["reg_count"]
                        preview = res["preview"]
                        s.preview = preview["points"]
                        s.colors = preview.get("colors")
                        s.intensity = preview.get("intensity")
                        overlay = res.get("overlay") or {}
                        s.edge_points = overlay.get("edge_points")
                        s.plane_points = overlay.get("plane_points")
                        s.features = res.get("features") or {}
                        s.cleaned = int(res.get("cleaned", 0))
                        s.cleaned_pct = float(res.get("cleaned_pct", 0.0))
                        s.bbox = res.get("bbox")
                        s.orientation = str(res.get("orientation") or s.orientation)
                        s.up = res.get("up")
                        s.up_confidence = float(res.get("up_confidence", 0.0))
                        s.up_mode = str(res.get("up_mode", "unknown"))
                        s.tilt_deg = float(res.get("tilt_deg", 0.0))
                done += 1
                label = self.scans[sid].ref.label if sid in self.scans else sid
                progress(lo + (hi - lo) * done / total, f"loaded {label}")
        finally:
            if cancelled:
                self._terminate_pool(pool)
                for s in self.scans.values():
                    if s.loading:
                        s.loading = False
            else:
                pool.shutdown(wait=True)
            self._pools.discard(pool)
        if not self.pairs:
            self._recompute_groups_from_pairs()
        self._layout_groups()
        return {
            "loaded": len(targets),
            "scans": len([s for s in self.scans.values() if s.loaded]),
            "bbox": self.scene_bbox(),
        }

    def _fine_reg_paths(
        self,
        progress: Callable,
        ids: list[str],
        voxel: float,
        *,
        lo: float = 0.0,
        hi: float = 1.0,
    ) -> dict[str, str]:
        """Registration clouds at ``voxel`` without touching the scan state.

        Matching runs on coarse bundles (wide ICP basin, tolerant scoring);
        only the final polish and the accuracy measurement need fine clouds.
        Bundles are disk-cached per voxel, so this is cheap after the first
        time.  Scan state, params and the link cache signature stay coarse.
        """
        tasks = []
        for sid in ids:
            s = self.scans.get(sid)
            if s is None or not s.loaded:
                continue
            tasks.append(
                {
                    "sid": s.sid,
                    "ref": {
                        "file": s.ref.file,
                        "index": s.ref.index,
                        "name": s.ref.name,
                        "point_count": s.ref.point_count,
                        "fields": s.ref.fields,
                        "pose": s.ref.pose.tolist(),
                        "has_pose": s.ref.has_pose,
                        "bounds": s.ref.bounds,
                    },
                    "reg_voxel": float(voxel),
                    "preview_points": int(self.params["preview_points"]),
                    "apply_pose": bool(s.apply_pose),
                    # same flag as the normal load so existing fine bundles
                    # (e.g. from earlier 2 cm runs) are cache hits
                    "features": bool(self.params["detect_features"]),
                    "cleanup": str(self.params.get("cleanup", "light")),
                    "orientation": s.orientation,
                    "file_hash": s.file_hash,
                }
            )
        out: dict[str, str] = {}
        if not tasks:
            return out
        total = len(tasks)
        workers = max(1, min(int(self.params["load_workers"]), total))
        pool = process_pool(workers)
        self._pools.add(pool)
        futures = {pool.submit(prepare_scan_task, t): t["sid"] for t in tasks}
        cancelled = False
        try:
            for done, fut in enumerate(as_completed(futures), start=1):
                if self._cancelled(self._busy_job or ""):
                    cancelled = True
                    break
                sid = futures[fut]
                try:
                    out[sid] = fut.result()["reg_path"]
                except Exception:
                    traceback.print_exc()
                progress(lo + (hi - lo) * done / total, f"fine {voxel * 100:.0f} cm: {self._label(sid)}")
        finally:
            if cancelled:
                self._terminate_pool(pool)
            else:
                pool.shutdown(wait=True)
            self._pools.discard(pool)
        return out

    # ------------------------------------------------------------- registration

    def _scan_key(self, sid: str) -> str:
        s = self.scans.get(sid)
        if s is None:
            return str(sid)
        if s.file_hash:
            return f"{s.file_hash}:{s.ref.index}"
        try:
            st = os.stat(s.ref.file)
            return f"{os.path.realpath(s.ref.file)}|{st.st_size}|{st.st_mtime_ns}|{s.ref.index}"
        except Exception:
            return f"{s.ref.file}|{s.ref.index}"

    def _params_sig(self) -> str:
        import hashlib

        payload = (
            round(float(self.params.get("voxel", 0.05)), 6),
            str(self.params.get("cleanup", "light")),
            bool(self.params.get("use_pose", True)),
            FEATURES_VERSION,
            ENGINE_CACHE,
            tuple(
                sorted(
                    (sid, s.orientation)
                    for sid, s in self.scans.items()
                    if s.orientation != "auto"
                )
            ),
        )
        return hashlib.sha1(repr(payload).encode()).hexdigest()[:16]

    def _label(self, sid: str) -> str:
        s = self.scans.get(sid)
        if s is not None:
            return s.ref.label
        if isinstance(sid, str) and sid.startswith("m") and sid[1:].isdigit():
            return f"group {chr(65 + int(sid[1:]))}"
        return str(sid)

    def _store_pair(self, res: dict[str, Any]) -> None:
        with self._lock:
            self._store_pair_unlocked(res)

    def _store_pair_unlocked(self, res: dict[str, Any]) -> None:
        if self.is_blocked(res["a"], res["b"]):
            return
        res.setdefault("params_sig", self._params_sig())
        key = f"{res['a']}|{res['b']}"
        reverse = f"{res['b']}|{res['a']}"
        other = self.pairs.get(reverse)
        if other is not None and (
            res.get("accepted")
            or not other.get("accepted")
            or float(res.get("score") or 0.0) >= float(other.get("score") or 0.0)
        ):
            self.pairs.pop(reverse, None)
        sa, sb = self.scans.get(res["a"]), self.scans.get(res["b"])
        if sa and sb:
            try:
                res["feature_matches"] = self._match_features(sa, sb, np.asarray(res["transform"]))
            except Exception:
                res["feature_matches"] = {}
        fm = res.get("feature_matches") or {}
        planes_tight = int(fm.get("planes_tight", 0) or 0)
        edges_tight = int(fm.get("edges_tight", 0) or 0)
        plane_area = float(fm.get("planes_area_matched", 0.0) or 0.0)
        corner_matches = int(res.get("corner_matches", 0) or 0)
        feature_score = min(
            1.0,
            0.18 * planes_tight
            + 0.10 * edges_tight
            + 0.12 * min(1.0, corner_matches / 4.0)
            + min(0.45, plane_area / 40.0),
        )
        res["feature_score"] = float(feature_score)
        structure_overlap = float(res.get("structure_overlap", res.get("score", 0.0)) or 0.0)
        structural_weak = (
            not res.get("area_ok")
            and planes_tight == 0
            and edges_tight == 0
            and structure_overlap < max(0.4, float(self.params.get("accept_score", 0.2)) * 1.5)
        )
        flat_only = bool(res.get("flat_only") or structural_weak)
        res["flat_only"] = flat_only
        wall_ok = bool(
            res.get("wall_ok")
            if res.get("wall_ok") is not None
            else (res.get("wall_score") is not None and float(res.get("wall_score") or 0.0) >= float(self.params.get("accept_score", 0.2)) * 1.2)
        )
        corner_ok = bool(res.get("corner_ok") if res.get("corner_ok") is not None else corner_matches >= 2)
        res["wall_ok"] = wall_ok
        res["corner_ok"] = corner_ok
        res["verified"] = bool(res.get("accepted") and not flat_only and (wall_ok or corner_ok))
        ambiguous = bool(res.get("ambiguous", False))
        score = float(res.get("score", 0.0) or 0.0)
        # Floor-only links still connect scans, but must not dominate the
        # weighted pose graph when walls are available from other links.
        if res.get("manual"):
            res["verified"] = True
            res["weight"] = max(0.6, score)
        elif flat_only or ambiguous:
            res["weight"] = score * 0.25
        else:
            res["weight"] = score * (0.7 + 0.3 * feature_score)
        self.pairs[key] = res

    def _match_features(self, sa: ScanState, sb: ScanState, T: np.ndarray) -> dict[str, Any]:
        if not sa.features or not sb.features:
            return {}
        fa = _features_from_summary(sa.features)
        fb = _features_from_summary(sb.features)
        # Candidate tolerance is generous (clusters get split differently by
        # DBSCAN); ``edges_tight`` only counts matches within 10 cm and is what
        # the accuracy figure uses.
        return match_features(
            fa,
            fb,
            T,
            tol=max(0.35, float(self.params["voxel"]) * 7),
            up_a=np.asarray(sa.up) if sa.up is not None else None,
            up_b=np.asarray(sb.up) if sb.up is not None else None,
        )

    # ------------------------------------------------------------ pipeline

    def _run_logger(self, name: str) -> tuple[Callable[[str], None], Callable[[], None], Path]:
        run_dir = CACHE_DIR / "runs" / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "run.log"
        session_path = run_dir / "session.json"

        def log(msg: str) -> None:
            with open(log_path, "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

        def autosave() -> None:
            tmp = session_path.with_suffix(".tmp.json")
            tmp.write_text(json.dumps(self.session_data(), indent=1))
            os.replace(tmp, session_path)

        log(f"{name}: run directory {run_dir}")
        return log, autosave, run_dir

    def register_scans(
        self, *, fresh: bool = False, keep_improving: bool = True, hours: float = 10.0,
        voxel: float | None = 0.05, refine_voxel: float | None = 0.02,
    ) -> str:
        """Stages 1-4 (minutes), then optionally stage 5 in the same job."""
        from .pipeline import Pipeline

        self.params["auto_links"] = True  # pressing Register switches automatic links back on

        if voxel and abs(float(voxel) - float(self.params["voxel"])) > 1e-6:
            with self._lock:
                self.params["voxel"] = float(voxel)
                for sc in self.scans.values():  # bundles are per voxel: reload
                    sc.loaded = False
                    sc.loading = False

        def job(progress: Callable) -> dict[str, Any]:
            log, autosave, run_dir = self._run_logger("register")
            plan = ["Load scans", "Neighbours", "Neighbours: try harder", "Find candidates", "Panoramas", "Choose links",
                    "Refine", "Polish"]
            if keep_improving:
                plan.append("Improve (background)")
            progress(0.0, "starting", plan=plan, open_ended=bool(keep_improving))
            ids = [i for i in self.scans if self.scans[i].included]
            if any(not self.scans[i].loaded for i in ids):
                progress(0.0, "loading scans", stage="Load scans")
                self._load_impl(progress, ids, lo=0.0, hi=0.05)
            if fresh:
                # Forcing only recomputes the stages this run reaches, so older results
                # (piece rescue, heavy matchers) would still feed selection.  Saved
                # candidates moved a benchmark B fresh run 12 cm off a clean one.
                from .db import DB

                dropped = DB.drop_hyps([self._scan_key(i) for i in ids])
                log(f"fresh: dropped {dropped} stored result(s)")
            pipe = Pipeline(self)
            result = pipe.register(progress, log, fresh=fresh, refine_voxel=refine_voxel)
            autosave()
            if keep_improving and not self._cancelled(self._busy_job or ""):
                result["improve"] = pipe.improve(progress, log, hours=hours, refine_voxel=refine_voxel, autosave=autosave)
                autosave()
            summary = self.summary()
            summary.update(result)
            summary["log_path"] = str(run_dir / "run.log")
            return summary

        return self._new_job("register", job)

    def improve(self, *, hours: float = 10.0, refine_voxel: float | None = 0.02) -> str:
        """Stage 5 only: heavy matchers where they can still help."""
        from .pipeline import Pipeline

        self.params["auto_links"] = True

        def job(progress: Callable) -> dict[str, Any]:
            log, autosave, run_dir = self._run_logger("improve")
            progress(0.0, "starting", plan=["Improve (background)"], open_ended=True)
            result = Pipeline(self).improve(progress, log, hours=hours, refine_voxel=refine_voxel, autosave=autosave)
            autosave()
            summary = self.summary()
            summary.update(result)
            summary["log_path"] = str(run_dir / "run.log")
            return summary

        return self._new_job("improve", job)

    def realign(self) -> str:
        """Re-run the global selection on the stored hypotheses (fast)."""
        from .pipeline import Pipeline

        def job(progress: Callable) -> dict[str, Any]:
            Pipeline(self).select_and_apply()
            self._refine_impl(progress, 0.2, 0.9, rounds=1, refresh=False)
            self._solve_and_apply(progress, 0.9, 1.0, preserve=True)
            return self.summary()

        return self._new_job("realign", job)

    def rematch_pair(self, a: str, b: str) -> str:
        """Run every heavy matcher on one pair, then re-select."""
        from .pipeline import Pipeline
        from .register import heavy_job

        if self.is_blocked(a, b):
            raise ValueError("this link is blocked: unblock it in the links list first")

        def job(progress: Callable) -> dict[str, Any]:
            pipe = Pipeline(self)
            pipe.run([pipe.job(a, b)], heavy_job, "heavy", progress, 0.0, 0.9, "heavy", force=True)
            pipe.select_and_apply()
            return self.summary()

        return self._new_job("rematch", job)

    def solve(self) -> str:
        def job(progress: Callable) -> dict[str, Any]:
            # Toggling a link should only change connectivity, never move the
            # already-refined scans.
            self._solve_and_apply(progress, preserve=True)
            return self.summary()

        return self._new_job("solve", job)

    def _solve_and_apply(
        self, progress: Callable, lo: float = 0.0, hi: float = 1.0, *, preserve: bool = False
    ) -> None:
        with self._lock:
            self._solve_and_apply_unlocked(progress, lo, hi, preserve=preserve)

    def _solve_and_apply_unlocked(
        self, progress: Callable, lo: float = 0.0, hi: float = 1.0, *, preserve: bool = False
    ) -> None:
        # Pose solving only needs the link graph and transforms, not the point
        # clouds.  Include scans whose clouds are not loaded (e.g. just after a
        # server restart) so re-solving does not silently produce empty groups.
        ids = list(self.scans.keys())
        progress(lo, "solving global poses")
        fallback = {i: self.scans[i].transform for i in ids}
        accepted = [p for p in self.pairs.values() if p.get("accepted")]
        ignored: set[str] = set()
        solution = None
        if preserve:
            solution = solve_global_poses(
                ids,
                accepted,
                reference=self.reference,
                fallback=fallback,
                preserve_poses=True,
            )
        else:
            # Iteratively drop links that the pose graph itself calls
            # inconsistent.  Removing them can split a scan off into its own
            # group, which is the honest answer for a scan that only shares a
            # repeated facade.
            for _ in range(4):
                active = [p for p in accepted if f"{p['a']}|{p['b']}" not in ignored]
                solution = solve_global_poses(ids, active, reference=self.reference, fallback=fallback)
                found = set()
                for g in solution["groups"]:
                    for out in g.get("outliers", []):
                        found.add(f"{out['a']}|{out['b']}")
                found -= ignored
                found = {k for k in found if not (self.pairs.get(k) or {}).get("manual")}
                if not found:
                    break
                ignored |= found
                progress(lo + (hi - lo) * 0.4, f"re-solving without {len(ignored)} inconsistent link(s)")

        if solution is None:
            return
        for sid, pose in solution["poses"].items():
            if sid in self.scans:
                P = np.asarray(pose, dtype=np.float64)
                if LOCK_TILT and self.scans[sid].up_mode == "level":
                    # tilt belongs to the IMU (± the refine's small
                    # correction); links made before the tilt lock or by
                    # 6-DoF matchers must not tilt the scan further
                    P = clamp_tilt(P, TILT_MAX_DEG)
                self.scans[sid].transform = P
        self.groups = solution["groups"]
        self.group_of = solution["group_of"]
        if solution.get("reference"):
            self.reference = solution["reference"]
        for s in self.scans.values():
            s.group = self.group_of.get(s.sid, -1)

        for p in self.pairs.values():
            p["outlier"] = False
            p.pop("outlier_reason", None)
        outlier_keys = set(ignored)
        for g in self.groups:
            for key, residual in g.get("residuals", {}).items():
                if key in self.pairs:
                    self.pairs[key]["residual"] = residual
            for out in g.get("outliers", []):
                outlier_keys.add(f"{out['a']}|{out['b']}")
                if f"{out['a']}|{out['b']}" in self.pairs:
                    self.pairs[f"{out['a']}|{out['b']}"]["outlier_reason"] = (
                        f"{out.get('rot_deg', 0):.1f}° / {out.get('trans', 0):.2f} m"
                    )
        for key in outlier_keys:
            if key in self.pairs:
                self.pairs[key]["outlier"] = True
        try:
            cycles = cycle_consistency(
                [p for p in self.pairs.values() if not p.get("synthetic")]
            )
            for p in self.pairs.values():
                if not p.get("accepted") or p.get("synthetic"):
                    continue
                a, b = p.get("a"), p.get("b")
                key = f"{a}|{b}" if a < b else f"{b}|{a}"
                if key in cycles:
                    p["cycle"] = cycles[key]
        except Exception:
            traceback.print_exc()
        self._layout_groups()
        progress(hi, "poses solved")

    def _refine_impl(
        self,
        progress: Callable,
        lo: float,
        hi: float,
        *,
        rounds: int = 2,
        refresh: bool = True,
        paths: dict[str, str] | None = None,
        voxel: float | None = None,
        on_round: Callable[[int, float, float], None] | None = None,
    ) -> None:
        """Multi-view ICP over each group.

        ``paths``/``voxel`` override the registration clouds, so the final
        polish can run on finer bundles than the ones used for matching.
        """
        ids = [i for i in self.scans if self.scans[i].loaded]
        if paths is None:
            paths = {i: self.scans[i].reg_path for i in ids if self.scans[i].reg_path}
        refine_voxel = float(voxel if voxel is not None else self.params["voxel"])
        if len(ids) < 2:
            return
        poses = {i: self.scans[i].transform.copy() for i in ids}
        groups = self.groups or [{"members": ids, "reference": self.reference}]
        n = max(1, len(groups))
        for gi, g in enumerate(groups):
            members = [m for m in g.get("members", []) if m in paths]
            if len(members) < 2:
                continue
            refined = refine_multiview(
                members,
                {m: paths[m] for m in members},
                list(self.pairs.values()),
                {m: poses[m] for m in members},
                refine_voxel,
                reference=g.get("reference"),
                rounds=max(1, int(rounds)),
                min_score=float(self.params["accept_score"]),
                progress=lambda p, gi=gi: progress(
                    lo + (hi - lo) * ((gi + p) / n), "multi-view refine"
                ),
                should_cancel=lambda: self._cancelled(self._busy_job or ""),
                on_round=on_round,
                levelled=[m for m in members if self.scans[m].up_mode == "level"],
            )
            if self._cancelled(self._busy_job or ""):
                break
            poses.update(refined)
        for sid, P in poses.items():
            if sid in self.scans:
                self.scans[sid].transform = P
        if refresh:
            self._refresh_pair_residuals()
        self._layout_groups()

    def _consensus_impl(self, log: Callable[[str], None] | None = None) -> None:
        """Fit each scan to every overlapping scan of its group.  Scans with a
        manual link do not move, and neither does a group's reference."""
        from . import register as R

        if not R.CONSENSUS_FIT:
            return
        manual = {m for p in self.pairs.values() if p.get("manual") and p.get("accepted") for m in (p["a"], p["b"])}
        t0 = time.time()
        moved = 0.0
        for g in self.groups:
            members = [m for m in g.get("members", []) if self.scans[m].loaded and self.scans[m].reg_path]
            if len(members) < 2:
                continue
            fixed = manual | ({g["reference"]} if g.get("reference") else set())
            fitted = R.consensus_fit(
                members,
                {m: self.scans[m].reg_path for m in members},
                {m: self.scans[m].transform for m in members},
                float(self.params["voxel"]),
                fixed=fixed,
                should_cancel=lambda: self._cancelled(self._busy_job or ""),
            )
            for m, P in fitted.items():
                moved = max(moved, float(np.linalg.norm(P[:3, 3] - self.scans[m].transform[:3, 3])))
                self.scans[m].transform = P
        if log:
            log(f"fit to all overlapping scans: largest move {moved * 1000:.0f} mm, {time.time() - t0:.0f} s")

    def refine(self, voxel: float = 0.02, rounds: int = 8) -> str:
        """Whole-building multi-view ICP on fine clouds; kept only if the fit improves."""
        from .pipeline import Pipeline

        def job(progress: Callable) -> dict[str, Any]:
            log, autosave, run_dir = self._run_logger("refine")
            result = Pipeline(self).refine_all(progress, log, voxel=voxel, rounds=rounds)
            autosave()
            summary = self.summary()
            summary.update(result)
            summary["log_path"] = str(run_dir / "run.log")
            return summary

        return self._new_job("refine", job)

    # ------------------------------------------------------- merged-model growth

    def _group_summary(self) -> str:
        if not self.groups:
            return "no groups"
        parts = []
        for g in self.groups:
            tag = "unlinked" if g.get("unconnected") else f"{g.get('edges', 0)} links"
            parts.append(f"{chr(65 + g['id'])}={len(g['members'])} ({tag})")
        return ", ".join(parts)

    def _build_group_models(
        self, progress: Callable | None = None, lo: float = 0.0, hi: float = 1.0
    ) -> dict[int, dict[str, Any]]:
        """Synthesize one merged cloud per registered group, in that group's frame."""
        with self._lock:
            groups = [dict(g) for g in self.groups if len(g.get("members", [])) >= 2]
            snapshots = {}
            for g in groups:
                members = [m for m in g["members"] if m in self.scans and self.scans[m].reg_path]
                snapshots[g["id"]] = (
                    members,
                    {m: self.scans[m].transform.copy() for m in members},
                    {m: self.scans[m].reg_path for m in members},
                    g.get("reference"),
                )
        models: dict[int, dict[str, Any]] = {}
        voxel = float(self.params["voxel"])
        out_dir = CACHE_DIR / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        n = max(1, len(snapshots))
        for k, (gid, (members, poses, paths, reference)) in enumerate(snapshots.items()):
            if len(members) < 2:
                continue
            parts = []
            for m in members:
                pts = np.load(paths[m])
                # The merged cloud is capped at 600 k points anyway; sampling
                # each member first keeps this cheap at 2 cm resolution.
                if len(pts) > 400_000:
                    rng = np.random.default_rng(0)
                    pts = pts[rng.choice(len(pts), 400_000, replace=False)]
                P = poses[m]
                parts.append((pts @ P[:3, :3].T + P[:3, 3]).astype(np.float32, copy=False))
            merged = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
            merged = voxel_down(merged, voxel)
            if len(merged) > 600_000:
                rng = np.random.default_rng(0)
                merged = merged[rng.choice(len(merged), 600_000, replace=False)]
            key = e57io._cache_key(
                "model",
                tuple(sorted(members)),
                tuple(np.round(np.concatenate([poses[m] for m in members]), 4).tolist()),
                round(voxel, 6),
            )
            path = out_dir / f"{key}.npy"
            if not path.exists():
                np.save(path, np.ascontiguousarray(merged, dtype=np.float32))
            models[gid] = {
                "path": str(path),
                "members": members,
                "reference": reference,
                "count": int(len(merged)),
            }
            if progress:
                progress(
                    lo + (hi - lo) * (k + 1) / n,
                    f"merged model {chr(65 + gid)}: {len(merged):,} pts from {len(members)} scans",
                )
        return models

    # ---------------------------------------------------------------- go deep

    def verify_alignment(self) -> dict[str, Any]:
        """Cheap sanity gate before an unattended export.

        A good pair score is not enough: a link can be locally plausible yet
        disagree with alternate paths (repeated facades).  Refuse to publish an
        automatic output when such a link is present; the user can inspect it
        and export deliberately.
        """
        accepted = [p for p in self.pairs.values() if p.get("accepted")]
        reasons: list[str] = []
        if not accepted:
            reasons.append("no accepted links")
        bad = []
        for p in accepted:
            if p.get("synthetic"):
                continue
            cyc = p.get("cycle") or {}
            if float(cyc.get("min_trans", 0.0)) > 1.0 or float(cyc.get("min_rot", 0.0)) > 10.0:
                bad.append(
                    f"{self._label(p['a'])}↔{self._label(p['b'])} "
                    f"{float(cyc.get('min_trans', 0.0)):.2f}m/{float(cyc.get('min_rot', 0.0)):.1f}°"
                )
        if bad:
            reasons.append("cycle-inconsistent links: " + ", ".join(bad[:4]))
        flat = [p for p in accepted if p.get("flat_only")]
        if flat:
            reasons.append(f"{len(flat)} flat-only / underconstrained link(s)")
        ambiguous = [p for p in accepted if p.get("ambiguous")]
        if ambiguous:
            reasons.append(f"{len(ambiguous)} ambiguous link(s)")
        if self.groups and len(self.groups[0].get("members", [])) < 2:
            reasons.append("main group has fewer than two scans")
        bbox = self.scene_bbox()
        if bbox:
            diag = float(np.linalg.norm(np.asarray(bbox["max"]) - np.asarray(bbox["min"])))
            if diag > 100_000:
                reasons.append(f"implausible scene size ({diag:.0f} m)")
        acc = self.summary().get("accuracy")
        if acc and acc.get("links", 0) >= 3 and float(acc.get("mean_mm") or 0) > 300:
            reasons.append(f"mean surface deviation {acc['mean_mm']:.0f} mm")
        return {
            "ok": not reasons,
            "reasons": reasons,
            "accepted": len(accepted),
            "groups": len(self.groups),
        }

    def _measure_accuracy_impl(
        self,
        progress: Callable,
        lo: float,
        hi: float,
        *,
        paths: dict[str, str] | None = None,
        voxel: float | None = None,
    ) -> None:
        pairs = [p for p in self.pairs.values() if p.get("accepted")]
        total = max(1, len(pairs))
        for i, p in enumerate(pairs):
            if self._cancelled(self._busy_job or ""):
                break
            a, b = p.get("a"), p.get("b")
            try:
                res = self.pair_accuracy(a, b, paths=paths, voxel=voxel)
                p["accuracy"] = res["accuracy"]
                p["feature_matches"] = res["feature_matches"]
            except Exception:
                traceback.print_exc()
            progress(lo + (hi - lo) * (i + 1) / total, f"measured {self._label(a)} ↔ {self._label(b)}")

    def restore_session(self, data: dict[str, Any]) -> dict[str, Any]:
        """Re-apply transforms/links/groups from a saved session JSON."""
        with self._lock:
            id_map: dict[str, str] = {}
            for entry in data.get("scans", []):
                file = str(entry.get("file", ""))
                index = int(entry.get("index", 0))
                if not file:
                    continue
                existing = next(
                    (s for s in self.scans.values() if s.ref.file == file and s.ref.index == index),
                    None,
                )
                if existing is None:
                    try:
                        refs = e57io.probe_file(file)
                        ref = refs[index]
                    except Exception:
                        continue
                    sid = f"s{self._counter}"
                    self._counter += 1
                    ref.id = sid
                    self.scans[sid] = ScanState(
                        sid=sid,
                        ref=ref,
                        color=PALETTE[len(self.scans) % len(PALETTE)],
                        apply_pose=bool(self.params.get("use_pose", True)),
                    )
                    existing = self.scans[sid]
                id_map[str(entry.get("id"))] = existing.sid
                existing.transform = np.asarray(entry.get("transform", np.eye(4)), dtype=np.float64)

            self.pairs = {}
            self.unresolved_pairs = [
                [id_map.get(str(u[0]), str(u[0])), id_map.get(str(u[1]), str(u[1]))]
                for u in data.get("unresolved_pairs") or []
            ]
            for p in data.get("pairs", []):
                a = id_map.get(str(p.get("a")), str(p.get("a")))
                b = id_map.get(str(p.get("b")), str(p.get("b")))
                if a not in self.scans or b not in self.scans or not p.get("transform"):
                    continue
                if is_unused_candidate(p):  # saved before these stopped being listed
                    self.unresolved_pairs.append([a, b])
                    continue
                q = dict(p)
                q["a"], q["b"] = a, b
                q.setdefault("accepted", True)
                q.setdefault("params_sig", self._params_sig())
                self.pairs[f"{a}|{b}"] = q
            self.drop_blocked_rows()  # a restored file may predate the block
            self.reference = id_map.get(str(data.get("reference")), data.get("reference"))
            baseline = data.get("baseline") or {}
            self.baseline = {
                id_map.get(str(k), str(k)): v for k, v in baseline.items() if id_map.get(str(k)) in self.scans
            }
            try:
                from .cleanup import normalize_region

                self.cleanups = []
                for c in data.get("cleanups") or []:
                    region = normalize_region(c)
                    if region.get("scans"):
                        region["scans"] = [
                            id_map.get(str(sid), str(sid)) for sid in region["scans"]
                        ]
                    self.cleanups.append(region)
            except Exception:
                self.cleanups = []
            self.groups = []
            for g in data.get("groups", []):
                members = [id_map.get(str(m), str(m)) for m in g.get("members", [])]
                members = [m for m in members if m in self.scans]
                if not members:
                    continue
                outliers = g.get("outliers", [])
                self.groups.append(
                    {
                        **g,
                        "id": len(self.groups),
                        "members": members,
                        "reference": id_map.get(str(g.get("reference")), g.get("reference")),
                        "outliers": outliers if isinstance(outliers, list) else [],
                        "outlier_count": int(outliers) if isinstance(outliers, (int, float)) else len(outliers),
                    }
                )
                for m in members:
                    self.scans[m].group = len(self.groups) - 1
            self.groups.sort(key=lambda g: len(g["members"]), reverse=True)
            for gi, g in enumerate(self.groups):
                g["id"] = gi
                for m in g["members"]:
                    self.scans[m].group = gi
            self.group_of = {m: g["id"] for g in self.groups for m in g["members"]}
            self._layout_groups()
            self._notify()
            return self.state()

    # ---------------------------------------------------------- manual assist
    #
    # The UI lets the user drag a scan (or a whole group) roughly into place
    # over another group.  "Snap" then runs a *local* registration from that
    # guess against the nearest target scans, "Find it for me" lists the best
    # automatic placements to choose from, and "Accept" stores the result as
    # user-confirmed (manual) links that later automatic runs never drop.

    def _world_center(self, sid: str, pose: np.ndarray | None = None) -> np.ndarray:
        s = self.scans[sid]
        T = np.asarray(pose if pose is not None else s.transform, dtype=np.float64)
        if s.bbox:
            c = (np.asarray(s.bbox[0], dtype=np.float64) + np.asarray(s.bbox[1], dtype=np.float64)) / 2
        else:
            c = np.zeros(3)
        return T[:3, :3] @ c + T[:3, 3]

    def _assist_target_ids(self, moving: list[str], target_group: int | None) -> list[str]:
        moving_set = set(moving)
        if target_group is not None:
            g = next((g for g in self.groups if int(g.get("id", -1)) == int(target_group)), None)
            members = list(g.get("members", [])) if g else []
        else:
            members = [i for i in self.scans]
        return [
            i for i in members
            if i not in moving_set and i in self.scans and self.scans[i].loaded and self.scans[i].reg_path
        ]

    @staticmethod
    def _assist_quality(res: dict[str, Any], accept: float) -> dict[str, Any]:
        acc = res.get("accuracy") or {}
        mm = acc.get("mean_mm")
        area = float(res.get("wall_area") or 0.0)
        score = float(res.get("score") or 0.0)
        if res.get("accepted") and mm is not None and mm < 30:
            level, head = "good", "Great fit"
        elif (res.get("accepted") or area >= 60 or score >= accept * 0.75) and (mm is None or mm < 80):
            level, head = "ok", "Plausible fit — check the walls"
        else:
            level, head = "bad", "Doesn't fit here"
        detail = []
        if mm is not None:
            detail.append(f"{mm:.0f} mm surface error")
        if area:
            detail.append(f"{area:.0f} m² shared walls")
        detail.append(f"{round(score * 100)}% overlap")
        return {"level": level, "headline": head, "detail": " · ".join(detail)}

    def assist_snap(
        self,
        moving: list[str],
        anchor: str,
        pose: list[list[float]],
        target_group: int | None = None,
        targets: int = 4,
    ) -> str:
        """Local registration of ``anchor`` from a hand-placed world pose."""
        guess = np.asarray(pose, dtype=np.float64)
        if anchor not in self.scans or not self.scans[anchor].reg_path:
            raise ValueError("the scan to place is not loaded")
        cand = self._assist_target_ids(moving, target_group)
        if not cand:
            raise ValueError("no loaded scans to snap against in that group")
        here = self._world_center(anchor, guess)
        # nearest in 3D, not on the floor plan: in a stairwell the scans
        # straight above/below are "nearest" on the plan but see other floors
        cand.sort(key=lambda i: float(np.linalg.norm(self._world_center(i) - here)))
        cand = cand[: max(1, int(targets))]

        def job(progress: Callable) -> dict[str, Any]:
            from .pipeline import Pipeline

            voxel = float(self.params["voxel"])
            accept = float(self.params["accept_score"])
            sa = self.scans[anchor]
            pipe = Pipeline(self)
            results: list[dict[str, Any]] = []
            pool = process_pool(max(1, min(len(cand), int(self.params["pair_workers"]))), initializer=_pair_worker_init)
            self._pools.add(pool)
            snap: dict[str, Any] = {}
            try:
                # 1) one local, trust-region search against the merged neighbours
                progress(0.05, f"snapping {self._label(anchor)} onto {len(cand)} nearby scan(s)")
                level = sa.up_mode == "level" and all(self.scans[t].up_mode == "level" for t in cand)
                task = {
                    "path": sa.reg_path,
                    "targets": [(self.scans[t].reg_path, self.scans[t].transform.tolist()) for t in cand],
                    "init": guess.tolist(),
                    "voxel": voxel,
                    "level": level,
                }
                try:
                    snap = pool.submit(snap_to_model, task).result()
                except Exception:
                    traceback.print_exc()
                    snap = {"error": "snap crashed"}
                if self._cancelled(self._busy_job or ""):
                    return {"error": "cancelled"}
                if snap.get("pose"):
                    world_guess = np.asarray(snap["pose"], dtype=np.float64)
                    fn, label = evaluate_fixed_job, "checking"
                else:
                    # nothing fits near the placement: keep it where the user
                    # put it and report the fit there (never jump elsewhere)
                    world_guess, fn, label = guess, evaluate_fixed_job, "checking"
                # 2) evidence per neighbour for exactly that pose
                jobs = [pipe.job(anchor, t, init=invert(self.scans[t].transform) @ world_guess,
                                 source=snap.get("method") or "manual") for t in cand]
                futs = {pool.submit(fn, j): j for j in jobs}
                for k, fut in enumerate(as_completed(futs), start=1):
                    if self._cancelled(self._busy_job or ""):
                        break
                    try:
                        r = fut.result()
                        r["snap_shift_m"] = snap.get("shift_m", 0.0)
                        r["snap_turn_deg"] = snap.get("turn_deg", 0.0)
                        results.append(r)
                    except Exception:
                        traceback.print_exc()
                    progress(0.3 + 0.6 * k / len(jobs), f"{label} against {k}/{len(jobs)}")
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
                self._pools.discard(pool)
            if not results:
                return {"error": snap.get("error") or "snap failed"}

            def rank(r: dict[str, Any]) -> tuple:
                wall = r.get("wall_score")
                q = 0.55 * float(r.get("score") or 0) + 0.45 * float(wall if wall is not None else r.get("score") or 0)
                return (bool(r.get("accepted")), q, float(r.get("wall_area") or 0))

            results.sort(key=rank, reverse=True)
            best = results[0]
            T_b = self.scans[best["b"]].transform
            world = T_b @ np.asarray(best["transform"], dtype=np.float64)
            # Re-express every result relative to the chosen pose so stored
            # links are mutually consistent; keep those that agree with it.
            links = []
            for r in results:
                Tt = self.scans[r["b"]].transform
                own = Tt @ np.asarray(r["transform"], dtype=np.float64)
                if r is not best and not (r.get("accepted") and _pose_close(own, world, trans=0.10, rot_deg=1.5)):
                    continue
                q = dict(r)
                q["transform"] = (invert(Tt) @ world).tolist()
                links.append(q)
            quality = self._assist_quality(best, accept)
            with self._lock:
                self._assist_last = {
                    "anchor": anchor,
                    "moving": list(moving),
                    "pose": world.tolist(),
                    "links": links,
                }
            progress(1.0, quality["headline"])
            return {
                "pose": world.tolist(),
                "quality": quality,
                "method": snap.get("method") or best.get("method"),
                "snap_note": None if snap.get("pose") else snap.get("error"),
                "best": {
                    "target": best["b"],
                    "target_label": self._label(best["b"]),
                    "accepted": bool(best.get("accepted")),
                    "reason": best.get("rejection_reason"),
                    "shift_m": best.get("snap_shift_m"),
                    "turn_deg": best.get("snap_turn_deg"),
                    "mean_mm": (best.get("accuracy") or {}).get("mean_mm"),
                    "wall_area": best.get("wall_area"),
                    "score": best.get("score"),
                },
                "links": len(links),
                "targets": [self._label(t) for t in cand],
            }

        return self._new_job("assist-snap", job, lane="assist")

    # ------------------------------------------------------- panorama markers

    def _pano_folder(self, sid: str):
        from .panorama import pano_dir

        sc = self.scans.get(sid)
        if sc is None:
            raise ValueError(f"unknown scan {sid}")
        return pano_dir(self._scan_key(sid), bool(sc.apply_pose))

    def pano_prepare(self, ids: list[str]) -> str:
        """Build (or reuse) the panoramas of ``ids``; ~15 s per scan the first time."""
        from . import panorama as P

        ids = [i for i in ids if i in self.scans]
        if not ids:
            raise ValueError("no such scans")

        def job(progress: Callable) -> dict[str, Any]:
            out = {}
            for k, sid in enumerate(ids):
                folder = self._pano_folder(sid)
                if not P.is_built(folder):
                    progress(k / len(ids), f"building panorama of {self._label(sid)} ({k + 1}/{len(ids)})")
                    P.build(self.scans[sid].ref, bool(self.scans[sid].apply_pose), folder)
                m = P.meta(folder)
                out[sid] = {"w": m["w"], "h": m["h"], "origin": m["origin"], "colour": m.get("colour", False)}
            progress(1.0, "panoramas ready")
            return {"panoramas": out}

        return self._new_job("panorama", job, lane="assist")

    def pano_image(self, sid: str):
        from . import panorama as P

        folder = self._pano_folder(sid)
        if not P.is_built(folder):
            raise ValueError("panorama not built yet")
        return folder / "pano.jpg"

    def pano_pick(self, sid: str, u: float, v: float, pose: list[list[float]] | None = None) -> dict[str, Any]:
        """3D point under a panorama pixel: scan-local and in the world (with
        ``pose`` instead of the scan's transform, e.g. a scan being placed)."""
        from . import panorama as P

        folder = self._pano_folder(sid)
        if not P.is_built(folder):
            raise ValueError("panorama not built yet")
        pk = P.pick(folder, u, v)
        if pk is None:
            return {"ok": False, "reason": "no measurement there (sky, glass or too far)"}
        T = np.asarray(pose, dtype=np.float64) if pose is not None else self.scans[sid].transform
        loc = np.asarray(pk["local"], dtype=np.float64)
        pk["world"] = (T[:3, :3] @ loc + T[:3, 3]).tolist()
        pk["ok"] = True
        return pk

    def pano_solve(self, moving: str, target: str, pairs: list[dict[str, Any]]) -> dict[str, Any]:
        """World pose of ``moving`` from marker pairs [{m: [u, v], t: [u, v]}]."""
        from . import panorama as P

        fm, ft = self._pano_folder(moving), self._pano_folder(target)
        Tt = self.scans[target].transform
        world, local, used = [], [], []
        for k, pr in enumerate(pairs):
            a = P.pick(fm, *pr["m"])
            b = P.pick(ft, *pr["t"])
            if a is None or b is None:
                continue
            local.append(a["local"])
            world.append((Tt[:3, :3] @ np.asarray(b["local"]) + Tt[:3, 3]).tolist())
            used.append(k)
        level = self.scans[moving].up_mode == "level" and self.scans[target].up_mode == "level"
        res = P.solve(np.asarray(world), np.asarray(local), level=level)
        res["used"] = used
        res["level"] = level  # levelled: yaw + translation only, tilt stays the IMU's
        return res

    def assist_suggest(self, moving: list[str], anchor: str, target_group: int | None = None) -> str:
        """Best placements of ``anchor`` in the target group from all stored
        hypotheses (computing floor-plan candidates for missing pairs)."""
        from .pipeline import Pipeline
        from .register import candidates_job

        if anchor not in self.scans or not self.scans[anchor].reg_path:
            raise ValueError("the scan to place is not loaded")
        cand = self._assist_target_ids(moving, target_group)
        if not cand:
            raise ValueError("no loaded scans in that group")

        def job(progress: Callable) -> dict[str, Any]:
            pipe = Pipeline(self)
            res = pipe.run([pipe.job(anchor, t) for t in cand], candidates_job, "candidates",
                           progress, 0.0, 0.9, "searching")
            stored = pipe.stored()
            options: list[dict[str, Any]] = []
            for t in cand:
                for (a, b) in ((anchor, t), (t, anchor)):
                    for st in (stored.get((a, b)) or {}).values():
                        for h in st.get("hyps") or []:
                            T = np.asarray(h["transform"], dtype=np.float64)
                            if a != anchor:
                                T = invert(T)
                            options.append({
                                "pose": (self.scans[t].transform @ T).tolist(),
                                "rank": float(h.get("rank") or 0.0),
                                "score": float(h.get("score") or 0.0),
                                "wall": h.get("wall_score"),
                                "plausible": bool(h.get("plausible")),
                                "target": t,
                                "target_label": self._label(t),
                                "method": h.get("source"),
                            })
            options.sort(key=lambda o: (o["plausible"], o["rank"]), reverse=True)
            unique: list[dict[str, Any]] = []
            for o in options:
                P = np.asarray(o["pose"])
                same = next((u for u in unique if _pose_close(P, np.asarray(u["pose"]), trans=0.5, rot_deg=5.0)), None)
                if same is not None:
                    same["votes"] += 1
                    continue
                o["votes"] = 1
                unique.append(o)
                if len(unique) >= 5:
                    break
            progress(1.0, f"{len(unique)} placement(s) found")
            return {"options": unique}

        return self._new_job("assist-suggest", job, lane="assist")

    def _record_manual(self, a: str, b: str, T: np.ndarray, kind: str, evidence: dict[str, Any] | None = None,
                       markers: Any = None) -> None:
        """Append a manual placement to the permanent store (never wiped)."""
        if not self.use_manual_store:
            return
        try:
            from .manual_store import store

            sa, sb = self.scans[a], self.scans[b]
            store().add(project=self.project_name, a_key=self._scan_key(a), b_key=self._scan_key(b),
                        a_label=self._label(a), b_label=self._label(b), a_file=str(sa.ref.file), b_file=str(sb.ref.file),
                        transform=T, world_a=sa.transform, world_b=sb.transform, kind=kind,
                        evidence=evidence, markers=markers)
        except Exception:
            traceback.print_exc()

    def _retract_manual(self, a: str, b: str, reason: str) -> None:
        if not self.use_manual_store:
            return
        try:
            from .manual_store import store

            store().retract(self._scan_key(a), self._scan_key(b), reason)
        except Exception:
            traceback.print_exc()

    def blocked_links(self) -> list[dict[str, Any]]:
        """The user's banned scan pairs among this session's scans."""
        if not self.use_manual_store:
            return []
        from .manual_store import store

        sid_of = {self._scan_key(i): i for i in self.scans}
        return [{"a": sid_of[r["a_key"]], "b": sid_of[r["b_key"]], "created": r["created"], "project": r["project"]}
                for r in store().blocked(sid_of)]

    def blocked_pairs(self) -> set[frozenset[str]]:
        return {frozenset((r["a"], r["b"])) for r in self.blocked_links()}

    def is_blocked(self, a: str, b: str) -> bool:
        return frozenset((a, b)) in self.blocked_pairs()

    def drop_blocked_rows(self) -> None:
        blocked = self.blocked_pairs()
        if not blocked:
            return
        self.pairs = {k: p for k, p in self.pairs.items() if frozenset((p["a"], p["b"])) not in blocked}
        self.unresolved_pairs = [u for u in self.unresolved_pairs if frozenset(u) not in blocked]

    def set_link_blocked(self, a: str, b: str, blocked: bool) -> bool:
        """Ban a scan pair from every kind of link, or lift the ban.  Stored by
        scan content outside the project, so resets, Start clean and restoring
        an older save cannot bring the link back."""
        from .manual_store import store

        if a not in self.scans or b not in self.scans or a == b:
            raise ValueError("unknown scan pair")
        if not self.use_manual_store:
            raise ValueError("blocking needs the permanent store")
        pair = frozenset((a, b))
        stash = self._block_stash
        restored = False
        with self._lock:
            if blocked:
                # keep the row aside so Undo can put it back exactly; no scan
                # moves here, the layout only changes on the next re-choose
                rows = [copy.deepcopy(p) for p in self.pairs.values() if frozenset((p["a"], p["b"])) == pair]
                unresolved = [list(u) for u in self.unresolved_pairs if frozenset(u) == pair]
                was_manual = any(p.get("manual") for p in rows)
                store().block(project=self.project_name, a_key=self._scan_key(a), b_key=self._scan_key(b),
                              a_label=self._label(a), b_label=self._label(b))
                self._retract_manual(a, b, "blocked by you")
                self.drop_blocked_rows()
                if not was_manual:
                    stash[pair] = {"rows": rows, "unresolved": unresolved,
                                   "poses": {i: self.scans[i].transform.copy() for i in (a, b)}}
            else:
                store().unblock(self._scan_key(a), self._scan_key(b))
                kept = stash.pop(pair, None)
                same_poses = kept is not None and all(
                    np.allclose(self.scans[i].transform, kept["poses"][i]) for i in (a, b))
                exists = any(frozenset((p["a"], p["b"])) == pair for p in self.pairs.values())
                if same_poses and not exists:
                    for p in kept["rows"]:
                        self.pairs[f"{p['a']}|{p['b']}"] = p
                    self.unresolved_pairs.extend(kept["unresolved"])
                    restored = bool(kept["rows"])
            self._recompute_groups_from_pairs()
        self._notify()
        return restored

    def manual_overview(self) -> dict[str, list[dict[str, Any]]]:
        """Per scan: its active manual links (partner, kind, when, markers)."""
        if not self.use_manual_store:
            return {}
        try:
            from .manual_store import store

            sid_of = {self._scan_key(i): i for i in self.scans}
            marks = store().marker_counts(sid_of)
            out: dict[str, list[dict[str, Any]]] = {}
            for row in store().active(sid_of):
                a, b = sid_of[row["a_key"]], sid_of[row["b_key"]]
                n = marks.get(frozenset((row["a_key"], row["b_key"])), 0)
                p = self.pairs.get(f"{a}|{b}") or self.pairs.get(f"{b}|{a}") or {}
                status = "applied" if p.get("accepted") else ("contradicted" if p else "pending")
                for x, y in ((a, b), (b, a)):
                    out.setdefault(x, []).append({"partner": y, "partner_label": self._label(y), "kind": row["kind"],
                                                  "created": row["created"], "markers": n, "id": row["id"],
                                                  "status": status, "reason": p.get("rejection_reason")})
            return out
        except Exception:
            traceback.print_exc()
            return {}

    def import_project_manual_links(self) -> int:
        """Copy manual links that exist only in the project into the store."""
        if not self.use_manual_store:
            return 0
        from .manual_store import store

        have = {frozenset((r["a_key"], r["b_key"])) for r in store().active({self._scan_key(i) for i in self.scans})}
        n = 0
        for p in list(self.pairs.values()):
            if not (p.get("manual") and p.get("accepted") and p.get("transform")):
                continue
            a, b = p["a"], p["b"]
            if a not in self.scans or b not in self.scans:
                continue
            if frozenset((self._scan_key(a), self._scan_key(b))) in have:
                continue
            self._record_manual(a, b, np.asarray(p["transform"]), "imported",
                                {"note": "placed before the manual store existed", "method": p.get("method")})
            n += 1
        return n

    def wipe_automatic_links(self) -> dict[str, Any]:
        """Remove every automatic link and its cached candidates; keep manual
        placements; switch automatic linking off until Register is pressed."""
        from .db import DB
        from .pipeline import Pipeline

        imported = self.import_project_manual_links()
        DB.drop_hyps([self._scan_key(i) for i in self.scans])
        with self._lock:
            self.params["auto_links"] = False
            self.pairs = {k: p for k, p in self.pairs.items() if p.get("manual") and p.get("accepted")}
            self.unresolved_pairs = []
        Pipeline(self).select_and_apply(world=False)
        self._notify()
        return {"imported": imported, "manual": len(self.pairs), "summary": self.summary()}

    def delete_manual_link(self, a: str, b: str) -> dict[str, Any]:
        """Stop using the manual link between a and b.  The store keeps it as
        retracted history (never deleted); the scans are re-solved without it."""
        from .pipeline import Pipeline

        if a not in self.scans or b not in self.scans:
            raise ValueError("unknown scan")
        self._retract_manual(a, b, "deleted by you")
        with self._lock:
            for key in (f"{a}|{b}", f"{b}|{a}"):
                p = self.pairs.get(key)
                if p is not None and p.get("manual"):
                    self.pairs.pop(key, None)
        Pipeline(self).select_and_apply(world=False)
        self._notify()
        return {"ok": True, "summary": self.summary()}

    def save_markers(self, moving: str, target: str, pairs: list[dict[str, Any]]) -> None:
        if not self.use_manual_store:
            return
        from .manual_store import store

        store().save_markers(project=self.project_name, moving_key=self._scan_key(moving), target_key=self._scan_key(target),
                             pairs=pairs, moving_label=self._label(moving), target_label=self._label(target))

    def load_markers(self, moving: str, target: str) -> dict[str, Any] | None:
        from .manual_store import store

        return store().latest_markers(self._scan_key(moving), self._scan_key(target))

    def assist_accept(self, moving: list[str], anchor: str, pose: list[list[float]], force: bool = False,
                      markers: Any = None, kind: str | None = None) -> dict[str, Any]:
        """Move ``moving`` rigidly so ``anchor`` sits at ``pose`` and link it."""
        with self._lock:
            last = getattr(self, "_assist_last", None)
            final = np.asarray(pose, dtype=np.float64)
            if (
                last is None
                or last.get("anchor") != anchor
                or not _pose_close(np.asarray(last["pose"]), final, trans=0.02, rot_deg=0.2)
            ):
                raise ValueError("snap the scan first (it moved since the last snap)")
            links = [dict(l) for l in last["links"]]
            allowed = [l for l in links if not self.is_blocked(l["a"], l["b"])]
            if links and not allowed:
                raise ValueError("this link is blocked: unblock it in the links list first")
            links = allowed
            if not force and not any(l.get("accepted") for l in links):
                raise ValueError("the snap did not verify; use 'accept anyway' if you are sure")
            moving = [m for m in moving if m in self.scans]
            delta = final @ invert(self.scans[anchor].transform)
            undo = {
                "transforms": {m: self.scans[m].transform.tolist() for m in moving},
                "pairs": {},
                "added": [],
            }
            for m in moving:
                self.scans[m].transform = delta @ self.scans[m].transform
            # Automatic links that tie the moved scans to the old (wrong) place
            # would pull them back on the next re-solve: reject them.
            moving_set = set(moving)
            rejected = 0
            for key, p in self.pairs.items():
                if not p.get("accepted") or p.get("manual"):
                    continue
                pa_, pb_ = p.get("a"), p.get("b")
                if (pa_ in moving_set) == (pb_ in moving_set) or pa_ not in self.scans or pb_ not in self.scans:
                    continue
                now = invert(self.scans[pb_].transform) @ self.scans[pa_].transform
                if not _pose_close(now, np.asarray(p["transform"], dtype=np.float64), trans=0.15, rot_deg=2.0):
                    undo["pairs"].setdefault(key, dict(p))
                    p["accepted"] = False
                    p["rejection_reason"] = "contradicts a manual placement"
                    rejected += 1
            for l in links:
                a, b = l["a"], l["b"]
                for key in (f"{a}|{b}", f"{b}|{a}"):
                    if key in self.pairs:
                        undo["pairs"].setdefault(key, dict(self.pairs[key]))
                        self.pairs.pop(key, None)
                l["transform"] = (invert(self.scans[b].transform) @ self.scans[a].transform).tolist()
                verified = bool(l.get("accepted"))
                l["accepted"] = True
                l["manual"] = True
                l["rejection_reason"] = None
                l["params_sig"] = self._params_sig()
                l["manual_created"] = time.time()
                self._store_pair_unlocked(l)
                undo["added"].append(f"{a}|{b}")
                ev = {k: l.get(k) for k in ("score", "wall_score", "wall_area", "see_through", "rejection_reason")}
                ev["mean_mm"] = (l.get("accuracy") or {}).get("mean_mm")
                ev["verified"] = verified
                self._record_manual(a, b, np.asarray(l["transform"]),
                                    kind or ("markers" if markers else ("force" if force else "snap")), ev, markers)
            self._assist_undo = undo
            self._assist_last = None
            self._solve_and_apply_unlocked(lambda *a, **k: None, preserve=True)
        self._notify()
        return {"ok": True, "links": len(links), "rejected": rejected, "summary": self.summary()}

    def place_exact(self, moving: list[str], anchor: str, pose: list[list[float]], partners: list[str],
                    kind: str = "assistant", note: str = "") -> dict[str, Any]:
        """Put ``moving`` rigidly so ``anchor`` sits exactly at ``pose`` and
        record manual links anchor<->partners (no Snap).  For placements whose
        pose was already checked elsewhere (zero contradictions, real overlap).
        Undo-able like a normal manual placement."""
        from .pipeline import Pipeline

        final = np.asarray(pose, dtype=np.float64)
        allowed = [p for p in partners if not self.is_blocked(anchor, p)]
        if partners and not allowed:
            raise ValueError("this link is blocked: unblock it in the links list first")
        partners = allowed
        with self._lock:
            moving = [m for m in moving if m in self.scans]
            if anchor not in moving:
                raise ValueError("anchor must be one of the moving scans")
            delta = final @ invert(self.scans[anchor].transform)
            undo = {"transforms": {m: self.scans[m].transform.tolist() for m in moving}, "pairs": {}, "added": []}
            for m in moving:
                self.scans[m].transform = delta @ self.scans[m].transform
        pipe = Pipeline(self)
        jobs = [pipe.job(anchor, p, init=invert(self.scans[p].transform) @ self.scans[anchor].transform,
                         source=f"manual ({kind})") for p in partners if p in self.scans and p not in moving]
        evid = {}
        pool = process_pool(max(1, min(len(jobs), 4)), initializer=_pair_worker_init)
        try:
            for j, r in zip(jobs, pool.map(evaluate_fixed_job, jobs)):
                evid[j.b] = r
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            for p, r in evid.items():
                for key in (f"{anchor}|{p}", f"{p}|{anchor}"):
                    if key in self.pairs:
                        undo["pairs"].setdefault(key, dict(self.pairs[key]))
                        self.pairs.pop(key, None)
                l = dict(r)
                l.update({"a": anchor, "b": p, "accepted": True, "manual": True, "rejection_reason": None,
                          "transform": (invert(self.scans[p].transform) @ self.scans[anchor].transform).tolist(),
                          "params_sig": self._params_sig(), "manual_created": time.time()})
                self._store_pair_unlocked(l)
                undo["added"].append(f"{anchor}|{p}")
                ev = {k: r.get(k) for k in ("score", "wall_score", "wall_area", "see_through", "plausible", "reason")}
                ev["mean_mm"] = (r.get("accuracy") or {}).get("mean_mm")
                ev["note"] = note
                self._record_manual(anchor, p, np.asarray(l["transform"]), kind, ev)
            self._assist_undo = undo
            self._solve_and_apply_unlocked(lambda *a, **k: None, preserve=True)
        self._notify()
        return {"ok": True, "links": [{"partner": p, "plausible": bool(r.get("plausible")), "reason": r.get("reason"),
                                       "mean_mm": (r.get("accuracy") or {}).get("mean_mm"), "score": r.get("score")}
                                      for p, r in evid.items()]}

    def assist_undo(self) -> dict[str, Any]:
        with self._lock:
            undo = getattr(self, "_assist_undo", None)
            if not undo:
                raise ValueError("nothing to undo")
            for key in undo["added"]:
                self.pairs.pop(key, None)
                a, b = key.split("|", 1)
                if a in self.scans and b in self.scans:
                    self._retract_manual(a, b, "undo")
            for key, p in undo["pairs"].items():
                if not self.is_blocked(p["a"], p["b"]):
                    self.pairs[key] = p
            for sid, T in undo["transforms"].items():
                if sid in self.scans:
                    self.scans[sid].transform = np.asarray(T, dtype=np.float64)
            self._assist_undo = None
            self._solve_and_apply_unlocked(lambda *a, **k: None, preserve=True)
        self._notify()
        return {"ok": True}

    def recover_scan(self, sid: str, from_scan: str | None = None) -> dict[str, Any]:
        """Re-place a scan from its accepted links, without moving the others.

        Use after accidentally resetting one scan's pose.  The link transforms
        are independent measurements, so the scan can be put back exactly.
        """
        s = self.scans.get(sid)
        if s is None:
            raise ValueError("unknown scan")
        candidates = [
            p
            for p in self.pairs.values()
            if p.get("accepted") and sid in (p.get("a"), p.get("b")) and p.get("transform")
        ]
        if from_scan:
            chosen = [
                p
                for p in candidates
                if (p.get("a") == from_scan and p.get("b") == sid)
                or (p.get("b") == from_scan and p.get("a") == sid)
            ]
            if chosen:
                candidates = chosen
        if not candidates:
            raise ValueError("this scan has no accepted links to recover from")
        best = max(candidates, key=lambda p: float(p.get("score") or 0.0))
        a, b = best["a"], best["b"]
        T = np.asarray(best["transform"], dtype=np.float64)
        if sid == a:
            other = b
            P = np.asarray(self.scans[other].transform, dtype=np.float64) @ T
        else:
            other = a
            P = np.asarray(self.scans[other].transform, dtype=np.float64) @ invert(T)
        previous = s.transform.tolist()
        s.transform = P
        self._layout_groups()
        self._notify()
        return {
            "id": sid,
            "transform": P.tolist(),
            "previous": previous,
            "from": other,
            "score": float(best.get("score") or 0.0),
        }

    def measure_accuracy(self) -> str:
        """Compute the surface deviation (mm) for every accepted link."""

        def job(progress: Callable) -> dict[str, Any]:
            self._measure_accuracy_impl(progress, 0.0, 1.0)
            self._notify()
            return self.summary()

        return self._new_job("accuracy", job)

    def pair_accuracy(
        self,
        a: str,
        b: str,
        *,
        paths: dict[str, str] | None = None,
        voxel: float | None = None,
    ) -> dict[str, Any]:
        from .register import pair_transform_from_poses, surface_accuracy

        sa, sb = self.scans.get(a), self.scans.get(b)
        if sa is None or sb is None or not sa.reg_path or not sb.reg_path:
            raise ValueError("scans are not loaded")
        T = pair_transform_from_poses({a: sa.transform, b: sb.transform}, a, b)
        # Use the fine bundles only when both scans have one; never compare a
        # fine cloud against a coarse one.
        fine = paths or {}
        if a in fine and b in fine:
            pa, pb = fine[a], fine[b]
        else:
            pa, pb, voxel = sa.reg_path, sb.reg_path, None
        acc = surface_accuracy(pa, pb, T, float(voxel if voxel is not None else self.params["voxel"]))
        feats = self._match_features(sa, sb, T)
        pair = self.pairs.get(f"{a}|{b}") or self.pairs.get(f"{b}|{a}") or {}
        return {
            "a": a,
            "b": b,
            "transform": T.tolist(),
            "accuracy": acc,
            "feature_matches": feats,
            "score": pair.get("score"),
            "structure_overlap": pair.get("structure_overlap"),
            "accepted": pair.get("accepted", False),
            "group_of": {a: sa.group, b: sb.group},
        }

    def _refresh_pair_residuals(self) -> None:
        from .register import pair_transform_from_poses, score_transform

        for key, p in self.pairs.items():
            a, b = p["a"], p["b"]
            if a not in self.scans or b not in self.scans:
                continue
            sa, sb = self.scans[a], self.scans[b]
            if sa.reg_path and sb.reg_path:
                try:
                    pa = np.load(sa.reg_path)
                    pb = np.load(sb.reg_path)
                    T = pair_transform_from_poses(
                        {a: sa.transform, b: sb.transform}, a, b
                    )
                    s = score_transform(pa, pb, T, float(self.params["voxel"]))
                    p["residual"] = {
                        "overlap": s["overlap"],
                        "rmse": s["rmse"],
                    }
                except Exception:
                    pass

    def set_pair_accepted(self, a: str, b: str, accepted: bool) -> None:
        with self._lock:
            self._set_pair_accepted_unlocked(a, b, accepted)

    def _set_pair_accepted_unlocked(self, a: str, b: str, accepted: bool) -> None:
        """The user's call on a link is a constraint, not a toggle: links are
        re-chosen from stored candidates, so a rejection must stop that pose
        from coming back, and an acceptance pins it like a manual placement."""
        p = self.pairs.get(f"{a}|{b}") or self.pairs.get(f"{b}|{a}")
        if p is None:
            return
        was_manual = bool(p.get("manual"))
        p["accepted"] = bool(accepted)
        if accepted:
            p["manual"] = True
            p["manual_created"] = time.time()
            p.pop("user_rejected", None)
            p["rejection_reason"] = None
            if p.get("transform") and p["a"] in self.scans and p["b"] in self.scans:
                self._record_manual(p["a"], p["b"], np.asarray(p["transform"]), "accepted-link",
                                    {"score": p.get("score"), "method": p.get("method")})
        else:
            if was_manual and p["a"] in self.scans and p["b"] in self.scans:
                self._retract_manual(p["a"], p["b"], "rejected by you")
            p["user_rejected"] = True
            p.pop("manual", None)
            p["rejection_reason"] = "rejected by you"
            if p.get("transform"):
                self.link_vetoes.append({"a": p["a"], "b": p["b"], "transform": p["transform"]})
        if accepted:
            self.link_vetoes = [v for v in self.link_vetoes if {v["a"], v["b"]} != {p["a"], p["b"]}]
        self._notify()

    def set_included(self, sid: str, included: bool) -> None:
        if sid in self.scans:
            self.scans[sid].included = bool(included)
            self._notify()

    def add_cleanup(self, data: dict[str, Any]) -> dict[str, Any]:
        from .cleanup import normalize_region

        region = normalize_region(data)
        with self._lock:
            if region.get("clip"):
                # only one clip box per project: adding a new one replaces it
                self.cleanups = [c for c in self.cleanups if not c.get("clip")]
            self.cleanups.append(region)
        self._notify()
        return region

    def remove_cleanup(self, cid: str) -> None:
        with self._lock:
            self.cleanups = [c for c in self.cleanups if c.get("id") != cid]
        self._notify()

    def update_cleanup(self, cid: str, **changes: Any) -> dict[str, Any]:
        from .cleanup import normalize_region

        with self._lock:
            for i, c in enumerate(self.cleanups):
                if c.get("id") == cid:
                    merged = dict(c)
                    merged.update(changes)
                    merged["id"] = cid
                    region = normalize_region(merged)
                    self.cleanups[i] = region
                    self._notify()
                    return region
        raise ValueError(f"no such cleanup region: {cid}")

    def enabled_cleanups(self) -> list[dict[str, Any]]:
        return [c for c in self.cleanups if c.get("enabled", True)]

    def set_orientation(self, sid: str, orientation: str) -> dict[str, Any]:
        mode = str(orientation or "auto").lower()
        if mode not in ("auto", "level", "free"):
            raise ValueError("orientation must be auto, level or free")
        s = self.scans.get(sid)
        if s is None:
            raise ValueError(f"no such scan: {sid}")
        s.orientation = mode
        s.loaded = False
        s.loading = False
        self._notify()
        return s.to_dict()

    def set_transform(self, sid: str, matrix: list[list[float]]) -> None:
        with self._lock:
            self._set_transform_unlocked(sid, matrix)

    def _set_transform_unlocked(self, sid: str, matrix: list[list[float]]) -> None:
        if sid in self.scans:
            self.scans[sid].transform = np.asarray(matrix, dtype=np.float64)
            self._layout_groups()
        self._notify()

    def set_reference(self, sid: str) -> None:
        if sid in self.scans:
            self.reference = sid
        self._notify()

    def reset_registration(self, *, cleanups: bool = False, vetoes: bool = False, cached: bool = False) -> None:
        """Forget links/poses/groups but keep the loaded scans."""
        if cached:
            from .db import DB

            DB.drop_hyps([self._scan_key(i) for i in self.scans])
        with self._lock:
            if vetoes:
                self.link_vetoes = []
            for s in self.scans.values():
                s.transform = np.eye(4)
                s.group = -1
            self.pairs.clear()
            self.unresolved_pairs = []
            self.groups.clear()
            self.group_of.clear()
            self.baseline = {}
            self.reference = None
            self.last_result = None
            if cleanups:
                self.cleanups.clear()
        self._notify()

    def reset_alignment(self) -> None:
        with self._lock:
            self._reset_alignment_unlocked()

    def _reset_alignment_unlocked(self) -> None:
        for s in self.scans.values():
            s.transform = np.eye(4)
            s.group = -1
        self.groups = []
        self.group_of = {}
        self.baseline = {}
        self._layout_groups()
        self._notify()

    # ---------------------------------------------------------------- display

    def _preview_world_bbox(
        self, members: list[str], *, with_display: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Exact world bbox from each scan's full bounding box (not the preview).

        The preview is a random subsample; thin far-away parts of a scan (a slim
        facade, a corridor) can be missed entirely, which made the auto-fit
        frame far too tight.
        """
        pts = []
        for sid in members:
            s = self.scans.get(sid)
            if s is None or not s.bbox:
                continue
            lo, hi = np.asarray(s.bbox[0]), np.asarray(s.bbox[1])
            corners = np.array(
                [[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
                dtype=np.float64,
            )
            M = (s.display_offset if with_display else np.eye(4)) @ s.transform
            pts.append((corners @ M.T)[:, :3])
        if not pts:
            return None
        allp = np.concatenate(pts, axis=0)
        return allp.min(axis=0), allp.max(axis=0)

    def _layout_groups(self) -> None:
        """Place disconnected groups side by side for the *preview only*.

        The exported transforms are untouched; this only keeps unrelated rooms
        from stacking on top of each other in the viewport.
        """
        for s in self.scans.values():
            s.display_offset = np.eye(4)
        if not self.groups:
            return
        primary = self.groups[0]
        bbox = self._preview_world_bbox(primary["members"], with_display=True)
        if bbox is None:
            return
        lo, hi = bbox
        cursor = float(hi[0] + 3.0)
        center_y = float((lo[1] + hi[1]) / 2.0)
        for g in self.groups[1:]:
            gb = self._preview_world_bbox(g["members"], with_display=False)
            if gb is None:
                continue
            glo, ghi = gb
            off = np.eye(4)
            off[0, 3] = cursor - float(glo[0])
            off[1, 3] = center_y - float((glo[1] + ghi[1]) / 2.0)
            for m in g["members"]:
                if m in self.scans:
                    self.scans[m].display_offset = off
            cursor += float(ghi[0] - glo[0]) + 3.0

    def _recompute_groups_from_pairs(self) -> None:
        ids = [i for i in self.scans if self.scans[i].loaded]
        if not ids:
            return
        solution = solve_global_poses(
            ids,
            list(self.pairs.values()),
            reference=self.reference,
            fallback={i: self.scans[i].transform for i in ids},
        )
        self.groups = solution["groups"]
        self.group_of = solution["group_of"]
        for s in self.scans.values():
            s.group = self.group_of.get(s.sid, -1)
        self._layout_groups()

    def scene_bbox(self) -> dict[str, list[float]] | None:
        bbox = self._preview_world_bbox(list(self.scans.keys()), with_display=True)
        if bbox is None:
            return None
        lo, hi = bbox
        return {"min": lo.tolist(), "max": hi.tolist()}

    # ------------------------------------------------------------------ state

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return self._summary_unlocked()

    def _summary_unlocked(self) -> dict[str, Any]:
        links_with_acc = [
            p["accuracy"]
            for p in self.pairs.values()
            if p.get("accepted") and p.get("accuracy") and p["accuracy"].get("n")
        ]
        accuracy = None
        if links_with_acc:
            n = sum(int(a["n"]) for a in links_with_acc)
            accuracy = {
                "mean_mm": float(sum(a["mean_mm"] * a["n"] for a in links_with_acc) / max(n, 1)),
                "p95_worst_mm": float(max(a["p95_mm"] for a in links_with_acc)),
                "links": len(links_with_acc),
                "points": int(n),
            }
        return {
            "scans": len(self.scans),
            "loaded": sum(1 for s in self.scans.values() if s.loaded),
            "pairs": len(self.pairs),
            "accepted": sum(1 for p in self.pairs.values() if p.get("accepted")),
            "groups": [
                {
                    "id": g["id"],
                    "members": g["members"],
                    "reference": g["reference"],
                    "unconnected": g.get("unconnected", False),
                    "edges": g.get("edges", 0),
                    "outliers": len(g.get("outliers", [])) if isinstance(g.get("outliers"), (list, tuple)) else int(g.get("outliers") or 0),
                }
                for g in self.groups
            ],
            "reference": self.reference,
            "accuracy": accuracy,
            "bbox": self.scene_bbox(),
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state_unlocked()

    def _state_unlocked(self) -> dict[str, Any]:
        # Keep a stable score order: toggling a link must not make its row jump
        # to the bottom (that made unchecked links hard to find again).
        pairs = sorted(
            self.pairs.values(),
            key=lambda p: -float(p.get("score") or 0.0),
        )
        return {
            "scans": [s.to_dict() for s in self.scans.values()],
            "pairs": pairs,
            "blocked_links": self.blocked_links(),
            "groups": self._summary_unlocked()["groups"],
            "accuracy": self._summary_unlocked().get("accuracy"),
            "cache": DB.stats(),
            "params": self.params,
            "reference": self.reference,
            "baseline": self.baseline,
            "cleanups": list(self.cleanups),
            "bbox": self.scene_bbox(),
            "busy": self._busy_job,
            "busy_kind": (self.jobs.get(self._busy_job) or {}).get("kind") if self._busy_job else None,
            "manual": self.manual_overview(),
            "auto_links": bool(self.params.get("auto_links", True)),
            "locked": bool(self.params.get("locked", False)),
        }

    # ------------------------------------------------------------------ export

    def _write_manifest(self, path: str, data: dict[str, Any]) -> str:
        from pathlib import Path

        target = Path(str(path) + ".json")
        payload = {
            "created": time.time(),
            "path": str(path),
            "params": dict(self.params),
            "reference": self.reference,
            "groups": self.summary().get("groups"),
            "accuracy": self.summary().get("accuracy"),
            "cleanups": self.enabled_cleanups(),
            **data,
        }
        target.write_text(json.dumps(payload, indent=1))
        return str(target)

    def export(self, opts: dict[str, Any]) -> str:
        def job(progress: Callable) -> dict[str, Any]:
            fmt = str(opts.get("format", "e57")).lower()
            path = str(opts.get("path") or "merged.e57")
            mode = str(opts.get("mode", "merged"))
            scope = str(opts.get("scope", "active"))
            explicit_ids = opts.get("ids")
            voxel = float(opts.get("voxel") or 0.0)
            precision = str(opts.get("precision", "double"))
            include_colors = bool(opts.get("colors", True))
            include_intensity = bool(opts.get("intensity", True))
            density_cap = int(opts.get("density_cap") or 0)
            density_voxel = float(opts.get("density_voxel") or 0.02)
            filter_noise = bool(opts.get("filter_noise", True))

            ids = [i for i in (explicit_ids or []) if i in self.scans] or self._export_ids(scope)
            if not ids:
                raise ValueError("no scans selected for export")
            if fmt != "e57":
                mode = "merged"
            # formats that only store one vertex list
            if fmt in {"ply", "pts", "xyz"}:
                mode = "merged"

            total_points = sum(self.scans[i].ref.point_count for i in ids)
            done_points = [0]

            def make_chunks(sid: str) -> Callable[[], Iterator[dict[str, np.ndarray]]]:
                s = self.scans[sid]

                # ``display_offset`` is a viewport-only parking transform for
                # disconnected groups.  Never bake it into survey output: that
                # would shift whole rooms by metres.  Opt in explicitly if a
                # presentation file should match the side-by-side view.
                export_transform = np.asarray(s.transform)
                if opts.get("spread_groups"):
                    export_transform = np.asarray(s.display_offset) @ export_transform

                def gen() -> Iterator[dict[str, np.ndarray]]:
                    if self._cancelled(self._busy_job or ""):
                        raise RuntimeError("cancelled by user")
                    noise_ref = (
                        s.reg_path
                        if filter_noise
                        and str(self.params.get("cleanup", "light")) not in ("off", "none", "false", "0", "")
                        else None
                    )
                    for ch in e57io.scan_chunks(
                        s.ref,
                        export_transform,
                        apply_pose=s.apply_pose,
                        chunk=1_000_000,
                        voxel=voxel if voxel > 0 and not density_cap else None,
                        density_cap=density_cap,
                        density_voxel=density_voxel,
                        noise_ref=noise_ref,
                        noise_radius=max(2.5 * float(self.params["voxel"]), 0.15),
                        want_colors=include_colors,
                        want_intensity=include_intensity,
                        cleanup=self.enabled_cleanups(),
                        cleanup_scan=sid,
                        cancel=lambda: self._cancelled(self._busy_job or ""),
                    ):
                        done_points[0] += len(ch["points"])
                        progress(
                            min(0.99, done_points[0] / max(1, total_points)),
                            f"writing {s.ref.label}",
                        )
                        yield ch

                return gen

            def has_colors(sid: str) -> bool:
                s = self.scans[sid]
                return include_colors and bool(s.colors is not None and s.colors.size)

            def has_intensity(sid: str) -> bool:
                s = self.scans[sid]
                return include_intensity and bool(s.intensity is not None and s.intensity.size)

            if mode == "separate" and fmt == "e57":
                streams = [
                    StreamSpec(
                        name=self.scans[sid].ref.label,
                        chunks=make_chunks(sid),
                        has_colors=has_colors(sid),
                        has_intensity=has_intensity(sid),
                    )
                    for sid in ids
                ]
            else:
                streams = [
                    StreamSpec(
                        name=str(opts.get("name") or "merged"),
                        chunks=lambda: itertools.chain.from_iterable(make_chunks(sid)() for sid in ids),
                        has_colors=any(has_colors(sid) for sid in ids),
                        has_intensity=any(has_intensity(sid) for sid in ids),
                    )
                ]

            writer = e57io.STREAM_WRITERS.get(fmt)
            if writer is None:
                raise ValueError(f"unsupported format: {fmt}")
            if fmt == "e57":
                writer(path, streams, precision=precision)
            else:
                writer(path, streams)
            progress(1.0, "export complete")
            size = os.path.getsize(path) if os.path.exists(path) else 0
            manifest = self._write_manifest(
                path,
                {
                    "kind": "points",
                    "format": fmt,
                    "mode": mode,
                    "scans": ids,
                    "bytes": size,
                    "filter_noise": filter_noise,
                    "colors": include_colors,
                    "intensity": include_intensity,
                },
            )
            return {
                "path": path,
                "format": fmt,
                "mode": mode,
                "scans": ids,
                "bytes": size,
                "manifest": manifest,
            }

        return self._new_job("export", job)

    def export_mesh(self, opts: dict[str, Any]) -> str:
        """Triangulate the registered cloud and write a mesh (PLY/OBJ/GLB)."""

        def job(progress: Callable) -> dict[str, Any]:
            path = str(opts.get("path") or "registered_mesh.ply")
            scope = str(opts.get("scope", "active"))
            ids = [i for i in (opts.get("ids") or []) if i in self.scans] or self._export_ids(scope)
            ids = [i for i in ids if self.scans[i].loaded]
            if not ids:
                raise ValueError("no loaded scans selected for mesh export")
            voxel = float(opts.get("voxel") or 0.02)
            method = str(opts.get("method") or "poisson")
            depth = int(opts.get("depth") or 9)
            include_colors = bool(opts.get("colors", True))
            include_intensity = bool(opts.get("intensity", True))

            parts: list[np.ndarray] = []
            cols: list[np.ndarray] = []
            inten: list[np.ndarray] = []
            for k, sid in enumerate(ids):
                if self._cancelled(self._busy_job or ""):
                    raise RuntimeError("cancelled by user")
                s = self.scans[sid]
                if not s.reg_path:
                    continue
                preview = Path(s.reg_path).parent / "preview.npz"
                if not preview.exists():
                    continue
                with np.load(preview, allow_pickle=False) as z:
                    pts = np.asarray(z["points"], dtype=np.float64)
                    c = np.asarray(z["colors"], dtype=np.uint8) if "colors" in z.files else None
                    it = np.asarray(z["intensity"], dtype=np.float32) if "intensity" in z.files else None
                T = np.asarray(s.transform)
                if opts.get("spread_groups"):
                    T = np.asarray(s.display_offset) @ T
                pts = pts @ T[:3, :3].T + T[:3, 3]
                from .cleanup import removal_mask

                hidden = removal_mask(pts, self.enabled_cleanups(), scan=sid)
                if hidden.any():
                    keep = ~hidden
                    pts = pts[keep]
                    if c is not None and len(c) == len(hidden):
                        c = c[keep]
                    if it is not None and len(it) == len(hidden):
                        it = it[keep]
                parts.append(pts.astype(np.float32))
                if c is not None and len(c) == len(pts):
                    cols.append(c)
                if it is not None and len(it) == len(pts):
                    inten.append(it)
                progress(0.01 + 0.09 * (k + 1) / len(ids), f"mesh: collecting {s.ref.label}")

            if not parts:
                raise ValueError("no preview points available for meshing")
            points = np.concatenate(parts, axis=0)
            colors = np.concatenate(cols, axis=0) if len(cols) == len(parts) else None
            intensity = np.concatenate(inten, axis=0) if len(inten) == len(parts) else None
            # previews are random subsamples; thin them to the requested mesh voxel
            if voxel > 0:
                points, keep = e57io._voxel_subsample(points, voxel)
                if colors is not None:
                    colors = colors[keep]
                if intensity is not None:
                    intensity = intensity[keep]
            if len(points) > 900_000:
                sel = np.random.default_rng(22).choice(len(points), 900_000, replace=False)
                points = points[sel]
                if colors is not None:
                    colors = colors[sel]
                if intensity is not None:
                    intensity = intensity[sel]
            progress(0.12, f"mesh: {len(points):,} points")
            result = mesh_mod.reconstruct_safely(
                points,
                colors=colors if include_colors else None,
                intensity=intensity if include_intensity else None,
                voxel=voxel,
                method=method,
                depth=depth,
                progress=lambda p, msg=None: progress(0.12 + 0.78 * float(p), msg),
                should_cancel=lambda: self._cancelled(self._busy_job or ""),
            )
            progress(0.92, "mesh: writing file")
            suffix = Path(path).suffix
            tmp_path = f"{path}.part{suffix}"
            mesh_mod.write_mesh(
                tmp_path,
                result,
                intensity_ramp=bool(opts.get("intensity_ramp")),
            )
            os.replace(tmp_path, path)
            size = os.path.getsize(path) if os.path.exists(path) else 0
            manifest = self._write_manifest(
                path,
                {
                    "kind": "mesh",
                    "format": Path(path).suffix.lower().lstrip("."),
                    "bytes": size,
                    "vertices": int(len(result["vertices"])),
                    "triangles": int(len(result["faces"])),
                    "method": result["method"],
                    "depth": result["depth"],
                    "scans": ids,
                    "colors": include_colors,
                    "intensity": include_intensity,
                },
            )
            progress(1.0, "mesh complete")
            return {
                "path": path,
                "format": Path(path).suffix.lower().lstrip("."),
                "bytes": size,
                "vertices": int(len(result["vertices"])),
                "triangles": int(len(result["faces"])),
                "method": result["method"],
                "depth": result["depth"],
                "scans": ids,
                "manifest": manifest,
            }

        return self._new_job("mesh", job)

    def session_data(self) -> dict[str, Any]:
        """Compact, portable description of the current alignment."""
        return {
            "version": 1,
            "params": dict(self.params),
            "reference": self.reference,
            "baseline": dict(self.baseline or {}),
            "cleanups": list(self.cleanups),
            "link_vetoes": list(self.link_vetoes),
            "unresolved_pairs": [list(u) for u in self.unresolved_pairs],
            "scans": [
                {
                    "id": s.sid,
                    "file": s.ref.file,
                    "index": s.ref.index,
                    "label": s.ref.label,
                    "transform": s.transform.tolist(),
                    "group": s.group,
                    "included": bool(s.included),
                    "file_hash": s.file_hash,
                    "orientation": s.orientation,
                }
                for s in self.scans.values()
            ],
            "pairs": [
                {
                    "a": p["a"],
                    "b": p["b"],
                    "score": p.get("score"),
                    "structure_overlap": p.get("structure_overlap"),
                    "overlap": p.get("overlap"),
                    "rmse": p.get("rmse"),
                    "method": p.get("method"),
                    "accepted": p.get("accepted", False),
                    "transform": p.get("transform"),
                    "params_sig": p.get("params_sig"),
                    "synthetic": bool(p.get("synthetic")),
                    "manual": bool(p.get("manual")),
                    "user_rejected": bool(p.get("user_rejected")),
                    "rejection_reason": p.get("rejection_reason"),
                    "wall_score": p.get("wall_score"),
                    "wall_area": p.get("wall_area"),
                    "accuracy": p.get("accuracy"),
                    "cycle": p.get("cycle"),
                    "cycle_warning": bool(p.get("cycle_warning")),
                    "alternatives": p.get("alternatives"),
                    "outlier": bool(p.get("outlier")),
                    "residual": p.get("residual"),
                }
                for p in self.pairs.values()
            ],
            "groups": self.summary()["groups"],
        }

    def _export_ids(self, scope: str) -> list[str]:
        loaded = [i for i in self.scans if self.scans[i].loaded]
        if scope == "all":
            return loaded
        if scope == "selected":
            return [i for i in loaded if self.scans[i].included]
        if self.groups:
            return [i for i in self.groups[0]["members"] if i in self.scans]
        return loaded


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _features_from_summary(summary: dict[str, Any]) -> ScanFeatures:
    from .features import EdgeCluster, PlaneFeature

    feats = ScanFeatures()
    for p in summary.get("planes", []):
        extent = np.asarray(p["extent"], dtype=np.float32)
        area = float(p.get("area") if p.get("area") is not None else np.prod(np.sort(extent)[-2:]))
        significant = p.get("significant")
        if significant is None:
            significant = int(p["inliers"]) >= 400 and area >= 0.8
        feats.planes.append(
            PlaneFeature(
                normal=np.asarray(p["normal"], dtype=np.float32),
                offset=float(p["offset"]),
                centroid=np.asarray(p["centroid"], dtype=np.float32),
                inliers=int(p["inliers"]),
                extent=extent,
                points=np.empty((0, 3), np.float32),
                area=area,
                significant=bool(significant),
            )
        )
    for e in summary.get("edges", []):
        extent = np.asarray(e["extent"], dtype=np.float32)
        significant = e.get("significant")
        if significant is None:
            significant = (
                int(e["size"]) >= 60
                and float(e["length"]) >= 0.35
                and float(extent.max()) >= 0.40
            )
        feats.edges.append(
            EdgeCluster(
                centroid=np.asarray(e["centroid"], dtype=np.float32),
                points=np.empty((0, 3), np.float32),
                size=int(e["size"]),
                extent=extent,
                direction=np.asarray(e["direction"], dtype=np.float32),
                length=float(e["length"]),
                significant=bool(significant),
            )
        )
    return feats


# global session for the local single-user server
SESSION = Session()
