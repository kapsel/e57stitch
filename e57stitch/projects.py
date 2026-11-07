"""Projects: isolated registration jobs with their own scans, links and groups.

Each project is a separate :class:`~e57stitch.session.Session` persisted as
``~/.cache/e57stitch/projects/<id>/project.json``.  Scans from one project are
never matched against scans from another.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .e57io import CACHE_DIR, ScanRef, probe_file
from .session import PALETTE, ScanState, Session, is_unused_candidate

PROJECTS_DIR = CACHE_DIR / "projects"


@dataclass
class Project:
    id: str
    name: str
    created: float
    session: Session | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def summary(self, active: bool = False) -> dict[str, Any]:
        s = self.session
        if s is not None:
            return {
                "id": self.id,
                "name": self.name,
                "created": self.created,
                "scans": len(s.scans),
                "loaded": sum(1 for x in s.scans.values() if x.loaded),
                "pairs": len(s.pairs),
                "accepted": sum(1 for p in s.pairs.values() if p.get("accepted")),
                "groups": len(s.groups),
                "busy": s._busy_job is not None,
                "active": active,
            }
        meta = self.meta or {}
        return {
            "id": self.id,
            "name": self.name,
            "created": self.created,
            "scans": len(meta.get("scans", [])),
            "loaded": 0,
            "pairs": len(meta.get("pairs", [])),
            "accepted": sum(1 for p in meta.get("pairs", []) if p.get("accepted")),
            "groups": len(meta.get("groups", [])),
            "busy": False,
            "active": active,
        }


class ProjectManager:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else PROJECTS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.projects: dict[str, Project] = {}
        self.active: str | None = None
        self._lock = threading.RLock()
        self._load_index()
        if not self.projects:
            self.create("My scans", activate=False)
        if self.active not in self.projects:
            self.active = next(iter(self.projects))
        self.activate(self.active, persist=False)

    # ------------------------------------------------------------- persistence

    def _index_path(self) -> Path:
        return self.root / "index.json"

    def _load_index(self) -> None:
        active = None
        idx = self._index_path()
        if idx.exists():
            try:
                active = json.loads(idx.read_text()).get("active")
            except Exception:
                active = None
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            pj = folder / "project.json"
            if not pj.exists():
                continue
            try:
                meta = json.loads(pj.read_text())
            except Exception:
                continue
            pid = str(meta.get("id") or folder.name)
            self.projects[pid] = Project(
                id=pid,
                name=str(meta.get("name") or "Untitled"),
                created=float(meta.get("created") or folder.stat().st_mtime),
                session=None,
                meta=meta,
                path=pj,
            )
        self.active = active

    def _save_index(self) -> None:
        tmp = self._index_path().with_suffix(".tmp")
        tmp.write_text(json.dumps({"active": self.active}, indent=1))
        os.replace(tmp, self._index_path())

    def save(self, pid: str | None = None) -> None:
        pid = pid or self.active
        if pid is None:
            return
        with self._lock:
            project = self.projects.get(pid)
            if project is None or project.session is None:
                return
            data = dict(project.meta or {})
            data.update(project.session.session_data())
            data.update({"id": project.id, "name": project.name, "created": project.created, "version": 2})
            project.path = project.path or (self.root / project.id / "project.json")
            project.path.parent.mkdir(parents=True, exist_ok=True)
            # Keep the previous good state.  A registration reset clears the
            # in-memory links; without this a save/crash could lose an
            # alignment the user spent hours on.
            try:
                if project.path.exists():
                    shutil.copy2(project.path, project.path.with_suffix(".bak"))
            except Exception:
                pass
            tmp = project.path.with_name(f"{project.path.name}.{uuid.uuid4().hex[:8]}.tmp")
            tmp.write_text(json.dumps(data, indent=1))
            os.replace(tmp, project.path)
            project.meta = data
            self._save_index()

    def snapshot(self, pid: str | None = None, label: str = "manual", keep: int = 60) -> str:
        """Write a timestamped, immutable copy of the current alignment.

        Unlike ``save()`` the snapshot is never overwritten by later rounds, so
        an accidental reset or a bad overnight run can always be rolled back.
        """
        pid = pid or self.active
        if pid is None:
            raise KeyError("no active project")
        with self._lock:
            project = self.projects.get(pid)
            if project is None or project.session is None:
                raise KeyError(pid)
            data = dict(project.meta or {})
            with project.session._lock:
                data.update(project.session.session_data())
            data.update(
                {"id": project.id, "name": project.name, "created": project.created, "version": 2}
            )
            folder = (project.path or (self.root / project.id / "project.json")).parent
            snap_dir = folder / "snapshots"
            snap_dir.mkdir(parents=True, exist_ok=True)
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label)[:40] or "manual"
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = snap_dir / f"{stamp}-{safe}.json"
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=1))
            os.replace(tmp, path)
            files = sorted(snap_dir.glob("*.json"))
            for old in files[:-keep]:
                try:
                    old.unlink()
                except OSError:
                    pass
            return str(path)

    def save_all(self) -> None:
        for pid in list(self.projects):
            self.save(pid)

    # -------------------------------------------------------------- lifecycle

    def _build_session(self, meta: dict[str, Any]) -> Session:
        session = Session()
        session.params.update(meta.get("params") or {})
        session.link_vetoes = list(meta.get("link_vetoes") or [])
        session.project_name = str(meta.get("name") or "")
        for i, entry in enumerate(meta.get("scans") or []):
            file = str(entry.get("file") or "")
            index = int(entry.get("index") or 0)
            if not file:
                continue
            sid = str(entry.get("id") or f"s{i}")
            try:
                refs = probe_file(file)
                ref = refs[index]
            except Exception:
                # The file may be on an unmounted drive.  Keep the scan and its
                # links in the project instead of silently deleting both on the
                # next save; it will simply show as unloaded.
                ref = ScanRef(
                    id=sid,
                    file=file,
                    index=index,
                    name=Path(file).stem,
                    point_count=0,
                    fields=[],
                    pose=np.eye(4),
                    has_pose=False,
                    bounds=None,
                )
            ref.id = sid
            session.scans[sid] = ScanState(
                sid=sid,
                ref=ref,
                color=PALETTE[i % len(PALETTE)],
                apply_pose=bool(session.params.get("use_pose", True)),
                included=bool(entry.get("included", True)),
                file_hash=str(entry.get("file_hash") or ""),
                orientation=str(entry.get("orientation") or "auto"),
            )
            session.scans[sid].transform = np.asarray(entry.get("transform", np.eye(4)), dtype=np.float64)
            try:
                session._counter = max(session._counter, int(sid[1:]) + 1)
            except Exception:
                session._counter += 1
        sig = session._params_sig()
        session.unresolved_pairs = [[str(u[0]), str(u[1])] for u in meta.get("unresolved_pairs") or []]
        for p in meta.get("pairs") or []:
            if not p.get("transform"):
                continue
            a, b = str(p.get("a")), str(p.get("b"))
            if a not in session.scans or b not in session.scans:
                continue
            if is_unused_candidate(p):  # saved before these stopped being listed
                session.unresolved_pairs.append([a, b])
                continue
            q = dict(p)
            q["a"], q["b"] = a, b
            q.setdefault("accepted", True)
            # older projects predate the signature: they were saved together
            # with the same params, so treat them as valid for this signature
            q.setdefault("params_sig", sig)
            session.pairs[f"{a}|{b}"] = q
        session.drop_blocked_rows()  # a restored save may predate the block
        try:
            from .cleanup import normalize_region

            session.cleanups = [normalize_region(c) for c in (meta.get("cleanups") or [])]
        except Exception:
            session.cleanups = []
        session.reference = meta.get("reference")
        session.baseline = meta.get("baseline") or {}
        session.groups = []
        for g in meta.get("groups") or []:
            members = [str(m) for m in g.get("members") or [] if str(m) in session.scans]
            if not members:
                continue
            outliers = g.get("outliers", [])
            session.groups.append(
                {
                    **g,
                    "id": len(session.groups),
                    "members": members,
                    "outliers": outliers if isinstance(outliers, list) else [],
                }
            )
            for m in members:
                session.scans[m].group = len(session.groups) - 1
        session.groups.sort(key=lambda g: len(g.get("members") or []), reverse=True)
        for gi, g in enumerate(session.groups):
            g["id"] = gi
            for m in g["members"]:
                session.scans[m].group = gi
        session.group_of = {m: g["id"] for g in session.groups for m in g["members"]}
        session._layout_groups()
        session.on_change = lambda pid=self.projects_key(meta): self.save(pid)
        return session

    @staticmethod
    def projects_key(meta: dict[str, Any]) -> str:
        return str(meta.get("id") or "")

    def ingest(
        self,
        pid: str,
        paths: list[str],
        *,
        force: bool = False,
        recursive: bool = True,
        auto: bool = False,
    ) -> dict[str, Any]:
        """Add E57 files to a project, with SHA-256 deduplication."""
        from . import ingest as ingest_mod

        project = self.projects.get(pid)
        if project is None:
            raise KeyError(pid)
        session = self.activate(pid, persist=False)
        files = ingest_mod.expand_paths(paths, recursive=recursive)
        added: list[str] = []
        skipped: list[dict[str, Any]] = []
        new_ids: list[str] = []
        for path in files:
            try:
                sha = ingest_mod.sha256_file(path)
            except Exception as exc:
                skipped.append({"path": path, "reason": f"cannot hash: {exc}"})
                continue
            entry = ingest_mod.lookup(sha)
            if entry is not None:
                projects = entry.get("projects") or {}
                if pid in projects and not force:
                    # The registry can outlive a "clear" or a removed scan.
                    # Only trust it if the file is actually in the session.
                    resolved_path = str(Path(path).resolve())
                    present = any(
                        scan.file_hash == sha
                        or str(Path(scan.ref.file).resolve()) == resolved_path
                        for scan in session.scans.values()
                    )
                    if present:
                        skipped.append(
                            {"path": path, "sha256": sha, "reason": "already in this project"}
                        )
                        continue
                    ingest_mod.unregister(sha, pid)
                if projects and pid not in projects and not force:
                    other_id = next(iter(projects))
                    other = self.projects.get(other_id)
                    skipped.append(
                        {
                            "path": path,
                            "sha256": sha,
                            "reason": f"already ingested in project “{other.name if other else other_id}”",
                            "project": other_id,
                        }
                    )
                    continue
            before = set(session.scans)
            try:
                session.add_files([path])
            except Exception as exc:
                skipped.append({"path": path, "sha256": sha, "reason": f"cannot read: {exc}"})
                continue
            fresh = set(session.scans) - before
            resolved = str(Path(path).resolve())
            for sid in fresh:
                if str(Path(session.scans[sid].ref.file).resolve()) == resolved:
                    session.scans[sid].file_hash = sha
                    new_ids.append(sid)
            if not fresh:
                skipped.append(
                    {"path": path, "sha256": sha, "reason": "already in this project"}
                )
                continue
            ingest_mod.register(sha, path, pid, os.path.getsize(path))
            try:
                from .db import DB

                st = os.stat(path)
                DB.put_file(sha, path, st.st_size, st.st_mtime_ns, len(fresh))
            except Exception:
                pass
            added.append(path)
        self.save(pid)
        return {
            "added": added,
            "skipped": skipped,
            "files": len(files),
            "new_ids": new_ids,
            "auto": auto,
            "scans": len(session.scans),
        }

    def activate(self, pid: str, *, persist: bool = True) -> Session:
        with self._lock:
            project = self.projects.get(pid)
            if project is None:
                raise KeyError(pid)
            if project.session is None:
                project.session = self._build_session(project.meta)
                project.session.on_change = lambda pid=pid: self.save(pid)
            self.active = pid
            if persist:
                self._save_index()
            self._seed_hashes(project)
            return project.session

    def _seed_hashes(self, project: Project) -> None:
        """Register existing project files in the global SHA-256 registry."""
        from . import ingest as ingest_mod

        if project.session is None:
            return
        changed = False
        seen: dict[str, str] = {}
        for scan in project.session.scans.values():
            path = str(Path(scan.ref.file).resolve())
            sha = scan.file_hash
            if not sha:
                if path not in seen:
                    try:
                        seen[path] = ingest_mod.sha256_file(path)
                    except Exception:
                        continue
                sha = seen[path]
                scan.file_hash = sha
                changed = True
            try:
                st = os.stat(path)
                ingest_mod.register(sha, path, project.id, st.st_size)
                from .db import DB

                DB.put_file(sha, path, st.st_size, st.st_mtime_ns, 0)
            except Exception:
                pass
        try:
            from .db import DB

            sig = project.session._params_sig()
            for p in project.session.pairs.values():
                a, b = p.get("a"), p.get("b")
                if p.get("manual") or p.get("synthetic"):
                    continue  # not matcher results; never seed the link cache
                if a in project.session.scans and b in project.session.scans and p.get("transform"):
                    DB.put_link(
                        project.session._scan_key(a),
                        project.session._scan_key(b),
                        sig,
                        p,
                        overwrite=False,
                    )
        except Exception:
            pass
        if changed:
            self.save(project.id)

    def current(self) -> Session:
        if self.active is None:
            raise RuntimeError("no active project")
        project = self.projects[self.active]
        if project.session is None:
            return self.activate(self.active, persist=False)
        return project.session

    def active_project(self) -> Project:
        return self.projects[self.active]

    def create(self, name: str = "Untitled project", *, activate: bool = True) -> Project:
        with self._lock:
            pid = uuid.uuid4().hex[:10]
            project = Project(id=pid, name=name.strip() or "Untitled project", created=time.time())
            project.session = Session()
            project.session.on_change = lambda pid=pid: self.save(pid)
            project.path = self.root / pid / "project.json"
            self.projects[pid] = project
            self.save(pid)
            if activate:
                self.active = pid
                self._save_index()
            return project

    def restore_backup(self, pid: str) -> dict[str, Any]:
        project = self.projects.get(pid)
        if project is None or project.path is None:
            raise KeyError(pid)
        bak = project.path.with_suffix(".bak")
        if not bak.exists():
            raise FileNotFoundError(str(bak))
        data = json.loads(bak.read_text())
        project.session = self._build_session(data)
        project.session.on_change = lambda pid=pid: self.save(pid)
        project.meta = data
        self.activate(pid, persist=False)
        self.save(pid)
        return self.state()

    def rename(self, pid: str, name: str) -> None:
        with self._lock:
            project = self.projects.get(pid)
            if project is None:
                return
            project.name = name.strip() or project.name
            self.save(pid)

    def delete(self, pid: str) -> None:
        with self._lock:
            project = self.projects.pop(pid, None)
            if project is None:
                return
            if project.session is not None and project.session._busy_job is not None:
                self.projects[pid] = project
                raise RuntimeError("project has a running job")
            if project.path is not None:
                # The folder holds subfolders (inbox, snapshots). If project.json
                # survived, the project would come back at the next restart, so
                # a failure here must not look like success.
                try:
                    shutil.rmtree(project.path.parent)
                except OSError as exc:
                    self.projects[pid] = project
                    raise RuntimeError(f"could not delete project files: {exc}") from exc
            try:
                from . import ingest as ingest_mod

                ingest_mod.release_project(pid)
            except Exception:
                pass
            if self.active == pid:
                if self.projects:
                    self.active = next(iter(self.projects))
                    self.activate(self.active, persist=False)
                else:
                    self.active = None
                    self.create("My scans")
            self._save_index()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            active = self.active
            return [p.summary(active=p.id == active) for p in self.projects.values()]

    def job(self, job_id: str) -> dict[str, Any] | None:
        for project in self.projects.values():
            if project.session is not None:
                job = project.session.job(job_id)
                if job is not None:
                    return job
        return None

    def state(self) -> dict[str, Any]:
        session = self.current()
        state = session.state()
        project = self.active_project()
        state["project"] = {
            "id": project.id,
            "name": project.name,
            "created": project.created,
        }
        state["projects"] = self.list()
        return state

    def state_for(self, pid: str) -> dict[str, Any]:
        self.activate(pid)
        return self.state()


    def ensure_session(self, pid: str) -> Session:
        with self._lock:
            project = self.projects[pid]
            if project.session is None:
                project.session = self._build_session(project.meta)
                project.session.on_change = lambda pid=pid: self.save(pid)
                self._seed_hashes(project)
            return project.session

    def reveal(self, path: str) -> None:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(str(p))
        if p.is_dir():
            opener = ["open"] if sys.platform == "darwin" else ["xdg-open"]
            subprocess.run(opener + [str(p)], check=False)
            return
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", str(p)], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(p.parent)], check=False)
        else:
            subprocess.run(["explorer", "/select,", str(p)], check=False)


MANAGER = ProjectManager()
