"""File hashing + a global ingest registry.

The same physical scan file is never ingested twice into the same project, and
by default not into another project either (a copy with a different path is
still caught by its SHA-256).  ``force`` overrides the cross-project check.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .e57io import CACHE_DIR

HASH_CACHE_PATH = CACHE_DIR / "hashes.json"
REGISTRY_PATH = CACHE_DIR / "files.json"
_LOCK = threading.RLock()


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def sha256_file(path: str, *, chunk: int = 4 * 1024 * 1024) -> str:
    """SHA-256 of a file, cached by (path, size, mtime)."""
    resolved = str(Path(path).resolve())
    st = os.stat(resolved)
    with _LOCK:
        cache = _read_json(HASH_CACHE_PATH, {})
    entry = cache.get(resolved)
    if entry and entry.get("size") == st.st_size and entry.get("mtime") == st.st_mtime_ns:
        return str(entry["sha256"])
    digest = hashlib.sha256()
    with open(resolved, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    sha = digest.hexdigest()
    with _LOCK:
        cache = _read_json(HASH_CACHE_PATH, {})
        cache[resolved] = {"size": st.st_size, "mtime": st.st_mtime_ns, "sha256": sha}
        _write_json(HASH_CACHE_PATH, cache)
    return sha


def lookup(sha: str) -> dict[str, Any] | None:
    with _LOCK:
        return _read_json(REGISTRY_PATH, {}).get(sha)


def register(sha: str, path: str, project_id: str, size: int) -> dict[str, Any]:
    with _LOCK:
        reg = _read_json(REGISTRY_PATH, {})
        entry = reg.get(sha) or {
            "sha256": sha,
            "path": path,
            "size": size,
            "first_seen": int(time.time()),
            "projects": {},
        }
        entry.setdefault("projects", {})[project_id] = int(time.time())
        reg[sha] = entry
        _write_json(REGISTRY_PATH, reg)
        return entry


def unregister(sha: str, project_id: str) -> None:
    """Remove one file/project pair after the scan is deleted from a project."""
    with _LOCK:
        reg = _read_json(REGISTRY_PATH, {})
        entry = reg.get(sha)
        if entry is None:
            return
        projects = entry.get("projects") or {}
        if project_id in projects:
            projects.pop(project_id, None)
            entry["projects"] = projects
            _write_json(REGISTRY_PATH, reg)


def release_project(project_id: str) -> None:
    """Forget a deleted project from the registry (files stay registered)."""
    with _LOCK:
        reg = _read_json(REGISTRY_PATH, {})
        changed = False
        for entry in reg.values():
            projects = entry.get("projects") or {}
            if project_id in projects:
                projects.pop(project_id, None)
                changed = True
        if changed:
            _write_json(REGISTRY_PATH, reg)


def expand_paths(paths: list[str], *, recursive: bool = True) -> list[str]:
    out: list[str] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            for child in sorted(it):
                if child.is_file() and child.suffix.lower() == ".e57":
                    out.append(str(child.resolve()))
        elif p.is_file() and p.suffix.lower() == ".e57":
            out.append(str(p.resolve()))
    # stable, unique
    seen = set()
    unique = []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique
