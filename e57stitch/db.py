"""SQLite cache for expensive registration artefacts.

Everything is keyed by *content* (file SHA-256 + scan index) and a settings
signature, never by project.  Switching a project or changing a parameter back
to a previous value is therefore instant: the scan bundle (registration cloud,
preview, features, noise stats) and the pairwise link results are reused.

Large payloads stay on disk (npy/npz); SQLite only stores the index/metadata.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .e57io import CACHE_DIR

DB_PATH = Path(os.environ.get("E57STITCH_DB", CACHE_DIR / "e57stitch.sqlite"))


class CacheDB:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create()

    def _create(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS files(
                    sha TEXT PRIMARY KEY, path TEXT, size INTEGER, mtime INTEGER,
                    scan_count INTEGER, updated REAL
                );
                CREATE TABLE IF NOT EXISTS artifacts(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_key TEXT NOT NULL, kind TEXT NOT NULL,
                    params TEXT NOT NULL, path TEXT NOT NULL, meta TEXT,
                    updated REAL,
                    UNIQUE(scan_key, kind, params)
                );
                CREATE TABLE IF NOT EXISTS links(
                    a_key TEXT NOT NULL, b_key TEXT NOT NULL, params TEXT NOT NULL,
                    result TEXT NOT NULL, updated REAL,
                    PRIMARY KEY(a_key, b_key, params)
                );
                CREATE TABLE IF NOT EXISTS noise(
                    scan_key TEXT NOT NULL, params TEXT NOT NULL,
                    cleaned INTEGER, pct REAL, updated REAL,
                    PRIMARY KEY(scan_key, params)
                );
                CREATE TABLE IF NOT EXISTS hyps(
                    a_key TEXT NOT NULL, b_key TEXT NOT NULL, params TEXT NOT NULL,
                    stage TEXT NOT NULL, result TEXT NOT NULL, updated REAL,
                    PRIMARY KEY(a_key, b_key, params, stage)
                );
                CREATE INDEX IF NOT EXISTS links_params ON links(params);
                CREATE INDEX IF NOT EXISTS artifacts_key ON artifacts(scan_key, kind);
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------ files

    def put_file(self, sha: str, path: str, size: int, mtime: int, scan_count: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO files(sha,path,size,mtime,scan_count,updated) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(sha) DO UPDATE SET path=excluded.path,size=excluded.size,"
                "mtime=excluded.mtime,scan_count=excluded.scan_count,updated=excluded.updated",
                (sha, path, size, mtime, scan_count, time.time()),
            )
            self._conn.commit()

    # -------------------------------------------------------------- artifacts

    def put_artifact(
        self, scan_key: str, kind: str, params: str, path: str, meta: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO artifacts(scan_key,kind,params,path,meta,updated) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(scan_key,kind,params) DO UPDATE SET path=excluded.path,"
                "meta=excluded.meta,updated=excluded.updated",
                (scan_key, kind, params, path, json.dumps(meta or {}), time.time()),
            )
            self._conn.commit()

    def get_artifact(self, scan_key: str, kind: str, params: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT path, meta FROM artifacts WHERE scan_key=? AND kind=? AND params=?",
                (scan_key, kind, params),
            ).fetchone()
        if row is None:
            return None
        path = Path(row[0])
        if not path.exists():
            return None
        try:
            meta = json.loads(row[1] or "{}")
        except Exception:
            meta = {}
        return {"path": str(path), "meta": meta}

    def put_noise(self, scan_key: str, params: str, cleaned: int, pct: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO noise(scan_key,params,cleaned,pct,updated) VALUES(?,?,?,?,?) "
                "ON CONFLICT(scan_key,params) DO UPDATE SET cleaned=excluded.cleaned,"
                "pct=excluded.pct,updated=excluded.updated",
                (scan_key, params, int(cleaned), float(pct), time.time()),
            )
            self._conn.commit()

    # ------------------------------------------------------------------ links

    def get_links(self, keys: Iterable[tuple[str, str]], params: str) -> dict[tuple[str, str], dict[str, Any]]:
        keys = list(keys)
        if not keys:
            return {}
        out: dict[tuple[str, str], dict[str, Any]] = {}
        with self._lock:
            for a, b in keys:
                row = self._conn.execute(
                    "SELECT result FROM links WHERE a_key=? AND b_key=? AND params=?",
                    (a, b, params),
                ).fetchone()
                if row is not None:
                    try:
                        out[(a, b)] = json.loads(row[0])
                    except Exception:
                        pass
        return out

    # ------------------------------------------------------------ hypotheses

    def get_hyps(self, pairs: Iterable[tuple[str, str]], params: str) -> dict[tuple[str, str, str], dict[str, Any]]:
        """All stored stage results for the given (a_key, b_key) pairs."""
        wanted = set(pairs)
        out: dict[tuple[str, str, str], dict[str, Any]] = {}
        if not wanted:
            return out
        with self._lock:
            rows = self._conn.execute(
                "SELECT a_key,b_key,stage,result FROM hyps WHERE params=?", (params,)
            ).fetchall()
        for a, b, stage, res in rows:
            if (a, b) in wanted:
                try:
                    out[(a, b, stage)] = json.loads(res)
                except Exception:
                    pass
        return out

    def put_hyps(self, a_key: str, b_key: str, params: str, stage: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO hyps(a_key,b_key,params,stage,result,updated) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(a_key,b_key,params,stage) DO UPDATE SET result=excluded.result,updated=excluded.updated",
                (a_key, b_key, params, stage, json.dumps(result), time.time()),
            )
            self._conn.commit()

    def drop_hyps(self, keys: Iterable[str]) -> int:
        keys = list(keys)
        if not keys:
            return 0
        q = ",".join("?" * len(keys))
        with self._lock:
            n = self._conn.execute(f"DELETE FROM hyps WHERE a_key IN ({q}) OR b_key IN ({q})", keys + keys).rowcount
            self._conn.commit()
        return int(n)

    def put_link(
        self, a_key: str, b_key: str, params: str, result: dict[str, Any], *, overwrite: bool = True
    ) -> None:
        """Store a matcher result.  ``overwrite=False`` only fills gaps: used
        when seeding from a project file, whose pairs are a slim summary that
        must never replace a full matcher result (it lacks the evidence the
        acceptance rule needs)."""
        payload = {k: v for k, v in result.items() if k not in ("a", "b")}
        conflict = (
            "DO UPDATE SET result=excluded.result,updated=excluded.updated" if overwrite else "DO NOTHING"
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO links(a_key,b_key,params,result,updated) VALUES(?,?,?,?,?) "
                f"ON CONFLICT(a_key,b_key,params) {conflict}",
                (a_key, b_key, params, json.dumps(payload), time.time()),
            )
            self._conn.commit()

    # ------------------------------------------------------------------ stats

    def stats(self) -> dict[str, Any]:
        with self._lock:
            files = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            bundles = self._conn.execute(
                "SELECT COUNT(*) FROM artifacts WHERE kind='bundle'"
            ).fetchone()[0]
            links = self._conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
            noise = self._conn.execute(
                "SELECT COALESCE(SUM(cleaned),0), COALESCE(AVG(pct),0) FROM noise"
            ).fetchone()
        return {
            "files": int(files),
            "bundles": int(bundles),
            "links": int(links),
            "noise_points": int(noise[0]),
            "noise_pct": float(noise[1]),
        }


DB = CacheDB()
