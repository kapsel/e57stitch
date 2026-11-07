"""Permanent record of manual placements.

Everything the user places by hand (Snap + Accept, Accept anyway, panorama
markers, accepting a link) is appended here and **never deleted**: resets,
fresh registrations, cache clean-ups and benchmark runs do not touch it.
Undo / reject only marks a row as retracted (with time and reason).

Lives outside the cache (``~/Library/Application Support/e57stitch`` on
macOS, ``~/.local/share/e57stitch`` elsewhere, or ``$E57STITCH_MANUAL_DB``),
with a plain-text JSON-lines mirror next to it as a second copy.

Scans are identified by content (``file_hash:index``), so a placement follows
the scan into any project that contains it.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable


def _default_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "e57stitch"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "e57stitch"


MANUAL_DB = Path(os.environ.get("E57STITCH_MANUAL_DB", _default_dir() / "manual_links.sqlite"))


class ManualStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else MANUAL_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = self.path.with_suffix(".jsonl")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS manual_links(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created REAL NOT NULL,
                project TEXT,
                a_key TEXT NOT NULL, b_key TEXT NOT NULL,
                a_label TEXT, b_label TEXT, a_file TEXT, b_file TEXT,
                transform TEXT NOT NULL,          -- a-local -> b-local, 4x4 rows
                world_a TEXT, world_b TEXT,       -- both poses when placed
                kind TEXT,                        -- snap | force | markers | accepted-link
                evidence TEXT, markers TEXT,
                retracted REAL, retract_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS manual_pair ON manual_links(a_key, b_key);
            -- every saved set of panorama markers (history; latest = current)
            CREATE TABLE IF NOT EXISTS marker_sets(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created REAL NOT NULL,
                project TEXT,
                moving_key TEXT NOT NULL, target_key TEXT NOT NULL,
                moving_label TEXT, target_label TEXT,
                pairs TEXT NOT NULL               -- [{t:[u,v], m:[u,v], ...}]
            );
            CREATE INDEX IF NOT EXISTS marker_pair ON marker_sets(moving_key, target_key);
            -- scan pairs the user banned: no link between them, whatever matches
            CREATE TABLE IF NOT EXISTS blocked_links(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created REAL NOT NULL,
                project TEXT,
                a_key TEXT NOT NULL, b_key TEXT NOT NULL,
                a_label TEXT, b_label TEXT,
                unblocked REAL
            );
            CREATE INDEX IF NOT EXISTS blocked_pair ON blocked_links(a_key, b_key);
            """
        )
        self._conn.commit()

    def _mirror(self, rec: dict[str, Any]) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def add(self, *, project: str, a_key: str, b_key: str, transform: Any, a_label: str = "", b_label: str = "",
            a_file: str = "", b_file: str = "", world_a: Any = None, world_b: Any = None, kind: str = "snap",
            evidence: dict[str, Any] | None = None, markers: Any = None) -> int:
        rec = {
            "op": "add", "created": time.time(), "project": project, "a_key": a_key, "b_key": b_key,
            "a_label": a_label, "b_label": b_label, "a_file": a_file, "b_file": b_file,
            "transform": _rows(transform), "world_a": _rows(world_a), "world_b": _rows(world_b),
            "kind": kind, "evidence": evidence or {}, "markers": markers,
        }
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO manual_links(created, project, a_key, b_key, a_label, b_label, a_file, b_file,"
                " transform, world_a, world_b, kind, evidence, markers) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec["created"], project, a_key, b_key, a_label, b_label, a_file, b_file,
                 json.dumps(rec["transform"]), json.dumps(rec["world_a"]), json.dumps(rec["world_b"]), kind,
                 json.dumps(rec["evidence"]), json.dumps(markers)),
            )
            self._conn.commit()
            rec["id"] = int(cur.lastrowid)
            self._mirror(rec)
        return rec["id"]

    def retract(self, a_key: str, b_key: str, reason: str) -> int:
        """Mark every active placement of this scan pair as retracted."""
        now = time.time()
        with self._lock:
            n = self._conn.execute(
                "UPDATE manual_links SET retracted=?, retract_reason=? WHERE retracted IS NULL AND "
                "((a_key=? AND b_key=?) OR (a_key=? AND b_key=?))",
                (now, reason, a_key, b_key, b_key, a_key),
            ).rowcount
            self._conn.commit()
            if n:
                self._mirror({"op": "retract", "time": now, "a_key": a_key, "b_key": b_key, "reason": reason})
        return int(n)

    def active(self, keys: Iterable[str]) -> list[dict[str, Any]]:
        """Newest active placement per scan pair, for pairs inside ``keys``."""
        keys = set(keys)
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, created, project, a_key, b_key, a_label, b_label, transform, kind FROM manual_links "
                "WHERE retracted IS NULL ORDER BY created DESC"
            ).fetchall()
        out, seen = [], set()
        for rid, created, project, a, b, al, bl, T, kind in rows:
            pair = frozenset((a, b))
            if a not in keys or b not in keys or pair in seen:
                continue
            seen.add(pair)
            out.append({"id": rid, "created": created, "project": project, "a_key": a, "b_key": b,
                        "a_label": al, "b_label": bl, "transform": json.loads(T), "kind": kind})
        return out

    def block(self, *, project: str, a_key: str, b_key: str, a_label: str = "", b_label: str = "") -> None:
        """Ban a scan pair until it is explicitly unblocked (idempotent)."""
        now = time.time()
        with self._lock:
            active = self._conn.execute(
                "SELECT 1 FROM blocked_links WHERE unblocked IS NULL AND "
                "((a_key=? AND b_key=?) OR (a_key=? AND b_key=?))",
                (a_key, b_key, b_key, a_key),
            ).fetchone()
            if active:
                return
            self._conn.execute(
                "INSERT INTO blocked_links(created, project, a_key, b_key, a_label, b_label) VALUES (?,?,?,?,?,?)",
                (now, project, a_key, b_key, a_label, b_label),
            )
            self._conn.commit()
            self._mirror({"op": "block", "time": now, "project": project, "a_key": a_key, "b_key": b_key,
                          "a_label": a_label, "b_label": b_label})

    def unblock(self, a_key: str, b_key: str) -> int:
        """Lift a ban; the row stays as history."""
        now = time.time()
        with self._lock:
            n = self._conn.execute(
                "UPDATE blocked_links SET unblocked=? WHERE unblocked IS NULL AND "
                "((a_key=? AND b_key=?) OR (a_key=? AND b_key=?))",
                (now, a_key, b_key, b_key, a_key),
            ).rowcount
            self._conn.commit()
            if n:
                self._mirror({"op": "unblock", "time": now, "a_key": a_key, "b_key": b_key})
        return int(n)

    def blocked(self, keys: Iterable[str]) -> list[dict[str, Any]]:
        """Active bans for pairs inside ``keys``."""
        keys = set(keys)
        with self._lock:
            rows = self._conn.execute(
                "SELECT created, project, a_key, b_key FROM blocked_links WHERE unblocked IS NULL ORDER BY created"
            ).fetchall()
        return [{"created": created, "project": project, "a_key": a, "b_key": b}
                for created, project, a, b in rows if a in keys and b in keys]

    def save_markers(self, *, project: str, moving_key: str, target_key: str, pairs: list[dict[str, Any]],
                     moving_label: str = "", target_label: str = "") -> int:
        """Append the current marker set of a scan pair (kept forever)."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO marker_sets(created, project, moving_key, target_key, moving_label, target_label, pairs)"
                " VALUES (?,?,?,?,?,?,?)",
                (now, project, moving_key, target_key, moving_label, target_label, json.dumps(pairs)),
            )
            self._conn.commit()
            self._mirror({"op": "markers", "time": now, "project": project, "moving_key": moving_key,
                          "target_key": target_key, "pairs": pairs})
            return int(cur.lastrowid)

    def latest_markers(self, moving_key: str, target_key: str) -> dict[str, Any] | None:
        """Latest marker set for this pair, in either direction (t/m swapped
        back so that ``t`` is always on ``target_key``)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT created, moving_key, pairs FROM marker_sets WHERE (moving_key=? AND target_key=?) "
                "OR (moving_key=? AND target_key=?) ORDER BY created DESC LIMIT 1",
                (moving_key, target_key, target_key, moving_key),
            ).fetchone()
        if row is None:
            return None
        created, mk, pairs = row
        pairs = json.loads(pairs)
        if mk != moving_key:
            pairs = [{"t": p.get("m"), "m": p.get("t")} for p in pairs]
        return {"created": created, "pairs": pairs}

    def marker_counts(self, keys: Iterable[str]) -> dict[frozenset, int]:
        """Number of markers in the latest set per scan pair."""
        keys = set(keys)
        with self._lock:
            rows = self._conn.execute(
                "SELECT moving_key, target_key, pairs FROM marker_sets ORDER BY created DESC").fetchall()
        out: dict[frozenset, int] = {}
        for mk, tk, pairs in rows:
            k = frozenset((mk, tk))
            if mk in keys and tk in keys and k not in out:
                out[k] = len(json.loads(pairs))
        return out

    def count(self) -> dict[str, int]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM manual_links").fetchone()[0]
            act = self._conn.execute("SELECT COUNT(*) FROM manual_links WHERE retracted IS NULL").fetchone()[0]
        return {"total": int(total), "active": int(act)}


def _rows(T: Any) -> Any:
    if T is None:
        return None
    try:
        import numpy as np

        return np.asarray(T, dtype=float).tolist()
    except Exception:
        return T


_STORE: ManualStore | None = None


def store() -> ManualStore:
    global _STORE
    if _STORE is None:
        _STORE = ManualStore()
    return _STORE
