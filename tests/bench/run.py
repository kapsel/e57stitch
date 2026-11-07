"""Fresh registration of a benchmark project, scored and recorded.

    python tests/bench/run.py NAME                 # fresh run, score, append history
    python tests/bench/run.py NAME --note "try"
    python tests/bench/run.py NAME --score-only SESSION.json

NAME is a benchmark from tests/bench/benches.json, which maps it to a
project name and a truth file:

    {"office": ["Office scans", "office_truth.json"]}

Without that file every tests/bench/<name>_truth.json is a benchmark whose
project has the same name.  Truth files and benches.json describe your own
scans and are not part of the repository.

Runs in a private Session: the project's saved links are never touched.
"Fresh" = cached candidates of these scans dropped, no links, no vetoes.
Every run appends one line to tests/bench/history.jsonl (commit, engine,
scores, runtime).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def _benches() -> dict[str, tuple[str, Path]]:
    cfg = HERE / "benches.json"
    if cfg.exists():
        return {k: (v[0], HERE / v[1]) for k, v in json.loads(cfg.read_text()).items()}
    return {p.name[: -len("_truth.json")]: (p.name[: -len("_truth.json")], p) for p in sorted(HERE.glob("*_truth.json"))}


BENCH = _benches()
TOL_M, TOL_DEG = 0.10, 1.0


def rot_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def score(state: dict, truth: dict) -> dict:
    lab = {s["id"]: s["label"] for s in state["scans"]}
    T = {s["label"]: np.asarray(s["transform"], dtype=float) for s in state["scans"]}
    adj: dict[str, set] = {l: set() for l in T}
    links = []
    for p in state.get("pairs", []):
        if p.get("accepted"):
            a, b = lab[p["a"]], lab[p["b"]]
            adj[a].add(b); adj[b].add(a); links.append((a, b))
    comp: dict[str, str] = {}
    for l in T:
        if l in comp:
            continue
        stack = [l]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp[x] = l
            stack += adj[x]
    tg = {l: i for i, g in enumerate(truth["groups"]) for l in g}
    multi = {i for i, g in enumerate(truth["groups"]) if len(g) > 1}

    def rel_ok(a: str, b: str) -> tuple[bool, float, float]:
        k = f"{a}|{b}"
        want = np.asarray(truth["relative"][k]) if k in truth["relative"] else np.linalg.inv(np.asarray(truth["relative"][f"{b}|{a}"]))
        d = np.linalg.inv(want) @ (np.linalg.inv(T[b]) @ T[a])
        et, er = float(np.linalg.norm(d[:3, 3])), rot_deg(d[:3, :3])
        return et <= TOL_M and er <= TOL_DEG, et, er

    pairs = {"correct": 0, "wrong": 0, "missing": 0}
    for g in truth["groups"]:
        for a, b in itertools.combinations(g, 2):
            if a not in T or b not in T:
                continue
            if comp[a] != comp[b]:
                pairs["missing"] += 1
            else:
                pairs["correct" if rel_ok(a, b)[0] else "wrong"] += 1
    cross = [(a, b) for a, b in links if tg.get(a) in multi and tg.get(b) in multi and tg[a] != tg[b]]

    # scans placed correctly: per truth group, the largest set of scans that
    # are mutually right (a reference scan + everyone correct relative to it)
    # Pieces: repeatedly take the largest set of still-unassigned scans that
    # are mutually right; scans sharing a component with a piece but not in
    # it are misplaced, single leftovers are unattached.
    placed, wrong_scans, unattached, pieces = [], [], [], []
    for gi in multi:
        left = [l for l in truth["groups"][gi] if l in T]
        group_pieces = []
        while left:
            best: list[str] = []
            for r in left:
                ok = [r] + [s for s in left if s != r and comp[s] == comp[r] and rel_ok(s, r)[0]]
                if len(ok) > len(best):
                    best = ok
            if len(best) < 2:
                break
            group_pieces.append(best)
            left = [s for s in left if s not in best]
        if group_pieces:
            placed += group_pieces[0]
        pieces += [len(p) for p in group_pieces]
        in_piece = {s for p in group_pieces for s in p}
        piece_comps = {comp[p[0]] for p in group_pieces}
        for s in left:
            (wrong_scans if comp[s] in piece_comps or any(comp[s] == comp[x] for x in T if x != s) else unattached).append(s)
        for p in group_pieces[1:]:
            if comp[p[0]] == comp[group_pieces[0][0]]:
                wrong_scans += p          # attached to the main piece, but in the wrong place
    n_truth = sum(len([l for l in truth["groups"][i] if l in T]) for i in multi)
    return {
        "scans": len(T), "scans_in_truth": n_truth, "placed_right": len(placed), "pieces": sorted(pieces, reverse=True),
        "wrong_scans": sorted(wrong_scans), "unattached": sorted(unattached),
        "links": len(links), "groups": len(set(comp.values())), "pairs": pairs, "cross": len(cross),
        "cross_links": cross,
    }


def git_info() -> dict:
    def run(*a: str) -> str:
        try:
            return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
        except Exception:
            return ""
    if os.environ.get("E57STITCH_COMMIT"):  # container: no .git, deploy.sh passes the version
        return {"commit": os.environ["E57STITCH_COMMIT"], "branch": os.environ.get("E57STITCH_BRANCH", ""),
                "dirty": bool(os.environ.get("E57STITCH_DIRTY"))}
    return {"commit": run("rev-parse", "--short", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "e57stitch"))}


def fresh_run(project: str, reuse: bool = False) -> tuple[dict, float, str]:
    sys.path.insert(0, str(ROOT))
    from e57stitch.db import DB
    from e57stitch.projects import ProjectManager

    meta = None
    for f in glob.glob(os.path.expanduser("~/.cache/e57stitch/projects/*/project.json")):
        m = json.loads(Path(f).read_text())
        if m.get("name") == project:
            meta = m
    if meta is None:
        raise SystemExit(f"project {project!r} not found")
    s = ProjectManager._build_session(object.__new__(ProjectManager), meta)
    s.on_change = None                    # never save into the real project
    s.use_manual_store = False            # measure the automatic engine only

    def wait(jid: str) -> dict:
        while True:
            j = s.job(jid)
            if j["status"] in ("done", "error", "cancelled"):
                if j["status"] != "done":
                    raise SystemExit(f"job failed: {j.get('error')}")
                return j.get("result") or {}
            time.sleep(1)

    wait(s.load())
    for i, sc in s.scans.items():
        if "thermal" in (_label(meta, i) or ""):  # same geometry as the real scan: not a registration input
            sc.included = False
    # reuse: keep the computed candidates, re-run only linking/selection
    s.reset_registration(vetoes=True, cached=not reuse)
    if not reuse:
        DB.drop_hyps([s._scan_key(i) for i in s.scans])
    t0 = time.time()
    r = wait(s.register_scans(fresh=not reuse, keep_improving=False))
    secs = time.time() - t0
    return s.session_data() | {"scans": [
        {"id": i, "label": _label(meta, i), "transform": sc.transform.tolist()}
        for i, sc in s.scans.items()]}, secs, r.get("log_path", "")


def _label(meta: dict, sid: str) -> str:
    for e in meta["scans"]:
        if e.get("id") == sid:
            return e.get("label") or sid
    return sid


def apply_overrides(items: list[str]) -> dict[str, Any]:
    """--set pipeline.SELECTION_MODE=growth etc.: patch engine constants for this run."""
    import ast
    import importlib

    sys.path.insert(0, str(ROOT))
    out = {}
    for item in items:
        key, val = item.split("=", 1)
        mod, name = key.rsplit(".", 1)
        m = importlib.import_module(f"e57stitch.{mod}")
        try:
            v = ast.literal_eval(val)
        except Exception:
            v = val
        setattr(m, name, v)
        if mod == "pipeline" and name == "SELECTION_MODE":
            m.SELECTION_VERSION = "s9" if v == "growth" else m.SELECTION_VERSION
        out[key] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=sorted(BENCH))
    ap.add_argument("--score-only", help="score a saved state/session JSON instead of running")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="keep cached candidates; only linking is re-run (fast)")
    ap.add_argument("--set", action="append", default=[], metavar="MODULE.NAME=VALUE",
                    help="engine setting for this run only, e.g. pipeline.SELECTION_MODE=growth, register.LOCK_TILT=False")
    args = ap.parse_args()
    project, truth_path = BENCH[args.bench]
    overrides = apply_overrides(args.set)
    truth = json.loads(truth_path.read_text())
    if args.score_only:
        state, secs, log = json.loads(Path(args.score_only).read_text()), None, ""
    else:
        state, secs, log = fresh_run(project, reuse=args.reuse)
        out = HERE / "runs"
        out.mkdir(exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        (out / f"{args.bench}-{stamp}.json").write_text(json.dumps(state))
    res = score(state, truth)
    sys.path.insert(0, str(ROOT))
    from e57stitch.pipeline import PIPELINE_VERSION, SELECTION_VERSION
    from e57stitch.register import ENGINE_VERSION

    rec = {"time": dt.datetime.now().isoformat(timespec="seconds"), **git_info(), "reuse": bool(getattr(args, "reuse", False)), "engine": ENGINE_VERSION,
           "pipeline": PIPELINE_VERSION, "selection": SELECTION_VERSION, "bench": args.bench, "seconds": None if secs is None else round(secs),
           "note": args.note, "settings": overrides, **{k: v for k, v in res.items() if k != "cross_links"}}
    print(json.dumps(rec, indent=1))
    if log and Path(log).exists():
        print(Path(log).read_text())
    if not args.no_record and not args.score_only:
        with open(HERE / "history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
