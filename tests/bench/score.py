"""Score a registration result against a frozen ground truth.

    python tests/bench/score.py --truth T.json                     # live server state
    python tests/bench/score.py --truth T.json --session run.json  # a saved session/snapshot

Pairs within one truth group: correct (within --trans/--rot), wrong pose, or
not linked.  Links between two *different* multi-scan truth groups are hard
errors.  Links involving scans with no confirmed placement are listed for
review (they may be right; the truth does not know).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np


def rot_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def load_result(args: argparse.Namespace) -> dict:
    if args.session:
        return json.loads(Path(args.session).read_text())
    return json.load(urllib.request.urlopen(f"{args.url}/api/state"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="truth file, e.g. tests/bench/office_truth.json")
    ap.add_argument("--session")
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--trans", type=float, default=0.10, help="m")
    ap.add_argument("--rot", type=float, default=1.0, help="degrees")
    args = ap.parse_args()
    truth = json.loads(Path(args.truth).read_text())
    res = load_result(args)
    lab = {s["id"]: s["label"] for s in res["scans"]}
    T = {s["label"]: np.asarray(s["transform"], dtype=float) for s in res["scans"]}
    adj: dict[str, set] = {l: set() for l in T}
    links = []
    for p in res.get("pairs", []):
        if p.get("accepted"):
            a, b = lab[p["a"]], lab[p["b"]]
            adj[a].add(b); adj[b].add(a); links.append((a, b))
    comp, seen = {}, set()
    for l in T:
        if l in seen:
            continue
        stack, cid = [l], l
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp[x] = cid; stack += adj[x]
    tgroup = {l: i for i, g in enumerate(truth["groups"]) for l in g}
    multi = {i for i, g in enumerate(truth["groups"]) if len(g) > 1}
    ok, bad, missing = [], [], []
    for g in truth["groups"]:
        for a, b in itertools.combinations(g, 2):
            if a not in T or b not in T:
                continue
            if comp[a] != comp[b]:
                missing.append((a, b)); continue
            want = np.asarray(truth["relative"][f"{a}|{b}"])
            got = np.linalg.inv(T[b]) @ T[a]
            d = np.linalg.inv(want) @ got
            e_t, e_r = float(np.linalg.norm(d[:3, 3])), rot_deg(d[:3, :3])
            (ok if e_t <= args.trans and e_r <= args.rot else bad).append((a, b, e_t, e_r))
    cross = [(a, b) for a, b in links if tgroup.get(a) in multi and tgroup.get(b) in multi and tgroup[a] != tgroup[b]]
    review = [(a, b) for a, b in links if tgroup.get(a) not in multi or tgroup.get(b) not in multi]
    n = len(ok) + len(bad) + len(missing)
    print(f"truth pairs: {n}  correct: {len(ok)}  wrong pose: {len(bad)}  not linked: {len(missing)}")
    print(f"links between different truth groups (errors): {len(cross)}")
    for a, b, et, er in bad:
        print(f"  WRONG  {a} ↔ {b}: {et:.2f} m / {er:.1f}°")
    for a, b in missing:
        print(f"  MISSING {a} ↔ {b}")
    for a, b in cross:
        print(f"  CROSS  {a} ↔ {b}")
    for a, b in review:
        print(f"  review {a} ↔ {b} (no confirmed placement in the truth)")
    if ok:
        print(f"  correct pairs: max error {max(e for *_, e, _r in ok) * 100:.1f} cm / {max(r for *_, r in ok):.2f}°")
    return 1 if bad or cross else 0


if __name__ == "__main__":
    sys.exit(main())
