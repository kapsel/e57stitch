"""Registration pipeline: generate hypotheses, decide globally, keep improving.

Stages
------
0. prepare   per-scan bundles (levelled cloud, walls, features) - Session._load_impl
1+2. candidates  every pair: floor-plan search -> ICP -> evaluated hypotheses (~2 s/pair);
                 pairs without floor-plan overlap drop out in ~0.4 s
3. select    choose at most one hypothesis per pair so that the most links agree
             (consensus over all pairs; order independent); ambiguous or weak
             single links are not trusted, loop closures must agree
4. refine    multi-view ICP on the chosen links, 2 cm polish, surface accuracy
5. improve   background, by priority: unlinked scans, scans held by one link,
             unresolved pairs, non-levelled pairs -> every heavy matcher and
             merged-group matching add hypotheses -> stage 3 again

Hypotheses live in the SQLite ``hyps`` table keyed by scan content + settings,
so they survive restarts and are shared by projects.  Links, groups and poses
are always *derived* from them (plus the user's manual links); nothing patches
a link directly.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
import traceback
from concurrent.futures import FIRST_COMPLETED, wait
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .db import DB
from .photometric import FS_MAX, NCC_MARGIN, NCC_MIN
from .workers import process_pool
from .register import (
    BEV_GREY,
    BEV_MIN,
    PairJob,
    _pose_close,
    _bridges_of,
    candidates_job,
    evaluate_job,
    heavy_job,
    local_job,
    invert,
    register_pair_job,
    rotation_angle_deg,
    strong_link,
)

if TYPE_CHECKING:  # pragma: no cover
    from .session import Session

PIPELINE_VERSION = "p1"
# Link selection rules (re-run on cached hypotheses, so no cache change):
# s1 consensus + stations + triangles; s2 co-location merges deferred;
# s2m manual placements applied first (always win), read from the manual store;
# s3 whole-merge contradiction check (implemented, off until benchmarked).
WORLD_CHECK = False
# s4: build the model scan by scan, every step verified against all of it
SELECTION_MODE = "growth"          # "growth" | "consensus" (default since Engine 15, consensus kept for comparison)
# s5: growth + appearance (panorama NCC) check; s6: + piece joins; s7: + panorama free-space check
# s8: + storey twins (height-only alternatives tie; capture order breaks the tie, flagged)
#     and growth decides scans closest in capture order to the model first
STOREY_BY_ORDER = True
ORDER_FREE = True                  # s10: selection and route ignore capture order (engine 16)
EVIDENCE_K = 4.0                   # evidence = NCC - k * see-through share; same result for k = 4..12 on benchmark B
GROWTH_STARTS = 4                  # other seeds to try when an assembly contradicts itself
VIOLATION_RADIUS = 15.0            # m; scan pairs compared when an assembly is checked
SINGLE_EV_MIN = 0.25               # mean evidence a placement on 1-2 links needs: wrong ones reached 0.22 (benchmark B), the lowest right one is 0.28 (benchmark A, scans 5-6)
EV_MARGIN = 0.10                   # mean evidence by which one of two rival positions must lead (benchmark A, scan 7: 0.70 vs 0.53)
RESTART_VIOL = 5                   # violations before an assembly counts as wrong (bad starts: dozens; one marginal pair is not a reason)
RESCUE_BATCH = 16                  # local-search pairs between pieces per round
RESCUE_PAIRS = 64                  # at most this many per run: real separate buildings must not cost all pairs
RESCUE_LINKS = 2                   # agreeing links a joined piece needs before the rescue stops early
RESCUE_IDLE = 2                    # rounds in a row without a join before giving up (benchmark A: 4 rounds, 11 min, nothing; benchmark B joins in round 2)
SELECTION_VERSION = "s11" if SELECTION_MODE == "growth" else ("s3" if WORLD_CHECK else "s2m")
WORLD_RADIUS = 20.0       # m: scan pairs further apart are not compared
WORLD_OPTIONS = 4         # alternative joins of two pieces compared per merge
WORLD_MAX_BAD = 3         # a join contradicting this many scan pairs (and ...)
WORLD_MAX_FRAC = 0.15     # ... this share of the compared pairs) is not made
AMBIGUOUS = 0.04          # rank gap below which two plausible poses of one pair are a tie
AGREE_TRANS = 0.25        # m
AGREE_ROT = 2.5           # degrees
MANUAL_WEIGHT = 10.0


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")


def pool_shape(requested: int) -> tuple[int, int]:
    """(workers, threads per worker).  Fewer workers with a few threads each
    finish each pair sooner (links show up earlier) at about the same total
    throughput as one thread per core."""
    import os

    cores = max(1, os.cpu_count() or 4)
    workers = max(1, min(int(requested), max(2, cores // 2)))
    return workers, max(1, cores // workers)


def _worker_init(threads: int = 1) -> None:
    import os

    from . import register as _reg

    for key in THREAD_VARS:
        os.environ[key] = str(threads)
    _reg.KD_WORKERS = threads
    # Open3D ignores OMP_NUM_THREADS on macOS; its own limit works.  With 8
    # workers each spinning up a thread per core, a pair cost ~5x the CPU.
    try:
        import open3d as o3d

        o3d.utility.set_max_threads(threads)
    except Exception:
        pass


def _single_threaded_children() -> dict[str, str | None]:
    """Spawned workers import Open3D while unpickling their first task, i.e.
    *before* any initializer runs, so thread limits must already be in the
    environment they inherit.  Returns the previous values to restore."""
    import os

    old = {k: os.environ.get(k) for k in THREAD_VARS}
    for k in THREAD_VARS:
        os.environ[k] = "1"
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    import os

    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# --------------------------------------------------------------------------- #
# global selection
# --------------------------------------------------------------------------- #


STATION_SEP = 1.0         # m: scanners closer than this see the room from the same place
# Two *separate* setups (different E57 files) almost never stand on the same
# spot: benchmark B truth has 2 such pairs in 253, while repeated geometry (stacked
# stairwell landings, identical bays) produced 21 of 65 wrong candidates.
# A merge that claims it is therefore made last and must pass the strict
# single-link test; it never counts as independent confirmation.
COLOC_SEP = 0.5           # m


def _world_pick(best, opts, members, pose, station, world_check):
    """Among alternative joins of the same two pieces, the one that makes the
    fewest scan pairs contradict each other; a string (reason) if every
    option makes the merged model contradict itself."""
    _key, ci, cj, _X, _support, keys = best
    distinct: list[tuple[tuple, np.ndarray, dict]] = []
    for o in sorted(opts, key=lambda o: o[0], reverse=True):
        if not o[0][0]:
            continue  # co-location joins are handled elsewhere
        if any(_pose_close(o[1], d[1], trans=AGREE_TRANS, rot_deg=AGREE_ROT) for d in distinct):
            continue
        distinct.append(o)
        if len(distinct) >= WORLD_OPTIONS:
            break
    scored = []
    for rank, (key, X, support) in enumerate(distinct):
        sup = {frozenset(k) for k in support}
        pairs = []
        for x in members[ci]:
            px = station(x)
            for y in members[cj]:
                if frozenset((x, y)) in sup:
                    continue
                Py = X @ pose[y]
                py = Py[:3, :3] @ (invert(pose[y]) @ np.append(station(y), 1.0))[:3] + Py[:3, 3]
                if float(np.linalg.norm(px - py)) > WORLD_RADIUS:
                    continue
                pairs.append((x, y, invert(Py) @ pose[x]))
        bad = world_check(pairs) if pairs else []
        scored.append((len(bad), rank, key, X, support, bad, len(pairs)))
    if not scored:
        return best
    nbad, _rank, key, X, support, bad, n = min(scored, key=lambda t: (t[0], t[1]))
    if nbad >= WORLD_MAX_BAD and nbad >= WORLD_MAX_FRAC * max(1, n):
        ex = ", ".join(f"{a}×{b}" for a, b, *_ in bad[:3])
        return f"joining here would make {nbad} scan pairs contradict each other (e.g. {ex})"
    return (key, ci, cj, X, support, keys)


GROWTH_RADIUS = 20.0      # m: placed scans a candidate is checked against
GROWTH_TIE = 0.10         # a second verified position within 10 % of the best score: wait


SLICE_DZ_MAX = 2.5        # m: slice contradiction test only between scans this close in height (s9)
STOREY_MIN = 1.0          # m: candidates this far apart in height only are storey twins
STOREY_XY = 0.5           # m
STOREY_YAW = 3.0          # deg


def _storey_twin(A: np.ndarray, B: np.ndarray, origin: np.ndarray) -> bool:
    """Same scanner x/y and heading, different height (world z is up)."""
    sa = A[:3, :3] @ origin + A[:3, 3]
    sb = B[:3, :3] @ origin + B[:3, 3]
    d = sa - sb
    yaw = math.degrees(abs(math.remainder(math.atan2(A[1, 0], A[0, 0]) - math.atan2(B[1, 0], B[0, 0]), 2 * math.pi)))
    return float(np.hypot(d[0], d[1])) < STOREY_XY and abs(float(d[2])) >= STOREY_MIN and yaw < STOREY_YAW


def _link_components(ids: list[str], chosen: list[dict[str, Any]]) -> dict[str, str]:
    """Scan id -> id of its piece, from the chosen links."""
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in chosen:
        if c["a"] in parent and c["b"] in parent:
            parent[find(c["a"])] = find(c["b"])
    return {i: find(i) for i in ids}


def select_growth(ids: list[str], edges: dict[tuple[str, str], list[dict[str, Any]]],
                  origins: dict[str, np.ndarray], setups: dict[str, str],
                  contradicts: Callable[[str, str, np.ndarray], bool],
                  log: Callable[[str], None] | None = None,
                  appearance: Callable[[str, str, np.ndarray], float | None] | None = None,
                  order: dict[str, int] | None = None,
                  twin_choice: Callable[..., Any] | None = None,
                  evidence: Callable[[str, str, dict[str, Any]], float] | None = None,
                  seed_pair: tuple[str, str] | None = None,
                  single_min: float | None = None) -> dict[str, Any]:
    """Build the model scan by scan, checking every step against all of it.

    Manual links first (they always win).  Then, repeatedly, every scan not
    yet placed gets its candidate world poses from hypotheses with placed
    scans; candidates are clustered (agreeing placed scans add up) and
    verified: no other setup on the same spot (COLOC_SEP), no contradiction
    (height-slice see-through) with any placed scan within GROWTH_RADIUS.
    The scan with the best verified, unambiguous candidate is placed; a scan
    whose best verified candidate has a near-equal verified rival waits.

    s8 storey twins: verified candidates that differ only in height (same
    x/y and heading, >= STOREY_MIN apart) are a physical tie whatever their
    votes - on identical storeys the votes are not independent evidence
    (engine 7).  With `order` (capture order) the twin backed by the
    scans closest in capture order wins and the scan is flagged "check
    storey"; without it the scan waits.  With `twin_choice` (no capture order
    needed) ties wait until nothing else can be placed, then the callback
    gets (pending, W, piece) with pending = [(s, p_id, [candidates])] and
    returns (s, p_id, candidate) or None.  With `evidence` (a, b, hyp) -> float
    the scan and the seed with the strongest evidence are decided first,
    instead of those with the most votes.
    `single_min`: a pose backed by one or two placed scans needs at least this mean evidence,
    otherwise the scan waits (a missing link can be rescued, a wrong one spreads).
    s9: a candidate cluster is verified pose by pose (strongest member first),
    not only by its first member's pose; the caller's slice test skips pairs
    more than SLICE_DZ_MAX apart in height.
    Scans that can never be placed start a new piece (new seed).
    """
    hyp_of: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (a, b), hs in edges.items():
        if a in origins and b in origins:
            hyp_of.setdefault((a, b), []).extend(hs)
            hyp_of.setdefault((b, a), []).extend({**h, "transform": invert(np.asarray(h["transform"])).tolist()} for h in hs)
    W: dict[str, np.ndarray] = {}
    piece: dict[str, int] = {}
    storey_flag: set[str] = set()
    chosen: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    station = lambda i: W[i][:3, :3] @ origins[i] + W[i][:3, 3]   # noqa: E731

    def weight(h: dict[str, Any]) -> float:
        return MANUAL_WEIGHT if h.get("manual") else float(h.get("rank") or 0.0)

    # 1) manual components, placed as given
    manual = sorted(((k, h) for k, hs in edges.items() for h in hs if h.get("manual")),
                    key=lambda kh: -float(kh[1].get("manual_created") or 0.0))
    for (a, b), h in manual:
        T = np.asarray(h["transform"])          # a-local -> b-local
        if a in W and b in W:
            continue
        if a not in W and b not in W:
            W[b] = np.eye(4); piece[b] = len(set(piece.values()))
        if a not in W:
            W[a] = W[b] @ T; piece[a] = piece[b]
        else:
            W[b] = W[a] @ invert(T); piece[b] = piece[a]
        chosen.append({"a": a, "b": b, "hyp": h, "votes": "manual"})

    def candidates(s: str, p_id: int) -> list[dict[str, Any]]:
        cl: list[dict[str, Any]] = []
        for p in [x for x in W if piece[x] == p_id]:
            for h in hyp_of.get((s, p), []):
                Ws = W[p] @ np.asarray(h["transform"])
                for c in cl:
                    if _pose_close(Ws, c["W"], trans=AGREE_TRANS, rot_deg=AGREE_ROT):
                        if p not in c["by"]:
                            c["by"][p] = h
                            c["score"] += weight(h)
                        c["poses"].append((weight(h), Ws))
                        break
                else:
                    cl.append({"W": Ws, "by": {p: h}, "score": weight(h), "poses": [(weight(h), Ws)]})
        return cl

    def verify_any(s: str, c: dict[str, Any], p_id: int) -> bool:
        # s9: a cluster spans AGREE_TRANS; its first member's pose may be the
        # one that is off (benchmark B, scan 025: 028's hypothesis 0.21 m low, rejected by
        # 027, while 026/027's own poses in the same cluster are right).
        # Try every member pose, strongest first; the first that passes is it.
        for _w, Ws in sorted(c["poses"], key=lambda wp: -wp[0]):
            c["W"] = Ws
            if verify(s, c, p_id) is None:
                return True
        return False

    def verify(s: str, c: dict[str, Any], p_id: int) -> str | None:
        Ws = c["W"]
        st = Ws[:3, :3] @ origins[s] + Ws[:3, 3]
        for x in W:
            if piece[x] != p_id:
                continue
            if setups.get(x) != setups.get(s) and float(np.linalg.norm(station(x) - st)) < COLOC_SEP:
                return f"on the spot of {x}"
        for x in W:
            if piece[x] != p_id or float(np.linalg.norm(station(x) - st)) > GROWTH_RADIUS:
                continue
            if contradicts(s, x, invert(W[x]) @ Ws):
                return f"contradicts {x}"
        return None

    def seed_value(a: str, b: str, h: dict[str, Any]) -> float:
        return evidence(a, b, h) if evidence is not None else weight(h)

    unplaced = [i for i in ids if i not in W]
    while unplaced:
        best = None
        pending: list[tuple[str, int, list[dict[str, Any]]]] = []
        for p_id in sorted(set(piece.values())):
            for s in unplaced:
                cl = sorted(candidates(s, p_id), key=lambda c: (-len(c["by"]), -c["score"]))
                verified = [c for c in cl[:6] if verify_any(s, c, p_id)]
                if appearance is not None:
                    # repeated geometry looks different: drop poses whose
                    # appearance does not match their partners
                    kept = []
                    for c in verified:
                        vals = [v for v in (appearance(s, p, invert(W[p]) @ c["W"]) for p in list(c["by"])[:3]) if v is not None]
                        c["ncc"] = max(vals) if vals else None
                        if c["ncc"] is None or c["ncc"] >= NCC_MIN:
                            kept.append(c)
                    verified = kept
                if not verified:
                    continue
                top = verified[0]
                twins = [c for c in verified[1:] if _storey_twin(c["W"], top["W"], origins[s])]
                if twins:
                    if twin_choice is not None:
                        pending.append((s, p_id, [top] + twins))
                        continue
                    if order is None:
                        continue
                    group = [top] + twins
                    group.sort(key=lambda c: (sorted(abs(order[s] - order[p]) for p in c["by"]), -len(c["by"]), -c["score"]))
                    top = dict(group[0], storey_tie=True)
                    gid = {id(c) for c in group}
                    verified = [top] + [c for c in verified if id(c) not in gid]
                rival = [c for c in verified[1:] if not _pose_close(c["W"], top["W"], trans=AGREE_TRANS, rot_deg=AGREE_ROT)
                         and len(c["by"]) >= len(top["by"]) and c["score"] >= top["score"] * (1 - GROWTH_TIE)]
                if rival:
                    allc = [top] + rival
                    if evidence is not None:
                        # s11: mean over all partners; the best single NCC cannot split
                        # two poses a few decimetres apart (benchmark A, scan 7: 0.788 vs 0.782)
                        by_ev = sorted(((float(np.mean([evidence(s, p, h) for p, h in c["by"].items()])), k)
                                        for k, c in enumerate(allc)), reverse=True)
                        if by_ev[0][0] < by_ev[1][0] + EV_MARGIN:
                            continue
                        top = allc[by_ev[0][1]]
                    else:
                        # equal geometry: clearly better appearance decides, else wait
                        scored = [c for c in allc if c.get("ncc") is not None]
                        if len(scored) < len(allc):
                            continue
                        scored.sort(key=lambda c: -c["ncc"])
                        if scored[0]["ncc"] < scored[1]["ncc"] + NCC_MARGIN:
                            continue
                        top = scored[0]
                if single_min is not None and evidence is not None and len(top["by"]) <= 2 \
                        and not any(h.get("manual") for h in top["by"].values()):
                    if float(np.mean([evidence(s, p, h) for p, h in top["by"].items()])) < single_min:
                        continue
                key = (len(top["by"]), top["score"])
                if evidence is not None:
                    key = (float(np.mean([evidence(s, p, h) for p, h in top["by"].items()])),) + key
                if order is not None:
                    # s8: decide the scan closest in capture order to its partners first
                    key = (-min(abs(order[s] - order[p]) for p in top["by"]),) + key
                if best is None or key > best[0]:
                    best = (key, s, p_id, top)
        if best is None and pending and twin_choice is not None:
            picked = twin_choice(pending, W, piece)
            if picked is not None:
                s, p_id, top = picked
                best = ((), s, p_id, dict(top, storey_tie=True))
        if best is None:
            # nothing can join an existing piece: seed a new piece with the
            # strongest pair among the remaining scans (unambiguous best pose)
            seed = None
            for (a, b), hs in hyp_of.items():
                if a in unplaced and b in unplaced and a < b and hs:
                    hs_sorted = sorted(hs, key=weight, reverse=True)
                    h = hs_sorted[0]
                    second = next((o for o in hs_sorted[1:] if not _pose_close(np.asarray(o["transform"]), np.asarray(h["transform"]),
                                                                              trans=AGREE_TRANS, rot_deg=AGREE_ROT)), None)
                    if second is not None and weight(second) >= weight(h) * (1 - GROWTH_TIE):
                        continue
                    Th = np.asarray(h["transform"])
                    if setups.get(a) != setups.get(b) and float(np.linalg.norm(Th[:3, :3] @ origins[a] + Th[:3, 3] - origins[b])) < COLOC_SEP:
                        continue   # repeated geometry, not a seed
                    if appearance is not None:
                        v = appearance(a, b, Th)
                        if v is not None and v < NCC_MIN:
                            continue
                    if single_min is not None and evidence is not None and evidence(a, b, h) < single_min:
                        continue
                    if seed is None or seed_value(a, b, h) > seed_value(*seed):
                        seed = (a, b, h)
            if seed_pair is not None and not W:
                forced = [(a, b, h) for (a, b), hs in hyp_of.items() if a < b and {a, b} == set(seed_pair) for h in hs]
                if forced:
                    seed = max(forced, key=lambda x: seed_value(*x))
            if seed is None:
                break
            a, b, h = seed
            nid = (max(piece.values()) + 1) if piece else 0
            W[b] = np.eye(4); piece[b] = nid
            W[a] = W[b] @ np.asarray(h["transform"]); piece[a] = nid   # T: a-local -> b-local
            chosen.append({"a": a, "b": b, "hyp": h, "votes": "seed"})
            unplaced = [i for i in unplaced if i not in (a, b)]
            if log:
                log(f"growth: new piece {nid} seeded with {a}–{b}")
            continue
        _key, s, p_id, top = best
        W[s] = top["W"]; piece[s] = p_id
        for p, h in top["by"].items():
            chosen.append({"a": s, "b": p, "hyp": h, "votes": "storey by order" if top.get("storey_tie") else len(top["by"])})
        if top.get("storey_tie"):
            storey_flag.add(s)
        unplaced.remove(s)
        if log:
            log(f"growth: placed {s} via {', '.join(top['by'])} (score {top['score']:.2f})"
                + (" — storey tie decided, check it" if top.get("storey_tie") else ""))

    # 2) join whole pieces: a piece moves rigidly onto another when
    # hypotheses between them agree, nothing lands on the same spot, no scan
    # pair sees through the other (within GROWTH_RADIUS) and the bridging
    # scans look alike; equal rivals without a clear appearance winner wait
    while len(set(piece.values())) > 1:
        best_join = None
        pids = sorted(set(piece.values()))
        for P in pids:
            for Q in pids:
                if Q <= P:
                    continue
                # G: moves piece Q's world into piece P's
                cl: list[dict[str, Any]] = []
                for x in [i for i in W if piece[i] == Q]:
                    for y in [i for i in W if piece[i] == P]:
                        for h in hyp_of.get((x, y), []):
                            G = W[y] @ np.asarray(h["transform"]) @ invert(W[x])
                            for c in cl:
                                if _pose_close(G, c["G"], trans=AGREE_TRANS, rot_deg=AGREE_ROT):
                                    if (x, y) not in c["by"]:
                                        c["by"][(x, y)] = h
                                        c["score"] += weight(h)
                                    break
                            else:
                                cl.append({"G": G, "by": {(x, y): h}, "score": weight(h)})
                cl.sort(key=lambda c: (-len(c["by"]), -c["score"]))
                ok = []
                for c in cl[:6]:
                    G = c["G"]
                    bad = None
                    qs = [i for i in W if piece[i] == Q]
                    ps = [i for i in W if piece[i] == P]
                    for x in qs:
                        sx = G[:3, :3] @ station(x) + G[:3, 3]
                        for y in ps:
                            d = float(np.linalg.norm(station(y) - sx))
                            if setups.get(x) != setups.get(y) and d < COLOC_SEP:
                                bad = f"{x} on the spot of {y}"
                                break
                            if d <= GROWTH_RADIUS and contradicts(x, y, invert(W[y]) @ G @ W[x]):
                                bad = f"{x} contradicts {y}"
                                break
                        if bad:
                            break
                    if bad:
                        continue
                    if appearance is not None:
                        vals = [v for v in (appearance(x, y, invert(W[y]) @ G @ W[x]) for (x, y) in list(c["by"])[:3])
                                if v is not None]
                        c["ncc"] = max(vals) if vals else None
                        if c["ncc"] is not None and c["ncc"] < NCC_MIN:
                            continue
                    ok.append(c)
                if not ok:
                    continue
                top = ok[0]
                if single_min is not None and evidence is not None and len(top["by"]) <= 2:
                    if float(np.mean([evidence(x, y, h) for (x, y), h in top["by"].items()])) < single_min:
                        continue
                rival = [c for c in ok[1:] if not _pose_close(c["G"], top["G"], trans=AGREE_TRANS, rot_deg=AGREE_ROT)
                         and len(c["by"]) >= len(top["by"]) and c["score"] >= top["score"] * (1 - GROWTH_TIE)]
                if rival:
                    allc = [top] + rival
                    if any(c.get("ncc") is None for c in allc):
                        continue
                    allc.sort(key=lambda c: -c["ncc"])
                    if allc[0]["ncc"] < allc[1]["ncc"] + NCC_MARGIN:
                        continue
                    top = allc[0]
                key = (len(top["by"]), top["score"])
                if best_join is None or key > best_join[0]:
                    best_join = (key, P, Q, top)
        if best_join is None:
            break
        _k, P, Q, top = best_join
        for x in [i for i in W if piece[i] == Q]:
            W[x] = top["G"] @ W[x]
            piece[x] = P
        for (x, y), h in top["by"].items():
            chosen.append({"a": x, "b": y, "hyp": h, "votes": f"join {len(top['by'])}"})
        if log:
            log(f"growth: joined piece {Q} onto piece {P} via {', '.join(f'{x}–{y}' for x, y in top['by'])}"
                f" (score {top['score']:.2f}, appearance {top.get('ncc')})")

    # every pair hypothesis between placed scans that agrees with the model is a link too
    have = {frozenset((c["a"], c["b"])) for c in chosen}
    for (a, b), hs in edges.items():
        if a in W and b in W and piece[a] == piece[b] and frozenset((a, b)) not in have:
            implied = invert(W[b]) @ W[a]
            agree = [h for h in hs if _pose_close(np.asarray(h["transform"]), implied, trans=AGREE_TRANS, rot_deg=AGREE_ROT)]
            if agree:
                chosen.append({"a": a, "b": b, "hyp": max(agree, key=weight), "votes": "loop"})
            elif hs:
                h = max(hs, key=weight)
                d = invert(np.asarray(h["transform"])) @ implied
                rejected.append({"a": a, "b": b, "hyp": h, "reason": (
                    f"contradicts the model: off by {float(np.linalg.norm(d[:3, 3])):.2f} m / {rotation_angle_deg(d[:3, :3]):.0f}°")})
    poses = {i: W.get(i, np.eye(4)) for i in ids}
    # confidence: placed with >= 2 agreeing scans or closing loops = confirmed
    deg: dict[str, int] = {i: 0 for i in ids}
    for c in chosen:
        deg[c["a"]] += 1; deg[c["b"]] += 1
    confidence = {i: ("unlinked" if i not in W or deg[i] == 0 else "check storey" if i in storey_flag
                      else "confirmed" if deg[i] >= 2 else "single link") for i in ids}
    return {"chosen": chosen, "rejected": rejected, "poses": poses, "confidence": confidence}


def select_links(ids: list[str], edges: dict[tuple[str, str], list[dict[str, Any]]], accept: float = 0.2,
                 origins: dict[str, np.ndarray] | None = None,
                 setups: dict[str, str] | None = None,
                 world_check: Callable[[list[tuple[str, str, np.ndarray]]], list[tuple[str, str, float, float]]] | None = None,
                 ) -> dict[str, Any]:
    """Choose links from plausible hypotheses by agreement.

    ``edges[(a, b)]`` holds plausible hypotheses (a→b transforms with a
    ``rank``; manual ones carry ``manual=True``).  Components are merged one
    at a time, always the merge supported by the most agreeing pairs (then
    the highest total rank).  A merge resting on a single pair is only made
    when that pair has one clear winner and passes the strict single-link
    test; otherwise the pair waits, and is accepted later only if other links
    bring its scans together *and* it agrees with them (loop closure).
    """
    comp = {i: i for i in ids}
    members = {i: [i] for i in ids}
    pose = {i: np.eye(4) for i in ids}
    org = {i: np.asarray((origins or {}).get(i, np.zeros(3)), dtype=np.float64) for i in ids}

    def station(i: str) -> np.ndarray:
        return pose[i][:3, :3] @ org[i] + pose[i][:3, 3]

    def station_ids(scans: set[str]) -> dict[str, int]:
        # scans from (almost) one spot share one viewpoint, and so the same
        # symmetric ambiguity: they are one witness, not several
        reps: list[np.ndarray] = []
        out: dict[str, int] = {}
        for x in sorted(scans):
            p = station(x)
            for n, r in enumerate(reps):
                if np.linalg.norm(p - r) <= STATION_SEP:
                    out[x] = n
                    break
            else:
                out[x] = len(reps)
                reps.append(p)
        return out
    setup = {i: (setups or {}).get(i, i) for i in ids}

    def colocated(ci: str, cj: str, X: np.ndarray) -> bool:
        """Would merging cj into ci (by X) put two separate setups on one spot?"""
        if setups is None:
            return False
        a_pts = [(x, station(x)) for x in members[ci]]
        for y in members[cj]:
            q = X[:3, :3] @ station(y) + X[:3, 3]
            for x, p in a_pts:
                if setup[x] != setup[y] and float(np.linalg.norm(p - q)) < COLOC_SEP:
                    return True
        return False

    pending = {k: v for k, v in edges.items() if k[0] in comp and k[1] in comp and v}
    chosen: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    deferred: dict[tuple[str, str], str] = {}

    # Manual placements always win: they are applied first, newest first, and
    # no automatic evidence can outvote them.  A pair with a manual link uses
    # only that link.  An older manual link that contradicts newer ones is
    # reported, not applied.
    manual = sorted(
        ((k, h) for k, hs in pending.items() for h in hs if h.get("manual")),
        key=lambda kh: -float(kh[1].get("manual_created") or 0.0),
    )
    for k, h in manual:
        a, b = k
        T = np.asarray(h["transform"])
        if comp[a] != comp[b]:
            ci, cj = comp[a], comp[b]
            X = pose[a] @ invert(T) @ invert(pose[b])   # puts b where the manual link says
            for m in members[cj]:
                pose[m] = X @ pose[m]
                comp[m] = ci
            members[ci].extend(members.pop(cj))
            chosen.append({"a": a, "b": b, "hyp": h, "votes": "manual"})
        else:
            implied = invert(pose[b]) @ pose[a]
            if _pose_close(T, implied, trans=AGREE_TRANS, rot_deg=AGREE_ROT):
                chosen.append({"a": a, "b": b, "hyp": h, "votes": "manual"})
            else:
                d = invert(T) @ implied
                rejected.append({"a": a, "b": b, "hyp": h, "reason": (
                    f"older manual placement contradicts newer ones: off by {float(np.linalg.norm(d[:3, 3])):.2f} m / "
                    f"{rotation_angle_deg(d[:3, :3]):.0f}°")})
    manual_keys = {k for k, _h in manual}
    pending = {k: v for k, v in pending.items() if k not in manual_keys}
    # automatic hypotheses between scans already joined by manual links can
    # only confirm them (as loop closures, below), never move them
    for k in pending:
        if comp[k[0]] == comp[k[1]]:
            deferred[k] = "already placed by manual links"

    def weight(h: dict[str, Any]) -> float:
        return MANUAL_WEIGHT if h.get("manual") else float(h.get("rank") or 0.0)

    # Triangle support: how many third scans c confirm a hypothesis a→b
    # (some a→c and c→b compose to the same pose).  At the start every scan
    # stands alone, so every merge has one vote; without this the highest
    # ranked hypothesis wins even when it is a 180° turn that nothing else
    # agrees with.
    nbrs: dict[str, set[str]] = {i: set() for i in ids}
    for (a, b) in pending:
        nbrs[a].add(b)
        nbrs[b].add(a)

    def trans(x: str, y: str) -> list[np.ndarray]:
        if (x, y) in pending:
            return [np.asarray(h["transform"]) for h in pending[(x, y)]]
        if (y, x) in pending:
            return [invert(np.asarray(h["transform"])) for h in pending[(y, x)]]
        return []

    tri: dict[int, int] = {}
    for (a, b), hs in pending.items():
        for h in hs:
            T = np.asarray(h["transform"])
            n = 0
            for c in nbrs[a] & nbrs[b]:
                for T1 in trans(a, c):
                    # c must stand somewhere else to be an independent witness
                    oc_in_a = invert(T1)[:3, :3] @ org[c] + invert(T1)[:3, 3]
                    oc_in_b = T[:3, :3] @ oc_in_a + T[:3, 3]
                    ob = org[b]
                    if np.linalg.norm(oc_in_a - org[a]) <= STATION_SEP or np.linalg.norm(oc_in_b - ob) <= STATION_SEP:
                        continue
                    if any(_pose_close(T2 @ T1, T, trans=AGREE_TRANS * 1.5, rot_deg=AGREE_ROT * 1.5)
                           for T2 in trans(c, b)):
                        n += 1
                        break
            tri[id(h)] = n

    def implied_X(a: str, b: str, T: np.ndarray, ci: str) -> np.ndarray:
        # rigid transform mapping the other component's frame into ci's frame
        if comp[a] == ci:
            return pose[a] @ invert(T) @ invert(pose[b])
        return pose[b] @ T @ invert(pose[a])

    while True:
        by_comp: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for (a, b) in pending:
            if (a, b) in deferred:
                continue
            ca, cb = comp[a], comp[b]
            if ca != cb:
                by_comp.setdefault((min(ca, cb), max(ca, cb)), []).append((a, b))
        if not by_comp:
            break
        best = None
        options: dict[tuple[str, str], list[tuple[tuple, np.ndarray, dict]]] = {}
        for (ci, cj), keys in by_comp.items():
            cands = []
            for k in keys:
                for h in pending[k]:
                    cands.append((implied_X(k[0], k[1], np.asarray(h["transform"]), ci), weight(h), k, h))
            for X, _w, _k, _h in cands:
                support: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
                for X2, w2, k2, h2 in cands:
                    if _pose_close(X, X2, trans=AGREE_TRANS, rot_deg=AGREE_ROT):
                        if k2 not in support or w2 > support[k2][0]:
                            support[k2] = (w2, h2)
                si = station_ids({x for k2 in support for x in k2 if comp[x] == ci})
                sj = station_ids({x for k2 in support for x in k2 if comp[x] == cj})
                indep = len({(si[x], sj[y]) if comp[x] == ci else (si[y], sj[x]) for (x, y) in support})
                manual = any(v[1].get("manual") for v in support.values())
                if manual:
                    indep = max(indep, 2)
                co = not manual and colocated(ci, cj, X)
                if co:
                    indep = 1   # repeated geometry agrees with itself: one witness at most
                key = (not co, indep, sum(tri.get(id(v[1]), 0) for v in support.values()),
                       sum(v[0] for v in support.values()))
                options.setdefault((ci, cj), []).append((key, X, support))
                if best is None or key > best[0]:
                    best = (key, ci, cj, X, support, keys)
        if world_check is not None and best[0][0]:
            picked = _world_pick(best, options[(best[1], best[2])], members, pose, station, world_check)
            if picked is None:
                continue
            if isinstance(picked, str):
                for o in options[(best[1], best[2])][:WORLD_OPTIONS * 3]:
                    for k in o[2]:
                        deferred[k] = picked
                continue
            best = picked
        (_not_co, votes, _tri, _wsum), ci, cj, X, support, keys = best
        if not _not_co:
            # only co-location merges are left: they never build the model,
            # they can only close a loop once other links agree (see below)
            for k in support:
                deferred[k] = "co-located: puts two separate setups on one spot (repeated geometry?) — waits for another link"
            continue
        if votes == 1:
            k, (w, h) = max(support.items(), key=lambda kv: kv[1][0])
            if not h.get("manual"):
                others = [o for o in pending[k] if o is not h
                          and not _pose_close(np.asarray(o["transform"]), np.asarray(h["transform"]), trans=AGREE_TRANS, rot_deg=AGREE_ROT)]
                close_rival = max((weight(o) for o in others), default=-1.0)
                ok, why = strong_link(h, accept)
                if tri.get(id(h), 0) > 0 and _not_co:
                    ok, why = True, None  # confirmed by a third scan
                if close_rival >= w - AMBIGUOUS and not (tri.get(id(h), 0) > 0 and not any(
                        tri.get(id(o), 0) for o in others)):
                    deferred[k] = f"ambiguous: {len(others) + 1} plausible poses, no other link to decide"
                    continue
                if not ok:
                    deferred[k] = f"single link, evidence too weak: {why}"
                    continue
        for m in members[cj]:
            pose[m] = X @ pose[m]
            comp[m] = ci
        members[ci].extend(members.pop(cj))
        for k, (_w, h) in support.items():
            chosen.append({"a": k[0], "b": k[1], "hyp": h, "votes": votes})
        for k in keys:
            if k in support:
                continue
            h = max(pending[k], key=weight)
            d = invert(np.asarray(h["transform"])) @ (invert(pose[k[1]]) @ pose[k[0]])
            rejected.append({"a": k[0], "b": k[1], "hyp": h, "reason": (
                f"contradicts {votes} agreeing link(s): off by {float(np.linalg.norm(d[:3, 3])):.2f} m / "
                f"{rotation_angle_deg(d[:3, :3]):.0f}°")})
        pending = {k: v for k, v in pending.items() if comp[k[0]] != comp[k[1]] or k in deferred}

    # deferred pairs: accept as loop closures if their scans ended up together
    # and one of their poses agrees with the solution; otherwise report why.
    for k, why in deferred.items():
        a, b = k
        hyps = pending.get(k) or edges.get(k) or []
        if comp[a] == comp[b]:
            implied = invert(pose[b]) @ pose[a]
            agree = [h for h in hyps if _pose_close(np.asarray(h["transform"]), implied, trans=AGREE_TRANS, rot_deg=AGREE_ROT)]
            if agree:
                chosen.append({"a": a, "b": b, "hyp": max(agree, key=weight), "votes": "loop"})
                continue
            h = max(hyps, key=weight)
            d = invert(np.asarray(h["transform"])) @ implied
            why = (f"contradicts its group: off by {float(np.linalg.norm(d[:3, 3])):.2f} m / "
                   f"{rotation_angle_deg(d[:3, :3]):.0f}°")
        if hyps:
            rejected.append({"a": a, "b": b, "hyp": max(hyps, key=weight), "reason": why})

    # Confidence per scan, judged on viewpoints (scanner positions), not
    # scans: co-located scans can only confirm each other's relative pose,
    # never a placement relative to the rest of the building.
    st_of: dict[str, str] = {}
    for c_id, ms in members.items():
        for x, n in station_ids(set(ms)).items():
            st_of[x] = f"{c_id}#{n}"
    st_edges = {}
    for c in chosen:
        sa, sb = st_of[c["a"]], st_of[c["b"]]
        if sa != sb:
            k = tuple(sorted((sa, sb)))
            st_edges[k] = (sa, sb, f"{k[0]}|{k[1]}")
    st_nodes = sorted(set(st_of.values()))
    st_bridges = _bridges_of(st_nodes, list(st_edges.values()))
    linked = {x for c in chosen for x in (c["a"], c["b"])}
    conf = {}
    for i in ids:
        mine = [e for e in st_edges.values() if st_of[i] in (e[0], e[1])]
        if i not in linked:
            conf[i] = "unlinked"
        elif any(e[2] not in st_bridges for e in mine):
            conf[i] = "confirmed"
        else:
            conf[i] = "single link"
    return {"poses": pose, "chosen": chosen, "rejected": rejected, "confidence": conf}


# --------------------------------------------------------------------------- #
# pipeline bound to a session
# --------------------------------------------------------------------------- #


class Pipeline:
    def __init__(self, session: "Session") -> None:
        self.s = session

    # ---------------------------------------------------------------- basics

    def sequence_index(self, ids: list[str] | None = None) -> dict[str, int]:
        """Capture order of the scans: natural sort of their names, so
        scan_017 < scan_018 and scan2 < scan10."""
        import re

        s = self.s
        ids = list(s.scans) if ids is None else ids

        def key(i: str):
            sc = s.scans[i]
            name = str(getattr(sc.ref, "name", "") or i)
            parts = re.split(r"(\d+)", f"{Path(str(sc.ref.file)).stem}/{name}/{sc.ref.index:04d}")
            return [int(p) if p.isdigit() else p.lower() for p in parts]

        return {i: k for k, i in enumerate(sorted(ids, key=key))}

    def by_sequence(self, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Neighbours in capture order first (10-11, 11-12, ...), then gaps of
        2, 3, ...: the scanner moves progressively, so the closest captures
        are the most likely to overlap and links appear as a growing chain."""
        order = self.sequence_index(sorted({x for p in pairs for x in p}))
        return sorted(pairs, key=lambda p: (abs(order[p[0]] - order[p[1]]), min(order[p[0]], order[p[1]])))

    def sig(self) -> str:
        return f"{self.s._params_sig()}|{PIPELINE_VERSION}"

    def ids(self) -> list[str]:
        return [i for i, sc in self.s.scans.items() if sc.loaded and sc.included and sc.reg_path]

    def _origin(self, sid: str) -> np.ndarray:
        sc = self.s.scans[sid]
        if sc.apply_pose and getattr(sc.ref, "has_pose", False):
            return np.asarray(sc.ref.pose, dtype=np.float64)[:3, 3].copy()
        return np.zeros(3)

    def job(self, a: str, b: str, *, init: np.ndarray | None = None, path_b: str | None = None,
            source: str = "") -> PairJob:
        s = self.s
        sa = s.scans[a]
        j = PairJob(
            a, b, sa.reg_path, path_b or s.scans[b].reg_path,
            np.eye(4) if init is None else np.asarray(init, dtype=np.float64),
            float(s.params["voxel"]), float(s.params["accept_score"]), int(s.params.get("max_attempts", 3)),
        )
        j.a_key = s._scan_key(a)
        j.b_key = s._scan_key(b) if b in s.scans else b
        j.origin_a = self._origin(a)
        j.origin_b = self._origin(b) if b in s.scans else np.zeros(3)
        j.source = source
        return j

    def run(self, jobs: list[PairJob], worker: Callable[[PairJob], dict[str, Any]], stage: str | None,
            progress: Callable, lo: float, hi: float, label: str, *, force: bool = False,
            on_interval: Callable[[], None] | None = None, interval: float = 20.0,
            stage_label: str | None = None) -> list[dict[str, Any]]:
        """Run jobs in a process pool; cached per (pair, settings, stage).

        A dead worker (out of memory, native crash) restarts the pool and
        retries what was in flight once; a pair that kills a worker twice is
        skipped.  Stop is honoured between results.
        """
        s = self.s
        sig = self.sig()
        blocked = s.blocked_pairs()
        jobs = [j for j in jobs if frozenset((j.a, j.b)) not in blocked]
        out: list[dict[str, Any]] = []
        todo: list[PairJob] = []
        if stage and not force:
            cached = DB.get_hyps([(j.a_key, j.b_key) for j in jobs], sig)
            for j in jobs:
                hit = cached.get((j.a_key, j.b_key, stage))
                if hit is not None:
                    hit = dict(hit)
                    hit["a"], hit["b"] = j.a, j.b
                    out.append(hit)
                else:
                    todo.append(j)
        else:
            todo = list(jobs)
        total = max(1, len(jobs))
        done = len(out)
        stage_name = stage_label or label
        progress(lo + (hi - lo) * done / total, f"{label}: {done} of {len(jobs)} cached" if done else label,
                 stage=stage_name, done=done, total=len(jobs))
        if done:
            if on_interval is not None:
                on_interval()  # show what is already known right away
        if not todo:
            return out
        workers, threads = pool_shape(int(s.params["pair_workers"]))
        workers = min(workers, len(todo))
        env = _single_threaded_children()
        pool = process_pool(workers, initializer=_worker_init, initargs=(threads,))
        s._pools.add(pool)
        queue = iter(todo)
        last_tick = time.time()
        futures: dict[Any, PairJob] = {}
        retries: dict[int, int] = {}
        cancelled = False

        def submit(j: PairJob) -> None:
            futures[pool.submit(worker, j)] = j

        for _ in range(workers * 2):
            j = next(queue, None)
            if j is None:
                break
            submit(j)
        try:
            while futures:
                finished, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                broken = False
                for fut in finished:
                    j = futures.pop(fut)
                    try:
                        res = fut.result()
                    except BrokenProcessPool:
                        inflight = [j] + list(futures.values())
                        futures.clear()
                        s._terminate_pool(pool)
                        s._pools.discard(pool)
                        pool = process_pool(workers, initializer=_worker_init, initargs=(threads,))
                        s._pools.add(pool)
                        print(f"[pipeline] a worker died; retrying {len(inflight)} pair(s)", flush=True)
                        for jj in inflight:
                            n = retries.get(id(jj), 0)
                            if n >= 1:
                                done += 1
                                continue
                            retries[id(jj)] = n + 1
                            submit(jj)
                        broken = True
                        break
                    except Exception:
                        traceback.print_exc()
                        res = None
                    if res is not None:
                        out.append(res)
                        if stage and j.a_key and j.b_key:
                            store = {k: v for k, v in res.items() if k not in ("a", "b")}
                            DB.put_hyps(j.a_key, j.b_key, sig, stage, store)
                    done += 1
                    progress(lo + (hi - lo) * done / total, f"{s._label(j.a)} ↔ {s._label(j.b)}",
                             stage=stage_name, done=done, total=len(jobs))
                    if s._cancelled(s._busy_job or ""):
                        cancelled = True
                        break
                    nxt = next(queue, None)
                    if nxt is not None:
                        submit(nxt)
                if cancelled:
                    break
                if on_interval is not None and time.time() - last_tick >= interval:
                    last_tick = time.time()
                    try:
                        on_interval()
                    except Exception:
                        traceback.print_exc()
                if broken:
                    continue
        finally:
            if cancelled:
                s._terminate_pool(pool)
            else:
                pool.shutdown(wait=True)
            s._pools.discard(pool)
            _restore_env(env)
        return out

    # ------------------------------------------------------------ hypotheses

    def stored(self) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
        """All stage results per scan pair, keyed by our scan ids."""
        s = self.s
        ids = self.ids()
        key_of = {i: s._scan_key(i) for i in ids}
        id_of = {v: k for k, v in key_of.items()}
        pairs = [(key_of[a], key_of[b]) for a in ids for b in ids if a != b]
        raw = DB.get_hyps(pairs, self.sig())
        out: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        for (ka, kb, stage), res in raw.items():
            a, b = id_of[ka], id_of[kb]
            out.setdefault((a, b), {})[stage] = res
        return out

    def plausible_edges(self) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Plausible hypotheses per unordered pair (as a→b with a<b), plus manual links."""
        edges: dict[tuple[str, str], list[dict[str, Any]]] = {}
        blocked = self.s.blocked_pairs()  # banned by the user: no source may link them

        def add(a: str, b: str, h: dict[str, Any]) -> None:
            if frozenset((a, b)) in blocked:
                return
            if a > b:
                a, b = b, a
                h = {**h, "transform": invert(np.asarray(h["transform"])).tolist()}
            edges.setdefault((a, b), []).append(h)

        from .register import hypothesis_verdict

        accept = float(self.s.params["accept_score"])
        # poses the user rejected never come back (other poses of the pair may)
        vetoed: dict[frozenset, list[tuple[str, np.ndarray]]] = {}
        for v in getattr(self.s, "link_vetoes", []):
            vetoed.setdefault(frozenset((v["a"], v["b"])), []).append((v["a"], np.asarray(v["transform"])))

        def is_vetoed(a: str, b: str, h: dict[str, Any]) -> bool:
            T = np.asarray(h["transform"])
            for va, VT in vetoed.get(frozenset((a, b)), []):
                VT_ab = VT if va == a else invert(VT)
                if _pose_close(T, VT_ab, trans=AGREE_TRANS, rot_deg=AGREE_ROT):
                    return True
            return False
        auto = bool(self.s.params.get("auto_links", True))
        for (a, b), stages in (self.stored().items() if auto else ()):
            for res in stages.values():
                for h in res.get("hyps") or []:
                    # thresholds may have changed since this was stored
                    v = hypothesis_verdict(h, accept)
                    if v["plausible"] and not is_vetoed(a, b, h):
                        add(a, b, {**h, **v})
        for p in self.s.pairs.values():
            if p.get("manual") and p.get("accepted"):
                add(p["a"], p["b"], {**p, "manual": True, "plausible": True, "source": "manual",
                                     "manual_created": float(p.get("manual_created") or 0.0)})
        # placements from the permanent manual store (survive every reset)
        if getattr(self.s, "use_manual_store", False):
            try:
                from .manual_store import store

                sid_of = {self.s._scan_key(i): i for i in self.s.scans}
                for row in store().active(sid_of):
                    a, b = sid_of[row["a_key"]], sid_of[row["b_key"]]
                    add(a, b, {"transform": row["transform"], "manual": True, "plausible": True, "accepted": True,
                               "source": f"manual ({row['kind']}, saved)", "manual_created": float(row["created"]),
                               "manual_id": row["id"]})
            except Exception:
                traceback.print_exc()
        # de-duplicate near-identical poses from different stages
        from .register import _distinct

        for k, hs in edges.items():
            man = sorted((h for h in hs if h.get("manual")), key=lambda h: -float(h.get("manual_created") or 0.0))
            if man:
                # the newest manual placement of a pair is the answer for it
                edges[k] = man[:1]
                continue
            edges[k] = _distinct([h for h in hs if not h.get("manual")])
        return edges

    # ------------------------------------------------------------- stage 3

    def ensure_panoramas(self, progress: Callable | None = None, lo: float = 0.0, hi: float = 1.0) -> None:
        """Build the panoramas the appearance check needs (cached on disk)."""
        from . import panorama as P

        s = self.s
        todo = [i for i in self.ids() if not P.is_built(s._pano_folder(i))]
        if not todo:
            return
        workers, _threads = pool_shape(int(s.params["pair_workers"]))
        with process_pool(max(1, min(workers, len(todo)))) as ex:
            futs = [ex.submit(P.build, s.scans[i].ref, bool(s.scans[i].apply_pose), s._pano_folder(i)) for i in todo]
            for k, f in enumerate(futs, 1):
                try:
                    f.result()
                except Exception:
                    traceback.print_exc()
                if progress:
                    progress(lo + (hi - lo) * k / len(futs), f"panoramas {k}/{len(futs)}", stage="Panoramas")

    def appearance_checker(self):
        """Callback (s, p, T_sp) -> NCC of s's panorama re-observed by p, cached."""
        from . import panorama as P
        from .photometric import ncc

        s = self.s
        cache: dict[tuple, float | None] = getattr(self, "_ncc_cache", {})
        self._ncc_cache = cache

        def check(a: str, b: str, T: np.ndarray) -> float | None:
            fa, fb = s._pano_folder(a), s._pano_folder(b)
            if not (P.is_built(fa) and P.is_built(fb)):
                return None
            yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
            key = (a, b, *np.round(T[:3, 3] / 0.05).astype(int).tolist(), round(yaw / 0.5))
            if key not in cache:
                try:
                    cache[key] = ncc(fa, fb, T)[0]
                except Exception:
                    traceback.print_exc()
                    cache[key] = None
            return cache[key]

        return check

    def freespace_checker(self):
        """Callback (a, b, T_ab) -> worst see-through share of the pose, from
        the depth panoramas (photometric.see_through), cached; None = no verdict."""
        from . import panorama as P
        from .photometric import see_through

        s = self.s
        cache: dict[tuple, float | None] = getattr(self, "_fs_cache", {})
        self._fs_cache = cache

        def check(a: str, b: str, T: np.ndarray) -> float | None:
            fa, fb = s._pano_folder(a), s._pano_folder(b)
            if not (P.is_built(fa) and P.is_built(fb)):
                return None
            yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
            key = (a, b, *np.round(T[:3, 3] / 0.05).astype(int).tolist(), round(yaw / 0.5))
            if key not in cache:
                try:
                    cache[key] = see_through(fa, fb, T)[0]
                except Exception:
                    traceback.print_exc()
                    cache[key] = None
            return cache[key]

        return check

    def world_checker(self, origins: dict[str, np.ndarray]):
        """Callback for select_links: which of these scan pairs contradict
        each other at the given relative pose (cached)."""
        from .register import CONTRA_GLOBAL, CONTRA_SLICE, _load_prepared, slice_contradiction

        s = self.s
        voxel = float(s.params["voxel"])
        cache: dict[tuple, tuple[float, float]] = getattr(self, "_contra_cache", {})
        self._contra_cache = cache

        def check(pairs: list[tuple[str, str, np.ndarray]]) -> list[tuple[str, str, float, float]]:
            out = []
            for x, y, T in pairs:
                a = math.degrees(math.atan2(T[1, 0], T[0, 0]))
                key = (x, y, *np.round(T[:3, 3] / 0.05).astype(int).tolist(), round(a / 0.5))
                if key not in cache:
                    pa = _load_prepared(s.scans[x].reg_path, voxel)
                    pb = _load_prepared(s.scans[y].reg_path, voxel)
                    cache[key] = slice_contradiction(pa, pb, T, origins.get(x), origins.get(y))
                g, w = cache[key]
                if w >= CONTRA_SLICE and g >= CONTRA_GLOBAL:
                    out.append((x, y, g, w))
            return out

        return check

    def select_and_apply(self, log: Callable[[str], None] | None = None, world: bool | None = None) -> dict[str, Any]:
        s = self.s
        ids = [i for i in s.scans if s.scans[i].loaded and s.scans[i].included]
        edges = self.plausible_edges()
        origins = {i: self._origin(i) for i in ids}
        setups = {i: str(getattr(s.scans[i].ref, "file", i)) for i in ids}
        # the whole-merge check loads every scan and compares many pairs: only
        # for the final selection of a run, not the live refresh
        final = bool(log) if world is None else world
        if final and SELECTION_MODE == "growth":
            check = self.world_checker(origins)
            free = self.freespace_checker()

            def contradicts(x: str, y: str, T: np.ndarray) -> bool:
                # s7: panorama free-space first (cheap, all heights), then the slice test
                share = free(x, y, T)
                if share is not None and share > FS_MAX:
                    return True
                if abs(float(T[2, 3])) > SLICE_DZ_MAX:
                    # s9: scans storeys apart see each other only through the
                    # stair eye at grazing angles; 10 cm of growth drift flips
                    # the slice test there (benchmark B, scan 024-032: 0.19 -> 0.84)
                    return False
                return bool(check([(x, y, T)]))

            if ORDER_FREE:
                result = self._select_order_free(ids, edges, origins, setups, contradicts, free, log)
            else:
                result = select_growth(ids, edges, origins, setups, contradicts, log, self.appearance_checker(),
                                       self.sequence_index(ids) if STOREY_BY_ORDER else None)
        else:
            world = final and WORLD_CHECK
            result = select_links(ids, edges, float(s.params["accept_score"]), origins, setups,
                                  self.world_checker(origins) if world else None)
        with s._lock:
            manual_keep = {k: p for k, p in s.pairs.items() if p.get("manual") or p.get("user_rejected")}
            s.pairs = {}
            for c in result["chosen"]:
                h = c["hyp"]
                p = {k: v for k, v in h.items() if k not in ("plausible",)}
                p.update({"a": c["a"], "b": c["b"], "accepted": True, "consensus_votes": c["votes"],
                          "method": h.get("source"), "params_sig": s._params_sig()})
                if h.get("manual"):
                    p["manual"] = True
                s._store_pair_unlocked(p)
            # Rejected candidates are not links: only remember the pairs, so
            # Improve links can retry them with the slower matchers.
            s.unresolved_pairs = [[r["a"], r["b"]] for r in result["rejected"]]
            for k, p in manual_keep.items():
                if k not in s.pairs and f"{p['b']}|{p['a']}" not in s.pairs:
                    s.pairs[k] = p
            for sid, P in result["poses"].items():
                s.scans[sid].transform = np.asarray(P, dtype=np.float64)
            for sid in s.scans:
                s.scans[sid].confidence = result["confidence"].get(sid, "unlinked")
            s._solve_and_apply_unlocked(lambda *a, **k: None, preserve=True)
        if log:
            conf = result["confidence"]
            log(
                f"selection: {len(result['chosen'])} link(s), {len(result['rejected'])} rejected · "
                f"{sum(1 for v in conf.values() if v == 'confirmed')} confirmed, "
                f"{sum(1 for v in conf.values() if v == 'single link')} on a single link, "
                f"{sum(1 for v in conf.values() if v == 'unlinked')} unlinked · {s._group_summary()}"
            )
            for r in result["rejected"]:
                log(f"  ✗ {s._label(r['a'])} ↔ {s._label(r['b'])}: {r['reason']}")
        s._notify()
        return result

    def _select_order_free(self, ids, edges, origins, setups, contradicts, free, log) -> dict[str, Any]:
        """s10: growth ranked by physical and colour evidence, no capture order.

        The finished assembly is checked against itself: every scan pair close
        enough to compare, at its assembled pose, must not see through the
        other's surfaces.  A wrong start shows up as dozens of violations (benchmark B
        43-74 against 0), so a failed assembly restarts from the next-best seed.
        """
        app = self.appearance_checker()

        def evidence(a: str, b: str, h: dict[str, Any]) -> float:
            n = app(a, b, np.asarray(h["transform"]))
            st = h.get("see_through")
            return (0.0 if n is None else n) - EVIDENCE_K * (0.1 if st is None else st)

        def check(res: dict[str, Any]) -> tuple[int, float]:
            # see-through violations and mean evidence of nearby pairs at their assembled poses
            W, n, vals = res["poses"], 0, []
            comp = _link_components(ids, res["chosen"])
            for k, a in enumerate(ids):
                for b in ids[k + 1:]:
                    if comp[a] != comp[b]:   # separate pieces have unrelated frames
                        continue
                    T = invert(np.asarray(W[b])) @ np.asarray(W[a])
                    if float(np.linalg.norm(T[:3, 3])) >= VIOLATION_RADIUS:
                        continue
                    share, v = free(a, b, T), app(a, b, T)
                    n += share is not None and share > FS_MAX
                    vals.append((0.0 if v is None else v) - EVIDENCE_K * (0.1 if share is None else share))
            return n, float(np.mean(vals)) if vals else 0.0

        def grow(seed: tuple[str, str] | None, lg) -> dict[str, Any]:
            return select_growth(ids, edges, origins, setups, contradicts, lg, app, None, None, evidence, seed, SINGLE_EV_MIN)

        best = grow(None, log)
        viol, mean_ev = check(best)
        if log:
            log(f"selection: {len(set(_link_components(ids, best['chosen']).values()))} piece(s), {viol} see-through violation(s), evidence {mean_ev:.2f}")
        if viol < RESTART_VIOL or any(h.get("manual") for hs in edges.values() for h in hs):
            return best      # manual placements stay the start
        # next starts: the strongest pairs, each scan used once
        ranked = sorted(((max(evidence(a, b, h) for h in hs), a, b) for (a, b), hs in edges.items() if hs), reverse=True)
        used: set[str] = set()
        tries = 0
        for _v, a, b in ranked:
            if tries >= GROWTH_STARTS:
                break
            if a in used or b in used:
                continue
            used |= {a, b}
            tries += 1
            res = grow((a, b), None)
            v2, e2 = check(res)
            if log:
                log(f"restart from {a}–{b}: {v2} violation(s), evidence {e2:.2f}")
            if v2 < viol and e2 > mean_ev:
                best, viol, mean_ev = res, v2, e2
            if viol < RESTART_VIOL:
                break
        if log and viol >= RESTART_VIOL:
            log(f"warning: best assembly still has {viol} see-through violation(s), check it")
        return best

    def _rescue_pieces(self, result: dict[str, Any], progress: Callable, log: Callable[[str], None],
                       fresh: bool) -> dict[str, Any]:
        """s10: join pieces without capture order: local search (both scans
        cropped near their scanners) on pairs between pieces, least-linked
        scans first.  The bridge of a thin graph lies between two scans with
        few links; floor-plan overlap ranks it badly (benchmark B, scan 032-033: 71st of 112,
        by links 28th).  One link is not enough to stop on: the same search also
        finds wrong poses (benchmark B, scan 026/027/030-033)."""
        s = self.s
        ids = self.ids()
        tried: set[frozenset] = set()
        idle = 0
        while True:
            comp = _link_components(ids, result["chosen"])
            if len(set(comp.values())) == 1 or len(tried) >= RESCUE_PAIRS or idle >= RESCUE_IDLE or s._cancelled(s._busy_job or ""):
                return result
            deg = {i: 0 for i in ids}
            for c in result["chosen"]:
                deg[c["a"]] += 1
                deg[c["b"]] += 1
            todo = [(a, b) for k, a in enumerate(ids) for b in ids[k + 1:]
                    if comp[a] != comp[b] and frozenset((a, b)) not in tried]
            todo.sort(key=lambda p: (deg[p[0]] + deg[p[1]], p))
            batch = todo[:min(RESCUE_BATCH, RESCUE_PAIRS - len(tried))]
            if not batch:
                return result
            tried |= {frozenset(p) for p in batch}
            log(f"{len(set(comp.values()))} pieces → local search on {len(batch)} pairs between them")
            self.run([self.job(a, b) for a, b in batch], local_job, "local", progress, 0.78, 0.80,
                     "pieces: local search", force=fresh, stage_label="Join pieces")
            next_result = self.select_and_apply(log)
            new = _link_components(ids, next_result["chosen"])
            crossing: dict[frozenset, int] = {}
            for c in next_result["chosen"]:
                if comp[c["a"]] != comp[c["b"]]:
                    key = frozenset((comp[c["a"]], comp[c["b"]]))
                    crossing[key] = crossing.get(key, 0) + 1
            result = next_result
            joined = len(set(new.values())) < len(set(comp.values()))
            idle = 0 if joined else idle + 1
            if joined and len(set(new.values())) == 1 and min(crossing.values(), default=0) >= RESCUE_LINKS:
                return result

    # -------------------------------------------------------- stages 1 to 4

    def register(self, progress: Callable, log: Callable[[str], None], *, fresh: bool = False,
                 refine_voxel: float | None = 0.02) -> dict[str, Any]:
        s = self.s
        t0 = time.time()
        ids = self.ids()
        pairs = self.by_sequence([(a, b) for k, a in enumerate(ids) for b in ids[k + 1:]])
        order = self.sequence_index(ids)
        seq = sorted(ids, key=lambda i: order[i])
        log(f"register: {len(ids)} scans, {len(pairs)} pairs in capture order, voxel {float(s.params['voxel']) * 100:.0f} cm")
        refresh = lambda: self.select_and_apply()  # noqa: E731

        def found(a: str, b: str) -> bool:
            k = (a, b) if a < b else (b, a)
            return any(not h.get("manual") for h in self.plausible_edges().get(k, []))

        # 1) the walking route: every scan against the next one, fast matcher
        # (s10: no route, every pair gets the same search; the rescue below is order-free)
        chain = [] if ORDER_FREE else [(seq[k], seq[k + 1]) for k in range(len(seq) - 1)]
        res = self.run([self.job(a, b) for a, b in chain], candidates_job, "candidates",
                       progress, 0.05, 0.20, "neighbours", force=fresh,
                       on_interval=refresh, interval=10.0, stage_label="Neighbours")
        if s._cancelled(s._busy_job or ""):
            return {"cancelled": True}
        refresh()
        # 2) a neighbour pair that did not link breaks the route: try much
        #    harder right away (every heavy matcher), then bridge the gap via
        #    the scans either side (k-1 -> k+1, k -> k+2)
        missing = [p for p in chain if not found(*p)]
        bridges: list[tuple[str, str]] = []
        if missing:
            log("neighbours without a link: " + ", ".join(f"{s._label(a)}–{s._label(b)}" for a, b in missing)
                + " → heavy matchers")
            res += self.run([self.job(a, b) for a, b in missing], heavy_job, "heavy",
                            progress, 0.20, 0.35, "neighbours: heavy matchers", force=fresh,
                            on_interval=refresh, interval=10.0, stage_label="Neighbours: try harder")
            refresh()
            still = [p for p in missing if not found(*p)]
            if still and not s._cancelled(s._busy_job or ""):
                # engine 12: look near the two scanners only
                log("still missing: " + ", ".join(f"{s._label(a)}–{s._label(b)}" for a, b in still)
                    + " → local search near the scanners")
                res += self.run([self.job(a, b) for a, b in still], local_job, "local",
                                progress, 0.35, 0.37, "neighbours: local search", force=fresh,
                                stage_label="Neighbours: try harder")
                refresh()
                still = [p for p in missing if not found(*p)]
            for a, b in still:
                i, j = order[a], order[b]
                for x, y in ((i - 1, j), (i, j + 1)):
                    if 0 <= x < len(seq) and 0 <= y < len(seq) and x != y:
                        bp = (seq[x], seq[y])
                        if bp not in bridges and not found(*bp):
                            bridges.append(bp)
            if bridges and not s._cancelled(s._busy_job or ""):
                log("still missing: " + ", ".join(f"{s._label(a)}–{s._label(b)}" for a, b in still)
                    + f" → bridging via {len(bridges)} pair(s) one scan further")
                res += self.run([self.job(a, b) for a, b in bridges], candidates_job, "candidates",
                                progress, 0.35, 0.40, "bridges", force=fresh, stage_label="Neighbours: try harder")
                hard = [p for p in bridges if not found(*p)]
                if hard:
                    res += self.run([self.job(a, b) for a, b in hard], heavy_job, "heavy",
                                    progress, 0.40, 0.50, "bridges: heavy matchers", force=fresh,
                                    on_interval=refresh, interval=10.0, stage_label="Neighbours: try harder")
                refresh()
            linked = len(chain) - sum(1 for p in chain if not found(*p))
            log(f"route: {linked} of {len(chain)} neighbour pairs linked")
        if s._cancelled(s._busy_job or ""):
            return {"cancelled": True}
        # 3) everything else (loops back to earlier rooms), still in order
        done_pairs = {(a, b) for a, b in chain + bridges} | {(b, a) for a, b in chain + bridges}
        rest = [p for p in pairs if p not in done_pairs]
        # Links appear while candidates are still being computed: the global
        # selection is cheap, so it re-runs on whatever is stored so far.
        res += self.run([self.job(a, b) for a, b in rest], candidates_job, "candidates",
                        progress, 0.50, 0.75, "candidates", force=fresh,
                        on_interval=refresh, interval=10.0, stage_label="Find candidates")
        overlap = sum(1 for r in res if r.get("hyps"))
        plaus = sum(1 for r in res for h in r.get("hyps") or [] if h.get("plausible"))
        log(f"candidates: {overlap} of {len(pairs)} pairs overlap on the floor plan, "
            f"{plaus} plausible pose(s) · {time.time() - t0:.0f} s")
        if s._cancelled(s._busy_job or ""):
            return {"cancelled": True}
        if SELECTION_MODE == "growth":
            progress(0.75, "panoramas for the appearance check", stage="Panoramas")
            self.ensure_panoramas(progress, 0.75, 0.78)
        progress(0.78, "choosing links by agreement", stage="Choose links")
        selected = self.select_and_apply(log)
        if ORDER_FREE and SELECTION_MODE == "growth":
            self._rescue_pieces(selected, progress, log, fresh)
            if s._cancelled(s._busy_job or ""):
                return {"cancelled": True}
        progress(0.80, "multi-view ICP", stage="Refine")
        s._refine_impl(progress, 0.80, 0.90, rounds=2, refresh=False)
        s._solve_and_apply(progress, 0.90, 0.91, preserve=True)
        s._notify()
        log(f"registered in {time.time() - t0:.0f} s: {s._group_summary()}")
        if not s._cancelled(s._busy_job or ""):
            progress(0.91, "loading fine clouds", stage="Polish")
            self.polish(progress, 0.91, 0.99, refine_voxel, log)
        return {"seconds": round(time.time() - t0, 1)}

    def polish(self, progress: Callable, lo: float, hi: float, refine_voxel: float | None,
               log: Callable[[str], None]) -> None:
        s = self.s
        t0 = time.time()
        paths, fine = None, None
        if refine_voxel and float(refine_voxel) < float(s.params["voxel"]) - 1e-6:
            fine = float(refine_voxel)
            members = [i for i in s.scans if s.scans[i].loaded and s.scans[i].included]
            paths = s._fine_reg_paths(progress, members, fine, lo=lo, hi=lo + (hi - lo) * 0.4) or None
        if paths:
            allp = {i: s.scans[i].reg_path for i in s.scans if s.scans[i].reg_path}
            allp.update(paths)
            s._refine_impl(progress, lo + (hi - lo) * 0.4, lo + (hi - lo) * 0.8, rounds=2, refresh=True,
                           paths=allp, voxel=fine)
        else:
            s._refine_impl(progress, lo, lo + (hi - lo) * 0.8, rounds=2, refresh=True)
        s._consensus_impl(log)
        t1 = time.time()
        s._solve_and_apply(progress, lo + (hi - lo) * 0.8, lo + (hi - lo) * 0.85, preserve=True)
        t2 = time.time()
        try:
            s._measure_accuracy_impl(progress, lo + (hi - lo) * 0.85, hi, paths=paths, voxel=fine)
        except Exception:
            traceback.print_exc()
        from .register import POLISH_ICP
        log(f"polish timing ({POLISH_ICP}): fine clouds + refine {t1 - t0:.0f} s, solve {t2 - t1:.0f} s, "
            f"measure {time.time() - t2:.0f} s")
        mm = [float((p.get("accuracy") or {}).get("mean_mm")) for p in s.pairs.values()
              if p.get("accepted") and (p.get("accuracy") or {}).get("mean_mm") is not None]
        if mm:
            log(f"polish{f' at {fine * 100:.0f} cm' if fine else ''}: mean surface error {np.mean(mm):.1f} mm over {len(mm)} link(s)")
        s._notify()

    @staticmethod
    def _fit(pairs: dict) -> dict[str, float] | None:
        mm = [(p.get("accuracy") or {}) for p in pairs.values() if p.get("accepted")]
        mm = [a for a in mm if a.get("mean_mm") is not None]
        if not mm:
            return None
        return {
            "links": len(mm),
            "mean_mm": float(np.mean([a["mean_mm"] for a in mm])),
            "median_mm": float(np.mean([a.get("median_mm", a["mean_mm"]) for a in mm])),
            "worst_mm": float(max(a["mean_mm"] for a in mm)),
        }

    def refine_all(self, progress: Callable, log: Callable[[str], None], *,
                   voxel: float = 0.02, rounds: int = 8) -> dict[str, Any]:
        """Polish every pose against all its linked neighbours at ``voxel``.

        Links are not re-chosen.  The fit is measured on the same fine clouds
        before and after, and the old poses are restored unless it improved.
        """
        import copy

        s = self.s
        t0 = time.time()
        voxel = float(voxel)
        progress(0.0, "loading fine clouds", stage="Load fine clouds",
                 plan=["Load fine clouds", "Measure before", "Polish", "Measure after"])
        members = [i for i in s.scans if s.scans[i].loaded and s.scans[i].included]
        fine = None
        if voxel < float(s.params["voxel"]) - 1e-6:
            fine = s._fine_reg_paths(progress, members, voxel, lo=0.0, hi=0.30) or None
        allp = {i: s.scans[i].reg_path for i in s.scans if s.scans[i].reg_path}
        allp.update(fine or {})
        mvox = voxel if fine else None
        cancelled = lambda: s._cancelled(s._busy_job or "")

        progress(0.30, "measuring the current fit", stage="Measure before")
        s._measure_accuracy_impl(progress, 0.30, 0.40, paths=fine, voxel=mvox)
        before = self._fit(s.pairs)
        saved_poses = {i: st.transform.copy() for i, st in s.scans.items()}
        saved_pairs = copy.deepcopy(s.pairs)

        progress(0.40, "multi-view polish", stage="Polish")
        log(f"refine at {voxel * 100:g} cm: {len(members)} scans, up to {rounds} rounds")

        def on_round(n: int, mm: float, deg: float) -> None:
            log(f"  round {n}: largest move {mm:.2f} mm / {deg:.3f}°")

        s._refine_impl(progress, 0.40, 0.85, rounds=rounds, refresh=True,
                       paths=allp, voxel=voxel, on_round=on_round)
        s._solve_and_apply(progress, 0.85, 0.86, preserve=True)

        def restore(why: str) -> None:
            for i, P in saved_poses.items():
                if i in s.scans:
                    s.scans[i].transform = P
            s.pairs = saved_pairs
            s._layout_groups()
            s._notify()
            log(f"refine: kept the previous poses ({why})")

        if cancelled():
            restore("cancelled")
            return {"refine": {"kept": False, "before": before, "after": None, "reason": "cancelled"}}

        progress(0.86, "measuring the new fit", stage="Measure after")
        s._measure_accuracy_impl(progress, 0.86, 0.99, paths=fine, voxel=mvox)
        after = self._fit(s.pairs)
        kept = True
        reason = ""
        if before and after:
            log(f"refine: mean surface error {before['mean_mm']:.2f} → {after['mean_mm']:.2f} mm, "
                f"worst link {before['worst_mm']:.2f} → {after['worst_mm']:.2f} mm "
                f"over {after['links']} link(s) · {time.time() - t0:.0f} s")
            if after["mean_mm"] > before["mean_mm"] + 0.05:
                kept, reason = False, "the fit got worse"
                restore(reason)
        s._notify()
        return {"refine": {"kept": kept, "before": before, "after": after, "reason": reason,
                           "voxel": voxel, "seconds": round(time.time() - t0, 1)}}

    # -------------------------------------------------------------- stage 5

    def improve(self, progress: Callable, log: Callable[[str], None], *, hours: float = 10.0,
                refine_voxel: float | None = 0.02, autosave: Callable[[], None] | None = None) -> dict[str, Any]:
        """Fire every heavy matcher where it can still help, by priority."""
        s = self.s
        t0 = time.time()
        deadline = t0 + max(0.05, float(hours)) * 3600
        done_heavy: set[tuple[str, str]] = set()
        done_model: set[tuple[str, tuple]] = set()
        rounds = 0
        changed_any = False
        while time.time() < deadline and not s._cancelled(s._busy_job or ""):
            rounds += 1
            stored = self.stored()
            conf = {i: getattr(s.scans[i], "confidence", "unlinked") for i in self.ids()}
            linked = {(p["a"], p["b"]) for p in s.pairs.values() if p.get("accepted")}
            linked |= {(b, a) for a, b in linked}

            def info(a: str, b: str) -> dict[str, Any]:
                st = stored.get((a, b)) or stored.get((b, a)) or {}
                return st.get("candidates") or {}

            def heavy_done(a: str, b: str) -> bool:
                st = stored.get((a, b)) or stored.get((b, a)) or {}
                return "heavy" in st or (a, b) in done_heavy or (b, a) in done_heavy

            ids = self.ids()
            tiers: list[tuple[str, list[tuple[str, str]]]] = []
            unl = [i for i in ids if conf.get(i) == "unlinked"]
            single = [i for i in ids if conf.get(i) == "single link"]
            t1 = []
            for u in unl:
                near = [x for x in ids if x != u and float(info(u, x).get("bev_peak") or 0) >= BEV_GREY]
                t1 += [(u, x) for x in (near or [x for x in ids if x != u])]
            tiers.append(("unlinked scans", t1))
            t2 = []
            for u in single:
                g = s.group_of.get(u)
                for x in ids:
                    if x == u or (u, x) in linked:
                        continue
                    if s.group_of.get(x) == g or float(info(u, x).get("bev_peak") or 0) >= BEV_GREY:
                        t2.append((u, x))
            tiers.append(("scans held by one link", t2))
            t3 = [(a, b) for (a, b) in [(p["a"], p["b"]) for p in s.pairs.values() if not p.get("accepted")]]
            t3 += [(a, b) for a, b in s.unresolved_pairs if a in s.scans and b in s.scans]
            t3 += [(a, b) for (a, b), st in stored.items() if a < b
                   and float((st.get("candidates") or {}).get("bev_peak") or 0) >= BEV_MIN
                   and not any(h.get("plausible") for r in st.values() for h in r.get("hyps") or [])]
            tiers.append(("unresolved pairs", t3))
            t4 = [(a, b) for (a, b), st in stored.items() if a < b and (st.get("candidates") or {}).get("needs_heavy")]
            tiers.append(("non-levelled / weak pairs", t4))

            work = None
            for name, plist in tiers:
                seen = set()
                todo = []
                for a, b in plist:
                    key = (min(a, b), max(a, b))
                    if key in seen or heavy_done(a, b):
                        continue
                    seen.add(key)
                    todo.append(key)
                if todo:
                    work = (name, self.by_sequence(todo))
                    break

            if work is not None:
                name, todo = work
                log(f"improve round {rounds}: {name}: heavy matchers on {len(todo)} pair(s)")
                res = self.run([self.job(a, b) for a, b in todo], heavy_job, "heavy", progress, 0.05, 0.85,
                               f"heavy ({name})", stage_label=f"Improve {rounds}: {name}")
                for a, b in todo:
                    done_heavy.add((a, b))
                new_plaus = sum(1 for r in res for h in r.get("hyps") or [] if h.get("plausible"))
                log(f"  {new_plaus} plausible pose(s) from {len(res)} pair(s)")
            else:
                # last resort for unlinked scans: match against merged groups
                models = s._build_group_models(progress, 0.05, 0.2)
                unl = [i for i in ids if conf.get(i) == "unlinked"]
                mjobs = []
                for u in unl:
                    for gid, m in models.items():
                        key = (u, tuple(sorted(m["members"])))
                        if key in done_model:
                            continue
                        done_model.add(key)
                        mjobs.append(self.job(u, f"m{gid}", path_b=m["path"], source=f"group {chr(65 + gid)}"))
                if not mjobs:
                    log("improve: nothing left to try")
                    break
                log(f"improve round {rounds}: merged-group matching for {len(unl)} unlinked scan(s)")
                mres = self.run(mjobs, register_pair_job, None, progress, 0.2, 0.7, "scan → group",
                                stage_label=f"Improve {rounds}: match against groups")
                ejobs = []
                for r, j in zip(mres, mjobs):
                    T = np.asarray(r.get("transform") or np.eye(4))
                    gid = int(j.b[1:])
                    m = models[gid]
                    for mem in m["members"][:4]:
                        init = invert(s.scans[mem].transform) @ T
                        ejobs.append(self.job(j.a, mem, init=init, source=f"merged group {chr(65 + gid)}"))
                eres = self.run(ejobs, evaluate_job, None, progress, 0.7, 0.85, "checking group matches")
                for r, j in zip(eres, ejobs):
                    if j.a_key and j.b_key:
                        old = (stored.get((j.a, j.b)) or {}).get("model", {}).get("hyps") or []
                        from .register import _distinct

                        DB.put_hyps(j.a_key, j.b_key, self.sig(), "model",
                                    {"stage": "model", "hyps": _distinct(old + (r.get("hyps") or []))})
            if s._cancelled(s._busy_job or ""):
                break
            before = {k: p.get("transform") for k, p in s.pairs.items() if p.get("accepted")}
            self.select_and_apply(log)
            after = {k: p.get("transform") for k, p in s.pairs.items() if p.get("accepted")}
            if before != after:
                changed_any = True
                s._refine_impl(progress, 0.86, 0.95, rounds=1, refresh=False)
                s._solve_and_apply(progress, 0.95, 0.96, preserve=True)
            s._notify()
            if autosave:
                autosave()
        if changed_any and not s._cancelled(s._busy_job or ""):
            self.polish(progress, 0.96, 0.995, refine_voxel, log)
        log(f"improve finished after {rounds} round(s), {(time.time() - t0) / 60:.1f} min: {s._group_summary()}")
        return {"rounds": rounds, "changed": changed_any}
