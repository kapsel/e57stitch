"""Global link selection on synthetic graphs (no point clouds needed)."""

from __future__ import annotations

import unittest

import numpy as np

from e57stitch.pipeline import select_links


def pose(x: float = 0.0, y: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    a = np.radians(yaw_deg)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T[:2, 3] = [x, y]
    return T


# true world poses of four scans in a long room
TRUE = {"a": pose(0, 0), "b": pose(6, 0), "c": pose(12, 0, 3), "d": pose(18, 1, -2)}


def rel(a: str, b: str, err: np.ndarray | None = None) -> np.ndarray:
    """a→b transform (a-local into b-local), optionally corrupted."""
    T = np.linalg.inv(TRUE[b]) @ TRUE[a]
    return T @ err if err is not None else T


def hyp(T: np.ndarray, rank: float = 0.5, **kw) -> dict:
    # evidence that passes the strict single-link test unless overridden
    h = {"transform": T.tolist(), "rank": rank, "score": 0.5, "wall_score": 0.45,
         "accuracy": {"mean_mm": 8.0}, "margin": 1.0}
    h.update(kw)
    return h


def rel_err(result: dict, a: str, b: str) -> float:
    P = result["poses"]
    got = np.linalg.inv(P[b]) @ P[a]
    return float(np.linalg.norm((np.linalg.inv(rel(a, b)) @ got)[:3, 3]))


class SelectLinks(unittest.TestCase):
    def test_majority_beats_strong_wrong_link(self):
        flip = pose(5.3, 0, 180)  # 180° turn + one bay: what hit A4/A7
        edges = {
            ("a", "b"): [hyp(rel("a", "b"))],
            ("a", "c"): [hyp(rel("a", "c"))],
            ("b", "c"): [hyp(rel("b", "c"))],
            ("c", "d"): [hyp(rel("c", "d"), rank=0.6)],
            ("b", "d"): [hyp(rel("b", "d"))],
            # the highest-ranked hypothesis of all, but wrong
            ("a", "d"): [hyp(rel("a", "d", flip), rank=0.9)],
        }
        r = select_links(list(TRUE), edges)
        self.assertTrue(all(r["confidence"][s] == "confirmed" for s in TRUE))
        self.assertLess(rel_err(r, "a", "d"), 0.05)
        bad = [x for x in r["rejected"] if {x["a"], x["b"]} == {"a", "d"}]
        self.assertEqual(len(bad), 1)
        self.assertIn("contradicts", bad[0]["reason"])

    def test_ambiguous_single_link_is_not_trusted(self):
        slide = pose(5.3, 0, 0)
        edges = {("a", "b"): [hyp(rel("a", "b"), rank=0.50), hyp(rel("a", "b", slide), rank=0.49)]}
        r = select_links(["a", "b"], edges)
        self.assertEqual(r["confidence"]["a"], "unlinked")
        self.assertIn("ambiguous", r["rejected"][0]["reason"])

    def test_ambiguity_resolved_by_other_links(self):
        slide = pose(5.3, 0, 0)
        edges = {
            ("a", "b"): [hyp(rel("a", "b"), rank=0.49), hyp(rel("a", "b", slide), rank=0.50)],
            ("a", "c"): [hyp(rel("a", "c"))],
            ("b", "c"): [hyp(rel("b", "c"))],
        }
        r = select_links(["a", "b", "c"], edges)
        self.assertLess(rel_err(r, "a", "b"), 0.05)
        self.assertTrue(all(r["confidence"][s] == "confirmed" for s in "abc"))

    def test_weak_single_link_waits_then_closes_loop(self):
        edges = {
            ("a", "b"): [hyp(rel("a", "b"), accuracy={"mean_mm": 35.0})],  # too weak alone
            ("b", "c"): [hyp(rel("b", "c"))],
            ("a", "c"): [hyp(rel("a", "c"))],
        }
        r = select_links(["a", "b", "c"], edges)
        chosen = {frozenset((c["a"], c["b"])): c for c in r["chosen"]}
        self.assertIn(frozenset("ab"), chosen)
        self.assertTrue(all(r["confidence"][s] == "confirmed" for s in "abc"))

    def test_single_strong_link_is_flagged(self):
        r = select_links(["a", "b"], {("a", "b"): [hyp(rel("a", "b"))]})
        self.assertEqual(r["confidence"]["a"], "single link")
        self.assertLess(rel_err(r, "a", "b"), 0.01)

    def test_manual_link_wins(self):
        wrong = pose(3, 0, 90)
        edges = {("a", "b"): [hyp(rel("a", "b", wrong), rank=0.9),
                              {"transform": rel("a", "b").tolist(), "manual": True}]}
        r = select_links(["a", "b"], edges)
        self.assertLess(rel_err(r, "a", "b"), 0.01)

    def test_colocated_scans_are_one_witness(self):
        # three scans from one spot (like A8, A8 #1, A8 #2) all agree
        # on a pose for x: that is one viewpoint, not three confirmations
        P = {"s1": pose(0, 0), "s2": pose(0.002, 0, 0.1), "s3": pose(0, 0.003, -0.1), "x": pose(9, 2, 30)}
        r_ = lambda a, b: np.linalg.inv(P[b]) @ P[a]
        edges = {
            ("s1", "s2"): [hyp(r_("s1", "s2"))], ("s1", "s3"): [hyp(r_("s1", "s3"))],
            ("s2", "s3"): [hyp(r_("s2", "s3"))],
            ("s1", "x"): [hyp(r_("s1", "x"))], ("s2", "x"): [hyp(r_("s2", "x"))], ("s3", "x"): [hyp(r_("s3", "x"))],
        }
        r = select_links(list(P), edges)
        self.assertEqual(r["confidence"]["x"], "single link")

    def test_stacked_floors_do_not_collapse(self):
        # Two identical floors 3 m apart (benchmark B stairwell): c is right under a,
        # d right under b.  The stair scan s sees both floors.  Repeated
        # geometry offers strong "same floor" poses (c on a's spot, d on b's)
        # that even agree with each other; the true floors must win.
        P = {"a": pose(0, 0), "b": pose(6, 0), "s": pose(3, 1), "c": pose(0, 0), "d": pose(6, 0)}
        P["c"][2, 3] = P["d"][2, 3] = -3.0
        P["s"][2, 3] = -1.5
        r_ = lambda x, y: np.linalg.inv(P[y]) @ P[x]
        collapse_ac = np.linalg.inv(P["a"]) @ P["a"]       # c claimed at a's spot
        collapse_bd = np.linalg.inv(P["b"]) @ P["b"]
        edges = {
            ("a", "b"): [hyp(r_("a", "b"))],
            ("c", "d"): [hyp(r_("c", "d"))],
            ("a", "s"): [hyp(r_("a", "s"), rank=0.45)],
            ("b", "s"): [hyp(r_("b", "s"), rank=0.45)],
            ("c", "s"): [hyp(r_("c", "s"), rank=0.45)],
            ("d", "s"): [hyp(r_("d", "s"), rank=0.45)],
            ("a", "c"): [hyp(collapse_ac, rank=0.9)],
            ("b", "d"): [hyp(collapse_bd, rank=0.9)],
            ("a", "d"): [hyp(r_("a", "b"), rank=0.8)],      # d on b's spot, seen from a
            ("b", "c"): [hyp(np.linalg.inv(P["a"]) @ P["b"], rank=0.8)],   # c on a's spot, seen from b
        }
        setups = {k: k for k in P}
        origins = {k: np.zeros(3) for k in P}
        r = select_links(list(P), edges, origins=origins, setups=setups)
        got = r["poses"]
        dz = (np.linalg.inv(got["a"]) @ got["c"])[2, 3]
        self.assertAlmostEqual(dz, -3.0, places=2)
        self.assertAlmostEqual((np.linalg.inv(got["b"]) @ got["d"])[2, 3], -3.0, places=2)

    def test_same_file_scans_may_share_a_spot(self):
        # A8, A8 #1, A8 #2: one setup, three scans — co-location is real
        P = {"s1": pose(0, 0), "s2": pose(0.01, 0), "x": pose(5, 1, 20)}
        r_ = lambda a, b: np.linalg.inv(P[b]) @ P[a]
        edges = {("s1", "s2"): [hyp(r_("s1", "s2"))], ("s1", "x"): [hyp(r_("s1", "x"))], ("s2", "x"): [hyp(r_("s2", "x"))]}
        r = select_links(list(P), edges, origins={k: np.zeros(3) for k in P}, setups={"s1": "f", "s2": "f", "x": "g"})
        self.assertEqual(len({frozenset((c["a"], c["b"])) for c in r["chosen"]}), 3)

    def test_manual_beats_any_number_of_agreeing_auto_links(self):
        # four scans agree on a wrong place for d; one manual link says otherwise
        wrong = pose(4, 0, 180)
        edges = {
            ("a", "b"): [hyp(rel("a", "b"))], ("a", "c"): [hyp(rel("a", "c"))], ("b", "c"): [hyp(rel("b", "c"))],
            ("a", "d"): [hyp(rel("a", "d", wrong), rank=0.9)],
            ("b", "d"): [hyp(rel("b", "d", wrong), rank=0.9)],
            ("c", "d"): [hyp(rel("c", "d", wrong), rank=0.9)],
            ("d", "c"): [{"transform": rel("d", "c").tolist(), "manual": True, "manual_created": 1.0}],
        }
        r = select_links(list(TRUE), edges)
        self.assertLess(rel_err(r, "c", "d"), 0.01)
        self.assertLess(rel_err(r, "a", "d"), 0.05)

    def test_newest_manual_placement_wins(self):
        old = {"transform": rel("a", "b", pose(2, 0, 0)).tolist(), "manual": True, "manual_created": 1.0}
        new = {"transform": rel("a", "b").tolist(), "manual": True, "manual_created": 2.0}
        r = select_links(["a", "b"], {("a", "b"): [old, new]})
        self.assertLess(rel_err(r, "a", "b"), 0.01)


class ManualStoreTest(unittest.TestCase):
    def test_add_retract_never_delete(self):
        import tempfile
        from e57stitch.manual_store import ManualStore

        with tempfile.TemporaryDirectory() as d:
            st = ManualStore(f"{d}/m.sqlite")
            st.add(project="p", a_key="A", b_key="B", transform=np.eye(4), kind="snap")
            st.add(project="p", a_key="B", b_key="C", transform=np.eye(4), kind="markers", markers=[{"t": [1, 2]}])
            self.assertEqual(len(st.active({"A", "B", "C"})), 2)
            self.assertEqual(st.retract("B", "A", "undo"), 1)   # either direction
            self.assertEqual([r["b_key"] for r in st.active({"A", "B", "C"})], ["C"])
            self.assertEqual(st.count(), {"total": 2, "active": 1})
            lines = open(f"{d}/m.jsonl").read().splitlines()
            self.assertEqual([json_op(l) for l in lines], ["add", "add", "retract"])


class BlockedLinkTest(unittest.TestCase):
    """A blocked scan pair gets no link from any source, across any reset."""

    def setUp(self):
        import tempfile
        from types import SimpleNamespace
        from unittest import mock

        from e57stitch.manual_store import ManualStore
        from e57stitch.session import Session

        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.store = ManualStore(f"{d.name}/m.sqlite")
        patcher = mock.patch("e57stitch.manual_store.store", lambda: self.store)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.s = Session()
        self.s.on_change = None
        for i, sid in enumerate(["a", "b", "c"]):
            self.s.scans[sid] = SimpleNamespace(file_hash=f"h{sid}", ref=SimpleNamespace(index=i, file=f"{sid}.e57", label=sid),
                                                transform=np.eye(4), loaded=False)

    def edges(self):
        from e57stitch.pipeline import Pipeline

        pipe = Pipeline(self.s)
        h = hyp(rel("a", "b"), plausible=True)
        pipe.stored = lambda: {("a", "b"): {"heavy": {"hyps": [h]}}, ("b", "c"): {"heavy": {"hyps": [h]}}}
        return pipe.plausible_edges()

    def test_blocked_pair_gets_no_edge_until_unblocked(self):
        self.assertIn(("a", "b"), self.edges())
        self.s.set_link_blocked("b", "a", True)  # either order
        edges = self.edges()
        self.assertNotIn(("a", "b"), edges)
        self.assertIn(("b", "c"), edges)
        self.s.set_link_blocked("a", "b", False)
        self.assertIn(("a", "b"), self.edges())

    def test_block_removes_rows_and_refuses_new_ones(self):
        self.s.pairs["a|b"] = {"a": "a", "b": "b", "accepted": True, "manual": True, "transform": np.eye(4).tolist()}
        self.s.unresolved_pairs = [["b", "a"], ["b", "c"]]
        self.s.set_link_blocked("a", "b", True)
        self.assertEqual(self.s.pairs, {})
        self.assertEqual(self.s.unresolved_pairs, [["b", "c"]])
        self.s._store_pair_unlocked({"a": "b", "b": "a", "accepted": True, "manual": True, "transform": np.eye(4).tolist()})
        self.assertEqual(self.s.pairs, {})

    def test_block_survives_reset_and_restored_rows_are_dropped(self):
        self.s.set_link_blocked("a", "b", True)
        self.s.reset_registration(cleanups=True, vetoes=True)
        self.assertTrue(self.s.is_blocked("a", "b"))
        self.s.pairs["a|b"] = {"a": "a", "b": "b", "accepted": True}
        self.s.drop_blocked_rows()
        self.assertEqual(self.s.pairs, {})

    def test_undo_of_a_block_restores_the_row_exactly(self):
        row = {"a": "a", "b": "b", "accepted": True, "score": 0.7, "transform": np.eye(4).tolist()}
        self.s.pairs["a|b"] = dict(row)
        self.s.set_link_blocked("a", "b", True)
        self.assertEqual(self.s.pairs, {})
        self.assertTrue(np.allclose(self.s.scans["a"].transform, np.eye(4)))  # nothing moved
        self.assertTrue(self.s.set_link_blocked("a", "b", False))
        self.assertEqual(self.s.pairs, {"a|b": row})

    def test_unblock_after_a_move_does_not_restore_the_old_row(self):
        self.s.pairs["a|b"] = {"a": "a", "b": "b", "accepted": True, "transform": np.eye(4).tolist()}
        self.s.set_link_blocked("a", "b", True)
        moved = np.eye(4)
        moved[0, 3] = 2.0
        self.s.scans["b"].transform = moved
        self.assertFalse(self.s.set_link_blocked("a", "b", False))
        self.assertEqual(self.s.pairs, {})

    def test_hand_placement_on_a_blocked_pair_is_refused(self):
        self.s.set_link_blocked("a", "b", True)
        with self.assertRaisesRegex(ValueError, "blocked"):
            self.s.place_exact(["a"], "a", np.eye(4).tolist(), ["b"])
        self.s._assist_last = {"anchor": "a", "pose": np.eye(4).tolist(), "links": [{"a": "a", "b": "b", "accepted": True}]}
        with self.assertRaisesRegex(ValueError, "blocked"):
            self.s.assist_accept(["a"], "a", np.eye(4).tolist())
        self.assertTrue(np.allclose(self.s.scans["a"].transform, np.eye(4)))

    def test_bench_session_ignores_blocks(self):
        self.s.set_link_blocked("a", "b", True)
        self.s.use_manual_store = False
        self.assertIn(("a", "b"), self.edges())


def json_op(line: str) -> str:
    import json
    return json.loads(line)["op"]


if __name__ == "__main__":
    unittest.main()
