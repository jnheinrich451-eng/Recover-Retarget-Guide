"""Module 2 anchor selection v2 — the reviewer's amended rule (a'), 2026-10-17: gates, then ε-lexicographic ranking, then seed.

Pure arithmetic over a table of per-candidate measurements (the scorer's scores.csv plus the heading flag); no model runs here.
Every number it needs is in the lock (`m2.selection`): tau_road, tau_q, eps_iou, eps_clip, the gate list, the order, the tie rule.
A candidate with any metric missing FAILS the "computable" gate with its reason recorded — never ranked last, never rescued.
An empty survivor set returns `selected = None` and the pair is flagged "no anchor" (it counts in the pass rate).

Road agreement convention (scripts/score_m2d_bear_anchor_seeds.py): the NEGATIVE of the worst visible stroke's p95 boundary distance
divided by the image diagonal — always ≤ 0, higher (closer to 0) is better; 0 is a perfect boundary. So tau_road is a negative number
(e.g. −0.02 = the worst stroke within 2 % of the diagonal). The self-test that fixes this reading on data (source frame against itself
must give the maximum, against its mirror a low value) runs on Colab before the rule is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

REQUIRED = ("object_iou", "road_agreement", "subject_similarity", "quality_full")


@dataclass
class Candidate:
    seed: int
    metrics: dict                       # object_iou, road_agreement, subject_similarity, quality_full (+ optional quality_subject)
    object_found: bool = True           # OWLv2 found the class and SAM2 gave one connected mask
    heading_consistent: bool | None = True   # None = not evaluated (gate fails: "not computable")
    reasons: list = field(default_factory=list)


def gates(c: Candidate, lock: dict) -> list[str]:
    """Every failed gate, by name. Empty = survivor."""
    out = []
    if not c.object_found:
        out.append("object_not_found")
    missing = [k for k in REQUIRED if c.metrics.get(k) is None]
    if missing:
        out.append("not_computable:" + ",".join(missing))
    if c.heading_consistent is None:
        out.append("not_computable:heading")
    elif not c.heading_consistent:
        out.append("heading_inconsistent")
    r = c.metrics.get("road_agreement")
    if r is not None and r < float(lock["tau_road"]):
        out.append("road_below_tau")
    q = c.metrics.get("quality_full")
    if q is not None and q < float(lock["tau_q"]):
        out.append("quality_below_tau")
    return out


def select(cands: list[Candidate], lock: dict) -> dict:
    """gates → S; S1 = {IoU ≥ max IoU − eps_iou}; pick = argmax CLIP within S1, ties within eps_clip → lowest seed."""
    order = list(lock.get("order", ["proxy_iou", "clip_identity"]))
    assert order == ["proxy_iou", "clip_identity"], f"only the reviewer's order is implemented: {order}"
    assert str(lock.get("tie", "lowest_seed")) == "lowest_seed"
    rec = {"rule": "m2.selection v2 (gates, eps-lexicographic proxy_iou then clip_identity, lowest seed)", "candidates": [], "selected": None}
    S = []
    for c in sorted(cands, key=lambda x: x.seed):
        f = gates(c, lock)
        rec["candidates"].append({"seed": c.seed, "gates_failed": f, "metrics": dict(c.metrics)})
        if not f:
            S.append(c)
    rec["n_survivors"] = len(S)
    if not S:
        rec["flag"] = "no_anchor"
        return rec
    iou_max = max(c.metrics["object_iou"] for c in S)
    S1 = [c for c in S if c.metrics["object_iou"] >= iou_max - float(lock["eps_iou"])]
    clip_max = max(c.metrics["subject_similarity"] for c in S1)
    S2 = [c for c in S1 if c.metrics["subject_similarity"] >= clip_max - float(lock["eps_clip"])]
    pick = min(S2, key=lambda c: c.seed)
    rec.update({"selected": pick.seed, "S1_seeds": [c.seed for c in S1], "S2_seeds": [c.seed for c in S2], "iou_max": iou_max, "clip_max": clip_max,
                "tie_broken_by_seed": len(S2) > 1})
    return rec
