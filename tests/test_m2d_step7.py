"""m2_d build step 7 — SKILL_m2_d v1.1 §8 tests 5 (selection determinism), 6 (refusal emits), 7 (hash chain). CPU.

The thresholds used here are TEST values handed in as a dict; the real ones live in calibration.lock.yaml.

Run:  python tests/test_m2d_step7.py
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2d import select                                       # noqa: E402

REPORT = ["# m2_d step 7 — selection: tests 5, 6, 7", ""]
TH = {"m2d_e_loc_rel_max": 0.5, "m2d_e_area_max": float(np.log(2)), "m2d_heading": "human", "m2d_eps_shape_max": None}
H = W = 1024


def cand(seed, e_loc=0.005, e_area=0.05, found=True, A=0.04):
    c = {"seed": seed, "object_found": found, "obj_fidelity": 0.9 if found else 0.0, "bg_fidelity": 0.9, "A_obj_rendered_frac": A}
    if found:
        c.update({"E_loc": e_loc, "E_area": e_area, "log_area_ratio": e_area})
    return c


C = [cand(k) for k in range(4)]
L = {k: "away" for k in range(4)}

# ---- re-normalisation: 16 px on a 12.9 % object vs on a 0.7 % object
r_big, r_small = select.e_loc_rel(16 / np.hypot(H, W), 0.129, H, W), select.e_loc_rel(16 / np.hypot(H, W), 0.007, H, W)
assert abs(r_big - 16 / np.sqrt(0.129 * H * W)) < 1e-12 and r_small > 4 * r_big
REPORT.append(f"E_loc_rel: the same 16 px offset is {r_big:.3f} of a 12.9 % object's scale and {r_small:.3f} of a 0.7 % object's")

# ---- test 5: same candidates, shuffled order -> same winner; ties break to the lowest seed
ranks = {0: 2, 1: 1, 2: 1, 3: 3}                                  # seeds 1 and 2 tie at rank 1
wins = set()
for i in range(20):
    cc = C[:]; random.Random(i).shuffle(cc)
    wins.add(select.select(cc, L, ranks, "away", 0.3, TH, H, W, 4, 8)["selected_seed"])
assert wins == {1}, wins
REPORT.append("test 5 determinism: 20 shuffles -> the same winner; a rank tie between seeds 1 and 2 breaks to seed 1")

# Phase 1 rejects on each condition separately, and Phase 2 never sees a rejected seed even if the human ranked it best
bad = [cand(0, e_loc=0.2), cand(1, e_area=0.8), cand(2), cand(3, found=False)]
q = select.select(bad, {0: "away", 1: "away", 2: "away", 3: "away"}, {0: 1, 1: 1, 2: 2, 3: 1}, "away", 0.3, TH, H, W, 4, 8)
fails = {r["seed"]: [g for g, v in r["gates"].items() if v["outcome"] == "fail"] for r in q["candidates"]}
assert fails == {0: ["g_e_loc_rel"], 1: ["g_e_area"], 2: [], 3: ["g_object_found"]} and q["selected_seed"] == 2 and q["n_survivors"] == 1, fails
assert "E_loc" not in q["candidates"][3]                           # nothing fabricated for an object that was not found
qh = select.select(C, {0: "away", 1: "lateral_left_to_right", 2: "unclear", 3: "away"}, {0: 2, 3: 1}, "away", 0.3, TH, H, W, 4, 8)
assert [r["phase1_pass"] for r in qh["candidates"]] == [True, False, False, True] and qh["selected_seed"] == 3
ex = select.exercised(q["candidates"])
assert ex["g_e_loc_rel"]["exercised"] and ex["g_eps_shape"]["n_ungated"] == 4 and not ex["g_heading"]["exercised"]
assert not any(v["exercised"] for v in select.exercised(select.phase1(C, L, "away", 0.3, TH, H, W)).values())   # the first real run's situation
REPORT.append("phase 1: each condition rejects alone (location, area, object not found, heading label != control class, 'unclear'); a rejected seed "
              "ranked best by the human is never selected; `exercised()` reports which conditions ever fired (none, on an all-pass set)")

# a null threshold is 'ungated'; a set eps_shape ceiling is clip-level and rejects every candidate
assert all(r["gates"]["g_eps_shape"]["outcome"] == "ungated" for r in q["candidates"])
qe = select.select(C, L, ranks, "away", 1.02, {**TH, "m2d_eps_shape_max": 0.5}, H, W, 4, 8)
assert qe["n_survivors"] == 0 and qe["outcome"] == "resample"
# human inputs missing -> AwaitingHuman, never a silent default; a non-human heading mode is refused
for kw, exc in (({"labels": {0: "away"}}, select.AwaitingHuman), ({"ranks": {0: 1}}, select.AwaitingHuman)):
    try:
        select.select(C, kw.get("labels", L), kw.get("ranks", ranks), "away", 0.3, TH, H, W, 4, 8); raise RuntimeError("selected without the human's input")
    except exc:
        pass
try:
    select.select(C, L, ranks, "away", 0.3, {**TH, "m2d_heading": None}, H, W, 4, 8); raise RuntimeError("heading ran without an instrument")
except AssertionError as e:
    assert "human" in str(e)
REPORT.append("human in the loop: missing heading labels or Phase-2 ranks raise AwaitingHuman; `m2d_heading` other than 'human' is refused; "
              "a null threshold reads 'ungated'")

# ---- test 6: zero survivors -> qc_m2d.json written with the outcome, NO anchor.png (a stale one is removed)
with tempfile.TemporaryDirectory() as td:
    out = Path(td); (out / "candidates").mkdir()
    for k in range(4):
        (out / "candidates" / f"seed_{k}.png").write_bytes(b"png%d" % k)
    ok = select.select(C, L, ranks, "away", 0.3, TH, H, W, 4, 8)
    assert select.write_selection(out, ok) is not None and (out / "anchor.png").read_bytes() == b"png1"
    none = [cand(k, e_loc=0.3) for k in range(4)]
    q_res = select.select(none, L, ranks, "away", 0.3, TH, H, W, n_generated=4, budget_total=8)
    q_ref = select.select(none, L, ranks, "away", 0.3, TH, H, W, n_generated=8, budget_total=8)
    assert q_res["outcome"] == "resample" and q_ref["outcome"] == "refused" and q_ref["selected_seed"] is None
    assert select.write_selection(out, q_ref) is None and not (out / "anchor.png").exists()
    j = json.loads((out / "qc_m2d.json").read_text(encoding="utf-8"))
    assert j["schema"] == select.QC_SCHEMA and j["outcome"] == "refused" and j["gate_outcomes"]["g_phase1_survivors"]["outcome"] == "fail"
    assert all((out / "candidates" / f"seed_{k}.png").exists() for k in range(4))          # non-winners are never discarded
REPORT.append("test 6 refusal emits: zero survivors -> 'resample' below the budget, 'refused' at it; qc_m2d.json written, anchor.png absent "
              "(a stale anchor from an earlier selection is removed); all candidates retained")

# ---- test 7: hash chain — corrupt an upstream hash -> hard exit
rec = {"c3": "aa" * 32, "c1": "bb" * 32, "m2a": "cc" * 32, "proxy_cache_key": "5fbdb89abb315954"}
select.check_chain(rec, dict(rec))
for k in rec:
    cur = dict(rec); cur[k] = "00" + cur[k][2:]
    try:
        select.check_chain(rec, cur); raise RuntimeError(f"corrupted {k} accepted")
    except select.HashMismatch as e:
        assert k in str(e)
try:
    select.check_chain(rec, {k: v for k, v in rec.items() if k != "m2a"}); raise RuntimeError("a missing upstream hash was accepted")
except select.HashMismatch:
    pass
REPORT.append("test 7 hash chain: corrupting any one of c3 / c1 / m2a / proxy_cache_key raises HashMismatch naming it; a hash that cannot be re-derived is refused too")

REPORT += ["", "all m2_d step-7 tests passed"]
# automatic Phase 2 (2026-10-05): lowest surviving seed; a seed that fails Phase 1 is never chosen; deterministic; manual path unchanged
_recs = select.phase1(bad, {0: "away", 1: "away", 2: "away", 3: "away"}, "away", 0.3, TH, H, W)
_ar = select.auto_ranks(_recs)
assert _ar == {2: 1} and select.phase2(_recs, _ar) == 2
_all = select.phase1(C, L, "away", 0.3, TH, H, W)
assert select.auto_ranks(_all) == {0: 1, 1: 2, 2: 3, 3: 4} and select.phase2(_all, select.auto_ranks(_all)) == 0
assert select.auto_ranks(list(reversed(_all))) == select.auto_ranks(_all)
REPORT.append("automatic Phase 2: lowest Phase-1 survivor; a failing seed is never selected; order-independent")
Path("docs/report_m2d_step7_tests.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
print("\n".join(REPORT))
