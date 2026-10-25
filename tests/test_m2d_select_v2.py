"""Known answers for the v2 selection rule (pure arithmetic; no model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2d_select_v2 import Candidate, select  # noqa: E402

LOCK = {"tau_road": -0.05, "tau_q": 40.0, "eps_iou": 0.02, "eps_clip": 0.01, "order": ["proxy_iou", "clip_identity"], "tie": "lowest_seed"}


def C(seed, iou, road, clip, q, **kw):
    return Candidate(seed, {"object_iou": iou, "road_agreement": road, "subject_similarity": clip, "quality_full": q}, **kw)


def main():
    # 1. the highest IoU wins when it is clear of the band, even with a lower CLIP
    r = select([C(0, 0.90, -0.01, 0.30, 70), C(1, 0.95, -0.01, 0.20, 70)], LOCK); assert r["selected"] == 1, r
    # 2. inside the IoU band, CLIP decides
    r = select([C(0, 0.94, -0.01, 0.20, 70), C(1, 0.95, -0.01, 0.30, 70)], LOCK); assert r["selected"] == 1 and r["S1_seeds"] == [0, 1], r
    # 3. inside both bands, the lowest seed
    r = select([C(3, 0.95, -0.01, 0.300, 70), C(2, 0.945, -0.01, 0.295, 70)], LOCK); assert r["selected"] == 2 and r["tie_broken_by_seed"], r
    # 4. gates: road below tau, quality below tau, object not found, heading, missing metric -> each fails with its reason; never ranked
    r = select([C(0, 0.99, -0.10, 0.9, 70), C(1, 0.50, -0.01, 0.1, 70)], LOCK); assert r["selected"] == 1 and "road_below_tau" in r["candidates"][0]["gates_failed"], r
    r = select([C(0, 0.99, -0.01, 0.9, 10), C(1, 0.50, -0.01, 0.1, 70)], LOCK); assert r["selected"] == 1 and "quality_below_tau" in r["candidates"][0]["gates_failed"]
    r = select([C(0, 0.99, -0.01, 0.9, 70, object_found=False), C(1, 0.50, -0.01, 0.1, 70)], LOCK); assert r["selected"] == 1
    r = select([C(0, 0.99, -0.01, 0.9, 70, heading_consistent=False), C(1, 0.50, -0.01, 0.1, 70)], LOCK); assert r["selected"] == 1
    r = select([C(0, 0.99, None, 0.9, 70), C(1, 0.50, -0.01, 0.1, 70)], LOCK); assert r["selected"] == 1 and r["candidates"][0]["gates_failed"][0].startswith("not_computable"), r
    r = select([C(0, 0.99, -0.01, 0.9, 70, heading_consistent=None)], LOCK); assert r["selected"] is None and r["flag"] == "no_anchor", r
    # 5. no survivors -> no anchor, never rescued
    r = select([C(0, 0.99, -0.2, 0.9, 70), C(1, 0.99, -0.3, 0.9, 70)], LOCK); assert r["selected"] is None and r["n_survivors"] == 0
    # 6. the 035 bear scores of record under this rule (scores.csv, crossclip v1): seed 4 unscorable -> fails computable; seed 6 road -0.247 fails tau_road -0.05 -> no anchor
    r = select([C(4, 0.963, None, 0.251, 61.0), C(6, 0.936, -0.247, 0.245, 71.2)], LOCK); assert r["selected"] is None, r
    print("ok")


if __name__ == "__main__":
    main()
