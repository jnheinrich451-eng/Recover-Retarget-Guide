"""m2_d build step 6 — the fidelity statistics on synthetic masks and fields (CPU; the segmenter and the
estimator are real models and are NOT stood in for here — they run on Colab).

Run:  python tests/test_m2d_step6.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2d import fidelity                                     # noqa: E402

REPORT = ["# m2_d step 6 — fidelity statistics: synthetic tests", ""]
H = W = 256
M = np.zeros((H, W), bool); M[100:160, 80:180] = True                    # control object: 60 x 100, centroid (130, 130)
p_ta = [130.0, 130.0]

# identical mask: IoU 1, E_loc 0, E_area 0
s = fidelity.object_stats(M, M, p_ta)
assert s["obj_fidelity"] == 1.0 and s["E_loc"] < 1e-12 and s["E_area"] < 1e-12 and s["E_loc_vs_rendered_centroid"] < 1e-12
# shifted by (dx, dy) = (30, 40): E_loc = 50 / diag; area unchanged; IoU known
G = np.roll(np.roll(M, 40, 0), 30, 1)
s = fidelity.object_stats(G, M, p_ta)
inter = (60 - 40) * (100 - 30); iou = inter / (2 * 6000 - inter)
assert abs(s["E_loc"] - 50.0 / np.hypot(H, W)) < 1e-12 and s["E_area"] < 1e-12 and abs(s["obj_fidelity"] - iou) < 1e-12
# doubled area about the same centre: E_area = log 2, sign kept in log_area_ratio
G2 = np.zeros((H, W), bool); G2[70:190, 80:180] = True
s2 = fidelity.object_stats(G2, M, p_ta)
assert abs(s2["E_area"] - np.log(2)) < 1e-12 and s2["log_area_ratio"] > 0 and s2["E_loc"] < 1e-12 and abs(s2["obj_fidelity"] - 0.5) < 1e-12
half = fidelity.object_stats(M, G2, p_ta)
assert abs(half["log_area_ratio"] + np.log(2)) < 1e-12                   # E_area is symmetric, the signed ratio is not
# p_ta differs from the rendered centroid (delta_p): both distances are reported
sp = fidelity.object_stats(M, M, [133.0, 134.0])
assert abs(sp["E_loc"] - 5.0 / np.hypot(H, W)) < 1e-12 and sp["E_loc_vs_rendered_centroid"] < 1e-12
# object not found: a measurement, not an exception, and no fabricated E_loc / E_area
e = fidelity.object_stats(np.zeros((H, W), bool), M, p_ta)
assert e["object_found"] is False and e["obj_fidelity"] == 0.0 and "E_loc" not in e and "E_area" not in e
REPORT.append(f"object stats: identical -> IoU 1, E_loc 0, E_area 0; shift (30, 40) px -> E_loc 50/diag, IoU {iou:.4f} exact; doubled area -> "
              "E_area log 2 with the sign kept in log_area_ratio; p_ta vs rendered centroid reported separately; an empty segmentation "
              "returns object_found False with NO E_loc / E_area fields")

# background correlation: monotone transform -> +1, inverse -> -1, and the object / sky / generated object are excluded
rng = np.random.default_rng(0)
zc = np.tile(np.linspace(0.05, 1.0, H)[:, None], (1, W)) + 0.01 * rng.standard_normal((H, W))      # near at the bottom
valid = np.ones((H, W), bool); valid[:20] = False                        # sky rows
zh = 3.0 * np.sqrt(zc - zc.min() + 1e-3) + 7.0                           # affine-free, monotone
b = fidelity.bg_fidelity(zh, zc, M, valid)
assert b["bg_fidelity"] > 0.999 and abs(b["bg_domain_frac"] - (valid & ~M).mean()) < 1e-12
assert fidelity.bg_fidelity(-zh, zc, M, valid)["bg_fidelity"] < -0.999
zh_bad = zh.copy(); zh_bad[M] = -1e3; zh_bad[:20] = 1e3                  # garbage INSIDE the object and in the sky must not matter
assert fidelity.bg_fidelity(zh_bad, zc, M, valid)["bg_fidelity"] > 0.999
zh_g = zh.copy(); zh_g[G2] = zh.max() + 5.0                              # a generated object bigger than its outline is near, legitimately
with_g = fidelity.bg_fidelity(zh_g, zc, M, valid, mask_gen=G2)["bg_fidelity"]
without = fidelity.bg_fidelity(zh_g, zc, M, valid)["bg_fidelity"]
assert with_g > 0.999 and without < with_g
try:
    fidelity.bg_fidelity(zh, zc, np.ones((H, W), bool), valid); raise RuntimeError("an empty background domain was accepted")
except AssertionError:
    pass
REPORT.append(f"bg fidelity: monotone transform -> rho {b['bg_fidelity']:+.4f}, inverse -> -1; garbage inside M_obj and in the sky is ignored; "
              f"an oversized generated object counts against the background unless its own mask is excluded ({without:+.3f} vs {with_g:+.3f}); "
              "an empty domain is refused")

REPORT += ["", "all m2_d step-6 tests passed"]
Path("docs/report_m2d_step6_tests.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
print("\n".join(REPORT))
