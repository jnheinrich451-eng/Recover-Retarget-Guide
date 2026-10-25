"""S3 metric on known answers (brief v2.6 §9.1 self-tests). CPU.  Run:  python tests/test_m3_refresh_depth.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generate.m3 import refresh_depth as RD  # noqa: E402

rng = np.random.default_rng(0)
Hp, Wp, px = 30, 52, 16

# 1. patch median: a plane with a hole; patches under 50 % valid are NaN, the rest hold the median of their valid pixels
D = 5.0 + 0.01 * np.arange(Hp * px)[:, None] + 0.002 * np.arange(Wp * px)[None, :]
D[100:140, 200:300] = np.nan                                                   # a hole: patches 6-8 rows x 12-18 cols partly invalid
P = RD.patch_depth(D, px, Hp, Wp)
assert P.shape == (Hp, Wp) and np.isfinite(P).sum() > 0.9 * Hp * Wp
assert np.isnan(P[6:8, 13:18]).all() and np.isfinite(P[0, 0]) and abs(P[0, 0] - np.median(D[:16, :16])) < 1e-9

# 2. an exact affine change of disparity aligns to AbsRel 0; the coefficients come back
D_ref = np.where(np.isfinite(P), P, np.nan)
D_sig = 1.0 / (0.8 * (1.0 / D_ref) + 0.01)
row = RD.metric_row(D_sig, D_ref)
assert row["absrel_all"] < 1e-9 and abs(row["c"] - 0.8) < 1e-9 and abs(row["d"] - 0.01) < 1e-9 and row["r2"] > 0.999999
assert row["absrel_border_extrap"] < 1e-9

# 3. a 10 % depth scale on half the patches reads about 0.05 after alignment (the fit lands between the halves)
D_half = D_ref.copy(); D_half[:, : Wp // 2] *= 1.10
row3 = RD.metric_row(D_half, D_ref)
assert 0.035 <= row3["absrel_all"] <= 0.065, row3["absrel_all"]

# 4. test_refresh_align_direction: with noise on the noisy map, regressing the reference ON the noisy map attenuates the slope;
#    the §9.1 direction (noisy on reference) recovers it
noise = 1.0 / (1.0 / D_ref + rng.normal(scale=0.02, size=D_ref.shape))
fit_right = RD.align_disparity(noise, D_ref, np.isfinite(noise) & np.isfinite(D_ref))
fit_wrong = RD.align_disparity(D_ref, noise, np.isfinite(noise) & np.isfinite(D_ref))     # the other way round
assert abs(fit_right["c"] - 1.0) < 0.05 and fit_wrong["c"] < fit_right["c"] - 0.05, (fit_right["c"], fit_wrong["c"])

# 5. induced flow: identical depths give 0 error; a 10 % depth error under a sideways translation gives a non-zero, bounded EPE
Kp = np.array([[154.4 / 16, 0, 127.7 / 16 * 832 / 256 / 16 * 16], [0, 273.2 / 16, 127.3 / 16], [0, 0, 1]])
Kp = np.array([[9.65, 0, 26.0], [0, 17.0, 15.0], [0, 0, 1]])
E_b = np.eye(4); E_t = np.eye(4); E_t[0, 3] = 0.3
f0 = RD.induced_flow_error(D_ref, D_ref, Kp, E_b, [E_t])
f1 = RD.induced_flow_error(D_ref * 1.1, D_ref, Kp, E_b, [E_t])
assert f0["epe_median_patch"] < 1e-9 and 0 < f1["epe_median_patch"] < 2.0

# 6. the rule for R
assert RD.refresh_rule([0.95, 0.85, 0.70, 0.55, 0.45, 0.30]) == [0.85, 0.70, 0.55]
assert RD.refresh_rule([0.55, 0.45, 0.30]) == [0.55, 0.45]
assert RD.refresh_rule([0.95, 0.30]) == [] and RD.refresh_rule([0.70]) == [0.70]
print(f"test_m3_refresh_depth: passed (affine exact -> 0; half-scaled 10 % -> {row3['absrel_all']:.3f}; slope right {fit_right['c']:.3f} vs wrong {fit_wrong['c']:.3f}; "
      f"induced EPE 10 % depth: {f1['epe_median_patch']:.3f} patch)")
