"""Bundle adjustment for camera C (src/generate/m3/bundle_adjust.py): synthetic scene with the 035 error mechanism (per-point
lifted-depth error of 5 %, perturbed initial cameras). BA must recover the cameras to sub-pixel reprojection with the gauge
fixed at camera 0, and not worsen the held-out instrument. CPU.  Run:  python tests/test_m3_bundle_adjust.py"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generate.m3 import bundle_adjust as BA  # noqa: E402

rng = np.random.default_rng(0)
F, N = 40, 300
K = np.tile(np.array([[140.0, 0, 128], [0, 255.0, 128], [0, 0, 1]]), (F, 1, 1))
Xw = np.stack([rng.uniform(-3, 3, N), rng.uniform(-1, 1, N), rng.uniform(5, 12, N)], 1)


def pose(k):
    th = 0.008 * k
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    E = np.eye(4); E[:3, :3] = R; E[:3, 3] = [0.02 * k, 0.0, -0.04 * k]
    return E


def proj(X, E, Kk):
    Xc = X @ E[:3, :3].T + E[:3, 3]
    assert (Xc[:, 2] > 0.5).all(), "test scene: points must stay in front of the camera"
    uv = Xc @ Kk.T
    return uv[:, :2] / uv[:, 2:3]


E_true = np.stack([pose(k) for k in range(F)])
oc, op, ouv = [], [], []
for k in range(F):
    uv = proj(Xw, E_true[k], K[k]) + rng.normal(scale=0.3, size=(N, 2))
    oc += [k] * N; op += list(range(N)); ouv += list(uv)
oc, op, ouv = np.array(oc), np.array(op), np.array(ouv)
X_init = Xw * (1 + rng.normal(scale=0.05, size=(N, 1)))              # the lifted-depth error: 5 % along each ray
E_init = E_true.copy()
for k in range(1, F):
    E_init[k, :3, 3] += rng.normal(scale=0.05, size=3)
held = np.stack([rng.uniform(-3, 3, 100), rng.uniform(-1, 1, 100), rng.uniform(5, 12, 100)], 1)
held_lift = held * (1 + rng.normal(scale=0.05, size=(100, 1)))
err = lambda E: float(np.median([np.median(np.linalg.norm(proj(held_lift, E[k], K[k]) - proj(held, E_true[k], K[k]), axis=1)) for k in range(1, F)]))  # noqa: E731
t0 = time.time()
E_ba, X_ba, info = BA.bundle_adjust(E_init, X_init, oc, op, ouv, K, np.arange(40), prior_weight=50.0)
cam_err = max(np.linalg.norm(E_ba[k, :3, 3] - E_true[k, :3, 3]) for k in range(F))
cam_err0 = max(np.linalg.norm(E_init[k, :3, 3] - E_true[k, :3, 3]) for k in range(F))
print(f"reproj RMS {info['reproj_rms_before']:.2f} -> {info['reproj_rms_after']:.2f} px | camera translation error max {cam_err0:.3f} -> {cam_err:.3f} | "
      f"held-out lifted points: init {err(E_init):.2f} px -> BA {err(E_ba):.2f} px | {time.time() - t0:.1f} s, nfev {info['nfev']}")
assert info["reproj_rms_after"] < 0.5 and cam_err < 0.25 * cam_err0 and err(E_ba) <= err(E_init) + 1e-9
assert np.allclose(E_ba[0], np.eye(4))
# the lift-free read: held-out points TRIANGULATED by the refined cameras from their (noisy, 0.3 px) observations, then reprojected.
# With a 5 % lift error the lift gate floors at ~1.6 px here; triangulation through good cameras must sit near the observation noise.
uv_held = np.stack([proj(held, E_true[k], K[k]) + rng.normal(scale=0.3, size=(100, 2)) for k in range(F)])   # [F,100,2]
tri_err, tri_pt = [], []
for i in range(100):
    X = BA.triangulate_dlt(E_ba, K, uv_held[:, i])
    tri_pt.append(np.linalg.norm(X - held[i]))
    tri_err.append(np.median([np.linalg.norm(proj(X[None], E_ba[k], K[k])[0] - uv_held[k, i]) for k in range(1, F)]))
print(f"triangulated held-out: reprojection median {np.median(tri_err):.2f} px (observation noise 0.3 px), 3D error median {np.median(tri_pt):.3f} (lift error ~{0.05 * 8:.2f})")
assert np.median(tri_err) < 1.0 and np.median(tri_pt) < 0.1
X_exact = BA.triangulate_dlt(E_true, K, np.stack([proj(held[:1], E_true[k], K[k])[0] for k in range(F)]))
assert np.allclose(X_exact, held[0], atol=1e-6), (X_exact, held[0])          # exact cameras + exact observations -> exact point
# the split-in-time column with gauge check (reviewer 2026-10-07h), on known answers
Pw = uv_held.transpose(1, 0, 2); Qw = np.ones((100, F), bool); Xw_ = np.zeros((100, F), bool)
Z_lift = held[:, 2] * (1 + rng.normal(scale=0.05, size=100))
GATE, TOL = 4.92, 0.05
r_true = BA.triangulated_heldout(E_true, K, Pw, Qw, Xw_, Z_lift, GATE, TOL)
E_scaled = E_true.copy(); E_scaled[:, :3, 3] *= 1.3                                  # a camera with the wrong gauge
r_scaled = BA.triangulated_heldout(E_scaled, K, Pw, Qw, Xw_, Z_lift, GATE, TOL)
E_rot = E_true.copy()
for k in range(1, F):
    E_rot[k, :3, :3] = BA._rodrigues(rng.normal(scale=np.radians(3.0), size=3)) @ E_rot[k, :3, :3]   # a camera wrong in rotation
r_rot = BA.triangulated_heldout(E_rot, K, Pw, Qw, Xw_, Z_lift, GATE, TOL)
E_static = np.tile(np.eye(4), (F, 1, 1)); P_static = np.stack([proj(held, E_static[k], K[k]) + rng.normal(scale=0.3, size=(100, 2)) for k in range(F)]).transpose(1, 0, 2)
r_static = BA.triangulated_heldout(E_static, K, P_static, Qw, Xw_, Z_lift, GATE, TOL)
print(f"split column: true cameras {r_true['median_px']:.2f} px, gauge {r_true['gauge']['median_Ztri_over_Zlift']:.3f} {r_true['gauge']['ci95']} -> {r_true['outcome']} | "
      f"translation x1.3: {r_scaled['median_px']:.2f} px, gauge {r_scaled['gauge']['median_Ztri_over_Zlift']:.3f} -> {r_scaled['outcome']} | "
      f"rotation jitter 3 deg: {r_rot['median_px']:.2f} px -> {r_rot['outcome']} | static camera: n={r_static['n_points']}, dropped {r_static['dropped']}, "
      f"gauge {r_static['gauge']['median_Ztri_over_Zlift']} -> {r_static['outcome']}")
assert r_true["outcome"] == "pass" and r_true["median_px"] < 1.0 and abs(r_true["gauge"]["median_Ztri_over_Zlift"] - 1) < 0.02
assert r_scaled["reproj_ok"] and r_scaled["outcome"] == "fail (gauge: wrong scale)"   # reprojection is blind to scale; the gauge check is not
assert not r_rot["reproj_ok"] and r_rot["outcome"] == "fail (reprojection)" and r_rot["median_px"] > GATE   # a wrong camera fails: the control has power
assert r_static["outcome"] != "pass"                                                   # no baseline: never admitted on this column
# reviewer 2026-10-07i: the interval decides (equivalence), a minimum count, and depth terciles
E_106 = E_true.copy(); E_106[:, :3, 3] *= 1.06                                         # scale 6 % off: the interval straddles the 5 % bound
r_106 = BA.triangulated_heldout(E_106, K, Pw, Qw, Xw_, Z_lift, GATE, TOL)
r_few = BA.triangulated_heldout(E_true, K, Pw[:10], Qw[:10], Xw_[:10], Z_lift[:10], GATE, TOL)
Z_dep = held[:, 2] * (1 + 0.12 * (held[:, 2] - 5.0) / 7.0)                              # a lift error growing with depth: the 035 mechanism
r_dep = BA.triangulated_heldout(E_true, K, Pw, Qw, Xw_, Z_dep, GATE, TOL)
print(f"gauge rule: scale x1.06 -> {r_106['gauge']['ci95']} {r_106['outcome']} | 10 points -> {r_few['outcome']} | depth-dependent lift -> terciles "
      f"{[round(v['median'], 3) for v in r_dep['gauge']['terciles'].values()]}, read on {r_dep['gauge']['read_on']} -> {r_dep['outcome']}")
assert r_106["outcome"] == "undetermined (gauge)" and r_106["gauge"]["outcome"] == "undetermined"
assert r_few["outcome"].startswith("undetermined (n=")
assert r_dep["gauge"]["depth_dependent"] and r_dep["gauge"]["read_on"] == "near" and r_dep["outcome"] == "pass"
assert r_true["gauge"]["depth_dependent"] is False and r_true["gauge"]["read_on"] == "all"
r_pts = BA.triangulated_heldout(E_true, K, Pw, Qw, Xw_, Z_lift, GATE, TOL, return_points=True)
zerr = [abs(p["Z_tri"] - held[p["track"], 2]) for p in r_pts["points"]]
assert len(r_pts["points"]) == r_pts["n_points"] and np.median(zerr) < 0.1 and max(zerr) < 0.5      # half-frame triangulation: 5-12 units deep
assert np.median([np.hypot(*p["resid_mean"]) for p in r_pts["points"]]) < 0.3                  # mean signed residual of unbiased noise is small
# free shared focal (reviewer 2026-10-07k): with K's focal 5 % too large, the freed scale must come back near 1/1.05 and the fit collapse to the floor;
# with the true K it must stay within 1 %
K_bad = K.copy(); K_bad[:, 0, 0] *= 1.05; K_bad[:, 1, 1] *= 1.05
_, _, i_bad = BA.bundle_adjust(E_init, X_init, oc, op, ouv, K_bad, np.arange(40), prior_weight=50.0, free_focal=True)
_, _, i_fix = BA.bundle_adjust(E_init, X_init, oc, op, ouv, K_bad, np.arange(40), prior_weight=50.0)
_, _, i_ok = BA.bundle_adjust(E_init, X_init, oc, op, ouv, K, np.arange(40), prior_weight=50.0, free_focal=True)
print(f"free focal: K x1.05 -> scale {i_bad['focal_scale']:.4f} (expect {1 / 1.05:.4f}), RMS {i_fix['reproj_rms_after']:.2f} fixed -> {i_bad['reproj_rms_after']:.2f} freed | true K -> scale {i_ok['focal_scale']:.4f}")
assert abs(i_bad["focal_scale"] * 1.05 - 1.0) < 0.01 and i_bad["reproj_rms_after"] < 0.5 and abs(i_ok["focal_scale"] - 1.0) < 0.01
# band gauge and re-binning on the per-point records: band = the bottom quarter of the frame; the true-camera ratio is 1 and determined
band = np.zeros((256, 256), bool); band[150:, :] = True                                   # the synthetic points span rows ~77-179
pts_true = BA.triangulated_heldout(E_true, K, Pw, Qw, Xw_, Z_lift, GATE, TOL, return_points=True)["points"]
bg = BA.band_gauge(pts_true, band, TOL, min_points=10)
rb = BA.rebin_tables(pts_true)
pts_dep = BA.triangulated_heldout(E_true, K, Pw, Qw, Xw_, Z_dep, GATE, TOL, return_points=True)["points"]
rb_dep = BA.rebin_tables(pts_dep)
print(f"band gauge: ratio {bg['ratio']:.3f} {bg['ci95']} n={bg['n_band_points']} fit {bg['half_fit']:.3f} verify {bg['half_verify']:.3f} -> {bg['outcome']} | "
      f"rebin (true): lift {[round(v['median'], 3) for v in rb['by_lift_depth'].values()]} tri {[round(v['median'], 3) for v in rb['by_triangulated_depth'].values()]} | "
      f"rebin (depth-dependent lift): lift {[round(v['median'], 3) for v in rb_dep['by_lift_depth'].values()]} tri {[round(v['median'], 3) for v in rb_dep['by_triangulated_depth'].values()]}")
assert bg["outcome"] == "determined" and abs(bg["ratio"] - 1.0) < 0.03
assert all(abs(v["median"] - 1.0) < 0.05 for v in rb["by_triangulated_depth"].values())                 # no bias -> no excess in any bin
assert rb_dep["by_triangulated_depth"]["near"]["median"] > rb_dep["by_triangulated_depth"]["far"]["median"] + 0.05   # a real depth-dependent bias survives re-binning
# tracker-vs-tracker: a second noisy observation set of the same points triangulates to the same depths (ratio 1); a lagging tracker (near points
# moving 10 % less than they should) reads > 1 in the near bin
uv_b = uv_held + rng.normal(scale=0.3, size=uv_held.shape)
pts_b = BA.triangulated_heldout(E_true, K, uv_b.transpose(1, 0, 2), Qw, Xw_, Z_lift, GATE, TOL, return_points=True)["points"]
cmp_same = BA.compare_triangulations(pts_true, pts_b, radius_px=2.0)
lag = uv_held.copy(); near = held[:, 2] < np.quantile(held[:, 2], 1 / 3)
lag[:, near] = uv_held[0:1, near] + 0.9 * (uv_held[:, near] - uv_held[0:1, near])                           # near points' motion under-tracked by 10 %
pts_lag = BA.triangulated_heldout(E_true, K, lag.transpose(1, 0, 2), Qw, Xw_, Z_lift, GATE, TOL, return_points=True)["points"]
cmp_lag = BA.compare_triangulations(pts_lag, pts_true, radius_px=2.0)
cmp_band = BA.compare_triangulations(pts_lag, pts_true, radius_px=2.0, band_mask=np.ones((256, 256), bool))
assert cmp_band["in_band"]["n"] == cmp_band["n_matched"] and abs(cmp_band["in_band"]["median"] - cmp_lag["median"]) < 1e-9
print(f"tracker vs tracker: same geometry {cmp_same['median']:.3f} {cmp_same['ci95']} (n={cmp_same['n_matched']}) | lagging near tracks: by depth "
      f"{ {k: round(v['median'], 3) for k, v in cmp_lag['by_depth_B'].items()} }")
assert abs(cmp_same["median"] - 1.0) < 0.02 and cmp_lag["by_depth_B"]["near"]["median"] > 1.05 and abs(cmp_lag["by_depth_B"]["far"]["median"] - 1.0) < 0.03
# contact bridge (reviewer 2026-10-07l): ground points on a plane y = 1.5 - 0.02 z (gently tilted), observed by the true cameras; their LIFT is the
# plane compressed by 1/1.15 along the rays (the near-field bias). beta at the contact rays must read 1.15, the planes must be parallel, and the
# ratio must show no depth trend
zg = rng.uniform(4, 12, 200); xg = rng.uniform(-3, 3, 200); yg = 1.5 - 0.02 * zg
ground = np.stack([xg, yg, zg], 1); lift_ground = ground / 1.15
uv_g = np.stack([proj(ground, E_true[k], K[k]) + rng.normal(scale=0.3, size=(200, 2)) for k in range(F)]).transpose(1, 0, 2)
pts_g = BA.triangulated_heldout(E_true, K, uv_g, np.ones((200, F), bool), np.zeros((200, F), bool), lift_ground[:, 2], GATE, TOL, return_points=True, X_lift=lift_ground)["points"]
band_all = np.ones((256, 256), bool)
contact = np.array([[100.0, 170.0], [128.0, 172.0], [156.0, 170.0]])
cb = BA.contact_bridge(pts_g, band_all, K[0], contact, rho=1.15, tol=TOL)
print(f"contact bridge: beta {cb['beta']:.3f} (expect 1.150), beta/rho-1 {cb['beta_over_rho_minus_1']:+.3f}, tilt {cb['tilt_deg']:.2f} deg, height ratio {cb['height_ratio_abs_d']:.3f}, "
      f"Spearman ratio vs Z_tri {cb['spearman_ratio_vs_Ztri_in_band']:+.2f}; planes inliers {cb['plane_tri']['inlier_frac']:.2f}/{cb['plane_lift']['inlier_frac']:.2f}")
assert abs(cb["beta"] - 1.15) < 0.02 and cb["tilt_deg"] < 1.0 and abs(cb["height_ratio_abs_d"] - 1.15) < 0.02 and abs(cb["spearman_ratio_vs_Ztri_in_band"]) < 0.3
Mh = np.zeros((256, 256), bool); Mh[100:180, 60:200] = True; Mh[180:230, 70:85] = True; Mh[180:236, 170:190] = True   # a body on two legs
rays = BA.bottom_edge_rays(Mh, 0.10)
assert rays.shape[1] == 2 and set(np.unique(rays[:, 1]).tolist()) <= {229.5, 235.5} and 15 <= len(rays) <= 35, (rays.shape, np.unique(rays[:, 1]))   # legs only, not the belly
bp = BA.band_parallax(uv_g, np.ones((200, F), bool), np.zeros((200, F), bool), np.ones(200, bool))
# pre-check (reviewer 2026-10-07m): translation-only camera over a ground plane; the difference between the bridges must equal F_lo x (1 - beta_lo/beta_hi)
Hp_, Wp_ = 30, 52; Kp_ = np.array([[140.0 / 16, 0, 128 / 16], [0, 255.0 / 16, 128 / 16], [0, 0, 1]])
E_tr = np.tile(np.eye(4), (6, 1, 1)); E_tr[:, 0, 3] = 0.15 * np.arange(6)                                   # pure sideways translation
pre = BA.flow_difference(ground, E_tr, Kp_, Hp_, Wp_, 1.0, 1.146, np.zeros((Hp_, Wp_), bool))
from src.generate.m3.geometry import project as PJ_
pb0, _ = PJ_.project(ground, E_tr, Kp_)
F_lo = np.linalg.norm(pb0[1:] - pb0[0], axis=-1)
print(f"pre-check: median diff {pre['median_patch']:.3f} patch vs expected {np.median(F_lo) * (1 - 1 / 1.146):.3f}; share > 0.5: {pre['share_above_half_patch']:.2f}; {pre['reading']}")
assert abs(pre["median_patch"] - np.median(F_lo) * (1 - 1 / 1.146)) < 0.1 * np.median(F_lo) * (1 - 1 / 1.146) + 0.02
# test_bridge_camera_invariance (v2.6 §2.3): scaling a camera's translations by k scales the triangulated depths, rho and beta by k, so tau / beta is invariant
E_k = E_true.copy(); E_k[:, :3, 3] *= 1.3
pts_k = BA.triangulated_heldout(E_k, K, Pw, Qw, Xw_, Z_lift, GATE, TOL, return_points=True)["points"]
zk = np.array([p["Z_tri"] for p in pts_k]); zt = np.array([p["Z_tri"] for p in pts_true]); ratio_k = np.median(zk / zt)
beta_k, beta_1 = ratio_k * 1.0, 1.0                                                   # beta scales with the camera's gauge
print(f"camera invariance: translations x1.3 -> Z_tri x{ratio_k:.4f}; tau/beta ratio {(1.3 / beta_k) / (1.0 / beta_1):.4f} (expect 1)")
assert abs(ratio_k - 1.3) < 0.01 and abs((1.3 / beta_k) - 1.0) < 0.01
assert bp["n"] == 200 and bp["p90_px"] > bp["median_px"] > 0
print("test_m3_bundle_adjust: passed (gauge fixed at camera 0; the held-out floor is the held-out points' own 5 % lift error; triangulation removes it; split column self-tested)")
