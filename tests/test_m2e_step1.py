"""m2_e build steps 1-2 — SKILL m2_e v2 §9 tests 1, 1b, 2, 3(a, b, c), 7, plus review item 7's square-pixel check. CPU, no models.

Synthetic scene: anisotropic K, a horizontal ground plane below the camera, a vertical planar object standing on it at a known yaw.
Disparities are exact ray-plane intersections; the "estimator" is an affine map plus persistent per-pixel error.

Run:  python tests/test_m2e_step1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2e import alpha as A                                   # noqa: E402
from src.anchor.m2b2 import body                                        # noqa: E402

REPORT = ["# m2_e steps 1-2 — alpha, fit region, point map: synthetic tests", ""]
H = W = 512
K = np.array([[1800.0, 0, 256.0], [0, 2600.0, 256.0], [0, 0, 1.0]])          # anisotropic, like the controls (fy > fx)
N_GROUND = np.array([0.0, -1.0, 0.0]); D_CAM = 1.2                          # ground: -y + d = 0  ->  y = d (camera above the ground)
rng = np.random.default_rng(0)


def rays():
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    return np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T


def ground_disparity(r):
    z = np.where(r[..., 1] > 1e-9, D_CAM / np.maximum(r[..., 1], 1e-9), np.nan)      # y = z r_y = d
    return np.where(np.isfinite(z), 1.0 / z, np.nan)


def object_plane(yaw_deg, P0=(0.15, 0.0, 6.0), width=1.6, height=1.0):
    """Vertical plane through P0 with normal (sin psi, 0, cos psi); returns (disparity over its rectangle, mask, normal, d_pl)."""
    psi = np.radians(yaw_deg); n = np.array([np.sin(psi), 0.0, np.cos(psi)]); t = np.array([np.cos(psi), 0.0, -np.sin(psi)])
    P0 = np.array(P0, float); P0[1] = D_CAM - height / 2
    d_pl = float(n @ P0); r = rays()
    den = r @ n; z = np.where(np.abs(den) > 1e-9, d_pl / den, np.nan)
    X = z[..., None] * r
    inside = np.isfinite(z) & (z > 0) & (np.abs((X - P0) @ t) <= width / 2) & (X[..., 1] >= D_CAM - height) & (X[..., 1] <= D_CAM)
    return np.where(inside, 1.0 / z, np.nan), inside, n, d_pl


def scene(yaw_deg, **kw):
    r = rays(); zg = ground_disparity(r); zo, M, n, d_pl = object_plane(yaw_deg, **kw)
    zeta = np.where(M, zo, zg)                                              # the object occludes the ground where present
    return zeta, M, n, d_pl, np.isfinite(zg)


def yaw_from_pointmap(depth, M):
    """Principal-plane normal of the back-projected object pixels, yaw about y."""
    r = rays()[M]; X = np.asarray(depth)[M][:, None] * r
    Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc, full_matrices=False); n = Vt[-1]
    n = n if n[2] > 0 else -n
    return float(np.degrees(np.arctan2(n[0], n[2])))


# ---- scene in the source gauge: zeta_ctrl = ground (C2 background) + proxy at yaw 0
a_true, b_true = 2.3, -0.15
zeta_ctrl, M_obj, n_proxy, d_proxy, valid = scene(35.0)            # the proxy at 35 deg: a fronto-parallel plane has NO disparity span
sky = ~valid
hole = np.zeros((H, W), bool); hole[300:340, 200:320] = True                # "C2's filled hole": excluded from the band

# ---- test 1: round trip (no error) on band and object, to 1e-6
zeta_hat = (zeta_ctrl - b_true) / a_true
band, bprov = A.ground_band(M_obj, M_obj, K, N_GROUND, valid, hole, inner_px=8, outer_frac=1.0)
for reg, name in ((band, "band"), (M_obj, "object")):
    f = A.affine_fit(zeta_hat, zeta_ctrl, reg)
    assert abs(f["alpha"] - a_true) < 1e-6 and abs(f["b"] - b_true) < 1e-6, (name, f["alpha"], f["b"])
REPORT.append(f"test 1 round trip: (a, b) = ({a_true}, {b_true}) recovered to 1e-6 on the band ({bprov['n_pixels']} px) and on the object ({int(M_obj.sum())} px)")

# ---- test 1b: persistent per-pixel error, sigma swept (review item 12: sigma is a synthetic parameter, not 0.2)
x = zeta_ctrl[M_obj]; s2 = np.var(x) / a_true ** 2                          # variance of the TRUE zeta_hat over the object
lines = []
for k in (0.5, 1.0, 2.0):
    sig = k * np.sqrt(s2)
    e = rng.normal(0, sig, (H, W))                                            # persistent per pixel: fixed for the image
    zh = (zeta_ctrl - b_true) / a_true + e
    f = A.affine_fit(zh, zeta_ctrl, M_obj); ci = A.bootstrap_alpha(zh, zeta_ctrl, M_obj, 0, 200)["ci"]
    lam = s2 / (s2 + sig ** 2)
    assert ci[0] <= a_true <= ci[1], (k, f["alpha"], ci)
    assert abs(f["alpha_naive"] - lam * a_true) <= 0.10 * lam * a_true, (k, f["alpha_naive"], lam * a_true)
    lines.append(f"sigma={k:.1f}s: reversed {f['alpha']:.3f} (CI {ci[0]:.3f}-{ci[1]:.3f} contains {a_true}), naive {f['alpha_naive']:.3f} vs lambda*a {lam * a_true:.3f}")
REPORT.append("test 1b persistent error: " + "; ".join(lines) + " — the reversed direction recovers a within its interval, the naive one is attenuated by the analytic lambda")

# ---- test 7: the fit region — object fit over the INTERSECTION; the band excludes the inner margin, the hole, sky and above-horizon
anchor_mask = np.roll(M_obj, 12, axis=1)                                     # a generated object shifted 12 px from the proxy
inter = anchor_mask & M_obj
f = A.affine_fit(zeta_hat, zeta_ctrl, inter)
assert f["n_pixels"] == int(inter.sum()) and inter.sum() < anchor_mask.sum() and inter.sum() < M_obj.sum()
band2, bp2 = A.ground_band(anchor_mask, M_obj, K, N_GROUND, valid, hole, 16, 1.0)
import cv2                                                                    # noqa: E402
inner = cv2.dilate((anchor_mask | M_obj).astype(np.uint8), np.ones((33, 33), np.uint8)).astype(bool)
assert not (band2 & inner).any() and not (band2 & hole).any() and not (band2 & sky).any() and (band2 <= A.below_horizon(K, N_GROUND, H, W)).all()
assert band2.sum() > 0 and np.isfinite(zeta_ctrl[band2]).all()
REPORT.append(f"test 7 fit region: object fit uses exactly the intersection ({int(inter.sum())} px of {int(anchor_mask.sum())} / {int(M_obj.sum())}); "
              f"the band ({bp2['n_pixels']} px, inner {bp2['inner_px']} px, outer {bp2['outer_px']} px) contains no inner-margin, hole, sky or above-horizon pixel")

# ---- test 2: circularity guards (API label and numerical tripwire)
for bad in ("zeta_ctrl", "control", "proxy"):
    try:
        A.pointmap_source_guard(bad); raise RuntimeError("accepted")
    except A.PointMapCircular:
        pass
A.pointmap_source_guard("alpha_scaled_zeta_hat")
depth_proxy = 1.0 / zeta_ctrl
try:
    A.assert_not_proxy(depth_proxy, zeta_ctrl, M_obj); raise RuntimeError("proxy depth accepted")
except A.PointMapCircular:
    pass
zeta_anchor, M_a, _, _, _ = scene(45.0)
_, depth_anchor, _ = A.anchor_pointmap((zeta_anchor - b_true) / a_true, a_true, b_true, K, valid)
chk = A.assert_not_proxy(depth_anchor, zeta_ctrl, M_obj)
assert chk["checked"] and chk["relative_residual"] > 1e-3
REPORT.append(f"test 2 circularity: labels zeta_ctrl/control/proxy refused; 1/zeta_obj as the point map refused by the tripwire; "
              f"the anchor's own depth (yaw 45 deg vs proxy 35) passes with relative residual {chk['relative_residual']:.2e}")

# ---- test 3a: alpha scaling (b = 0) preserves yaw for alpha in {0.3, 1, 3}
for psi in (10.0, -25.0):
    zt, Mt, _, _, _ = scene(psi)
    for al in (0.3, 1.0, 3.0):
        _, dep, _ = A.anchor_pointmap(zt / al, al, 0.0, K, valid)             # zeta_hat = zeta / alpha; depth = 1 / (alpha zeta_hat)
        assert abs(yaw_from_pointmap(dep, Mt) - psi) < 0.1, (psi, al, yaw_from_pointmap(dep, Mt))
REPORT.append("test 3a: with b = 0, alpha in {0.3, 1, 3} leaves a 10 deg and a -25 deg object yaw unchanged to 0.1 deg")

# ---- test 3b: a shift b != 0 rotates a tilted plane's normal toward the optical axis by the analytic amount
zt, Mt, n_o, d_pl, _ = scene(45.0)
lines = []
for frac in (0.1, 0.5):
    bshift = frac * float(np.nanmean(zt[Mt]))
    _, dep, _ = A.anchor_pointmap(zt, 1.0, bshift, K, valid)                   # depth = 1 / (zeta + b)
    n_new = n_o / d_pl + bshift * np.array([0, 0, 1.0])
    yaw_analytic = float(np.degrees(np.arctan2(n_new[0], n_new[2]))); yaw_meas = yaw_from_pointmap(dep, Mt)
    assert abs(yaw_meas - yaw_analytic) < 0.1, (frac, yaw_meas, yaw_analytic)
    lines.append(f"shift {int(100 * frac)} % -> yaw 45 -> {yaw_meas:.1f} deg (analytic {yaw_analytic:.1f})")
REPORT.append("test 3b: " + "; ".join(lines) + " — a disparity shift is a yaw change, which is why alpha is not fitted on the object")

# ---- test 3c: the circularity in action — object-mask fit pulls the yaw toward the proxy, the band fit does not
zeta_A, M_A, _, _, _ = scene(45.0)                                             # the generated anchor: same object, yawed 10 deg past the proxy's 35
zh = (zeta_A - b_true) / a_true + rng.normal(0, 0.002, (H, W))                # the estimator sees the ANCHOR, with small error
th = {"m2e_alpha_pixels_min": None, "m2e_delta_width_max": None, "m2e_alpha_ci_rel_max": None}
band3, _ = A.ground_band(M_A, M_obj, K, N_GROUND, valid, hole, 16, 1.0)
rec = A.fit_alpha(zh, zeta_ctrl, band3, M_A & M_obj, th, blocks=(32, 64), n_boot=60)
_, dep_band, _ = A.anchor_pointmap(zh, rec["band_fit"]["alpha"], rec["band_fit"]["b"], K, valid)
_, dep_obj, _ = A.anchor_pointmap(zh, rec["object_fit_uncorrected"]["alpha"], rec["object_fit_uncorrected"]["b"], K, valid)
yaw_band, yaw_obj = yaw_from_pointmap(dep_band, M_A), yaw_from_pointmap(dep_obj, M_A)
assert abs(yaw_band - 45.0) < 0.5, yaw_band
assert abs(yaw_obj - 45.0) > 0.5 and abs(yaw_obj - 35.0) < abs(yaw_band - 35.0), (yaw_obj, yaw_band)   # pulled toward the proxy's 35
assert rec["alpha_source"] == "ground_band" and rec["alpha"] is not None
REPORT.append(f"test 3c: anchor yawed 45 deg, proxy 35 deg — band fit alpha {rec['band_fit']['alpha']:.3f} (true {a_true}) recovers yaw {yaw_band:.2f} deg; "
              f"uncorrected object-mask fit alpha {rec['object_fit_uncorrected']['alpha']:.3f} recovers {yaw_obj:.2f} deg, pulled toward the proxy; alpha_source = {rec['alpha_source']}; "
              f"band/object(uncorrected) ratio {rec['band_vs_object_uncorrected_ratio']:.3f}; band intervals block-32 {np.round(rec['band_bootstrap']['block_32']['ci'], 4).tolist()} "
              f"({rec['band_bootstrap']['block_32']['n_units']} blocks), block-64 {np.round(rec['band_bootstrap']['block_64']['ci'], 4).tolist()} ({rec['band_bootstrap']['block_64']['n_units']} blocks), "
              f"pixel {np.round(rec['band_bootstrap']['pixel']['ci'], 4).tolist()}")

# ---- test 3d (reviewer 2026-09-23): the object comparison against the proxy ROTATED by the measured delta (here the true 10 deg)
zeta_proxy_rot, M_rot, _, _, _ = scene(45.0)                                   # the proxy at the anchor's yaw = the delta-corrected proxy
rec_d = A.decide_alpha(rec, zh, zeta_proxy_rot, M_A & M_rot, th, blocks=(32, 64), n_boot=60)
_, dep_corr, _ = A.anchor_pointmap(zh, rec_d["object_fit_corrected"]["alpha"], rec_d["object_fit_corrected"]["b"], K, valid)
yaw_corr = yaw_from_pointmap(dep_corr, M_A)
assert abs(rec_d["object_fit_corrected"]["alpha"] - a_true) < 0.01 * a_true and abs(rec_d["band_vs_object_ratio"] - 1) < 0.02 and abs(yaw_corr - 45.0) < 0.5, \
    (rec_d["object_fit_corrected"]["alpha"], rec_d["band_vs_object_ratio"], yaw_corr)
assert rec_d["alpha_source"] == "ground_band" and rec_d["flag"] is None
REPORT.append(f"test 3d: against the proxy rotated by the true 10 deg, the object fit recovers alpha {rec_d['object_fit_corrected']['alpha']:.3f} (true {a_true}), "
              f"yaw {yaw_corr:.2f} deg, band/object ratio {rec_d['band_vs_object_ratio']:.3f} — the ratio is a flattening diagnostic only after the delta correction")

# ---- review item 7: the square-pixel resample changes K, not the 3-D geometry: no "undo" of the height rescale
frame = np.zeros((H, W, 3), np.uint8); sq = body.square_pixel_frame(frame, M_obj, K, pp_max_px=1e9)
Kp = sq.K if hasattr(sq, "K") else sq[1]
Xs = np.array([[0.3, -0.2, 5.0], [-0.4, 0.5, 7.0]])
for X in Xs:
    u = K @ X; u = u[:2] / u[2]; up = np.array([u[0], u[1] * Kp[1, 1] / K[1, 1]])            # the resample scales v by fx/fy
    Xb = X[2] * (np.linalg.inv(Kp) @ np.array([up[0], up[1], 1.0]))
    assert np.allclose(Xb, X, atol=1e-9), (Xb, X)
d_dir = Xs[1] - Xs[0]
REPORT.append(f"review item 7: two 3-D points projected under the anisotropic K, height-rescaled by fx/fy, back-projected under K' return the identical points "
              f"(direction {np.round(d_dir / np.linalg.norm(d_dir), 4).tolist()} unchanged) — Body's keypoints need no undo")

# ---- test 1c (reviewer 2026-09-23): a LOW-FREQUENCY error field — the block bootstrap must widen, the pixel interval must undercover
def smooth_field(sig, cell=96, seed=1):
    g = np.random.default_rng(seed).normal(0, sig, (H // cell + 2, W // cell + 2))
    yy, xx = np.mgrid[0:H, 0:W] / cell
    y0, x0 = yy.astype(int), xx.astype(int); fy, fx = yy - y0, xx - x0
    return (g[y0, x0] * (1 - fy) * (1 - fx) + g[y0 + 1, x0] * fy * (1 - fx) + g[y0, x0 + 1] * (1 - fy) * fx + g[y0 + 1, x0 + 1] * fy * fx)
cover_pix, cover_b64, widths = [], [], []
for sd in range(12):
    zh = (zeta_ctrl - b_true) / a_true + smooth_field(0.6 * float(np.nanstd(zeta_ctrl[band])) / a_true, seed=sd)
    cp = A.bootstrap_alpha(zh, zeta_ctrl, band, 0, 80, seed=sd)["ci"]; c64 = A.bootstrap_alpha(zh, zeta_ctrl, band, 64, 80, seed=sd)["ci"]
    cover_pix.append(cp[0] <= a_true <= cp[1]); cover_b64.append(c64[0] <= a_true <= c64[1]); widths.append((cp[1] - cp[0], c64[1] - c64[0]))
widths = np.array(widths)
assert (widths[:, 1] > widths[:, 0]).mean() >= 0.9, widths
assert np.mean(cover_b64) > np.mean(cover_pix), (np.mean(cover_pix), np.mean(cover_b64))
REPORT.append(f"test 1c low-frequency error field (12 draws): block-64 interval wider than the pixel interval in {100 * (widths[:, 1] > widths[:, 0]).mean():.0f} % of draws "
              f"(median widths {np.median(widths[:, 1]):.4f} vs {np.median(widths[:, 0]):.4f}); coverage of the true a: pixel {100 * np.mean(cover_pix):.0f} %, block-64 {100 * np.mean(cover_b64):.0f} % "
              "— the pixel interval undercovers on a smooth field, the block bootstrap does its job")

# ---- Body gauge alignment (reviewer 2026-09-23): a mesh depth at the wrong scale and offset recovers (k, t) with rotation unchanged
from src.anchor.m2e import delta as Dm                                  # noqa: E402
zt, Mt, n_o, _, _ = scene(30.0)
depth_anchor = 1.0 / zt
k_true, t_true = 1.7, 0.4
depth_body = k_true * depth_anchor + t_true
ga_m = Dm.gauge_align(depth_body, depth_anchor, Mt)                     # the mode of record (reviewer 2026-09-27): ratio of medians, t = 0
assert ga_m["mode"] == "median_ratio" and ga_m["shift_t"] == 0.0 and abs(ga_m["scale_k"] - np.median(depth_body[Mt]) / np.median(depth_anchor[Mt])) < 1e-12, ga_m
assert abs(ga_m["k_regression"] - k_true) < 1e-9 and abs(ga_m["t_regression"] - t_true) < 1e-9 and ga_m["resid_rms_regression"] < 1e-9, ga_m   # the regression is recorded beside it
assert ga_m["k_regression_nonpositive"] is False and ga_m["k_regression_ci95"][0] - 1e-9 <= k_true <= ga_m["k_regression_ci95"][1] + 1e-9, ga_m   # an exact fit: the CI collapses to float width
ga = Dm.gauge_align(depth_body, depth_anchor, Mt, mode="regression")      # the affine recovery itself: (k, t) exact; rotation untouched by apply_gauge
assert abs(ga["scale_k"] - k_true) < 1e-9 and abs(ga["shift_t"] - t_true) < 1e-9 and ga["resid_rms_used"] < 1e-9, ga
r_ = rays()[Mt]; Vb = depth_body[Mt][:, None] * r_                      # Body's vertices along the same rays at Body's depth
Vs = Dm.apply_gauge(Vb, ga["scale_k"], ga["shift_t"])
Va = depth_anchor[Mt][:, None] * r_
assert np.allclose(Vs, Va, atol=1e-9)
REPORT.append(f"Body gauge alignment: depth_body = {k_true} depth_anchor + {t_true} -> regression k {ga['scale_k']:.6f}, t {ga['shift_t']:.6f}, residual {ga['resid_rms_used']:.1e}; the mode of record (median ratio) gives k {ga_m['scale_k']:.4f} with the regression recorded beside it; "
              "the re-gauged vertices coincide with the anchor's along their rays (rotation untouched by construction: x, y scale with z along the ray)")

# ---- far field: a bridge whose intercept sends the far background negative is accepted over the object/band region and RECORDED for the frame
zh_far = (zeta_ctrl - b_true) / a_true
_, dep_f, prov_f = A.anchor_pointmap(zh_far, a_true, b_true - 0.05, K, valid, region=M_obj)            # shift the intercept down: far ground (small zeta) goes negative; the object (z ~ 6) stays positive
assert prov_f["region_nonpositive_frac"] < 0.05 and prov_f["far_field_nonpositive_frac"] > prov_f["region_nonpositive_frac"]
try:
    A.anchor_pointmap(zh_far, a_true, b_true - 0.05, K, valid, region=None); assert prov_f["far_field_nonpositive_frac"] < 0.05, "whole-frame positivity should have failed"
except AssertionError:
    pass
REPORT.append(f"far field: with the intercept shifted, {100 * prov_f['far_field_nonpositive_frac']:.1f} % of valid pixels (the far background) go non-positive and are recorded, "
              f"while the object region stays positive ({100 * prov_f['region_nonpositive_frac']:.1f} %) — positivity is required on the object and the band the bridge was fitted on, the far-field fraction is the record")

REPORT += ["", "all m2_e step 1-2 tests passed"]
Path("docs/report_m2e_step12_tests.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
print("\n".join(REPORT))
