"""m2_c C3 — build order step 0/1 groundwork (SKILL_m2_c_C3 v1.1): test 0 (z-buffer exactness) and
the C1 placement record round trip (review B1). Composition tests 1-10 follow in later steps.

Run:  python tests/test_m2c_c3.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2c import placement, proxy, synthetic                 # noqa: E402

REPORT = ["# m2_c C3 — test 0 z-buffer exactness, placement record", ""]

# ---------------------------------------------------------------- test 0: z-buffer bit-exact vs the loop
R = synthetic.rot_y(30.0) @ synthetic.rot_x(10.0)
lines = []
for kind in ("sphere", "ellipsoid", "box", "asym", "billboard"):
    m = proxy.uv_sphere(n_lat=150, n_lon=300) if kind == "sphere" else \
        proxy.uv_sphere(n_lat=150, n_lon=300, radii=(1.3, 0.8, 0.6), name="ellipsoid") if kind == "ellipsoid" else proxy.make_proxy(kind)
    for px, z in ((256, 6.0), (1024, 3.5)):
        K = np.array([[px, 0, px / 2], [0, px, px / 2], [0, 0, 1.0]])
        V = np.array([0.4, -0.2, z]) + proxy.aligned_vertices(m, R)
        t0 = time.time(); zb_loop, mk_loop = proxy.render_perspective(V, m.F, K, px, px, fast=False); t_loop = time.time() - t0
        t0 = time.time(); zb_vec, mk_vec = proxy.render_perspective(V, m.F, K, px, px, fast=True); t_vec = time.time() - t0
        assert mk_loop.shape == mk_vec.shape and np.array_equal(mk_loop, mk_vec), (kind, px, int((mk_loop ^ mk_vec).sum()))
        assert np.array_equal(zb_loop, zb_vec), (kind, px, float(np.nanmax(np.abs(np.where(mk_loop, zb_loop - zb_vec, 0)))))
        lines.append(f"{kind}@{px}: {m.F.shape[0]} tris, {int(mk_vec.sum())} px, bit-exact (mask and depth), loop {t_loop:.2f}s -> vec {t_vec:.3f}s")
# orthographic path (constants raster) too
m = proxy.ellipsoid()
Va = proxy.aligned_vertices(m, R)
zo_l, mo_l = proxy.rasterize(*(lambda uv: (uv[m.F], Va[m.F][..., 2]))((Va[:, :2] + 1.4) / (2.8 / 800)), 800, 800, perspective=False)
zo_v, mo_v = proxy.rasterize_vec(*(lambda uv: (uv[m.F], Va[m.F][..., 2]))((Va[:, :2] + 1.4) / (2.8 / 800)), 800, 800, perspective=False)
assert np.array_equal(mo_l, mo_v) and np.array_equal(zo_l, zo_v)
REPORT.append("test 0 z-buffer exactness: " + "; ".join(lines) + "; orthographic 800-grid bit-exact")

# ---------------------------------------------------------------- placement record round trip (review B1)
mesh = proxy.uv_sphere(n_lat=150, n_lon=300, radii=(1.3, 0.8, 0.6), name="ellipsoid")
st0, mk, dp, mm, tr = synthetic.make_scene("ellipsoid", R_A=R, z_true=8.0, rho_x=0.0, mesh=mesh, grid_n=800)
st = placement.source_stats(mk, dp, st0["K_a"].astype(float))
pl = placement.place(st0, st, mm, consts={"a_0": tr["a_0"], "h_b": tr["h_b"], "delta_p": tr["delta_p"]})
pl_r, hist = placement.refine(st0, st, mm, pl, iters=1)
with tempfile.TemporaryDirectory() as td:
    f = placement.write_placement(Path(td) / "clip" / "vehicle", pl_r, hist, "clip", "vehicle", "abcd", "right", st, st0,
                                  {"m2a": "h1", "m2b2": "h2"})
    rec = placement.read_placement(Path(td) / "clip" / "vehicle")
    assert rec["admitted"] == pl_r.admitted and rec["proxy_cache_key"] == "abcd" and rec["proxy_facing"] == "right"
    assert np.allclose(rec["placement"]["R_A"], pl_r.R_A) and abs(rec["placement"]["z"] - pl_r.z) < 1e-12
    assert rec["heading_out_of_plane_deg"] == pl_r.pitch_deg and rec["source_stats"]["v_contact"] == st.v_contact
    try:
        placement.read_placement(Path(td) / "clip" / "human"); raise AssertionError("missing record accepted")
    except FileNotFoundError:
        pass
    (Path(td) / "clip" / "vehicle" / "placement.json").write_text('{"schema": "other"}', encoding="utf-8")
    try:
        placement.read_placement(Path(td) / "clip" / "vehicle"); raise AssertionError("wrong schema accepted")
    except AssertionError as e:
        assert "schema" in str(e)
REPORT.append(f"placement record: write/read round trip (z {pl_r.z:.4f}, s {pl_r.s:.4f}, admitted {pl_r.admitted}, pitch {pl_r.pitch_deg:+.2f} deg); "
              "missing and wrong-schema records refused")

REPORT.append("step 0 passed")

# ============================================================ C3 tests 1-10 (SKILL v1.1 §8)
from src.anchor.m2c import compose, sanitise                            # noqa: E402

P = {"p_low": 0.02, "p_high": 0.98, "render_px": 1024, "coarse_n": 32, "contact_band_rows": 8, "c2_border_band_px_256": 26}
H = W = 1024; N = 32
K256 = np.array([[256.0, 0, 128], [0, 256.0, 128], [0, 0, 1]])
K1024 = K256.copy(); K1024[:2] *= 4
K32 = K256.copy(); K32[:2] *= 32 / 256.0
n_g, d_g = np.array([0.0, -1.0, 0.0]), 1.5                      # level ground 1.5 below the camera
zeta_c = sanitise.plane_disparity(n_g, d_g, K32, N, N)           # exact plane at the coarse grid (cell centres)
valid_c = zeta_c > 0                                              # above the horizon: sky (invalid)
zeta_c = np.where(valid_c, zeta_c, 0.0)
sph = proxy.uv_sphere(n_lat=150, n_lon=300)


def rec_for(X, s, R_A=np.eye(3), admitted=True):
    return {"clip": "syn", "category": "test", "admitted": admitted, "proxy_cache_key": "k", "proxy_facing": "right",
            "placement": {"X": np.asarray(X, float), "s": float(s), "R_A": np.asarray(R_A, float), "rho_tilde": np.zeros(3)}}


# ---- test 1: domain round trip (the one conversion)
z_plane = 1.0 / np.where(sanitise.plane_disparity(n_g, d_g, K1024, H, W) > 0, sanitise.plane_disparity(n_g, d_g, K1024, H, W), 1.0)
M_all = np.ones((H, W), bool)
back = 1.0 / compose.object_disparity(z_plane, M_all)
assert np.abs(back - z_plane).max() < 1e-12 * z_plane.max()
z_sph, m_sph = compose.render_object(sph.V, sph.F, [0.0, 0.5, 6.0], 1.0, np.eye(3), K1024, H, W)
zt = compose.object_disparity(z_sph, m_sph)
assert np.array_equal(zt == 0, ~m_sph) and np.abs(1.0 / zt[m_sph] - z_sph[m_sph]).max() < 1e-12
REPORT.append("test 1 domain round trip: depth -> 1/z -> depth exact to 1e-12 relative on a full-frame plane and on a rendered sphere; zeta = 0 exactly off the mask")

# ---- test 2: composition sign (max, not min)
zb, vb = compose.background_full(zeta_c, valid_c, H, W)
zf, mf = compose.render_object(sph.V, sph.F, [0.0, 0.5, 6.0], 1.0, np.eye(3), K1024, H, W)        # bottom at y = 1.5: ON the ground
_, _, occ_front = compose.compose(zb, vb, compose.object_disparity(zf, mf), mf)
zbk, mbk = compose.render_object(sph.V, sph.F, [0.0, 3.0, 20.0], 1.0, np.eye(3), K1024, H, W)     # under the ground: behind it along every ray
_, _, occ_back = compose.compose(zb, vb, compose.object_disparity(zbk, mbk), mbk)
assert occ_front[mf].mean() < 0.01 and occ_back[mbk].mean() == 1.0, (occ_front[mf].mean(), occ_back[mbk].mean())
REPORT.append(f"test 2 composition sign: object on the ground wins on {100 * (1 - occ_front[mf].mean()):.1f} % of its pixels; object below the ground loses on 100 %")

# ---- test 3: partial occlusion follows the ridge, not the silhouette
ridge_row = 700
zeta_wall = np.max(compose.object_disparity(zf, mf)) * 2.0
zb_r = zb.copy(); zb_r[ridge_row:] = zeta_wall; vb_r = vb | (np.arange(H)[:, None] >= ridge_row)
_, _, occ_r = compose.compose(zb_r, vb_r, compose.object_disparity(zf, mf), mf)
rows = np.arange(H)[:, None]
assert np.array_equal(occ_r, mf & (rows >= ridge_row)), "occlusion boundary must be the ridge row"
assert (mf & (rows < ridge_row)).any() and (mf & (rows >= ridge_row)).any()
REPORT.append(f"test 3 partial occlusion: with a wall from row {ridge_row}, lost pixels == M_obj & (row >= {ridge_row}) exactly")

# ---- test 4: contact continuity
step, v_bottom = compose.contact_step(compose.object_disparity(zf, mf), zb, mf, 8)
quantum = compose.contact_quantum(n_g, d_g, K1024, H / N)
assert abs(step) < quantum, (step, quantum)
REPORT.append(f"test 4 contact continuity: sphere on the plane, contact step {step:+.2e} vs one coarse-cell quantum {quantum:.2e} (row {v_bottom})")

# ---- test 5: normalisation invariance
zc_ctrl, valid, _ = compose.compose(zb, vb, compose.object_disparity(zf, mf), mf)
D0, ql, qh = compose.normalise(zc_ctrl, valid, P["p_low"], P["p_high"])
D1, _, _ = compose.normalise(3.7 * zc_ctrl + 0.21, valid, P["p_low"], P["p_high"])
inv = float(np.abs(D1 - D0).max())
assert inv < 1e-12, inv
REPORT.append(f"test 5 normalisation invariance: D_ctrl under zeta -> 3.7 zeta + 0.21 differs by {inv:.1e} (float rounding; the gauge is gone)")

# ---- test 6: invalid handling (sky)
assert (zb[~vb] == 0).all() and np.isfinite(zc_ctrl).all()
_, ql_all, _ = compose.normalise(zc_ctrl, np.ones_like(valid), P["p_low"], P["p_high"])
assert ql_all < ql, "sky included would drag q_low down"
assert (D0[~valid] == 0).all(), "sky clips to the far end"
REPORT.append(f"test 6 invalid: sky zeta = 0, no nan; q_low over valid {ql:.4f} vs with sky {ql_all:.4f}; sky -> D = 0 (far end); sky frac {100 * (1 - vb.mean()):.1f} %")

# ---- test 7: resolution consistency (horizon row)
a_, b_, g_ = sanitise.affine_from_plane(n_g, d_g, K256)
v_h256 = -(a_ * (128 + 0.5) + g_) / b_ - 0.5                      # continuous row index where zeta = 0 at column 128
col = zb[:, 512]; below = np.nonzero(col > 0)[0]
r0 = below.min()
# MEASURED 2026-09-15: the upsampled field is the exact affine only where the Keys kernel's support
# (2 coarse cells = 64 px at 1024) lies entirely in valid cells; within 64 px of the sky boundary the
# zeroed sky cells bend it (first-two-rows extrapolation gave 500.4 vs 511.5). So the horizon is
# extrapolated from rows >= 2 cells below the first ground row, where the field is affine.
rows_fit = np.arange(r0 + 64, r0 + 400)
(slope, icpt), *_ = np.linalg.lstsq(np.stack([rows_fit + 0.5, np.ones_like(rows_fit, float)], 1), col[rows_fit], rcond=None)
resid = float(np.abs(col[rows_fit] - (slope * (rows_fit + 0.5) + icpt)).max())
v_cross = -icpt / slope - 0.5
v_expected = 4 * (v_h256 + 0.5) - 0.5
assert resid < 1e-9 and abs(v_cross - v_expected) < 0.5, (v_cross, v_expected, resid)
z1, z2 = col[r0], col[r0 + 1]; v_naive = (r0 + 0.5) - z1 / (z2 - z1)
REPORT.append(f"test 7 resolution: horizon row at 1024 {v_cross:.3f} vs 4 x the 256 horizon {v_expected:.3f} (affine residual {resid:.1e} on rows "
              f">= 2 cells below the sky); naive first-rows extrapolation {v_naive:.1f} — the field is bent within 2 cells (64 px) of the sky boundary")

# ---- test 8 / 9 / 10 through the stage function
c2_rec = {"components": [{"filled": False, "reason": compose.SKY_REASON}]}
arrays, qc = compose.compose_control(sph.V, sph.F, rec_for([0.0, 0.5, 6.0], 1.0), zeta_c, valid_c, c2_rec, K256, n_g, d_g, P)
assert qc["emitted"] and np.array_equal(arrays["D_ctrl_inv"], (np.float32(1.0) - arrays["D_ctrl"]).astype(np.float32))
REPORT.append("test 8 polarity complement: D_ctrl_inv == 1 - D_ctrl exactly on the stored float32 field; polarity 'uncalibrated'")
arr9, qc9 = compose.compose_control(sph.V, sph.F, rec_for([0.0, 0.1, 0.735], -0.113), zeta_c, valid_c, c2_rec, K256, n_g, d_g, P)
assert arr9 is None and qc9["gates"]["g_depth_sign"]["outcome"] == "fail" and not qc9["emitted"]
with tempfile.TemporaryDirectory() as td:
    assert compose.write_control(Path(td) / "neg", None, qc9, {}) is None and (Path(td) / "neg" / "qc_c3.json").exists()
    assert not (Path(td) / "neg" / "control.npz").exists()
    arr_bad, qc_bad = compose.compose_control(sph.V, sph.F, rec_for([0.0, 0.5, 6.0], 1.0), zeta_c, valid_c,
                                              {"components": [{"filled": False, "reason": "harmonic_solve_failed"}]}, K256, n_g, d_g, P)
    assert arr_bad is None and qc_bad["gates"]["g_invalid_reason"]["outcome"] == "fail"
    REPORT.append("test 9 refusals: s = -0.113 (clip 284's pre-fix value) -> g_depth_sign fail, only qc_c3.json written; a non-sky unfilled component -> g_invalid_reason fail")
    f = compose.write_control(Path(td) / "ok", arrays, qc, {"c1": "h"})
    back_arr, back_qc = compose.read_control(Path(td) / "ok")
    zc_saved = back_arr["zeta_ctrl"].astype(np.float64)
    a0, b0 = 0.37, 0.011
    zhat = (zc_saved - b0) / a0                                   # a synthetic anchor estimate in another gauge
    Mfit = back_arr["M_obj"]
    A = np.stack([zhat[Mfit], np.ones(int(Mfit.sum()))], 1)
    (a_hat, b_hat), *_ = np.linalg.lstsq(A, zc_saved[Mfit], rcond=None)
    assert abs(a_hat - a0) < 1e-6 and abs(b_hat - b0) < 1e-6, (a_hat, b_hat)
    assert (Path(td) / "ok" / "control.png").exists()
REPORT.append(f"test 10 zeta_ctrl retention: pre-normalisation field written and read back; alpha-style affine fit recovers (a, b) = ({a_hat:.6f}, {b_hat:.6f}) vs (0.37, 0.011)")

# ---- percentile domain (reviewer 2026-09-15): a close object saturates with the object included
near_rec = rec_for([0.0, 0.9, 2.2], 0.6)                          # a big near sphere on the ground (bottom at y = 1.5)
arr_bg, qc_bg = compose.compose_control(sph.V, sph.F, near_rec, zeta_c, valid_c, c2_rec, K256, n_g, d_g, {**P, "percentile_domain": "background"})
arr_all, qc_all = compose.compose_control(sph.V, sph.F, near_rec, zeta_c, valid_c, c2_rec, K256, n_g, d_g, {**P, "percentile_domain": "all"})
# MEASURED 2026-09-15: the reviewer expected the object-included percentiles to be the problem; on a near
# object it is the background-only domain that saturates it (nothing in the background is as near), and
# only p_high = 1.0 removes saturation. Recorded, not asserted in either direction; the human locks it.
assert qc_bg["q_background"] == qc_all["q_background"] and qc_bg["q_all_valid"] == qc_all["q_all_valid"]   # both pairs recorded either way
assert qc_bg["obj_saturated_frac_if_p_high_max"] < 0.01          # only the single nearest pixel maps to exactly 1
arr_fc, qc_fc = compose.compose_control(sph.V, sph.F, near_rec, zeta_c, valid_c, c2_rec, K256, n_g, d_g, {**P, "percentile_domain": "background_floor_object_ceiling"})
assert qc_fc["obj_saturated_frac"] < 0.03 and qc_fc["q_low"] == qc_bg["q_background"][0] and qc_fc["q_high"] >= qc_bg["q_background"][1]
assert 0 < qc_fc["background_range_compression"] <= 1.0
REPORT.append(f"percentile domain (finding): near sphere ({100 * qc_bg['obj_area_frac']:.0f} % of the frame) saturates "
              f"{100 * qc_bg['obj_saturated_frac_if_background']:.0f} % over background-only percentiles, {100 * qc_all['obj_saturated_frac_if_all']:.0f} % "
              f"with the object included, {100 * qc_bg['obj_saturated_frac_if_p_high_max']:.0f} % with p_high = max, "
              f"{100 * qc_fc['obj_saturated_frac']:.0f} % with background floor + object ceiling (background range compression "
              f"{100 * (1 - qc_fc['background_range_compression']):.0f} %); all recorded per clip")

# ---- contact-step ratio gate refuses when the lock ceiling is set (reviewer A1(b), 2026-09-18)
float_rec = rec_for([0.0, -0.4, 6.0], 1.0)                      # sphere floating 0.9 above the ground: feet nearer than the ground under them
a_f, q_f = compose.compose_control(sph.V, sph.F, float_rec, zeta_c, valid_c, c2_rec, K256, n_g, d_g, {**P, "thresholds": {"m2c_contact_step_max": 5}})
a_u, q_u = compose.compose_control(sph.V, sph.F, float_rec, zeta_c, valid_c, c2_rec, K256, n_g, d_g, P)
a_ok, q_ok = compose.compose_control(sph.V, sph.F, rec_for([0.0, 0.5, 6.0], 1.0), zeta_c, valid_c, c2_rec, K256, n_g, d_g, {**P, "thresholds": {"m2c_contact_step_max": 5}})
assert a_f is None and q_f["refused_by"] == "g_contact_step" and abs(q_f["contact_step_ratio"]) > 5, q_f.get("contact_step_ratio")
assert a_u is not None and q_u["gates"]["g_contact_step"]["outcome"] == "ungated"           # null ceiling -> recorded only
assert a_ok is not None and q_ok["gates"]["g_contact_step"]["outcome"] == "pass"
REPORT.append(f"contact-step gate: a floating sphere (ratio {q_f['contact_step_ratio']:+.1f}) is refused under the ceiling 5 and emitted when the "
              f"ceiling is null; the grounded sphere (ratio {q_ok['contact_step_ratio']:+.2f}) passes")

# ---- locked polarity: one image, in the locked sense; unlocked: both, exact complements
a_nb, q_nb = compose.compose_control(sph.V, sph.F, rec_for([0.0, 0.5, 6.0], 1.0), zeta_c, valid_c, c2_rec, K256, n_g, d_g, {**P, "thresholds": {"m2d_polarity": "near_bright"}})
a_nd, q_nd = compose.compose_control(sph.V, sph.F, rec_for([0.0, 0.5, 6.0], 1.0), zeta_c, valid_c, c2_rec, K256, n_g, d_g, {**P, "thresholds": {"m2d_polarity": "near_dark"}})
assert "D_ctrl_inv" not in a_nb and "D_ctrl_inv" not in a_nd and q_nb["polarity"] == "near_bright" and q_nd["polarity"] == "near_dark"
assert np.array_equal(a_nb["D_ctrl"], arrays["D_ctrl"]) and np.array_equal(a_nd["D_ctrl"], arrays["D_ctrl_inv"])
assert np.array_equal(a_nb["zeta_ctrl"], a_nd["zeta_ctrl"]), "the retained disparity is polarity-free"
assert qc["polarity"] == "uncalibrated" and "D_ctrl_inv" in arrays
REPORT.append("locked polarity: near_bright emits D_ctrl alone, near_dark emits its exact complement alone, zeta_ctrl identical in both; unlocked emits both")

# ---- PNG writer applies no transfer curve (reviewer 2026-09-18): bytes == round(D_ctrl * 255), 3 equal channels, no gamma chunk
import cv2                                                               # noqa: E402
with tempfile.TemporaryDirectory() as td:
    compose.write_control(Path(td) / "png", arrays, qc, {})
    raw = (Path(td) / "png" / "control.png").read_bytes()
    img = cv2.imread(str(Path(td) / "png" / "control.png"), cv2.IMREAD_UNCHANGED)
    want = (np.clip(arrays["D_ctrl"], 0, 1) * 255.0 + 0.5).astype(np.uint8)
    assert img.shape == (1024, 1024, 3) and all(np.array_equal(img[..., c], want) for c in range(3))
    assert b"gAMA" not in raw and b"sRGB" not in raw and b"iCCP" not in raw, "PNG carries a colour-transfer chunk"
REPORT.append("PNG writer: control.png bytes == round(D_ctrl * 255) on all three channels; no gAMA / sRGB / iCCP chunk (no transfer curve between D_ctrl and the file)")

REPORT += ["", "all C3 tests 0-10 passed"]
Path("docs/report_m2c_c3_tests.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
print("\n".join(REPORT[-12:]))
