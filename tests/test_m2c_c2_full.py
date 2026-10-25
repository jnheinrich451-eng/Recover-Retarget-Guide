"""m2_c / C2 harness — steps 3–6 (SKILL_m2_c_C2 §8): T2 (exact fill + depth-domain
comparison), T3 (maximum principle), T4 (coarsening preserves affine), T5
(high-frequency suppression), T6 (leakage), T7 (dilation vs halo), T9 (invalid
pixels), T10 (ring conditioning gate), T8 (fill order, reported, no winner).
HARD STOP after this file: human review before any real clip or lock value.

Run:  python tests/test_m2c_c2_full.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2c import sanitise as S, bg_synthetic as B      # noqa: E402

PARAMS = {"r": 4, "w": 6, "N": 32, "sigma1": 2.0, "sigma2": 0.5, "fill_order": "fill_then_coarsen"}
HARNESS_TH = {"m2c_ring_cond_max": 1000.0, "m2c_leakage_max": 0.2}   # harness-only; lock stays null
REPORT = ["# m2_c C2 harness — steps 3–6 report", "",
          f"params (harness, NOT lock): {PARAMS}; harness gate values {HARNESS_TH}", ""]


def log(s):
    REPORT.append(s); print(s)


def ground_valid(bg):
    return bg["zeta_true"] > 0


# ----------------------------------------------------------------------------
def T2_exact_fill_and_depth_domain():
    bg = B.make_background()
    H, W, K = bg["H"], bg["W"], bg["K"]
    valid = ground_valid(bg)
    zf, vf, rec = S.fill(bg["zeta_obs"], valid, bg["mask"], PARAMS, K)
    omega = S.dilate(bg["mask"], PARAMS["r"])
    err = np.abs(zf - bg["zeta_true"])[omega].max()
    assert err < 1e-12, err
    resid_max = max(c["resid_max_abs"] for c in rec["components"] if c["is_object"])
    assert resid_max < 1e-12, resid_max
    # --- depth-domain comparison, predicted before running: error ∝ z^3 ---------------
    # masks of the SAME pixel size at several rows (several mean z); fill in z with the same
    # affine + harmonic machinery; error vs mean z on a log-log fit -> exponent
    rows, errs, zbar = [130, 150, 175, 200, 230], [], []
    z_true = np.where(valid, 1.0 / np.where(valid, bg["zeta_true"], 1.0), 0.0)
    for v0 in rows:
        m = B.ellipse_mask(H, W, W * 0.5, v0, 14, 14)
        om = S.dilate(m, PARAMS["r"])
        known = valid & ~om
        ring = S.ring_of(om, PARAMS["w"], known)
        coef_z, _ = S.ring_fit(z_true, ring, H, W)
        base_z = S.eval_affine(coef_z, H, W)
        resid = S.harmonic_solve(np.where(known, z_true - base_z, 0.0), om, known)
        z_fill = base_z + resid
        e = np.abs(z_fill - z_true)[om]
        errs.append(float(e.max())); zbar.append(float(z_true[om].mean()))
    slope = np.polyfit(np.log(zbar), np.log(errs), 1)[0]
    zeta_errs = []
    for v0 in rows:                                 # the disparity fill on the same masks: exact
        m = B.ellipse_mask(H, W, W * 0.5, v0, 14, 14)
        zf2, _, _ = S.fill(bg["zeta_true"], valid, m, PARAMS, K)
        zeta_errs.append(float(np.abs(zf2 - bg["zeta_true"])[S.dilate(m, PARAMS["r"])].max()))
    assert 2.7 <= slope <= 3.3, slope
    log(f"T2 PASS exact fill on planar ground: max |fill - true| inside omega {err:.1e}; harmonic residual "
        f"{resid_max:.1e} (both < 1e-12). DEPTH-DOMAIN comparison (same affine+harmonic machinery on z): "
        f"max error at mean z {', '.join(f'{z:.2f}: {e:.2e}' for z, e in zip(zbar, errs))}; log-log slope "
        f"{slope:.2f} (predicted 3, accepted [2.7, 3.3]); the disparity fill on the same masks: max "
        f"{max(zeta_errs):.1e}. The domain, not the solver, is the difference.")


def T3_maximum_principle():
    bg = B.make_background(bump=(150.0, 190.0, 25.0, 0.02))      # bump overlapping the mask
    H, W, K = bg["H"], bg["W"], bg["K"]
    valid = ground_valid(bg)
    om = S.dilate(bg["mask"], PARAMS["r"])
    known = valid & ~om
    ring = S.ring_of(om, PARAMS["w"], known)
    coef, _ = S.ring_fit(bg["zeta_true"], ring, H, W)
    base = S.eval_affine(coef, H, W)
    rk = np.where(known, bg["zeta_true"] - base, 0.0)
    resid = S.harmonic_solve(rk, om, known)
    # the Dirichlet data actually used: known 4-neighbours of omega
    nb = S.dilate(om, 1) & known
    lo, hi = rk[nb].min(), rk[nb].max()
    inside = resid[om]
    assert inside.min() >= lo - 1e-12 and inside.max() <= hi + 1e-12, (inside.min(), inside.max(), lo, hi)
    assert hi - lo > 1e-4                                        # the test has teeth
    log(f"T3 PASS maximum principle (plane + bump): solved residual in [{inside.min():.5f}, {inside.max():.5f}] "
        f"within boundary [{lo:.5f}, {hi:.5f}]")


def T4_coarsen_preserves_affine():
    bg = B.make_background()
    H, W, K = bg["H"], bg["W"], bg["K"]
    valid = np.ones((H, W), bool)                                # analytic field, all valid
    zc, vc = S.coarsen(bg["zeta_true"], valid, PARAMS)
    up = S.upsample_bicubic(zc, H, W)
    err = np.abs(up - bg["zeta_true"])
    # border-affected width: smallest margin beyond which err < 1e-6
    margin = next(m for m in range(0, 64) if err[m:H - m, m:W - m].max() < 1e-6)
    interior = err[margin:H - margin, margin:W - margin].max()
    # slope and horizon from the coarse grid itself (interior cells)
    cell = H // PARAMS["N"]
    uc = (np.arange(PARAMS["N"]) + 0.5) * cell
    Uc, Vc = np.meshgrid(uc, uc)
    sel = np.ones_like(zc, bool); sel[:2] = sel[-2:] = sel[:, :2] = sel[:, -2:] = False
    A = np.stack([Uc[sel], Vc[sel], np.ones(sel.sum())], 1)
    c, *_ = np.linalg.lstsq(A, zc[sel], rcond=None)
    rel = np.abs(c[1] - bg["affine"][1]) / abs(bg["affine"][1])
    # horizon row at the image centre column: alpha u + beta v + gamma = 0
    v_h_true = -(bg["affine"][0] * W / 2 + bg["affine"][2]) / bg["affine"][1]
    v_h_fit = -(c[0] * W / 2 + c[2]) / c[1]
    assert interior < 1e-6 and rel < 1e-4 and abs(v_h_fit - v_h_true) < cell * 0.5, (interior, rel, v_h_fit, v_h_true)
    log(f"T4 PASS coarsening preserves affine: interior round-trip error {interior:.1e} (< 1e-6) beyond a "
        f"{margin}-px border band; ground slope beta re-fit rel err {rel:.1e} (< 1e-4); horizon row "
        f"{v_h_fit:.2f} vs true {v_h_true:.2f} (sub-cell, cell = {cell} px)")


def T5_high_frequency():
    H = W = 256
    cell = H // PARAMS["N"]
    widths = [4, 8, 16, 32, 64, 128]
    surv = []
    for wpx in widths:
        bg = B.make_background(features=[(wpx, 0.01)])
        valid = np.ones((H, W), bool)
        zc, vc = S.coarsen(bg["zeta_true"], valid, PARAMS)
        up = S.upsample_bicubic(zc, H, W)
        plane = S.eval_affine(bg["affine"], H, W)
        m = 24
        a_in = (bg["zeta_true"] - plane)[m:-m, m:-m].std()
        a_out = (up - plane)[m:-m, m:-m].std()
        surv.append(a_out / a_in)
    assert all(surv[i] <= surv[i + 1] + 1e-9 for i in range(len(surv) - 1)), surv
    assert surv[0] < 0.1 and surv[-1] > 0.9, surv
    half = np.interp(0.5, surv, widths)
    log(f"T5 PASS high-frequency suppression (N={PARAMS['N']}, cell {cell} px, sigma1 {PARAMS['sigma1']}): surviving "
        f"amplitude by feature width " + ", ".join(f"{w}px: {s:.3f}" for w, s in zip(widths, surv))
        + f"; half-amplitude width {half:.1f} px (prediction: near one cell = {cell} px, wider than Nyquist because "
        f"sigma1 attenuates before the downsample)")


def T6_leakage():
    out = {}
    for label, kw in (("planar", {}), ("bump under object", {"bump": (150.0, 190.0, 25.0, 0.02)})):
        bg = B.make_background(**kw)
        zc, vc, rec = S.sanitise(bg["depth16"], bg["mask"], bg["K"], PARAMS, bg["valid"], HARNESS_TH,
                                 zeta_obj_true=bg["zeta_obj"])
        out[label] = rec["leakage"]
        assert rec["gates"]["g_leakage"]["outcome"] == "recorded"
    p = out["planar"]
    assert abs(p["corr_detrended"]) < HARNESS_TH["m2c_leakage_max"] or np.isnan(p["corr_detrended"]), p
    assert p["ridge_ratio"] < 1.5, p
    log("T6 leakage (recorded, never refuses): " + "; ".join(
        f"{k}: corr_raw {v['corr_raw']:+.3f}, corr_detrended {v['corr_detrended']:+.3f}, ridge_ratio {v['ridge_ratio']:.3f}"
        for k, v in out.items()) + ". Planar: no silhouette ridge, detrended correlation below the harness ceiling; "
        "the bump case is the one to watch on real clips (structure under the object is unobservable).")


def T7_halo():
    k = 3
    res = {}
    for r in (1, 4):
        bg = B.make_background(halo_px=k)
        prm = dict(PARAMS, r=r)
        zc, vc, rec = S.sanitise(bg["depth16"], bg["mask"], bg["K"], prm, bg["valid"], HARNESS_TH,
                                 zeta_obj_true=bg["zeta_obj"])
        up = S.upsample_bicubic(zc, bg["H"], bg["W"])
        inside = bg["mask"]
        res[r] = float(np.abs(up - bg["zeta_true"])[inside].max())
    assert res[4] < 1e-3 and res[1] > 3e-3 and res[1] > 5 * res[4], res
    log(f"T7 PASS dilation vs halo (halo k={k} px, object +0.03 disparity, halo +0.015): max |bg - true| inside the "
        f"mask with r=1 (< k): {res[1]:.2e}; with r=4 (>= k): {res[4]:.2e}. r does the job it is locked for.")


def T9_invalid():
    holes = [(120.0, 210.0, 12.0),      # overlaps the ring / omega of the default mask (mask centre 141,174)
             (60.0, 230.0, 10.0),       # ground hole away from the object
             (200.0, 60.0, 15.0)]       # sky (already invalid)
    bg = B.make_background(holes=holes)
    zc, vc, rec = S.sanitise(bg["depth16"], bg["mask"], bg["K"], PARAMS, bg["valid"], HARNESS_TH)
    assert np.isfinite(zc).all() and np.isfinite(vc.astype(float)).all()
    # ring fit ignores invalid: on an exact plane any valid subset gives the same coefficients
    bg0 = B.make_background()
    _, _, rec0 = S.sanitise(bg0["depth16"], bg0["mask"], bg0["K"], PARAMS, bg0["valid"], HARNESS_TH)
    d = np.abs(np.array(rec["affine"]) - np.array(rec0["affine"])).max() / np.abs(rec0["affine"]).max()
    assert d < 1e-3, d                                   # f16 noise on different ring pixels, not structure
    # the ground hole is filled (valid after fill); sky cells stay invalid and are counted
    assert rec["invalid_frac_coarse"] > 0.2 and rec["invalid_frac_in"] > 0.33
    filled_ground = [c for c in rec["components"] if not c["is_object"] and c["filled"]]
    assert len(filled_ground) >= 1
    log(f"T9 PASS invalid pixels: no nan/inf in outputs; ring fit with holes vs without differs {d:.1e} "
        f"(f16 pixel noise only); invalid_frac_in {rec['invalid_frac_in']:.3f} (sky + holes) -> invalid_frac_coarse "
        f"{rec['invalid_frac_coarse']:.3f} (sky only; ground holes filled: {len(filled_ground)} component(s)); "
        f"{rec['n_components']} unknown components solved against their own rings")


def T10_ring_gate():
    H = W = 256
    bg = B.make_background()
    strip = np.zeros((H, W), bool); strip[170:172, 60:200] = True
    zc, vc, rec = S.sanitise(bg["depth16"], strip, bg["K"], dict(PARAMS, r=1, w=1), bg["valid"], HARNESS_TH)
    assert rec["gates"]["g_ring_cond"]["outcome"] == "fail" and not rec["admitted"]
    assert zc.shape == (PARAMS["N"], PARAMS["N"])                # files/values still emitted
    # border-truncated mask: ring only on three sides — measured, the SKILL expects it to fire
    border = B.ellipse_mask(H, W, 128.0, 256.0, 30.0, 40.0)      # centred on the bottom edge
    zc2, vc2, rec2 = S.sanitise(bg["depth16"], border, bg["K"], PARAMS, bg["valid"], HARNESS_TH)
    _, _, rec0 = S.sanitise(bg["depth16"], bg["mask"], bg["K"], PARAMS, bg["valid"], HARNESS_TH)
    log(f"T10 ring conditioning gate: thin strip cond {rec['ring_cond']:.0f} -> g_ring_cond FAIL, refused, "
        f"outputs still emitted (PASS). Border-truncated mask: cond {rec2['ring_cond']:.1f} vs compact {rec0['ring_cond']:.1f} "
        f"-> {rec2['gates']['g_ring_cond']['outcome']} at the harness ceiling {HARNESS_TH['m2c_ring_cond_max']}. "
        + ("FINDING: a three-sided ring is worse-conditioned but NOT singular; whether it should refuse is a "
           "threshold decision, and the truncation itself is m2_a's gate." if rec2["admitted"] else ""))


def T8_fill_order():
    out = {}
    for label, kw in (("planar", {}), ("bump", {"bump": (150.0, 190.0, 25.0, 0.02)})):
        bg = B.make_background(**kw)
        H, W, K = bg["H"], bg["W"], bg["K"]
        zeta, ok = S.disparity(bg["depth16"], bg["valid"])
        # fill-then-coarsen (the module)
        zf, vf, _ = S.fill(zeta, ok, bg["mask"], PARAMS, K)
        zc_a, vc_a = S.coarsen(zf, vf, PARAMS)
        # coarsen-then-fill: coarsen with omega invalid, then fill on the coarse grid.
        # A cell PARTLY covered by omega averages only its valid pixels, i.e. it reports the
        # value at the valid-pixel centroid, not the cell centre: a bias of |grad zeta| x up to
        # half a cell (measured 0.011 on the plane, 2026-09-11). So order B must treat any
        # partially covered cell as unknown, or the comparison is not apples to apples.
        om = S.dilate(bg["mask"], PARAMS["r"])
        N = PARAMS["N"]
        # ... and the same bias one stage down: the normalised Gaussian is one-sided within
        # 3*sigma1 of an unknown pixel (bias <= sigma1 |grad zeta|, measured 2.3e-3 residual),
        # so B's known cells are those whose whole blur support is valid
        reach = int(np.ceil(3 * PARAMS["sigma1"]))
        unknown_px = S.dilate(om | ~ok, reach)
        frac_unknown = S.area_downsample(unknown_px.astype(float), np.ones_like(om), N)[0]
        full_valid = frac_unknown == 0
        zc_b0, vc_b0 = S.coarsen(zeta, ok & ~om, PARAMS)
        vc_b0 &= full_valid
        mask_c = (S.area_downsample(om.astype(float), np.ones_like(om), N)[0] > 0)
        zc_b, vc_b, _ = S.fill(zc_b0, vc_b0, mask_c, {"r": 0, "w": 1}, None)
        both = vc_a & vc_b
        diff = np.abs(zc_a - zc_b)[both]
        out[label] = (float(diff.max()), float(np.sqrt((diff ** 2).mean())))
    assert out["planar"][0] < 1e-3, out                          # f16 noise only
    log(f"T8 fill order (REPORTED, no recommendation): planar max|A-B| {out['planar'][0]:.1e} RMS {out['planar'][1]:.1e} "
        f"(indistinguishable up to f16 noise); bump max {out['bump'][0]:.2e} RMS {out['bump'][1]:.2e} on the coarse grid "
        f"(disparity units; compare with the bump amplitude 0.02). Human sets fill_order in the lock. "
        "FINDING (both orders): a coarse cell with partial validity reports the value at its valid-pixel centroid, "
        "not its centre (bias |grad zeta| x <= half a cell, 0.011 here), and the normalised Gaussian carries the same "
        "one-sided bias within 3 sigma1 of any invalid pixel (<= sigma1 |grad zeta|, 2.3e-3 here). Both act wherever "
        "validity is partial — the sky boundary in the canonical order. Recorded; the remedy would be a per-cell "
        "validity-fraction flag that C3 can weight by.")


if __name__ == "__main__":
    T2_exact_fill_and_depth_domain()
    T3_maximum_principle()
    T4_coarsen_preserves_affine()
    T5_high_frequency()
    T6_leakage()
    T7_halo()
    T9_invalid()
    T10_ring_gate()
    T8_fill_order()
    REPORT.append("")
    REPORT.append("HARD STOP (SKILL §8 step 6): human review before C2 runs on any real clip or any lock value is written.")
    Path("docs/report_m2c_c2_tests.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print("\nall C2 harness tests passed -> docs/report_m2c_c2_tests.md")
