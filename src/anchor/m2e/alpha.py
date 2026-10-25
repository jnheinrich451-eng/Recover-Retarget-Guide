"""m2_e — the scale bridge `alpha` (SKILL m2_e v2 §2, amended by review items 2, 3, 8, 9, 12 on 2026-09-22).

Depth-Anything on the anchor gives a relative disparity zeta_hat; the control's disparity zeta_ctrl is in the source gauge.
Model (estimator error on the estimator's side):   zeta_hat(p) = (zeta_ctrl(p) - b) / a + e(p)
Fit with the EXACT variable as the regressor and invert:  zeta_hat ~ c zeta_ctrl + d,  alpha = a = 1/c,  b = -d/c.
Never regress zeta_ctrl on zeta_hat: an OLS slope is attenuated by noise in its regressor (dilution, established on real data in
the 404 investigation); the naive direction is computed and RECORDED as Depth-Anything's empirical lambda, never used.

WHERE to fit (review item 3): NOT over the object. Over M, zeta_ctrl is the proxy's rendered disparity, and an affine (a, b)
fitted there absorbs part of a genuine yaw difference between the generated object and the proxy (a shift b tilts a plane's
normal toward the optical axis), so R2 would reconstruct a point map already rotated toward the proxy and bias A toward identity.
The estimator's ambiguity is one (a, b) per image, so it is fitted on a GROUND BAND around the object — an annulus outside
dilate(anchor_mask | M_obj), below the horizon, on observed Stage A depth (not C2's filled hole), not sky — where zeta_ctrl is
C2's sanitised background in the source gauge and no proxy exists. The object-mask fit is kept ALONGSIDE as a check: agreement
tests the global-affine assumption; disagreement flags a within-object scale error (flattening). If the band's interval is too
wide the object fit is the fallback and delta is flagged "possibly proxy-pulled".
"""
from __future__ import annotations

import numpy as np

from ..m2d import instrument

FIT_DIRECTION = "zeta_hat_on_zeta_ctrl"


def below_horizon(K, n_plane, H: int, W: int) -> np.ndarray:
    """Pixels whose ray meets the contact plane in front of the camera: n . r < 0 with r = K^-1 [u, v, 1] (camera above
    the plane, n.X + d = 0 with d > 0)."""
    K = np.asarray(K, np.float64); n = np.asarray(n_plane, np.float64); n = n / np.linalg.norm(n)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    r = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    return (r @ n) < 0


def ground_band(anchor_mask, M_obj, K, n_plane, valid, hole, inner_px: int, outer_frac: float) -> tuple[np.ndarray, dict]:
    """The proxy-free fit region. inner: dilate(anchor | M_obj) by inner_px; outer: dilate by outer_frac x object height."""
    import cv2
    A = np.asarray(anchor_mask, bool); O = np.asarray(M_obj, bool); H, W = O.shape
    assert A.shape == O.shape and np.asarray(valid).shape == O.shape and np.asarray(hole).shape == O.shape
    both = (A | O).astype(np.uint8)
    rows = np.nonzero(both.any(1))[0]; obj_h = int(rows.max() - rows.min() + 1) if len(rows) else 0
    assert obj_h > 0, "empty object"
    k_in = 2 * int(inner_px) + 1; k_out = 2 * int(round(outer_frac * obj_h)) + 1
    inner = cv2.dilate(both, np.ones((k_in, k_in), np.uint8)).astype(bool)
    outer = cv2.dilate(both, np.ones((k_out, k_out), np.uint8)).astype(bool)
    band = outer & ~inner & below_horizon(K, n_plane, H, W) & np.asarray(valid, bool) & ~np.asarray(hole, bool)
    prov = {"definition": "annulus(dilate(anchor|M_obj), inner_px, outer_frac*obj_h) & below_horizon & valid & ~hole",
            "inner_px": int(inner_px), "outer_px": int(round(outer_frac * obj_h)), "object_height_px": obj_h,
            "n_pixels": int(band.sum()), "frac_below_horizon": float(below_horizon(K, n_plane, H, W).mean())}
    return band, prov


def affine_fit(zeta_hat, zeta_ref, region) -> dict:
    """zeta_hat ~ c zeta_ref + d over the region; alpha = 1/c, b = -d/c. Also the NAIVE direction (recorded, never used)."""
    m = np.asarray(region, bool) & np.isfinite(zeta_hat) & np.isfinite(zeta_ref)
    x = np.asarray(zeta_ref, np.float64)[m]; y = np.asarray(zeta_hat, np.float64)[m]
    n = int(m.sum()); assert n >= 3, f"fit region has {n} pixels"
    xm, ym = x.mean(), y.mean(); sxx = ((x - xm) ** 2).sum(); syy = ((y - ym) ** 2).sum(); sxy = ((x - xm) * (y - ym)).sum()
    c = sxy / max(sxx, 1e-300); d = ym - c * xm
    resid = y - (c * x + d)
    span = float(np.percentile(x, 98) - np.percentile(x, 2))
    out = {"fit_direction": FIT_DIRECTION, "c": float(c), "d": float(d), "n_pixels": n, "resid_rms": float(np.sqrt((resid ** 2).mean())),
           "disparity_span": span, "corr": float(sxy / np.sqrt(max(sxx * syy, 1e-300)))}
    out["alpha"] = float(1.0 / c) if c != 0 else float("nan"); out["b"] = float(-d / c) if c != 0 else float("nan")
    c_n = sxy / max(syy, 1e-300); d_n = xm - c_n * ym                      # naive: zeta_ref on zeta_hat (regressor = the noisy one)
    out["alpha_naive"] = float(c_n); out["b_naive"] = float(d_n)
    out["naive_over_reversed"] = float(c_n * c) if c != 0 else float("nan")   # = empirical lambda of the estimator on this region
    return out


def bootstrap_alpha(zeta_hat, zeta_ref, region, block: int, n_boot: int = 300, seed: int = 0) -> dict:
    """Interval on alpha. block = 0: pixel bootstrap; block > 0: block bootstrap over block x block tiles (the estimator's error
    is a smooth field, so pixels are not independent). Reports the block count; fewer than ~10 is flagged unreliable."""
    reg = np.asarray(region, bool); H, W = reg.shape
    zh = np.asarray(zeta_hat, np.float64); zr = np.asarray(zeta_ref, np.float64)
    rng = np.random.default_rng(seed)
    if block <= 0:
        jj, ii = np.nonzero(reg); units = [np.array([k]) for k in range(len(jj))]; flat_j, flat_i = jj, ii
    else:
        by, bx = np.meshgrid(np.arange(0, H, block), np.arange(0, W, block), indexing="ij")
        units, flat_j, flat_i = [], [], []
        for y0, x0 in zip(by.ravel(), bx.ravel()):
            sub = reg[y0:y0 + block, x0:x0 + block]
            if sub.sum() >= max(4, 0.05 * block * block):
                jj, ii = np.nonzero(sub); units.append(np.arange(len(flat_j), len(flat_j) + len(jj))); flat_j += (jj + y0).tolist(); flat_i += (ii + x0).tolist()
        flat_j, flat_i = np.array(flat_j), np.array(flat_i)
    n_units = len(units)
    if n_units < 2:
        return {"block": int(block), "n_units": n_units, "ci": [float("nan"), float("nan")], "unreliable": True}
    x_all = zr[flat_j, flat_i]; y_all = zh[flat_j, flat_i]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_units, n_units)
        sel = np.concatenate([units[k] for k in pick])
        x, y = x_all[sel], y_all[sel]
        if np.ptp(x) < 1e-12:
            continue
        c = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
        vals.append(1.0 / c if c != 0 else np.nan)
    vals = np.array(vals); vals = vals[np.isfinite(vals)]
    lo, hi = (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if len(vals) > 10 else (np.nan, np.nan)
    return {"block": int(block), "n_units": int(n_units), "ci": [float(lo), float(hi)], "unreliable": bool(n_units < 10 or len(vals) <= 10)}


def _gate(value, threshold, cond):
    if threshold is None:
        return {"value": value, "threshold": None, "outcome": "ungated"}
    return {"value": value, "threshold": threshold, "outcome": "pass" if cond(threshold) else "fail"}


def fit_alpha(zeta_hat, zeta_ctrl, band, object_region, th: dict, blocks=(32, 64), n_boot=300) -> dict:
    """Band fit (primary) + UNCORRECTED object-mask fit (against the proxy at R_A: recorded as a check only. It absorbs the
    anchor-vs-proxy yaw, so its ratio to the band is NOT a flattening diagnostic; the delta-corrected object fit and the fallback
    decision are made in the delta step, `decide_alpha`). `alpha`/`b` are the band values to USE when the band is positive with
    an interval excluding zero; otherwise None until the delta step decides the fallback."""
    band_fit = affine_fit(zeta_hat, zeta_ctrl, band)
    obj_fit = affine_fit(zeta_hat, zeta_ctrl, object_region)
    boots_band = {f"block_{bk}": bootstrap_alpha(zeta_hat, zeta_ctrl, band, bk, n_boot) for bk in blocks}
    boots_band["pixel"] = bootstrap_alpha(zeta_hat, zeta_ctrl, band, 0, min(n_boot, 100))
    boots_obj = {f"block_{bk}": bootstrap_alpha(zeta_hat, zeta_ctrl, object_region, bk, n_boot) for bk in blocks}
    ci_band = boots_band[f"block_{blocks[0]}"]["ci"]; ci_obj = boots_obj[f"block_{blocks[0]}"]["ci"]
    def positive(fit, ci):
        return bool(fit["c"] > 0 and np.isfinite(ci).all() and ci[0] > 0)
    rel_w = lambda fit, ci: float((ci[1] - ci[0]) / abs(fit["alpha"])) if np.isfinite(ci).all() and fit["alpha"] else float("inf")   # noqa: E731
    band_ok = positive(band_fit, ci_band); obj_ok = positive(obj_fit, ci_obj)
    ci_max = th.get("m2e_alpha_ci_rel_max")
    band_wide = (ci_max is not None and rel_w(band_fit, ci_band) > ci_max) or not np.isfinite(ci_band).all()
    if band_ok and not band_wide:
        used, source, flag = band_fit, "ground_band", None
    else:                                                                   # the fallback needs the delta-corrected object fit (delta step)
        used, source, flag = None, None, ("band_interval_wide" if band_ok else "band_not_positive")
    rho_obj = instrument.spearman(zeta_hat, zeta_ctrl, mask=object_region)
    rec = {"band_fit": band_fit, "object_fit_uncorrected": obj_fit, "band_bootstrap": boots_band, "object_bootstrap_uncorrected": boots_obj,
           "alpha": None if used is None else used["alpha"], "b": None if used is None else used["b"], "alpha_source": source, "flag": flag,
           "alpha_band": band_fit["alpha"], "alpha_object_uncorrected": obj_fit["alpha"],
           "band_vs_object_uncorrected_ratio": float(band_fit["alpha"] / obj_fit["alpha"]) if obj_fit["alpha"] not in (0, None) and np.isfinite(obj_fit["alpha"]) else float("nan"),
           "spearman_over_object": rho_obj, "alpha_ci_rel_width_band": rel_w(band_fit, ci_band),
           "gates": {"g_alpha_positive": {"band": band_ok, "object": obj_ok, "outcome": "pass" if used is not None else "fail"},
                     "g_alpha_ci": _gate(rel_w(band_fit, ci_band), ci_max, lambda t: rel_w(band_fit, ci_band) <= t),
                     "g_alpha_pixels": _gate(band_fit["n_pixels"], th.get("m2e_alpha_pixels_min"), lambda t: band_fit["n_pixels"] >= t)}}
    rec["alpha_low_confidence"] = bool(rec["gates"]["g_alpha_pixels"]["outcome"] == "fail" or boots_band[f"block_{blocks[0]}"]["unreliable"])
    return rec


def decide_alpha(rec: dict, zeta_hat, zeta_proxy_corrected, object_region, th: dict, blocks=(32, 64), n_boot=300) -> dict:
    """Delta step (reviewer 2026-09-23): the object comparison is made against the proxy ROTATED by the measured delta, so the
    shift has no yaw difference to absorb; a band/object ratio away from 1 now genuinely means within-object scale error
    (flattening), and the fallback (band interval too wide) is no longer proxy-pulled."""
    obj = affine_fit(zeta_hat, zeta_proxy_corrected, object_region)
    boots = {f"block_{bk}": bootstrap_alpha(zeta_hat, zeta_proxy_corrected, object_region, bk, n_boot) for bk in blocks}
    ci = boots[f"block_{blocks[0]}"]["ci"]; obj_ok = bool(obj["c"] > 0 and np.isfinite(ci).all() and ci[0] > 0)
    out = dict(rec); out["object_fit_corrected"] = obj; out["object_bootstrap_corrected"] = boots
    out["alpha_object_corrected"] = obj["alpha"]
    out["band_vs_object_ratio"] = float(rec["band_fit"]["alpha"] / obj["alpha"]) if obj["alpha"] and np.isfinite(obj["alpha"]) else float("nan")
    if rec["alpha_source"] == "ground_band":
        out["flag"] = None if abs(out["band_vs_object_ratio"] - 1) < 0.1 else "within_object_scale_error"
    elif obj_ok:
        out["alpha"], out["b"], out["alpha_source"], out["flag"] = obj["alpha"], obj["b"], "object_mask_fallback_delta_corrected", rec["flag"]
    else:
        out["alpha"], out["b"], out["alpha_source"], out["flag"] = None, None, None, "alpha_undetermined"
    out["gates"] = {**rec["gates"], "g_alpha_positive": {**rec["gates"]["g_alpha_positive"], "object_corrected": obj_ok, "outcome": "pass" if out["alpha"] is not None else "fail"}}
    return out


def anchor_pointmap(zeta_hat, a: float, b: float, K, valid=None, z_bg=None, region=None, max_bad_frac: float = 0.05):
    """R2's point map: the ANCHOR's depth brought into the source gauge, depth = 1 / (a zeta_hat + b). Never zeta_ctrl (§3).

    Positivity is required over `region` — the object (anchor mask | M_obj) and the band the bridge was fitted on — not over the
    whole frame: the affine bridge holds where it was fitted, and the estimator's relative disparity saturates toward zero in the
    FAR field, where a zeta_hat + b goes negative (first run, 035: 43.9 % of valid pixels, all far background). Far pixels are not
    part of the object's gauge; they get the flat background depth opencv_pointmap gives invalid pixels. The far-field fraction is
    RECORDED (`far_field_nonpositive_frac`) as the measured reach of the affine model, not hidden."""
    from ..m2b2.pointmap import opencv_pointmap
    zh = np.asarray(zeta_hat, np.float64)
    disp = a * zh + b
    val = np.ones_like(zh, bool) if valid is None else np.asarray(valid, bool)
    ok = np.isfinite(disp) & (disp > 0) & val
    depth = np.where(ok, 1.0 / np.where(ok, disp, 1.0), np.nan)
    frac_bad_all = float(1 - ok[val].mean()) if val.any() else 1.0
    reg = val if region is None else (np.asarray(region, bool) & val)
    frac_bad = float(1 - ok[reg].mean()) if reg.any() else 1.0
    assert frac_bad < max_bad_frac, f"alpha-scaled disparity non-positive on {100 * frac_bad:.1f} % of the object/band region (limit {100 * max_bad_frac:.0f} %)"
    zb = float(np.nanmedian(depth[ok]) * 3) if z_bg is None else float(z_bg)
    return opencv_pointmap(np.where(ok, depth, zb), ok, K, zb), depth, {"pointmap_source": "alpha_scaled_zeta_hat", "a": float(a), "b": float(b),
                                                                        "region_nonpositive_frac": frac_bad, "far_field_nonpositive_frac": frac_bad_all, "z_bg": zb}


class PointMapCircular(RuntimeError):
    """The point map handed to R2 is (numerically) the control's own disparity — the circularity §3 forbids."""


def assert_not_proxy(depth, zeta_obj, M_obj, rel_tol: float = 1e-6):
    """Tripwire (review item 8): over M_obj, if 1/depth is affinely identical to zeta_obj (residual relative to the disparity
    span below rel_tol), the point map IS the proxy's, whatever it was labelled. Numerical identity, not R^2."""
    m = np.asarray(M_obj, bool) & np.isfinite(depth) & np.isfinite(zeta_obj) & (np.asarray(depth) > 0)
    x = np.asarray(zeta_obj, np.float64)[m]; y = 1.0 / np.asarray(depth, np.float64)[m]
    if len(x) < 10 or np.ptp(x) < 1e-12:
        return {"checked": False}
    A = np.stack([x, np.ones_like(x)], 1); coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.sqrt(((y - A @ coef) ** 2).mean())); span = float(np.percentile(y, 98) - np.percentile(y, 2))
    rel = resid / max(span, 1e-300)
    if rel < rel_tol:
        raise PointMapCircular(f"point map is affinely identical to zeta_obj over M_obj (relative residual {rel:.1e})")
    return {"checked": True, "relative_residual": rel}


def pointmap_source_guard(source: str):
    if str(source).lower() in ("zeta_ctrl", "control", "proxy", "zeta_obj"):
        raise PointMapCircular(f"pointmap_source {source!r} is the control's own disparity (SKILL m2_e §3)")
