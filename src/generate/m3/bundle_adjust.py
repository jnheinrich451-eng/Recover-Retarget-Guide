"""Bundle adjustment for camera C (reviewer 2026-10-07b): cameras and static 3-D points refined jointly from their multi-view
track observations, with the gauge fixed at t_a. CPU, scipy sparse Levenberg-Marquardt (trust-region reflective on a sparse
Jacobian pattern). Rotations as axis-angle; camera 0 (= t_a) fixed to the identity; scale fixed by a prior on the lifted
depths of a handful of frame-t_a points (the lift's gauge is a PRIOR, not a constraint on every point).

Residuals: reprojection (u, v) per observation, in pixels at the tracker's resolution; prior: (z_i - z_lift_i) / z_lift_i
for the anchor points, weighted by `prior_weight` pixels per unit relative error.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def _rodrigues(rv: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    S = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * S + (1 - np.cos(th)) * S @ S


def _rotvec(R: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.Rodrigues(np.asarray(R, np.float64))[0][:, 0]


def bundle_adjust(E0: np.ndarray, X0: np.ndarray, obs_cam: np.ndarray, obs_pt: np.ndarray, obs_uv: np.ndarray, K: np.ndarray,
                  anchor_pts: np.ndarray, prior_weight: float, max_nfev: int = 200, loss: str = "huber", f_scale: float = 2.0,
                  free_focal: bool = False):
    """E0 [F,4,4] initial world->camera (E0[0] = I kept fixed), X0 [N,3] initial world points, observations (camera index,
    point index, pixel) and per-camera K [F,3,3]. anchor_pts: indices of points whose initial depth (in camera 0) is the scale
    prior. Returns E [F,4,4], X [N,3], info dict."""
    F, N = len(E0), len(X0)
    assert np.allclose(E0[0], np.eye(4)), "gauge: camera 0 must be the identity (world = camera-t_a)"
    rv0 = np.stack([_rotvec(E0[c, :3, :3]) for c in range(1, F)])
    tv0 = E0[1:, :3, 3]
    # free_focal (reviewer 2026-10-07k, the Stage-A confirming test): one shared log focal scale s, fx,fy -> e^s fx, e^s fy on every
    # frame; a camera-model error shows as s moving by several percent with the residual collapsing, a correct K keeps s within ~1 %.
    p0 = np.concatenate([rv0.ravel(), tv0.ravel(), X0.ravel()] + ([np.zeros(1)] if free_focal else []))
    z_prior = X0[anchor_pts, 2].copy()
    n_obs = len(obs_cam)
    n_core = 6 * (F - 1) + 3 * N

    def unpack(p):
        rv = p[: 6 * (F - 1)][: 3 * (F - 1)].reshape(F - 1, 3)
        tv = p[3 * (F - 1): 6 * (F - 1)].reshape(F - 1, 3)
        X = p[6 * (F - 1): n_core].reshape(N, 3)
        return rv, tv, X

    def residuals(p):
        rv, tv, X = unpack(p)
        fs = float(np.exp(p[n_core])) if free_focal else 1.0
        R = np.concatenate([np.eye(3)[None], np.stack([_rodrigues(r) for r in rv])])
        t = np.concatenate([np.zeros((1, 3)), tv])
        Xc = np.einsum("oij,oj->oi", R[obs_cam], X[obs_pt]) + t[obs_cam]
        z = np.maximum(Xc[:, 2], 1e-6)
        u = fs * K[obs_cam, 0, 0] * Xc[:, 0] / z + K[obs_cam, 0, 2]
        v = fs * K[obs_cam, 1, 1] * Xc[:, 1] / z + K[obs_cam, 1, 2]
        r_rep = np.concatenate([u - obs_uv[:, 0], v - obs_uv[:, 1]])
        r_prior = prior_weight * (X[anchor_pts, 2] - z_prior) / z_prior
        return np.concatenate([r_rep, r_prior])

    # sparsity: each reprojection residual depends on its camera's 6 params (unless camera 0) and its point's 3
    m = 2 * n_obs + len(anchor_pts)
    S = lil_matrix((m, len(p0)), dtype=int)
    for k in range(n_obs):
        c, j = obs_cam[k], obs_pt[k]
        for row in (k, n_obs + k):
            if c > 0:
                S[row, 3 * (c - 1): 3 * c] = 1
                S[row, 3 * (F - 1) + 3 * (c - 1): 3 * (F - 1) + 3 * c] = 1
            S[row, 6 * (F - 1) + 3 * j: 6 * (F - 1) + 3 * j + 3] = 1
    for i, j in enumerate(anchor_pts):
        S[2 * n_obs + i, 6 * (F - 1) + 3 * j + 2] = 1
    if free_focal:
        S[: 2 * n_obs, n_core] = 1
    r0 = residuals(p0)
    # Two stages (measured on the synthetic test, 2026-10-07): a robust loss from the start stalls at the initial residual level
    # (huber f_scale 2 px: 1.65 px, status 3); plain least squares converges to the noise floor in 7 evaluations. So: plain first,
    # then one robust refit from that solution with f_scale set by the residual spread (3 x MAD), which only trims outliers.
    sol = least_squares(residuals, p0, jac_sparsity=S, method="trf", loss="linear", max_nfev=max_nfev, x_scale="jac", verbose=0)
    stages = [{"loss": "linear", "nfev": int(sol.nfev), "status": int(sol.status)}]
    if loss != "linear":
        rr = residuals(sol.x)[: 2 * n_obs]
        fs = max(float(f_scale), 3.0 * 1.4826 * float(np.median(np.abs(rr - np.median(rr)))))
        sol2 = least_squares(residuals, sol.x, jac_sparsity=S, method="trf", loss=loss, f_scale=fs, max_nfev=max_nfev, x_scale="jac", verbose=0)
        if sol2.cost <= sol.cost * 1.5:                       # the robust refit may not blow up the fit; otherwise keep the plain solution
            sol = sol2
        stages.append({"loss": loss, "f_scale": fs, "nfev": int(sol2.nfev), "status": int(sol2.status), "kept": bool(sol is sol2)})
    rv, tv, X = unpack(sol.x)
    E = np.tile(np.eye(4), (F, 1, 1))
    for c in range(1, F):
        E[c, :3, :3] = _rodrigues(rv[c - 1]); E[c, :3, 3] = tv[c - 1]
    r1 = residuals(sol.x)
    info = {"nfev": int(sum(s["nfev"] for s in stages)), "status": int(sol.status), "stages": stages, "reproj_rms_before": float(np.sqrt(np.mean(r0[: 2 * n_obs] ** 2))),
            "reproj_rms_after": float(np.sqrt(np.mean(r1[: 2 * n_obs] ** 2))), "n_obs": int(n_obs), "n_points": int(N), "n_cams": int(F),
            "prior_rms_rel_after": float(np.sqrt(np.mean((r1[2 * n_obs:] / prior_weight) ** 2))) if len(anchor_pts) else None,
            "focal_scale": float(np.exp(sol.x[n_core])) if free_focal else None}
    return E, X, info


def triangulate_dlt(E: np.ndarray, K: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of one point from n >= 2 views. E [n,4,4] world->camera, K [n,3,3], uv [n,2] pixels.
    Rows are built in K-normalised coordinates (unit focal) so pixel magnitudes do not dominate the SVD. Returns X_world [3]."""
    E = np.asarray(E, np.float64); K = np.asarray(K, np.float64); uv = np.asarray(uv, np.float64)
    assert E.shape[1:] == (4, 4) and K.shape[1:] == (3, 3) and uv.shape == (E.shape[0], 2) and E.shape[0] >= 2, (E.shape, K.shape, uv.shape)
    rows = []
    for k in range(E.shape[0]):
        xn = np.linalg.solve(K[k], np.array([uv[k, 0], uv[k, 1], 1.0]))
        Pm = E[k, :3, :]
        rows.append(xn[0] * Pm[2] - Pm[0]); rows.append(xn[1] * Pm[2] - Pm[1])
    Xh = np.linalg.svd(np.array(rows))[2][-1]
    if abs(Xh[3]) <= 1e-12:
        raise ValueError("point at infinity (rays parallel: no baseline)")
    X = Xh[:3] / Xh[3]
    if not np.all(np.isfinite(X)):
        raise ValueError("non-finite triangulation")
    return X


def _project_nan(X: np.ndarray, E: np.ndarray, K: np.ndarray) -> np.ndarray:
    """X [3] world -> pixels under each of E [n,4,4], K [n,3,3]; NaN where the point is behind the camera."""
    Xc = np.einsum("nij,j->ni", E[:, :3, :3], X) + E[:, :3, 3]
    uv = np.einsum("nij,nj->ni", K, Xc)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(Xc[:, 2:3] > 0, uv[:, :2] / uv[:, 2:3], np.nan)


def _boot_ci(vals: np.ndarray, rng, n_boot: int):
    boots = np.array([np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)])
    return float(np.median(vals)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _equivalence(lo: float, hi: float, tol: float) -> str:
    """Three outcomes (reviewer 2026-10-07i): the whole interval inside 1 +- tol is a pass (equivalence shown); entirely outside is a
    fail (wrong scale); straddling a bound is undetermined. A median alone cannot support an equivalence claim."""
    if lo >= 1.0 - tol and hi <= 1.0 + tol:
        return "pass"
    if hi < 1.0 - tol or lo > 1.0 + tol:
        return "fail"
    return "undetermined"


def triangulated_heldout(E: np.ndarray, K: np.ndarray, P: np.ndarray, Q: np.ndarray, excl: np.ndarray, Z_lift: np.ndarray,
                         gate_px: float, gauge_tol: float, min_obs: int = 8, min_span=None, n_boot: int = 1000, seed: int = 0,
                         min_points: int = 30, return_points: bool = False, X_lift=None) -> dict:
    """The lift-free read of a camera over a pair window (reviewer 2026-10-07h: gate of record for every clip and every camera).
    E [F,4,4] world->camera with E[0] = I (world = camera t_a); K [F,3,3]; P [n,F,2] tracked pixels; Q [n,F] valid; excl [n,F]
    inside the object mask; Z_lift [n] single-frame lifted depth at t_a (NaN where absent); gate_px the lift gate's threshold.
      condition 1 (split in time): each held-out point is triangulated from its EVEN-k observations and scored on its ODD-k
        observations; a point needs >= min_obs observations spanning >= min_span frames (default half the window);
      condition 3 (gauge): triangulation is blind to global scale, so Z_tri / Z_lift at t_a is an equivalence claim: the bootstrap
        95 % interval of its median must lie wholly within gauge_tol of 1 (lock m3.trajectory.triangulated_gauge_tol); wholly
        outside = wrong scale; straddling = undetermined. The ratio is also given per lifted-depth tercile (near/mid/far): if the
        tercile medians differ by more than gauge_tol the gauge is "depth-dependent" and the gate is read on the NEAR tercile,
        where Module 3 takes its scale (the ground band behind k_bg and the object's placement).
      fewer than min_points scorable points -> undetermined, whatever the interval says.
    Points whose triangulation lands behind camera t_a or has no baseline are counted, never scored."""
    n, F = Q.shape
    assert E.shape == (F, 4, 4) and K.shape == (F, 3, 3) and P.shape == (n, F, 2) and excl.shape == (n, F) and Z_lift.shape == (n,)
    assert np.allclose(E[0], np.eye(4), atol=1e-9), "E[0] must be the identity (world = camera t_a)"
    min_span = (F - 1) // 2 if min_span is None else int(min_span)
    errs, ratios, zl, pts = [], [], [], []
    drops = {"too_few_obs": 0, "span_too_short": 0, "too_few_even_or_odd": 0, "no_baseline": 0, "behind_camera": 0}
    for i in range(n):
        ks = np.nonzero(Q[i] & ~excl[i])[0]
        if len(ks) < min_obs:
            drops["too_few_obs"] += 1; continue
        if ks.max() - ks.min() < min_span:
            drops["span_too_short"] += 1; continue
        ev, od = ks[ks % 2 == 0], ks[ks % 2 == 1]
        if len(ev) < 2 or len(od) < 1:
            drops["too_few_even_or_odd"] += 1; continue
        try:
            X = triangulate_dlt(E[ev], K[ev], P[i, ev])
        except ValueError:
            drops["no_baseline"] += 1; continue
        e = np.linalg.norm(_project_nan(X, E[od], K[od]) - P[i, od], axis=1)
        if X[2] <= 0 or not np.isfinite(e).all():
            drops["behind_camera"] += 1; continue
        errs.append(float(np.median(e)))
        if np.isfinite(Z_lift[i]) and Z_lift[i] > 0:
            ratios.append(float(X[2] / Z_lift[i])); zl.append(float(Z_lift[i]))
        if return_points:
            res = _project_nan(X, E[od], K[od]) - P[i, od]                   # signed residual per odd frame (predicted - observed)
            k0 = int(ks.min())
            pts.append({"track": int(i), "uv_first": [float(P[i, k0, 0]), float(P[i, k0, 1])], "k_first": k0,
                        "uv_ta": [float(P[i, 0, 0]), float(P[i, 0, 1])] if (Q[i, 0] and not excl[i, 0]) else None,
                        "Z_lift": float(Z_lift[i]) if np.isfinite(Z_lift[i]) else None, "Z_tri": float(X[2]), "X_tri": [float(v) for v in X],
                        "X_lift": None if (X_lift is None or not np.all(np.isfinite(X_lift[i]))) else [float(v) for v in X_lift[i]],
                        "err_median": float(np.median(e)), "resid_mean": [float(res[:, 0].mean()), float(res[:, 1].mean())], "n_odd": int(len(od))})
    errs, ratios, zl = np.array(errs), np.array(ratios), np.array(zl)
    out = {"median_px": float(np.median(errs)) if len(errs) else None, "p90_px": float(np.quantile(errs, 0.9)) if len(errs) else None,
           "n_points": int(len(errs)), "dropped": drops, "min_obs": min_obs, "min_span": min_span, "split": "even k triangulate, odd k score"}
    rng = np.random.default_rng(seed)
    if len(ratios) >= 5:
        med, lo, hi = _boot_ci(ratios, rng, n_boot)
        g = {"median_Ztri_over_Zlift": med, "ci95": [lo, hi], "n": int(len(ratios)), "tol": gauge_tol, "outcome": _equivalence(lo, hi, gauge_tol),
             "read_on": "all"}
        order = np.argsort(zl); parts = np.array_split(order, 3)
        g["terciles"] = {name: {"median": float(np.median(ratios[idx])), "n": int(len(idx)), "z_lift_range": [float(zl[idx].min()), float(zl[idx].max())]}
                         for name, idx in zip(("near", "mid", "far"), parts) if len(idx)}
        tm = [v["median"] for v in g["terciles"].values()]
        g["depth_dependent"] = bool(len(tm) == 3 and (max(tm) - min(tm)) > gauge_tol)
        if g["depth_dependent"] and len(parts[0]) >= 5:
            nmed, nlo, nhi = _boot_ci(ratios[parts[0]], rng, n_boot)
            g["near"] = {"median_Ztri_over_Zlift": nmed, "ci95": [nlo, nhi], "n": int(len(parts[0]))}
            g["outcome"] = _equivalence(nlo, nhi, gauge_tol); g["read_on"] = "near"
    else:
        g = {"median_Ztri_over_Zlift": None, "ci95": None, "n": int(len(ratios)), "tol": gauge_tol, "outcome": "undetermined", "read_on": "all",
             "terciles": {}, "depth_dependent": None}
    out["gauge"] = g
    if return_points:
        out["points"] = pts
    out["min_points"] = int(min_points)
    out["reproj_ok"] = bool(out["median_px"] is not None and out["median_px"] <= gate_px)
    if out["median_px"] is None:
        out["outcome"] = "no points"
    elif out["n_points"] < min_points:
        out["outcome"] = f"undetermined (n={out['n_points']}<{min_points})"
    elif not out["reproj_ok"]:
        out["outcome"] = "fail (reprojection)"
    elif g["outcome"] == "pass":
        out["outcome"] = "pass"
    elif g["outcome"] == "fail":
        out["outcome"] = "fail (gauge: wrong scale)"
    else:
        out["outcome"] = "undetermined (gauge)"
    return out


def band_gauge(points: list, band_mask: np.ndarray, tol: float, min_points: int = 30, n_boot: int = 1000, seed: int = 0) -> dict:
    """The gauge as a MEASUREMENT against the ground band (reviewer 2026-10-07k): median Z_tri / Z_lift over held-out static points
    whose t_a position lies inside the band (selected by image position, so free of the lifted-depth binning artefact). The band is
    where Module 3 takes its scale (k_bg and the object's contact). Checks: the median is fitted on a random half of the band points
    and verified on the other half within tol; fewer than min_points, or a bootstrap half-width above tol, is "undetermined".
    The bridge (replacing sigma_hat, D-M3-08 amendment for John) divides the camera translations by the determined ratio."""
    H, W = band_mask.shape
    sel = []
    for p in points:
        if p["uv_ta"] is None or p["Z_lift"] is None or p["Z_lift"] <= 0:
            continue
        u, v = int(round(p["uv_ta"][0])), int(round(p["uv_ta"][1]))
        if 0 <= u < W and 0 <= v < H and band_mask[v, u]:
            sel.append(p)
    ratios = np.array([p["Z_tri"] / p["Z_lift"] for p in sel]); errs = np.array([p["err_median"] for p in sel])
    out = {"n_band_points": int(len(sel)), "min_points": int(min_points), "tol": float(tol), "band_pixels": int(band_mask.sum())}
    if len(sel) < max(min_points, 6):
        out.update({"ratio": None, "ci95": None, "half_fit": None, "half_verify": None, "verify_rel_diff": None,
                    "band_reproj_median_px": float(np.median(errs)) if len(errs) else None, "outcome": f"undetermined (n={len(sel)}<{min_points})"})
        return out
    rng = np.random.default_rng(seed)
    med, lo, hi = _boot_ci(ratios, rng, n_boot)
    perm = rng.permutation(len(ratios)); a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
    fit, ver = float(np.median(ratios[a])), float(np.median(ratios[b]))
    rel = abs(ver / fit - 1.0)
    half_width = (hi - lo) / 2.0
    ok = rel <= tol and half_width <= tol
    out.update({"ratio": med, "ci95": [lo, hi], "ci_half_width": float(half_width), "half_fit": fit, "half_verify": ver, "verify_rel_diff": float(rel),
                "band_reproj_median_px": float(np.median(errs)), "outcome": "determined" if ok else "undetermined (half-split or interval wider than tol)"})
    return out


def rebin_tables(points: list, side: int = 256) -> dict:
    """The selection-artefact check (reviewer 2026-10-07k): binning Z_tri/Z_lift by LIFTED depth puts the points whose lift came out
    too shallow into the near bin, so a near excess can be pure selection. Re-bin the same ratios by TRIANGULATED depth and by image
    row (both free of the lift's noise); a near excess that persists is a bias, one that vanishes is selection."""
    pts = [p for p in points if p["Z_lift"] is not None and p["Z_lift"] > 0]
    if len(pts) < 15:
        return {}
    ratio = np.array([p["Z_tri"] / p["Z_lift"] for p in pts]); zl = np.array([p["Z_lift"] for p in pts]); zt = np.array([p["Z_tri"] for p in pts])
    row = np.array([(p["uv_ta"] or p["uv_first"])[1] for p in pts])

    def terc(key, names):
        q = np.quantile(key, [1 / 3, 2 / 3]); i = np.digitize(key, q)
        return {n: {"median": float(np.median(ratio[i == k])), "n": int((i == k).sum()), "range": [float(key[i == k].min()), float(key[i == k].max())]}
                for k, n in enumerate(names) if (i == k).sum() >= 5}
    return {"by_lift_depth": terc(zl, ("near", "mid", "far")), "by_triangulated_depth": terc(zt, ("near", "mid", "far")),
            "by_image_row": terc(row, ("top", "middle", "bottom"))}


def compare_triangulations(pts_a: list, pts_b: list, radius_px: float = 3.0, n_boot: int = 1000, seed: int = 0, side: int = 256, band_mask=None) -> dict:
    """Z_tri from tracker A over Z_tri from tracker B at matched t_a positions (within radius_px): the lift never enters, so a ratio
    near 1 says the two trackers agree on the geometry and any excess against the lift is the lift's; a ratio away from 1 in the
    near field says one tracker lags there (reviewer 2026-10-07k: tracker lag would bias exactly the near points)."""
    from scipy.spatial import cKDTree
    A = [p for p in pts_a if p["uv_ta"] is not None]; B = [p for p in pts_b if p["uv_ta"] is not None]
    if len(A) < 5 or len(B) < 5:
        return {"n_matched": 0}
    tree = cKDTree(np.array([p["uv_ta"] for p in B]))
    r, row, zb, inb = [], [], [], []
    for p in A:
        d, j = tree.query(p["uv_ta"])
        if d <= radius_px:
            r.append(p["Z_tri"] / B[j]["Z_tri"]); row.append(p["uv_ta"][1]); zb.append(B[j]["Z_tri"])
            u_, v_ = int(round(p["uv_ta"][0])), int(round(p["uv_ta"][1]))
            inb.append(bool(band_mask is not None and 0 <= u_ < band_mask.shape[1] and 0 <= v_ < band_mask.shape[0] and band_mask[v_, u_]))
    r, row, zb, inb = np.array(r), np.array(row), np.array(zb), np.array(inb, bool)
    out = {"n_matched": int(len(r)), "radius_px": radius_px}
    if len(r) < 5:
        return out
    rng = np.random.default_rng(seed)
    med, lo, hi = _boot_ci(r, rng, n_boot)
    out.update({"median": med, "ci95": [lo, hi]})
    if band_mask is not None:                                                 # the tracker factor r (reviewer 2026-10-07n): matched points INSIDE the band
        if inb.sum() >= 5:
            bm, bl, bh = _boot_ci(r[inb], rng, n_boot)
            out["in_band"] = {"median": bm, "ci95": [bl, bh], "n": int(inb.sum())}
        else:
            out["in_band"] = {"median": None, "ci95": None, "n": int(inb.sum())}
    for key, name, labels in ((row, "by_image_row", ("top", "middle", "bottom")), (zb, "by_depth_B", ("near", "mid", "far"))):
        q = np.quantile(key, [1 / 3, 2 / 3]); i = np.digitize(key, q)
        out[name] = {n: {"median": float(np.median(r[i == k])), "n": int((i == k).sum())} for k, n in enumerate(labels) if (i == k).sum() >= 5}
    return out


def fit_plane(X: np.ndarray, iters: int = 3, k_mad: float = 3.0) -> dict:
    """Robust plane n.X + d = 0 through 3-D points: SVD about the median, re-weighted by a 3xMAD inlier cut. n is unit, oriented
    with n_y < 0 (up, in OpenCV camera axes)."""
    X = np.asarray(X, np.float64); assert X.ndim == 2 and X.shape[1] == 3 and len(X) >= 5, X.shape
    sel = np.ones(len(X), bool)
    for _ in range(iters):
        c = np.median(X[sel], 0); n = np.linalg.svd(X[sel] - c)[2][-1]; d = float(-n @ c)
        r = np.abs(X @ n + d); mad = 1.4826 * float(np.median(r[sel]))
        sel = r <= k_mad * max(mad, 1e-9)
        if sel.sum() < 5:
            sel[:] = True; break
    if n[1] > 0:
        n, d = -n, -d
    r = np.abs(X @ n + d)
    return {"n": [float(v) for v in n], "d": d, "inlier_frac": float(sel.mean()), "resid_median": float(np.median(r[sel])), "n_points": int(len(X))}


def ray_plane_depth(K: np.ndarray, uv: np.ndarray, n, d: float) -> np.ndarray:
    """z-depth where the pixel rays of uv [m,2] meet the plane n.X + d = 0 (NaN where the ray is parallel or hits behind)."""
    r = np.linalg.solve(np.asarray(K, np.float64), np.c_[np.asarray(uv, np.float64), np.ones(len(uv))].T).T
    den = r @ np.asarray(n, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(np.abs(den) > 1e-9, -float(d) / den, np.nan)
    return np.where(z > 0, z, np.nan)


def contact_bridge(points: list, band_mask: np.ndarray, K_ta: np.ndarray, contact_uv: np.ndarray, rho: float, tol: float) -> dict:
    """Reviewer 2026-10-07l item 1: beta = Z^tri_c / Z^lift_c at the source object's bottom-edge rays, from planes fitted to the
    TRIANGULATED and to the LIFTED band points (band shape, item 2: tilt between the normals, height ratio, Spearman of the ratio
    against Z_tri inside the band). Expected |beta/rho - 1| inside the band interval's half-width; tilt <= 1 degree; no depth trend."""
    from scipy.stats import spearmanr
    H, W = band_mask.shape
    sel = [p for p in points if p["uv_ta"] is not None and p["X_lift"] is not None and p["Z_lift"] and p["Z_lift"] > 0
           and 0 <= int(round(p["uv_ta"][0])) < W and 0 <= int(round(p["uv_ta"][1])) < H and band_mask[int(round(p["uv_ta"][1])), int(round(p["uv_ta"][0]))]]
    out = {"n_band_points": int(len(sel))}
    if len(sel) < 8 or len(contact_uv) == 0:
        out.update({"beta": None, "outcome": "undetermined (too few band points or no contact rays)"}); return out
    Xt = np.array([p["X_tri"] for p in sel]); Xl = np.array([p["X_lift"] for p in sel])
    pt, pl = fit_plane(Xt), fit_plane(Xl)
    zt, zl = ray_plane_depth(K_ta, contact_uv, pt["n"], pt["d"]), ray_plane_depth(K_ta, contact_uv, pl["n"], pl["d"])
    ok = np.isfinite(zt) & np.isfinite(zl)
    ratio = np.array([p["Z_tri"] / p["Z_lift"] for p in sel]); zt_pts = np.array([p["Z_tri"] for p in sel])
    beta = float(np.median(zt[ok] / zl[ok])) if ok.sum() else None
    out.update({"beta": beta, "n_contact_rays": int(ok.sum()), "rho": rho, "beta_over_rho_minus_1": None if (beta is None or not rho) else float(beta / rho - 1.0),
                "plane_tri": pt, "plane_lift": pl, "tilt_deg": float(np.degrees(np.arccos(np.clip(np.dot(pt["n"], pl["n"]), -1, 1)))),
                "height_ratio_abs_d": float(abs(pt["d"]) / max(abs(pl["d"]), 1e-9)), "Z_contact_tri_median": float(np.nanmedian(zt)), "Z_contact_lift_median": float(np.nanmedian(zl)),
                "spearman_ratio_vs_Ztri_in_band": float(spearmanr(zt_pts, ratio).correlation)})
    return out


def band_parallax(P_win: np.ndarray, Q_win: np.ndarray, excl_win: np.ndarray, uv_ta_in_band: np.ndarray) -> dict:
    """Band parallax (reviewer 2026-10-07l item 4): per band track, the largest displacement from its t_a position over the window;
    median and p90 in pixels. ΔF = p90 x |beta_CoT - beta_D4RT| / beta is the flow error the tracker choice can cost."""
    vals = []
    for i in np.nonzero(uv_ta_in_band)[0]:
        ok = Q_win[i] & ~excl_win[i]
        if ok[0] and ok.sum() >= 2:
            vals.append(float(np.linalg.norm(P_win[i, ok] - P_win[i, 0], axis=1).max()))
    return {"n": int(len(vals)), "median_px": float(np.median(vals)) if vals else None, "p90_px": float(np.quantile(vals, 0.9)) if vals else None}


def object_bias_over_time(Xl: np.ndarray, P: np.ndarray, Q: np.ndarray, y_dyn: np.ndarray, M: np.ndarray, K: np.ndarray, frames, base_region_fn,
                          contact_rows: float, min_tracks: int = 5) -> dict:
    """Reviewer 2026-10-07l item 3: rho_o(t) = lifted depth of the object's base tracks over the lifted ground under them, at the listed
    frames, every lift in its own camera-t frame (X_local). The ground is a plane fitted to static lifted tracks near the contact (rows
    from one object height above the mask bottom, columns within 1.5 object widths); its depth is read on the base pixels' rays.
    A scalar bridge is exact only if rho_o is constant along the path (expected within 5 %)."""
    out = {}
    for t in frames:
        Mt = M[t]
        if not Mt.any():
            out[int(t)] = {"rho_o": None, "note": "empty mask"}; continue
        rows = np.nonzero(Mt.any(1))[0]; cols = np.nonzero(Mt.any(0))[0]
        h_obj, w_obj = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
        base = base_region_fn(Mt, contact_rows)
        if base.sum() == 0:
            base = np.zeros_like(Mt); base[max(rows.max() - int(0.1 * h_obj), 0): rows.max() + 1] = Mt[max(rows.max() - int(0.1 * h_obj), 0): rows.max() + 1]
        obj = y_dyn & Q[:, t]
        u = np.rint(P[obj, t, 0]).astype(int).clip(0, Mt.shape[1] - 1); v = np.rint(P[obj, t, 1]).astype(int).clip(0, Mt.shape[0] - 1)
        in_base = base[v, u]
        if in_base.sum() < min_tracks:                                   # widen to the bottom 10 % of the object when the base is too thin for sparse tracks
            lo = max(rows.max() - int(round(0.1 * h_obj)), 0); in_base = (v >= lo) & Mt[v, u]
        oi = np.nonzero(obj)[0][in_base]
        stat = ~y_dyn & Q[:, t]
        us = np.rint(P[stat, t, 0]).astype(int).clip(0, Mt.shape[1] - 1); vs = np.rint(P[stat, t, 1]).astype(int).clip(0, Mt.shape[0] - 1)
        near = (~Mt[vs, us]) & (vs >= rows.max() - h_obj) & (np.abs(us - cols.mean()) <= 1.5 * w_obj)
        si = np.nonzero(stat)[0][near]
        if len(oi) < min_tracks or len(si) < 8:
            out[int(t)] = {"rho_o": None, "n_base_tracks": int(len(oi)), "n_ground_tracks": int(len(si)), "note": "too few tracks"}; continue
        pl = fit_plane(Xl[si, t])
        zg = ray_plane_depth(K[t], P[oi, t], pl["n"], pl["d"])
        zb = Xl[oi, t, 2]
        ok = np.isfinite(zg) & (zb > 0)
        out[int(t)] = {"rho_o": float(np.median(zb[ok] / zg[ok])) if ok.sum() >= min_tracks else None, "n_base_tracks": int(len(oi)), "n_ground_tracks": int(len(si)),
                       "Z_base_median": float(np.median(zb)), "Z_ground_at_base_median": float(np.nanmedian(zg)), "plane_inlier_frac": pl["inlier_frac"]}
    vals = [v["rho_o"] for v in out.values() if v.get("rho_o")]
    return {"per_frame": out, "spread_rel": (max(vals) / min(vals) - 1.0) if len(vals) >= 2 else None, "n_frames_with_value": int(len(vals))}


def bottom_edge_rays(M: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """Pixel centres of the mask's bottom-most pixel in every column, kept only where that pixel lies within the lowest `frac` of the
    object's height (the feet, hooves and wheels; not a belly between legs). The lowest-rows rule (base_region) gave one hoof tip on
    265.2.005 (n = 1 ray); this is the per-column reading Module 2's contact column B uses."""
    M = np.asarray(M, bool)
    rows = np.nonzero(M.any(1))[0]
    if not len(rows):
        return np.zeros((0, 2))
    lo = rows.max() - frac * (rows.max() - rows.min() + 1)
    out = []
    for j in np.nonzero(M.any(0))[0]:
        b = int(np.nonzero(M[:, j])[0].max())
        if b >= lo:
            out.append([j + 0.5, b + 0.5])
    return np.array(out, np.float64).reshape(-1, 2)


def flow_difference(X_bg: np.ndarray, E_l: np.ndarray, Kp: np.ndarray, Hp: int, Wp: int, beta_lo: float, beta_hi: float, excluded_patches: np.ndarray) -> dict:
    """Reviewer 2026-10-07m condition 5: || F_geo(beta_lo) - F_geo(beta_hi) || over the gated lattice, in patches. F_geo(beta) is the
    background flow from the admitted camera with its translations divided by beta, on the lifted background X_bg (world = camera t_a,
    target gauge), read per patch as the median displacement of the points that start in that patch. Gated = patches holding points,
    not excluded (sky, object). Below 0.5 patch (the block-20 readout error) the method cannot tell the two bridges apart."""
    from src.generate.m3.geometry import project as PJ
    def flow(beta):
        E = np.array(E_l, np.float64).copy(); E[:, :3, 3] /= float(beta)
        pb, z = PJ.project(X_bg, E, Kp)
        return pb, z
    pb_lo, z_lo = flow(beta_lo); pb_hi, _ = flow(beta_hi)
    T = len(E_l)
    u0, v0 = np.floor(pb_lo[0, :, 0]).astype(int), np.floor(pb_lo[0, :, 1]).astype(int)
    inside = (u0 >= 0) & (u0 < Wp) & (v0 >= 0) & (v0 < Hp) & (z_lo[0] > 0)
    cell = v0 * Wp + u0
    diffs, per_frame = [], []
    for j in range(1, T):
        d = np.linalg.norm((pb_lo[j] - pb_lo[0]) - (pb_hi[j] - pb_hi[0]), axis=1)
        ok = inside & np.isfinite(d) & (z_lo[j] > 0)
        med = np.full(Hp * Wp, np.nan)
        for c in np.unique(cell[ok]):
            med[c] = np.median(d[ok & (cell == c)])
        gated = np.isfinite(med) & ~excluded_patches.reshape(-1)
        vals = med[gated]
        diffs.append(vals); per_frame.append({"latent_frame": int(j), "median_patch": float(np.median(vals)) if len(vals) else None, "n_gated": int(gated.sum())})
    allv = np.concatenate(diffs) if diffs else np.array([])
    return {"beta_lo": float(beta_lo), "beta_hi": float(beta_hi), "median_patch": float(np.median(allv)) if len(allv) else None,
            "p90_patch": float(np.quantile(allv, 0.9)) if len(allv) else None, "share_above_half_patch": float((allv > 0.5).mean()) if len(allv) else None,
            "n_gated_patch_frames": int(len(allv)), "per_frame": per_frame,
            "reading": None if not len(allv) else ("informative: the two bridges differ by more than the readout error on part of the lattice" if (allv > 0.5).mean() > 0 else "not informative: below the readout error everywhere")}
