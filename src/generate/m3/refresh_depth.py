"""S3 metric (brief v2.6 §9.1): depth read from a decoded x_hat_0 at patch scale, aligned to a reference with the lower-noise map
as the regressor, scored by median AbsRel. Pure numpy; every function is unit-tested on a known answer before any real
number is read (CLAUDE.md §3a.13). Conventions: depth maps are [H, W] camera z with NaN where invalid; patch maps are [Hp, Wp]
with NaN where a patch has fewer than `min_valid` of its pixels valid.

    patch_depth          pixel depth -> patch median (valid iff >= 50 % of the pixels are valid)
    align_disparity      1/D_sigma ~ c (1/D_ref) + d on the fit mask; D_tilde = ((1/D_sigma - d) / c)^-1     (§9.1 step 3)
    absrel               median |D_tilde - D_ref| / D_ref over the score mask
    metric_row           absrel_all (fit and score on all valid patches) and absrel_border_extrap (fit interior, score the outer ring)
    induced_flow_error   patch centres lifted with D_tilde and with D_ref, moved by E_b -> E_t, EPE in patches
"""
from __future__ import annotations

import numpy as np


def patch_depth(depth: np.ndarray, px: int, Hp: int, Wp: int, min_valid: float = 0.5) -> np.ndarray:
    """depth [Hp*px, Wp*px] with NaN outside the valid mask -> [Hp, Wp] patch medians; NaN where < min_valid of the patch is valid."""
    D = np.asarray(depth, np.float64)
    assert D.shape == (Hp * px, Wp * px), (D.shape, Hp * px, Wp * px)
    blocks = D.reshape(Hp, px, Wp, px).transpose(0, 2, 1, 3).reshape(Hp, Wp, px * px)
    valid = np.isfinite(blocks) & (blocks > 0)
    frac = valid.mean(-1)
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                             # all-NaN patches are expected (sky, holes); they read NaN
        med = np.nanmedian(np.where(valid, blocks, np.nan), axis=-1)
    return np.where(frac >= min_valid, med, np.nan)


def align_disparity(D_sigma: np.ndarray, D_ref: np.ndarray, fit_mask: np.ndarray) -> dict:
    """Affine fit in disparity with the REFERENCE (lower-noise) map as the regressor: 1/D_sigma ~ c (1/D_ref) + d.
    Returns c, d, r2, n and D_tilde = ((1/D_sigma - d) / c)^-1 over the whole map (NaN where either input is invalid)."""
    ok = np.asarray(fit_mask, bool) & np.isfinite(D_sigma) & np.isfinite(D_ref) & (D_sigma > 0) & (D_ref > 0)
    n = int(ok.sum())
    assert n >= 3, f"alignment needs >= 3 valid patches, got {n}"
    x = 1.0 / D_ref[ok]; y = 1.0 / D_sigma[ok]
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum(); sxy = ((x - xm) * (y - ym)).sum(); syy = ((y - ym) ** 2).sum()
    c = float(sxy / max(sxx, 1e-300)); d = float(ym - c * xm)
    r2 = float(sxy * sxy / max(sxx * syy, 1e-300))
    with np.errstate(all="ignore"):
        inv = (1.0 / D_sigma - d) / c
        D_tilde = np.where(np.isfinite(inv) & (inv > 0), 1.0 / inv, np.nan)
    return {"c": c, "d": d, "r2": r2, "n_fit": n, "D_tilde": D_tilde}


def absrel(D_tilde: np.ndarray, D_ref: np.ndarray, score_mask: np.ndarray):
    ok = np.asarray(score_mask, bool) & np.isfinite(D_tilde) & np.isfinite(D_ref) & (D_ref > 0)
    if ok.sum() == 0:
        return None, 0
    return float(np.median(np.abs(D_tilde[ok] - D_ref[ok]) / D_ref[ok])), int(ok.sum())


def metric_row(D_sigma: np.ndarray, D_ref: np.ndarray, border: int = 4) -> dict:
    """The §9.1 columns for one (sigma, frame) against its reference, both at patch scale."""
    Hp, Wp = D_ref.shape
    both = np.isfinite(D_sigma) & np.isfinite(D_ref) & (D_sigma > 0) & (D_ref > 0)
    out = {"valid_share_both": float(both.mean()), "n_valid_both": int(both.sum())}
    if both.sum() < 3:
        out.update({"absrel_all": None, "absrel_border_extrap": None, "c": None, "d": None, "r2": None, "unaligned_ratio_median": None})
        return out
    fit = align_disparity(D_sigma, D_ref, both)
    out["absrel_all"], out["n_scored_all"] = absrel(fit["D_tilde"], D_ref, both)
    out.update({"c": fit["c"], "d": fit["d"], "r2": fit["r2"], "n_fit_all": fit["n_fit"]})
    out["unaligned_ratio_median"] = float(np.median(D_sigma[both] / D_ref[both]))
    ring = np.zeros((Hp, Wp), bool); ring[:border] = ring[-border:] = True; ring[:, :border] = ring[:, -border:] = True
    interior = both & ~ring
    if interior.sum() >= 3 and (both & ring).sum() >= 3:
        fit_i = align_disparity(D_sigma, D_ref, interior)
        out["absrel_border_extrap"], out["n_scored_border"] = absrel(fit_i["D_tilde"], D_ref, both & ring)
        out["c_interior"], out["d_interior"] = fit_i["c"], fit_i["d"]
    else:
        out["absrel_border_extrap"], out["n_scored_border"] = None, 0
    return out


def induced_flow_error(D_tilde: np.ndarray, D_ref: np.ndarray, Kp: np.ndarray, E_b: np.ndarray, E_later: np.ndarray) -> dict:
    """Patch centres of frame b lifted with D_tilde and with D_ref (camera b), moved to each later frame by E_t E_b^-1, projected with K^p:
    the displacement error the aligned depth would induce in the geometric flow. EPE median / p90 in patches over valid patches and frames."""
    Hp, Wp = D_ref.shape
    jj, ii = np.meshgrid(np.arange(Wp) + 0.5, np.arange(Hp) + 0.5)
    ok = np.isfinite(D_tilde) & np.isfinite(D_ref) & (D_tilde > 0) & (D_ref > 0)
    if ok.sum() < 3 or len(E_later) == 0:
        return {"epe_median_patch": None, "epe_p90_patch": None, "n": 0}
    uv1 = np.stack([jj[ok], ii[ok], np.ones(int(ok.sum()))], 0)
    rays = np.linalg.inv(np.asarray(Kp, np.float64)) @ uv1                                   # [3, n] unit-depth rays in camera b
    Xb_t, Xb_r = (rays * D_tilde[ok]).T, (rays * D_ref[ok]).T
    Eb_inv = np.linalg.inv(np.asarray(E_b, np.float64))
    errs = []
    for E_t in np.asarray(E_later, np.float64):
        M = E_t @ Eb_inv

        def proj(X):
            Xc = X @ M[:3, :3].T + M[:3, 3]
            p = (np.asarray(Kp, np.float64) @ Xc.T).T
            with np.errstate(all="ignore"):
                return np.where(Xc[:, 2:3] > 0, p[:, :2] / p[:, 2:3], np.nan)
        e = np.linalg.norm(proj(Xb_t) - proj(Xb_r), axis=1)
        errs.append(e[np.isfinite(e)])
    e = np.concatenate(errs) if errs else np.array([])
    return {"epe_median_patch": float(np.median(e)) if len(e) else None, "epe_p90_patch": float(np.quantile(e, 0.9)) if len(e) else None, "n": int(len(e))}


def refresh_rule(passing_sigmas: list, lo: float = 0.45, hi: float = 0.85, max_n: int = 3) -> list:
    """§9.1: R = the highest passing sigma inside [lo, hi] and, if they pass, the next one or two lower grid values (at most three).
    Empty when nothing passes in the window."""
    P = sorted([s for s in passing_sigmas if lo - 1e-9 <= s <= hi + 1e-9], reverse=True)
    if not P:
        return []
    out = [P[0]]
    for s in P[1:]:
        if len(out) >= max_n:
            break
        out.append(s)
    return out
