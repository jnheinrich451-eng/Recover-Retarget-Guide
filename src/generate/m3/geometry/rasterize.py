"""Depth-aware rasteriser on the QUERY frame t (brief v2.4 §3.4) and the anchor schedule (§3.5).

Bilinear footprint, radius 1 patch: a primitive at patch coordinate (u, v) spreads over the 4 nearest patch centres
(centre of (i, j) = (j + .5, i + .5)). Displacements are BACKWARD: F^{t->a}(q) = location in a minus location in t.
C_geo is a bounded coverage 1 - prod(1 - r), computed as 1 - exp(sum log(1 - r)).
"""
from __future__ import annotations

import numpy as np

_R_MAX = 1.0 - 1e-9          # numerical guard for log(1 - r) only; not a tunable


def _corners(p: np.ndarray, Hp: int, Wp: int):
    """p [M,2] finite. Returns flat cell index [M,4], weight [M,4] (0 where the corner is outside the lattice)."""
    x, y = p[:, 0] - 0.5, p[:, 1] - 0.5
    j0, i0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - j0, y - i0
    cells, wts = [], []
    for di, wy in ((0, 1 - fy), (1, fy)):
        for dj, wx in ((0, 1 - fx), (1, fx)):
            i, j = i0 + di, j0 + dj
            ok = (i >= 0) & (i < Hp) & (j >= 0) & (j < Wp)
            cells.append(np.where(ok, i * Wp + j, 0)); wts.append(np.where(ok, wy * wx, 0.0))
    return np.stack(cells, 1), np.stack(wts, 1)


def pair_fields(pbar_a, pbar_t, vis_a, vis_t, Hp: int, Wp: int, eps: float):
    """One pair (a, t). pbar_* [N,2], vis_* [N] bool. Returns F_geo [Hp,Wp,2] (patch units, backward), C_geo [Hp,Wp] in [0,1]."""
    m = vis_a & vis_t
    F = np.zeros((Hp * Wp, 2)); den = np.zeros(Hp * Wp); logacc = np.zeros(Hp * Wp)
    if m.any():
        cells, w = _corners(pbar_t[m], Hp, Wp)
        d = (pbar_a[m] - pbar_t[m])
        for k in range(4):
            np.add.at(den, cells[:, k], w[:, k])
            np.add.at(F, cells[:, k], w[:, k, None] * d)
            np.add.at(logacc, cells[:, k], np.log1p(-np.minimum(w[:, k], _R_MAX)))
    F = F / (den[:, None] + eps)
    C = 1.0 - np.exp(logacc)
    assert np.isfinite(F).all() and (C >= 0).all() and (C <= 1).all()
    return F.reshape(Hp, Wp, 2), C.reshape(Hp, Wp)


def occupancy(pbar_t, vis_t, is_obj, Hp: int, Wp: int) -> np.ndarray:
    """O_t(q) = 1 - prod over visible OBJECT primitives of (1 - r). Pair-independent."""
    m = vis_t & is_obj
    logacc = np.zeros(Hp * Wp)
    if m.any():
        cells, w = _corners(pbar_t[m], Hp, Wp)
        for k in range(4):
            np.add.at(logacc, cells[:, k], np.log1p(-np.minimum(w[:, k], _R_MAX)))
    return (1.0 - np.exp(logacc)).reshape(Hp, Wp)


def coverage_curve(C0: np.ndarray, O: np.ndarray, eps: float) -> np.ndarray:
    """C0 [T,Hp,Wp] = C_geo^{t->0}, O [T,Hp,Wp]. Background coverage from frame 0: sum C (1 - O) / sum (1 - O)."""
    w = 1.0 - O
    return (C0 * w).sum((1, 2)) / (w.sum((1, 2)) + eps)


def anchor_schedule(cov: np.ndarray, tau_anchor: float, max_anchors: int) -> list[int]:
    """a_1 = first t >= 1 with cov < tau; a_{k+1} = a_k + a_1; at most max_anchors, all < T. Never below tau: {0}, no refresh."""
    T = len(cov)
    below = [t for t in range(1, T) if cov[t] < tau_anchor]
    if not below:
        return [0]
    a1 = below[0]
    return [a for a in range(0, T, a1)][:max_anchors]


def warp_mask_occupancy(pm0: np.ndarray, p_from: np.ndarray, p_to: np.ndarray, Hp: int, Wp: int):
    """Occupancy of record (reviewer 2026-10-06g): the PAINTED object's pixels pm0 [M,2] (patch units, frame 0) are moved by the
    low-pass (affine) displacement field fitted to the object primitives' own displacements p_from -> p_to [N,2] (visible in both
    frames), and splatted with the bilinear footprint. Returns O [Hp,Wp] and the affine fit's median residual (patches)."""
    if len(p_from) >= 6:
        A = np.concatenate([p_from, np.ones((len(p_from), 1))], 1)
        coef, *_ = np.linalg.lstsq(A, p_to - p_from, rcond=None)
        pm_t = pm0 + np.concatenate([pm0, np.ones((len(pm0), 1))], 1) @ coef
        resid = float(np.median(np.linalg.norm(A @ coef - (p_to - p_from), axis=1)))
    else:
        pm_t, resid = pm0.copy(), None
    inside = (pm_t[:, 0] >= 0) & (pm_t[:, 0] < Wp) & (pm_t[:, 1] >= 0) & (pm_t[:, 1] < Hp)
    logacc = np.zeros(Hp * Wp)
    if inside.any():
        cells, w = _corners(pm_t[inside], Hp, Wp)
        for k in range(4):
            np.add.at(logacc, cells[:, k], np.log1p(-np.minimum(w[:, k], _R_MAX)))
    return (1.0 - np.exp(logacc)).reshape(Hp, Wp), resid
