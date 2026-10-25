"""Projection onto the token lattice and visibility (brief v2.4 §3.3).

pbar [T,N,2] in patch units (u = column direction, v = row direction), depth [T,N] = camera z, vis [T,N] bool.
Visibility = inside the lattice, z > 0, and passing a z-buffer at `zbuf_scale` x the lattice resolution built from ALL
primitives (background and object together: background behind the object is invisible).
"""
from __future__ import annotations

import numpy as np


def project(X_w: np.ndarray, E: np.ndarray, Kp: np.ndarray):
    """X_w [N,3] (static) or [T,N,3] (per frame); E [T,4,4] world->camera; Kp [3,3]. Returns pbar [T,N,2], depth [T,N]."""
    E = np.asarray(E, np.float64); Kp = np.asarray(Kp, np.float64); X = np.asarray(X_w, np.float64)
    T = len(E)
    if X.ndim == 2:
        X = np.broadcast_to(X, (T,) + X.shape)
    assert X.shape[0] == T and X.shape[2] == 3, X.shape
    Xc = X @ E[:, :3, :3].transpose(0, 2, 1) + E[:, None, :3, 3]
    z = Xc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = Xc @ Kp.T
        pbar = uvw[..., :2] / uvw[..., 2:3]
    pbar = np.where((z > 0)[..., None], pbar, np.nan)      # behind the camera: no patch coordinate, never a wrapped one
    return pbar, z


def visibility(pbar: np.ndarray, depth: np.ndarray, Hp: int, Wp: int, zbuf_rel_tol: float, zbuf_scale: int = 2) -> np.ndarray:
    """vis [T,N]. z-buffer cell = floor(pbar * zbuf_scale); a point is visible if its depth is within (1 + tol) of the
    nearest depth in its cell."""
    T, N = depth.shape
    inside = np.isfinite(pbar).all(-1) & (depth > 0)
    u, v = np.where(inside, pbar[..., 0], 0.0), np.where(inside, pbar[..., 1], 0.0)
    inside &= (u >= 0) & (u < Wp) & (v >= 0) & (v < Hp)
    vis = np.zeros((T, N), bool)
    Hz, Wz = Hp * zbuf_scale, Wp * zbuf_scale
    for t in range(T):
        idx = np.nonzero(inside[t])[0]
        if len(idx) == 0:
            continue
        cell = np.minimum((v[t, idx] * zbuf_scale).astype(int), Hz - 1) * Wz + np.minimum((u[t, idx] * zbuf_scale).astype(int), Wz - 1)
        zmin = np.full(Hz * Wz, np.inf)
        np.minimum.at(zmin, cell, depth[t, idx])
        vis[t, idx] = depth[t, idx] <= zmin[cell] * (1.0 + zbuf_rel_tol)
    return vis


def move_object(X_obj0: np.ndarray, R_t: np.ndarray, c0: np.ndarray, c_t: np.ndarray) -> np.ndarray:
    """X_{i,t} = R_t (X_i - c0) + c_t  (brief §3.2). X_obj0 [N,3] -> [T,N,3]. No re-centring, no A, no vendor pose."""
    X = np.asarray(X_obj0, np.float64) - np.asarray(c0, np.float64)
    return X[None] @ np.asarray(R_t, np.float64).transpose(0, 2, 1) + np.asarray(c_t, np.float64)[:, None]
