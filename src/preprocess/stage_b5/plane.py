"""Stage B.5 — scene up by pooled static plane (SKILL v1.1 §2.3).

Pool = de-gauged static points over ALL frames; on branch clips the
followed subject is LABELED STATIC (Stage B pathology) so `exclude` MUST
carry fc_blob_idx there (v1.1 delta 3). RANSAC → eigen refine; planarity π
AND spread ς both reported (π alone passes confidently-wrong strip normals).
"""
from __future__ import annotations

import numpy as np


def static_pool(X0, Q, y_dyn, eta, seg_curve, exclude=None, eta_q=0.5,
                cap=20000, seed=0):
    N, F = Q.shape
    keep = ~y_dyn.copy()
    if exclude is not None and len(exclude):
        keep[np.asarray(exclude, int)] = False
    eta_min = np.nanquantile(eta[Q], eta_q)
    sel = keep[:, None] & Q & (eta > eta_min)
    Xd = X0 / seg_curve[None, :, None]
    pts = Xd[sel]
    if len(pts) > cap:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), cap, replace=False)]
    return pts


def fit_plane(pts, iters=200, tol=0.05, seed=0):
    """RANSAC n·X+d=0 → eigen refinement on inliers.
    Returns dict(n, d, planarity, spread, n_inlier) or None."""
    if len(pts) < 3:
        return None
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(int(iters)):
        i = rng.choice(len(pts), 3, replace=False)
        a, b, c = pts[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        dist = np.abs((pts - a) @ n)
        cnt = int((dist < tol).sum())
        if cnt > best_n:
            best_n, best = cnt, (n, a)
    if best is None:
        return None
    n0, a0 = best
    inl = pts[np.abs((pts - a0) @ n0) < tol]
    mu = inl.mean(0)
    S = np.cov((inl - mu).T)
    w, V = np.linalg.eigh(S)          # ascending
    n = V[:, 0]
    planarity = float(1.0 - w[0] / max(w[1], 1e-12))
    spread = float(w[1] / max(w[2], 1e-12))
    return {"n": n, "d": float(-n @ mu), "mu": mu,
            "planarity": planarity, "spread": spread,
            "n_inlier": int(len(inl))}


def orient_and_cross(pl, centroid, G, t_c):
    """Sign n toward the subject centroid; camera no-roll prior
    u_b = -R_{t_c} e_y; angles between and vs slope (§2.3)."""
    n = pl["n"]
    if (centroid - pl["mu"]) @ n < 0:
        n = -n
    # The plane is n.X + d = 0 (contract v1.2): re-signing n re-signs d. MEASURED 2026-09-14 on
    # clip 284 when the R1 proxy first met C1: the emitted d kept the unsigned fit's sign, so the
    # source sat BELOW its own ground plane and the placement solved a negative scale.
    pl["d"] = float(-n @ pl["mu"])
    assert abs(n @ pl["mu"] + pl["d"]) < 1e-9
    assert (centroid @ n) + pl["d"] > 0, "subject centroid must lie on the +n side of the oriented plane"
    u_b = -G[t_c, :3, :3] @ np.array([0.0, 1.0, 0.0])
    u_b = u_b / np.linalg.norm(u_b)
    ang = float(np.degrees(np.arccos(np.clip(n @ u_b, -1, 1))))
    return n, u_b, ang
