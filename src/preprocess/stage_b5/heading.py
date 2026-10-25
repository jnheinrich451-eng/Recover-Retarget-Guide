"""Stage B.5 — centroid track + heading by windowed WLS (SKILL v1.1 §2.1-2.2).

Channel-dispatched per the O_pre contract: x0 clips measure the centroid
directly from X0; branch clips COMPOSE it (trajectory ∘ camera-frame
centroid) because their X0 carries no object motion. Never finite-difference;
never span an un-de-gauged encode boundary (§2.2 gauge discipline).
Pure NumPy.
"""
from __future__ import annotations

import numpy as np


def seg_scale_curve(segment_scales, F):
    """Per-frame relative gauge from Stage B's segment_scales record."""
    g = np.ones(F)
    for seg in segment_scales:
        g[int(seg["start"]):int(seg["end"])] = float(seg["rel_scale"])
    return g


def centroid_track_x0(X0, Q, P, M, subject, eta=None):
    """x0 channel: weighted per-frame centroid of the subject set.
    w = Q ∧ inside-mask ∧ η (η only on this channel — §2.1)."""
    N, F = Q.shape
    H, W = M.shape[1:]
    xb = np.full((F, 3), np.nan)
    for t in range(F):
        idx = subject[Q[subject, t]]
        if not len(idx):
            continue
        u = np.clip(P[idx, t, 0].round().astype(int), 0, W - 1)
        v = np.clip(P[idx, t, 1].round().astype(int), 0, H - 1)
        inm = M[t, v, u]
        idx = idx[inm]
        if not len(idx):
            continue
        w = eta[idx, t] if eta is not None else np.ones(len(idx))
        sw = w.sum()
        if sw > 0:
            xb[t] = (w[:, None] * X0[idx, t]).sum(0) / sw
    return xb


def centroid_track_branch(G_traj, c_cam_ts, tau_rel, rel_valid):
    """Branch channel (§2.1 v1.1): X̄_t = R_t (c^cam_{t_s} + τ^rel_t) + t_t.
    c_cam_ts = 3D camera-frame subject centroid at the seed frame (from
    X_local of blob∪dense — NEVER the 2D seed pixels). Also returns
    branch_ratio = median ‖dX̄‖/‖dt_t‖ (Gate-1 statistic; ≈1 healthy)."""
    F = G_traj.shape[0]
    xb = np.full((F, 3), np.nan)
    for t in range(F):
        if rel_valid[t]:
            p = c_cam_ts + tau_rel[t]
            xb[t] = G_traj[t, :3, :3] @ p + G_traj[t, :3, 3]
    ok = np.isfinite(xb[:, 0])
    dx = np.linalg.norm(np.diff(xb[ok], axis=0), axis=1)
    dc = np.linalg.norm(np.diff(G_traj[ok][:, :3, 3], axis=0), axis=1)
    ratio = float(np.median(dx / np.maximum(dc, 1e-12))) if len(dx) else float("nan")
    return xb, ratio


def heading_window(xb, t_c, L, L_min, boundaries, segment_scales):
    """§2.2 window + gauge decision, shared by EVERY heading estimator so
    v1.1 and v1.2 are measured on identical frames. Window = [t_c, t_c+L]
    clipped to the encode segment; if that leaves < L_min frames, keep the
    full window and de-gauge it with segment_scales instead.
    Returns (ts, scale, rec): frames used (finite in xb), per-frame
    multiplicative de-gauge factor (all 1 unless degauged), window record."""
    F = len(xb)
    nxt = next((int(b) for b in boundaries if int(b) > t_c), F)
    end = min(t_c + L, F - 1)
    degauged = False
    if nxt - 1 < t_c + L_min:
        degauged = True
    else:
        end = min(end, nxt - 1)
    ts = np.arange(t_c, end + 1)
    ts = ts[np.isfinite(xb[ts, 0])]
    g = seg_scale_curve(segment_scales, F)
    scale = (1.0 / g[ts]) if degauged else np.ones(len(ts))
    rec = {"t0": int(t_c), "t1": int(end), "degauged": bool(degauged),
           "n": int(len(ts))}
    return ts, scale, rec


def wls_slope(X, d, w=None):
    """Closed-form WLS line fit X ~ b + v d on one 3-D track (v1.1 formula,
    extended with optional per-frame weights w >= 0; unit weights reproduce
    v1.1 exactly). Returns v [3] (with magnitude), sigma_theta (rad),
    per-frame residual norms."""
    X = np.asarray(X, np.float64)
    d = np.asarray(d, np.float64)
    n = len(d)
    w = np.ones(n) if w is None else np.asarray(w, np.float64)
    w = w * (n / max(w.sum(), 1e-12))
    dm = (w * d).sum() / n
    Xm = (w[:, None] * X).sum(0) / n
    denom = (w * (d - dm) ** 2).sum()
    v = ((w * (d - dm))[:, None] * (X - Xm)).sum(0) / max(denom, 1e-12)
    resid = X - (Xm + np.outer(d - dm, v))
    r = np.linalg.norm(resid, axis=1)
    sigma2 = (w * r ** 2).sum() / max(n - 2, 1)
    sigma_v = np.sqrt(sigma2 / max(denom, 1e-12))
    speed = np.linalg.norm(v)
    sigma_theta = float(sigma_v / speed) if speed > 1e-12 else float("inf")
    return v, sigma_theta, r


def irls_slope(X, d, w0=None, passes=3):
    """Huber IRLS on per-frame residual norms; delta = 1.4826 MAD of the
    plain pass. Used only for the sigma_spec spread (v1.2 §2.5)."""
    v, s, r = wls_slope(X, d, w0)
    delta = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
    w = np.ones(len(d)) if w0 is None else np.asarray(w0, np.float64).copy()
    for _ in range(int(passes)):
        hub = np.minimum(1.0, delta / np.maximum(r, 1e-12))
        v, s, r = wls_slope(X, d, w * hub)
    return v, s, r


def wls_heading(xb, t_c, L, L_min, boundaries, segment_scales):
    """v1.1 §2.2: windowed WLS on a centroid track. Returns d_src [3] (unit)
    or None, sigma_theta, window record. Kept verbatim in behaviour as the
    v1.2 foil (`heading_estimator: v1.1`)."""
    ts, scale, rec = heading_window(xb, t_c, L, L_min, boundaries, segment_scales)
    if len(ts) < 3:
        return None, float("nan"), rec
    d = (ts - t_c).astype(np.float64)
    v, sigma_theta, _ = wls_slope(xb[ts] * scale[:, None], d)
    speed = np.linalg.norm(v)
    if speed < 1e-12:
        return None, float("inf"), rec
    return v / speed, sigma_theta, rec


# ----------------------------------------------------------------------------
# v1.2 — common-support centroid (SKILL v1.2 §2.1b; Gate A 2026-09-10)
# ----------------------------------------------------------------------------
def _in_mask(P, M, idx, t):
    H, W = M.shape[1:]
    u = np.clip(P[idx, t, 0].round().astype(int), 0, W - 1)
    v = np.clip(P[idx, t, 1].round().astype(int), 0, H - 1)
    return M[t, v, u]


def common_support(Q, P, M, subject, ts):
    """S_star = tracks valid AND inside the mask at EVERY window frame.
    Returns (S_star sorted int array, n_union, per-frame counts). Rationale
    (measured 2026-09-07/09): a support set that varies with t shifts the
    centroid by the depth difference of entering/leaving tracks — on the
    verdict clips this manufactured a 46-83 deg heading error with no 1/N
    suppression, while the true per-frame displacement was ~0.003."""
    sets, counts = [], []
    for t in ts:
        idx = subject[Q[subject, t]]
        idx = idx[_in_mask(P, M, idx, t)] if len(idx) else idx
        sets.append(set(idx.tolist()))
        counts.append(int(len(idx)))
    if not sets:
        return np.array([], int), 0, counts
    star = np.array(sorted(set.intersection(*sets)), int)
    return star, int(len(set.union(*sets))), counts


def centroid_common(X0, S_star, ts, scale, weights=None):
    """Centroid over a FIXED support on the window frames (de-gauged).
    weights: None (v1.2 production, unweighted) or [n_star] window-constant."""
    Xs = X0[S_star][:, ts] * scale[None, :, None]          # [n, |W|, 3]
    if weights is None:
        return Xs.mean(axis=0)
    w = np.asarray(weights, np.float64)
    return (w[:, None, None] * Xs).sum(0) / max(w.sum(), 1e-12)


def mad(x):
    """1.4826 x median absolute deviation (sigma-consistent). Contractual
    convention for every rho_* statistic (contract_stage_b5_v1_2 §3)."""
    x = np.asarray(x, np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def rho(x):
    """|median| / MAD — signal-to-spread of a per-track velocity component."""
    m = mad(x)
    return float(abs(np.median(x)) / m) if m > 0 else float("inf")


def identifiability(V, G, t_a, c_bar_world):
    """Per-track velocity signal-to-spread in the CAMERA-t_a FRAME:
    radial = along the unit ray from the camera-t_a origin to the window-mean
    centroid; lateral = the two perpendicular directions. NOT rotation-
    invariant — the frame is part of the definition (contract §3)."""
    R_a, p_a = G[t_a, :3, :3], G[t_a, :3, 3]
    ray = R_a.T @ (np.asarray(c_bar_world, np.float64) - p_a)
    ray = ray / max(np.linalg.norm(ray), 1e-12)
    Va = np.asarray(V, np.float64) @ R_a                    # rows: R_a.T @ v
    e1 = np.cross(ray, np.array([0.0, -1.0, 0.0]))
    if np.linalg.norm(e1) < 1e-9:
        e1 = np.cross(ray, np.array([1.0, 0.0, 0.0]))
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(ray, e1)
    v_rad = Va @ ray
    return {"rho_radial": rho(v_rad), "rho_lat1": rho(Va @ e1),
            "rho_lat2": rho(Va @ e2), "median_v_radial": float(np.median(v_rad)),
            "frame": "camera_t_a", "mad_convention": "1.4826*MAD"}


def _angle_deg(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))


def heading_v12(X0, Q, P, M, subject, eta, present, G, t_c, L, L_min,
                boundaries, segment_scales, irls_passes=3, min_support=3):
    """v1.2 x0-channel heading: common support (valid + in-mask at every
    window frame), UNWEIGHTED centroid, plain WLS. Selected by blind human
    Gate A 2026-09-10 (8/9 vs v1.1 6/9); mechanism measured, not assumed.

    Also returns the v1.1 heading on the same window (foil; `churn_deg` =
    angle between them), support statistics, `sigma_spec` (max angle from
    d_src to the four nuisance variants V1c / Vmed / Vmedw / V1cr — the
    heading's sensitivity to specification choices) and the camera-t_a
    identifiability statistics. All scalars f64; arrays f64."""
    xb11 = centroid_track_x0(X0, Q, P, M, subject, eta)
    xb11[~present] = np.nan
    ts, scale, rec = heading_window(xb11, t_c, L, L_min, boundaries, segment_scales)
    out = {"window": rec, "X_bar_v11": xb11, "heading_defined": False,
           "d_src": None, "sigma_theta": float("nan"), "X_bar_common": None,
           "d_v11": None, "churn_deg": float("nan"), "sigma_spec_deg": float("nan"),
           "n_star": 0, "n_union": 0, "support_frac": float("nan"),
           "per_frame_counts": [], "ident": None}
    if len(ts) < 3:
        return out
    d = (ts - t_c).astype(np.float64)
    S_star, n_union, counts = common_support(Q, P, M, subject, ts)
    out.update({"n_star": int(len(S_star)), "n_union": int(n_union),
                "per_frame_counts": counts,
                "support_frac": float(len(S_star) / max(n_union, 1))})
    # v1.1 foil on the identical window
    v11, s11, _ = wls_slope(xb11[ts] * scale[:, None], d)
    if np.linalg.norm(v11) > 1e-12:
        out["d_v11"] = v11 / np.linalg.norm(v11)
    if len(S_star) < min_support:
        return out
    Xc = centroid_common(X0, S_star, ts, scale)
    v, sigma_theta, _ = wls_slope(Xc, d)
    speed = np.linalg.norm(v)
    if speed < 1e-12:
        return out
    d_src = v / speed
    Xfull = np.full((Q.shape[1], 3), np.nan)
    Xfull[ts] = Xc
    out.update({"heading_defined": True, "d_src": d_src,
                "sigma_theta": float(sigma_theta), "X_bar_common": Xfull})
    if out["d_v11"] is not None:
        out["churn_deg"] = _angle_deg(d_src, out["d_v11"])
    # nuisance variants -> sigma_spec
    Xs = X0[S_star][:, ts] * scale[None, :, None]
    V = np.stack([wls_slope(Xs[j], d)[0] for j in range(len(S_star))])
    eta_bar = np.median(eta[S_star][:, ts], axis=1)
    vs = []
    vs.append(wls_slope(centroid_common(X0, S_star, ts, scale, eta_bar), d)[0])     # V1c
    vs.append(np.median(V, axis=0))                                                  # Vmed
    order = np.argsort(V, axis=0)
    cw = np.cumsum(eta_bar[order], axis=0) / max(eta_bar.sum(), 1e-12)
    vs.append(np.array([V[order[np.searchsorted(cw[:, c], 0.5), c], c] for c in range(3)]))  # Vmedw
    vs.append(irls_slope(centroid_common(X0, S_star, ts, scale, eta_bar), d, passes=irls_passes)[0])  # V1cr
    out["sigma_spec_deg"] = float(max(_angle_deg(d_src, u) for u in vs))
    out["ident"] = identifiability(V, G, t_c, np.nanmean(Xc, axis=0))
    return out
