"""Module 3 guidance losses (brief v2.6 §5) and the latent update (§5.4). Pure torch, unit-tested on known answers before any real
latent is touched (CLAUDE.md §3a.13). Every F here is the WINDOWED readout of §4 (readout.windowed_expected_displacement), backward
displacement in patch units on the frame-t lattice; targets are the geometry cache's fields on the same lattice.

    bg_loss              L_bg   = sum_q w (F_hat - F_geo)^2 / (sum w + eps),  w = C_geo (1 - O_t)
    centroid_loss        L_cent = || mean_{O_t} F_hat - target ||^2  (object mode `centroid`: one vector per frame)
    perp_loss            L_perp = sum_q [1 - C^max] d(q)^2 / (sum [1 - C^max] + eps), d = distance of the expected source to the epipolar line
                                  (line mode) or to the homography-mapped point (homography mode)
    mass_terms           L_mass (D-M3-10, off until S4 decides) and its two diagnostics: mass at the geometric target, share of queries
                         whose readout window contains the target
    guided_update        z' = z - eta * g / (RMS(g) + eps), frame-0 slice zeroed
"""
from __future__ import annotations

import torch

EPS = 1e-6


def bg_loss(F_hat: torch.Tensor, F_geo: torch.Tensor, C_geo: torch.Tensor, O_t: torch.Tensor) -> torch.Tensor:
    """F_hat, F_geo [N,2]; C_geo, O_t [N] in [0,1]. Weighted squared displacement error on covered background patches."""
    w = (C_geo * (1.0 - O_t)).clamp_min(0.0)
    return (w * ((F_hat - F_geo) ** 2).sum(-1)).sum() / (w.sum() + EPS)


def centroid_loss(F_hat: torch.Tensor, target: torch.Tensor, O_t: torch.Tensor) -> torch.Tensor:
    """Centroid limit of L_obj (§5.2): the occupancy-weighted mean readout on object patches against the object's expected
    displacement (one [2] vector for this frame, backward: p_0(centroid) - p_t(centroid))."""
    w = O_t.clamp_min(0.0)
    if w.sum() < EPS:
        return F_hat.new_zeros(())
    mean_hat = (w[:, None] * F_hat).sum(0) / (w.sum() + EPS)
    return ((mean_hat - target) ** 2).sum()


def perp_loss(F_hat: torch.Tensor, c: torch.Tensor, lines: torch.Tensor, H_pts: torch.Tensor, line_mode: torch.Tensor, C_max: torch.Tensor) -> torch.Tensor:
    """§5.3. c [N,2] patch centres; lines [N,3] (A,B,C) of the epipolar line of each query in the key frame (line mode);
    H_pts [N,2] the homography image of each query (homography mode); line_mode [N] bool selects per patch; C_max [N].
    The expected source q' = c + F_hat; d = |A u' + B v' + C| / sqrt(A^2+B^2) or ||q' - H q||. Weight 1 - C_max."""
    qp = c + F_hat
    num = (lines[:, 0] * qp[:, 0] + lines[:, 1] * qp[:, 1] + lines[:, 2]).abs()
    den = torch.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2).clamp_min(EPS)
    d_line = num / den
    d_h = torch.linalg.norm(qp - H_pts, dim=-1)
    d = torch.where(line_mode, d_line, torch.nan_to_num(d_h, nan=0.0))
    w = (1.0 - C_max).clamp(0.0, 1.0)
    return (w * d ** 2).sum() / (w.sum() + EPS)


def mass_terms(A: torch.Tensor, c: torch.Tensor, F_geo: torch.Tensor, r_w: float, w: torch.Tensor, F_hat: torch.Tensor = None) -> dict:
    """A [heads,N,N] (or [N,N]); c [N,2]; F_geo [N,2]; r_w window radius (Chebyshev, as the readout's window); w [N] the L_bg weights.
    Returns L_mass (negative weighted log mass inside the window around the geometric target), the weighted mean mass at the
    target (diagnostic), and the share of weighted queries whose READOUT window (around the detached argmax) contains the target."""
    P = A.mean(0) if A.ndim == 3 else A
    tgt = c + F_geo                                                              # [N,2] where the source should be
    inside = ((c[None, :, :] - tgt[:, None, :]).abs().amax(-1) <= r_w).to(P.dtype)   # [N(query), N(key)]
    mass = (P * inside).sum(-1)                                                   # [N]
    L_mass = -(w * torch.log(mass + EPS)).sum() / (w.sum() + EPS)
    out = {"L_mass": L_mass, "mass_at_target_mean": float(((w * mass).sum() / (w.sum() + EPS)).detach())}
    am = P.argmax(-1)
    contains = ((c[am] - tgt).abs().amax(-1) <= r_w).to(P.dtype)
    out["window_contains_target_share"] = float((w * contains).sum() / (w.sum() + EPS))
    return out


def guided_update(z: torch.Tensor, g: torch.Tensor, eta: float) -> torch.Tensor:
    """§5.4: RMS-normalised descent on the latent, the frame-0 (I2V anchor) slice left untouched. z, g [1,C,T,h,w]."""
    g = g.clone()
    g[:, :, 0] = 0.0
    rms = torch.sqrt((g.float() ** 2).mean()) + EPS
    return (z.float() - float(eta) * g.float() / rms).to(z.dtype)


def tile_score(frame_u8, bg_mask, min_lag_px: int = 8, max_lag_px: int = 96) -> dict:
    """Reviewer 2026-10-10, the grader's tile detector: on the background of one decoded frame, the height of the SECONDARY peak of the
    normalised 2-D autocorrelation (lags between min_lag_px and max_lag_px, the central peak excluded), relative to the central value.
    A natural ground reads low (texture decorrelates with lag); repeated tiles read high (a second peak at the tile period). Numpy only.
    frame_u8 [H,W,3]; bg_mask [H,W] bool (background: outside the object, no sky). Returns the peak height, its lag and the share of
    background pixels used."""
    import numpy as np
    g = frame_u8.astype(np.float64).mean(-1)
    m = np.asarray(bg_mask, bool)
    if m.mean() < 0.05:
        return {"secondary_autocorr_peak": None, "lag_px": None, "bg_share": float(m.mean())}
    g = np.where(m, g, np.nan)
    g = g - np.nanmean(g)
    g = np.where(m, g, 0.0)                                                      # masked pixels contribute zero (mean removed first)
    Hh, W = g.shape
    F_ = np.fft.rfft2(g); ac = np.fft.irfft2(F_ * np.conj(F_), s=(Hh, W))
    Fm = np.fft.rfft2(m.astype(np.float64)); n_ov = np.fft.irfft2(Fm * np.conj(Fm), s=(Hh, W))    # overlap count per lag, for normalisation
    with np.errstate(divide="ignore", invalid="ignore"):
        acn = np.where(n_ov > 0.2 * m.sum(), ac / np.maximum(n_ov, 1.0), np.nan)
    c0 = acn[0, 0]
    if not np.isfinite(c0) or c0 <= 0:
        return {"secondary_autocorr_peak": None, "lag_px": None, "bg_share": float(m.mean())}
    acn = acn / c0
    yy, xx = np.meshgrid(np.arange(Hh), np.arange(W), indexing="ij")
    dy = np.minimum(yy, Hh - yy); dx = np.minimum(xx, W - xx)
    r = np.sqrt(dx ** 2 + dy ** 2)
    ring = (r >= min_lag_px) & (r <= max_lag_px) & np.isfinite(acn)
    if not ring.any():
        return {"secondary_autocorr_peak": None, "lag_px": None, "bg_share": float(m.mean())}
    k = np.argmax(np.where(ring, acn, -np.inf))
    return {"secondary_autocorr_peak": float(acn.reshape(-1)[k]), "lag_px": float(r.reshape(-1)[k]), "bg_share": float(m.mean())}
