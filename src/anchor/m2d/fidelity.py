"""Fidelity statistics (SKILL_m2_d v1.1 §4-5): which part of the control the generator honours.

    obj_fidelity = IoU(segment(generated), M_obj)
    bg_fidelity  = corr(zeta_hat(generated), zeta_bg) outside M_obj      (Spearman: the estimator is affine-free)
    E_loc        = |centroid(generated mask) - p_ta| / sqrt(H^2 + W^2)
    E_area       = |log(A_generated / A_obj_rendered)|                   (the rendered proxy's area, NOT A_src)

Pure functions, CPU. RECORDED, never gated here (§7): they measure the hypothesis under test.
"""
from __future__ import annotations

import numpy as np

from . import instrument


def _centroid(mask):
    jj, ii = np.nonzero(mask)
    assert len(jj), "empty mask has no centroid"
    return np.array([ii.mean() + 0.5, jj.mean() + 0.5])                  # pixel-centre convention (u, v)


def object_stats(mask_gen, M_obj, p_ta_uv) -> dict:
    """mask_gen: segment(generated) [H,W] bool; M_obj: the control's object mask; p_ta_uv: the placement's
    anchor point in THIS image's pixels. An empty generated mask is a measurement (object not found), not an error."""
    g = np.asarray(mask_gen, bool); m = np.asarray(M_obj, bool)
    assert g.shape == m.shape and m.any(), (g.shape, m.shape)
    H, W = m.shape; diag = float(np.hypot(H, W))
    if not g.any():
        return {"object_found": False, "obj_fidelity": 0.0, "A_generated_frac": 0.0, "A_obj_rendered_frac": float(m.mean())}
    cg, cm = _centroid(g), _centroid(m)
    return {"object_found": True, "obj_fidelity": float((g & m).sum() / (g | m).sum()),
            "E_loc": float(np.linalg.norm(cg - np.asarray(p_ta_uv, float)) / diag),
            "E_loc_vs_rendered_centroid": float(np.linalg.norm(cg - cm) / diag),   # p_ta and the rendered centroid differ by delta_p
            "E_area": float(abs(np.log(g.sum() / m.sum()))), "log_area_ratio": float(np.log(g.sum() / m.sum())),
            "A_generated_frac": float(g.mean()), "A_obj_rendered_frac": float(m.mean()),
            "centroid_generated_uv": cg.tolist()}


def bg_fidelity(zeta_hat, zeta_ctrl, M_obj, valid, mask_gen=None) -> dict:
    """Spearman rho between the estimated disparity of the generated image and the control's background
    disparity, over pixels that are background IN BOTH: outside M_obj, outside the generated object (a
    generated object larger than its outline would otherwise count as background error), and valid (sky
    has no disparity to honour)."""
    zh = np.asarray(zeta_hat, np.float64); zc = np.asarray(zeta_ctrl, np.float64)
    assert zh.shape == zc.shape == np.asarray(M_obj).shape
    dom = ~np.asarray(M_obj, bool) & np.asarray(valid, bool) & np.isfinite(zh)
    if mask_gen is not None:
        dom &= ~np.asarray(mask_gen, bool)
    assert dom.sum() > 100, "no background domain to correlate over"
    return {"bg_fidelity": float(instrument.spearman(zh, zc, mask=dom)), "bg_domain_frac": float(dom.mean())}
