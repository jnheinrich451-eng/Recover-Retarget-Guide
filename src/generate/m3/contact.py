"""Object-ground contact check for the frame-0 lift (reviewer 2026-10-06c; brief §3.2 `g_obj_contact`).

Question: is the base of G_tar at the depth of the lifted ground it stands on? Parallax scales with 1/Z, so a base that
floats or sinks slides the object over the ground under camera motion. All arrays live on the square-pixel depth grid.
Nothing here changes the lift: the check reports r_c; the re-gauge (scaling G_tar about the camera centre) is a separate,
explicit function that is applied only on John's decision.
"""
from __future__ import annotations

import numpy as np


def base_region(M_g: np.ndarray, contact_rows: float) -> np.ndarray:
    """Lowest `contact_rows` fraction of the rendered object's image height (at least one row)."""
    rows = np.nonzero(M_g.any(1))[0]
    assert len(rows), "empty object render"
    n = max(1, int(round(contact_rows * (rows.max() - rows.min() + 1))))
    out = np.zeros_like(M_g)
    out[rows.max() - n + 1: rows.max() + 1] = M_g[rows.max() - n + 1: rows.max() + 1]
    return out


def ground_band_below(mask: np.ndarray, valid: np.ndarray, cols: np.ndarray, band_px: int) -> np.ndarray:
    """Background pixels in the `band_px` rows immediately below the mask's bottom edge, per column, restricted to `cols`
    (columns under the object's base) and to valid depth. Empty when the mask reaches the bottom of the frame."""
    H, W = mask.shape
    out = np.zeros_like(mask)
    for j in np.nonzero(cols & mask.any(0))[0]:
        b = int(np.nonzero(mask[:, j])[0].max())
        out[b + 1: min(b + 1 + band_px, H), j] = True
    return out & valid & ~mask


def contact_ratio(zbuf_g, M_g, depth_src, valid, anchor_mask, contact_rows, band_px, block_px, n_boot, seed):
    """r_c = Z_base(G_tar) / Z_ground(k_bg x depth) with a column-block bootstrap; plus the upper-half body ratio."""
    base = base_region(M_g, contact_rows) & np.isfinite(zbuf_g)
    band = ground_band_below(anchor_mask | M_g, valid & np.isfinite(depth_src) & (depth_src > 0), base.any(0), band_px)
    rec = {"n_base_px": int(base.sum()), "n_ground_px": int(band.sum())}
    rows = np.nonzero(M_g.any(1))[0]
    top = np.zeros_like(M_g); top[rows.min(): (rows.min() + rows.max()) // 2 + 1] = True
    ub_g, ub_i = M_g & top & np.isfinite(zbuf_g), anchor_mask & top & valid & np.isfinite(depth_src)
    rec["upper_body_ratio"] = float(np.median(zbuf_g[ub_g]) / np.median(depth_src[ub_i])) if ub_g.any() and ub_i.any() else None
    if band.sum() == 0:
        return {**rec, "contact_band_empty": True, "r_c": None, "ci95": None,
                "mask_bottom_row": int(rows.max()), "frame_rows": int(M_g.shape[0])}
    zb, zg = float(np.median(zbuf_g[base])), float(np.median(depth_src[band]))
    cols = np.nonzero(band.any(0))[0]
    blocks = np.unique(cols // block_px)
    rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        pick = rng.choice(blocks, len(blocks))
        sel_b = np.concatenate([zbuf_g[:, (np.arange(M_g.shape[1]) // block_px) == p][base[:, (np.arange(M_g.shape[1]) // block_px) == p]] for p in pick])
        sel_g = np.concatenate([depth_src[:, (np.arange(M_g.shape[1]) // block_px) == p][band[:, (np.arange(M_g.shape[1]) // block_px) == p]] for p in pick])
        if len(sel_b) and len(sel_g):
            boots.append(np.median(sel_b) / np.median(sel_g))
    lo, hi = (np.quantile(boots, [0.025, 0.975]) if boots else (np.nan, np.nan))
    # Known instrument bias (self-test: +4 % on a standing box): the band lies below the base, where a receding ground is
    # NEARER. Reported beside r_c, never replacing it: on a plane 1/z is linear in the image row, so fit it over the band and
    # read the ground at the base's own rows.
    vb, _ = np.nonzero(band); vbase, _ = np.nonzero(base)
    r_ex = None
    if len(np.unique(vb)) >= 3:
        a, b = np.polyfit(vb.astype(float), 1.0 / depth_src[band], 1)
        zg_ex = 1.0 / (a * float(np.median(vbase)) + b)
        r_ex = float(zb / zg_ex) if np.isfinite(zg_ex) and zg_ex > 0 else None
    return {**rec, "contact_band_empty": False, "Z_base_gtar": zb, "Z_ground_image": zg, "r_c": zb / zg,
            "ci95": [float(lo), float(hi)], "n_col_blocks": int(len(blocks)), "r_c_ground_extrapolated_to_base": r_ex}


def fit_ground_plane(depth_src, valid, obj_mask, Kq, nbhd_px: int, iters: int, rel_tol: float, min_inliers: int, seed: int):
    """RANSAC plane n.X + d = 0 on the lifted ground (k_bg x depth) in a neighbourhood of the object: pixels within nbhd_px of
    the mask, below its mid row, outside it, with valid depth. Fitted from the DEPTH MODEL's ground, never from the handoff's
    plane (that one shares G_tar's gauge and would put one witness on both sides of the ratio). Returns dict or None."""
    import cv2
    H, W = depth_src.shape
    rows = np.nonzero(obj_mask.any(1))[0]
    k = 2 * int(nbhd_px) + 1
    near = cv2.dilate(obj_mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & ~obj_mask
    near[: (rows.min() + rows.max()) // 2] = False
    ok = near & valid & np.isfinite(depth_src) & (depth_src > 0)
    vv, uu = np.nonzero(ok)
    if len(vv) < max(min_inliers, 3):
        return None
    X = (np.linalg.inv(Kq) @ np.stack([uu + 0.5, vv + 0.5, np.ones(len(uu))], 0)).T * depth_src[vv, uu][:, None]
    rng = np.random.default_rng(seed); best = None
    for _ in range(int(iters)):
        a, b, c = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn; d = -float(n @ a)
        inl = np.abs(X @ n + d) / X[:, 2] < rel_tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < min_inliers:
        return None
    Xi = X[best]; c0 = Xi.mean(0)
    n = np.linalg.svd(Xi - c0, full_matrices=False)[2][-1]                 # least-squares refit on the inliers
    d = -float(n @ c0)
    res = np.abs(Xi @ n + d) / Xi[:, 2]
    return {"n": n, "d": d, "n_candidates": int(len(X)), "n_inliers": int(best.sum()), "inlier_frac": float(best.mean()),
            "resid_rel_median": float(np.median(res))}


def contact_ratio_plane(zbuf_g, M_g, plane, Kq, contact_rows: float, block_px: int, n_boot: int, seed: int, anchor_mask=None):
    """Corrected contact ratio (reviewer 2026-10-06d): both sides read ALONG THE SAME RAYS. For every column whose lowest object
    pixel lies in the base rows: Z_base = G_tar's depth at that pixel; Z_ground = depth at which that pixel's ray meets the
    fitted ground plane, Z = -d / (n . K^-1 u~). r_c = median of the per-column ratios; column-block bootstrap."""
    base = base_region(M_g, contact_rows) & np.isfinite(zbuf_g)
    cols = np.nonzero(base.any(0))[0]
    if plane is None or len(cols) == 0:
        return {"plane_unreliable": plane is None, "r_c_plane": None, "ci95_plane": None, "n_cols": int(len(cols))}
    bot = np.array([np.nonzero(base[:, j])[0].max() for j in cols])
    # The renderer covers a pixel when its CENTRE is inside the object, so the true bottom edge lies between the centre of
    # row `bot` and the centre of row bot+1: on average at v = bot + 1.0. Reading the ray at the pixel centre (bot + 0.5)
    # looks half a pixel too high and reads the ground too far (-1.3 % on the synthetic box at depth 9, pitch 0.25).
    rays = (np.linalg.inv(Kq) @ np.stack([cols + 0.5, bot + 1.0, np.ones(len(cols))], 0)).T
    den = rays @ plane["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        zg = np.where(np.abs(den) > 1e-12, -plane["d"] / den, np.nan)
    ok = np.isfinite(zg) & (zg > 0)
    if not ok.any():
        return {"plane_unreliable": True, "r_c_plane": None, "ci95_plane": None, "n_cols": int(len(cols))}
    ratio = zbuf_g[bot[ok], cols[ok]] / zg[ok]
    blk = cols[ok] // block_px; ids = np.unique(blk)
    rng = np.random.default_rng(seed)
    boots = [float(np.median(np.concatenate([ratio[blk == b] for b in rng.choice(ids, len(ids))]))) for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    extra = {}
    # Diagnostic (2026-10-06): the ground's depth changes fast with the image row, so a vertical misalignment between G_tar's
    # rendered bottom edge and the PAINTED object's bottom edge reads as a contact failure with no depth error at all.
    # Report the row offset, the ground's relative depth change per row at the base, and the ratio re-read with the ground
    # taken at the painted object's own bottom edge.
    Kinv = np.linalg.inv(Kq)
    r1 = (Kinv @ np.stack([cols + 0.5, bot + 2.0, np.ones(len(cols))], 0)).T @ plane["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        zg1 = -plane["d"] / r1
    sens = np.abs(zg1[ok] - zg[ok]) / zg[ok]
    extra["ground_rel_depth_change_per_row"] = float(np.median(sens[np.isfinite(sens)])) if np.isfinite(sens).any() else None
    if anchor_mask is not None:
        has = anchor_mask[:, cols].any(0)
        ba = np.array([np.nonzero(anchor_mask[:, j])[0].max() if h else -1 for j, h in zip(cols, has)])
        use = ok & has
        if use.any():
            ra = (Kinv @ np.stack([cols[use] + 0.5, ba[use] + 1.0, np.ones(int(use.sum()))], 0)).T @ plane["n"]
            with np.errstate(divide="ignore", invalid="ignore"):
                zga = -plane["d"] / ra
            g = np.isfinite(zga) & (zga > 0)
            extra["bottom_row_offset_px_gtar_minus_anchor"] = float(np.median((bot[use] - ba[use])[g])) if g.any() else None
            extra["r_c_ground_at_painted_edge"] = float(np.median(zbuf_g[bot[use], cols[use]][g] / zga[g])) if g.any() else None
            extra["n_cols_painted_edge"] = int(g.sum())
    return {**extra, "plane_unreliable": False, "r_c_plane": float(np.median(ratio)), "ci95_plane": [float(lo), float(hi)], "n_cols": int(ok.sum()),
            "n_col_blocks": int(len(ids)), "plane_inliers": plane["n_inliers"], "plane_inlier_frac": plane["inlier_frac"],
            "plane_resid_rel_median": plane["resid_rel_median"]}


def reading_plane(rc: dict, depth_gate_failed: bool, contact_tol: float) -> str:
    """Pre-registered reading (reviewer 2026-10-06c/d/e). The gate column is B: the ground plane read along the PAINTED
    object's bottom-edge rays against G_tar's base depth (`r_c_ground_at_painted_edge`). Column A (`r_c_plane`, the ground at the
    mesh's own bottom edge) multiplies the depth question by a shape question and is a diagnostic only."""
    return decision(rc, contact_tol)["reading"] + (" (median depth gate failed by extent)" if depth_gate_failed and decision(rc, contact_tol)["s_c"] == 1.0 else "")


def decision(rc: dict, contact_tol: float, lattice_px: int = 16) -> dict:
    """The rule as a record: s_c = 1 (carry forward) or s_c = 1 / B (re-gauge the OBJECT to the ground, never the reverse; the
    object's displacement is scaled by the same s_c). No contact line in view -> s_c = 1, labelled. Applied only by John's decision."""
    B = rc.get("r_c_ground_at_painted_edge")
    off = rc.get("bottom_row_offset_px_gtar_minus_anchor")
    out = {"gate_column": "B: ground at the painted object's bottom edge", "r_c_B": B, "r_c_A": rc.get("r_c_plane"), "edge_offset_rows": off,
           "edge_offset_exceeds_half_patch": None if off is None else bool(abs(off) > lattice_px / 2), "contact_tol": contact_tol}
    if rc.get("object_reaches_frame_bottom"):
        return {**out, "s_c": 1.0, "outcome": "no_contact_line_in_view", "reading": "object cut by the frame bottom: no contact line in view; nothing to re-gauge against"}
    if B is None:
        return {**out, "s_c": 1.0, "outcome": "no_ground_plane", "reading": "no ground plane under the object: gate stands, nothing to re-gauge against"}
    if abs(B - 1.0) <= contact_tol:
        return {**out, "s_c": 1.0, "outcome": "contact_holds", "reading": "contact holds: carry forward"}
    return {**out, "s_c": float(1.0 / B), "outcome": "contact_fails_regauge",
            "reading": f"contact fails: object re-gauged to the ground, s_c = {1.0 / B:.4f} (displacement scaled alike; D-M3-03 for this pair)"}


def rescale_about_camera(V: np.ndarray, s_c: float) -> np.ndarray:
    """Scale G_tar about the camera centre: every vertex keeps its viewing ray, so the projected footprint is unchanged and
    every depth is multiplied by s_c. The object's displacement must be scaled by the same s_c (D-M3-03, per pair)."""
    assert np.isfinite(s_c) and s_c > 0
    return np.asarray(V, np.float64) * float(s_c)


def reading(rc: dict, depth_gate_failed: bool, contact_tol: float) -> str:
    """The pre-registered reading (reviewer 2026-10-06c). A label for the report, not an action."""
    if rc.get("contact_band_empty"):
        return "contact_band_empty: gate stands, nothing to re-gauge against"
    if abs(rc["r_c"] - 1.0) <= contact_tol:
        return "contact holds: carry forward" + (" (depth gate failed by extent, no change to the lift)" if depth_gate_failed else "")
    return f"contact fails: re-gauge the object to the ground proposed, s_c = {1.0 / rc['r_c']:.4f}"
