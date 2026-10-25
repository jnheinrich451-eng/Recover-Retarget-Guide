"""Render target (reviewer 2026-10-10, v2.7 pending): the frame-0 content moved by the geometry. Pure numpy, unit-tested.

    lift_dense        every valid background pixel of the anchor's square-pixel grid -> world point (camera t_a, target gauge) + colour
    render_frame      z-buffered point render of coloured points into an H x W pixel grid under (E_t, K): image, coverage, depth
    object_footprint  z-buffered footprint of the object's primitives at a frame (translation-only path)
    latent_mask       pixel masks of the 4 frames of a latent chunk -> one latent-grid mask (all four covered, min-pooled 8x8)

The mask M_t = rendered (covered) & not object & not sky, eroded by one latent pixel; frame 0 is the anchor itself (E_0 = I).
Uncovered pixels are filled with the anchor frame so the VAE sees a full image; the mask keeps them out of the correction.
"""
from __future__ import annotations

import numpy as np


def lift_dense(depth_src: np.ndarray, valid: np.ndarray, Kq: np.ndarray, colours_u8: np.ndarray):
    """depth_src [H,W] (NaN invalid), valid [H,W] bool, Kq [3,3] (square-pixel grid), colours [H,W,3] uint8 on the same grid.
    Returns X [N,3] world points (camera t_a) and C [N,3] uint8."""
    H, W = depth_src.shape
    ok = np.asarray(valid, bool) & np.isfinite(depth_src) & (depth_src > 0)
    vv, uu = np.nonzero(ok)
    u, v, z = uu + 0.5, vv + 0.5, depth_src[vv, uu]
    X = (np.linalg.inv(np.asarray(Kq, np.float64)) @ np.stack([u, v, np.ones_like(u)], 0)).T * z[:, None]
    return X, colours_u8[vv, uu]


def render_frame(X: np.ndarray, C: np.ndarray, E_t: np.ndarray, K: np.ndarray, H: int, W: int, splat: int = 1):
    """Z-buffered nearest-pixel render. Returns image [H,W,3] uint8 (0 where uncovered), covered [H,W] bool, depth [H,W] (inf uncovered).
    splat = 1 renders each point into a (2*splat+1)^2 block so a stride-sampled cloud leaves no pinholes."""
    Xc = X @ np.asarray(E_t, np.float64)[:3, :3].T + np.asarray(E_t, np.float64)[:3, 3]
    z = Xc[:, 2]
    front = z > 1e-6
    p = (np.asarray(K, np.float64) @ Xc[front].T).T
    u = np.floor(p[:, 0] / p[:, 2]).astype(int); v = np.floor(p[:, 1] / p[:, 2]).astype(int)
    zf = z[front]; Cf = C[front]
    img = np.zeros((H, W, 3), np.uint8); depth = np.full((H, W), np.inf)
    order = np.argsort(-zf)                                                          # far first, near last wins (painter's), exact for nearest-pixel
    for du in range(-splat, splat + 1):
        for dv in range(-splat, splat + 1):
            uu, vv = u[order] + du, v[order] + dv
            inside = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            ui, vi, zi, ci = uu[inside], vv[inside], zf[order][inside], Cf[order][inside]
            # painter's order within this pass; across passes keep the nearer
            closer = zi < depth[vi, ui]
            depth[vi[closer], ui[closer]] = zi[closer]
            img[vi[closer], ui[closer]] = ci[closer]
    return img, np.isfinite(depth), depth


def object_footprint(V: np.ndarray, shift: np.ndarray, E_t: np.ndarray, K: np.ndarray, H: int, W: int, bg_depth: np.ndarray, dilate_px: int = 8) -> np.ndarray:
    """Pixels where the object's primitives V [M,3] (world, frame 0) translated by `shift` [3] project in front of the rendered background."""
    import cv2
    Xc = (V + shift) @ np.asarray(E_t, np.float64)[:3, :3].T + np.asarray(E_t, np.float64)[:3, 3]
    z = Xc[:, 2]; front = z > 1e-6
    p = (np.asarray(K, np.float64) @ Xc[front].T).T
    u = np.floor(p[:, 0] / p[:, 2]).astype(int); v = np.floor(p[:, 1] / p[:, 2]).astype(int)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    m = np.zeros((H, W), bool)
    zf = z[front][inside]; ui, vi = u[inside], v[inside]
    in_front = zf <= bg_depth[vi, ui] * 1.02 + 1e-6
    m[vi[in_front], ui[in_front]] = True
    if dilate_px > 0:
        m = cv2.dilate(m.astype(np.uint8), np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)).astype(bool)
    return m


def latent_mask(masks_px: np.ndarray, latent_px: int = 8) -> np.ndarray:
    """masks_px [n_frames, H, W] bool (the frames of one latent chunk) -> [H/8, W/8] bool: a latent pixel is kept iff every frame's
    mask covers its whole 8x8 block."""
    m = np.asarray(masks_px, bool).all(0)
    H, W = m.shape
    assert H % latent_px == 0 and W % latent_px == 0, (H, W)
    return m.reshape(H // latent_px, latent_px, W // latent_px, latent_px).all((1, 3))


def flow_at_queries(X: np.ndarray, E: np.ndarray, K: np.ndarray, q0: np.ndarray, H: int, W: int, cell_px: int = 16, min_pts: int = 3, shift=None):
    """The target geometry's flow at query pixels. X [N,3] world points (the render's dense lift), E [T,4,4] with E[0] = I, K [3,3] pixel
    intrinsics, q0 [Q,2] query pixels at frame 0. For each query, the median displacement over frames (pixels) of the points whose frame-0
    projection lands in the query's cell_px x cell_px cell and that stay in front of the camera at every frame. Returns F [T,Q,2] in pixels
    (NaN where fewer than min_pts points) and the median frame-0 depth per query [Q] (NaN likewise).
    The grader's expected flow is taken here, at its own queries, so measured and expected share one sampling (rule 15)."""
    E = np.asarray(E, np.float64); K = np.asarray(K, np.float64); X = np.asarray(X, np.float64)
    if shift is not None:                                                            # shift [T,3]: the points translate rigidly per frame (the object along its path)
        Xs = X[None] + np.asarray(shift, np.float64)[:, None, :]
        Xc = np.einsum("tij,tnj->tni", E[:, :3, :3], Xs) + E[:, None, :3, 3]
    else:
        Xc = np.einsum("tij,nj->tni", E[:, :3, :3], X) + E[:, None, :3, 3]
    p = np.einsum("ij,tnj->tni", K, Xc); z = Xc[..., 2]
    p = p[..., :2] / np.maximum(p[..., 2:3], 1e-9)
    keep = (z > 1e-6).all(0) & (p[0, :, 0] >= 0) & (p[0, :, 0] < W) & (p[0, :, 1] >= 0) & (p[0, :, 1] < H)
    p, z = p[:, keep], z[:, keep]
    ncol = W // cell_px + 1
    cell = (p[0, :, 1] // cell_px).astype(int) * ncol + (p[0, :, 0] // cell_px).astype(int)
    order = np.argsort(cell, kind="stable"); cs = cell[order]
    bounds = np.searchsorted(cs, np.arange(cs.max() + 2)) if len(cs) else None
    q0 = np.asarray(q0, np.float64)
    qcell = (np.clip(q0[:, 1], 0, H - 1) // cell_px).astype(int) * ncol + (np.clip(q0[:, 0], 0, W - 1) // cell_px).astype(int)
    F = np.full((E.shape[0], len(q0), 2), np.nan); zq = np.full(len(q0), np.nan)
    disp = p - p[0][None]
    for i, c in enumerate(qcell):
        if bounds is None or c >= len(bounds) - 1:
            continue
        pts = order[bounds[c]: bounds[c + 1]]
        if len(pts) < min_pts:
            continue
        F[:, i] = np.median(disp[:, pts], axis=1); zq[i] = np.median(z[0, pts])
    return F, zq


def nearest_fill(img: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Fill every uncovered pixel of img [H,W,3] with the colour of its nearest covered pixel (Euclidean, via the distance transform's labels).
    Reviewer 2026-10-11 §3: the fill is content that moves with the camera (it is the frame's own render), never the anchor at the same pixel,
    which has zero motion and biased every newly revealed strip toward standing still (and showed the anchor's object as a ghost)."""
    from scipy import ndimage
    cov = np.asarray(covered, bool)
    if cov.all():
        return img.copy()
    assert cov.any(), "nothing covered: nothing to fill from"
    # the exact Euclidean distance transform of the uncovered set returns, per pixel, the indices of its nearest covered pixel
    _, (src_v, src_u) = ndimage.distance_transform_edt(~cov, return_indices=True)
    out = img[src_v, src_u]
    out[cov] = img[cov]
    return out


def pushpull_fill(img: np.ndarray, covered: np.ndarray, max_levels: int = 12) -> np.ndarray:
    """Push-pull fill: every uncovered pixel of img [H,W,3] takes the coverage-weighted mean of the covered content at the coarsest pyramid level
    that has any, so a hole is filled with a smooth blend of its surroundings instead of the streaks of a nearest-pixel fill. Covered pixels are
    returned unchanged. (Reviewer 2026-10-11 §3: fill from the same frame's covered render, push-pull or nearest-valid; push-pull chosen after a
    nearest-valid fill produced Voronoi streaks across the sky and the hole behind the object.)"""
    import cv2
    cov = np.asarray(covered, bool)
    if cov.all():
        return img.copy()
    assert cov.any(), "nothing covered: nothing to fill from"
    w = cov.astype(np.float32); c = img.astype(np.float32) * w[..., None]
    pyr_c, pyr_w = [c], [w]
    for _ in range(max_levels):
        if pyr_w[-1].min() > 0 or min(pyr_w[-1].shape) < 2:
            break
        pyr_c.append(cv2.pyrDown(pyr_c[-1])); pyr_w.append(cv2.pyrDown(pyr_w[-1]))
    # pull: from the coarsest level down, where a level has no weight take the coarser level's (normalised) value
    cc, ww = pyr_c[-1], pyr_w[-1]
    val = cc / np.maximum(ww, 1e-6)[..., None]
    for lev in range(len(pyr_c) - 2, -1, -1):
        up = cv2.resize(val, (pyr_c[lev].shape[1], pyr_c[lev].shape[0]), interpolation=cv2.INTER_LINEAR)
        wl = pyr_w[lev]; vl = pyr_c[lev] / np.maximum(wl, 1e-6)[..., None]
        # blend: where the level is fully covered keep it, where empty take the coarser value, in between mix by weight
        a = np.clip(wl, 0, 1)[..., None]
        val = a * vl + (1 - a) * up
    out = np.clip(np.round(val), 0, 255).astype(np.uint8)
    out[cov] = img[cov]
    return out


def sky_at_infinity(anchor: np.ndarray, sky0: np.ndarray, E_t: np.ndarray, K: np.ndarray):
    """The anchor's sky (pixels with no depth) carried to frame t as content at infinity: a pure-rotation warp p_t = K R_t K^-1 p_0 (E_t = [R_t | t_t],
    E_0 = I; translation does not move a point at infinity). Returns the warped image [H,W,3] and the warped sky mask [H,W] bool. The sky then
    moves exactly as the camera's rotation says, which is the only motion a sky has."""
    import cv2
    H, W = sky0.shape
    R = np.asarray(E_t, np.float64)[:3, :3]
    Hm = np.asarray(K, np.float64) @ R @ np.linalg.inv(np.asarray(K, np.float64))
    img = cv2.warpPerspective(anchor, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    msk = cv2.warpPerspective(sky0.astype(np.uint8), Hm, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE).astype(bool)
    return img, msk
