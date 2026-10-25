"""Pixel-exact silhouette of a camera-frame mesh — the runtime self-check (SKILL §10 test 1).

A pixel is inside when its centre (i+0.5, j+0.5) lies in any projected triangle: the same
rule as the m2_c harness z-buffer (`proxy.rasterize`), vectorised over (triangle, bbox-pixel)
pairs. Validated 2026-09-13 against the harness on far/near ellipsoids and a 12-triangle box:
IoU 1.0000 with identical pixel counts. NOT cv2.fillPoly: with many polygons in one call it
fills by the even-odd rule, so front and back layers of a closed mesh cancel to a wireframe
(measured: two overlapping triangles -> 920 px vs 1281 for the union).
"""
from __future__ import annotations

import numpy as np


def silhouette(X_cv, F, K, H, W, max_pairs=4_000_000) -> np.ndarray:
    X = np.asarray(X_cv, np.float64)
    F = np.asarray(F, int)
    K = np.asarray(K, np.float64)
    keep = (X[F][..., 2] > 1e-6).all(1)
    img = np.zeros((H, W), bool)
    if not keep.any():
        return img
    P = X @ K.T
    uv = P[:, :2] / P[:, 2:3]
    T = uv[F[keep]]
    i0 = np.clip(np.floor(T[:, :, 0].min(1) - 0.5), 0, W - 1).astype(np.int64)
    i1 = np.clip(np.ceil(T[:, :, 0].max(1) - 0.5), 0, W - 1).astype(np.int64)
    j0 = np.clip(np.floor(T[:, :, 1].min(1) - 0.5), 0, H - 1).astype(np.int64)
    j1 = np.clip(np.ceil(T[:, :, 1].max(1) - 0.5), 0, H - 1).astype(np.int64)
    off = (T[:, :, 0].max(1) < 0) | (T[:, :, 0].min(1) > W) | (T[:, :, 1].max(1) < 0) | (T[:, :, 1].min(1) > H)
    w = i1 - i0 + 1; h = j1 - j0 + 1
    n = np.where(off | (w <= 0) | (h <= 0), 0, w * h)
    (u0, v0), (u1, v1), (u2, v2) = T[:, 0].T, T[:, 1].T, T[:, 2].T
    den = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
    n = np.where(np.abs(den) < 1e-15, 0, n)
    csum = np.cumsum(n)
    a = 0
    while a < len(n):
        base = csum[a] - n[a]
        b = max(int(np.searchsorted(csum - base, max_pairs, side="right")), a + 1)
        nn = n[a:b]
        tot = int(nn.sum())
        if tot == 0:
            a = b; continue
        tid = np.repeat(np.arange(a, b), nn)
        k = np.arange(tot) - np.repeat(np.cumsum(nn) - nn, nn)
        i = i0[tid] + k % w[tid]; j = j0[tid] + k // w[tid]
        px, py = i + 0.5, j + 0.5
        d = den[tid]
        w0 = ((v1[tid] - v2[tid]) * (px - u2[tid]) + (u2[tid] - u1[tid]) * (py - v2[tid])) / d
        w1 = ((v2[tid] - v0[tid]) * (px - u2[tid]) + (u0[tid] - u2[tid]) * (py - v2[tid])) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-12) & (w1 >= -1e-12) & (w2 >= -1e-12)
        img[j[inside], i[inside]] = True
        a = b
    return img


def iou(mask_a, mask_b) -> float:
    a, b = np.asarray(mask_a, bool), np.asarray(mask_b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def boundary_edges(F):
    """(edges used by exactly one face, total edges): watertight iff the first is 0."""
    F = np.asarray(F, int)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return int((cnt == 1).sum()), int(len(cnt))


def centroid_aligned_iou(X_cv, F, K, H, W, mask, iters=2):
    """Translate the mesh in camera x, y (at its own median depth) until its silhouette centroid
    meets the mask centroid, then IoU: shape + orientation only, translation removed (the probe's
    P1/P2 instrument). Returns (iou_aligned, shift_px (du, dv) applied). The GEOMETRY is not
    modified by callers: this is a selection criterion and a QC number (review, 2026-09-14)."""
    X = np.asarray(X_cv, np.float64).copy()
    K = np.asarray(K, np.float64)
    my, mx = np.nonzero(np.asarray(mask, bool))
    if len(mx) == 0:
        return float("nan"), (float("nan"), float("nan"))
    cm = np.array([mx.mean() + 0.5, my.mean() + 0.5])
    zbar = float(np.median(X[:, 2]))
    shift = np.zeros(2)
    for _ in range(iters):
        ry, rx = np.nonzero(silhouette(X, F, K, H, W))
        if len(rx) == 0:
            return float("nan"), (float("nan"), float("nan"))
        d = cm - np.array([rx.mean() + 0.5, ry.mean() + 0.5])
        shift += d
        X[:, 0] += d[0] * zbar / K[0, 0]; X[:, 1] += d[1] * zbar / K[1, 1]
    return iou(mask, silhouette(X, F, K, H, W)), (float(shift[0]), float(shift[1]))
