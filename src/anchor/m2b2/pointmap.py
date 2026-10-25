"""Point maps for SAM 3D Objects (SKILL §2): mandatory wherever depth exists.

Measured justification (P0, 2026-09-12): the vendor's internal MoGe map returned z in
[1.38, 1.63] against a true 6.0, and `translation`/`scale` are expressed against that map,
so without ours the layout sits in an unknown gauge. Sources:
  "stage_a"          real frames: Stage A D[t_a] (f16 on disk -> f64 here) + K[t_a]
  "fixture"          exact depth from a synthetic render
  "vendor_internal"  generated proxy images (R1): no depth exists; every downstream constant
                     is then scale-free and the record says so (SKILL §2)
The adapter RAISES if the map is None on a path where one is available.
"""
from __future__ import annotations

import numpy as np

SOURCES = ("stage_a", "fixture", "vendor_internal")


class PointMapMissing(ValueError):
    """pointmap is None on a path where one is available (SKILL §2: raise, never warn)."""


def opencv_pointmap(depth, valid, K, z_bg) -> np.ndarray:
    """z-depth [H,W] + K -> HxWx3 f32 points in the OpenCV camera frame (x right, y down,
    z forward), pixel centres at (i+0.5, j+0.5). Invalid pixels get a flat background at z_bg."""
    D = np.asarray(depth, np.float64)
    M = np.asarray(valid, bool)
    K = np.asarray(K, np.float64)
    assert D.shape == M.shape and D.ndim == 2, (D.shape, M.shape)
    assert K.shape == (3, 3) and K[0, 0] > 0 and K[1, 1] > 0
    assert np.isfinite(z_bg) and z_bg > 0
    H, W = D.shape
    z = np.where(M & np.isfinite(D) & (D > 0), D, z_bg)
    jj, ii = np.mgrid[0:H, 0:W]
    x = (ii + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (jj + 0.5 - K[1, 2]) / K[1, 1] * z
    pm = np.stack([x, y, z], -1).astype(np.float32)
    assert np.isfinite(pm).all()
    return pm


def stage_a_pointmap(D, K, mask, bg_depth_mult: float):
    """The P1/P2 construction on a Stage A frame: valid depth everywhere it exists, a flat
    background at bg_depth_mult x the mask's median depth elsewhere (the vendor clips points
    beyond k x median mask depth to NaN itself). Returns (pointmap, z_med)."""
    D = np.asarray(D, np.float64)                       # f16 on disk -> f64 before arithmetic
    M = np.asarray(mask, bool)
    valid = np.isfinite(D) & (D > 0)
    assert (M & valid).any(), "no finite positive depth under the mask"
    z_med = float(np.median(D[M & valid]))
    return opencv_pointmap(np.where(valid, D, z_med * bg_depth_mult), np.ones_like(valid), K,
                           z_med * bg_depth_mult), z_med


def require_pointmap(pointmap, source: str):
    """Guard at the vendor call: a None map is refused on every path but vendor_internal."""
    assert source in SOURCES, f"unknown pointmap source {source!r}; one of {SOURCES}"
    if source == "vendor_internal":
        assert pointmap is None, "vendor_internal source must not carry a point map"
        return None
    if pointmap is None:
        raise PointMapMissing(f"pointmap is None on source {source!r} — depth is available on this path "
                              "(SKILL §2); build it with stage_a_pointmap / opencv_pointmap")
    pm = np.asarray(pointmap, np.float32)
    assert pm.ndim == 3 and pm.shape[2] == 3, f"pointmap must be HxWx3, got {pm.shape}"
    assert np.isfinite(pm).all(), "non-finite point map"
    return pm


def fit_pinhole(pointmap, mask) -> np.ndarray:
    """Least-squares pinhole (fx, cx, fy, cy) from an OpenCV point map — the Stage A K rule —
    used only to project under the vendor's own camera on vendor_internal paths (P4)."""
    pm = np.asarray(pointmap, np.float64)
    M = np.asarray(mask, bool)
    jj, ii = np.nonzero(M & np.isfinite(pm).all(-1) & (pm[..., 2] > 0))
    assert len(ii) >= 8, "too few valid points to fit a pinhole"
    P = pm[jj, ii]
    x, y = P[:, 0] / P[:, 2], P[:, 1] / P[:, 2]
    (fx, cx), _, _, _ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), ii + 0.5, rcond=None)
    (fy, cy), _, _, _ = np.linalg.lstsq(np.stack([y, np.ones_like(y)], 1), jj + 0.5, rcond=None)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
