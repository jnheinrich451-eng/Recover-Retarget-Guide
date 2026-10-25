"""Synthetic backgrounds for the C2 harness (SKILL_m2_c_C2 §5). Harness-only.

A ground plane (n, d, K) in disparity — with the horizon INSIDE the image so
zeta <= 0 (sky) is exercised — plus optional: a non-planar bump, an object
occupying a mask with its own nearer disparity, a halo of object disparity
bleeding k px past the silhouette, holes (invalid pixels), and sinusoidal
features of known width for T5.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import sanitise as S


def default_camera(H=256, W=256, f=300.0, tilt_deg=8.0, height=1.5):
    """Camera `height` above level ground, pitched DOWN by tilt_deg so the horizon
    sits above the image centre (row ~ c_y - f tan(tilt))."""
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    t = np.radians(tilt_deg)
    # ground normal is world-up = -y_world; camera pitched down by t: in camera coords
    # up = (0, -cos t, sin t)... a downward pitch rotates world-up toward +z_cam
    n = np.array([0.0, -np.cos(t), -np.sin(t)])
    # sign check: a ground pixel below the horizon must have zeta > 0 (n·rho < 0)
    return K, n, float(height)


def ellipse_mask(H, W, cu, cv, a, b):
    u, v = S.pixel_grid(H, W)
    return ((u - cu) / a) ** 2 + ((v - cv) / b) ** 2 <= 1.0


def make_background(H=256, W=256, K=None, n=None, d=None, mask=None, bump=None,
                    object_near=0.03, halo_px=0, holes=None, features=None, seed=0):
    """Returns dict with zeta_true (background only), zeta_obs (what Stage A
    would see: object + halo + holes), valid, mask, zeta_obj (object disparity
    inside mask ∪ halo), affine (closed form), K, n, d."""
    rng = np.random.default_rng(seed)
    if K is None:
        K, n, d = default_camera(H, W)
    zeta_true = S.plane_disparity(n, d, K, H, W)
    if bump is not None:                                    # non-planar background
        cu, cv, sig, amp = bump
        u, v = S.pixel_grid(H, W)
        zeta_true = zeta_true + amp * np.exp(-((u - cu) ** 2 + (v - cv) ** 2) / (2 * sig ** 2))
    if features is not None:                                # T5: stripes of known width
        u, v = S.pixel_grid(H, W)
        for width, amp in features:
            zeta_true = zeta_true + amp * np.sin(2 * np.pi * u / (2 * width))
    if mask is None:
        mask = ellipse_mask(H, W, W * 0.55, H * 0.68, 22, 30)
    mask = np.asarray(mask, bool)
    # object: nearer than the ground by a dome profile (peak object_near in disparity)
    dist = ndimage.distance_transform_edt(mask)
    dome = dist / max(dist.max(), 1e-9)
    zeta_obj = zeta_true + object_near * np.sqrt(np.clip(dome, 0, 1))
    zeta_obs = zeta_true.copy()
    zeta_obs[mask] = zeta_obj[mask]
    if halo_px > 0:                                         # T7: object disparity bleeds past the silhouette
        halo = S.dilate(mask, halo_px) & ~mask
        zeta_obs[halo] = (zeta_true + object_near * 0.5)[halo]
        zeta_obj = np.where(halo, zeta_obs, zeta_obj)
    valid = np.ones((H, W), bool)
    if holes:                                               # T9: invalid pixels
        for (cu, cv, rad) in holes:
            valid &= ~ellipse_mask(H, W, cu, cv, rad, rad)
    # what Stage A stores: depth, f16, 0 where invalid; sky (zeta <= 0) is invalid there too
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where((zeta_obs > 0) & valid, 1.0 / np.where(zeta_obs > 0, zeta_obs, 1.0), 0.0)
    # f16 cannot hold z > 65504: near-horizon ground overflows to inf in storage. Mark those
    # pixels invalid (0) as the stored map effectively does (disparity() rejects non-finite).
    depth = np.where(depth > 6.0e4, 0.0, depth)
    valid = valid & (depth > 0)
    depth16 = depth.astype(np.float16)
    return {"zeta_true": zeta_true, "zeta_obs": zeta_obs, "zeta_obj": zeta_obj, "valid": valid,
            "mask": mask, "depth16": depth16, "affine": S.affine_from_plane(n, d, K),
            "K": K, "n": np.asarray(n, float), "d": float(d), "H": H, "W": W}
