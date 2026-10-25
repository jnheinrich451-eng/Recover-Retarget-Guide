"""MoGe-2 adapter for Module 3 (brief §3.2, §2.6; probe: docs/report_probe_moge.md; reviewer decision 2026-10-06b).

Measured conventions, ASSERTED on every call (never assumed):
  depth is camera z (equals points[..., 2]); intrinsics are normalised by (W, H) with pixel centres at +0.5;
  mask arrives float32 and is cast; depth is not used outside the mask.
Interface rule (no tunable): the model only represents square pixels, and its free focal guess was 0.64-1.08 x ours, so
  (1) the 1024^2 anchor is resampled ONCE to a square-pixel grid H x W' with W' = round(W_wan * fy / fx) and intrinsics K';
  (2) the known horizontal FOV of K' is always passed, so the model's camera equals K' by construction (asserted).
Lifted points are projected onto the token lattice with K^p afterwards; the lift never needs Wan's own grid.
"""
from __future__ import annotations

import numpy as np


def square_pixel_grid(K_1024: np.ndarray, anchor_px: int, wan_w: int, wan_h: int):
    """Returns (W', H, K') for the square-pixel resample of the anchor. K' = diag(W'/anchor, H/anchor, 1) K_1024, so the
    resize and the intrinsics move together (CLAUDE.md §3.6). Residual |fx'/fy' - 1| comes only from rounding W'."""
    K = np.asarray(K_1024, np.float64)
    fx_w, fy_w = K[0, 0] * wan_w / anchor_px, K[1, 1] * wan_h / anchor_px
    Wp = int(round(wan_w * fy_w / fx_w))
    Kq = np.diag([Wp / anchor_px, wan_h / anchor_px, 1.0]) @ K
    return Wp, int(wan_h), Kq


def fov_x_deg(Kq: np.ndarray, W: int) -> float:
    return float(np.degrees(2.0 * np.arctan(W / (2.0 * Kq[0, 0]))))


def check_output(out: dict, Kq: np.ndarray, focal_rel_tol: float, z_rel_tol: float = 1e-5):
    """Validate one model output against the measured conventions and the imposed camera. Returns (depth f64, mask bool)."""
    for k in ("points", "depth", "mask", "intrinsics"):
        assert k in out, f"MoGe output lacks {k!r} (keys: {sorted(out)})"
    dep = np.asarray(out["depth"], np.float64); pts = np.asarray(out["points"], np.float64)
    mask = np.asarray(out["mask"]) > 0.5
    H, W = dep.shape
    assert pts.shape == (H, W, 3) and mask.shape == (H, W) and mask.any(), (pts.shape, mask.shape)
    assert np.isfinite(dep[mask]).all() and (dep[mask] > 0).all(), "non-finite or non-positive depth inside the validity mask"
    rel = np.abs(dep[mask] - pts[..., 2][mask]) / dep[mask]
    assert rel.max() < z_rel_tol, f"depth is not points-z (max rel {rel.max():.2e}): convention changed"
    Kn = np.asarray(out["intrinsics"], np.float64)
    fx, fy = Kn[0, 0] * W, Kn[1, 1] * H
    assert abs(fx / Kq[0, 0] - 1) <= focal_rel_tol and abs(fy / Kq[1, 1] - 1) <= focal_rel_tol, \
        f"model camera ({fx:.1f}, {fy:.1f}) != imposed K' ({Kq[0, 0]:.1f}, {Kq[1, 1]:.1f}): FOV was not honoured"
    return dep, mask


def estimate_depth(model, anchor_rgb_u8: np.ndarray, K_1024: np.ndarray, wan_w: int, wan_h: int, focal_rel_tol: float, device: str):
    """anchor [1024,1024,3] uint8 RGB -> dict(depth [H,W'] f64 (NaN outside mask), mask, K_sq, W_sq, H, fov_x_deg).
    Depth is in the MODEL's units on THIS grid; the source-gauge scalar k_bg is fitted on this same grid and never reused."""
    import cv2
    import torch
    A = anchor_rgb_u8.shape[0]
    assert anchor_rgb_u8.shape == (A, A, 3) and anchor_rgb_u8.dtype == np.uint8
    Wq, H, Kq = square_pixel_grid(K_1024, A, wan_w, wan_h)
    img = cv2.resize(anchor_rgb_u8, (Wq, H), interpolation=cv2.INTER_AREA)
    fov = fov_x_deg(Kq, Wq)
    x = torch.from_numpy(img).to(device).float().permute(2, 0, 1) / 255.0
    with torch.no_grad():
        out = model.infer(x, fov_x=fov)
    out = {k: (v.detach().float().cpu().numpy() if hasattr(v, "detach") else v) for k, v in out.items()}
    dep, mask = check_output(out, Kq, focal_rel_tol)
    return {"depth": np.where(mask, dep, np.nan), "mask": mask, "K_sq": Kq, "W_sq": Wq, "H": H, "fov_x_deg": fov, "image_sq": img}


def lift_background(depth_src: np.ndarray, valid: np.ndarray, Kq: np.ndarray, stride: int) -> np.ndarray:
    """X = D(u, v) K'^-1 [u + .5, v + .5, 1] on a stride grid over valid pixels (pixel centres at +0.5, as measured).
    depth_src must already be in the source gauge (k_bg applied). Returns [N,3] in world = camera-t_a."""
    H, W = depth_src.shape
    vv, uu = np.mgrid[stride // 2:H:stride, stride // 2:W:stride]
    ok = valid[vv, uu] & np.isfinite(depth_src[vv, uu]) & (depth_src[vv, uu] > 0)
    u, v, z = uu[ok] + 0.5, vv[ok] + 0.5, depth_src[vv, uu][ok]
    X = (np.linalg.inv(Kq) @ np.stack([u, v, np.ones_like(u)], 0)).T * z[:, None]
    assert np.isfinite(X).all()
    return X
