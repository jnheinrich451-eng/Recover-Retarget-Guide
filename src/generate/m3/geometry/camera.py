"""Cameras, lattice intrinsics, object pose, per-pair lengths (brief v2.4 §1.2, §1.3, §2.2, §2.5).

Conventions (asserted, not commented): E maps world -> camera, 4x4; world = camera at the first frame, E[0] = I;
K is PASSED, never estimated; patch coordinate (u, v) is continuous, in patch units, centre of index (i, j) = (j + .5, i + .5).
No tunable lives here: tolerances are arguments (lock file: m3.int.*).
"""
from __future__ import annotations

import numpy as np


def frames_for_pair(n_frames_after_t_a: int) -> tuple[int, int]:
    """Largest 4k+1 <= n (Wan's 1+4k temporal compression) and T_l = 1 + (F_pair - 1) / 4. Never a constant 21."""
    n = int(n_frames_after_t_a)
    assert n >= 1, n
    F_pair = 4 * ((n - 1) // 4) + 1
    return F_pair, 1 + (F_pair - 1) // 4


def decompose_projection(P: np.ndarray, K: np.ndarray, ortho_tol: float) -> np.ndarray:
    """P [F,4,4] with top rows [K R_t | K t_t]  ->  E [F,4,4], world->camera, rebased so E[0] = I.
    K [3,3] is mandatory (v2.1 §2.2'). Rotations are checked orthonormal with det +1 within ortho_tol; a failure is a finding."""
    P = np.asarray(P, np.float64); K = np.asarray(K, np.float64)
    assert P.ndim == 3 and P.shape[1:] == (4, 4) and K.shape == (3, 3), (P.shape, K.shape)
    assert np.isfinite(P).all() and np.isfinite(K).all()
    Kinv = np.linalg.inv(K)
    E = np.tile(np.eye(4), (len(P), 1, 1))
    E[:, :3, :3] = Kinv @ P[:, :3, :3]
    E[:, :3, 3] = (Kinv @ P[:, :3, 3, None])[..., 0]
    R = E[:, :3, :3]
    err = np.abs(R.transpose(0, 2, 1) @ R - np.eye(3)).max()
    det = np.linalg.det(R)
    assert err <= ortho_tol and np.all(np.abs(det - 1.0) <= ortho_tol), f"non-rigid camera block: |RtR-I| {err:.2e}, det in [{det.min():.4f}, {det.max():.4f}]"
    E = E @ np.linalg.inv(E[0])
    assert np.allclose(E[0], np.eye(4), atol=1e-9)
    return E


def lattice_intrinsics(K_wan: np.ndarray, px_per_patch: int) -> np.ndarray:
    """K^p = diag(1/p, 1/p, 1) K_wan: every projection onto the token lattice uses this (brief §1.3)."""
    return np.diag([1.0 / px_per_patch, 1.0 / px_per_patch, 1.0]) @ np.asarray(K_wan, np.float64)


def wan_intrinsics(K_1024: np.ndarray, wan_w: int, wan_h: int, anchor_px: int) -> np.ndarray:
    """K_wan = diag(W/anchor, H/anchor, 1) K_1024 (brief §2.5): the resize is a geometry transform and K follows it."""
    return np.diag([wan_w / anchor_px, wan_h / anchor_px, 1.0]) @ np.asarray(K_1024, np.float64)


def object_pose(R_t: np.ndarray, c0: np.ndarray, c_t: np.ndarray) -> np.ndarray:
    """G_t = [R_t | c_t - R_t c0] (brief §1.3): centroid-anchored, maps c0 -> c_t for any R_t. R_t [T,3,3], c_t [T,3]."""
    R_t = np.asarray(R_t, np.float64); c0 = np.asarray(c0, np.float64); c_t = np.asarray(c_t, np.float64)
    T = len(R_t)
    assert R_t.shape == (T, 3, 3) and c_t.shape == (T, 3) and c0.shape == (3,)
    G = np.tile(np.eye(4), (T, 1, 1))
    G[:, :3, :3] = R_t
    G[:, :3, 3] = c_t - (R_t @ c0)
    assert np.isfinite(G).all()
    return G
