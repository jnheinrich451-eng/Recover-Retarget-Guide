"""Epipolar geometry on the token lattice for L_perp (brief v2.4 §3.6).

For a query patch q in frame t and its key anchor a:  x_a = R x_t + tau  with [R | tau] = T_ta (camera t -> camera a).
Line in a:  l(q) = (K^p)^-T [tau]_x R (K^p)^-1 q~ .  If ||tau|| < tau_rot_only the line is undefined: homography
H = K^p R (K^p)^-1 gives a point target instead. Background: T = E_a E_t^-1. Object: T = E_a G_a G_t^-1 E_t^-1.
"""
from __future__ import annotations

import numpy as np


def key_anchor(t: int, anchors: list[int]) -> int:
    """a(t) = max{a in A : a < t}."""
    c = [a for a in anchors if a < t]
    assert c, f"no anchor before frame {t}"
    return max(c)


def relative_pose(E_a, E_t, G_a=None, G_t=None) -> np.ndarray:
    """Camera t -> camera a, for a static point (G None) or a point riding the object pose G."""
    T = np.asarray(E_a, np.float64) @ np.linalg.inv(np.asarray(E_t, np.float64))
    if G_a is not None:
        T = np.asarray(E_a, np.float64) @ np.asarray(G_a, np.float64) @ np.linalg.inv(np.asarray(G_t, np.float64)) @ np.linalg.inv(np.asarray(E_t, np.float64))
    return T


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], np.float64)


def fundamental(T_ta: np.ndarray, Kp: np.ndarray) -> np.ndarray:
    Kinv = np.linalg.inv(np.asarray(Kp, np.float64))
    return Kinv.T @ _skew(T_ta[:3, 3]) @ T_ta[:3, :3] @ Kinv


def homography(T_ta: np.ndarray, Kp: np.ndarray) -> np.ndarray:
    Kp = np.asarray(Kp, np.float64)
    return Kp @ T_ta[:3, :3] @ np.linalg.inv(Kp)


def patch_centres(Hp: int, Wp: int) -> np.ndarray:
    """[Hp,Wp,3] homogeneous patch centres (j + .5, i + .5, 1)."""
    jj, ii = np.meshgrid(np.arange(Wp) + 0.5, np.arange(Hp) + 0.5)
    return np.stack([jj, ii, np.ones_like(jj)], -1)


def lines_and_points(T_ta: np.ndarray, Kp: np.ndarray, Hp: int, Wp: int, tau_rot_only: float):
    """Returns mode ('line' | 'homography'), lines [Hp,Wp,3] (zeros in homography mode), H_pts [Hp,Wp,2] (NaN in line mode)."""
    q = patch_centres(Hp, Wp)
    if np.linalg.norm(T_ta[:3, 3]) < tau_rot_only:
        h = q @ homography(T_ta, Kp).T
        return "homography", np.zeros((Hp, Wp, 3)), h[..., :2] / h[..., 2:3]
    return "line", q @ fundamental(T_ta, Kp).T, np.full((Hp, Wp, 2), np.nan)


def perp_distance(lines: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Signed perpendicular distance (patch units) of pts [...,2] from lines [...,3] = (A, B, C)."""
    A, B, C = lines[..., 0], lines[..., 1], lines[..., 2]
    return (A * pts[..., 0] + B * pts[..., 1] + C) / np.sqrt(A * A + B * B)
