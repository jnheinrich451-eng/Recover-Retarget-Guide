"""Phase-2 camera E'_t from the archived ViPE trajectory (brief §2.3; reviewer 2026-10-06: D-M3-07 (a), D-M3-08 "settled for build").

ViPE poses are cam2world, metric, OpenCV axes (A.5 V-probe, signed 2026-08-06). World of Module 3 = camera at source t_a in
the SOURCE gauge (D4RT at t_a). The one-frame bridge divides ViPE's translations by sigma_hat_{t_a} (metric = sigma x source):
a gauge choice at one frame, always on, not a drift correction. No per-frame g_t is applied here.

    P_{t_a<-t} = P_{t_a}^{-1} P_t            (camera t -> camera t_a, metric)
    E'_t       = [R | tau / sigma_ta]^{-1}    (world -> camera t),  E'_{t_a} = I
"""
from __future__ import annotations

import numpy as np


def phase2_camera(P_c2w: np.ndarray, t_a: int, sigma_ta: float, F_pair: int, ortho_tol: float):
    """P_c2w [F,4,4] ViPE cam2world (any world anchor: it cancels). Returns E_full [F_pair,4,4] for source frames
    t_a .. t_a+F_pair-1 and E_l [T_l,4,4] at the latent frames (every 4th). A missing or non-positive sigma is refused."""
    P = np.asarray(P_c2w, np.float64)
    assert P.ndim == 3 and P.shape[1:] == (4, 4) and np.isfinite(P).all()
    assert sigma_ta is not None and np.isfinite(sigma_ta) and sigma_ta > 0, f"sigma_hat at t_a = {sigma_ta}: refuse, never assume 1"
    assert 0 <= t_a and t_a + F_pair <= len(P) and (F_pair - 1) % 4 == 0, (t_a, F_pair, len(P))
    R = P[:, :3, :3]
    err = np.abs(R.transpose(0, 2, 1) @ R - np.eye(3)).max()
    assert err <= ortho_tol and np.all(np.abs(np.linalg.det(R) - 1) <= ortho_tol), f"ViPE rotations not rigid: {err:.2e}"
    rel = np.linalg.inv(P[t_a]) @ P[t_a: t_a + F_pair]            # camera t -> camera t_a
    rel[:, :3, 3] /= float(sigma_ta)                              # into the source gauge
    E = np.linalg.inv(rel)                                        # world (camera t_a) -> camera t
    assert np.allclose(E[0], np.eye(4), atol=1e-9) and np.isfinite(E).all()
    return E, E[::4]


def object_path_world(Xbar_cam0: np.ndarray, G_0_from_t: np.ndarray, E_l: np.ndarray) -> np.ndarray:
    """Object centroid per latent frame in world = camera-t_a, source gauge (brief §2.3 unified rule, drift factor 1).
    Xbar_cam0 [T,3]: centroid in D4RT camera-0 coordinates at the T latent source frames; G_0_from_t [T,4,4]: D4RT camera-t ->
    camera-0 at the same frames (bookkeeping only: it takes the centroid back into its own camera-t); E_l [T,4,4]: the bridged
    Phase-2 camera, world -> camera-t. The camera's motion enters once, through E_l."""
    Ginv = np.linalg.inv(np.asarray(G_0_from_t, np.float64))
    X_cam = (Ginv[:, :3, :3] @ np.asarray(Xbar_cam0, np.float64)[..., None])[..., 0] + Ginv[:, :3, 3]
    Pt = np.linalg.inv(np.asarray(E_l, np.float64))
    Xw = (Pt[:, :3, :3] @ X_cam[..., None])[..., 0] + Pt[:, :3, 3]
    assert np.isfinite(Xw).all()
    return Xw


def object_path_target(Xbar_cam0: np.ndarray, G_0_from_t: np.ndarray, E_bridged: np.ndarray, s_delta: float) -> np.ndarray:
    """Object centroid per frame in the TARGET world (v2.6 §2.3-§2.4, D-M3-08 amended): the target scene is a similarity of the true
    scene, so the object's camera-relative position (lift gauge) is scaled by s_delta = rho_c / beta and carried by the bridged camera
    (translations already divided by beta):  c^tar_t = R_t^T (s_delta X^L_cam,t - tau^tri_t / beta).
    Scaling the whole world displacement by s_delta instead would scale the camera's own contribution a second time."""
    Ginv = np.linalg.inv(np.asarray(G_0_from_t, np.float64))
    X_cam = (Ginv[:, :3, :3] @ np.asarray(Xbar_cam0, np.float64)[..., None])[..., 0] + Ginv[:, :3, 3]
    Pt = np.linalg.inv(np.asarray(E_bridged, np.float64))
    Xw = (Pt[:, :3, :3] @ (float(s_delta) * X_cam)[..., None])[..., 0] + Pt[:, :3, 3]
    assert np.isfinite(Xw).all()
    return Xw
