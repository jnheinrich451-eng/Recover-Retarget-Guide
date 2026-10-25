"""Synthetic scene generator for the placement harness (SKILL_m2_c_C1 v1.1 §2).

Produces production-shaped inputs: a stage0 dict (m2_a schema, camera-t_a
frame), plus mask and depth at process resolution rendered from the posed
proxy, plus the analytic ground truth. Nothing here is used in production.
"""
from __future__ import annotations

import numpy as np

from . import proxy as _proxy
from .placement import desired_basis, desired_basis_in_plane


def rot_x(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def random_rotation(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def make_scene(proxy="sphere", R_A=None, plane_n=(0.0, -1.0, 0.0), cam_height=1.5,
               u_scene=None, K=None, H=256, W=256, z_true=8.0, s_true=1.0,
               scale_lambda=1.0, projection="perspective", grid_n=1600, mesh=None,
               render=True, rho_x=0.15):
    """Returns (stage0, mask_ta, depth_ta, mesh, truth).

    The object centre is placed ON the ground-contact solution: given
    (z_true, s_true) and the ray through the requested image point, the plane
    offset d is chosen so that the proxy's lowest point along n touches the
    plane. Then everything is consistent by construction and the solve must
    return (z_true, s_true) exactly (up to the projection model).
    `cam_height` only sets the ray: the centroid image point is derived so the
    object sits `cam_height` below the camera along n before s-scaling — i.e.
    d = cam_height, which is the camera's height above the plane. `d = 0` is
    legal (camera on the ground)."""
    rng = np.random.default_rng(0)
    mesh = mesh or _proxy.make_proxy(proxy)
    if R_A is None:
        R_A = np.eye(3)
    R_A = np.asarray(R_A, float)
    n = np.asarray(plane_n, float); n /= np.linalg.norm(n)
    u = n if u_scene is None else np.asarray(u_scene, float) / np.linalg.norm(u_scene)
    if K is None:
        K = np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1.0]])
    K = np.asarray(K, float)
    lam = float(scale_lambda)

    # alignment FIRST (T10 fix, 2026-09-11): if the caller sets a slope (u_scene != n)
    # the pose the solve will reconstruct is R_A = B_des(d_src, u_scene) B_tar^T, so the
    # scene must be built with that R_A, not the requested one. d_src is the requested
    # forward axis; u_scene overrides the up target.
    B_req = R_A @ mesh.B_tar
    d_src = B_req[:, 0]
    u_for_stage0 = B_req[:, 1] if u_scene is None else u
    # §6.2 amended 2026-09-15: with a contact plane the solve reconstructs the IN-PLANE basis
    # (f = d_src projected into the plane, u = n), whatever u_scene says — so the scene is built
    # with that R_A; the requested pose's pitch (e.g. rot_x in R_NI) is what the projection removes
    B_des2, _, _ = desired_basis_in_plane(d_src, n)
    R_A = B_des2 @ mesh.B_tar.T

    # pose in camera-t_a frame at unit gauge, then scale the whole scene by lambda
    Va = _proxy.aligned_vertices(mesh, R_A)                  # unit scale, centred
    hb = _proxy.h_b(mesh, R_A, n)
    d = float(cam_height)                                    # camera height above plane
    # centre X: on the plane-contact surface n·X + d = s h_b, along a ray with depth z_true
    # choose lateral position so that n·X = s*hb - d  ->  X = z_true * rho, solve for rho_y
    # given rho_x from a mild offset; if n is (0,-1,0): -X_y + d = s hb -> X_y = d - s hb
    s, z = s_true, z_true
    target = s * hb - d                                      # required n·X
    # pick rho with rho_x fixed (slight off-axis) and rho_y from the plane condition when
    # n has a y component; otherwise fall back to solving along the largest component
    rho = np.array([float(rho_x), 0.0, 1.0])
    k = int(np.argmax(np.abs(n[:2])))
    others = float(n @ rho - n[k] * rho[k])
    rho[k] = (target / z - others) / n[k]
    X = z * rho
    V_world = X + s * Va
    contact = float(n @ (X + s * Va[np.argmin(Va @ n)]) + d)
    assert abs(contact) < 1e-9, contact

    # scale the scene by lambda (null direction of the pixel observables)
    X_l, s_l, d_l = lam * X, lam * s, lam * d
    V_l = lam * V_world
    if render:
        zbuf, mask = _proxy.render_perspective(V_l, mesh.F, K, H, W)
        assert mask.any(), "object rendered outside the image"
        depth = np.where(mask, zbuf, 0.0).astype(np.float16)     # as Stage A stores it
    else:
        zbuf = mask = depth = None
    stage0 = {"t_a": np.int32(0), "K_a": K.astype(np.float32),
              "d_src": d_src.astype(np.float32), "u_scene": u_for_stage0.astype(np.float32),
              "n_plane": n.astype(np.float32), "d_plane": np.float32(d_l),
              "admitted": np.bool_(True)}
    # f64 twin of stage0 for closed-form tests (the contract stores f32; that rounding is
    # a property of the m2_a schema, measured separately, not of the solve)
    stage0_f64 = {"t_a": 0, "K_a": K, "d_src": d_src, "u_scene": u_for_stage0,
                  "n_plane": n, "d_plane": d_l, "admitted": True}
    # analytic weak-perspective source statistics (T1a) at the scaled pose
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    c = _proxy.orthographic_constants(mesh, R_A, grid_n)
    p_low = Va[np.argmin(Va @ n)]
    X_low = X_l + s_l * p_low
    truth = {"z": float(z * lam), "s": float(s_l), "X": X_l, "R_A": R_A, "d": d_l,
             "h_b": hb, "a_0": c["a_0"], "delta_p": c["delta_p"], "lambda": lam,
             "u0_wp": float(fx * X_l[0] / X_l[2] + cx), "v0_wp": float(fy * X_l[1] / X_l[2] + cy),
             "A_wp": float(c["a_0"] * s_l ** 2 * fx * fy / X_l[2] ** 2),
             "v_contact_wp": float(fy * X_low[1] / X_low[2] + cy),
             "z_src_wp": float(X_l[2] - s_l * c["delta_p"]),
             "n_dot_rho": float(n @ rho), "projection": projection, "stage0_f64": stage0_f64}
    return stage0, mask, depth, mesh, truth
