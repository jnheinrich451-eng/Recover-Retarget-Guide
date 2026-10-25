"""Module 3 geometry core — brief v2.4 §13 unit tests that need no external model and no open decision (CPU, float64).
Covers: frames_for_pair, decompose_projection, project round trip, rasteriser on a uniform translation, z-buffer occlusion,
coverage bounds / monotonicity, occupancy, anchor schedule, epipolar lines (background and object), homography switch,
G_t composition. Synthetic scenes with analytic ground truth stand in for Kubric (none rendered; owner 2026-10-04).

Run:  python tests/test_m3_geometry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generate.m3.geometry import camera, epipolar, project, rasterize  # noqa: E402

rng = np.random.default_rng(0)
Hp, Wp, PX = 30, 52, 16
K_1024 = np.array([[560.0, 0, 512.0], [0, 1020.0, 512.0], [0, 0, 1]])           # squashed anchor pixels (fx/fy ~ 0.55)
K_wan = camera.wan_intrinsics(K_1024, 832, 480, 1024)
Kp = camera.lattice_intrinsics(K_wan, PX)
assert abs(K_wan[0, 0] / K_wan[1, 1] - (560 * 832) / (1020 * 480)) < 1e-12 and np.allclose(Kp[:2], K_wan[:2] / PX)
n = 0


def rot(axis, th):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a); S = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * S + (1 - np.cos(th)) * S @ S


def pose(R, t):
    E = np.eye(4); E[:3, :3] = R; E[:3, 3] = t; return E


# 1. per-pair lengths: never a constant 21
assert camera.frames_for_pair(81) == (81, 21) and camera.frames_for_pair(77) == (77, 20)
assert camera.frames_for_pair(34) == (33, 9) and camera.frames_for_pair(80) == (77, 20) and camera.frames_for_pair(1) == (1, 1)
n += 1

# 2. decompose_projection: K passed; E rebased to E[0] = I; a non-rigid block is refused
E_true = np.stack([pose(rot([0.2, 1, 0.1], 0.05 * t + 0.3), np.array([0.1 * t, -0.02 * t, 0.3 + 0.05 * t])) for t in range(6)])
K256 = np.array([[140.0, 0, 128], [0, 255.0, 128], [0, 0, 1]])
P = np.tile(np.eye(4), (6, 1, 1)); P[:, :3, :3] = K256 @ E_true[:, :3, :3]; P[:, :3, 3] = (K256 @ E_true[:, :3, 3, None])[..., 0]
E = camera.decompose_projection(P, K256, ortho_tol=1e-9)
assert np.allclose(E, E_true @ np.linalg.inv(E_true[0]), atol=1e-10) and np.allclose(E[0], np.eye(4))
Pbad = P.copy(); Pbad[2, :3, :3] *= 1.05
try:
    camera.decompose_projection(Pbad, K256, ortho_tol=1e-6); raise RuntimeError("non-rigid block accepted")
except AssertionError:
    pass
n += 1

# 3. project round trip: lift a patch coordinate with depth, project with E = I, recover it; behind-camera -> NaN, not wrapped
pb = np.stack([rng.uniform(0, Wp, 200), rng.uniform(0, Hp, 200)], 1); z = rng.uniform(2, 9, 200)
Xw = (np.linalg.inv(Kp) @ np.concatenate([pb, np.ones((200, 1))], 1).T).T * z[:, None]
p2, d2 = project.project(Xw, np.eye(4)[None], Kp)
assert np.allclose(p2[0], pb, atol=1e-10) and np.allclose(d2[0], z)
pn, dn = project.project(np.array([[0.0, 0.0, -1.0]]), np.eye(4)[None], Kp)
assert np.isnan(pn).all() and dn[0, 0] < 0
n += 1

# 4. rasteriser, pure camera x-translation of a fronto-parallel plane: constant backward flow, C -> 1 inside, 0 at the revealed edge
Z, shift = 5.0, 2.0                                               # shift in patches
us, vs = np.meshgrid(np.arange(2, 832, 4), np.arange(2, 480, 4))  # stride-4 pixels: 16 primitives per patch (brief §3.2)
pix = np.stack([us.ravel(), vs.ravel(), np.ones(us.size)], 1).astype(float)
Xpl = (np.linalg.inv(K_wan) @ pix.T).T * Z
b = shift * Z / Kp[0, 0]                                          # camera moves +b in x -> image moves -shift patches
E2 = np.stack([np.eye(4), pose(np.eye(3), [-b, 0, 0])])
pbar, dep = project.project(Xpl, E2, Kp)
vis = project.visibility(pbar, dep, Hp, Wp, zbuf_rel_tol=0.03)
assert vis[0].all()
Fg, Cg = rasterize.pair_fields(pbar[0], pbar[1], vis[0], vis[1], Hp, Wp, eps=1e-6)
good = Cg > 0.8
assert good[:, : Wp - 3].all() and np.allclose(Fg[good], [shift, 0.0], atol=1e-6), (Cg.min(), np.abs(Fg[good] - [shift, 0]).max())
assert Cg[:, : Wp - 3].min() > 0.99 and (Cg[:, Wp - 1] == 0).all(), (Cg[:, : Wp - 3].min(), Cg[:, Wp - 1].max())
Fs, Cs = rasterize.pair_fields(pbar[0], pbar[0], vis[0], vis[0], Hp, Wp, eps=1e-6)     # pair (a, a): zero flow
assert np.abs(Fs).max() < 1e-12
n += 1

# 5. z-buffer: a background point behind an object point in the same cell is invisible; within tolerance both survive
ray = np.linalg.inv(Kp) @ np.array([20.3, 12.4, 1.0])
pts = np.stack([ray * 3.0, ray * 6.0, ray * 3.05])
pb5, d5 = project.project(pts, np.eye(4)[None], Kp)
v5 = project.visibility(pb5, d5, Hp, Wp, zbuf_rel_tol=0.03)
assert v5[0].tolist() == [True, False, True]
n += 1

# 6. coverage bounded and monotone in support; occupancy counts object primitives only
half = np.arange(len(Xpl)) % 2 == 0
_, C_half = rasterize.pair_fields(pbar[0], pbar[1], vis[0] & half, vis[1], Hp, Wp, eps=1e-6)
assert (C_half <= Cg + 1e-12).all() and C_half.min() >= 0 and Cg.max() <= 1
is_obj = (np.abs(pbar[0][:, 0] - 26) < 4) & (np.abs(pbar[0][:, 1] - 15) < 3)
O = rasterize.occupancy(pbar[0], vis[0], is_obj, Hp, Wp)
assert O[15, 26] > 0.99 and O[2, 2] == 0 and 0 <= O.min() and O.max() <= 1
assert rasterize.occupancy(pbar[0], vis[0], np.zeros(len(Xpl), bool), Hp, Wp).max() == 0
n += 1

# 7. anchor schedule: decay -> {0, a1, 2 a1} capped; never below tau -> {0}; coverage curve ignores object patches
cov = np.linspace(1.0, 0.0, 21)
assert rasterize.anchor_schedule(cov, 0.5, 3) == [0, 11]
assert rasterize.anchor_schedule(np.linspace(1.0, 0.0, 41), 0.5, 3) == [0, 21] and rasterize.anchor_schedule(np.linspace(1, 0, 41), 0.8, 3) == [0, 9, 18]
assert rasterize.anchor_schedule(np.full(21, 0.9), 0.5, 3) == [0]
C0 = np.ones((2, Hp, Wp)); C0[1, :, 26:] = 0.0
Oc = np.zeros((2, Hp, Wp)); Oc[1, :, 26:] = 1.0                    # the uncovered half is entirely object -> not counted
assert np.allclose(rasterize.coverage_curve(C0, Oc, 1e-6), [1.0, 1.0], atol=1e-5)
n += 1

# 8. epipolar: GT correspondences lie on the predicted line (background), and with the object pose (object)
Ea, Et = pose(rot([0, 1, 0], 0.05), [0.1, 0.0, 0.05]), pose(rot([0.1, 1, 0], -0.2), [-0.6, 0.05, 0.3])
Xs = np.stack([rng.uniform(-2, 2, 300), rng.uniform(-1, 1, 300), rng.uniform(4, 9, 300)], 1)
pa, _ = project.project(Xs, Ea[None], Kp); pt, _ = project.project(Xs, Et[None], Kp)
T_bg = epipolar.relative_pose(Ea, Et)
lines = np.concatenate([pt[0], np.ones((300, 1))], 1) @ epipolar.fundamental(T_bg, Kp).T
assert np.abs(epipolar.perp_distance(lines, pa[0])).max() < 1e-9
c0 = np.array([0.2, 0.1, 6.0])
G = camera.object_pose(np.stack([np.eye(3), rot([0, 1, 0], 0.4)]), c0, np.stack([c0, c0 + [0.8, 0.0, -0.5]]))
Xo = c0 + rng.normal(scale=0.3, size=(100, 3))
Xo_t = project.move_object(Xo, G[:, :3, :3], c0, np.stack([c0, c0 + [0.8, 0.0, -0.5]]))
assert np.allclose(Xo_t[1], Xo @ G[1, :3, :3].T + G[1, :3, 3]) and np.allclose(Xo_t[0], Xo)
pa_o, _ = project.project(Xo_t[0], Ea[None], Kp); pt_o, _ = project.project(Xo_t[1], Et[None], Kp)
T_obj = epipolar.relative_pose(Ea, Et, G[0], G[1])
lo = np.concatenate([pt_o[0], np.ones((100, 1))], 1) @ epipolar.fundamental(T_obj, Kp).T
assert np.abs(epipolar.perp_distance(lo, pa_o[0])).max() < 1e-9
assert np.abs(epipolar.perp_distance(np.concatenate([pt_o[0], np.ones((100, 1))], 1) @ epipolar.fundamental(T_bg, Kp).T, pa_o[0])).max() > 0.1   # wrong pose is visible
assert epipolar.key_anchor(7, [0, 5, 10]) == 5 and epipolar.key_anchor(5, [0, 5, 10]) == 0
n += 1

# 9. homography switch: pure rotation -> point target equals GT; with translation -> line mode
Er = pose(rot([0, 1, 0], 0.08), [0, 0, 0])
mode, L9, Hpts = epipolar.lines_and_points(epipolar.relative_pose(np.eye(4), Er), Kp, Hp, Wp, tau_rot_only=0.05)
assert mode == "homography" and np.isfinite(Hpts).all() and not L9.any()
q = epipolar.patch_centres(Hp, Wp)[10, 20]
Xq = np.linalg.inv(Er) @ np.append(np.linalg.inv(Kp) @ q * 7.0, 1.0)            # the world point seen at q in the rotated camera
pa9, _ = project.project(Xq[None, :3], np.eye(4)[None], Kp)
assert np.allclose(pa9[0, 0], Hpts[10, 20], atol=1e-9)
mode2, L2, H2 = epipolar.lines_and_points(T_bg, Kp, Hp, Wp, tau_rot_only=0.05)
assert mode2 == "line" and np.isnan(H2).all() and L2.shape == (Hp, Wp, 3)
n += 1

# 10. G_t composition: maps c0 -> c_t for any R_t; G_0 = I
Rs = np.stack([np.eye(3), rot([0.3, 1, 0.2], 0.7), rot([1, 0, 0], -1.1)]); cs = np.stack([c0, c0 + [1, 2, 3], c0 - [0.5, 0, 2]])
G3 = camera.object_pose(Rs, c0, cs)
assert np.allclose(G3[0], np.eye(4)) and np.allclose((G3 @ np.append(c0, 1.0))[:, :3], cs)
n += 1

print(f"test_m3_geometry: {n}/10 passed")

# 11. Phase-2 camera from ViPE cam2world poses: direction, the one-frame gauge bridge, world-anchor invariance, refusals
from src.generate.m3.geometry import vipe_camera  # noqa: E402
F11, ta, sig = 60, 7, 2.5
W2C = np.stack([pose(rot([0.1, 1, 0.05], 0.01 * t), np.array([0.05 * t, 0.0, 0.02 * t])) for t in range(F11)])   # metric world->camera
C2W = np.linalg.inv(W2C)
E_full, E_l = vipe_camera.phase2_camera(C2W, ta, sig, 33, ortho_tol=1e-9)
assert E_full.shape == (33, 4, 4) and E_l.shape == (9, 4, 4) and np.allclose(E_full[0], np.eye(4))
Xm_w = np.array([0.4, -0.2, 6.0, 1.0])                                  # a static metric world point
Xs_ta = (W2C[ta] @ Xm_w)[:3] / sig                                      # the same point in world = camera-t_a, SOURCE gauge
for k in (0, 5, 32):                                                    # E' X_source = (metric camera coords) / sigma: same pixel
    assert np.allclose(E_full[k, :3, :3] @ Xs_ta + E_full[k, :3, 3], (W2C[ta + k] @ Xm_w)[:3] / sig, atol=1e-10)
anchor = pose(rot([1, 0, 0], 0.7), [3, -1, 2])                          # a different world anchor for ViPE's poses cancels
assert np.allclose(vipe_camera.phase2_camera(anchor @ C2W, ta, sig, 33, 1e-9)[0], E_full, atol=1e-9)
bad_dir = vipe_camera.phase2_camera(W2C, ta, sig, 33, 1e-9)[0]          # feeding world2cam instead is detectably different
assert not np.allclose(bad_dir[5], E_full[5], atol=1e-4)
for kw in ({"sigma_ta": None}, {"sigma_ta": 0.0}, {"F_pair": 34}, {"t_a": 40}):
    a = {"t_a": ta, "sigma_ta": sig, "F_pair": 33, **kw}
    try:
        vipe_camera.phase2_camera(C2W, a["t_a"], a["sigma_ta"], a["F_pair"], 1e-9); raise RuntimeError(f"accepted {kw}")
    except AssertionError:
        pass
print("test 11 (Phase-2 camera from ViPE): passed")

# 12. MoGe adapter, model-free parts: square-pixel grid, output validator on planted conventions, background lift round trip
from src.generate.m3 import depth_moge  # noqa: E402
K370 = np.array([[705.0, 0, 512.0], [0, 1363.0, 512.0], [0, 0, 1]])      # 370-like: fx/fy at 832x480 ~ 0.897
Wq, Hq, Kq = depth_moge.square_pixel_grid(K370, 1024, 832, 480)
assert Hq == 480 and Wq == round(832 * (1363 * 480 / 1024) / (705 * 832 / 1024)) and abs(Kq[0, 0] / Kq[1, 1] - 1) < 1e-3
assert abs(Kq[1, 1] - 1363 * 480 / 1024) < 1e-9 and abs(Kq[0, 2] - 512 * Wq / 1024) < 1e-9
vv, uu = np.mgrid[0:Hq, 0:Wq].astype(float); zz = 3.0 + 0.004 * vv
ptsq = np.stack([(uu + 0.5 - Kq[0, 2]) / Kq[0, 0] * zz, (vv + 0.5 - Kq[1, 2]) / Kq[1, 1] * zz, zz], -1)
mk = np.ones((Hq, Wq), np.float32); mk[:100] = 0.0                       # "sky"
depq = zz.copy(); depq[:100] = np.inf
good = {"points": ptsq, "depth": depq, "mask": mk, "intrinsics": np.array([[Kq[0, 0] / Wq, 0, 0.5], [0, Kq[1, 1] / Hq, 0.5], [0, 0, 1]])}
d12, m12 = depth_moge.check_output(good, Kq, focal_rel_tol=0.01)
assert m12.dtype == bool and m12.sum() == (Hq - 100) * Wq
for bad in ({"depth": np.linalg.norm(ptsq, axis=-1)},                                             # ray length instead of z
            {"intrinsics": good["intrinsics"] * np.array([[0.8], [0.8], [1.0]])},                 # free focal (FOV not honoured)
            {"mask": np.ones((Hq, Wq), np.float32)}):                                            # invalid depth inside the mask
    try:
        depth_moge.check_output({**good, **bad}, Kq, focal_rel_tol=0.01); raise RuntimeError(f"accepted {list(bad)}")
    except AssertionError:
        pass
Xb = depth_moge.lift_background(np.where(m12, d12, np.nan), m12, Kq, stride=4)
pb12, db12 = project.project(Xb, np.eye(4)[None], Kq)                     # re-project with K' itself: lands on the sampled pixel centres
assert np.allclose(pb12[0] % 4, 2.5, atol=1e-9) and np.allclose(db12[0], 3.0 + 0.004 * (pb12[0][:, 1] - 0.5), atol=1e-9) and pb12[0][:, 1].min() > 100
print("test 12 (MoGe adapter, model-free parts): passed")


# 13. object path: D4RT centroid (camera-0, source gauge) -> world = camera-t_a via the bridged ViPE camera. Known scene:
#     metric world, D4RT gauge = metric / sigma with the SAME camera rotations; the recovered displacement must equal the true
#     one in the source gauge, expressed in camera-t_a, and must not contain the camera's own motion.
sigma13, ta13, T13 = 2.5, 8, 6
fr = ta13 + 4 * np.arange(T13)
obj_w = np.stack([np.array([0.3 + 0.05 * t, 0.0, 7.0 - 0.03 * t]) for t in range(F11)])               # metric world path of the object
cam0 = (W2C[0, :3, :3] @ obj_w.T).T + W2C[0, :3, 3]                                                  # in metric camera-0
Xbar13 = cam0 / sigma13                                                                              # D4RT: same frame, source gauge
Gd = np.linalg.inv(W2C) ; Gd = np.linalg.inv(Gd[0]) @ Gd                                             # metric camera-t -> camera-0
Gd[:, :3, 3] /= sigma13                                                                              # D4RT gauge
_, El13 = vipe_camera.phase2_camera(C2W, ta13, sigma13, 4 * (T13 - 1) + 1, 1e-9)
Xw13 = vipe_camera.object_path_world(Xbar13[fr], Gd[fr], El13)
true_ta = ((W2C[ta13, :3, :3] @ obj_w[fr].T).T + W2C[ta13, :3, 3]) / sigma13                         # truth: camera-t_a coords, source gauge
assert np.allclose(Xw13, true_ta, atol=1e-9)
static = np.tile(obj_w[ta13], (F11, 1)); cs = ((W2C[0, :3, :3] @ static.T).T + W2C[0, :3, 3]) / sigma13
assert np.allclose(np.ptp(vipe_camera.object_path_world(cs[fr], Gd[fr], El13), axis=0), 0, atol=1e-9)   # a static point stays put while the camera moves
print("test 13 (object path in the world of the Phase-2 camera): passed")
# test 13b (v2.6 §2.3-§2.4): the target path = R^T (s_delta X_cam - tau / beta). With s_delta = 1 and beta = 1 it is test 13's path; with beta and
# s_delta, the camera's own contribution must NOT be scaled by s_delta (scaling the world displacement would double-scale it)
beta_, sd_ = 1.3, 0.9
El_b = El13.copy(); El_b[:, :3, 3] /= beta_
Xw_t = vipe_camera.object_path_target(Xbar13[fr], Gd[fr], El_b, sd_)
Ginv_ = np.linalg.inv(Gd[fr]); Xc_ = (Ginv_[:, :3, :3] @ Xbar13[fr][..., None])[..., 0] + Ginv_[:, :3, 3]
Pt_ = np.linalg.inv(El_b); hand = (Pt_[:, :3, :3] @ (sd_ * Xc_)[..., None])[..., 0] + Pt_[:, :3, 3]
assert np.allclose(Xw_t, hand, atol=1e-9)
assert np.allclose(vipe_camera.object_path_target(Xbar13[fr], Gd[fr], El13, 1.0), Xw13, atol=1e-9)
wrong = (vipe_camera.object_path_world(Xbar13[fr], Gd[fr], El_b) - vipe_camera.object_path_world(Xbar13[fr], Gd[fr], El_b)[0]) * sd_
assert not np.allclose(wrong, Xw_t - Xw_t[0], atol=1e-6)                                 # the double-scaled form differs whenever the camera moves
print("test 13b (target object path: s_delta on the camera-relative part only): passed")

# 14. camera check instrument (scripts/check_m3_camera.py): on a scene where D4RT gauge = metric / sigma, the D4RT camera and the
#     ViPE camera bridged with the RIGHT scale both reproduce static points exactly; a wrong bridge scale is detected in pixels.
from scripts.check_m3_camera import reproj_px, rot_angle_deg  # noqa: E402
K256 = np.array([[140.0, 0, 128], [0, 255.0, 128], [0, 0, 1]])
pts_w = np.stack([rng.uniform(-3, 3, 200), rng.uniform(-1, 1, 200), rng.uniform(5, 12, 200)], 1)
X_ta = ((W2C[ta13, :3, :3] @ pts_w.T).T + W2C[ta13, :3, 3]) / sigma13                 # lifted at t_a, source gauge
k14 = 20
obs = reproj_px((W2C[ta13 + k14, :3, :3] @ pts_w.T).T + W2C[ta13 + k14, :3, 3], np.eye(4), K256)   # what the video shows
Gd14 = np.linalg.inv(W2C); Gd14 = np.linalg.inv(Gd14[0]) @ Gd14; Gd14[:, :3, 3] /= sigma13
E_d14 = np.linalg.inv(Gd14[ta13 + k14]) @ Gd14[ta13]
E_s14 = vipe_camera.phase2_camera(C2W, ta13, sigma13, 21, 1e-9)[0][k14]
E_bad = vipe_camera.phase2_camera(C2W, ta13, sigma13 * 1.4, 21, 1e-9)[0][k14]
err = lambda E: float(np.nanmedian(np.linalg.norm(reproj_px(X_ta, E, K256) - obs, axis=1)))   # noqa: E731
assert err(E_d14) < 1e-6 and err(E_s14) < 1e-6 and err(E_bad) > 0.5, (err(E_d14), err(E_s14), err(E_bad))
assert rot_angle_deg(E_d14[:3, :3], E_s14[:3, :3]) < 1e-5
print(f"test 14 (camera check instrument): right scale 0 px, 1.4x wrong bridge scale {err(E_bad):.2f} px")

# 15. occupancy from the warped painted mask: a translating + scaling object whose painted mask equals its primitives' footprint
#     gives the same occupancy as the primitives; with the mask OFFSET from the mesh (the 265.2.005 case) the occupancy follows the
#     painted pixels, not the mesh; the affine field fits an exact affine motion with zero residual.
prims0 = np.stack([rng.uniform(10, 20, 400), rng.uniform(8, 14, 400)], 1)             # object primitives at frame 0 (patch units)
Aff = np.array([[1.1, 0.0], [0.0, 1.1]]); bvec = np.array([3.0, -1.0])
prims1 = prims0 @ Aff.T + bvec
O_mask, resid = rasterize.warp_mask_occupancy(prims0, prims0, prims1, Hp, Wp)       # painted mask == the primitives themselves
O_prim = rasterize.occupancy(prims1, np.ones(400, bool), np.ones(400, bool), Hp, Wp)
assert resid < 1e-9 and np.allclose(O_mask, O_prim, atol=1e-9)
painted = prims0 + np.array([-2.0, 0.0])                                             # the painting sits 2 patches LEFT of the mesh
O_off, _ = rasterize.warp_mask_occupancy(painted, prims0, prims1, Hp, Wp)
cm = lambda O: np.array([(O.sum(0) * (np.arange(Wp) + 0.5)).sum() / O.sum(), (O.sum(1) * (np.arange(Hp) + 0.5)).sum() / O.sum()])   # noqa: E731
assert abs((cm(O_off) - cm(O_prim))[0] - (-2.0 * 1.1)) < 0.15 and abs((cm(O_off) - cm(O_prim))[1]) < 0.15
O_few, r_few = rasterize.warp_mask_occupancy(painted, prims0[:3], prims1[:3], Hp, Wp)   # too few primitives: no warp, mask stays, residual None
assert r_few is None and O_few.sum() > 0
print("test 15 (occupancy from the warped painted mask): passed")
