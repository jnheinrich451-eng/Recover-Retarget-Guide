"""Render target on known answers (reviewer 2026-10-10). CPU.  Run:  python tests/test_m3_render.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generate.m3 import render_target as RT  # noqa: E402

rng = np.random.default_rng(0)
H, W = 96, 160
K = np.array([[120.0, 0, 80.0], [0, 120.0, 48.0], [0, 0, 1]])
# a fronto-parallel plane at depth 6 with a smooth colour field; lifted densely from its own grid
yy, xx = np.mgrid[0:H, 0:W]
depth = np.full((H, W), 6.0)
col = np.stack([(xx * 255 // W), (yy * 255 // H), np.full((H, W), 128)], -1).astype(np.uint8)
X, C = RT.lift_dense(depth, np.ones((H, W), bool), K, col)
# 1. identity render reproduces the colours exactly, full coverage
img0, cov0, d0 = RT.render_frame(X, C, np.eye(4), K, H, W, splat=0)
assert cov0.all() and np.array_equal(img0, col), (cov0.mean(),)
# 2. a sideways camera translation renders the image shifted by f * tx / z pixels; the uncovered strip is where no content exists
E = np.eye(4); E[0, 3] = -0.5                                                        # camera moves +x in world -> content shifts left by 120*0.5/6 = 10 px
img1, cov1, _ = RT.render_frame(X, C, E, K, H, W, splat=0)
assert cov1[:, : W - 10].all() and not cov1[:, W - 10:].any()
assert np.array_equal(img1[:, : W - 10], col[:, 10:])
# 3. the z-buffer: a nearer plane occludes the far one where they overlap
Xn, Cn = RT.lift_dense(np.full((H, W), 3.0), (xx < W // 2), K, np.full((H, W, 3), 7, np.uint8))
img2, cov2, d2 = RT.render_frame(np.concatenate([X, Xn]), np.concatenate([C, Cn]), np.eye(4), K, H, W, splat=0)
assert (img2[:, : W // 2] == 7).all() and np.array_equal(img2[:, W // 2:], col[:, W // 2:]) and np.allclose(d2[:, : W // 2], 3.0)
# 4. the object footprint: primitives at depth 4 in the centre occlude the background plane at 6 and dilate by the given radius
V = np.stack([rng.uniform(-0.3, 0.3, 500), rng.uniform(-0.2, 0.2, 500), np.full(500, 4.0)], 1)
fp = RT.object_footprint(V, np.zeros(3), np.eye(4), K, H, W, d0, dilate_px=4)
assert fp[48, 80] and not fp[5, 5] and 0.02 < fp.mean() < 0.3
# 5. the latent mask keeps a block only when all frames cover all its pixels
mp = np.ones((4, 16, 24), bool); mp[2, 3, 10] = False
lm = RT.latent_mask(mp, 8)
assert lm.shape == (2, 3) and lm[0, 0] and not lm[0, 1] and lm[1, 2]
# 6. flow at queries: the plane at depth 6 under the 0.5 sideways move gives exactly -10 px at every query cell; with the nearer plane (depth 3)
#    on the left half the median there is the nearer plane's -20 px (the z-buffer's content), and the depth per query reports 3 and 6
q = np.array([[20.0, 30.0], [120.0, 30.0], [150.0, 90.0]])
F, zq = RT.flow_at_queries(X, np.stack([np.eye(4), E]), K, q, H, W, cell_px=16)
assert np.allclose(F[1], [[-10, 0]] * 3) and np.allclose(F[0], 0) and np.allclose(zq, 6)
F2, zq2 = RT.flow_at_queries(np.concatenate([X, Xn, Xn]), np.stack([np.eye(4), E]), K, q, H, W, cell_px=16)
assert np.isclose(F2[1, 0, 0], -20) and np.isclose(F2[1, 1, 0], -10) and np.isclose(zq2[0], 3) and np.isclose(zq2[1], 6)
F3, _ = RT.flow_at_queries(X[:5], np.stack([np.eye(4), E]), K, q, H, W, cell_px=16)
assert np.isnan(F3).all()                                                           # too few points per cell -> NaN, never a number
# 7. the object channel's compositing (render script, --object): an object cloud at depth 4 drawn into the background's z-buffer wins exactly the
#    pixels where it is nearer; shifted by +x it wins the shifted pixels; background and object masks never overlap
Xo, Co = RT.lift_dense(np.full((H, W), 4.0), (xx >= 60) & (xx < 100) & (yy >= 30) & (yy < 60), K, np.full((H, W, 3), 200, np.uint8))
img_b, cov_b, dep_b = RT.render_frame(X, C, np.eye(4), K, H, W, splat=0)
img_o, cov_o, dep_o = RT.render_frame(Xo, Co, np.eye(4), K, H, W, splat=0)
won = cov_o & (dep_o <= dep_b + 1e-6)
assert won.sum() == 40 * 30 and won[30:60, 60:100].all() and not (won & ~((xx >= 60) & (xx < 100) & (yy >= 30) & (yy < 60))).any()
shift = np.array([0.5, 0.0, 0.0])                                                      # +0.5 at depth 4 under f = 120 -> 15 px to the right
img_o2, cov_o2, dep_o2 = RT.render_frame(Xo + shift, Co, np.eye(4), K, H, W, splat=0)
won2 = cov_o2 & (dep_o2 <= dep_b + 1e-6)
assert won2[30:60, 75:115].all() and won2.sum() == 40 * 30
bg_mask = cov_b & ~won2
assert not (bg_mask & won2).any() and bg_mask.sum() == H * W - 40 * 30
# 8. nearest fill: a left half of colour 10 and a right half of colour 200 with an uncovered band in the middle -> the band takes 10 on its
#    left part and 200 on its right part (nearest covered column), covered pixels untouched; a one-pixel hole takes a neighbour's colour
imf = np.zeros((H, W, 3), np.uint8); imf[:, : W // 2] = 10; imf[:, W // 2:] = 200
covf = np.ones((H, W), bool); covf[:, 70:90] = False; imf[:, 70:90] = 0
filled = RT.nearest_fill(imf, covf)
assert (filled[:, 70:80] == 10).all() and (filled[:, 80:90] == 200).all() and (filled[covf] == imf[covf]).all()
covh = np.ones((H, W), bool); covh[10, 10] = False; imh = imf.copy(); imh[10, 10] = 0
assert (RT.nearest_fill(imh, covh)[10, 10] == 10).all()
# 9. push-pull fill: a hole in a constant image is filled with that constant exactly; a hole between 10 and 200 takes values between them that
#    rise from the 10 side to the 200 side; covered pixels untouched
imc = np.full((H, W, 3), 77, np.uint8); covc = np.ones((H, W), bool); covc[20:40, 50:90] = False
assert (RT.pushpull_fill(imc, covc) == 77).all()
pp = RT.pushpull_fill(imf, covf).astype(int)
assert (pp[covf] == imf[covf]).all() and (pp[:, 70:90] >= 10).all() and (pp[:, 70:90] <= 200).all() and pp[48, 72, 0] < pp[48, 88, 0]
# 10. sky at infinity: a pure yaw of the camera by theta moves a sky pixel at the principal point by f*tan(theta); translation moves nothing
anc = np.zeros((H, W, 3), np.uint8); anc[:, 80] = 255; sky0 = np.ones((H, W), bool)
Ey = np.eye(4); th = np.radians(5.0); Ey[:3, :3] = [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
wimg, wm = RT.sky_at_infinity(anc, sky0, Ey, K)
col = np.argmax(wimg[48, :, 0]); assert abs(col - (80 + 120 * np.tan(th))) <= 1.5, col
Et = np.eye(4); Et[0, 3] = 3.0
assert np.array_equal(RT.sky_at_infinity(anc, sky0, Et, K)[0], anc)
print("test_m3_render: passed (identity exact; 10 px shift exact; z-buffer; footprint; latent mask; flow at queries exact, nearer plane wins, NaN when sparse; "
      "object compositing exact, 15 px shift exact, masks disjoint; nearest fill exact; push-pull fill; sky at infinity exact)")
