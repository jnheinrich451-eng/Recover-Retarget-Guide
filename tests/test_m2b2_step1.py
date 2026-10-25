"""m2_b2 build step 1 — SKILL_m2_b2 v1.1 §10 tests 1, 2, 10 (+ the point-map builder check).

CPU only, no external model, no binary fixture (human decision 2026-09-14): a procedurally
generated ellipsoid with a known camera-frame pose stands in for the vendor output by
INVERTING the E1 composition, so the ground truth is exact.

Run:  python tests/test_m2b2_step1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2b2 import compose, pointmap, silhouette              # noqa: E402
from src.anchor.m2c import proxy                                        # noqa: E402
from src.anchor.m2c.synthetic import rot_x, rot_y                       # noqa: E402

TOL = 1e-6
REPORT = ["# m2_b2 step 1 — tests 1, 2, 10", ""]
H = W = 1024
K = np.array([[1024.0, 0, 512], [0, 1024.0, 512], [0, 0, 1]])
rng = np.random.default_rng(7)


def fabricate(X_true, q, t, s):
    """Vendor-output stand-in: V_local such that the E1 composition returns X_true exactly."""
    R = compose.quat_wxyz_to_matrix(q)
    return ((X_true @ np.linalg.inv(compose.P3D_TO_CV) - t) @ np.linalg.inv(R)) / s


def wrong_variants(V, q, t, s):
    """The compositions a reader might assume instead (probe P1 candidates), column form."""
    R = compose.quat_wxyz_to_matrix(q)
    Rx = compose.quat_wxyz_to_matrix(np.asarray(q)[[3, 0, 1, 2]])       # read as xyzw
    return {"no conversion": (V * s) @ R + t,
            "R for R^T (column)": (R @ (V * s).T).T @ compose.P3D_TO_CV + t @ compose.P3D_TO_CV,
            "xyzw": ((V * s) @ Rx + t) @ compose.P3D_TO_CV}


# ---------------------------------------------------------------- test 1: composition round trip
mesh = proxy.ellipsoid()
R_A = rot_y(35.0) @ rot_x(15.0)
X_true = np.array([0.9, 0.6, 6.0]) + proxy.aligned_vertices(mesh, R_A)       # OFF-centre: the conversion must matter
_, m_true = proxy.render_perspective(X_true, mesh.F, K, H, W)
# a GENERIC rotation: the "R for R^T" variant applies R twice (x = R R p), so it is invisible
# when the angle is near 0 or 180 deg (first draw of this test: 0.993 IoU at ~178 deg) — 50 deg here
axis = rng.normal(size=3); axis /= np.linalg.norm(axis); ang = np.radians(50.0)
q = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])
t = np.array([0.3, -0.2, 6.0]) @ compose.P3D_TO_CV
s = 2.6
V = fabricate(X_true, q, t, s)
X, Rcl, t_cv, s_read = compose.compose_objects(V, q, t, [s, s, s], TOL)
assert np.abs(X - X_true).max() < 1e-9, np.abs(X - X_true).max()
assert abs(s_read - s) < 1e-12
# column form agrees with the row form
assert np.abs((Rcl @ (V * s).T).T + t_cv - X).max() < 1e-9
iou_ok = silhouette.iou(m_true, silhouette.silhouette(X, mesh.F, K, H, W))
assert iou_ok > 0.99, iou_ok
drops = {}
for name, Xw in wrong_variants(V, q, t, s).items():
    if (Xw[:, 2] <= 0).mean() > 0.5:
        drops[name] = 0.0; continue
    drops[name] = silhouette.iou(m_true, silhouette.silhouette(Xw, mesh.F, K, H, W))
    assert drops[name] < 0.9, (name, drops[name])
REPORT.append(f"test 1 composition round trip: max vertex error {np.abs(X - X_true).max():.1e}, IoU {iou_ok:.4f}; "
              "wrong variants: " + ", ".join(f"{k} {v:.3f}" for k, v in drops.items()) +
              " (probe measured xyzw 0.744->0.547, no conversion 0.795->0.000 on real anchors)")

# ---------------------------------------------------------------- test 10: scale isotropy gate
assert abs(compose.read_scale([2.0, 2.0, 2.0], TOL) - 2.0) < 1e-12
assert abs(compose.read_scale([2.0], TOL) - 2.0) < 1e-12
try:
    compose.read_scale([1.0, 1.0, 1.01], TOL)
    raise AssertionError("anisotropic scale was accepted")
except compose.ScaleAnisotropic:
    pass
try:
    compose.compose_objects(V, q, t, [1.0, 1.0, 1.01], TOL)
    raise AssertionError("anisotropic scale reached the composition")
except compose.ScaleAnisotropic:
    pass
REPORT.append("test 10 scale isotropy gate: (2,2,2) -> 2.0; (1,1,1.01) refused (ScaleAnisotropic); tol 1e-6 relative")

# ---------------------------------------------------------------- test 2: point-map refusal + builder
for src in ("stage_a", "fixture"):
    try:
        pointmap.require_pointmap(None, src)
        raise AssertionError(f"None accepted on {src}")
    except pointmap.PointMapMissing:
        pass
assert pointmap.require_pointmap(None, "vendor_internal") is None
try:
    pointmap.require_pointmap(np.zeros((4, 4, 3), np.float32), "vendor_internal")
    raise AssertionError("a map was accepted on vendor_internal")
except AssertionError as e:
    assert "must not carry" in str(e)
try:
    pointmap.require_pointmap(None, "moge")
    raise AssertionError("unknown source accepted")
except AssertionError as e:
    assert "unknown pointmap source" in str(e)
# builder: points lie on the rendered surface and reproject to their pixel centres
zb, m = proxy.render_perspective(X_true, mesh.F, K, H, W)
pm = pointmap.opencv_pointmap(zb, m, K, 18.0)
P = pm[m].astype(np.float64)
uv = (P @ K.T)[:, :2] / P[:, 2:3]
jj, ii = np.nonzero(m)
reproj = np.abs(uv - np.stack([ii + 0.5, jj + 0.5], 1)).max()
assert reproj < 1e-2, reproj
assert np.allclose(pm[~m][:, 2], 18.0)
# stage_a construction: f16 depth in, background at k x median
D16 = np.where(m, zb, np.nan).astype(np.float16)
pm_a, zmed = pointmap.stage_a_pointmap(D16, K, m, 3.0)
assert abs(zmed - float(np.median(zb[m]))) < 1e-2 and np.allclose(pm_a[~m][:, 2], 3.0 * zmed)
# fitted pinhole recovers K from the map
Kf = pointmap.fit_pinhole(pm, m)
assert np.abs(Kf - K).max() < 1e-3, Kf
REPORT.append(f"test 2 point-map refusal: None refused on stage_a/fixture, accepted only on vendor_internal; "
              f"builder reprojection error {reproj:.1e} px, background z 18.0, f16 stage_a path median {zmed:.3f}, "
              f"fitted pinhole error {np.abs(Kf - K).max():.1e}")

REPORT += ["", "all step-1 tests passed"]
Path("docs/report_m2b2_step1_tests.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
print("\n".join(REPORT))
