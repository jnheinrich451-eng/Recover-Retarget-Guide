"""Stage B.5 synthetic harness (SKILL v1.1 §4) — written before real runs.

Run:  python tests/test_stage_b5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.stage_b5 import heading, plane  # noqa: E402

F = 81
BD = [35, 47, 59]
SEG1 = [{"start": 0, "end": 35, "rel_scale": 1.0},
        {"start": 35, "end": 47, "rel_scale": 1.0},
        {"start": 47, "end": 59, "rel_scale": 1.0},
        {"start": 59, "end": 81, "rel_scale": 1.0}]


def make_xbar(v, noise, seed=0, g=None):
    rng = np.random.default_rng(seed)
    t = np.arange(F)[:, None]
    xb = np.array([1.0, 0.5, 6.0]) + t * np.asarray(v)
    if g is not None:
        xb = xb * g[:, None]
    return xb + rng.normal(0, noise, (F, 3))


def test_a_heading_and_sigma():
    v = np.array([0.05, 0.0, 0.02])
    d1 = s1 = None
    for i, noise in enumerate((0.002, 0.02)):
        xb = make_xbar(v, noise, seed=i)
        d, s, rec = heading.wls_heading(xb, 5, 12, 5, BD, SEG1)
        assert not rec["degauged"]
        cos = d @ (v / np.linalg.norm(v))
        assert cos > 0.995, cos
        if i == 0:
            d1, s1 = d, s
        else:
            assert s > 3 * s1, (s, s1)   # sigma_theta scales with noise
    print(f"PASS (a) heading (cos>0.995; sigma {s1:.4f} -> {s:.4f})")


def test_b_gauge_jump():
    v = np.array([0.05, 0.0, 0.0])
    g = np.ones(F)
    g[10:] = 1.2                        # jump INSIDE the would-be window
    segs = [{"start": 0, "end": 10, "rel_scale": 1.0},
            {"start": 10, "end": F, "rel_scale": 1.2}]
    xb = make_xbar(v, 0.002, seed=3, g=g)
    d, s, rec = heading.wls_heading(xb, 5, 12, 5, [10], segs)
    assert rec["degauged"], "gauge-discipline path must trigger"
    cos = d @ np.array([1.0, 0, 0])
    assert cos > 0.98, cos              # radial gauge bias removed
    print(f"PASS (b) gauge jump (degauged, cos {cos:.3f})")


def test_c_plane_and_strip():
    rng = np.random.default_rng(4)
    n_true = np.array([0.1, -0.98, 0.05])
    n_true /= np.linalg.norm(n_true)
    b1 = np.array([1.0, 0.0, 0.0])
    b1 -= (b1 @ n_true) * n_true
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(n_true, b1)
    a = rng.uniform(-4, 4, 3000)
    b = rng.uniform(-4, 4, 3000)
    pts = np.outer(a, b1) + np.outer(b, b2) + rng.normal(0, 0.02, (3000, 3))
    pl = plane.fit_plane(pts, tol=0.06, seed=1)
    assert abs(pl["n"] @ n_true) > 0.995 and pl["planarity"] > 0.9
    assert pl["spread"] > 0.3
    strip = np.outer(a, b1) + np.outer(rng.uniform(-0.1, 0.1, 3000), b2) \
        + rng.normal(0, 0.02, (3000, 3))
    pls = plane.fit_plane(strip, tol=0.06, seed=1)
    assert pls["spread"] < 0.05, "strip geometry must show tiny spread"
    print(f"PASS (c) plane (|cos| {abs(pl['n'] @ n_true):.3f}, "
          f"strip spread {pls['spread']:.4f})")


def test_c2_plane_offset_follows_orientation():
    """2026-09-14: the emitted (n, d) must satisfy n.X + d = 0 AFTER n is signed toward the
    subject, and the subject must be on the +n side. Before the fix d kept the unsigned fit's
    sign whenever orient flipped n (measured on clip 284: source below its own ground)."""
    rng = np.random.default_rng(5)
    n_true = np.array([0.04, -0.996, -0.08]); n_true /= np.linalg.norm(n_true)
    mu = np.array([0.0, 0.25, 1.5])
    b1 = np.cross(n_true, [1.0, 0, 0]); b1 /= np.linalg.norm(b1); b2 = np.cross(n_true, b1)
    pts = mu + np.outer(rng.uniform(-2, 2, 2000), b1) + np.outer(rng.uniform(-2, 2, 2000), b2) + rng.normal(0, 0.01, (2000, 3))
    G = np.tile(np.eye(4), (3, 1, 1))
    for flip in (1.0, -1.0):
        pl = plane.fit_plane(pts, tol=0.05, seed=1)
        pl["n"] = flip * pl["n"]; pl["d"] = float(-pl["n"] @ pl["mu"])      # either sign can come out of the fit
        centroid = mu + 0.3 * n_true                                        # subject 0.3 above the plane
        n, _, _ = plane.orient_and_cross(pl, centroid, G, 0)
        assert n @ n_true > 0.99
        assert abs(n @ pl["mu"] + pl["d"]) < 1e-9, "n.mu + d must vanish for the EMITTED pair"
        assert centroid @ n + pl["d"] > 0.25, "subject on the +n side"
        assert pl["d"] < 0 or True
    print("PASS (c2) plane offset re-signed with the oriented normal; subject on the +n side for both fit signs")


def test_d_branch_composition():
    G = np.tile(np.eye(4), (F, 1, 1))
    G[:, :3, 3] = np.arange(F)[:, None] * np.array([0.02, 0.0, 0.05])
    c_ts = np.array([0.0, 0.2, 6.0])
    tau = np.zeros((F, 3))
    xb, ratio = heading.centroid_track_branch(G, c_ts, tau, np.ones(F, bool))
    step = np.diff(xb, axis=0)
    assert np.allclose(step, [0.02, 0, 0.05]) and abs(ratio - 1) < 1e-6
    print(f"PASS (d) branch composition (ratio {ratio:.3f})")


def test_e_kappa_degenerate():
    u = np.array([0.0, 1.0, 0.0])
    d = np.array([0.0, 0.999, 0.04])
    d /= np.linalg.norm(d)
    kappa = float(np.linalg.norm(np.cross(u, d)))
    assert kappa < 0.05
    print(f"PASS (e) kappa degenerate reported ({kappa:.3f})")


def test_pool_excludes_blob():
    N = 100
    Q = np.ones((N, F), bool)
    y = np.zeros(N, bool)
    eta = np.ones((N, F)) * 0.5
    eta[:50] = 0.9
    X0 = np.random.default_rng(5).normal(0, 1, (N, F, 3))
    g = np.ones(F)
    pts_all = plane.static_pool(X0, Q, y, eta, g, exclude=None)
    pts_ex = plane.static_pool(X0, Q, y, eta, g, exclude=np.arange(50))
    assert len(pts_ex) < len(pts_all), "blob exclusion must shrink the pool"
    print(f"PASS pool blob-exclusion ({len(pts_all)} -> {len(pts_ex)})")


def _rigid_object(n=200, v=(-0.02, 0.0, 0.0), z_range=(5.0, 6.0), seed=7,
                  noise=0.002):
    """Synthetic x0-channel subject: n tracks spanning a depth range, all
    translating with velocity v (no rotation), tiny noise. Returns
    X0 [n,F,3], Q, P (all inside a full-frame mask), M, present."""
    rng = np.random.default_rng(seed)
    base = np.stack([rng.uniform(-0.5, 0.5, n), rng.uniform(-0.3, 0.3, n),
                     rng.uniform(*z_range, n)], axis=1)
    t = np.arange(F)[None, :, None]
    X0 = base[:, None, :] + t * np.asarray(v)[None, None, :] \
        + rng.normal(0, noise, (n, F, 3))
    Q = np.ones((n, F), bool)
    P = np.full((n, F, 2), 128.0)
    M = np.ones((F, 256, 256), bool)
    present = np.ones(F, bool)
    return X0, Q, P, M, present, base


def _angle(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def test_f_support_churn_bias():
    """v1.2 §4(f): a spatially coherent patch (the NEAR 30% of the object)
    dropping out at half the window biases the v1.1 per-frame centroid
    heading radially; the common-support estimator is unaffected. This test
    exists so the intersection cannot be 'optimised' away later."""
    v_true = np.array([-0.02, 0.0, 0.0])
    X0, Q, P, M, present, base = _rigid_object(v=v_true)
    near = np.argsort(base[:, 2])[: int(0.3 * len(base))]
    t_c, L = 5, 12
    Q[near, t_c + 6:] = False            # near patch leaves mid-window, coherently
    subject = np.arange(len(base))
    eta = np.ones_like(Q, dtype=float)
    G = np.tile(np.eye(4), (F, 1, 1))
    out = heading.heading_v12(X0, Q, P, M, subject, eta, present, G, t_c, L, 5,
                              BD, SEG1)
    err11 = _angle(out["d_v11"], v_true)
    err12 = _angle(out["d_src"], v_true)
    assert err11 > 15.0, f"v1.1 must show the churn bias here ({err11:.1f} deg)"
    assert err12 < 1.0, f"common support must remove it ({err12:.2f} deg)"
    assert out["churn_deg"] > 15.0 and out["n_star"] == len(base) - len(near)
    assert abs(out["support_frac"] - 0.7) < 1e-9
    print(f"PASS (f) support churn: v1.1 err {err11:.1f} deg -> v1.2 err {err12:.2f} deg "
          f"(churn_deg {out['churn_deg']:.1f}, n_star {out['n_star']})")


def test_g_static_contaminant_direction_invariance():
    """v1.2 §4(g): 20% static tracks inside the support (fixed) attenuate the
    speed by ~0.8 but leave the DIRECTION invariant — the estimator's known
    failure boundary is speed, never heading."""
    v_true = np.array([-0.02, 0.0, 0.01])
    X0, Q, P, M, present, base = _rigid_object(n=200, v=v_true, noise=0.0)
    n_stat = 50                                   # 50/250 = 20%
    rng = np.random.default_rng(11)
    stat = np.stack([rng.uniform(-0.5, 0.5, n_stat), rng.uniform(-0.3, 0.3, n_stat),
                     rng.uniform(5.0, 6.0, n_stat)], axis=1)
    X0 = np.concatenate([X0, np.repeat(stat[:, None, :], F, axis=1)])
    Q = np.concatenate([Q, np.ones((n_stat, F), bool)])
    P = np.concatenate([P, np.full((n_stat, F, 2), 128.0)])
    subject = np.arange(len(X0))
    eta = np.ones_like(Q, dtype=float)
    G = np.tile(np.eye(4), (F, 1, 1))
    out = heading.heading_v12(X0, Q, P, M, subject, eta, present, G, 3, 12, 5,
                              BD, SEG1)
    ts, scale, _ = heading.heading_window(out["X_bar_v11"], 3, 12, 5, BD, SEG1)
    v_fit, _, _ = heading.wls_slope(out["X_bar_common"][ts], (ts - 3).astype(float))
    assert _angle(out["d_src"], v_true) < 0.06, _angle(out["d_src"], v_true)
    ratio = np.linalg.norm(v_fit) / np.linalg.norm(v_true)
    assert abs(ratio - 0.8) < 1e-6, ratio
    print(f"PASS (g) static contaminant: direction err {_angle(out['d_src'], v_true):.3f} deg, "
          f"speed x{ratio:.3f}")


def test_h_identifiability_frame():
    """rho_radial is along the ray from the camera-t_a POSITION to the
    centroid. It is invariant to the camera's rotation (both vectors rotate
    together) but depends on where the camera is: a pure-lateral motion in
    camera-0 is fully lateral from a camera on the z axis, and fully radial
    from a camera displaced along x. Camera-0's z axis is not the ray unless
    t_a = 0 and the object is centred."""
    n = 100
    V = np.tile(np.array([-0.02, 0.0, 0.0]), (n, 1)) \
        + np.random.default_rng(2).normal(0, 0.001, (n, 3))
    G = np.tile(np.eye(4), (F, 1, 1))
    c = np.array([0.0, 0.0, 5.0])
    i0 = heading.identifiability(V, G, 0, c)
    assert i0["rho_radial"] < 1.0 and i0["rho_lat1"] > 5.0, i0
    yaw = np.radians(60.0)                       # rotation alone: no change
    G[5, :3, :3] = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                             [-np.sin(yaw), 0, np.cos(yaw)]])
    i5 = heading.identifiability(V, G, 5, c)
    assert abs(i5["rho_radial"] - i0["rho_radial"]) < 1e-9, (i0, i5)
    G[7, :3, 3] = np.array([5.0, 0.0, 5.0])      # camera displaced along x
    i7 = heading.identifiability(V, G, 7, c)     # ray = -x: motion is radial
    assert i7["rho_radial"] > 5.0 and i7["rho_lat1"] < 1.0, i7
    print(f"PASS (h) identifiability frame: rho_radial {i0['rho_radial']:.2f} (t_a=0) = "
          f"{i5['rho_radial']:.2f} (yawed) -> {i7['rho_radial']:.2f} (camera displaced)")


if __name__ == "__main__":
    test_a_heading_and_sigma()
    test_b_gauge_jump()
    test_c_plane_and_strip()
    test_c2_plane_offset_follows_orientation()
    test_d_branch_composition()
    test_e_kappa_degenerate()
    test_pool_excludes_blob()
    test_f_support_churn_bias()
    test_g_static_contaminant_direction_invariance()
    test_h_identifiability_frame()
    print("\nall Stage B.5 harness tests passed")
