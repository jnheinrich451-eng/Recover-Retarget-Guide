"""Reference constants and the seed medoid (SKILL §5 level 2, §7).

These are NOT the constants m2_c consumes: a_0, h_b, delta_p depend on R_A, which is per clip,
so computing them at the working pose would make the medoid clip-dependent and break the
category cache. Reference constants are computed once at R_A = I with up = u~_can on the
NORMALISED mesh, purely for medoid selection and QC, and travel under the `ref_` namespace
(test 6: they never reach m2_c).

h_b here is the LOW PERCENTILE of the spec amendment (§9): the P4 proxy cars gave p01/min
0.84-0.89 (stray geometry below the body, +11-16 % on a strict min) where clean fixtures gave
p01 - min = -0.03; p01 under-reads a smooth extremum by ~3 % (tessellation density, measured).
a_0 and delta_p come from the C1 harness raster unchanged (`proxy.orthographic_constants`);
C1's own `proxy.h_b` keeps its min until the spec amendment lands.
"""
from __future__ import annotations

import numpy as np

from src.anchor.m2c import proxy


def normalise(V):
    """SKILL §7: centre at the vertex centroid o_p, scale so max|v - o_p| = 1.
    Returns (V_norm, o_p, norm_scale) with V_norm = (V - o_p) / norm_scale."""
    V = np.asarray(V, np.float64)
    assert V.ndim == 2 and V.shape[1] == 3 and np.isfinite(V).all()
    o_p = V.mean(0)
    norm_scale = float(np.linalg.norm(V - o_p, axis=1).max())
    assert norm_scale > 0, "degenerate mesh (all vertices coincide)"
    return (V - o_p) / norm_scale, o_p, norm_scale


def h_b_percentile(V_centred, n, q: float):
    """-quantile_q(n . v) and the strict -min, both returned so p01 - min is a QC field."""
    proj = np.asarray(V_centred, np.float64) @ np.asarray(n, np.float64)
    hb_q = float(-np.quantile(proj, q))
    hb_min = float(-proj.min())
    assert hb_q > 0, f"h_b must be positive (got {hb_q}); check the sign of the up axis"
    return hb_q, hb_min


def reference_constants(V_norm, F, up, q: float, grid_n: int) -> dict:
    """At R_A = I on the normalised mesh. up = u~_can (unit); the plane normal is `up` itself
    (points away from the ground). Returns the ref_ namespace only."""
    up = np.asarray(up, np.float64); up = up / np.linalg.norm(up)
    mesh = proxy.ProxyMesh("ref", np.asarray(V_norm, np.float64), np.asarray(F, int), np.zeros(3), proxy.CANON_B_TAR.copy())
    c = proxy.orthographic_constants(mesh, np.eye(3), grid_n)          # a_0, delta_p — harness code, unchanged
    hb_q, hb_min = h_b_percentile(mesh.V, up, q)
    return {"ref_a0": float(c["a_0"]), "ref_dp": float(c["delta_p"]), "ref_hb": hb_q,
            "ref_hb_min": hb_min, "ref_hb_p01_minus_min": hb_q - hb_min, "ref_q": q, "ref_grid": grid_n}


def medoid_features(consts: list[dict]) -> np.ndarray:
    """SKILL §5: (hb_ref, dp_ref/hb_ref, a0_ref/hb_ref^2) per seed."""
    return np.array([[c["ref_hb"], c["ref_dp"] / c["ref_hb"], c["ref_a0"] / c["ref_hb"] ** 2] for c in consts], np.float64)


def medoid(features) -> tuple[int, np.ndarray]:
    """Standardise each feature by its across-seed median and 1.4826*MAD (a zero MAD drops the
    feature), L1 distance, medoid = argmin of the summed distance to the other seeds; ties ->
    lowest index. Deterministic under any reordering of the seeds (test 4)."""
    X = np.asarray(features, np.float64)
    assert X.ndim == 2 and len(X) >= 1 and np.isfinite(X).all()
    if len(X) == 1:
        return 0, np.zeros(1)
    med = np.median(X, 0)
    mad = 1.4826 * np.median(np.abs(X - med), 0)
    keep = mad > 0
    if not keep.any():                                   # all seeds identical in every feature
        return 0, np.zeros(len(X))
    Z = (X[:, keep] - med[keep]) / mad[keep]
    D = np.abs(Z[:, None, :] - Z[None, :, :]).sum(-1)    # [N,N] L1
    total = D.sum(1)
    idx = int(np.flatnonzero(total <= total.min() + 1e-12)[0])
    return idx, total
