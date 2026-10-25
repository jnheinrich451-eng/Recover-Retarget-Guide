"""Proxy geometry for placement: triangle meshes, a small exact rasteriser,
and the weak-perspective proxy constants (SKILL_m2_c_C1 v1.1 §3).

Meshes rather than point clouds so the z-buffer has no holes and silhouettes
are hard (spec v3 Remark 1). All arithmetic f64. Camera axes: x right,
y down, z forward. Image convention: pixel (col i, row j) has its centre at
continuous coordinates (u, v) = (i + 0.5, j + 0.5); K maps 3D -> (u, v).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProxyMesh:
    name: str
    V: np.ndarray            # [Nv,3] vertices, canonical frame, unit scale
    F: np.ndarray            # [Nf,3] int triangle indices
    o_p: np.ndarray          # [3] proxy centre (mean vertex)
    B_tar: np.ndarray        # [3,3] canonical basis: columns forward, up, right


CANON_B_TAR = np.array([[1.0, 0.0, 0.0],      # forward = image-right (+x)
                        [0.0, -1.0, 0.0],     # up = -y
                        [0.0, 0.0, 1.0]]).T   # right = forward x up = (0,0,1)... see below
# columns: f=(1,0,0), u=(0,-1,0), r = f x u = (1,0,0)x(0,-1,0) = (0,0,-1)
CANON_B_TAR = np.stack([np.array([1.0, 0, 0]), np.array([0, -1.0, 0]),
                        np.cross([1.0, 0, 0], [0, -1.0, 0])], axis=1)
assert abs(np.linalg.det(CANON_B_TAR) - 1.0) < 1e-12


# ----------------------------------------------------------------------------
# mesh generators (unit scale, centred so o_p == mean vertex)
# ----------------------------------------------------------------------------
def uv_sphere(n_lat=300, n_lon=600, radii=(1.0, 1.0, 1.0), offset=(0.0, 0.0, 0.0), name="sphere"):
    """Faceting sag ~ (pi/n_lat)^2/8 pulls the rendered surface INSIDE the true
    sphere, biasing delta_p low: measured 1.5e-4 at 150x300 (2026-09-10), so the
    default is 300x600 (~4e-5). Convergence is reported by the harness step 2."""
    th = np.linspace(0, np.pi, n_lat + 1)
    ph = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    V = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3)
    V = V * np.asarray(radii) + np.asarray(offset)
    idx = np.arange((n_lat + 1) * n_lon).reshape(n_lat + 1, n_lon)
    a = idx[:-1, :]; b = idx[1:, :]; c = idx[1:, (np.arange(n_lon) + 1) % n_lon]
    d = idx[:-1, (np.arange(n_lon) + 1) % n_lon]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                        np.stack([a, c, d], -1).reshape(-1, 3)])
    return _finish(name, V, F)


def box(half=(1.0, 1.0, 1.0), name="box"):
    h = np.asarray(half, float)
    s = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float) * h
    F = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                  [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    return _finish(name, s, F)


def billboard(half=(1.0, 1.0), name="billboard"):
    """2x2 flat square in the x-y plane at z = 0 (faces the camera)."""
    hx, hy = half
    V = np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0]], float)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    return _finish(name, V, F)


def ellipsoid(radii=(1.3, 0.8, 0.6), name="ellipsoid"):
    """Axis-aligned ellipsoid: anisotropic but still analytic — viewed along z
    with up = -y, h_b = b, a_0 = pi a b, delta_p = c / sqrt(2) (the sphere's
    CDF argument scales axis by axis)."""
    return uv_sphere(radii=radii, name=name)


def asym(name="asym"):
    """Deliberately lopsided: an ellipsoid (1.3, 0.8, 0.6) with a bump so no
    symmetry makes a constant coincide with a tidy number."""
    m = uv_sphere(radii=(1.3, 0.8, 0.6), name=name)
    V = m.V.copy()
    bump = np.exp(-np.sum((V - np.array([0.9, -0.4, 0.2])) ** 2, 1) / 0.15)
    V += 0.35 * bump[:, None] * (V / np.linalg.norm(V, axis=1, keepdims=True))
    return _finish(name, V, m.F)


def _finish(name, V, F):
    V = np.asarray(V, np.float64)
    return ProxyMesh(name, V, np.asarray(F, int), V.mean(0), CANON_B_TAR.copy())


def make_proxy(kind: str) -> ProxyMesh:
    return {"sphere": uv_sphere, "ellipsoid": ellipsoid, "box": box, "billboard": billboard,
            "asym": asym}[kind]()


# ----------------------------------------------------------------------------
# rasteriser: nearest-depth z-buffer of a triangle mesh on a pixel grid
# ----------------------------------------------------------------------------
def rasterize(P2, Z, H, W, perspective):
    """P2 [T,3,2] continuous (u,v) per vertex; Z [T,3] vertex depths (> 0 for
    perspective). Cell (j,i) sampled at its centre (i+0.5, j+0.5). Perspective
    interpolates 1/z linearly (perspective-correct). Returns zbuf [H,W]
    (inf where empty) and mask."""
    zbuf = np.full((H, W), np.inf)
    Zi = 1.0 / Z if perspective else Z
    for t in range(P2.shape[0]):
        (u0, v0), (u1, v1), (u2, v2) = P2[t]
        i0, i1 = int(np.floor(min(u0, u1, u2) - 0.5)), int(np.ceil(max(u0, u1, u2) - 0.5))
        j0, j1 = int(np.floor(min(v0, v1, v2) - 0.5)), int(np.ceil(max(v0, v1, v2) - 0.5))
        i0, j0 = max(i0, 0), max(j0, 0)
        i1, j1 = min(i1, W - 1), min(j1, H - 1)
        if i1 < i0 or j1 < j0:
            continue
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
        px, py = ii + 0.5, jj + 0.5
        den = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        if abs(den) < 1e-15:
            continue
        w0 = ((v1 - v2) * (px - u2) + (u2 - u1) * (py - v2)) / den
        w1 = ((v2 - v0) * (px - u2) + (u0 - u2) * (py - v2)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-12) & (w1 >= -1e-12) & (w2 >= -1e-12)
        if not inside.any():
            continue
        zi = w0 * Zi[t, 0] + w1 * Zi[t, 1] + w2 * Zi[t, 2]
        z = (1.0 / zi) if perspective else zi
        sub = zbuf[j0:j1 + 1, i0:i1 + 1]
        upd = inside & (z < sub)
        sub[upd] = z[upd]
    return zbuf, np.isfinite(zbuf)


def rasterize_vec(P2, Z, H, W, perspective, max_pairs=4_000_000):
    """The same z-buffer as `rasterize`, vectorised over (triangle, bbox-pixel) pairs (SKILL C3
    v1.1 step 1 / m2_b2 §10b). Same bbox rule, same barycentric expressions in the same order,
    same -1e-12 inside tolerance, same perspective-correct 1/z: the per-(triangle, pixel) depths
    are bit-identical, and np.minimum.at over them equals the loop's "replace if strictly
    smaller" rule (both are the minimum over covering triangles; a shared-edge tie carries the
    same value either way). Tested bit-for-bit against `rasterize` on every harness mesh at 256
    and 1024 (tests/test_m2c_c3.py test 0). Chunked so the pair arrays stay under max_pairs."""
    P2 = np.asarray(P2, np.float64); Z = np.asarray(Z, np.float64)
    zbuf = np.full(H * W, np.inf)
    T = P2.shape[0]
    if T == 0:
        return zbuf.reshape(H, W), np.zeros((H, W), bool)
    Zi = 1.0 / Z if perspective else Z
    u0, v0 = P2[:, 0, 0], P2[:, 0, 1]; u1, v1 = P2[:, 1, 0], P2[:, 1, 1]; u2, v2 = P2[:, 2, 0], P2[:, 2, 1]
    i0 = np.floor(np.minimum(np.minimum(u0, u1), u2) - 0.5); i1 = np.ceil(np.maximum(np.maximum(u0, u1), u2) - 0.5)
    j0 = np.floor(np.minimum(np.minimum(v0, v1), v2) - 0.5); j1 = np.ceil(np.maximum(np.maximum(v0, v1), v2) - 0.5)
    i0 = np.maximum(i0, 0); j0 = np.maximum(j0, 0); i1 = np.minimum(i1, W - 1); j1 = np.minimum(j1, H - 1)
    ok = np.isfinite(i0) & np.isfinite(i1) & np.isfinite(j0) & np.isfinite(j1)
    i0 = np.where(ok, i0, 0).astype(np.int64); i1 = np.where(ok, i1, -1).astype(np.int64)
    j0 = np.where(ok, j0, 0).astype(np.int64); j1 = np.where(ok, j1, -1).astype(np.int64)
    den = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
    w = i1 - i0 + 1; h = j1 - j0 + 1
    n = np.where((w > 0) & (h > 0) & (np.abs(den) >= 1e-15), w * h, 0)
    csum = np.cumsum(n)
    a = 0
    while a < T:
        base = csum[a] - n[a]
        b = max(int(np.searchsorted(csum - base, max_pairs, side="right")), a + 1)
        nn = n[a:b]; tot = int(nn.sum())
        if tot == 0:
            a = b; continue
        tid = np.repeat(np.arange(a, b), nn)
        k = np.arange(tot) - np.repeat(np.cumsum(nn) - nn, nn)
        ii = i0[tid] + k % w[tid]; jj = j0[tid] + k // w[tid]
        px, py = ii + 0.5, jj + 0.5
        d = den[tid]
        w0 = ((v1[tid] - v2[tid]) * (px - u2[tid]) + (u2[tid] - u1[tid]) * (py - v2[tid])) / d
        w1 = ((v2[tid] - v0[tid]) * (px - u2[tid]) + (u0[tid] - u2[tid]) * (py - v2[tid])) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-12) & (w1 >= -1e-12) & (w2 >= -1e-12)
        zi = w0 * Zi[tid, 0] + w1 * Zi[tid, 1] + w2 * Zi[tid, 2]
        z = (1.0 / zi) if perspective else zi
        sel = inside
        np.minimum.at(zbuf, (jj[sel] * W + ii[sel]), z[sel])
        a = b
    zbuf = zbuf.reshape(H, W)
    return zbuf, np.isfinite(zbuf)


def render_perspective(V_world, F, K, H, W, fast=True):
    """Mesh in camera coordinates (z > 0) -> (zbuf, mask) at process resolution.
    fast=True uses the vectorised z-buffer (bit-identical to the loop, test 0 of C3)."""
    P = V_world @ K.T
    uv = P[:, :2] / P[:, 2:3]
    fn = rasterize_vec if fast else rasterize
    return fn(uv[F], V_world[F][..., 2], H, W, perspective=True)


def render_orthographic(V, F, grid_n, extent=None):
    """Orthographic along +z: (u,v) = (x, y) scaled to a grid of grid_n cells
    over [-extent, extent]^2. Returns (zbuf, mask, cell_size)."""
    if extent is None:
        extent = float(np.abs(V[:, :2]).max()) * 1.02
    cell = 2.0 * extent / grid_n
    uv = (V[:, :2] + extent) / cell            # continuous cell coordinates
    zbuf, mask = rasterize_vec(uv[F], V[F][..., 2], grid_n, grid_n, perspective=False)
    return zbuf, mask, cell


# ----------------------------------------------------------------------------
# weak-perspective proxy constants (SKILL §3): ORTHOGRAPHIC, unit scale
# ----------------------------------------------------------------------------
def aligned_vertices(mesh: ProxyMesh, R_A):
    """R_A (p - o_p): the proxy posed, centred at the origin, unit scale."""
    return (mesh.V - mesh.o_p) @ np.asarray(R_A, float).T


def h_b(mesh: ProxyMesh, R_A, n):
    """Extent below the centre along the plane normal n: -min_i n·R_A(p_i - o_p)
    (spec 19). Extremes of a mesh are attained at vertices."""
    val = -float(np.min(aligned_vertices(mesh, R_A) @ np.asarray(n, float)))
    assert val > 0, f"h_b must be positive (got {val}); check the sign of n"
    return val


def orthographic_constants(mesh: ProxyMesh, R_A, grid_n=1600):
    """Grid default 1600, measured 2026-09-10 on the unit sphere (300x600 mesh):
    delta_p error -8.5e-5 at 800 cells (the TRUE sphere sampled on the same grid
    gives -7.8e-5, so it is the raster, not the mesh), -3.8e-6 at 1600, -1.8e-6 at
    3200; a_0 error 3.4e-4 / 3.3e-5 / 3.2e-5. Runtime is set by the triangle loop,
    not the grid (~12 s either way for 360k triangles)."""
    """a_0 = projected area at unit scale (orthographic along camera z);
    delta_p = z_centre - median visible depth = -median(zbuf) since the
    centre is at the origin."""
    Va = aligned_vertices(mesh, R_A)
    zbuf, mask, cell = render_orthographic(Va, mesh.F, grid_n)
    a0 = float(mask.sum()) * cell * cell
    dp = -float(np.median(zbuf[mask]))
    return {"a_0": a0, "delta_p": dp, "cell": cell, "n_cells": int(mask.sum())}


def constants(mesh: ProxyMesh, R_A, n, grid_n=1600):
    c = orthographic_constants(mesh, R_A, grid_n)
    c["h_b"] = h_b(mesh, R_A, n)
    c["projection"] = "orthographic"
    return c
