"""m2_c / C2 — background depth sanitisation (SKILL_m2_c_C2_depth_sanitisation.md).

Three operations, all on disparity zeta = 1/z: dilate, fill (affine ring fit +
harmonic residual solve per component), coarsen (Gaussian, area-average to an
N x N grid). Canonical output is the N x N grid. No normalisation, no
composition, no proxy (those are C3).

Why disparity (SKILL §1): a ground plane n·X + d = 0 gives zeta = -(n·rho)/d,
affine in pixel coordinates, so the ring's affine model CONTAINS the truth and
the harmonic residual of planar ground is identically zero. In depth the
Laplacian is -2|grad A|^2 z^3 / d^2 — nonzero and growing as z^3.

Camera axes x right, y down, z forward; pixel (col i, row j) centre at
(u, v) = (i + 0.5, j + 0.5); K maps 3-D to (u, v). f64 throughout; stored
depth (f16) is cast before arithmetic.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

PARAM_KEYS = ("r", "w", "N", "sigma1", "sigma2", "fill_order")


# ----------------------------------------------------------------------------
# §1 closed forms
# ----------------------------------------------------------------------------
def pixel_grid(H, W):
    """Continuous pixel-centre coordinates (u, v) for every cell."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    return u + 0.5, v + 0.5


def affine_from_plane(n, d, K):
    """(alpha, beta, gamma) with zeta(u,v) = alpha u + beta v + gamma for the
    plane n·X + d = 0 (SKILL §1 boxed formula)."""
    n = np.asarray(n, np.float64); K = np.asarray(K, np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    alpha = -n[0] / (d * fx)
    beta = -n[1] / (d * fy)
    gamma = (n[0] * cx / fx + n[1] * cy / fy - n[2]) / d
    return np.array([alpha, beta, gamma])


def plane_from_affine(coef, K):
    """Invert (alpha, beta, gamma) -> (n_hat, d) with the scale fixed by |n| = 1.
    Sign: d > 0 (camera above the plane on the object's side)."""
    a, b, g = (float(x) for x in coef)
    K = np.asarray(K, np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    nx, ny = -a * fx, -b * fy                          # with d = 1
    nz = nx * cx / fx + ny * cy / fy - g
    n = np.array([nx, ny, nz])
    norm = np.linalg.norm(n)
    return n / norm, 1.0 / norm


def plane_disparity(n, d, K, H, W):
    """Exact disparity field of a plane on the pixel grid — computed from the
    3-D definition (not the affine formula) so the two can be cross-checked."""
    n = np.asarray(n, np.float64); K = np.asarray(K, np.float64)
    u, v = pixel_grid(H, W)
    rho = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    return -(rho @ n) / d


def disparity(depth, valid=None):
    """zeta = 1/z from stored depth (cast first); validity excludes z <= 0,
    non-finite, and any caller-supplied invalid mask."""
    z = np.asarray(depth).astype(np.float64)
    ok = np.isfinite(z) & (z > 0)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    zeta = np.zeros_like(z)
    zeta[ok] = 1.0 / z[ok]
    return zeta, ok


# ----------------------------------------------------------------------------
# §2.1 dilate, §2.2 ring + affine fit
# ----------------------------------------------------------------------------
def disc(r):
    r = int(r)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def dilate(mask, r):
    mask = np.asarray(mask, bool)
    return ndimage.binary_dilation(mask, structure=disc(r)) if r > 0 else mask.copy()


def ring_of(omega, w, valid):
    """Boundary ring: omega dilated outward by w, minus omega, intersected with valid."""
    return dilate(omega, w) & ~omega & np.asarray(valid, bool)


def ring_fit(zeta, ring, H, W):
    """Least-squares affine fit on the ring, in CENTRED, SCALED coordinates so the
    condition number measures ring geometry (collinearity), not pixel numerics
    (SKILL review note 1). Returns coef in raw (u,v) units, plus the record."""
    u, v = pixel_grid(H, W)
    su, sv = u[ring], v[ring]
    n_ring = int(ring.sum())
    if n_ring < 3:
        return None, {"ring_px": n_ring, "ring_cond": float("inf"), "ring_residual_rms": float("nan")}
    x = (su - W / 2.0) / W
    y = (sv - H / 2.0) / H
    A = np.stack([x, y, np.ones_like(x)], 1)
    N = A.T @ A
    cond = float(np.linalg.cond(N))
    if not np.isfinite(cond):
        return None, {"ring_px": n_ring, "ring_cond": cond, "ring_residual_rms": float("nan")}
    c, *_ = np.linalg.lstsq(A, zeta[ring], rcond=None)
    rms = float(np.sqrt(np.mean((A @ c - zeta[ring]) ** 2)))
    # back to raw units: zeta = c0 (u - W/2)/W + c1 (v - H/2)/H + c2
    alpha, beta = c[0] / W, c[1] / H
    gamma = c[2] - c[0] * 0.5 - c[1] * 0.5
    return np.array([alpha, beta, gamma]), {"ring_px": n_ring, "ring_cond": cond,
                                             "ring_residual_rms": rms}


def eval_affine(coef, H, W):
    u, v = pixel_grid(H, W)
    return coef[0] * u + coef[1] * v + coef[2]


# ----------------------------------------------------------------------------
# §2.3 harmonic residual solve (Dirichlet, 5-point Laplacian, per component)
# ----------------------------------------------------------------------------
def harmonic_solve(boundary_vals, interior, known):
    """Solve grad^2 f = 0 on `interior` with f = boundary_vals on `known`
    (4-neighbours of interior cells that are known). Interior cells whose
    neighbours are neither interior nor known see a Neumann (mirror) condition
    at the image border. Returns f on the full grid (0 outside interior)."""
    H, W = interior.shape
    idx = -np.ones((H, W), int)
    ii, jj = np.nonzero(interior)
    idx[ii, jj] = np.arange(len(ii))
    n = len(ii)
    if n == 0:
        return np.zeros((H, W))
    rows, cols, vals = [], [], []
    b = np.zeros(n)
    diag = np.zeros(n)
    for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nj, ni = ii + dj, jj + di
        inb = (nj >= 0) & (nj < H) & (ni >= 0) & (ni < W)
        k = np.arange(n)
        # neighbour inside the image
        kk, nj_, ni_ = k[inb], nj[inb], ni[inb]
        nidx = idx[nj_, ni_]
        is_int = nidx >= 0
        is_known = ~is_int & known[nj_, ni_]
        # interior neighbour: off-diagonal -1
        rows.append(kk[is_int]); cols.append(nidx[is_int]); vals.append(-np.ones(is_int.sum()))
        diag[kk[is_int]] += 1
        # known neighbour: moves to RHS
        b[kk[is_known]] += boundary_vals[nj_[is_known], ni_[is_known]]
        diag[kk[is_known]] += 1
        # neighbour neither interior nor known (unknown & unfilled, or off-image): Neumann,
        # contributes nothing (mirror)
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    L = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    # guard: a component with no known neighbour at all is singular -> zero residual
    if not np.any(diag > 0) or L.diagonal().min() <= 0:
        f = np.zeros((H, W)); return f
    x = spsolve(L.tocsc(), b)
    f = np.zeros((H, W))
    f[ii, jj] = x
    return f


def fill(zeta, valid, mask, params, K=None):
    """§2.1–2.3. Unknown region = dilate(mask, r) ∪ invalid. Each connected
    component gets its own ring, affine fit, and residual solve. Returns
    (zeta_fill, valid_out, record). The mask's own component(s) define the
    reported affine / plane; hole components are filled but reported separately."""
    zeta = np.asarray(zeta, np.float64); valid = np.asarray(valid, bool)
    H, W = zeta.shape
    r, w = int(params["r"]), int(params["w"])
    omega_obj = dilate(mask, r)
    unknown = omega_obj | ~valid
    out = zeta.copy()
    known = valid & ~unknown
    labels, n_comp = ndimage.label(unknown)
    comps = []
    affine_obj, rec_obj = None, None
    for c in range(1, n_comp + 1):
        comp = labels == c
        touches_obj = bool((comp & omega_obj).any())
        on_border = bool(comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any())
        entry = {"component": c, "n_px": int(comp.sum()), "is_object": touches_obj,
                 "touches_border": on_border}
        # A non-object unknown region that reaches the image border has no enclosing ring —
        # the sky above the horizon is the standing example (34% of a synthetic frame, ring =
        # one straight line, cond 2e4, measured 2026-09-11): filling it extrapolates the ground
        # plane into a region nothing observed. Structural rule, no threshold: leave it invalid
        # and record it. Interior holes are filled; the object's component is always attempted.
        if on_border and not touches_obj:
            entry.update({"filled": False, "reason": "border_component_no_enclosing_ring",
                          "ring_px": None, "ring_cond": None, "ring_residual_rms": None})
            comps.append(entry)
            continue
        ring = ring_of(comp, w, known)
        coef, rec = ring_fit(zeta, ring, H, W)
        entry.update(rec)
        if coef is None:
            entry["filled"] = False
            comps.append(entry)
            continue
        base = eval_affine(coef, H, W)
        resid_known = np.where(known, zeta - base, 0.0)
        resid = harmonic_solve(resid_known, comp, known)
        out[comp] = base[comp] + resid[comp]
        entry.update({"filled": True, "affine": coef.tolist(),
                      "resid_max_abs": float(np.abs(resid[comp]).max()) if comp.any() else 0.0})
        comps.append(entry)
        if touches_obj and (affine_obj is None or entry["n_px"] > rec_obj["n_px"]):
            affine_obj, rec_obj = coef, entry
    filled = np.array([e["filled"] for e in comps], bool) if comps else np.zeros(0, bool)
    valid_out = valid.copy()
    for e in comps:
        if e["filled"]:
            valid_out[labels == e["component"]] = True
    record = {"n_components": int(n_comp), "components": comps,
              "invalid_frac_in": float(1 - valid.mean()),
              "omega_frac": float(omega_obj.mean()),
              "affine": None if affine_obj is None else affine_obj.tolist(),
              "ring_px": None if rec_obj is None else rec_obj["ring_px"],
              "ring_cond": None if rec_obj is None else rec_obj["ring_cond"],
              "ring_residual_rms": None if rec_obj is None else rec_obj["ring_residual_rms"]}
    if affine_obj is not None and K is not None:
        n_hat, d_hat = plane_from_affine(affine_obj, K)
        record["plane_from_fit"] = [*n_hat.tolist(), float(d_hat)]
    return out, valid_out, record


# ----------------------------------------------------------------------------
# §2.4 coarsen (validity-aware), and the deterministic upsample C3 will use
# ----------------------------------------------------------------------------
def gaussian_valid(field, valid, sigma):
    """Normalised convolution: blur only over valid pixels; invalid stay invalid."""
    if sigma <= 0:
        return field.copy(), valid.copy()
    f = np.where(valid, field, 0.0)
    num = ndimage.gaussian_filter(f, sigma, mode="nearest")
    den = ndimage.gaussian_filter(valid.astype(np.float64), sigma, mode="nearest")
    out = np.zeros_like(field)
    ok = den > 1e-9
    out[ok] = num[ok] / den[ok]
    return out, ok & valid


def area_downsample(field, valid, N):
    """Area average onto an N x N grid dividing by the count of VALID
    contributors per cell; cells with no valid contributor are invalid."""
    H, W = field.shape
    assert H % N == 0 and W % N == 0, f"{H}x{W} not divisible by N={N}"
    h, w = H // N, W // N
    f = np.where(valid, field, 0.0).reshape(N, h, N, w).sum((1, 3))
    c = valid.reshape(N, h, N, w).sum((1, 3)).astype(np.float64)
    out = np.zeros((N, N))
    ok = c > 0
    out[ok] = f[ok] / c[ok]
    return out, ok


def keys_kernel(x, a=-0.5):
    x = np.abs(np.asarray(x, np.float64))
    return np.where(x < 1, (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1,
                    np.where(x < 2, a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a, 0.0))


def _resample_matrix(n_src, n_dst, a=-0.5):
    """Separable cubic-convolution matrix [n_dst, n_src] with pixel-centre
    alignment (dst centre (j+0.5)/s - 0.5 in source cells), edge-clamped."""
    s = n_dst / n_src
    xs = (np.arange(n_dst) + 0.5) / s - 0.5
    i0 = np.floor(xs).astype(int)
    M = np.zeros((n_dst, n_src))
    for k in range(-1, 3):
        idx = np.clip(i0 + k, 0, n_src - 1)
        np.add.at(M, (np.arange(n_dst), idx), keys_kernel(xs - (i0 + k), a))
    return M


def upsample_bicubic(coarse, H, W):
    """Deterministic bicubic upsample (C3's; provided here so T4 can round-trip).
    Keys cubic convolution with a = -0.5 (Catmull-Rom), pixel-centre aligned.
    MEASURED 2026-09-11: OpenCV's INTER_CUBIC (a = -0.75) does NOT reproduce
    linear functions (first-moment error 4.8e-2 -> 8.4e-4 on a plane), a = -0.5
    does to 2e-16 — so a plane survives coarsening exactly only with this kernel.
    The spec's "bicubic" is pinned to a = -0.5 here and in C3."""
    c = np.asarray(coarse, np.float64)
    Mv = _resample_matrix(c.shape[0], H)
    Mu = _resample_matrix(c.shape[1], W)
    return Mv @ c @ Mu.T


def coarsen(zeta_fill, valid, params):
    N, s1, s2 = int(params["N"]), float(params["sigma1"]), float(params["sigma2"])
    f1, v1 = gaussian_valid(np.asarray(zeta_fill, np.float64), np.asarray(valid, bool), s1)
    zc, vc = area_downsample(f1, v1, N)
    zc2, vc2 = gaussian_valid(zc, vc, s2)      # sigma2 applied on the coarse grid (cells)
    return zc2, vc2


# ----------------------------------------------------------------------------
# gates (thresholds are lock values, all unset -> ungated)
# ----------------------------------------------------------------------------
def _gate(stat, threshold, cond):
    if threshold is None:
        return {**stat, "threshold": None, "pass": None, "outcome": "ungated"}
    ok = bool(cond(float(threshold)))
    return {**stat, "threshold": threshold, "pass": ok, "outcome": "pass" if ok else "fail"}


def evaluate_gates(record, valid_in, valid_coarse, leakage, th):
    th = th or {}
    g = {}
    cond = record.get("ring_cond")
    g["g_ring_valid"] = {"ring_px": record.get("ring_px"), "collinear": cond is None or not np.isfinite(cond),
                         "pass": bool(record.get("affine") is not None),
                         "outcome": "pass" if record.get("affine") is not None else "fail"}
    g["g_ring_cond"] = _gate({"ring_cond": cond}, th.get("m2c_ring_cond_max"),
                             lambda t: cond is not None and np.isfinite(cond) and cond <= t)
    g["g_mask_frac"] = _gate({"omega_frac": record["omega_frac"]}, th.get("m2c_mask_frac_max"),
                             lambda t: record["omega_frac"] <= t)
    vf = float(np.mean(valid_in))
    g["g_invalid_frac"] = _gate({"valid_frac": vf}, th.get("m2c_valid_frac_min"), lambda t: vf >= t)
    g["g_leakage"] = {**leakage, "threshold": th.get("m2c_leakage_max"), "pass": None,
                      "outcome": "recorded"}                 # record, never refuse (SKILL §6)
    refusing = ("g_ring_valid", "g_ring_cond", "g_mask_frac", "g_invalid_frac")
    admitted = all(g[k]["pass"] is not False for k in refusing)
    return g, admitted


def sanitise(depth, mask, K, params, valid=None, thresholds=None, zeta_obj_true=None):
    """The C2 stage end to end. Returns (zeta_coarse, valid_coarse, record).
    zeta_obj_true is HARNESS-ONLY (leakage correlation against the removed
    object); production passes None and the correlation term is absent."""
    zeta, ok = disparity(depth, valid)
    H, W = zeta.shape
    zf, vf, rec = fill(zeta, ok, mask, params, K)
    zc, vc = coarsen(zf, vf, params)
    leak = leakage_score(zc, vc, mask, rec.get("affine"), H, W, zeta_obj_true)
    gates, admitted = evaluate_gates(rec, ok, vc, leak, thresholds)
    rec.update({"gates": gates, "admitted": bool(admitted), "leakage": leak,
                "invalid_frac_coarse": float(1 - vc.mean()),
                "params": {k: params.get(k) for k in PARAM_KEYS}})
    return zc, vc, rec


# ----------------------------------------------------------------------------
# T6 leakage measures (also reported in production; the correlation needs the
# harness's knowledge of the removed object and is absent otherwise)
# ----------------------------------------------------------------------------
def leakage_score(zeta_coarse, valid_coarse, mask, affine, H, W, zeta_obj_true=None):
    up = upsample_bicubic(zeta_coarse, H, W)
    vup = upsample_bicubic(valid_coarse.astype(np.float64), H, W) > 0.5
    base = eval_affine(np.asarray(affine), H, W) if affine is not None else np.zeros((H, W))
    detr = up - base
    M = np.asarray(mask, bool)
    out = {}
    # gradient ridge along the original silhouette vs background at the same scale
    gy, gx = np.gradient(up)
    gmag = np.hypot(gx, gy)
    sil = M & ~ndimage.binary_erosion(M)
    sil = dilate(sil, 1) & vup
    bg = ~dilate(M, 6) & vup
    out["ridge_ratio"] = float(np.median(gmag[sil]) / max(np.median(gmag[bg]), 1e-15)) if sil.any() and bg.any() else float("nan")
    out["ridge_ratio_definition"] = "median |grad zeta_bg| on the 1-px silhouette band / median over background >6 px away"
    if zeta_obj_true is not None:
        inside = M & vup
        a = detr[inside]; b = (np.asarray(zeta_obj_true, np.float64) - base)[inside]
        if inside.sum() > 3 and a.std() > 0 and b.std() > 0:
            out["corr_detrended"] = float(np.corrcoef(a, b)[0, 1])
            out["corr_raw"] = float(np.corrcoef(up[inside], np.asarray(zeta_obj_true, np.float64)[inside])[0, 1])
        else:
            out["corr_detrended"] = out["corr_raw"] = float("nan")
    return out
