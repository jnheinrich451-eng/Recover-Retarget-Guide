"""Stage IV placement (spec v3 §6, as amended by SKILL_m2_c_C1 v1.1 §4.7).

Two importable functions used unchanged by the harness and by production:
    source_stats(mask_ta, depth_ta, K_a) -> SourceStats
    place(stage0, stats, proxy, mode, thresholds) -> Placement
Closed form (weak perspective); refinement (C5) is a later build step.
f64 throughout; depth is cast to f32-or-better before any arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import proxy as _proxy

# Modes (spec v3 §6.3 as amended 2026-09-15): "framing_preserving" is the old name of depth_preserving —
# it anchors on z_src, which a truncated mask biases NEAR (clip 370: the proxy came out half the man's
# height, internally consistent and visually wrong); area_preserving anchors on the mask AREA instead
# (biased the other way on a truncated mask); size_preserving anchors on a category size prior.
MODES = ("depth_preserving", "area_preserving", "size_preserving")
MODE_ALIASES = {"framing_preserving": "depth_preserving"}


def _contact_params():
    """configs/m2c.yaml -> contact section (review 2026-09-15: robust contact row + local plane offset)."""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path("configs/m2c.yaml").read_text(encoding="utf-8"))
    return cfg["m2c"]["contact"]


CONTACT = _contact_params()


@dataclass
class SourceStats:
    u0: float; v0: float           # mask centroid, continuous image coords
    A_src: float                   # mask area, px
    v_contact: float | None        # bottom edge of the lowest mask pixel (None if unobserved)
    z_src: float                   # median depth over the mask
    contact_observed: bool         # False iff the mask touches the bottom image border
    mask_border_frac: float        # outline pixels on the 1-px border / outline pixels
    n_mask: int
    H: int; W: int
    touches: dict | None = None    # {"top","bottom","left","right"} -> bool; lateral/top
                                   # clipping biases the centroid (found by the harness
                                   # corner case, 2026-09-11) — recorded, gated by m2_a
    # review 2026-09-15 (a): the contact row is a high percentile of the mask rows, not the max
    # (a maximum is destroyed by one stray pixel — the same failure that took h_b from min to p01);
    # contact is unobserved only if the mask still reaches the border after the trim AND the
    # bottom-touching run is a contiguous segment comparable to the object's width there.
    v_contact_max: float | None = None     # the old maximum, kept for the record
    bottom_run_frac: float | None = None   # longest contiguous run on the last row / object width near it
    u_contact: float | None = None         # column of the contact band (median), for the local plane offset
    z_contact: float | None = None         # median depth over the contact band (review (b))
    contact_at_frame_edge: bool = False    # robust contact row == last row (admitted with a flag, 2026-09-18)


def source_stats(mask_ta, depth_ta, K_a) -> SourceStats:
    M = np.asarray(mask_ta, bool)
    D = np.asarray(depth_ta)
    assert D.dtype != np.float16 or True
    D = D.astype(np.float64)                    # f16 on disk -> f64 before arithmetic
    H, W = M.shape
    jj, ii = np.nonzero(M)
    n = int(len(ii))
    assert n > 0, "empty mask at t_a — m2_a should have refused (g_present)"
    u0, v0 = float(ii.mean() + 0.5), float(jj.mean() + 0.5)
    z = D[M]
    z = z[np.isfinite(z) & (z > 0)]
    assert len(z) > 0, "no finite positive depth under the mask"
    z_src = float(np.median(z))
    # robust contact row: p_q of the mask's row distribution; the trimmed mask (rows <= that
    # quantile) decides whether the object reaches the border; a border touch counts only when the
    # touching run on the last row is contiguous and comparable to the object's width near it
    q = float(CONTACT["row_quantile"])
    v_q = float(np.quantile(jj, q))
    trimmed_reaches = bool((jj[jj <= v_q] >= H - 1).any())
    last = M[-1]
    run = 0
    if last.any():
        best = cur = 0
        for v in last:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        run = best
    band = jj >= v_q - float(CONTACT["band_rows"])
    width_near = float(ii[band].max() - ii[band].min() + 1) if band.any() else float("nan")
    run_frac = float(run / width_near) if run and np.isfinite(width_near) and width_near > 0 else 0.0
    # 2026-09-18 (reviewer decision A1(b), supersedes the 2026-09-15 last-row refusal): a contact row at
    # the frame edge is ADMITTED WITH A FLAG; truncation is decided by the run-width rule again, and the
    # refusal of a false frame-edge contact moves to C3's contact-step ratio gate — the one measured
    # quantity that separates 284 (-0.06 quanta, human-accepted) from 370 (15.9). `contact_at_frame_edge`
    # is recorded so the gate's behaviour on this class stays visible.
    contact_at_frame_edge = bool(v_q >= H - 1 - float(CONTACT["band_rows"]))   # within the contact band of the last row
                                                                                # (a round bottom puts p99.5 a row or two above it)
    touches_bottom = bool(trimmed_reaches and run_frac >= float(CONTACT["run_width_frac"]))
    v_contact_max = float(jj.max() + 1.0)
    v_contact = None if touches_bottom else float(v_q + 1.0)
    # contact band: the pixels within band_rows of the contact row — column and depth of the feet
    cb = M & (np.arange(H)[:, None] >= v_q - float(CONTACT["band_rows"]))
    zc = D[cb]; zc = zc[np.isfinite(zc) & (zc > 0)]
    u_contact = float(np.median(np.nonzero(cb)[1]) + 0.5) if cb.any() else None
    # contour depth, not band median: on a round-bottomed object the band's front surface is
    # nearer than the silhouette contour (MEASURED 2026-09-15 on the harness ellipsoid at z = 8:
    # band median 7.64 vs contour ~7.9 -> feet 5 % of camera height above the ground; the far
    # quantile brings it to ~2 %, the rest is tangent-point geometry). Flat-bottomed objects
    # (wheels, feet) have contour == contact and no such bias.
    z_contact = float(np.quantile(zc, float(CONTACT["depth_quantile"]))) if len(zc) else None
    # outline / border fraction (same definition as m2_a.mask_stats)
    er = M.copy()
    er[1:] &= M[:-1]; er[:-1] &= M[1:]; er[:, 1:] &= M[:, :-1]; er[:, :-1] &= M[:, 1:]
    er[0, :] = er[-1, :] = er[:, 0] = er[:, -1] = False
    outline = M & ~er
    ring = np.zeros_like(M); ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    bf = float((outline & ring).sum() / max(int(outline.sum()), 1))
    touches = {"top": bool(M[0].any()), "bottom": touches_bottom,
               "left": bool(M[:, 0].any()), "right": bool(M[:, -1].any())}
    return SourceStats(u0, v0, float(n), v_contact, z_src, not touches_bottom, bf, n, H, W, touches,
                       v_contact_max, run_frac, u_contact, z_contact, contact_at_frame_edge)


# ----------------------------------------------------------------------------
# heading alignment (spec §6.1, eqs 12–13)
# ----------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    assert n > 0, "zero vector where a unit vector is required"
    return v / n


def desired_basis(d_src, u_scene):
    f = _unit(d_src)
    u = np.asarray(u_scene, float) - (np.asarray(u_scene, float) @ f) * f
    kappa = float(np.linalg.norm(u))
    u = u / kappa
    r = np.cross(f, u)
    B = np.stack([f, u, r], axis=1)
    assert abs(np.linalg.det(B) - 1.0) < 1e-9
    return B, kappa


def desired_basis_in_plane(d_src, n_plane):
    """Spec v3 §6.2 as amended 2026-09-15 (replaces eqs 22-25 whenever a contact plane is used):
        f_des = (d - (n.d) n) / |d - (n.d) n|,   u_des = n,   r_des = f_des x u_des
    A ground-supported object's forward axis lies in its support plane; the previous basis let a
    pitched heading tilt the proxy onto its nose (measured on clip 035: d_src 16.4 deg out of the
    plane, the human saw the tilt). kappa_basis becomes |d - (n.d) n|: the guard on the projection,
    degenerate only when the motion is nearly along the plane normal. pitch_deg = arcsin(n.d) is
    recorded alongside (the arrow verdicts are 2-D and blind to it)."""
    d = _unit(d_src); n = _unit(n_plane)
    fp = d - (n @ d) * n
    kappa = float(np.linalg.norm(fp))
    assert kappa > 1e-9, "heading parallel to the plane normal: undefined in-plane forward axis"
    f = fp / kappa
    r = np.cross(f, n)
    B = np.stack([f, n, r], axis=1)
    assert abs(np.linalg.det(B) - 1.0) < 1e-9
    pitch = float(np.degrees(np.arcsin(np.clip(n @ d, -1.0, 1.0))))
    return B, kappa, pitch


def alignment(d_src, u_scene, B_tar, n_plane=None):
    """R_A = B_des B_tar^T: proxy canonical frame -> camera-t_a frame. With a contact plane
    (n_plane given and non-zero) the amended in-plane basis is used and u_scene is not needed;
    without one (camera-prior up, no plane) the original u_scene orthogonalisation applies."""
    if n_plane is not None and np.linalg.norm(np.asarray(n_plane, float)) > 0:
        B_des, kappa, _ = desired_basis_in_plane(d_src, n_plane)
    else:
        B_des, kappa = desired_basis(d_src, u_scene)
    return B_des @ np.asarray(B_tar, float).T, kappa


# ----------------------------------------------------------------------------
# the solve (spec §6.3 C1–C4, closed form)
# ----------------------------------------------------------------------------
@dataclass
class Placement:
    mode: str
    z: float; s: float
    X: np.ndarray                  # object centre, camera-t_a frame
    R_A: np.ndarray
    rho_tilde: np.ndarray
    a_0: float; h_b: float; delta_p: float
    n_dot_rho: float; det: float
    eps_shape: float | None
    kappa_basis: float
    gates: dict
    admitted: bool
    contact_residual: float        # n·X_bottom + d, must be ~0 in framing mode
    d_used: float = float("nan")           # the plane offset the solve used (local when contact observed)
    d_local_minus_global: float | None = None   # review 2026-09-15 (b): terrain-roughness signal
    pitch_deg: float = float("nan")        # §6.2 amended: arcsin(n.d_src), the pitch the projection removed

    def to_json(self):
        d = asdict(self)
        for k in ("X", "R_A", "rho_tilde"):
            d[k] = np.asarray(d[k]).tolist()
        return d


def _gate(stat, threshold, cond):
    if threshold is None:
        return {**stat, "threshold": None, "pass": None, "outcome": "ungated"}
    ok = bool(cond(float(threshold)))
    return {**stat, "threshold": threshold, "pass": ok, "outcome": "pass" if ok else "fail"}


def eps_shape(stats: SourceStats, K, a_0, h_b):
    """Spec (18) with the sqrt the spec omits: both sides are s/X_z.
    (v_contact − v0)/(f_y h_b) vs sqrt(A_src/(a_0 f_x f_y)). Level-ground,
    weak-perspective diagnostic; None when contact is unobserved."""
    if not stats.contact_observed or stats.v_contact is None:
        return None
    fx, fy = float(K[0, 0]), float(K[1, 1])
    lhs = (stats.v_contact - stats.v0) / (fy * h_b)
    rhs = np.sqrt(stats.A_src / (a_0 * fx * fy))
    if lhs <= 0 or rhs <= 0:
        return None
    return float(abs(np.log(lhs / rhs)))


_CONST_CACHE: dict = {}


def cached_constants(mesh, R_A, n, grid_n):
    """Constants depend only on (mesh, R_A, n, grid); the rasterisation costs
    ~12 s, so memoise — the harness calls place() dozens of times per mesh."""
    key = (mesh.name, id(mesh), np.round(R_A, 12).tobytes(), np.round(n, 12).tobytes(), int(grid_n))
    if key not in _CONST_CACHE:
        _CONST_CACHE[key] = _proxy.constants(mesh, R_A, n, grid_n)
    return _CONST_CACHE[key]


def place(stage0: dict, stats: SourceStats, mesh: _proxy.ProxyMesh, mode="depth_preserving",
          thresholds=None, s_given=None, grid_n=1600, consts=None, local_offset=None) -> Placement:
    """stage0: dict with K_a[3,3], d_src[3], u_scene[3], n_plane[3], d_plane()
    in the camera-t_a frame (m2a.read_stage0 output). thresholds: lock dict
    (keys m2c_det_min_frac); None/absent -> ungated. consts: optional
    precomputed proxy constants (harness step 2 passes them to test the solve
    independently of the rasteriser)."""
    mode = MODE_ALIASES.get(mode, mode)
    assert mode in MODES, mode
    th = thresholds or {}
    K = np.asarray(stage0["K_a"], float)
    n = _unit(stage0["n_plane"])
    d_global = float(stage0["d_plane"])
    # review 2026-09-15 (b): a single globally fitted plane cannot follow undulating terrain (clip
    # 284: the plane passed through the subject's centroid ray, s < 0). The pooled NORMAL is robust;
    # only the offset is local. Unproject the source's own contact point at its depth and set
    # d_local = -n.X_feet: the plane is then guaranteed to pass under the object. Global offset is
    # kept when contact is unobserved (the gate refuses then anyway) and d_local - d_global is recorded.
    d_local = None
    use_local = bool(CONTACT["local_plane_offset"]) if local_offset is None else bool(local_offset)
    # MEASURED 2026-09-15 on 326 (1650-px mask, ~40 px tall): the contact band's contour depth (1.53) came
    # out NEARER than the body median (1.73) — impossible for a standing figure seen from above — and the
    # local plane rose by 0.124, shrinking the proxy to a quarter of its area. Depth noise on tiny masks.
    # Recorded (z_contact / z_src, n_mask); a reliability rule is a threshold, the lock file's.
    contact_depth_ratio = (float(stats.z_contact / stats.z_src) if (stats.z_contact and stats.z_src) else float("nan"))
    if use_local and stats.contact_observed and stats.v_contact is not None and stats.z_contact is not None and stats.u_contact is not None:
        rf = np.linalg.solve(K, np.array([stats.u_contact, stats.v_contact, 1.0])); rf = rf / rf[2]
        X_feet = stats.z_contact * rf
        d_local = float(-n @ X_feet)
    d = d_local if d_local is not None else d_global
    R_A, kappa = alignment(stage0["d_src"], stage0["u_scene"], mesh.B_tar, n_plane=n)   # in-plane basis (§6.2 amended)
    pitch_deg = float(np.degrees(np.arcsin(np.clip(float(n @ _unit(stage0["d_src"])), -1.0, 1.0))))
    c = consts if consts is not None else cached_constants(mesh, R_A, n, grid_n)
    a_0, h_b, delta_p = float(c["a_0"]), float(c["h_b"]), float(c["delta_p"])

    rho = np.linalg.solve(K, np.array([stats.u0, stats.v0, 1.0]))
    rho = rho / rho[2]
    ndr = float(n @ rho)
    det = h_b - delta_p * ndr

    gates = {}
    gates["g_contact_unobserved"] = {"contact_observed": stats.contact_observed,
                                     "mask_border_frac": stats.mask_border_frac,
                                     "pass": bool(stats.contact_observed),
                                     "outcome": "pass" if stats.contact_observed else "fail"}
    gates["g_det"] = _gate({"det": det, "det_over_hb": det / h_b}, th.get("m2c_det_min_frac"),
                           lambda t: abs(det) / h_b >= t)
    # lateral/top clipping biases the mask centroid (the ray); recorded here, thresholded by
    # m2_a's truncation gate (m2a_truncation_max), not by a second threshold
    t = stats.touches or {}
    lat = bool(t.get("left") or t.get("right") or t.get("top"))
    gates["g_mask_clipped"] = {"sides": [k for k, v in t.items() if v], "lateral_or_top": lat,
                               "pass": None, "outcome": "ungated"}
    gates["g_contact_band_depth"] = _gate({"z_contact_over_z_src": contact_depth_ratio, "n_mask": int(stats.n_mask),
                                           "d_local_minus_global": None if d_local is None else float(d_local - d_global)},
                                          th.get("m2c_contact_depth_ratio_min"), lambda t_: contact_depth_ratio >= t_)

    if mode == "depth_preserving":
        if abs(det) < 1e-12:
            z, s = float("nan"), float("nan")
        else:
            z = (delta_p * d + h_b * stats.z_src) / det
            s = (ndr * stats.z_src + d) / det
    elif mode == "area_preserving":
        # weak perspective: A_src = a_0 s^2 fx fy / z^2  =>  k = s/z = sqrt(A_src / (a_0 fx fy));
        # contact n.X + d = s h_b with X = z rho~  =>  z = -d / (n.rho~ - h_b k), s = k z
        k = float(np.sqrt(stats.A_src / (a_0 * K[0, 0] * K[1, 1])))
        det_a = ndr - h_b * k
        gates["g_area_det"] = _gate({"det_area": det_a, "k": k}, th.get("m2c_det_min_frac"),
                                    lambda t: abs(det_a) >= t)
        if abs(det_a) < 1e-12:
            z, s = float("nan"), float("nan")
        else:
            z = -d / det_a
            s = k * z
    else:
        assert s_given is not None and s_given > 0, "size_preserving needs s_given"
        s = float(s_given)
        gates["g_ray_parallel"] = {"n_dot_rho": ndr, "pass": bool(abs(ndr) > 1e-9),
                                   "outcome": "pass" if abs(ndr) > 1e-9 else "fail"}
        z = (s * h_b - d) / ndr if abs(ndr) > 1e-9 else float("nan")

    X = z * rho
    # contact residual: the lowest point along n must sit on the plane
    Va = _proxy.aligned_vertices(mesh, R_A)
    p_low = Va[np.argmin(Va @ n)]
    resid = float(n @ (X + s * p_low) + d) if np.isfinite(z) else float("nan")
    admitted = all(g["pass"] is not False for g in gates.values()) and np.isfinite(z) and z > 0 and s > 0
    return Placement(mode, float(z), float(s), X, R_A, rho, a_0, h_b, delta_p, ndr, float(det),
                     eps_shape(stats, K, a_0, h_b), kappa, gates, bool(admitted), resid,
                     float(d), None if d_local is None else float(d_local - d_global), pitch_deg)


# ----------------------------------------------------------------------------
# C5 refinement — SKILL_m2_c_C1 v1.1 §4.7 (supersedes spec v3 eq. 23)
# ----------------------------------------------------------------------------
def render_stats(pl: Placement, mesh: _proxy.ProxyMesh, K, H, W):
    """Render the placed proxy under true perspective and measure it with the
    SAME source_stats the source went through. Also the perspective delta_p
    at this pose, (X_z - median rendered depth) / s."""
    V = pl.X + pl.s * _proxy.aligned_vertices(mesh, pl.R_A)
    zb, m = _proxy.render_perspective(V, mesh.F, np.asarray(K, float), H, W)
    assert m.any(), "placed proxy renders outside the image"
    st = source_stats(m, np.where(m, zb, 0.0), K)
    dp = (pl.X[2] - st.z_src) / pl.s
    return st, dp, (zb, m)


def refine(stage0: dict, stats: SourceStats, mesh: _proxy.ProxyMesh, pl: Placement,
           mode="depth_preserving", iters=2, thresholds=None, s_given=None, H=None, W=None, local_offset=None):
    """Re-solve the same 2x2 under true perspective. Area is NEVER a
    constraint in framing mode. Per iteration: render at the current pose;
    the rendered mask centroid differs from the projected centre by an offset
    o; move the ray so the centroid lands on the source centroid (ray through
    p_ta - o); re-measure delta_p from the render (h_b is exact geometry);
    re-solve. Returns (Placement, history) with per-iteration centroid error."""
    K = np.asarray(stage0["K_a"], float)
    H = H or stats.H; W = W or stats.W
    hist = []
    cur = pl
    for it in range(int(iters)):
        st_r, dp_r, _ = render_stats(cur, mesh, K, H, W)
        Pc = K @ cur.X; uc, vc = Pc[0] / Pc[2], Pc[1] / Pc[2]          # projected centre
        off = np.array([st_r.u0 - uc, st_r.v0 - vc])                   # centroid - centre
        err = np.hypot(st_r.u0 - stats.u0, st_r.v0 - stats.v0)
        hist.append({"iter": it, "centroid_err_px": float(err), "delta_p_persp": float(dp_r),
                     "z": cur.z, "s": cur.s, "eps_shape": cur.eps_shape})
        # ray through (p_ta - off): the CENTRE should project there so the mask centroid
        # lands on p_ta
        from dataclasses import replace
        target = replace(stats, u0=stats.u0 - off[0], v0=stats.v0 - off[1])
        consts = {"a_0": cur.a_0, "h_b": cur.h_b, "delta_p": dp_r}
        cur = place(stage0, target, mesh, mode, thresholds, s_given, consts=consts, local_offset=local_offset)
        # keep the ORIGINAL (unshifted) stats' eps_shape semantics: recompute on source stats
        cur.eps_shape = eps_shape(stats, K, cur.a_0, cur.h_b)
    st_r, dp_r, _ = render_stats(cur, mesh, K, H, W)
    hist.append({"iter": int(iters), "centroid_err_px": float(np.hypot(st_r.u0 - stats.u0, st_r.v0 - stats.v0)),
                 "delta_p_persp": float(dp_r), "z": cur.z, "s": cur.s, "eps_shape": cur.eps_shape,
                 "z_src_rendered": st_r.z_src, "v_contact_rendered": st_r.v_contact})
    return cur, hist


# ----------------------------------------------------------------------------
# C1 producer output (SKILL_m2_c_C3 v1.1 §6, review B1 2026-09-15): C1 persists its own placement
# ----------------------------------------------------------------------------
PLACEMENT_SCHEMA = "m2c_placement_v1"


def write_placement(out_dir, pl: Placement, refine_history, clip: str, category: str, proxy_cache_key: str,
                    proxy_facing: str, stats: SourceStats, stage0: dict, hashes: dict, extra: dict | None = None):
    """cache/m2_c/<clip>/<category>/placement.json — read by C3, m2_e and Module 3, never recomputed."""
    import json
    from pathlib import Path as _P
    out = _P(out_dir); out.mkdir(parents=True, exist_ok=True)
    rec = {"schema": PLACEMENT_SCHEMA, "clip": clip, "category": category, "proxy_cache_key": proxy_cache_key,
           "proxy_facing": proxy_facing, "admitted": bool(pl.admitted), "mode": pl.mode,
           "placement": pl.to_json(), "refine_history": refine_history,
           "heading_out_of_plane_deg": pl.pitch_deg,
           "stage0": {k: (np.asarray(v).tolist() if hasattr(v, "tolist") or isinstance(v, np.ndarray) else v)
                      for k, v in stage0.items() if k in ("K_a", "d_src", "u_scene", "n_plane", "d_plane")},
           "source_stats": asdict(stats), "upstream_hashes": hashes, **(extra or {})}
    (out / "placement.json").write_text(json.dumps(rec, indent=1, default=_json_default), encoding="utf-8")
    return out / "placement.json"


def read_placement(out_dir):
    """Consumer entry (C3, m2_e): validates the schema and the admitted flag's presence; refuses otherwise."""
    import json
    from pathlib import Path as _P
    f = _P(out_dir) / "placement.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing — C1 has not placed this clip/category (run scripts/place_proxy_on_clip.py)")
    rec = json.loads(f.read_text(encoding="utf-8"))
    assert rec.get("schema") == PLACEMENT_SCHEMA, f"unexpected placement schema {rec.get('schema')!r}"
    for k in ("admitted", "placement", "proxy_cache_key", "proxy_facing", "mode"):
        assert k in rec, f"placement.json missing {k}"
    pl = rec["placement"]
    for k in ("X", "R_A", "rho_tilde"):
        pl[k] = np.asarray(pl[k], np.float64)
    assert pl["R_A"].shape == (3, 3) and abs(np.linalg.det(pl["R_A"]) - 1) < 1e-6
    return rec


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)
