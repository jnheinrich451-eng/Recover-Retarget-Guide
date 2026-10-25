"""m2_e — the residual alignment A = R_u(delta) (SKILL m2_e v2.2 §5).

Vehicles (Objects): the pose head's decomposition is unreliable (P3), so delta is a RESTRICTED SILHOUETTE SCAN: rotate the aligned
proxy about the plane normal through its placed centroid, R_A(theta) = R_n(theta) R_A with X fixed, render under K_1024 with C3's
renderer, IoU against the ANCHOR MASK (never G_tar as the primary), theta in [-45, +45] deg at 1 deg steps, parabolic refinement of
the peak; the theta = 0 point must reproduce m2_d's obj_fidelity (same render). Positive theta about the UP normal.
Humans (Body): from the keypoints, f = l x u with l the mean unit left-minus-right shoulder/hip vector and u = (0, -1, 0) — the
working cross-product order validated 5/5 against the human (measurements/body_facing/); delta = signed angle in the ground plane
between that facing and B_des's forward column, R_A @ B_tar[:, 0]. No undo of the square-pixel rescale (review item 7).
"""
from __future__ import annotations

import numpy as np

from ..m2c import compose as _compose

U_IMAGE = np.array([0.0, -1.0, 0.0])                                     # OpenCV y points down: "up" is -y
# mhr70 indices read from sam_3d_body.metadata.mhr70.mhr_names at probe time (2026-09-17); asserted against the names at runtime
MHR70 = {"left-shoulder": 5, "right-shoulder": 6, "left-hip": 9, "right-hip": 10}


def rot_about(n, theta_deg: float) -> np.ndarray:
    n = np.asarray(n, np.float64); n = n / np.linalg.norm(n); th = np.radians(theta_deg)
    Kx = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def iou(a, b) -> float:
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    return float((a & b).sum() / max((a | b).sum(), 1))


def signed_yaw_deg(v_from, v_to, n) -> float:
    """Angle from v_from to v_to about the normal n (positive = the rotation rot_about(n, +theta) would apply), in the plane ⊥ n."""
    n = np.asarray(n, np.float64); n = n / np.linalg.norm(n)
    a = np.asarray(v_from, np.float64); b = np.asarray(v_to, np.float64)
    a = a - (a @ n) * n; b = b - (b @ n) * n
    assert np.linalg.norm(a) > 1e-9 and np.linalg.norm(b) > 1e-9, "a vector has no component in the ground plane"
    return float(np.degrees(np.arctan2(np.cross(a, b) @ n, a @ b)))


def yaw_scan(V_norm, F, X, s, R_A, n_plane, K, anchor_mask, scan_deg=45.0, step_deg=1.0, tol=0.01, refine=True) -> dict:
    """Restricted silhouette scan. Returns delta_deg, the curve, its width and the IoU at theta = 0."""
    H, W = np.asarray(anchor_mask).shape
    thetas = np.arange(-scan_deg, scan_deg + 1e-9, step_deg)
    ious = []
    for th in thetas:
        _, mk = _compose.render_object(V_norm, F, X, s, rot_about(n_plane, th) @ np.asarray(R_A, np.float64), K, H, W)
        ious.append(iou(mk, anchor_mask))
    ious = np.array(ious); k = int(np.argmax(ious)); peak = float(ious[k]); delta = float(thetas[k])
    if refine and 0 < k < len(thetas) - 1:                                  # parabolic refinement of the peak (review item 6)
        y0, y1, y2 = ious[k - 1], ious[k], ious[k + 1]; den = (y0 - 2 * y1 + y2)
        if den < 0:
            delta = float(thetas[k] + 0.5 * (y0 - y2) / den * step_deg)
    above = np.nonzero(ious >= peak - tol)[0]
    width = float(thetas[above.max()] - thetas[above.min()]) if len(above) else 0.0
    at_edge = bool(k == 0 or k == len(thetas) - 1)
    i0 = int(np.argmin(np.abs(thetas)))
    return {"delta_deg": delta, "iou_peak": peak, "iou_at_zero": float(ious[i0]), "curve": [[float(t), float(v)] for t, v in zip(thetas, ious)],
            "curve_width_deg": width, "peak_at_scan_edge": at_edge, "scan_deg": float(scan_deg), "step_deg": float(step_deg), "tol": float(tol),
            "instrument": "silhouette_scan"}


def body_facing(kp3d_cam, idx: dict = MHR70) -> tuple[np.ndarray, dict]:
    """f = l x u (validated order). Also the shoulder/hip angle as the twist check the probe used."""
    kp = np.asarray(kp3d_cam, np.float64); assert kp.ndim == 2 and kp.shape[1] == 3 and kp.shape[0] > max(idx.values())
    unit = lambda v: v / np.linalg.norm(v)                                  # noqa: E731
    l_sh = unit(kp[idx["left-shoulder"]] - kp[idx["right-shoulder"]]); l_hip = unit(kp[idx["left-hip"]] - kp[idx["right-hip"]])
    l = unit(l_sh + l_hip)
    f = unit(np.cross(l, U_IMAGE))
    twist = float(np.degrees(np.arccos(np.clip(l_sh @ l_hip, -1, 1))))
    return f, {"shoulder_hip_angle_deg": twist, "cross_order": "l x u", "instrument": "keypoints"}


def body_delta(kp3d_cam, R_A, B_tar, n_plane, idx: dict = MHR70) -> dict:
    f, info = body_facing(kp3d_cam, idx)
    forward = np.asarray(R_A, np.float64) @ np.asarray(B_tar, np.float64)[:, 0]      # B_des's forward column
    d = signed_yaw_deg(forward, f, n_plane)
    return {"delta_deg": d, "facing_cam": f.tolist(), "forward_des_cam": forward.tolist(), **info}


DELTA_DEFINITION = "delta = yaw of the ANCHOR minus yaw of the POSED PROXY, positive about the UP plane normal (rot_about(n, +delta) turns the proxy onto the anchor); shared by the scan and the keypoint instrument"


def paired_scan(V_norm, F, X, s, R_A, n_plane, K, anchor_mask, **kw) -> dict:
    """Reviewer 2026-09-23: a flipped anchor scores a plausible, quiet delta (test 5: +11.1 deg at IoU 0.905, inside the range of
    correct real anchors 0.63-0.97), so the absolute peak is not a tell. Scan the proxy as posed AND rotated 180 deg against the
    same anchor mask; the generator's shape deviation hits both alike, what remains is the proxy's own front/back asymmetry.
    margin = peak(posed) - peak(flipped): positive = posed wins; small = ambiguous pair, recorded rather than resolved."""
    posed = yaw_scan(V_norm, F, X, s, R_A, n_plane, K, anchor_mask, **kw)
    flipped = yaw_scan(V_norm, F, X, s, rot_about(n_plane, 180.0) @ np.asarray(R_A, np.float64), n_plane, K, anchor_mask, **kw)
    return {**posed, "flipped_peak_iou": flipped["iou_peak"], "flipped_delta_deg": flipped["delta_deg"], "front_back_margin": posed["iou_peak"] - flipped["iou_peak"],
            "flipped_curve": flipped["curve"]}


def gauge_align(depth_body, depth_anchor, region, mode: str = "median_ratio") -> dict:
    """Post-R2 gauge (SKILL v2.2 §1): bring a mesh in the vendor's gauge into the source gauge over the region, no rotation.
    mode "median_ratio" (reviewer 2026-09-27): k = median(depth_body) / median(depth_anchor), t = 0 — a least-squares depth
    regression would fit its slope to the flattened band-alpha depth and shrink the mesh by the flattening; the ratio of medians
    converts the gauge without absorbing within-object scale. The regression (k_reg, t_reg, residual) is recorded alongside."""
    reg = np.asarray(region, bool) & np.isfinite(depth_body) & np.isfinite(depth_anchor) & (np.asarray(depth_body) > 0) & (np.asarray(depth_anchor) > 0)
    assert reg.sum() > 100, "gauge alignment: no overlap"
    x = np.asarray(depth_anchor, np.float64)[reg]; y = np.asarray(depth_body, np.float64)[reg]
    (k_reg, t_reg), *_ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)
    resid_reg = float(np.sqrt(((y - (k_reg * x + t_reg)) ** 2).mean()))
    k_med = float(np.median(y) / np.median(x))
    k, tt = (k_med, 0.0) if mode == "median_ratio" else (float(k_reg), float(t_reg))
    # R1 (reviewer 2026-10-19b): the assertion is on the scale the mode USES. k_reg is recorded with its conditioning: over one subject
    # the anchor depth spans a few tenths of a unit, close to its noise, so the slope's sign is near random and is a flag, not a refusal.
    assert k > 0, f"gauge alignment: non-positive scale ({mode}: k {k:.4f})"
    xr = x - x.mean(); sxx = float((xr ** 2).sum()); n_ = int(reg.sum())
    se = float(np.sqrt(((y - (k_reg * x + t_reg)) ** 2).sum() / max(n_ - 2, 1) / sxx)) if sxx > 0 else float("inf")
    lo, hi = float(np.percentile(x, 5)), float(np.percentile(x, 95))
    return {"mode": mode, "scale_k": float(k), "shift_t": float(tt), "n_pixels": n_, "k_median_ratio": k_med, "k_regression": float(k_reg), "t_regression": float(t_reg),
            "resid_rms_regression": resid_reg, "resid_rms_used": float(np.sqrt(((y - (k * x + tt)) ** 2).mean())), "median_depth_body": float(np.median(y)), "median_depth_anchor": float(np.median(x)),
            "k_regression_nonpositive": bool(k_reg <= 0), "k_regression_se": se, "k_regression_ci95": [float(k_reg - 1.96 * se), float(k_reg + 1.96 * se)],
            "anchor_depth_range_p5_p95": [lo, hi], "anchor_depth_range_rel": float((hi - lo) / np.median(x)),
            "conditioning_note": "k_reg is a record: its sign is informative only when the anchor depth range over the subject is large against the depth noise"}


def flatten_predicted_yaw(f: float, heading_az_deg: float) -> dict:
    """Reviewer 2026-09-27: compressing an object's disparity by f about its mean rotates every surface normal toward the optical
    axis; a face at angle theta from the axis ends at tan(theta') = f tan(theta), Delta = theta - arctan(f tan theta). The dominant
    face is the one nearer the axis: front/rear (normal along the heading) or side (normal across it). Direction: the object turns
    toward the square view of that face — heading azimuth toward the nearest of {+-90} (frontal) or {0, 180} (side). Sign in the
    scan's convention (+ = rot_about(up, +), az increasing)."""
    az = float(heading_az_deg); h = np.array([np.cos(np.radians(az)), 0.0, np.sin(np.radians(az))])
    th_fr = float(np.degrees(np.arccos(min(1.0, abs(h[2])))))                # front/rear face normal along the heading
    th_side = float(np.degrees(np.arccos(min(1.0, abs(h[0])))))              # side face normal across it
    face, th, targets = ("frontal", th_fr, (-90.0, 90.0)) if th_fr <= th_side else ("side", th_side, (0.0, 180.0, -180.0))
    mag = th - float(np.degrees(np.arctan(f * np.tan(np.radians(th)))))
    wrap = lambda d: (d + 180.0) % 360.0 - 180.0                              # noqa: E731
    tgt = min(targets, key=lambda a_: abs(wrap(a_ - az))); sgn = 1.0 if wrap(tgt - az) > 0 else -1.0
    return {"f": float(f), "dominant_face": face, "face_angle_from_axis_deg": th, "predicted_abs_deg": mag, "predicted_signed_deg": sgn * mag,
            "toward": f"square {face}", "heading_az_deg": az}



def apply_gauge(V, k: float, t: float) -> np.ndarray:
    """Vertices in Body's gauge -> source gauge along their own rays: z' = (z - t) / k, x' = x z'/z, y' = y z'/z."""
    V = np.asarray(V, np.float64); z = V[:, 2]; zp = (z - t) / k
    return np.stack([V[:, 0] * zp / z, V[:, 1] * zp / z, zp], 1)


def major_axis_yaw(V, n_plane, ref_axis=None) -> dict:
    """Yaw of a vertex set's MAJOR AXIS in the ground plane (reviewer 2026-09-27: orientation from shape, not from silhouette
    overlap). The axis has a 180-degree ambiguity; it is resolved toward ref_axis when given. Returns the axis and the anisotropy
    (major / minor eigenvalue ratio: near 1 = the axis is undetermined, e.g. a standing human seen from above)."""
    n = np.asarray(n_plane, np.float64); n = n / np.linalg.norm(n)
    V = np.asarray(V, np.float64); P = V - (V @ n)[:, None] * n
    Pc = P - P.mean(0); C = Pc.T @ Pc / len(Pc)
    w, U = np.linalg.eigh(C); order = np.argsort(w)[::-1]; ax = U[:, order[0]]; ax = ax - (ax @ n) * n; ax /= np.linalg.norm(ax)
    if ref_axis is not None and ax @ np.asarray(ref_axis, np.float64) < 0:
        ax = -ax
    return {"axis": ax.tolist(), "anisotropy": float(w[order[0]] / max(w[order[1]], 1e-12))}


def gtar_yaw_vs_proxy(V_gtar, V_proxy_posed, n_plane) -> dict:
    """Signed yaw from the posed proxy's major axis to G_tar's, about the up normal; the 180 ambiguity resolved to the nearest.
    Pre-registered discriminator: if this measures orientation, its deviation does NOT track R2 IoU."""
    a_p = major_axis_yaw(V_proxy_posed, n_plane); a_g = major_axis_yaw(V_gtar, n_plane, ref_axis=a_p["axis"])
    return {"delta_gtar_shape_deg": signed_yaw_deg(a_p["axis"], a_g["axis"], n_plane), "proxy_anisotropy": a_p["anisotropy"], "gtar_anisotropy": a_g["anisotropy"],
            "instrument": "major_axis_in_ground_plane"}


def principal_extents(V) -> dict:
    """Extents along the principal axes (2 x sqrt of the covariance eigenvalues, largest first: length, width, height in the
    body's own frame). Ratios are shape, independent of pose and scale; a reconstruction shallower than its proxy along one axis
    shows as a smaller ratio there (reviewer 2026-09-28: whether the depth compression reaches G_tar's shape, not only its placement)."""
    V = np.asarray(V, np.float64); C = np.cov((V - V.mean(0)).T); w = np.sort(np.linalg.eigvalsh(C))[::-1]
    e = 2.0 * np.sqrt(np.maximum(w, 0))
    return {"extents": e.tolist(), "ratios_to_largest": (e / max(e[0], 1e-12)).tolist(), "width_over_length": float(e[1] / max(e[0], 1e-12)), "height_over_length": float(e[2] / max(e[0], 1e-12))}
