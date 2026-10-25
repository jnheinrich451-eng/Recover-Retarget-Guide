"""m2_c / C3 — hybrid control composition (SKILL_m2_c_C3_composition.md v1.1).

Domain ledger (§1): C1 quantities and the object render are DEPTH; C2's coarse field and
everything C3 emits are DISPARITY ζ = 1/z. C3 converts exactly once — the object render —
and composes with a `max` (nearer is larger in disparity; ζ_obj = 0 outside the mask makes
occlusion fall out). Invalid background (sky) is ζ = 0, the honest value, excluded from the
percentiles. The gauge dies at normalisation; `zeta_ctrl` is retained for m2_e's α.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import proxy as _proxy
from . import sanitise as _san

CONTROL_SCHEMA = "m2c_control_v1"
SKY_REASON = "border_component_no_enclosing_ring"


def _c3_params():
    import yaml
    cfg = yaml.safe_load(Path("configs/m2c.yaml").read_text(encoding="utf-8"))["m2c"]
    return {**cfg["c3"], "c2_border_band_px_256": cfg["c2_border_band_px_256"]}


# ----------------------------------------------------------------------------
# step 1: object render at 1024 under K_1024, depth-sign guard
# ----------------------------------------------------------------------------
def render_object(V_norm, F, X, s, R_A, K, H, W):
    """Pose the normalised proxy with C1's placement and z-buffer it (depth, camera z)."""
    mesh = _proxy.ProxyMesh("proxy", np.asarray(V_norm, np.float64), np.asarray(F, int),
                            np.asarray(V_norm, np.float64).mean(0), _proxy.CANON_B_TAR.copy())
    V = np.asarray(X, np.float64) + float(s) * _proxy.aligned_vertices(mesh, np.asarray(R_A, np.float64))
    zb, mk = _proxy.render_perspective(V, mesh.F, np.asarray(K, np.float64), H, W)
    return zb, mk


def depth_sign_gate(s, z_obj, M_obj):
    """§1/§7 g_depth_sign: a negative scale inverts the mesh (measured: clip 284 before the C1
    fixes, s = −0.11) and 1/z on non-positive depths is silent nonsense. Refuse."""
    zmin = float(z_obj[M_obj].min()) if M_obj.any() else float("nan")
    ok = bool(s > 0 and M_obj.any() and np.isfinite(zmin) and zmin > 0)
    return {"s": float(s), "z_obj_min": zmin, "pass": ok, "outcome": "pass" if ok else "fail"}


def object_disparity(z_obj, M_obj):
    """ζ_obj = 1/z on the mask, 0 elsewhere (0 disparity = infinitely far)."""
    zeta = np.zeros(z_obj.shape, np.float64)
    zeta[M_obj] = 1.0 / z_obj[M_obj]
    assert np.isfinite(zeta).all()
    return zeta


# ----------------------------------------------------------------------------
# steps 2-4: background upsample (C2's kernel), composition, invalid handling
# ----------------------------------------------------------------------------
def background_full(zeta_coarse, valid_coarse, H, W):
    """32 → H×W directly with C2's Keys a = −0.5 resampler (never cv2: a = −0.75 breaks the
    affine field C2 verified); validity through the same resampler as a float, thresholded at
    0.5 as C2's leakage score does. Invalid → ζ = 0 (sky)."""
    zb = _san.upsample_bicubic(np.asarray(zeta_coarse, np.float64), H, W)
    vb = _san.upsample_bicubic(np.asarray(valid_coarse, np.float64), H, W) > 0.5
    zb = np.where(vb, zb, 0.0)
    assert np.isfinite(zb).all()
    return zb, vb


def invalid_all_sky(record_c2) -> bool:
    """C2 tags every unfilled component with a reason; sky is the only accepted one."""
    comps = (record_c2 or {}).get("components", []) or []
    unfilled = [c for c in comps if c.get("filled") is False]
    return all(c.get("reason") == SKY_REASON for c in unfilled)


def compose(zeta_bg, valid_bg, zeta_obj, M_obj, fill_mask=None):
    """ζ_ctrl = max(ζ_bg, ζ_obj). Occlusion: object pixels where the background won. With `fill_mask` (R2, reviewer 2026-10-19b): a
    FILL pixel (inpainted over the removed subject, never observed) is an estimate of what lies behind the subject and never occludes
    the proxy: occludes(p) = [ζ_bg > ζ_obj] and p not in the fill. The fill's depth values are left alone; only the test changes."""
    zeta_ctrl = np.maximum(zeta_bg, zeta_obj)
    valid = valid_bg | M_obj
    occluded = M_obj & (zeta_bg > zeta_obj)
    if fill_mask is not None:
        occluded &= ~np.asarray(fill_mask, bool)
    return zeta_ctrl, valid, occluded


# ----------------------------------------------------------------------------
# step 5: normalisation, both polarities
# ----------------------------------------------------------------------------
def normalise(zeta_ctrl, valid, p_low, p_high):
    vals = zeta_ctrl[valid]
    assert vals.size > 0, "no valid pixels to normalise over"
    q_low, q_high = np.percentile(vals, [100 * p_low, 100 * p_high])
    assert q_high > q_low, (q_low, q_high)
    D = np.clip((zeta_ctrl - q_low) / (q_high - q_low), 0.0, 1.0)
    return D, float(q_low), float(q_high)


# ----------------------------------------------------------------------------
# step 6: contact continuity
# ----------------------------------------------------------------------------
def contact_step(zeta_obj, zeta_bg, M_obj, band_rows):
    """ζ_obj − ζ_bg over the object's lowest `band_rows` rows: the two surfaces should meet."""
    rows = np.nonzero(M_obj.any(1))[0]
    if len(rows) == 0:
        return float("nan"), int(-1)
    v_bottom = int(rows.max())
    band = M_obj & (np.arange(M_obj.shape[0])[:, None] > v_bottom - band_rows)
    return float(np.median(zeta_obj[band] - zeta_bg[band])), v_bottom


def contact_quantum(n_plane, d_plane, K, cell_px):
    """One coarse-cell step of the ground's disparity along v: |β| · cell (SKILL §8 test 4)."""
    a, b, g = _san.affine_from_plane(n_plane, d_plane, K)
    return float(abs(b) * cell_px)


# ----------------------------------------------------------------------------
# the stage: placement + proxy + C2 field -> control record
# ----------------------------------------------------------------------------
def compose_control(V_norm, F, placement_rec, zeta_coarse, valid_coarse, record_c2, K_256, n_plane, d_plane,
                    params=None, fill_mask_256=None):
    p = params or _c3_params()
    H = W = int(p["render_px"]); N = int(p["coarse_n"])
    K = np.asarray(K_256, np.float64).copy(); K[:2] *= H / 256.0            # diag(4,4,1) K_256
    pl = placement_rec["placement"]
    gates = {"g_placement_admitted": {"admitted": bool(placement_rec["admitted"]), "pass": bool(placement_rec["admitted"]),
                                      "outcome": "pass" if placement_rec["admitted"] else "fail"}}
    qc = {"schema": CONTROL_SCHEMA, "clip": placement_rec.get("clip"), "category": placement_rec.get("category"),
          "proxy_cache_key": placement_rec["proxy_cache_key"], "proxy_facing": placement_rec["proxy_facing"],
          "render_px": H, "coarse_n": N, "K_1024": K.tolist(), "gates": gates}
    if not gates["g_placement_admitted"]["pass"]:
        qc["emitted"] = False
        return None, qc
    z_obj, M_obj = render_object(V_norm, F, pl["X"], pl["s"], pl["R_A"], K, H, W)
    gates["g_depth_sign"] = depth_sign_gate(pl["s"], z_obj, M_obj)
    if not gates["g_depth_sign"]["pass"]:
        qc["emitted"] = False
        return None, qc
    zeta_obj = object_disparity(z_obj, M_obj)
    zeta_bg, valid_bg = background_full(zeta_coarse, valid_coarse, H, W)
    all_sky = invalid_all_sky(record_c2)
    gates["g_invalid_reason"] = {"invalid_all_sky": all_sky, "invalid_frac_coarse": float(1 - np.mean(valid_coarse)),
                                 "pass": all_sky, "outcome": "pass" if all_sky else "fail"}
    if not all_sky:
        qc["emitted"] = False
        return None, qc
    fill_full = None
    if fill_mask_256 is not None:                                                  # C2's fill region (dilate(mask, r) at 256^2) at the control's resolution
        import cv2 as _cv2
        fill_full = _cv2.resize(np.asarray(fill_mask_256, np.uint8), (W, H), interpolation=_cv2.INTER_NEAREST).astype(bool)
    zeta_ctrl, valid, occluded = compose(zeta_bg, valid_bg, zeta_obj, M_obj, fill_full)
    occ_frac = float(occluded.sum() / max(M_obj.sum(), 1))
    qc["occlusion_rule"] = "scene_only (fill pixels never occlude; R2 2026-10-19b)" if fill_full is not None else "all_background (record of the six pairs)"
    if fill_full is not None:
        qc["occluded_by_fill_frac_under_old_rule"] = float((M_obj & (zeta_bg > zeta_obj) & fill_full).sum() / max(M_obj.sum(), 1))
    gates["g_obj_visible"] = {"obj_px": int(M_obj.sum()), "occluded_frac": occ_frac, "pass": bool(M_obj.any() and occ_frac < 1.0),
                              "outcome": "pass" if (M_obj.any() and occ_frac < 1.0) else "fail"}
    if not gates["g_obj_visible"]["pass"]:
        qc["emitted"] = False
        return None, qc
    # percentile domain (reviewer 2026-09-15, measured on 370: with the object included a close subject
    # saturates to a flat silhouette because q_high lands inside the scene): the percentiles map the
    # SCENE's depth range, so by default they are computed over background pixels only; both pairs are
    # recorded so the choice can be compared on real clips before it is locked
    # Revised 2026-09-15 on the six-clip data (265 vehicle: 56 % of the car clipped to flat white under
    # background-only; saturation detects large NEAR objects, not bad placements): the background sets
    # the floor, the object may extend the ceiling — q_low = background p_low, q_high = max(background
    # p_high, object p_high). Cost measured: a 13 % range compression on 265-vehicle's background,
    # none on 284 / 035 / 326. All variants stay recorded.
    dom = p.get("percentile_domain", "background_floor_object_ceiling")
    assert dom in ("background", "all", "background_floor_object_ceiling"), dom
    D_bg, ql_bg, qh_bg = normalise(zeta_ctrl, valid & ~M_obj, float(p["p_low"]), float(p["p_high"]))
    D_all, ql_all, qh_all = normalise(zeta_ctrl, valid, float(p["p_low"]), float(p["p_high"]))
    qh_obj = float(np.percentile(zeta_ctrl[M_obj], 100 * float(p["p_high"]))) if M_obj.any() else qh_bg
    ql_fc, qh_fc = ql_bg, max(qh_bg, qh_obj)
    D_fc = np.clip((zeta_ctrl - ql_fc) / (qh_fc - ql_fc), 0.0, 1.0)
    D, q_low, q_high = {"background": (D_bg, ql_bg, qh_bg), "all": (D_all, ql_all, qh_all),
                        "background_floor_object_ceiling": (D_fc, ql_fc, qh_fc)}[dom]
    obj_sat = float((D[M_obj] >= 1.0).mean()) if M_obj.any() else float("nan")
    obj_sat_all = float((D_all[M_obj] >= 1.0).mean()) if M_obj.any() else float("nan")
    obj_sat_bg = float((D_bg[M_obj] >= 1.0).mean()) if M_obj.any() else float("nan")
    obj_sat_fc = float((D_fc[M_obj] >= 1.0).mean()) if M_obj.any() else float("nan")
    bg_compression = float((qh_bg - ql_bg) / (qh_fc - ql_fc))          # 1.0 = no cost to the background's range
    # MEASURED 2026-09-15 (test): background-only percentiles saturate a NEAR object completely (100 % vs
    # 6 % with the object included on a synthetic near sphere); only an upper percentile at the maximum
    # removes saturation. All three recorded so the human locks the choice on real numbers.
    D_max, ql_max, qh_max = normalise(zeta_ctrl, valid, float(p["p_low"]), 1.0)
    obj_sat_max = float((D_max[M_obj] >= 1.0).mean()) if M_obj.any() else float("nan")
    step, v_bottom = contact_step(zeta_obj, zeta_bg, M_obj, int(p.get("contact_band_rows", 8)))
    quantum = contact_quantum(n_plane, d_plane, K, H / N)
    band_px = int(round(p["c2_border_band_px_256"] * H / 256.0))
    ys, xs = np.nonzero(M_obj)
    in_band = bool(v_bottom >= H - band_px or xs.mean() < band_px or xs.mean() > W - band_px or ys.mean() < band_px)
    # MEASURED 2026-09-15 (test 7): the upsampled background is the exact affine only where the Keys
    # kernel's 2-cell support lies in valid cells; within 2 cells (64 px at 1024) of the sky boundary
    # the zeroed sky bends it. Flag a contact row inside that band like the border band.
    horizon_band_px = int(2 * H / N)
    sky_rows = np.nonzero(~valid_bg.all(1))[0]
    horizon_row = int(sky_rows.max()) if len(sky_rows) else -1
    in_horizon_band = bool(horizon_row >= 0 and abs(v_bottom - horizon_row) <= horizon_band_px)
    # reviewer 2026-09-15: contact_step / quantum separates the clips cleanly (284 0.06, 035 0.67, 370 15.9) and
    # measures AGREEMENT between two independent estimates (C1's plane vs C2's background) — a gate candidate;
    # its ceiling (~2 quanta proposed) is the lock file's once more clips exist -> ungated until then
    ratio = float(step / quantum) if quantum > 0 else float("nan")
    th = (p.get("thresholds") or {}).get("m2c_contact_step_max")
    gates["g_contact_step"] = {"contact_step": step, "contact_step_quantum": quantum, "ratio": ratio, "threshold": th,
                               "pass": None if th is None else bool(abs(ratio) <= th), "outcome": "ungated" if th is None else ("pass" if abs(ratio) <= th else "fail")}
    if gates["g_contact_step"]["pass"] is False:
        # reviewer decision A1(b), 2026-09-18: PROVISIONAL ceiling from six points (largest accepted 2.26,
        # the refusal 15.9 — a factor of 7 with the ceiling in an empty gap). It measures agreement between
        # two independent estimates (C1's plane vs C2's background), which is how it saw 370's bias.
        qc.update({"emitted": False, "contact_step": step, "contact_step_quantum": quantum, "contact_step_ratio": ratio,
                   "contact_row_1024": v_bottom, "refused_by": "g_contact_step"})
        return None, qc
    # a wholly saturated object (nearer than 98 % of the background) is a second, independent detector of the
    # same placement failure, computed from different quantities — recorded
    # 2026-09-15: refuted as a failure detector (265 vehicle 0.56, 265 human 0.80 under background-only, both
    # accepted placements — it detects large near objects); recorded as a cost number only
    gates["g_obj_saturated"] = {"obj_saturated_frac": obj_sat, "pass": None, "outcome": "record (cost, not a detector)"}
    gates["g_obj_area"] = {"obj_area_frac": float(M_obj.mean()), "pass": None, "outcome": "record"}
    qc.update({"emitted": True, "q_low": q_low, "q_high": q_high, "p_low": p["p_low"], "p_high": p["p_high"],
               "percentile_domain": dom, "q_background": [ql_bg, qh_bg], "q_all_valid": [ql_all, qh_all],
               "obj_saturated_frac": obj_sat, "obj_saturated_frac_if_all": obj_sat_all, "obj_saturated_frac_if_background": obj_sat_bg,
               "obj_saturated_frac_if_p_high_max": obj_sat_max, "q_all_p_high_max": [ql_max, qh_max],
               "q_background_floor_object_ceiling": [ql_fc, qh_fc], "obj_saturated_frac_if_floor_ceiling": obj_sat_fc,
               "background_range_compression": bg_compression,
               "upsample_path": "32->1024 keys a=-0.5", "border_band_px_1024": band_px,
               "contact_step": step, "contact_step_quantum": quantum, "contact_step_ratio": ratio, "contact_row_1024": v_bottom,
               "contact_in_border_band": in_band, "horizon_band_px_1024": horizon_band_px, "horizon_row_1024": horizon_row,
               "contact_in_horizon_band": in_horizon_band, "obj_area_frac": float(M_obj.mean()), "occluded_frac": occ_frac,
               "invalid_frac_coarse": float(1 - np.mean(valid_coarse)), "invalid_all_sky": all_sky,
               "z_obj_min": gates["g_depth_sign"]["z_obj_min"], "png_encoding": "uint8 0-255 = D_ctrl * 255, near = bright UNTIL calibrated"})
    D32 = D.astype(np.float32)
    # polarity: once the human has locked `m2d_polarity` (m2_d probe), C3 emits ONE image in that sense;
    # until then both, marked uncalibrated. `D_near_bright` is always the large-is-near field.
    pol = (p.get("thresholds") or {}).get("m2d_polarity")
    assert pol in (None, "near_bright", "near_dark"), pol
    D_emit = D32 if pol in (None, "near_bright") else (np.float32(1.0) - D32).astype(np.float32)
    qc["polarity"] = pol or "uncalibrated"
    arrays = {"D_ctrl": D_emit, **({} if pol else {"D_ctrl_inv": (np.float32(1.0) - D32).astype(np.float32)}),   # exact complement of the STORED field
              "zeta_ctrl": zeta_ctrl.astype(np.float32), "M_obj": M_obj, "valid": valid, "zeta_obj": zeta_obj.astype(np.float32)}
    return arrays, qc


def write_control(out_dir, arrays, qc, hashes):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    qc = {**qc, "upstream_hashes": hashes}
    (out / "qc_c3.json").write_text(json.dumps(qc, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    if arrays is None:
        return None
    np.savez_compressed(out / "control.npz", schema=CONTROL_SCHEMA, **arrays)
    import cv2
    png = np.repeat((np.clip(arrays["D_ctrl"], 0, 1) * 255.0 + 0.5).astype(np.uint8)[..., None], 3, axis=2)
    cv2.imwrite(str(out / "control.png"), png)
    return out / "control.npz"


def read_control(out_dir):
    out = Path(out_dir)
    qc = json.loads((out / "qc_c3.json").read_text(encoding="utf-8"))
    assert qc.get("schema") == CONTROL_SCHEMA, qc.get("schema")
    if not qc.get("emitted"):
        return None, qc
    with np.load(out / "control.npz") as z:
        assert str(z["schema"]) == CONTROL_SCHEMA
        arrays = {k: z[k] for k in z.files if k != "schema"}
    return arrays, qc
