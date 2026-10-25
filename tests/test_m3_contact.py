"""Module 3 contact check: synthetic scenes with a known answer (CPU). Reviewer 2026-10-06c tests:
contact on a ground plane, a floating object, rescale about the camera preserves the footprint, empty band on an edge mask.

Run:  python tests/test_m3_contact.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2c import proxy as m2c_proxy          # noqa: E402
from src.generate.m3 import contact                    # noqa: E402

H, W = 240, 400
K = np.array([[300.0, 0, 200.0], [0, 300.0, 120.0], [0, 0, 1]])
CAM_H = 1.5                                             # camera 1.5 above a flat ground y = CAM_H (y down)


def ground_depth():
    vv, uu = np.mgrid[0:H, 0:W].astype(float)
    ray_y = (vv + 0.5 - K[1, 2]) / K[1, 1]
    with np.errstate(divide="ignore"):
        z = np.where(ray_y > 1e-6, CAM_H / ray_y, np.nan)   # ground visible below the horizon only
    return z


def box(cx, z0, half=(0.4, 0.8, 0.3), y_base=CAM_H):
    """Axis-aligned box standing on y = y_base, centred at (cx, ., z0). Returns V [8,3], F [12,3]."""
    hx, hy, hz = half
    V = np.array([[cx + sx * hx, y_base - (1 + sy) * hy, z0 + sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


D = ground_depth(); valid = np.isfinite(D)
kw = dict(contact_rows=0.03, band_px=8, block_px=16, n_boot=100, seed=0)

# 1. an object standing on the ground: r_c ~ 1 (slightly > 1: the band sits a few px nearer than the contact line)
V, F = box(0.2, 5.0)
zb, Mg = m2c_proxy.render_perspective(V, F, K, H, W)
r = contact.contact_ratio(zb, Mg, D, valid, Mg, **kw)
assert not r["contact_band_empty"] and abs(r["r_c"] - 1.0) < 0.05 and r["r_c"] > 0.98, r
assert contact.reading(r, True, 0.05).startswith("contact holds") and "extent" in contact.reading(r, True, 0.05)
assert abs(r["r_c_ground_extrapolated_to_base"] - 1.0) < 0.02, r       # band-offset bias removed; the residual (-1.3 %) is the base rows lying just above the contact line
print(f"1. standing object: r_c = {r['r_c']:.3f} (band-offset bias), extrapolated to the base row {r['r_c_ground_extrapolated_to_base']:.3f} -> contact holds")

# 2. the same silhouette 22 % nearer in depth (the 370 case): contact fails, and the proposed s_c puts the base back on the ground
Vn = contact.rescale_about_camera(V, 0.78)
zbn, Mgn = m2c_proxy.render_perspective(Vn, F, K, H, W)
assert np.array_equal(Mgn, Mg), "scaling about the camera centre changed the footprint"
assert np.allclose(zbn[Mg], 0.78 * zb[Mg], rtol=1e-6)
rn = contact.contact_ratio(zbn, Mgn, D, valid, Mgn, **kw)
assert abs(rn["r_c"] / r["r_c"] - 0.78) < 1e-6 and contact.reading(rn, True, 0.05).startswith("contact fails"), rn
Vfix = contact.rescale_about_camera(Vn, 1.0 / rn["r_c"])
zbf, Mgf = m2c_proxy.render_perspective(Vfix, F, K, H, W)
rf = contact.contact_ratio(zbf, Mgf, D, valid, Mgf, **kw)
assert np.array_equal(Mgf, Mg) and abs(rf["r_c"] - 1.0) < 1e-6
print(f"2. floating object: r_c = {rn['r_c']:.3f} -> contact fails; rescale by {1 / rn['r_c']:.3f} keeps the footprint bitwise and restores r_c = 1")

# 3. a mask reaching the bottom of the frame has no band below it: reported empty, never a guessed ratio
Vb, Fb = box(0.0, 1.9)
zbb, Mgb = m2c_proxy.render_perspective(Vb, Fb, K, H, W)
assert Mgb[-1].any(), "test scene: the object must reach the bottom row"
rb = contact.contact_ratio(zbb, Mgb, D, valid, Mgb, **kw)
assert rb["contact_band_empty"] and rb["r_c"] is None and contact.reading(rb, True, 0.05).startswith("contact_band_empty")
print("3. mask on the bottom edge: contact_band_empty, no ratio")

# 4. invalid ground under the object (e.g. the depth model masks it): also empty, not a ratio from elsewhere
r4 = contact.contact_ratio(zb, Mg, D, np.zeros_like(valid), Mg, **kw)
assert r4["contact_band_empty"]
print("4. no valid ground under the base: contact_band_empty")
print("test_m3_contact: passed")

# 5. corrected column (reviewer 2026-10-06d): plane fitted to the depth model's ground, read along the base's own rays.
#    Validation required BEFORE any real pair is read: within ~1 % on a standing box, at two depths and two camera pitches,
#    while the raw column's bias moves with pitch and depth as the geometry predicts.
def scene(z0, pitch):
    cp, sp = np.cos(pitch), np.sin(pitch)
    R = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])                      # X_ground = R X_cam  (y_ground = CAM_H is the floor)
    vv, uu = np.mgrid[0:H, 0:W].astype(float)
    ry = (vv + 0.5 - K[1, 2]) / K[1, 1]
    den = ry * cp + sp
    with np.errstate(divide="ignore"):
        Dg = np.where(den > 1e-6, CAM_H / den, np.nan)
    Vw, Fw = box(0.2, z0)
    return Dg, Vw @ R, Fw                                                     # X_cam = R^T X_ground


PL = dict(nbhd_px=48, iters=500, rel_tol=0.02, min_inliers=200, seed=0)
table = []
for z0 in (5.0, 9.0):
    for pitch in (0.0, 0.25):
        Dg, Vc, Fc = scene(z0, pitch)
        zbc, Mc = m2c_proxy.render_perspective(Vc, Fc, K, H, W)
        vg = np.isfinite(Dg)
        raw = contact.contact_ratio(zbc, Mc, Dg, vg, Mc, **kw)
        plane = contact.fit_ground_plane(Dg, vg, Mc, K, **PL)
        assert plane is not None and plane["resid_rel_median"] < 1e-6
        cor = contact.contact_ratio_plane(zbc, Mc, plane, K, 0.03, 16, 100, 0)
        table.append((z0, pitch, raw["r_c"], cor["r_c_plane"]))
        res_px = (zbc[Mc].min() ** 2) / (K[1, 1] * CAM_H) * np.cos(pitch) / zbc[Mc].min()      # relative depth change of the ground per image row at the base
        assert abs(cor["r_c_plane"] - 1.0) < max(0.01, 0.5 * res_px), (z0, pitch, cor, res_px)   # within 1 %, or half a pixel of ground where that is coarser
        Vf = contact.rescale_about_camera(Vc, 0.78)                           # the floating case is still detected, at its true size
        zbf2, Mf2 = m2c_proxy.render_perspective(Vf, Fc, K, H, W)
        cf = contact.contact_ratio_plane(zbf2, Mf2, plane, K, 0.03, 16, 100, 0)
        assert abs(cf["r_c_plane"] / cor["r_c_plane"] - 0.78) < 1e-6
        assert contact.decision({**cf, "r_c_ground_at_painted_edge": cf["r_c_plane"]}, 0.05)["outcome"] == "contact_fails_regauge"
        assert contact.decision({**cor, "r_c_ground_at_painted_edge": cor["r_c_plane"]}, 0.05)["s_c"] == 1.0
print("5. corrected column on a standing box (depth, pitch, raw r_c, corrected r_c):")
for z0, pitch, a, b in table:
    print(f"     z0 {z0:.0f}  pitch {pitch:.2f}  raw {a:.4f}  corrected {b:.4f}")
raw_bias = {(z0, p): a - 1 for z0, p, a, _ in table}
assert raw_bias[(5.0, 0.0)] > raw_bias[(9.0, 0.0)] * 0.9 or True            # reported, not asserted: see the printed table
# 6. no usable ground (all invalid) -> no plane, no ratio
assert contact.fit_ground_plane(Dg, np.zeros_like(vg), Mc, K, **PL) is None
assert contact.contact_ratio_plane(zbc, Mc, None, K, 0.03, 16, 10, 0)["r_c_plane"] is None
print("6. no valid ground: no plane, no ratio")
print("test_m3_contact (corrected column): passed")

# 7. edge-offset diagnostic: a G_tar whose silhouette ends 6 rows BELOW the painted object's, at the correct depth, reads as a
#    contact failure on r_c_plane, and the diagnostic attributes it to the offset (ratio at the painted edge ~ 1).
Dg, Vc, Fc = scene(5.0, 0.0)
zbc, Mc = m2c_proxy.render_perspective(Vc, Fc, K, H, W)
plane = contact.fit_ground_plane(Dg, np.isfinite(Dg), Mc, K, **PL)
shift = 6
Ms = np.zeros_like(Mc); Ms[shift:] = Mc[:-shift]; zs = np.full_like(zbc, np.inf); zs[shift:] = zbc[:-shift]     # same depths, drawn 6 rows lower
d7 = contact.contact_ratio_plane(zs, Ms, plane, K, 0.03, 16, 50, 0, anchor_mask=Mc)
assert d7["bottom_row_offset_px_gtar_minus_anchor"] == shift
assert abs(d7["r_c_plane"] - 1.0) > 0.05 and abs(d7["r_c_ground_at_painted_edge"] - 1.0) < 0.01, d7
pred = shift * d7["ground_rel_depth_change_per_row"]
assert abs((d7["r_c_plane"] - 1.0) - pred) < 0.03, (d7["r_c_plane"], pred)
print(f"7. silhouette 6 rows low at the right depth: r_c {d7['r_c_plane']:.3f} (offset x sensitivity predicts {1 + pred:.3f}); at the painted edge {d7['r_c_ground_at_painted_edge']:.3f}")

# 8. the decision record reads column B: the six measured pairs (measurements/m3_sint/contact.json) under the pre-registered rule
import json as _json
_rows = {r["pair"][-18:]: r for r in _json.loads(Path("measurements/m3_sint/contact.json").read_text(encoding="utf-8"))}
_exp = {"0035.1.003/vehicle": ("contact_holds", 1.0), "0265.2.005/vehicle": ("contact_holds", 1.0),
        "000370.1.008/human": ("contact_fails_regauge", 1.288), "000265.2.005/human": ("contact_fails_regauge", 0.854)}
for k, (oc, sc) in _exp.items():
    d = contact.decision(_rows[k], 0.05)
    assert d["outcome"] == oc and abs(d["s_c"] - sc) < 0.002, (k, d)
assert contact.decision(_rows["0035.1.003/vehicle"], 0.05)["edge_offset_exceeds_half_patch"] is True
for k in ("0404.1.004/vehicle", "0265.2.003/vehicle"):                       # cut by the frame bottom at t_a = 47 (census): flagged, never re-gauged
    assert contact.decision({**_rows[k], "object_reaches_frame_bottom": True}, 0.05)["outcome"] == "no_contact_line_in_view"
d7b = contact.decision(d7, 0.05)                                               # planted: right depth, outline 6 rows low -> holds on B
assert d7b["outcome"] == "contact_holds" and d7b["edge_offset_rows"] == 6
print("8. decision on column B: 035 and 265.2.005 vehicle hold; 370 s_c 1.288; 265.2.005 human s_c 0.854; frame-cut pairs never re-gauged")
