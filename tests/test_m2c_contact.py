"""m2_c C1 — review items of 2026-09-15: (a) robust contact row, (b) local plane offset.

Run:  python tests/test_m2c_contact.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2c import placement, proxy, synthetic                 # noqa: E402

REPORT = ["# m2_c C1 — robust contact row and local plane offset (review 2026-09-15)", ""]
mesh = proxy.uv_sphere(n_lat=150, n_lon=300, radii=(1.3, 0.8, 0.6), name="ellipsoid")
R = synthetic.rot_y(30.0) @ synthetic.rot_x(10.0)
st0, mk, dp, m, tr = synthetic.make_scene("ellipsoid", R_A=R, z_true=8.0, rho_x=0.0, mesh=mesh, grid_n=800)
K = st0["K_a"].astype(float)
H, W = mk.shape

# ---------------------------------------------------------------- (a) robust contact row
clean = placement.source_stats(mk, dp, K)
assert clean.contact_observed and abs(clean.v_contact - clean.v_contact_max) < 1.5
# three stray pixels on the bottom row (shadow speckle, clip 035's signature): a max would declare
# contact unobserved and lose the row; the percentile keeps it
speck = mk.copy(); dps = dp.copy()
for u in (40, 41, 200):
    speck[H - 1, u] = True; dps[H - 1, u] = clean.z_src
s_speck = placement.source_stats(speck, dps, K)
assert s_speck.contact_observed, "stray bottom-row pixels must not kill the contact row"
assert abs(s_speck.v_contact - clean.v_contact) < 1.5, (s_speck.v_contact, clean.v_contact)
assert s_speck.v_contact_max == H and s_speck.bottom_run_frac < 0.1
# the object genuinely cut by the border: shift the mask down so its bottom rows leave the image;
# the last row then carries a contiguous run comparable to the object's width -> unobserved
shift = int(H - clean.v_contact_max + 12)
cut = np.zeros_like(mk); cut[shift:] = mk[:H - shift]
dpc = np.zeros_like(dp); dpc[shift:] = dp[:H - shift]
s_cut = placement.source_stats(cut, dpc, K)
assert not s_cut.contact_observed and s_cut.v_contact is None and s_cut.bottom_run_frac > 0.5, (s_cut.bottom_run_frac,)
assert s_cut.contact_at_frame_edge and not clean.contact_at_frame_edge
# 2026-09-18: a robust contact row AT the frame edge with a NARROW last-row run (clips 284 / 370) is admitted with
# a flag; the refusal of a false contact is C3's contact-step ratio gate
edge = np.zeros_like(mk); sh2 = int(H - clean.v_contact_max); edge[sh2:] = mk[:H - sh2]        # bottom exactly on the last row
dpe = np.zeros_like(dp); dpe[sh2:] = dp[:H - sh2]
s_edge = placement.source_stats(edge, dpe, K)
assert s_edge.contact_at_frame_edge and s_edge.contact_observed and s_edge.bottom_run_frac < 0.5, (s_edge.bottom_run_frac,)
REPORT.append(f"(a) robust contact: clean v_contact {clean.v_contact:.1f} (max {clean.v_contact_max:.1f}); 3 stray bottom-row pixels -> "
              f"still observed, v_contact {s_speck.v_contact:.1f}, run frac {s_speck.bottom_run_frac:.2f}; object cut by the border -> "
              f"unobserved, run frac {s_cut.bottom_run_frac:.2f} (threshold {placement.CONTACT['run_width_frac']})")

# ---------------------------------------------------------------- (b) local plane offset
n = st0["n_plane"]; d_true = float(st0["d_plane"])
consts = {"a_0": tr["a_0"], "h_b": tr["h_b"], "delta_p": tr["delta_p"]}
pl_g = placement.place(st0, clean, m, consts=consts, local_offset=False)
pl_l = placement.place(st0, clean, m, consts=consts, local_offset=True)
gap = pl_l.d_local_minus_global
bias = abs(gap) / d_true
assert pl_g.d_local_minus_global is None and abs(pl_g.d_used - d_true) < 1e-12
assert bias < 0.03, f"local offset bias {bias:.3f} of the camera height on a round-bottomed object"
# a corrupted global plane (rough terrain: offset 20 % too high) — the local offset is immune to it
bad = dict(st0); bad["d_plane"] = d_true + 0.2 * d_true
pl_bad_g = placement.place(bad, clean, m, consts=consts, local_offset=False)
pl_bad_l = placement.place(bad, clean, m, consts=consts, local_offset=True)
err_g = abs(pl_bad_g.s - tr["s"]) / tr["s"]; err_l = abs(pl_bad_l.s - tr["s"]) / tr["s"]
assert err_g > 0.10 and err_l < 0.05, (err_g, err_l)
assert abs(pl_bad_l.d_used - pl_l.d_used) < 1e-12, "the local offset must not depend on the global one"
assert abs(pl_bad_l.d_local_minus_global - (gap - 0.2 * d_true)) < 1e-9, "the recorded gap must reflect the corruption"
REPORT.append(f"(b) local plane offset on the harness ellipsoid (round bottom, worst case): d_local - d_global = {gap:+.4f} "
              f"({100 * bias:.1f} % of camera height {d_true}; contour-depth quantile {placement.CONTACT['depth_quantile']}); "
              f"with the global offset corrupted by +20 %: scale error {100 * err_g:.1f} % (global) vs {100 * err_l:.1f} % (local), "
              f"recorded gap {pl_bad_l.d_local_minus_global:+.3f}")

# ---------------------------------------------------------------- §6.2 amended: in-plane basis (clip 035's case)
d035, n035 = np.array([0.409, -0.195, 0.892]), np.array([0.14, -0.989, 0.036])
d035 /= np.linalg.norm(d035); n035 /= np.linalg.norm(n035)
B, kappa, pitch = placement.desired_basis_in_plane(d035, n035)
assert abs(B[:, 0] @ n035) < 1e-12 and np.allclose(B[:, 1], n035) and abs(np.linalg.det(B) - 1) < 1e-12
assert abs(pitch - 16.4) < 0.1 and abs(kappa - np.cos(np.radians(pitch))) < 1e-9
R_pl, k_pl = placement.alignment(d035, np.array([0, -1.0, 0]), proxy.CANON_B_TAR, n_plane=n035)
fwd = R_pl @ proxy.CANON_B_TAR[:, 0]; up = R_pl @ proxy.CANON_B_TAR[:, 1]
assert abs(fwd @ n035) < 1e-12 and np.allclose(up, n035)
# the projected forward keeps the azimuth: same direction as d within the plane
assert fwd @ (d035 - (n035 @ d035) * n035) > 0.999 * np.linalg.norm(d035 - (n035 @ d035) * n035)
# without a plane the original u_scene rule is untouched
R_old, _ = placement.alignment(d035, np.array([0, -1.0, 0]), proxy.CANON_B_TAR)
assert abs((R_old @ proxy.CANON_B_TAR[:, 0]) @ d035 - 1) < 1e-12
try:
    placement.desired_basis_in_plane(n035, n035); raise AssertionError("heading along the normal accepted")
except AssertionError as e:
    assert "parallel" in str(e)
REPORT.append(f"§6.2 in-plane basis on clip 035's numbers: pitch removed {pitch:+.1f} deg, kappa_basis {kappa:.4f} = cos(pitch); forward "
              "in the plane, up == n, azimuth preserved; no-plane path unchanged; heading along the normal refused")

# ---------------------------------------------------------------- area_preserving mode (spec §6.3 amended 2026-09-15)
from src.anchor.m2c.placement import SourceStats
wp = SourceStats(tr["u0_wp"], tr["v0_wp"], tr["A_wp"], tr["v_contact_wp"], tr["z_src_wp"], True, 0.0, int(tr["A_wp"]), 256, 256)
st0f = tr["stage0_f64"]
pl_d = placement.place(st0f, wp, m, "depth_preserving", consts=consts, local_offset=False)
pl_a = placement.place(st0f, wp, m, "area_preserving", consts=consts, local_offset=False)
pl_f = placement.place(st0f, wp, m, "framing_preserving", consts=consts, local_offset=False)      # alias
assert abs(pl_a.z - tr["z"]) < 1e-6 * tr["z"] and abs(pl_a.s - tr["s"]) < 1e-6, (pl_a.z, pl_a.s, tr["z"], tr["s"])
assert abs(pl_d.z - pl_f.z) < 1e-12 and pl_f.mode == "depth_preserving"
assert abs(pl_a.contact_residual) < 1e-9
# a near-biased z_src (a truncated mask: clip 370's mechanism) shrinks the depth mode's answer but not the area mode's
wp_near = SourceStats(tr["u0_wp"], tr["v0_wp"], tr["A_wp"], tr["v_contact_wp"], 0.8 * tr["z_src_wp"], True, 0.0, int(tr["A_wp"]), 256, 256)
pl_dn = placement.place(st0f, wp_near, m, "depth_preserving", consts=consts, local_offset=False)
pl_an = placement.place(st0f, wp_near, m, "area_preserving", consts=consts, local_offset=False)
# MEASURED 2026-09-15: with a level ground and a downward ray (n.rho < 0) a NEAR-biased z_src makes the
# depth mode's scale LARGER (s = (n.rho z_src + d)/det), not smaller — the reviewer's mechanism for 370's
# small proxy has the wrong sign here; 370 shrank through the local plane offset on a truncated contact
# (the frame-bottom row is the man's thighs). The area mode is invariant to the depth bias either way.
assert abs(pl_dn.s - tr["s"]) > 0.05 * tr["s"] and abs(pl_an.s - tr["s"]) < 1e-6, (pl_dn.s, pl_an.s)
REPORT.append(f"area_preserving: exact on the weak-perspective truth (z {pl_a.z:.4f} vs {tr['z']}, s {pl_a.s:.4f} vs {tr['s']}); with z_src biased "
              f"20 % near, depth_preserving gives s {pl_dn.s:.3f} (LARGER: n.rho < 0 on a level ground) while area_preserving keeps {pl_an.s:.3f}; "
              "framing_preserving accepted as the alias of depth_preserving")

REPORT += ["", "all contact/offset tests passed"]
Path("docs/report_m2c_contact_tests.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
print("\n".join(REPORT))
