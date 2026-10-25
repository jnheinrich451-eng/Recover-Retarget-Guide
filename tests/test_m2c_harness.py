"""m2_c / C1 placement harness — build steps 1–2 (SKILL_m2_c_C1 v1.1 §7).

Step 1: generator + source_stats + stage0 round trip through m2_a's writer/reader;
        `place` reads the triple.
Step 2: proxy constants under ORTHOGRAPHIC projection vs the analytic table (§3);
        perspective gap REPORTED at z/R in {5,10,20,50}.
HARD STOP after step 2 (report to the human) — T1…T11 are later steps.

Run:  python tests/test_m2c_harness.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor import m2a                                  # noqa: E402
from src.anchor.m2c import proxy, placement, synthetic      # noqa: E402

# (label, proxy kind, R_A factory, analytic constants). The ellipsoid (a,b,c) = (1.3,0.8,0.6)
# under three poses gives every axis a different role: h_b = extent along -y, delta_p =
# half-extent along z / sqrt2, a_0 = pi * (x half-extent) * (y half-extent). An axis swap or
# a hardcoded h_b = 1 fails at least one row; the sphere/box/billboard rows cannot see that
# (three share h_b = 1, and the box has h_b == delta_p).
ORTHO = [
    ("sphere", "sphere", lambda: np.eye(3), {"h_b": 1.0, "a_0": np.pi, "delta_p": 1 / np.sqrt(2)}),
    ("ellipsoid", "ellipsoid", lambda: np.eye(3),
     {"h_b": 0.8, "a_0": np.pi * 1.3 * 0.8, "delta_p": 0.6 / np.sqrt(2)}),
    ("ellipsoid R_y(90)", "ellipsoid", lambda: synthetic.rot_y(90.0),      # x-axis (1.3) -> z
     {"h_b": 0.8, "a_0": np.pi * 0.6 * 0.8, "delta_p": 1.3 / np.sqrt(2)}),
    ("ellipsoid R_x(90)", "ellipsoid", lambda: synthetic.rot_x(90.0),      # y-axis (0.8) -> z
     {"h_b": 0.6, "a_0": np.pi * 1.3 * 0.6, "delta_p": 0.8 / np.sqrt(2)}),
    ("box", "box", lambda: np.eye(3), {"h_b": 1.0, "a_0": 4.0, "delta_p": 1.0}),
    ("billboard", "billboard", lambda: np.eye(3), {"h_b": 1.0, "a_0": 4.0, "delta_p": 0.0}),
]
REPORT = []


def step1_io_round_trip():
    R_A = synthetic.rot_y(30.0) @ synthetic.rot_x(10.0)     # non-identity, always (§6)
    st0, mask, depth, mesh, truth = synthetic.make_scene("box", R_A=R_A)
    assert depth.dtype == np.float16                         # as Stage A stores it
    with tempfile.TemporaryDirectory() as tmp:
        qc = {"schema": m2a.STAGE0_SCHEMA, "heading_defined": True, "up_defined": True,
              "clip_id": "synthetic"}
        out = m2a.write_stage0(Path(tmp), "synthetic", st0, qc)
        back = m2a.read_stage0(out)                          # the frozen schema validates
    stats = placement.source_stats(mask, depth, back["K_a"])
    assert stats.contact_observed and stats.n_mask > 100
    pl = placement.place(back, stats, mesh)                  # `place` reads the triple
    assert np.allclose(pl.R_A, truth["R_A"], atol=1e-9), "alignment must reproduce R_A"
    REPORT.append(f"step 1: stage0 round trip through m2_a writer/reader OK; mask {stats.n_mask} px, "
                  f"centroid ({stats.u0:.2f},{stats.v0:.2f}), z_src {stats.z_src:.4f}; "
                  f"place() returned z={pl.z:.4f} s={pl.s:.4f} (truth {truth['z']:.4f}, {truth['s']:.4f}; "
                  f"perspective inputs, closed form — gap is expected here and measured in step 2)")
    print(REPORT[-1])


def step2_constants():
    n = np.array([0.0, -1.0, 0.0])
    rows = []
    for label, kind, RA, ref in ORTHO:
        mesh = proxy.make_proxy(kind)
        c = proxy.constants(mesh, RA(), n, grid_n=1600)
        rows.append((label, c, ref))
        for k, v in ref.items():
            err = abs(c[k] - v)
            tol = 1e-4 if k != "a_0" else 5e-3        # a_0 is a raster count (§4.10: measured)
            assert err < tol, f"{label} {k}: numerical {c[k]:.6f} vs analytic {v:.6f} (err {err:.2e})"
    # raster convergence measured, not chosen (§4.10): grid 400..3200, mesh fixed
    sp = proxy.make_proxy("sphere")
    grid_rows = [(g, proxy.orthographic_constants(sp, np.eye(3), g)) for g in (400, 800, 1600, 3200)]
    a_lo, a_hi = grid_rows[2][1]["a_0"], grid_rows[3][1]["a_0"]
    # pose dependence (T5 preview, asserted fully later): box h_b under 45 deg about x
    hb45 = proxy.h_b(proxy.make_proxy("box"), synthetic.rot_x(45.0), n)
    rng = np.random.default_rng(3)
    hb_sph = [proxy.h_b(sp, synthetic.random_rotation(rng), n) for _ in range(3)]
    asym_c = proxy.constants(proxy.make_proxy("asym"), synthetic.rot_y(25.0), n)

    lines = ["step 2: orthographic proxy constants (grid 1600, unit scale)",
             "| proxy | h_b | a_0 | delta_p | analytic |",
             "|---|---|---|---|---|"]
    for kind, c, ref in rows:
        lines.append(f"| {kind} | {c['h_b']:.6f} | {c['a_0']:.6f} | {c['delta_p']:.6f} | "
                     f"{ref['h_b']:.4f} / {ref['a_0']:.5f} / {ref['delta_p']:.5f} |")
    lines.append(f"| asym (R_y 25°) — SMOKE TEST, not evidence | {asym_c['h_b']:.6f} | {asym_c['a_0']:.6f} | "
                 f"{asym_c['delta_p']:.6f} | self-computed |")
    lines.append("sphere constants vs RASTER grid (300x600 mesh): " + ", ".join(
        f"{g}: dp {c['delta_p']-1/np.sqrt(2):+.1e} / a0 {c['a_0']-np.pi:+.1e}" for g, c in grid_rows))
    lines.append(f"a_0 raster tolerance (1600 vs 3200 cells): {abs(a_lo - a_hi):.2e}; "
                 f"delta_p floor at 800 is the raster (true sphere sampled on that grid: -7.8e-5), gone at 1600")
    # mesh faceting convergence of delta_p (sag ~ (pi/n_lat)^2/8): measured, not assumed
    conv = []
    for nl in (150, 300, 600):
        m = proxy.uv_sphere(n_lat=nl, n_lon=2 * nl)
        conv.append((nl, proxy.orthographic_constants(m, np.eye(3), 1600)["delta_p"]))
    lines.append("sphere delta_p vs MESH resolution (grid 1600): " + ", ".join(
        f"{nl}x{2*nl}: {dp:.6f} ({dp - 1/np.sqrt(2):+.1e})" for nl, dp in conv)
        + f"  [analytic {1/np.sqrt(2):.6f}; error should fall ~4x per doubling]")
    # box vertices are exact -> 1e-9. A UV sphere's extreme along an arbitrary n is a
    # vertex only if one lies there: shortfall <= 1 - cos(theta/2) with theta = pi/n_lat
    # (mesh faceting, measured not assumed) -> tolerance derived from the mesh, not chosen.
    n_lat = 300
    sph_bound = 1.0 - np.cos(np.pi / n_lat / np.sqrt(2))   # quad-centre bound (half diagonal)
    lines.append(f"pose: box h_b under R_x(45°) = {hb45:.9f} (expect {np.sqrt(2):.9f}, exact); "
                 f"sphere h_b under 3 random R_A = {', '.join(f'{v:.7f}' for v in hb_sph)} "
                 f"(expect 1 - faceting <= {sph_bound:.1e})")
    assert abs(hb45 - np.sqrt(2)) < 1e-9, hb45
    assert all(0 < 1 - v <= sph_bound + 1e-12 for v in hb_sph), hb_sph

    # perspective gap, REPORTED (SKILL §3): pixel-area-weighted median depth and area of the
    # rendered unit sphere at z/R in {5,10,20,50}, f = 400 px
    lines.append("perspective gap (unit sphere rendered; f = 400*z so the silhouette spans ~400 px "
                 "radius at every z — otherwise pixel-counting an 8-px disc contaminates a_0):")
    lines.append("| z/R | delta_p persp | vs 0.70711 | a_0 eff | vs pi |")
    lines.append("|---|---|---|---|---|")
    for zc in (5, 10, 20, 50):
        f = 400.0 * zc
        K = np.array([[f, 0, 512], [0, f, 512], [0, 0, 1]])
        zb, m = proxy.render_perspective(sp.V + np.array([0, 0, zc]), sp.F, K, 1024, 1024)
        dp = zc - float(np.median(zb[m]))
        a0e = float(m.sum()) * zc ** 2 / f ** 2
        lines.append(f"| {zc} | {dp:.5f} | {dp - 1 / np.sqrt(2):+.5f} | {a0e:.5f} | {a0e / np.pi - 1:+.5f} |")
    REPORT.extend(lines)
    print("\n".join(lines))


if __name__ == "__main__":
    step1_io_round_trip()
    step2_constants()
    out = Path("docs/report_m2c_harness_step2.md")
    out.write_text("# m2_c C1 harness — steps 1–2 report\n\n" + "\n".join(REPORT)
                   + "\n\nHARD STOP (SKILL §7): human review before steps 3–5.\n", encoding="utf-8")
    print(f"\nsteps 1-2 OK -> {out}")
