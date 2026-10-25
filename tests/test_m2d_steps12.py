"""m2_d build steps 1-2 — SKILL_m2_d v1.1 §8 tests 1 (polarity complement) and 4 (correlation sign),
plus the stimuli's own premises. CPU only.

Run:  python tests/test_m2d_steps12.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2d import instrument, stimuli                          # noqa: E402

REPORT = ["# m2_d steps 1-2 — correlation instrument and synthetic stimuli", ""]
rng = np.random.default_rng(1)

# ---- test 4: correlation sign (guards the probe's own instrument)
z = rng.uniform(0.1, 2.0, (256, 256))
mono = np.sqrt(z) * 3.0 + 0.2                                  # a monotone re-encoding (what a relative estimator does)
assert abs(instrument.spearman(mono, z) - 1.0) < 1e-12
assert abs(instrument.spearman(-mono, z) + 1.0) < 1e-12
assert abs(instrument.spearman(1.0 / mono, z) + 1.0) < 1e-12    # depth instead of disparity flips the sign
noise = rng.normal(size=z.shape)
assert abs(instrument.spearman(noise, z)) < 0.02
ties = np.round(z * 4) / 4                                      # heavy ties must not break the ranks
assert instrument.spearman(ties, z) > 0.95
m = np.zeros_like(z, bool); m[:64] = True
assert abs(instrument.spearman(np.where(m, mono, -mono), z, mask=m) - 1.0) < 1e-12
REPORT.append("test 4 correlation sign: monotone re-encoding rho = +1, its negation and its reciprocal (depth for disparity) rho = -1, "
              "noise |rho| < 0.02, heavy ties rho > 0.95, mask respected")

# ---- stimuli
S = stimuli.all_synthetic()
lines = []
for s in S:
    D, Di = s["D_ctrl"], s["D_ctrl_inv"]
    # test 1: exact complements, before any probe runs
    assert np.array_equal(Di, (np.float32(1) - D).astype(np.float32)), s["name"]
    assert np.isfinite(s["zeta_ctrl"]).all() and D.min() >= 0 and D.max() <= 1
    u8, u8i = stimuli.to_control_uint8(D), stimuli.to_control_uint8(Di)
    assert u8.shape == (1024, 1024, 3) and (u8[..., 0] == u8[..., 1]).all()
    assert np.abs(u8.astype(int) + u8i.astype(int) - 255).max() <= 1        # uint8 complements to within rounding
    lines.append(f"{s['name']}: valid {100 * s['valid'].mean():.0f} %, D range [{D[s['valid']].min():.2f}, {D[s['valid']].max():.2f}], "
                 f"checksum {instrument.checksum(u8)} / inv {instrument.checksum(u8i)}")
s1, s2, s3 = S
# S1: the ground is nearer at the bottom than toward the horizon, monotonically down a column
col = s1["zeta_ctrl"][:, 40]; g = s1["regions"]["ground"][:, 40]
assert (np.diff(col[g]) > 0).all(), "ground disparity must increase toward the bottom of the image"
# S2: topological premise — the bar beats the panel on every overlapping pixel, and D orders them the same way
ov = s2["regions"]["overlap"]; far = s2["regions"]["far_panel"]; near = s2["regions"]["near_bar"]
assert s2["D_ctrl"][near].min() > s2["D_ctrl"][far].max() - 1e-6, "in the control the bar is brighter (nearer) than the whole panel"
assert s2["D_ctrl_inv"][near].max() < s2["D_ctrl_inv"][far].min() + 1e-6, "and darker in the inverted control"
# S3: the vanishing point is the FARTHEST point and sits at the image centre
zc = s3["zeta_ctrl"]; cy, cx = np.unravel_index(np.argmin(np.where(s3["valid"], zc, np.inf)), zc.shape)
assert abs(cy - 511.5) < 3 and abs(cx - 511.5) < 3, (cy, cx)
assert zc[1000, 512] > zc[600, 512] > zc[520, 512]
REPORT.append("test 1 polarity complement + stimuli premises: " + "; ".join(lines) +
              f"; S1 ground monotone down the image; S2 bar nearer than the panel on all {int(ov.sum())} overlap px and in D; "
              f"S3 vanishing point at ({cx}, {cy})")

REPORT += ["", "all m2_d step 1-2 tests passed"]
Path("docs/report_m2d_steps12_tests.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
print("\n".join(REPORT))
