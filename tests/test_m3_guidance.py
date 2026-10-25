"""Guidance losses on known answers (brief v2.6 §5, §13). CPU.  Run:  python tests/test_m3_guidance.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generate.m3 import guidance as GD, readout as RO  # noqa: E402

torch.manual_seed(0)
Hp, Wp = 6, 10; N = Hp * Wp
jj, ii = torch.meshgrid(torch.arange(Wp) + 0.5, torch.arange(Hp) + 0.5, indexing="xy")
c = torch.stack([jj.reshape(-1), ii.reshape(-1)], -1).float()


def peaked_attention(F_true, sharp=6.0):
    """logits peaked at the key nearest c_q + F_true(q): a model that 'knows' the displacement. Returns logits [1,N,N] (grad-able)."""
    tgt = c + F_true
    d = torch.cdist(tgt, c)
    return (-sharp * d)[None].clone().requires_grad_(True)


# 1. L_bg is zero when the readout matches the target and positive otherwise; its gradient on the logits reduces it
F_geo = torch.stack([torch.full((N,), 1.0), torch.full((N,), -0.5)], -1)
tgt_in = ((c + F_geo)[:, 0] > 0) & ((c + F_geo)[:, 0] < Wp) & ((c + F_geo)[:, 1] > 0) & ((c + F_geo)[:, 1] < Hp)
C_geo = tgt_in.float(); O_t = torch.zeros(N)                                      # coverage is zero where the source lies outside the lattice
lg = peaked_attention(F_geo)
A = torch.softmax(lg, -1)
F_hat = RO.windowed_expected_displacement(A, c, 3.0)
L0 = GD.bg_loss(F_hat, F_geo, C_geo, O_t)
assert L0.item() < 0.05, L0.item()
F_wrong = F_geo + torch.tensor([2.0, 0.0])
lg_w = peaked_attention(F_wrong)
A_w = torch.softmax(lg_w, -1)
L_w = GD.bg_loss(RO.windowed_expected_displacement(A_w, c, 3.0), F_geo, C_geo, O_t)
assert L_w.item() > 1.0, L_w.item()
L_w.backward()
g = lg_w.grad
with torch.no_grad():
    lg2 = lg_w - 20.0 * g
L_after = GD.bg_loss(RO.windowed_expected_displacement(torch.softmax(lg2, -1), c, 3.0), F_geo, C_geo, O_t)
assert L_after.item() < L_w.item(), (L_w.item(), L_after.item())          # a descent step on the logits lowers L_bg (gradient path through the window)
# the object patches carry no weight in L_bg
O_half = torch.zeros(N); O_half[: N // 2] = 1.0
F_mixed = F_geo.clone(); F_mixed[: N // 2] += 5.0
assert GD.bg_loss(F_mixed, F_geo, C_geo, O_half).item() < 1e-9

# 2. centroid loss: mean readout over object patches against the target vector
tgt = torch.tensor([0.7, -1.2])
F_obj = torch.zeros(N, 2); F_obj[: N // 2] = tgt
assert GD.centroid_loss(F_obj, tgt, O_half).item() < 1e-12
assert abs(GD.centroid_loss(F_obj + torch.tensor([1.0, 0.0]), tgt, O_half).item() - 1.0) < 1e-5     # the eps in the weight denominator
assert GD.centroid_loss(F_obj, tgt, torch.zeros(N)).item() == 0.0                 # no object patches: no term

# 3. L_perp: a point on its epipolar line costs 0; off the line by d costs d^2 (weight 1 where uncovered)
lines = torch.zeros(N, 3); lines[:, 0] = 0.0; lines[:, 1] = 1.0; lines[:, 2] = -(c[:, 1] + 0.5)   # horizontal line one row below each query
F_on = torch.stack([torch.zeros(N), torch.full((N,), 0.5)], -1)                  # expected source exactly on the line
H_pts = torch.full((N, 2), float("nan")); mode = torch.ones(N, dtype=torch.bool); C_max = torch.zeros(N)
assert GD.perp_loss(F_on, c, lines, H_pts, mode, C_max).item() < 1e-6
F_off = F_on + torch.tensor([0.0, 0.3])
assert abs(GD.perp_loss(F_off, c, lines, H_pts, mode, C_max).item() - 0.09) < 1e-5
assert GD.perp_loss(F_off, c, lines, H_pts, mode, torch.ones(N)).item() < 1e-9    # fully covered patches carry no perpendicular weight
# homography mode: distance to the mapped point
H_pts2 = c + torch.tensor([0.5, 0.0]); mode2 = torch.zeros(N, dtype=torch.bool)
assert abs(GD.perp_loss(torch.stack([torch.full((N,), 0.5), torch.zeros(N)], -1), c, lines, H_pts2, mode2, C_max).item()) < 1e-6

# 4. mass terms: a peaked attention at the target has mass near 1 inside the window and its window contains the target
m = GD.mass_terms(A.detach(), c, F_geo, 3.0, C_geo)
assert m["mass_at_target_mean"] > 0.9 and m["window_contains_target_share"] > 0.9 and m["L_mass"].item() < 0.2
m_w = GD.mass_terms(A_w.detach(), c, F_geo, 1.0, C_geo)
assert m_w["mass_at_target_mean"] < 0.5 and m_w["L_mass"].item() > m["L_mass"].item()

# 5. the update: RMS-normalised step, frame 0 untouched, dtype preserved
z = torch.randn(1, 4, 5, 3, 3, dtype=torch.bfloat16); g = torch.randn(1, 4, 5, 3, 3)
z2 = GD.guided_update(z, g, 0.5)
assert z2.dtype == z.dtype and torch.equal(z2[:, :, 0], z[:, :, 0])
step = (z2.float() - z.float())[:, :, 1:]
g0 = g.clone(); g0[:, :, 0] = 0                                                   # the brief zeroes the frame-0 slice BEFORE the RMS (§7 pseudo-code)
assert abs(torch.sqrt((step ** 2).mean()).item() - 0.5 * torch.sqrt((g0[:, :, 1:] ** 2).mean() / (g0 ** 2).mean()).item()) < 0.05
# 6. the tile detector: a natural-like random texture reads low; the same texture tiled at a 32 px period reads a high secondary peak at ~32 px
rng = np.random.default_rng(1)
base = rng.normal(size=(120, 160)); base = np.stack([base] * 3, -1)
from scipy.ndimage import gaussian_filter
nat = gaussian_filter(base, (3, 3, 0)); nat = ((nat - nat.min()) / (nat.max() - nat.min()) * 255).astype(np.uint8)
tile = np.tile(nat[:32, :32], (4, 5, 1))[:120, :160]
mask = np.ones((120, 160), bool)
s_nat, s_tile = GD.tile_score(nat, mask), GD.tile_score(tile, mask)
assert s_nat["secondary_autocorr_peak"] < 0.5 and s_tile["secondary_autocorr_peak"] > 0.8 and abs(s_tile["lag_px"] - 32) < 2.5, (s_nat, s_tile)
print(f"tile detector: natural {s_nat['secondary_autocorr_peak']:.2f}, tiled {s_tile['secondary_autocorr_peak']:.2f} at lag {s_tile['lag_px']:.0f} px")
print(f"test_m3_guidance: passed (L_bg at target {L0.item():.4f}, off target {L_w.item():.2f} -> {L_after.item():.2f} after one logit step; "
      f"mass at target {m['mass_at_target_mean']:.2f} vs {m_w['mass_at_target_mean']:.2f} off target)")
