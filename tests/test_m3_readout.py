"""Attention readout arithmetic on known answers (brief §13 test_expected_displacement_identity; CLAUDE.md §3a.13: self-test the
instrument before interpreting its output on data). CPU torch.  Run:  python tests/test_m3_readout.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generate.m3 import readout as R  # noqa: E402

torch.manual_seed(0)
Hp, Wp, heads, d = 6, 10, 4, 8
N = Hp * Wp
jj, ii = np.meshgrid(np.arange(Wp) + 0.5, np.arange(Hp) + 0.5)
c = torch.from_numpy(np.stack([jj.ravel(), ii.ravel()], 1)).float()

# 1. one-hot attention on the true source -> F_hat equals the ground-truth displacement exactly (identity test)
shift = torch.tensor([2.0, -1.0])
src_idx = torch.arange(N).view(Hp, Wp)
src_idx = torch.roll(src_idx, shifts=(1, -2), dims=(0, 1)).reshape(-1)          # query (i,j) attends to key (i-1, j+2): displacement (+2, -1)
A = torch.zeros(heads, N, N); A[:, torch.arange(N), src_idx] = 1.0
F = R.expected_displacement(A, c)
interior = ((ii >= 1) & (jj <= Wp - 3)).ravel()                                  # torch.roll wraps at the edges; the fixture is exact inside
assert torch.allclose(F[torch.from_numpy(interior)], shift.expand(int(interior.sum()), 2)), F[:3]
print("1. one-hot attention on the true source: F_hat == displacement exactly")

# 2. uniform attention -> F_hat points to the lattice centre (the diffuse signature the diagnostics must name)
U = torch.full((heads, N, N), 1.0 / N)
Fu = R.expected_displacement(U, c)
assert torch.allclose(Fu, (c.mean(0) - c), atol=1e-5)
diag = R.diagnostics(U, c, shift.expand(N, 2), torch.ones(N, dtype=torch.bool))
assert abs(diag["centre_pull_corr"] - 1.0) < 1e-4 and abs(diag["entropy_bits_mean"] - diag["uniform_bits"]) < 1e-4 and diag["hit_mass_median"] < 0.3
print(f"2. uniform attention: F_hat = vector to centre (centre-pull corr {diag['centre_pull_corr']:.3f}), entropy = uniform, hit mass {diag['hit_mass_median']:.2f}")

# 3. attention_probs from Q/K: a key equal to the query (and others orthogonal, cold) gives a peaked row; argmax-self on the self pair
Q = torch.randn(N, heads, d) * 4
A3 = R.attention_probs(Q, Q)
d3 = R.diagnostics(A3, c, torch.zeros(N, 2), torch.ones(N, dtype=torch.bool))
assert d3["argmax_self_frac"] >= 0.9 and torch.allclose(A3.sum(-1), torch.ones(heads, N), atol=1e-5)
print(f"3. self pair with matching keys: argmax-self fraction {d3['argmax_self_frac']:.2f}, rows sum to 1")

# 4. head subset: a head that is one-hot on the truth among diffuse heads -> per-head EPE isolates it, subset readout recovers the shift
Am = U.clone(); Am[0] = A[0]
d4 = R.diagnostics(Am, c, shift.expand(N, 2), torch.ones(N, dtype=torch.bool))
assert d4["per_head_epe"][0] < 1e-5 and min(d4["per_head_epe"][1:]) > 1.0
assert torch.allclose(R.expected_displacement(Am, c, heads=[0])[torch.from_numpy(interior)], shift.expand(int(interior.sum()), 2))   # same edge wrap as test 1
print("4. head subset: the one informative head is found by per-head EPE and reads the truth")
print("test_m3_readout: passed")

# 5. heavy-tailed attention (a peak on the true source + a uniform background): the plain expectation is pulled toward the
#    centre and misses by patches; the windowed soft-argmax recovers the shift. This is the measured situation (2026-10-07:
#    self-hit 0.96-1.00 but |F_self| 2-6 patches, hit mass 0.15-0.43, centre-pull corr 0.84-0.97).
mix = 0.3 * A + 0.7 * U                                                          # 30 % mass on the true source, 70 % spread uniformly
Fplain = R.expected_displacement(mix, c)
Fwin = R.windowed_expected_displacement(mix, c, radius=2.0)
inner = torch.from_numpy(interior)
err_plain = torch.linalg.norm(Fplain[inner] - shift, dim=-1).median()
err_win = torch.linalg.norm(Fwin[inner] - shift, dim=-1).median()
assert err_plain > 1.0 and err_win < 0.3, (err_plain, err_win)
# head subset chosen on another clip's per-head EPE: top-2 of a planted list
assert R.select_heads([3.0, 0.5, None, 0.9, 2.0], 2) == [1, 3]
print(f"5. heavy-tailed attention: plain readout misses by {err_plain:.2f} patches, windowed (r=2) by {err_win:.2f}; head selection by rank")
print("test_m3_readout (readout forms): passed")
