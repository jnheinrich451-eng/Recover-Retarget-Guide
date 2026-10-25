"""Attention readout (brief v2.5 §4): expected displacement from a sub-block softmax, and its diagnostics.

F_hat^{t->a}(q) = sum_{q'} A(q, q') (c_{q'} - c_q), A = softmax over the keys of frame a of (Q_t . K_a / sqrt(d)), heads averaged
(lock attention.head_mode = mean) unless a head subset is given. Pure torch; no model here, so the arithmetic is unit-tested
on a known answer (brief §13 test_expected_displacement_identity) before any real number is read (CLAUDE.md §3a.13).
"""
from __future__ import annotations

import torch


def attention_probs(Qt: torch.Tensor, Ka: torch.Tensor) -> torch.Tensor:
    """Qt [N, heads, d] (queries of frame t), Ka [N, heads, d] (keys of frame a) -> A [heads, N, N], rows sum to 1."""
    assert Qt.ndim == 3 and Ka.ndim == 3 and Qt.shape[1:] == Ka.shape[1:], (Qt.shape, Ka.shape)
    d = Qt.shape[-1]
    return torch.softmax(torch.einsum("qhd,khd->hqk", Qt.float(), Ka.float()) / (d ** 0.5), dim=-1)


def expected_displacement(A: torch.Tensor, c: torch.Tensor, heads=None) -> torch.Tensor:
    """A [heads, N, N], c [N, 2] patch centres -> F_hat [N, 2] = E_A[c_source] - c_query, averaged over `heads` (all if None)."""
    Ah = A if heads is None else A[list(heads)]
    P = Ah.mean(0)                                                               # mean of probabilities over heads, not of logits
    return (P @ c.float()) - c.float()


def diagnostics(A: torch.Tensor, c: torch.Tensor, F_gt: torch.Tensor, mask: torch.Tensor, hit_radius: float = 2.0) -> dict:
    """What a failed gate means (reviewer 2026-10-07f): is the attention diffuse, mis-ordered, or biased?
    A [heads,N,N]; c [N,2]; F_gt [N,2] ground-truth backward displacement; mask [N] bool (gated patches).
      argmax_self_frac    share of queries whose most-attended key is their OWN position (self pair: should be high if token order is right)
      hit_mass            attention mass within hit_radius patches of the ground-truth source location, median over gated queries
      entropy_bits        mean attention entropy (log2 N is uniform)
      centre_pull_corr    correlation of F_hat with (lattice centre - c_q): the signature of diffuse attention
      per_head_epe        median EPE per head on gated patches
    """
    heads, N, _ = A.shape
    P = A.mean(0)
    Fh = (P @ c.float()) - c.float()
    am = P.argmax(-1)
    out = {"argmax_self_frac": float((am == torch.arange(N, device=A.device)).float().mean())}
    src = c[None, :, :].float() + 0 * Fh[:, None, :]                             # c_{q'} broadcast
    tgt = (c.float() + F_gt.float())[:, None, :]                                 # ground-truth source location per query
    near = (torch.linalg.norm(src - tgt, dim=-1) <= hit_radius).float()         # [N, N]
    out["hit_mass_median"] = float(((P * near).sum(-1))[mask].median()) if mask.any() else None
    ent = -(P * torch.log2(P.clamp_min(1e-12))).sum(-1)
    out["entropy_bits_mean"] = float(ent.mean()); out["uniform_bits"] = float(torch.log2(torch.tensor(float(N))))
    centre = c.float().mean(0)
    v = (centre - c.float())[mask]; f = Fh[mask]
    if mask.sum() > 3:
        vv = (v - v.mean(0)).flatten(); ff = (f - f.mean(0)).flatten()
        out["centre_pull_corr"] = float((vv * ff).sum() / (vv.norm() * ff.norm() + 1e-12))
    else:
        out["centre_pull_corr"] = None
    pe = []
    for h in range(heads):
        Fhh = (A[h] @ c.float()) - c.float()
        pe.append(float(torch.linalg.norm(Fhh[mask] - F_gt.float()[mask], dim=-1).median()) if mask.any() else None)
    out["per_head_epe"] = pe
    out["fhat_median_magnitude"] = float(torch.linalg.norm(Fh[mask], dim=-1).median()) if mask.any() else None
    return out


def windowed_expected_displacement(A: torch.Tensor, c: torch.Tensor, radius: float, heads=None) -> torch.Tensor:
    """Windowed soft-argmax (reviewer 2026-10-07g, the G1-reported fix for a heavy-tailed attention): per query, the attention
    mass within `radius` patches of the most-attended key is renormalised and the expected location taken over that window
    only. The argmax itself is not used as the answer (it is quantised to a patch); the window removes the far-field mass that
    drags the plain expectation toward the lattice centre. A [heads,N,N], c [N,2]; returns F_hat [N,2]."""
    Ah = A if heads is None else A[list(heads)]
    P = Ah.mean(0)
    am = P.argmax(-1)                                                            # [N] most-attended key per query
    d = torch.cdist(c[am].float(), c.float())                                    # [N,N] distance from the argmax key to every key
    W = (d <= radius).float()
    Pw = P * W
    Pw = Pw / Pw.sum(-1, keepdim=True).clamp_min(1e-12)
    return (Pw @ c.float()) - c.float()


def select_heads(per_head_epe: list, k: int) -> list:
    """Top-k heads by per-head EPE, measured on a DIFFERENT clip from the one being graded (prohibition 7: a head subset only
    with a G1 report; choosing heads on the graded clip would be tuning to it)."""
    ranked = sorted((v, i) for i, v in enumerate(per_head_epe) if v is not None)
    return [i for _, i in ranked[:k]]
