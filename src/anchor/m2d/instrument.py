"""Rank correlation between an estimated disparity and a control disparity (SKILL §2; shared with
m2_e's alpha machinery). Spearman, not Pearson: a monocular estimate is defined up to an unknown
monotone map, and the question here is ordering, i.e. a sign."""
from __future__ import annotations

import hashlib

import numpy as np


def _ranks(x):
    """Average ranks (ties share their mean rank), O(n log n)."""
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    xs = x[order]
    # average over runs of equal values
    starts = np.flatnonzero(np.concatenate([[True], xs[1:] != xs[:-1]]))
    ends = np.concatenate([starts[1:], [len(xs)]])
    for a, b in zip(starts, ends):
        if b - a > 1:
            r[order[a:b]] = 0.5 * (a + b - 1)
    return r


def spearman(a, b, mask=None, max_n=400_000, seed=0) -> float:
    """Spearman rho over the mask (finite values only); subsampled deterministically above max_n."""
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    assert a.shape == b.shape, (a.shape, b.shape)
    m = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m &= np.asarray(mask, bool).ravel()
    idx = np.flatnonzero(m)
    assert len(idx) >= 16, "too few valid pixels for a rank correlation"
    if len(idx) > max_n:
        idx = np.random.default_rng(seed).choice(idx, max_n, replace=False)
    ra, rb = _ranks(a[idx]), _ranks(b[idx])
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float((ra @ rb) / den) if den > 0 else float("nan")


def checksum(arr) -> str:
    """sha256 of an array's bytes with its dtype and shape — logged at the generator call site so a
    preprocessor slipping back into the path is visible (SKILL §4, reviewer addition A3)."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()[:16]
