"""Stage B.5 contract writer (SKILL v1.1 §3)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_b5(out_dir, clip_id, rec, arrays, hashes):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{clip_id}.npz",
                        **{k: np.asarray(v) for k, v in arrays.items()})
    (out / f"{clip_id}_qc_b5.json").write_text(
        json.dumps({**rec, "hashes": hashes}, indent=2, default=float),
        encoding="utf-8")
    return out / f"{clip_id}.npz"
