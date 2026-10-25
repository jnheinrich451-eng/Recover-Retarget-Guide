"""m2_e QC assembly (SKILL m2_e v2.2 §7-8). Labelled pairs run and are reported, never in accepted-set statistics (test 8)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

QC_SCHEMA = "m2e_qc_v1"


def accepted_summary(records: list) -> dict:
    """Statistics over ACCEPTED pairs only. Labelled pairs are listed by name so their absence is visible, not silent."""
    acc = [r for r in records if r.get("disposition") == "accepted"]
    lab = [r for r in records if r.get("disposition") == "labelled"]
    exp = [r for r in records if r.get("disposition") == "experiment"]          # the experiment set: admitted by gates alone, no human disposition (plan v1)
    assert all(r.get("disposition") in ("accepted", "labelled", "experiment") for r in records), "every record needs a disposition"
    def stat(key):
        v = np.array([r[key] for r in acc if r.get(key) is not None], float)
        return {"n": int(len(v)), "median": float(np.median(v)) if len(v) else None, "min": float(v.min()) if len(v) else None, "max": float(v.max()) if len(v) else None}
    return {"accepted_pairs": [f"{r['clip']}/{r['category']}" for r in acc], "labelled_pairs_excluded": [f"{r['clip']}/{r['category']}" for r in lab],
            "experiment_pairs_excluded": [f"{r['clip']}/{r['category']}" for r in exp],
            "alpha": stat("alpha"), "delta_deg_abs": {**stat("delta_deg_abs")}, "r2_iou_vs_anchor_mask": stat("r2_iou_vs_anchor_mask")}


def write_qc(out_dir, rec: dict) -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "qc_m2e.json").write_text(json.dumps({"schema": QC_SCHEMA, **rec}, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    return out / "qc_m2e.json"
