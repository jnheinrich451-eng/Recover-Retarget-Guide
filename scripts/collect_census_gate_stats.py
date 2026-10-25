#!/usr/bin/env python
"""Reviewer 2026-10-18: the A-G2b statistic of EVERY census clip, passes beside refusals (passes: cache/stage_a/<clip>/qc.json; refusals:
viz/stage_a/<clip>/qc_failed.json), so the thesis can say whether the threshold cuts through one population. CPU, seconds, read-only.

    MT_ROOT=/content/drive/MyDrive/motion_transfer_exp python scripts/collect_census_gate_stats.py
Writes <root>/reports/census/stage_a_gates.{json,md}.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


def main():
    root = Path(os.environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer_exp"))
    cen = yaml.safe_load(Path("configs/census.yaml").read_text(encoding="utf-8"))
    rows = []
    for c in cen["clips"]:
        clip = c["id"]; src = None
        for p_, kind in ((root / "cache" / "stage_a" / clip / "qc.json", "pass"), (root / "viz" / "stage_a" / clip / "qc_failed.json", "refused")):
            if p_.exists():
                src = (json.loads(p_.read_text(encoding="utf-8")), kind); break
        if src is None:
            rows.append({"clip": clip, "pool": c["pool"], "category": c["category"], "stage_a": "no record"}); continue
        q, kind = src; g = q.get("gates", {})
        rows.append({"clip": clip, "pool": c["pool"], "category": c["category"], "stage_a": kind,
                     "gates": {k: {kk: v.get(kk) for kk in ("median_px", "median_rel", "value", "tau", "pass") if isinstance(v, dict) and kk in v} for k, v in g.items() if isinstance(v, dict)},
                     "a_g2b_median_px": (g.get("A-G2b") or {}).get("median_px"), "report_only_gates": q.get("report_only_gates")})
    rep = root / "reports" / "census"; rep.mkdir(parents=True, exist_ok=True)
    (rep / "stage_a_gates.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    have = [r for r in rows if r.get("a_g2b_median_px") is not None]
    md = ["# Stage A gate statistics, every census clip", "", "| clip | pool | target | Stage A | A-G2b median px |", "|---|---|---|---|---|"]
    for r in sorted(have, key=lambda r: r["a_g2b_median_px"]):
        md.append(f"| {r['clip']} | {r['pool']} | {r['category']} | {r['stage_a']} | {r['a_g2b_median_px']:.2f} |")
    pv = np.array([r["a_g2b_median_px"] for r in have if r["stage_a"] == "pass"]); rv = np.array([r["a_g2b_median_px"] for r in have if r["stage_a"] == "refused"])
    f = lambda a: f"n={len(a)}, min {a.min():.2f}, median {np.median(a):.2f}, max {a.max():.2f}" if len(a) else "n=0"   # noqa: E731
    md += ["", f"A-G2b on the passes (Stage A pass): {f(pv)}", f"A-G2b on the refused: {f(rv)}",
           "", "Reading (reviewer): if the passes run continuously up to 4 px and the refusals continue from 4 px, the threshold cuts one population; if the passes cluster well below 4, the refused clips are a separate group."]
    (rep / "stage_a_gates.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[-4:-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
