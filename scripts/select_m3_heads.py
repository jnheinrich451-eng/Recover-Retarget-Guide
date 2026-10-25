#!/usr/bin/env python
"""Pick a head subset per block from a DIAGNOSED readout json of one clip (top-k by per-head EPE, median over the sigma grid), to be
applied to a DIFFERENT clip with `run_m3_s2_readout.py --readout heads --heads-from <out>`. Prohibition 7: a head subset only with a
G1 report, never selected on the clip being graded.

    python scripts/select_m3_heads.py --from measurements/m3_s2/readout_fp8_F81.json --k 4 --out measurements/m3_s2/heads_from_265.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generate.m3.readout import select_heads  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = json.loads(Path(args.src).read_text(encoding="utf-8"))
    clips = sorted({r["clip"] for r in rows}); assert len(clips) == 1, "select on ONE clip"
    out = {}
    for b in sorted({r["block"] for r in rows}):
        per = [r["diagnostics_mid_frame"]["per_head_epe_cross"] for r in rows if r["block"] == b and r.get("diagnostics_mid_frame")]
        assert per, f"block {b}: no diagnostics (run the readout with --diagnose)"
        med = np.nanmedian(np.array([[np.nan if v is None else v for v in p] for p in per], float), axis=0)
        out[str(b)] = select_heads([None if not np.isfinite(v) else float(v) for v in med], args.k)
    Path(args.out).write_text(json.dumps({"selected_on": clips[0], "k": args.k, "rule": "top-k heads by per-head EPE, median over sigma", **out}, indent=1), encoding="utf-8")
    print(json.dumps(out)); print(f"-> {args.out} (apply only to clips other than {clips[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
