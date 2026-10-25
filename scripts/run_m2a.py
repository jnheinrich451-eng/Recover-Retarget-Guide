#!/usr/bin/env python
"""m2_a runner: B.5 + Stage C + bundle -> cache/m2_a/<clip>/{stage0.npz, qc_m2a.json}.
CPU, milliseconds per clip. Refusal is a recorded outcome; only an upstream
hash mismatch exits non-zero for that clip.

    python scripts/run_m2a.py --clip-id <id> [...] | --curated <tier> | --verdict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml                                              # noqa: E402
from src.preprocess.bundle import read_bundle            # noqa: E402
from src.preprocess.config import load_config            # noqa: E402
from src.preprocess.stage_c import finalize              # noqa: E402
from src.anchor import m2a                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", nargs="*", default=[])
    ap.add_argument("--curated", default=None, help="tier in configs/curated_clips.yaml | all")
    ap.add_argument("--verdict", action="store_true", help="configs/b5_verdict_clips.yaml")
    ap.add_argument("--config", default="configs/stage_a.yaml")
    ap.add_argument("--config-m2a", default="configs/m2a.yaml")
    args = ap.parse_args()
    if args.curated:
        cur = yaml.safe_load(Path("configs/curated_clips.yaml").read_text(encoding="utf-8"))
        tiers = list(cur) if args.curated == "all" else [args.curated]
        args.clip_id += [c for t in tiers for c in cur[t]]
    if args.verdict:
        v = yaml.safe_load(Path("configs/b5_verdict_clips.yaml").read_text(encoding="utf-8"))
        args.clip_id += [c for t in v.values() for c in t]
    assert args.clip_id, "give --clip-id ..., --curated <tier>, or --verdict"

    cfg = load_config(args.config)
    raw = yaml.safe_load(Path(args.config_m2a).read_text(encoding="utf-8"))
    drive = str(cfg["paths"]["drive_root"])
    paths = {k: v.replace("${paths.drive_root}", drive) for k, v in raw["paths"].items()}
    th = cfg["thresholds"]
    chash = m2a.m2a_config_hash(th, raw["m2a"])
    assert raw["m2a"]["b5_schema"] == m2a.B5_SCHEMA
    print(f"m2_a config hash {chash}; thresholds {{{', '.join(f'{k}={th.get(k)}' for k in m2a.LOCK_KEYS)}}}")

    rc = 0
    for clip in args.clip_id:
        try:
            b = read_bundle(cfg, clip)
            cdir = Path(cfg["paths"]["stage_c_cache"]) / clip
            M, present, t_c = finalize.read_mask(cdir)
            qc_c = json.loads((cdir / "qc_c.json").read_text(encoding="utf-8"))
            b5_arr, b5_qc = m2a.load_b5(Path(paths["b5_cache"]), clip)
            arrays, qc = m2a.run_clip(
                clip, b["G"].astype(np.float64), b["K"].astype(np.float64), M, present,
                int(t_c), b["provenance"]["config_hash"], qc_c, b5_arr, b5_qc, th, chash)
        except m2a.HashMismatch as e:
            print(f"{clip}: HASH MISMATCH — {e}")
            rc = 1
            continue
        out = m2a.write_stage0(Path(paths["out_dir"]), clip, arrays, qc)
        m2a.read_stage0(out)          # the consumer must be able to open what we wrote
        go = qc["gate_outcomes"]
        print(f"{clip}: t_a={qc['t_a']} admitted={qc['admitted']} heading={qc['heading_defined']} "
              f"up={qc['up_defined']}({qc['up_source']}) kappa={qc['kappa_basis']:.3f} "
              f"area={qc['mask_area_px']} trunc={qc['mask_border_frac']:.3f} "
              f"gates={{{', '.join(f'{k}:{v}' for k, v in go.items())}}}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
