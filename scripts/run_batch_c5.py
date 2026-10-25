#!/usr/bin/env python
"""Stage C.5 batch driver: a curated tier (or explicit clips) -> one
run_stage_c5.py subprocess per clip (isolated failures; the D4RT model
reloads per clip, same cost pattern as the Stage A batch), skipping clips
whose qc_c5 already matches the current config hash. Resumable.

    python scripts/run_batch_c5.py --curated tier_a_good [--force]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml                                    # noqa: E402
from src.preprocess.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", default="tier_a_good")
    ap.add_argument("--clip-id", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config("configs/stage_a.yaml")
    clips = list(args.clip_id)
    if not clips:
        cur = yaml.safe_load(Path("configs/curated_clips.yaml")
                             .read_text(encoding="utf-8"))
        tiers = list(cur) if args.curated == "all" else [args.curated]
        clips = [c for t in tiers for c in cur[t]]
    out_root = Path(str(cfg["paths"]["drive_root"]) + "/cache/stage_c5")
    rows, ok, fail = [], 0, 0
    for clip in clips:
        qc = out_root / f"{clip}_qc_c5.json"
        if qc.exists() and not args.force:
            try:
                if json.loads(qc.read_text(encoding="utf-8")).get(
                        "config_hash") == cfg["_meta"]["config_hash"]:
                    rows.append(f"- {clip}: cached")
                    ok += 1
                    continue
            except Exception:  # noqa: BLE001
                pass
        r = subprocess.run([sys.executable, "scripts/run_stage_c5.py",
                            "--clip-id", clip], capture_output=True, text=True)
        if r.returncode == 0:
            last = [l for l in r.stdout.strip().splitlines() if l.startswith(clip)]
            rows.append(f"- {last[-1] if last else clip + ': OK'}")
            ok += 1
        else:
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:]
            rows.append(f"- {clip}: FAILED — {tail[0] if tail else '?'}")
            fail += 1
        print(rows[-1])
    out = Path(cfg["paths"]["stage_c_viz"]) / "batch_stage_c5.md"
    out.write_text(f"# Stage C.5 batch ({ok} ok, {fail} failed / {len(clips)})\n\n"
                   + "\n".join(rows), encoding="utf-8")
    print(f"\nsummary -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
