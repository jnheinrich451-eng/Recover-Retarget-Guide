#!/usr/bin/env python
"""Stage C batch driver: every clip with Stage B outputs -> run_stage_c.py,
one subprocess per clip (isolated failures; ~6 s SAM2 reload each), skipping
clips whose qc_c already matches the current config hash. Resumable.

    python scripts/run_batch_c.py [--calibrate] [--force]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default="configs/stage_a.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bdir = Path(cfg["paths"]["stage_b_cache"])
    cdir = Path(cfg["paths"]["stage_c_cache"])
    clips = sorted(p.name[:-len("_qc_b.json")]
                   for p in bdir.glob("*_qc_b.json") if "__" not in p.name)
    rows, done, fail = [], 0, 0
    for clip in clips:
        qc = cdir / clip / "qc_c.json"
        if qc.exists() and not args.force:
            try:
                if json.loads(qc.read_text(encoding="utf-8")).get(
                        "config_hash") == cfg["_meta"]["config_hash"]:
                    rows.append(f"- {clip}: cached")
                    done += 1
                    continue
            except Exception:  # noqa: BLE001 — malformed cache -> re-run
                pass
        cmd = [sys.executable, "scripts/run_stage_c.py", "--clips", clip]
        if args.calibrate:
            cmd.append("--calibrate")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            rows.append(f"- {clip}: OK")
            done += 1
        else:
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:]
            rows.append(f"- {clip}: FAILED — {tail[0] if tail else '?'}")
            fail += 1
        print(rows[-1])
    out = Path(cfg["paths"]["stage_c_viz"]) / "batch_stage_c.md"
    out.write_text(f"# Stage C batch ({done} ok, {fail} failed / "
                   f"{len(clips)})\n\n" + "\n".join(rows), encoding="utf-8")
    print(f"\nsummary -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
