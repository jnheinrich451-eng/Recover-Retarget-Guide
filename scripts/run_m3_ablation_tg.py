"""The one new ablation of D5: the guidance window T_g at two settings on the six pairs' root (the five gated pairs of the batch of record,
33 frames, seed 0, the A3 arm of record per category, the lock's window replaced by --bg-window HI LO). Phase 2, A100, ~2 h for two settings.
Resumable: a video that exists is kept. Grades every new video against the source with the pipeline grader (measure_m3_a0.py) and appends
one row per (pair, window) to <root>/reports/m3_ablation/tg_window.json (rule 24). The two settings are the OWNER's (registered in
docs/plan_experiment_v1.md before the run); this script takes them as arguments and chooses nothing.

    python scripts/run_m3_ablation_tg.py --windows 0.85,0.75 0.85,0.45          # two settings: HI,LO each
    python scripts/run_m3_ablation_tg.py --windows 0.85,0.75 --pairs 000000000265.2.005/vehicle --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3_batch import video_name   # noqa: E402

PY = sys.executable


def sh(cmd, dry):
    print("[ablation] $ " + " ".join(map(str, cmd)), flush=True)
    if dry:
        return 0, ""
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    lines = []
    for ln in p.stdout:
        print("    " + ln.rstrip(), flush=True); lines.append(ln)
    return p.wait(), "".join(lines[-40:])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=os.environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--videos-dir", default=os.environ.get("MT_DATA", "/content/data") + "/videos")
    ap.add_argument("--config", default="configs/m3_batch.yaml", help="the batch of record: its gated pairs, frames, arms")
    ap.add_argument("--windows", nargs="+", required=True, help="HI,LO per setting in sigma, e.g. 0.85,0.75 0.85,0.45 (the owner's registered settings)")
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); frames = int(cfg["frames"])
    pairs = [p for p in cfg["pairs"] if not p.get("camera_gate_failed") and (not args.pairs or p["pair"] in args.pairs)]
    windows = []
    for w in args.windows:
        hi, lo = (float(v) for v in w.split(",")); assert 0 < lo < hi <= 1, w; windows.append((hi, lo))
    root = Path(args.drive_root); rep = root / "reports" / "m3_ablation"; rep.mkdir(parents=True, exist_ok=True)
    jf = rep / "tg_window.json"; rows = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else []
    lock_win = yaml.safe_load((ROOT / "configs" / "calibration.lock.yaml").read_text(encoding="utf-8"))["m3"]["guidance"]["T_g_sigma"]
    for p in pairs:
        pair = p["pair"]; clip, cat = pair.split("/"); d = root / "cache" / "m3" / clip / cat
        flags = list(cfg["arms"][f"A3_{cat}"])
        a0 = f"{video_name('A0', args.seed, frames, [])}.mp4"
        for hi, lo in windows:
            stem = f"{video_name('A3', args.seed, frames, flags)}_bw{lo:g}"; vp = d / f"{stem}.mp4"
            if vp.exists():
                print(f"[ablation] {pair} window [{hi}, {lo}]: {vp.name} exists; kept", flush=True)
            else:
                rc, tail = sh([PY, "scripts/run_m3_s4_guided.py", "--drive-root", args.drive_root, "--pair", pair, "--seed", str(args.seed), "--frames", str(frames), "--arm", "A3"] + flags + ["--bg-window", str(hi), str(lo)], args.dry_run)
                if rc != 0:
                    rows.append({"pair": pair, "window": [hi, lo], "video": vp.name, "rc": rc, "tail": tail[-800:], "time": time.strftime("%Y-%m-%d %H:%M")})
                    if not args.dry_run: jf.write_text(json.dumps(rows, indent=1), encoding="utf-8")
                    print(f"[ablation] {pair} window [{hi}, {lo}]: FAILED rc {rc} (recorded)", flush=True); continue
            vids = [vp.name] + ([a0] if (d / a0).exists() else [])
            rc, _ = sh([PY, "scripts/measure_m3_a0.py", "--drive-root", args.drive_root, "--videos-dir", args.videos_dir, "--pairs", pair, "--force", "--source-frames", str(frames), "--videos"] + vids + ["source"], args.dry_run)
            rows.append({"pair": pair, "window": [hi, lo], "lock_window": lock_win, "video": vp.name, "a0": a0, "flags": flags, "seed": args.seed, "frames": frames, "graded_rc": rc, "time": time.strftime("%Y-%m-%d %H:%M")})
            if not args.dry_run:
                jf.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    # export for the scoring agent: the ten videos beside the pair's A0 and lock-window A3 of record, with an index (nothing regenerated)
    import shutil
    ex = rep / "export"; index = []
    for p in pairs:
        pair = p["pair"]; clip, cat = pair.split("/"); d = root / "cache" / "m3" / clip / cat; pdir = ex / f"{clip}_{cat}"; pdir.mkdir(parents=True, exist_ok=True)
        flags = list(cfg["arms"][f"A3_{cat}"])
        names = [f"{video_name('A0', args.seed, frames, [])}.mp4", f"{video_name('A3', args.seed, frames, flags)}.mp4"] + [f"{video_name('A3', args.seed, frames, flags)}_bw{lo:g}.mp4" for _, lo in windows]
        for n_ in names:
            if (d / n_).exists():
                if not args.dry_run: shutil.copy2(d / n_, pdir / n_)
                index.append({"pair": pair, "video": f"{clip}_{cat}/{n_}", "arm": "A0" if n_.startswith("A0") else "A3", "window": (lock_win if "_bw" not in n_ else [w for w in windows if n_.endswith(f"_bw{w[1]:g}.mp4")][0]),
                              "seed": args.seed, "frames": frames, "role": "lock window (of record)" if ("_bw" not in n_ and n_.startswith("A3")) else ("backbone" if n_.startswith("A0") else "ablation")})
        for extra in ("anchor.png",):
            src_ = root / "cache" / "m2_d" / clip / cat / extra
            if src_.exists() and not args.dry_run: shutil.copy2(src_, pdir / extra)
    if not args.dry_run:
        (ex / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"[ablation] {len(rows)} rows -> {jf}; export {len(index)} videos -> {ex} (the scoring agent scores the ablation videos beside A0 and the lock-window A3 of the same pair)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
