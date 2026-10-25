"""Guidance ablations with the evaluation metrics, on the development pairs (registered in
docs/ablation_guidance_sheet_2026-10-24.md before any run). Every setting is run on the five gated pairs of the batch of
record (33 frames, seed 0) unless noted, and is read against the pair's A0 and A3 of record, which already exist:

  --noobj      the vehicle arm without the object render (vehicle pairs only): does the object render carry the object channel?
  --gammas G.. the A3 arm with --gamma G (gamma_obj stays at the lock): why gamma = 1?
  --euler      the A3 arm with the first-order Euler scheduler instead of UniPC: the sampler row of Table 4.11 with numbers
  --coff       the A0 control with UniPC's corrector off at the corrected steps, no correction: the scheduler switch alone
  --hardmask   the A3 arm with the hard latent mask instead of the soft one
  --skyout     the A3 arm with the sky left out of the correction mask

A setting whose render does not exist yet (vehicles without the object render; any arm without the sky) is rendered first
with scripts/render_m3_target.py, as the batch's Phase 1 would. Same form as scripts/run_m3_ablation_tg.py: resumable (an existing video is kept), every new video graded with the pipeline
grader (measure_m3_a0.py), one row per (pair, setting) appended to <root>/reports/m3_ablation/guidance.json (never replaced),
and the videos exported beside the pair's A0 and A3 of record under <root>/reports/m3_ablation/export_guidance/ with an
index, for the evaluation to score by content hash with the Chapter 4 metrics (as the window ablation, v2). Nothing of
record is regenerated; the lock is not edited. Development pairs only: never the test set.

    python scripts/run_m3_ablation_guidance.py --noobj --euler --coff --hardmask --skyout --gammas 0.5 0.25
    python scripts/run_m3_ablation_guidance.py --noobj --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    lines = []
    for ln in p.stdout:
        print("    " + ln.rstrip(), flush=True); lines.append(ln)
    return p.wait(), "".join(lines[-40:])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=os.environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--videos-dir", default=os.environ.get("MT_DATA", "/content/data") + "/videos")
    ap.add_argument("--config", default="configs/m3_batch.yaml", help="the batch of record: its gated pairs, frames, arms")
    ap.add_argument("--noobj", action="store_true"); ap.add_argument("--gammas", nargs="*", type=float, default=None)
    ap.add_argument("--euler", action="store_true"); ap.add_argument("--coff", action="store_true")
    ap.add_argument("--hardmask", action="store_true"); ap.add_argument("--skyout", action="store_true")
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    assert any([args.noobj, args.gammas, args.euler, args.coff, args.hardmask, args.skyout]), "nothing to run"
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); frames = int(cfg["frames"])
    pairs = [p for p in cfg["pairs"] if not p.get("camera_gate_failed") and (not args.pairs or p["pair"] in args.pairs)]
    lock = yaml.safe_load((ROOT / "configs" / "calibration.lock.yaml").read_text(encoding="utf-8"))["m3"]["guidance"]
    root = Path(args.drive_root); rep = root / "reports" / "m3_ablation"; rep.mkdir(parents=True, exist_ok=True)
    jf = rep / "guidance.json"; rows = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else []
    done = {(r["pair"], r["setting"]) for r in rows if r.get("graded_rc") == 0}

    # settings per pair: (label, arm, flags for S4, output stem). Stems follow run_m3_s4_guided.py's naming.
    def settings(cat, rec, all=False):
        out = []
        if (args.noobj or all) and cat == "vehicle":
            fl = [f for f in rec if f != "--object"]; out.append(("noobj", "A3", fl, video_name("A3", args.seed, frames, fl)))
        for g in (args.gammas or ([0.5, 0.25] if all else [])):
            assert 0.0 < g < 1.0, f"gamma {g} is not an ablation value (the lock holds {lock['gamma_render']})"
            out.append((f"gamma{g:g}", "A3", rec + ["--gamma", f"{g:g}"], f"{video_name('A3', args.seed, frames, rec)}_g{g:g}"))
        if args.euler or all:
            # run_m3_s4_guided.py names the scheduler tag BEFORE the mask tags: A3_seed0_rec_F33[_obj_ow0.75]_euler[_sm][_sky]
            euler_stem = f"A3_seed{args.seed}_rec_F{frames}" + ("_obj_ow0.75" if "--object" in rec else "") + "_euler" + ("_sm" if "--soft-mask" in rec else "") + ("_sky" if "--sky-mask" in rec else "")
            out.append(("euler", "A3", rec + ["--euler"], euler_stem))
        if args.coff or all:
            out.append(("coff", "A0", ["--corrector-off-control"], f"{video_name('A0', args.seed, frames, [])}_coff"))
        if args.hardmask or all:
            fl = [f for f in rec if f != "--soft-mask"]; out.append(("hardmask", "A3", fl, video_name("A3", args.seed, frames, fl)))
        if args.skyout or all:
            fl = [f for f in rec if f != "--sky-mask"]; out.append(("skyout", "A3", fl, video_name("A3", args.seed, frames, fl)))
        return out

    for p in pairs:
        pair = p["pair"]; clip, cat = pair.split("/"); d = root / "cache" / "m3" / clip / cat
        rec = list(cfg["arms"][f"A3_{cat}"])
        a0 = f"{video_name('A0', args.seed, frames, [])}.mp4"; a3 = f"{video_name('A3', args.seed, frames, rec)}.mp4"
        for label, arm, fl, stem in settings(cat, rec):
            if (pair, label) in done:
                print(f"[ablation] {pair} {label}: already graded; kept", flush=True); continue
            vp = d / f"{stem}.mp4"
            if vp.exists():
                print(f"[ablation] {pair} {label}: {vp.name} exists; kept", flush=True)
            else:
                if arm == "A3":   # the render this setting reads; build it if the batch of record did not (its Phase 1 made only the renders of record)
                    suf = ("_obj" if "--object" in fl else "") + ("_sky" if "--sky-mask" in fl else "") + ("_nodyn" if "--nodyn-mask" in fl else "")
                    rp_ = d / f"render{suf}.npz"
                    if not rp_.exists():
                        rflags = (["--object"] if "--object" in fl else []) + (["--sky-in-mask"] if "--sky-mask" in fl else [])
                        rc, tail = sh([PY, "scripts/render_m3_target.py", "--drive-root", args.drive_root, "--pairs", pair, "--bridge", "rec"] + rflags, args.dry_run)
                        if rc != 0:
                            rows.append({"pair": pair, "setting": label, "render": rp_.name, "rc": rc, "tail": tail[-800:], "time": time.strftime("%Y-%m-%d %H:%M")})
                            if not args.dry_run: jf.write_text(json.dumps(rows, indent=1), encoding="utf-8")
                            print(f"[ablation] {pair} {label}: RENDER FAILED rc {rc} (recorded)", flush=True); continue
                rc, tail = sh([PY, "scripts/run_m3_s4_guided.py", "--drive-root", args.drive_root, "--pair", pair, "--seed", str(args.seed),
                               "--frames", str(frames), "--arm", arm, "--ablation-label", label] + fl, args.dry_run)
                if rc != 0:
                    rows.append({"pair": pair, "setting": label, "video": vp.name, "rc": rc, "tail": tail[-800:], "time": time.strftime("%Y-%m-%d %H:%M")})
                    if not args.dry_run: jf.write_text(json.dumps(rows, indent=1), encoding="utf-8")
                    print(f"[ablation] {pair} {label}: FAILED rc {rc} (recorded)", flush=True); continue
            vids = [vp.name] + ([a0] if (d / a0).exists() else [])
            rc, _ = sh([PY, "scripts/measure_m3_a0.py", "--drive-root", args.drive_root, "--videos-dir", args.videos_dir, "--pairs", pair, "--force",
                        "--source-frames", str(frames), "--videos"] + vids, args.dry_run)
            rows.append({"pair": pair, "setting": label, "arm": arm, "flags": fl, "lock_gamma": lock["gamma_render"], "lock_gamma_obj": lock["gamma_obj"],
                         "video": vp.name, "a0": a0, "a3_record": a3, "seed": args.seed, "frames": frames, "graded_rc": rc,
                         "time": time.strftime("%Y-%m-%d %H:%M")})
            if not args.dry_run:
                jf.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    # export beside the pair's A0 and A3 of record (nothing regenerated); every setting whose video exists, whatever this run was given
    ex = rep / "export_guidance"; index = []
    for p in pairs:
        pair = p["pair"]; clip, cat = pair.split("/"); d = root / "cache" / "m3" / clip / cat; pdir = ex / f"{clip}_{cat}"
        rec = list(cfg["arms"][f"A3_{cat}"])
        names = [(f"{video_name('A0', args.seed, frames, [])}.mp4", "backbone"), (f"{video_name('A3', args.seed, frames, rec)}.mp4", "A3 of record")]
        names += [(f"{stem}.mp4", label) for label, _, _, stem in settings(cat, rec, all=True)]
        for n_, role in names:
            if (d / n_).exists():
                if not args.dry_run:
                    pdir.mkdir(parents=True, exist_ok=True); shutil.copy2(d / n_, pdir / n_)
                index.append({"pair": pair, "video": f"{clip}_{cat}/{n_}", "role": role, "seed": args.seed, "frames": frames})
        src_ = root / "cache" / "m2_d" / clip / cat / "anchor.png"
        if src_.exists() and not args.dry_run:
            pdir.mkdir(parents=True, exist_ok=True); shutil.copy2(src_, pdir / "anchor.png")
    if not args.dry_run:
        (ex / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"[ablation] {len(rows)} rows -> {jf}; export {len(index)} videos -> {ex}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
