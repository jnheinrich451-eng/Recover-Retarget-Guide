#!/usr/bin/env python
"""Module 3 batch driver (configs/m3_batch.yaml): the same pipeline of record on every handed-over pair, in the order the modules run.

  phase1   CPU runtime: lift (run_m3_sint.py), contact (check_m3_contact.py), geometry (run_m3_sint2.py), dynamic mask where the
           config asks for it, renders (render_m3_target.py) per arm configuration.  Idempotent: a step whose outputs exist is skipped
           unless --force; a failing step is recorded in the manifest and the pair goes on (a failing gate is a finding).
  phase2   GPU runtime: run_m3_s4_guided.py for every pair x arm x seed of the config (the lock's scheduler, 33 frames).
  grade    GPU runtime: measure_m3_a0.py on every video of the pair plus 'source' (--source-frames = the batch's frames), the
           mask-centroid arbiter on the same videos, both with --force (the batch is one grading pass).
  pack     CPU: the evaluation pack per pair (reports/m3_batch/eval_pack/<clip>_<cat>/) for the experiment agent: geometry_eval.npz
           (target camera E_full per frame, E_l, K_wan, c_tar, t_a, src_frames, beta, s_delta), runs.json (every video with its arm,
           seed, config and the gate flags), and the pointers to the source clip and the Stage C / C.5 caches.

Every phase appends to reports/m3_batch/manifest.json, one entry per attempt, written after each step (rule 24: never replaced). --dry-run prints the
commands. Nothing here chooses a tunable: arms and flags come from the config, thresholds from the lock.

    python scripts/run_m3_batch.py phase1 [--pairs ...] [--dry-run]
    python scripts/run_m3_batch.py phase2 --seeds 0 1
    python scripts/run_m3_batch.py grade
    python scripts/run_m3_batch.py pack
    python scripts/run_m3_batch.py status      # read-only: what exists, how old, stale or not, and what the manifest says
    python scripts/run_m3_batch.py export      # every video + source + anchor + index.json -> reports/m3_batch/export/, zipped
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def sh(cmd, dry, log):
    line = " ".join(str(c) for c in cmd)
    print(f"[batch] $ {line}", flush=True)
    if dry:
        return 0, ""
    t0 = time.time(); lines = []
    import os
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    for ln in p.stdout:                                                              # live: every line of the child shows in the cell as it comes
        print("    " + ln.rstrip(), flush=True); lines.append(ln)
    rc = p.wait(); out = "".join(lines[-200:])
    log.append({"cmd": line, "rc": rc, "wall_s": round(time.time() - t0, 1), "tail": out[-1500:]})
    print(f"[batch] {'done' if rc == 0 else f'FAILED rc {rc}'} in {time.time() - t0:.0f} s", flush=True)
    return rc, out


GEOMETRY_CONTRACT = ("bridge_run", "object_path_source", "beta", "s_delta", "c_tar", "E_full")


def geometry_status(d):
    """'absent' | 'off-record: <missing keys>' | 'ok' — the cache's contract, not its existence, decides whether a step is done."""
    gp = d / "geometry.npz"
    if not gp.exists():
        return "absent"
    with np.load(gp) as z:
        missing = [k for k in GEOMETRY_CONTRACT if k not in z.files]
    return f"off-record: missing {missing}" if missing else "ok"



def tbl(table: dict, key: str):
    """A table keyed on the thesis's categories (vehicle / human) serves a registered subject through its kind:
    A3_bear -> A3_vehicle when bear is rigid; bear_nodyn -> human_nodyn when bear is of the human kind."""
    if key in table:
        return table[key]
    from src.anchor import subjects
    parts = key.split("_")
    for cat in sorted(subjects.load(), key=len, reverse=True):
        if cat in parts:
            k2 = "_".join(subjects.default_name(cat) if part == cat else part for part in parts)
            if k2 in table:
                return table[k2]
    raise KeyError(f"no entry {key!r} in the batch config, and no registered subject maps it to one of {sorted(table)}")

def video_name(arm, seed, frames, flags):
    """The sampler's output stem for an arm's flags (mirrors run_m3_s4_guided.py's naming of the flags the config uses)."""
    base = f"{arm}_seed{seed}_rec_F{frames}"
    if "--object" in flags:
        base += "_obj_ow0.75"                                                        # the lock's object window is written into the name
    if "--soft-mask" in flags:
        base += "_sm"
    if "--sky-mask" in flags:
        base += "_sky"
    if "--nodyn-mask" in flags:
        base += "_nodyn"
    return base


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["phase1", "phase2", "grade", "pack", "export", "status"])
    ap.add_argument("--config", default="configs/m3_batch.yaml")
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--pairs", nargs="*", default=None, help="subset of the config's pairs")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="subset of the config's seeds")
    ap.add_argument("--videos-dir", default=__import__("os").environ.get("MT_DATA", "/content/data") + "/videos")
    ap.add_argument("--force", action="store_true", help="redo steps whose outputs exist; replace manifest entries")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    frames, seeds = int(cfg["frames"]), [int(s) for s in (args.seeds or cfg["seeds"])]
    pairs = [p for p in cfg["pairs"] if not args.pairs or p["pair"] in args.pairs]
    for pr in (args.pairs or []):                                                   # a pair outside the config (a reader's clip): a plain entry, labels from the registry
        if pr not in {p["pair"] for p in pairs}:
            from src.anchor import subjects
            lab = subjects.arbiter_label(pr.split("/")[1]) or "person"
            pairs.append({"pair": pr, "label_source": lab, "label_target": lab, "note": "outside the batch config: entry synthesised by --pairs"})
            print(f"[batch] {pr}: not in {args.config}; running it as a plain entry (labels {lab})", flush=True)
    root = Path(args.drive_root); rep = root / "reports" / "m3_batch"; rep.mkdir(parents=True, exist_ok=True)
    mf = rep / "manifest.json"
    manifest = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else []
    log = []

    def record(entry):
        """Append-only (rule 24: the record is extended, never replaced): a repeated step is a new attempt; written to disk at once."""
        key = (entry["phase"], entry["pair"], entry.get("step") or entry.get("video") or "")
        entry["attempt"] = 1 + sum((e["phase"], e["pair"], e.get("step") or e.get("video") or "") == key for e in manifest)
        manifest.append(entry)
        if not args.dry_run:
            mf.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")

    for p in pairs:
        pair = p["pair"]; clip, cat = pair.split("/"); d = root / "cache" / "m3" / clip / cat
        arm_keys = ["A0", f"A3_{cat}"] + ([f"A3_{cat}_nodyn"] if p.get("nodyn") else [])
        if args.phase == "phase1":
            steps = []
            exp_flags = ["--experiment"] if cfg.get("path") == "experiment" else []
            cam_flags = ["--camera", str(cfg["camera"])] if cfg.get("camera") else []
            if args.force or not (d / "lift.npz").exists():
                steps.append(("lift", [PY, "scripts/run_m3_sint.py", "--drive-root", args.drive_root, "--pairs", pair] + exp_flags))
            if args.force or not (d / "contact_decision.json").exists():
                steps.append(("contact", [PY, "scripts/check_m3_contact.py", "--drive-root", args.drive_root]))
            gstat = geometry_status(d)
            if p.get("camera_gate_failed"):
                print(f"[batch] {pair}: flagged (camera gate failed): no geometry or renders can be built (no bridge); A0 only", flush=True)
                record({"phase": "phase1", "pair": pair, "step": "geometry+renders", "rc": None, "skipped": "camera gate failed (flagged pair)", "time": time.strftime("%Y-%m-%d %H:%M")})
            elif args.force or gstat != "ok":
                print(f"[batch] {pair}: geometry {gstat}; running sint2", flush=True)
                steps.append(("geometry", [PY, "scripts/run_m3_sint2.py", "--drive-root", args.drive_root, "--pairs", pair] + cam_flags))
            if p.get("nodyn") and (args.force or not (d / "dynamic_mask.png").exists()):
                steps.append(("dynamic_mask", [PY, "scripts/make_m3_dynamic_mask.py", "--drive-root", args.drive_root, "--pair", pair]))
            rk = [] if p.get("camera_gate_failed") else [cat] + ([f"{cat}_nodyn"] if p.get("nodyn") else [])
            for k in rk:
                flags = list(tbl(cfg["render"], k))
                if "--exclude-mask" in flags:
                    flags[flags.index("--exclude-mask") + 1] = str(d / flags[flags.index("--exclude-mask") + 1])
                suf = ("_obj" if "--object" in flags else "") + ("_sky" if "--sky-in-mask" in flags else "") + ("_nodyn" if "--exclude-mask" in flags else "")
                rp_ = d / f"render{suf}.npz"
                stale = (not rp_.exists()) or any(n == "geometry" for n, _ in steps) or ((d / "geometry.npz").exists() and rp_.stat().st_mtime < (d / "geometry.npz").stat().st_mtime)
                if args.force or stale:
                    steps.append((f"render{suf}", [PY, "scripts/render_m3_target.py", "--drive-root", args.drive_root, "--pairs", pair] + flags))
            failed = None
            for name, cmd in steps:
                needs = {"geometry": ("lift",), "dynamic_mask": (), "contact": ("lift",)}.get(name, ("lift", "geometry") if name.startswith("render") else ())
                if failed and failed in needs:
                    print(f"[batch] {pair}: {name} skipped — {failed} failed before it", flush=True)
                    record({"phase": "phase1", "pair": pair, "step": name, "rc": None, "skipped": f"{failed} failed", "time": time.strftime("%Y-%m-%d %H:%M")}); continue
                rc, out_ = sh(cmd, args.dry_run, log)
                record({"phase": "phase1", "pair": pair, "step": name, "rc": rc, "cmd": " ".join(map(str, cmd)), "tail": out_[-600:] if rc else "", "time": time.strftime("%Y-%m-%d %H:%M")})
                if rc != 0:
                    failed = failed or name
                    print(f"[batch] {pair}: {name} failed (recorded); steps that depend on it are skipped", flush=True)
            if not steps:
                print(f"[batch] {pair}: phase 1 outputs exist (lift, contact, geometry, renders); nothing to do", flush=True)
            # a runner that returned 0 without its output has REFUSED the pair (handoff gate, camera, bridge): the reason from its report is the record
            for name_, f_, repf in (("lift", d / "lift.npz", rep / ".." / "m3_sint" / "sint_step1.json"), ("geometry", d / "geometry.npz", rep / ".." / "m3_sint" / "sint_step2.json")):
                if not f_.exists() and repf.exists():
                    why = next((r.get("refused") or r.get("error") for r in json.loads(repf.read_text(encoding="utf-8")) if r.get("pair") == pair and (r.get("refused") or r.get("error"))), None)
                    if why:
                        record({"phase": "phase1", "pair": pair, "step": f"{name_}_refused", "rc": None, "refused": why, "time": time.strftime("%Y-%m-%d %H:%M")})
                        print(f"[batch] {pair}: {name_} REFUSED: {why}", flush=True); break
        elif args.phase == "phase2":
            gstat = geometry_status(d)
            for ak in arm_keys:
                arm = ak.split("_")[0]; flags = list(tbl(cfg["arms"], ak))
                if arm == "A3" and (p.get("camera_gate_failed") or gstat != "ok"):
                    why = "camera gate failed (flagged pair)" if p.get("camera_gate_failed") else f"geometry {gstat}: not the configuration of record; run phase1 first"
                    print(f"[batch] {pair}: {ak} skipped — {why}", flush=True)
                    for seed in seeds:
                        record({"phase": "phase2", "pair": pair, "video": f"{video_name(arm, seed, frames, flags)}.mp4", "arm": arm, "arm_key": ak, "seed": seed, "flags": flags, "rc": None, "skipped": why, "time": time.strftime("%Y-%m-%d %H:%M")})
                    continue
                for seed in seeds:
                    stem = video_name(arm, seed, frames, flags); vp = d / f"{stem}.mp4"
                    stale = False
                    if vp.exists() and (d / f"{stem}.npz").exists():                          # a record without `scheduler` predates the sampler of record (2026-10-14)
                        with np.load(d / f"{stem}.npz") as zr:
                            if not json.loads(str(zr["config"])).get("scheduler"):
                                stale = True; print(f"[batch] {pair}: {stem} has a pre-2026-10-14 record (no scheduler field): stale", flush=True)
                    if vp.exists() and arm == "A3" and not stale:
                        rsuf = ("_obj" if "--object" in flags else "") + ("_sky" if "--sky-mask" in flags else "") + ("_nodyn" if "--nodyn-mask" in flags else "")
                        deps = [d / "geometry.npz", d / f"render{rsuf}.npz"]
                        stale = any(x.exists() and vp.stat().st_mtime < x.stat().st_mtime for x in deps)
                    if not args.force and vp.exists() and not stale:
                        print(f"[batch] {pair}: {stem}.mp4 exists; skipped", flush=True); continue
                    if stale:
                        print(f"[batch] {pair}: {stem}.mp4 is older than its geometry / render: re-run (the sampler's record is replaced with --force)", flush=True)
                    cmd = [PY, "scripts/run_m3_s4_guided.py", "--drive-root", args.drive_root, "--pair", pair, "--seed", str(seed), "--frames", str(frames), "--arm", arm] + flags + (["--force"] if (args.force or stale) else [])
                    rc, _ = sh(cmd, args.dry_run, log)
                    record({"phase": "phase2", "pair": pair, "video": f"{stem}.mp4", "arm": arm, "arm_key": ak, "seed": seed, "flags": flags, "rc": rc, "time": time.strftime("%Y-%m-%d %H:%M")})
        elif args.phase == "grade":
            vids = [f"{video_name(ak.split('_')[0], s, frames, list(tbl(cfg['arms'], ak)))}.mp4" for ak in arm_keys for s in seeds]
            vids = [v for v in vids if args.dry_run or (d / v).exists()]
            if not vids:
                print(f"[batch] {pair}: no videos to grade", flush=True); continue
            cmd = [PY, "scripts/measure_m3_a0.py", "--drive-root", args.drive_root, "--videos-dir", args.videos_dir, "--pairs", pair, "--force", "--source-frames", str(frames), "--videos"] + vids + ["source"]
            if p.get("region_mask"):
                cmd += ["--region-mask", str(d / p["region_mask"])]
            if p.get("ground_mask"):
                cmd += ["--ground-mask", str(d / p["ground_mask"])]
            rc, _ = sh(cmd, args.dry_run, log)
            record({"phase": "grade", "pair": pair, "step": "measure_m3_a0", "videos": vids, "rc": rc, "time": time.strftime("%Y-%m-%d %H:%M")})
            for lab, vv in ((p["label_target"], vids), (p["label_source"], ["source"])):
                cmd = [PY, "scripts/check_m3_mask_centroid.py", "--drive-root", args.drive_root, "--videos-dir", args.videos_dir, "--pair", pair, "--label", lab, "--frames", str(frames), "--force", "--videos"] + vv
                rc, _ = sh(cmd, args.dry_run, log)
                record({"phase": "grade", "pair": pair, "step": f"mask_centroid_{lab}_{'source' if vv == ['source'] else 'generated'}", "rc": rc, "time": time.strftime("%Y-%m-%d %H:%M")})
        elif args.phase == "status":
            def age(x):
                return time.strftime("%m-%d %H:%M", time.localtime(x.stat().st_mtime)) if x.exists() else "absent"
            def last(key):
                es = [e for e in manifest if (e["phase"], e["pair"], e.get("step") or e.get("video") or "") == key]
                if not es:
                    return "no manifest entry"
                e = es[-1]
                if e.get("skipped"):
                    return f"attempt {e.get('attempt', '?')}: skipped ({e['skipped']})"
                tail = (e.get("tail") or "").strip().splitlines()
                return f"attempt {e.get('attempt', '?')}: rc {e.get('rc')}" + (f" — {tail[-1][:140]}" if e.get("rc") and tail else "")
            gp = d / "geometry.npz"
            print(f"\n=== {pair}{'  [FLAGGED: camera gate failed]' if p.get('camera_gate_failed') else ''}", flush=True)
            print(f"  geometry: {geometry_status(d)} ({age(gp)}); manifest: {last(('phase1', pair, 'geometry'))}", flush=True)
            for rf in sorted(d.glob("render*.npz")):
                st = " STALE (older than geometry)" if gp.exists() and rf.stat().st_mtime < gp.stat().st_mtime else ""
                print(f"  {rf.name}: {age(rf)}{st}", flush=True)
            for ak in arm_keys:
                arm = ak.split("_")[0]; flags = list(tbl(cfg["arms"], ak))
                for seed in seeds:
                    stem = video_name(arm, seed, frames, flags); vp = d / f"{stem}.mp4"
                    st = ""
                    if vp.exists() and arm == "A3":
                        rsuf = ("_obj" if "--object" in flags else "") + ("_sky" if "--sky-mask" in flags else "") + ("_nodyn" if "--nodyn-mask" in flags else "")
                        if any(x.exists() and vp.stat().st_mtime < x.stat().st_mtime for x in (gp, d / f"render{rsuf}.npz")):
                            st = " STALE"
                    print(f"  {stem}.mp4: {age(vp)}{st}; manifest: {last(('phase2', pair, f'{stem}.mp4'))}", flush=True)
        elif args.phase == "export":
            import shutil
            ex = rep / "export"; pdir = ex / f"{clip}_{cat}"; pdir.mkdir(parents=True, exist_ok=True)
            idx_f = ex / "index.json"; index = json.loads(idx_f.read_text(encoding="utf-8")) if idx_f.exists() else []
            index = [e for e in index if e["pair"] != pair]                                  # the export is a copy of the current cache, rebuilt per pair
            qc = root / "cache" / "m2_d" / clip / cat / "qc_m2d.json"
            prompt = json.loads(qc.read_text(encoding="utf-8"))["prompt"] if qc.exists() else None
            t_a = None
            if (d / "geometry.npz").exists():
                with np.load(d / "geometry.npz") as z:
                    t_a = int(z["src_frames"][0])
            src = Path(args.videos_dir) / f"{clip}.mp4"
            if src.exists():
                shutil.copy2(src, pdir / f"source_{clip}.mp4")                               # unchanged: no resize, no trim (the window is in the index)
            # the reference of record for the grader: the census cut, found by content (configs/experiment_video_hashes.yaml), never by folder;
            # and the file Stage A consumed (bundle provenance), which on a MiraData census clip can be the flat 16:9 original (grader Q1)
            ref_path = None; stage_a_src = None
            hf_ = Path("configs/experiment_video_hashes.yaml")
            if hf_.exists():
                import hashlib as _hl
                want = (yaml.safe_load(hf_.read_text(encoding="utf-8")) or {}).get("sha256", {}).get(clip)
                for c_ in sorted(Path(args.videos_dir).glob(f"*/{clip}.mp4")) + [Path(args.videos_dir) / f"{clip}.mp4"]:
                    if c_.exists() and want and _hl.sha256(c_.read_bytes()).hexdigest() == want:
                        ref_path = c_; shutil.copy2(c_, pdir / f"reference_{clip}.mp4"); break
            pf_ = root / "cache" / "stage_a" / clip / "provenance.json"
            if pf_.exists():
                stage_a_src = json.loads(pf_.read_text(encoding="utf-8")).get("clip_path")
            anc = root / "cache" / "m2_d" / clip / cat / "anchor.png"
            if anc.exists():
                shutil.copy2(anc, pdir / "anchor.png")
            for ak in arm_keys:
                arm = ak.split("_")[0]; flags = list(tbl(cfg["arms"], ak))
                for seed in seeds:
                    stem = video_name(arm, seed, frames, flags); vp = d / f"{stem}.mp4"
                    if not vp.exists():
                        continue
                    shutil.copy2(vp, pdir / vp.name)
                    run_cfg = {}
                    if (d / f"{stem}.npz").exists():
                        with np.load(d / f"{stem}.npz") as zr:
                            c_ = json.loads(str(zr["config"])); run_cfg = {k: c_.get(k) for k in ("scheduler", "steps", "shift", "cfg", "weights", "text_tokens", "corrector_off_steps", "bridge", "beta")}
                    index.append({"pair": pair, "arm": arm, "arm_key": ak, "method": "no guidance (Wan2.1 I2V)" if arm == "A0" else "ours (Wan2.1 I2V + geometric guidance)",
                                  "seed": seed, "video": f"{clip}_{cat}/{vp.name}", "frames": frames, "fps": 16, "size": "832x480", "prompt": prompt,
                                  "anchor": f"{clip}_{cat}/anchor.png" if anc.exists() else None, "source": f"{clip}_{cat}/source_{clip}.mp4" if src.exists() else None,
                                  "reference": (f"{clip}_{cat}/reference_{clip}.mp4" if ref_path else None), "reference_note": "the census cut, matched by sha256 to configs/experiment_video_hashes.yaml; the grader's reference",
                                  "stage_a_source": stage_a_src, "stage_a_source_note": "the file Stage A consumed (bundle provenance): the Wan frame is its FULL field of view resized to 832x480",
                                  "t_a": t_a, "window": (f"source frames {t_a}..{t_a + frames - 1}; generated frame k = source frame {t_a} + k" if t_a is not None else None),
                                  "flags": flags, "camera_gate_failed": bool(p.get("camera_gate_failed")), "geometry_status": geometry_status(d), "run_config": run_cfg})
            index.sort(key=lambda e: (e["pair"], e["arm_key"], e["seed"]))
            idx_f.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
            n_ = sum(e["pair"] == pair for e in index)
            print(f"[batch] {pair}: exported {n_} videos + source + anchor to {pdir}", flush=True)
            record({"phase": "export", "pair": pair, "step": "export", "rc": 0, "n_videos": n_, "time": time.strftime("%Y-%m-%d %H:%M")})
        elif args.phase == "pack":
            out = rep / "eval_pack" / f"{clip}_{cat}"; out.mkdir(parents=True, exist_ok=True)
            missing_files = [n for n in ("geometry.npz", "lift.npz") if not (d / n).exists()]
            if missing_files:
                print(f"[batch] {pair}: pack skipped, missing {missing_files}", flush=True)
                record({"phase": "pack", "pair": pair, "step": "eval_pack", "rc": 1, "note": f"missing {missing_files}"}); continue
            with np.load(d / "geometry.npz") as z, np.load(d / "lift.npz") as zl:
                want = ("E_full", "E_l", "c_tar", "src_frames", "beta", "s_delta", "bridge_run", "F_pair", "T_l", "Kp", "object_path_source")
                g = {k: z[k] for k in want if k in z.files}; absent = [k for k in want if k not in z.files]
                np.savez_compressed(out / "geometry_eval.npz", **g, K_wan=zl["K_wan"], K_sq=zl["K_sq"], anchor_mask_sq=zl["anchor_mask_sq"], depth_src_sq=zl["depth_src_sq"])
            gstat = geometry_status(d)
            gates = {}
            for jn in ("contact_decision.json", "sint_step2.json", "sint_step1.json"):
                if (d / jn).exists():
                    try:
                        gates[jn] = json.loads((d / jn).read_text(encoding="utf-8"))
                    except Exception as e_:                                                 # noqa: BLE001
                        gates[jn] = f"unreadable: {e_}"
            runs = []
            for ak in arm_keys:
                for s_ in seeds:
                    stem = video_name(ak.split("_")[0], s_, frames, list(tbl(cfg["arms"], ak)))
                    vp = d / f"{stem}.mp4"
                    cfg_run = None
                    if (d / f"{stem}.npz").exists():
                        with np.load(d / f"{stem}.npz") as zr:
                            cfg_run = json.loads(str(zr["config"]))
                    runs.append({"arm": ak.split("_")[0], "arm_key": ak, "seed": s_, "video": str(vp), "exists": vp.exists(), "flags": list(tbl(cfg["arms"], ak)), "run_config": cfg_run})
            pack = {"pair": pair, "batch": cfg["batch"], "frames": frames, "note": p.get("note"), "camera_gate_failed": bool(p.get("camera_gate_failed")), "geometry_status": gstat,
                    "geometry_keys_absent": absent, "label_source": p["label_source"], "label_target": p["label_target"],
                    "source_clip": str(Path(args.videos_dir) / f"{clip}.mp4"), "t_a": int(g["src_frames"][0]), "src_frames": g["src_frames"].tolist(),
                    "reference_window": f"frames t_a .. t_a + {frames} - 1 of the source clip (the generated video's frame k is source frame t_a + k)",
                    "caches": {"m3": str(d), "stage_c_mask": str(root / "cache" / "stage_c" / clip), "stage_c5_tracks": str(root / "cache" / "stage_c5" / f"{clip}.npz"),
                               "stage_a_bundle": str(root / "cache" / "stage_a" / clip), "handoff": str(root / "cache" / "m2_e" / clip / cat)},
                    "geometry_eval": str(out / "geometry_eval.npz"),
                    "geometry_eval_semantics": {"E_full": "[F_pair, 4, 4] the TARGET camera per source frame (the reference camera bridged by beta, rebased to t_a, cam2world in the anchor's frame = the expected camera of a generated video; rotation is bridge-invariant, translation is in the target's units)",
                                                "E_l": "[T_l, 4, 4] the same at the latent frames (every 4th)", "c_tar": "[T_l, 3] the retargeted object centroid path in the same frame",
                                                "K_wan": "[3, 3] intrinsics at 832x480", "beta": "the bridge scale of record", "s_delta": "the object displacement scale"},
                    "gates": gates, "runs": runs, "grader_records": {"a0_motion": str(root / "reports" / "m3_s4" / "a0_motion.json"), "mask_centroid": str(root / "reports" / "m3_s4" / f"mask_centroid_{clip}_{cat}.json")}}
            (out / "runs.json").write_text(json.dumps(pack, indent=1, default=str), encoding="utf-8")
            record({"phase": "pack", "pair": pair, "step": "eval_pack", "rc": 0, "path": str(out), "time": time.strftime("%Y-%m-%d %H:%M")})
            print(f"[batch] {pair}: eval pack at {out} ({sum(r['exists'] for r in runs)}/{len(runs)} videos present)", flush=True)
    if args.phase == "export" and not args.dry_run:
        import shutil
        z_ = shutil.make_archive(str(rep / "m3_batch_export"), "zip", root_dir=str(rep / "export"))
        print(f"[batch] export zipped: {z_}", flush=True)
    if not args.dry_run and args.phase != "status":
        mf.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
        (rep / f"log_{args.phase}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        print(f"[batch] manifest {mf}: {len(manifest)} entries", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
