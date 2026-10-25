#!/usr/bin/env python
"""The whole pipeline on one clip, in order, from one command — with the human stops it has (2026-10-15), or on the experiment set
with the gates-only path (2026-10-19, plan v1: the two Module 2 review stops become gate checks; the anchor is selected by the rule of
record; the m2_e handoff verdict is the gate outcomes).

    python scripts/run_all.py --env colab --clip 000000000265.2.005 --category human --module 1      # Module 1: A -> B -> C -> C.5 -> B.5
    python scripts/run_all.py --env colab --clip 000000000265.2.005 --category human --module 2      # Module 2: m2_a -> b2 -> c3 -> d -> e (stops at its reviews)
    python scripts/run_all.py --env colab --clip 000000000265.2.005 --category human --module 3      # Module 3: geometry, renders, A0 + A3, grading, export
    python scripts/run_all.py --env local --clip ... --category ... --module all                      # the three in sequence on one machine (three interpreters)
    python scripts/run_all.py --env colab --clip ... --category ... --status                          # read-only: what exists per stage

    # the experiment set (configs/experiment_set.yaml; root motion_transfer_exp), no human step: every pair in turn, a failure is a recorded
    # outcome and the next pair runs; the proxies of record are copied from the six pairs' root the first time
    python scripts/run_all.py --env colab_exp --set configs/experiment_set.yaml --module 2 --auto --proxy-from /content/drive/MyDrive/motion_transfer
    python scripts/run_all.py --env colab_exp --set configs/experiment_set.yaml --module 3 --auto     # Phase-2 session: the batch of the set
    python scripts/run_all.py --env colab_exp --set configs/experiment_set.yaml --status

Each stage is one of the existing runners as a subprocess; a stage whose output exists and is of record is skipped; a failing stage stops
the chain with its last lines (a failing gate is a finding, never a parameter to adjust). The root (cache/, reports/) and the data folder
(videos/) come from configs/environment.yaml via MT_ROOT / MT_DATA, so the same command runs on Colab and on a local card.

Module 2 is closed at v1 and keeps its hard stops on the six pairs' root: the proxy cache is written only after human review
(--commit-cache), the control images are reviewed before the anchors are generated, and the anchor is selected by the human's ranks.
Without --auto this driver stops at each with the exact command the human runs next, then resumes from there on the next call.
With --auto (the experiment path) the proxy is the one of record (copied, never regenerated), the control review is its own gate record
(qc_c3.json: emitted, no gate failed), the anchor is selected by scripts/select_m2d_anchors.py (rule a' of record, --write-anchor), and
the handoff verdict is the gate outcomes of qc_m2e.json (any 'fail' refuses the pair, recorded in handoff_refused.json).
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


LAST_TAIL = []


def sh(cmd, env, dry):
    """Runs a stage, streams its output, and keeps its last lines (LAST_TAIL) so a failure's cause is in the record, not only on the console."""
    print(f"[all] $ {' '.join(map(str, cmd))}", flush=True)
    LAST_TAIL.clear()
    if dry:
        return 0
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for ln in p.stdout:
        print("    " + ln.rstrip(), flush=True); LAST_TAIL.append(ln.rstrip())
        del LAST_TAIL[:-25]
    return p.wait()


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def copy_proxy_of_record(R: Path, src_root: Path, cat: str, key: str, dry: bool):
    """The experiment path never regenerates a proxy: the one of record (cache/m2_b2/<cat>/<key> under the six pairs' root) is copied
    whole, with its qc_b2.json (the key re-derives from it), and the estimator-sign record m2_d needs (probe/m2d/estimator_sign) comes
    along once. The copy is recorded next to the proxy."""
    src = src_root / "cache" / "m2_b2" / cat / key; dst = R / "cache" / "m2_b2" / cat / key
    print(f"[all] copy proxy of record {cat}/{key}: {src} -> {dst}", flush=True)
    if dry:
        return
    assert (src / "proxy.npz").exists() and (src / "qc_b2.json").exists(), f"proxy of record {src} is not a complete cache entry"
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(src, dst)
    (dst / "copied_from.json").write_text(json.dumps({"source": str(src), "time": time.strftime("%Y-%m-%d %H:%M"), "note": "proxy of record copied, not regenerated (experiment path)"}, indent=1), encoding="utf-8")
    es_src = src_root / "probe" / "m2d" / "estimator_sign" / "estimator_sign.json"; es_dst = R / "probe" / "m2d" / "estimator_sign" / "estimator_sign.json"
    if es_src.exists() and not es_dst.exists():
        es_dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(es_src, es_dst)
        print(f"[all] copied the estimator-sign record (m2_d --measure reads it): {es_dst}", flush=True)


def heading_verdict(R: Path, clip: str):
    """Placement turns the proxy onto the source heading d_src; m2_a records heading_defined (the g_heading gate itself is ungated in the
    lock, so admission passes without one). No heading = a recorded Module 2 refusal on the experiment path, never a crash in placement."""
    q = read_json(R / "cache/m2_a" / clip / "qc_m2a.json")
    if q is None:
        return False
    if q.get("heading_defined") is False:
        return "refused: no source heading (m2_a heading_defined = False: B.5 found no heading); C1 placement needs d_src"
    return True


OCCLUSION_EXCLUDES_FILL = False


def locked_polarity():
    lock = yaml.safe_load((ROOT / "configs" / "calibration.lock.yaml").read_text(encoding="utf-8")) or {}
    return (lock.get("thresholds") or {}).get("m2d_polarity") or (lock.get("m2") or {}).get("m2d_polarity")


def c3_exists_in_locked_polarity(R: Path, clip: str, cat: str):
    """A control composed under a different polarity than the lock's (e.g. 'uncalibrated' while the lock read as absent) is stale: C3 re-runs."""
    q = read_json(R / "cache/m2_c" / clip / cat / "qc_c3.json")
    if q is None:
        return False
    pol = locked_polarity()
    if q.get("emitted") and pol and q.get("polarity") != pol:
        print(f"[all] c3 control of {clip}/{cat} was composed as {q.get('polarity')!r}, the lock says {pol!r}: stale, re-composed", flush=True)
        return False
    if OCCLUSION_EXCLUDES_FILL and not q.get("emitted") and "g_obj_visible" in [k for k, v in (q.get("gates") or {}).items() if isinstance(v, dict) and v.get("outcome") == "fail"] \
            and not str(q.get("occlusion_rule", "")).startswith("scene_only"):
        print(f"[all] c3 control of {clip}/{cat} was refused by g_obj_visible under the old occlusion rule: re-composed under R2 (fill pixels never occlude)", flush=True)
        return False
    return True


def importable(interp: str, module: str, env) -> bool:
    return subprocess.run([interp, "-c", f"import {module}"], env=env, capture_output=True).returncode == 0


def archive_m2e_from_r2(R: Path, clip: str, cat: str):
    """Move a pair's m2_e outputs from R2 onward into a dated archive folder (the record is kept, never replaced: rule 24); alpha.json and
    anchor_masks.npz stay (R2 reads them and D7 does not change them)."""
    d = R / "cache/m2_e" / clip / cat
    if not d.exists():
        return
    names = [n for n in ("r2.json", "r2_body.json", "r2_failed.json", "r2_refused.json", "r2", "g_tar_mesh.npz", "g_tar_gauss.npz", "delta.json", "qc_m2e.json", "handoff.json", "handoff_refused.json") if (d / n).exists()]
    if not names:
        return
    arc = d / "_archive" / time.strftime("%Y%m%dT%H%M%S"); arc.mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.move(str(d / n), str(arc / n))
    (arc / "why.json").write_text(json.dumps({"reason": "D7 re-run: Body R2 re-centred by padding (reviewer 2026-10-19); outputs from R2 onward redone", "moved": names}, indent=1), encoding="utf-8")
    print(f"[all] m2_e outputs of {clip}/{cat} from R2 onward archived to {arc} ({', '.join(names)}); redone below", flush=True)


def m3_outcome(R: Path, clip: str, cat: str, m3_config):
    """True when the pair has its lift, geometry, renders and the videos of both arms; a refusal string naming the first missing piece
    and the runner's recorded reason (reports/m3_sint/sint_step{1,2}.json) otherwise. The batch driver returns 0 on recorded failures,
    so the chain reads the cache, not the return code."""
    sys.path.insert(0, str(ROOT))
    from scripts.run_m3_batch import video_name
    cfg = yaml.safe_load(Path(m3_config or "configs/m3_batch.yaml").read_text(encoding="utf-8"))
    d = R / "cache/m3" / clip / cat; frames = int(cfg["frames"]); seeds = [int(s) for s in cfg["seeds"]]
    def why(step):
        f = R / "reports" / "m3_sint" / f"sint_step{step}.json"
        rows = read_json(f) or []
        return next((r.get("refused") or r.get("error") for r in rows if r.get("pair") == f"{clip}/{cat}" and (r.get("refused") or r.get("error"))), None)
    if not (d / "lift.npz").exists():
        return f"refused: Module 3 lift wrote nothing: {why(1) or 'no reason recorded'}"
    if not (d / "geometry.npz").exists():
        return f"refused: Module 3 geometry wrote nothing: {why(2) or 'no reason recorded'}"
    missing = []
    for ak in ["A0", f"A3_{cat}"]:
        for s in seeds:
            name = f"{video_name(ak.split('_')[0], s, frames, list(cfg['arms'][ak]))}.mp4"
            if not (d / name).exists():
                missing.append(name)
    if missing:
        rf = R / "reports" / "m3_batch" / "manifest.json"
        tails = [e for e in (read_json(rf) or []) if e.get("pair") == f"{clip}/{cat}" and e.get("rc") not in (0, None)]
        last = (tails[-1].get("tail") or "").strip().splitlines()[-1:] if tails else []
        return f"refused: Module 3 videos missing {missing}" + (f"; last failed step {tails[-1].get('step') or tails[-1].get('video')}: {last[0][:200]}" if tails and last else "")
    return True


def c3_gate_verdict(R: Path, clip: str, cat: str):
    """The control review as a gate record (plan v1: 'the two review stops become gates-only'): qc_c3.json emitted, no gate failed."""
    q = read_json(R / "cache/m2_c" / clip / cat / "qc_c3.json")
    if q is None:
        return False
    failed = sorted(k for k, v in (q.get("gates") or {}).items() if isinstance(v, dict) and (v.get("outcome") == "fail" or v.get("pass") is False))
    if not q.get("emitted") or failed:
        why = []
        if "g_placement_admitted" in failed:
            pl = read_json(R / "cache/m2_c" / clip / cat / "placement.json") or {}; p_ = pl.get("placement") or {}
            g_ = {k: v.get("outcome") for k, v in (p_.get("gates") or {}).items() if isinstance(v, dict)}
            why.append(f"C1 not admitted: gates {g_ or 'n/a'}, z {p_.get('z')}, s {p_.get('s')}, mode {pl.get('mode')}")
        if "g_obj_visible" in failed:
            v = q["gates"]["g_obj_visible"]; why.append(f"object not visible in the control: obj_px {v.get('obj_px')}, occluded_frac {v.get('occluded_frac')}")
        return f"refused: control not emitted; gates failed {failed} (qc_c3.json)" + ("; " + "; ".join(why) if why else "")
    return True


def selection_verdict(R: Path, clip: str, cat: str):
    q = read_json(R / "cache/m2_d" / clip / cat / "qc_m2d.json")
    if q is None or not str(q.get("phase2_instrument", "")).startswith("selection v2"):
        return False
    if q.get("selected_seed") is None:
        return f"refused: no anchor (selection v2: {q.get('n_survivors')} survivors)"
    return (R / "cache/m2_d" / clip / cat / "anchor.png").exists()


def handoff_verdict(R: Path, clip: str, cat: str):
    d = R / "cache/m2_e" / clip / cat
    if (d / "handoff.json").exists():
        return True
    r = read_json(d / "handoff_refused.json")
    if r is not None:
        return f"refused: {r.get('failed_gates') or r.get('reason')}"
    return False


def run_pair(args, e, env, py, clip, cat, exp):
    R = Path(os.path.expandvars(str(e["root"]))); data = os.path.expandvars(str(e["data"])); pair = f"{clip}/{cat}"
    video = Path(data) / "videos" / f"{clip}.mp4"
    auto = bool(args.auto); es = ["--experiment-set", args.set] if (args.set and auto) else []
    key = (exp or {}).get("proxy_keys", {}).get(cat) if auto else None
    print(f"[all] env {args.env}: root {R}; data {data}; clip {clip} ({cat}); video {'present' if video.exists() else 'ABSENT'}; path {'automatic (gates only)' if auto else 'with human stops'}", flush=True)
    if args.redo_from == "r2" and cat in (args.redo_categories or []) and (not args.redo_pairs or pair in args.redo_pairs) and not args.status and not args.dry_run:
        archive_m2e_from_r2(R, clip, cat)

    # ---- every stage: (name, done-check -> True | False | 'refused: ...', command(s), interpreter key, human-stop message or None) --------
    m1 = [("M1 A..B5 (depth, motion, mask, dense tracks, heading)", lambda: (R / "cache/stage_b5" / f"{clip}_qc_b5.json").exists(),
           ["scripts/run_pipeline.py", "--clip-id", clip, "--videos-dir", str(Path(data) / "videos")], "default", None)]
    proxy_done = (lambda: (R / "cache/m2_b2" / cat / key / "proxy.npz").exists()) if auto else \
                 (lambda: any((R / "cache/m2_b2").glob(f"{cat}/*")) if (R / "cache/m2_b2").exists() else False)
    proxy_cmd = (("copy_proxy", lambda: copy_proxy_of_record(R, Path(args.proxy_from), cat, key, args.dry_run)) if (auto and args.proxy_from) else None)
    m2 = [
        ("M2 a admission (stage0)", lambda: (R / "cache/m2_a" / clip / "qc_m2a.json").exists(), ["scripts/run_m2a.py", "--clip-id", clip], "default", None),
        ("M2 b2 R1 proxy" + (" of record (copied)" if auto else " (reconstructor)"), proxy_done, proxy_cmd, "sam3d_objects" if cat == "vehicle" else "default",
         None if (auto and args.proxy_from) else (
         "the proxy of record is absent under this root: pass --proxy-from <root of the six pairs> (experiment path, copied, never regenerated)" if auto else
         "HARD STOP (SKILL m2_b2 §12): the R1 proxy is generated from a candidate image and reviewed before it enters the cache:\n"
         "      python scripts/run_m2b2.py --r1 --category {cat} --image <proxy png> --mask <mask png> --facing <left|right> --selected-k 0\n"
         "      python scripts/run_m2b2.py --promote <KEY>        # after the review (the _review folder names KEY)")),
        ("M2 precondition: heading defined (m2_a)", lambda: heading_verdict(R, clip), ("check", None), "default", None),
        ("M2 c1 placement", lambda: (R / "cache/m2_c" / clip / cat / "placement.json").exists(),
         ["scripts/place_proxy_on_clip.py", "--clip", clip, "--category", cat] + (["--key", key] if key else []), "default", None),
        ("M2 c3 control", lambda: c3_exists_in_locked_polarity(R, clip, cat),
         ["scripts/run_m2c_c3.py", "--clip", clip, "--category", cat] + (["--occlusion-excludes-fill"] if (auto and args.occlusion_excludes_fill) else []), "default", None),
        ("M2 c3 review" + (" (gate record)" if auto else ""), (lambda: c3_gate_verdict(R, clip, cat)) if auto else (lambda: args.c3_reviewed), (("check", None) if auto else None), "default",
         None if auto else "HARD STOP (SKILL C3 §10): review the composed control image under viz/m2c_control/<clip>_<cat>.png, then re-run with --c3-reviewed"),
        ("M2 d anchors (FLUX.1 Depth)", lambda: (R / "cache/m2_d" / clip / cat / "generation.json").exists(), ["scripts/run_m2d.py", "--clip", clip, "--category", cat] + es, "default", None),
        ("M2 d fidelity", lambda: (R / "cache/m2_d" / clip / cat / "fidelity.json").exists(), ["scripts/run_m2d.py", "--clip", clip, "--category", cat, "--measure"] + es, "default", None),
        ("M2 d selection" + (" (rule of record)" if auto else ""), (lambda: selection_verdict(R, clip, cat)) if auto else (lambda: (R / "cache/m2_d" / clip / cat / "anchor.png").exists()),
         ([["scripts/select_m2d_anchors.py", "measure", "--pairs", pair], ["scripts/select_m2d_anchors.py", "select", "--pairs", pair, "--write-anchor", "--no-report"]] if auto else None), "default",
         None if auto else
         "HARD STOP: the anchor is selected from the retained candidates — the human's Phase-2 ranks (`python scripts/run_m2d.py --clip {clip} --category {cat} --select`)\n"
         "      or the recorded automatic rule (`python scripts/select_m2d_anchors.py select --pairs {clip}/{cat} --write-anchor`; the selection of record on the six pairs is the human's)"),
        ("M2 e alpha", lambda: (R / "cache/m2_e" / clip / cat / "alpha.json").exists(), ["scripts/run_m2e.py", "--alpha", "--clip", clip, "--category", cat] + (["--auto"] if auto else []), "default", None),
    ]
    def r2_verdict():
        d_ = R / "cache/m2_e" / clip / cat
        if any((d_ / n).exists() for n in ("r2.json", "r2_body.json")):
            return True
        rr = read_json(d_ / "r2_refused.json")
        return f"refused: R2 {rr.get('stage')}: {rr.get('reason')}" if rr else False
    r2_stage = ("M2 e r2 reconstruction", r2_verdict,
                ["scripts/run_m2e.py", "--r2", "--clip", clip, "--category", cat] + (["--auto"] if auto else []), "sam3d_objects" if cat == "vehicle" else "default", None)
    delta_stage = ("M2 e delta", lambda: (R / "cache/m2_e" / clip / cat / "delta.json").exists(), ["scripts/run_m2e.py", "--delta", "--clip", clip, "--category", cat] + (["--auto"] if auto else []), "default", None)
    body_env = ("M2 e Body environment (detectron2 in the default interpreter)", lambda: importable(py.get("default", "python"), "detectron2", env),
                ("env", "SAM 3D Body needs detectron2 in the default interpreter (scripts/setup_sam3d.sh installs it; its FAILED line is in <root>/envs/setup_sam3d_*.log): "
                        "re-run `bash scripts/setup_sam3d.sh body` and read the log before this pair is re-run"), "default", None)
    # vehicles: the delta step's fallback decision fixes alpha BEFORE R2 (run_m2e asserts it); humans: the Body R2 record feeds delta
    m2 += ([delta_stage, r2_stage] if cat == "vehicle" else [body_env, r2_stage, delta_stage]) + [
        ("M2 e qc", lambda: (R / "cache/m2_e" / clip / cat / "qc_m2e.json").exists(), ["scripts/run_m2e.py", "--qc", "--clip", clip, "--category", cat] + (["--auto"] if auto else []), "default", None),
        ("M2 e handoff" + (" (verdict = gate outcomes)" if auto else ""), (lambda: handoff_verdict(R, clip, cat)) if auto else (lambda: (R / "cache/m2_e" / clip / cat / "handoff.json").exists()),
         ["scripts/run_m2e.py", "--handoff", "--clip", clip, "--category", cat] + (["--auto"] if auto else []), "default", None),
    ]
    force = ["--force"] if args.force else []; m3cfg = ["--config", args.m3_config] if args.m3_config else []
    m3 = [                                                                            # the batch driver decides per step what exists, is stale, or is off-record
        ("M3 precondition: Module 2 handoff", lambda: (True if (R / "cache/m2_e" / clip / cat / "handoff.json").exists() else f"refused: no Module 2 handoff for this pair under {R} (Module 2 did not complete: see reports/experiment/run_all.json)"), ("check", None), "default", None),
        ("M3 phase 1 (geometry, renders)", lambda: False, ["scripts/run_m3_batch.py", "phase1", "--pairs", pair] + m3cfg + force, "default", None),
        ("M3 phase 2 (A0 + A3, seeds)", lambda: False, ["scripts/run_m3_batch.py", "phase2", "--pairs", pair, "--seeds"] + [str(s) for s in args.seeds] + m3cfg + force, "phase2", None),
        ("M3 grade", lambda: False, ["scripts/run_m3_batch.py", "grade", "--pairs", pair, "--seeds"] + [str(s) for s in args.seeds] + m3cfg, "phase2", None),
        ("M3 pack", lambda: False, ["scripts/run_m3_batch.py", "pack", "--pairs", pair] + m3cfg, "default", None),
        ("M3 export", lambda: False, ["scripts/run_m3_batch.py", "export", "--pairs", pair] + m3cfg, "default", None),
        ("M3 outcome (lift, geometry, renders, both arms' videos)", lambda: m3_outcome(R, clip, cat, args.m3_config), ("check", None), "default", None),
    ]
    chain = {"1": m1, "2": m2, "3": m3, "all": m1 + m2 + m3}[args.module]

    for name, done, cmd, pykey, stop in chain:
        try:
            ok = done()
        except Exception as ex:                                                       # noqa: BLE001
            ok = False; print(f"[all] {name}: done-check raised {ex!r}; treated as not done", flush=True)
        if isinstance(ok, str):                                                       # a recorded refusal upstream: the chain ends here as a finding
            print(f"[all] {name}: {ok}", flush=True)
            if args.status:
                continue
            return {"outcome": "refused", "stage": name, "detail": ok}
        if args.status:
            print(f"[all] {'done   ' if ok else ('runner decides' if name.startswith('M3') else 'pending')}  {name}", flush=True); continue
        if ok:
            print(f"[all] skip  {name} (output exists)", flush=True); continue
        if cmd is None:
            msg = (stop or "no command for this stage").format(clip=clip, cat=cat)
            print(f"[all] STOP at {name}:\n      {msg}\n[all] re-run this command when it is done; the chain resumes here", flush=True)
            return {"outcome": "stopped", "stage": name, "detail": msg.splitlines()[0]}
        if isinstance(cmd, tuple) and cmd[0] == "env":                                # an environment precondition: a missing module is a failure of the VM, not of the pair
            if args.dry_run:
                print(f"[all] {name}: environment checked here (dry run)", flush=True); continue
            print(f"[all] {name}: FAILED: {cmd[1]}", flush=True)
            return {"outcome": "failed", "stage": name, "detail": cmd[1]}
        if isinstance(cmd, tuple) and cmd[0] == "check":                              # a pure gate-record check: nothing to run; a missing record is a failure, never a stop
            if args.dry_run:
                print(f"[all] {name}: gate record checked here (dry run)", flush=True); continue
            print(f"[all] {name}: the gate record is missing (the stage before it wrote no qc); the chain stops here", flush=True)
            return {"outcome": "failed", "stage": name, "detail": "gate record missing"}
        if isinstance(cmd, tuple) and callable(cmd[1]):                               # an in-process step (the proxy copy)
            cmd[1]()
        else:
            cmds = cmd if isinstance(cmd[0], list) else [cmd]
            for c in cmds:
                rc = sh([py.get(pykey, "python")] + c, env, args.dry_run)
                if rc != 0:
                    print(f"[all] {name} FAILED (rc {rc}); the chain stops here — read the stage's own report before anything is changed", flush=True)
                    err = next((l for l in reversed(LAST_TAIL) if l.strip() and not l.startswith(" ")), "")
                    return {"outcome": "failed", "stage": name, "detail": f"rc {rc}: {err[:300]}", "tail": list(LAST_TAIL)}
        try:                                                                          # a stage may have produced a recorded refusal (no anchor, handoff refused, gate failed)
            after = done()
        except Exception:                                                             # noqa: BLE001
            after = True
        if isinstance(after, str):
            print(f"[all] {name}: {after}", flush=True)
            return {"outcome": "refused", "stage": name, "detail": after}
        if stop:
            print(f"[all] STOP after {name}: {stop.format(clip=clip, cat=cat)}", flush=True)
            return {"outcome": "stopped", "stage": name, "detail": stop.splitlines()[0]}
    if not args.status:
        print(f"[all] chain complete for {pair} ({args.module}) at {time.strftime('%Y-%m-%d %H:%M')}", flush=True)
    return {"outcome": "complete" if not args.status else "status", "stage": None, "detail": None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="colab", help="preset in configs/environment.yaml")
    ap.add_argument("--clip"); ap.add_argument("--category", choices=["vehicle", "human"])
    ap.add_argument("--set", default=None, help="configs/experiment_set.yaml: run every pair of the set (or --clip/--category as a subset of it)")
    ap.add_argument("--module", default="all", choices=["1", "2", "3", "all"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="Module 3 seeds (default: the batch config's; the set's config says seed 0)")
    ap.add_argument("--status", action="store_true", help="read-only: report what exists per stage and stop")
    ap.add_argument("--force", action="store_true", help="re-run stages whose outputs exist (Module 3 only; Modules 1 and 2 keep their hash guards)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--c3-reviewed", action="store_true", help="Module 2: the composed control image has been reviewed (SKILL C3 §10 hard stop)")
    ap.add_argument("--auto", action="store_true", help="the experiment path: no human stop; review stops are gate records; selection by the rule of record; handoff verdict = gate outcomes")
    ap.add_argument("--proxy-from", default=None, help="--auto: root holding the proxies of record (cache/m2_b2/<cat>/<key>), copied into this root once")
    ap.add_argument("--m3-config", default=None, help="Module 3 batch config (default: configs/m3_batch_exp.yaml with --set, else the batch driver's default)")
    ap.add_argument("--stop-on-fail", action="store_true", help="--set: stop at the first pair that fails (default: record it and go on)")
    ap.add_argument("--include-follow-up", action="store_true", help="Module 3 on the set: also run the follow-up rows of configs/experiment_handoffs.yaml (reported separately)")
    ap.add_argument("--redo-from", default=None, choices=["r2"], help="Module 2: archive each pair's m2_e outputs from this stage onward (cache/m2_e/<clip>/<cat>/_archive/<time>/) and redo them; D7 re-run")
    ap.add_argument("--redo-categories", nargs="*", default=["human"], help="--redo-from: categories it applies to (D7 touches the Body path only)")
    ap.add_argument("--redo-pairs", nargs="*", default=None, help="--redo-from: only these pairs (clip/category); default: every pair of the redo categories")
    ap.add_argument("--occlusion-excludes-fill", action="store_true", help="R2 adopted (after scripts/check_m2c_fill_occlusion.py reads 0 on every handoff): C3 runs with fill pixels never occluding; g_obj_visible refusals under the old rule are re-composed")
    args = ap.parse_args(argv)
    envs = yaml.safe_load((ROOT / "configs" / "environment.yaml").read_text(encoding="utf-8"))
    assert args.env in envs, f"--env must be one of {list(envs)}"
    e = envs[args.env]
    root = os.path.expandvars(str(e["root"])); data = os.path.expandvars(str(e["data"]))
    assert "$" not in root and "$" not in data, f"environment {args.env}: set MT_ROOT / MT_DATA in the shell or edit configs/environment.yaml ({root}, {data})"
    env = {**os.environ, "MT_ROOT": root, "MT_DATA": data, "PYTHONUNBUFFERED": "1"}
    py = {k: str(v) for k, v in e["python"].items()}
    exp = None
    if args.set:
        exp = yaml.safe_load(Path(args.set).read_text(encoding="utf-8"))["experiment_set"]
        if args.m3_config is None:
            args.m3_config = "configs/m3_batch_exp.yaml"
        if args.seeds is None:
            args.seeds = [int(s) for s in exp.get("seeds", [0])]
        todo = [(p["clip"], p["category"]) for p in exp["pairs"] if not args.clip or (p["clip"] == args.clip and (not args.category or p["category"] == args.category))]
        if args.module == "3" and not args.clip:                                     # Module 3 runs on the FROZEN handoff set (configs/experiment_handoffs.yaml), main rows only
            hf = ROOT / "configs" / "experiment_handoffs.yaml"
            assert hf.exists(), "Module 3 on the set needs the frozen handoff list configs/experiment_handoffs.yaml"
            fz = yaml.safe_load(hf.read_text(encoding="utf-8"))["experiment_handoffs"]
            keep = set(fz["main"]) | (set(fz.get("follow_up") or {}) if args.include_follow_up else set())
            todo = [(c, k) for c, k in todo if f"{c}/{k}" in keep]
            print(f"[all] Module 3 on the frozen handoff set ({fz['frozen']}): {len(todo)} pairs" + (" incl. follow-up rows" if args.include_follow_up else " (main rows; --include-follow-up adds the R2 row)"), flush=True)
        assert todo, f"{args.clip}/{args.category} is not in the set {args.set}"
    else:
        assert args.clip and args.category, "give --clip and --category, or --set"
        if args.seeds is None:
            args.seeds = [0, 1]
        todo = [(args.clip, args.category)]
    if args.auto:
        assert args.set, "--auto is the experiment path: it takes --set (the proxy keys and the target text come from the set)"
    global OCCLUSION_EXCLUDES_FILL
    if args.occlusion_excludes_fill:
        vf = Path(root) / "reports" / "m2c_fill_occlusion" / "fill_occlusion.json"
        v = (read_json(vf) or {}).get("verdict") or {}
        assert v.get("adopt_R2") is True, (f"--occlusion-excludes-fill refused: the non-regression check under {root} says adopt_R2 = {v.get('adopt_R2')} "
                                           f"(handoffs with fill occlusion: {v.get('handoffs_with_fill_occlusion')}); R2 is a registered follow-up, not the rule of record")
    OCCLUSION_EXCLUDES_FILL = bool(args.occlusion_excludes_fill)
    results = []
    for clip, cat in todo:
        r = run_pair(args, e, env, py, clip, cat, exp); r.update({"pair": f"{clip}/{cat}", "module": args.module, "time": time.strftime("%Y-%m-%d %H:%M")})
        results.append(r)
        if args.set and not args.status and not args.dry_run:                        # append-only record of every pair's outcome under the root
            rep = Path(root) / "reports" / "experiment"; rep.mkdir(parents=True, exist_ok=True); f = rep / "run_all.json"
            log = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []; log.append(r); f.write_text(json.dumps(log, indent=1), encoding="utf-8")
        if r["outcome"] in ("failed", "refused", "stopped") and (args.stop_on_fail or not args.set):
            break
    if args.set and not args.status:
        print(f"\n[all] set {args.set}, module {args.module}: " + ", ".join(f"{o} {sum(r['outcome'] == o for r in results)}" for o in ("complete", "refused", "failed", "stopped")), flush=True)
        for r in results:
            if r["outcome"] != "complete":
                print(f"    {r['pair']}: {r['outcome']} at {r['stage']}: {r['detail']}", flush=True)
        failed = [r for r in results if r["outcome"] == "failed" and r.get("tail")]
        if failed:
            print(chr(10) + f"[all] last lines of the first failed stage ({failed[0]['pair']}, {failed[0]['stage']}):", flush=True)
            for l in failed[0]["tail"][-12:]:
                print("    | " + l, flush=True)
    bad = [r for r in results if r["outcome"] in ("failed", "stopped")]
    return 0 if not bad or args.set else (2 if bad[0]["outcome"] == "stopped" else 1)


if __name__ == "__main__":
    sys.exit(main())
