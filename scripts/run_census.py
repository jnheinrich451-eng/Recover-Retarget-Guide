#!/usr/bin/env python
"""The census (reviewer 2026-10-17): which clips can enter the experiment at all, decided by gates BEFORE any Module 2 or Module 3 work.

Per clip, in order; a failing step records its reason and the clip is a census FAIL (never adjusted, never rescued):
  1. Module 1  run_pipeline (A -> B -> C -> C.5 -> B.5) — its own gates; a clip with no bundle / mask / B.5 record fails here.
  2. m2_a      the admission reader (CPU, no model, no human stop): stage0.npz with t_a, K_a, d_src, u_scene, n_plane.
  3. source-side handoff  cache/census/<clip>/<cat>/handoff.json — the five fields the camera-gate instruments read (t_a,
               n_frames_after_t_a, d_src, K_1024 = diag(4,4,1) K_a, n_plane), from Module 1 only; and the SOURCE ground band
               (the m2_e annulus definition applied to the source mask at t_a on Stage A's valid depth) -> cache/census/<clip>/ground_band_256.npz.
  4. camera gate of record  check_m3_camera_c --census, then refine_m3_camera_ba --census: the triangulated split at
               lock m3.trajectory.triangulated_gate_patch, with the band gauge read on the source band.
  5. support surface  band points >= lock m3.selection.band_points_min and the gauge's half-split within band_halfsplit_max; a band
               that cannot be determined = "bridge undefined: no support surface".
Category per clip: the pre-registered rule in configs/census.yaml (vehicle sources -> car, everything else -> man) — the category names
the target; the source subject is whatever Stage C masked. The clip list comes from the config (or --clips).

Writes reports/census/census.json (append-only per clip) and census.md: clip | M1 | admission | band n / half-split | camera gate (px, outcome)
| support | CENSUS. The set that enters the experiment is then taken from the PASS rows by the rule written in the plan, never by eye.

    python scripts/run_census.py --config configs/census.yaml            # every clip in the config
    python scripts/run_census.py --clips bear car-turn --category-rule default
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.anchor.m2e import alpha as m2e_alpha                                     # noqa: E402  (the band's definition of record)
from src.preprocess.stage_c import finalize                                      # noqa: E402

VEHICLE_WORDS = ("car", "cars", "bus", "truck", "jeep", "rover", "van", "motorcycle", "vehicle", "carriage", "cart", "boat")   # whole words, in the caption's SUBJECT (its first five words)


def sh(cmd, env):
    print(f"[census] $ {' '.join(map(str, cmd))}", flush=True)
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for ln in p.stdout:
        print("    " + ln.rstrip(), flush=True); tail.append(ln)
    return p.wait(), "".join(tail[-40:])


def category_of(clip: str, caption: str | None, rule: str) -> str:
    """vehicle sources -> 'vehicle' (the red car); everything else -> 'human' (the man). The rule is fixed before the census runs."""
    assert rule == "default", rule
    import re
    head = " ".join((caption or clip.replace("-", " ")).lower().split()[:5])
    words = set(re.findall(r"[a-z]+", head))                                        # whole words only: "scarf" or "carts" never count as "car"
    return "vehicle" if words & set(VEHICLE_WORDS) else "human"


def source_handoff_and_band(root: Path, clip: str, cat: str, lock_m3: dict, m2e_cfg: dict) -> dict:
    """The five handoff fields from Module 1 + stage0, and the source-side ground band at 1024^2 -> 256^2."""
    import cv2
    s0 = root / "cache" / "m2_a" / clip / "stage0.npz"
    with np.load(s0) as z:
        t_a = int(z["t_a"]); K_a = z["K_a"].astype(np.float64); d_src = z["d_src"].astype(np.float64); n_plane = z["n_plane"].astype(np.float64)
    with np.load(root / "cache" / "stage_a" / clip / "bundle.npz") as z:
        F = int(z["D"].shape[0]); D_ta = z["D"][t_a].astype(np.float64)
    M, present, _ = finalize.read_mask(root / "cache" / "stage_c" / clip)
    K_1024 = np.diag([4.0, 4.0, 1.0]) @ K_a
    h = {"t_a": t_a, "n_frames_after_t_a": F - t_a, "d_src": d_src.tolist(), "K_1024": K_1024.tolist(), "n_plane": n_plane.tolist(),
         "source": "census: Module 1 + m2_a stage0 (no Module 2)", "heading_defined": bool(np.linalg.norm(d_src) > 0), "plane_defined": bool(np.linalg.norm(n_plane) > 0)}
    out = root / "cache" / "census" / clip / cat; out.mkdir(parents=True, exist_ok=True)
    (out / "handoff.json").write_text(json.dumps(h, indent=1), encoding="utf-8")
    rec = {"t_a": t_a, "n_frames": F, "heading_defined": h["heading_defined"], "plane_defined": h["plane_defined"]}
    if not h["plane_defined"]:
        rec["band"] = {"n_points": 0, "reason": "no ground plane (B.5 up vector undefined)"}
        return rec
    m_ta = M[t_a].astype(bool)
    big = lambda a, interp: cv2.resize(a.astype(np.uint8 if a.dtype == bool else np.float32), (1024, 1024), interpolation=interp)   # noqa: E731
    mask1024 = big(m_ta, cv2.INTER_NEAREST).astype(bool)
    valid1024 = big(np.isfinite(D_ta) & (D_ta > 0), cv2.INTER_NEAREST).astype(bool)
    acfg = m2e_cfg["alpha"]
    band, info = m2e_alpha.ground_band(mask1024, mask1024, K_1024, n_plane, valid1024, np.zeros_like(mask1024), int(acfg["band_inner_px"]), float(acfg["band_outer_frac"]))
    band_s = cv2.resize(band.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA) > 0.5
    np.savez_compressed(root / "cache" / "census" / clip / "ground_band_256.npz", band=band_s,
                        definition="SOURCE-side band: m2_e ground_band on the Stage C mask at t_a over Stage A valid depth, 1024^2 -> 256^2 (> 0.5); census only")
    rec["band"] = {"n_pixels_256": int(band_s.sum()), "info": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in (info or {}).items()}}
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/census.yaml")
    ap.add_argument("--drive-root", default=os.environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--videos-dir", default=os.environ.get("MT_DATA", "/content/data") + "/videos")
    ap.add_argument("--clips", nargs="*", default=None, help="subset of the config's clips")
    ap.add_argument("--category-rule", default="default")
    ap.add_argument("--category", default=None, help="a clip outside the config (named in --clips): its target category (vehicle, human, or a registered subject)")
    ap.add_argument("--caption", default=None, help="a clip outside the config: its caption (recorded; the category rule reads it when --category is absent)")
    ap.add_argument("--skip-m1", action="store_true", help="Module 1 already run for these clips (its outputs are checked, not re-made)")
    ap.add_argument("--only-failed", action="store_true", help="re-examine only the clips whose last verdict is not PASS (a reasons pass after a code change)")
    ap.add_argument("--sensitivity-g2b", action="store_true", help="SENSITIVITY RUN (reviewer 2026-10-18): Stage A's A-G2b gate report-only; use a THIRD root (never the record's); "
                    "the result is 'k more would pass', never an addition to the experiment set")
    ap.add_argument("--hf-dataset", default=None, help="download this Hugging Face dataset (e.g. John00451/experiment-testset: DAVIS/<clip>.mp4, miradata/<clip>.mp4) into --videos-dir first")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    lock = yaml.safe_load((ROOT / "configs" / "calibration.lock.yaml").read_text(encoding="utf-8"))
    lock_m3 = lock["m3"]; sel = lock_m3["selection"]
    m2e_cfg = yaml.safe_load((ROOT / "configs" / "m2e.yaml").read_text(encoding="utf-8"))["m2e"]
    root = Path(args.drive_root); rep = root / "reports" / "census"; rep.mkdir(parents=True, exist_ok=True)
    jf = rep / "census.json"; rows = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else []
    env = {**os.environ, "MT_ROOT": str(root), "PYTHONUNBUFFERED": "1"}
    if args.sensitivity_g2b:
        assert "exp" not in str(root).rstrip("/").split("/")[-1] or "sens" in str(root), f"the sensitivity run needs its own root, not {root}"
        env["STAGE_A_REPORT_ONLY_GATES"] = "A-G2b"; print("[census] SENSITIVITY RUN: A-G2b report-only; root", root, flush=True)
    clips = [c for c in cfg["clips"] if not args.clips or c["id"] in args.clips]
    if args.clips and (args.category or args.caption):                             # a listed clip run as another target: the given category / caption win
        clips = [{**c, **({"category": args.category} if args.category else {}), **({"caption": args.caption} if args.caption else {})}
                 if c["id"] in args.clips else c for c in clips]
    for cid in (args.clips or []):                                                  # a clip outside the config: the reader's own
        if cid not in {c["id"] for c in clips}:
            assert args.category or args.caption, f"{cid} is not in {args.config}: give --category (or --caption for the category rule)"
            clips.append({"id": cid, "pool": "own", "caption": args.caption, "category": args.category})
    tri_patch = float(lock_m3["trajectory"]["triangulated_gate_patch"])
    if not args.skip_m1:
        sa = yaml.safe_load((ROOT / "configs" / "stage_a.yaml").read_text(encoding="utf-8"))["paths"]
        d4rt = Path(str(sa.get("d4rt_dir", "/content/d4rt")))
        assert d4rt.exists() and any(d4rt.rglob("infer_track_3d*")), (f"the OpenD4RT repo is not at {d4rt}: run `bash scripts/bootstrap.sh` in this session first "
                                                                       "(Stage A fails on every clip without it, and that is not a census result)")
        import importlib.util
        assert importlib.util.find_spec("sam2") is not None, "SAM2 is not installed (Stage C): run `bash scripts/bootstrap.sh` first; a missing dependency is not a census result"
    if args.hf_dataset:
        from huggingface_hub import snapshot_download
        print(f"[census] videos <- {args.hf_dataset} -> {args.videos_dir}", flush=True)
        snapshot_download(repo_id=args.hf_dataset, repo_type="dataset", local_dir=args.videos_dir)
    POOL_DIRS = {"davis": ("DAVIS", "davis"), "miradata": ("miradata", "MiraData")}
    prev_verdict = {r["clip"]: r.get("census") for r in rows}
    for c in clips:
        clip = c["id"]; cat = c.get("category") or category_of(clip, c.get("caption"), args.category_rule)
        if args.only_failed and prev_verdict.get(clip) == "PASS":
            continue
        row = {"clip": clip, "pool": c.get("pool"), "category": cat, "caption": c.get("caption"), "time": time.strftime("%Y-%m-%d %H:%M"), "steps": {}}
        print(f"\n=== census {clip} ({c.get('pool')}) -> target {cat}", flush=True)
        cands = [Path(args.videos_dir) / f"{clip}.mp4"] + [Path(args.videos_dir) / d / f"{clip}.mp4" for d in POOL_DIRS.get(str(c.get("pool")), ())]
        video = next((v for v in cands if v.exists()), None)
        if video is None:
            row["steps"]["video"] = f"missing: looked in {[str(v) for v in cands]}"; row["census"] = "FAIL: video missing"; rows = [r for r in rows if r["clip"] != clip] + [row]; continue
        vdir = str(video.parent)
        # 1. Module 1
        if not args.skip_m1:
            rc, tail = sh([sys.executable, "scripts/run_pipeline.py", "--clip-id", clip, "--videos-dir", vdir], env)
            row["steps"]["module1"] = {"rc": rc, "tail": tail[-1200:]}
        m1_ok = all((root / p).exists() for p in (f"cache/stage_a/{clip}/bundle.npz", f"cache/stage_c/{clip}/stage_c.npz", f"cache/stage_b5/{clip}_qc_b5.json"))
        row["steps"]["module1_outputs"] = m1_ok
        if not m1_ok:
            tail_ = (row["steps"].get("module1", {}) or {}).get("tail", "") or ""
            import re as _re
            m_ = _re.search(r"FAILED at (\w+): ([^|]{0,160})", tail_)
            stage, msg = (m_.group(1), m_.group(2).strip()) if m_ else ("?", "")
            qf = root / "viz" / "stage_a" / clip / "qc_failed.json"
            gates_failed = None
            if stage == "?" and qf.exists() and not (root / "cache" / "stage_a" / clip / "bundle.npz").exists():
                stage, msg = "A", "[stage_a] GATES FAILED (from qc_failed.json; Module 1 not re-run)"
            if stage == "A" and qf.exists():
                try:
                    q_ = json.loads(qf.read_text(encoding="utf-8")); g_ = q_.get("gates") or {}
                    gates_failed = {k: v for k, v in g_.items() if isinstance(v, dict) and v.get("pass") is False} or {k: v for k, v in g_.items() if v is False}
                except Exception as e_:                                             # noqa: BLE001
                    gates_failed = f"unreadable: {e_}"
            crash = ("Error" in msg and "GATES FAILED" not in msg)
            row["module1_refusal"] = {"stage": stage, "message": msg[:160], "gates_failed": gates_failed, "kind": "crash (re-run, not a verdict)" if crash else "gate refusal (a verdict)"}
            row["census"] = (f"UNDETERMINED: Module 1 stage {stage} crashed — {msg[:80]}" if crash else f"FAIL: Module 1 stage {stage} gate refused it" + (f" ({', '.join(gates_failed)})" if isinstance(gates_failed, dict) and gates_failed else ""))
            rows = [r for r in rows if r["clip"] != clip] + [row]; jf.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8"); continue
        # 2. admission reader (stage0)
        rc, tail = sh([sys.executable, "scripts/run_m2a.py", "--clip-id", clip], env)
        row["steps"]["m2a"] = {"rc": rc, "tail": tail[-1200:]}
        if rc != 0 and "hash" in tail.lower():
            # the reader refused a stale upstream record: a B.5 record from before the plane_d sign fix is regenerated by B.5 alone (CPU);
            # any other mismatch rebuilds Module 1 with the current config; then the reader is retried once
            print(f"[census] {clip}: admission reader refused on a hash mismatch; Module 1 rebuilt from this video (every stage, --force), then retry", flush=True)
            cmd1 = [sys.executable, "scripts/run_pipeline.py", "--clip-id", clip, "--videos-dir", vdir, "--force", "--stages", "A,B,C,C5,B5"]   # every stage from THIS video: a partial rebuild mixes frame counts
            rc1, tail1 = sh(cmd1, env)
            rc, tail = sh([sys.executable, "scripts/run_m2a.py", "--clip-id", clip], env)
            row["steps"]["module1_forced"] = {"rc": rc1, "tail": tail1[-1200:]}; row["steps"]["m2a_retry"] = {"rc": rc, "tail": tail[-1200:]}
        if not (root / "cache" / "m2_a" / clip / "stage0.npz").exists():
            row["census"] = "FAIL: admission reader wrote no stage0 (object absent at t_a or B.5 artefact missing)"; rows.append(row); continue
        # 3. source handoff + band
        try:
            rec = source_handoff_and_band(root, clip, cat, lock_m3, m2e_cfg); row["steps"]["source_handoff"] = rec
        except Exception as e:                                                       # noqa: BLE001
            row["steps"]["source_handoff"] = f"{type(e).__name__}: {e}"; row["census"] = "FAIL: source handoff / band could not be built"; rows.append(row); continue
        # 4. camera gate of record (census mode)
        rc1, t1 = sh([sys.executable, "scripts/check_m3_camera_c.py", "--census", "--clips", clip], env)
        rc2, t2 = sh([sys.executable, "scripts/refine_m3_camera_ba.py", "--census", "--clips", clip], env)
        row["steps"]["camera_c"] = {"rc": rc1, "tail": t1[-800:]}; row["steps"]["camera_ba"] = {"rc": rc2, "tail": t2[-800:]}
        ba = {}
        bf = rep / "camera_c_ba.json"
        if bf.exists():
            ba = next((r for r in json.loads(bf.read_text(encoding="utf-8")) if r.get("pair") == f"{clip}/{cat}"), {})
        tri = ba.get("heldout_triangulated", {}); bg = ba.get("band_gauge", {}); gor = ba.get("gate_of_record", {})
        outcome = gor.get("outcome") or ("error: " + ba["error"][:120] if "error" in ba else "not run")
        if "too few points survive" in str(outcome):
            outcome = "undetermined (too few static points for the camera solve)"              # a verdict of the instrument, recorded as such, not an error
        elif "camera C must be solved on every frame" in str(outcome):
            outcome = "undetermined (camera C unsolved on some frames: too few static tracks there)"
        elif "zero-size array" in (t1 + t2):
            outcome = "undetermined (no static tracks at all for the camera solve)"            # the solver's reduction over an empty set: a verdict, not a crash
        elif str(outcome) == "not run":
            outcome = "UNDETERMINED: camera instruments did not run (" + (t1 or t2).strip().splitlines()[-1][:100] + ")" if (t1 or t2).strip() else "UNDETERMINED: camera instruments did not run"
        row["camera"] = {"triangulated_median_px": tri.get("median_px"), "n_points": tri.get("n_points"), "gate_patch": tri_patch, "outcome": outcome}
        cam_pass = str(row["camera"]["outcome"]).startswith("pass")
        # 5. support surface
        n_band = bg.get("n_band_points"); half = bg.get("verify_rel_diff")
        sup_ok = (n_band is not None and n_band >= int(sel["band_points_min"])) and (half is not None and abs(float(half)) <= float(sel["band_halfsplit_max"]))
        evaluated = n_band is not None
        row["support"] = {"n_band_points": n_band, "half_split_rel": half, "ratio": bg.get("ratio"), "ci95": bg.get("ci95"), "pass": bool(sup_ok),
                          "reason": None if sup_ok else ("bridge undefined: no support surface (band points or half-split)" if evaluated else "not evaluated (the camera did not reach the gate)")}
        cam_s = str(row["camera"]["outcome"])
        if cam_s.startswith("UNDETERMINED"):
            row["census"] = cam_s                                                       # instruments did not run: re-run, not a verdict
        else:
            parts = ([] if cam_pass else [f"camera {cam_s}"]) + ([] if (sup_ok or not evaluated) else ["support surface undetermined"])
            row["census"] = "PASS" if (cam_pass and sup_ok) else "FAIL: " + "; ".join(parts)
        rows = [r for r in rows if r["clip"] != clip] + [row]
        jf.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
        print(f"[census] {clip}: {row['census']}  (camera {row['camera']['outcome']} {row['camera']['triangulated_median_px']}; band n {n_band}, half-split {half})", flush=True)
    jf.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    f = lambda v, n=2: "-" if v is None else (f"{v:.{n}f}" if isinstance(v, (int, float)) else str(v))                # noqa: E731
    md = ["# Census — which clips enter the experiment (gates only; set rule in docs/plan_experiment_v1.md)", "",
          f"Camera gate: triangulated split, median held-out reprojection <= {tri_patch} patch. Support: band points >= {sel['band_points_min']}, half-split <= {sel['band_halfsplit_max']}.", "",
          "| clip | pool | target | Module 1 | admission | camera gate: median px (n) -> outcome | band n / half-split / ratio [CI] | support | CENSUS |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (str(r.get("pool")), r["clip"])):
        cam = r.get("camera", {}); sp = r.get("support", {})
        md.append(f"| {r['clip']} | {r.get('pool') or '-'} | {r['category']} | {'ok' if r['steps'].get('module1_outputs') else 'FAIL'} | {'ok' if 'source_handoff' in r['steps'] and isinstance(r['steps']['source_handoff'], dict) else '-'} | "
                  f"{f(cam.get('triangulated_median_px'))} ({cam.get('n_points') or '-'}) -> {cam.get('outcome') or '-'} | {sp.get('n_band_points') or '-'} / {f(sp.get('half_split_rel'), 3)} / {f(sp.get('ratio'), 3)} {sp.get('ci95') or ''} | "
                  f"{'pass' if sp.get('pass') else (sp.get('reason') or '-')} | **{r.get('census')}** |")
    n_pass = sum(r.get("census") == "PASS" for r in rows)
    md += ["", f"Pass rate: {n_pass} / {len(rows)}. A FAIL row is reported, never adjusted; the experiment set is taken from the PASS rows by the pre-written rule."]
    (rep / "census.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[census] {n_pass}/{len(rows)} pass; {rep / 'census.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
