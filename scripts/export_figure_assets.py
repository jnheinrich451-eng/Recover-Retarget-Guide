#!/usr/bin/env python
"""Export clean per-pair image assets + a manifest for thesis figures. READ-ONLY on every cache; writes only under
<drive_root>/reports/figure_assets/. No model is loaded, nothing is recomputed, no debug text is drawn.

Per pair it collects, where present:
  ref_frame.png      source frame at t_a (native resolution, from the video)
  ref_mask.png       Stage C mask at t_a, resized (nearest) to the frame, 0/255
  control.png        C3 depth control as fed to the image model (cache/m2_c)
  seed_<k>.png       the K' candidates (cache/m2_d/.../candidates)
  anchor.png         the selected anchor (cache/m2_d)
  manifest.json      pair id, t_a, disposition, selected seed, human ranks, alpha, A, R2 IoU, file list, missing list

    python scripts/export_figure_assets.py                 # all pairs with a handoff
    python scripts/export_figure_assets.py --pairs 000000000265.2.005/vehicle
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--videos", default=__import__("os").environ.get("MT_DATA", "/content/data") + "/videos")
    ap.add_argument("--pairs", nargs="+", default=None, help="<clip>/<category>; default: every pair with a handoff.json")
    args = ap.parse_args(argv)
    root = Path(args.drive_root); cache = root / "cache"
    out_root = root / "reports" / "figure_assets"
    pairs = args.pairs or sorted(f"{p.parent.parent.name}/{p.parent.name}" for p in (cache / "m2_e").glob("*/*/handoff.json"))
    ranks = {}
    for rf in (cache / "m2_d").glob("**/phase2_ranks.csv"):
        for row in csv.reader(open(rf, encoding="utf-8")):
            if len(row) >= 5 and row[2].isdigit():
                ranks.setdefault(f"{row[0]}/{row[1]}", {})[int(row[2])] = row[4]
    index = []
    for pair in pairs:
        clip, cat = pair.split("/")
        out = out_root / clip / cat; out.mkdir(parents=True, exist_ok=True)
        files, missing = [], []

        def take(src: Path, name: str):
            if src.exists():
                shutil.copyfile(src, out / name); files.append(name)
            else:
                missing.append(f"{name} <- {src}")

        hp = cache / "m2_e" / clip / cat / "handoff.json"
        h = json.loads(hp.read_text(encoding="utf-8")) if hp.exists() else {}
        t_a = int(h.get("t_a", 0))
        take(cache / "m2_c" / clip / cat / "control.png", "control.png")
        take(cache / "m2_d" / clip / cat / "anchor.png", "anchor.png")
        for s in sorted((cache / "m2_d" / clip / cat / "candidates").glob("seed_*.png")):
            take(s, s.name)
        # reference frame + mask at t_a
        try:
            import cv2
            cap = cv2.VideoCapture(str(Path(args.videos) / f"{clip}.mp4"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, t_a); ok, fr = cap.read(); cap.release()
            assert ok, "frame read failed"
            cv2.imwrite(str(out / "ref_frame.png"), fr); files.append("ref_frame.png")
            from src.preprocess.stage_c import finalize
            M, _, _ = finalize.read_mask(cache / "stage_c" / clip)
            m = cv2.resize(M[t_a].astype(np.uint8) * 255, (fr.shape[1], fr.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out / "ref_mask.png"), m); files.append("ref_mask.png")
        except Exception as e:
            missing.append(f"ref_frame/ref_mask: {type(e).__name__}: {e}")
        # the two overlay panels: placed proxy on the reference frame; rebuilt target on the anchor (outlines only, no text)
        try:
            import cv2

            def outline(img, mask, colour):
                cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                o = img.copy(); cv2.drawContours(o, cs, -1, colour, 4, cv2.LINE_AA); return o

            with np.load(cache / "m2_c" / clip / cat / "control.npz") as z:
                keys = list(z.files)
                assert "M_obj" in keys, f"control.npz has no M_obj (keys: {keys})"
                Mo = z["M_obj"].astype(bool)
            ref = cv2.imread(str(out / "ref_frame.png")); assert ref is not None, "ref_frame.png not exported"
            ref_sq = cv2.resize(ref, (Mo.shape[1], Mo.shape[0]), interpolation=cv2.INTER_AREA)      # the control's own (squashed) frame
            wide = cv2.resize(outline(ref_sq, Mo, (60, 200, 255)), (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / "proxy_on_ref.png"), wide); cv2.imwrite(str(out / "proxy_mask.png"), Mo.astype(np.uint8) * 255)
            files += ["proxy_on_ref.png", "proxy_mask.png"]
        except Exception as e:
            missing.append(f"proxy_on_ref: {type(e).__name__}: {e}")
        try:
            import cv2
            from src.anchor.m2c import proxy as _proxy
            with np.load(cache / "m2_e" / clip / cat / "g_tar_mesh.npz") as z:
                Vg, Fg, Kg = z["vertices"].astype(np.float64), z["faces"].astype(int), z["K_1024"].astype(np.float64)
            anc = cv2.imread(str(out / "anchor.png")); assert anc is not None, "anchor.png not exported"
            _, Mg = _proxy.render_perspective(Vg, Fg, Kg, anc.shape[0], anc.shape[1])
            cs, _ = cv2.findContours(Mg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            o = anc.copy(); cv2.drawContours(o, cs, -1, (255, 60, 255), 4, cv2.LINE_AA)
            cv2.imwrite(str(out / "gtar_on_anchor.png"), o); cv2.imwrite(str(out / "gtar_mask.png"), Mg.astype(np.uint8) * 255)
            files += ["gtar_on_anchor.png", "gtar_mask.png"]
        except Exception as e:
            missing.append(f"gtar_on_anchor: {type(e).__name__}: {e}")
        qd = cache / "m2_d" / clip / cat / "qc_m2d.json"
        sel = json.loads(qd.read_text(encoding="utf-8")).get("selected_seed") if qd.exists() else None
        man = {"pair": pair, "t_a": t_a, "disposition": h.get("disposition"), "label": h.get("label"), "category": cat,
               "selected_seed": sel, "human_ranks": ranks.get(pair), "alpha": h.get("alpha"), "A_deg": h.get("A_deg"),
               "R2_iou": h.get("R2_iou"), "n_frames_after_t_a": h.get("n_frames_after_t_a"), "files": files, "missing": missing}
        (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
        index.append(man)
        print(f"{pair}: {len(files)} files, {len(missing)} missing")
    (out_root / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"-> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
