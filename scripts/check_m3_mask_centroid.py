#!/usr/bin/env python
"""Reviewer 2026-10-14, vehicle table item 3 / human item 5: an arbiter for the object's screen motion that uses NO tracks — the per-frame
centroid of the object's semantic mask (SegFormer-B2 ADE20K, the instrument of scripts/make_m3_dynamic_mask.py; CPU) on the source clip, the
A0 videos and the A3 videos, with the mask's area and its IoU with the previous frame (figure continuity). The question it decides: does the
object hold its screen position (follow-cam) or drift across the frame, and by how many patches, in the source and in each generated video.

Not SAM2 (the reviewer's instrument of choice): SegFormer is what runs on CPU here; a class mask, not an instance mask — on a frame with one
object of the class the two agree, and the area / IoU columns show when they would not.

    python scripts/check_m3_mask_centroid.py --pair 000000000265.2.005/vehicle --label car --videos A0_seed0_rec_F33.mp4 A3_seed0_rec_F33_obj_ow0.75_sm_sky.mp4 source
    python scripts/check_m3_mask_centroid.py --pair 000000000265.2.005/human --label person --cache-m3 measurements/m3 --reports measurements --videos A0_seed0_rec_F33.mp4 ...

Writes <reports>/m3_s4/mask_centroid_<clip>_<cat>.{json,md} (rule 24: adds rows per video, refuses to replace without --force).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

WAN_W, WAN_H, PX = 832, 480, 16
MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"
ADE = {"person": 12, "car": 20, "animal": 126, "horse": 126, "truck": 83, "bus": 80, "van": 102, "bicycle": 127, "motorbike": 116, "boat": 76}



def source_video_path(videos_dir, clip, n_frames_expected=None, sha256_expected=None):
    """The source video of record for a clip (reviewer 2026-10-20: asserted by CONTENT, never by lookup order). Candidates: the pool
    subfolders and the flat file under <videos_dir>; the one whose frame count equals the record's (the Stage C mask length) and, when the
    set carries a hash (configs/experiment_video_hashes.yaml), whose sha256 equals it. Exactly one candidate may satisfy that."""
    import hashlib
    cands = sorted(Path(videos_dir).glob(f"*/{clip}.mp4")) + ([Path(videos_dir) / f"{clip}.mp4"] if (Path(videos_dir) / f"{clip}.mp4").exists() else [])
    assert cands, f"source video {clip}.mp4 not found under {videos_dir} (a pool subfolder or flat)"
    if n_frames_expected is None and sha256_expected is None:
        assert len(cands) == 1, f"source video {clip}.mp4 is ambiguous under {videos_dir} and no frame count / hash was given: {cands}"
        return cands[0]
    ok = []
    for c in cands:
        cap = cv2.VideoCapture(str(c)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        h = hashlib.sha256(c.read_bytes()).hexdigest() if sha256_expected else None
        if sha256_expected is not None:
            if h == sha256_expected:
                ok.append((c, n, h))
        elif n_frames_expected is None or n >= int(n_frames_expected):           # Stage A consumed the FIRST n_frames_expected frames: a longer file (the six pairs' originals) qualifies
            ok.append((c, n, h))
    if len(ok) > 1 and n_frames_expected is not None:
        exact = [o for o in ok if o[1] == int(n_frames_expected)]
        ok = exact or ok
    assert len(ok) == 1, f"source video {clip}.mp4: {len(ok)} candidates match frames >= {n_frames_expected} / hash {sha256_expected and sha256_expected[:12]} under {videos_dir}: {[str(c) for c in cands]}"
    return ok[0][0]


def expected_source(root, clip, cat):
    """(frame count, sha256 or None) of the clip's source of record: the handoff's n_frames_source and the set's recorded hash when present."""
    h = json.loads((Path(root) / "cache" / "m2_e" / clip / cat / "handoff.json").read_text(encoding="utf-8"))
    hf = ROOT / "configs" / "experiment_video_hashes.yaml"
    hashes = (yaml.safe_load(hf.read_text(encoding="utf-8")) or {}).get("sha256", {}) if hf.exists() else {}
    return int(h["n_frames_source"]), hashes.get(clip)

def read_video(path, n, W, H):
    cap = cv2.VideoCapture(str(path)); fr = []
    while len(fr) < n:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[:2] != (H, W):
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        fr.append(f[..., ::-1].copy())
    cap.release(); assert len(fr) == n, (path, len(fr), n)
    return np.stack(fr)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--pair", required=True)
    ap.add_argument("--label", required=True, choices=sorted(ADE), help="ADE20K class of the object")
    ap.add_argument("--videos", nargs="+", required=True, help="video names in the pair's cache, or 'source' (the clip from t_a, --frames frames)")
    ap.add_argument("--videos-dir", default=__import__("os").environ.get("MT_DATA", "/content/data") + "/videos")
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--cache-m3", default=None, help="local copy of cache/m3 (e.g. measurements/m3)")
    ap.add_argument("--reports", default=None, help="local reports root (e.g. measurements)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    torch.set_num_threads(max(1, torch.get_num_threads()))
    root = Path(args.drive_root); clip, cat = args.pair.split("/")
    d = (Path(args.cache_m3) if args.cache_m3 else root / "cache" / "m3") / clip / cat
    rep = (Path(args.reports) if args.reports else root / "reports") / "m3_s4"; rep.mkdir(parents=True, exist_ok=True)
    with np.load(d / "geometry.npz") as z:
        t_a = int(z["src_frames"][0]); c_tar = z["c_tar"].astype(np.float64); E_l = z["E_l"].astype(np.float64); Kp = z["Kp"].astype(np.float64)
    # the expectation the grader uses for the object: c_tar projected on the lattice, displacement from frame 0 (patches), at the latent frames
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.generate.m3.geometry import project as PJ
    pc, _ = PJ.project(c_tar[:, None, :], E_l, Kp); exp_obj = [float(np.linalg.norm(pc[j, 0] - pc[0, 0])) for j in range(len(E_l))]
    proc = SegformerImageProcessor.from_pretrained(MODEL); model = SegformerForSemanticSegmentation.from_pretrained(MODEL).eval()
    cls = ADE[args.label]
    rows = []
    for name in args.videos:
        if name == "source":
            n_exp, sha_exp = expected_source(root, clip, cat); video = read_video(source_video_path(args.videos_dir, clip, n_exp, sha_exp), t_a + args.frames, WAN_W, WAN_H)[t_a:]
        else:
            video = read_video(d / name, args.frames, WAN_W, WAN_H)
        cen = []; area = []; ious = []; prev = None
        for f in video:
            with torch.no_grad():
                out = model(**proc(images=f, return_tensors="pt")).logits
            seg = torch.nn.functional.interpolate(out, size=(WAN_H, WAN_W), mode="bilinear", align_corners=False)[0].argmax(0).numpy()
            m = seg == cls
            # the largest connected component: one object of the class (a second instance would show as an area jump)
            n_, lab = cv2.connectedComponents(m.astype(np.uint8))
            if n_ > 2:
                sizes = [(lab == i).sum() for i in range(1, n_)]; m = lab == (1 + int(np.argmax(sizes)))
            if m.sum() < 50:
                cen.append([np.nan, np.nan]); area.append(float(m.mean())); ious.append(np.nan); prev = None; continue
            ys, xs = np.nonzero(m); cen.append([float(xs.mean()), float(ys.mean())]); area.append(float(m.mean()))
            ious.append(float((m & prev).sum() / max((m | prev).sum(), 1)) if prev is not None else np.nan); prev = m
        cen = np.array(cen); disp = (cen - cen[0]) / PX
        T_l = 1 + (len(video) - 1) // 4
        lat = [float(np.linalg.norm(disp[4 * j])) if np.isfinite(disp[4 * j]).all() else None for j in range(T_l)]
        vel = np.linalg.norm(np.diff(cen, axis=0), axis=1) / PX
        row = {"pair": args.pair, "video": name, "label": args.label, "frames": len(video), "t_a": t_a if name == "source" else None,
               "centroid_disp_patch_at_latent_frames": lat, "centroid_disp_last_patch": lat[-1], "expected_obj_disp_last_patch": exp_obj[T_l - 1],
               "centroid_disp_vector_last_patch": disp[-1].tolist() if np.isfinite(disp[-1]).all() else None,
               "path_length_patch": float(np.nansum(vel)), "mask_area_share": {"first": area[0], "median": float(np.median(area)), "min": float(np.min(area)), "last": area[-1]},
               "mask_iou_prev": {"median": float(np.nanmedian(ious)), "min": float(np.nanmin(ious)) if np.isfinite(ious).any() else None},
               "n_frames_without_mask": int(np.isnan(cen[:, 0]).sum()), "instrument": MODEL}
        rows.append(row)
        print(f"[mask centroid] {name}: {args.label} centroid moved {row['centroid_disp_last_patch'] if row['centroid_disp_last_patch'] is not None else float('nan'):.2f} patch by the last frame "
              f"(expected {exp_obj[T_l - 1]:.2f}; path {row['path_length_patch']:.2f}); area first/median/last {area[0]:.3f}/{np.median(area):.3f}/{area[-1]:.3f}; IoU median {np.nanmedian(ious):.2f} min {np.nanmin(ious):.2f}; "
              f"frames without mask {row['n_frames_without_mask']}", flush=True)
    jf = rep / f"mask_centroid_{clip}_{cat}.json"
    prev_rows = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else []
    key = lambda r: (r["pair"], r["video"])                                                                         # noqa: E731
    dup = {key(r) for r in prev_rows} & {key(r) for r in rows}
    assert not dup or args.force, f"rule 24: {sorted(dup)} already recorded; pass --force"
    rows = sorted([r for r in prev_rows if key(r) not in {key(x) for x in rows}] + rows, key=key)
    jf.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    f = lambda v: "-" if v is None else f"{v:.2f}"                                                                   # noqa: E731
    md = [f"# Mask-centroid arbiter — {args.pair} ({args.label}, {MODEL}, no tracks)", "",
          "| video | centroid displacement at the last frame (patch) | expected (c_tar projected) | path length | mask area first / median / last | IoU with previous frame, median / min | frames without mask |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        a = r["mask_area_share"]; io = r["mask_iou_prev"]
        md.append(f"| {r['video']} | {f(r['centroid_disp_last_patch'])} | {f(r['expected_obj_disp_last_patch'])} | {f(r['path_length_patch'])} | {a['first']:.3f} / {a['median']:.3f} / {a['last']:.3f} | {f(io['median'])} / {f(io['min'])} | {r['n_frames_without_mask']} |")
    md += ["", "Reading: the displacement is the object's semantic-mask centroid against its own frame-0 position, in patches of 16 px; 'expected' is the target's object path on the lattice "
           "(the grader's expectation). A source row far from 'expected' says the placement / composition differs from the reference's screen motion, not that the tracks slid. "
           "Area and IoU are the figure's continuity: a collapse in IoU or area marks a broken figure."]
    (rep / f"mask_centroid_{clip}_{cat}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[mask centroid] wrote {jf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
