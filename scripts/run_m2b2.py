"""m2_b2 runner (SKILL_m2_b2 v1.1 §12 step 5) — R1 proxy per category, R2 anchor per clip.

Objects variant: run with the SAM 3D Objects env python (/content/micromamba/envs/sam3d-objects/bin/python).
Body variant: plain python (base env).

  R1:  python scripts/run_m2b2.py --r1 --category vehicle --image <png> --mask <png> --facing right --selected-k 0
  R2:  python scripts/run_m2b2.py --r2 --category vehicle --clip 000000000284.0.006

HARD STOP (SKILL §12): human review before m2_b2 runs on a real target category and before any
category cache is populated — this runner therefore writes under <cache>/_review/ until --commit-cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2b2 import cache, pipeline, pointmap          # noqa: E402
from src.anchor.m2b2.objects import load_m2b2_config           # noqa: E402

RENDER_PX = 1024


def anchor_inputs(cfg_a, clip):
    """Stage A depth + K at 256, Stage C mask + t_c, native frame (the probe's load_anchor_geometry)."""
    import cv2
    from src.preprocess.bundle import read_bundle
    from src.preprocess.stage_c import finalize
    b = read_bundle(cfg_a, clip)
    M, present, t_c = finalize.read_mask(Path(cfg_a["paths"]["stage_c_cache"]) / clip)
    assert bool(present[t_c]), f"{clip}: subject not present at t_c={t_c}"
    cap = cv2.VideoCapture(b["provenance"]["clip_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_c)); ok, fr = cap.read(); cap.release()
    assert ok, f"{clip}: frame {t_c} not readable"
    frame = np.ascontiguousarray(fr[..., ::-1])
    Hn, Wn = frame.shape[:2]
    D = b["D"][t_c].astype(np.float64); K256 = b["K"][t_c].astype(np.float64)
    K_native = K256.copy(); K_native[0] *= Wn / 256.0; K_native[1] *= Hn / 256.0
    hashes = {"stage_a": b["provenance"].get("config_hash"), "stage_c": None}
    qc_c = Path(cfg_a["paths"]["stage_c_cache"]) / clip / "qc_c.json"
    if qc_c.exists():
        hashes["stage_c"] = json.loads(qc_c.read_text(encoding="utf-8")).get("config_hash")
    return {"clip": clip, "t_c": int(t_c), "frame": frame, "mask256": M[t_c], "D256": D, "K256": K256,
            "K_native": K_native, "hashes": hashes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", action="store_true"); ap.add_argument("--r2", action="store_true")
    ap.add_argument("--category", required=True)
    ap.add_argument("--image"); ap.add_argument("--mask"); ap.add_argument("--facing"); ap.add_argument("--selected-k", type=int, default=0)
    ap.add_argument("--gen", help="name of a generated candidate in <drive>/probe/sam3d/gen, e.g. p4_vehicle_proxy_s0 "
                                  "(sets --image/--mask/--selected-k from it)")
    ap.add_argument("--clip")
    ap.add_argument("--config", default="configs/stage_a.yaml"); ap.add_argument("--config-m2b2", default="configs/m2b2.yaml")
    ap.add_argument("--commit-cache", action="store_true", help="write to the real cache instead of <cache>/_review/ (after the hard stop)")
    ap.add_argument("--promote", help="move <cache_r1>/_review/<category>/<KEY> to the cache after re-deriving KEY from its record (no model)")
    args = ap.parse_args()
    from src.preprocess.config import load_config
    cfg_a = load_config(args.config)
    cfg = load_m2b2_config(args.config_m2b2, drive_root=cfg_a["paths"]["drive_root"])
    variant = cache.route(args.category, cfg)
    print(f"[m2b2] category {args.category} -> {variant}; config hash {cfg['config_hash']}")
    if args.promote:
        dst = cache.promote(Path(cfg["paths"]["cache_r1"]) / "_review", Path(cfg["paths"]["cache_r1"]), args.category,
                            args.promote, cfg["m2b2"]["proxy_prompt_template"])
        print(f"[m2b2] promoted {args.category}/{args.promote} -> {dst} (key re-derived from its record)"); return 0
    if variant == "objects":
        from src.anchor.m2b2.objects import ObjectsReconstructor
        recon = ObjectsReconstructor(cfg)
    else:
        from src.anchor.m2b2.body import BodyReconstructor
        recon = BodyReconstructor(cfg)
    print(f"[m2b2] {variant} loaded in {recon.load_seconds:.0f}s; model {recon.model_version}")
    import cv2
    if args.r1:
        if args.gen:
            gen_dir = Path(cfg_a["paths"]["drive_root"]) / "probe" / "sam3d" / "gen"
            args.image, args.mask = str(gen_dir / f"{args.gen}.png"), str(gen_dir / f"{args.gen}_mask.png")
            args.selected_k = int(args.gen.rsplit("_s", 1)[1])
        assert args.image and args.mask and args.facing, "--r1 needs --image --mask --facing (or --gen <name> --facing)"
        assert Path(args.image).exists() and Path(args.mask).exists(), (args.image, args.mask)
        img = cv2.imread(args.image)[..., ::-1].copy()
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE) > 127
        root = Path(cfg["paths"]["cache_r1"]) / ("" if args.commit_cache else "_review")
        out = pipeline.r1_proxy(recon, variant, args.category, img, mask, args.facing, args.image, args.selected_k, cfg, root)
        print(f"[m2b2] R1 proxy -> {out['out_dir']} (medoid seed {out['medoid_seed']})")
        q = out["qc"]
        for p in q["per_seed"]:
            print(f"  seed {p['seed']}: ref_hb {p['ref_hb']:.4f} ref_dp/hb {p['ref_dp'] / p['ref_hb']:.3f} ref_a0/hb2 {p['ref_a0'] / p['ref_hb'] ** 2:.3f} "
                  f"p01-min {p['ref_hb_p01_minus_min']:+.4f} IoU {p['iou_vs_input_mask']:.3f} {'watertight' if p['watertight'] else 'OPEN'} {p['seconds']:.0f}s")
        print(f"  medoid distances {q['medoid_distances']}; cache_key {q['cache_key']}")
    if args.r2:
        assert args.clip, "--r2 needs --clip"
        g = anchor_inputs(cfg_a, args.clip)
        root = Path(cfg["paths"]["cache_r2"]) / ("" if args.commit_cache else "_review")
        if variant == "objects":
            img = cv2.resize(g["frame"], (RENDER_PX, RENDER_PX), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(g["mask256"].astype(np.uint8), (RENDER_PX, RENDER_PX), interpolation=cv2.INTER_NEAREST).astype(bool)
            K1024 = g["K256"].copy(); K1024[:2] *= RENDER_PX / 256.0
            pm, zmed = pointmap.stage_a_pointmap(g["D256"], g["K256"], g["mask256"], cfg["m2b2"]["objects"]["bg_depth_mult"])
            out = pipeline.r2_anchor(recon, variant, args.category, args.clip, img, mask, K1024, pm, g["t_c"], cfg, root, g["hashes"])
        else:
            Hn, Wn = g["frame"].shape[:2]
            mask_n = cv2.resize(g["mask256"].astype(np.uint8), (Wn, Hn), interpolation=cv2.INTER_NEAREST).astype(bool)
            out = pipeline.r2_anchor(recon, variant, args.category, args.clip, None, None, None, None, g["t_c"], cfg, root, g["hashes"],
                                     K_native=g["K_native"], native_frame=g["frame"], native_mask=mask_n)
        print(f"[m2b2] R2 anchor -> {out['out_dir']} (medoid seed {out['medoid_seed']})")
        for p in out["qc"]["per_seed"]:
            print("  " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in p.items() if k not in ("t_cv",)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
