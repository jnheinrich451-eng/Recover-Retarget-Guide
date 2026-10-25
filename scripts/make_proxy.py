#!/usr/bin/env python
"""A proxy for a new target subject, without the review stop (experimental: outside the thesis's evaluated pipeline).

    python scripts/make_proxy.py --category bear --object "a brown bear" --kind rigid [--action walking] [--query "a bear"]
                                 [--seeds 3] [--facing right] [--env colab_public] [--drive-root <root>]

Steps, each resumable: (1) the subject is registered in configs/subjects.yaml; (2) FLUX.1-dev generates `--seeds` candidate images
from the fixed proxy prompt template of configs/m2b2.yaml ("{object}, complete body visible, strict side profile, facing image-right,
..."); (3) OWLv2 + SAM 2 mask each candidate with the subject's query; (4) one candidate is chosen by a fixed rule (a single mask
that covers 5-60 % of the image and does not touch its border; else the largest); (5) scripts/run_m2b2.py --r1 reconstructs it
(SAM 3D Objects for `rigid`, in that environment's interpreter; SAM 3D Body for `human`) and commits the proxy to the cache; (6) the
proxy's key is written back to the registry, so scripts/run_all.py --auto finds it. The facing is taken from the template
(image-right) unless --facing says otherwise; the thesis read it by eye, this script records it as assumed and saves an overlay
(<root>/probe/m2d/proxies/<category>/s<k>_overlay.png) for a glance. Phase-2 environment (diffusers) plus the reconstructor's.
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
from src.anchor import subjects  # noqa: E402

FLUX = "black-forest-labs/FLUX.1-dev"


def choose(masks: list[np.ndarray]) -> tuple[int, str]:
    """The fixed rule: first seed whose mask covers 5-60 % of the image and keeps 2 px from every border; else the largest."""
    ok = []
    for k, m in enumerate(masks):
        frac = float(m.mean())
        border = bool(m[:2].any() or m[-2:].any() or m[:, :2].any() or m[:, -2:].any())
        if 0.05 <= frac <= 0.60 and not border:
            ok.append(k)
    if ok:
        return ok[0], "first seed with a single mask covering 5-60 % of the image and clear of the border"
    k = int(np.argmax([m.sum() for m in masks]))
    return k, "no seed met the rule; the largest mask (check the overlay)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, help="a short name, e.g. bear (becomes the category of every pair that targets it)")
    ap.add_argument("--object", required=True, help="the subject's identity, e.g. 'a brown bear'")
    ap.add_argument("--kind", required=True, choices=subjects.KINDS, help="rigid: SAM 3D Objects + the object render; human: SAM 3D Body, camera only")
    ap.add_argument("--action", default="moving", help="the verb of the anchor prompt, e.g. walking, driving, running")
    ap.add_argument("--query", default=None, help="the detector's text for the mask (default: the object text)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--facing", default="right", choices=["left", "right"], help="the candidate's facing (the template asks for image-right)")
    ap.add_argument("--steps", type=int, default=28); ap.add_argument("--guidance", type=float, default=3.5); ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--env", default="colab_public", help="preset in configs/environment.yaml (the SAM 3D Objects interpreter)")
    ap.add_argument("--drive-root", default=os.environ.get("MT_ROOT"), help="the root that holds cache/ (MT_ROOT)")
    ap.add_argument("--skip-generate", action="store_true", help="use candidates already under probe/m2d/proxies/<category>/")
    args = ap.parse_args(argv)
    assert args.drive_root, "set MT_ROOT or pass --drive-root"
    root = Path(args.drive_root)
    os.environ["MT_ROOT"] = str(root)

    # 1. register
    rec = subjects.register(args.category, kind=args.kind, object=args.object, action=args.action, query=args.query or args.object,
                            arbiter_label=args.object.split()[-1], facing_offset_deg=0)
    print(f"[proxy] registered {args.category}: {rec}", flush=True)

    # 2. candidates
    out = root / "probe" / "m2d" / "proxies" / args.category; out.mkdir(parents=True, exist_ok=True)
    tmpl = yaml.safe_load((ROOT / "configs" / "m2b2.yaml").read_text(encoding="utf-8"))["m2b2"]["proxy_prompt_template"]
    prompt = tmpl.format(object=args.object)
    todo = [k for k in range(args.seeds) if not (out / f"s{k}.png").exists()]
    if todo and not args.skip_generate:
        import torch
        from diffusers import FluxPipeline
        t0 = time.time()
        pipe = FluxPipeline.from_pretrained(FLUX, torch_dtype=torch.bfloat16).to("cuda")
        print(f"[proxy] FLUX.1-dev loaded in {time.time() - t0:.0f} s; prompt: {prompt!r}", flush=True)
        for k in todo:
            img = pipe(prompt, height=args.size, width=args.size, num_inference_steps=args.steps, guidance_scale=args.guidance,
                       generator=torch.Generator("cuda").manual_seed(k)).images[0]
            img.save(out / f"s{k}.png"); print(f"[proxy] candidate s{k} written", flush=True)
        del pipe; torch.cuda.empty_cache()
    cands = [out / f"s{k}.png" for k in range(args.seeds) if (out / f"s{k}.png").exists()]
    assert cands, f"no candidate images under {out}"

    # 3. masks
    import cv2
    from src.anchor.m2d import segment as S
    need = [p for p in cands if not p.with_name(p.stem + "_mask.png").exists()]
    if need:
        seg = S.load_segmenter()
        for p in need:
            img = cv2.imread(str(p))[..., ::-1].copy()
            m, info = S.segment(seg, img, args.category)
            cv2.imwrite(str(p.with_name(p.stem + "_mask.png")), (m.astype(np.uint8) * 255))
            ov = img.astype(np.float32); ov[m] = 0.5 * ov[m] + 0.5 * np.array([232, 200, 106]); cv2.imwrite(str(p.with_name(p.stem + "_overlay.png")), ov[..., ::-1].astype(np.uint8))
            print(f"[proxy] mask for {p.name}: {info}", flush=True)
    masks = [cv2.imread(str(p.with_name(p.stem + "_mask.png")), cv2.IMREAD_GRAYSCALE) > 127 for p in cands]

    # 4. choose
    k, why = choose(masks)
    choice = {"category": args.category, "object": args.object, "kind": args.kind, "prompt": prompt, "selected_seed": int(cands[k].stem[1:]),
              "rule": why, "facing": args.facing, "facing_source": "assumed from the template (facing image-right)" if args.facing == "right" else "given",
              "mask_fraction": [float(m.mean()) for m in masks], "time": time.strftime("%Y-%m-%d %H:%M")}
    (out / "choice.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    print(f"[proxy] chose s{choice['selected_seed']}: {why}; facing {args.facing} ({choice['facing_source']}); overlay {out / f's{choice['selected_seed']}_overlay.png'}", flush=True)

    # 5. reconstruct and commit
    envs = yaml.safe_load((ROOT / "configs" / "environment.yaml").read_text(encoding="utf-8"))
    py = envs.get(args.env, {}).get("python", {})
    interp = str(py.get("sam3d_objects", sys.executable)) if args.kind == "rigid" else sys.executable
    cmd = [interp, "scripts/run_m2b2.py", "--r1", "--category", args.category, "--image", str(cands[k]), "--mask", str(cands[k].with_name(cands[k].stem + "_mask.png")),
           "--facing", args.facing, "--selected-k", str(choice["selected_seed"]), "--commit-cache"]
    print("[proxy] $", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "MT_ROOT": str(root), "PYTHONUNBUFFERED": "1"})
    assert r.returncode == 0, "the reconstructor stopped; read its log"

    # 6. the proxy key back into the registry
    cdir = root / "cache" / "m2_b2" / args.category
    keys = sorted((d for d in cdir.iterdir() if (d / "proxy.npz").exists()), key=lambda d: d.stat().st_mtime)
    assert keys, f"no proxy under {cdir}"
    key = keys[-1].name
    subjects.register(args.category, proxy_key=key)
    print(f"[proxy] {args.category}: proxy {key} in the cache and registered; run_all.py --auto --category {args.category} will use it", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
