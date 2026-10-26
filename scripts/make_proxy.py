#!/usr/bin/env python
"""A proxy for a new target subject, without the review stop (experimental: outside the thesis's evaluated pipeline).

    python scripts/make_proxy.py --category bear --object "a brown bear" --kind rigid [--action walking] [--query "a bear"]
                                 [--seeds 3] [--facing right] [--env colab_public] [--drive-root <root>]

Steps, each resumable: (1) the subject is registered in configs/subjects.yaml; (2) FLUX.1-dev generates `--seeds` candidate images
from the fixed proxy prompt template of configs/m2b2.yaml ("{object}, complete body visible, strict side profile, facing image-right,
..."); (3) OWLv2 + SAM 2 mask each candidate with the subject's query; (4) one candidate is chosen by a fixed rule (a single mask
that covers 5-60 % of the image and does not touch its border; else the largest); (5) scripts/run_m2b2.py --r1 reconstructs it
(SAM 3D Objects for `rigid`, in that environment's interpreter; SAM 3D Body for `human`) and commits the proxy to the cache; (6) the
proxy's key is written back to the registry, so scripts/run_all.py --auto finds it.

The facing (left or right) is READ, not assumed: the generator does not obey "facing image-right" reliably (measured 2026-10-27, the
bear's seed 0 faced left). The thesis read it by eye once per category; here the detector finds the subject's head inside its mask
and the head's side of the body's centroid is the facing, guarded by a mirror fixture (the flipped image must read the other way)
and a margin floor (a head at the centre is a view toward or away from the camera). The rule then prefers candidates whose facing
could be read, and among them the template's side. A candidate that cannot be read is recorded as assumed, with a warning; --facing
is the human's reading and wins over the detector's, with the disagreement recorded; --select-seed picks the candidate by eye after a
look at the overlays (<root>/probe/m2d/proxies/<category>/s<k>_overlay.png). Phase-2 environment (diffusers) plus the reconstructor's.
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
TEMPLATE_FACING = "right"                                        # what the proxy prompt template asks for ("facing image-right")
FACING_MIN_MARGIN = 0.10                                         # the head's centre at least this fraction of the body's width from the centroid; a reading floor, not a gate


def read_facing(seg: dict, img_rgb: np.ndarray, mask: np.ndarray, object_text: str, head_query: str | None = None) -> dict:
    """Which way a candidate faces, from where its head is. The detector (OWLv2, the one that boxed the subject) is asked for the head;
    the best box whose centre lies on the mask and that is smaller than half the mask's box is the head; its centre's offset from the
    mask's centroid, in units of the mask's width, is the margin: positive = facing image-right. Two guards: |margin| below
    FACING_MIN_MARGIN is unread (toward or away from the camera, or a wrong box), and the mirrored image must read the other way
    (the fixture with the known answer, rule 17); if it does not, the reading is unread."""
    import torch
    from PIL import Image
    from src.anchor.m2d import segment as S
    owl_p, owl = seg["owl_p"], seg["owl"]
    q = head_query or f"the head of {object_text}"

    def head(img, m):
        H, W = m.shape
        ys, xs = np.nonzero(m)
        if not len(xs):
            return None
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        inputs = owl_p(text=[[q]], images=Image.fromarray(np.ascontiguousarray(img)), return_tensors="pt").to("cuda")
        with torch.no_grad():
            post = getattr(owl_p, "post_process_grounded_object_detection", None) or owl_p.post_process_object_detection
            res = post(owl(**inputs), threshold=S.DET_THRESHOLD, target_sizes=torch.tensor([[max(H, W), max(H, W)]]))[0]
        best = None
        for b, sc in zip(res["boxes"].detach().cpu().numpy().astype(float), res["scores"].detach().cpu().numpy().astype(float)):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            on_mask = x0 <= cx <= x1 and y0 <= cy <= y1 and bool(m[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))])
            small = (b[2] - b[0]) * (b[3] - b[1]) <= 0.5 * max(x1 - x0, 1) * max(y1 - y0, 1)
            if on_mask and small and (best is None or sc > best[1]):
                best = (b, float(sc))
        if best is None:
            return None
        return {"margin": float((((best[0][0] + best[0][2]) / 2) - xs.mean()) / max(x1 - x0, 1)), "score": best[1], "box": [float(v) for v in best[0]]}

    out = {"query": q, "read": None, "margin": None, "mirror_margin": None, "head_score": None, "head_box": None, "why": ""}
    r = head(img_rgb, mask)
    if r is None:
        out["why"] = "no head found on the mask"
        return out
    out.update(margin=r["margin"], head_score=r["score"], head_box=r["box"])
    if abs(r["margin"]) < FACING_MIN_MARGIN:
        out["why"] = f"head within {FACING_MIN_MARGIN:.2f} of the body's centre (margin {r['margin']:+.2f}): toward or away from the camera, or a wrong box"
        return out
    rm = head(img_rgb[:, ::-1], mask[:, ::-1])
    out["mirror_margin"] = None if rm is None else rm["margin"]
    if rm is None or (rm["margin"] > 0) == (r["margin"] > 0):
        out["why"] = f"mirror fixture failed: margin {r['margin']:+.2f}, mirrored {'none' if rm is None else format(rm['margin'], '+.2f')}"
        return out
    out["read"] = "right" if r["margin"] > 0 else "left"
    out["why"] = f"head centre {r['margin']:+.2f} of the body width from the centroid (mirrored {rm['margin']:+.2f}), detector score {r['score']:.2f}"
    return out


def choose(masks: list[np.ndarray], readings: list[dict] | None = None, want: str = TEMPLATE_FACING) -> tuple[int, str]:
    """The fixed rule: among seeds whose mask covers 5-60 % of the image and keeps 2 px from every border, the first whose facing was
    read as `want`; else the first whose facing was read at all (its own facing is then used); else the first unread one, with a
    warning; with no seed clear of the border, the largest mask."""
    ok = []
    for k, m in enumerate(masks):
        frac = float(m.mean())
        border = bool(m[:2].any() or m[-2:].any() or m[:, :2].any() or m[:, -2:].any())
        if 0.05 <= frac <= 0.60 and not border:
            ok.append(k)
    base = "mask covering 5-60 % of the image and clear of the border"
    if ok and readings is not None:
        got = [k for k in ok if readings[k]["read"] == want]
        if got:
            return got[0], f"first seed with a {base} whose facing reads {want}"
        got = [k for k in ok if readings[k]["read"] is not None]
        if got:
            return got[0], f"no seed reads {want}; first seed with a {base} whose facing could be read"
        return ok[0], f"WARNING no seed's facing could be read; first seed with a {base} (look at the overlay, pass --facing)"
    if ok:
        return ok[0], f"first seed with a {base}"
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
    ap.add_argument("--facing", default=None, choices=["left", "right"], help="the chosen candidate's facing, read by eye; wins over the detector's reading (default: the reading; the template's image-right if unread)")
    ap.add_argument("--head-query", default=None, help="the detector's text for the facing (default: 'the head of <object>'; e.g. 'the front of a car')")
    ap.add_argument("--steps", type=int, default=28); ap.add_argument("--guidance", type=float, default=3.5); ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--env", default="colab_public", help="preset in configs/environment.yaml (the SAM 3D Objects interpreter)")
    ap.add_argument("--drive-root", default=os.environ.get("MT_ROOT"), help="the root that holds cache/ (MT_ROOT)")
    ap.add_argument("--skip-generate", action="store_true", help="use candidates already under probe/m2d/proxies/<category>/")
    ap.add_argument("--select-seed", type=int, default=None, help="use this candidate seed instead of the fixed rule's choice (after looking at the overlays)")
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
    seg = S.load_segmenter() if need else None
    if need:
        for p in need:
            img = cv2.imread(str(p))[..., ::-1].copy()
            m, info = S.segment(seg, img, args.category)
            cv2.imwrite(str(p.with_name(p.stem + "_mask.png")), (m.astype(np.uint8) * 255))
            ov = img.astype(np.float32); ov[m] = 0.5 * ov[m] + 0.5 * np.array([232, 200, 106]); cv2.imwrite(str(p.with_name(p.stem + "_overlay.png")), ov[..., ::-1].astype(np.uint8))
            print(f"[proxy] mask for {p.name}: {info}", flush=True)
    masks = [cv2.imread(str(p.with_name(p.stem + "_mask.png")), cv2.IMREAD_GRAYSCALE) > 127 for p in cands]

    # 4. facing, read from the head of every candidate (the thesis read it by eye once per category)
    seg = seg or S.load_segmenter()
    readings = []
    for p, m in zip(cands, masks):
        img = cv2.imread(str(p))[..., ::-1].copy()
        rd = read_facing(seg, img, m, args.object, args.head_query); readings.append(rd)
        print(f"[proxy] facing of {p.name}: {rd['read'] or 'UNREAD'} ({rd['why']})", flush=True)

    # 5. choose
    k, why = choose(masks, readings, TEMPLATE_FACING)
    if args.select_seed is not None:
        k = next(i for i, c in enumerate(cands) if c.stem == f"s{args.select_seed}"); why = f"seed {args.select_seed} chosen by eye (--select-seed)"
    read = readings[k]["read"]
    if args.facing:
        facing = args.facing
        src = "given by --facing" + ("" if read in (None, facing) else f"; DISAGREES with the detector's reading ({read})")
    elif read:
        facing, src = read, f"read by the head detector ({readings[k]['why']})"
    else:
        facing, src = TEMPLATE_FACING, f"WARNING assumed from the template (facing image-{TEMPLATE_FACING}); unread: {readings[k]['why']}"
    choice = {"category": args.category, "object": args.object, "kind": args.kind, "prompt": prompt, "selected_seed": int(cands[k].stem[1:]),
              "rule": why, "facing": facing, "facing_source": src, "facing_readings": {c.stem: r for c, r in zip(cands, readings)},
              "mask_fraction": [float(m.mean()) for m in masks], "time": time.strftime("%Y-%m-%d %H:%M")}
    (out / "choice.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    print(f"[proxy] chose s{choice['selected_seed']}: {why}; facing {facing} ({src}); overlay {out / f's{choice['selected_seed']}_overlay.png'}", flush=True)

    # 6. reconstruct and commit
    envs = yaml.safe_load((ROOT / "configs" / "environment.yaml").read_text(encoding="utf-8"))
    py = envs.get(args.env, {}).get("python", {})
    interp = str(py.get("sam3d_objects", sys.executable)) if args.kind == "rigid" else sys.executable
    cmd = [interp, "scripts/run_m2b2.py", "--r1", "--category", args.category, "--image", str(cands[k]), "--mask", str(cands[k].with_name(cands[k].stem + "_mask.png")),
           "--facing", facing, "--selected-k", str(choice["selected_seed"]), "--commit-cache"]
    print("[proxy] $", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "MT_ROOT": str(root), "PYTHONUNBUFFERED": "1"})
    assert r.returncode == 0, "the reconstructor stopped; read its log"

    # 7. the proxy key back into the registry
    cdir = root / "cache" / "m2_b2" / args.category
    keys = sorted((d for d in cdir.iterdir() if (d / "proxy.npz").exists()), key=lambda d: d.stat().st_mtime)
    assert keys, f"no proxy under {cdir}"
    key = keys[-1].name
    subjects.register(args.category, proxy_key=key)
    print(f"[proxy] {args.category}: proxy {key} in the cache and registered; run_all.py --auto --category {args.category} will use it", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
