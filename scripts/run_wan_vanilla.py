#!/usr/bin/env python
"""Vanilla Wan 2.1 I2V on any image and prompt, through the diffusers pipeline as shipped (its own scheduler and loop): the backbone baseline that
needs no Module 3 cache (stock control of 2026-10-13: the pipeline walks the man and drives the car where our loop did not). Same settings as the
sampler's runs of record (lock: steps, weights; configs/m3.yaml: guidance scale, negative prompt), so a vanilla video here is comparable to an
A0 of the sampler. Writes <out-dir>/<name>.mp4 and <name>.json (settings, prompt verbatim, continuity: first step vs median step). Phase 2.

    python scripts/run_wan_vanilla.py --image /content/drive/MyDrive/motion_transfer/cache/m2_d/000000000035.1.003/bear/anchor.png \\
        --prompt "A brown bear walking on all fours on a coastal cliff road" --out-dir /content/drive/MyDrive/motion_transfer/cache/m3/000000000035.1.003/bear --name vanilla_seed0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_m3_s2_readout import REPO, WAN_H, WAN_W                         # noqa: E402
from src.generate.m3 import handoff as H                                          # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--prompt", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--steps", type=int, default=None, help="default: the lock's step count")
    ap.add_argument("--weights", default=None, choices=["fp8", "bf16"], help="default: the lock's weights")
    ap.add_argument("--negative-prompt", default=None, help="default: configs/m3.yaml sampling.negative_prompt (Wan's default); '' for none")
    ap.add_argument("--cfg", type=float, default=None, help="default: configs/m3.yaml sampling.guidance_scale")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import cv2
    import torch
    import yaml
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel
    lock = H.load_m3_lock()
    m3cfg = yaml.safe_load(Path("configs/m3.yaml").read_text(encoding="utf-8"))["m3"]; samp = m3cfg.get("sampling", {})
    steps = args.steps or int(lock["scheduler"]["steps"]); weights = args.weights or lock.get("weights", "fp8")
    cfg_scale = float(args.cfg if args.cfg is not None else samp.get("guidance_scale", 5.0))
    neg = str(samp.get("negative_prompt", "")) if args.negative_prompt is None else args.negative_prompt
    assert (args.frames - 1) % 4 == 0, "frames must be 4k + 1"
    dev, dt = "cuda", torch.bfloat16
    bgr = cv2.imread(args.image); assert bgr is not None, args.image
    img = Image.fromarray(cv2.resize(bgr[..., ::-1], (WAN_W, WAN_H), interpolation=cv2.INTER_AREA))
    tr = WanTransformer3DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=dt).requires_grad_(False).eval()
    if weights == "fp8":
        from torchao.quantization import Float8WeightOnlyConfig, quantize_
        quantize_(tr, Float8WeightOnlyConfig())
    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(REPO, transformer=tr, vae=vae, torch_dtype=dt).to(dev)
    print(f"[vanilla] scheduler {type(pipe.scheduler).__name__}, {steps} steps, cfg {cfg_scale}, {args.frames} frames, weights {weights}, seed {args.seed}", flush=True)
    print(f"[vanilla] prompt: {args.prompt!r}\n[vanilla] negative prompt ({len(neg)} chars): {neg!r}", flush=True)
    t0 = time.time()
    with torch.no_grad():
        frames = pipe(image=img, prompt=args.prompt, negative_prompt=neg, height=WAN_H, width=WAN_W, num_frames=args.frames, num_inference_steps=steps,
                      guidance_scale=cfg_scale, generator=torch.Generator(dev).manual_seed(args.seed), output_type="np").frames[0]
    u8 = (np.clip(frames, 0, 1) * 255.0).round().astype(np.uint8)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    name = args.name or f"vanilla_seed{args.seed}_F{args.frames}" + ("" if weights == "fp8" else f"_{weights}")
    vw = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 16, (WAN_W, WAN_H))
    for f_ in u8:
        vw.write(f_[..., ::-1])
    vw.release()
    fd = [float(np.abs(u8[i + 1].astype(np.float32) - u8[i].astype(np.float32)).mean()) for i in range(len(u8) - 1)]
    rec = {"image": args.image, "prompt": args.prompt, "negative_prompt": neg, "seed": args.seed, "frames": args.frames, "steps": steps, "cfg": cfg_scale, "weights": weights,
           "scheduler": type(pipe.scheduler).__name__, "pipeline": "diffusers WanImageToVideoPipeline.__call__ (stock)", "wall_s": time.time() - t0,
           "continuity": {"first_step": fd[0], "median_step": float(np.median(fd[1:])), "first_over_median": float(fd[0] / max(np.median(fd[1:]), 1e-6))},
           "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"[vanilla] -> {out / name}.mp4 in {rec['wall_s']:.0f} s; continuity first step {fd[0]:.1f} vs median {rec['continuity']['median_step']:.1f}; peak {rec['peak_gb']:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
