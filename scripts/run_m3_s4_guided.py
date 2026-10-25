#!/usr/bin/env python
"""Module 3 S4: the guided sampler (brief v2.6 §4, §5, §7), first pass: anchor set {0} (no refresh, R = []), object mode `centroid`,
mass term off unless the lock says otherwise, every loss on the windowed readout at the locked block. One simultaneous RMS-normalised
latent update per guided step inside T_g, frame-0 slice untouched. Weights never change.

Arms:  A2 = attention guidance (bg + obj + perp);  A0 = the same sampling with guidance off (identical noise: the paired baseline);
       A3 = the RENDER TARGET (reviewer 2026-10-10, v2.7 pending): inside T_g the x_hat_0 is corrected toward the encoded render where the
            mask holds, z' = z + (1 - sigma) gamma m (z^r - x_hat_0), no backward pass; the attention readout is logged as the in-loop monitor.
            Needs cache/m3/<clip>/<cat>/render{tag}.npz from scripts/render_m3_target.py; gamma from lock guidance.gamma_render (John sets it
            from --gamma-response).
       --bridge {rec, mid, far} picks the geometry cache (beta_tag). --frames 33 reads the first T_l latent frames of the cache.
       --smoke: 8 steps, 33 frames, nothing recorded (checks the code path and the per-step log in ~2 min).
Per guided step the log carries L_bg, L_obj, L_perp, (L_mass), mass at target, share of queries whose window contains the target, the
readout's EPE to F_geo on covered background, ||g|| RMS, mean C_geo, wall time, peak memory (§7). Outputs: cache/m3/<clip>/<cat>/<arm>_seed<k>_<tag>.mp4
+ .npz (latents, log, configuration); rows in reports/m3_s4/guided_runs.json (rule 24). Grade with scripts/measure_m3_a0.py --videos.
Phase 2 (bootstrap_phase2.sh), A100 40 GB.

    python scripts/run_m3_s4_guided.py --pair 000000000265.2.005/vehicle --seed 0 --frames 33 --smoke
    python scripts/run_m3_s4_guided.py --pair 000000000265.2.005/vehicle --seed 0 --frames 33 --arm A0
    python scripts/run_m3_s4_guided.py --pair 000000000265.2.005/vehicle --seed 0 --frames 33 --arm A2
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
from src.generate.m3 import guidance as GD, handoff as H, readout as RO          # noqa: E402
from src.generate.m3.geometry import epipolar as EP, project as PJ              # noqa: E402


def build_targets(geo: dict, T_run: int, Hp: int, Wp: int, tau_rot_only: float) -> dict:
    """Per latent frame t > 0 against anchor 0: background flow target, coverage, occupancy, object centroid target, epipolar lines."""
    E_l, Kp, O, F0, C0, c_tar, sky = geo["E_l"][:T_run], geo["Kp"], geo["O"][:T_run], geo["Fgeo0"][:T_run], geo["Cgeo0"][:T_run], geo["c_tar"][:T_run], geo["sky_lattice"]
    pc, _ = PJ.project(c_tar[:, None, :], E_l, Kp)                                 # [T,1,2] the object centroid on the lattice per frame
    N = Hp * Wp
    out = {"N": N, "T": T_run}
    cen = EP.patch_centres(Hp, Wp)
    for t in range(1, T_run):
        obj_tgt = pc[0, 0] - pc[t, 0]                                               # backward: where the centroid was at frame 0, from frame t
        T_bg = EP.relative_pose(E_l[0], E_l[t])
        G0 = np.eye(4); G0[:3, 3] = c_tar[0]; Gt = np.eye(4); Gt[:3, 3] = c_tar[t]  # object pose: translation only (R_t = I, centroid mode)
        T_obj = EP.relative_pose(E_l[0], E_l[t], G0, Gt)
        mode_bg, lines_bg, H_bg = EP.lines_and_points(T_bg, Kp, Hp, Wp, tau_rot_only)
        mode_ob, lines_ob, H_ob = EP.lines_and_points(T_obj, Kp, Hp, Wp, tau_rot_only)
        is_obj = O[t] > 0.5
        lines = np.where(is_obj[..., None], lines_ob, lines_bg); H_pts = np.where(is_obj[..., None], H_ob, H_bg)
        line_mode = np.where(is_obj, mode_ob == "line", mode_bg == "line") & ~sky  # sky patches always take the homography branch
        H_hom = np.asarray([[(EP.homography(T_bg, Kp) @ cen[i, j])[:2] / (EP.homography(T_bg, Kp) @ cen[i, j])[2] for j in range(Wp)] for i in range(Hp)])
        H_pts = np.where(np.isfinite(H_pts), H_pts, H_hom)                           # homography point everywhere a line is not used
        out[t] = {"F_geo": F0[t].reshape(N, 2), "C": C0[t].reshape(N), "O": O[t].reshape(N), "sky": sky.reshape(N),
                  "obj_target": obj_tgt, "lines": lines.reshape(N, 3), "H_pts": H_pts.reshape(N, 2), "line_mode": line_mode.reshape(N),
                  "epi_mode_bg": mode_bg, "epi_mode_obj": mode_ob}
    return out


TEXT_LEN = 512                                                                      # WanImageToVideoPipeline.__call__ encodes at max_sequence_length=512


def match_text_embeds(pipe, texts, embeds, max_len=TEXT_LEN):
    """The text goes through the pipeline's own encode_prompt at max_sequence_length=512 (its __call__'s value; the method's default is 226,
    which this loop used before 2026-10-14 — T5 at another padding length and 226 tokens at the transformer). This helper only asserts the
    length the pipeline feeds; it pads nothing (reviewer 2026-10-14: the fix belongs at the source)."""
    for e, t in zip(embeds, texts):
        assert e.shape[1] == max_len, f"text embedding for {t[:40]!r} has {e.shape[1]} tokens, the pipeline feeds {max_len}: encode with max_sequence_length={max_len}"
    return list(embeds), max_len


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--pair", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=None, help="4k+1 frames to generate; default = the pair's full window")
    ap.add_argument("--bridge", default="rec", choices=["rec", "mid", "far"])
    ap.add_argument("--arm", default="A2", choices=["A0", "A2", "A3"])
    ap.add_argument("--smoke", action="store_true", help="8 steps, 33 frames, nothing recorded")
    ap.add_argument("--steps", type=int, default=None, help="override the lock's step count (smoke only; a run of record uses the lock)")
    ap.add_argument("--checkpoint-moge", default=None, help="unused here; kept for symmetry with S3")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--object", action="store_true", help="arm A3: use render{tag}_obj.npz (the object rendered along its path) and apply lock guidance.gamma_obj on the object's latent mask")
    ap.add_argument("--motion-clause", default=None, help="append this phrase to the frozen anchor prompt for the run (reviewer 2026-10-12: the text carries what kind of motion; "
                    "the clause is a Module 3 input, John's decision, recorded verbatim); output name carries _mc")
    ap.add_argument("--no-processor", action="store_true", help="control: leave the stock attention processor on block 20 (no injection processor; the attention readout is then unavailable); _noproc")
    ap.add_argument("--stock-scheduler", action="store_true", help="override the lock's scheduler.name with the pipeline's own UniPC multistep (as shipped); _unipc when it is not the lock's")
    ap.add_argument("--euler", action="store_true", help="override the lock's scheduler.name with FlowMatchEuler at the lock's shift (the sampler's scheduler before 2026-10-14); _euler when it is not the lock's")
    ap.add_argument("--corrector-off-control", action="store_true", help="arm A0 control: UniPC's corrector off at the steps A3 corrects (the lock's T_g window), with no correction; separates "
                    "the predictor-only steps from the correction itself and tests continuity; _coff")
    ap.add_argument("--keep-corrector", action="store_true", help="control: leave UniPC's corrector on at the corrected steps (it rebuilds the sample from the previous one, discarding the z-form correction); _corr")
    ap.add_argument("--demo-label", default=None, help="UNREGISTERED demonstration (reviewer 2026-10-13): outputs to <drive>/demo/<label>/ with the label json; nothing is added to "
                    "guided_runs.json; --anchor-override / --prompt-override apply to A3 too; the A3 render is render_demo.npz; beta = 1 (the geometry as is)")
    ap.add_argument("--stock", action="store_true", help="arm A0 only: generate with the diffusers pipeline's own __call__ and its own scheduler (UniPC as shipped), none of this script's "
                    "sampling loop; the control for 'is our configuration of Wan the problem'; output name carries _stock")
    ap.add_argument("--weights", default=None, choices=["fp8", "bf16"], help="override the lock's weights for a control run (bf16 = the 80 GB reference, as in S3); the output name carries the tag")
    ap.add_argument("--anchor-override", default=None, help="arm A0 only: use this image (any size; resized to the Wan frame) instead of the pair's anchor, e.g. the source clip's own first frame "
                    "for a backbone sanity check on a natural image; output name carries _anchor<stem>")
    ap.add_argument("--prompt-override", default=None, help="arm A0 only: use this prompt instead of the pair's frozen prompt (recorded verbatim)")
    ap.add_argument("--bg-window", nargs=2, type=float, default=None, metavar=("HI", "LO"), help="arm A3: the background correction window [LO, HI] in sigma, overriding the "
                    "lock's T_g_sigma for this run (reviewer 2026-10-13 check 3: [0.85, 0.75], the free late steps animate texture); output name carries _bw<LO>")
    ap.add_argument("--clause-cfg", type=float, default=None, help="composable text guidance: a separate weight W on the (anchor+clause) branch, v = v_neg + cfg (v_anchor - v_neg) + W (v_clause - v_neg); needs --motion-clause; _ccfg<W>")
    ap.add_argument("--image-cfg", type=float, default=None, help="image-condition guidance scale S: v = v_noimg + S (v_img - v_noimg) + cfg (v_txt - v_img); S = 1 is the pipeline; S < 1 weakens the first frame; _icfg<S>")
    ap.add_argument("--correct-from", type=float, default=None, help="arm A3: raise the correction window's upper bound to this sigma (reviewer 2026-10-12 round 2: 0.99; the lock's "
                    "T_g_sigma[0] = 0.85 was inherited from the attention readout); frame 0 untouched; output name carries _cf<sigma>")
    ap.add_argument("--nodyn-mask", action="store_true", help="arm A3: use the render made with --exclude-mask (dynamic texture out of the mask); output name carries _nodyn")
    ap.add_argument("--sky-mask", action="store_true", help="arm A3: use render{tag}[_obj]_sky.npz (the sky at infinity inside the correction mask); output name carries _sky")
    ap.add_argument("--obj-taper", type=int, default=None, help="arm A3 --object: taper gamma_obj linearly to 0 over the last N object-corrected steps of the object window; _tp<N>")
    ap.add_argument("--soft-mask", action="store_true", help="arm A3: use the render's soft latent mask (fraction of each chunk / block covered) for the init and the correction "
                    "instead of the hard 4-frame min-pool; output name carries _sm")
    ap.add_argument("--obj-window", nargs=2, type=float, default=None, metavar=("HI", "LO"),
                    help="arm A3 --object: apply the object correction only for sigma in [LO, HI] (the background keeps the lock's T_g window); reviewer 2026-10-11 round 4: "
                    "[0.85, 0.75] and [0.85, 0.70]; output name carries _ow<LO>")
    ap.add_argument("--obj-lowpass", type=float, default=None, help="arm A3 --object: spatial Gaussian low-pass (sigma in latent px) of the object correction z^r - x_hat0 before the "
                    "object mask, so silhouette / position / scale are enforced and texture (wheel spokes, shading) is left to the model (reviewer 2026-10-11 decision 3); "
                    "output name carries _lp<s>")
    ap.add_argument("--gamma", type=float, default=None, help="arm A3: override the lock's guidance.gamma_render for this run (the gamma ablation, registered in docs/ablation_guidance_sheet_2026-10-24.md; gamma_obj stays at the lock); output name carries _g<value>")
    ap.add_argument("--ablation-label", default=None, help="label of a registered ablation setting (docs/ablation_guidance_sheet_2026-10-24.md); part of the run record's key, so an ablation run never collides with a run of record of the same flags")
    ap.add_argument("--gamma-obj-response", nargs="*", type=float, default=None, help="smoke only, arm A3 --object: at the first guided step apply the background correction at gamma_render and "
                    "the object correction at each gamma_obj given (default 1.0 0.5 0.2 0.1 0.05); report the object-mask residual after; the basis for the lock's gamma_obj")
    ap.add_argument("--grad-breakdown", action="store_true", help="smoke only: at the first guided step, the share of gradient energy at Q/K from each term (basis for lambda_mass)")
    ap.add_argument("--step-response", nargs="*", type=float, default=None, help="smoke only: at the first guided step, apply the update at each eta given (default 0.5 0.2 0.1 0.05 0.02), "
                    "re-run the readout and report EPE / contains-share AFTER the update; the basis for the lock's eta")
    ap.add_argument("--gamma-response", nargs="*", type=float, default=None, help="smoke only, arm A3: at the first guided step, apply the render correction at each gamma given "
                    "(default 1.0 0.5 0.2 0.1 0.05), re-run the readout monitor and report EPE / masked residual AFTER; the basis for the lock's gamma_render")
    args = ap.parse_args(argv)
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import cv2
    import torch
    import torch.nn.functional as Fnn
    from PIL import Image
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanImageToVideoPipeline
    from src.generate.ditflow_wan.attention import WanInjectionProcessor
    from src.generate.ditflow_wan.transformer import ControlledWanTransformer
    lock = H.load_m3_lock(); lat = lock["lattice"]; Hp, Wp, px = int(lat["Hp"]), int(lat["Wp"]), int(lat["px_per_patch"])
    for k in ("attention", "readout", "guidance", "refresh"):
        assert k in lock, f"lock m3.{k} missing (v2.6 §8; John adds it)"
    gcfg, rcfg = lock["guidance"], lock["readout"]
    assert rcfg["form"] == "windowed", rcfg
    block, r_w = int(lock["attention"]["block"]), float(rcfg["window_radius_patches"])
    assert lock["attention"].get("head_mode", "mean") == "mean"
    T_g_hi, T_g_lo = float(gcfg["T_g_sigma"][0]), float(gcfg["T_g_sigma"][1])
    lam_bg, lam_obj, lam_perp, eta = float(gcfg["lambda_bg"]), float(gcfg["lambda_obj"]), float(gcfg["lambda_perp"]), float(gcfg["eta"])
    lam_mass = gcfg.get("lambda_mass"); lam_mass = None if lam_mass is None else float(lam_mass)
    assert gcfg.get("obj_mode", "centroid") == "centroid", "first pass runs centroid mode (rotation.source is null)"
    tau_rot = float(gcfg.get("tau_rot_only", 0.05))
    assert len(lock["refresh"].get("R_sigma") or []) == 0, "refresh (S5) is not built: refresh.R_sigma must be [] for this sampler"
    steps_lock, shift = int(lock["scheduler"]["steps"]), float(lock["scheduler"]["shift"])
    sched_lock = str(lock["scheduler"]["name"]); assert sched_lock in ("FlowMatchEulerDiscrete", "UniPCMultistep"), sched_lock
    corrector_policy = str(lock["scheduler"].get("corrector_off", "correction_steps")); corrector_policy_source = "lock" if "corrector_off" in lock["scheduler"] else "code default (lock key absent)"
    assert corrector_policy in ("correction_steps", "none"), corrector_policy
    weights, ckpt = (args.weights or lock.get("weights", "fp8")), bool(lock.get("checkpointing", True))
    m3cfg = __import__("yaml").safe_load(Path("configs/m3.yaml").read_text(encoding="utf-8"))["m3"]
    samp = m3cfg.get("sampling", {}); cfg_scale = float(samp.get("guidance_scale", 5.0)); neg = str(samp.get("negative_prompt", ""))
    steps = (args.steps or 8) if args.smoke else steps_lock
    assert args.smoke or args.steps is None, "--steps is for --smoke only; a run of record uses the lock's step count"
    frames = 33 if args.smoke and args.frames is None else args.frames
    dev, dt = "cuda", torch.bfloat16
    root = Path(args.drive_root); clip, cat = args.pair.split("/")
    d = root / "cache" / "m3" / clip / cat
    tag = "" if args.bridge == "rec" else f"_beta_{args.bridge}"
    with np.load(d / f"geometry{tag}.npz") as z:
        geo = {k: z[k] for k in z.files}
    if "bridge_run" not in geo:
        assert args.arm == "A0" or args.demo_label, f"{args.pair}: geometry.npz carries no bridge (undetermined under its camera): only A0 (no guidance) or a labelled demo runs on it"
        print(f"[S4] {args.pair}: geometry without a bridge — A0 on the anchor alone; recorded bridge none", flush=True)
    else:
        assert args.demo_label or str(geo["bridge_run"]) == args.bridge, (geo["bridge_run"], args.bridge)
    T_cache = int(geo["T_l"]); F_pair = int(geo["F_pair"])
    if frames is None:
        frames = F_pair
    assert (frames - 1) % 4 == 0 and frames <= F_pair, (frames, F_pair)
    T_run = 1 + (frames - 1) // 4
    anchors = [int(a) for a in geo["anchors"] if int(a) < T_run]
    assert anchors == [0] or len(lock["refresh"]["R_sigma"]) == 0, "anchors beyond 0 need the refresh (S5)"
    tg = build_targets(geo, T_run, Hp, Wp, tau_rot)
    N = Hp * Wp
    render = None
    if args.arm == "A3":
        rp = d / (f"render{tag}_demo.npz" if args.demo_label else f"render{tag}{'_obj' if args.object else ''}{'_sky' if args.sky_mask else ''}{'_nodyn' if args.nodyn_mask else ''}.npz")
        assert rp.exists(), f"{rp} missing: run scripts/render_m3_target.py --pairs {args.pair} --bridge {args.bridge}{' --object' if args.object else ''} first (Phase 1)"
        with np.load(rp) as z:
            render = {"frames": z["frames"][:frames], "mask_latent": z["mask_latent"][:T_run].astype(bool), "beta_tag": str(z["beta_tag"]),
                      "mask_obj_latent": z["mask_obj_latent"][:T_run].astype(bool) if args.object else None}
            if args.soft_mask:
                assert "mask_latent_soft" in z.files, f"{rp} has no soft latent mask: re-render with the current scripts/render_m3_target.py"
                render["mask_latent"] = z["mask_latent_soft"][:T_run].astype(np.float32)
                if args.object:
                    render["mask_obj_latent"] = z["mask_obj_latent_soft"][:T_run].astype(np.float32)
        assert (args.demo_label or render["beta_tag"] == args.bridge) and render["frames"].shape[0] == frames, (render["beta_tag"], render["frames"].shape)
        gamma = gcfg.get("gamma_render")
        gamma = None if gamma is None else float(gamma)
        assert gamma is not None or (args.smoke and args.gamma_response is not None), "lock guidance.gamma_render missing (John sets it from --gamma-response)"
        if args.gamma is not None:
            assert args.arm == "A3" and not args.smoke and 0.0 < args.gamma <= 1.0, "--gamma is an A3 ablation override in (0, 1]"
            print(f"[S4 A3] gamma overridden: lock {gamma} -> {args.gamma} (ablation run)", flush=True); gamma = float(args.gamma)
        if args.obj_window is None and gcfg.get("obj_window_sigma") is not None and args.object:
            args.obj_window = [float(v) for v in gcfg["obj_window_sigma"]]
            print(f"[S4 A3 object] object window from the lock: {sorted(args.obj_window, reverse=True)}", flush=True)
        gamma_obj = None
        if args.object:
            gamma_obj = gcfg.get("gamma_obj"); gamma_obj = None if gamma_obj is None else float(gamma_obj)
            assert gamma_obj is not None or (args.smoke and args.gamma_obj_response is not None), "lock guidance.gamma_obj missing (John sets it from --gamma-obj-response)"
            assert gamma is not None, "the object run needs gamma_render in the lock"
    prompt_anchor = json.loads((root / "cache" / "m2_d" / clip / cat / "qc_m2d.json").read_text(encoding="utf-8"))["prompt"]
    if args.prompt_override is not None:
        assert args.arm == "A0" or args.demo_label, "--prompt-override is a backbone control (arm A0) or part of a labelled demo"
        prompt_anchor = args.prompt_override
    prompt = (prompt_anchor.rstrip().rstrip(".") + ", " + args.motion_clause.strip()) if args.motion_clause else prompt_anchor
    print(f"[S4] prompt (anchor, frozen): {prompt_anchor!r}", flush=True)
    print(f"[S4] negative prompt ({'empty' if not neg else str(len(neg)) + ' chars'}): {neg!r}", flush=True)
    if args.motion_clause:
        print(f"[S4] prompt with the motion clause: {prompt!r}", flush=True)
    anchor_path = Path(args.anchor_override) if args.anchor_override else root / "cache" / "m2_d" / clip / cat / "anchor.png"
    if args.anchor_override:
        assert args.arm == "A0" or args.demo_label, "--anchor-override is a backbone control (arm A0) or part of a labelled demo"
    anchor = cv2.imread(str(anchor_path)); assert anchor is not None, anchor_path
    anchor = anchor[..., ::-1]
    anchor_wan = cv2.resize(anchor, (WAN_W, WAN_H), interpolation=cv2.INTER_AREA)

    # ---- models
    tr = ControlledWanTransformer.from_pretrained(REPO, subfolder="transformer", torch_dtype=dt).requires_grad_(False).eval()
    if weights == "fp8":
        from torchao.quantization import Float8WeightOnlyConfig, quantize_
        quantize_(tr, Float8WeightOnlyConfig())
    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(REPO, transformer=tr, vae=vae, torch_dtype=dt)
    stock_sched = pipe.scheduler                                                    # the scheduler the pipeline ships with (UniPC multistep), kept for --stock
    assert not (args.euler and args.stock_scheduler), "--euler and --stock-scheduler contradict"
    # the lock's scheduler.name decides (parity 2026-10-14: the pipeline's UniPC inside this loop lands on the pipeline's figure, Euler on a
    # twisted one; the lock is the human's to edit); --euler / --stock-scheduler override it and the output name says so
    use_euler = args.euler or (sched_lock == "FlowMatchEulerDiscrete" and not args.stock_scheduler)
    if use_euler:
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(stock_sched.config, shift=shift)
    else:
        assert abs(float(stock_sched.config.flow_shift) - shift) < 1e-6, (stock_sched.config.flow_shift, shift)
        pipe.scheduler = stock_sched                                                  # UniPC multistep as the pipeline ships it, flow shift = the lock's
    scheduler_name = type(pipe.scheduler).__name__
    sched_tag = "" if (use_euler == (sched_lock == "FlowMatchEulerDiscrete")) else ("_euler" if use_euler else "_unipc")
    print(f"[S4] scheduler {scheduler_name} (lock: {sched_lock}{', OVERRIDDEN' if sched_tag else ''}); shift {shift}; steps {steps}", flush=True)
    pipe.to(dev)
    if args.stock:
        assert args.arm == "A0", "--stock is the vanilla-pipeline control (arm A0)"
        pipe.scheduler = stock_sched
        print(f"[S4 stock] diffusers WanImageToVideoPipeline.__call__: scheduler {type(stock_sched).__name__} (config shift {getattr(stock_sched.config, 'flow_shift', None)}), "
              f"{steps} steps, cfg {cfg_scale}, {frames} frames, weights {weights}; prompt {prompt!r}", flush=True)
        t_run = time.time()
        with torch.no_grad():
            frames_out = pipe(image=Image.fromarray(anchor_wan), prompt=prompt, negative_prompt=neg, height=WAN_H, width=WAN_W, num_frames=frames, num_inference_steps=steps,
                              guidance_scale=cfg_scale, generator=torch.Generator(dev).manual_seed(args.seed), output_type="np").frames[0]
        u8 = (np.clip(frames_out, 0, 1) * 255.0).round().astype(np.uint8)
        out_dir = d / ("smoke" if args.smoke else ""); out_dir.mkdir(parents=True, exist_ok=True)
        name = f"A0_seed{args.seed}_{args.bridge}_F{frames}" + ("" if weights == "fp8" else f"_{weights}") + (f"_anchor{Path(args.anchor_override).stem[:12]}" if args.anchor_override else "") + ("_noproc" if args.no_processor else "") + "_stock"
        vw = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 16, (WAN_W, WAN_H))
        for f_ in u8:
            vw.write(f_[..., ::-1])
        vw.release()
        fd_ = [float(np.abs(u8[i + 1].astype(np.float32) - u8[i].astype(np.float32)).mean()) for i in range(len(u8) - 1)]
        print(f"[S4 stock] -> {out_dir / name}.mp4 in {time.time() - t_run:.0f} s; continuity first step {fd_[0]:.1f} vs median {np.median(fd_[1:]):.1f}", flush=True)
        return 0
    if ckpt:
        tr.gradient_checkpointing = True
        tr.enable_gradient_checkpointing()
    if args.no_processor:
        assert args.arm in ("A0", "A3"), "--no-processor: the attention readout monitor is unavailable (A2 needs the processor)"
        class _NoProc:                                                             # a stand-in with the flags the loop sets; the stock processor stays installed
            copy_kv = False
            def clear(self): pass
        proc = _NoProc()
    else:
        proc = WanInjectionProcessor(f"blocks.{block - 1}.attn1"); tr.blocks[block - 1].attn1.set_processor(proc)
    lm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(dev)
    ls = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(dev)
    c_t = torch.from_numpy(EP.patch_centres(Hp, Wp)[..., :2].reshape(N, 2)).float().to(dev)
    tt = {t: {k: (torch.from_numpy(np.asarray(v)).to(dev) if isinstance(v, np.ndarray) else v) for k, v in tg[t].items()} for t in range(1, T_run)}
    for t in tt:
        for k in ("F_geo", "C", "O", "lines", "H_pts"):
            tt[t][k] = tt[t][k].float()
        tt[t]["obj_target"] = torch.as_tensor(tt[t]["obj_target"], dtype=torch.float32, device=dev)
        tt[t]["w_bg"] = (tt[t]["C"] * (1.0 - tt[t]["O"]) * (~tt[t]["sky"]).float())

    with torch.no_grad():
        pe, ne = pipe.encode_prompt(prompt, negative_prompt=neg, do_classifier_free_guidance=True, device=dev, max_sequence_length=TEXT_LEN)   # the pipeline's own path (its __call__ passes 512)
        (pe, ne), L_txt = match_text_embeds(pipe, [prompt, neg], [pe, ne])         # asserts the length; pads nothing (reviewer 2026-10-14)
        print(f"[S4] text embeddings padded to {L_txt} tokens", flush=True)
        pe_anchor = None
        if args.clause_cfg is not None:
            assert args.motion_clause, "--clause-cfg needs --motion-clause"
            pe_anchor, _ = pipe.encode_prompt(prompt_anchor, negative_prompt=neg, do_classifier_free_guidance=True, device=dev, max_sequence_length=TEXT_LEN)   # the anchor prompt alone, for the composable branch
            (pe_anchor,), _ = match_text_embeds(pipe, [prompt_anchor], [pe_anchor])
        img = Image.fromarray(anchor_wan); ie = pipe.encode_image(img, dev)
        img_t = pipe.video_processor.preprocess(img, height=WAN_H, width=WAN_W).to(dev, dtype=torch.float32)
        zT, cond = pipe.prepare_latents(img_t, 1, vae.config.z_dim, WAN_H, WAN_W, frames, torch.float32, dev, torch.Generator(dev).manual_seed(args.seed))   # float32, as the pipeline does
    pipe.text_encoder.to("cpu"); pipe.image_encoder.to("cpu"); torch.cuda.empty_cache()
    assert zT.shape[2] == T_run
    z_r = m_lat = None
    if render is not None:
        with torch.no_grad():
            vr = torch.from_numpy(render["frames"]).permute(3, 0, 1, 2)[None].float().to(dev) / 127.5 - 1.0
            z_r = ((vae.encode(vr).latent_dist.mode() - lm) / ls).float()                 # [1,C,T,h,w] normalised, the encoded render
        m_lat = torch.from_numpy(render["mask_latent"]).float().to(dev)[None, None]       # [1,1,T,h,w]
        m_lat[:, :, 0] = 0.0                                                               # frame 0 is the anchor itself: never corrected
        m_obj = None
        if args.object:
            m_obj = torch.from_numpy(render["mask_obj_latent"]).float().to(dev)[None, None]; m_obj[:, :, 0] = 0.0
            ov = (m_obj > 0) & (m_lat > 0)                                                 # cells where the two latent masks meet (soft masks at the object's edge)
            overlap_cells = int(ov.sum()); overlap_mass = float((m_obj * m_lat).sum())
            if overlap_cells:
                m_lat = m_lat * (~(m_obj > 0)).float()                                     # the pixel rule of the render (a pixel belongs to one mask; the object wins) applied at the latent level
                print(f"[S4 A3 object] {overlap_cells} latent cells carried both masks (mass {overlap_mass:.3f}); background zeroed there (the render's pixel rule); recorded", flush=True)
            assert float((m_obj * m_lat).sum()) == 0.0, "object and background latent masks overlap"
            rec_overlap = {"cells": overlap_cells, "mass": overlap_mass, "rule": "background zeroed where the object mask is nonzero (render pixel rule); 0 on the six pairs of record"}
            print(f"[S4 A3 object] object latent mask share {float(m_obj.mean()):.3f} (last frame {float(m_obj[:, :, -1].mean()):.3f})", flush=True)
        assert z_r.shape[2] == T_run and m_lat.shape[2:] == z_r.shape[2:], (z_r.shape, m_lat.shape)
        print(f"[S4 A3] render encoded: latent mask share {float(m_lat.mean()):.3f} (last frame {float(m_lat[:, :, -1].mean()):.3f})", flush=True)
        del vr; torch.cuda.empty_cache()

    def lowpass(x):
        """Separable Gaussian blur over the two spatial latent axes of x [1,C,T,h,w], sigma = args.obj_lowpass latent px, reflect padding."""
        sg = float(args.obj_lowpass); rad = int(np.ceil(3 * sg)); k1 = torch.exp(-0.5 * (torch.arange(-rad, rad + 1, device=x.device, dtype=torch.float32) / sg) ** 2); k1 = k1 / k1.sum()
        B, C, T, h, w = x.shape
        y = x.float().reshape(B * C * T, 1, h, w)
        y = Fnn.conv2d(Fnn.pad(y, (rad, rad, 0, 0), mode="reflect"), k1.view(1, 1, 1, -1))
        y = Fnn.conv2d(Fnn.pad(y, (0, 0, rad, rad), mode="reflect"), k1.view(1, 1, -1, 1))
        return y.reshape(B, C, T, h, w)

    def readout_eval(z_eval, ts):
        """No-grad forward to the hooked block; the readout's EPE to the background target (weighted) and the contains share, over pairs (0, t)."""
        tr.stop_after_block = block - 1; proc.copy_kv = True
        with torch.no_grad():
            tr(torch.cat([z_eval.to(dt), cond.to(dt)], 1), ts, pe.to(dt), encoder_hidden_states_image=ie.to(dt), return_dict=False)
            Qe, Ke = proc.query, proc.key; K0e = Ke[0, :N]
            epes, cont, mass = [], [], []
            for t in range(1, T_run):
                Ae = RO.attention_probs(Qe[0, t * N:(t + 1) * N], K0e)
                Fe = RO.windowed_expected_displacement(Ae, c_t, r_w); g_ = tt[t]; wb = g_["w_bg"]
                epes.append(float((wb * torch.linalg.norm(Fe - g_["F_geo"], dim=-1)).sum() / (wb.sum() + 1e-6)))
                m_ = GD.mass_terms(Ae, c_t, g_["F_geo"], r_w, wb); cont.append(m_["window_contains_target_share"]); mass.append(m_["mass_at_target_mean"])
        proc.clear(); proc.copy_kv = False; tr.stop_after_block = None
        return {"epe_bg_mean": float(np.mean(epes)), "epe_bg_last": epes[-1], "contains": float(np.mean(cont)), "mass": float(np.mean(mass))}

    sch = pipe.scheduler; sch.set_timesteps(steps, device=dev)
    sig = sch.sigmas.float().cpu().numpy()[:steps]
    corrector_off = []
    z = zT.float(); log = []; t_run = time.time()                                       # the latent trajectory stays float32; only the model input is cast
    # reviewer 2026-10-12 round 2: "early start" = the correction window extended upward (the (1 - sigma) factor ramps it in), frame 0 untouched
    T_g_hi_eff = float(args.correct_from) if args.correct_from is not None else T_g_hi
    if args.bg_window is not None:
        assert args.arm == "A3" and args.correct_from is None, "--bg-window sets both bounds; do not combine with --correct-from"
        T_g_hi_eff, T_g_lo = max(args.bg_window), min(args.bg_window)
        guided_steps = [k for k in range(steps) if T_g_lo - 1e-9 <= float(sig[k]) <= T_g_hi_eff + 1e-9]
        first_guided = guided_steps[0] if guided_steps else -1
        print(f"[S4 A3] background window overridden to [{T_g_hi_eff:.2f}, {T_g_lo:.2f}]: {len(guided_steps)} corrected steps", flush=True)
    if args.correct_from is not None:
        assert args.arm == "A3" and T_g_hi_eff >= T_g_hi, "--correct-from raises the A3 window's upper bound above the lock's T_g_sigma[0]"
        guided_steps = [k for k in range(steps) if T_g_lo - 1e-9 <= float(sig[k]) <= T_g_hi_eff + 1e-9]
        first_guided = guided_steps[0] if guided_steps else -1
        print(f"[S4 A3] correction window raised to [{T_g_hi_eff:.2f}, {T_g_lo:.2f}]: {len(guided_steps)} corrected steps (frame-0 slice untouched)", flush=True)
    if args.corrector_off_control:
        assert args.arm == "A0" and not use_euler, "--corrector-off-control is an A0 control on the UniPC scheduler"
    if ((args.arm == "A3" and corrector_policy == "correction_steps" and not args.keep_corrector) or args.corrector_off_control) and not use_euler:
        # UniPC's step(k) first runs its corrector, which REBUILDS the sample from the sample of step k-1 and the model outputs: the sample passed in
        # survives only through its model output. A z-form correction applied at step k and passed into step(k) would be discarded there, so the
        # corrector is switched off at the corrected steps (the scheduler's own list; index k-1 names step k). At step k+1 the corrector starts
        # from the corrected sample, so the correction carries forward. Self-test: tests/test_m3_unipc_corrector.py.
        corrector_off = [k for k in range(steps) if T_g_lo - 1e-9 <= float(sig[k]) <= T_g_hi_eff + 1e-9]
        sch.disable_corrector = sorted(set(list(sch.disable_corrector) + [k - 1 for k in corrector_off]))
        print(f"[S4 {args.arm}] {scheduler_name}: corrector off at the {len(corrector_off)} steps {corrector_off[0]}..{corrector_off[-1]} (0-based; sigma {float(sig[corrector_off[0]]):.3f}..{float(sig[corrector_off[-1]]):.3f}; "
              f"scheduler disable list {corrector_off[0] - 1}..{corrector_off[-1] - 1}); predictor-only there; policy {corrector_policy}", flush=True)
    for k, tstep in enumerate(sch.timesteps):
        s_k = float(sig[k]); ts = tstep.expand(1)
        tr.stop_after_block = None; proc.copy_kv = False
        with torch.no_grad():
            v_c = tr(torch.cat([z.to(dt), cond.to(dt)], 1), ts, pe.to(dt), encoder_hidden_states_image=ie.to(dt), return_dict=False)[0].float()
            v_u = tr(torch.cat([z.to(dt), cond.to(dt)], 1), ts, ne.to(dt), encoder_hidden_states_image=ie.to(dt), return_dict=False)[0].float()
            if pe_anchor is not None:                                                  # composable: anchor branch at cfg, clause branch at W
                v_a = tr(torch.cat([z.to(dt), cond.to(dt)], 1), ts, pe_anchor.to(dt), encoder_hidden_states_image=ie.to(dt), return_dict=False)[0].float()
                v = v_u + cfg_scale * (v_a - v_u) + float(args.clause_cfg) * (v_c - v_u)
            else:
                v = v_u + cfg_scale * (v_c - v_u)
            if args.image_cfg is not None:                                             # the image condition at its own scale S
                v_ni = tr(torch.cat([z.to(dt), torch.zeros_like(cond).to(dt)], 1), ts, ne.to(dt), encoder_hidden_states_image=torch.zeros_like(ie).to(dt), return_dict=False)[0].float()
                v = v_ni + float(args.image_cfg) * (v_u - v_ni) + (v - v_u)
        rec = {"step": k, "sigma": s_k, "guided": False}
        if args.arm == "A2" and T_g_lo - 1e-9 <= s_k <= T_g_hi + 1e-9:
            t0 = time.time(); torch.cuda.reset_peak_memory_stats()
            zg = z.detach().clone().to(dt).requires_grad_(True)
            tr.stop_after_block = block - 1; proc.copy_kv = True
            tr(torch.cat([zg, cond.to(dt)], 1), ts, pe.to(dt), encoder_hidden_states_image=ie.to(dt), return_dict=False)
            Q, K = proc.query, proc.key                                             # [1, T*N, heads, D] after RoPE, graph attached
            assert Q is not None and Q.shape[1] == T_run * N, (None if Q is None else Q.shape, T_run * N)
            K0 = K[0, :N]
            gQ = torch.zeros_like(Q); gK = torch.zeros_like(K)
            acc = {"L_bg": 0.0, "L_obj": 0.0, "L_perp": 0.0, "L_mass": 0.0, "epe_bg": [], "mass": [], "contains": [], "C_mean": []}
            terms_rms = {}
            for t in range(1, T_run):
                A = RO.attention_probs(Q[0, t * N:(t + 1) * N], K0)                  # [heads, N, N], float32, differentiable
                F_hat = RO.windowed_expected_displacement(A, c_t, r_w)
                g_ = tt[t]
                L_bg = GD.bg_loss(F_hat, g_["F_geo"], g_["w_bg"], torch.zeros_like(g_["O"]))   # w_bg = C_geo (1 - O_t) (not sky): §5.1 with the sky rule
                L_obj = GD.centroid_loss(F_hat, g_["obj_target"], g_["O"])
                L_perp = GD.perp_loss(F_hat, c_t, g_["lines"], g_["H_pts"], g_["line_mode"], g_["C"])
                L_t = lam_bg * L_bg + lam_obj * L_obj + lam_perp * L_perp
                mt = GD.mass_terms(A, c_t, g_["F_geo"], r_w, g_["w_bg"])
                if lam_mass is not None:
                    L_t = L_t + lam_mass * mt["L_mass"]
                if args.smoke and args.grad_breakdown and k == first_guided:          # per-term gradients FIRST (they need the pair's graph intact)
                    for nm, Lx in (("bg", lam_bg * L_bg), ("obj", lam_obj * L_obj), ("perp", lam_perp * L_perp), ("mass", mt["L_mass"])):
                        if Lx.requires_grad:
                            dq_, dk_ = torch.autograd.grad(Lx, [Q, K], retain_graph=True, allow_unused=True)
                            terms_rms[nm] = terms_rms.get(nm, 0.0) + float(sum((x.float() ** 2).sum() for x in (dq_, dk_) if x is not None))
                dq, dk = torch.autograd.grad(L_t, [Q, K], retain_graph=False, allow_unused=True)   # into the captured Q/K only; A freed per pair
                if dq is not None:
                    gQ += dq
                if dk is not None:
                    gK += dk
                with torch.no_grad():
                    wb = g_["w_bg"]; epe = (wb * torch.linalg.norm(F_hat.detach() - g_["F_geo"], dim=-1)).sum() / (wb.sum() + 1e-6)
                acc["L_bg"] += float(L_bg); acc["L_obj"] += float(L_obj); acc["L_perp"] += float(L_perp); acc["L_mass"] += float(mt["L_mass"])
                acc["epe_bg"].append(float(epe)); acc["mass"].append(mt["mass_at_target_mean"]); acc["contains"].append(mt["window_contains_target_share"]); acc["C_mean"].append(float(g_["C"].mean()))
                del A, F_hat, L_t
            torch.autograd.backward([Q, K], [gQ, gK])                                    # ONE backward through the transformer for all pairs
            g = zg.grad
            assert g is not None and torch.isfinite(g).all(), "no finite gradient reached the latent"
            if terms_rms:
                tot = sum(terms_rms.values()) or 1.0
                rec["grad_share_by_term_at_QK"] = {nm: v / tot for nm, v in terms_rms.items()}
                print(f"[S4 smoke] gradient energy at Q/K by term (first guided step): " + ", ".join(f"{nm} {v / tot:.3f}" for nm, v in terms_rms.items()), flush=True)
            g_rms = float(torch.sqrt((g.float() ** 2).mean()))
            if args.smoke and args.step_response is not None and k == first_guided:
                etas = args.step_response or [0.5, 0.2, 0.1, 0.05, 0.02]
                before = readout_eval(z, ts); resp = {"eta_0": before}
                print(f"[S4 step-response] before the update: EPE {before['epe_bg_mean']:.2f} (last {before['epe_bg_last']:.2f}), contains {before['contains']:.3f}, mass {before['mass']:.3f}", flush=True)
                for e_ in etas:
                    after = readout_eval(GD.guided_update(z, g, e_).detach(), ts); resp[f"eta_{e_}"] = after
                    print(f"[S4 step-response] eta {e_}: EPE {after['epe_bg_mean']:.2f} (last {after['epe_bg_last']:.2f}), contains {after['contains']:.3f}, mass {after['mass']:.3f}", flush=True)
                rec["step_response"] = resp
            z = GD.guided_update(z, g, eta).detach()
            proc.clear(); del Q, K, K0, zg, gQ, gK; torch.cuda.empty_cache()
            n_pairs = T_run - 1
            rec.update({"guided": True, "L_bg": acc["L_bg"] / n_pairs, "L_obj": acc["L_obj"] / n_pairs, "L_perp": acc["L_perp"] / n_pairs, "L_mass": acc["L_mass"] / n_pairs,
                        "epe_bg_patch_mean": float(np.mean(acc["epe_bg"])), "epe_bg_patch_last": acc["epe_bg"][-1], "mass_at_target_mean": float(np.mean(acc["mass"])),
                        "window_contains_target_share": float(np.mean(acc["contains"])), "C_geo_mean": float(np.mean(acc["C_mean"])), "grad_rms": g_rms,
                        "wall_s": time.time() - t0, "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30})
            print(f"[S4 {args.arm}] step {k + 1}/{steps} sigma {s_k:.3f}: L_bg {rec['L_bg']:.3f} L_obj {rec['L_obj']:.3f} L_perp {rec['L_perp']:.3f} | EPE bg {rec['epe_bg_patch_mean']:.2f} "
                  f"(last {rec['epe_bg_patch_last']:.2f}) | mass {rec['mass_at_target_mean']:.2f} contains {rec['window_contains_target_share']:.2f} | g_rms {g_rms:.2e} | {rec['wall_s']:.0f} s {rec['peak_gb']:.1f} GB", flush=True)
        elif args.arm == "A3" and T_g_lo - 1e-9 <= s_k <= T_g_hi_eff + 1e-9:
            t0 = time.time()
            xhat = z.float() - s_k * v.float()
            resid = m_lat * (z_r - xhat)
            in_obj_window = (args.obj_window is None) or (min(args.obj_window) - 1e-9 <= s_k <= max(args.obj_window) + 1e-9)
            taper = 1.0
            if args.obj_taper and m_obj is not None and in_obj_window:
                ow_steps = [k_ for k_ in range(steps) if (args.obj_window is None or min(args.obj_window) - 1e-9 <= float(sig[k_]) <= max(args.obj_window) + 1e-9)
                            and T_g_lo - 1e-9 <= float(sig[k_]) <= T_g_hi_eff + 1e-9]
                pos_from_end = len(ow_steps) - 1 - ow_steps.index(k) if k in ow_steps else None
                if pos_from_end is not None and pos_from_end < args.obj_taper:
                    taper = (pos_from_end + 1) / (args.obj_taper + 1)             # ... 3/4, 2/4, 1/4 over the last N steps (N = 3)
            resid_obj = (m_obj * (lowpass(z_r - xhat) if args.obj_lowpass else (z_r - xhat))) if (m_obj is not None and in_obj_window) else None
            def corrected(gm, gmo=None):
                out_ = z.float() + (1.0 - s_k) * gm * resid
                if resid_obj is not None:
                    gmo_ = gmo if gmo is not None else (gamma_obj if gamma_obj is not None else 0.0)   # a probe of the background alone leaves the object uncorrected
                    out_ = out_ + (1.0 - s_k) * gmo_ * taper * resid_obj
                return out_.float()
            def obj_resid_rms(zc):
                return float(torch.sqrt(((m_obj * (z_r - (zc.float() - s_k * v.float()))) ** 2).sum() / (m_obj.sum() * z_r.shape[1] + 1e-6)))
            mon_before = readout_eval(z, ts)
            if args.smoke and args.object and args.gamma_obj_response is not None and k == first_guided:
                ro_before = obj_resid_rms(z); resp_o = {"gamma_obj_0": ro_before}
                print(f"[S4 gamma-obj-response] before: object-mask residual {ro_before:.3f}", flush=True)
                for gmo in (args.gamma_obj_response or [1.0, 0.5, 0.2, 0.1, 0.05]):
                    zc = corrected(gamma, gmo); r_o = obj_resid_rms(zc); after = readout_eval(zc, ts)
                    resp_o[f"gamma_obj_{gmo}"] = {"object_resid_rms_after": r_o, "bg_monitor_epe_after": after["epe_bg_mean"]}
                    print(f"[S4 gamma-obj-response] gamma_obj {gmo}: object-mask residual {r_o:.3f} | bg monitor EPE {after['epe_bg_mean']:.2f}", flush=True)
                rec["gamma_obj_response"] = resp_o
            if args.smoke and args.gamma_response is not None and k == first_guided:
                resp = {"gamma_0": mon_before}
                print(f"[S4 gamma-response] before: EPE {mon_before['epe_bg_mean']:.2f} (last {mon_before['epe_bg_last']:.2f}), contains {mon_before['contains']:.3f}", flush=True)
                for gm in (args.gamma_response or [1.0, 0.5, 0.2, 0.1, 0.05]):
                    zc = corrected(gm); after = readout_eval(zc, ts)
                    r2 = float(torch.sqrt(((m_lat * (z_r - (zc.float() - s_k * v.float()))) ** 2).sum() / (m_lat.sum() * z_r.shape[1] + 1e-6)))
                    after["masked_resid_rms_after"] = r2; resp[f"gamma_{gm}"] = after
                    print(f"[S4 gamma-response] gamma {gm}: EPE {after['epe_bg_mean']:.2f} (last {after['epe_bg_last']:.2f}), contains {after['contains']:.3f}, masked residual {r2:.3f}", flush=True)
                rec["gamma_response"] = resp
            gm_use = gamma if gamma is not None else (args.gamma_response or [1.0])[0]
            gmo_use = (gamma_obj if gamma_obj is not None else (args.gamma_obj_response or [1.0])[0]) if args.object else None
            r_before = float(torch.sqrt((resid ** 2).sum() / (m_lat.sum() * z_r.shape[1] + 1e-6)))
            ro_before = obj_resid_rms(z) if m_obj is not None else None
            z = corrected(gm_use, gmo_use).detach()
            rec.update({"guided": True, "gamma": gm_use, "gamma_obj": gmo_use, "object_taper": taper, "object_corrected_this_step": bool(resid_obj is not None), "masked_resid_rms_before": r_before, "object_resid_rms_before": ro_before,
                        "monitor_epe_bg": mon_before["epe_bg_mean"], "monitor_contains": mon_before["contains"],
                        "monitor_mass": mon_before["mass"], "wall_s": time.time() - t0, "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30})
            print(f"[S4 A3] step {k + 1}/{steps} sigma {s_k:.3f}: masked residual {r_before:.3f} | monitor EPE {mon_before['epe_bg_mean']:.2f} contains {mon_before['contains']:.2f} | gamma {gm_use} | {rec['wall_s']:.0f} s", flush=True)
        else:
            if k % 10 == 0:
                print(f"[S4 {args.arm}] step {k + 1}/{steps} sigma {s_k:.3f}: unguided", flush=True)
        log.append(rec)
        tr.stop_after_block = None
        z = sch.step(v, tstep, z, return_dict=False)[0].float()
    wall = time.time() - t_run
    with torch.no_grad():
        video = vae.decode(z.float() * ls + lm).sample
    u8 = ((video[0].permute(1, 2, 3, 0).float().clamp(-1, 1) + 1.0) * 127.5).round().cpu().numpy().astype(np.uint8)
    out_dir = (root / "demo" / args.demo_label) if args.demo_label else d / ("smoke" if args.smoke else ""); out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.arm}_seed{args.seed}_{args.bridge}_F{frames}" + ("" if weights == "fp8" else f"_{weights}") + ("_obj" if args.object else "") + (f"_lp{args.obj_lowpass:g}" if args.obj_lowpass else "") + (f"_ow{min(args.obj_window):g}" if args.obj_window else "") + (f"_cf{args.correct_from:g}" if args.correct_from is not None else "") + sched_tag + ("_corr" if args.keep_corrector else "") + ("_coff" if args.corrector_off_control else "") + ("_sm" if args.soft_mask else "") + ("_sky" if args.sky_mask else "") + ("_nodyn" if args.nodyn_mask else "") + (f"_tp{args.obj_taper}" if args.obj_taper else "") + ("_mc" if args.motion_clause else "") + (f"_anchor{Path(args.anchor_override).stem[:12]}" if args.anchor_override else "") + ("_noproc" if args.no_processor else "") + (f"_bw{min(args.bg_window):g}" if args.bg_window else "") + (f"_ccfg{args.clause_cfg:g}" if args.clause_cfg is not None else "") + (f"_icfg{args.image_cfg:g}" if args.image_cfg is not None else "") + (f"_g{args.gamma:g}" if args.gamma is not None else "")
    vw = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 16, (WAN_W, WAN_H))
    for f_ in u8:
        vw.write(f_[..., ::-1])
    vw.release()
    config = {"pair": args.pair, "gamma_override": args.gamma, "ablation_label": args.ablation_label, "arm": args.arm, "seed": args.seed, "frames": frames, "T_run": T_run, "bridge": (args.bridge if "bridge_run" in geo else "none"), "beta": (float(geo["beta"]) if "beta" in geo else None), "s_delta": (float(geo["s_delta"]) if "s_delta" in geo else None),
              "object_render": bool(args.object), "gamma_obj": (gamma_obj if args.arm == "A3" and args.object else None), "obj_lowpass_sigma_latent": args.obj_lowpass, "obj_window_sigma": (sorted(args.obj_window, reverse=True) if args.obj_window else None), "correct_from_sigma": args.correct_from, "prompt_anchor": prompt_anchor, "motion_clause": args.motion_clause, "prompt_used": prompt, "anchor_override": args.anchor_override, "no_processor": bool(args.no_processor), "scheduler": scheduler_name, "scheduler_lock": sched_lock, "keep_corrector": bool(args.keep_corrector), "corrector_policy": corrector_policy, "corrector_policy_source": corrector_policy_source, "corrector_off_control": bool(args.corrector_off_control), "latent_dtype": "float32 (from 2026-10-13; earlier runs bf16)", "text_tokens": L_txt, "clause_cfg": args.clause_cfg, "image_cfg": args.image_cfg, "bg_window_override": (sorted(args.bg_window, reverse=True) if args.bg_window else None), "negative_prompt_used": neg, "soft_mask": bool(args.soft_mask), "sky_mask": bool(args.sky_mask), "nodyn_mask": bool(args.nodyn_mask), "obj_taper_steps": args.obj_taper, "T_g_effective": [T_g_hi_eff, T_g_lo],
              "block": block, "r_w": r_w, "T_g": [T_g_hi, T_g_lo], "lambdas": {"bg": lam_bg, "obj": lam_obj, "perp": lam_perp, "mass": lam_mass}, "eta": eta, "steps": steps, "shift": shift,
              "cfg": cfg_scale, "weights": weights, "checkpointing": ckpt, "anchors": anchors, "smoke": args.smoke, "wall_s": wall, "gpu": torch.cuda.get_device_name(0), "corrector_off_steps": corrector_off,
              "latent_mask_overlap": (rec_overlap if (args.arm == "A3" and args.object) else None)}
    np.savez_compressed(out_dir / f"{name}.npz", latents=z.float().cpu().numpy(), log=json.dumps(log), config=json.dumps(config))
    guided = [r for r in log if r["guided"]]
    ep = lambda r: r.get("epe_bg_patch_mean", r.get("monitor_epe_bg"))                                              # noqa: E731
    summary = {**config, "video": str(out_dir / f"{name}.mp4"), "n_guided_steps": len(guided), "gamma": (gamma if args.arm == "A3" else None),
               "epe_bg_first_guided": ep(guided[0]) if guided else None, "epe_bg_last_guided": ep(guided[-1]) if guided else None,
               "L_bg_first": guided[0].get("L_bg") if guided else None, "L_bg_last": guided[-1].get("L_bg") if guided else None,
               "mass_first": guided[0].get("mass_at_target_mean", guided[0].get("monitor_mass")) if guided else None, "mass_last": guided[-1].get("mass_at_target_mean", guided[-1].get("monitor_mass")) if guided else None,
               "masked_resid_first": guided[0].get("masked_resid_rms_before") if guided else None, "masked_resid_last": guided[-1].get("masked_resid_rms_before") if guided else None,
               "n_object_corrected_steps": sum(1 for r_ in guided if r_.get("object_corrected_this_step")),
               "object_resid_first": guided[0].get("object_resid_rms_before") if guided else None, "object_resid_last": guided[-1].get("object_resid_rms_before") if guided else None,
               "peak_gb": max((r.get("peak_gb") or 0 for r in guided), default=None), "guided_step_wall_s": float(np.mean([r["wall_s"] for r in guided])) if guided else None}
    print(json.dumps({k: v for k, v in summary.items() if k not in ("lambdas",)}, indent=1))
    if args.demo_label:
        label = {"registered": False, "claim_eligible": False, "camera_gate": "undetermined (failed under ViPE, D4RT, BA)", "band_ratio": "undetermined (12-16 pts < 30)",
                 "beta": {"value": 1.0, "source": "gauge identity, not measured"}, "purpose": "poster demonstration", "label": args.demo_label, "pair": args.pair,
                 "arm": args.arm, "seed": args.seed, "prompt_used": prompt, "anchor": args.anchor_override, "video": str(out_dir / f"{name}.mp4"),
                 "config": {k: v for k, v in config.items() if k not in ("lambdas",)}, "summary": {k: v for k, v in summary.items() if k not in ("lambdas",)}}
        (out_dir / f"{name}.json").write_text(json.dumps(label, indent=1, default=str), encoding="utf-8")
        print(f"[S4 demo] UNREGISTERED run written to {out_dir} (not in guided_runs.json)", flush=True)
    elif not args.smoke:
        rep = root / "reports" / "m3_s4"; rep.mkdir(parents=True, exist_ok=True)
        jf = rep / "guided_runs.json"
        prev = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else []
        key = lambda r: tuple(str(v) for v in (r["pair"], r["arm"], r["seed"], r["bridge"], r["frames"], r["weights"], bool(r.get("object_render")), r.get("obj_lowpass_sigma_latent"), tuple(r.get("obj_window_sigma") or ()), r.get("correct_from_sigma"), bool(r.get("soft_mask")), bool(r.get("sky_mask")), r.get("obj_taper_steps"), r.get("motion_clause"), r.get("anchor_override"), bool(r.get("no_processor")), r.get("scheduler"), bool(r.get("keep_corrector")), bool(r.get("corrector_off_control")), r.get("clause_cfg"), r.get("image_cfg"), tuple(r.get("bg_window_override") or ()), bool(r.get("nodyn_mask")), r.get("gamma_override"), r.get("ablation_label")))          # noqa: E731
        dup = {key(r) for r in prev} & {key(summary)}
        assert not dup or args.force, f"rule 24: {sorted(dup)} already recorded; pass --force to replace"
        rows = sorted([r for r in prev if key(r) != key(summary)] + [summary], key=key)
        jf.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"-> {jf} ({len(rows)} runs). Grade: python scripts/measure_m3_a0.py --pairs {args.pair} --videos {summary['video']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
