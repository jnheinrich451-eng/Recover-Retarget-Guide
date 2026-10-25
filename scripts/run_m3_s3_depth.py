#!/usr/bin/env python
"""Module 3 S3 (brief v2.6 §9.1): at which noise levels is the depth read from the decoded x_hat_0 good enough to lift new background
from? Two arms, one metric (src/generate/m3/refresh_depth.py, unit-tested):
  T  teacher-forced: the clean SOURCE clip encoded, noised to sigma, one I2V forward, x_hat_0 = z - sigma v_hat; reference = depth of the
     clean source frame. The instrument and upper bound (S2's harness).
  F  free-running: an unguided A0 sampling on the TARGET pair (frozen I2V configuration, CFG as generation uses it); x_hat_0 at each grid
     sigma from the same v_hat the scheduler steps with; reference = depth of the FINAL frame of the same run. Arm of record for R.
     Its videos are saved with their seeds and are the A0 row of S4.
Self-tests before any number is read: prefix decode == full decode (causal VAE), the VAE round-trip floor, arm F's last x_hat_0 == final frame.
Gate: absrel_all <= lock gates.G2_depth_absrel_patch_max on every row of record. R by refresh_depth.refresh_rule; John writes refresh.R_sigma.
Phase 2 (bootstrap_phase2.sh, which installs MoGe; in an older session: pip install --no-deps git+https://github.com/microsoft/MoGe.git), A100.
Rule 24: rows extend reports/m3_s3/s3_depth.json; --force replaces.

    python scripts/run_m3_s3_depth.py --pairs 000000000265.2.005/vehicle 000000000265.2.005/human
    python scripts/run_m3_s3_depth.py --pairs 000000000370.1.008/human --seeds 0        # labelled: one seed, reported outside
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_cotracker import read_video                                   # noqa: E402
from scripts.run_m3_s2_readout import REPO, SIGMAS, WAN_H, WAN_W                 # noqa: E402
from src.generate.m3 import depth_moge, handoff as H, refresh_depth as RD        # noqa: E402

FOCAL_REL_TOL = 0.01


def depth_wan_frame(model, frame_u8: np.ndarray, K_wan: np.ndarray, device: str) -> np.ndarray:
    """MoGe on an 832x480 frame through the §2.6 rule (square pixels, FOV passed), returned on the Wan grid [480, 832] with NaN outside the mask."""
    import cv2
    import torch
    assert frame_u8.shape == (WAN_H, WAN_W, 3) and frame_u8.dtype == np.uint8
    Wq = int(round(WAN_W * K_wan[1, 1] / K_wan[0, 0]))
    Kq = np.diag([Wq / WAN_W, 1.0, 1.0]) @ np.asarray(K_wan, np.float64)
    img = cv2.resize(frame_u8, (Wq, WAN_H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).to(device).float().permute(2, 0, 1) / 255.0
    with torch.no_grad():
        out = model.infer(x, fov_x=depth_moge.fov_x_deg(Kq, Wq))
    out = {k: (v.detach().float().cpu().numpy() if hasattr(v, "detach") else v) for k, v in out.items()}
    dep, mask = depth_moge.check_output(out, Kq, FOCAL_REL_TOL)
    d = np.where(mask, dep, np.nan).astype(np.float32)
    d_wan = cv2.resize(d, (WAN_W, WAN_H), interpolation=cv2.INTER_NEAREST)           # back to the Wan grid; NaN stays NaN under nearest
    return d_wan.astype(np.float64)


def to_u8(frames):
    """[1,3,F,H,W] in [-1,1] -> [F,H,W,3] uint8."""
    return ((frames[0].permute(1, 2, 3, 0).float().clamp(-1, 1) + 1.0) * 127.5).round().to("cpu").numpy().astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--videos", default=__import__("os").environ.get("MT_DATA", "/content/data") + "/videos")
    ap.add_argument("--pairs", nargs="+", required=True, help="clip/category")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="default: lock m3.eval.seeds")
    ap.add_argument("--sigmas", nargs="+", type=float, default=list(SIGMAS))
    ap.add_argument("--arms", nargs="+", default=["T", "F"], choices=["T", "F"])
    ap.add_argument("--checkpoint", default="Ruicheng/moge-2-vitl-normal")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--weights", default=None, choices=["fp8", "bf16"], help="default: lock m3.weights. bf16 = the 80 GB reference (FP8-vs-bf16 generation check); rows carry the tag")
    args = ap.parse_args(argv)
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import cv2
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanImageToVideoPipeline
    from src.generate.ditflow_wan.transformer import ControlledWanTransformer
    lock = H.load_m3_lock(); lat = lock["lattice"]; Hp, Wp, px = int(lat["Hp"]), int(lat["Wp"]), int(lat["px_per_patch"])
    assert "G2_depth_absrel_patch_max" in lock["gates"], "lock m3.gates.G2_depth_absrel_patch_max missing (v2.6 §8: 0.15); John adds it"
    g2 = float(lock["gates"]["G2_depth_absrel_patch_max"])
    seeds = args.seeds or (lock.get("eval") or {}).get("seeds")
    assert seeds, "lock m3.eval.seeds missing (v2.6 §8 SIGN line); John adds it or passes --seeds"
    m3cfg = __import__("yaml").safe_load(Path("configs/m3.yaml").read_text(encoding="utf-8"))["m3"]
    samp = m3cfg.get("sampling", {})
    cfg_scale = float(samp.get("guidance_scale", 5.0)); neg = str(samp.get("negative_prompt", ""))
    steps, shift = int(lock["scheduler"]["steps"]), float(lock["scheduler"]["shift"])
    weights = args.weights or lock.get("weights", "fp8")
    dev, dt = "cuda", torch.bfloat16
    root = Path(args.drive_root)

    # ---- models: transformer under the generation numerics, VAE f32, MoGe
    tr = ControlledWanTransformer.from_pretrained(REPO, subfolder="transformer", torch_dtype=dt).requires_grad_(False).eval()
    if weights == "fp8":
        from torchao.quantization import Float8WeightOnlyConfig, quantize_
        quantize_(tr, Float8WeightOnlyConfig())
    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(REPO, transformer=tr, vae=vae, torch_dtype=dt)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=shift)
    pipe.to(dev)
    from moge.model.v2 import MoGeModel
    moge = MoGeModel.from_pretrained(args.checkpoint).to(dev).eval()
    lm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(dev)
    ls = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(dev)

    def decode(lat_norm, n_latent=None):
        """normalised latents [1,C,T,h,w] -> frames [1,3,F,H,W] in [-1,1]; prefix of n_latent latent frames (causal VAE)."""
        z = lat_norm if n_latent is None else lat_norm[:, :, :n_latent]
        with torch.no_grad():
            return vae.decode((z.float() * ls + lm)).sample

    written_this_run = set()

    def score_controls(maps, pair, frames_b, seeds, g2, weights):
        """N1 / N2 for every arm-F (seed, b, sigma) with the needed maps present. Returns rows that carry the control columns and `power`."""
        out = []
        for seed in seeds:
            others = [s2 for s2 in seeds if s2 != seed and ("ref", "F", s2, frames_b[0]) in maps]
            for b in frames_b:
                b_other = [bb for bb in frames_b if bb != b]
                for key, D_s in list(maps.items()):
                    if len(key) != 4 or key[0] != "F" or key[1] != seed or key[2] != b:
                        continue
                    sig = key[3]
                    n1 = [RD.metric_row(D_s, maps[("ref", "F", s2, b)])["absrel_all"] for s2 in others if ("ref", "F", s2, b) in maps]
                    n2 = [RD.metric_row(D_s, maps[("ref", "F", seed, bb)])["absrel_all"] for bb in b_other if ("ref", "F", seed, bb) in maps]
                    n1v = None if not n1 else float(np.nanmin([v for v in n1 if v is not None] or [np.nan]))
                    n2v = None if not n2 else float(np.nanmin([v for v in n2 if v is not None] or [np.nan]))
                    power = (n1v is not None and n1v > g2) and (n2v is not None and n2v > g2)
                    out.append({"pair": pair, "arm": "F", "seed": seed, "frame_b": b, "sigma": sig, "weights": weights, "control_only": True,
                                "N1_cross_seed_absrel": n1v, "N2_cross_frame_absrel": n2v, "power": bool(power) if (n1v is not None and n2v is not None) else None})
        return out

    def write_record(rows, selftests, final=False, controls=False):
        """Rule 24 record + report, written after every (pair, seed) so a dead session costs at most one unit of work."""
        # ---- record (rule 24), gate, rule for R
        rep = root / "reports" / "m3_s3"; rep.mkdir(parents=True, exist_ok=True)
        jf = rep / "s3_depth.json"
        prev = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else {"rows": [], "selftests": []}
        key = lambda r: (r["pair"], r["arm"], r["seed"], r["frame_b"], r["sigma"], r.get("weights", "fp8"))           # noqa: E731
        dup = ({key(r) for r in prev["rows"]} & {key(r) for r in rows}) - written_this_run
        assert not dup or args.force, f"rule 24: {len(dup)} rows already recorded; pass --force to replace"
        written_this_run.update(key(r) for r in rows)
        if controls:
            by = {key(r): r for r in prev["rows"]}
            for c in rows:
                k = key(c)
                if k in by:
                    by[k].update({kk: vv for kk, vv in c.items() if kk.startswith("N1") or kk.startswith("N2") or kk == "power"})
            allrows = list(by.values()); rows = []
        new_keys = {key(r) for r in rows}
        allrows = ([r for r in prev["rows"] if key(r) not in new_keys] + rows) if not controls else allrows
        tkey = lambda t: (t["pair"], t["test"], t.get("seed", 0), t["frame_b"])                                           # noqa: E731
        alltests = [t for t in prev["selftests"] if tkey(t) not in {tkey(x) for x in selftests}] + selftests
        jf.write_text(json.dumps({"rows": allrows, "selftests": alltests, "gate": g2, "scheduler": {"steps": steps, "shift": shift, "cfg": cfg_scale}}, indent=1), encoding="utf-8")
        # the rule: a sigma passes iff absrel_all <= G2 on EVERY row of record (arm F, both pairs of record, every seed, every frame, FP8).
        # A row whose controls both pass (power False) counts as "no power", neither pass nor fail (reviewer 2026-10-09): it leaves R untouched.
        record_pairs = [p for p in sorted({r["pair"] for r in allrows}) if p.startswith("000000000265.2.005")]
        passing, no_power = [], []
        for s in sorted({r["sigma"] for r in allrows}, reverse=True):
            rs = [r for r in allrows if r["arm"] == "F" and r["sigma"] == s and r["pair"] in record_pairs and r.get("weights", "fp8") == "fp8"]
            if rs and all(r["gate"] == "pass" for r in rs):
                passing.append(s)
            if any(r.get("power") is False for r in rs):
                no_power.append(s)
        R = RD.refresh_rule(passing)
        f = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"                                                           # noqa: E731
        md = [f"# Module 3 S3 — depth from x_hat_0 at patch scale (v2.6 §9.1); gate absrel_all <= {g2}; scheduler {steps} steps, shift {shift}, CFG {cfg_scale}", "",
              "## Self-tests", "", "| pair | test | seed | frame b | value | pass |", "|---|---|---|---|---|---|"]
        for t in alltests:
            val = t.get("max_abs_diff", t.get("absrel_all"))
            extra = f" (median {f(t['median_abs_diff'], 5)}; depth AbsRel prefix vs full {f(t.get('depth_absrel_prefix_vs_full'), 4)})" if "median_abs_diff" in t else ""
            md.append(f"| {t['pair'][-22:]} | {t['test']} | {t.get('seed', 0)} | {t['frame_b']} | {f(val, 4)}{extra} | {t.get('pass', '-')} |")
        md += ["", "## Rows", "", "| pair | arm | seed | b | sigma (actual) | weights | absrel_all | absrel_border_extrap | N1 cross-seed / N2 cross-frame (power) | c / d / R² | unaligned ratio | valid share | rgb MAD | induced EPE median / p90 | wall s / peak GB | gate |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(allrows, key=lambda r: (r["pair"], r["arm"], r["seed"], r["frame_b"], -r["sigma"])):
            fl = r.get("induced_flow") or {}
            md.append(f"| {r['pair'][-22:]} | {r['arm']} | {r['seed']} | {r['frame_b']} | {r['sigma']} ({f(r['sigma_actual'])}) | {r.get('weights', 'fp8')} | **{f(r['absrel_all'])}** | {f(r['absrel_border_extrap'])} | "
                      f"{f(r.get('N1_cross_seed_absrel'))} / {f(r.get('N2_cross_frame_absrel'))} ({'-' if r.get('power') is None else ('yes' if r['power'] else 'NO POWER')}) | "
                      f"{f(r['c'])} / {f(r['d'], 4)} / {f(r['r2'])} | {f(r['unaligned_ratio_median'])} | {f(r['valid_share_both'], 2)} | {f(r['rgb_mad_patch'], 1)} | "
                      f"{f(fl.get('epe_median_patch'), 2)} / {f(fl.get('epe_p90_patch'), 2)} | {f(r['refresh_wall_s'], 1)} / {f(r['refresh_peak_gb'], 1)} | {r['gate']} |")
        md += ["", f"## Rule for R (pre-registered): passing sigmas on every row of record (arm F, {record_pairs}): {passing} → **R = {R}**; sigmas with a no-power row: {no_power}",
               "", "John writes `refresh.R_sigma` from this; CC does not. Registered expectations (reviewer, §9.1): VAE floor < 0.03; arm T <= arm F at every sigma; arm F fails at "
               "0.95 and 0.85, passes from 0.55 down, 0.70 open; absrel_all falls monotonically on >= 5 of 6 steps; the later frame is worse at fixed sigma; "
               "absrel_border_extrap > absrel_all at every sigma. Stated limit: an unguided run does not follow E'_t, so the refresh's trusted/candidate split does not exist here; "
               "absrel_border_extrap is its proxy (S5 logs the pipeline-form number)."]
        (rep / "s3_depth.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        if final:
            print("\n".join(md[-6:]))
        return allrows

    rows, selftests = [], []
    maps_dir = root / "reports" / "m3_s3" / "depth_maps"; maps_dir.mkdir(parents=True, exist_ok=True)
    for pair in args.pairs:
        maps = {}                                                               # (arm, seed, b, sigma) -> patch depth of x_hat_0; ("ref", arm, seed, b) -> reference
        clip, cat = pair.split("/")
        h = json.loads((root / "cache" / "m2_e" / clip / cat / "handoff.json").read_text(encoding="utf-8"))
        t_a = int(h["t_a"]); F_pair = int(h["n_frames_after_t_a"]); F_pair = 4 * ((F_pair - 1) // 4) + 1; T_l = 1 + (F_pair - 1) // 4
        prompt = json.loads((root / "cache" / "m2_d" / clip / cat / "qc_m2d.json").read_text(encoding="utf-8"))["prompt"]
        with np.load(root / "cache" / "m3" / clip / cat / "lift.npz") as z:
            K_wan, Kp = z["K_wan"].astype(np.float64), z["Kp"].astype(np.float64)
        gz = root / "cache" / "m3" / clip / cat / "geometry.npz"
        with np.load(gz) as z:
            E_l, anchors, sky = z["E_l"].astype(np.float64), [int(a) for a in z["anchors"]], z["sky_lattice"].astype(bool)
            beta_tag = str(z["bridge_run"]) if "bridge_run" in z.files else "v2.5"
        frames_b = [a for a in anchors if a != 0] or [T_l // 3, 2 * T_l // 3]
        frames_b = sorted({b for b in frames_b if 0 < b < T_l})
        # the source clip from t_a (arm T) and the target anchor frame (arm F)
        video = read_video(Path(args.videos) / f"{clip}.mp4", t_a + F_pair, WAN_W, WAN_H)[t_a: t_a + F_pair]
        anchor = cv2.imread(str(root / "cache" / "m2_d" / clip / cat / "anchor.png"))[..., ::-1]
        anchor_wan = cv2.resize(anchor, (WAN_W, WAN_H), interpolation=cv2.INTER_AREA)
        for seed in seeds:
            gen = torch.Generator(dev).manual_seed(seed)
            with torch.no_grad():
                pe, ne = pipe.encode_prompt(prompt, negative_prompt=neg, do_classifier_free_guidance=True, device=dev)
                # ---- arm T inputs: the source's own first frame as the I2V condition, the clean clip encoded
                img_src = Image.fromarray(video[0])
                ie_src = pipe.encode_image(img_src, dev)
                img_t = pipe.video_processor.preprocess(img_src, height=WAN_H, width=WAN_W).to(dev, dtype=torch.float32)
                _, cond_src = pipe.prepare_latents(img_t, 1, vae.config.z_dim, WAN_H, WAN_W, F_pair, dt, dev, gen)
                vid = torch.from_numpy(video).permute(3, 0, 1, 2)[None].float().to(dev) / 127.5 - 1.0
                x0_src = ((vae.encode(vid).latent_dist.mode() - lm) / ls).to(dt)
                # ---- arm F inputs: the target anchor as the I2V condition, random initial latents
                img_tar = Image.fromarray(anchor_wan)
                ie_tar = pipe.encode_image(img_tar, dev)
                img_tt = pipe.video_processor.preprocess(img_tar, height=WAN_H, width=WAN_W).to(dev, dtype=torch.float32)
                zT, cond_tar = pipe.prepare_latents(img_tt, 1, vae.config.z_dim, WAN_H, WAN_W, F_pair, dt, dev, torch.Generator(dev).manual_seed(seed))
            pipe.text_encoder.to("cpu"); pipe.image_encoder.to("cpu"); torch.cuda.empty_cache()
            assert x0_src.shape[2] == T_l and zT.shape[2] == T_l
            # ---- self-tests (seed 0 only; they do not depend on the seed)
            if seed == seeds[0]:
                full = decode(x0_src)
                full2 = decode(x0_src)
                d2 = float((full2 - full).abs().max() / 2.0)
                selftests.append({"pair": pair, "test": "full_decode_twice", "frame_b": -1, "max_abs_diff": d2, "pass": d2 < 1e-3,
                                  "note": "if this also differs at ~1e-3 the prefix-decode threshold measured GPU numerics, not causality (reviewer 2026-10-09)"})
                del full2
                for b in frames_b:
                    pre = decode(x0_src, b + 1)
                    dpx = (pre[:, :, 4 * b] - full[:, :, 4 * b]).abs() / 2.0                                # in [0,1] units
                    diff, dmed = float(dpx.max()), float(dpx.median())
                    D_pre = RD.patch_depth(depth_wan_frame(moge, to_u8(pre)[4 * b], K_wan, dev), px, Hp, Wp)
                    D_ful = RD.patch_depth(depth_wan_frame(moge, to_u8(full)[4 * b], K_wan, dev), px, Hp, Wp)
                    r_pf = RD.metric_row(D_pre, D_ful)
                    selftests.append({"pair": pair, "test": "prefix_decode_equals_full", "frame_b": b, "max_abs_diff": diff, "median_abs_diff": dmed,
                                      "depth_absrel_prefix_vs_full": r_pf["absrel_all"], "pass": diff < 1e-3})
                rt = to_u8(full)
                floors = []
                for b in frames_b:
                    D_clean = RD.patch_depth(depth_wan_frame(moge, video[4 * b], K_wan, dev), px, Hp, Wp)
                    D_rt = RD.patch_depth(depth_wan_frame(moge, rt[4 * b], K_wan, dev), px, Hp, Wp)
                    r = RD.metric_row(D_rt, D_clean); floors.append(r["absrel_all"])
                    selftests.append({"pair": pair, "test": "vae_floor", "frame_b": b, **{k: r[k] for k in ("absrel_all", "absrel_border_extrap", "c", "d", "r2", "valid_share_both")}})
                del full, pre, rt; torch.cuda.empty_cache()
                if max(v for v in floors if v is not None) > g2:
                    print(f"[S3] {pair}: VAE floor {floors} above the gate {g2}: the instrument cannot grade; S3 stops on this pair")
                    continue
            sig_list = sorted(args.sigmas, reverse=True)
            # ---- arm T
            if "T" in args.arms:
                D_clean = {b: RD.patch_depth(depth_wan_frame(moge, video[4 * b], K_wan, dev), px, Hp, Wp) for b in frames_b}
                for s in sig_list:
                    eps = torch.randn(x0_src.shape, generator=torch.Generator(dev).manual_seed(seed * 1000 + int(s * 100)), device=dev, dtype=dt)
                    z = (1.0 - s) * x0_src + s * eps
                    ts = torch.tensor([int(round(s * 1000))], device=dev)
                    tr.stop_after_block = None
                    with torch.no_grad():
                        v = tr(torch.cat([z, cond_src.to(dt)], 1), ts, pe.to(dt), encoder_hidden_states_image=ie_src.to(dt), return_dict=False)[0]
                    xhat = z.float() - s * v.float()
                    for b in frames_b:
                        t0 = time.time(); torch.cuda.reset_peak_memory_stats()
                        fr = to_u8(decode(xhat, b + 1))[4 * b]
                        D_s = RD.patch_depth(depth_wan_frame(moge, fr, K_wan, dev), px, Hp, Wp)
                        wall = time.time() - t0; peak = torch.cuda.max_memory_allocated() / 2 ** 30
                        r = RD.metric_row(D_s, D_clean[b]); maps[("T", seed, b, s)] = D_s; maps[("ref", "T", seed, b)] = D_clean[b]
                        fit = RD.align_disparity(D_s, D_clean[b], np.isfinite(D_s) & np.isfinite(D_clean[b])) if r["absrel_all"] is not None else None
                        flow = RD.induced_flow_error(fit["D_tilde"], D_clean[b], Kp, E_l[b], E_l[b + 1:]) if fit else {}
                        rgb = float(np.abs(fr.astype(np.float32) - video[4 * b].astype(np.float32)).mean())
                        rows.append({"pair": pair, "arm": "T", "seed": seed, "frame_b": b, "sigma": s, "sigma_actual": s, **r, "sky_iou": None, "weights": weights,
                                     "rgb_mad_patch": rgb, "induced_flow": flow, "beta_tag": beta_tag, "refresh_wall_s": wall, "refresh_peak_gb": peak,
                                     "gate": "pass" if (r["absrel_all"] is not None and r["absrel_all"] <= g2) else "fail"})
                        print(f"[S3 T] {pair} seed {seed} b={b} sigma={s}: absrel {r['absrel_all']} ({rows[-1]['gate']}), border {r['absrel_border_extrap']}, {wall:.1f} s, {peak:.1f} GB")
            # ---- arm F: the unguided A0 sampling, x_hat_0 captured at the grid sigmas, final video saved
            if "F" in args.arms:
                sch = pipe.scheduler; sch.set_timesteps(steps, device=dev)
                sig_sched = sch.sigmas.float().cpu().numpy()[:steps]
                want = {s: int(np.argmin(np.abs(sig_sched - s))) for s in sig_list}                      # nearest scheduler step per grid sigma
                z = zT.to(dt); xhat_at = {}
                tr.stop_after_block = None
                t_run = time.time()
                for k, t in enumerate(sch.timesteps):
                    s_k = float(sig_sched[k])
                    ts = t.expand(1)
                    with torch.no_grad():
                        v_c = tr(torch.cat([z, cond_tar.to(dt)], 1), ts, pe.to(dt), encoder_hidden_states_image=ie_tar.to(dt), return_dict=False)[0]
                        v_u = tr(torch.cat([z, cond_tar.to(dt)], 1), ts, ne.to(dt), encoder_hidden_states_image=ie_tar.to(dt), return_dict=False)[0]
                    v = v_u + cfg_scale * (v_c - v_u)
                    for s, kk in want.items():
                        if kk == k:
                            xhat_at[s] = ((z.float() - s_k * v.float()).detach(), s_k)
                    if k == steps - 1:
                        xhat_last = (z.float() - s_k * v.float()).detach()
                    z = sch.step(v, t, z, return_dict=False)[0].to(dt)
                    if k % 10 == 0 or k == steps - 1:
                        print(f"[S3 F] {pair} seed {seed}: step {k + 1}/{steps} (sigma {s_k:.3f}), {time.time() - t_run:.0f} s", flush=True)
                wall_run = time.time() - t_run
                print(f"[S3 F] {pair} seed {seed}: sampling done in {wall_run:.0f} s; decoding the final video and the self-test frames", flush=True)
                final = decode(z); final_u8 = to_u8(final)
                out_dir = root / "cache" / "m3" / clip / cat; out_dir.mkdir(parents=True, exist_ok=True)
                wtag = "" if weights == "fp8" else f"_{weights}"
                np.savez_compressed(out_dir / f"A0_seed{seed}{wtag}.npz", latents=z.float().cpu().numpy(), seed=seed, steps=steps, shift=shift, cfg=cfg_scale, prompt=prompt, beta_tag=beta_tag, weights=weights,
                                    **{f"xhat_sigma_{s}": xhat_at[s][0].to(torch.float16).cpu().numpy() for s in xhat_at}, **{f"sigma_actual_{s}": np.float64(xhat_at[s][1]) for s in xhat_at})
                vw = cv2.VideoWriter(str(out_dir / f"A0_seed{seed}{wtag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 16, (WAN_W, WAN_H))
                for f_ in final_u8:
                    vw.write(f_[..., ::-1])
                vw.release()
                D_final = {b: RD.patch_depth(depth_wan_frame(moge, final_u8[4 * b], K_wan, dev), px, Hp, Wp) for b in frames_b}
                # arm F sanity: the last step's x_hat_0 against the final frame
                last_u8 = to_u8(decode(xhat_last))
                for b in frames_b:
                    r = RD.metric_row(RD.patch_depth(depth_wan_frame(moge, last_u8[4 * b], K_wan, dev), px, Hp, Wp), D_final[b])
                    selftests.append({"pair": pair, "seed": seed, "test": "armF_last_xhat_equals_final", "frame_b": b, "absrel_all": r["absrel_all"], "pass": r["absrel_all"] is not None and r["absrel_all"] < 0.02})
                del final, last_u8; torch.cuda.empty_cache()
                for s in sig_list:
                    xhat, s_act = xhat_at[s]
                    for b in frames_b:
                        t0 = time.time(); torch.cuda.reset_peak_memory_stats()
                        fr = to_u8(decode(xhat, b + 1))[4 * b]
                        D_s = RD.patch_depth(depth_wan_frame(moge, fr, K_wan, dev), px, Hp, Wp)
                        wall = time.time() - t0; peak = torch.cuda.max_memory_allocated() / 2 ** 30
                        r = RD.metric_row(D_s, D_final[b]); maps[("F", seed, b, s)] = D_s; maps[("ref", "F", seed, b)] = D_final[b]
                        fit = RD.align_disparity(D_s, D_final[b], np.isfinite(D_s) & np.isfinite(D_final[b])) if r["absrel_all"] is not None else None
                        flow = RD.induced_flow_error(fit["D_tilde"], D_final[b], Kp, E_l[b], E_l[b + 1:]) if fit else {}
                        rgb = float(np.abs(fr.astype(np.float32) - final_u8[4 * b].astype(np.float32)).mean())
                        rows.append({"pair": pair, "arm": "F", "seed": seed, "frame_b": b, "sigma": s, "sigma_actual": float(s_act), **r, "sky_iou": None,
                                     "rgb_mad_patch": rgb, "induced_flow": flow, "beta_tag": beta_tag, "refresh_wall_s": wall, "refresh_peak_gb": peak,
                                     "run_wall_s": wall_run, "video": str(out_dir / f"A0_seed{seed}{wtag}.mp4"), "weights": weights,
                                     "gate": "pass" if (r["absrel_all"] is not None and r["absrel_all"] <= g2) else "fail"})
                        print(f"[S3 F] {pair} seed {seed} b={b} sigma={s} ({s_act:.3f}): absrel {r['absrel_all']} ({rows[-1]['gate']}), border {r['absrel_border_extrap']}, {wall:.1f} s, {peak:.1f} GB")
                del xhat_at; torch.cuda.empty_cache()
            write_record(rows, selftests); rows, selftests = [], []                       # checkpoint the record after every (pair, seed)
        # per pair, after every seed: save the maps and score the controls on the arm-F rows that have the data
        np.savez_compressed(maps_dir / f"{pair.replace('/', '_')}_{weights}.npz", **{"|".join(str(x) for x in k): v for k, v in maps.items()})
        ctrl_rows = score_controls(maps, pair, frames_b, seeds, g2, weights)
        if ctrl_rows:
            write_record(ctrl_rows, [], controls=True)
    write_record(rows, selftests, final=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
