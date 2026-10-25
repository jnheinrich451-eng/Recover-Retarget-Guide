#!/usr/bin/env python
"""Module 3 S2 — attention readout gate G1 (brief v2.5 §4, §9): does Wan's attention at block l, read as an expected
displacement F_hat, agree with real motion on real clips? Measured under the generation numerics (FP8, lock m3.weights)
on the A100 40 GB; a bf16 reference run on an 80 GB card is the pre-registered FP8 test (reviewer 2026-10-07).

Per clip: the SOURCE video (832x480, F frames) is encoded by Wan's VAE to real latents, noised at each sigma of the grid
(flow matching: z = (1 - s) x0 + s eps), run with I2V conditioning (first frame + CLIP image embeds + the pair's prompt)
to block l with Q/K captured after RoPE (the August-probed processor), and F_hat^{t->0} is computed per pair (0, t) from the
sub-block softmax (queries of frame t over keys of frame 0), heads averaged. Ground truth: CoTracker3 tracks from a grid
of patch centres at frame 0, rasterised onto the frame-t lattice with the same bilinear rasteriser as F_geo (so the
comparison is on identical footing). Metrics on covered patches (C_gt > 0.5): median EPE in patches, Spearman between
F_hat and F_gt components. Gates: lock m3.gates.G1_fhat_epe_patch_max, G1_fhat_spearman_min. Report only; John picks l
and the sigma window from the table.

    python scripts/run_m3_s2_readout.py --clips 000000000265.2.005 --frames 81 --blocks 8 12 16 20 24
    python scripts/run_m3_s2_readout.py --clips 000000000265.2.005 --frames 33            # the short-pair re-check (contract 4a)
    python scripts/run_m3_s2_readout.py ... --weights bf16                                # the 80 GB reference run
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
from src.generate.m3 import handoff as H                                          # noqa: E402
from src.generate.m3 import readout as RO                                         # noqa: E402
from src.generate.m3.geometry import rasterize                                    # noqa: E402

REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
WAN_W, WAN_H = 832, 480
SIGMAS = (0.95, 0.85, 0.70, 0.55, 0.45, 0.30)


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else 0.0


def gt_fields(tracks, vis, Hp, Wp, px, T_l, eps=1e-6):
    """tracks [F,N,2] px (x, y) from frame-0 queries at patch centres; vis [F,N]. Per latent frame j (video frame 4j):
    F_gt^{t->0} on the frame-t lattice = (p_0 - p_t) rasterised at p_t, C_gt its coverage."""
    p0 = tracks[0] / px
    Fg, Cg, Ng = np.zeros((T_l, Hp, Wp, 2)), np.zeros((T_l, Hp, Wp)), np.zeros((T_l, Hp, Wp), int)
    for j in range(T_l):
        f = 4 * j
        pt = tracks[f] / px
        v = (vis[0] > 0.5) & (vis[f] > 0.5)
        Fg[j], Cg[j] = rasterize.pair_fields(p0, pt, v, v, Hp, Wp, eps)
        u_, v_ = np.floor(pt[v, 0]).astype(int), np.floor(pt[v, 1]).astype(int)       # tracks landing inside each frame-t patch
        ok = (u_ >= 0) & (u_ < Wp) & (v_ >= 0) & (v_ < Hp)
        np.add.at(Ng[j], (v_[ok], u_[ok]), 1)
    return Fg, Cg, Ng


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--videos", default=__import__("os").environ.get("MT_DATA", "/content/data") + "/videos")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--blocks", nargs="+", type=int, default=[8, 12, 16, 20, 24])
    ap.add_argument("--sigmas", nargs="+", type=float, default=list(SIGMAS))
    ap.add_argument("--weights", default=None, help="fp8 (lock default) | bf16 (the 80 GB reference run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readout", default="expected", choices=["expected", "windowed", "heads"],
                    help="expected = plain expectation (brief §4, the form that failed 2026-10-07); windowed = soft-argmax within configs/m3.yaml "
                         "readout.window_radius_patches (reviewer 2026-10-07g); heads = plain expectation over a head subset from --heads-from")
    ap.add_argument("--heads-from", default=None, help="json {block: [head indices]} selected on a DIFFERENT clip (scripts/select_m3_heads.py)")
    ap.add_argument("--force", action="store_true", help="replace (clip, sigma, block) rows already recorded in the output json (rule 24: refused by default)")
    ap.add_argument("--diagnose", action="store_true", help="what a failed gate means (reviewer 2026-10-07f): self-pair argmax, hit mass at the "
                    "ground-truth source, entropy, centre-pull correlation, per-head EPE; at every (block, sigma), mid latent frame")
    args = ap.parse_args(argv)
    import torch
    from PIL import Image
    from src.generate.ditflow_wan.attention import WanInjectionProcessor
    from src.generate.ditflow_wan.transformer import ControlledWanTransformer
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    lock = H.load_m3_lock(); lat = lock["lattice"]; Hp, Wp, px = int(lat["Hp"]), int(lat["Wp"]), int(lat["px_per_patch"])
    m3cfg = __import__("yaml").safe_load(Path("configs/m3.yaml").read_text(encoding="utf-8"))["m3"]
    win_r = float(m3cfg["readout"]["window_radius_patches"])
    heads_by_block = {int(k): v for k, v in json.loads(Path(args.heads_from).read_text(encoding="utf-8")).items()} if args.heads_from else {}
    assert args.readout != "heads" or heads_by_block, "--readout heads needs --heads-from"
    weights = args.weights or lock.get("weights", "bf16")
    assert "gates" in lock, "lock m3.gates missing (brief §8: G1_fhat_epe_patch_max 1.0, G1_fhat_spearman_min 0.6); John adds it"
    g1_epe, g1_rho = float(lock["gates"]["G1_fhat_epe_patch_max"]), float(lock["gates"]["G1_fhat_spearman_min"])
    dev = "cuda"; dt = torch.bfloat16
    assert (args.frames - 1) % 4 == 0
    T_l = 1 + (args.frames - 1) // 4
    root = Path(args.drive_root)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    N = Hp * Wp
    jj, ii = np.meshgrid(np.arange(Wp) + 0.5, np.arange(Hp) + 0.5)
    centres = np.stack([jj.ravel(), ii.ravel()], 1)                                   # patch units (the readout's query lattice)
    # tracker queries: 2x2 per patch (reviewer 2026-10-07e), so the ">= 3 tracks per patch" column exists; stride px/2 in pixels
    qj, qi = np.meshgrid(np.arange(2 * Wp) + 0.5, np.arange(2 * Hp) + 0.5)
    q_px = np.stack([qj.ravel(), qi.ravel()], 1) * (px / 2)
    # ---- ground truth FIRST (CoTracker needs 11.7 GB on its own: probe C5), cached per clip and length, GPU freed before Wan
    gt = {}
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(dev).eval()
    for clip in args.clips:
        cf = root / "cache" / "m3" / clip / f"cotracker_F{args.frames}_q4.npz"
        if cf.exists():
            with np.load(cf) as z:
                gt[clip] = (z["tracks"], z["vis"])
            continue
        video = read_video(Path(args.videos) / f"{clip}.mp4", args.frames, WAN_W, WAN_H)
        vt = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(dev)
        tr_all, vi_all = [], []
        for ch in np.array_split(np.arange(len(q_px)), 4):                               # 4 chunks of 1560: the probe's measured 11.7 GB each
            q = torch.from_numpy(np.concatenate([np.zeros((len(ch), 1)), q_px[ch]], 1)).float().to(dev)[None]
            with torch.no_grad():
                tracks, vis = cot(vt, queries=q)                                         # conventions per probe_cotracker C1/C2
            tr_all.append(tracks[0].float().cpu().numpy()); vi_all.append(vis[0].float().cpu().numpy())
            del tracks, vis; torch.cuda.empty_cache()
        gt[clip] = (np.concatenate(tr_all, 1), np.concatenate(vi_all, 1))
        cf.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cf, tracks=gt[clip][0], vis=gt[clip][1], grid="2x2_per_16px_patch_stride8", query_frame=0, input="clean_source_video")
        del vt
    del cot; torch.cuda.empty_cache()

    # ---- models (Phase 2): transformer under the generation numerics, VAE f32, encoders bf16 then offloaded
    tr = ControlledWanTransformer.from_pretrained(REPO, subfolder="transformer", torch_dtype=dt).requires_grad_(False).eval()
    if weights == "fp8":
        from torchao.quantization import Float8WeightOnlyConfig, quantize_
        quantize_(tr, Float8WeightOnlyConfig())
    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(REPO, transformer=tr, vae=vae, torch_dtype=dt)
    pipe.to(dev)
    procs = {}
    for b in args.blocks:
        p = WanInjectionProcessor(f"blocks.{b - 1}.attn1"); p.copy_kv = True
        tr.blocks[b - 1].attn1.set_processor(p); procs[b] = p
    c_t = torch.from_numpy(centres).float().to(dev)
    lm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(dev)
    ls = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(dev)
    c6 = None
    pc = root / "reports" / "probe_cotracker" / "probe_cotracker.json"
    if pc.exists():
        c6 = json.loads(pc.read_text(encoding="utf-8")).get("C6")
    c6_floor = None if not c6 or "gap_patches_median" not in c6 else float(c6["gap_patches_median"])
    rows = []
    for clip in args.clips:
        video = read_video(Path(args.videos) / f"{clip}.mp4", args.frames, WAN_W, WAN_H)   # [F,H,W,3] uint8 RGB
        Fg, Cg, Ng = gt_fields(gt[clip][0], gt[clip][1], Hp, Wp, px, T_l)
        # the pair's prompt (first handoff on this clip) and I2V conditioning on the source's own first frame
        hp = sorted((root / "cache" / "m2_e" / clip).glob("*/handoff.json"))[0]
        prompt = json.loads((root / "cache" / "m2_d" / clip / hp.parent.name / "qc_m2d.json").read_text(encoding="utf-8"))["prompt"]
        img0 = Image.fromarray(video[0])
        with torch.no_grad():
            pe, _ = pipe.encode_prompt(prompt, negative_prompt=None, do_classifier_free_guidance=False, device=dev)
            ie = pipe.encode_image(img0, dev)
            img_t = pipe.video_processor.preprocess(img0, height=WAN_H, width=WAN_W).to(dev, dtype=torch.float32)
            _, cond = pipe.prepare_latents(img_t, 1, vae.config.z_dim, WAN_H, WAN_W, args.frames, dt, dev, torch.Generator(dev).manual_seed(args.seed))
            vid = torch.from_numpy(video).permute(3, 0, 1, 2)[None].float().to(dev) / 127.5 - 1.0          # [1,3,F,H,W] in [-1,1]
            x0 = vae.encode(vid).latent_dist.mode()
            x0 = ((x0 - lm) / ls).to(dt)
        assert x0.shape[2] == T_l and cond.shape[2] == T_l, (x0.shape, cond.shape)
        pipe.text_encoder.to("cpu"); pipe.image_encoder.to("cpu"); torch.cuda.empty_cache()
        # trivial predictors (reviewer 2026-10-07e): a pass is a pass RELATIVE to these. Zero flow; the clip's mean flow per latent frame.
        b0, bm = [], []
        for j in range(1, T_l):
            m = Cg[j] > 0.5
            if m.sum() >= 20:
                b0.append(float(np.median(np.linalg.norm(Fg[j][m], axis=-1))))
                bm.append(float(np.median(np.linalg.norm(Fg[j][m] - Fg[j][m].mean(0), axis=-1))))
        base0, base_mean = (float(np.median(b0)) if b0 else None), (float(np.median(bm)) if bm else None)
        gen = torch.Generator(dev).manual_seed(args.seed)
        for s in args.sigmas:
            eps = torch.randn(x0.shape, generator=gen, device=dev, dtype=dt)
            z = (1.0 - s) * x0 + s * eps
            ts = torch.tensor([int(round(s * 1000))], device=dev)
            tr.stop_after_block = max(args.blocks) - 1
            t0 = time.time()
            with torch.no_grad():
                tr(torch.cat([z, cond.to(dt)], 1), ts, pe.to(dt), encoder_hidden_states_image=ie.to(dt), return_dict=False)
            wall = time.time() - t0
            for b in args.blocks:
                Qf, Kf = procs[b].query, procs[b].key                                   # [1, S, heads, D] after RoPE (probe P2)
                assert Qf is not None and Qf.shape[1] == T_l * N, (None if Qf is None else Qf.shape, T_l * N)
                K0 = Kf[0, :N].float()                                                  # keys of frame 0
                epe, rho, epe3, n_gated, diag_rec = [], [], [], [], None
                for j in range(1, T_l):
                    Qt = Qf[0, j * N:(j + 1) * N]
                    with torch.no_grad():
                        A = RO.attention_probs(Qt, Kf[0, :N])                              # [heads, N, N] over the keys of frame 0 (tested: tests/test_m3_readout.py)
                        if args.readout == "windowed":
                            Fh = RO.windowed_expected_displacement(A, c_t, win_r)
                        elif args.readout == "heads":
                            Fh = RO.expected_displacement(A, c_t, heads=heads_by_block[b])
                        else:
                            Fh = RO.expected_displacement(A, c_t)                          # backward displacement, patch units, heads averaged
                        if args.diagnose and j == T_l // 2:
                            mk_ = torch.from_numpy((Cg[j] > 0.5).reshape(-1)).to(dev)
                            Fg_t = torch.from_numpy(Fg[j].reshape(-1, 2)).to(dev)
                            dg = RO.diagnostics(A, c_t, Fg_t, mk_)
                            A_self = RO.attention_probs(Qt, Kf[0, j * N:(j + 1) * N])        # the self pair (t, t): token order and locality check
                            ds = RO.diagnostics(A_self, c_t, torch.zeros_like(Fg_t), mk_)
                            diag_rec = {"cross_pair": {k: v for k, v in dg.items() if k != "per_head_epe"}, "per_head_epe_cross": dg["per_head_epe"],
                                        "self_pair": {"argmax_self_frac": ds["argmax_self_frac"], "fhat_median_magnitude": ds["fhat_median_magnitude"], "entropy_bits_mean": ds["entropy_bits_mean"]}}
                    Fh = Fh.cpu().numpy().reshape(Hp, Wp, 2)
                    m = Cg[j] > 0.5
                    n_gated.append(int(m.sum()))
                    if m.sum() < 20:
                        continue
                    epe.append(float(np.median(np.linalg.norm(Fh[m] - Fg[j][m], axis=-1))))
                    m3 = m & (Ng[j] >= 3)
                    if m3.sum() >= 20:
                        epe3.append(float(np.median(np.linalg.norm(Fh[m3] - Fg[j][m3], axis=-1))))
                    rho.append(0.5 * (spearman(Fh[m][:, 0], Fg[j][m][:, 0]) + spearman(Fh[m][:, 1], Fg[j][m][:, 1])))
                r = {"clip": clip, "frames": args.frames, "weights": weights, "sigma": s, "block": b, "readout": args.readout,
                     "window_radius_patches": win_r if args.readout == "windowed" else None, "heads": heads_by_block.get(b) if args.readout == "heads" else None,
                     "epe_patch_median": float(np.median(epe)) if epe else None,
                     "gated_patches_per_latent_frame": n_gated, "gated_frac_last": (n_gated[-1] / N) if n_gated else None,
                     "spearman_median": float(np.median(rho)) if rho else None, "epe_patch_median_ge3_tracks": float(np.median(epe3)) if epe3 else None, "n_pairs": len(epe), "wall_s_forward": round(wall, 1),
                     "gt_median_flow_patches_last": float(np.median(np.linalg.norm(Fg[-1][Cg[-1] > 0.5], axis=-1))) if (Cg[-1] > 0.5).any() else None}
                r["G1_epe"] = bool(r["epe_patch_median"] is not None and r["epe_patch_median"] <= g1_epe)
                r["G1_spearman"] = bool(r["spearman_median"] is not None and r["spearman_median"] >= g1_rho)
                r["G1"] = "pass" if (r["G1_epe"] and r["G1_spearman"]) else "fail"
                r["diagnostics_mid_frame"] = diag_rec
                r["at_tracker_floor"] = bool(r["epe_patch_median"] is not None and c6_floor is not None and r["epe_patch_median"] < c6_floor)
                r["baseline_epe_zero_flow"] = base0; r["baseline_epe_mean_flow"] = base_mean
                rows.append(r); procs[b].clear()
                print(f"[S2] {clip} F={args.frames} {weights} sigma={s} block={b}: EPE {r['epe_patch_median']} patch, rho {r['spearman_median']} -> {r['G1']}")
    rep = root / "reports" / "m3_s2"; rep.mkdir(parents=True, exist_ok=True)
    tag = f"_{weights}_F{args.frames}" + ("" if args.readout == "expected" else f"_{args.readout}")
    jf = rep / f"readout{tag}.json"
    prev = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else []
    key = lambda r: (r["clip"], r["frames"], r["weights"], r["sigma"], r["block"], r.get("readout", "expected"))     # noqa: E731
    dup = {key(r) for r in prev} & {key(r) for r in rows}
    assert not dup or args.force, f"rule 24: {len(dup)} (clip, sigma, block) rows already recorded in {jf.name}; pass --force to replace them"
    new_keys = {key(r) for r in rows}
    rows = [r for r in prev if key(r) not in new_keys] + rows                 # the archived record, extended
    clips_md = list(dict.fromkeys(r["clip"] for r in rows)); sigmas_md = sorted({r["sigma"] for r in rows}, reverse=True); blocks_md = sorted({r["block"] for r in rows})
    jf.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    md = [f"# Module 3 S2 — attention readout vs CoTracker ({weights}, {args.frames} frames, {torch.cuda.get_device_name(0)}; readout = {args.readout}"
          + (f", window radius {win_r} patches" if args.readout == "windowed" else "") + (f", heads {heads_by_block}" if args.readout == "heads" else "") + ")", "",
          f"Tracker error bar inherited by this gate (probe C6, CoTracker vs Stage-A static tracks): {json.dumps(c6)}. EPE below the median gap "
          f"({c6_floor}) is reported as 'at the tracker floor'; p90 is reported, never gated.", "",
          "Trivial baselines (median EPE of predicting zero flow / the mean flow), per clip: " +
          "; ".join(f"{c[-9:]}: {next((x['baseline_epe_zero_flow'] for x in rows if x['clip'] == c), None)} / {next((x['baseline_epe_mean_flow'] for x in rows if x['clip'] == c), None)}" for c in clips_md), "",
          f"Gate G1: median EPE <= {g1_epe} patch and Spearman >= {g1_rho} on covered patches. Report only: John picks block l and the sigma window.", "",
          "| clip | sigma | " + " | ".join(f"block {b}: EPE all (≥3 tracks) / rho" for b in blocks_md) + " |", "|---|---|" + "---|" * len(blocks_md)]
    for clip in clips_md:
        for s in sigmas_md:
            cells = []
            for b in blocks_md:
                r = next((x for x in rows if x["clip"] == clip and x["sigma"] == s and x["block"] == b), None)
                cells.append("-" if r is None or r["epe_patch_median"] is None else
                             f"{r['epe_patch_median']:.2f}{'*' if r['at_tracker_floor'] else ''} ({'-' if r['epe_patch_median_ge3_tracks'] is None else format(r['epe_patch_median_ge3_tracks'], '.2f')}) / {r['spearman_median']:.2f} "
                             f"{'✓' if r['G1'] == 'pass' else ('E' if r['G1_epe'] else '') + ('S' if r['G1_spearman'] else '') or '✗'} [{r['gated_frac_last']:.2f}]")
            md.append(f"| {clip[-9:]} | {s} | " + " | ".join(cells) + " |")
    md += ["", "Cell: EPE all patches (EPE on patches with ≥ 3 tracks) / Spearman, then ✓ both gates, E = EPE only, S = Spearman only, ✗ neither; "
           "[gated share of the lattice at the last latent frame]; * = EPE at the tracker floor.",
           "", "✓ = both G1 thresholds met at that (block, sigma). The guidance window T_g is the sigma range where the chosen block passes; the readout must pass on the "
           "accepted clips at BOTH 81 and 33 frames before the short pairs are generated (contract 4a)."]
    if args.diagnose:
        md += ["", "## Diagnostics at the mid latent frame (what a fail means)", "",
               "| clip | sigma | block | self-pair: argmax-self frac / |F_hat| | cross: hit mass at GT source (r=2) / entropy bits (uniform) / centre-pull corr / |F_hat| | best 3 heads by EPE |",
               "|---|---|---|---|---|---|"]
        for r in rows:
            d_ = r.get("diagnostics_mid_frame")
            if not d_:
                continue
            cp, sp = d_["cross_pair"], d_["self_pair"]
            ph = sorted(((v, i) for i, v in enumerate(d_["per_head_epe_cross"]) if v is not None))[:3]
            md.append(f"| {r['clip'][-9:]} | {r['sigma']} | {r['block']} | {sp['argmax_self_frac']:.2f} / {sp['fhat_median_magnitude']:.2f} | "
                      f"{cp['hit_mass_median']:.3f} / {cp['entropy_bits_mean']:.1f} ({cp['uniform_bits']:.1f}) / {cp['centre_pull_corr']:.2f} / {cp['fhat_median_magnitude']:.2f} | "
                      + ", ".join(f"h{i}: {v:.2f}" for v, i in ph) + " |")
        md += ["", "Readings fixed in advance: self-pair argmax-self near 1 and |F_hat| near 0 -> token order and locality are right and the cross-pair failure is the model's attention, "
               "not the instrument; low hit mass + entropy near uniform + centre-pull corr near 1 -> diffuse attention (a readout-form question: sharpening or a head subset, "
               "with a G1 report per prohibition 7); a head whose EPE is far below the mean -> the head-subset case."]
    (rep / f"readout{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
