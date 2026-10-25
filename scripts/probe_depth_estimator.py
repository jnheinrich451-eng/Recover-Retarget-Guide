"""m2_d build step 0 — the depth estimator's own SIGN, measured per frame (SKILL v1.1 §2, review A2).

`rho = corr(zeta_hat(generated), zeta_ctrl)` decides nothing unless zeta_hat is known to increase with
nearness, and estimators differ (relative inverse depth vs metric depth). This probe runs the chosen
estimator on the real source frames of the nine B.5 verdict clips, where Stage A's zeta = 1/D is known,
and records Spearman rho PER FRAME — a stable convention is positive (or negative) on all nine; a flip
on any frame is a different failure from a weak correlation and must be visible.

  python scripts/probe_depth_estimator.py            # base env; GPU optional (small model)

Writes <drive>/probe/m2d/estimator_sign/{report_estimator_sign.md, estimator_sign.json}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2d import instrument                                       # noqa: E402


def load_estimator(repo):
    import torch
    import transformers
    from transformers import pipeline
    pipe = pipeline("depth-estimation", model=repo, device=0 if torch.cuda.is_available() else -1)
    return pipe, transformers.__version__


def estimate(pipe, rgb_uint8, H, W):
    """predicted_depth as a float array resized to H x W (bilinear). Whatever it means — that is what
    this probe measures."""
    import cv2
    from PIL import Image
    out = pipe(Image.fromarray(rgb_uint8))
    pd = out["predicted_depth"]
    pd = pd.detach().float().cpu().numpy() if hasattr(pd, "detach") else np.asarray(pd, np.float32)
    pd = np.squeeze(pd)
    return cv2.resize(pd.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float64)


def main():
    import cv2
    import yaml
    from src.preprocess.bundle import read_bundle
    from src.preprocess.config import load_config
    from src.preprocess.stage_c import finalize
    cfg_a = load_config("configs/stage_a.yaml"); drive = str(cfg_a["paths"]["drive_root"])
    m2d = yaml.safe_load(Path("configs/m2d.yaml").read_text(encoding="utf-8"))
    repo = m2d["m2d"]["estimator"]["hf_repo"]
    out = Path(str(m2d["paths"]["probe_root"]).replace("${paths.drive_root}", drive)) / "estimator_sign"
    out.mkdir(parents=True, exist_ok=True)
    clips = [c for t in yaml.safe_load(Path("configs/b5_verdict_clips.yaml").read_text(encoding="utf-8")).values() for c in t]
    pipe, tv = load_estimator(repo)
    lines = [f"# Depth estimator sign probe — {repo} (transformers {tv})", "",
             "Spearman rho between the estimator's `predicted_depth` and Stage A's zeta = 1/D on the real source frame at t_c, "
             "over valid Stage A pixels. rho > 0 means the output increases with NEARNESS (disparity-like); rho < 0 means it is depth-like.", "",
             "| clip | t_c | valid px | rho (all valid) | rho (background only) | sign |", "|---|---|---|---|---|---|"]
    rows = []
    for clip in clips:
        try:
            b = read_bundle(cfg_a, clip)
            M, present, t_c = finalize.read_mask(Path(cfg_a["paths"]["stage_c_cache"]) / clip)
            cap = cv2.VideoCapture(b["provenance"]["clip_path"]); cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_c)); ok, fr = cap.read(); cap.release()
            assert ok, "frame not readable"
            D = b["D"][int(t_c)].astype(np.float64)
            valid = np.isfinite(D) & (D > 0)
            zeta = np.where(valid, 1.0 / np.where(valid, D, 1.0), np.nan)
            est = estimate(pipe, np.ascontiguousarray(fr[..., ::-1]), *D.shape)
            rho = instrument.spearman(est, zeta, mask=valid)
            rho_bg = instrument.spearman(est, zeta, mask=valid & ~M[int(t_c)].astype(bool))
            rows.append({"clip": clip, "t_c": int(t_c), "n_valid": int(valid.sum()), "rho": rho, "rho_background": rho_bg})
            lines.append(f"| {clip} | {int(t_c)} | {int(valid.sum())} | {rho:+.3f} | {rho_bg:+.3f} | {'+' if rho > 0 else '-'} |")
        except Exception as e:                                        # noqa: BLE001
            lines.append(f"| {clip} | – | – | error: {type(e).__name__}: {str(e)[:70]} | | |")
    signs = {np.sign(r["rho"]) for r in rows}
    weak = [r["clip"] for r in rows if abs(r["rho"]) < 0.5]
    sign = int(signs.pop()) if len(signs) == 1 and rows else None
    lines += ["", f"Frames measured: {len(rows)}/{len(clips)}. Sign unanimous: {sign is not None} "
              f"({'+1: predicted_depth increases with nearness' if sign == 1 else '-1: predicted_depth is depth-like' if sign == -1 else 'MIXED — do not use this estimator'}). "
              f"|rho| < 0.5 on: {weak or 'none'}.",
              "The polarity probe's secondary readout multiplies the estimator output by this sign; with a mixed sign it refuses to run."]
    (out / "estimator_sign.json").write_text(json.dumps({"estimator": repo, "transformers": tv, "sign": sign, "rows": rows}, indent=1), encoding="utf-8")
    (out / "report_estimator_sign.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
