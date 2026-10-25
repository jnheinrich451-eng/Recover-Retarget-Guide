"""Stage C CPU tests: prompt geometry, referee, presence, finalize round-trip.
No SAM2 anywhere (CLAUDE.md §3.2 — the adapter is exercised on Colab only).

Run:  python tests/test_stage_c.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml                                            # noqa: E402
from src.preprocess.stage_c import finalize, presence, prompts  # noqa: E402
from src.preprocess.stage_c.mask import referee        # noqa: E402

CFG_C = yaml.safe_load(Path("configs/stage_c.yaml").read_text(encoding="utf-8"))


def fake_clip(branch=True, N=200, F=81, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.random((N, F)) > 0.05
    P = np.zeros((N, F, 2), np.float32)
    P[:50] = rng.uniform(100, 150, (50, 1, 2))          # subject cluster
    P[50:] = rng.uniform(0, 256, (N - 50, 1, 2))        # background
    b = {"Q": Q, "P": P}
    sb = {"y_dyn": np.zeros(N, bool), "m": np.full(N, np.nan)}
    sb["y_dyn"][:60] = True
    sb["m"][:60] = 3.0
    sb["m"][:50] += 2.0                                  # subject high-m
    if branch:
        sb["fc_blob_idx"] = np.arange(0, 55)             # subject + 5 halo
        qc_b = {"follow_cam": {"branch_active": True, "fired": True,
                               "seed_frame": 40,
                               "seed_bbox": [90, 90, 160, 160]}}
    else:
        sb["o_src_members"] = np.arange(0, 60)           # subject + strays
        qc_b = {"follow_cam": {"fired": False}}
    return b, sb, qc_b


def test_prompts_branch():
    b, sb, qc_b = fake_clip(branch=True)
    t_c, core, variants, prov = prompts.build_variants(b, sb, qc_b, CFG_C)
    assert t_c == 40 and prov["channel"] == "branch"
    pts, lab, box = variants["V2_negs"]
    pos, neg = pts[lab == 1], pts[lab == 0]
    assert (np.abs(pos[:, 0] - 125) < 40).mean() > 0.8, "positives off subject"
    assert len(neg) >= 4, "negative pool starved (radius inflation?)"
    subj = sb["fc_blob_idx"]
    subj_v = subj[b["Q"][subj, t_c]]
    c_ext = b["P"][subj_v, t_c].mean(axis=0)
    d = np.linalg.norm(neg - c_ext, axis=1)
    assert (d > CFG_C["prompts"]["neg_margin"] * prov["r_s"] * 0.99).all(), \
        "negative inside the margin"
    print(f"PASS prompts branch (t_c {t_c}, {len(pos)}+/{len(neg)}-, "
          f"r_s {prov['r_s']})")


def test_prompts_x0_fallback():
    b, sb, qc_b = fake_clip(branch=False)
    t_c, core, variants, prov = prompts.build_variants(b, sb, qc_b, CFG_C)
    assert prov["channel"] == "x0" and prov["anchor_n"] >= 4
    core_set = set(core.tolist())
    assert len(core_set & set(range(50))) / len(core_set) > 0.9, \
        "core contaminated by strays"
    print(f"PASS prompts x0 fallback (anchor {prov['anchor_n']}, core pure)")


def test_referee_band():
    stats = {"big": {"med_iou": 0.5, "area_hull_ratio": 8.0, "area_med": 0.3},
             "good": {"med_iou": 0.3, "area_hull_ratio": 1.2, "area_med": 0.06}}
    th = {"c_size_lo": 0.5, "c_size_hi": 3.0, "c_abs_area_max": 0.2}
    best, inband = referee(stats, th)
    assert best == "good" and inband, \
        "area band must screen before IoU (Task-1 lesson)"
    # C-ABS (2026-08-21): whole-scene mask passes the RELATIVE band when the
    # corrupted core hull is equally huge (67.1.004: area 0.645, ratio 0.87)
    stats2 = {"scene": {"med_iou": 0.54, "area_hull_ratio": 0.87,
                        "area_med": 0.645},
              "obj": {"med_iou": 0.2, "area_hull_ratio": 1.1,
                      "area_med": 0.05}}
    best2, inband2 = referee(stats2, th)
    assert best2 == "obj" and inband2, \
        "absolute ceiling must screen the whole-scene mask"
    print("PASS referee (relative band + C-ABS ceiling)")


def test_presence_and_finalize(tmp="C:/Users/12856/AppData/Local/Temp/stage_c_test"):
    F, Hn = 20, 64
    core = np.arange(5)
    Q = np.zeros((10, F), bool)
    Q[:, 8:] = True                                      # subject enters at 8
    masks = {t: np.zeros((Hn, Hn), bool) for t in range(F)}
    for t in range(6, F):
        masks[t][10:20, 10:20] = True                    # incl. pre-entry 6,7
    pres, masks = presence.presence(masks, core, Q, F, n_pos_min=4)
    assert not pres[:8].any() and pres[8:].all()
    assert not masks[6].any(), "pre-entry mask must be zeroed"
    cfg = {"video": {"process_height": 32},
           "paths": {"stage_c_cache": tmp},
           "_meta": {"config_hash": "t"}}
    finalize.write_stage_c(cfg, "clip", masks, pres, 10,
                           {"channel": "x"}, {}, [], "V1_pts", "up")
    M, pres2, t_c = finalize.read_mask(Path(tmp) / "clip")
    assert M.shape == (F, 32, 32) and t_c == 10
    assert (pres2 == pres).all() and M[9].any() and not M[6].any()
    print("PASS presence + finalize round-trip (honest absence, packed M_src)")


if __name__ == "__main__":
    test_prompts_branch()
    test_prompts_x0_fallback()
    test_referee_band()
    test_presence_and_finalize()
    print("\nall Stage C CPU tests passed")
