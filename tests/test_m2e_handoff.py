"""m2_e handoff check (reviewer 2026-09-28: Module 2's build closes when the handoff is written AND checked against the contract).
CPU only, reads measurements/m2e/handoff_*.json. Every contract §1 field must be present per pair, in the contract's frame/gauge
labels; gaussians_available false on Body pairs; d_used present and used for contact; 284 absent (refused). Records the temporal
envelope (n_frames_after_t_a vs Wan's 81) per pair, which is Module 3's item, not a gate here.

Run:  python tests/test_m2e_handoff.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPORT = ["# m2_e handoff check against docs/contract_m2e_to_m3.md §1", ""]
FIELDS = ["disposition", "label", "anchor_image", "K_1024", "K_a", "t_a", "n_frames_after_t_a", "g_tar_mesh", "g_tar_gauss", "gaussians_available",
          "pointmap_source", "placement_variant", "A_deg", "A_instrument", "A_already_applied_in_g_tar", "alpha", "b", "alpha_ci_block32", "gauge_k",
          "placement_C1", "n_plane", "d_plane", "d_used", "d_src", "frame", "gauge", "R2_iou", "gtar_shape_yaw_deg", "upstream_hashes", "hard_stop"]
files = sorted(glob.glob("measurements/m2e/handoff_*.json"))
assert len(files) == 6, files
assert not any("284" in f for f in files), "284 was refused and must have no handoff"
rows = []
for f in files:
    h = json.loads(Path(f).read_text(encoding="utf-8"))
    missing = [k for k in FIELDS if k not in h]
    assert not missing, (f, missing)
    assert h["schema"] == "m2e_handoff_v1" and h["frame"] == "camera_t_a" and h["gauge"].startswith("source"), f
    assert h["hard_stop"]["verdict"] == "pass" and h["A_already_applied_in_g_tar"] is True
    assert h["disposition"] in ("accepted", "labelled") and (h["label"] is None) == (h["disposition"] == "accepted")
    K = np.array(h["K_1024"]); assert K.shape == (3, 3) and 0.5 < K[0, 0] / K[1, 1] < 0.6, "anchor pixels must carry the squash"
    assert h["gaussians_available"] == (h["category"] == "vehicle") and (h["g_tar_gauss"] is None) == (h["category"] == "human")
    assert h["pointmap_source"] in ("alpha_scaled_zeta_hat", "n/a (Body estimates its own depth)"), h["pointmap_source"]
    assert h["A_instrument"] == ("keypoints" if h["category"] == "human" else "silhouette_scan")
    assert np.isfinite(h["d_used"]) and np.isfinite(h["d_plane"]) and h["alpha"] > 0 and h["alpha_ci_block32"][0] > 0
    assert (h["gauge_k"] is None) == (h["category"] == "vehicle"), "gauge_k only on Body / fallback pairs"
    assert h["n_frames_after_t_a"] == h["n_frames_source"] - h["t_a"] and h["generated_frame0_is_source_frame"] == h["t_a"]
    for k in ("X", "s", "R_A", "z"):
        assert k in h["placement_C1"]
    assert abs(np.linalg.det(np.array(h["placement_C1"]["R_A"])) - 1) < 1e-6
    rows.append((h["clip"][-8:], h["category"], h["disposition"], h["t_a"], h["n_frames_after_t_a"], h["gaussians_available"], round(h["d_used"], 3), round(h["d_plane"], 3), round(h["A_deg"], 2), round(h["R2_iou"], 3)))
REPORT += ["| clip | cat | disp. | t_a | frames after t_a (Wan needs 81) | gaussians | d_used | d_plane | A (deg) | R2 IoU |", "|---|---|---|---|---|---|---|---|---|---|"]
REPORT += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]}{' **short**' if r[4] < 81 else ''} | {r[5]} | {r[6]} | {r[7]} | {r[8]:+.2f} | {r[9]} |" for r in rows]
REPORT += ["", "All 30 contract §1 fields present on all six pairs; frame camera_t_a and source gauge labelled; A marked as already applied in G_tar; "
           "d_used carried beside d_plane; Gaussians on vehicles only; 284 absent (refused). Temporal envelope: 404 and 265.2.003 have 34 source frames after t_a "
           "against Wan's 81 - Module 3's policy item (contract §4), recorded here, not gated.", "", "handoff check passed"]
Path("docs/report_m2e_handoff_check.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
print("\n".join(REPORT))
