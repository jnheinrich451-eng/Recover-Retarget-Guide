"""Module 2 -> Module 3 handoff reader and the JSON-level integration gates (brief v2.5 §2.1, §2.7, §9 S-int).

Consumes the v1 contract (docs/contract_m2e_to_m3.md), never Module 2's implementation. A gate that fails is a recorded
refusal or a hard error as the brief states; nothing here adjusts a value. Lock values come from the `m3:` section of
configs/calibration.lock.yaml, which John owns; a missing key is an error, never a default.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import camera

SCHEMA = "m2e_handoff_v1"
REQUIRED_LOCK = ("depth_model", "lattice", "int", "lift", "anchors", "rasterizer", "trajectory", "depth")


class HandoffRefused(Exception):
    """A recorded refusal (CLAUDE.md §3a rule 18): the pair is skipped with this reason, the run goes on."""


def load_m3_lock(lock_path="configs/calibration.lock.yaml") -> dict:
    import yaml
    lock = yaml.safe_load(Path(lock_path).read_text(encoding="utf-8")) or {}
    assert "m3" in lock and isinstance(lock["m3"], dict), \
        f"{lock_path} has no `m3:` section (brief §8). John adds it; Claude Code never edits the lock file."
    missing = [k for k in REQUIRED_LOCK if k not in lock["m3"]]
    assert not missing, f"lock `m3:` section lacks {missing} (brief §8)"
    return lock["m3"]


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gate_handoff(h: dict, sha256=None, frozen=None, experiment: bool = False) -> dict:
    """g_handoff: schema, disposition, declared hard stop; and, when the closure manifest is given, the frozen hash. On the experiment
    path (plan v1) the disposition is 'experiment' (admitted by Module 2's gates, verdict = gate outcomes) and the frozen set is
    configs/experiment_handoffs.yaml, not the v1 closure manifest."""
    assert h.get("schema") == SCHEMA, f"schema {h.get('schema')!r} != {SCHEMA}"
    allowed = ("accepted", "labelled", "experiment") if experiment else ("accepted", "labelled")
    if h.get("disposition") not in allowed:
        raise HandoffRefused(f"disposition {h.get('disposition')!r}")
    assert (h.get("label") is None) == (h["disposition"] in ("accepted", "experiment")), "label / disposition inconsistent"
    assert h.get("hard_stop", {}).get("verdict") == "pass", "handoff written without a declared hard stop"
    out = {"disposition": h["disposition"], "label": h.get("label"), "hash_checked": False, "verdict_source": (h.get("hard_stop") or {}).get("source")}
    if frozen is not None:
        key = f"{h['clip']}/{h['category']}/handoff.json"
        assert key in frozen, f"{key} is not in the closure manifest"
        assert sha256 == frozen[key], f"{key}: handoff differs from the frozen v1 file (hard error)"
        out["hash_checked"] = True
    return out


def gate_frame0(h: dict) -> dict:
    """g_frame0 (hard error): world = camera at t_a, source gauge, A already inside G_tar, generated frame 0 = source t_a."""
    assert h["frame"] == "camera_t_a", h["frame"]
    assert str(h["gauge"]).startswith("source"), h["gauge"]
    assert h["A_already_applied_in_g_tar"] is True, "Module 3 never applies A"
    assert h["generated_frame0_is_source_frame"] == h["t_a"]
    assert h["n_frames_after_t_a"] == h["n_frames_source"] - h["t_a"]
    K = np.asarray(h["K_1024"], np.float64); Ka = np.asarray(h["K_a"], np.float64)
    assert K.shape == (3, 3) and np.allclose(K, np.diag([4.0, 4.0, 1.0]) @ Ka, rtol=1e-6), "K_1024 != diag(4,4,1) K_256"
    return {"t_a": int(h["t_a"]), "fx_over_fy_1024": float(K[0, 0] / K[1, 1])}


def gate_envelope(h: dict, min_frames) -> dict:
    """g_envelope: F_pair, T_l recorded; refuse only below lock.int.min_frames when that is set (contract §4a)."""
    F_pair, T_l = camera.frames_for_pair(h["n_frames_after_t_a"])
    if min_frames is not None and F_pair < int(min_frames):
        raise HandoffRefused(f"F_pair {F_pair} < min_frames {min_frames}")
    return {"F_pair": F_pair, "T_l": T_l, "lost_frames_before_t_a": int(h["t_a"]), "min_frames": min_frames}


def gate_mode(mode) -> dict:
    """g_mode: s = 1 holds only for depth_preserving placement; other modes refuse until D-M3-03."""
    if mode != "depth_preserving":
        raise HandoffRefused(f"placement mode {mode!r}: scale anchor undecided (D-M3-03)")
    return {"mode": mode, "s": 1.0}


def read_pair(drive_root, clip: str, category: str, lock_m3: dict, frozen=None, experiment: bool = False) -> dict:
    """Everything S-int needs for one pair that is a plain read: handoff + the gates above, prompt, mode, K_wan, K^p and
    the paths of the arrays (loaded by the caller)."""
    cache = Path(drive_root) / "cache"
    hp = cache / "m2_e" / clip / category / "handoff.json"
    if not hp.exists():
        raise HandoffRefused("no handoff.json (pair refused or never built in Module 2)")
    h = json.loads(hp.read_text(encoding="utf-8"))
    gates = {"g_handoff": gate_handoff(h, sha256_file(hp), frozen, experiment), "g_frame0": gate_frame0(h),
             "g_envelope": gate_envelope(h, lock_m3["int"].get("min_frames"))}
    pl = json.loads((cache / "m2_c" / clip / category / "placement.json").read_text(encoding="utf-8"))
    assert "mode" in pl, f"placement.json has no `mode` (keys: {sorted(pl)})"
    gates["g_mode"] = gate_mode(pl["mode"])
    qd = json.loads((cache / "m2_d" / clip / category / "qc_m2d.json").read_text(encoding="utf-8"))
    lat = lock_m3["lattice"]
    px = int(lat["px_per_patch"])
    K_wan = camera.wan_intrinsics(np.asarray(h["K_1024"]), int(lat["Wp"]) * px, int(lat["Hp"]) * px, 1024)
    d = cache / "m2_e" / clip / category
    return {"handoff": h, "gates": gates, "prompt": qd["prompt"], "K_wan": K_wan, "Kp": camera.lattice_intrinsics(K_wan, px),
            "paths": {"anchor": cache / "m2_d" / clip / category / "anchor.png", "anchor_masks": d / "anchor_masks.npz",
                      "g_tar_mesh": d / "g_tar_mesh.npz", "control": cache / "m2_c" / clip / category / "control.npz"}}
