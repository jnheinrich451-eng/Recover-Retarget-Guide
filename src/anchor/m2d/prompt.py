"""Anchor prompt (SKILL_m2_d v1.1 §3; spec v3 Table 1, Stage VI).

y_anchor keeps the target's object, action and scene and adds coarse camera-relative language from the
deterministic geometry-to-language dictionary. Principle 1: the heading class is read in the
camera-t_a frame — a heading left in the frame-0 gauge selects the wrong row SILENTLY, since every row
yields a grammatical prompt — so the frame is READ FROM THE stage0 STRUCT (m2a.read_stage0 returns it with the vectors) and asserted; it is
not an argument a caller can mistype. No computed value is ever written into a prompt: the DICTIONARY
phrases are digit-free by assert (the user's own text, 'a 1967 Corvette', is not the risk surface).

Azimuth convention (OpenCV camera axes, as in the Body facing probe): 0 deg = +x image-right,
+90 deg = +z away from the camera, -90 deg = toward the camera, 180 deg = image-left. The heading used is
the one C1 uses: d_src projected into the contact plane (spec §6.2 amended).
"""
from __future__ import annotations

import re

import numpy as np

FRAME = "camera_t_a"
HEADING_CLASSES = ("toward", "away", "lateral_left_to_right", "lateral_right_to_left")


class WrongFrame(ValueError):
    """The heading was not supplied in the camera-t_a frame (Principle 1)."""


def heading_azimuth_deg(d_src, n_plane) -> float:
    d = np.asarray(d_src, np.float64); n = np.asarray(n_plane, np.float64)
    assert np.linalg.norm(d) > 0, "zero heading: m2_a refused this clip"
    if np.linalg.norm(n) > 0:
        n = n / np.linalg.norm(n)
        d = d - (n @ d) * n                                  # the in-plane heading C1 aligns the proxy to
    assert np.linalg.norm(d[[0, 2]]) > 1e-9, "heading has no horizontal component"
    return float(np.degrees(np.arctan2(d[2], d[0])))


def heading_class(stage0: dict, half_width_deg: float) -> tuple[str, float]:
    """Exactly one class for every heading (test 3): away / toward within half_width of +-90 deg,
    otherwise lateral, split by the sign of the x component. Boundaries belong to away / toward.
    stage0 is m2a.read_stage0's output: the frame label travels with the vectors."""
    frame = stage0.get("frame")
    if frame != FRAME:
        raise WrongFrame(f"stage0 carries frame {frame!r}; the dictionary is camera-relative at t_a and reads {FRAME!r} only "
                         "(pass m2a.read_stage0's dict, not loose vectors)")
    assert 0 < half_width_deg < 90
    az = heading_azimuth_deg(stage0["d_src"], stage0["n_plane"])
    if abs(az - 90.0) <= half_width_deg:
        return "away", az
    if abs(az + 90.0) <= half_width_deg:
        return "toward", az
    return ("lateral_left_to_right" if abs(az) < 90.0 else "lateral_right_to_left"), az


def shot_class(obj_area_frac: float, small_max: float, medium_max: float) -> str:
    assert 0 <= obj_area_frac <= 1 and 0 < small_max < medium_max
    if obj_area_frac < small_max:
        return "small_area"
    return "medium_area" if obj_area_frac < medium_max else "large_area"


def heights(placement: dict, V_aligned, n_plane) -> tuple[float, float]:
    """(camera height, object height), both DERIVED. The camera is the origin of the camera-t_a frame, so its
    height above the plane the object stands on is |d_used| (n unit). The object's height is
    s x (extent of the posed unit-scale proxy along n); V_aligned = proxy.aligned_vertices(mesh, R_A)."""
    n = np.asarray(n_plane, np.float64); n = n / np.linalg.norm(n)
    h = np.asarray(V_aligned, np.float64) @ n
    obj = float(placement["s"]) * float(h.max() - h.min())
    cam = abs(float(placement["d_used"]))
    assert np.isfinite([obj, cam]).all() and obj > 0 and cam > 0, (obj, cam)
    return cam, obj


def build_anchor_prompt(target_object: str, action: str, scene: str, stage0: dict, placement: dict, V_aligned,
                        obj_area_frac: float, dictionary: dict) -> dict:
    """Returns the prompt and everything SKILL §3 asks to be recorded per clip."""
    ph = dictionary["phrases"]
    need = set(HEADING_CLASSES) | {"small_area", "medium_area", "large_area", "camera_low"}
    assert need <= set(ph), f"dictionary coverage: missing {sorted(need - set(ph))}"
    for k in need:                                          # computed geometry must never reach the prompt as a number
        assert not re.search(r"\d", ph[k]), f"numeric value in dictionary phrase {k!r}: {ph[k]!r}"
    hc, az = heading_class(stage0, float(dictionary["toward_away_half_width_deg"]))
    sc = shot_class(obj_area_frac, float(dictionary["small_area_max_frac"]), float(dictionary["medium_area_max_frac"]))
    cam_h, obj_h = heights(placement, V_aligned, stage0["n_plane"])
    low = bool(cam_h / obj_h < float(dictionary["camera_low_max_height_ratio"]))
    edges = (float(dictionary["small_area_max_frac"]), float(dictionary["medium_area_max_frac"]))
    margin = min(abs(obj_area_frac - e) / e for e in edges)                    # relative distance to the nearest band edge
    hwid = float(dictionary["toward_away_half_width_deg"])
    h_margin = min(abs(abs(abs(az) - 90.0) - hwid), 360.0)                     # degrees to the nearest heading-class boundary
    rows = [hc, sc] + (["camera_low"] if low else [])
    parts = [f"{target_object.strip()} {action.strip()}".strip(), scene.strip()] + [ph[r] for r in rows]
    return {"prompt": ", ".join(p for p in parts if p), "dictionary_rows": rows, "heading_class": hc,
            "heading_azimuth_deg": az, "heading_frame": stage0["frame"], "heading_margin_deg": float(h_margin), "shot_class": sc,
            "shot_band_margin_rel": float(margin), "shot_class_near_band_edge": bool(margin < float(dictionary["band_edge_flag_rel"])), "obj_area_frac": float(obj_area_frac),
            "camera_low": low, "camera_height": cam_h, "object_height": obj_h, "camera_height_over_object_height": cam_h / obj_h}
