"""Lexicographic anchor selection (SKILL_m2_d v1.1 §5; thresholds per reviewer 2026-09-18). Pure, CPU.

Phase 1 — a guard against gross failure, NOT a discriminator (first run: 20/20 passed every condition, so
none was exercised; the thresholds are derived from what a failure would BE, not from where the sample lies):
    E_loc_rel = |centroid(generated) - p_ta| / sqrt(A_obj_rendered)   <= m2d_e_loc_rel_max   (half the object's own scale)
    E_area    = |log(A_generated / A_obj_rendered)|                   <= m2d_e_area_max      (twice / half the size asked for)
    heading   = the HUMAN's label == the control's class              (m2d_heading: human — no validated instrument)
    eps_shape                                                        <= m2d_eps_shape_max   (clip-level, from C1; null = ungated)
  The spec's E_loc divides by the image diagonal, which makes 16 px on a 400 px car equal 16 px on a 30 px man;
  both are recorded, the object-scale one is gated.
Phase 2 — quality alone, MANUAL (spec §4.2): the human ranks the survivors; ties break to the lowest seed.

A null threshold is 'ungated', never a pass by default-value. Thresholds come from the lock; none lives here.
"""
from __future__ import annotations

import math

HEADING_LABELS = ("toward", "away", "lateral_left_to_right", "lateral_right_to_left", "unclear")


class AwaitingHuman(RuntimeError):
    """Selection needs a human input that is not there yet (heading labels or Phase-2 ranks)."""


class HashMismatch(RuntimeError):
    """An upstream artifact changed after the candidates were generated / measured."""


def e_loc_rel(E_loc_diag: float, A_obj_frac: float, H: int, W: int) -> float:
    """Re-normalise the recorded E_loc (|d| / image diagonal) by the object's own scale sqrt(A_obj) in pixels."""
    assert A_obj_frac > 0
    return float(E_loc_diag * math.hypot(H, W) / math.sqrt(A_obj_frac * H * W))


def _gate(value, threshold, passed_if):
    if threshold is None:
        return {"value": value, "threshold": None, "outcome": "ungated"}
    return {"value": value, "threshold": threshold, "outcome": "pass" if passed_if(threshold) else "fail"}


def check_chain(recorded: dict, current: dict):
    """Every hash the generation recorded must equal the artifact on disk now (test 7): hard exit otherwise."""
    for k, v in recorded.items():
        if k not in current:
            raise HashMismatch(f"upstream hash {k!r} recorded at generation but not re-derivable now")
        if current[k] != v:
            raise HashMismatch(f"upstream {k!r} changed since generation: recorded {str(v)[:16]}, now {str(current[k])[:16]} — regenerate (steps 5-6)")


def phase1(candidates: list, labels: dict, control_class: str, eps_shape, th: dict, H: int, W: int) -> list:
    """candidates: fidelity.json rows. labels: {seed: human heading label}. Returns one record per candidate."""
    mode = th.get("m2d_heading")
    assert mode == "human", f"m2d_heading is {mode!r}: the only heading instrument that exists is the human (lock: m2d_heading: human)"
    out = []
    for c in sorted(candidates, key=lambda c_: c_["seed"]):
        k = c["seed"]
        if k not in labels or not labels[k]:
            raise AwaitingHuman(f"no heading label for seed {k}: fill heading_labels.csv")
        assert labels[k] in HEADING_LABELS, f"seed {k}: label {labels[k]!r} is not in {HEADING_LABELS}"
        g = {"g_object_found": {"value": bool(c["object_found"]), "outcome": "pass" if c["object_found"] else "fail"},
             "g_heading": {"value": labels[k], "control_class": control_class, "instrument": "human",
                           "outcome": "pass" if labels[k] == control_class else "fail"},
             "g_eps_shape": _gate(eps_shape, th.get("m2d_eps_shape_max"), lambda t: eps_shape is not None and eps_shape <= t)}
        rec = {"seed": k, "heading_class": labels[k], "obj_fidelity": c["obj_fidelity"], "bg_fidelity": c["bg_fidelity"]}
        if c["object_found"]:
            rel = e_loc_rel(c["E_loc"], c["A_obj_rendered_frac"], H, W)
            rec.update({"E_loc": c["E_loc"], "E_loc_rel": rel, "E_area": c["E_area"], "log_area_ratio": c["log_area_ratio"]})
            g["g_e_loc_rel"] = _gate(rel, th.get("m2d_e_loc_rel_max"), lambda t: rel <= t)
            g["g_e_area"] = _gate(c["E_area"], th.get("m2d_e_area_max"), lambda t: c["E_area"] <= t)
        rec["gates"] = g
        rec["phase1_pass"] = all(v["outcome"] != "fail" for v in g.values())
        out.append(rec)
    return out


def phase2(records: list, ranks: dict):
    """ranks: {seed: human rank, 1 = best} over the SURVIVORS. Winner = min (rank, seed): order-independent (test 5)."""
    surv = [r for r in records if r["phase1_pass"]]
    if not surv:
        return None
    missing = [r["seed"] for r in surv if ranks.get(r["seed"]) is None]
    if missing:
        raise AwaitingHuman(f"no Phase-2 rank for surviving seeds {missing}: fill phase2_ranks.csv (1 = best; ties allowed)")
    for r in records:
        if r["phase1_pass"]:
            r["phase2_rank"] = int(ranks[r["seed"]])
    return min(surv, key=lambda r: (r["phase2_rank"], r["seed"]))["seed"]


def auto_ranks(records: list) -> dict:
    """Automatic Phase 2 (added 2026-10-05, owner's request; additive, the manual path is unchanged). Rule: among the Phase-1
    survivors, rank by seed order, so the winner is the LOWEST surviving seed - the rule the manual path already uses to break
    ties. It uses no image-quality judgement: it is a deterministic default, not a quality instrument, and is recorded as such."""
    surv = sorted(r["seed"] for r in records if r["phase1_pass"])
    return {k: i + 1 for i, k in enumerate(surv)}


def exercised(all_records: list) -> dict:
    """Which Phase-1 conditions ever FIRED across the set. A gate that never fires is an assumption, not a validated threshold."""
    names = sorted({g for r in all_records for g in r["gates"]})
    return {n: {"n_fail": sum(r["gates"].get(n, {}).get("outcome") == "fail" for r in all_records),
                "n_ungated": sum(r["gates"].get(n, {}).get("outcome") == "ungated" for r in all_records),
                "exercised": any(r["gates"].get(n, {}).get("outcome") == "fail" for r in all_records)} for n in names}


def select(candidates: list, labels: dict, ranks: dict, control_class: str, eps_shape, th: dict, H: int, W: int,
           n_generated: int, budget_total: int) -> dict:
    recs = phase1(candidates, labels, control_class, eps_shape, th, H, W)
    n_surv = sum(r["phase1_pass"] for r in recs)
    if n_surv == 0:
        outcome = "resample" if n_generated < budget_total else "refused"
        return {"selected_seed": None, "outcome": outcome, "n_survivors": 0, "candidates": recs,
                "gate_outcomes": {"g_phase1_survivors": {"value": 0, "outcome": "fail", "action": outcome}}}
    win = phase2(recs, ranks)
    return {"selected_seed": win, "outcome": "selected", "n_survivors": n_surv, "candidates": recs,
            "gate_outcomes": {"g_phase1_survivors": {"value": n_surv, "outcome": "pass"}}}


QC_SCHEMA = "m2d_qc_v1"


def write_selection(out_dir, qc: dict):
    """qc_m2d.json ALWAYS; anchor.png only when a seed was selected (test 6: a refusal emits qc and NO anchor —
    a stale anchor.png from an earlier selection is removed, never left to be read as current)."""
    import json
    import shutil
    from pathlib import Path
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    anchor = out / "anchor.png"
    if anchor.exists():
        anchor.unlink()
    if qc.get("selected_seed") is not None:
        src = out / "candidates" / f"seed_{qc['selected_seed']}.png"
        assert src.exists(), f"{src} missing"
        shutil.copyfile(src, anchor)                       # candidates are all retained; the anchor is a copy
    (out / "qc_m2d.json").write_text(json.dumps({"schema": QC_SCHEMA, **qc}, indent=1), encoding="utf-8")
    return anchor if anchor.exists() else None
