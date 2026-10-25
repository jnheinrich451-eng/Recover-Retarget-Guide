"""m2_d runner — build steps 5 (anchor-candidate generation) 6 (fidelity measurement) and 7 (selection), SKILL_m2_d v1.1 §3-6.

  python scripts/run_m2d.py --clip 000000000284.0.006 --category vehicle --dry-run     # CPU: prompt + checks, no model
  python scripts/run_m2d.py --all                                                      # GPU: the admitted set, K' seeds each
  python scripts/run_m2d.py --all --measure                                            # GPU (small): step 6 on the retained candidates
  python scripts/run_m2d.py --all --select                                             # CPU: step 7 (needs heading_labels.csv, then phase2_ranks.csv)

Reads C3's control.npz (ONE image, emitted under the locked `m2d_polarity` — a control composed before the
lock is refused: re-run scripts/run_m2c_c3.py), C1's placement.json, m2_a's stage0 (the frame travels with
it) and the R1 proxy. Writes cache/m2_d/<clip>/<category>/candidates/seed_<k>.png (all K' retained) and
generation.json; a contact sheet under <drive>/viz/m2d_anchors/. Resumable: an existing seed is skipped
only if its recorded control checksum and prompt match.

The control must reach the model untouched: checksum + tensor deviation asserted AT EVERY CALL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor import m2a                                                 # noqa: E402
from src.anchor.m2b2 import cache                                          # noqa: E402
from src.anchor.m2b2.objects import load_m2b2_config                       # noqa: E402
from src.anchor.m2c import compose, placement, proxy                       # noqa: E402
from src.anchor.m2d import fidelity, instrument, prompt, select, stimuli   # noqa: E402

GENERATION_SCHEMA = "m2d_generation_v1"


def prepare(clip: str, category: str, ctx: dict) -> dict:
    """Everything a generation call needs, all CPU. Refuses rather than guesses."""
    cfg, drive, lock = ctx["cfg"], ctx["drive"], ctx["lock"]
    tkey = f"{clip}/{category}"
    if tkey in cfg["targets"]:
        tgt = dict(cfg["targets"][tkey]); tgt["target_source"] = "configs/m2d.yaml m2d.targets (the human's text)"
    else:
        exp = ctx.get("experiment_targets") or {}
        assert tkey in exp, f"no target text for {tkey}: not in configs/m2d.yaml m2d.targets (the human's text) and not in the experiment set (--experiment-set)"
        tgt = {k: exp[tkey]["target"][k] for k in ("object", "action", "scene")}; tgt["target_source"] = f"experiment set, fixed rule ({exp[tkey]['target'].get('scene_source')})"
    c_dir = ctx["cache_m2c"] / clip / category
    rec = placement.read_placement(c_dir)
    assert rec["admitted"], f"{tkey}: C1 placement not admitted"
    arrays, qc3 = compose.read_control(c_dir)
    assert arrays is not None, f"{tkey}: C3 emitted no control (qc_c3.json: {[k for k, v in qc3['gates'].items() if v.get('outcome') == 'fail']})"
    pol = lock.get("m2d_polarity")
    assert pol in ("near_bright", "near_dark"), "m2d_polarity is not locked; run the polarity probe first"
    assert qc3.get("polarity") == pol and "D_ctrl_inv" not in arrays, \
        f"{tkey}: control.npz was composed as {qc3.get('polarity')!r}, the lock says {pol!r} — re-run scripts/run_m2c_c3.py for this clip"
    st0 = m2a.read_stage0(ctx["m2a_out"] / clip)                           # carries frame = camera_t_a
    for k in ("d_src", "n_plane"):                                         # the placement was solved on THIS stage0, not a stale one
        assert np.allclose(np.asarray(rec["stage0"][k], float), st0[k].astype(float), atol=1e-6), \
            f"{tkey}: placement.json's {k} differs from m2_a's stage0 — re-place the clip"
    b2 = ctx["b2"]; variant = cache.route(category, b2)
    prox = cache.read_proxy(Path(b2["paths"]["cache_r1"]) / category / rec["proxy_cache_key"], variant, b2["m2b2"]["adapter_version"])
    mesh = proxy.ProxyMesh("proxy", prox["vertices"], prox["faces"], prox["vertices"].mean(0), prox["B_tar"])
    V_al = proxy.aligned_vertices(mesh, rec["placement"]["R_A"])
    pr = prompt.build_anchor_prompt(tgt["object"], tgt["action"], tgt["scene"], st0, rec["placement"], V_al,
                                    float(qc3["obj_area_frac"]), cfg["prompt_dictionary"])
    u8 = stimuli.to_control_uint8(arrays["D_ctrl"])
    return {"clip": clip, "category": category, "target": tgt, "prompt": pr, "control_u8": u8, "control_checksum": instrument.checksum(u8),
            "polarity": pol, "M_obj": arrays["M_obj"].astype(bool), "zeta_ctrl": arrays["zeta_ctrl"].astype(np.float64),
            "valid": arrays["valid"].astype(bool), "eps_shape": rec["placement"].get("eps_shape"),
            "p_ta_uv": [float(rec["source_stats"]["u0"]) * u8.shape[1] / 256.0, float(rec["source_stats"]["v0"]) * u8.shape[0] / 256.0],
            "m2b2_proxy_sha256": cache.sha256_file(Path(b2["paths"]["cache_r1"]) / category / rec["proxy_cache_key"] / "proxy.npz"),
            "upstream_hashes": {"c3": cache.sha256_file(c_dir / "control.npz"), "c1": cache.sha256_file(c_dir / "placement.json"),
                                "m2a": cache.sha256_file(ctx["m2a_out"] / clip / "stage0.npz"), "proxy_cache_key": rec["proxy_cache_key"],
                                **{k: v for k, v in (qc3.get("upstream_hashes") or {}).items() if k in ("c2_params",)}}}


def load_pipe(g: dict):
    import diffusers
    import torch
    from diffusers import FluxControlPipeline
    pipe = FluxControlPipeline.from_pretrained(g["hf_repo"], torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload() if g.get("cpu_offload") else pipe.to("cuda")
    comps = sorted(pipe.components.keys())
    suspicious = [c for c in comps if any(k in c.lower() for k in ("depth", "preprocess", "estimator", "annotator"))]
    assert not suspicious, f"the pipeline carries a preprocessor-like component {suspicious} — the control would not reach the model untouched"
    print(f"[m2d] {type(pipe).__name__} (diffusers {diffusers.__version__}); preprocessor-like components: none")
    return pipe, {"pipeline": type(pipe).__name__, "diffusers": diffusers.__version__, "gpu": torch.cuda.get_device_name(0)}


def generate(item: dict, pipe, pinfo: dict, ctx: dict) -> Path:
    import cv2
    import torch
    from PIL import Image
    cfg = ctx["cfg"]; g = cfg["generator"]; Kp = int(ctx.get("seeds_total") or cfg["generation"]["seeds"])
    assert int(cfg["generation"]["seeds"]) <= Kp <= int(cfg["generation"]["resample_budget_total"]), "seed count outside [K', resample budget]"
    out = ctx["cache_m2d"] / item["clip"] / item["category"]; cand = out / "candidates"; cand.mkdir(parents=True, exist_ok=True)
    f_rec = out / "generation.json"
    old = json.loads(f_rec.read_text(encoding="utf-8")) if f_rec.exists() else {}
    # a cached seed is reused iff everything that reaches the model is unchanged: control, prompt string, steps, guidance, size, repo
    same = (old.get("control_checksum") == item["control_checksum"] and old.get("prompt", {}).get("prompt") == item["prompt"]["prompt"]
            and old.get("guidance") == float(g["guidance"]) and old.get("steps") == int(g["steps"]) and old.get("seeds", 0) >= 1
            and old.get("generator_repo", g["hf_repo"]) == g["hf_repo"])
    calls = {c["seed"]: c for c in old.get("calls", [])} if same else {}
    u8 = item["control_u8"]; ctrl = Image.fromarray(u8)
    expect = torch.from_numpy(u8.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
    for k in range(Kp):
        f = cand / f"seed_{k}.png"
        if k in calls and f.exists():
            print(f"[m2d] {item['clip']}/{item['category']} seed {k}: cached (same control, prompt, config)"); continue
        t = pipe.image_processor.preprocess(ctrl, height=g["size"], width=g["size"])
        dev = float((t.float().cpu() - expect).abs().max()); cs = instrument.checksum(u8)
        assert cs == item["control_checksum"] and dev < 1e-5, f"control altered on the way to the model (checksum {cs}, dev {dev})"
        print(f"[m2d] {item['clip']}/{item['category']} seed {k}: control checksum {cs}, tensor-vs-D max dev {dev:.1e}", flush=True)
        img = pipe(prompt=item["prompt"]["prompt"], control_image=ctrl, height=g["size"], width=g["size"], num_inference_steps=int(g["steps"]),
                   guidance_scale=float(g["guidance"]), generator=torch.Generator("cuda").manual_seed(k)).images[0]
        img.save(f)
        calls[k] = {"seed": k, "control_checksum": cs, "tensor_max_dev": dev, "png_sha256": cache.sha256_file(f)}
        rec = {"schema": GENERATION_SCHEMA, "clip": item["clip"], "category": item["category"], "polarity": item["polarity"],
               "guidance": float(g["guidance"]), "steps": int(g["steps"]), "size": int(g["size"]), "generator_repo": g["hf_repo"], "seeds": Kp, "target": item["target"], "prompt": item["prompt"],
               "control_checksum": item["control_checksum"], "calls": [calls[s] for s in sorted(calls)], **pinfo,
               "config_hash": ctx["config_hash"], "upstream_hashes": item["upstream_hashes"]}
        f_rec.write_text(json.dumps(rec, indent=1), encoding="utf-8")       # after EVERY seed: Colab VMs die
    # contact sheet: the control (object outlined) and the K' candidates with the same outline
    edge = item["M_obj"] ^ cv2.erode(item["M_obj"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    tiles = [u8.copy()] + [cv2.imread(str(cand / f"seed_{k}.png"))[..., ::-1].copy() for k in range(Kp)]
    for i, im in enumerate(tiles):
        im[edge] = [0, 255, 0]
        cv2.putText(im, "control" if i == 0 else f"seed {i - 1}", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 0), 3)
    sheet = np.concatenate([cv2.resize(im, (640, 640), interpolation=cv2.INTER_AREA) for im in tiles], 1)
    viz = Path(ctx["drive"]) / "viz" / "m2d_anchors"; viz.mkdir(parents=True, exist_ok=True)
    png = viz / f"{item['clip']}_{item['category']}.png"
    cv2.imwrite(str(png), sheet[..., ::-1])
    return png


def measure(item: dict, seg, est, ctx: dict) -> dict:
    """Step 6: per candidate obj_fidelity, bg_fidelity, E_loc, E_area. RECORDED, never gated (§7)."""
    import cv2
    from scripts.probe_depth_estimator import estimate
    from src.anchor.m2d import segment as _seg
    out = ctx["cache_m2d"] / item["clip"] / item["category"]
    gen = json.loads((out / "generation.json").read_text(encoding="utf-8"))
    assert gen["control_checksum"] == item["control_checksum"], "the candidates were generated from a different control: re-run step 5"
    H, W = item["M_obj"].shape
    edge = item["M_obj"] ^ cv2.erode(item["M_obj"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    rows, tiles = [], []
    for c in gen["calls"]:
        f = out / "candidates" / f"seed_{c['seed']}.png"
        assert cache.sha256_file(f) == c["png_sha256"], f"{f} is not the image generation.json recorded"
        img = np.ascontiguousarray(cv2.imread(str(f))[..., ::-1])
        assert img.shape[:2] == (H, W)
        mask, sm = _seg.segment(seg, img, item["category"])
        zh = est["sign"] * estimate(est["pipe"], img, H, W)
        r = {"seed": c["seed"], **fidelity.object_stats(mask, item["M_obj"], item["p_ta_uv"]),
             **fidelity.bg_fidelity(zh, item["zeta_ctrl"], item["M_obj"], item["valid"], mask_gen=mask), "segmenter": sm}
        rows.append(r)
        ov = img.copy(); ov[mask] = (0.5 * ov[mask] + [127, 0, 127]).astype(np.uint8); ov[edge] = [0, 255, 0]
        cv2.putText(ov, f"seed {c['seed']}  IoU {r['obj_fidelity']:.2f}  bg rho {r['bg_fidelity']:+.2f}", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        tiles.append(cv2.resize(ov, (640, 640), interpolation=cv2.INTER_AREA))
        print(f"[m2d] {item['clip']}/{item['category']} seed {c['seed']}: obj IoU {r['obj_fidelity']:.3f}, bg rho {r['bg_fidelity']:+.3f}, "
              f"E_loc {r.get('E_loc', float('nan')):.4f}, E_area {r.get('E_area', float('nan')):.3f} (det {sm['det_score']:.2f})", flush=True)
    rec = {"schema": "m2d_fidelity_v1", "clip": item["clip"], "category": item["category"], "control_checksum": item["control_checksum"],
           "p_ta_uv_1024": item["p_ta_uv"], "eps_shape": item["eps_shape"], "estimator": est["repo"], "estimator_sign": est["sign"],
           "segmenter": {"owl": _seg.OWL, "sam2": _seg.SAM2, "query": _seg.QUERIES[item["category"]]}, "candidates": rows,
           "gates": {"g_obj_fidelity": {"outcome": "recorded"}, "g_bg_fidelity": {"outcome": "recorded"}}}
    (out / "fidelity.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    viz = Path(ctx["drive"]) / "viz" / "m2d_anchors"; viz.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(viz / f"{item['clip']}_{item['category']}_fidelity.png"), np.concatenate(tiles, 1)[..., ::-1])   # magenta = segment(generated): REVIEW it
    (viz / f"fidelity_{item['clip'][-8:]}_{item['category']}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def report(recs: list, items: list, ctx: dict) -> Path:
    import csv
    L = ["# m2_d step 6 — fidelity on the retained candidates (recorded, not gated)", "",
         "| clip | cat | seed | obj IoU | bg rho | E_loc | E_loc vs rendered centroid | log(A_gen/A_obj) | det |", "|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        for c in r["candidates"]:
            g = lambda k, f: (format(c[k], f) if k in c else "n/a")                                  # noqa: E731
            L.append(f"| {r['clip'][-8:]} | {r['category']} | {c['seed']} | {c['obj_fidelity']:.3f} | {c['bg_fidelity']:+.3f} | {g('E_loc', '.4f')} | "
                     f"{g('E_loc_vs_rendered_centroid', '.4f')} | {g('log_area_ratio', '+.3f')} | {c['segmenter']['det_score']:.2f} |")
    allc = [c for r in recs for c in r["candidates"] if c["object_found"]]
    L += ["", f"{len(allc)} candidates with an object found, of {sum(len(r['candidates']) for r in recs)}. Distributions (median [min, max]):", ""]
    for k in ("obj_fidelity", "bg_fidelity", "E_loc", "E_area"):
        v = np.array([c[k] for c in allc])
        L.append(f"- `{k}`: {np.median(v):.4f} [{v.min():.4f}, {v.max():.4f}]")
    L += ["", "Per-clip median (obj IoU, bg rho): " + "; ".join(
        f"{r['clip'][-8:]}/{r['category']} ({np.median([c['obj_fidelity'] for c in r['candidates']]):.3f}, {np.median([c['bg_fidelity'] for c in r['candidates']]):+.3f})" for r in recs),
          "", "The magenta overlay in `*_fidelity.png` is segment(generated): a mask on the wrong thing is an instrument error, not a generator finding. Review it first.",
          "Heading class of each candidate is the HUMAN's label (no validated instrument): fill `heading_labels.csv` from the full-resolution candidates; the control's class is deliberately not shown.",
          "Tolerances are NOT proposed here by design: the human and the reviewer set them from this table (SKILL §5), then step 7 runs."]
    viz = Path(ctx["drive"]) / "viz" / "m2d_anchors"
    f = viz / "report_m2d_fidelity.md"; f.write_text(chr(10).join(L) + chr(10), encoding="utf-8")
    fc = viz / "heading_labels.csv"                                        # APPEND rows for new clip/category pairs; existing rows (the
    have = set()                                                           # human's labels) are never touched (gap found 2026-09-21)
    if fc.exists():
        with open(fc, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        have = {(r[0].strip(), r[1].strip(), int(r[2])) for r in rows[1:] if len(r) > 2 and r[2].strip().isdigit()}
        ncol = len(rows[0])
    else:
        ncol = 5
        with open(fc, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["clip", "category", "seed", "label (toward | away | lateral_left_to_right | lateral_right_to_left | unclear) - judge at full resolution", "note"])
    with open(fc, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for r, it in zip(recs, items):
            for c in r["candidates"]:
                if (r["clip"], r["category"], int(c["seed"])) not in have:
                    w.writerow([r["clip"], r["category"], c["seed"]] + [""] * (ncol - 3))     # label column blind, whatever the sheet's layout
    return f


def _read_csv(f: Path, value_header: str) -> dict:
    """The value column is found BY HEADER (label sheets written before 2026-09-18 carry an extra control-class column)."""
    import csv
    out = {}
    with open(f, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    hits = [i for i, h in enumerate(rows[0]) if h.strip().lower().startswith(value_header)]
    assert len(hits) == 1, f"{f}: expected exactly one column starting with {value_header!r}, header is {rows[0]}"
    value_col = hits[0]
    for r in rows[1:]:
        if len(r) > value_col:
            out[(r[0].strip(), r[1].strip(), int(r[2]))] = r[value_col].strip()
    return out


def run_select(items: list, ctx: dict) -> int:
    """Step 7. Phase 1 = guard (lock thresholds + the human's heading labels); Phase 2 = the human's ranks."""
    import csv
    cfg, th = ctx["cfg"], ctx["lock"]
    for k in ("m2d_e_loc_rel_max", "m2d_e_area_max", "m2d_heading", "m2d_eps_shape_max"):
        assert k in th, f"{k} is not in calibration.lock.yaml (the human adds it; null = ungated)"
    viz = Path(ctx["drive"]) / "viz" / "m2d_anchors"
    f_lab, f_rank = viz / "heading_labels.csv", viz / "phase2_ranks.csv"
    assert f_lab.exists(), f"{f_lab} missing: run --measure, then label it"
    labels = _read_csv(f_lab, "label")
    ranks = {k: (int(v) if v else None) for k, v in _read_csv(f_rank, "rank").items()} if f_rank.exists() else {}
    budget = int(cfg["generation"]["resample_budget_total"])
    done, p1_all, waiting = [], [], []
    for it in items:
        out = ctx["cache_m2d"] / it["clip"] / it["category"]
        gen = json.loads((out / "generation.json").read_text(encoding="utf-8"))
        fid = json.loads((out / "fidelity.json").read_text(encoding="utf-8"))
        select.check_chain(gen["upstream_hashes"], it["upstream_hashes"])                 # test 7: hard exit
        select.check_chain({"control_checksum": gen["control_checksum"], "fidelity_control_checksum": fid["control_checksum"]},
                           {"control_checksum": it["control_checksum"], "fidelity_control_checksum": it["control_checksum"]})
        assert {c["seed"] for c in fid["candidates"]} == {c["seed"] for c in gen["calls"]}, "fidelity.json does not cover the generated seeds: re-run --measure"
        H, W = it["M_obj"].shape
        lab = {s: v for (c, k, s), v in labels.items() if (c, k) == (it["clip"], it["category"])}
        rk = {s: v for (c, k, s), v in ranks.items() if (c, k) == (it["clip"], it["category"])}
        args7 = (fid["candidates"], lab, rk, it["prompt"]["heading_class"], it["eps_shape"], th, H, W, len(gen["calls"]), budget)
        try:
            q = select.select(*args7)
        except select.AwaitingHuman as e:
            recs = select.phase1(fid["candidates"], lab, it["prompt"]["heading_class"], it["eps_shape"], th, H, W)   # labels missing re-raises here
            p1_all += recs; waiting.append((it, recs)); print(f"[m2d] {it['clip']}/{it['category']}: {e}"); continue
        p1_all += q["candidates"]
        p = it["prompt"]
        qc = {"clip": it["clip"], "category": it["category"], "polarity": gen["polarity"], "guidance": gen["guidance"], "steps": gen["steps"],
              "seeds": [c["seed"] for c in gen["calls"]], **q, "prompt": p["prompt"], "dictionary_row": p["dictionary_rows"],
              "heading_anchor_frame": {"class": p["heading_class"], "azimuth_deg": p["heading_azimuth_deg"], "frame": p["heading_frame"]},
              "shot_class_near_band_edge": p["shot_class_near_band_edge"], "eps_shape": it["eps_shape"],
              "heading_instrument": "human label (heading_labels.csv); NOT blind on the first run: the control's class was printed beside the label",
              "phase2_instrument": "human rank (phase2_ranks.csv), spec §4.2", "resample_count": max(0, len(gen["calls"]) - int(cfg["generation"]["seeds"])),
              "thresholds": {k: th[k] for k in ("m2d_e_loc_rel_max", "m2d_e_area_max", "m2d_heading", "m2d_eps_shape_max")},
              "config_hash": ctx["config_hash"], "upstream_hashes": {**gen["upstream_hashes"], "m2b2": it["m2b2_proxy_sha256"],
                                                                     "generation": cache.sha256_file(out / "generation.json"), "fidelity": cache.sha256_file(out / "fidelity.json")}}
        a = select.write_selection(out, qc)
        (viz / f"qc_m2d_{it['clip'][-8:]}_{it['category']}.json").write_text(json.dumps({"schema": select.QC_SCHEMA, **qc}, indent=1), encoding="utf-8")
        done.append(qc)
        print(f"[m2d] {it['clip']}/{it['category']}: {q['outcome']}, survivors {q['n_survivors']}/{len(gen['calls'])}, selected seed {q['selected_seed']} -> {a}")
    if waiting:                                                            # the ranks sheet: survivors only, never overwriting the human's entries
        rows = [(it["clip"], it["category"], r["seed"], r["phase1_pass"], "" if ranks.get((it["clip"], it["category"], r["seed"])) is None
                 else ranks[(it["clip"], it["category"], r["seed"])]) for it, recs in waiting for r in recs]
        keep = [(c, k, s, "True", v) for (c, k, s), v in ranks.items() if (c, k) not in {(it["clip"], it["category"]) for it, _ in waiting}]
        with open(f_rank, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["clip", "category", "seed", "phase1_pass", "rank (1 = best; ties allowed; leave rejected seeds blank)"])
            w.writerows(keep + rows)
        print(f"[m2d] Phase 2 is MANUAL: rank the surviving seeds in {f_rank} at FULL RESOLUTION (cache/m2_d/<clip>/<category>/candidates/seed_<k>.png; contact sheets are for navigation only), then re-run --select.")
    ex = select.exercised(p1_all)
    L = ["# m2_d step 7 — selection", "", f"Clips selected: {len(done)}; awaiting the human's ranks: {len(waiting)}.", "",
         "| gate | fails | ungated | exercised |", "|---|---|---|---|"] + [f"| {n} | {v['n_fail']} | {v['n_ungated']} | {v['exercised']} |" for n, v in ex.items()]
    if not any(v["exercised"] for v in ex.values()):
        L += ["", f"**All Phase-1 conditions passed on every one of the {len(p1_all)} candidates, so none was exercised.** The thresholds are derived from failure "
              "semantics (half the object's own scale; twice / half the requested area), not from this sample, and remain UNVALIDATED, in the same category as "
              "the dictionary's ±45° boundary and the low-angle ratio. With every candidate through Phase 1, **selection here is entirely Phase 2, which is "
              "manual**: the anchor is chosen by the human's eye among the seeds, and the lexicographic design's value on this set is defensive, not discriminative."]
    L += ["", "| clip | cat | outcome | survivors | selected | E_loc_rel max | E_area max | eps_shape (ungated) | band-edge flag |", "|---|---|---|---|---|---|---|---|---|"]
    for q in done:
        c = [r for r in q["candidates"] if "E_loc_rel" in r]
        L.append(f"| {q['clip'][-8:]} | {q['category']} | {q['outcome']} | {q['n_survivors']}/{len(q['candidates'])} | {q['selected_seed']} | "
                 f"{max(r['E_loc_rel'] for r in c):.3f} | {max(r['E_area'] for r in c):.3f} | {q['eps_shape']:.3f} | {q['shot_class_near_band_edge']} |")
    L += ["", "Known property (reviewer): E_area has a size-dependent floor ~ perimeter / area; at 0.7 % object size (265/human, +0.33 to +0.47) the metric "
          "is close to uninformative and the log 2 threshold has only a 1.5x margin there.", "",
          "HARD STOP (SKILL §10): human review of the anchors before m2_e."]
    (viz / "report_m2d_selection.md").write_text(chr(10).join(L) + chr(10), encoding="utf-8")
    print(f"[m2d] wrote {viz / 'report_m2d_selection.md'}")
    return 0


def run_auto_report(items: list, ctx: dict) -> int:
    """READ-ONLY on cache/m2_d (Module 2 v1 is closed; nothing under a v1 root is written). Per pair: what the automatic Phase-2
    rule (lowest surviving seed) would select, beside the human's recorded selection. Output: viz/m2d_anchors/auto_selection.*"""
    th = ctx["lock"]; viz = Path(ctx["drive"]) / "viz" / "m2d_anchors"
    f_lab = viz / "heading_labels.csv"
    labels = _read_csv(f_lab, "label") if f_lab.exists() else {}
    rows = []
    for it in items:
        out = ctx["cache_m2d"] / it["clip"] / it["category"]
        fid = json.loads((out / "fidelity.json").read_text(encoding="utf-8"))
        qf = out / "qc_m2d.json"
        human = json.loads(qf.read_text(encoding="utf-8")).get("selected_seed") if qf.exists() else None
        H, W = it["M_obj"].shape
        lab = {s: v for (c, k, s), v in labels.items() if (c, k) == (it["clip"], it["category"])}
        row = {"clip": it["clip"], "category": it["category"], "human_selected_seed": human}
        try:
            recs = select.phase1(fid["candidates"], lab, it["prompt"]["heading_class"], it["eps_shape"], th, H, W)
            ar = select.auto_ranks(recs)
            row.update({"auto_selected_seed": select.phase2(recs, ar), "n_survivors": len(ar), "agrees_with_human": None if human is None else bool(select.phase2(recs, ar) == human)})
        except select.AwaitingHuman as e:        # the heading label is still a human input (lock: m2d_heading: human); never bypassed here
            row.update({"auto_selected_seed": None, "blocked": f"needs the human heading label: {e}"})
        rows.append(row)
        print(f"[m2d] auto {it['clip']}/{it['category']}: auto={row.get('auto_selected_seed')} human={human} {row.get('blocked', '')}")
    (viz / "auto_selection.json").write_text(json.dumps({"rule": "lowest Phase-1 surviving seed", "rows": rows}, indent=1), encoding="utf-8")
    n = [r for r in rows if r.get("agrees_with_human") is not None]
    L = ["# m2_d automatic selection (report only; cache/m2_d untouched)", "", "Rule: lowest Phase-1 surviving seed. The heading label stays a human input (lock `m2d_heading: human`).", "",
         "| clip | cat | survivors | auto | human | agree |", "|---|---|---|---|---|---|"]
    L += [f"| {r['clip'][-8:]} | {r['category']} | {r.get('n_survivors', '-')} | {r.get('auto_selected_seed')} | {r['human_selected_seed']} | {r.get('agrees_with_human', r.get('blocked', '-'))} |" for r in rows]
    L += ["", f"Agreement with the human's selection: {sum(r['agrees_with_human'] for r in n)} of {len(n)} pairs."]
    (viz / "auto_selection.md").write_text(chr(10).join(L) + chr(10), encoding="utf-8")
    print(f"[m2d] wrote {viz / 'auto_selection.md'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip"); ap.add_argument("--category"); ap.add_argument("--all", action="store_true", help="every key of m2d.targets")
    ap.add_argument("--measure", action="store_true", help="step 6: fidelity on the retained candidates (OWLv2 + SAM2 + depth estimator)")
    ap.add_argument("--select", action="store_true", help="step 7 (CPU): Phase 1 guard + the human's Phase-2 ranks -> anchor.png, qc_m2d.json")
    ap.add_argument("--auto-report", action="store_true", help="CPU, read-only: what the automatic Phase-2 rule (lowest surviving seed) would select vs the human's selection")
    ap.add_argument("--seeds", type=int, help="step 5 only: generate this many seeds in total (resampling after a zero-survivor selection)")
    ap.add_argument("--dry-run", action="store_true", help="CPU only: inputs checked, prompt printed, nothing generated")
    ap.add_argument("--experiment-set", default=None, help="configs/experiment_set.yaml: target text by the fixed caption rule for pairs without human text")
    ap.add_argument("--caption", default=None, help="a clip outside the experiment set: its caption; target text by the fixed rule of src/anchor/m2d/auto_target.py")
    ap.add_argument("--config", default="configs/stage_a.yaml"); ap.add_argument("--config-m2b2", default="configs/m2b2.yaml")
    args = ap.parse_args()
    import yaml
    from src.preprocess.config import load_config
    cfg_a = load_config(args.config); drive = str(cfg_a["paths"]["drive_root"])
    sub = lambda s: Path(str(s).replace("${paths.drive_root}", drive))                       # noqa: E731
    m2d = yaml.safe_load(Path("configs/m2d.yaml").read_text(encoding="utf-8"))
    cfg = m2d["m2d"]
    ctx = {"cfg": cfg, "drive": drive, "lock": cfg_a.get("thresholds", {}), "b2": load_m2b2_config(args.config_m2b2, drive_root=drive),
           "cache_m2c": sub(yaml.safe_load(Path("configs/m2c.yaml").read_text(encoding="utf-8"))["paths"]["cache_m2c"]),
           "m2a_out": sub(yaml.safe_load(Path("configs/m2a.yaml").read_text(encoding="utf-8"))["paths"]["out_dir"]),
           "cache_m2d": sub(m2d["paths"]["cache_m2d"]),
           "config_hash": hashlib.sha256(json.dumps({k: cfg[k] for k in ("generator", "generation", "prompt_dictionary")}, sort_keys=True).encode()).hexdigest()[:16]}
    ctx["seeds_total"] = args.seeds
    if args.experiment_set:
        es = yaml.safe_load(Path(args.experiment_set).read_text(encoding="utf-8"))["experiment_set"]
        ctx["experiment_targets"] = {p_["pair"]: p_ for p_ in es["pairs"]}
    if args.caption:
        from src.anchor.m2d.auto_target import target_for
        ctx.setdefault("experiment_targets", {})[f"{args.clip}/{args.category}"] = {"target": target_for(args.category, args.caption), "caption": args.caption}
    todo = [tuple(k.split("/")) for k in cfg["targets"]] if args.all else [(args.clip, args.category)]
    assert all(c and k for c, k in todo), "give --clip and --category, or --all"
    items = []
    refused = []
    for clip, cat in todo:                                                 # ALL inputs are checked before the model is loaded
        try:
            it = prepare(clip, cat, ctx)
        except (AssertionError, FileNotFoundError) as e:                   # an upstream refusal is a RESULT: report it, skip the pair, go on
            if not args.all:
                raise
            refused.append((clip, cat, str(e).splitlines()[0][:160])); print(f"[m2d] {clip}/{cat}: SKIPPED (upstream refusal): {refused[-1][2]}"); continue
        items.append(it); p = it["prompt"]
        print(f"[m2d] {clip}/{cat}: heading {p['heading_class']} ({p['heading_azimuth_deg']:+.1f} deg, frame {p['heading_frame']}); "
              f"area {100 * p['obj_area_frac']:.1f} % -> {p['shot_class']}; camera/object height {p['camera_height_over_object_height']:.2f} "
              f"-> low-angle {p['camera_low']}; control {it['control_checksum']} ({it['polarity']})\n      \"{p['prompt']}\"")
    if refused:
        print(f"[m2d] {len(refused)} pair(s) skipped on upstream refusal: " + "; ".join(f"{c}/{k}" for c, k, _ in refused))
    assert items, "no admitted pair to run"
    if args.dry_run:
        print("[m2d] dry run: nothing generated"); return 0
    if args.auto_report:
        return run_auto_report(items, ctx)
    if args.select:
        return run_select(items, ctx)
    if args.measure:
        from scripts.probe_depth_estimator import load_estimator
        from src.anchor.m2d import segment as _seg
        sj = sub(m2d["paths"]["probe_root"]) / "estimator_sign" / "estimator_sign.json"
        assert sj.exists(), "run scripts/probe_depth_estimator.py first (the estimator's sign is measured, never assumed)"
        es = json.loads(sj.read_text(encoding="utf-8"))
        assert es["sign"] in (1, -1) and es["estimator"] == cfg["estimator"]["hf_repo"], f"estimator sign record unusable: {es.get('sign')} for {es.get('estimator')}"
        est = {"pipe": load_estimator(es["estimator"])[0], "sign": int(es["sign"]), "repo": es["estimator"]}
        seg = _seg.load_segmenter()
        f = report([measure(it, seg, est, ctx) for it in items], items, ctx)
        print(f"[m2d] step 6 done -> {f}. Copy report_m2d_fidelity.md, fidelity_*.json and *_fidelity.png to measurements/m2d_anchors/; label heading_labels.csv.")
        return 0
    pipe, pinfo = load_pipe(cfg["generator"])
    for it in items:
        print(f"[m2d] wrote {generate(it, pipe, pinfo, ctx)}")
    print("[m2d] step 5 done. Copy viz/m2d_anchors/*.png and each generation.json to measurements/m2d_anchors/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
