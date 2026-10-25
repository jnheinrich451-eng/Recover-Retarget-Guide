#!/usr/bin/env python
"""The project page (GitHub Pages), second design: generated from the batch export, the sealed tables and the thesis figures.

    python scripts/make_project_page.py --config configs/project_page.yaml [--no-encode]

Reads configs/project_page.yaml (paths and prose), the export's index.json (our A0 / A3 videos, anchor, source, prompt per
pair), the per-pair / main / cross-method tables of experiment_tables_v2.md and the per-pair ablation tables of
ablation_guidance_tables.md; re-encodes every video it places to a small H.264 copy (the reference is cut to the pair's
window and retimed to the generated frame rate); writes <out>/index.html, <out>/assets/site.{css,js}, <out>/videos/test/*,
<out>/videos/abl/*, <out>/figures/*. The development-set videos already in <out>/videos are referenced as they are. A missing
video leaves an "n/a" tile, never a wrong one. Numbers are parsed from the records, never typed. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- video
def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(src: Path):
    out = subprocess.run([ffmpeg(), "-i", str(src), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"], capture_output=True, text=True).stderr
    fr = re.findall(r"frame=\s*(\d+)", out)
    fps = re.search(r"([\d.]+) fps", out)
    return (int(fr[-1]) if fr else None), (float(fps.group(1)) if fps else None)


def encode(src: Path, dst: Path, crf: int, max_h: int, fps: int, start: int | None = None, n: int | None = None, image_seq: bool = False):
    """Small H.264 copy. With start / n the input is cut by frame index and retimed to `fps` (frame k of the cut = frame k of
    the generated video). image_seq: src is a printf pattern of frames (DAVIS)."""
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = []
    if start is not None:
        vf.append(f"select='between(n\\,{start}\\,{start + n - 1})'")
    vf.append(f"setpts=N/{fps}/TB")
    vf.append(f"scale=-2:'min({max_h},ih)'")
    cmd = [ffmpeg(), "-y", "-loglevel", "error"]
    if image_seq:
        cmd += ["-framerate", str(fps)]
    cmd += ["-i", str(src), "-vf", ",".join(vf), "-an", "-r", str(fps), "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


# ----------------------------------------------------------------------------- records
def md_tables(text: str):
    """Every Markdown table in `text`, in order, as (header cells, rows of cells), with the section heading that precedes it."""
    tables, heading, buf = [], "", []
    for ln in text.splitlines() + [""]:
        if ln.startswith("#"):
            heading = ln.strip("# ").strip()
        if ln.startswith("|"):
            buf.append([c.strip() for c in ln.strip().strip("|").split("|")])
        elif buf:
            if len(buf) >= 2:
                tables.append((heading, buf[0], [r for r in buf[2:] if r]))
            buf = []
    return tables


def pair_key(p: str) -> str:
    return p.replace("/", "_")


def short_pair(p: str) -> str:
    clip, cat = p.split("/")
    return re.sub(r"^0+(?=\d)", "", clip) + " · " + cat


def num(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def esc(s) -> str:
    return html.escape(str(s), quote=True)


# ----------------------------------------------------------------------------- svg chart
def dumbbell_svg(rows, key_a, key_b, title, unit, log=False, better_lower=True):
    """One panel: per pair, A0 (hollow) to ours (filled); gold when ours is better, crimson otherwise."""
    import math
    W, H, left, right, top, rowh = 560, 24 * len(rows) + 60, 150, 24, 36, 24
    vals = [v for r in rows for v in (r[key_a], r[key_b]) if v is not None]
    lo, hi = min(vals), max(vals)
    if log:
        lo, hi = math.log10(max(lo, 1e-3)), math.log10(hi)
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad

    def x(v):
        t = math.log10(max(v, 1e-3)) if log else v
        return left + (t - lo) / (hi - lo) * (W - left - right)

    out = [f"<svg class='chart' viewBox='0 0 {W} {H}' role='img' aria-label='{esc(title)}'>",
           f"<text class='c-title' x='{left}' y='18'>{esc(title)}</text>"]
    # ticks
    ticks = []
    if log:
        for e in range(-1, 3):
            for m in (1, 2, 5):
                v = m * 10 ** e
                if lo <= math.log10(v) <= hi:
                    ticks.append(v)
    else:
        step = 10 ** math.floor(math.log10(hi - lo)) / 2
        v = math.floor(lo / step) * step
        while v <= hi:
            if v >= lo:
                ticks.append(round(v, 6))
            v += step
    for v in ticks:
        out.append(f"<line class='c-grid' x1='{x(v):.1f}' y1='{top - 6}' x2='{x(v):.1f}' y2='{H - 20}'/>")
        out.append(f"<text class='c-tick' x='{x(v):.1f}' y='{H - 6}' text-anchor='middle'>{v:g}</text>")
    out.append(f"<text class='c-tick' x='{W - right}' y='{top - 12}' text-anchor='end'>{esc(unit)}</text>")
    for i, r in enumerate(rows):
        y = top + i * rowh + 10
        a, b = r[key_a], r[key_b]
        out.append(f"<text class='c-row' x='{left - 8}' y='{y + 4}' text-anchor='end'>{esc(r['short'])}</text>")
        if a is None or b is None:
            out.append(f"<text class='c-tick' x='{left}' y='{y + 4}'>n/a</text>")
            continue
        better = (b < a) if better_lower else (b > a)
        cls = "win" if better else "loss"
        out.append(f"<line class='c-link {cls}' x1='{x(a):.1f}' y1='{y}' x2='{x(b):.1f}' y2='{y}'/>")
        out.append(f"<circle class='c-a' cx='{x(a):.1f}' cy='{y}' r='4.5'/>")
        out.append(f"<circle class='c-b {cls}' cx='{x(b):.1f}' cy='{y}' r='4.5'/>")
    out.append("</svg>")
    return "\n".join(out)


# ----------------------------------------------------------------------------- page
CSS_FILE = ROOT / "scripts" / "project_page" / "site.css"
JS_FILE = ROOT / "scripts" / "project_page" / "site.js"


def video_tile(rel: str | None, label: str, sub: str = "", poster: str | None = None) -> str:
    lab = f"<p class='lab'>{esc(label)}</p>" + (f"<p class='sub'>{sub}</p>" if sub else "")
    if rel is None:
        return f"<div class='cell'><div class='na'>n/a</div>{lab}</div>"
    return f"<div class='cell'><video src='{rel}' muted loop playsinline preload='metadata'></video>{lab}</div>"


def image_tile(rel: str | None, label: str) -> str:
    if rel is None:
        return f"<div class='cell'><div class='na'>n/a</div><p class='lab'>{esc(label)}</p></div>"
    return f"<div class='cell'><img src='{rel}' alt='{esc(label)}' loading='lazy'><p class='lab'>{esc(label)}</p></div>"


def arrow(a, b, better_lower=True, fmt="{:.2f}"):
    if a is None or b is None:
        return "<span class='num'>n/a</span>"
    better = (b < a) if better_lower else (b > a)
    cls = "win" if better else "loss"
    return f"<span class='num'>{fmt.format(a)} <span class='to'>→</span> <b class='{cls}'>{fmt.format(b)}</b></span>"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "project_page.yaml"))
    ap.add_argument("--no-encode", action="store_true", help="write the HTML only; reuse the videos already in out/")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    P = cfg["paths"]
    thesis = Path(P["thesis_root"])
    export = ROOT / P["export"]
    abl = ROOT / P["ablation_export"]
    out = Path(P["out"])
    V = cfg["video"]
    (out / "assets").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    # ---- records
    tables = md_tables((thesis / P["tables_md"]).read_text(encoding="utf-8"))
    main_tab = next(t for t in tables if t[0].startswith("1."))
    per_pair_tab = next(t for t in tables if t[0].startswith("3."))
    cross_tab = next(t for t in tables if t[0].startswith("4."))
    per_pair = {}
    for r in per_pair_tab[2]:
        d = dict(zip(per_pair_tab[1], r))
        per_pair[d["pair"]] = {
            "flags": {"footprint": d["footprint"], "depth": d["depth"],
                      "contact": {"in_contact": "in view", "contact_out_of_view": "out of view"}.get(d["contact"], d["contact"].replace("_", " ")),
                      "heading": "unchecked" if d["heading unchecked"] == "True" else "checked"},
            "cam_a0": num(d["CamRotErr A0"]), "cam_ours": num(d["ours"]) if "ours" in d else None,
        }
    # the header has two columns named "ours"; take them by position
    hdr = per_pair_tab[1]
    i_cam_a0, i_obj_a0 = hdr.index("CamRotErr A0"), hdr.index("ObjMC motion A0")
    for r in per_pair_tab[2]:
        p = r[0]
        per_pair[p]["cam_a0"], per_pair[p]["cam_ours"] = num(r[i_cam_a0]), num(r[i_cam_a0 + 1])
        per_pair[p]["obj_a0"], per_pair[p]["obj_ours"] = num(r[i_obj_a0]), num(r[i_obj_a0 + 1])
        per_pair[p]["align"] = r[hdr.index("v2 align A0 / ours")]
    index = json.loads((export / "index.json").read_text(encoding="utf-8"))
    by_pair = {}
    for row in index:
        by_pair.setdefault(row["pair"], {})[row["arm"]] = row      # "A0" / "A3" (arm_key carries the setting suffix)
    test_pairs = [p for p in per_pair if p in by_pair]
    missing = [p for p in per_pair if p not in by_pair]
    if missing:
        print("[page] pairs in the record but not in the export:", missing, file=sys.stderr)

    # ---- videos of the test pairs
    vdir = out / "videos" / "test"
    rows = []
    for p in test_pairs:
        rec = per_pair[p]
        a0, a3 = by_pair[p].get("A0"), by_pair[p].get("A3")
        base = a0 or a3
        prompt = base["prompt"]
        t_a = int(base.get("t_a", 0))
        tiles = {}
        # reference: the export's source cut (full clip; cut from t_a) or DAVIS frames
        ref_src = export / base["source"] if base.get("source") else None
        ref_dst = vdir / f"{pair_key(p)}_reference.mp4"
        if not args.no_encode:
            if ref_src and ref_src.exists():
                encode(ref_src, ref_dst, V["crf"], V["max_height"], V["fps"], start=t_a, n=int(base["frames"]))
            else:
                davis = Path(P.get("davis_frames", "")) / p.split("/")[0]
                if davis.exists():
                    encode(davis / "%05d.jpg", ref_dst, V["crf"], V["max_height"], V["fps"], start=t_a, n=int(base["frames"]), image_seq=True)
        tiles["reference"] = f"videos/test/{ref_dst.name}" if ref_dst.exists() else None
        anchor_src = export / base["anchor"]
        anchor_dst = out / "figures" / "anchors" / f"{pair_key(p)}.jpg"
        if anchor_src.exists() and not anchor_dst.exists():
            from PIL import Image
            anchor_dst.parent.mkdir(parents=True, exist_ok=True)
            im = Image.open(anchor_src).convert("RGB")
            im.thumbnail((2 * V["max_height"], 2 * V["max_height"]))        # the tile is ~1/4 of the frame; 720 px is plenty
            im.save(anchor_dst, quality=86, optimize=True)
        tiles["anchor"] = f"figures/anchors/{anchor_dst.name}" if anchor_dst.exists() else None
        for arm, r in (("A0", a0), ("A3", a3)):
            dst = vdir / f"{pair_key(p)}_{arm}.mp4"
            src = export / r["video"] if r else None
            if src and src.exists() and not args.no_encode:
                encode(src, dst, V["crf"], V["max_height"], V["fps"])
            tiles[arm] = f"videos/test/{dst.name}" if dst.exists() else None
        # the other methods on this pair (the grader's pairs folder), when present
        extra = []
        B = cfg.get("baselines")
        if B:
            clip, cat = p.split("/")
            for m in B["methods"]:
                src = Path(B["root"]) / f"{clip}__{cat}" / f"{clip}__{cat}__{m['key']}__seed0.mp4"
                dst = vdir / f"{pair_key(p)}_{m['key']}.mp4"
                if src.exists() and not args.no_encode:
                    encode(src, dst, V["crf"], V["max_height"], V["fps"])
                extra.append((m["label"], f"videos/test/{dst.name}" if dst.exists() else None))
            if not any(rel for _, rel in extra):
                extra = []
        rows.append({"pair": p, "short": short_pair(p), "cat": p.split("/")[1], "prompt": prompt, "tiles": tiles, "extra": extra, **rec})

    # ---- main table
    main_rows = []
    for r in main_tab[2]:
        d = dict(zip(main_tab[1], r))
        med = re.match(r"([\d.]+) \((\d+)\) / ([\d.]+) \((\d+)\)", d["v2 median A0 / ours"])
        wl = d["v2 W / L / T / excl"].split(" / ")
        main_rows.append({"metric": d["Metric"].replace("**", "").replace(" (primary)", ""), "primary": "(primary)" in d["Metric"],
                          "better": d["better"], "a0": med.group(1) if med else "-", "ours": med.group(3) if med else "-",
                          "n": med.group(2) if med else "-", "w": wl[0], "l": wl[1], "p": d["v2 p"], "sig": d["v2 significant"].replace("**", "")})

    # ---- cross-method table (A)
    cols = cfg["other_methods"]["cross_table_columns"]
    hdr_c = cross_tab[1]
    # the record's header lacks the final CLIP-IQA column name; the last cell is CLIP-IQA
    hdr_c = hdr_c + (["CLIP-IQA (higher)"] if len(cross_tab[2][0]) > len(hdr_c) else [])
    cross_rows = [[dict(zip(hdr_c, r)).get(c, "-") for c in cols] for r in cross_tab[2]]

    # ---- ablation summary: per setting vs the fixed row, over the five pairs
    abl_tabs = md_tables((thesis / P["ablation_md"]).read_text(encoding="utf-8"))
    A = cfg["ablations"]
    counts = {s["row"]: {"cam": [0, 0, 0], "obj": [0, 0, 0]} for s in A["settings"]}   # better, worse, scorable
    clip_vals = {}
    for heading, h, rs in abl_tabs:
        d = {r[0]: dict(zip(h, r)) for r in rs}
        fixed = d.get(A["fixed_row"])
        if not fixed:
            continue
        for s in A["settings"]:
            row = d.get(s["row"])
            if not row:
                continue
            for key, col in (("cam", "CamRotErr (deg)"), ("obj", "ObjMC motion (px)")):
                a, b = num(fixed[col]), num(row[col])
                if a is None or b is None:
                    continue
                counts[s["row"]][key][2] += 1
                if b < a:
                    counts[s["row"]][key][0] += 1
                elif b > a:
                    counts[s["row"]][key][1] += 1
        if heading.replace("## ", "") == A["clip_pair"]:
            clip_vals = {k: (num(v["CamRotErr (deg)"]), num(v["ObjMC motion (px)"])) for k, v in d.items()}

    # ---- ablation clips
    abl_pair = A["clip_pair"]
    abl_dir = abl / pair_key(abl_pair)
    abl_tiles = []
    stem = None
    for f in abl_dir.glob("A3_seed0_*_obj_ow0.75_sm_sky.mp4"):
        stem = f.name.split("_obj_")[0]       # A3_seed0_rec_F33
    for c in A["clips"]:
        name = (stem.replace("A3", "A3") + c["suffix"] + ".mp4") if stem else None
        src = abl_dir / name if name else None
        dst = out / "videos" / "abl" / f"{pair_key(abl_pair)}_{c['key']}.mp4"
        if src and src.exists() and not args.no_encode:
            encode(src, dst, V["crf"], V["max_height"], V["fps"])
        rowname = {"fixed": A["fixed_row"], "noobj": "A3 without the object render (vehicles)", "hard": "A3 hard mask",
                   "skyout": "A3 sky out of the mask", "euler": "A3 Euler scheduler", "g025": "A3 gamma 0.25"}[c["key"]]
        cv = clip_vals.get(rowname, (None, None))
        sub = (f"camera {cv[0]:.2f}°<br>path {cv[1]:.1f} px" if cv[0] is not None and cv[1] is not None else "")
        abl_tiles.append(video_tile(f"videos/abl/{dst.name}" if dst.exists() else None, c["label"], sub))

    # ---- method figure
    fig_src = thesis / P["method_figure"]
    fig_dst = out / "figures" / fig_src.name
    if fig_src.exists():
        shutil.copyfile(fig_src, fig_dst)
    # the dark variant of the method figure, translated from its SVG: no background, ink text to cream, the gold and purple
    # model boxes half-transparent, gold text kept gold, the blue connectors lightened
    svg_src = fig_src.with_suffix(".svg")
    if svg_src.exists():
        s = svg_src.read_text(encoding="utf-8")
        s = re.sub(r'<rect width="\d+" height="\d+" fill="white"/>', "", s, count=1)
        for a, b in {"#17212B": "#EFE3CD", "#33404D": "#D8CDB8", "#187EC1": "#6FB7E9", "#F5F7F9": "none", "white": "none", "#A87A06": "#E8C86A"}.items():
            s = s.replace(f'fill="{a}"', f'fill="{b}"').replace(f'stroke="{a}"', f'stroke="{b}"')
        s = s.replace('fill="#F7DC6F"', 'fill="#C9A227" fill-opacity="0.45"').replace('fill="#CDB8EE"', 'fill="#8E6CC4" fill-opacity="0.45"')
        fig_dst = out / "figures" / (fig_src.stem + "-dark.svg")
        fig_dst.write_text(s, encoding="utf-8", newline="\n")

    # ---- development-set rows (existing videos)
    dev_html = []
    OM = cfg["other_methods"]
    for pr in OM["pairs"]:
        k = pair_key(pr["pair"])
        cells = []
        for m in OM["methods"]:
            f = out / P["dev_videos"] / f"{k}_{m['key']}.mp4"
            cells.append(video_tile(f"{P['dev_videos']}/{f.name}" if f.exists() else None, m["label"]))
        flag = f" <span class='flag'>{esc(pr['flag'])}</span>" if pr.get("flag") else ""
        dev_html.append(f"<article class='pair dev'><header><p class='label'>{esc(short_pair(pr['pair']))}{flag}</p>"
                        f"<p class='prompt'>{esc(pr['title'])}</p><div class='ctl'></div></header>"
                        f"<div class='grid g{len(OM['methods'])}'>{''.join(cells)}</div></article>")

    # ---- HTML
    S = cfg["site"]
    teaser = S["teaser"]["pair"]
    tk = pair_key(teaser)
    teaser_ref = out / P["dev_videos"] / f"{tk}_reference.mp4"
    teaser_ours = out / P["dev_videos"] / f"{tk}_ours.mp4"

    def row_html(r):
        t = r["tiles"]
        cam = arrow(r["cam_a0"], r["cam_ours"])
        obj = arrow(r["obj_a0"], r["obj_ours"], fmt="{:.1f}")
        fl = r["flags"]
        flags = f"footprint {fl['footprint']} · depth {fl['depth']} · contact {fl['contact']} · heading {fl['heading']} · alignment {esc(r['align'])}"
        return (f"<article class='pair' data-cat='{r['cat']}'><header><p class='label'>{esc(r['short'])}</p>"
                f"<p class='prompt'>{esc(r['prompt'])}</p><div class='ctl'></div></header>"
                f"<div class='grid g4'>{video_tile(t['reference'], 'reference')}{image_tile(t['anchor'], 'anchor')}"
                f"{video_tile(t['A0'], 'backbone, no guidance')}{video_tile(t['A3'], 'ours')}</div>"
                f"<p class='nums'><span class='k'>camera rotation error, deg</span> {cam} <span class='k'>subject path error, px</span> {obj}</p>"
                f"<p class='flags'>{flags}</p>"
                + (f"<p class='label others'>{esc(cfg['baselines']['note'])}</p><div class='grid g{len(r['extra'])}'>"
                   + "".join(video_tile(rel, lab) for lab, rel in r["extra"]) + "</div>" if r.get("extra") else "")
                + "</article>")

    n_pairs = len(rows)
    cam_wins = sum(1 for r in rows if r["cam_ours"] is not None and r["cam_ours"] < r["cam_a0"])
    obj_wins = sum(1 for r in rows if r["obj_ours"] is not None and r["obj_ours"] < r["obj_a0"])
    prim = {m["metric"]: m for m in main_rows}
    cam_m, obj_m = prim.get("CamRotErr (deg)"), prim.get("ObjMC motion-only (px)")

    def main_table_html():
        h = "<table class='tbl'><thead><tr><th>metric</th><th>better</th><th>backbone</th><th>ours</th><th>wins / losses</th><th>p</th></tr></thead><tbody>"
        for m in main_rows:
            cls = " class='primary'" if m["primary"] else ""
            p = m["p"] if m["p"] not in ("exploratory",) else "—"
            h += f"<tr{cls}><td>{esc(m['metric'])}</td><td>{esc(m['better'])}</td><td>{esc(m['a0'])}</td><td>{esc(m['ours'])}</td><td>{esc(m['w'])} / {esc(m['l'])}</td><td>{esc(p)}</td></tr>"
        return h + "</tbody></table>"

    def cross_table_html():
        h = "<table class='tbl'><thead><tr>" + "".join(f"<th>{esc(c.replace(' (lower)', ' ↓').replace(' (higher)', ' ↑'))}</th>" for c in cols) + "</tr></thead><tbody>"
        for r in cross_rows:
            name = r[0].replace(", different prompt subject", "<sup>†</sup>").replace("(CLIP-IQA only; not a ceiling)", "(CLIP-IQA only)")
            h += "<tr>" + f"<td>{name}</td>" + "".join(f"<td>{esc(c)}</td>" for c in r[1:]) + "</tr>"
        return h + "</tbody></table>"

    def abl_table_html():
        h = "<table class='tbl center'><thead><tr><th>setting (one change from the fixed configuration)</th><th>camera error: better / worse</th><th>subject path: better / worse</th><th>scorable pairs</th></tr></thead><tbody>"
        for s in A["settings"]:
            c = counts[s["row"]]
            h += f"<tr><td>{esc(s['name'])}</td><td>{c['cam'][0]} / {c['cam'][1]}</td><td>{c['obj'][0]} / {c['obj'][1]}</td><td>{c['cam'][2]}</td></tr>"
        return h + "</tbody></table>"

    steps_html = "".join(
        f"<li class='step'><p class='label'>{i + 1:02d} · <span class='status s-{st['status'].split(',')[0].replace(' ', '-')}'>{esc(st['status'])}</span></p>"
        f"<h3>{esc(st['title'])}</h3><p>{esc(st['text'])}</p><p class='evidence'><span class='k'>evidence</span> {esc(st['evidence'])}</p></li>"
        for i, st in enumerate(cfg["process"]["steps"]))

    chart_rows = [{"short": r["short"], "cam_a0": r["cam_a0"], "cam_ours": r["cam_ours"], "obj_a0": r["obj_a0"], "obj_ours": r["obj_ours"]} for r in rows]
    chart = ("<div class='charts'>" + dumbbell_svg(chart_rows, "cam_a0", "cam_ours", "camera rotation error", "degrees, log scale", log=True)
             + dumbbell_svg(chart_rows, "obj_a0", "obj_ours", "subject path error", "pixels at 720×480") + "</div>"
             "<p class='cap'>Hollow mark: the backbone without guidance. Filled mark: ours. Gold where ours is lower, red where it is higher.</p>")

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(S['title'])} — {esc(S['subtitle'])}</title>
<meta name="description" content="{esc(S['subtitle'])}. {esc(S['author'])}.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400..700&family=IBM+Plex+Mono:wght@400;500&family=Josefin+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
<header class="top"><a class="brand" href="#top">{esc(S['title'])}</a>
<nav><a href="#method">How it works</a><a href="#results">Results</a><a href="#others">Other methods</a><a href="#process">How we got here</a><a href="#ablations">Ablations</a><a href="#notes">Notes</a></nav></header>

<section class="hero" id="top"><div class="frame">
<p class="label">{esc(S['eyebrow'])}</p>
<h1>{esc(S['title'])}</h1>
<p class="statement">{esc(S['subtitle'])}</p>
<p class="author">{esc(S['author'])}</p>
<p class="lead">{esc(S['lead'])}</p>
<p class="links">{''.join(f"<a href='{l['href']}'>{esc(l['label'])}</a>" for l in S['links'])}</p>
<div class="teaser pair"><header><div class="ctl"></div></header><div class="grid g2">
{video_tile(f"{P['dev_videos']}/{teaser_ref.name}" if teaser_ref.exists() else None, 'reference')}
{video_tile(f"{P['dev_videos']}/{teaser_ours.name}" if teaser_ours.exists() else None, 'ours')}
</div><p class="cap">{esc(S['teaser']['note'])}</p></div>
</div></section>

<section class="band" id="method"><div class="frame">
<p class="label">How it works</p><h2>Three stages, one frozen generator</h2>
<figure><img src="figures/{fig_dst.name}" alt="The three stages of the method"></figure>
<div class="stages">{''.join(f"<div class='stage'><p class='label'>{i + 1:02d}</p><h3>{esc(s['name'])}</h3><p>{esc(s['text'])}</p></div>" for i, s in enumerate(cfg['stages']))}</div>
</div></section>

<section class="band" id="results"><div class="frame">
<p class="label">Results</p><h2>Fifteen test pairs, against the same generator without guidance</h2>
<p class="lede">{esc(cfg['results']['intro'])}</p>
<div class="headline">
<div><p class="label">camera rotation error, median</p><p class="big">{esc(cam_m['a0'] if cam_m else '-')} <span class='to'>→</span> {esc(cam_m['ours'] if cam_m else '-')}<span class='unit'> deg</span></p><p class="small">lower on {cam_wins} of {n_pairs} pairs · {'significant after Holm correction (one-sided sign test)' if cam_m and cam_m['sig'] == 'yes' else 'not significant'}</p></div>
<div><p class="label">subject path error, median</p><p class="big">{esc(obj_m['a0'] if obj_m else '-')} <span class='to'>→</span> {esc(obj_m['ours'] if obj_m else '-')}<span class='unit'> px</span></p><p class="small">lower on {obj_wins} of {n_pairs} pairs · not significant</p></div>
</div>
{chart}
<p class="lede">{esc(cfg['results']['reading'])}</p>
<div class="filters"><span class="label">show</span><button data-f="all" class="on">all</button><button data-f="human">human targets</button><button data-f="vehicle">vehicle targets</button><span class="label hint">space plays or pauses the row under the pointer; , and . step frames</span></div>
<div class="rows">{''.join(row_html(r) for r in rows)}</div>
<details class="more"><summary class="label">All metrics of the paired comparison</summary>{main_table_html()}<p class="cap">Medians over the fifteen pairs; wins and losses of ours against the backbone; the two primary metrics carry the test, the others are exploratory. ObjErr3D excludes four backbone videos without a valid camera alignment.</p></details>
</div></section>

<section class="band" id="others"><div class="frame">
<p class="label">Other methods</p><h2>Beside the published methods, on the development set</h2>
<p class="lede">{esc(OM['intro'])}</p>
<div class="rows">{''.join(dev_html)}</div>
<details class="more"><summary class="label">The cross-method table on the thirteen MiraData test pairs</summary>{cross_table_html()}<p class="cap">{esc(OM['cross_table_note'])} <sup>†</sup> different prompt subject.</p></details>
</div></section>

<section class="band" id="process"><div class="frame">
<p class="label">How we got here</p><h2>What was tried, what was kept, what was set aside</h2>
<p class="lede">{esc(cfg['process']['intro'])}</p>
<ol class="steps">{steps_html}</ol>
</div></section>

<section class="band" id="ablations"><div class="frame">
<p class="label">Ablations</p><h2>One change at a time</h2>
<p class="lede">{esc(A['intro'])}</p>
<article class="pair"><header><p class="label">{esc(short_pair(abl_pair))}</p><div class="ctl"></div></header><div class="grid g6">{''.join(abl_tiles)}</div></article>
{abl_table_html()}
<p class="cap">Counts of the five development pairs on which the setting is better or worse than the fixed configuration on each primary metric; the remainder are unscorable videos. Descriptive.</p>
</div></section>

<section class="band" id="notes"><div class="frame">
<p class="label">Notes</p><h2>What this does not do, and what it is built on</h2>
<ul class="limits">{''.join(f"<li>{esc(l)}</li>" for l in cfg['limitations'])}</ul>
<p class="credits"><span class="k">models</span> {esc(cfg['credits']['models'])}</p>
<p class="credits">{esc(cfg['credits']['note'])}</p>
<p class="foot">{esc(S['author'])} · Australian National University · 2026 · <a href="{S['links'][0]['href']}">code</a></p>
</div></section>
<script src="assets/site.js"></script>
</body></html>
"""
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    shutil.copyfile(CSS_FILE, out / "assets" / "site.css")
    shutil.copyfile(JS_FILE, out / "assets" / "site.js")
    n_vid = len(list((out / "videos").rglob("*.mp4")))
    size = sum(f.stat().st_size for f in (out / "videos").rglob("*.mp4")) / 1e6
    print(f"page written: {out / 'index.html'}; test pairs {n_pairs}; videos {n_vid} ({size:.1f} MB); missing in export: {missing}")


if __name__ == "__main__":
    main()
