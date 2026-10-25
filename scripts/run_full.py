#!/usr/bin/env python
"""The whole pipeline in one line: a reference clip in, a generated video out (the automatic path the thesis's test set used).

    # the thesis's two targets, a new background
    python scripts/run_full.py --video /path/camel.mp4 --target human --scene "on a rocky riverbank at dusk"
    # a new subject from one sentence: subject before the motion verb, the verb, the scene after it
    python scripts/run_full.py --video /path/camel.mp4 --prompt "a brown bear walking on a snowy hill" --kind rigid
    # the same with the parts given separately
    python scripts/run_full.py --video /path/camel.mp4 --target bear --object "a brown bear" --kind rigid --action walking --scene "on a snowy hill"

Steps, each resumable (finished stages are skipped on a rerun): [setup] the environments (Colab: two virtual environments on the
system torch and the SAM 3D Objects environment; --setup none on a machine that has them); [1] Module 1 and the screening checks the
test set passed (scripts/run_census.py; a FAIL is a recorded outcome with its reason); [proxy] the proxies of record from the release
asset, and a new subject's proxy (scripts/make_proxy.py) when the target is new; [2] Module 2 on the automatic path; [3] Module 3 with
the final configuration (configs/m3_batch_exp.yaml: camera C + bundle adjustment, 21 frames, seed 0). The subject of the prompt is
always the target's identity (the generator is given its geometry); --scene is the background, empty = the reference's own scene read
from the caption. Root: --root or MT_ROOT (cache/, reports/, videos/ live there).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.anchor import subjects                                                     # noqa: E402
from src.anchor.m2d.auto_target import MOTION_VERBS, scene_from_caption             # noqa: E402

PROXY_ASSET = "https://github.com/jnheinrich451-eng/Recover-Retarget-Guide/releases/download/v1.0/proxies_of_record.zip"
VENV1, VENV2 = Path("/content/venv-phase1"), Path("/content/venv-phase2")


def sh(cmd, env=None, cwd=ROOT, shell=False):
    print("\n$", cmd if shell else " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, shell=shell, executable="/bin/bash" if shell else None)
    if r.returncode != 0:
        raise SystemExit(f"stopped: {'the command above' if not shell else cmd.split('&&')[-1].strip()} returned {r.returncode}; read its log")


DOUBLED = {"run": "running", "swim": "swimming", "spin": "spinning"}


def _base(w: str):
    """The listed motion verb a word stands for (walks, walking, walk -> walk), or None."""
    w = w.lower()
    if w in MOTION_VERBS:
        return w
    if w.endswith("ing"):
        for cand in (w[:-3], w[:-3] + "e", w[:-4]):
            if cand in MOTION_VERBS:
                return cand
    return None


def gerund(v: str) -> str:
    v = v.lower().rstrip("s") if v.lower().endswith("s") and v.lower()[:-1] in MOTION_VERBS else v.lower()
    if v.endswith("ing"):
        return v
    return DOUBLED.get(v) or (v[:-1] + "ing" if v.endswith("e") else v + "ing")


VEHICLE_WORDS = ("car", "bus", "truck", "jeep", "rover", "van", "motorcycle", "vehicle", "carriage", "cart", "boat")   # the census's rule
PERSON_WORDS = ("man", "woman", "person", "people", "boy", "girl", "child", "warrior", "soldier", "knight", "hiker", "runner", "walker",
                "worker", "player", "guy", "lady", "gentleman", "pedestrian", "human", "figure")


def infer_target(subject: str, kind: str | None) -> str:
    """A sentence without --target: a vehicle word -> the red car; a person word -> the man; otherwise a new subject (needs --kind),
    named after the subject's words. The thesis's two identities map to their targets whatever the exact wording."""
    words = set(re.findall(r"[a-z]+", subject.lower()))
    if words & set(VEHICLE_WORDS):
        return "vehicle"
    if words & set(PERSON_WORDS):
        return "human"
    if not kind:
        raise SystemExit(f"{subject!r} is neither a vehicle nor a person, so it is a new subject: add --kind rigid (animals, machines) or --kind human")
    return re.sub(r"[^a-z0-9]+", "_", subject.lower().removeprefix("a ").removeprefix("an ").removeprefix("the ")).strip("_")


def parse_prompt(prompt: str):
    """One sentence -> (subject text, action as a gerund, scene): the first motion verb (any form) splits the subject from the
    scene; the scene starts at the first locative after the verb, else is the remainder, and ends at a full stop or clause opener."""
    from src.anchor.m2d.auto_target import _first_locative, CLAUSE_OPENERS
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", prompt)
    verb = next((w for w in words if _base(w)), None)
    if verb is None:
        raise SystemExit(f"--prompt needs a motion verb (walk, run, drive, ride, travel, gallop, move, swim, skate, jump, fly, sail, climb, cross, ... in any form): {prompt!r}")
    m = re.search(r"\b" + re.escape(verb) + r"\b", prompt)
    subject = prompt[: m.start()].strip().rstrip(",")
    rest = prompt[m.end():].strip()
    loc = _first_locative(rest)
    scene = rest[loc[0]:] if loc else rest
    scene = re.split(r"[.]|, (?=(?:" + "|".join(CLAUSE_OPENERS) + r")\b)|, (?=\w+(?:ed|ing)\b)", scene)[0].strip().rstrip(",")
    return subject, gerund(verb), scene


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="the reference clip (mp4, at least 24 frames)")
    ap.add_argument("--clip", default=None, help="its id (default: the file name without extension)")
    ap.add_argument("--target", default=None, help="human | vehicle | a new subject's name (with --object/--kind, or --prompt)")
    ap.add_argument("--prompt", default=None, help="one sentence: '<subject> <motion verb> <scene>' (a new subject unless the subject is the target's own)")
    ap.add_argument("--object", default=None, help="new subject: its identity text, e.g. 'a brown bear'")
    ap.add_argument("--kind", default=None, choices=subjects.KINDS, help="new subject: rigid (SAM 3D Objects, rendered along the path) or human (SAM 3D Body)")
    ap.add_argument("--action", default=None, help="new subject: the verb of its prompt (default: the prompt's verb, else 'moving')")
    ap.add_argument("--scene", default=None, help="the background, e.g. 'on a rocky riverbank at dusk'; empty = the reference's own scene from the caption")
    ap.add_argument("--caption", default=None, help="a caption of the reference clip (default: built from the target and the scene; the prompt when given)")
    ap.add_argument("--root", default=os.environ.get("MT_ROOT"), help="data root (MT_ROOT): cache/, reports/, videos/")
    ap.add_argument("--env", default="colab_oneclick", help="interpreter preset in configs/environment.yaml")
    ap.add_argument("--setup", default="colab", choices=["colab", "none"], help="build the environments (Colab) or assume they exist")
    ap.add_argument("--proxy-asset", default=PROXY_ASSET, help="URL or local path of proxies_of_record.zip")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--module", default="all", choices=["1", "2", "3", "all"], help="run only one module (the others must be done)")
    args = ap.parse_args(argv)
    assert args.root, "give --root or set MT_ROOT"
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True); (root / "videos").mkdir(exist_ok=True)
    env = {"MT_ROOT": str(root), "MT_DATA": str(root), "PYTHONUNBUFFERED": "1", "SKIP_VIDEOS": "1"}
    clip = args.clip or Path(args.video).stem
    dst = root / "videos" / f"{clip}.mp4"
    if not dst.exists():
        shutil.copyfile(args.video, dst)

    # ---- the target, from the parts or from one sentence
    subject_text, verb, scene = (None, None, None)
    if args.prompt:
        subject_text, verb, scene = parse_prompt(args.prompt)
    target = args.target
    new = False
    if target is None and args.prompt:
        target = infer_target(subject_text, args.kind)
        print(f"[full] the sentence's subject {subject_text!r} -> target {target}", flush=True)
    assert target, "give --target (human | vehicle | a new name) or --prompt"
    new = target not in subjects.load()
    obj = args.object or subject_text
    kind = args.kind
    action = args.action or verb or "moving"
    if new:
        assert obj and kind, f"{target!r} is a new subject: give --object and --kind (or --prompt and --kind)"
    scene = (args.scene or scene or "").strip()
    ident = subjects.identity(target) if not new else {"object": obj, "action": action}
    caption = args.caption or args.prompt or f"{ident['object']} {ident['action']} {scene}".strip()
    print(f"[full] clip {clip} | target {target}{' (new: ' + obj + ', ' + kind + ')' if new else ''} | scene {scene or '(the reference’s own)'} | caption {caption!r}", flush=True)

    # ---- interpreters
    envs = yaml.safe_load((ROOT / "configs" / "environment.yaml").read_text(encoding="utf-8"))
    py = {k: str(v) for k, v in envs[args.env]["python"].items()}
    P1, P2 = py.get("phase1", py.get("default", sys.executable)), py.get("default", sys.executable)

    # ---- [setup]
    if args.setup == "colab":
        for v in (VENV1, VENV2):
            if not ((v / "bin" / "activate").exists() and (v / "bin" / "pip").exists()):        # absent, or half-built by a failed attempt
                shutil.rmtree(v, ignore_errors=True)
                # Colab's python has no ensurepip (python3-venv is not installed), so `python -m venv` fails at the pip step:
                # virtualenv (pure python, brings its own pip) builds the same thing
                r = subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(v)], capture_output=True, text=True)
                if r.returncode != 0:
                    shutil.rmtree(v, ignore_errors=True)
                    sh([sys.executable, "-m", "pip", "install", "-q", "virtualenv"])
                    sh([sys.executable, "-m", "virtualenv", "-q", "--system-site-packages", str(v)])
                assert (v / "bin" / "python").exists(), f"no interpreter in {v}"
        sh(f"source {VENV1}/bin/activate && bash scripts/bootstrap.sh", env=env, shell=True)
        sh(f"source {VENV2}/bin/activate && bash scripts/bootstrap_phase2.sh && (python -c 'import sam2' 2>/dev/null || pip install -q 'git+https://github.com/facebookresearch/sam2.git')", env=env, shell=True)
        cap = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() or "8.0"
        which = "both" if (kind == "rigid" or (not new and subjects.kind(target) == "rigid")) else "body"
        sh(f"source {VENV2}/bin/activate && bash scripts/setup_sam3d.sh {which}", env={**env, "DRIVE_ROOT": str(root), "TORCH_CUDA_ARCH_LIST": cap}, shell=True)

    # ---- [1] Module 1 and the screening
    if args.module in ("1", "all"):
        sh([P1, "scripts/run_census.py", "--clips", clip, "--category", target, "--caption", caption, "--drive-root", str(root), "--videos-dir", str(root / "videos")], env=env)
        rows = json.loads((root / "reports" / "census" / "census.json").read_text(encoding="utf-8"))
        verdict = next((r for r in rows if r["clip"] == clip), {}).get("census", "?")
        print(f"\n[full] census verdict for {clip}: {verdict}", flush=True)
        if verdict != "PASS":
            raise SystemExit(f"the clip did not pass the screening ({verdict}); reasons in {root / 'reports' / 'census' / 'census.md'}")
        if args.module == "1":
            return 0

    # ---- [proxy]
    if not (root / "cache" / "m2_b2").exists():
        src = args.proxy_asset
        z = Path(src) if Path(src).exists() else Path("/tmp/proxies_of_record.zip")
        if not z.exists():
            print(f"[full] downloading {src}", flush=True); urllib.request.urlretrieve(src, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
        print(f"[full] proxies of record unpacked under {root}", flush=True)
    if new and not subjects.proxy_key(target):
        sh([P2, "scripts/make_proxy.py", "--category", target, "--object", obj, "--kind", kind, "--action", action, "--env", args.env, "--drive-root", str(root)], env=env)

    # ---- [2] [3]
    run = [P2, "scripts/run_all.py", "--env", args.env, "--clip", clip, "--category", target]
    if args.module in ("2", "all"):
        sh(run + ["--module", "2", "--auto", "--proxy-from", str(root), "--caption", caption] + (["--scene", scene] if scene else []), env=env)
    if args.module in ("3", "all"):
        sh(run + ["--module", "3", "--auto", "--m3-config", "configs/m3_batch_exp.yaml", "--seeds"] + [str(s) for s in args.seeds], env=env)
    subprocess.run(run + ["--status"], cwd=ROOT, env={**os.environ, **env})
    out = root / "cache" / "m3" / clip / target
    vids = sorted(out.glob("A*_seed*_*.mp4")) if out.exists() else []
    print("\n[full] videos:" + ("".join(f"\n  {v}" for v in vids) if vids else " none written; read the log above"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
