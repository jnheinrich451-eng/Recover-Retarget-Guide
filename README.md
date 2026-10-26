<div align="center">

# Recover, Retarget, Guide

### Camera and object motion transfer with a frozen video model

**Jiazhouxin Zeng** · supervisor **Dr Jing Zhang** · Australian National University, honours thesis, 2026

[![Project page](https://img.shields.io/badge/Project%20page-videos-c9a227?style=flat-square)](https://jnheinrich451-eng.github.io/camera-object-motion-transfer/)
[![Open in Colab](https://img.shields.io/badge/Colab-quick%20start-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/jnheinrich451-eng/Recover-Retarget-Guide/blob/main/colab_quickstart.ipynb)
[![Release asset](https://img.shields.io/badge/Release-bundled%20example-6b3fa0?style=flat-square)](https://github.com/jnheinrich451-eng/Recover-Retarget-Guide/releases)
[![Licence](https://img.shields.io/badge/Licence-MIT-2d6a4f?style=flat-square)](LICENSE)

🧊 **Training-free** &nbsp;·&nbsp; 🧭 **Geometry in the target scene** &nbsp;·&nbsp; 🎞️ **Frozen Wan 2.1 I2V** &nbsp;·&nbsp; 📏 **Measured, per pair**

</div>

## Transfer a reference's camera and subject motion to a new subject in a new scene

<table align="center">
<tr><th>Reference video</th><th>Ours, "a red car driving, on open scrubland"</th><th>Ours, "a man in a grey jacket walking, on a rocky riverbank"</th></tr>
<tr>
<td><img src="assets/readme/teaser_vehicle_reference.gif" width="300" alt="reference"></td>
<td><img src="assets/readme/teaser_vehicle_ours.gif" width="300" alt="ours, vehicle target"></td>
<td><img src="assets/readme/teaser_human_ours.gif" width="300" alt="ours, human target"></td>
</tr>
</table>

Given **one reference video** and **one text prompt**, the pipeline generates a video whose camera and subject move like the
reference's, in the scene the prompt describes, with a subject of another category. No target image is supplied: the target's
first frame is synthesised. The video generator, Wan 2.1 I2V 14B, is used frozen: its weights, its sampler and its text guidance
are unchanged.

- 🔍 **Recover.** The reference is lifted to a camera trajectory, a subject path and a ground plane in one frame and one scale with a frozen 3D tracker; the subject is masked without a human prompt; a clip whose camera, subject or support surface cannot be recovered is refused.
- 🎯 **Retarget.** A proxy of the target category is placed where the source subject stood, facing its heading on its ground; a depth control built from that placement drives a depth-conditioned image generator; the anchor is selected automatically and reconstructed in 3D.
- 🧲 **Guide.** The anchor's own content is rendered along the recovered camera path, and the object along the subject's path for rigid targets; inside a window of noise levels the frozen generator's clean estimate is corrected toward that render wherever a mask marks it reliable. One encode, no gradient, sixteen of fifty steps.
- 📊 **Evaluation.** Fifteen test pairs against the same generator without guidance: camera rotation error falls on 15 of 15 (median 3.59° to 1.08°, significant after Holm correction); the subject's on-screen path is not improved on human targets (6 of 15), whose arm renders no object. Every number on the [project page](https://jnheinrich451-eng.github.io/camera-object-motion-transfer/) is read from the sealed records in [`measurements/`](measurements/).

<p align="center"><img src="assets/readme/method-overview.png" width="900" alt="the three stages"></p>

## Results

Four test pairs, seed 0, 21 frames: the reference, the backbone without guidance, and ours. All fifteen pairs, the other methods, the ablations and what was tried along the way are on the [project page](https://jnheinrich451-eng.github.io/camera-object-motion-transfer/).

<table align="center">
<tr><th></th><th>Reference</th><th>Backbone, no guidance</th><th>Ours</th></tr>
<tr><td><b>191.6.037</b><br>→ human</td>
<td><img src="assets/readme/000000000191.6.037_human_reference.gif" width="260"></td>
<td><img src="assets/readme/000000000191.6.037_human_backbone.gif" width="260"></td>
<td><img src="assets/readme/000000000191.6.037_human_ours.gif" width="260"></td></tr>
<tr><td><b>217.0.006</b><br>→ vehicle</td>
<td><img src="assets/readme/000000000217.0.006_vehicle_reference.gif" width="260"></td>
<td><img src="assets/readme/000000000217.0.006_vehicle_backbone.gif" width="260"></td>
<td><img src="assets/readme/000000000217.0.006_vehicle_ours.gif" width="260"></td></tr>
<tr><td><b>431.5.009</b><br>→ human</td>
<td><img src="assets/readme/000000000431.5.009_human_reference.gif" width="260"></td>
<td><img src="assets/readme/000000000431.5.009_human_backbone.gif" width="260"></td>
<td><img src="assets/readme/000000000431.5.009_human_ours.gif" width="260"></td></tr>
<tr><td><b>camel</b> (DAVIS)<br>→ human</td>
<td><img src="assets/readme/camel_human_reference.gif" width="260"></td>
<td><img src="assets/readme/camel_human_backbone.gif" width="260"></td>
<td><img src="assets/readme/camel_human_ours.gif" width="260"></td></tr>
</table>

## Quick start: the bundled example

[![Open in Colab](https://img.shields.io/badge/Colab-open%20colab__quickstart.ipynb-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/jnheinrich451-eng/Recover-Retarget-Guide/blob/main/colab_quickstart.ipynb)

The clip `000000000265.2.005` is the running example of the thesis, in two retargetings: a red car on scrubland (`vehicle`) and a
man on a rocky riverbank (`human`). Its intermediate caches (the Module 2 anchor and handoff, the Module 3 lift, geometry and
renders) are published as the release asset `m3_example_265.2.005.zip`, so that Module 3 can be run on its own. The notebook does
the steps below on an A100 runtime; by hand:

1. Unpack `m3_example_265.2.005.zip` into a folder, say `DATA/`; it has the layout `DATA/cache/{m2_c,m2_d,m2_e,m3}/<clip>/<category>/`. The renders carry the targets the fixed configuration uses (`render_sky.npz` for the human pair, `render_obj_sky.npz` for the vehicle pair); `scripts/render_m3_target.py --sky-in-mask [--object]` rebuilds them from `lift.npz` and `geometry.npz` on a CPU.
2. Put the source clip at `DATA/videos/000000000265.2.005.mp4` (a MiraData clip; not redistributed here).
3. On a GPU machine (A100 40 GB), install the Module 3 environment and run the sampler:

```bash
SKIP_VIDEOS=1 bash scripts/bootstrap_phase2.sh      # diffusers, transformers, MoGe; no Drive and no token needed
python scripts/run_m3_s4_guided.py --drive-root DATA --pair 000000000265.2.005/human --seed 0 --frames 33 --arm A0                          # no guidance
python scripts/run_m3_s4_guided.py --drive-root DATA --pair 000000000265.2.005/human --seed 0 --frames 33 --arm A3 --soft-mask --sky-mask   # ours
python scripts/measure_m3_a0.py    --drive-root DATA --videos-dir DATA/videos --pairs 000000000265.2.005/human --source-frames 33 \
       --videos A0_seed0_rec_F33.mp4 A3_seed0_rec_F33_sm_sky.mp4 source                                                                     # grade both against the reference
```

For the vehicle pair add `--object --obj-window 0.85 0.75` to the guided run. The videos land in `DATA/cache/m3/000000000265.2.005/<category>/`;
the grader's table in `DATA/reports/m3_s4/a0_motion.md`. The whole batch of the thesis (every pair, both arms, grading, export) is one driver:

```bash
python scripts/run_m3_batch.py phase1 --drive-root DATA     # CPU: geometry and renders (skipped when present and of record)
python scripts/run_m3_batch.py phase2 --drive-root DATA     # GPU: sampling
python scripts/run_m3_batch.py grade  --drive-root DATA     # GPU: CoTracker grader + mask-centroid arbiter
python scripts/run_m3_batch.py export --drive-root DATA     # every video + source + anchor + index.json, zipped
```

`--drive-root` is any folder with the `cache/` and `reports/` layout; `MT_ROOT` in the shell is its default.

## Running the whole pipeline on a new clip

The three modules run in three environments and are never loaded together (Phase 1 preprocessing and Phase 2 generation are
memory-disjoint by rule). Each module is a runner over one clip; each writes to `cache/<stage>/<clip>/` and refuses to overwrite a
cache whose config hash matches. Thresholds live in `configs/calibration.lock.yaml` and are not edited by the code.

| step | runner | runtime | external models |
|---|---|---|---|
| Module 1: depth, tracks, camera, subject mask, dense subject tracks, heading | `scripts/run_pipeline.py --clip-id <clip>` (A → B → C → C.5 → B.5) | GPU 24 GB, `scripts/bootstrap.sh` | OpenD4RT, ViPE, SAM 2.1 |
| Module 2: admission, proxy, placement and control, anchor, reconstruction and handoff | `scripts/run_m2a.py`, `run_m2b2.py`, `run_m2c_c3.py`, `run_m2d.py`, `run_m2e.py` (`run_m2a.py --clip-id <clip>`; the others `--clip <clip> --category vehicle\|human`) | GPU 24 GB, `scripts/setup_sam3d.sh` for the reconstructor | FLUX.1 (dev and Depth), SAM 3D, Depth Anything |
| Module 3: geometry, renders, guided sampling, grading | `scripts/run_m3_batch.py` (or the per-step scripts it calls) | A100 40 GB, `scripts/bootstrap_phase2.sh` | MoGe-2 (phase 1), Wan 2.1 I2V 14B (phase 2), CoTracker3 (grading) |

One command chains them for a clip, with done-checks per stage and the human review points Module 2 keeps (it stops there with
the command to run next, and resumes on the next call):

```bash
python scripts/run_all.py --env colab --clip 000000000265.2.005 --category human --module 1     # Module 1 (Phase-1 runtime)
python scripts/run_all.py --env colab --clip 000000000265.2.005 --category human --module 2     # Module 2 (stops at its reviews)
python scripts/run_all.py --env colab --clip 000000000265.2.005 --category human --module 3     # Module 3 (Phase-2 runtime)
python scripts/run_all.py --env local --clip 000000000265.2.005 --category human --status       # what exists, read-only
```

### The experiment set, with no human step

The screened clips (`configs/experiment_set.yaml`, frozen 2026-10-19) run under their own root (`colab_exp`) with the gates-only
path: the category proxies are copied (never regenerated), the control review is its own gate record, the anchor is selected by the
fixed rule (`scripts/select_m2d_anchors.py`), the prompt comes from the caption by a fixed rule (`src/anchor/m2d/auto_target.py`), and
the handoff verdict is the gate outcomes. A pair that fails a gate is a recorded outcome (`reports/experiment/run_all.json`) and the
next pair runs.

```bash
python scripts/run_all.py --env colab_exp --set configs/experiment_set.yaml --module 2 --auto --proxy-from <root with the proxies>
python scripts/run_all.py --env colab_exp --set configs/experiment_set.yaml --module 3 --auto      # Phase-2 session (configs/m3_batch_exp.yaml: 21 frames, seed 0)
python scripts/run_all.py --env colab_exp --set configs/experiment_set.yaml --status
python scripts/select_m2d_anchors.py select --pairs <all pairs>                                     # the set-level selection report (CPU, after Module 2)
```

`--env` picks a preset in `configs/environment.yaml`: the root that holds `cache/` and `reports/`, the folder that holds `videos/`, and
the interpreter per module step. On your own machine set `MT_ROOT` and `MT_DATA` in the shell (or edit the `local` preset); every
runner takes its paths from them, and paths are outside every config hash, so caches made on Colab stay valid locally. The module
environments stay separate: the Phase-2 requirements are never installed into the Phase-1 environment, and the reconstructor has its
own. The prompts of the six development pairs are in `docs/m3_batch_prompts.json`.

## Running it yourself

### The environments

The pipeline runs in three environments that cannot share one Python installation (the tracker pins one torch, the generators
another, the rigid reconstructor needs its own compiled stack). Each is built by one script; the one-line runner below builds all
three on Colab by itself (`--setup colab`), and on your own machine you build them once and pass `--setup none`.

| environment | holds | built by | used for | GPU |
|---|---|---|---|---|
| Phase 1 | `requirements.txt`, OpenD4RT (tracks, depth, camera), SAM 2.1 (masks) | `scripts/bootstrap.sh` | Module 1 and the screening checks | 24 GB |
| Phase 2 | diffusers, transformers, MoGe-2; SAM 3D Body (people) and SAM 2 for the anchor's checks | `scripts/bootstrap_phase2.sh`, then `scripts/setup_sam3d.sh body` | Module 2 (anchor, reconstruction, handoff) and Module 3 (renders, guided sampling, grading) | 24 GB for Module 2, 40 GB for Module 3 |
| SAM 3D Objects | the rigid reconstructor in its own conda-style environment (micromamba, compiled once, saved under `<root>/envs/`) | `scripts/setup_sam3d.sh objects` | rigid targets only (the red car, or a new rigid subject) | 24 GB |

`configs/environment.yaml` names the interpreter of each environment per preset: `colab_oneclick` (the three side by side in one
Colab VM, two virtual environments on Colab's torch), `colab_public` (one environment per session), `local` (edit the paths). Every
runner reads its data root from `MT_ROOT` (`cache/`, `reports/`, `videos/` live there) and overrides it with `--root` or
`--drive-root`. Model weights download on first use: Wan 2.1, MoGe-2, SAM 2.1 and the tracker are public; FLUX.1-dev (a new subject's proxy candidates), FLUX.1-Depth-dev (the anchors), SAM
3D Body and SAM 3D Objects are gated on Hugging Face, so Module 2 needs an `HF_TOKEN` in the environment from an account that
accepted their terms (the bootstrap scripts also read it from `<Drive>/secrets/hf_token.env` when a Drive is mounted). No GitHub
token is needed. The reference clips of the thesis are MiraData and DAVIS clips and are not redistributed; `SKIP_VIDEOS=1` tells the
bootstrap scripts not to try the author's dataset.

### One line: a reference clip in, a generated video out

`scripts/run_full.py` runs the automatic path the thesis's test set used, end to end and resumable: Module 1 with the screening
checks, the proxies, Module 2, Module 3 in the final configuration (camera from the bundle-adjusted static-track solve, 21 frames,
seed 0). A clip refused by a check is a recorded outcome with its reason, not an error.

```bash
# the thesis's human target, a new background
python scripts/run_full.py --video camel.mp4 --target human --scene "on a rocky riverbank at dusk" --root DATA --setup colab
# one sentence: subject, motion verb, scene; a person or vehicle word picks the thesis's target, anything else is a new subject
python scripts/run_full.py --video camel.mp4 --prompt "a man walking on a beach at sunset" --root DATA --setup none
python scripts/run_full.py --video camel.mp4 --prompt "a brown bear walking on a snowy hill" --kind rigid --root DATA --setup none
```

| parameter | meaning |
|---|---|
| `--video` | the reference clip, mp4, at least 24 frames (the pipeline reads up to 81); its motion is what is transferred |
| `--clip` | the clip's id in the data root (default: the file name) |
| `--target` | `human` (a man in a grey jacket and dark trousers, walking), `vehicle` (a red car, driving), or a new subject's name |
| `--prompt` | one sentence, `<subject> <motion verb> <scene>`; the rule splits it: subject → target, verb → action, the rest → scene |
| `--scene` | the background, e.g. `"on a rocky riverbank at dusk"`; empty = the reference's own scene, as in the thesis |
| `--object`, `--kind`, `--action` | a new subject: its identity text, `rigid` (reconstructed as an object, rendered along the path) or `human` (SAM 3D Body), and its verb; from a sentence only `--kind` is needed |
| `--caption` | a caption of the clip, recorded with the run (default: built from the target and the scene) |
| `--root` | the data root: `cache/`, `reports/`, `videos/`; on Colab a Drive folder so a disconnected run resumes |
| `--env` | the interpreter preset of `configs/environment.yaml` (`colab_oneclick`, `colab_public`, `local`) |
| `--setup` | `colab` builds the three environments in this VM; `none` assumes they exist |
| `--proxy-asset` | URL or path of `proxies_of_record.zip` (default: the release asset) |
| `--hf-token-file` | a file with your Hugging Face token; default: `HF_TOKEN` in the environment, `<Drive>/secrets/hf_token.env`, or the saved login of `huggingface-cli login` |
| `--seeds`, `--module` | the sampling seeds (default 0); run one module only when the others are done |

The subject of the generated video is always the target's identity, because the generator is given the target's geometry as its
depth control; the sentence or `--scene` decides the background, and the clip decides the motion. The outputs land under
`<root>/cache/m3/<clip>/<target>/` (the backbone's video `A0_*.mp4` and ours `A3_*.mp4`) and the grader's table under
`<root>/reports/m3_s4/`.

### On Colab

- `colab_quickstart.ipynb`: Module 3 only, on the bundled example from the release asset, in one runtime (A100); the quickest way to see the sampler work and to compare its grading with the thesis's rows.
- `colab_full_pipeline.ipynb`: the line above in a one-click cell, with a settings form (target, caption, scene, a new subject's fields, Drive on or off); the three-part path below it runs the modules in three runtimes and is the fallback if an environment refuses to share a VM.
- `notebooks/development/`: the author's session records, bound to the author's Drive; not entry points.

**Target subjects.** Two category proxies exist in this repository, the ones the thesis validated: a red car (`vehicle`, rigid,
rendered along its path in Module 3) and a man in a grey jacket and dark trousers (`human`, articulated, no object render). The
proxy fixes the target's identity; the caption of a clip supplies only the scene. A proxy is made once per category, not per clip,
and almost all of it is automatic: a candidate image from a fixed prompt template, its mask from a detector and a segmenter, a 3D
reconstruction by SAM 3D Objects (rigid) or SAM 3D Body (articulated), the medoid over seeds, a scale-free normalisation. The one
reading taken by eye is which way the candidate faces, left or right, because placement defines the proxy's forward axis from it;
`scripts/run_m2b2.py --r1 --category <name> --image <png> --mask <png> --facing left|right` then `--promote` is that step. A subject beyond
these two is one command, `scripts/make_proxy.py --category bear --object "a brown bear" --kind rigid|human` (what `run_full.py` runs
for a new subject): it generates the candidates from the same template, masks them, picks one by a fixed rule, reconstructs it by
kind, records the facing as the template asked for it, and registers the subject in `configs/subjects.yaml`, after which every runner
accepts `--category bear`. This path is experimental: the thesis's measurements say nothing about how such a subject performs, and
the facing is assumed rather than read.

## Repository layout

```
configs/      stage_*.yaml, m2*.yaml, m3.yaml, m3_batch.yaml; calibration.lock.yaml (thresholds, human-owned)
src/          preprocess/ (Module 1), anchor/ (Module 2), generate/m3/ (Module 3)
scripts/      the pipeline runners (run_*.py), the Stage III checks the runners call (check_*.py), the page generator
docs/         the module contracts, the Module 2 closure record, the pipeline guide, the registered ablation sheet
tests/        contract and synthetic tests (no external model needed): python tests/<file>.py
results/      measured tables and figures
measurements/ the sealed records of the experiment and its instruments (json, md, csv); never videos or weights
notebooks/    the author's Colab session records (development/)
```

Secrets are read from files outside the tree and never written into it; weights, videos and caches are never committed.

## Citation and licence

Honours thesis, Australian National University, 2026. Licence: MIT ([`LICENSE`](LICENSE)). The external models keep their own
licences: OpenD4RT, ViPE, SAM 2.1, SAM 3D, FLUX.1, Depth Anything, MoGe-2, Wan 2.1, CoTracker3.

```bibtex
@thesis{zeng2026rrg,
  title  = {Recover, Retarget, Guide: camera and object motion transfer with a frozen video model},
  author = {Jiazhouxin, Zeng},
  school = {Australian National University},
  year   = {2026},
  type   = {Honours thesis}
}
```
