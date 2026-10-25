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

**On Colab.** `colab_quickstart.ipynb` is the entry point: it clones this repository (no token), installs the Module 3 environment,
downloads the release asset, takes the source clip from a path or a URL you give, runs the backbone and ours on one pair, grades
both and shows the videos. It needs an A100 (40 GB) runtime; Google Drive is optional (mount it to keep the outputs between
sessions). The four notebooks under `notebooks/development/` are the author's session records, bound to the author's Drive; they
are not entry points.

**On your own machine.** Module 3 needs a GPU with 40 GB (Wan 2.1 I2V 14B in fp8; bf16 needs 80 GB); Modules 1 and 2 run on
24 GB. Tokens: none for Module 3 (Wan 2.1 is public on Hugging Face); an `HF_TOKEN` with access to FLUX.1-dev for Module 2; no
GitHub token. The reference clips of the thesis are MiraData and DAVIS clips and are not redistributed; `scripts/bootstrap*.sh`
try the author's dataset and continue without it (`SKIP_VIDEOS=1` skips the attempt).

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
