# Results

Measured outputs of the pipeline of record (sampler: Wan 2.1 I2V 14B, UniPC 50 steps, shift 3.0, CFG 5.0, 512-token text; guidance at
sampling steps 18..33). Every number here is one the instruments produced; nothing is quoted from a draft. Videos are not stored in the
repository: the batch export (every video, the source clip, the anchor, an index) is attached to the release that this folder's tag names.

## Reference example: 000000000265.2.005

Two retargetings of one reference clip (a man walking along a river, camera following): a red car on scrubland and a man on a rocky riverbank.
Anchors (the synthesised first frames): `figures/anchor_265.2.005_vehicle.png`, `figures/anchor_265.2.005_human.png`.

### Camera channel (CoTracker grader, 33 frames, seeds 0 and 1)

| | vehicle A0 (no guidance) | vehicle A3 (ours) | human A0 | human A3 (sky) | human A3 (sky, water excluded) |
|---|---|---|---|---|---|
| outside cosine vs source (direction of the background against the reference's tracks) | −0.09 / −0.41 | **0.84 / 0.98** | – | – | uncovered part 0.98 / 0.99 |
| inside ratio to own expectation (magnitude) | 1.01 / 1.07 | 0.95 / 0.94 | 8.8 / 6.3 | 0.97 / 0.98 | 1.06 / 1.16 |
| object screen displacement, patches (reference: 0.44 horse, 0.84 person) | 8.7 / 8.9 | **0.52 / 0.63** | 2.5 / 1.6 | 1.9 / – | 3.4 / 1.7 |

Without guidance the model moves the background by the expected amount in the wrong direction and drives the car across the frame; with it
the background follows the reference camera and the car holds its screen position as the reference's horse does.

### Subject channel (the open problem)

| | human A0 | human A3 (sky) | human A3 (sky, water excluded) |
|---|---|---|---|
| gait oscillation, patches | 0.30 / – | 0.14 / – | 0.15 / 0.18 |
| walk against rock-only ground, patches (registered: ≥ 10 = real) | unscorable | unscorable | – / 11.0 |
| water non-rigid residual, median (reference 0.88) | 10.8 / 8.7 | 0.09 / 0.11 (frozen) | 11.0 / 8.9 |

The person keeps the model's own gait, and water inside the correction mask freezes unless it is excluded from the mask. On the vehicle pair
the object correction draws the render's copy of the car beside the model's own from frame 8 on (both seeds). The subject's articulation is
not transferred by the method as it stands.

Full record with every intermediate reading: `../docs/report_m3_s2_first.md`, section "Re-based first light (2026-10-14)" onward;
per-video JSON in `../measurements/m3_s4/a0_motion.json` and `../measurements/m3_s4/mask_centroid_*.json`.

## The batch (six pairs)

Configuration: `../configs/m3_batch.yaml`. Pairs: 035.1.003/vehicle (flagged: its camera failed the gate, no guided run), 265.2.003/vehicle,
265.2.005/vehicle, 265.2.005/human, 370.1.008/human, 404.1.004/vehicle. The cross-method table (CogVideoX, DiTFlow, Wan alone, ours) is
computed by the experiment's grader from the exported videos; it is added here with its pair count, seeds, aggregation and alignment stated
once that computation is documented.

## Figures

- `figures/panel2_guide_vehicle_preview.png`, `figures/panel2_guide_human_nodyn_preview.png`: rendered target, region mask, corrected and
  uncorrected frames, and the sampling schedule with the correction band (made from the Euler-era runs; regenerated from the batch of record
  with `scripts/make_poster_panel2_guide.py`).
