# Experiment set, Module 3 on the frozen handoffs: outcome per pair (record of 2026-10-20; m3_set_records.json is the per-pair record, run_all.json the append-only log)

16 main pairs (configs/experiment_handoffs.yaml); 21 frames, seed 0, both arms (A0 backbone, A3 of record per category); the census camera of record (camera C + BA, source gauge, no bridge); grading against the set's canonical cuts (the first pass had read the flat originals: rescored). No threshold adjusted.

| outcome | n | pairs |
|---|---|---|
| complete: both arms, graded, exported | 15 | 000000000191.6.037/human, 000000000200.1.006/human, 000000000217.0.006/vehicle, 000000000221.7.011/human, 000000000252.1.018/vehicle, 000000000252.2.007/human, 000000000256.3.010/human, 000000000265.4.002/human, 000000000362.2.005/human, 000000000391.0.016/human, 000000000396.3.002/vehicle, 000000000404.1.006/human, 000000000431.5.009/human, camel/human, horsejump-high/human |
| refused at the lift (F_pair below lock min_frames 21) | 1 | bmx-bumps/human (t_a 13: 11 source frames after t_a, F_pair 9) |
| follow-up row (R2 not adopted), not run | 1 | hockey/human |

## Module 3 integration gates on the completed pairs (recorded, the pair runs on; lock m3.lift)

| gate | threshold | pass | fail (pairs) |
|---|---|---|---|
| g_obj_footprint: rendered G_tar vs anchor mask IoU | >= 0.7 | 9 | 6: 000000000217.0.006/vehicle, 000000000252.1.018/vehicle, 000000000252.2.007/human, 000000000362.2.005/human, 000000000396.3.002/vehicle, 000000000404.1.006/human |
| g_obj_depth: G_tar visible depth vs k_bg image depth, relative | <= 0.15 | 6 | 9: 000000000191.6.037/human, 000000000252.2.007/human, 000000000265.4.002/human, 000000000362.2.005/human, 000000000391.0.016/human, 000000000396.3.002/vehicle, 000000000404.1.006/human, camel/human, horsejump-high/human |
| both | | 4: 000000000200.1.006/human, 000000000221.7.011/human, 000000000256.3.010/human, 000000000431.5.009/human | |

| pair | R2 IoU (handoff) | footprint IoU | depth rel diff | k_bg | cam path | rot deg | obj net disp | coverage first / last | note |
|---|---|---|---|---|---|---|---|---|---|
| 000000000191.6.037/human | 0.816 | 0.816 pass | 0.624 fail | 0.198 | 0.704 | 8.4 | 0.634 | 0.91 / 0.76 |  |
| 000000000200.1.006/human | 0.718 | 0.718 pass | 0.110 pass | 0.362 | 1.144 | 0.8 | 0.595 | 0.95 / 0.95 |  |
| 000000000217.0.006/vehicle | 0.564 | 0.564 fail | 0.073 pass | 0.118 | 1.379 | 6.7 | 1.152 | 0.90 / 0.51 |  |
| 000000000221.7.011/human | 0.850 | 0.850 pass | 0.140 pass | 0.244 | 1.325 | 15.3 | 0.893 | 0.97 / 0.73 |  |
| 000000000252.1.018/vehicle | 0.642 | 0.642 fail | 0.114 pass | 0.156 | 2.265 | 4.4 | 1.894 | 0.79 / 0.44 |  |
| 000000000252.2.007/human | 0.631 | 0.631 fail | 0.307 fail | 0.187 | 1.827 | 14.1 | 2.496 | 0.53 / 0.30 |  |
| 000000000256.3.010/human | 0.789 | 0.789 pass | 0.035 pass | 0.168 | 1.410 | 2.4 | 1.010 | 0.73 / 0.58 |  |
| 000000000265.4.002/human | 0.798 | 0.798 pass | 0.206 fail | 0.135 | 0.934 | 3.6 | 0.646 | 0.97 / 0.77 |  |
| 000000000362.2.005/human | 0.418 | 0.418 fail | 2.122 fail | 0.554 | 0.824 | 14.0 | 0.533 | 0.75 / 0.58 |  |
| 000000000391.0.016/human | 0.852 | 0.852 pass | 0.955 fail | 0.311 | 0.589 | 1.6 | 0.398 | 0.80 / 0.84 |  |
| 000000000396.3.002/vehicle | 0.655 | 0.655 fail | 0.154 fail | 0.072 | 1.040 | 7.2 | 0.898 | 0.83 / 0.47 | latent masks met in 32 cells (mass 0.98): background zeroed there by the render pixel rule |
| 000000000404.1.006/human | 0.693 | 0.693 fail | 0.515 fail | 0.173 | 1.920 | 1.4 | 1.587 | 0.83 / 0.67 |  |
| 000000000431.5.009/human | 0.796 | 0.796 pass | 0.056 pass | 0.261 | 1.012 | 1.7 | 0.676 | 0.93 / 0.85 |  |
| camel/human | 0.900 | 0.900 pass | 0.231 fail | 0.615 | 0.357 | 6.4 | 0.664 | 0.92 / 0.73 |  |
| horsejump-high/human | 0.854 | 0.854 pass | 0.160 fail | 0.505 | 0.339 | 29.1 | 2.207 | 0.90 / 0.43 |  |

## Reading

- The six pairs of record all passed both lift gates (284 was refused at the footprint gate and never ran). On the census set the gates were applied as record-only flags, as the lift runner does: 6 of 15 fail the footprint gate and 9 of 15 the depth gate; 4 pass both.
- Which rows the main table scores is the reviewer's ruling (D8 in the brief of 2026-10-20): every completed pair with its flags shown; footprint-pass pairs only; or both-gates-pass pairs only. The plan's D2 wording ('pairs that fail a gate appear as flagged rows, not scored ones') was written for Module 1's gates.
- The depth gate compares the reconstructed object's depth with the k_bg-bridged image depth under the anchor mask; its failures on humans with footprint pass (191, 265.4.002, 391, camel, horsejump-high) say the Body mesh sits at a different depth from the anchor's own depth estimate, the known gauge limitation of the human path (contract §3).
- bmx-bumps: the census recorded 24 frames after t_a; the handoff has 11 (t_a 13 in a 24-frame clip). The census field was the clip length, not the frames after t_a; a recorded error of the census, not of Module 3.

## Corrections (reviewer 2026-10-20)

- Ours completed **13** MiraData pairs (not 12): 191, 200, 217, 221, 252.1.018, 252.2.007, 256, 265.4.002, 362, 391, 396, 404, 431. The baselines'
  subset median uses the same 13.
- Per-row flag columns of record: footprint, depth, the R1 `k_reg <= 0` flag, contact label, heading unchecked, latent-mask overlap.
- D8: all 15 scored, flags per row (`docs/plan_experiment_v1.md`, section D8); blindness statement in `blind_video_verdicts_2026-10-20.md`.

## Records added (reviewer 2026-10-20): beta = 1 and s_delta = 1 on the set, with the evidence beside them

- rho, the census band ratio (camera C + BA against the lift at t_a), lies in 0.952-1.035 on every frozen pair: the camera and the lift agree
  within 3.5 % on the completed pairs, the evidence for beta = 1 (values per pair in m3_set_records.md).
- The six-pair formula s_delta = Z^M_c / Z^L_gnd,base (scripts/check_m3_bridge_interval.py --camera ba) is defined on 10 of the 15:

| pair | rho (census band ratio) | s_delta, six-pair formula | depth gate | reviewer's prediction (>5 % from 1 <-> depth fail) |
|---|---|---|---|---|
| 000000000191.6.037/human | 0.968 | 0.635 | fail | agrees |
| 000000000200.1.006/human | 1.035 | 0.923 | pass | disagrees |
| 000000000217.0.006/vehicle | 0.989 | 0.744 | pass | disagrees |
| 000000000256.3.010/human | 0.980 | 1.420 | pass | disagrees |
| 000000000391.0.016/human | 0.977 | 0.510 | fail | agrees |
| 000000000396.3.002/vehicle | 1.011 | 1.319 | fail | agrees |
| 000000000404.1.006/human | 1.017 | 0.559 | fail | agrees |
| 000000000431.5.009/human | 1.005 | 1.078 | pass | disagrees |
| camel/human | 0.986 | 0.763 | fail | agrees |
| horsejump-high/human | 1.033 | 1.409 | fail | agrees |

  The prediction holds on 6 of 10: the four exceptions all pass the depth gate with s_delta away from 1: 217.0.006 (0.74) and 256.3.010 (1.42)
  far from it, 200.1.006 (0.92) and 431.5.009 (1.08) at 8 %. Reported; s_delta stays 1 on the set.
- Body integrity (scripts/check_m2e_body_integrity.py): every regenerated human R2 with an archived predecessor equals it to float precision
  (max |dv| <= 2e-6 m on 8 pairs); the other human pairs had no earlier reconstruction (principal-point refusals before D7).
- Latent-mask overlap: 32 cells on 396.3.002 (footprint-flagged); 0 on every other pair.
