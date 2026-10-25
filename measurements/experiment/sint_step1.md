# Module 3 S-int step 1 — frame-0 lift (report; hard stop: John reviews before S-int step 2)

Depth model `Ruicheng/moge-2-vitl-normal`, square-pixel input, FOV imposed. GPU: NVIDIA A100-SXM4-80GB.

| pair | disp. | F_pair / T_l | grid | valid frac | k_bg [CI95] | CI rel | band px / blocks | alpha-depth / k_bg-depth: band, object | footprint IoU (R2 IoU) | G_tar depth / image depth | sky patches | n_bg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bmx-bumps/human | F_pair 9 < min_frames 21 |

Gates: footprint IoU against `lift.obj_place_iou_min`, depth agreement against `lift.obj_depth_rel_tol`; `g_gauge_scalar` is record-only (`int.kbg_ci_rel_max` is null). A failing gate is a finding with its numbers above; nothing was adjusted.

## Decisions needed from John

1. Review k_bg and its interval per pair; decide whether `int.kbg_ci_rel_max` gets a value.
2. Any pair failing the footprint or depth gate: recorded refusal for Module 3, or investigate.
3. Pre-registered expectation 2 (brief §14): alpha-bridged depth agrees with k_bg depth over the band within a few percent — read the column.
