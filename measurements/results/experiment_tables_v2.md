# Experiment set, v2 scoring (proper similarity) beside v1
v2 CSV sealed 2026-10-20T08:24:26 (875efed9eace); v1 sealed 2026-10-20T06:53:43 (8dd54f6f81bf). The alignment fix was made AFTER the v1 tables were read (brief v2, 2026-10-20).

## 1. Main, paired: ours (A3) vs A0, the 15 pairs, seed 0, v1 and v2

| Metric | better | v2 median A0 / ours | v1 W / L / T / excl | v2 W / L / T / excl | v1 p | v2 p | v2 Holm thr. | v2 significant |
|---|---|---|---|---|---|---|---|---|
| **CamRotErr (deg)** (primary) | lower | 3.590 (15) / 1.084 (15) | 15 / 0 / 0 / 0 | 15 / 0 / 0 / 0 | 0.0000 | 0.0000 | 0.025 | **yes** |
| **ObjMC motion-only (px)** (primary) | lower | 14.112 (15) / 21.579 (15) | 6 / 9 / 0 / 0 | 6 / 9 / 0 / 0 | 0.8491 | 0.8491 | 0.050 | no |
| RPE (deg/frame) | lower | 0.571 (15) / 0.242 (15) | 15 / 0 / 0 / 0 | 15 / 0 / 0 / 0 | exploratory | exploratory | - | - |
| ATE sim(3) | lower | 0.058 (15) / 0.021 (15) | 13 / 1 / 1 / 0 | 13 / 1 / 1 / 0 | exploratory | exploratory | - | - |
| RotErr, CameraCtrl (rad sum) | lower | 1.316 (15) / 0.397 (15) | 15 / 0 / 0 / 0 | 15 / 0 / 0 / 0 | exploratory | exploratory | - | - |
| TransErr, CameraCtrl (sq sum) | lower | 5.120 (15) / 0.474 (15) | 14 / 1 / 0 / 0 | 14 / 1 / 0 / 0 | exploratory | exploratory | - | - |
| ObjMC (px) | lower | 39.562 (15) / 43.838 (15) | 8 / 7 / 0 / 0 | 8 / 7 / 0 / 0 | exploratory | exploratory | - | - |
| Box-IoU | higher | 0.301 (15) / 0.208 (15) | 3 / 12 / 0 / 0 | 3 / 12 / 0 / 0 | exploratory | exploratory | - | - |
| ObjErr3D | lower | 0.181 (11) / 0.069 (15) | 10 / 5 / 0 / 0 | 8 / 3 / 0 / 4 | exploratory | exploratory | - | - |
| MF | higher | 0.693 (15) / 0.906 (15) | 14 / 1 / 0 / 0 | 14 / 1 / 0 / 0 | exploratory | exploratory | - | - |
| Direction cos | higher | 0.770 (15) / 0.987 (15) | 13 / 2 / 0 / 0 | 13 / 2 / 0 / 0 | exploratory | exploratory | - | - |
| IQ (CLIPScore) | higher | 0.324 (15) / 0.299 (15) | 1 / 14 / 0 / 0 | 1 / 14 / 0 / 0 | exploratory | exploratory | - | - |
| CLIP-IQA | higher | 0.755 (15) / 0.526 (15) | 2 / 13 / 0 / 0 | 2 / 13 / 0 / 0 | exploratory | exploratory | - | - |

- CamRotErr (deg): W-L v1 = 15, v2 = 15: direction unchanged.
- ObjMC motion-only (px): W-L v1 = -3, v2 = -3: direction unchanged.
- ObjErr3D (the one metric in the table that uses the scale-only fit): v1 n' = 15, v2 n' = 11 (4 pairs excluded: A0 has no valid alignment or is unscorable).

## 2. Secondary lines, v2

| Line | Metric | W / L / T / excl | n' | p one-sided | wins needed (alpha 0.05) |
|---|---|---|---|---|---|
| footprint-pass, n = 9 | CamRotErr (deg) | 9 / 0 / 0 / 0 | 9 | 0.0020 | 8 |
| footprint-pass, n = 9 | ObjMC motion-only (px) | 3 / 6 / 0 / 0 | 9 | 0.9102 | 8 |
| both gates, n = 4 (descriptive) | CamRotErr (deg) | 4 / 0 / 0 / 0 | 4 | - | not attainable |
| both gates, n = 4 (descriptive) | ObjMC motion-only (px) | 1 / 3 / 0 / 0 | 4 | - | not attainable |

## 3. Per pair, v2: the alignment record beside the primaries

| pair | footprint | depth | kreg<=0 | contact | heading unchecked | cam_scale v1 A0 / ours | v2 align A0 / ours | CamRotErr A0 | ours | result | ObjMC motion A0 | ours | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 000000000191.6.037/human | pass | fail | False | contact_out_of_view | False | 0.533 / 0.809 | ok / ok | 3.590 | 1.156 | win | 13.867 | 12.250 | win |
| 000000000200.1.006/human | pass | pass | False | in_contact | False | 0.754 / 0.656 | ok / ok | 6.108 | 1.084 | win | 44.246 | 36.041 | win |
| 000000000217.0.006/vehicle | fail | pass | not_provided | in_contact | False | 51.221 / 1.040 | ok / ok | 2.115 | 0.506 | win | 11.038 | 8.913 | win |
| 000000000221.7.011/human | pass | pass | False | in_contact | False | 26.438 / 0.822 | ok / ok | 6.857 | 2.824 | win | 3.888 | 17.640 | loss |
| 000000000252.1.018/vehicle | fail | pass | not_provided | in_contact | True | 9.637 / 1.503 | ok / ok | 5.767 | 0.862 | win | 39.943 | 27.754 | win |
| 000000000252.2.007/human | fail | fail | False | in_contact | False | 20.337 / 1.027 | ok / ok | 10.035 | 1.873 | win | 7.143 | 17.636 | loss |
| 000000000256.3.010/human | pass | pass | False | in_contact | False | -0.556 / 0.807 | unalignable / ok | 1.749 | 1.003 | win | 11.769 | 18.922 | loss |
| 000000000265.4.002/human | pass | fail | False | contact_out_of_view | False | 23.446 / 0.853 | ok / ok | 2.854 | 0.854 | win | 15.216 | 26.485 | loss |
| 000000000362.2.005/human | fail | fail | True | in_contact | True | -3.089 / 8.201 | unalignable / ok | 7.853 | 5.983 | win | 8.455 | 53.689 | loss |
| 000000000391.0.016/human | pass | fail | False | contact_out_of_view | False | 2.787 / 0.986 | ok / ok | 2.192 | 0.577 | win | 28.060 | 9.481 | win |
| 000000000396.3.002/vehicle | fail | fail | not_provided | in_contact | True | 11.960 / 0.693 | ok / ok | 2.635 | 0.542 | win | 20.508 | 14.921 | win |
| 000000000404.1.006/human | fail | fail | False | in_contact | False | 8.883 / 1.666 | ok / ok | 9.975 | 1.819 | win | 14.112 | 21.579 | loss |
| 000000000431.5.009/human | pass | pass | True | in_contact | False | -6.129 / 0.685 | unalignable / ok | 1.822 | 0.362 | win | 4.685 | 62.369 | loss |
| camel/human | pass | fail | True | in_contact | False | -0.309 / 0.360 | unalignable / ok | 1.978 | 1.171 | win | 28.730 | 35.777 | loss |
| horsejump-high/human | pass | fail | True | in_contact | False | 0.129 / 0.308 | ok / ok | 21.735 | 5.666 | win | 35.988 | 82.383 | loss |

## 4. Cross-method, MiraData, v2 (descriptive; D9 labels)

(A) On ours' subset: the 13 MiraData pairs, seed 0.

| Method | n videos | CamRotErr (deg) (lower) | ObjMC motion-only (px) (lower) | RPE (deg/frame) (lower) | ATE sim(3) (lower) | RotErr, CameraCtrl (rad sum) (lower) | TransErr, CameraCtrl (sq sum) (lower) | ObjMC (px) (lower) | Box-IoU (higher) | ObjErr3D (lower) | MF (higher) | Direction cos (higher) | IQ (CLIPScore) (higher) | CLIP-IQA (higher) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our method (A3) | 13 | 1.003 | 18.922 | 0.215 | 0.022 | 0.368 | 1.574 | 43.838 | 0.207 | 0.069 | 0.906 | 0.987 | 0.299 | 0.527 |
| No guidance (Wan2.1 I2V, A0) | 13 | 3.590 | 13.867 | 0.571 | 0.126 | 1.316 | 6.823 | 39.562 | 0.301 | 0.179 | 0.550 | 0.757 | 0.324 | 0.788 |
| No guidance (CogVideoX-5B), different prompt subject | 13 | 4.005 | 21.747 | 0.537 | 0.116 | 1.468 | 6.974 | 52.067 | 0.159 | 0.280 | 0.678 | 0.090 | 0.323 | 0.821 |
| DiTFlow (CogVideoX-5B), different prompt subject | 13 | 3.259 | 13.022 | 0.465 | 0.028 | 1.194 | 5.320 | 40.408 | 0.211 | 0.061 | 0.807 | 0.957 | 0.326 | 0.752 |
| SMM (re-impl., CogVideoX-5B), different prompt subject | 13 | 2.150 | 21.317 | 0.270 | 0.029 | 0.788 | 1.496 | 34.433 | 0.166 | 0.044 | 0.870 | 0.927 | 0.326 | 0.699 |
| MOFT (re-impl., CogVideoX-5B), different prompt subject | 13 | 3.484 | 37.101 | 0.615 | 0.031 | 1.277 | 7.206 | 68.066 | 0.112 | 0.162 | 0.769 | 0.869 | 0.350 | 0.708 |
| No guidance (Wan2.1 T2V 1.3B), different prompt subject | 13 | 4.872 | 8.746 | 0.566 | 0.137 | 1.786 | 8.230 | 44.727 | 0.167 | 0.168 | 0.678 | -0.074 | 0.337 | 0.881 |
| Follow-Your-Motion (Wan2.1 T2V 1.3B), different prompt subject | 9 | 1.100 | 5.843 | 0.182 | 0.012 | 0.403 | 0.155 | 13.158 | 0.447 | 0.022 | 0.928 | 0.991 | 0.314 | 0.445 |
| Reference value (CLIP-IQA only; not a ceiling) | 13 | - | - | - | - | - | - | - | - | - | - | - | - | 0.450 |

(B) Each row on its own MiraData pairs (baselines: 19; ours: 13).

| Method | n videos | CamRotErr (deg) (lower) | ObjMC motion-only (px) (lower) | RPE (deg/frame) (lower) | ATE sim(3) (lower) | RotErr, CameraCtrl (rad sum) (lower) | TransErr, CameraCtrl (sq sum) (lower) | ObjMC (px) (lower) | Box-IoU (higher) | ObjErr3D (lower) | MF (higher) | Direction cos (higher) | IQ (CLIPScore) (higher) | CLIP-IQA (higher) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our method (A3) | 13 | 1.003 | 18.922 | 0.215 | 0.022 | 0.368 | 1.574 | 43.838 | 0.207 | 0.069 | 0.906 | 0.987 | 0.299 | 0.527 |
| No guidance (Wan2.1 I2V, A0) | 13 | 3.590 | 13.867 | 0.571 | 0.126 | 1.316 | 6.823 | 39.562 | 0.301 | 0.179 | 0.550 | 0.757 | 0.324 | 0.788 |
| No guidance (CogVideoX-5B), different prompt subject | 19 | 5.053 | 16.592 | 0.582 | 0.116 | 1.852 | 6.206 | 52.067 | 0.108 | 0.268 | 0.676 | -0.192 | 0.323 | 0.783 |
| DiTFlow (CogVideoX-5B), different prompt subject | 19 | 3.661 | 10.609 | 0.465 | 0.034 | 1.342 | 5.320 | 44.039 | 0.211 | 0.056 | 0.839 | 0.960 | 0.326 | 0.752 |
| SMM (re-impl., CogVideoX-5B), different prompt subject | 19 | 2.532 | 22.945 | 0.415 | 0.035 | 0.928 | 1.987 | 52.058 | 0.185 | 0.048 | 0.860 | 0.893 | 0.329 | 0.699 |
| MOFT (re-impl., CogVideoX-5B), different prompt subject | 19 | 5.000 | 27.848 | 0.639 | 0.031 | 1.833 | 6.845 | 62.708 | 0.178 | 0.079 | 0.769 | 0.870 | 0.350 | 0.696 |
| No guidance (Wan2.1 T2V 1.3B), different prompt subject | 19 | 5.582 | 10.238 | 0.612 | 0.137 | 2.046 | 7.944 | 47.242 | 0.167 | 0.143 | 0.678 | 0.058 | 0.345 | 0.881 |
| Follow-Your-Motion (Wan2.1 T2V 1.3B), different prompt subject | 14 | 1.075 | 6.095 | 0.183 | 0.012 | 0.394 | 0.112 | 12.856 | 0.533 | 0.020 | 0.934 | 0.992 | 0.302 | 0.473 |
| Reference value (CLIP-IQA only; not a ceiling) | 19 | - | - | - | - | - | - | - | - | - | - | - | - | 0.416 |

### Ours vs Follow-Your-Motion on the same 9 pairs (FYM fine-tuned per video, different backbone (Wan2.1 T2V 1.3B), different prompt subject: reported, not claimed)

Restricted to the 9 MiraData pairs where FYM generated a video.

| Method | n videos | CamRotErr (deg) (lower) | ObjMC motion-only (px) (lower) | RPE (deg/frame) (lower) | ATE sim(3) (lower) | RotErr, CameraCtrl (rad sum) (lower) | TransErr, CameraCtrl (sq sum) (lower) | ObjMC (px) (lower) | Box-IoU (higher) | ObjErr3D (lower) | MF (higher) | Direction cos (higher) | IQ (CLIPScore) (higher) | CLIP-IQA (higher) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our method (A3) | 9 | 1.003 | 18.922 | 0.200 | 0.022 | 0.368 | 0.447 | 39.322 | 0.207 | 0.069 | 0.897 | 0.987 | 0.299 | 0.526 |
| Follow-Your-Motion (Wan2.1 T2V 1.3B), different prompt subject | 9 | 1.100 | 5.843 | 0.182 | 0.012 | 0.403 | 0.155 | 13.158 | 0.447 | 0.022 | 0.928 | 0.991 | 0.314 | 0.445 |
| Reference value (CLIP-IQA only; not a ceiling) | 9 | - | - | - | - | - | - | - | - | - | - | - | - | 0.450 |

| pair | CamRotErr (deg) FYM / ours | result | ObjMC motion-only (px) FYM / ours | result |
|---|---|---|---|---|
| 000000000191.6.037/human | 2.022 / 1.156 | win | 1.715 / 12.250 | loss |
| 000000000200.1.006/human | 1.394 / 1.084 | win | 14.438 / 36.041 | loss |
| 000000000217.0.006/vehicle | 0.660 / 0.506 | win | 5.843 / 8.913 | loss |
| 000000000221.7.011/human | 1.177 / 2.824 | loss | 6.346 / 17.640 | loss |
| 000000000252.1.018/vehicle | 1.050 / 0.862 | win | 10.034 / 27.754 | loss |
| 000000000256.3.010/human | 0.866 / 1.003 | loss | 4.940 / 18.922 | loss |
| 000000000265.4.002/human | 1.100 / 0.854 | win | 17.697 / 26.485 | loss |
| 000000000391.0.016/human | 0.451 / 0.577 | loss | 1.610 / 9.481 | loss |
| 000000000404.1.006/human | 1.189 / 1.819 | loss | 1.503 / 21.579 | loss |

CamRotErr (deg), ours vs FYM: 5 wins / 4 losses / 0 ties / 0 excluded (descriptive).

ObjMC motion-only (px), ours vs FYM: 0 wins / 9 losses / 0 ties / 0 excluded (descriptive).

FYM did not produce a video on 4 of the 13: 000000000252.2.007/human (FYM failed at collect_attn: empty_head_group (layers [13] ha), 000000000362.2.005/human (FYM failed at collect_attn: empty_head_group (layers [13] ha), 000000000396.3.002/vehicle (FYM failed at collect_attn: empty_head_group (layers [13] ha), 000000000431.5.009/human (FYM failed at collect_attn: empty_head_group (layers [13] ha)

Yield, v2 (videos generated / scored):

- Our method (A3): generated 13 of 13, scored 13, no valid alignment 0
- No guidance (Wan2.1 I2V, A0): generated 13 of 13, scored 13, no valid alignment 3
- No guidance (CogVideoX-5B), different prompt subject: generated 19 of 19, scored 19, no valid alignment 4
- DiTFlow (CogVideoX-5B), different prompt subject: generated 19 of 19, scored 19, no valid alignment 1
- SMM (re-impl., CogVideoX-5B), different prompt subject: generated 19 of 19, scored 19, no valid alignment 5
- MOFT (re-impl., CogVideoX-5B), different prompt subject: generated 19 of 19, scored 19, no valid alignment 3
- No guidance (Wan2.1 T2V 1.3B), different prompt subject: generated 19 of 19, scored 18, no valid alignment 5
- Follow-Your-Motion (Wan2.1 T2V 1.3B), different prompt subject: generated 14 of 19, scored 14, no valid alignment 0

## 5. Cross-method, DAVIS, v2: horsejump-high (baselines seed 1)

horsejump-high/human: ours seed 0; baselines seed 1, different prompt subject.

| Method | n videos | CamRotErr (deg) (lower) | ObjMC motion-only (px) (lower) | RPE (deg/frame) (lower) | ATE sim(3) (lower) | RotErr, CameraCtrl (rad sum) (lower) | TransErr, CameraCtrl (sq sum) (lower) | ObjMC (px) (lower) | Box-IoU (higher) | ObjErr3D (lower) | MF (higher) | Direction cos (higher) | IQ (CLIPScore) (higher) | CLIP-IQA (higher) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our method (A3) | 1 | 5.666 | 82.383 | 0.781 | 0.009 | 2.077 | 0.056 | 70.097 | 0.265 | 0.320 | 0.872 | 1.000 | 0.256 | 0.457 |
| No guidance (Wan2.1 I2V, A0) | 1 | 21.735 | 35.988 | 2.328 | 0.011 | 7.966 | 0.152 | 52.144 | 0.315 | 0.332 | 0.923 | 0.999 | 0.311 | 0.490 |
| No guidance (CogVideoX-5B), different prompt subject | 1 | 14.351 | 46.885 | 1.618 | 0.050 | 5.260 | 0.274 | 50.095 | 0.320 | 0.329 | 0.864 | 0.981 | 0.298 | 0.687 |
| DiTFlow (CogVideoX-5B), different prompt subject | 1 | 6.722 | 51.289 | 0.934 | 0.045 | 2.464 | 0.561 | 57.105 | 0.301 | - | 0.935 | 1.000 | 0.261 | 0.464 |
| SMM (re-impl., CogVideoX-5B), different prompt subject | 1 | 5.460 | 51.281 | 0.838 | 0.016 | 2.001 | 0.178 | 53.116 | 0.313 | 0.226 | 0.911 | 1.000 | 0.281 | 0.592 |
| MOFT (re-impl., CogVideoX-5B), different prompt subject | 1 | 4.235 | 69.481 | 1.029 | 0.036 | 1.552 | 0.364 | 83.663 | 0.216 | - | 0.909 | 0.958 | 0.276 | 0.688 |
| No guidance (Wan2.1 T2V 1.3B), different prompt subject | 1 | 16.254 | 41.333 | 1.789 | 0.030 | 5.957 | 0.622 | 47.282 | 0.263 | - | 0.887 | -0.999 | 0.250 | 0.302 |
| Follow-Your-Motion (Wan2.1 T2V 1.3B), different prompt subject | 1 | 1.769 | 16.681 | 0.502 | 0.020 | 0.648 | 0.182 | 19.828 | 0.540 | 0.239 | 0.958 | 1.000 | 0.288 | 0.659 |
| Reference value (CLIP-IQA only; not a ceiling) | 1 | - | - | - | - | - | - | - | - | - | - | - | - | 0.686 |

CLIP-IQA: the reference row is a reference value, not a ceiling (generated rows can score above it).
