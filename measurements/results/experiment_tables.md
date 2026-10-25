# Experiment set: ours vs backbone and cross-method (CSV sealed 2026-10-20T06:53:43, 8dd54f6f81bf)

## 1. Main, paired: ours (A3) vs its backbone (A0), the 15 pairs, seed 0

Win: ours better after rounding to 3 decimals. Tie: removed from n'. Ours unscorable, A0 scorable: loss. A0 unscorable: pair excluded. p = P(X >= wins | n', 1/2). Holm across the two primaries.

| Metric | better | median A0 (n) | median ours (n) | W / L / T / excl | n' | p one-sided | Holm threshold | significant |
|---|---|---|---|---|---|---|---|---|
| **CamRotErr (deg)** (primary) | lower | 3.590 (15) | 1.084 (15) | 15 / 0 / 0 / 0 | 15 | 0.0000 | 0.025 | **yes** |
| **ObjMC motion-only (px)** (primary) | lower | 14.112 (15) | 21.579 (15) | 6 / 9 / 0 / 0 | 15 | 0.8491 | 0.050 | no |
| RPE (deg/frame) | lower | 0.571 (15) | 0.242 (15) | 15 / 0 / 0 / 0 | 15 | exploratory | - | - |
| ATE sim(3) | lower | 0.058 (15) | 0.021 (15) | 13 / 1 / 1 / 0 | 14 | exploratory | - | - |
| RotErr, CameraCtrl (rad sum) | lower | 1.316 (15) | 0.397 (15) | 15 / 0 / 0 / 0 | 15 | exploratory | - | - |
| TransErr, CameraCtrl (sq sum) | lower | 5.120 (15) | 0.474 (15) | 14 / 1 / 0 / 0 | 15 | exploratory | - | - |
| ObjMC (px) | lower | 39.562 (15) | 43.838 (15) | 8 / 7 / 0 / 0 | 15 | exploratory | - | - |
| Box-IoU | higher | 0.301 (15) | 0.208 (15) | 3 / 12 / 0 / 0 | 15 | exploratory | - | - |
| ObjErr3D | lower | 0.176 (15) | 0.069 (15) | 10 / 5 / 0 / 0 | 15 | exploratory | - | - |
| MF | higher | 0.693 (15) | 0.906 (15) | 14 / 1 / 0 / 0 | 15 | exploratory | - | - |
| Direction cos | higher | 0.770 (15) | 0.987 (15) | 13 / 2 / 0 / 0 | 15 | exploratory | - | - |
| IQ (CLIPScore) | higher | 0.324 (15) | 0.299 (15) | 1 / 14 / 0 / 0 | 15 | exploratory | - | - |
| CLIP-IQA | higher | 0.755 (15) | 0.526 (15) | 2 / 13 / 0 / 0 | 15 | exploratory | - | - |

CamRotErr (deg): at n' = 15, 12 wins needed its Holm threshold 0.025.

ObjMC motion-only (px): at n' = 15, 12 wins needed its Holm threshold 0.050.

## 2. Secondary lines (stated before the run)

| Line | Metric | W / L / T / excl | n' | p one-sided | wins needed (alpha 0.05) |
|---|---|---|---|---|---|
| footprint-pass, n = 9 | CamRotErr (deg) | 9 / 0 / 0 / 0 | 9 | 0.0020 | 8 |
| footprint-pass, n = 9 | ObjMC motion-only (px) | 3 / 6 / 0 / 0 | 9 | 0.9102 | 8 |
| both gates, n = 4 (descriptive) | CamRotErr (deg) | 4 / 0 / 0 / 0 | 4 | - | not attainable |
| both gates, n = 4 (descriptive) | ObjMC motion-only (px) | 1 / 3 / 0 / 0 | 4 | - | not attainable |

CamRotErr (deg): primary W-L = 15, footprint-pass W-L = 9: same sign.

ObjMC motion-only (px): primary W-L = -3, footprint-pass W-L = -3: same sign.

## 3. Per pair, main set (flags disclosed, never dropped)

| pair | footprint | depth | kreg<=0 | contact | heading unchecked | overlap cells | A-G2b over (ref / A0 / ours) | CamRotErr A0 | ours | result | ObjMC motion A0 | ours | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 000000000191.6.037/human | pass | fail | not_provided | contact_out_of_view | False | 0 | False / False / False | 3.590 | 1.156 | win | 13.867 | 12.250 | win |
| 000000000200.1.006/human | pass | pass | not_provided | in_contact | False | 0 | False / False / False | 6.108 | 1.084 | win | 44.246 | 36.041 | win |
| 000000000217.0.006/vehicle | fail | pass | not_provided | in_contact | False | 0 | False / False / False | 2.115 | 0.506 | win | 11.038 | 8.913 | win |
| 000000000221.7.011/human | pass | pass | not_provided | in_contact | False | 0 | False / False / False | 6.857 | 2.824 | win | 3.888 | 17.640 | loss |
| 000000000252.1.018/vehicle | fail | pass | not_provided | in_contact | True | 0 | False / False / False | 5.767 | 0.862 | win | 39.943 | 27.754 | win |
| 000000000252.2.007/human | fail | fail | not_provided | in_contact | False | 0 | False / False / False | 10.035 | 1.873 | win | 7.143 | 17.636 | loss |
| 000000000256.3.010/human | pass | pass | not_provided | in_contact | False | 0 | False / False / False | 1.749 | 1.003 | win | 11.769 | 18.922 | loss |
| 000000000265.4.002/human | pass | fail | not_provided | contact_out_of_view | False | 0 | False / False / False | 2.854 | 0.854 | win | 15.216 | 26.485 | loss |
| 000000000362.2.005/human | fail | fail | True | in_contact | True | 0 | False / False / False | 7.853 | 5.983 | win | 8.455 | 53.689 | loss |
| 000000000391.0.016/human | pass | fail | not_provided | contact_out_of_view | False | 0 | False / False / False | 2.192 | 0.577 | win | 28.060 | 9.481 | win |
| 000000000396.3.002/vehicle | fail | fail | not_provided | in_contact | True | 32 | False / False / True | 2.635 | 0.542 | win | 20.508 | 14.921 | win |
| 000000000404.1.006/human | fail | fail | not_provided | in_contact | False | 0 | False / False / False | 9.975 | 1.819 | win | 14.112 | 21.579 | loss |
| 000000000431.5.009/human | pass | pass | True | in_contact | False | 0 | False / False / False | 1.822 | 0.362 | win | 4.685 | 62.369 | loss |
| camel/human | pass | fail | True | in_contact | False | 0 | False / False / False | 1.978 | 1.171 | win | 28.730 | 35.777 | loss |
| horsejump-high/human | pass | fail | True | in_contact | False | 0 | False / False / False | 21.735 | 5.666 | win | 35.988 | 82.383 | loss |

Unscorable in the main set: none.

## 4. Cross-method, MiraData (descriptive; D9: baselines are the existing caption-prompt runs)

(A) On ours' subset: the 13 MiraData pairs of the set, seed 0.

| Method | n videos | CamRotErr (deg) (lower) | ObjMC motion-only (px) (lower) | RPE (deg/frame) (lower) | ATE sim(3) (lower) | RotErr, CameraCtrl (rad sum) (lower) | TransErr, CameraCtrl (sq sum) (lower) | ObjMC (px) (lower) | Box-IoU (higher) | ObjErr3D (lower) | MF (higher) | Direction cos (higher) | IQ (CLIPScore) (higher) | CLIP-IQA (higher) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our method (A3) | 13 | 1.003 | 18.922 | 0.215 | 0.022 | 0.368 | 1.574 | 43.838 | 0.207 | 0.069 | 0.906 | 0.987 | 0.299 | 0.527 |
| No guidance (Wan2.1 I2V, A0) | 13 | 3.590 | 13.867 | 0.571 | 0.126 | 1.316 | 6.823 | 39.562 | 0.301 | 0.176 | 0.550 | 0.757 | 0.324 | 0.788 |
| No guidance (CogVideoX-5B), different prompt subject | 13 | 4.005 | 21.747 | 0.537 | 0.116 | 1.468 | 6.974 | 52.067 | 0.159 | 0.293 | 0.678 | 0.090 | 0.323 | 0.821 |
| DiTFlow (CogVideoX-5B), different prompt subject | 13 | 3.259 | 13.022 | 0.465 | 0.028 | 1.194 | 5.320 | 40.408 | 0.211 | 0.073 | 0.807 | 0.957 | 0.326 | 0.752 |
| SMM (re-impl., CogVideoX-5B), different prompt subject | 13 | 2.150 | 21.317 | 0.270 | 0.029 | 0.788 | 1.496 | 34.433 | 0.166 | 0.062 | 0.870 | 0.927 | 0.326 | 0.699 |
| MOFT (re-impl., CogVideoX-5B), different prompt subject | 13 | 3.484 | 37.101 | 0.615 | 0.031 | 1.277 | 7.206 | 68.066 | 0.112 | 0.202 | 0.769 | 0.869 | 0.350 | 0.708 |
| No guidance (Wan2.1 T2V 1.3B), different prompt subject | 13 | 4.872 | 8.746 | 0.566 | 0.137 | 1.786 | 8.230 | 44.727 | 0.167 | 0.168 | 0.678 | -0.074 | 0.337 | 0.881 |
| Follow-Your-Motion (Wan2.1 T2V 1.3B), different prompt subject | 9 | 1.100 | 5.843 | 0.182 | 0.012 | 0.403 | 0.155 | 13.158 | 0.447 | 0.022 | 0.928 | 0.991 | 0.314 | 0.445 |
| Reference (ceiling, CLIP-IQA only) | 13 | - | - | - | - | - | - | - | - | - | - | - | - | 0.450 |

(B) Each row on its own MiraData pairs (baselines: the 19 census clips; ours: 13).

| Method | n videos | CamRotErr (deg) (lower) | ObjMC motion-only (px) (lower) | RPE (deg/frame) (lower) | ATE sim(3) (lower) | RotErr, CameraCtrl (rad sum) (lower) | TransErr, CameraCtrl (sq sum) (lower) | ObjMC (px) (lower) | Box-IoU (higher) | ObjErr3D (lower) | MF (higher) | Direction cos (higher) | IQ (CLIPScore) (higher) | CLIP-IQA (higher) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our method (A3) | 13 | 1.003 | 18.922 | 0.215 | 0.022 | 0.368 | 1.574 | 43.838 | 0.207 | 0.069 | 0.906 | 0.987 | 0.299 | 0.527 |
| No guidance (Wan2.1 I2V, A0) | 13 | 3.590 | 13.867 | 0.571 | 0.126 | 1.316 | 6.823 | 39.562 | 0.301 | 0.176 | 0.550 | 0.757 | 0.324 | 0.788 |
| No guidance (CogVideoX-5B), different prompt subject | 19 | 5.053 | 16.592 | 0.582 | 0.116 | 1.852 | 6.206 | 52.067 | 0.108 | 0.293 | 0.676 | -0.192 | 0.323 | 0.783 |
| DiTFlow (CogVideoX-5B), different prompt subject | 19 | 3.661 | 10.609 | 0.465 | 0.034 | 1.342 | 5.320 | 44.039 | 0.211 | 0.062 | 0.839 | 0.960 | 0.326 | 0.752 |
| SMM (re-impl., CogVideoX-5B), different prompt subject | 19 | 2.532 | 22.945 | 0.415 | 0.035 | 0.928 | 1.987 | 52.058 | 0.185 | 0.075 | 0.860 | 0.893 | 0.329 | 0.699 |
| MOFT (re-impl., CogVideoX-5B), different prompt subject | 19 | 5.000 | 27.848 | 0.639 | 0.031 | 1.833 | 6.845 | 62.708 | 0.178 | 0.086 | 0.769 | 0.870 | 0.350 | 0.696 |
| No guidance (Wan2.1 T2V 1.3B), different prompt subject | 19 | 5.582 | 10.238 | 0.612 | 0.137 | 2.046 | 7.944 | 47.242 | 0.167 | 0.141 | 0.678 | 0.058 | 0.345 | 0.881 |
| Follow-Your-Motion (Wan2.1 T2V 1.3B), different prompt subject | 14 | 1.075 | 6.095 | 0.183 | 0.012 | 0.394 | 0.112 | 12.856 | 0.533 | 0.020 | 0.934 | 0.992 | 0.302 | 0.473 |
| Reference (ceiling, CLIP-IQA only) | 19 | - | - | - | - | - | - | - | - | - | - | - | - | 0.416 |

Yield (videos generated / scored by the instrument, of the pairs each row was run on):

- Our method (A3): generated 13 of 13, scored 13
- No guidance (Wan2.1 I2V, A0): generated 13 of 13, scored 13
- No guidance (CogVideoX-5B), different prompt subject: generated 19 of 19, scored 19
- DiTFlow (CogVideoX-5B), different prompt subject: generated 19 of 19, scored 19
- SMM (re-impl., CogVideoX-5B), different prompt subject: generated 19 of 19, scored 19
- MOFT (re-impl., CogVideoX-5B), different prompt subject: generated 19 of 19, scored 19
- No guidance (Wan2.1 T2V 1.3B), different prompt subject: generated 19 of 19, scored 18
- Follow-Your-Motion (Wan2.1 T2V 1.3B), different prompt subject: generated 14 of 19, scored 14

## 5. Cross-method, DAVIS: horsejump-high, a single descriptive row per method (baselines seed 1)

horsejump-high/human: ours seed 0; baselines seed 1, different prompt subject.

| Method | n videos | CamRotErr (deg) (lower) | ObjMC motion-only (px) (lower) | RPE (deg/frame) (lower) | ATE sim(3) (lower) | RotErr, CameraCtrl (rad sum) (lower) | TransErr, CameraCtrl (sq sum) (lower) | ObjMC (px) (lower) | Box-IoU (higher) | ObjErr3D (lower) | MF (higher) | Direction cos (higher) | IQ (CLIPScore) (higher) | CLIP-IQA (higher) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Our method (A3) | 1 | 5.666 | 82.383 | 0.781 | 0.009 | 2.077 | 0.056 | 70.097 | 0.265 | 0.320 | 0.872 | 1.000 | 0.256 | 0.457 |
| No guidance (Wan2.1 I2V, A0) | 1 | 21.735 | 35.988 | 2.328 | 0.011 | 7.966 | 0.152 | 52.144 | 0.315 | 0.332 | 0.923 | 0.999 | 0.311 | 0.490 |
| No guidance (CogVideoX-5B), different prompt subject | 1 | 14.351 | 46.885 | 1.618 | 0.050 | 5.260 | 0.274 | 50.095 | 0.320 | 0.329 | 0.864 | 0.981 | 0.298 | 0.687 |
| DiTFlow (CogVideoX-5B), different prompt subject | 1 | 6.722 | 51.289 | 0.934 | 0.045 | 2.464 | 0.561 | 57.105 | 0.301 | 0.504 | 0.935 | 1.000 | 0.261 | 0.464 |
| SMM (re-impl., CogVideoX-5B), different prompt subject | 1 | 5.460 | 51.281 | 0.838 | 0.016 | 2.001 | 0.178 | 53.116 | 0.313 | 0.226 | 0.911 | 1.000 | 0.281 | 0.592 |
| MOFT (re-impl., CogVideoX-5B), different prompt subject | 1 | 4.235 | 69.481 | 1.029 | 0.036 | 1.552 | 0.364 | 83.663 | 0.216 | 0.389 | 0.909 | 0.958 | 0.276 | 0.688 |
| No guidance (Wan2.1 T2V 1.3B), different prompt subject | 1 | 16.254 | 41.333 | 1.789 | 0.030 | 5.957 | 0.622 | 47.282 | 0.263 | 0.150 | 0.887 | -0.999 | 0.250 | 0.302 |
| Follow-Your-Motion (Wan2.1 T2V 1.3B), different prompt subject | 1 | 1.769 | 16.681 | 0.502 | 0.020 | 0.648 | 0.182 | 19.828 | 0.540 | 0.239 | 0.958 | 1.000 | 0.288 | 0.659 |
| Reference (ceiling, CLIP-IQA only) | 1 | - | - | - | - | - | - | - | - | - | - | - | - | 0.686 |

A-G2b report-only: 14 of 144 scored videos have the reference or the video over the uncalibrated 4 px check (scored, flagged; per row in the CSV).
