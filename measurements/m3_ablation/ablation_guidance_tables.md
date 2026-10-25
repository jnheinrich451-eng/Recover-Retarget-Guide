# Guidance ablations: five pairs, seed 0, frames 0-20 (CSV sealed 2026-10-24T06:01:03, a0496647f78e)

Lower is better except Box-IoU, MF, direction cosine, IQ, CLIP-IQA. Scorer v2 (proper similarity). Descriptive: five pairs.

## 000000000265.2.003/vehicle

| Arm | CamRotErr (deg) | RPE (deg/fr) | ObjMC motion (px) | ObjMC (px) | Box-IoU | ObjErr3D | MF | Dir cos | IQ (CLIPScore) | CLIP-IQA | align | A-G2b over |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 (no guidance) | 13.815 | 1.530 | 55.629 | 162.543 | 0.024 | - | 0.794 | -0.021 | 0.367 | 0.808 | unalignable | False |
| A3 of record (gamma 1, UniPC, soft mask, sky in) | 0.462 | 0.120 | 2.277 | 107.488 | 0.221 | 0.096 | 0.915 | 0.995 | 0.357 | 0.667 | ok | False |
| A3 without the object render (vehicles) | 1.091 | 0.154 | 34.695 | 31.831 | 0.361 | 0.045 | 0.915 | 0.962 | 0.374 | 0.572 | ok | False |
| A3 gamma 0.5 | 0.607 | 0.107 | 4.850 | 120.199 | 0.125 | 0.109 | 0.916 | 0.995 | 0.358 | 0.644 | ok | False |
| A3 gamma 0.25 | 0.635 | 0.126 | 2.853 | 120.324 | 0.155 | 0.100 | 0.916 | 0.995 | 0.363 | 0.565 | ok | False |
| A3 Euler scheduler | 0.417 | 0.112 | 1.999 | 109.971 | 0.189 | 0.148 | 0.922 | 0.996 | 0.362 | 0.682 | ok | False |
| A0, corrector off at the corrected steps | 14.033 | 1.542 | 39.179 | 38.584 | 0.295 | - | 0.794 | 0.018 | 0.365 | 0.798 | unalignable | False |
| A3 hard mask | 0.537 | 0.122 | 17.418 | 17.582 | 0.385 | 0.019 | 0.909 | 0.999 | 0.338 | 0.512 | ok | False |
| A3 sky out of the mask | 3.646 | 0.398 | 26.138 | 157.116 | 0.151 | 0.028 | 0.917 | -0.120 | 0.347 | 0.746 | ok | True |

## 000000000265.2.005/vehicle

| Arm | CamRotErr (deg) | RPE (deg/fr) | ObjMC motion (px) | ObjMC (px) | Box-IoU | ObjErr3D | MF | Dir cos | IQ (CLIPScore) | CLIP-IQA | align | A-G2b over |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 (no guidance) | 1.961 | 0.156 | 12.054 | 42.827 | 0.364 | 0.046 | 0.531 | 0.964 | 0.354 | 0.804 | ok | False |
| A3 of record (gamma 1, UniPC, soft mask, sky in) | 0.256 | 0.085 | 7.990 | 11.316 | 0.662 | 0.012 | 0.855 | 0.983 | 0.344 | 0.756 | ok | False |
| A3 without the object render (vehicles) | 0.362 | 0.081 | 19.570 | 55.467 | 0.327 | 0.031 | 0.816 | 0.972 | 0.350 | 0.691 | ok | False |
| A3 gamma 0.5 | 0.513 | 0.121 | 10.849 | 10.422 | 0.654 | 0.013 | 0.826 | 0.998 | 0.336 | 0.739 | ok | False |
| A3 gamma 0.25 | 0.498 | 0.105 | 7.385 | 12.078 | 0.646 | 0.014 | 0.826 | 0.999 | 0.321 | 0.766 | ok | False |
| A3 Euler scheduler | 0.314 | 0.100 | 27.399 | 89.208 | 0.399 | 0.014 | 0.828 | 0.996 | 0.347 | 0.756 | ok | False |
| A0, corrector off at the corrected steps | 1.925 | 0.154 | 9.984 | 90.518 | 0.201 | 0.045 | 0.555 | 0.965 | 0.354 | 0.815 | ok | False |
| A3 hard mask | 0.287 | 0.091 | 11.975 | 13.137 | 0.651 | 0.013 | 0.824 | 0.983 | 0.338 | 0.714 | ok | False |
| A3 sky out of the mask | 0.313 | 0.092 | 21.639 | 10.422 | 0.657 | 0.015 | 0.801 | 1.000 | 0.343 | 0.743 | ok | False |

## 000000000265.2.005/human

| Arm | CamRotErr (deg) | RPE (deg/fr) | ObjMC motion (px) | ObjMC (px) | Box-IoU | ObjErr3D | MF | Dir cos | IQ (CLIPScore) | CLIP-IQA | align | A-G2b over |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 (no guidance) | 7.387 | 0.650 | 12.752 | 73.033 | 0.290 | 0.023 | 0.440 | 0.964 | 0.333 | 0.767 | ok | False |
| A3 of record (gamma 1, UniPC, soft mask, sky in) | 0.222 | 0.050 | 30.215 | 18.858 | 0.212 | 0.019 | 0.843 | 0.990 | 0.318 | 0.660 | ok | False |
| A3 gamma 0.5 | 0.149 | 0.043 | 31.178 | 26.206 | 0.469 | 0.029 | 0.828 | 0.990 | 0.318 | 0.620 | ok | False |
| A3 gamma 0.25 | 0.229 | 0.047 | 7.135 | 127.128 | 0.000 | 0.025 | 0.824 | 0.990 | 0.323 | 0.629 | ok | False |
| A3 Euler scheduler | - | - | - | - | - | - | 0.847 | 0.989 | 0.311 | 0.525 | unscorable_gen |  |
| A0, corrector off at the corrected steps | 7.372 | 0.649 | 7.601 | 41.061 | 0.285 | 0.152 | 0.455 | 0.964 | 0.334 | 0.781 | ok | False |
| A3 hard mask | 0.204 | 0.058 | 35.337 | 129.623 | 0.088 | 0.077 | 0.829 | 0.989 | 0.322 | 0.723 | ok | False |
| A3 sky out of the mask | 0.584 | 0.095 | 19.307 | 111.518 | 0.164 | 0.023 | 0.836 | 0.989 | 0.320 | 0.576 | ok | False |

## 000000000370.1.008/human

| Arm | CamRotErr (deg) | RPE (deg/fr) | ObjMC motion (px) | ObjMC (px) | Box-IoU | ObjErr3D | MF | Dir cos | IQ (CLIPScore) | CLIP-IQA | align | A-G2b over |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 (no guidance) | 3.358 | 0.257 | 75.140 | 112.055 | 0.135 | 0.012 | 0.758 | -0.949 | 0.322 | 0.940 | ok | False |
| A3 of record (gamma 1, UniPC, soft mask, sky in) | 0.430 | 0.060 | 3.780 | 25.424 | 0.538 | 0.006 | 0.914 | 0.996 | 0.331 | 0.895 | ok | False |
| A3 gamma 0.5 | 0.319 | 0.052 | 12.185 | 14.192 | 0.148 | 0.008 | 0.911 | 0.994 | 0.325 | 0.914 | ok | False |
| A3 gamma 0.25 | 0.313 | 0.046 | 9.468 | 7.715 | 0.247 | 0.005 | 0.930 | 0.997 | 0.332 | 0.904 | ok | False |
| A3 Euler scheduler | 0.408 | 0.057 | 33.310 | 113.583 | 0.224 | 0.007 | 0.909 | 0.995 | 0.332 | 0.892 | ok | False |
| A0, corrector off at the corrected steps | 3.174 | 0.242 | 6.361 | 5.799 | 0.278 | 0.003 | 0.725 | -0.951 | 0.325 | 0.943 | ok | False |
| A3 hard mask | 0.392 | 0.056 | 13.170 | 53.809 | 0.319 | 0.009 | 0.908 | 0.996 | 0.335 | 0.882 | ok | False |
| A3 sky out of the mask | 0.537 | 0.071 | 4.472 | 25.582 | 0.609 | 0.004 | 0.922 | 0.994 | 0.335 | 0.903 | ok | False |

## 000000000404.1.004/vehicle

| Arm | CamRotErr (deg) | RPE (deg/fr) | ObjMC motion (px) | ObjMC (px) | Box-IoU | ObjErr3D | MF | Dir cos | IQ (CLIPScore) | CLIP-IQA | align | A-G2b over |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 (no guidance) | 0.746 | 0.175 | 21.752 | 70.541 | 0.045 | 0.009 | 0.799 | -0.238 | 0.336 | 0.878 | ok | True |
| A3 of record (gamma 1, UniPC, soft mask, sky in) | 0.533 | 0.081 | 17.276 | 120.974 | 0.393 | 0.011 | 0.871 | -0.887 | 0.344 | 0.740 | ok | False |
| A3 without the object render (vehicles) | 0.480 | 0.090 | 14.313 | 143.439 | 0.123 | 0.012 | 0.865 | -0.171 | 0.346 | 0.817 | ok | True |
| A3 gamma 0.5 | 0.392 | 0.082 | 16.198 | 120.240 | 0.366 | 0.011 | 0.885 | -0.850 | 0.340 | 0.773 | ok | True |
| A3 gamma 0.25 | 0.279 | 0.063 | 15.911 | 116.811 | 0.396 | 0.013 | 0.879 | -0.656 | 0.337 | 0.713 | ok | True |
| A3 Euler scheduler | 0.453 | 0.082 | 17.692 | 137.461 | 0.295 | 0.015 | 0.872 | -0.896 | 0.346 | 0.703 | ok | True |
| A0, corrector off at the corrected steps | 0.734 | 0.170 | 66.548 | 48.765 | 0.353 | 0.015 | 0.776 | -0.270 | 0.335 | 0.869 | ok | True |
| A3 hard mask | 0.362 | 0.080 | 29.245 | 141.803 | 0.177 | 0.009 | 0.896 | -0.353 | 0.334 | 0.643 | ok | True |
| A3 sky out of the mask | 0.269 | 0.082 | 19.521 | 131.988 | 0.358 | 0.013 | 0.909 | -0.343 | 0.344 | 0.776 | ok | False |

## Registered predictions, scored as registered (sheet of 2026-10-24)

- P0 (correction on vs off): A3 of record has lower CamRotErr than A0 on 0 of 0 pairs (registered: 5 of 5).
- P1 (object render off, vehicles): ObjMC-motion higher without the object render on 0 of 0 pairs (registered: 3 of 3); CamRotErr noobj - lock per pair: 265.2.003 1.091 vs 0.462, 265.2.005 0.362 vs 0.256, 404.1.004 0.480 vs 0.533; CLIP-IQA sign {'n/a': 3}, IQ sign {'n/a': 3}.
- P2 (gamma): CamRotErr rises monotonically from gamma 1 to 0.5 to 0.25 on 1 of 5 pairs (registered: 4 of 5); toward A0 at gamma 0.25: IQ {'n/a': 5}, CLIP-IQA {'n/a': 5}; ObjMC-motion sign g0.5 {'n/a': 5}, g0.25 {'n/a': 5}.
- P3 (Euler): CamRotErr at or above the lock's on 0 of 0 pairs (registered: 4 of 5); Box-IoU below the lock's on 0 of 0; IQ sign {'n/a': 5}, CLIP-IQA sign {'n/a': 5}.
- P4 (corrector off, no correction) coff - A0 per pair: CamRotErr (deg) [0.218, -0.036, -0.015, -0.184, -0.012]; ObjMC motion (px) [-16.45, -2.07, -5.151, -68.779, 44.796]; Box-IoU [0.271, -0.163, -0.005, 0.143, 0.308]; CLIP-IQA [-0.01, 0.011, 0.014, 0.003, -0.009] (registered: within the seed-to-seed spread of A0 on 4 of 5; judged against the recorded seeds 0/1 spread of A0).
- P5 (hard mask): CamRotErr at or above the lock's on 0 of 0 pairs (registered: 4 of 5); CLIP-IQA sign {'n/a': 5}.
- P6 (sky out): CamRotErr above the lock's on 0 of 0 pairs (registered: 4 of 5).

A0 and lock-window A3 are the six pairs' rows of record (frames identical to the scored files): all
