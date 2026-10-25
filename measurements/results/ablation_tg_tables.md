# T_g window ablation (plan D5): five pairs, seed 0, frames 0-20 (CSV sealed 2026-10-20T22:31:28, 743956909591)

Lower is better except Box-IoU, MF, direction cosine, IQ, CLIP-IQA. Scorer v2 (proper similarity). Descriptive: five pairs.

## 000000000265.2.003/vehicle

| Arm | CamRotErr (deg) | RPE (deg/fr) | ObjMC motion (px) | ObjMC (px) | Box-IoU | ObjErr3D | MF | Dir cos | IQ (CLIPScore) | CLIP-IQA | align | A-G2b over |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 (no guidance) | 13.815 | 1.530 | 55.629 | 162.543 | 0.024 | - | 0.794 | -0.021 | 0.367 | 0.808 | unalignable | False |
| A3 [0.85, 0.75] (A: ends earlier) | 0.727 | 0.119 | 5.837 | 112.725 | 0.208 | 0.124 | 0.911 | 0.995 | 0.343 | 0.588 | ok | False |
| A3 [0.85, 0.60] (lock, of record) | 0.462 | 0.120 | 2.277 | 107.488 | 0.221 | 0.096 | 0.915 | 0.995 | 0.357 | 0.667 | ok | False |
| A3 [0.85, 0.45] (B: ends later) | 0.494 | 0.115 | 3.973 | 118.596 | 0.172 | 0.093 | 0.927 | 0.994 | 0.347 | 0.634 | ok | False |

## 000000000265.2.005/vehicle

| Arm | CamRotErr (deg) | RPE (deg/fr) | ObjMC motion (px) | ObjMC (px) | Box-IoU | ObjErr3D | MF | Dir cos | IQ (CLIPScore) | CLIP-IQA | align | A-G2b over |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 (no guidance) | 1.961 | 0.156 | 12.054 | 42.827 | 0.364 | 0.046 | 0.531 | 0.964 | 0.354 | 0.804 | ok | False |
| A3 [0.85, 0.75] (A: ends earlier) | 0.478 | 0.101 | 12.249 | 13.404 | 0.651 | 0.012 | 0.842 | 0.997 | 0.354 | 0.724 | ok | False |
| A3 [0.85, 0.60] (lock, of record) | 0.256 | 0.085 | 7.990 | 11.316 | 0.662 | 0.012 | 0.855 | 0.983 | 0.344 | 0.756 | ok | False |
| A3 [0.85, 0.45] (B: ends later) | 0.309 | 0.089 | 24.956 | 47.420 | 0.345 | 0.018 | 0.821 | 0.996 | 0.355 | 0.849 | ok | False |

## 000000000265.2.005/human

| Arm | CamRotErr (deg) | RPE (deg/fr) | ObjMC motion (px) | ObjMC (px) | Box-IoU | ObjErr3D | MF | Dir cos | IQ (CLIPScore) | CLIP-IQA | align | A-G2b over |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 (no guidance) | 7.387 | 0.650 | 12.752 | 73.033 | 0.290 | 0.023 | 0.440 | 0.964 | 0.333 | 0.767 | ok | False |
| A3 [0.85, 0.75] (A: ends earlier) | 0.368 | 0.105 | 11.453 | 12.311 | 0.154 | 0.010 | 0.845 | 0.991 | 0.308 | 0.474 | ok | True |
| A3 [0.85, 0.60] (lock, of record) | 0.222 | 0.050 | 30.215 | 18.858 | 0.212 | 0.019 | 0.843 | 0.990 | 0.318 | 0.660 | ok | False |
| A3 [0.85, 0.45] (B: ends later) | 0.158 | 0.048 | 22.710 | 15.809 | 0.165 | 0.017 | 0.830 | 0.989 | 0.336 | 0.674 | ok | False |

## 000000000370.1.008/human

| Arm | CamRotErr (deg) | RPE (deg/fr) | ObjMC motion (px) | ObjMC (px) | Box-IoU | ObjErr3D | MF | Dir cos | IQ (CLIPScore) | CLIP-IQA | align | A-G2b over |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 (no guidance) | 3.358 | 0.257 | 75.105 | 111.891 | 0.135 | 0.012 | 0.758 | -0.949 | 0.322 | 0.940 | ok | False |
| A3 [0.85, 0.75] (A: ends earlier) | 0.406 | 0.051 | 15.599 | 92.616 | 0.163 | 0.027 | 0.911 | 0.995 | 0.322 | 0.890 | ok | False |
| A3 [0.85, 0.60] (lock, of record) | 3.358 | 0.257 | 75.105 | 111.891 | 0.135 | 0.012 | 0.914 | 0.996 | 0.331 | 0.895 | ok | False |
| A3 [0.85, 0.45] (B: ends later) | 0.426 | 0.065 | 6.740 | 6.235 | 0.230 | 0.009 | 0.912 | 0.995 | 0.330 | 0.911 | ok | False |

## 000000000404.1.004/vehicle

| Arm | CamRotErr (deg) | RPE (deg/fr) | ObjMC motion (px) | ObjMC (px) | Box-IoU | ObjErr3D | MF | Dir cos | IQ (CLIPScore) | CLIP-IQA | align | A-G2b over |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 (no guidance) | 0.746 | 0.175 | 21.752 | 70.541 | 0.045 | 0.009 | 0.799 | -0.238 | 0.336 | 0.878 | ok | True |
| A3 [0.85, 0.75] (A: ends earlier) | 0.328 | 0.090 | 17.857 | 133.637 | 0.356 | 0.014 | 0.884 | -0.795 | 0.340 | 0.763 | ok | True |
| A3 [0.85, 0.60] (lock, of record) | 0.533 | 0.081 | 17.276 | 120.974 | 0.393 | 0.011 | 0.871 | -0.887 | 0.344 | 0.740 | ok | False |
| A3 [0.85, 0.45] (B: ends later) | 0.462 | 0.085 | 11.675 | 140.475 | 0.271 | 0.012 | 0.865 | -0.782 | 0.348 | 0.727 | ok | True |

## Registered predictions, scored as registered (plan D5)

(1) A ends earlier: CamRotErr and the image-quality scores (CLIPScore, CLIP-IQA, Box-IoU) move from the lock toward A0. (2) B ends later: CamRotErr at or below the lock's, image quality lower. (3) ObjMC-motion has no consistent direction.

| pair | (1) CamRotErr A toward A0 | (1) IQ | (1) CLIP-IQA | (1) Box-IoU | (2) CamRotErr B <= lock | (2) IQ lower | (2) CLIP-IQA lower | (2) Box-IoU lower | (3) ObjMC-motion sign A-lock / B-lock |
|---|---|---|---|---|---|---|---|---|---|
| 000000000265.2.003/vehicle | yes | no | no | yes | no | yes | yes | yes | + / + |
| 000000000265.2.005/vehicle | yes | yes | no | yes | no | no | no | yes | + / + |
| 000000000265.2.005/human | yes | no | no | no | yes | no | no | yes | - / - |
| 000000000370.1.008/human | n/a | yes | no | n/a | yes | yes | no | no | - / - |
| 000000000404.1.004/vehicle | no | yes | yes | yes | yes | no | yes | yes | + / - |
| **holds on** | 3 of 4 | 3 of 5 | 1 of 5 | 3 of 4 | 3 of 5 | 2 of 5 | 2 of 5 | 4 of 5 | A: {'+': 3, '-': 2}, B: {'+': 2, '-': 3} |

(3) setting A: ObjMC-motion moved up on 3 and down on 2 pairs: prediction (3) holds. (unchanged on 0, not computable on 0)

(3) setting B: ObjMC-motion moved up on 2 and down on 3 pairs: prediction (3) holds. (unchanged on 0, not computable on 0)

A0 and lock-window A3 are the six pairs' rows of record (frames identical to the scored files): all
