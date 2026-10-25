# Module 3 — readback: the readout on RE-ENCODED videos against the pair's geometric background target

| pair | video | sigma | EPE to F_geo, mean (last frame) | windows containing the target | mass at target |
|---|---|---|---|---|---|
| 00000265.2.005/vehicle | A0_seed0_rec_F33.mp4 | 0.3 | 3.89 (6.65) | 0.628 | 0.281 |
| 00000265.2.005/vehicle | A0_seed0_rec_F33.mp4 | 0.45 | 4.44 (7.51) | 0.598 | 0.274 |
| 00000265.2.005/vehicle | A0_seed0_rec_F33.mp4 | 0.55 | 4.35 (7.74) | 0.594 | 0.284 |
| 00000265.2.005/vehicle | A0_seed0_rec_F33.mp4 | 0.7 | 4.34 (7.28) | 0.584 | 0.283 |
| 00000265.2.005/vehicle | A0_seed0_rec_F33.mp4 | 0.85 | 5.01 (8.65) | 0.553 | 0.279 |
| 00000265.2.005/vehicle | A2_seed0_rec_F33.mp4 | 0.3 | 1.88 (3.17) | 0.879 | 0.353 |
| 00000265.2.005/vehicle | A2_seed0_rec_F33.mp4 | 0.45 | 1.71 (2.89) | 0.891 | 0.373 |
| 00000265.2.005/vehicle | A2_seed0_rec_F33.mp4 | 0.55 | 1.58 (3.19) | 0.899 | 0.382 |
| 00000265.2.005/vehicle | A2_seed0_rec_F33.mp4 | 0.7 | 1.84 (3.33) | 0.882 | 0.385 |
| 00000265.2.005/vehicle | A2_seed0_rec_F33.mp4 | 0.85 | 3.40 (5.85) | 0.716 | 0.310 |
| 00000265.2.005/vehicle | source | 0.3 | 1.36 (2.14) | 0.903 | 0.467 |
| 00000265.2.005/vehicle | source | 0.45 | 1.30 (2.08) | 0.911 | 0.470 |
| 00000265.2.005/vehicle | source | 0.55 | 1.36 (2.08) | 0.909 | 0.471 |
| 00000265.2.005/vehicle | source | 0.7 | 1.30 (2.05) | 0.912 | 0.477 |
| 00000265.2.005/vehicle | source | 0.85 | 1.50 (2.35) | 0.906 | 0.475 |

Reading fixed in advance: A2's readback EPE near A0's and far above its in-loop value -> the in-loop readout was satisfied by the latent perturbation, not by generated motion (guidance on an internal readout can be gamed); A2's readback near its in-loop value -> the content carries the motion and the pixel grader is the thing to re-examine. The source clip's own readback is the floor the target was built for.
