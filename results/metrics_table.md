| Method | Camera rotation error (deg) ↓ | Subject movement error, offset removed (px) ↓ | Subject position error (px) ↓ |
|---|---|---|---|
| No guidance (CogVideoX) | 2.36 | 20.69 | 86.16 |
| DiTFlow (CogVideoX) | 1.08 | **8.47** | 63.69 |
| SMM (re-impl., CogVideoX) | 0.90 | 12.86 | 58.55 |
| MOFT (re-impl., CogVideoX) | 1.12 | 19.19 | **55.22** |
| No guidance (Wan 2.1 I2V) | 5.37 | 34.19 | 92.46 |
| Ours (Wan 2.1 I2V + guidance) | **0.46** | 17.28 | 107.49 |

Medians over the 5 gated pairs (3 source clips), seed 0 for every method, first 21 frames of each pair's window; SMM and Wan no-guidance over 4 pairs (one video each missing at scoring time). Bold: column best. SMM and MOFT are the DiTFlow repository's re-implementations on CogVideoX, comparable to DiTFlow and to the CogVideoX no-guidance row, not to the published SMM and MOFT. The absolute position differs for ours because the subject is placed in a synthesised scene, not at the reference's pixel position.
