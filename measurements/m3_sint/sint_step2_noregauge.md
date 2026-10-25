# Module 3 S-int step 2 — camera path, object path, coverage, anchors (UN-RE-GAUGED sub-row)

R_t = I on every pair (rotation source not yet chosen). Coverage is the share of non-object patches in frame t that the frame-0 scene still explains.

| pair | ch | T_l | s_c | cam path / rot° | obj path | ViPE-vs-D4RT disp diff / path | heading vs d_src ° | cov t=1 / mid / last | same, sky excluded | anchors (sky excl.) | bg flow last: median / p95 patches | obj centre shift patches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0035.1.003/vehicle | branch | 20 | 1.000 | 9.742 / 57.5 | 9.181 | 0.679 | 20.8 | 0.46 / 0.12 / 0.02 | 0.78 / 0.16 / 0.01 | [0, 1, 2, 3, 4, 5] ([0, 3, 6, 9, 12, 15]) | 42.81 / 49.21 | 6.39 |
| 0265.2.003/vehicle | x0 | 9 | 1.000 | 1.218 / 10.5 | 1.947 | 0.170 | 41.8 | 0.39 / 0.33 / 0.24 | 0.78 / 0.67 / 0.45 | [0, 1, 2, 3, 4, 5] ([0, 7]) | 5.28 / 17.23 | 3.92 |
| 000265.2.005/human | x0 | 21 | 1.000 | 2.155 / 10.6 | 2.420 | 0.412 | 3.4 | 0.94 / 0.74 / 0.79 | 0.94 / 0.74 / 0.79 | [0] ([0]) | 4.05 / 23.49 | 4.17 |
| 0265.2.005/vehicle | x0 | 21 | 1.000 | 2.155 / 10.6 | 2.420 | 0.412 | 3.4 | 0.41 / 0.25 / 0.33 | 0.78 / 0.47 / 0.67 | [0, 1, 2, 3, 4, 5] ([0, 8, 16]) | 6.39 / 20.35 | 3.79 |
| 000370.1.008/human | x0 | 21 | 1.000 | 1.055 / 1.2 | 1.259 | 0.177 | 5.5 | 0.99 / 0.94 / 0.85 | 0.99 / 0.94 / 0.85 | [0] ([0]) | 5.96 / 11.70 | 1.47 |
| 0404.1.004/vehicle | x0 | 9 | 1.000 | 1.183 / 41.7 | 0.784 | 0.355 | 25.8 | 0.42 / 0.33 / 0.30 | 0.95 / 0.72 / 0.63 | [0, 1, 2, 3, 4, 5] ([0]) | 10.08 / 23.31 | 1.69 |

## Decisions needed from John (S-int hard stop)

1. Review per pair: is the object path plausible (heading vs d_src; ViPE-vs-D4RT difference), and does the coverage curve make sense for the camera motion?
2. Sky: anchors computed with and without sky patches in the denominator (reviewer's v2.6 item); choose which is the schedule of record.
3. Any pair with an ERROR row is not carried to the next stage until its cause is read.
