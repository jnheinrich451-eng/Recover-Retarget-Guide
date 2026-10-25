# Module 3 S-int step 2 — camera path, object path, coverage, anchors

R_t = I on every pair (rotation source not yet chosen). Coverage is the share of non-object patches in frame t that the frame-0 scene still explains.

| pair | ch | T_l | s_c | cam path / rot° | obj path | ViPE-vs-D4RT disp diff / path | heading vs d_src ° | cov t=1 / mid / last | same, sky excluded | anchors (sky excl.) | bg flow last: median / p95 patches | obj centre shift patches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0035.1.003/vehicle | ERROR AssertionError: 000000000035.1.003/vehicle: beta_far is not determined under vipe_over_sigma (v2.6 §2.3: the pair does not enter S4) |
| 0265.2.003/vehicle | x0 | 9 | 1.000 | 0.656 / 10.5 | 1.373 | 0.106 | 55.9 | 0.39 / 0.35 / 0.32 | 0.79 / 0.70 / 0.60 | [0, 1, 2, 3, 4, 5] ([0]) | 3.79 / 6.95 | 4.08 |
| 000265.2.005/human | x0 | 21 | 0.854 | 1.298 / 10.6 | 1.851 | 0.461 | 2.4 | 0.93 / 0.81 / 0.84 | 0.93 / 0.81 / 0.84 | [0] ([0]) | 4.69 / 11.85 | 4.91 |
| 0265.2.005/vehicle | x0 | 21 | 1.000 | 1.343 / 10.6 | 1.915 | 0.461 | 2.4 | 0.40 / 0.29 / 0.36 | 0.77 / 0.55 / 0.74 | [0, 1, 2, 3, 4, 5] ([0]) | 3.84 / 10.16 | 3.76 |
| 000370.1.008/human | x0 | 21 | 1.289 | 1.107 / 1.2 | 1.660 | 0.114 | 11.7 | 0.99 / 0.94 / 0.84 | 0.99 / 0.94 / 0.84 | [0] ([0]) | 6.18 / 12.24 | 1.64 |
| 0404.1.004/vehicle | x0 | 9 | 1.000 | 0.960 / 41.7 | 0.763 | 0.377 | 27.1 | 0.43 / 0.35 / 0.29 | 0.96 / 0.77 / 0.58 | [0, 1, 2, 3, 4, 5] ([0]) | 12.57 / 24.19 | 1.20 |

## Decisions needed from John (S-int hard stop)

1. Review per pair: is the object path plausible (heading vs d_src; ViPE-vs-D4RT difference), and does the coverage curve make sense for the camera motion?
2. Sky: anchors computed with and without sky patches in the denominator (reviewer's v2.6 item); choose which is the schedule of record.
3. Any pair with an ERROR row is not carried to the next stage until its cause is read.
