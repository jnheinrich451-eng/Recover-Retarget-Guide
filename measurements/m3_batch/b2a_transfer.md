# B2a transfer check - 20260923T072545Z

**PASS: no gate verdict flips on any clip**

Pass rule (pre-registered): no gate verdict flips on any clip; numeric deltas reported, not thresholded.

| clip | stage | verdict flips | largest numeric change | old | new | rel |
|---|---|---|---|---|---|---|
| 000000000001.5.002 | A | none | `gates.A-G2.median_rel` | 0.02278025448322296 | 0.02278025448322296 | 0.00% |
| 000000000001.5.002 | B | none | `follow_cam.n_quiet` | 153 | 153 | 0.00% |
| 000000000001.5.002 | C | none | `prompts.anchor_n` | 153 | 153 | 0.00% |
| 000000000001.5.002 | C5 | none | `n_dense` | 4637 | 4637 | 0.00% |
| 000000000265.2.005 | A | none | `gates.A-G2.median_rel` | 0.04860524833202362 | 0.04860524833202362 | 0.00% |
| 000000000265.2.005 | B | none | `follow_cam.n_quiet` | 74 | 74 | 0.00% |
| 000000000265.2.005 | C | none | `prompts.anchor_n` | 37 | 37 | 0.00% |
| 000000000265.2.005 | C5 | none | `n_dense` | 613 | 613 | 0.00% |
| 000000000035.1.003 | A | none | `gates.A-G2.median_rel` | 0.08222095668315887 | 0.08222095668315887 | 0.00% |
| 000000000035.1.003 | B | none | `follow_cam.n_quiet` | 1734 | 1734 | 0.00% |
| 000000000035.1.003 | C | none | `t_c` | 4 | 4 | 0.00% |
| 000000000035.1.003 | C5 | none | `n_dense` | 112 | 112 | 0.00% |

## T0.2 determinism (000000000265.2.005)

**deterministic - two-pass floor is zero; noise floor = static residual + Kubric error**

| field | bitwise equal | max abs diff |
|---|---|---|
| cam_extrinsics | True | 0.0 |
| depth | True | 0.0 |
| track_xy | True | 0.0 |
| track_xyz | True | 0.0 |
| track_visible | True | 0.0 |
| instances | True | 0.0 |
