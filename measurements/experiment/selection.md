# Module 2 selection v2 on 19 pairs under /content/drive/MyDrive/motion_transfer_exp (rule a' with Q1/Q2; lock values calibrated 2026-10-18 on the seven pairs)

| pair | survivors | S1 | pick | heading | human's pick | blind verdict of the pick | pick at eps/2 | pick at eps x2 |
|---|---|---|---|---|---|---|---|---|
| 000000000130.0.015/human | 0 | None | None | H1 | None | - | None | None |
| 000000000191.6.037/human | 4 | [0, 1, 2, 3] | 0 | H1 | None | - | 1 | 0 |
| 000000000200.1.006/human | 4 | [0, 1, 2, 3] | 0 | H1 | None | - | 0 | 0 |
| 000000000217.0.006/vehicle | 3 | [2, 3] | 2 | H1 | None | - | 2 | 1 |
| 000000000221.7.011/human | 4 | [0, 1, 2, 3] | 1 | H1 | None | - | 1 | 1 |
| 000000000252.1.018/vehicle | 1 | [2] | 2 | unchecked | None | - | 2 | 2 |
| 000000000252.2.007/human | 4 | [0, 2, 3] | 0 | H1 | None | - | 2 | 0 |
| 000000000256.3.010/human | 4 | [0, 1, 2, 3] | 1 | H1 | None | - | 3 | 0 |
| 000000000265.4.002/human | 4 | [0, 1, 2, 3] | 0 | H1 | None | - | 0 | 0 |
| 000000000362.2.005/human | 1 | [0] | 0 | unchecked | None | - | 0 | 0 |
| 000000000391.0.016/human | 2 | [0, 3] | 0 | H1 | None | - | 3 | 0 |
| 000000000396.3.002/vehicle | 4 | [0, 1, 3] | 0 | unchecked | None | - | 0 | 0 |
| 000000000404.1.006/human | 3 | [1, 3] | 1 | H1 | None | - | 3 | 0 |
| 000000000431.5.009/human | 4 | [0, 1, 2, 3] | 1 | H1 | None | - | 3 | 1 |
| bmx-bumps/human | 4 | [0, 1, 2, 3] | 0 | H1 | None | - | 1 | 0 |
| breakdance/human | 0 | None | None | H1 | None | - | None | None |
| camel/human | 4 | [0, 1, 2, 3] | 0 | H1 | None | - | 2 | 0 |
| goat/human | 0 | None | None | H1 | None | - | None | None |
| horsejump-high/human | 1 | [0] | 0 | H1 | None | - | 0 | 0 |

G2 DROPPED AND DISCLOSED: its self-test failed as measured (2026-10-19 02:05: 14 of 23 in-contact census clips below 0.8, median 0.57; see measurements/m2_selection/g2_selftest_labelled.md). Path support is reported per candidate, never gated; the gate list is G1, H, quality.

G1 pass rate over all candidates: 51 of 76 (G1 guarantees admissibility to Module 3; it does not rank).
Contact outcome labels of the picks: 191.6.037/human: contact_fails_regauge; 200.1.006/human: contact_holds; 217.0.006/vehicle: contact_fails_regauge; 221.7.011/human: contact_holds; 252.1.018/vehicle: contact_fails_regauge; 252.2.007/human: contact_fails_regauge; 256.3.010/human: contact_holds; 265.4.002/human: contact_fails_regauge; 362.2.005/human: no_contact_line_in_view; 391.0.016/human: contact_fails_regauge; 396.3.002/vehicle: contact_fails_regauge; 404.1.006/human: contact_fails_regauge; 431.5.009/human: contact_fails_regauge; bmx-bumps/human: contact_fails_regauge; camel/human: contact_fails_regauge; horsejump-high/human: contact_fails_regauge

Anchors: 16 of 19 pairs got an anchor; the rest are 'no anchor' (every candidate failed a gate). Picks that change when eps is halved or doubled: 9 of 16.
No blind verdicts exist for these pairs (the experiment set is never scored by eye); the acceptable-pick rate and the chance baseline are defined on the seven calibration pairs only.