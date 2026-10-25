# Module 2 selection v2 on the seven pairs (rule a' with Q1/Q2; lock values calibrated 2026-10-18)

| pair | survivors | S1 | pick | heading | human's pick | blind verdict of the pick | pick at eps/2 | pick at eps x2 |
|---|---|---|---|---|---|---|---|---|
| 000000000035.1.003/vehicle | 4 | [0, 2, 3] | 0 | H1 | 2 | fail | 0 | 0 |
| 000000000265.2.003/vehicle | 4 | [0, 1, 2, 3] | 1 | H1 | 2 | pass | 1 | 0 |
| 000000000265.2.005/vehicle | 4 | [0, 1, 2, 3] | 1 | H1 | 1 | pass | 1 | 1 |
| 000000000265.2.005/human | 4 | [0, 1, 2, 3] | 0 | H1 | 0 | pass | 1 | 0 |
| 000000000284.0.006/vehicle | 4 | [0, 1, 2, 3] | 3 | unchecked | 3 | fail | 0 | 1 |
| 000000000370.1.008/human | 4 | [0, 1, 2, 3] | 0 | H1 | 2 | pass | 1 | 0 |
| 000000000404.1.004/vehicle | 4 | [0, 1, 2, 3] | 0 | H1 | 1 | pass | 1 | 0 |

G2 DROPPED AND DISCLOSED: its self-test failed as measured (2026-10-19 02:05: 14 of 23 in-contact census clips below 0.8, median 0.57; see measurements/m2_selection/g2_selftest_labelled.md). Path support is reported per candidate, never gated; the gate list is G1, H, quality.

G1 pass rate over all candidates: 28 of 28 (G1 guarantees admissibility to Module 3; it does not rank).
Contact outcome labels of the picks: 035.1.003/vehicle: contact_holds; 265.2.003/vehicle: no_contact_line_in_view; 265.2.005/vehicle: contact_fails_regauge; 265.2.005/human: contact_holds; 284.0.006/vehicle: contact_fails_regauge; 370.1.008/human: no_contact_line_in_view; 404.1.004/vehicle: no_contact_line_in_view

Acceptable-pick rate (the thesis metric): 5 of 7 picks are in the blind pass set (7 of 7 pairs got an anchor; the rest are 'no anchor').
Agreement with the human's selection (descriptive only): 3 of 7. Picks that change when eps is halved or doubled: 5 of 7.
Chance baseline (reviewer 2026-10-19): a uniform random pick per pair is expected to land in the blind pass set 3.5 of 7 times (sum over pairs of the pair's blind pass share: 035.1.003/vehicle 0.00, 265.2.003/vehicle 0.50, 265.2.005/vehicle 0.75, 265.2.005/human 0.25, 284.0.006/vehicle 0.25, 370.1.008/human 0.75, 404.1.004/vehicle 1.00); the headline is read against this number.