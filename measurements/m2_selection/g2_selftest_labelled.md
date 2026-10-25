# G2 self-test, labelled aggregate (2026-10-19 02:05; run of record measurements/m2_selection/g2_selftest.json, labels measurements/census/contact_labels.csv)

Fixtures: composition median 1.02 px as record vs 44.9 px wrong way; the camera gate's own held-out median 1.22 px (n 32) -> composition fixture holds.
Frame-t_a hit (no camera composition involved): 13 of 23 in-contact clips; 0 of 3 negatives.

| clip | label | f_sup | frames in view | hit at t_a | composition px (record / wrong way / gate) |
|---|---|---|---|---|---|
| 000000000119.5.007 | contact_out_of_view | 0.429 | 21 | 0 | 1.40 / 24.6 / 1.67 |
| 000000000191.6.037 | contact_out_of_view | 0.143 | 21 | 0 | 0.74 / 59.5 / 0.89 |
| 000000000222.4.003 | contact_out_of_view | 0.000 | 21 | 0 | 0.53 / 30.2 / 0.58 |
| 000000000265.4.002 | contact_out_of_view | 0.333 | 21 | 0 | 0.77 / 38.9 / 1.09 |
| 000000000370.1.009 | contact_out_of_view | 0.000 | 21 | 0 | 0.51 / 23.7 / 0.59 |
| 000000000391.0.016 | contact_out_of_view | 0.048 | 21 | 0 | 0.44 / 15.2 / 0.49 |
| 000000000064.3.002 | in_contact | 0.882 | 17 | 1 | 0.62 / 69.0 / 0.79 |
| 000000000130.0.015 | in_contact | 1.000 | 21 | 1 | 1.18 / 95.4 / 1.64 |
| 000000000200.1.006 | in_contact | 1.000 | 21 | 1 | 0.90 / 39.2 / 1.22 |
| 000000000217.0.006 | in_contact | 0.048 | 21 | 1 | 1.01 / 47.8 / 1.21 |
| 000000000221.7.011 | in_contact | 0.000 | 21 | 0 | 1.04 / 63.1 / 1.44 |
| 000000000252.1.018 | in_contact | 0.667 | 21 | 0 | 0.83 / 109.3 / 1.46 |
| 000000000252.2.007 | in_contact | 0.286 | 21 | 0 | 1.42 / 111.9 / 1.96 |
| 000000000256.3.010 | in_contact | 0.333 | 21 | 1 | 1.03 / 40.3 / 1.40 |
| 000000000362.2.005 | in_contact | 0.286 | 21 | 0 | 0.76 / 44.0 / 0.87 |
| 000000000396.3.002 | in_contact | 0.000 | 21 | 0 | 1.30 / 40.9 / 1.75 |
| 000000000404.1.006 | in_contact | 0.143 | 21 | 1 | 1.70 / 73.1 / 2.24 |
| 000000000431.5.009 | in_contact | 0.095 | 21 | 0 | 0.69 / 45.8 / 1.03 |
| bmx-bumps | in_contact | 1.000 | 9 | 1 | 1.95 / 9.9 / 1.86 |
| breakdance | in_contact | 1.000 | 21 | 1 | 0.82 / 2.1 / 0.86 |
| breakdance-flare | in_contact | 0.476 | 21 | 1 | 0.34 / 11.3 / 0.33 |
| camel | in_contact | 1.000 | 21 | 1 | 1.63 / 71.2 / 1.45 |
| dancing | in_contact | 0.000 | 21 | 0 | 3.23 / 17.7 / 3.28 |
| dogs-scale | in_contact | 0.571 | 21 | 0 | 2.08 / 73.4 / 1.93 |
| elephant | in_contact | 0.000 | 21 | 0 | 1.57 / 24.2 / 1.59 |
| goat | in_contact | 0.905 | 21 | 0 | 1.15 / 104.6 / 1.22 |
| hockey | in_contact | 1.000 | 21 | 1 | 0.97 / 56.7 / 1.06 |
| horsejump-high | in_contact | 0.571 | 21 | 1 | 2.35 / 135.0 / 2.01 |
| kite-walk | in_contact | 1.000 | 21 | 1 | 0.85 / 9.3 / 0.79 |
| 000000000299.1.005 | not_in_contact | 0.000 | 21 | 0 | 1.29 / 118.9 / 1.16 |
| disc-jockey | not_in_contact | 0.000 | 21 | 0 | 1.93 / 59.3 / 1.40 |
| gold-fish | not_in_contact | 0.000 | 21 | 0 | 0.41 / 24.7 / 0.41 |

Contact set n 23: min 0.000, median 0.571, sigma_hat 0.400 -> tau_sup rule gives 0.800; contact clips below 0.8: 14 (000000000217.0.006, 000000000221.7.011, 000000000252.1.018, 000000000252.2.007, 000000000256.3.010, 000000000362.2.005, 000000000396.3.002, 000000000404.1.006, 000000000431.5.009, breakdance-flare, dancing, dogs-scale, elephant, horsejump-high).
Negative controls: {'000000000299.1.005': 0.0, 'disc-jockey': 0.0, 'gold-fish': 0.0} (at or above tau: none). Contact out of view (reported, neither set): {'000000000119.5.007': 0.42857142857142855, '000000000191.6.037': 0.14285714285714285, '000000000222.4.003': 0.0, '000000000265.4.002': 0.3333333333333333, '000000000370.1.009': 0.0, '000000000391.0.016': 0.047619047619047616}.

**G2 self-test FAILED as measured: G2 is dropped and disclosed (committed rule); path support stays reported, never gated**