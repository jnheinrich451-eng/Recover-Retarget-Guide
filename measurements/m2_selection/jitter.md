# Selection calibration: instrument jitter and the MUSIQ floor

28 of 28 candidates found by the detector.

| quantity | within-candidate sigma (pooled) | between-candidate sd of the base value | lock value = 2 sigma |
|---|---|---|---|
| iou | 0.0509 | 0.0994 | 0.1017 |
| clip | 0.0080 | 0.0187 | 0.0159 |
| m_H | 0.0170 | 0.1377 | 0.0339 |

MUSIQ over 140 census source frames: 5th percentile 38.31406497955322, median 65.71485137939453, min 19.203975677490234 -> tau_q = 38.31406497955322

Reading: if the within-candidate sigma of IoU is of the order of the between-candidate sd, the IoU stage does not discriminate between seeds and CLIP decides (reported as a finding, not adjusted).

Lock lines (John writes them; values are the rule's, not a choice):

    eps_iou: 0.1017, eps_clip: 0.0159, delta_h: 0.0339, tau_q: 38.31406497955322