# m2_d step 7 — selection

Clips selected: 1; awaiting the human's ranks: 0.

| gate | fails | ungated | exercised |
|---|---|---|---|
| g_e_area | 0 | 0 | False |
| g_e_loc_rel | 0 | 0 | False |
| g_eps_shape | 0 | 4 | False |
| g_heading | 0 | 0 | False |
| g_object_found | 0 | 0 | False |

**All Phase-1 conditions passed on every one of the 4 candidates, so none was exercised.** The thresholds are derived from failure semantics (half the object's own scale; twice / half the requested area), not from this sample, and remain UNVALIDATED, in the same category as the dictionary's ±45° boundary and the low-angle ratio. With every candidate through Phase 1, **selection here is entirely Phase 2, which is manual**: the anchor is chosen by the human's eye among the seeds, and the lexicographic design's value on this set is defensive, not discriminative.

| clip | cat | outcome | survivors | selected | E_loc_rel max | E_area max | eps_shape (ungated) | band-edge flag |
|---|---|---|---|---|---|---|---|---|
| 65.2.003 | vehicle | selected | 4/4 | 2 | 0.015 | 0.026 | 0.630 | False |

Known property (reviewer): E_area has a size-dependent floor ~ perimeter / area; at 0.7 % object size (265/human, +0.33 to +0.47) the metric is close to uninformative and the log 2 threshold has only a 1.5x margin there.

HARD STOP (SKILL §10): human review of the anchors before m2_e.
