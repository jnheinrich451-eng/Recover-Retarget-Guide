# Module 3 — camera C refined by bundle adjustment (held-out gate, same instrument as camera C)

| clip | prior px / anchors | points / obs (dropped: behind, outside, far) | reproj RMS before -> after (px) | init consistent? | held-out median px: camera C -> BA (p90) | x0 no-degradation | anchor-depth scale change | max rot change ° | gate |
|---|---|---|---|---|---|---|---|---|---|
| 000000000035.1.003 | 200 / 278 | 278 / 8200 (0, 265, 706) | 3.00 -> 1.38 | True | 7.23 -> **5.03** (6.48) | True | 1.000 | 7.0 | **fail** (≤ 4.92 px) |

Pre-registered expectation (reviewer 2026-10-07b): BA-refined camera C passes the 1-patch gate on 035 over the full window. A pass admits 035 under `trajectory.source = static_pnp_ba` (fallback, gated), with d_src on 035 recomputed from this camera (B.5 item).
