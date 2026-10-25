# Module 3 — camera C: D4RT re-solved from its own static tracks, chained across encoder-window boundaries (held-out gate)

| clip | t_a | keyframes | solve / held | frames solved | held-out px: all / first window / after boundaries (p90) | G on the same held-out: first / after | rot C vs G: median first / max | gate | object heading vs d_src @12f | jerk C / ViPE | observability: depth/path, worst-axis σ (src units/px) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| soapbox/vehicle | 0 | [0] | 72 / 24 | 21/21 | 0.47 / 0.47 / - (0.64) | 0.78 / - | 0.22 / 0.41 | **pass** (≤ 4.92 px) | 60.1 | 0.0106 / n/a (IndexError) | 36.0, 0.0127 |

Reading (reviewer, pre-stated): inside the first window camera C should coincide with G to about the px already measured; the test is whether the chaining holds the error across the boundaries. A pass admits 035 under `trajectory.source = static_pnp` (fallback, gated); a fail leaves 035 'camera undetermined'. Jerk is in each camera's own units (camera C in the source gauge, ViPE bridged) and is a shape comparison only.
