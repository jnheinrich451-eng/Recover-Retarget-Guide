# Module 3 — camera C: D4RT re-solved from its own static tracks, chained across encoder-window boundaries (held-out gate)

| clip | t_a | keyframes | solve / held | frames solved | held-out px: all / first window / after boundaries (p90) | G on the same held-out: first / after | rot C vs G: median first / max | gate | object heading vs d_src @12f | jerk C / ViPE | observability: depth/path, worst-axis σ (src units/px) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 000000000035.1.003 | 4 | [4, 35, 47, 59] | 2318 / 714 | 77/77 | 7.23 / 1.57 / 8.10 (9.17) | 3.78 / 18.88 | 1.49 / 15.68 | **fail** (≤ 4.92 px) | 15.5 | 0.0469 / 0.0040 | 0.9, 0.0004 |
| 000000000265.2.003 | 47 | [47, 59] | 697 / 546 | 33/33 | 2.35 / 0.64 / 2.52 (2.65) | 0.91 / 3.32 | 0.17 / 1.44 | **pass** (≤ 4.92 px) | 69.2 | 0.0348 / 0.0026 | 5.1, 0.0020 |
| 000000000265.2.005 | 0 | [0, 35, 47, 59] | 408 / 360 | 81/81 | 1.99 / 0.65 / 3.05 (3.56) | 0.82 / 12.10 | 0.19 / 3.04 | **pass** (≤ 4.92 px) | 6.7 | 0.0236 / 0.0030 | 3.5, 0.0033 |
| 000000000370.1.008 | 0 | [0, 35, 47, 59] | 606 / 599 | 81/81 | 1.69 / 0.65 / 2.68 (3.12) | 0.76 / 3.53 | 0.09 / 0.78 | **pass** (≤ 4.92 px) | 2.2 | 0.0062 / 0.0177 | 2.9, 0.0010 |
| 000000000404.1.004 | 47 | [47, 59] | 837 / 476 | 33/33 | 1.10 / 0.38 / 1.15 (1.27) | 0.51 / 2.83 | 0.20 / 1.16 | **pass** (≤ 4.92 px) | 17.2 | 0.0252 / 0.0170 | 4.2, 0.0021 |

Reading (reviewer, pre-stated): inside the first window camera C should coincide with G to about the px already measured; the test is whether the chaining holds the error across the boundaries. A pass admits 035 under `trajectory.source = static_pnp` (fallback, gated); a fail leaves 035 'camera undetermined'. Jerk is in each camera's own units (camera C in the source gauge, ViPE bridged) and is a shape comparison only.
