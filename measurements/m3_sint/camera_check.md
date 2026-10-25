# Module 3 — camera check: static tracks lifted at t_a, reprojected through each camera (px at 256², median)

One row per source clip. `same window` = frames in t_a's encoder window; `other` = after a window boundary (boundary ±1 frame excluded).

| clip | t_a (on boundary?) | static pts | D4RT: all / same / other | ViPE÷σ̂(t_a): all / same / other | ViPE÷sim3_s: all / same / other | rot D4RT-vs-ViPE °: median / max | sim3_s/σ̂−1 | hybrid (D4RT rot + ViPE trans): all | best-fit scale ÷ σ̂ (px there; flat?) | pure-D4RT heading vs d_src ° at 4/8/12/16 frames |
|---|---|---|---|---|---|---|---|---|---|---|
| 000000000035.1.003 | 4 (False) | 700 | 8.71 / 3.50 / 24.44 | nan / 16.50 / nan | 34.75 / 13.44 / 54.99 | 3.46 / 12.02 | +1.274 | nan | 3.00 (30.10; False) | 20.1 / 3.8 / 1.7 / 7.8 |
| 000000000265.2.003 | 47 (True) | 1136 | 3.27 / 0.97 / 3.53 | 3.61 / 0.85 / 3.97 | 3.82 / 1.09 / 4.32 | 1.21 / 2.04 | +0.231 | 5.43 | 1.06 (3.37; False) | 64.9 / 86.8 / 55.0 / 62.9 |
| 000000000265.2.005 | 0 (False) | 800 | 7.68 / 0.87 / 14.52 | 4.06 / 3.67 / 4.24 | 5.41 / 4.77 / 5.64 | 0.87 / 1.92 | -0.089 | 5.60 | 1.19 (3.15; False) | 16.2 / 19.5 / 2.1 / 3.6 |
| 000000000370.1.008 | 0 (False) | 957 | 2.07 / 0.73 / 4.15 | 3.55 / 3.06 / 4.25 | 2.93 / 2.69 / 3.62 | 0.39 / 1.02 | +0.062 | 3.93 | 1.29 (1.94; False) | 13.5 / 12.1 / 1.9 / 12.1 |
| 000000000404.1.004 | 47 (True) | 891 | 2.61 / 0.54 / 2.84 | 1.75 / 0.77 / 2.41 | 1.98 / 0.79 / 3.51 | 0.69 / 2.66 | +0.134 | 2.33 | 0.85 (1.71; False) | 30.9 / 20.9 / 22.6 / 24.2 |

Reading (reviewer, pre-stated): the camera with the smaller error on the observed static points is the one in the gauge of the lifted scene. If ViPE÷sim3_s beats ViPE÷σ̂ on the x0 clips, the translation bridge should use the trajectory scale. A large rotation column means the two cameras disagree in rotation, which no scale repairs. The last column needs no ViPE: if it matches d_src while the ViPE-composed heading does not, the camera is the cause; if it misses too, the cause is upstream of Module 3 (t_a on a window boundary, or d_src itself).
