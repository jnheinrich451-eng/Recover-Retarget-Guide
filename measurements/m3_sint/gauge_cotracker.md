# Module 3 — the gauge with CoTracker's tracks (independent of D4RT's 2-D tracks; reviewer 2026-10-07k)

| clip | camera | tracks static / with Z_lift (rejected at depth edges) | split px CoTracker (n) / D4RT | Z_tri CoTracker ÷ Z_tri D4RT at matched points: median [CI] (n); by row top/middle/bottom | BAND ratio [CI] (n; fit → verify) → outcome | by lift depth near/mid/far | by triangulated depth near/mid/far | by image row top/middle/bottom |
|---|---|---|---|---|---|---|---|---|
| 000000000035.1.003 | ba | 2093 / 386 (517) | 8.58 (103) / 4.16 | 1.104 [0.899, 2.058] (n=67); 3.679/0.988/0.603 | - [-] (n=0; - → -) → undetermined (n=0<30) | - | - | - |
| 000000000035.1.003 | vipe_over_sigma | 2093 / 386 (517) | 5.29 (214) / 5.67 | 0.636 [0.592, 0.757] (n=138); 0.815/1.018/0.515 | - [-] (n=0; - → -) → undetermined (n=0<30) | 3.439/1.919 | 1.254/3.424/4.280 | 3.355/3.424/1.633 |
| 000000000035.1.003 | ba | 2093 / 386 (517) | 8.58 (103) / 4.16 | 1.104 [0.899, 2.058] (n=67); 3.679/0.988/0.603 | - [-] (n=0; - → -) → undetermined (n=0<30) | - | - | - |
| 000000000035.1.003 | vipe_over_sigma | 2093 / 386 (517) | 5.29 (214) / 5.67 | 0.636 [0.592, 0.757] (n=138); 0.815/1.018/0.515 | - [-] (n=0; - → -) → undetermined (n=0<30) | 3.439/1.919 | 1.254/3.424/4.280 | 3.355/3.424/1.633 |
| 000000000265.2.003 | ba | 2629 / 1022 (201) | 0.86 (2478) / 0.88 | 1.053 [1.031, 1.091] (n=1598); 3.097/1.033/0.942 | 1.180 [1.143, 1.224] (n=231; 1.221 → 1.152) → undetermined (half-split or interval wider than tol) | 1.101/1.149/1.945 | 1.100/1.098/2.240 | 2.232/1.098/1.101 |
| 000000000265.2.003 | vipe_over_sigma | 2629 / 1022 (201) | 0.76 (2480) / 1.28 | 1.053 [1.028, 1.094] (n=1598); 3.073/1.034/0.941 | 1.202 [1.154, 1.251] (n=231; 1.251 → 1.174) → undetermined (half-split or interval wider than tol) | 1.087/1.140/1.903 | 1.085/1.089/2.209 | 2.193/1.089/1.086 |
| 000000000265.2.005 | ba | 4265 / 154 (1) | 1.09 (4125) / 0.65 | 3.149 [3.026, 3.301] (n=1425); 5.767/3.928/1.146 | - [-] (n=28; - → -) → undetermined (n=28<30) | 1.346/3.640/3.354 | 1.346/3.394/4.312 | 3.835/3.536/1.349 |
| 000000000265.2.005 | vipe_over_sigma | 4265 / 154 (1) | 0.68 (4112) / 1.33 | 3.449 [3.297, 3.665] (n=1423); 8.037/4.569/1.122 | - [-] (n=28; - → -) → undetermined (n=28<30) | 1.293/3.735/3.320 | 1.293/3.403/6.403 | 3.936/3.504/1.344 |
| 000000000370.1.008 | ba | 4737 / 1222 (166) | 0.64 (3107) / 0.65 | 0.968 [0.958, 0.980] (n=1427); 0.987/0.991/0.949 | 0.992 [0.982, 0.999] (n=509; 0.988 → 0.996) → determined | 1.014/1.017/1.038 | 1.002/1.020/1.059 | 1.057/0.993/1.016 |
| 000000000370.1.008 | vipe_over_sigma | 4737 / 1222 (166) | 1.04 (3103) / 1.12 | 0.978 [0.969, 0.989] (n=1421); 1.007/1.000/0.957 | 1.030 [1.022, 1.034] (n=509; 1.029 → 1.030) → determined | 1.064/1.033/1.067 | 1.048/1.031/1.113 | 1.085/1.022/1.058 |
| 000000000404.1.004 | ba | 3621 / 915 (78) | 1.26 (3063) / 0.52 | 1.414 [1.366, 1.460] (n=1567); 1.423/1.913/1.079 | 1.292 [1.275, 1.309] (n=306; 1.283 → 1.299) → determined | 1.287/1.576/1.449 | 1.191/1.356/2.162 | 1.288/1.674/1.291 |
| 000000000404.1.004 | vipe_over_sigma | 3621 / 915 (78) | 0.59 (3057) / 1.25 | 1.759 [1.660, 1.832] (n=1562); 1.727/2.831/1.097 | 1.088 [1.070, 1.168] (n=306; 1.080 → 1.091) → determined | 1.077/1.644/2.003 | 1.025/1.412/2.833 | 1.241/2.143/1.086 |

| clip | β CoTracker | β D4RT | band parallax p90 px | ΔF px / patch | rule outcome |
|---|---|---|---|---|---|
| 000000000035.1.003 | - | - | - | - / - | needs both band ratios and the D4RT band parallax (rerun refine_m3_camera_ba.py) |
| 000000000265.2.003 | 1.180 | 1.149 | 22.8 | 0.61 / 0.124 | immaterial |
| 000000000265.2.005 | - | 1.167 | - | - / - | needs both band ratios and the D4RT band parallax (rerun refine_m3_camera_ba.py) |
| 000000000370.1.008 | 0.992 | 1.067 | 68.5 | 4.79 / 0.974 | interval + sensitivity run |
| 000000000404.1.004 | 1.292 | 1.100 | 95.1 | 16.60 / 3.371 | interval + sensitivity run |

**000000000035.1.003, bundle adjustment on CoTracker's static tracks** (solve 236 points / 3134 obs, held out 1062): reproj RMS 4.82 → 1.30 px; held-out split 8.68 (initial cameras) → **3.47 px** (p90 5.71; n=104; gate fail); rotation change vs camera-C-BA 34.0°, translation change 399% max; band ratio - (n=0) → undetermined (n=0<30). Camera saved as cache/m3/<clip>/camera_cotracker_ba.npz (candidate, consumed by nothing).

Reading fixed in advance: the same near excess with CoTracker's tracks as with D4RT's -> the lift is the cause (tracks cleared); a smaller or absent excess -> D4RT's tracks lag on the nearest points and the D4RT-track ratio overstates the bias. The band ratio from CoTracker is the one to set beside the D4RT-track band ratio before any bridge amendment; Z_lift here is a nearest-neighbour proxy from D4RT's static tracks (stated above).
