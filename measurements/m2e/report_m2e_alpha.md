# m2_e step 1 — alpha (ground-band fit, object-mask check)

| clip | cat | disp. | alpha (band) | CI32 | CI64 | alpha (object, uncorrected) | band/obj (uncorr.) | naive/reversed (band) | rho_full | n_band | source | flag | per-candidate spread rel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 35.1.003 | vehicle | accepted | 0.2344 | [0.2295, 0.2397] | [0.2261, 0.2449] | 0.3373 | 0.695 | 0.965 | +0.760 | 300034 | ground_band |  | 0.15664776142085937 |
| 84.0.006 | vehicle | accepted | 0.3632 | [0.3585, 0.3681] | [0.3540, 0.3726] | 0.3522 | 1.031 | 0.980 | +0.969 | 346026 | ground_band |  | 0.2093825278935864 |
| 65.2.005 | vehicle | accepted | 0.0709 | [0.0704, 0.0715] | [0.0699, 0.0718] | 0.0941 | 0.754 | 0.991 | +0.858 | 517555 | ground_band |  | 0.09260537938423172 |
| 65.2.005 | human | accepted | 0.0758 | [0.0738, 0.0774] | [0.0726, 0.0791] | -5.4335 | -0.014 | 0.968 | +0.915 | 401049 | ground_band |  | 0.10596231742882518 |
| 70.1.008 | human | accepted | 0.2143 | [0.2115, 0.2166] | [0.2098, 0.2188] | 0.2156 | 0.994 | 0.986 | +0.973 | 388191 | ground_band |  | 0.02930184317255457 |
| 04.1.004 | vehicle | labelled | 0.1316 | [0.1307, 0.1325] | [0.1296, 0.1333] | 0.1636 | 0.804 | 0.993 | +0.857 | 343623 | ground_band |  | 0.06896487694808942 |
| 65.2.003 | vehicle | labelled | 0.0852 | [0.0846, 0.0859] | [0.0839, 0.0865] | 0.1168 | 0.730 | 0.989 | +0.743 | 421876 | ground_band |  | 0.11568946825513544 |

naive/reversed = Depth-Anything's empirical lambda on this set (reported, not predicted; review item 12).
band/obj (uncorrected) is NOT a flattening diagnostic: the uncorrected object fit absorbs the anchor-vs-proxy yaw (test 3c: ratio 1.42 with no flattening).
The delta-corrected object fit, the flattening flag and the fallback decision are made in the delta step (reviewer 2026-09-23).
Gates m2e_alpha_pixels_min / m2e_alpha_ci_rel_max are null (record only).
