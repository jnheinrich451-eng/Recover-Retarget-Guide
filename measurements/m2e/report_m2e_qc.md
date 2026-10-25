# m2_e — QC summary

| clip | cat | disp. | alpha | source | band/obj (corr.) | delta (deg) | instrument | width | front/back margin | g_front_back | vs G_tar (scan) | G_tar shape yaw | flatten-pred. yaw | R2 IoU (n ok) | fallback | z_obj / z_C1 | z_obj / z_proxy_vis | z_obj spread | obj in band range | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 35.1.003 | vehicle | accepted | 0.2344 | ground_band | 0.688 | +0.78 | silhouette_scan | 2.0 | 0.105 | pass | 7.1 | 4.5 | 6.7 | 0.773 (None) |  | 0.797 | 1.057 | 0.165 | True (1.0) | within_object_scale_error |
| 84.0.006 | vehicle | accepted | 0.3632 | ground_band | 1.034 | +1.30 | silhouette_scan | 1.0 | 0.076 | pass | 2.3 | 3.1 | -0.0 | 0.401 (5) | vendor_internal + gauge_align | 0.751 | 0.987 | 0.06 | True (1.0) |  |
| 65.2.005 | vehicle | accepted | 0.0709 | ground_band | 0.755 | -0.05 | silhouette_scan | 4.0 | 0.106 | pass | 24.3 | 2.8 | -3.5 | 0.738 (None) |  | 0.972 | 1.105 | 0.01 | True (1.0) | within_object_scale_error |
| 65.2.005 | human | accepted | 0.0758 | ground_band | None | -11.60 | keypoints | None | None | n/a (keypoints read front from back directly) | None | None | None | 0.764 (None) |  | 1.146 | 1.172 | 0.045 | True (1.0) |  |
| 70.1.008 | human | accepted | 0.2143 | ground_band | None | -12.30 | keypoints | None | None | n/a (keypoints read front from back directly) | None | None | None | 0.865 (None) |  | 1.138 | 1.168 | 0.062 | True (1.0) |  |
| 04.1.004 | vehicle | labelled | 0.1316 | ground_band | 0.805 | +0.11 | silhouette_scan | 0.0 | 0.106 | pass | -5.1 | -6.7 | -2.1 | 0.830 (None) |  | 0.764 | 1.163 | 0.056 | False (0.44) | within_object_scale_error |
| 65.2.003 | vehicle | labelled | 0.0852 | ground_band | 0.737 | -0.37 | silhouette_scan | 4.0 | 0.117 | pass | 5.7 | 1.6 | -7.0 | 0.837 (None) |  | 0.848 | 1.062 | 0.045 | False (0.34) | within_object_scale_error |

delta = yaw of the ANCHOR minus yaw of the POSED PROXY, positive about the UP plane normal (rot_about(n, +delta) turns the proxy onto the anchor); shared by the scan and the keypoint instrument. Read |delta| against the curve width, not as a point estimate (the scan resolves ~+-2 deg at best).
front/back margin = peak IoU(posed) - peak IoU(proxy flipped 180): small = ambiguous pair (a flipped anchor scores a plausible quiet delta; test 5).

Accepted-set summary (labelled pairs excluded): {"accepted_pairs": ["000000000035.1.003/vehicle", "000000000284.0.006/vehicle", "000000000265.2.005/vehicle", "000000000265.2.005/human", "000000000370.1.008/human"], "labelled_pairs_excluded": ["000000000404.1.004/vehicle", "000000000265.2.003/vehicle"], "alpha": {"n": 5, "median": 0.2142922674477744, "min": 0.07091094550200953, "max": 0.36319843972951865}, "delta_deg_abs": {"n": 5, "median": 1.299217122648597, "min": 0.046617970594704634, "max": 12.30182443525027}, "r2_iou_vs_anchor_mask": {"n": 5, "median": 0.7641223940820444, "min": 0.40081754648147333, "max": 0.8647833808284806}}

HARD STOP (SKILL §11): human review of G_tar renders and the delta curves before Module 3.
