# Module 3 — object-ground contact check (report only; the per-pair decision is John's at the S-int hard stop)

Corrected column (reviewer 2026-10-06d): a plane is fitted by RANSAC to k_bg x MoGe ground within 48 px of the object (inlier tol 0.02, at least 200 inliers); r_c = G_tar depth at the base's bottom edge / depth where the same ray meets that plane. Base = lowest 0.03 of the object's height. Tolerance 0.05 (unchanged).

Instrument validation (tests/test_m3_contact.py, standing box, no noise): corrected 1.003 / 1.003 / 1.005 / 0.996 at depth 5 and 9, pitch 0 and 0.25; raw band column 1.044 / 1.045 / 1.083 / 1.071 on the same scenes.

| pair | disp. | median ratio (old gate) | **r_c corrected** | CI95 | base cols | plane inliers (frac) | plane resid | raw band r_c (diagnostic) | upper-body ratio | bottom-edge offset px (G_tar − painted) | ground depth change per row | r_c with ground at the PAINTED edge | reading (corrected column) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 000191.6.037/human | experiment | 1.624 (fail) | **1.580** | [1.580, 1.580] | 25 | 14656 (0.99) | 0.0052 | 1.681 (200 px) | 1.617 | -4.0 | 0.0056 | 1.632 | contact fails: object re-gauged to the ground, s_c = 0.6126 (displacement scaled alike; D-M3-03 for this pair) |
| 000200.1.006/human | experiment | 1.110 (pass) | **1.092** | [1.092, 1.092] | 26 | 9811 (0.58) | 0.0032 | 1.100 (208 px) | 1.126 | -1.0 | 0.0051 | 1.098 | contact fails: object re-gauged to the ground, s_c = 0.9110 (displacement scaled alike; D-M3-03 for this pair) |
| 0217.0.006/vehicle | experiment | 1.073 (pass) | **1.037** | [1.024, 1.037] | 14 | 20104 (1.00) | 0.0015 | 1.057 (112 px) | 1.032 | 1.0 | 0.0067 | 1.030 | contact holds: carry forward |
| 000221.7.011/human | experiment | 1.140 (pass) | **1.095** | [1.091, 1.095] | 31 | 19536 (0.99) | 0.0073 | 1.113 (248 px) | 1.141 | -3.0 | 0.0035 | 1.105 | contact fails: object re-gauged to the ground, s_c = 0.9046 (displacement scaled alike; D-M3-03 for this pair) |
| 0252.1.018/vehicle | experiment | 1.114 (pass) | **1.196** | [1.194, 1.256] | 23 | 18840 (1.00) | 0.0012 | 1.272 (184 px) | 1.107 | 27.0 | 0.0065 | 0.972 | contact holds: carry forward |
| 000252.2.007/human | experiment | 1.307 (fail) | **1.313** | [1.302, 1.314] | 21 | 12310 (0.66) | 0.0102 | 1.334 (168 px) | 1.314 | -7.0 | 0.0038 | 1.348 | contact fails: object re-gauged to the ground, s_c = 0.7418 (displacement scaled alike; D-M3-03 for this pair) |
| 000256.3.010/human | experiment | 1.035 (pass) | **1.052** | [1.052, 1.052] | 18 | 10922 (0.84) | 0.0021 | 1.088 (144 px) | 1.027 | 0.0 | 0.0087 | 1.052 | contact fails: object re-gauged to the ground, s_c = 0.9502 (displacement scaled alike; D-M3-03 for this pair) |
| 000265.4.002/human | experiment | 1.206 (fail) | **1.114** | [1.114, 1.114] | 7 | 7805 (0.62) | 0.0083 | 1.335 (56 px) | 1.201 | 16.0 | 0.0060 | 1.013 | contact holds: carry forward |
| 000362.2.005/human | experiment | 3.122 (fail) | **0.823** | [0.823, 0.823] | 54 | 8173 (0.40) | 0.0082 | 7.023 (7 px) | 2.838 | 12.5 | 0.0024 | 0.801 | object cut by the frame bottom: no contact line in view (ratios above are NOT contact readings) |
| 000391.0.016/human | experiment | 1.955 (fail) | **1.692** | [1.692, 1.692] | 35 | 14819 (1.00) | 0.0075 | 1.863 (224 px) | 1.954 | 0.0 | 0.0058 | 1.692 | contact fails: object re-gauged to the ground, s_c = 0.5911 (displacement scaled alike; D-M3-03 for this pair) |
| 0396.3.002/vehicle | experiment | 0.846 (fail) | **0.837** | [0.836, 0.838] | 49 | 16791 (1.00) | 0.0025 | 0.839 (392 px) | 0.877 | 17.0 | 0.0016 | 0.814 | contact fails: object re-gauged to the ground, s_c = 1.2282 (displacement scaled alike; D-M3-03 for this pair) |
| 000404.1.006/human | experiment | 1.515 (fail) | **1.503** | [1.503, 1.503] | 7 | 15904 (1.00) | 0.0034 | 1.520 (56 px) | 1.523 | -1.0 | 0.0069 | 1.513 | contact fails: object re-gauged to the ground, s_c = 0.6610 (displacement scaled alike; D-M3-03 for this pair) |
| 000431.5.009/human | experiment | 1.056 (pass) | **1.103** | [1.101, 1.104] | 18 | 14847 (1.00) | 0.0030 | 1.121 (144 px) | 1.047 | -2.0 | 0.0042 | 1.110 | contact fails: object re-gauged to the ground, s_c = 0.9007 (displacement scaled alike; D-M3-03 for this pair) |
| camel/human | experiment | 1.231 (fail) | **1.187** | [1.187, 1.187] | 49 | 24295 (0.84) | 0.0025 | 1.244 (392 px) | 1.223 | -2.0 | 0.0036 | 1.198 | contact fails: object re-gauged to the ground, s_c = 0.8350 (displacement scaled alike; D-M3-03 for this pair) |
| rsejump-high/human | experiment | 0.840 (fail) | **0.848** | [0.844, 0.851] | 40 | 15359 (0.83) | 0.0064 | 0.874 (320 px) | 0.840 | -3.0 | 0.0075 | 0.860 | contact fails: object re-gauged to the ground, s_c = 1.1624 (displacement scaled alike; D-M3-03 for this pair) |

A low inlier fraction or a large plane residual means the ground near the object is not one plane in the depth model's output; read that row with care.

Edge-offset diagnostic: (offset px) x (depth change per row) is the part of r_c - 1 explained by G_tar's silhouette ending on a different row from the painted object, with no depth error. The last ratio column removes it. It is a diagnostic; the pre-registered reading is unchanged.
