# §2.6 alignment — 000000000265.2.005/vehicle: MoGe disparity = a · triangulated disparity + b

n = 137 held-out static points (110 Huber inliers); a = 0.4393 (1/a = 2.2764, against the band scalar k_bg = 0.3443), b = -0.00940 (-0.094 of the median disparity); residual MAD 0.01316 (disparity). Band ρ (lift / triangulation) was 1.167.

| depth / Z_tri at the points | overall | near | mid | far |
|---|---|---|---|---|
| before: k_bg · MoGe | 0.865 | 0.860 | 0.800 | 1.046 |
| after: aligned | 1.021 | 1.026 | 0.939 | 1.104 |
| n | 137 | 46 | 45 | 46 |

Tercile levels after alignment spread 0.176 (registered ≤ 0.05: failed); the registration of record is the pixel bridge's tercile levels on the aligned lift (`check_m3_pixel_bridge.py --lift lift_aligned.npz`, β_pix at 1.0 ± 0.05 per tercile).
Valid fraction 0.588 → 0.588; background primitives 11913 → 11913.

Written: `lift_aligned.npz` beside `lift.npz`; nothing downstream reads it until `--apply` (archives lift.npz as lift_kbg_v1.npz), after which S-int step 2, the contact check, the bridge interval and the render are rerun for the pair.

regressor = triangulated disparity (camera-C BA, source gauge); response = MoGe disparity (5x5 median at the point); Huber IRLS. After alignment the background is in the camera's own gauge: beta applies to the object only, through the contact; s_delta / s_c / contact gate must be recomputed before use.
