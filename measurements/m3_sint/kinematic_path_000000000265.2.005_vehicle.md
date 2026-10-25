# Kinematic object path — 000000000265.2.005/vehicle (unicycle fit on the ground plane, reviewer 2026-10-12 §2)

Rate: latent (4-frame): step 2 predates the full-rate save; the stride (8 frames) is aliased at this rate; 21 samples. Tracker noise level (from the raw path's second differences) 0.0134 world units; fit residual RMS 0.0427 (target 0.0134; model order v deg 3, κ deg 2). Path length raw 1.693 → fit 1.586.

Speed per frame: median 0.0930 (min 0.0113, max 0.1032). Curvature: median -2.7359 (right), net heading change -181.9°, max |Δψ| 181.9°.

| frame | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ψ − ψ₀ (deg) | +0.0 | +11.6 | +13.2 | +5.7 | -11.1 | -37.7 | -72.5 | -111.3 | -146.6 | -170.8 | -181.9 |
| v (per frame) | 0.1032 | 0.0936 | 0.0924 | 0.0955 | 0.0989 | 0.0986 | 0.0905 | 0.0724 | 0.0461 | 0.0203 | 0.0113 |
| κ | +1.221 | +0.359 | -0.488 | -1.319 | -2.135 | -2.935 | -3.720 | -4.490 | -5.244 | -5.983 | -6.346 |

in-plane unicycle fit; height = raw height Savitzky-Golay smoothed; R_t = yaw about the handoff's ground normal n_plane by psi_t - psi_0; the _kin geometry is written beside the geometry of record and is not consumed until the render and the sampler are pointed at it
