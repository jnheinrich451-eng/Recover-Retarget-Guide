# Module 3 — render target per pair (reviewer 2026-10-10): the anchor's background moved by the bridged camera

| pair | bridge | points | pixel coverage mean / last | mask (px) mean / last | latent mask mean / last | erode px | object dilate px |
|---|---|---|---|---|---|---|---|
| 0000000265.2.005/human | rec | 399766 | 0.79 / 0.81 | 0.70 / 0.68 | 0.65 / 0.58 | 8 | 8 |
| 0000000265.2.005/human | rec_nodyn | 399766 | 0.79 / 0.81 | 0.47 / 0.46 | 0.41 / 0.36 | 8 | 8 |
| 0000000265.2.005/human | rec_sky | 399766 | 0.79 / 0.81 | 0.70 / 0.68 | 0.65 / 0.58 | 8 | 8 |
| 0000000265.2.005/human | rec_sky_nodyn | 399766 | 0.79 / 0.81 | 0.09 / 0.12 | 0.07 / 0.08 | 8 | 8 |
| 00000265.2.005/vehicle | far | 190633 | 0.39 / 0.41 | 0.25 / 0.21 | 0.21 / 0.15 | 8 | 8 |
| 00000265.2.005/vehicle | rec | 181024 | 0.34 / 0.35 | 0.20 / 0.18 | 0.17 / 0.12 | 8 | 8 |
| 00000265.2.005/vehicle | rec_obj | 181024 | 0.41 / 0.41 | 0.30 / 0.28 | 0.17 / 0.12 | 8 | 8 |
| 00000265.2.005/vehicle | rec_obj_sky | 181024 | 0.41 / 0.41 | 0.69 / 0.67 | 0.54 / 0.50 | 8 | 8 |
| 00000265.2.005/vehicle | rec_sky | 181024 | 0.34 / 0.35 | 0.59 / 0.57 | 0.54 / 0.50 | 8 | 8 |

Review at full resolution: `render_<clip>_<cat>.png` (darkened = outside the mask) and `render.mp4` in the pair's cache folder. The mask is where the correction applies; everything else is left to the model. Seams at the mask edge and render artefacts are the known risks (reviewer); the pixel grader stays the arbiter.
