"""m2_b2 — reconstructor adapter (SKILL_m2_b2_reconstructor_adapter.md v1.1).

Turns an image into a posed mesh in the camera frame plus a canonical basis. Owns
variant routing, the vendor composition, the point-map requirement, seed handling
and B_tar. Does NOT own the working-pose constants (m2_c), placement, background or
any FLUX call.

Sub-modules (build order §12):
  compose    step 1  E1 composition for SAM 3D Objects + scale-isotropy gate
  pointmap   step 1  OpenCV point maps from Stage A depth / exact fixtures; refusal
  silhouette step 1  the pixel-exact silhouette raster (runtime self-check)
  objects    step 1  the GPU wrapper around the vendor pipeline (imports it lazily)
"""
