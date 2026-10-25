"""D7 (reviewer 2026-10-19): the Body adapter's principal-point re-centring by padding. CPU, no model; known answers.

  1. the nine experiment-set offsets that the 4 px contract refused land within half a pixel of the centre after the pad, with K updated;
  2. shift-crop fixture: a frame with a centred principal point, cropped by 20 px on the left (a 10 px offset), re-centred: the pad
     restores the original frame pixel-exactly where pixels exist, the mask, and K;
  3. 3-D invariance: a world point projects to the same content pixel under (K, frame) and (K', padded frame), so vertices need no
     correction and a 2-D keypoint returns to native coordinates by subtracting the pad before the height un-scale.

Run:  python tests/test_m2b2_recentre.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2b2 import body  # noqa: E402

OFFSETS = [(-8.23, 1.17), (-5.24, -2.48), (-14.20, 4.33), (-4.11, -0.24), (0.76, -7.40), (-4.25, -2.08), (8.07, -0.28), (2.27, -5.06), (-10.05, -0.57)]


def main():
    H = W = 1024; fx, fy = 1100.0, 1190.0
    frame = np.zeros((H, W, 3), np.uint8); frame[..., 0] = (np.arange(W) % 251).astype(np.uint8); frame[..., 1] = (np.arange(H)[:, None] % 241).astype(np.uint8)
    mask = np.zeros((H, W), bool); mask[400:900, 380:640] = True
    # 1. the nine refused offsets (after the rescale) -> re-centred within half a pixel, contract satisfied
    for dx, dy in OFFSETS:
        Hp = int(round(H * fx / fy)); a = Hp / H
        K = np.array([[fx, 0, W / 2 + dx], [0, fy, (Hp / 2 + dy) / a], [0, 0, 1.0]])
        sq = body.square_pixel_frame(frame, mask, K, 4.0, recentre=True)
        assert max(abs(sq.pp_offset_px[0]), abs(sq.pp_offset_px[1])) <= 0.5 + 1e-9, (dx, dy, sq.pp_offset_px)
        assert abs(sq.pp_offset_native_px[0] - dx) < 0.51 and abs(sq.pp_offset_native_px[1] - dy) < 0.51, (dx, dy, sq.pp_offset_native_px)
        assert sq.frame.shape[:2] == sq.mask.shape and sq.K[0, 2] == K[0, 2] + sq.pad[2]
    for dx, dy in OFFSETS[:3]:                                                   # without re-centring the contract still refuses them (the assert is kept)
        K = np.array([[fx, 0, W / 2 + dx], [0, fy, (int(round(H * fx / fy)) / 2 + dy) * fy / fx], [0, 0, 1.0]])
        if max(abs(dx), abs(dy)) > 4.0:
            try:
                body.square_pixel_frame(frame, mask, K, 4.0); raise RuntimeError("accepted")
            except AssertionError as e:
                assert "principal point" in str(e)
    # 2. shift-crop fixture: centred K, crop 20 px on the left -> cx moves 20 px, the centre 10 px: offset -10 px; re-centring pads 20 px back
    K0 = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])               # square pixels, centred: the rescale is the identity
    crop = 20; fr_c = frame[:, crop:].copy(); mk_c = mask[:, crop:].copy(); K_c = K0.copy(); K_c[0, 2] -= crop
    sq0 = body.square_pixel_frame(frame, mask, K0, 4.0, recentre=True); sq = body.square_pixel_frame(fr_c, mk_c, K_c, 4.0, recentre=True)
    assert sq.pad == (0, 0, crop, 0), sq.pad
    assert np.array_equal(sq.frame[:, crop:], frame[:, crop:]) and np.array_equal(sq.mask, mask) and np.allclose(sq.K, K0), "crop + pad did not restore the frame, the mask and K"
    assert sq0.pad == (0, 0, 0, 0) and np.array_equal(sq0.frame, frame)
    # 3. projection invariance: one world point, same content pixel under (K_c, cropped frame) and (K', padded frame); keypoint round trip to native
    X = np.array([0.25, 0.1, 6.0])
    u_c = K_c @ X; u_c = u_c[:2] / u_c[2]; u_p = sq.K @ X; u_p = u_p[:2] / u_p[2]
    assert np.allclose(u_p - u_c, [crop, 0]), (u_c, u_p)
    px_c = fr_c[int(u_c[1]), int(u_c[0])]; px_p = sq.frame[int(u_p[1]), int(u_p[0])]
    assert np.array_equal(px_c, px_p), "the padded frame moved the content under the projected point"
    kp_native = body.to_native(u_p[None] - np.array([sq.pad[2], sq.pad[0]], float), sq.scale_y)[0]
    assert np.allclose(kp_native, u_c), (kp_native, u_c)
    print("test_m2b2_recentre: ok (9 refused offsets re-centred to <= 0.5 px; the contract still refuses un-padded inputs; crop-20 fixture restored pixel-exactly; projection invariant; keypoint round trip)")


if __name__ == "__main__":
    main()
