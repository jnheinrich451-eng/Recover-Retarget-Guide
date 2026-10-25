Median over 4 pairs (one seed each). Lower is better except Box-IoU.

| Method | CamRotErr (deg) lower | RPE (deg/frame) lower | RotErr, CameraCtrl def. lower | TransErr, CameraCtrl def. lower | Box-IoU higher | ObjMC (px) lower | ObjMC motion-only (px) lower | ObjErr3D lower |
|---|---|---|---|---|---|---|---|---|
| No guidance (CogVideoX) | 2.644 | 0.346 | 0.969 | 1.337 | 0.277 | 82.595 | 36.853 | 0.017 |
| DiTFlow (baseline) | 1.603 | 0.198 | 0.588 | 0.910 | 0.331 | 65.452 | 11.613 | 0.041 |
| No guidance (Wan 2.1 I2V) | 4.674 | 0.403 | 1.713 | 1.562 | 0.324 | 83.790 | 14.438 | 0.039 |
| Our method | 0.536 | 0.097 | 0.196 | 0.806 | 0.227 | 107.664 | 9.955 | 0.022 |
