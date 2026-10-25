# Wan frame -> reference cut mapping per pair (grader Q1), from Stage A's provenance (rule column corrected 2026-10-20 from the recorded values)

| pair | pool | Stage A file (recorded path) | native W x H | frames | cut W x H / frames | is the cut | rule |
|---|---|---|---|---|---|---|---|
| 000000000191.6.037/human | miradata | /content/data/experiment-testset/miradata/000000000191.6.037.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| 000000000200.1.006/human | miradata | /content/data/experiment-testset/miradata/000000000200.1.006.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| 000000000217.0.006/vehicle | miradata | /content/data/experiment-testset/miradata/000000000217.0.006.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| 000000000221.7.011/human | miradata | /content/data/experiment-testset/miradata/000000000221.7.011.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| 000000000252.1.018/vehicle | miradata | /content/data/experiment-testset/miradata/000000000252.1.018.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| 000000000252.2.007/human | miradata | /content/data/experiment-testset/miradata/000000000252.2.007.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| 000000000256.3.010/human | miradata | /content/data/experiment-testset/miradata/000000000256.3.010.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| 000000000265.4.002/human | miradata | /content/data/experiment-testset/miradata/000000000265.4.002.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| 000000000362.2.005/human | miradata | /content/data/experiment-testset/miradata/000000000362.2.005.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| 000000000391.0.016/human | miradata | /content/data/experiment-testset/miradata/000000000391.0.016.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| 000000000396.3.002/vehicle | miradata | /content/data/experiment-testset/miradata/000000000396.3.002.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| 000000000404.1.006/human | miradata | /content/data/experiment-testset/miradata/000000000404.1.006.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| 000000000431.5.009/human | miradata | /content/data/experiment-testset/miradata/000000000431.5.009.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| bmx-bumps/human | davis | /content/data/experiment-testset/DAVIS/bmx-bumps.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| camel/human | davis | /content/data/experiment-testset/DAVIS/camel.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |
| horsejump-high/human | davis | /content/data/experiment-testset/DAVIS/horsejump-high.mp4 | [720, 480] | 24 | [720, 480] / 24 | True | (a) |

Reading: Stage A consumed the experiment-testset file of every pair (the census session's download folder was /content/data/experiment-testset/; the same dataset as the hashed cuts), at 720 x 480 and 24 frames, so the Wan frame is the WHOLE cut resized to 832 x 480: rule (a) on all 15. The flat 16:9 originals on the Phase-2 VM were never consumed by Stage A. K_wan per pair is in the eval pack; K_cut = diag(720/832, 1, 1) K_wan.
