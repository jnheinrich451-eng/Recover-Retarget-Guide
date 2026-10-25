# Blind qualitative verdicts on the 15 completed pairs (John, 2026-10-20, full resolution, written before any paired number was read)

**Blindness statement (D8 condition, reviewer 2026-10-20).** The pipeline grader (scripts/measure_m3_a0.py) ran inside the batch and its
per-video rows exist on Drive (reports/m3_s4/a0_motion.json under the experiment root). Nobody has read a per-pair ours-against-A0 number
from them: CC's reads of the record checked only that the rows exist (the "graded: yes" column), and John has read none. The poster grader
(the scoring agent, docs/poster-scoring-rules-v1.md) has not run on this export. D8 (option (a): all 15 pairs scored, flags per row) is
committed with this file, before the grader's re-score is opened.

Verdicts as written by John after watching A0 (backbone) and A3 (ours) of each pair; A0 = "no guidance", A3 = "ours". Transcribed verbatim
except for the pair id.

| pair | verdict |
|---|---|
| 000000000191.6.037/human | A0 the man running to left, not correct; A3 the man running forward, with camera tracking, but gets distorted in later frames. |
| 000000000200.1.006/human | A0 the man walking to right, not correct; A3 the man and camera movement are correct, but gets shaded by the tree, it maybe detects the tree as foreground. The motion and camera are both correct, maybe the generated image is not perfect. |
| 000000000217.0.006/vehicle | A0 the car is still, but the pedestrians are moving; A3 movements are correct, but gets distorted in later frames. |
| 000000000221.7.011/human | A0 and A3 movement alike, but A0 without camera tracking; background gets distorted in A3, subject is clear. |
| 000000000252.1.018/vehicle | A0 the car moves slowly forward; A3 moves as the reference video, but the generated image does not contain the slope, so the background gets distorted; both motions copied faithfully. |
| 000000000252.2.007/human | A0 the man walks very slowly to the right; A3 copies the direction and speed from the reference, but later frames distorted, subject is vague. |
| 000000000256.3.010/human | A0 the man walks to the right, the proxy alignment is not quite accurate; A3 copies camera and subject motion faithfully, later frames little distortion. |
| 000000000265.4.002/human | A0 no motion copy, moves forward slowly; A3 copies both motions, the background cannot adapt to the speed it moves, has distortion. |
| 000000000362.2.005/human | Both low fidelity, maybe due to the anchor image not being good. Subject is vague, but A3 can still copy motion. |
| 000000000391.0.016/human | A0 almost no motion; A3 the subject has no motion of its own, but the camera has movement and the subject translates. |
| 000000000396.3.002/vehicle | A0 the car moves slowly; A3 the car follows both motions, in later frames has distortion, some parts vanish; in the reference the car moves down a slope, but in the reference image the slope is not apparent. |
| 000000000404.1.006/human | A0 moves to the right in slow motion; A3 restores the motion faithfully, including the collision; but the human motion is unnatural and the fidelity is low. |
| 000000000431.5.009/human | A0 a man moves forward slowly; A3 restores both motions well, and the fidelity is better. A good candidate for presentation. |
| horsejump-high/human | A0 a man running on the left half of the video, no jumping; A3 the fidelity is too low, the foreground maybe influences the behaviour, motion cannot be identified, camera motion is kept. |
| camel/human | A0 a man walks to the right slowly; A3 the man walks to the right and at the end turns inward, matching the reference; end frames have minor distortion. |

**John's reading (verbatim, lightly punctuated).** The first generated anchor image matters; of the 15 only one has an unclear subject
(362.2.005), which speaks for the automatic selector. The distortion comes with the motion displacement against the image: the background,
or a mis-rendered foreground, hides the subject; some of it reads as shading. Almost all motion is copied faithfully: the guidance works.
In A3 some subject motion is carried over from A0, so it depends on the model's own understanding of the motion, which is what the method
expects of it.

Not to be revised after the numbers are read (rule 19).
