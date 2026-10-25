# Blind pass/fail of the 28 candidate anchors (John, 2026-10-17)

File: `blind_passfail_template.csv` (the verdicts were written into the template; the file keeps that name). Recorded before any
automatic pick existed under the v2 rule: commit 7b083e3 precedes the runner's rule step. Judging view: the candidate PNGs at full
resolution, no scores shown. 14 pass / 14 fail.

The reasons fall in two classes, which the thesis reports separately: **gross failure** (wrong class, broken anatomy, object off the
ground, distorted background: 265.2.005 vehicle s0 "cartoon car"; 265.2.005 human s1–s3 "background distorted"; 370.1.008 human s1
"twisted") and **layout** (the road's course or direction: 035 all seeds, 265.2.003 s0/s3, 284 s1–s3), which is what the G2 path-support
gate measures. The acceptable-pick rate k/7 is reported both ways: against the gross-failure set only, and against the full verdict set.
Verdicts are not re-taken; they stand as given.
