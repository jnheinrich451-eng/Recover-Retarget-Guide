# Module 3 S-int step 2 — camera path, object path, coverage, anchors

R_t = I on every pair (rotation source not yet chosen). Coverage is the share of non-object patches in frame t that the frame-0 scene still explains.

| pair | ch | T_l | s_c | cam path / rot° | obj path | ViPE-vs-D4RT disp diff / path | heading vs d_src ° | cov t=1 / mid / last | same, sky excluded | anchors (sky excl.) | bg flow last: median / p95 patches | obj centre shift patches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Decisions needed from John (S-int hard stop)

1. Review per pair: is the object path plausible (heading vs d_src; ViPE-vs-D4RT difference), and does the coverage curve make sense for the camera motion?
2. Sky: anchors computed with and without sky patches in the denominator (reviewer's v2.6 item); choose which is the schedule of record.
3. Any pair with an ERROR row is not carried to the next stage until its cause is read.
