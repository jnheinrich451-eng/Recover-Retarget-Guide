# Experiment set, Module 2 automatic path: outcome per pair (record of 2026-10-19, after D7; run_all.json is the append-only log)

32 census passes; no clip added or removed; no threshold adjusted. A refusal is a recorded outcome with its mechanism. The pass before D7 is in the log (16 refusals before R2 unchanged; 9 principal-point refusals re-run re-centred).

| outcome | n | pairs |
|---|---|---|
| no source heading (Stage B.5) | 2 | 000000000064.3.002/human, dogs-scale/human |
| placement: contact out of view | 3 | 000000000119.5.007/human, 000000000222.4.003/human, 000000000370.1.009/human |
| selection: no candidate passed the gates | 3 | 000000000130.0.015/human, breakdance/human, goat/human |
| complete: handoff written | 12 | 000000000191.6.037/human, 000000000200.1.006/human, 000000000217.0.006/vehicle, 000000000221.7.011/human, 000000000252.1.018/vehicle, 000000000252.2.007/human, 000000000256.3.010/human, 000000000265.4.002/human, 000000000391.0.016/human, 000000000396.3.002/vehicle, 000000000404.1.006/human, bmx-bumps/human |
| placement: solve not admitted (negative scale) | 5 | 000000000299.1.005/vehicle, dancing/human, disc-jockey/human, elephant/human, gold-fish/human |
| Body R2: gauge alignment, non-positive scale | 4 | 000000000362.2.005/human, 000000000431.5.009/human, camel/human, horsejump-high/human |
| control: proxy fully occluded by the background | 3 | breakdance-flare/human, hockey/human, kite-walk/human |

Handoffs to Module 3: 12 of 32 (vehicle 3, human 9).

After D7 (re-centring by padding, every clip): the nine principal-point refusals became 8 handoffs and 1 gauge refusal; the gauge-alignment refusals are now 4 (362.2.005, 431.5.009, camel, horsejump-high): the Body mesh depth against the band-alpha anchor depth over the subject has no positive median ratio; diagnosed by scripts/check_m2e_gauge_refusal.py.

Self-transfer fixture on the three occluded-proxy refusals (measurements/m2c_self_transfer/self_transfer.md): the SOURCE's own subject placed back into its own sanitised frame is occluded 65-92 % by the C2 fill, whose depth under the subject is nearer than the subject itself (breakdance-flare 3.85 vs 4.31; hockey 6.13 vs 6.24; kite-walk 4.49 vs 5.59). The gate refused the proxy for a fault of the fill, not of the clip: a C2 finding (static-camera clips with the ground close below the subject), reported to the reviewer; not adjusted.

## Addenda (reviewer 2026-10-19)

- **Selection, no candidate passed the gates (130.0.015, breakdance, goat):** the refusing gate on every one of the 12 candidates is the
  proxy IoU = 0.0 (the segmented subject does not meet the proxy footprint at all), which makes H1 not computable and leaves G1 with no
  ground plane. Not the heading rule and not the MUSIQ floor (MUSIQ 39-74 against tau_q 38.3). Whether the generator ignored the control or
  the segmenter missed the subject is a full-resolution look at the twelve candidates (John), recorded next.
- **Placement refusals:** the negative-scale solves are 299.1.005, disc-jockey, gold-fish (John's three not-in-contact labels) and dancing;
  the contact-out-of-view refusals are 119.5.007, 222.4.003, 370.1.009, exactly John's labels. Both are the instrument working as intended.
- **D7 decided by the reviewer: re-centre by padding, every clip, before any Module 3 sampling.** Implemented (commit ac34599; CPU known
  answers pass; the GPU fixture is scripts/check_m2b2_recentre_fixture.py on the two human pairs of record); the human pairs are re-run
  from R2 with their previous outputs archived (scripts/run_all.py --redo-from r2).
- **Proxy fully occluded (breakdance-flare, hockey, kite-walk):** self-transfer fixture scripts/check_m2c_self_transfer.py, to run before
  these are read as refusals.

## R2 check and the session of 2026-10-19 09:21 (record)

- `measurements/m2c_fill_occlusion/fill_occlusion.md`: under the rule of record, 9 of the 12 handoffs had proxy pixels occluded by fill
  pixels (max 0.263), so by the reviewer's rule **R2 is NOT adopted**: the three occluded-proxy pairs (breakdance-flare, hockey, kite-walk)
  are reported as refusals with the mechanism named, and R2 is a registered follow-up. For the no-candidate pairs the proxy was fully
  visible in the control (goat 1.0, horsejump-high 1.0 as a handoff reference): the generator ignored the control, not a hidden proxy.
- The 09:21 session nevertheless ran the chain with `--occlusion-excludes-fill`: the three controls were re-composed under R2 (their
  refusals of record stay in the log at 02:xx and above); breakdance-flare and kite-walk then reached selection and got no anchor;
  hockey stopped at the Body environment. These R2 rows are follow-up material only; the main table keeps the three as refusals under
  the rule of record. The chain now refuses the flag unless the check's verdict says adopt (commit of this note).
- The same session was a fresh VM without detectron2, and `--redo-from r2` had been applied to every human pair: all human m2_e outputs
  from R2 onward are archived under each pair (`_archive/<time>/`) and must be redone once the Body environment is installed; the
  three vehicle handoffs are untouched. Nothing is lost; the handoffs regenerate (Body is deterministic; D7 unchanged).

## Module 3 attempt of 2026-10-19 10:58-11:11: void (recorded)

The chain reported "complete 16" because the batch driver returns 0 on recorded failures. The batch manifest (`measurements/m3_batch/manifest.json`)
shows that on every pair the lift wrote nothing (the handoff gate refuses the 'experiment' disposition and checks the v1 closure manifest, which
lists the six pairs only), the geometry step would in any case have needed the ViPE camera and the A.5 sigma record that the census clips never
had, and the renders and the A0 arm then failed on the missing lift. No video was produced. Fixes (commit of this note): the experiment path for
Module 3 (`--experiment` on the lift; the census camera of record `camera_c_ba` on the geometry: camera C + bundle adjustment, source gauge by
the lift prior, beta = 1, s_delta = 1, s_c as on every pair), named in `configs/m3_batch_exp.yaml`; a lift or geometry that wrote nothing is a
recorded refusal with the runner's reason; the chain's Module 3 outcome is read from the pair's cache (lift, geometry, both arms' videos), never
from a return code. The export of that session (16 pairs, A0 videos absent) is void.
