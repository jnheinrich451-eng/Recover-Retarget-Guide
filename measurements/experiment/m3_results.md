# Experiment set: results of record (grader CSV sealed 2026-10-20T06:53:43, sha256 8dd54f6f…; tables opened 07:09:27; read by CC 2026-10-20 after the seal)

**Status (2026-10-20, after v2): the grader's alignment instrument was fixed and self-tested (section 7); the primaries are unchanged under the fix, so the HOLD on the object result is lifted and sections 1–3 stand as the reading of record. v2 (`experiment_tables_v2.md`, CSV 875efed9eace) is the table of record; v1 stays beside it.**

Source: `measurements/results/experiment_tables.md` and `experiment_pairs.csv` (the grader's; recomputed here from the CSV with the D8
rule). D8 was committed, and John's blind verdicts written, before the CSV existed (`blind_video_verdicts_2026-10-20.md`).

## 1. The paired result (Main table; 15 pairs, seed 0, same anchor, same noise, same prompt)

| primary | median A0 | median ours | wins / losses (n' = 15) | p one-sided | Holm | verdict |
|---|---|---|---|---|---|---|
| CamRotErr (deg, lower) | 3.590 | 1.084 | 15 / 0 | < 0.0001 | 0.025 | **significant** |
| ObjMC motion-only (px, lower) | 14.112 | 21.579 | 6 / 9 | 0.849 | 0.050 | **not significant; sign against ours** |

Secondary lines (stated before the run): footprint-pass n = 9: CamRot 9/0 (p 0.002), ObjMC 3/6; both-gates n = 4: CamRot 4/0, ObjMC 1/3.
Both primaries have the same sign on the primary and the footprint-pass line: the claim is not narrowed by the subset, it is split by metric.

**Reading of record.** The camera channel transfers: every one of the 15 pairs improves its camera rotation error under guidance, the
median by a factor of 3.3, and the exploratory camera metrics (RPE 15/0, ATE 13/1, CameraCtrl RotErr 15/0, TransErr 14/1), the motion
fidelity (MF 14/1) and the direction cosine (13/2) all agree. The object channel does not transfer on the humans: the subject's own
on-screen path is further from the reference's under guidance on 9 of 12 human pairs, and the object primary fails as registered.

By category (from the CSV):

| category | n | CamRot wins | ObjMC-motion wins |
|---|---|---|---|
| vehicle | 3 | 3 | 3 |
| human | 12 | 12 | 3 (191, 200, 391) |

The split follows the method as built: the vehicle arm renders the object along its path (object channel on); the human arm has no object
render (the object channel is deferred, contract §3), so the human's own path is left to the model while the camera is held, and the
reference-relative path error grows where the model's subject motion differs from the reference's. The pre-registered expectation on the
six pairs ("the camera channel holds; subject articulation is not transferred") is reproduced on the set, by metric.

**Appearance (exploratory, not a registered claim):** Box-IoU 3/12, CLIPScore 1/14, CLIP-IQA 2/13 against ours. Guidance costs image
quality on nearly every pair; John's blind verdicts say the same ("distorted in later frames" on 10 of 15).

## 2. Blind verdicts against the instrument (one observation per pair; disagreements reported, not resolved)

| pair | John (blind, before the CSV) | ObjMC-motion | reading |
|---|---|---|---|
| 431.5.009 | "restores both motions well; fidelity better; a presentation candidate" | 4.7 → 62.4, the largest loss | **disagree**: the eye saw the subject moving with the camera as right; the metric reads its path against the reference's own |
| 221.7.011 | "A0 and A3 movement alike; background distorted in A3" | 3.9 → 17.6, loss | partial: the eye saw no subject change, the metric did |
| 362.2.005 | "both low fidelity; subject vague; A3 still copies motion" | 8.5 → 53.7, loss | **disagree** on the subject; agree on fidelity (Box-IoU 0.26 both) |
| 404.1.006 | "restores the motion faithfully incl. the collision; human motion unnatural" | 14.1 → 21.6, loss | the eye reads the event, the metric the path |
| 391.0.016 | "subject has no motion of its own; camera moves" | 28.1 → 9.5, win | agree: the camera carries the subject's on-screen path |
| 191, 200, 217, 252.1, 396 | motion correct / copied | wins | agree |
| 252.2.007, 256, 265.4, camel, horsejump-high | motion copied; distortion | losses | **disagree**: the eye accepted the subject path, the metric did not |

Seven of 15 disagree in the direction "the eye accepts the subject motion the metric rejects". Two readings, both recorded: (i) ObjMC-motion
reads the subject's 2-D path relative to its own first frame against the reference's, so a subject carried by the camera at a different speed
scores badly while looking "faithful"; (ii) the eye is the blind instrument of record for fidelity (D8), not for path error. Neither replaces
the other; the thesis reports both.

## 3. Cross-method (D9: the baselines' existing caption-prompt runs, labelled; descriptive, no test)

On ours' 13 MiraData pairs: ours CamRot 1.003 against DiTFlow 3.259, SMM 2.150, MOFT 3.484, CogVideoX no-guidance 4.005 (all CogVideoX-5B,
different prompt subject); Follow-Your-Motion (Wan2.1 T2V 1.3B, fine-tuned per clip) 1.100 on the 9 of 13 it generated, and FYM's ObjMC-motion
5.843 against ours 18.922. On the object metric ours is behind DiTFlow (13.0) and FYM; on the camera metric ours is the best training-free row
and level with FYM. Yield: ours 13 of 19 MiraData passes through Module 2 (the automatic path's refusals), the CogVideoX rows 19 of 19, FYM 14
of 19. DAVIS: horsejump-high alone (ours CamRot 5.67 vs A0 21.7; FYM 1.77 at seed 1), one descriptive row.

## 4. Flags and instrument checks

- `kreg_nonpositive`: `not_provided` on 11 pairs in the CSV (the records table the grader read predates the collector's flag columns; the
  four pairs reconstructed after R1 carry it: 362, 431, camel, horsejump-high, all True). To be filled from the refreshed records, no rescoring.
- A-G2b report-only: 14 of 144 scored videos over the 4 px check (ours: 396.3.002); scored, flagged.
- Unscorable in the main set: none. `heading_unchecked` True on 252.1.018, 362, 396 (as selected).

## 5. What the thesis can claim from this record

Camera motion transfer: a registered, significant, paired result on 15 pairs (15/0), with the subset lines in the same sign. Object motion
transfer: not shown; on humans the registered primary fails and the appearance metrics fall; on vehicles 3 of 3 but n = 3 is descriptive.
The cross-method rows are descriptive and prompt-mismatched by D9. The yield (13 of 19) is a property of the automatic Module 2 and is
reported with its mechanisms (`m2_outcomes.md`).

## 6. Instrument check ordered after the results were seen (reviewer 2026-10-20; `docs/grader_brief_experiment_v2.md`)

The sim(3) alignment of the generated camera to the reference returns a negative scale on four A0 rows (256 −0.56, 362 −3.09, 431 −6.13,
camel −0.31) and an extreme one on four others (217 51.2, 221 26.4, 265.4 23.4, 252.2 20.3); ours lies in 0.31–1.67 except 362 (8.2). A
negative scale is a reflection: the solver mirrored a camera moving the wrong way to make it fit, which flatters A0 on every metric that
uses the alignment (ATE, TransErr, ObjErr3D, and possibly ObjMC-motion through its camera removal). 431.5.009, the largest object loss and
the strongest blind-verdict disagreement of section 2, is exactly the pattern a reflected A0 alignment would produce. Ordered: (1) name the
metrics that use the alignment and confirm CamRotErr / RPE / RotErr are rotation-only and alignment-free (the camera claim's safety); (2)
constrain scale > 0, det R = +1, report "no valid alignment" otherwise; (3) self-test on the references against themselves (0 error, scale 1);
(4) re-score every row of every method, v1 and v2 reported side by side; the thesis records that the fix came after the results were seen
and whether the object primary changed direction. Two pairs to look at by eye after v2: 217.0.006 (direction cosine −0.85 for ours against
+0.88 for A0) and 431.5.009.

Reviewer's verdict table of record: camera channel met (15/0, p ≈ 3e-5, Holm-significant; every camera-side metric agrees; the footprint-pass
line 9/0 same sign, and the camera wins hold on all 6 footprint-fail pairs because the background render does not use G_tar); object motion
consistent with the prediction (no win claimed; the object channel is deferred, D6); the class split (vehicles 3/3, humans 3/12) matches the
design but was not pre-registered and n = 3, descriptive only; image quality is a cost to report (consistent with the pilot's softness and
seams). Proposed thesis sentence, pending v2: "Against its own backbone, guidance reduces camera rotation error on all 15 pairs (sign test,
one-sided p = 3e-5, Holm-corrected). Object-motion error is not reduced (6 of 15). Image-quality scores decrease."

Cross-method: FYM's medians cover 9 of the 13 pairs (4 `empty_head_group` crashes); on those 9 ours is 5/9 on CamRot (1.003 vs 1.100), 0/9 on
ObjMC-motion (18.9 vs 5.8), 1/9 on Box-IoU: on camera rotation a training-free method roughly matches one fine-tuned per video; FYM is clearly
better on object motion; different backbone, prompt subject and per-video training: reported, not claimed. CLIP-IQA's "ceiling" label is
wrong (A0 0.788 and most baselines score above the reference 0.45): "reference value".

## 7. The instrument check, done (grader v2, CSV sealed 2026-10-20T08:24:26, 875efed9eace; v1 kept)

- **Self-test (rule 13):** every reference cut graded against itself reads 0 on every error metric, 1 on IoU, scale exactly 1, under v1 and v2
  (42 rows, `experiment_selftest.csv`, all pass). The scorer reproduces a known answer.
- **Which metrics used the alignment:** only ObjErr3D used the scale-only fit. CamRotErr, RPE, CC_RotErr are rotation-only; ObjMC and
  ObjMC-motion, Box-IoU, MF, direction, IQ are alignment-free. Under the proper-similarity constraint the two primaries are IDENTICAL on
  every pair (v1 = v2: CamRot 15/0, ObjMC-motion 6/9); ObjErr3D goes from 10/5 (n' 15) to 8/3 (n' 11), four A0 rows having no valid alignment.
- **Reflections found:** A0 unalignable on 256.3.010, 362.2.005, 431.5.009 and camel (the four negative v1 scales); 25 of 144 scored rows across
  all methods have no valid alignment under v2 (A0 3, CogVideoX no-guidance 4, SMM 5, MOFT 3, Wan-T2V no-guidance 5, DiTFlow 1, ours 0, FYM 0).
- **Consequence for section 2:** the reviewer's conjecture that 431's object loss was a reflected A0 fit is not supported: ObjMC-motion does
  not use the alignment and is unchanged; the loss (4.7 → 62.4) stands as scored, and the blind-verdict disagreement on 431 stands as a
  finding about the metric versus the eye. 431 and 217.0.006 (direction cosine −0.85 for ours vs +0.88 for A0) remain John's by-eye items.
- **FYM on its 9 pairs:** CamRot ours 5/9 (1.003 vs 1.100), ObjMC-motion 0/9 (18.9 vs 5.8), Box-IoU 1/9; FYM's 4 failures (`empty_head_group`,
  layer 13) in the yield line. CLIP-IQA relabelled "reference value".

**Sentence of record (reviewer's wording, confirmed by v2):** "Against its own backbone, guidance reduces camera rotation error on all 15
pairs (sign test, one-sided p = 3e-5, Holm-corrected). Object-motion error is not reduced (6 of 15). Image-quality scores decrease."

## 8. The guidance-window ablation (plan D5; grader CSV `ablation_tg.csv` sealed 2026-10-20T22:31:28, 743956909591; five gated pairs, 33 frames, seed 0)

Settings as run (recorded in the plan with the rule-19 disclosure): A = [0.85, 0.75] ends earlier than the lock's [0.85, 0.60]; B = [0.85, 0.45]
ends later. Per-pair tables: `measurements/results/ablation_tg_tables.md`. Read as within-pair differences against the lock-window A3 and A0 of
the same pair; five pairs, descriptive.

Registered predictions, scored as registered by the grader: (1) A moves CamRotErr toward A0: holds on 3 of 4 scorable pairs; A moves image
quality toward A0: CLIPScore 3/5, CLIP-IQA 1/5, Box-IoU 3/4 (mixed: not confirmed for CLIP-IQA). (2) B keeps CamRotErr at or below the lock's:
3 of 5; B lowers image quality: CLIPScore 2/5, CLIP-IQA 2/5, Box-IoU 4/5 (not confirmed). (3) ObjMC-motion has no consistent direction under
either setting: holds (A up on 3 / down on 2; B up on 2 / down on 3). Reading: the camera result is robust to the window on every pair where the
lock row is valid (every A3 window beats A0 on CamRotErr by a factor of 2 to 19); the object metric moves without a direction; image quality
does not move with the window in a consistent direction at n = 5, which agrees with section 7 of `m3_analysis.md` (the appearance cost is not
proportional to how much the correction does).

**Under query (`docs/grader_query_ablation_2026-10-21.md`):** the 370.1.008 lock-window A3 row carries camera and object scores identical to its
A0 row while the videos and their appearance scores differ; the grader reused an extraction by name from the six-pair store. Until the v2
ablation CSV re-extracts the A0 and lock rows by content, 370's lock row and the prediction entries that depend on it are not available.
