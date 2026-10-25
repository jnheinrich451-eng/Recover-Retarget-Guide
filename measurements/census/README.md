# measurements/census

Records of the census (Module 1 gates, m2a admission, camera gate of record, support surface) on the 70 candidate clips:
`census.json/md` (one row per clip, append-only), `camera_c.{json,md}` and `camera_c_ba.{json,md}` (the camera gate per pass).

## contact_labels_template.csv (G2 self-test set; rule 19, labelled blind)

One row per census pass. John fills `label` from the SOURCE video only, before `g2selftest` runs and without seeing any f_sup:
`in_contact` = the subject is in ground contact throughout the 24-frame window (walking, driving, standing); `not_in_contact` = it is
off the ground for part or all of the window (swimming, flying, jumping clips, gold-fish); `contact_out_of_view` = the subject is on
the ground but its contact is not in frame (half-body framing, feet or wheels cut off by the frame edge for part or all of the window).
Leave blank only when undecidable from the video.
Save the filled sheet as `contact_labels.csv` next to it (the runner reads that name) and commit it before the run. The contact set
gives τ_sup (min − 2σ̂, floor 0.8); the `not_in_contact` rows are the negative controls and must read below τ_sup. `contact_out_of_view` rows
enter neither set: their f_sup is reported as a recorded outcome (the instrument cannot see a contact that is not in frame).
