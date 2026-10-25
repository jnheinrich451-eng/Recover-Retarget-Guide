"""segment(generated): the instrument validated by eye on 18 generated images in the SAM 3D probe
(scripts/probe_sam3d.py: gen_p4, 2026-09-14) — OWLv2 text box (category query) -> SAM2 box + centre-point
prompt -> the largest candidate that is not background. Lifted UNCHANGED; its known failure is a bare
centre-point prompt (part masks). GPU, base env (plain python). Real models only — there is no CPU stand-in.
"""
from __future__ import annotations

import numpy as np

SAM2 = "facebook/sam2.1-hiera-large"
OWL = "google/owlv2-base-patch16-ensemble"
QUERIES = {"vehicle": ["a car"], "human": ["a person"]}
DET_THRESHOLD = 0.05                                             # the probe's value; a detection floor, not a gate


def load_segmenter():
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    return {"owl_p": Owlv2Processor.from_pretrained(OWL), "owl": Owlv2ForObjectDetection.from_pretrained(OWL).to("cuda").eval(),
            "sam": SAM2ImagePredictor.from_pretrained(SAM2)}


def segment(seg: dict, img_rgb, category: str) -> tuple[np.ndarray, dict]:
    import torch
    from PIL import Image
    img = np.ascontiguousarray(img_rgb); Hh, Ww = img.shape[:2]
    owl_p, owl, pred = seg["owl_p"], seg["owl"], seg["sam"]
    if category not in QUERIES:
        from src.anchor import subjects
        QUERIES[category] = [subjects.query(category)]
    inputs = owl_p(text=[QUERIES[category]], images=Image.fromarray(img), return_tensors="pt").to("cuda")
    with torch.no_grad():
        post = getattr(owl_p, "post_process_grounded_object_detection", None) or owl_p.post_process_object_detection   # renamed (measured 2026-09-14)
        res = post(owl(**inputs), threshold=DET_THRESHOLD, target_sizes=torch.tensor([[max(Hh, Ww), max(Hh, Ww)]]))[0]
    if not len(res["scores"]):
        return np.zeros((Hh, Ww), bool), {"box": None, "det_score": 0.0, "sam_score": None}      # object not found: a measurement
    bi = int(res["scores"].argmax())
    box = np.clip(res["boxes"][bi].detach().cpu().numpy().astype(float), 0, [Ww, Hh, Ww, Hh]); det = float(res["scores"][bi])
    c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred.set_image(img)
        m, sc, _ = pred.predict(point_coords=np.array([c], np.float32), point_labels=np.ones(1, np.int32),
                                box=box[None].astype(np.float32), multimask_output=True)
    cands = []
    for mm, ss in zip(m, sc):                                    # SAM2's candidates are sub-part / part / whole
        mb = mm.astype(bool)
        ring = np.concatenate([mb[0], mb[-1], mb[:, 0], mb[:, -1]]).mean()
        if mb.mean() < 0.8 and ring < 0.5:
            cands.append((mb.sum(), float(ss), mb))
    if not cands:
        cands = [(m[i].sum(), float(sc[i]), m[i].astype(bool)) for i in range(len(m))]
    _, score, mask = max(cands, key=lambda c_: c_[0])
    return mask, {"box": box.tolist(), "det_score": det, "sam_score": score}
