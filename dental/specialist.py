"""DentalSpecialist: the interface the Router / Grounding agent / RL critic call.

    from dental.specialist import DentalSpecialist
    spec = DentalSpecialist("Qwen/Qwen2.5-VL-3B-Instruct", adapter="outputs/dental_lora_3b/final",
                            detector="outputs/dentex_disease_yolo/weights/best.pt")
    out = spec.answer("pano.png", "Which tooth has caries? Locate it.")

Returns the answer, labelled boxes (model and original coordinates), FDI teeth mentioned,
token confidences, quadrant-level ACS of its own boxes, and detector confidences.
"""
from __future__ import annotations

import torch
from PIL import Image

from .common import (build_messages, parse_boxes, parse_teeth, quadrant_from_position, quadrant_of_tooth,
                     smart_resize, to_rgb)


class DentalSpecialist:
    def __init__(self, base_model, adapter=None, detector=None, min_pixels=3136, max_pixels=802816,
                 device_map="auto"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.processor = AutoProcessor.from_pretrained(base_model, min_pixels=min_pixels, max_pixels=max_pixels)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model, torch_dtype=dtype,
                                                                        device_map=device_map)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.detector = None
        if detector:
            from ultralytics import YOLO
            self.detector = YOLO(detector)

    def _prep(self, image):
        im = to_rgb(Image.open(image)) if isinstance(image, str) else to_rgb(image)
        W0, H0 = im.size
        h, w = smart_resize(H0, W0, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        return im.resize((w, h), Image.BICUBIC), (W0, H0), (w, h)

    @torch.no_grad()
    def answer(self, image, question, metadata=None, max_new_tokens=256):
        im, (W0, H0), (w, h) = self._prep(image)
        msgs = build_messages({"question": question, "meta": metadata or {}},
                              context="meta" if metadata else "none")
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.processor(text=[text], images=[im], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  output_scores=True, return_dict_in_generate=True)
        gen = out.sequences[:, enc["input_ids"].shape[1]:]
        ans = self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        probs = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0].float().exp()

        boxes = parse_boxes(ans)
        sx, sy = W0 / w, H0 / h
        acs = []   # label-free ACS: box position vs. quadrant of the tooth named in its label
        for b, lab in boxes:
            t = parse_teeth(lab)
            if t:
                acs.append(float(quadrant_from_position(b, w / 2, h / 2) == quadrant_of_tooth(t[0])))
        result = {
            "answer": ans,
            "boxes_model": [{"box": b, "label": l} for b, l in boxes],
            "boxes_original": [{"box": [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], "label": l} for b, l in boxes],
            "teeth_mentioned": parse_teeth(ans) if not boxes else sorted({t for _, l in boxes for t in parse_teeth(l)}),
            "acs_quadrant": sum(acs) / len(acs) if acs else None,
            "model_size": [w, h], "original_size": [W0, H0],
            "conf_mean": float(probs.mean()), "conf_min": float(probs.min()), "conf_first_token": float(probs[0]),
        }
        if self.detector is not None:
            det = self.detector.predict(im, verbose=False, imgsz=1024)[0]
            n = det.names
            result["detector"] = [{"box": b, "label": n[int(c)], "conf": float(s)} for b, c, s in
                                  zip(det.boxes.xyxy.tolist(), det.boxes.cls.tolist(), det.boxes.conf.tolist())]
            result["detector_conf_max"] = max((d["conf"] for d in result["detector"]), default=0.0)
        return result
