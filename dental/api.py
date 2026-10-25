"""One-call interface to the whole dental system, for GUIs and other programs.

    from dental.api import DentalSystem
    system = DentalSystem()                       # loads everything once (needs a GPU; ~1-3 min)
    r = system.ask("xray.png", "Is there deep caries in this panoramic radiograph?")
    r["final_answer"], r["critic_actions"], r["image"]   # answer text, decisions, PIL image with red boxes

Every question goes through: fine-tuned Qwen specialist -> YOLO tools + tooth classifier -> PPO critic.
Paths default to the repository layout (download outputs.zip and runs.zip from the GitHub releases).
"""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .common import estimate_axes, load_config, parse_boxes, resize_for_qwen
from .critic import DEFAULT_DET, QTYPES, PPOCritic, detector_answer, state_features
from .specialist import DentalSpecialist

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAME = {"orig": "specialist", "reground": "YOLO tools", "reroute": "tooth classifier"}

EXAMPLE_QUESTIONS = [
    "Is there caries in this panoramic radiograph?",
    "Is there deep caries in this panoramic radiograph?",
    "Is there an impacted tooth?",
    "Which teeth have caries? Give FDI numbers.",
    "What is the finding on tooth 46?",
    "Is the caries on tooth 36 deep or not deep?",
    "Is tooth 18 present?",
    "How many teeth are in quadrant 3?",
    "Locate all abnormal findings and label each with its diagnosis and tooth number.",
    "Locate all impacted teeth and output bounding boxes.",
    "Draw a box around tooth 43.",
]


def _load_script(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class DentalSystem:
    def __init__(self, config="configs/dental_config.yaml", adapter="outputs/dental_lora_3b/final",
                 critic="results/critic_clf/ppo_critic_train.pt", classifier="outputs/tooth_classifier/best.pt",
                 disease_weights="runs/detect/outputs/dentex_disease_yolo/weights/best.pt",
                 teeth_weights="runs/detect/outputs/dentex_enumeration_yolo/weights/best.pt",
                 disease_conf=None, use_critic=True):
        p = lambda x: str(ROOT / x) if not Path(x).is_absolute() else x  # noqa: E731
        self.cfg = load_config(p(config))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.spec = DentalSpecialist(self.cfg["train"]["model"], adapter=p(adapter),
                                     min_pixels=self.cfg["image"]["min_pixels"],
                                     max_pixels=self.cfg["image"]["max_pixels"])
        from ultralytics import YOLO
        self.dis_model, self.teeth_model = YOLO(p(disease_weights)), YOLO(p(teeth_weights))
        self.live = _load_script("live", "18_live_critic.py")
        self.rr = self.live.rr if hasattr(self.live, "rr") else _load_script("rr", "17_reroute_with_classifier.py")
        self.clf, self.th = self.rr.tc.load_classifier(p(classifier), self.device)
        ck = torch.load(p(critic), map_location="cpu")
        self.K, self.rc, self._ck = ck["K"], ck["rc"], ck
        self.critic = None
        self.use_critic = use_critic
        self.dcfg = dict(DEFAULT_DET)
        if disease_conf is not None:
            self.dcfg["disease_conf"] = disease_conf
        self._cache = {}
        self._tmp = Path(tempfile.mkdtemp(prefix="dental_api_"))

    # ------------------------------------------------------------------ per-image preparation (cached)
    def _prepare(self, image):
        """Resize to the model's input size (all boxes then share one coordinate frame), run YOLO + classifier."""
        key = image if isinstance(image, str) else id(image)
        if key in self._cache:
            return self._cache[key]
        src = image
        if not isinstance(image, (str, Path)):
            src = self._tmp / f"upload_{len(self._cache)}.png"
            image.convert("RGB").save(src)
        dst = self._tmp / f"model_{len(self._cache)}.png"
        (W0, H0), (w, h) = resize_for_qwen(src, dst, self.cfg["image"]["min_pixels"], self.cfg["image"]["max_pixels"])
        det = self.live.run_detectors(str(dst), self.dis_model, self.teeth_model)
        teeth_hi = [dict(quadrant=t // 10, box=b) for b, t, c in det["teeth"] if c >= 0.5]
        mid, occ = estimate_axes(teeth_hi, w, h)
        probs = self.rr.tooth_probs(str(dst), det, self.clf, self.device, 0.5)
        info = dict(path=str(dst), size=(w, h), original_size=(W0, H0), det=det, mid=mid, occ=occ, probs=probs)
        self._cache[key] = info
        return info

    def _get_critic(self, item):
        if self.critic is None:
            n_in = len(state_features(item, "orig", 0, [False] * 4, self.K))
            self.critic = PPOCritic(n_in, self.rc, self.K)
            self.critic.pi.load_state_dict(self._ck["pi"])
            self.critic.v.load_state_dict(self._ck["v"])
            self.critic.mu, self.critic.sd = self._ck["mu"], self._ck["sd"]
        return self.critic

    # ------------------------------------------------------------------ the one call a GUI needs
    def ask(self, image, question, draw=True):
        """image: file path or PIL image. Returns a dict (see README) incl. a PIL image with the final boxes."""
        info = self._prepare(image)
        w, h = info["size"]
        out = self.spec.answer(info["path"], question)
        res = dict(question=question, specialist=out["answer"], specialist_confidence=out["conf_first_token"],
                   yolo=None, classifier=None, classifier_confidence=None, question_type=None,
                   critic_used=False, critic_actions=[], final_source="specialist",
                   final_answer=out["answer"], abstained=False)
        qtype, dx, tooth = self.live.question_type(question)
        res["question_type"] = qtype
        final_text = out["answer"]
        if self.use_critic and qtype in QTYPES:
            row = dict(qtype=qtype, level=self.live.LEVEL[qtype], question=question, diagnosis=dx, tooth=tooth,
                       options=self.live.OPTIONS.get(qtype), width=w, height=h, midline_x=info["mid"],
                       occlusal_y=info["occ"], gold_boxes=[], label=None)
            cands = {"orig": dict(text=out["answer"], conf=[out["conf_first_token"], out["conf_mean"], out["conf_min"]])}
            dtext, dconf = detector_answer(row, info["det"], self.dcfg)
            if dtext is not None:
                cands["reground"] = dict(text=dtext, conf=[dconf] * 3)
                res["yolo"] = dtext
            ctext, cconf = self.rr.answer(row, info["probs"], self.th)
            if ctext is not None:
                cands["reroute"] = dict(text=ctext, conf=[cconf] * 3)
                res["classifier"], res["classifier_confidence"] = ctext, cconf
            item = self.live.build_item(row, cands)
            final, actions = self.live.decide(item, self._get_critic(item), self.K)
            res.update(critic_used=True, critic_actions=actions)
            if final is None:
                res.update(abstained=True, final_source=None,
                           final_answer="Not sure - please have a dentist check this.")
                final_text = ""
            else:
                final_text = cands[final]["text"]
                res.update(final_source=SOURCE_NAME[final], final_answer=final_text)
        boxes = parse_boxes(final_text)
        W0, H0 = info["original_size"]
        sx, sy = W0 / w, H0 / h
        res["final_boxes"] = [dict(box=b, label=l) for b, l in boxes]                     # model-image pixels
        res["final_boxes_original"] = [dict(box=[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], label=l)
                                       for b, l in boxes]                                 # uploaded-image pixels
        if draw:
            res["image"] = self.draw(image if not isinstance(image, (str, Path)) else Image.open(image),
                                     res["final_boxes_original"])
        return res

    def ask_many(self, image, questions):
        return [self.ask(image, q) for q in questions]

    @staticmethod
    def draw(image, boxes):
        """Red boxes + labels on a copy of the image (boxes in that image's pixel coordinates)."""
        im = image.convert("RGB").copy()
        d = ImageDraw.Draw(im)
        lw = max(2, im.width // 400)
        for b in boxes:
            x1, y1, x2, y2 = b["box"]
            d.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=lw)
            d.text((x1, max(0, y1 - 14)), b["label"], fill=(255, 40, 40))
        return im
