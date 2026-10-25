"""LIVE pipeline: X-ray + question -> specialist answer -> RL critic decides on the spot.

For each question:
  1. the fine-tuned Qwen specialist answers (with token confidences)        -> "orig"
  2. the YOLO detectors give their version of the answer                    -> "reground" option
  3. the tooth-crop classifier gives its version (diagnosis questions only) -> "reroute" option
  4. the trained PPO critic looks at all of this and chooses, step by step:
     Accept / Reground / Reroute / Abstain   (at most 3 steps, exactly as in training)
Nothing is retrained: it reuses your saved model, detectors, classifier and critic.

GPU job.
  sbatch cluster/live_critic.sbatch --from_test 3 --with_diagnosis "deep caries"
  sbatch cluster/live_critic.sbatch --image my_xrays/test1.png --question "Is there deep caries?"
Answers + every critic decision are printed in the log; pictures go to results/live/
(red = final accepted boxes, green = expert boxes for test X-rays).
"""
import argparse
import importlib.util
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dental.common import (estimate_axes, load_config, parse_boxes, parse_teeth, read_jsonl,  # noqa: E402
                           resize_for_qwen)
from dental.critic import (ABSTAIN, ACCEPT, ACTIONS, DEFAULT_DET, QTYPES, REGROUND, PPOCritic, action_mask,  # noqa: E402
                           agree, answer_acs, answer_key, detector_answer, quadrant_of_question,
                           state_features)
from dental.specialist import DentalSpecialist  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LEVEL = {"presence": 1, "tooth_present": 1, "which_teeth": 2, "which_quadrant": 2, "tooth_diagnosis": 2,
         "count_quadrant": 2, "caries_depth": 3, "ground_diagnosis": 4, "ground_all": 4, "ground_tooth": 4}
OPTIONS = {"presence": ["yes", "no"], "tooth_present": ["yes", "no"], "caries_depth": ["deep", "not deep"],
           "tooth_diagnosis": ["caries", "deep caries", "periapical lesion", "impacted"]}
DEFAULT_QUESTIONS = [
    "Is there deep caries in this panoramic radiograph?",
    "Which teeth have caries? Give FDI numbers.",
    "Locate all abnormal findings and label each with its diagnosis and tooth number.",
    "Locate all impacted teeth and output bounding boxes.",
]


# ------------------------------------------------------------------ understanding a free question
def question_type(q):
    """Map a question in your own words to one of the critic's question types (None = critic not used)."""
    s = q.lower()
    dx = ("deep caries" if re.search(r"deep car", s) else "periapical lesion" if "periapical" in s
          else "impacted" if "impact" in s else "caries" if re.search(r"carie|carious|decay", s) else None)
    teeth = parse_teeth(q)
    tooth = teeth[0] if len(teeth) == 1 else None
    if re.search(r"\bbox|locate|bounding|where (is|are)", s):
        if tooth and not dx:
            return "ground_tooth", None, tooth
        return ("ground_diagnosis", dx, None) if dx else ("ground_all", None, None)
    if re.search(r"how many|count|number of teeth", s):
        return "count_quadrant", None, None
    if "quadrant" in s and dx:
        return "which_quadrant", dx, None
    if re.search(r"which teeth|which tooth|list", s) and dx:
        return "which_teeth", dx, None
    if tooth and re.search(r"\bdeep\b.*\bor\b|how deep|\bdeep\?", s):
        return "caries_depth", None, tooth
    if tooth and re.search(r"finding|abnormal|pathology|wrong with|diagnosis", s):
        return "tooth_diagnosis", None, tooth
    if tooth and "present" in s:
        return "tooth_present", None, tooth
    if dx and re.search(r"is there|are there|any |does|do you see|can you see", s):
        return "presence", dx, None
    return None, dx, tooth


def run_detectors(image_path, dis_model, teeth_model):
    d = dis_model.predict(image_path, imgsz=1024, conf=0.05, verbose=False)[0]
    e = teeth_model.predict(image_path, imgsz=1024, conf=0.05, verbose=False)[0]
    return {"disease": [[b, d.names[int(c)], float(s)] for b, c, s in
                        zip(d.boxes.xyxy.tolist(), d.boxes.cls.tolist(), d.boxes.conf.tolist())],
            "teeth": [[b, int(e.names[int(c)]), float(s)] for b, c, s in
                      zip(e.boxes.xyxy.tolist(), e.boxes.cls.tolist(), e.boxes.conf.tolist())]}


def build_item(row, cands):
    """Same item structure the critic was trained on, but without any correctness labels."""
    for c in cands.values():
        c["key"] = answer_key(row, c["text"])
        c["acs"] = answer_acs(row, c["text"])
        bxs = [b for b, _ in parse_boxes(c["text"])]
        img = float(row["width"] * row["height"])
        c["n_boxes"] = len(bxs)
        c["max_area"] = max(((b[2] - b[0]) * (b[3] - b[1]) / img for b in bxs), default=0.0)
    names = list(cands)
    ag = {(a, b): agree(row, cands[a]["key"], cands[b]["key"]) for a in names for b in names if a != b}
    return dict(row=row, cands=cands, agree=ag)


def decide(item, critic, K):
    """Run the critic live: returns (final candidate name or None if abstained, list of actions)."""
    cur, step, used, actions = "orig", 0, [False] * 4, []
    while True:
        s = state_features(item, cur, step, used, K)
        m = action_mask(item, step, used, K)
        a = critic.act(s, m, greedy=True)
        actions.append(ACTIONS[a])
        if a == ACCEPT:
            return cur, actions
        if a == ABSTAIN:
            return None, actions
        used[a] = True
        cur = "reground" if a == REGROUND else "reroute"
        step += 1


def draw(path, final_boxes, gold, out):
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    for g in gold:
        d.rectangle(g["box"], outline=(0, 200, 0), width=3)
        d.text((g["box"][0], g["box"][3] + 2), f"{g['diagnosis']} {g['tooth']}", fill=(0, 200, 0))
    for b, lab in final_boxes:
        d.rectangle(b, outline=(255, 40, 40), width=3)
        d.text((b[0], max(0, b[1] - 12)), lab, fill=(255, 40, 40))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dental_config.yaml")
    ap.add_argument("--adapter", default="outputs/dental_lora_3b/final")
    ap.add_argument("--critic", default="results/critic_clf/ppo_critic_train.pt")
    ap.add_argument("--classifier", default="outputs/tooth_classifier/best.pt")
    ap.add_argument("--disease_weights", default="runs/detect/outputs/dentex_disease_yolo/weights/best.pt")
    ap.add_argument("--teeth_weights", default="runs/detect/outputs/dentex_enumeration_yolo/weights/best.pt")
    ap.add_argument("--image")
    ap.add_argument("--question", action="append")
    ap.add_argument("--from_test", type=int, default=0)
    ap.add_argument("--with_diagnosis")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disease_conf", type=float, default=None,
                    help="YOLO confidence threshold for Reground answers (default: the one the critic was trained with)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = []
    if args.from_test:
        recs = [r for r in read_jsonl(Path(cfg["paths"]["processed"]) / "splits" / "test.jsonl")
                if r["source"] == "dentex_disease"]
        if args.with_diagnosis:
            recs = [r for r in recs if any(f["diagnosis"] == args.with_diagnosis for f in r["findings"])]
        random.Random(args.seed).shuffle(recs)
        items = [(r["image_path"], r["findings"], r["id"]) for r in recs[:args.from_test]]
    if args.image:
        items.append((args.image, [], Path(args.image).stem))
    if not items:
        sys.exit("Give --image or --from_test N")
    questions = args.question or DEFAULT_QUESTIONS

    print("loading specialist, detectors, tooth classifier and critic ...")
    spec = DentalSpecialist(cfg["train"]["model"], adapter=args.adapter,
                            min_pixels=cfg["image"]["min_pixels"], max_pixels=cfg["image"]["max_pixels"])
    from ultralytics import YOLO
    dis_model, teeth_model = YOLO(args.disease_weights), YOLO(args.teeth_weights)
    rr = _load("rr", "17_reroute_with_classifier.py")
    clf, th = rr.tc.load_classifier(args.classifier, device)
    ck = torch.load(args.critic, map_location="cpu")
    K, rc = ck["K"], ck["rc"]
    dcfg = dict(DEFAULT_DET)
    if args.disease_conf is not None:
        dcfg["disease_conf"] = args.disease_conf
    print("YOLO confidence threshold for Reground:", dcfg["disease_conf"])

    critic = None
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="live_"))
    for path, gold, name in items:
        # resize once to the model's input size so the model, YOLO and classifier share one coordinate frame
        src, path = path, str(tmp / f"{name}.png")
        resize_for_qwen(src, path, cfg["image"]["min_pixels"], cfg["image"]["max_pixels"])
        im = Image.open(path)
        W, H = im.size
        det = run_detectors(path, dis_model, teeth_model)
        teeth_hi = [dict(quadrant=t // 10, box=b) for b, t, c in det["teeth"] if c >= 0.5]
        mid, occ = estimate_axes(teeth_hi, W, H)
        probs = rr.tooth_probs(path, det, clf, device, 0.5)
        print("=" * 90)
        print("X-ray:", src)
        if gold:
            print("Expert findings:", ", ".join(f"{g['diagnosis']} {g['tooth']}" for g in gold))
        final_boxes = []
        for q in questions:
            out = spec.answer(path, q)
            qtype, dx, tooth = question_type(q)
            print(f"\nQ: {q}")
            print(f"   specialist: {out['answer'][:300]}   (confidence {out['conf_first_token']:.2f})")
            if qtype not in QTYPES:
                print("   critic: not used for this kind of question; specialist answer shown as is")
                final_boxes += parse_boxes(out["answer"])
                continue
            row = dict(qtype=qtype, level=LEVEL[qtype], question=q, diagnosis=dx, tooth=tooth,
                       options=OPTIONS.get(qtype), width=W, height=H, midline_x=mid, occlusal_y=occ,
                       gold_boxes=[], label=None)
            cands = {"orig": dict(text=out["answer"], conf=[out["conf_first_token"], out["conf_mean"],
                                                            out["conf_min"]])}
            dtext, dconf = detector_answer(row, det, dcfg)
            if dtext is not None:
                cands["reground"] = dict(text=dtext, conf=[dconf] * 3)
                print(f"   YOLO tools: {dtext[:300]}")
            ctext, cconf = rr.answer(row, probs, th)
            if ctext is not None:
                cands["reroute"] = dict(text=ctext, conf=[cconf] * 3)
                print(f"   tooth classifier: {ctext}   (confidence {cconf:.2f})")
            item = build_item(row, cands)
            if critic is None:
                n_in = len(state_features(item, "orig", 0, [False] * 4, K))
                critic = PPOCritic(n_in, rc, K)
                critic.pi.load_state_dict(ck["pi"])
                critic.v.load_state_dict(ck["v"])
                critic.mu, critic.sd = ck["mu"], ck["sd"]
            final, actions = decide(item, critic, K)
            print(f"   critic decisions: {' -> '.join(actions)}")
            if final is None:
                print("   FINAL: not sure - refer to a dentist (abstained)")
            else:
                src = {"orig": "specialist", "reground": "YOLO tools", "reroute": "tooth classifier"}[final]
                print(f"   FINAL ({src}): {cands[final]['text'][:300]}")
                final_boxes += parse_boxes(cands[final]["text"])
        pic = Path("results/live") / f"{name}.png"
        draw(path, final_boxes, gold, pic)
        print(f"\nPicture: {pic}  (red = final answer, green = expert)")


if __name__ == "__main__":
    main()
