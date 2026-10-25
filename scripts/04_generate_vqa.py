"""Generate hierarchical dental VQA pairs.

Each question is only asked where the labels can answer it:
  DENTEX disease subset     -> diagnoses on abnormal teeth (healthy teeth are NOT labelled,
                               so no "is tooth X present?" questions here)
  DENTEX enumeration subset -> every tooth labelled (presence, counts, tooth grounding),
                               but no diagnoses (so no disease questions here)
  Tufts                     -> abnormality presence/boxes from expert masks, report from text
Writes <processed>/vqa/{train,val,test}.jsonl
"""
import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import (ALL_TEETH, DIAGNOSES, INVALID_TEETH, QUADRANT_NAME,  # noqa: E402
                           boxes_to_answer, load_config, read_jsonl, write_jsonl)

# "caries" questions mean ANY carious lesion (caries or deep caries); depth is asked separately.
GROUP = {"caries": {"caries", "deep caries"}, "deep caries": {"deep caries"},
         "periapical lesion": {"periapical lesion"}, "impacted": {"impacted"}}
DX_PHRASE = {"caries": "caries", "deep caries": "deep caries",
             "periapical lesion": "a periapical lesion", "impacted": "an impacted tooth"}
DX_PLURAL = {"caries": "carious lesions", "deep caries": "deep carious lesions",
             "periapical lesion": "periapical lesions", "impacted": "impacted teeth"}

T = {
    "presence": ["Is there {p} in this panoramic radiograph?", "Does this X-ray show {p}?",
                 "Can you see {p} on this image?"],
    "which_teeth": ["Which teeth have {dx}? Give FDI numbers.", "List the FDI numbers of teeth with {dx}.",
                    "On which teeth is {dx} seen?"],
    "which_quadrant": ["In which quadrant(s) is {dx} seen?", "Which quadrant(s) contain {dx}?",
                       "Where is the {dx}: which quadrant?"],
    "tooth_diagnosis": ["What is the finding on tooth {t}?", "What abnormality affects tooth {t}?",
                        "Describe the pathology of tooth {t}."],
    "caries_depth": ["Is the caries on tooth {t} deep or not deep?",
                     "How deep is the carious lesion on tooth {t}: deep or not deep?",
                     "Does the caries on tooth {t} reach deep into the tooth (deep / not deep)?"],
    "ground_diagnosis": ["Locate all {pl} and output bounding boxes.", "Draw boxes around all {pl}.",
                         "Where are the {pl}? Give boxes in JSON with tooth numbers."],
    "ground_all": ["Locate all abnormal findings and label each with its diagnosis and tooth number.",
                   "Detect every abnormal tooth and output boxes with diagnosis and FDI number.",
                   "Find all caries, deep caries, periapical lesions and impacted teeth; output boxes."],
    "tooth_present": ["Is tooth {t} present?", "Is tooth {t} visible in this radiograph?",
                      "Can you see tooth {t}?"],
    "count_quadrant": ["How many teeth are visible in quadrant {q}?", "Count the teeth in quadrant {q}.",
                       "How many teeth are in the {qn} quadrant?"],
    "ground_tooth": ["Locate tooth {t}.", "Draw a box around tooth {t}.", "Where is tooth {t}? Give its box."],
    "abnormality": ["Is there any abnormality in this panoramic radiograph?",
                    "Does this X-ray show any dental abnormality?", "Is anything abnormal on this image?"],
    "ground_abnormality": ["Locate all abnormalities and output bounding boxes.",
                           "Draw boxes around the abnormal regions.", "Where are the abnormalities?"],
    "report": ["Write a short radiology report for this panoramic X-ray.",
               "Summarise the findings of this dental radiograph.", "Give a brief report of the findings."],
    "absent": ["Is there caries on tooth {t}?", "Does tooth {t} have a periapical lesion?",
               "What is the finding on tooth {t}?"],
}
LEVEL = {"presence": 1, "abnormality": 1, "tooth_present": 1,
         "which_teeth": 2, "which_quadrant": 2, "tooth_diagnosis": 2, "count_quadrant": 2,
         "caries_depth": 3,
         "ground_diagnosis": 4, "ground_all": 4, "ground_tooth": 4, "ground_abnormality": 4,
         "report": 5, "absent": 0}
OPTIONS = {"presence": ["yes", "no"], "abnormality": ["yes", "no"], "tooth_present": ["yes", "no"],
           "caries_depth": ["deep", "not deep"], "tooth_diagnosis": DIAGNOSES,
           "absent": ["not present", "yes", "no"] + DIAGNOSES}


def cap(s):
    return s[:1].upper() + s[1:]


def quad_text(qs):
    return ", ".join(f"quadrant {q} ({QUADRANT_NAME[q]})" for q in qs)


def dentex_report(r):
    by = defaultdict(set)
    for f in r["findings"]:
        by[f["diagnosis"]].add(f["tooth"])
    parts = [f"{cap(dx)}: {', '.join(map(str, sorted(by[dx])))}." for dx in DIAGNOSES if by[dx]]
    return "Panoramic radiograph. " + (" ".join(parts) if parts else "No caries, periapical lesion or impaction identified.")


def enum_report(r):
    present = {t["tooth"] for t in r["teeth"]}
    missing = [t for t in ALL_TEETH if t not in present]
    s = f"Panoramic radiograph showing {len(present)} teeth."
    return s + (f" Not present: {', '.join(map(str, missing))}." if missing else " All 32 permanent teeth present.")


def items_disease(r, rng):
    F = r["findings"]
    for dx in DIAGNOSES:   # L1 presence for every diagnosis -> balanced yes/no
        hit = [f for f in F if f["diagnosis"] in GROUP[dx]]
        yield "presence", {"p": DX_PHRASE[dx]}, "Yes." if hit else "No.", "yes" if hit else "no", {"diagnosis": dx}
        if hit:
            teeth = sorted({f["tooth"] for f in hit})
            quads = sorted({f["quadrant"] for f in hit})
            yield "which_teeth", {"dx": dx}, ", ".join(map(str, teeth)) + ".", teeth, {"diagnosis": dx}
            yield "which_quadrant", {"dx": dx}, cap(quad_text(quads)) + ".", quads, {"diagnosis": dx}
            gold = [(f["box"], f"{f['diagnosis']} {f['tooth']}") for f in hit]
            yield "ground_diagnosis", {"pl": DX_PLURAL[dx]}, boxes_to_answer(gold), "boxes", {"diagnosis": dx, "gold": gold}
        elif rng.random() < 0.5:
            yield "ground_diagnosis", {"pl": DX_PLURAL[dx]}, "None found.", "none", {"diagnosis": dx, "gold": []}
    by_tooth = defaultdict(set)
    for f in F:
        by_tooth[f["tooth"]].add(f["diagnosis"])
    for t, dxs in sorted(by_tooth.items()):
        yield "tooth_diagnosis", {"t": t}, cap(" and ".join(sorted(dxs))) + ".", sorted(dxs), {"tooth": t}
        if dxs & {"caries", "deep caries"}:
            lab = "deep" if "deep caries" in dxs else "not deep"
            yield "caries_depth", {"t": t}, cap(lab) + ".", lab, {"tooth": t}
    gold = [(f["box"], f"{f['diagnosis']} {f['tooth']}") for f in F]
    yield "ground_all", {}, boxes_to_answer(gold) if gold else "None found.", "boxes" if gold else "none", {"gold": gold}
    yield "report", {}, dentex_report(r), None, {}


def items_enum(r, cfg, rng):
    v = cfg["vqa"]
    teeth = {t["tooth"]: t for t in r["teeth"]}
    present, missing = sorted(teeth), [t for t in ALL_TEETH if t not in teeth]
    k = v["tooth_present_per_image"]
    pick = rng.sample(present, min(k // 2, len(present))) + rng.sample(missing, min(k - k // 2, len(missing)))
    for t in pick:
        yield "tooth_present", {"t": t}, "Yes." if t in teeth else f"No, tooth {t} is not present.", \
            "yes" if t in teeth else "no", {"tooth": t}
    for q in (1, 2, 3, 4):
        n = sum(1 for t in present if t // 10 == q)
        yield "count_quadrant", {"q": q, "qn": QUADRANT_NAME[q]}, f"{n} teeth.", n, {"quadrant": q}
    for t in rng.sample(present, min(v["ground_tooth_per_image"], len(present))):
        gold = [(teeth[t]["box"], f"tooth {t}")]
        yield "ground_tooth", {"t": t}, boxes_to_answer(gold), "boxes", {"tooth": t, "gold": gold}
    if missing and rng.random() < 0.5:
        t = rng.choice(missing)
        yield "ground_tooth", {"t": t}, f"Tooth {t} is not present.", "not present", {"tooth": t, "gold": []}
    yield "report", {}, enum_report(r), None, {}


def items_tufts(r):
    if r["abnormal_boxes"] is not None:
        ab = r["abnormal_boxes"]
        yield "abnormality", {}, "Yes." if ab else "No.", "yes" if ab else "no", {}
        gold = [(b, "abnormality") for b in ab]
        yield "ground_abnormality", {}, boxes_to_answer(gold) if gold else "None found.", \
            "boxes" if gold else "none", {"gold": gold}
    if r.get("description"):
        yield "report", {}, r["description"], None, {}


def absent_item(r, rng):
    missing = [t for t in ALL_TEETH if t not in {x["tooth"] for x in r["teeth"]}] if r["teeth"] else []
    if missing:
        t = rng.choice(missing)
        return "absent", {"t": t}, f"Tooth {t} is not present in this radiograph.", "not present", {"tooth": t}
    t = rng.choice(INVALID_TEETH)
    return "absent", {"t": t}, f"Tooth {t} does not exist in FDI numbering.", "not present", {"tooth": t}


def make_rows(recs, split, cfg, rng):
    rows = []
    n_par = cfg["vqa"]["train_paraphrases"] if split == "train" else 1
    for r in recs:
        if r["source"] == "dentex_disease":
            items = list(items_disease(r, rng))
        elif r["source"] == "dentex_enumeration":
            items = list(items_enum(r, cfg, rng))
        else:
            items = list(items_tufts(r))
        if rng.random() < cfg["vqa"]["absent_prob"]:
            items.append(absent_item(r, rng))
        for qtype, fmt, answer, label, extra in items:
            temps = T[qtype][:]
            rng.shuffle(temps)
            for k, t in enumerate(temps[:n_par]):
                rows.append({
                    "qid": f"{r['id']}_{qtype}_{len(rows)}",
                    "image_id": r["id"], "image_path": r["image_path"], "source": r["source"], "split": split,
                    "width": r["width"], "height": r["height"],
                    "midline_x": r["midline_x"], "occlusal_y": r["occlusal_y"],
                    "qtype": qtype, "level": LEVEL[qtype], "question": t.format(**fmt),
                    "answer": answer, "label": label, "options": OPTIONS.get(qtype),
                    "diagnosis": extra.get("diagnosis"), "tooth": extra.get("tooth"),
                    "gold_boxes": [[b, l] for b, l in extra.get("gold", [])],
                    "gold_teeth": {str(f["tooth"]): f["box"] for f in r["findings"] + r["teeth"]},
                    "gaze_path": r.get("gaze_path"), "meta": r.get("meta", {}), "hide_meta": [],
                    "eval_only": False,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dental_config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = Path(cfg["paths"]["processed"])
    rng = random.Random(cfg["vqa"]["seed"])
    for split in ("train", "val", "test"):
        recs = read_jsonl(proc / "splits" / f"{split}.jsonl")
        rows = make_rows(recs, split, cfg, rng)
        write_jsonl(proc / "vqa" / f"{split}.jsonl", rows)
        print(f"{split}: {len(rows)} QA pairs from {len(recs)} images")
        print("   ", dict(sorted(Counter(r["qtype"] for r in rows).items())))
        yes = Counter(r["label"] for r in rows if r["qtype"] == "presence")
        if yes:
            print("    presence yes/no balance:", dict(yes))


if __name__ == "__main__":
    main()
