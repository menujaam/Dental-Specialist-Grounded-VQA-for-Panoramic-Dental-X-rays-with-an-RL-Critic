"""Full error analysis of a model's saved test answers (CPU only, no model needed).

Reads a predictions file written by 06_predict.py and counts every error by type over the WHOLE test set:
  1. Tooth boxes ("box tooth X"): right tooth / one tooth off / further in same quadrant / wrong quadrant / wrong jaw
  2. Diagnosis boxes: findings found vs missed (per diagnosis), extra boxes, wrong label on a found box
  3. Duplicate boxes, and the grounding scores after removing them
  4. Diagnosis mix-ups (confusion table): which diagnosis is called which
  5. Presence and depth questions: missed (false "no") vs invented (false "yes"), per diagnosis
  6. Tooth present / not present errors, by tooth type (front teeth, canines, premolars, molars, wisdom teeth)
  7. Counting errors (how far off), and "which teeth" lists (missed vs extra teeth)

  python scripts/15_error_analysis.py                                        # your v1 fine-tuned model
  python scripts/15_error_analysis.py --preds preds/dental_lora_3b_v2_image.jsonl --name v2

Saves results/error_analysis_<name>.json and results/error_analysis_<name>.md (tables to paste in the report).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import (DIAGNOSES, extract_all_labels, extract_label, iou, parse_boxes,  # noqa: E402
                           parse_number, parse_teeth, read_jsonl)

GROUND_DX = {"ground_diagnosis", "ground_all"}
TYPE = {1: "front teeth (incisors)", 2: "front teeth (incisors)", 3: "canines", 4: "premolars", 5: "premolars",
        6: "molars", 7: "molars", 8: "wisdom teeth"}


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def P(a, b):
    """Percentage as text for the tables."""
    return f"{pct(a, b)}%" if b else "—"


def P1(x):
    return "—" if x is None else f"{x}%"


def dx_of(label):
    return extract_label(label, DIAGNOSES)


def tooth_of(label):
    t = parse_teeth(label)
    return t[0] if t else None


def tooth_distance(a, b):
    """How many teeth apart two FDI teeth are along the arch; None if on different jaws."""
    qa, ia, qb, ib = a // 10, a % 10, b // 10, b % 10
    if qa == qb:
        return abs(ia - ib)
    if {qa, qb} in ({1, 2}, {3, 4}):          # same jaw, across the midline (11 and 21 are neighbours)
        return ia + ib - 1
    return None


def dedupe(boxes, thr=0.9):
    out = []
    for b, lab in boxes:
        if not any(iou(b, o) >= thr and dx_of(lab) == dx_of(ol) for o, ol in out):
            out.append((b, lab))
    return out


def ground_scores(rows, dedup=False):
    rec, prec = [], []
    for r in rows:
        gold = [g[0] for g in r["gold_boxes"]]
        if not gold:
            continue
        pb = [b for b, _ in (dedupe(parse_boxes(r["pred"])) if dedup else parse_boxes(r["pred"]))]
        rec += [any(iou(g, p) >= 0.5 for p in pb) for g in gold]
        prec += [any(iou(p, g) >= 0.5 for g in gold) for p in pb]
    return dict(recall_at_0_5=pct(sum(rec), len(rec)), precision_at_0_5=pct(sum(prec), len(prec)),
                n_model_boxes=len(prec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds/dental_lora_3b_image.jsonl")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    rows = read_jsonl(args.preds)
    by = defaultdict(list)
    for r in rows:
        by[r["qtype"]].append(r)
    R, md = {}, [f"# Error analysis: {args.name}", f"Predictions: `{args.preds}`, {len(rows)} test questions.", ""]

    # ---------------------------------------------------------------- 1. tooth boxes
    c, n = Counter(), 0
    by_type = defaultdict(Counter)
    for r in by["ground_tooth"]:
        gold = r["gold_boxes"]
        if not gold:                                   # tooth is missing: right answer is "not present"
            c["missing tooth: model drew a box anyway" if parse_boxes(r["pred"])
              else "missing tooth: correctly said not present"] += 1
            continue
        n += 1
        t = r["tooth"]
        pb = parse_boxes(r["pred"])
        if not pb:
            k = "said 'not present' / no box although the tooth exists"
        else:
            box = pb[0][0]
            if iou(box, gold[0][0]) >= 0.5:
                k = "right tooth (IoU >= 0.5)"
            else:                                      # which real tooth did the box land on?
                cand = [(iou(box, b), int(tt)) for tt, b in (r.get("gold_teeth") or {}).items()]
                best = max(cand, default=(0, None))
                if best[0] < 0.3 or best[1] is None:
                    k = "box not on any tooth"
                elif best[1] == t:
                    k = "right tooth, box too loose/tight (IoU < 0.5)"
                else:
                    d = tooth_distance(t, best[1])
                    k = ("one tooth off" if d == 1 else "2+ teeth off, same jaw" if d is not None
                         else "wrong jaw (upper vs lower)")
        c[k] += 1
        by_type[TYPE.get(t % 10, "?")]["right" if k.startswith("right tooth (") else "wrong"] += 1
    R["tooth_boxes"] = dict(counts=dict(c), n_present=n)
    md += ["## 1. Box around a named tooth", "", "| Outcome | Count | Share of present-tooth questions |",
           "| --- | --- | --- |"]
    for k, v in c.most_common():
        md.append(f"| {k} | {v} | {'—' if k.startswith('missing tooth') else P(v, n)} |")
    wrong = n - c["right tooth (IoU >= 0.5)"]
    md += ["", f"Of {wrong} wrong boxes on present teeth, {c['one tooth off']} "
               f"are exactly one tooth off ({P(c['one tooth off'], wrong)}).", "", "| Tooth type | Correct boxes |", "| --- | --- |"]
    for k, v in by_type.items():
        md.append(f"| {k} | {v['right']} of {v['right'] + v['wrong']} ({P(v['right'], v['right'] + v['wrong'])}) |")
    R["tooth_boxes"]["by_tooth_type"] = {k: dict(v) for k, v in by_type.items()}

    # ---------------------------------------------------- 2. diagnosis boxes: found / missed / extra / wrong label
    found, missed, extra, wrong_lab, wrong_tooth = Counter(), Counter(), Counter(), Counter(), Counter()
    for r in by["ground_diagnosis"] + by["ground_all"]:
        gold = [(g[0], dx_of(g[1]), tooth_of(g[1])) for g in r["gold_boxes"]]
        pb = parse_boxes(r["pred"])
        for gb, gdx, gt in gold:
            hit = [(p, lab) for p, lab in pb if iou(p, gb) >= 0.5]
            if hit:
                found[gdx] += 1
                if dx_of(hit[0][1]) != gdx:
                    wrong_lab[f"{gdx} called {dx_of(hit[0][1])}"] += 1
                if tooth_of(hit[0][1]) != gt:
                    wrong_tooth[gdx] += 1
            else:
                missed[gdx] += 1
        for p, lab in pb:
            if not any(iou(p, g[0]) >= 0.5 for g in gold):
                extra[dx_of(lab) or "unlabelled"] += 1
    R["diagnosis_boxes"] = dict(found=dict(found), missed=dict(missed), extra=dict(extra),
                                wrong_label=dict(wrong_lab), wrong_tooth_number=dict(wrong_tooth))
    md += ["", "## 2. Boxes around findings (diagnosis and 'find all' questions)", "",
           "| Diagnosis | Expert findings | Found | Missed | Extra model boxes (no expert finding there) |",
           "| --- | --- | --- | --- | --- |"]
    for dx in DIAGNOSES:
        tot = found[dx] + missed[dx]
        md.append(f"| {dx} | {tot} | {found[dx]} ({P(found[dx], tot)}) | {missed[dx]} | {extra[dx]} |")
    if wrong_lab:
        md += ["", "Found but labelled with the wrong diagnosis: "
               + ", ".join(f"{k} ({v})" for k, v in wrong_lab.most_common()) + "."]
    nf = sum(found.values())
    md += ["", f"Found boxes with the wrong tooth number: {sum(wrong_tooth.values())} of {nf} "
               f"({P(sum(wrong_tooth.values()), nf)})."]

    # ------------------------------------------------------------- 3. duplicate boxes
    dup_rows = sum(len(dedupe(parse_boxes(r["pred"]))) < len(parse_boxes(r["pred"]))
                   for qt in ("ground_diagnosis", "ground_all", "ground_tooth") for r in by[qt])
    before = {qt: ground_scores(by[qt]) for qt in ("ground_diagnosis", "ground_all", "ground_tooth")}
    after = {qt: ground_scores(by[qt], dedup=True) for qt in ("ground_diagnosis", "ground_all", "ground_tooth")}
    R["duplicates"] = dict(answers_with_duplicates=dup_rows, before=before, after=after)
    md += ["", "## 3. Duplicate boxes", "", f"Answers containing a repeated box: {dup_rows}.", "",
           "Repeated copies of a correct box make precision look higher, so the 'after' column is the more honest one.", "",
           "| Question type | Precision@0.5 before | after removing duplicates | Recall@0.5 (unchanged) |",
           "| --- | --- | --- | --- |"]
    for qt in before:
        md.append(f"| {qt} | {P1(before[qt]['precision_at_0_5'])} | {P1(after[qt]['precision_at_0_5'])} | "
                  f"{P1(before[qt]['recall_at_0_5'])} |")

    # -------------------------------------------------- 4. diagnosis mix-ups (tooth_diagnosis questions)
    conf, added = Counter(), Counter()
    for r in by["tooth_diagnosis"]:
        g, p = set(r["label"] or []), set(extract_all_labels(r["pred"], DIAGNOSES))
        for d in g:      # each expert diagnosis: kept, replaced by another, or dropped
            conf[(d, d if d in p else ("; ".join(sorted(p - g)) or "nothing"))] += 1
        if g <= p:       # nothing replaced, so anything else is a genuinely added diagnosis
            for d in p - g:
                added[d] += 1
    md += ["", "## 4. Diagnosis mix-ups ('What is the finding on tooth X?')", "",
           "| Expert said | Model said | Count |", "| --- | --- | --- |"]
    mix = {}
    for (g, p), v in conf.most_common():
        md.append(f"| {g} | {p} | {v} |")
        mix[f"{g} -> {p}"] = v
    extras = dict(added)
    if extras:
        md += ["", "Diagnoses added that the expert did not give: "
               + ", ".join(f"{k} ({v})" for k, v in sorted(extras.items(), key=lambda x: -x[1])) + "."]
    R["diagnosis_mixups"] = dict(pairs=mix, added=extras)

    # ------------------------------------------- 5. presence + depth: missed vs invented
    pres = defaultdict(Counter)
    for r in by["presence"]:
        p = extract_label(r["pred"], r["options"])
        pres[r["diagnosis"]][("ok" if p == r["label"] else "missed (said no)" if r["label"] == "yes"
                              else "invented (said yes)")] += 1
    depth = Counter((r["label"], extract_label(r["pred"], r["options"]) or "unclear") for r in by["caries_depth"])
    R["presence"] = {k: dict(v) for k, v in pres.items()}
    R["caries_depth"] = {f"{a} -> {b}": v for (a, b), v in depth.items()}
    md += ["", "## 5. 'Is there X?' questions: missed vs invented", "",
           "| Diagnosis | Correct | Missed (said no, was yes) | Invented (said yes, was no) |", "| --- | --- | --- | --- |"]
    for dx in DIAGNOSES:
        v = pres.get(dx, Counter())
        md.append(f"| {dx} | {v['ok']} | {v['missed (said no)']} | {v['invented (said yes)']} |")
    md += ["", "Caries depth ('Is the caries on tooth X deep?'):", "",
           "| Expert | Model said deep | Model said not deep |", "| --- | --- | --- |",
           f"| deep | {depth[('deep', 'deep')]} | {depth[('deep', 'not deep')]} |",
           f"| not deep | {depth[('not deep', 'deep')]} | {depth[('not deep', 'not deep')]} |"]
    dd = depth[("deep", "deep")] + depth[("deep", "not deep")]
    md += ["", f"Deep caries recognised as deep: {depth[('deep', 'deep')]} of {dd} ({P(depth[('deep', 'deep')], dd)})."]

    # ------------------------------------------- 6. tooth present / not present
    tp = defaultdict(Counter)
    for r in by["tooth_present"]:
        p = extract_label(r["pred"], r["options"])
        k = ("ok" if p == r["label"] else "said missing, but present" if r["label"] == "yes"
             else "said present, but missing")
        tp[TYPE.get(r["tooth"] % 10, "?")][k] += 1
    R["tooth_present"] = {k: dict(v) for k, v in tp.items()}
    md += ["", "## 6. 'Is tooth X present?' errors by tooth type", "",
           "| Tooth type | Correct | Said missing, but present | Said present, but missing |", "| --- | --- | --- | --- |"]
    for k, v in tp.items():
        md.append(f"| {k} | {v['ok']} | {v['said missing, but present']} | {v['said present, but missing']} |")

    # ------------------------------------------- 7. counting and lists
    cnt = Counter()
    for r in by["count_quadrant"]:
        x = parse_number(r["pred"])
        cnt["unreadable" if x is None else f"{int(round(x)) - r['label']:+d}"] += 1
    lst = Counter()
    for r in by["which_teeth"]:
        g, p = set(r["label"]), set(parse_teeth(r["pred"]))
        lst["exactly right" if g == p else "missed teeth only" if p < g else "extra teeth only" if p > g
            else "both missed and extra"] += 1
    R["counting_error"] = dict(cnt)
    R["which_teeth"] = dict(lst)
    md += ["", "## 7. Counting and lists", "", "Teeth counted in a quadrant (model minus expert):", "",
           "| Error | Count |", "| --- | --- |"]
    for k in sorted(cnt, key=lambda s: (s == "unreadable", abs(int(s)) if s != "unreadable" else 0, s)):
        md.append(f"| {k} | {cnt[k]} |")
    md += ["", "'Which teeth have X?' lists:", "", "| Outcome | Count |", "| --- | --- |"]
    for k, v in lst.most_common():
        md.append(f"| {k} | {v} |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(out / f"error_analysis_{args.name}.json", "w"), indent=2)
    (out / f"error_analysis_{args.name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nsaved {out}/error_analysis_{args.name}.json and .md")


if __name__ == "__main__":
    main()
