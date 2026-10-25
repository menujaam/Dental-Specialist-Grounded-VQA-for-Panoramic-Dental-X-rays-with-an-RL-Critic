"""Metrics for the dental specialist (synopsis Table 6, dental parts)."""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from .common import (DIAGNOSES, extract_all_labels, extract_label, iou, parse_boxes, parse_number,
                     parse_quadrants, parse_teeth, quadrant_from_position, quadrant_of_tooth)

CLOSED = {"presence", "abnormality", "tooth_present", "caries_depth"}
SETS = {"which_teeth": parse_teeth, "which_quadrant": parse_quadrants,
        "tooth_diagnosis": lambda t: extract_all_labels(t, DIAGNOSES)}
GROUNDING = {"ground_diagnosis", "ground_all", "ground_tooth", "ground_abnormality"}


def macro_f1(golds, preds, labels):
    f1s = []
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(golds, preds))
        fp = sum(g != c and p == c for g, p in zip(golds, preds))
        fn = sum(g == c and p != c for g, p in zip(golds, preds))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else None


def set_f1(g, p):
    g, p = set(g), set(p)
    if not g and not p:
        return 1.0
    return 2 * len(g & p) / (len(g) + len(p))


def report_pairs(text):
    """(diagnosis, tooth) pairs: diagnoses and FDI numbers co-occurring in the same clause."""
    pairs = set()
    for chunk in re.split(r"[.;\n]", str(text or "")):
        dxs, teeth = extract_all_labels(chunk, DIAGNOSES), parse_teeth(chunk)
        pairs |= {(d, t) for d in dxs for t in teeth}
    return pairs


def tooth_in_label(label):
    t = parse_teeth(label)
    return t[0] if t else None


def acs_for_row(r, pred_boxes, use_estimated_axes=True):
    """Anatomical Consistency for one grounding answer.
    quadrant-level: the box position matches the quadrant of the tooth number in its own label.
    tooth-level:    the box overlaps the gold box of that tooth (when a gold box is known)."""
    W, H = r["width"], r["height"]
    mid = r["midline_x"] if use_estimated_axes else W / 2
    occ = r["occlusal_y"] if use_estimated_axes else H / 2
    q_scores, t_scores = [], []
    for box, lab in pred_boxes:
        t = tooth_in_label(lab) or r.get("tooth")
        if not t:
            continue
        q_scores.append(float(quadrant_from_position(box, mid, occ) == quadrant_of_tooth(t)))
        g = r["gold_teeth"].get(str(t))
        if g is not None:
            t_scores.append(float(iou(box, g) >= r.get("_tooth_iou", 0.3)))
    return q_scores, t_scores


def compute_metrics(preds, acs_cfg=None):
    acs_cfg = acs_cfg or {}
    by = defaultdict(list)
    for p in preds:
        by[p["qtype"]].append(p)
    res = {"n": len(preds), "per_qtype": {}}
    level_acc = defaultdict(list)
    acs_q_est, acs_q_def, acs_t = [], [], []

    for qt, rows in by.items():
        m = {"n": len(rows)}
        if qt in CLOSED:
            g = [r["label"] for r in rows]
            pr = [extract_label(r["pred"], r["options"]) for r in rows]
            ok = [a == b for a, b in zip(g, pr)]
            m.update(accuracy=float(np.mean(ok)), macro_f1=macro_f1(g, pr, rows[0]["options"]),
                     unparsed=float(np.mean([x is None for x in pr])))
            if qt == "presence":   # per-diagnosis accuracy
                for dx in DIAGNOSES:
                    sub = [o for o, r in zip(ok, rows) if r["diagnosis"] == dx]
                    if sub:
                        m[f"acc_{dx.replace(' ', '_')}"] = float(np.mean(sub))
            level_acc[rows[0]["level"]] += ok
        elif qt in SETS:
            f = SETS[qt]
            sc = [(set(r["label"]), set(f(r["pred"]))) for r in rows]
            ok = [a == b for a, b in sc]
            m.update(exact_match=float(np.mean(ok)), set_f1=float(np.mean([set_f1(a, b) for a, b in sc])))
            level_acc[rows[0]["level"]] += ok
        elif qt == "count_quadrant":
            pairs = [(r["label"], parse_number(r["pred"])) for r in rows]
            okp = [(a, b) for a, b in pairs if b is not None]
            m["mae"] = float(np.mean([abs(a - b) for a, b in okp])) if okp else None
            m["exact"] = float(np.mean([b is not None and round(b) == a for a, b in pairs]))
            level_acc[2] += [b is not None and round(b) == a for a, b in pairs]
        elif qt == "absent":
            pr = [extract_label(r["pred"], r["options"]) for r in rows]
            m["hallucination_rate"] = float(np.mean([x != "not present" for x in pr]))
        elif qt in GROUNDING:
            pos = [r for r in rows if r["gold_boxes"]]
            neg = [r for r in rows if not r["gold_boxes"]]
            best_ious, recall50, prec50, point, label_ok = [], [], [], [], []
            for r in pos:
                pb = parse_boxes(r["pred"])
                gold = r["gold_boxes"]
                best_ious.append(max((iou(a, g[0]) for a, _ in pb for g in gold), default=0.0))
                recall50 += [any(iou(a, g[0]) >= 0.5 for a, _ in pb) for g in gold]
                prec50 += [any(iou(a, g[0]) >= 0.5 for g in gold) for a, _ in pb]
                if pb:
                    b = pb[0][0]
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    point.append(any(g[0][0] <= cx <= g[0][2] and g[0][1] <= cy <= g[0][3] for g in gold))
                else:
                    point.append(False)
                # box matched with the correct tooth number in its label
                for g in gold:
                    gt = tooth_in_label(g[1])
                    if gt:
                        label_ok.append(any(iou(a, g[0]) >= 0.5 and tooth_in_label(l) == gt for a, l in pb))
            if pos:
                m.update(n_pos=len(pos), mean_best_iou=float(np.mean(best_ious)),
                         recall_at_0_5=float(np.mean(recall50)) if recall50 else None,
                         precision_at_0_5=float(np.mean(prec50)) if prec50 else 0.0,
                         pointing_game=float(np.mean(point)))
                if label_ok:
                    m["recall_at_0_5_correct_tooth"] = float(np.mean(label_ok))
            if neg:
                m.update(n_neg=len(neg), false_box_rate=float(np.mean([bool(parse_boxes(r["pred"])) for r in neg])))
            if qt != "ground_abnormality":
                for r in rows:
                    pb = parse_boxes(r["pred"])
                    r["_tooth_iou"] = acs_cfg.get("tooth_iou", 0.3)
                    q, t = acs_for_row(r, pb, True)
                    qd, _ = acs_for_row(r, pb, False)
                    acs_q_est += q
                    acs_q_def += qd
                    acs_t += t
        elif qt == "report":
            pf = [set_f1(report_pairs(r["answer"]), report_pairs(r["pred"]))
                  for r in rows if report_pairs(r["answer"])]
            m["finding_pair_f1"] = float(np.mean(pf)) if pf else None
            try:
                from rouge_score import rouge_scorer
                sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                m["rougeL"] = float(np.mean([sc.score(r["answer"], r["pred"])["rougeL"].fmeasure for r in rows]))
            except ImportError:
                pass
            try:
                import sacrebleu
                m["bleu"] = sacrebleu.corpus_bleu([r["pred"] for r in rows], [[r["answer"] for r in rows]]).score
            except ImportError:
                pass
        res["per_qtype"][qt] = m

    res["acs"] = {
        "quadrant_estimated_axes": float(np.mean(acs_q_est)) if acs_q_est else None,
        "quadrant_image_centre": float(np.mean(acs_q_def)) if acs_q_def else None,
        "tooth_level": float(np.mean(acs_t)) if acs_t else None,
        "n_boxes": len(acs_q_est),
    }
    res["accuracy_by_level"] = {f"L{k}": float(np.mean(v)) for k, v in sorted(level_acc.items())}
    allc = [x for v in level_acc.values() for x in v]
    res["closed_accuracy"] = float(np.mean(allc)) if allc else None
    return res
