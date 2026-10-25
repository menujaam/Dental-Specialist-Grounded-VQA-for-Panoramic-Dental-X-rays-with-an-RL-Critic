"""Convert DENTEX (COCO-style, hierarchical labels) into unified dental records.

DENTEX annotation files carry up to three category ids per box:
  category_id_1 -> quadrant (1-4), category_id_2 -> tooth number (1-8), category_id_3 -> diagnosis.
The subset type is detected from which ids are present:
  disease      (quadrant + tooth + diagnosis, abnormal teeth only)
  enumeration  (quadrant + tooth, ALL teeth)  -> tooth presence / tooth grounding
  quadrant     (quadrant only)                -> skipped
Usage: python scripts/01_prepare_dentex.py --config configs/dental_config.yaml
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import (estimate_axes, index_images, load_config, quadrant_from_position,  # noqa: E402
                           resize_for_qwen, write_jsonl)


def cat_map(d, key, offset_if_missing=1):
    cats = d.get(key) or []
    m = {c["id"]: str(c.get("name", c["id"])) for c in cats}
    return lambda i: m.get(i, str(i + offset_if_missing))


def to_int(name, fallback):
    try:
        return int(str(name).strip())
    except ValueError:
        return fallback


def load_subsets(root):
    out = []
    for jp in sorted(Path(root).rglob("*.json")):
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        if not isinstance(d, dict) or not d.get("images") or not d.get("annotations"):
            continue
        keys = set().union(*[a.keys() for a in d["annotations"][:200]])
        kind = ("disease" if "category_id_3" in keys else "enumeration" if "category_id_2" in keys
                else "quadrant" if "category_id_1" in keys else None)
        if kind:
            out.append((kind, jp, d))
            print(f"  found {kind:11s} subset: {jp}  ({len(d['images'])} images, {len(d['annotations'])} boxes)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dental_config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P, IM, D, A = cfg["paths"], cfg["image"], cfg["dentex"], cfg["acs"]
    dx_map = {k.lower(): v for k, v in D["diagnosis_names"].items()}
    out_dir = Path(P["processed"]) / "images" / "dentex"

    records, problems = [], Counter()
    acs_check = Counter()
    for kind, jp, d in load_subsets(P["dentex_root"]):
        if kind not in D["use_subsets"]:
            print(f"  skipping {kind} subset")
            continue
        q_name, t_name, d_name = cat_map(d, "categories_1"), cat_map(d, "categories_2"), cat_map(d, "categories_3", 0)
        idx = index_images(jp.parent)
        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        for im in d["images"]:
            path = idx.get(Path(im["file_name"]).name.lower())
            if path is None:
                problems[f"{kind}: image not found"] += 1
                continue
            rid = f"dx_{kind[:3]}_{Path(im['file_name']).stem}"
            (W0, H0), (w, h) = resize_for_qwen(path, out_dir / f"{rid}.png", IM["min_pixels"], IM["max_pixels"])
            sx, sy = w / W0, h / H0
            items = []
            for a in anns[im["id"]]:
                x, y, bw, bh = a["bbox"]
                box = [x * sx, y * sy, (x + bw) * sx, (y + bh) * sy]
                q = to_int(q_name(a["category_id_1"]), a["category_id_1"] + 1)
                t = to_int(t_name(a["category_id_2"]), a["category_id_2"] + 1)
                if not (1 <= q <= 4 and 1 <= t <= 8):
                    problems[f"{kind}: invalid quadrant/tooth"] += 1
                    continue
                it = {"tooth": q * 10 + t, "quadrant": q, "box": box}
                if kind == "disease":
                    raw = d_name(a["category_id_3"]).lower().strip()
                    if raw not in dx_map:
                        problems[f"unknown diagnosis '{raw}'"] += 1
                        continue
                    it["diagnosis"] = dx_map[raw]
                items.append(it)
            mid, occ = estimate_axes(items, w, h, A["midline_frac"], A["occlusal_frac"])
            for it in items:   # sanity check of the quadrant heuristic on GOLD boxes
                acs_check["default"] += quadrant_from_position(it["box"], w * A["midline_frac"], h * A["occlusal_frac"]) == it["quadrant"]
                acs_check["estimated"] += quadrant_from_position(it["box"], mid, occ) == it["quadrant"]
                acs_check["n"] += 1
            records.append({
                "id": rid, "source": f"dentex_{kind}",
                "image_path": str((out_dir / f"{rid}.png").resolve()), "orig_path": str(path.resolve()),
                "orig_size": [W0, H0], "width": w, "height": h, "group": rid,
                "findings": items if kind == "disease" else [],
                "teeth": items if kind == "enumeration" else [],
                "abnormal_boxes": [], "description": None, "gaze_path": None,
                "midline_x": mid, "occlusal_y": occ, "meta": {},
            })

    out = Path(P["processed"]) / "records_dentex.jsonl"
    write_jsonl(out, records)
    print(f"Wrote {len(records)} records -> {out}")
    print("  by subset:", dict(Counter(r["source"] for r in records)))
    print("  diagnoses:", dict(Counter(f["diagnosis"] for r in records for f in r["findings"])))
    if acs_check["n"]:
        print(f"  quadrant-from-position agreement on gold boxes: image-centre axes "
              f"{acs_check['default']/acs_check['n']:.3f}, estimated axes {acs_check['estimated']/acs_check['n']:.3f}"
              "  (upper bound for quadrant-level ACS)")
    for k, v in problems.items():
        print(f"  WARNING: {v} -> {k}")


if __name__ == "__main__":
    main()
