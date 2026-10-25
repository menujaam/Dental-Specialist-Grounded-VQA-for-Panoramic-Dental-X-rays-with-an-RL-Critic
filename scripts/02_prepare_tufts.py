"""Convert Tufts Dental Database into unified dental records (optional; needs access approval).

Uses: radiographs, expert abnormality masks (-> boxes), expert text descriptions (-> report target),
and expert eye-gaze maps (-> gaze-alignment evaluation). Folder names are set in the config;
run scripts/00_inspect_dataset.py on the Tufts folder first to check them.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import index_images, load_config, resize_for_qwen, universal_to_fdi, write_jsonl  # noqa: E402


def load_descriptions(path, img_keys, txt_keys):
    if not path or not Path(path).exists():
        print(f"  no description file at {path}")
        return {}
    d = json.load(open(path))
    items = d if isinstance(d, list) else [dict(v, **{"__key": k}) if isinstance(v, dict) else
                                           {"__key": k, "__text": v} for k, v in d.items()]
    out = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        name = next((it[k] for k in img_keys + ["__key"] if k in it), None)
        src = it.get("Label") if isinstance(it.get("Label"), dict) else {}
        text = next((it[k] for k in txt_keys + ["__text"] if k in it), None) or \
            next((src[k] for k in txt_keys if k in src), None)
        if name and isinstance(text, str) and text.strip():
            out[Path(str(name)).stem.lower()] = text.strip()
    print(f"  loaded {len(out)} descriptions")
    return out


def to_fdi_text(text):
    """Replace Universal tooth numbers written as '#N' or 'tooth N' with FDI numbers."""
    def rep(m):
        f = universal_to_fdi(m.group(2))
        return f"{m.group(1)}{f}" if f else m.group(0)
    return re.sub(r"(#\s*|tooth\s+)(\d{1,2})\b", rep, text, flags=re.I)


def mask_boxes(mask_path, size, min_area):
    m = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
    a = np.asarray(m) > 0
    lab, n = ndimage.label(a)
    boxes = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        ys, xs = sl
        if (ys.stop - ys.start) * (xs.stop - xs.start) >= min_area:
            boxes.append([float(xs.start), float(ys.start), float(xs.stop), float(ys.stop)])
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dental_config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P, IM, T = cfg["paths"], cfg["image"], cfg["tufts"]
    if not T.get("enabled"):
        print("Tufts disabled in config")
        return
    root = Path(P["tufts_root"])
    rad = root / T["radiographs_dir"]
    if not rad.exists():
        sys.exit(f"Radiographs folder not found: {rad}\nRun scripts/00_inspect_dataset.py {root} and fix the config.")

    radiographs = sorted({p for p in index_images(rad).values()})
    masks = index_images(root / T["abnormality_mask_dir"]) if (root / T["abnormality_mask_dir"]).exists() else {}
    gazes = index_images(root / T["gaze_dir"]) if (root / T["gaze_dir"]).exists() else {}
    desc = load_descriptions(root / T["description_json"] if T.get("description_json") else None,
                             T["description_image_keys"], T["description_text_keys"])
    print(f"  {len(radiographs)} radiographs, masks dir: {bool(masks)}, gaze dir: {bool(gazes)}")

    out_dir = Path(P["processed"]) / "images" / "tufts"
    records, stats = [], Counter()
    for p in radiographs:
        stem = p.stem.lower()
        rid = f"tufts_{p.stem}"
        (W0, H0), (w, h) = resize_for_qwen(p, out_dir / f"{rid}.png", IM["min_pixels"], IM["max_pixels"])
        mp = masks.get(stem)
        boxes = mask_boxes(mp, (w, h), T["mask_min_area"]) if mp else None
        text = desc.get(stem)
        if text and T.get("convert_universal_to_fdi"):
            text = to_fdi_text(text)
        stats["mask"] += mp is not None
        stats["abnormal"] += bool(boxes)
        stats["description"] += text is not None
        stats["gaze"] += stem in gazes
        records.append({
            "id": rid, "source": "tufts",
            "image_path": str((out_dir / f"{rid}.png").resolve()), "orig_path": str(p.resolve()),
            "orig_size": [W0, H0], "width": w, "height": h, "group": rid,
            "findings": [], "teeth": [],
            "abnormal_boxes": boxes,             # None = no mask available (unknown)
            "description": text,
            "gaze_path": str(gazes[stem].resolve()) if stem in gazes else None,
            "midline_x": w * cfg["acs"]["midline_frac"], "occlusal_y": h * cfg["acs"]["occlusal_frac"],
            "meta": {},
        })
    out = Path(P["processed"]) / "records_tufts.jsonl"
    write_jsonl(out, records)
    print(f"Wrote {len(records)} records -> {out}")
    print(f"  with mask {stats['mask']}, abnormal {stats['abnormal']}, description {stats['description']}, "
          f"gaze {stats['gaze']}")


if __name__ == "__main__":
    main()
