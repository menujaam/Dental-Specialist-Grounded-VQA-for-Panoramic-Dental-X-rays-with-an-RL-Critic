"""Shared helpers for the dental specialist: images, FDI numbering, prompts, answer parsing."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import yaml

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

SYSTEM_PROMPT = (
    "You are a dental radiology assistant reading panoramic X-rays. Use FDI tooth numbering "
    "(11-18, 21-28, 31-38, 41-48). Quadrant 1 = upper right, 2 = upper left, 3 = lower left, "
    "4 = lower right (patient's side). Answer concisely, starting with the direct answer. When asked "
    'to localise, output a JSON list such as [{"bbox_2d": [x1, y1, x2, y2], "label": "caries 36"}] '
    "in pixel coordinates of the given image. If a tooth asked about is not present, say so."
)

DIAGNOSES = ["caries", "deep caries", "periapical lesion", "impacted"]
QUADRANT_NAME = {1: "upper right", 2: "upper left", 3: "lower left", 4: "lower right"}
ALL_TEETH = [q * 10 + t for q in (1, 2, 3, 4) for t in range(1, 9)]
INVALID_TEETH = [19, 29, 39, 49, 10, 20, 30, 40]   # do not exist in FDI


# --------------------------------------------------------------------------- io
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def index_images(root):
    idx = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            idx.setdefault(p.name.lower(), p)
            idx.setdefault(p.stem.lower(), p)
    return idx


# ------------------------------------------------------------------- images
def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280):
    """Qwen2.5-VL resize rule: dims multiple of 28 within the pixel budget."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def to_rgb(im):
    from PIL import Image
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        import numpy as np
        a = np.asarray(im, dtype=np.float32)
        lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        im = Image.fromarray(a.astype(np.uint8), "L")
    return im.convert("RGB")


def resize_for_qwen(src, dst, min_pixels, max_pixels):
    from PIL import Image
    with Image.open(src) as im:
        im = to_rgb(im)
        W, H = im.size
        h, w = smart_resize(H, W, min_pixels=min_pixels, max_pixels=max_pixels)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        im.resize((w, h), Image.BICUBIC).save(dst)
    return (W, H), (w, h)


# --------------------------------------------------------------- FDI helpers
def quadrant_of_tooth(fdi):
    return int(fdi) // 10


def universal_to_fdi(u):
    u = int(u)
    if 1 <= u <= 8:
        return 10 + (9 - u)
    if 9 <= u <= 16:
        return 20 + (u - 8)
    if 17 <= u <= 24:
        return 30 + (25 - u)
    if 25 <= u <= 32:
        return 40 + (u - 24)
    return None


def quadrant_from_position(box, midline_x, occlusal_y):
    """Image quadrant -> FDI quadrant. Patient's right appears on the image LEFT."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    upper, img_left = cy < occlusal_y, cx < midline_x
    if upper:
        return 1 if img_left else 2
    return 4 if img_left else 3


def estimate_axes(items, W, H, midline_frac=0.5, occlusal_frac=0.5):
    """Estimate midline x and occlusal y from labelled boxes [{'quadrant', 'box'}]."""
    cx = lambda b: (b[0] + b[2]) / 2  # noqa: E731
    cy = lambda b: (b[1] + b[3]) / 2  # noqa: E731
    left = [cx(i["box"]) for i in items if i["quadrant"] in (1, 4)]
    right = [cx(i["box"]) for i in items if i["quadrant"] in (2, 3)]
    up = [cy(i["box"]) for i in items if i["quadrant"] in (1, 2)]
    low = [cy(i["box"]) for i in items if i["quadrant"] in (3, 4)]
    mid = (max(left) + min(right)) / 2 if left and right and max(left) < min(right) else W * midline_frac
    occ = (max(up) + min(low)) / 2 if up and low and max(up) < min(low) else H * occlusal_frac
    return mid, occ


# ------------------------------------------------------------------ prompts
def build_prompt(question, meta=None, context="meta", hide=()):
    m = {k: v for k, v in (meta or {}).items() if v not in (None, "") and k not in set(hide)}
    if context == "none" or not m:
        return question
    ctx = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in m.items())
    return f"Patient context: {ctx}\nQuestion: {question}"


def build_messages(rec, context="meta", with_image=True, include_answer=False):
    q = build_prompt(rec["question"], rec.get("meta"), context, rec.get("hide_meta", []))
    user = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": q}]
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user}]
    if include_answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": rec["answer"]}]})
    return msgs


# ------------------------------------------------------------ answer parsing
LABEL_SYNONYMS = {
    "yes": ["yes"], "no": ["no"],
    "caries": ["caries", "carious lesion", "decay"],
    "deep caries": ["deep caries", "deep carious lesion"],
    "periapical lesion": ["periapical lesion", "periapical radiolucency", "apical lesion"],
    "impacted": ["impacted", "impaction"],
    "deep": ["deep"], "not deep": ["not deep", "superficial", "shallow"],
    "not present": ["not present", "missing", "does not exist", "no tooth", "absent", "not visible"],
    "none": ["none found", "no findings", "none", "no abnormal"],
}


def _matches(text, options):
    """All (start, end, option) matches, longest synonyms first, non-overlapping."""
    t = str(text or "").lower()
    cands = [(syn, opt) for opt in options for syn in LABEL_SYNONYMS.get(opt, [opt])]
    cands.sort(key=lambda x: -len(x[0]))
    taken, out = [], []
    for syn, opt in cands:
        for m in re.finditer(r"(?<![a-z])" + re.escape(syn) + r"(?![a-z])", t):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            out.append((m.start(), m.end(), opt))
    return sorted(out)


def extract_label(text, options):
    m = _matches(text, options)
    return m[0][2] if m else None


def extract_all_labels(text, options):
    return sorted({o for _, _, o in _matches(text, options)})


def parse_teeth(text):
    return sorted({int(x) for x in re.findall(r"(?<!\d)([1-4][1-8])(?!\d)", str(text or ""))})


def parse_quadrants(text):
    return sorted({int(x) for x in re.findall(r"\b(?:quadrant|q)\s*([1-4])\b", str(text or "").lower())})


_OBJ_RE = re.compile(r"\{[^{}]*\}")
_BOX_RE = re.compile(r'"?bbox_2d"?\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]')
_LAB_RE = re.compile(r'"?label"?\s*:\s*"([^"]*)"')


def parse_boxes(text):
    """Return [(box, label)] from the model's JSON-ish grounding output."""
    out = []
    for obj in _OBJ_RE.findall(str(text or "")):
        b = _BOX_RE.search(obj)
        if not b:
            continue
        x1, y1, x2, y2 = map(float, b.groups())
        lab = _LAB_RE.search(obj)
        out.append(([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], lab.group(1) if lab else ""))
    return out


def parse_number(text):
    m = re.search(r"[-+]?\d*\.?\d+", str(text or ""))
    return float(m.group()) if m else None


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def boxes_to_answer(items):
    """items: [(box, label)] -> compact JSON answer."""
    return json.dumps([{"bbox_2d": [int(round(v)) for v in b], "label": l} for b, l in items])
