"""Model heatmaps vs. radiologist eye-gaze (Tufts): CC, NSS, KL divergence.

Three kinds of model heatmap:
  * attention : answer-token -> image-token attention of the VLM (needs the model, GPU)
  * boxes     : Gaussian blobs at the predicted boxes (from a predictions file, CPU)
  * centre    : centre-bias baseline (a saliency model must beat this to be meaningful)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

EPS = 1e-8


def load_gaze(path, size):
    g = np.asarray(Image.open(path).convert("L").resize(size, Image.BILINEAR), dtype=np.float64)
    return g / (g.max() + EPS)


def resize_map(m, size):
    m = np.asarray(m, dtype=np.float32)
    img = Image.fromarray(m / (m.max() + EPS))
    return np.asarray(img.resize(size, Image.BILINEAR), dtype=np.float64)


def boxes_heatmap(boxes, size, sigma_scale=0.5):
    W, H = size
    yy, xx = np.mgrid[0:H, 0:W]
    m = np.zeros((H, W))
    for b in boxes:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        sx, sy = max((b[2] - b[0]) * sigma_scale, 4), max((b[3] - b[1]) * sigma_scale, 4)
        m += np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)
    return m


def centre_bias(size, sigma_frac=0.25):
    W, H = size
    return boxes_heatmap([[W / 2 - W * sigma_frac, H / 2 - H * sigma_frac,
                           W / 2 + W * sigma_frac, H / 2 + H * sigma_frac]], size, 0.5)


# ------------------------------------------------------------ metrics
def cc(pred, gaze):
    p, g = pred.ravel(), gaze.ravel()
    if p.std() < EPS or g.std() < EPS:
        return 0.0
    return float(np.corrcoef(p, g)[0, 1])


def nss(pred, gaze, top_percent=5):
    """NSS at 'fixation' pixels = top-k% of the gaze density map."""
    if pred.std() < EPS:
        return 0.0
    z = (pred - pred.mean()) / pred.std()
    fix = gaze >= np.percentile(gaze, 100 - top_percent)
    return float(z[fix].mean())


def kl(pred, gaze):
    """KL(gaze || pred) between maps normalised to probability distributions (lower is better)."""
    p = pred / (pred.sum() + EPS)
    g = gaze / (gaze.sum() + EPS)
    return float(np.sum(g * np.log(EPS + g / (p + EPS))))


def all_metrics(pred, gaze, top_percent=5):
    return {"cc": cc(pred, gaze), "nss": nss(pred, gaze, top_percent), "kl": kl(pred, gaze)}


# ------------------------------------------------------------ VLM attention
def attention_heatmap(model, processor, image, messages, answer_text, last_n_layers=8):
    """Average attention from answer tokens to image tokens over the last N layers.
    The model must be loaded with attn_implementation="eager". Returns a (h/28, w/28) grid."""
    import torch
    msgs = messages + [{"role": "assistant", "content": [{"type": "text", "text": answer_text}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    enc = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    ids = enc["input_ids"][0]
    img_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    img_pos = (ids == img_id).nonzero().squeeze(1)
    marker = processor.tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    lst, m = ids.tolist(), len(marker)
    start = next(j + m for j in range(len(lst) - m, -1, -1) if lst[j:j + m] == marker)
    att = torch.stack([a[0].float().mean(0) for a in out.attentions[-last_n_layers:]]).mean(0)
    a = att[start:, :][:, img_pos].mean(0)
    _, gh, gw = enc["image_grid_thw"][0].tolist()
    return a.reshape(gh // 2, gw // 2).cpu().numpy()   # 2x2 patches are merged into one token
