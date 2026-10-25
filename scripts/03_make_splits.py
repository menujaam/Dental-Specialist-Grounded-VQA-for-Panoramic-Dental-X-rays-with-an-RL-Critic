"""Stratified (or patient-grouped) train/val/test splits per dataset.

Writes <processed>/splits/{train,val,test}.jsonl (records with a "split" field).
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import load_config, read_jsonl, write_jsonl  # noqa: E402


def strat_key(r):
    dx = "+".join(sorted({f["diagnosis"] for f in r["findings"]})) or "none"
    abn = "abn" if r.get("abnormal_boxes") else "noabn"
    return f"{r['source']}|{dx}|{abn}"


def safe_keys(recs, min_count=6):
    keys = [strat_key(r) for r in recs]
    c = Counter(keys)
    # merge rare strata so sklearn can stratify, falling back to the source only
    keys = [k if c[k] >= min_count else k.split("|")[0] for k in keys]
    c = Counter(keys)
    return [k if c[k] >= 2 else "rare" for k in keys]


def stratified(recs, fr, seed):
    keys = safe_keys(recs)
    tr, rest, _, k_rest = train_test_split(recs, keys, train_size=fr["train"], stratify=keys, random_state=seed)
    val_share = fr["val"] / (fr["val"] + fr["test"])
    c = Counter(k_rest)
    k_rest = [k if c[k] >= 2 else "rare" for k in k_rest]
    try:
        va, te = train_test_split(rest, train_size=val_share, stratify=k_rest, random_state=seed)
    except ValueError:
        va, te = train_test_split(rest, train_size=val_share, random_state=seed)
    return tr, va, te


def grouped(recs, fr, seed):
    groups = sorted({r["group"] for r in recs})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    a, b = int(n * fr["train"]), int(n * (fr["train"] + fr["val"]))
    gsplit = {g: ("train" if i < a else "val" if i < b else "test") for i, g in enumerate(groups)}
    out = {"train": [], "val": [], "test": []}
    for r in recs:
        out[gsplit[r["group"]]].append(r)
    return out["train"], out["val"], out["test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dental_config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = Path(cfg["paths"]["processed"])
    fr, seed = cfg["splits"], cfg["splits"]["seed"]

    splits = {"train": [], "val": [], "test": []}
    for src in ("dentex", "tufts"):
        p = proc / f"records_{src}.jsonl"
        if not p.exists():
            print(f"  skipping {src}: {p} not found")
            continue
        recs = read_jsonl(p)
        has_groups = len({r["group"] for r in recs}) < len(recs)
        tr, va, te = (grouped if has_groups else stratified)(recs, fr, seed)
        print(f"{src}: {'patient-grouped' if has_groups else 'stratified'} -> "
              f"train {len(tr)} / val {len(va)} / test {len(te)}")
        for name, part in (("train", tr), ("val", va), ("test", te)):
            for r in part:
                r["split"] = name
            splits[name] += part
            print(f"    {name:5s}", dict(Counter(r['source'] for r in part)))

    for name, part in splits.items():
        write_jsonl(proc / "splits" / f"{name}.jsonl", part)


if __name__ == "__main__":
    main()
