"""Score prediction files; image-reliance gap from blind / swap runs.

  python scripts/07_metrics.py --preds preds/ft_image.jsonl --blind preds/ft_blind.jsonl \
      --swap preds/ft_swap.jsonl --out results/ft.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import load_config, read_jsonl  # noqa: E402
from dental.metrics import compute_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dental_config.yaml")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--blind")
    ap.add_argument("--swap")
    ap.add_argument("--out")
    args = ap.parse_args()
    acs = load_config(args.config).get("acs", {})

    res = {"main": compute_metrics(read_jsonl(args.preds), acs)}
    for name, path in (("blind", args.blind), ("swap", args.swap)):
        if path:
            res[name] = compute_metrics(read_jsonl(path), acs)
            a, b = res["main"]["closed_accuracy"], res[name]["closed_accuracy"]
            if a is not None and b is not None:
                res[f"image_reliance_gap_{name}"] = a - b

    rows = [(qt, k, v) for qt, m in res["main"]["per_qtype"].items() for k, v in m.items() if k != "n"]
    print(f"\n== {args.preds}  (n={res['main']['n']})")
    print(pd.DataFrame(rows, columns=["qtype", "metric", "value"]).to_string(index=False))
    print("ACS:", res["main"]["acs"])
    print("accuracy by level:", res["main"]["accuracy_by_level"])
    print("closed accuracy:", res["main"]["closed_accuracy"])
    for k in ("image_reliance_gap_blind", "image_reliance_gap_swap"):
        if k in res:
            print(f"{k}: {res[k]:+.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
