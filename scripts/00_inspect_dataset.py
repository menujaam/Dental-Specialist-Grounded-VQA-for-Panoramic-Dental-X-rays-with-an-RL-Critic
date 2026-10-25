"""Print the folder structure and JSON keys of a downloaded dataset, so the config paths
can be checked before preparing data.

  python scripts/00_inspect_dataset.py data/raw/TuftsDental
  python scripts/00_inspect_dataset.py data/raw/DENTEX
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main(root):
    root = Path(root)
    by_dir = defaultdict(Counter)
    for p in root.rglob("*"):
        if p.is_file():
            by_dir[str(p.parent.relative_to(root))][p.suffix.lower() or "<none>"] += 1
    print(f"== folders under {root}")
    for d in sorted(by_dir):
        print(f"  {d:60s} {dict(by_dir[d])}")
        examples = sorted((root / d).iterdir())[:3]
        print("      e.g.", [e.name for e in examples if e.is_file()])

    for jp in sorted(root.rglob("*.json"))[:10]:
        print(f"\n== {jp.relative_to(root)}")
        try:
            d = json.load(open(jp))
        except Exception as e:
            print("   could not read:", e)
            continue
        if isinstance(d, dict):
            print("   top-level keys:", list(d)[:20])
            for k, v in d.items():
                if isinstance(v, list) and v:
                    print(f"   {k}[0] =", json.dumps(v[0])[:400])
        elif isinstance(d, list) and d:
            print(f"   list of {len(d)}; first item:", json.dumps(d[0])[:800])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
