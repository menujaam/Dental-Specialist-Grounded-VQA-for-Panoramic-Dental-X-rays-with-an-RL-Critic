"""Train a DENTEX detector (specialist tool whose confidences feed the RL critic).

  --task disease      4 classes (caries, deep caries, periapical lesion, impacted)  [default]
  --task enumeration  32 classes (FDI teeth 11-48)
Requires: pip install ultralytics   (AGPL-3.0 licence)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import ALL_TEETH, DIAGNOSES, load_config, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dental_config.yaml")
    ap.add_argument("--task", choices=["disease", "enumeration"], default="disease")
    ap.add_argument("--weights", default="yolov8m.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--prepare_only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = Path(cfg["paths"]["processed"])
    names = DIAGNOSES if args.task == "disease" else [str(t) for t in ALL_TEETH]
    cls = {n: i for i, n in enumerate(names)}
    src, field = ("dentex_disease", "findings") if args.task == "disease" else ("dentex_enumeration", "teeth")
    yroot = (proc / f"yolo_dentex_{args.task}").resolve()

    for split in ("train", "val", "test"):
        recs = [r for r in read_jsonl(proc / "splits" / f"{split}.jsonl") if r["source"] == src]
        (yroot / "images" / split).mkdir(parents=True, exist_ok=True)
        (yroot / "labels" / split).mkdir(parents=True, exist_ok=True)
        for r in recs:
            img = yroot / "images" / split / Path(r["image_path"]).name
            if not img.exists():
                img.symlink_to(r["image_path"])
            W, H = r["width"], r["height"]
            lines = []
            for it in r[field]:
                b = it["box"]
                c = cls[it["diagnosis"]] if args.task == "disease" else cls[str(it["tooth"])]
                lines.append(f"{c} {(b[0]+b[2])/2/W:.6f} {(b[1]+b[3])/2/H:.6f} {(b[2]-b[0])/W:.6f} {(b[3]-b[1])/H:.6f}")
            (yroot / "labels" / split / (img.stem + ".txt")).write_text("\n".join(lines))
        print(f"{split}: {len(recs)} images")

    data_yaml = yroot / "data.yaml"
    data_yaml.write_text(f"path: {yroot}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                         + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    if args.prepare_only:
        return
    from ultralytics import YOLO
    model = YOLO(args.weights)
    # no left-right flips: they would swap quadrants 1<->2 and 3<->4 and corrupt tooth labels
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, project="outputs",
                name=f"dentex_{args.task}_yolo", fliplr=0.0, mosaic=0.5, degrees=3)
    print(model.val(data=str(data_yaml), split="test"))


if __name__ == "__main__":
    main()
