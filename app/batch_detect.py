# app/batch_detect.py
"""
YOLO 行人检测脚本：适配 100 tile master manifest。

替换位置：
  C:/Users/Shaw2/Downloads/svi_pedestrian_demo/app\batch_detect.py

输入支持：
1. --manifest outputs/manifests/master_yolo_manifest_unique.csv
   manifest 中可以是 image_path 或 save_path。
2. --image-dir data/images
   如果不传 manifest，则递归扫描图片。

输出：
  outputs/yolo/detections.csv
  outputs/yolo/boxes_json/*.json
  outputs/yolo/annotated/*.jpg，可选

推荐运行：
python app/batch_detect.py ^
  --manifest outputs/manifests/master_yolo_manifest_unique.csv ^
  --output-dir outputs/yolo ^
  --model yolo11n.pt ^
  --conf 0.25 ^
  --imgsz 960 ^
  --device 0 ^
  --save-vis
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_direction(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    try:
        return f"{int(float(value)):03d}"
    except Exception:
        return str(value)


def normalize_month(value) -> str:
    if pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    if len(digits) >= 6:
        return f"{digits[:4]}-{digits[4:6]}"
    return str(value)


def make_image_id(row: pd.Series, image_path: Path) -> str:
    if "image_id" in row and not pd.isna(row["image_id"]) and str(row["image_id"]).strip():
        return str(row["image_id"])

    tile_id = str(row.get("tile_id", "") or "")
    iid = str(row.get("iid", "") or "")
    actual = normalize_month(row.get("actual_date", row.get("actual_month", ""))).replace("-", "_")
    direction = normalize_direction(row.get("direction", row.get("heading", "")))
    if tile_id and iid and actual and direction:
        return f"{tile_id}_iid_{iid}_date_{actual}_dir_{direction}"

    return image_path.stem


def load_manifest(manifest_path: str | Path | None, image_dir: str | Path | None) -> pd.DataFrame:
    if manifest_path:
        df = pd.read_csv(manifest_path, encoding="utf-8-sig")
        if "image_path" not in df.columns:
            if "save_path" not in df.columns:
                raise ValueError("manifest 必须包含 image_path 或 save_path")
            df["image_path"] = df["save_path"].astype(str)
        df = df[df["image_path"].notna()].copy()
        df = df[df["image_path"].astype(str).str.len() > 0].copy()
        return df.reset_index(drop=True)

    if not image_dir:
        raise ValueError("--manifest 和 --image-dir 至少提供一个")

    root = Path(image_dir)
    paths = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES])
    return pd.DataFrame({"image_path": [str(p) for p in paths]})


def box_to_dict(xyxy, conf, cls_id) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "conf": float(conf),
        "cls": int(cls_id),
        "width": float(max(0.0, x2 - x1)),
        "height": float(max(0.0, y2 - y1)),
        "area": float(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
    }


def save_visualization(image_path: Path, boxes: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")

    for b in boxes:
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        conf = float(b["conf"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"person {conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), img)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch YOLO person detection for tile manifests.")
    parser.add_argument("--manifest", default=None, help="推荐：outputs/manifests/master_yolo_manifest_unique.csv")
    parser.add_argument("--image-dir", default=None, help="不传 manifest 时递归扫描该目录")
    parser.add_argument("--output-dir", default="outputs/yolo")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--dense-threshold", type=int, default=8, help="街景中 >=8 人建议复核")
    parser.add_argument("--low-conf-threshold", type=float, default=0.50)
    parser.add_argument("--limit", type=int, default=None, help="调试时只跑前 N 张")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = ensure_dir(args.output_dir)
    boxes_dir = ensure_dir(output_dir / "boxes_json")
    ann_dir = ensure_dir(output_dir / "annotated")

    manifest = load_manifest(args.manifest, args.image_dir)
    if args.limit:
        manifest = manifest.head(args.limit).copy()

    model = YOLO(args.model)
    rows = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="YOLO detecting"):
        image_path = Path(str(row["image_path"]))
        image_id = make_image_id(row, image_path)

        base_record = row.to_dict()
        base_record["image_id"] = image_id
        base_record["image_path"] = str(image_path)

        if not image_path.exists():
            rows.append({
                **base_record,
                "status": "missing",
                "person_count_yolo": None,
                "person_count_final": None,
                "mean_conf": None,
                "min_conf": None,
                "max_conf": None,
                "dense_flag": False,
                "low_conf_flag": False,
                "need_review": True,
                "boxes_json": "",
                "annotated_path": "",
                "error": f"Image missing: {image_path}",
            })
            continue

        try:
            results = model.predict(
                source=str(image_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=[0],
                device=args.device,
                verbose=False,
            )
            result = results[0]
            boxes = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)
                for b, c, k in zip(xyxy, confs, clss):
                    boxes.append(box_to_dict(b, c, k))

            person_count = len(boxes)
            conf_list = [float(b["conf"]) for b in boxes]
            mean_conf = float(np.mean(conf_list)) if conf_list else 0.0
            min_conf = float(np.min(conf_list)) if conf_list else 0.0
            max_conf = float(np.max(conf_list)) if conf_list else 0.0

            dense_flag = person_count >= args.dense_threshold
            low_conf_flag = person_count > 0 and mean_conf < args.low_conf_threshold
            need_review = bool(dense_flag or low_conf_flag)

            boxes_path = boxes_dir / f"{image_id}.json"
            with open(boxes_path, "w", encoding="utf-8") as f:
                json.dump(boxes, f, ensure_ascii=False, indent=2)

            annotated_path = ""
            if args.save_vis:
                annotated_path = str(ann_dir / f"{image_id}_annotated.jpg")
                save_visualization(image_path, boxes, Path(annotated_path))

            rows.append({
                **base_record,
                "status": "ok",
                "person_count_yolo": person_count,
                "person_count_final": person_count,
                "mean_conf": mean_conf,
                "min_conf": min_conf,
                "max_conf": max_conf,
                "dense_flag": dense_flag,
                "low_conf_flag": low_conf_flag,
                "need_review": need_review,
                "boxes_json": str(boxes_path),
                "annotated_path": annotated_path,
                "error": "",
            })

        except Exception as e:
            rows.append({
                **base_record,
                "status": "error",
                "person_count_yolo": None,
                "person_count_final": None,
                "mean_conf": None,
                "min_conf": None,
                "max_conf": None,
                "dense_flag": False,
                "low_conf_flag": False,
                "need_review": True,
                "boxes_json": "",
                "annotated_path": "",
                "error": repr(e),
            })

    out = pd.DataFrame(rows)
    out_path = output_dir / "detections.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = {
        "n_images": int(len(out)),
        "n_ok": int((out["status"] == "ok").sum()),
        "n_missing": int((out["status"] == "missing").sum()),
        "n_error": int((out["status"] == "error").sum()),
        "total_person_count": int(pd.to_numeric(out["person_count_final"], errors="coerce").fillna(0).sum()),
        "n_need_review": int(out["need_review"].fillna(False).sum()),
        "output": str(out_path),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
