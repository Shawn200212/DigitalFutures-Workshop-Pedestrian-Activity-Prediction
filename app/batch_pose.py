# app/batch_pose.py
"""
YOLO Pose 姿态检测脚本：适配 100 tile master manifest。

替换位置：
  C:/Users/Shaw2/Downloads/svi_pedestrian_demo/app\batch_pose.py

推荐运行：
python app/batch_pose.py ^
  --manifest outputs/manifests/master_yolo_manifest_unique.csv ^
  --output-dir outputs/pose ^
  --model yolo11n-pose.pt ^
  --conf 0.08 ^
  --imgsz 960 ^
  --device 0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

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


def classify_pose(kpts: np.ndarray, conf_threshold: float = 0.30) -> str:
    """
    简单规则分类：standing / sitting / walking_like / unknown
    YOLO pose keypoints: COCO 17 点。
    注意：街景小目标姿态召回低，该字段建议用于质控，不建议作为主标签。
    """
    if kpts is None or len(kpts) < 17:
        return "unknown"

    lh, rh = kpts[11], kpts[12]
    lk, rk = kpts[13], kpts[14]
    la, ra = kpts[15], kpts[16]

    valid = [p for p in [lh, rh, lk, rk, la, ra] if len(p) >= 3 and p[2] > conf_threshold]
    if len(valid) < 4:
        return "unknown"

    hip_y = np.mean([lh[1], rh[1]])
    knee_y = np.mean([lk[1], rk[1]])
    ankle_y = np.mean([la[1], ra[1]])

    leg_height = ankle_y - hip_y
    knee_gap = abs(lk[0] - rk[0])
    ankle_gap = abs(la[0] - ra[0])

    if leg_height <= 0:
        return "unknown"

    if (knee_y - hip_y) / leg_height < 0.45:
        return "sitting"

    if ankle_gap > 45 or knee_gap > 35:
        return "walking_like"

    return "standing"


def keypoints_to_list(result) -> list:
    if result.keypoints is None:
        return []
    data = result.keypoints.data.cpu().numpy()
    return [person.tolist() for person in data]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch YOLO pose for tile manifests.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/pose")
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    ann_dir = ensure_dir(output_dir / "pose_annotated")
    keypoints_dir = ensure_dir(output_dir / "keypoints_json")

    manifest = load_manifest(args.manifest, args.image_dir)
    if args.limit:
        manifest = manifest.head(args.limit).copy()

    model = YOLO(args.model)
    records = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="YOLO pose"):
        image_path = Path(str(row["image_path"]))
        image_id = make_image_id(row, image_path)

        base = row.to_dict()
        base["image_id"] = image_id
        base["image_path"] = str(image_path)

        record = {
            **base,
            "pose_status": "missing",
            "person_pose_count": 0,
            "standing_count": 0,
            "sitting_count": 0,
            "walking_like_count": 0,
            "unknown_pose_count": 0,
            "mean_pose_conf": 0.0,
            "keypoints_json": "",
            "pose_annotated_path": "",
            "pose_error": "",
        }

        if not image_path.exists():
            record["pose_error"] = f"Image missing: {image_path}"
            records.append(record)
            continue

        try:
            results = model.predict(
                source=str(image_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            result = results[0]
            people_kpts = keypoints_to_list(result)
            record["person_pose_count"] = len(people_kpts)

            labels = []
            confs = []
            for kpts in people_kpts:
                arr = np.array(kpts)
                label = classify_pose(arr)
                labels.append(label)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    valid_conf = arr[:, 2]
                    valid_conf = valid_conf[valid_conf > 0]
                    if len(valid_conf) > 0:
                        confs.append(float(np.mean(valid_conf)))

            record["standing_count"] = labels.count("standing")
            record["sitting_count"] = labels.count("sitting")
            record["walking_like_count"] = labels.count("walking_like")
            record["unknown_pose_count"] = labels.count("unknown")
            record["mean_pose_conf"] = float(np.mean(confs)) if confs else 0.0

            kp_payload = {"people": people_kpts, "pose_labels": labels}
            kp_path = keypoints_dir / f"{image_id}.json"
            with open(kp_path, "w", encoding="utf-8") as f:
                json.dump(kp_payload, f, ensure_ascii=False, indent=2)
            record["keypoints_json"] = str(kp_path)
            record["pose_status"] = "ok"

            if args.save_vis:
                plotted = result.plot()
                save_path = ann_dir / f"{image_id}_pose.jpg"
                cv2.imwrite(str(save_path), plotted)
                record["pose_annotated_path"] = str(save_path)

        except Exception as e:
            record["pose_status"] = "error"
            record["pose_error"] = repr(e)

        records.append(record)

    out = pd.DataFrame(records)
    out_path = output_dir / "pose_detections.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = {
        "n_images": int(len(out)),
        "n_ok": int((out["pose_status"] == "ok").sum()),
        "n_missing": int((out["pose_status"] == "missing").sum()),
        "n_error": int((out["pose_status"] == "error").sum()),
        "total_pose_person_count": int(pd.to_numeric(out["person_pose_count"], errors="coerce").fillna(0).sum()),
        "output": str(out_path),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
