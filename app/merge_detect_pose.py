# merge_detect_pose.py
"""
修正版：合并 detections.csv 与 pose_detections.csv。

核心修正：
1. 合并前先按 image_id 去重，避免 detections 与 pose 两侧都有重复 image_id 时发生多对多 merge，导致行数膨胀。
2. 自动补齐姿态字段，计算 standing/sitting/walking_like/unknown 比例。
3. 增加 pose_detect_gap 与 pose_recall_proxy，用于判断姿态模型相对人数检测的召回情况。

用法：
python merge_detect_pose.py \
  --detections outputs/detections.csv \
  --poses outputs/pose_detections.csv \
  --output outputs/detections_with_pose.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


POSE_COUNT_COLS = [
    "person_pose_count",
    "standing_count",
    "sitting_count",
    "walking_like_count",
    "unknown_pose_count",
    "mean_pose_conf",
]


def safe_div(a, b) -> float:
    if pd.isna(b) or float(b) == 0:
        return 0.0
    return float(a) / float(b)


def read_csv_checked(path: str | Path, table_name: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{table_name} 不存在：{path}")

    df = pd.read_csv(path)
    if "image_id" not in df.columns:
        raise ValueError(f"{table_name} 中必须包含 image_id 列")

    df["image_id"] = df["image_id"].astype(str)
    return df


def dedupe_by_image_id(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    df = df.copy()

    duplicate_col = f"{table_name}_duplicate_rows"
    duplicate_counts = (
        df.groupby("image_id", as_index=False)
        .size()
        .rename(columns={"size": duplicate_col})
    )

    status_col = "status" if table_name == "det" else "pose_status"
    if status_col in df.columns:
        df["_ok_rank"] = (df[status_col].astype(str).str.lower() == "ok").astype(int)
    else:
        df["_ok_rank"] = 0

    sort_cols = ["image_id", "_ok_rank"]
    ascending = [True, False]

    if table_name == "det" and "mean_conf" in df.columns:
        sort_cols.append("mean_conf")
        ascending.append(False)

    if table_name == "pose" and "mean_pose_conf" in df.columns:
        sort_cols.append("mean_pose_conf")
        ascending.append(False)

    df = df.sort_values(sort_cols, ascending=ascending)

    deduped = (
        df.drop_duplicates(subset=["image_id"], keep="first")
        .drop(columns=["_ok_rank"], errors="ignore")
        .merge(duplicate_counts, on="image_id", how="left")
    )

    return deduped


def normalize_count_columns(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()

    for col in POSE_COUNT_COLS:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    if "person_count_final" not in merged.columns:
        if "person_count_yolo" in merged.columns:
            merged["person_count_final"] = merged["person_count_yolo"]
        else:
            merged["person_count_final"] = 0

    merged["person_count_final"] = pd.to_numeric(
        merged["person_count_final"], errors="coerce"
    ).fillna(0)

    return merged


def add_pose_quality_fields(merged: pd.DataFrame) -> pd.DataFrame:
    merged = normalize_count_columns(merged)

    merged["standing_ratio"] = merged.apply(
        lambda r: safe_div(r["standing_count"], r["person_pose_count"]), axis=1
    )
    merged["sitting_ratio"] = merged.apply(
        lambda r: safe_div(r["sitting_count"], r["person_pose_count"]), axis=1
    )
    merged["walking_like_ratio"] = merged.apply(
        lambda r: safe_div(r["walking_like_count"], r["person_pose_count"]), axis=1
    )
    merged["unknown_pose_ratio"] = merged.apply(
        lambda r: safe_div(r["unknown_pose_count"], r["person_pose_count"]), axis=1
    )

    merged["pose_has_result"] = merged["person_pose_count"] > 0
    merged["pose_detect_gap"] = (
        merged["person_count_final"].fillna(0) - merged["person_pose_count"].fillna(0)
    )

    merged["pose_recall_proxy"] = np.where(
        merged["person_count_final"].fillna(0) > 0,
        merged["person_pose_count"].fillna(0) / merged["person_count_final"].fillna(0),
        np.nan,
    )

    return merged


def build_review_flags(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()

    if "need_review" not in merged.columns:
        merged["need_review"] = False

    merged["review_reason"] = ""

    if "mean_conf" in merged.columns:
        low_conf = (merged["person_count_final"] > 0) & (
            pd.to_numeric(merged["mean_conf"], errors="coerce").fillna(0) < 0.5
        )
        merged.loc[low_conf, "need_review"] = True
        merged.loc[low_conf, "review_reason"] += "det_low_conf;"

    possible_overcount = merged["person_count_final"] >= 8
    merged.loc[possible_overcount, "need_review"] = True
    merged.loc[possible_overcount, "review_reason"] += "possible_overcount;"

    pose_missing = (merged["person_count_final"] > 0) & (merged["person_pose_count"] == 0)
    merged.loc[pose_missing, "review_reason"] += "pose_missing;"

    pose_low_recall = (merged["person_count_final"] >= 3) & (
        merged["pose_recall_proxy"].fillna(0) < 0.5
    )
    merged.loc[pose_low_recall, "review_reason"] += "pose_low_recall;"

    merged["review_reason"] = merged["review_reason"].str.strip(";")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge detections.csv and pose_detections.csv with image_id deduplication."
    )
    parser.add_argument("--detections", required=True, help="detections.csv 路径")
    parser.add_argument("--poses", required=True, help="pose_detections.csv 路径")
    parser.add_argument(
        "--output",
        default="outputs/detections_with_pose.csv",
        help="输出合并后的 CSV 路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    det_raw = read_csv_checked(args.detections, "detections.csv")
    pose_raw = read_csv_checked(args.poses, "pose_detections.csv")

    det = dedupe_by_image_id(det_raw, "det")
    pose = dedupe_by_image_id(pose_raw, "pose")

    merged = det.merge(
        pose,
        on="image_id",
        how="left",
        suffixes=("", "_pose"),
    )

    merged = add_pose_quality_fields(merged)
    merged = build_review_flags(merged)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "raw_detection_rows": int(len(det_raw)),
        "raw_pose_rows": int(len(pose_raw)),
        "unique_detection_images": int(det["image_id"].nunique()),
        "unique_pose_images": int(pose["image_id"].nunique()),
        "merged_rows": int(len(merged)),
        "detection_duplicate_image_ids": int((det["det_duplicate_rows"] > 1).sum()),
        "pose_duplicate_image_ids": int((pose["pose_duplicate_rows"] > 1).sum()),
        "n_need_review": int(merged["need_review"].fillna(False).sum()),
        "output": str(output_path),
    }

    print(summary)


if __name__ == "__main__":
    main()