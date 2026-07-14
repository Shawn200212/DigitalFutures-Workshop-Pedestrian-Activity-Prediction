# scripts/prepare_yolo_manifest_from_tiles.py
"""
从 100 个 tile 爬取结果中生成 YOLO/VLM 可直接使用的唯一图片 manifest。

适配你现在的目录结构：
DATASET_ROOT/
  tile001/
    tile001_flat_image_manifest_for_training.csv
  tile002/
    tile002_flat_image_manifest_for_training.csv
  ...
  MASTER_flat_image_manifest_for_training.csv

核心功能：
1. 自动读取 MASTER manifest；如果没有，则扫描 tile*/tile*_flat_image_manifest_for_training.csv。
2. 只保留四方向 flat 图，不保留 panorama。
3. 将 save_path 统一重命名/复制为 image_path。
4. 去重：同一个 save_path / tile_id+iid+actual_date+direction 只保留一条，避免同一实际图片因多个 target_date 重复进入 YOLO。
5. 生成稳定 image_id，保留 tile_id、iid、lon、lat、actual_date、target_dates、direction、source_csv 等字段。
6. 输出 summary JSON，告诉你总量、重复量、各 tile 图片量。

运行：
python scripts/prepare_yolo_manifest_from_tiles.py ^
  --dataset-root data/Images ^
  --output outputs/manifests/master_yolo_manifest_unique.csv ^
  --summary outputs/manifests/master_yolo_manifest_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def read_csv_auto(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def normalize_month(value) -> str:
    if pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    if len(digits) >= 6:
        return f"{digits[:4]}-{digits[4:6]}"
    if len(digits) >= 4:
        return digits[:4]
    return str(value)


def normalize_direction(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    try:
        return f"{int(float(value)):03d}"
    except Exception:
        return str(value)


def tile_sort_key(path: Path):
    m = re.search(r"tile[_-]?(\d+)", str(path), re.IGNORECASE)
    return int(m.group(1)) if m else 10**12


def infer_tile_id(row) -> str:
    if "tile_id" in row and not pd.isna(row["tile_id"]) and str(row["tile_id"]).strip():
        return str(row["tile_id"]).strip()

    for col in ["source_csv_name", "source_csv_path", "save_path", "image_path"]:
        if col in row and not pd.isna(row[col]):
            m = re.search(r"tile[_-]?(\d+)", str(row[col]), re.IGNORECASE)
            if m:
                return f"tile{int(m.group(1)):03d}"
    return "tile_unknown"


def make_image_id(row) -> str:
    tile_id = str(row.get("tile_id", "tile_unknown"))
    iid = str(row.get("iid", ""))
    actual_month = str(row.get("actual_month", "unknown")).replace("-", "_")
    direction = normalize_direction(row.get("direction", ""))
    lon = ""
    lat = ""
    try:
        lon = f"{float(row.get('lon')):.8f}"
        lat = f"{float(row.get('lat')):.8f}"
    except Exception:
        pass

    parts = [tile_id, f"iid_{iid}", f"date_{actual_month}"]
    if lon and lat:
        parts += [f"lon_{lon}", f"lat_{lat}"]
    if direction:
        parts.append(f"dir_{direction}")
    return "_".join(parts)


def choose_source_manifest(dataset_root: Path, master_name: str) -> tuple[list[Path], str]:
    master = dataset_root / master_name
    if master.exists() and master.stat().st_size > 0:
        return [master], "master"

    files = sorted(
        dataset_root.glob("tile*/tile*_flat_image_manifest_for_training.csv"),
        key=tile_sort_key,
    )
    if not files:
        raise FileNotFoundError(
            f"没有找到 {master}，也没有找到 tile*/tile*_flat_image_manifest_for_training.csv"
        )
    return files, "tile_scan"


def build_manifest(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    dataset_root = Path(args.dataset_root)
    manifest_files, source_mode = choose_source_manifest(dataset_root, args.master_manifest_name)

    dfs = []
    for path in manifest_files:
        df = read_csv_auto(path)
        df["_manifest_source_file"] = str(path)
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if raw.empty:
        raise ValueError("输入 manifest 为空")

    df = raw.copy()

    if "image_path" not in df.columns:
        if "save_path" not in df.columns:
            raise ValueError("manifest 必须包含 save_path 或 image_path")
        df["image_path"] = df["save_path"].astype(str)
    else:
        df["image_path"] = df["image_path"].astype(str)

    # 成功图片
    if "success" in df.columns:
        df = df[truthy(df["success"])].copy()

    # 只保留 flat
    if "is_pano" in df.columns:
        df = df[~truthy(df["is_pano"])].copy()
    elif "image_type" in df.columns:
        df = df[df["image_type"].astype(str).str.lower().eq("flat")].copy()

    df = df[df["image_path"].notna()]
    df = df[df["image_path"].astype(str).str.len() > 0].copy()

    # 不强制检查文件存在：因为你可能在 Windows 本地跑，路径在 sandbox/Linux 不存在。
    # 在 Windows 项目里运行时，可加 --check-exists。
    if args.check_exists:
        before = len(df)
        df = df[df["image_path"].map(lambda p: Path(str(p)).exists())].copy()
        missing = before - len(df)
    else:
        missing = None

    # 规范字段
    df["tile_id"] = df.apply(infer_tile_id, axis=1)
    if "actual_date" in df.columns:
        df["actual_month"] = df["actual_date"].apply(normalize_month)
    else:
        df["actual_month"] = ""

    if "target_date" in df.columns:
        df["target_month"] = df["target_date"].apply(normalize_month)
    else:
        df["target_month"] = ""

    if "direction" in df.columns:
        df["heading"] = df["direction"].apply(normalize_direction)
    elif "heading" in df.columns:
        df["heading"] = df["heading"].apply(normalize_direction)
    else:
        df["heading"] = ""

    # 聚合同一实际图片对应的多个 target_date
    group_key_cols = ["image_path"]
    if args.dedupe_key == "semantic":
        semantic_cols = ["tile_id", "iid", "actual_month", "lon", "lat", "heading"]
        group_key_cols = [c for c in semantic_cols if c in df.columns]

    # 先保存 target_month 列表
    if "target_month" in df.columns:
        target_lists = (
            df.groupby(group_key_cols, dropna=False)["target_month"]
            .apply(lambda s: ";".join(sorted(set([x for x in s.astype(str) if x and x != "nan"]))))
            .reset_index()
            .rename(columns={"target_month": "target_months_joined"})
        )
    else:
        target_lists = None

    # 排序后去重：优先保留 status=saved，其次 skipped_resume_existing，其次 skipped_existing
    status_rank = {
        "saved": 0,
        "skipped_resume_existing": 1,
        "skipped_existing": 2,
        "skipped_existing_after_request": 2,
    }
    if "status" in df.columns:
        df["_status_rank"] = df["status"].astype(str).map(status_rank).fillna(9)
    else:
        df["_status_rank"] = 9

    sort_cols = group_key_cols + ["_status_rank"]
    df = df.sort_values(sort_cols, ascending=True)
    unique = df.drop_duplicates(subset=group_key_cols, keep="first").copy()
    unique = unique.drop(columns=["_status_rank"], errors="ignore")

    if target_lists is not None:
        unique = unique.merge(target_lists, on=group_key_cols, how="left")

    unique["image_id"] = unique.apply(make_image_id, axis=1)

    # image_id 仍可能重复，追加序号兜底
    dup = unique["image_id"].duplicated(keep=False)
    if dup.any():
        unique.loc[dup, "image_id"] = (
            unique.loc[dup, "image_id"] + "_n" +
            unique.groupby("image_id").cumcount().astype(str).str.zfill(3)
        )

    # YOLO/VLM 必需列靠前
    preferred = [
        "image_id", "image_path", "tile_id", "iid", "lon", "lat",
        "actual_date", "actual_month", "target_date", "target_month",
        "target_months_joined", "direction", "heading",
        "road_seg_id", "source_csv_name", "source_csv_path",
        "filename", "save_path", "udt_sid", "status",
    ]
    preferred = [c for c in preferred if c in unique.columns]
    other = [c for c in unique.columns if c not in preferred]
    unique = unique[preferred + other]

    summary = {
        "dataset_root": str(dataset_root),
        "source_mode": source_mode,
        "manifest_files": [str(p) for p in manifest_files],
        "raw_rows": int(len(raw)),
        "after_filter_rows": int(len(df)),
        "unique_rows": int(len(unique)),
        "duplicate_rows_removed": int(len(df) - len(unique)),
        "dedupe_key": args.dedupe_key,
        "missing_removed_by_check_exists": missing,
        "n_tiles": int(unique["tile_id"].nunique()) if "tile_id" in unique.columns else None,
        "rows_by_tile": unique["tile_id"].value_counts().sort_index().to_dict() if "tile_id" in unique.columns else {},
        "rows_by_actual_month": unique["actual_month"].value_counts().sort_index().to_dict() if "actual_month" in unique.columns else {},
        "rows_by_heading": unique["heading"].value_counts().sort_index().to_dict() if "heading" in unique.columns else {},
    }
    return unique, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="data/Images 或包含 tile001...tile100 的目录")
    parser.add_argument("--master-manifest-name", default="MASTER_flat_image_manifest_for_training.csv")
    parser.add_argument("--output", default="outputs/manifests/master_yolo_manifest_unique.csv")
    parser.add_argument("--summary", default="outputs/manifests/master_yolo_manifest_summary.json")
    parser.add_argument("--dedupe-key", choices=["path", "semantic"], default="path")
    parser.add_argument("--check-exists", action="store_true", help="在当前机器检查 image_path 是否真实存在，不存在则剔除")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df, summary = build_manifest(args)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"YOLO/VLM manifest saved: {out}")
    print(f"summary saved: {summary_path}")


if __name__ == "__main__":
    main()
