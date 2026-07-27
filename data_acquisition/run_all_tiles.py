"""
run_all_tiles.py

用途：
    自动顺序运行当前项目目录下所有 tile_*.csv。
    每一个 tile 单独输出到自己的文件夹中，并且每完成一个 tile，
    就立刻刷新一次 MASTER 总控文件，避免必须等全部 tile 跑完才有总表。

输出结构：
    DATASET_ROOT/
      tile001/
        actual_YYYY_MM/
        panorama_YYYY_MM/
        crawl_log_flat_actualdate_training_review.csv
        tile001_crawl_log.csv
        tile001_flat_image_manifest_for_training.csv

      tile002/
        ...

      MASTER_batch_summary.csv
      MASTER_crawl_log_all_tiles.csv
      MASTER_flat_image_manifest_for_training.csv
      MASTER_progress.json

核心特点：
    1. 不改变 flat_images_pano.py 的图片写出逻辑；
    2. 自动按 tile_001, tile_002, tile_003 ... 的顺序运行；
    3. 每个 tile 单独输出到 tile001、tile002 目录；
    4. 每个 tile 完成 / 跳过 / 报错后，立即刷新 MASTER 文件；
    5. 中途 Ctrl+C 时，也会尽量刷新当前已经完成的 MASTER；
    6. MASTER 会从 DATASET_ROOT/tile*/tile*_crawl_log.csv 重新扫描生成，
       因此即使之前中断过，也能基于已有 tile 日志重建总表。

运行方式：
    把本脚本保存为 run_all_tiles.py，
    放到和 flat_images_pano.py、tile_001.csv、tile_002.csv 同一个目录，
    然后运行：

        python run_all_tiles.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import flat_images_pano as fp


# ============================================================
# 1. 你主要修改这里
# ============================================================

# CSV 所在目录。
# 如果本脚本就在 data_acquisition/ 目录下，保持默认即可。
PROJECT_DIR = Path(__file__).resolve().parent

# 自动匹配 tiles/tile_001.csv, tiles/tile_002.csv, ...
TILE_PATTERN = "tiles/tile_*.csv"

# 总输出根目录。
# 每个 tile 会在这个目录下生成独立文件夹：
#   DATASET_ROOT/tile001/
#   DATASET_ROOT/tile002/
#
# 默认写到仓库根目录下的 data/images/（已被 .gitignore 排除，不会入库）。
# 也可以用环境变量覆盖：set SVI_DATA_ROOT=D:/somewhere/images
DATASET_ROOT = Path(
    os.environ.get(
        "SVI_DATA_ROOT",
        PROJECT_DIR.parent / "data" / "images",
    )
)

# 某个 tile 报错后是否继续跑后面的 tile。
CONTINUE_ON_ERROR = True

# 是否跳过已经跑过的 tile。
# 判断标准：
#   DATASET_ROOT/tile001/tile001_crawl_log.csv 存在且非空。
#
# 建议：
#   如果你是第一次跑全量，可以设为 False。
#   如果你是中断后续跑 / 只想重建 MASTER，建议设为 True。
SKIP_FINISHED_TILE_LOG = True

# 每完成一个 tile 是否立刻刷新 MASTER。
# 这是本版新增的核心开关。
REFRESH_MASTER_AFTER_EACH_TILE = True

# MASTER 刷新时是否从 DATASET_ROOT 下所有已存在 tile 日志重新扫描。
# 建议保持 True。这样即使中途断过，也能把已有 tile001/tile002... 重新汇总。
REBUILD_MASTER_FROM_EXISTING_TILE_LOGS = True

# 每次刷新 MASTER 时是否打印统计信息。
PRINT_MASTER_REFRESH_SUMMARY = True


# ============================================================
# 2. flat_images_pano.py 参数统一配置
#    这里会覆盖 flat_images_pano.py 文件顶部的默认值
# ============================================================

TARGET_DATES = [
    "201703", "201706", "201709", "201712",
    "201803", "201806", "201809", "201812",
    "201903", "201906", "201909", "201912",
]

DATE_MODE = "nearest"

Z_LEVEL = 3
FLAT_IMG_SIZE = 720
FLAT_FOV_X = 90
DIRECTIONS = [0, 90, 180, 270]
YAW_OFFSET = 90

SAVE_PANORAMA = True

FILENAME_STYLE = "compact"
COORD_DECIMALS = 8

# 每个 tile 独立文件夹后，resume 可以安全开启。
# duplicate_policy 仍建议 skip，防止同一 tile 内重复经纬度重复写出。
DUPLICATE_POLICY = "skip"
ENABLE_RESUME = True

SORT_POINTS_BY_LON_LAT = True
VERBOSE = True
LOG_FLUSH_EVERY = 5
N_TRIES = 3


# ============================================================
# 3. 工具函数
# ============================================================

def tile_number(path: Path) -> int:
    """
    从 tile_001.csv 或 tile001_crawl_log.csv 提取数字 1，用于自然排序。
    """
    m = re.search(r"tile[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if not m:
        return 10**12
    return int(m.group(1))


def tile_folder_name(path: Path) -> str:
    """
    把 tile_001.csv 标准化成 tile001。
    把 tile_061.csv 标准化成 tile061。
    """
    n = tile_number(path)
    if n == 10**12:
        return path.stem.lower().replace("_", "")
    return f"tile{n:03d}"


def find_tile_csvs(project_dir: Path, pattern: str):
    return sorted(project_dir.glob(pattern), key=lambda p: (tile_number(p), p.name))


def configure_flat_images_pano(input_csv: Path, out_dir: Path):
    """
    配置 flat_images_pano.py 中的全局参数。
    不修改 flat_images_pano.py 的写图逻辑，只通过参数驱动它。
    """
    fp.input_csv = str(input_csv)
    fp.out_dir = fp.Path(str(out_dir))

    fp.target_dates = TARGET_DATES
    fp.date_mode = DATE_MODE

    fp.z_level = Z_LEVEL
    fp.flat_img_size = FLAT_IMG_SIZE
    fp.flat_fov_x = FLAT_FOV_X
    fp.directions = DIRECTIONS
    fp.yaw_offset = YAW_OFFSET

    fp.save_panorama = SAVE_PANORAMA

    fp.filename_style = FILENAME_STYLE
    fp.coord_decimals = COORD_DECIMALS
    fp.duplicate_policy = DUPLICATE_POLICY
    fp.sort_points_by_lon_lat = SORT_POINTS_BY_LON_LAT

    fp.verbose = VERBOSE
    fp.log_flush_every = LOG_FLUSH_EVERY
    fp.n_tries = N_TRIES
    fp.enable_resume = ENABLE_RESUME


def read_csv_if_exists(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[warn] 读取 CSV 失败：{path}, err={repr(e)}")
            return None


def success_mask(series: pd.Series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def is_pano_mask(series: pd.Series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def enrich_tile_log(tile_log_df: pd.DataFrame, tile_id: str, source_csv: Path, batch_index: int):
    """
    给每个 tile 的原始日志增加 tile 信息。
    不删除 flat_images_pano.py 原有字段，保证训练追踪价值。
    """
    df = tile_log_df.copy()

    # 避免重复插入列
    for col in ["batch_index", "tile_id", "source_csv_name", "source_csv_path", "image_type"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df.insert(0, "batch_index", batch_index)
    df.insert(1, "tile_id", tile_id)
    df.insert(2, "source_csv_name", source_csv.name)
    df.insert(3, "source_csv_path", str(source_csv))

    if "is_pano" in df.columns:
        df["image_type"] = df["is_pano"].map(
            lambda x: "panorama" if str(x).lower() in ["true", "1", "yes"] else "flat"
        )
    else:
        df["image_type"] = ""

    return df


def make_training_manifest(log_df: pd.DataFrame):
    """
    生成训练用 manifest：
    - success=True
    - is_pano=False
    - save_path 非空

    也就是说默认只收录四方向透视图，不收录 360° panorama。

    注意：
    这里不做 save_path 去重。因为 MASTER 是原始总表。
    后续进入 YOLO 前，仍建议使用 prepare_yolo_manifest_from_tiles.py 做唯一图片去重。
    """
    if log_df is None or log_df.empty:
        return pd.DataFrame()

    df = log_df.copy()

    if "success" in df.columns:
        mask_success = success_mask(df["success"])
    else:
        mask_success = pd.Series([True] * len(df), index=df.index)

    if "is_pano" in df.columns:
        mask_flat = ~is_pano_mask(df["is_pano"])
    else:
        mask_flat = pd.Series([True] * len(df), index=df.index)

    if "save_path" in df.columns:
        mask_path = df["save_path"].fillna("").astype(str).str.len() > 0
    else:
        mask_path = pd.Series([True] * len(df), index=df.index)

    manifest = df[mask_success & mask_flat & mask_path].copy()

    if "manual_keep" not in manifest.columns:
        manifest["manual_keep"] = ""
    if "manual_note" not in manifest.columns:
        manifest["manual_note"] = ""

    return manifest


def write_named_tile_logs(tile_out_dir: Path, tile_id: str, source_csv: Path, batch_index: int):
    """
    flat_images_pano.py 默认写：
        crawl_log_flat_actualdate_training_review.csv

    本函数会在同一个 tile 目录下生成：
        tile001_crawl_log.csv
        tile001_flat_image_manifest_for_training.csv

    并返回 enrich 后的 tile log。
    """
    raw_log_path = tile_out_dir / "crawl_log_flat_actualdate_training_review.csv"
    named_log_path = tile_out_dir / f"{tile_id}_crawl_log.csv"
    tile_manifest_path = tile_out_dir / f"{tile_id}_flat_image_manifest_for_training.csv"

    if not raw_log_path.exists():
        return None, raw_log_path, named_log_path, tile_manifest_path

    # 复制一份命名日志，保留原始日志不动
    shutil.copy2(raw_log_path, named_log_path)

    raw_df = read_csv_if_exists(named_log_path)
    if raw_df is None:
        return None, raw_log_path, named_log_path, tile_manifest_path

    enriched_df = enrich_tile_log(raw_df, tile_id, source_csv, batch_index)

    # 用 enrich 版本覆盖 tile001_crawl_log.csv，让每个 tile 自带 source_csv / tile_id 信息
    enriched_df.to_csv(named_log_path, index=False, encoding="utf-8-sig")

    tile_manifest = make_training_manifest(enriched_df)
    tile_manifest.to_csv(tile_manifest_path, index=False, encoding="utf-8-sig")

    return enriched_df, raw_log_path, named_log_path, tile_manifest_path


# ============================================================
# 4. MASTER 实时刷新相关函数
# ============================================================

def find_existing_tile_logs(dataset_root: Path):
    """
    扫描 DATASET_ROOT 下所有 tile*/tile*_crawl_log.csv。
    这使得 MASTER 可以在中断后基于已有 tile 日志重建。
    """
    if not dataset_root.exists():
        return []

    logs = []
    for p in dataset_root.glob("tile*/tile*_crawl_log.csv"):
        if p.exists() and p.stat().st_size > 0:
            logs.append(p)

    return sorted(logs, key=lambda p: (tile_number(p), str(p)))


def read_existing_tile_logs(dataset_root: Path):
    logs = find_existing_tile_logs(dataset_root)
    dfs = []
    failed_logs = []

    for p in logs:
        df = read_csv_if_exists(p)
        if df is None:
            failed_logs.append(str(p))
            continue
        dfs.append(df)

    return dfs, logs, failed_logs


def compact_value_counts(series: pd.Series) -> dict[str, int]:
    if series is None:
        return {}
    return {
        str(k): int(v)
        for k, v in series.fillna("").astype(str).value_counts().to_dict().items()
    }


def build_master_progress_dict(
    dataset_root: Path,
    tile_csvs: list[Path],
    batch_summary: list[dict[str, Any]],
    master_log: pd.DataFrame,
    master_manifest: pd.DataFrame,
    refresh_reason: str,
    last_tile_id: str = "",
    failed_logs: list[str] | None = None,
):
    failed_logs = failed_logs or []

    finished_or_skipped = [
        row for row in batch_summary
        if str(row.get("status", "")).lower() in ["finished", "skipped_existing_tile_log"]
    ]
    error_tiles = [
        row for row in batch_summary
        if str(row.get("status", "")).lower() == "error"
    ]

    status_counts = Counter(str(row.get("status", "")) for row in batch_summary)

    unique_images = 0
    if not master_manifest.empty and "save_path" in master_manifest.columns:
        unique_images = int(master_manifest["save_path"].fillna("").astype(str).nunique())

    tile_ids_in_master = 0
    if not master_log.empty and "tile_id" in master_log.columns:
        tile_ids_in_master = int(master_log["tile_id"].dropna().astype(str).nunique())

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "refresh_reason": refresh_reason,
        "last_tile_id": last_tile_id,
        "dataset_root": str(dataset_root),
        "total_tile_csvs": int(len(tile_csvs)),
        "batch_summary_rows": int(len(batch_summary)),
        "finished_or_skipped_tiles_this_run": int(len(finished_or_skipped)),
        "error_tiles_this_run": int(len(error_tiles)),
        "batch_status_counts_this_run": dict(status_counts),
        "tile_ids_in_master_log": tile_ids_in_master,
        "master_log_rows": int(len(master_log)),
        "master_manifest_rows": int(len(master_manifest)),
        "master_manifest_unique_save_path": unique_images,
        "failed_tile_logs_read": failed_logs,
        "master_log_status_counts": compact_value_counts(master_log["status"]) if "status" in master_log.columns else {},
        "master_log_image_type_counts": compact_value_counts(master_log["image_type"]) if "image_type" in master_log.columns else {},
        "note": (
            "MASTER 文件会在每个 tile 完成/跳过/报错后刷新。"
            "进入 YOLO 前仍建议使用 prepare_yolo_manifest_from_tiles.py 对 save_path 去重。"
        ),
    }


def write_master_files(
    dataset_root: Path,
    tile_csvs: list[Path],
    batch_summary: list[dict[str, Any]],
    all_tile_logs_in_memory: list[pd.DataFrame] | None = None,
    refresh_reason: str = "manual",
    last_tile_id: str = "",
):
    """
    写出 / 刷新 MASTER 文件。

    当 REBUILD_MASTER_FROM_EXISTING_TILE_LOGS=True 时：
        每次都从 DATASET_ROOT/tile*/tile*_crawl_log.csv 重新扫描。
        这是最稳妥的方式。

    否则：
        使用内存中的 all_tile_logs_in_memory。
    """
    dataset_root.mkdir(parents=True, exist_ok=True)

    summary_path = dataset_root / "MASTER_batch_summary.csv"
    master_log_path = dataset_root / "MASTER_crawl_log_all_tiles.csv"
    master_manifest_path = dataset_root / "MASTER_flat_image_manifest_for_training.csv"
    progress_path = dataset_root / "MASTER_progress.json"

    # 1) 写 batch summary
    summary_df = pd.DataFrame(batch_summary)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 2) 生成 master log
    failed_logs = []
    if REBUILD_MASTER_FROM_EXISTING_TILE_LOGS:
        tile_log_dfs, existing_log_paths, failed_logs = read_existing_tile_logs(dataset_root)
    else:
        tile_log_dfs = all_tile_logs_in_memory or []
        existing_log_paths = []

    if tile_log_dfs:
        master_log = pd.concat(tile_log_dfs, ignore_index=True)
    else:
        master_log = pd.DataFrame()

    master_log.to_csv(master_log_path, index=False, encoding="utf-8-sig")

    # 3) 生成 master manifest
    master_manifest = make_training_manifest(master_log)
    master_manifest.to_csv(master_manifest_path, index=False, encoding="utf-8-sig")

    # 4) 写进度 JSON
    progress = build_master_progress_dict(
        dataset_root=dataset_root,
        tile_csvs=tile_csvs,
        batch_summary=batch_summary,
        master_log=master_log,
        master_manifest=master_manifest,
        refresh_reason=refresh_reason,
        last_tile_id=last_tile_id,
        failed_logs=failed_logs,
    )
    progress["summary_path"] = str(summary_path)
    progress["master_log_path"] = str(master_log_path)
    progress["master_manifest_path"] = str(master_manifest_path)
    progress["existing_tile_log_files_seen"] = int(len(existing_log_paths))

    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    if PRINT_MASTER_REFRESH_SUMMARY:
        print()
        print("[MASTER refreshed]")
        print(f"  reason          = {refresh_reason}")
        print(f"  last_tile       = {last_tile_id}")
        print(f"  master_log_rows = {len(master_log)}")
        print(f"  manifest_rows   = {len(master_manifest)}")
        if "save_path" in master_manifest.columns and not master_manifest.empty:
            print(f"  unique_paths    = {master_manifest['save_path'].fillna('').astype(str).nunique()}")
        print(f"  summary         = {summary_path}")
        print(f"  master_log      = {master_log_path}")
        print(f"  master_manifest = {master_manifest_path}")
        print(f"  progress_json   = {progress_path}")
        print()

    return {
        "summary_path": summary_path,
        "master_log_path": master_log_path,
        "master_manifest_path": master_manifest_path,
        "progress_path": progress_path,
        "master_log_rows": len(master_log),
        "master_manifest_rows": len(master_manifest),
    }


# ============================================================
# 5. 主流程
# ============================================================

def main():
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    tile_csvs = find_tile_csvs(PROJECT_DIR, TILE_PATTERN)

    if not tile_csvs:
        raise FileNotFoundError(f"没有在 {PROJECT_DIR} 找到匹配 {TILE_PATTERN} 的 CSV 文件")

    print("=" * 80)
    print("批量 tile 街景下载任务：tile 独立文件夹 + MASTER 实时刷新版本")
    print("=" * 80)
    print(f"PROJECT_DIR   = {PROJECT_DIR}")
    print(f"TILE_PATTERN  = {TILE_PATTERN}")
    print(f"DATASET_ROOT  = {DATASET_ROOT}")
    print(f"tile 数量      = {len(tile_csvs)}")
    print(f"SKIP_FINISHED_TILE_LOG = {SKIP_FINISHED_TILE_LOG}")
    print(f"REFRESH_MASTER_AFTER_EACH_TILE = {REFRESH_MASTER_AFTER_EACH_TILE}")
    print(f"REBUILD_MASTER_FROM_EXISTING_TILE_LOGS = {REBUILD_MASTER_FROM_EXISTING_TILE_LOGS}")
    print("输出结构：")
    print("  DATASET_ROOT/tile001/actual_YYYY_MM/*.png")
    print("  DATASET_ROOT/tile001/panorama_YYYY_MM/*.png")
    print("  DATASET_ROOT/tile001/tile001_crawl_log.csv")
    print("  DATASET_ROOT/tile001/tile001_flat_image_manifest_for_training.csv")
    print("  DATASET_ROOT/MASTER_batch_summary.csv")
    print("  DATASET_ROOT/MASTER_crawl_log_all_tiles.csv")
    print("  DATASET_ROOT/MASTER_flat_image_manifest_for_training.csv")
    print("  DATASET_ROOT/MASTER_progress.json")
    print("=" * 80)
    print("运行顺序：")
    for p in tile_csvs:
        print(f"  - {p.name} -> {tile_folder_name(p)}")
    print("=" * 80)
    print()

    all_tile_logs = []
    batch_summary = []

    # 启动时先刷新一次 MASTER，便于你马上看到当前已有 tile 日志状态。
    write_master_files(
        dataset_root=DATASET_ROOT,
        tile_csvs=tile_csvs,
        batch_summary=batch_summary,
        all_tile_logs_in_memory=all_tile_logs,
        refresh_reason="startup_existing_logs_scan",
        last_tile_id="",
    )

    try:
        for batch_index, csv_path in enumerate(tile_csvs, start=1):
            t0 = time.time()
            tile_id = tile_folder_name(csv_path)
            tile_out_dir = DATASET_ROOT / tile_id
            tile_out_dir.mkdir(parents=True, exist_ok=True)

            named_log_path = tile_out_dir / f"{tile_id}_crawl_log.csv"
            tile_manifest_path = tile_out_dir / f"{tile_id}_flat_image_manifest_for_training.csv"

            if SKIP_FINISHED_TILE_LOG and named_log_path.exists() and named_log_path.stat().st_size > 0:
                print(f"[skip tile] {tile_id}: 已存在日志 {named_log_path}")

                log_df = read_csv_if_exists(named_log_path)
                if log_df is not None:
                    all_tile_logs.append(log_df)

                batch_summary.append({
                    "batch_index": batch_index,
                    "tile_id": tile_id,
                    "source_csv": str(csv_path),
                    "out_dir": str(tile_out_dir),
                    "status": "skipped_existing_tile_log",
                    "seconds": 0,
                    "raw_log": str(tile_out_dir / "crawl_log_flat_actualdate_training_review.csv"),
                    "tile_log": str(named_log_path),
                    "tile_manifest": str(tile_manifest_path),
                    "error": "",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                })

                if REFRESH_MASTER_AFTER_EACH_TILE:
                    write_master_files(
                        dataset_root=DATASET_ROOT,
                        tile_csvs=tile_csvs,
                        batch_summary=batch_summary,
                        all_tile_logs_in_memory=all_tile_logs,
                        refresh_reason="after_tile_skipped_existing",
                        last_tile_id=tile_id,
                    )
                continue

            print()
            print("#" * 80)
            print(f"[{batch_index}/{len(tile_csvs)}] 开始运行 {tile_id}")
            print(f"CSV     : {csv_path}")
            print(f"OUT_DIR : {tile_out_dir}")
            print("#" * 80)

            try:
                configure_flat_images_pano(
                    input_csv=csv_path,
                    out_dir=tile_out_dir,
                )

                fp.main()

                tile_log_df, raw_log_path, named_log_path, tile_manifest_path = write_named_tile_logs(
                    tile_out_dir=tile_out_dir,
                    tile_id=tile_id,
                    source_csv=csv_path,
                    batch_index=batch_index,
                )

                if tile_log_df is not None:
                    all_tile_logs.append(tile_log_df)

                seconds = round(time.time() - t0, 2)

                batch_summary.append({
                    "batch_index": batch_index,
                    "tile_id": tile_id,
                    "source_csv": str(csv_path),
                    "out_dir": str(tile_out_dir),
                    "status": "finished",
                    "seconds": seconds,
                    "raw_log": str(raw_log_path),
                    "tile_log": str(named_log_path),
                    "tile_manifest": str(tile_manifest_path),
                    "error": "",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                })

                print(f"[finished tile] {tile_id}, cost={seconds:.2f}s")
                print(f"  raw_log       = {raw_log_path}")
                print(f"  named_log     = {named_log_path}")
                print(f"  tile_manifest = {tile_manifest_path}")

                if REFRESH_MASTER_AFTER_EACH_TILE:
                    write_master_files(
                        dataset_root=DATASET_ROOT,
                        tile_csvs=tile_csvs,
                        batch_summary=batch_summary,
                        all_tile_logs_in_memory=all_tile_logs,
                        refresh_reason="after_tile_finished",
                        last_tile_id=tile_id,
                    )

            except KeyboardInterrupt:
                print()
                print(f"[interrupt tile] {tile_id}: 检测到 Ctrl+C，尝试保存当前 tile 日志并刷新 MASTER。")

                seconds = round(time.time() - t0, 2)

                # 如果 flat_images_pano.py 已经写出了 raw log，则尽量转成 tile 命名日志。
                try:
                    tile_log_df, raw_log_path, named_log_path, tile_manifest_path = write_named_tile_logs(
                        tile_out_dir=tile_out_dir,
                        tile_id=tile_id,
                        source_csv=csv_path,
                        batch_index=batch_index,
                    )
                    if tile_log_df is not None:
                        all_tile_logs.append(tile_log_df)
                except Exception as e:
                    raw_log_path = tile_out_dir / "crawl_log_flat_actualdate_training_review.csv"
                    named_log_path = tile_out_dir / f"{tile_id}_crawl_log.csv"
                    tile_manifest_path = tile_out_dir / f"{tile_id}_flat_image_manifest_for_training.csv"
                    print(f"[warn] 中断时整理 tile 日志失败：{repr(e)}")

                batch_summary.append({
                    "batch_index": batch_index,
                    "tile_id": tile_id,
                    "source_csv": str(csv_path),
                    "out_dir": str(tile_out_dir),
                    "status": "interrupted",
                    "seconds": seconds,
                    "raw_log": str(raw_log_path),
                    "tile_log": str(named_log_path),
                    "tile_manifest": str(tile_manifest_path),
                    "error": "KeyboardInterrupt",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                })

                write_master_files(
                    dataset_root=DATASET_ROOT,
                    tile_csvs=tile_csvs,
                    batch_summary=batch_summary,
                    all_tile_logs_in_memory=all_tile_logs,
                    refresh_reason="keyboard_interrupt",
                    last_tile_id=tile_id,
                )

                print("已刷新 MASTER。程序停止。")
                return

            except Exception as e:
                seconds = round(time.time() - t0, 2)
                err = repr(e)

                # 如果 tile 虽然报错，但 raw log 已经存在，也尽量整理命名日志。
                raw_log_path = tile_out_dir / "crawl_log_flat_actualdate_training_review.csv"
                named_log_path = tile_out_dir / f"{tile_id}_crawl_log.csv"
                tile_manifest_path = tile_out_dir / f"{tile_id}_flat_image_manifest_for_training.csv"

                try:
                    if raw_log_path.exists():
                        tile_log_df, raw_log_path, named_log_path, tile_manifest_path = write_named_tile_logs(
                            tile_out_dir=tile_out_dir,
                            tile_id=tile_id,
                            source_csv=csv_path,
                            batch_index=batch_index,
                        )
                        if tile_log_df is not None:
                            all_tile_logs.append(tile_log_df)
                except Exception as e2:
                    print(f"[warn] error tile 后整理日志失败：{repr(e2)}")

                batch_summary.append({
                    "batch_index": batch_index,
                    "tile_id": tile_id,
                    "source_csv": str(csv_path),
                    "out_dir": str(tile_out_dir),
                    "status": "error",
                    "seconds": seconds,
                    "raw_log": str(raw_log_path),
                    "tile_log": str(named_log_path) if named_log_path.exists() else "",
                    "tile_manifest": str(tile_manifest_path) if tile_manifest_path.exists() else "",
                    "error": err,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                })

                print(f"[error tile] {tile_id}, cost={seconds:.2f}s, err={err}")

                if REFRESH_MASTER_AFTER_EACH_TILE:
                    write_master_files(
                        dataset_root=DATASET_ROOT,
                        tile_csvs=tile_csvs,
                        batch_summary=batch_summary,
                        all_tile_logs_in_memory=all_tile_logs,
                        refresh_reason="after_tile_error",
                        last_tile_id=tile_id,
                    )

                if not CONTINUE_ON_ERROR:
                    break

    finally:
        # 无论正常结束还是 break，都最终刷新一次 MASTER。
        final_info = write_master_files(
            dataset_root=DATASET_ROOT,
            tile_csvs=tile_csvs,
            batch_summary=batch_summary,
            all_tile_logs_in_memory=all_tile_logs,
            refresh_reason="final_refresh",
            last_tile_id=batch_summary[-1]["tile_id"] if batch_summary else "",
        )

        print()
        print("=" * 80)
        print("tile 任务结束，MASTER 已刷新")
        print("=" * 80)
        print(f"batch summary        : {final_info['summary_path']}")
        print(f"master log           : {final_info['master_log_path']}")
        print(f"master train manifest: {final_info['master_manifest_path']}")
        print(f"master rows          : {final_info['master_log_rows']}")
        print(f"manifest rows        : {final_info['master_manifest_rows']}")
        print(f"progress json        : {final_info['progress_path']}")
        print("=" * 80)


if __name__ == "__main__":
    main()
