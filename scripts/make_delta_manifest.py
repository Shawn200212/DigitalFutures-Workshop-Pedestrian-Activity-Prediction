# scripts/make_delta_manifest.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_csv_or_empty(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_image_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "image_id" in df.columns:
        df["image_id"] = df["image_id"].astype(str)
        return df

    if "image_path" in df.columns:
        df["image_id"] = df["image_path"].astype(str).map(lambda p: Path(p).stem)
        return df

    if "save_path" in df.columns:
        df["image_id"] = df["save_path"].astype(str).map(lambda p: Path(p).stem)
        return df

    raise ValueError("CSV 必须包含 image_id / image_path / save_path 中至少一列")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create delta manifest by excluding images already processed."
    )
    parser.add_argument("--manifest", required=True, help="当前 master_yolo_manifest_unique.csv")
    parser.add_argument(
        "--existing",
        nargs="*",
        default=[],
        help="已经跑过的 detections.csv，可传多个；不存在会自动忽略",
    )
    parser.add_argument("--output", required=True, help="输出本批新增 manifest")
    args = parser.parse_args()

    manifest = normalize_image_id(pd.read_csv(args.manifest))

    existing_ids = set()
    for p in args.existing:
        df = read_csv_or_empty(p)
        if df.empty:
            continue
        df = normalize_image_id(df)
        existing_ids.update(df["image_id"].astype(str).tolist())

    delta = manifest[~manifest["image_id"].astype(str).isin(existing_ids)].copy()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    delta.to_csv(out, index=False, encoding="utf-8-sig")

    print({
        "manifest_rows": len(manifest),
        "existing_images": len(existing_ids),
        "delta_rows": len(delta),
        "output": str(out),
    })


if __name__ == "__main__":
    main()