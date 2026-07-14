# scripts/qc/aggregate_final_to_segment_labels.py
"""
把 image_labels_final.csv 聚合成 road_seg_id 级 segment_labels.csv。

前提：
image_labels_final.csv 中必须有 road_seg_id。
如果 road_seg_id 仍然是 UNKNOWN，请先做 lon/lat 最近道路段匹配。

运行：
python scripts/qc/aggregate_final_to_segment_labels.py ^
  --input outputs/qc/image_labels_final.csv ^
  --output data/processed/segment_labels.csv ^
  --drop-unknown-road
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def weighted_mean(values, weights):
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0)
    mask = v.notna() & (w > 0)
    if mask.sum() == 0:
        return float(v.dropna().mean()) if v.notna().any() else np.nan
    return float(np.average(v[mask], weights=w[mask]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/processed/segment_labels.csv")
    p.add_argument("--drop-unknown-road", action="store_true")
    p.add_argument("--min-images-per-seg", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input, encoding="utf-8-sig")

    if "road_seg_id" not in df.columns:
        raise ValueError("input 缺少 road_seg_id。请先完成图片点位到道路段的匹配。")

    if "final_person_count" not in df.columns:
        if "person_count_final" in df.columns:
            df["final_person_count"] = df["person_count_final"]
        else:
            raise ValueError("input 缺少 final_person_count/person_count_final")

    if "final_keep" in df.columns:
        keep = df["final_keep"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        df = df[keep].copy()

    if args.drop_unknown_road:
        df = df[df["road_seg_id"].notna()].copy()
        df = df[~df["road_seg_id"].astype(str).str.upper().isin(["UNKNOWN", "", "NAN"])].copy()

    df["final_person_count"] = pd.to_numeric(df["final_person_count"], errors="coerce")
    df = df[df["final_person_count"].notna()].copy()

    if "final_quality_score" not in df.columns:
        df["final_quality_score"] = 1.0

    group = df.groupby("road_seg_id", as_index=False)
    rows = []
    for rid, g in group:
        row = {
            "road_seg_id": rid,
            "person_count_mean": float(g["final_person_count"].mean()),
            "person_count_median": float(g["final_person_count"].median()),
            "person_count_max": float(g["final_person_count"].max()),
            "person_count_weighted_mean": weighted_mean(g["final_person_count"], g["final_quality_score"]),
            "n_images": int(len(g)),
            "n_tiles": int(g["tile_id"].nunique()) if "tile_id" in g.columns else 0,
            "n_iid": int(g["iid"].nunique()) if "iid" in g.columns else 0,
            "n_actual_months": int(g["actual_month"].nunique()) if "actual_month" in g.columns else 0,
            "n_headings": int(g["heading"].nunique()) if "heading" in g.columns else 0,
            "mean_quality_score": float(pd.to_numeric(g["final_quality_score"], errors="coerce").fillna(0).mean()),
        }
        row["y"] = float(np.log1p(row["person_count_weighted_mean"]))
        row["has_label"] = True
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out[out["n_images"] >= args.min_images_per_seg].copy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")

    summary = {
        "input_rows": int(len(df)),
        "segment_rows": int(len(out)),
        "output": str(output),
        "unknown_road_remaining": int(df["road_seg_id"].astype(str).str.upper().eq("UNKNOWN").sum()),
    }
    with open(output.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
