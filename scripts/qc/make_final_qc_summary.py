# make_final_qc_summary.py
"""生成自动质控/VLM/final label 的摘要 JSON。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--routes", default="outputs/qc/auto_qc_routes.csv")
    p.add_argument("--final", default="outputs/qc/image_labels_final.csv")
    p.add_argument("--output", default="outputs/qc/final_qc_summary.json")
    args = p.parse_args()

    summary = {}
    if Path(args.routes).exists():
        r = pd.read_csv(args.routes)
        summary["routes_rows"] = int(len(r))
        if "auto_route" in r.columns:
            summary["routes"] = {str(k): int(v) for k, v in r["auto_route"].value_counts(dropna=False).items()}
        if "vlm_needed" in r.columns:
            summary["vlm_needed"] = int(r["vlm_needed"].fillna(False).astype(bool).sum())
        if "human_needed" in r.columns:
            summary["human_needed_initial"] = int(r["human_needed"].fillna(False).astype(bool).sum())

    if Path(args.final).exists():
        f = pd.read_csv(args.final)
        summary["final_rows"] = int(len(f))
        if "final_keep" in f.columns:
            summary["final_kept"] = int(f["final_keep"].fillna(False).astype(bool).sum())
        if "human_needed_final" in f.columns:
            summary["human_needed_final"] = int(f["human_needed_final"].fillna(False).astype(bool).sum())
        if "final_count_source" in f.columns:
            summary["final_count_source"] = {str(k): int(v) for k, v in f["final_count_source"].value_counts(dropna=False).items()}
        if "final_quality_score" in f.columns:
            summary["quality_mean"] = float(pd.to_numeric(f["final_quality_score"], errors="coerce").mean())
        if "road_seg_id" in f.columns:
            summary["road_unknown"] = int((f["road_seg_id"].astype(str) == "UNKNOWN").sum())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
