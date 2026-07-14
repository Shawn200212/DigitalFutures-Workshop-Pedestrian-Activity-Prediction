# resolve_final_counts.py
"""
合并 YOLO 自动路由结果与 VLM 审核结果，生成最终图片级标签 image_labels_final.csv。

用法：
python scripts/qc/resolve_final_counts.py ^
  --routes outputs/qc/auto_qc_routes.csv ^
  --vlm outputs/qc/vlm_audit_results.csv ^
  --output outputs/qc/image_labels_final.csv

后续路段聚合请使用本脚本输出的 final_person_count，而不是直接使用 person_count_final。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def f(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def b(value: Any, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def choose_vlm_count(row: pd.Series, final_count_mode: str) -> float:
    if final_count_mode == "pedestrian_strict":
        return f(row.get("pedestrian_count_strict"), np.nan)
    if final_count_mode == "visible_all":
        return f(row.get("visible_person_count_all"), np.nan)
    # corrected: default, recommended for activity label
    return f(row.get("corrected_count"), np.nan)


def resolve_one(row: pd.Series, args: argparse.Namespace) -> dict[str, Any]:
    route = str(row.get("auto_route", ""))
    yolo_count = f(row.get("person_count_final"), 0)
    yolo_conf = f(row.get("mean_conf"), 0)
    auto_quality = f(row.get("auto_quality_score"), 0.5)
    vlm_status = str(row.get("vlm_status", "")).lower()

    out = {
        "final_person_count": np.nan,
        "final_count_source": "unresolved",
        "final_quality_score": 0.0,
        "final_keep": False,
        "human_needed_final": False,
        "final_decision_reason": "",
    }

    if route == "exclude":
        out.update(
            final_person_count=np.nan,
            final_count_source="exclude_bad_input",
            final_quality_score=0.0,
            final_keep=False,
            final_decision_reason="route_exclude",
        )
        return out

    if route.startswith("accept_yolo"):
        out.update(
            final_person_count=int(round(yolo_count)),
            final_count_source=route,
            final_quality_score=auto_quality,
            final_keep=True,
            final_decision_reason="auto_accepted_by_router",
        )
        return out

    # route = send_vlm / human_conflict：优先读取 VLM。
    if vlm_status != "ok":
        if args.missing_vlm_policy == "use_yolo_low_quality":
            out.update(
                final_person_count=int(round(yolo_count)),
                final_count_source="yolo_pending_vlm_used_low_quality",
                final_quality_score=min(auto_quality, args.pending_vlm_quality),
                final_keep=True,
                human_needed_final=(route == "human_conflict"),
                final_decision_reason="vlm_missing_or_error_use_yolo_temporarily",
            )
        else:
            out.update(
                final_count_source="pending_vlm",
                final_quality_score=0.0,
                final_keep=False,
                human_needed_final=(route == "human_conflict"),
                final_decision_reason="vlm_required_but_missing",
            )
        return out

    use_for_training = b(row.get("use_for_training"), True)
    vlm_conf = f(row.get("confidence"), 0.5)
    vlm_count = choose_vlm_count(row, args.final_count_mode)
    uncertain = f(row.get("uncertain_count"), 0)
    risk = str(row.get("poster_or_reflection_risk", "")).lower()
    assessment = str(row.get("yolo_assessment", "")).lower()

    if not use_for_training or np.isnan(vlm_count):
        out.update(
            final_person_count=np.nan,
            final_count_source="vlm_exclude",
            final_quality_score=0.0,
            final_keep=False,
            human_needed_final=False,
            final_decision_reason="vlm_not_suitable_for_training_or_nan_count",
        )
        return out

    diff = abs(vlm_count - yolo_count)
    denom = max(yolo_count, vlm_count, 1.0)
    diff_ratio = diff / denom

    # YOLO/VLM 基本一致：用两者均值降低单模型噪声。
    if diff <= args.agree_abs_diff or diff_ratio <= args.agree_ratio:
        final_count = int(round((vlm_count + yolo_count) / 2.0))
        source = "yolo_vlm_agree"
        quality = max(0.75, min(0.95, 0.70 + 0.25 * vlm_conf))
        reason = "yolo_and_vlm_close"

    # 明显过检：VLM 更低且它认为有海报/反射/误检风险，采用 VLM。
    elif vlm_count < yolo_count and (risk in {"medium", "high"} or assessment == "likely_overcount"):
        final_count = int(round(vlm_count))
        source = "vlm_corrected_overcount"
        quality = 0.65 if vlm_conf >= 0.6 else 0.50
        reason = f"vlm_lower_with_risk={risk}_assessment={assessment}"

    # 明显漏检：VLM 更高且判断 YOLO 低估。
    elif vlm_count > yolo_count and assessment == "likely_undercount":
        final_count = int(round(vlm_count))
        source = "vlm_corrected_undercount"
        quality = 0.65 if vlm_conf >= 0.6 else 0.50
        reason = "vlm_higher_and_assessment_undercount"

    # 分歧大：尽量自动化，但标记低质量和 human_needed_final。
    else:
        if args.conflict_policy == "exclude":
            out.update(
                final_person_count=np.nan,
                final_count_source="conflict_exclude",
                final_quality_score=0.0,
                final_keep=False,
                human_needed_final=True,
                final_decision_reason="large_yolo_vlm_conflict_excluded",
            )
            return out
        if args.conflict_policy == "use_vlm_low_quality":
            final_count = int(round(vlm_count))
            source = "vlm_conflict_used_low_quality"
            quality = 0.35 if vlm_conf < 0.7 else 0.45
            reason = "large_conflict_use_vlm_low_quality"
        else:  # use_yolo_low_quality
            final_count = int(round(yolo_count))
            source = "yolo_conflict_used_low_quality"
            quality = 0.30
            reason = "large_conflict_use_yolo_low_quality"

    # 不确定人数多，降低质量权重。
    if uncertain >= args.uncertain_count_penalty_threshold:
        quality = min(quality, args.uncertain_quality_cap)
        reason += f";uncertain_count_ge_{args.uncertain_count_penalty_threshold}"

    human_needed = (diff >= args.human_abs_diff and diff_ratio >= args.human_ratio) or route == "human_conflict"

    out.update(
        final_person_count=max(0, final_count),
        final_count_source=source,
        final_quality_score=round(float(quality), 3),
        final_keep=quality >= args.min_quality_to_keep,
        human_needed_final=bool(human_needed),
        final_decision_reason=reason,
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve final image-level person counts from YOLO routes and VLM audit results.")
    parser.add_argument("--routes", required=True, help="outputs/qc/auto_qc_routes.csv")
    parser.add_argument("--vlm", default="outputs/qc/vlm_audit_results.csv", help="VLM 审核结果，可不存在")
    parser.add_argument("--output", default="outputs/qc/image_labels_final.csv")

    parser.add_argument("--final-count-mode", choices=["corrected", "visible_all", "pedestrian_strict"], default="corrected")
    parser.add_argument("--missing-vlm-policy", choices=["pending", "use_yolo_low_quality"], default="use_yolo_low_quality")
    parser.add_argument("--pending-vlm-quality", type=float, default=0.35)
    parser.add_argument("--conflict-policy", choices=["use_vlm_low_quality", "use_yolo_low_quality", "exclude"], default="use_vlm_low_quality")

    parser.add_argument("--agree-abs-diff", type=float, default=1.0)
    parser.add_argument("--agree-ratio", type=float, default=0.25)
    parser.add_argument("--human-abs-diff", type=float, default=3.0)
    parser.add_argument("--human-ratio", type=float, default=0.50)
    parser.add_argument("--uncertain-count-penalty-threshold", type=float, default=2.0)
    parser.add_argument("--uncertain-quality-cap", type=float, default=0.45)
    parser.add_argument("--min-quality-to-keep", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    routes = pd.read_csv(args.routes)
    if "image_id" not in routes.columns:
        raise ValueError("routes 必须包含 image_id 列")
    routes["image_id"] = routes["image_id"].astype(str)
    routes = routes.drop_duplicates("image_id", keep="first").copy()

    vlm_path = Path(args.vlm)
    if vlm_path.exists():
        vlm = pd.read_csv(vlm_path)
        if "image_id" not in vlm.columns:
            raise ValueError("vlm 文件必须包含 image_id 列")
        vlm["image_id"] = vlm["image_id"].astype(str)
        vlm = vlm.drop_duplicates("image_id", keep="last")
        df = routes.merge(vlm, on="image_id", how="left", suffixes=("", "_vlm"))
    else:
        df = routes.copy()
        df["vlm_status"] = "missing"

    for col in [
        "person_count_final", "mean_conf", "auto_quality_score", "visible_person_count_all",
        "pedestrian_count_strict", "corrected_count", "uncertain_count", "confidence",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    resolved = pd.DataFrame([resolve_one(row, args) for _, row in df.iterrows()])
    out_df = pd.concat([df.reset_index(drop=True), resolved], axis=1)

    ordered = [
        "final_keep", "final_person_count", "final_count_source", "final_quality_score", "human_needed_final", "final_decision_reason",
        "auto_route", "vlm_needed", "image_id", "image_path", "annotated_path", "actual_date", "heading", "lon", "lat", "road_seg_id",
        "person_count_final", "mean_conf", "person_pose_count", "pose_recall_proxy",
        "vlm_status", "corrected_count", "visible_person_count_all", "pedestrian_count_strict", "cyclist_or_rider_count", "uncertain_count",
        "poster_or_reflection_risk", "occlusion_level", "image_quality", "yolo_assessment", "confidence", "explanation",
    ]
    ordered = [c for c in ordered if c in out_df.columns]
    other = [c for c in out_df.columns if c not in ordered]
    out_df = out_df[ordered + other]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "rows": int(len(out_df)),
        "kept": int(out_df["final_keep"].fillna(False).sum()),
        "human_needed_final": int(out_df["human_needed_final"].fillna(False).sum()),
        "source_counts": {str(k): int(v) for k, v in out_df["final_count_source"].value_counts(dropna=False).items()},
        "output": str(output_path),
    }
    print(summary)


if __name__ == "__main__":
    main()
