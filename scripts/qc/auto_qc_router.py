# auto_qc_router.py
"""
自动质控路由器：从 detections_with_pose.csv 生成大规模 YOLO/VLM/人工仲裁分流表。

它替代“逐张人工复核”的第一步，用规则把图片分为：
- accept_yolo_zero          : 零人图，暂时接受为低活跃样本
- accept_yolo_high_conf     : YOLO 高置信、人数不过高，直接接受
- accept_yolo_medium_conf   : YOLO 中等可信，保留为中等质量弱标签
- send_vlm                  : 低置信/高人数/姿态严重缺失等难例，送 VLM 二次审核
- human_conflict            : 极端异常，建议人工或剔除
- exclude                   : 图片缺失/检测错误等不可用样本

用法：
python scripts/qc/auto_qc_router.py ^
  --input outputs/detections_with_pose.csv ^
  --output outputs/qc/auto_qc_routes.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def has_text(value: Any, token: str) -> bool:
    if pd.isna(value):
        return False
    return token.lower() in str(value).lower()


def qc_priority(row: pd.Series, overcount_threshold: int, p0_conf: float, p1_conf: float) -> str:
    count = to_float(row.get("person_count_final", 0))
    conf = to_float(row.get("mean_conf", 0))
    pose_count = to_float(row.get("person_pose_count", 0))

    if count >= overcount_threshold or (count > 0 and conf < p0_conf):
        return "P0_必须二次审核"
    if count > 0 and conf < p1_conf:
        return "P1_建议二次审核"
    if count > 0 and pose_count == 0:
        return "P2_姿态缺失但人数可用"
    if count == 0:
        return "P3_零人样本抽检"
    return "P4_可自动接受"


def route_one(row: pd.Series, args: argparse.Namespace) -> dict[str, Any]:
    status = str(row.get("status", "ok")).lower()
    image_path = str(row.get("image_path", ""))
    count = to_float(row.get("person_count_final", 0))
    conf = to_float(row.get("mean_conf", 0))
    pose_count = to_float(row.get("person_pose_count", 0))
    pose_recall = to_float(row.get("pose_recall_proxy", np.nan), default=np.nan)
    review_reason = str(row.get("review_reason", ""))

    route = "accept_yolo_medium_conf"
    reason = []
    vlm_needed = False
    human_needed = False
    final_quality_score = 0.65

    # 基本不可用样本
    if status in {"missing", "error"} or image_path.strip() == "":
        return {
            "auto_route": "exclude",
            "route_reason": f"status={status};missing_or_empty_path",
            "vlm_needed": False,
            "human_needed": False,
            "auto_final_person_count": np.nan,
            "auto_final_count_source": "exclude_bad_input",
            "auto_quality_score": 0.0,
        }

    # 零人样本：大规模任务中不能全送 VLM，只做抽检；若后续接入高活跃道路上下文，可在这里升级为 send_vlm_sample。
    if count == 0:
        return {
            "auto_route": "accept_yolo_zero",
            "route_reason": "yolo_zero_auto_accept_low_activity_candidate",
            "vlm_needed": False,
            "human_needed": False,
            "auto_final_person_count": 0,
            "auto_final_count_source": "yolo_auto_zero",
            "auto_quality_score": args.zero_quality,
        }

    # 极端高人数：YOLO 在街景里容易把骑行者、橱窗、海报、远处模糊人形误计为人，优先送 VLM。
    if count >= args.overcount_threshold:
        route = "send_vlm"
        vlm_needed = True
        reason.append(f"count_ge_{args.overcount_threshold}")
        final_quality_score = 0.45

    # 低置信正样本：送 VLM。
    if conf < args.low_conf_threshold:
        route = "send_vlm"
        vlm_needed = True
        reason.append(f"mean_conf_lt_{args.low_conf_threshold}")
        final_quality_score = min(final_quality_score, 0.45)

    # 中等置信但仍偏低：可按预算选择是否送 VLM。
    elif conf < args.medium_conf_threshold:
        if args.audit_medium_conf:
            route = "send_vlm"
            vlm_needed = True
            reason.append(f"mean_conf_lt_{args.medium_conf_threshold}")
            final_quality_score = min(final_quality_score, 0.55)
        else:
            route = "accept_yolo_medium_conf"
            reason.append("medium_conf_yolo_weak_label")
            final_quality_score = 0.65

    # 高置信且人数不过高：自动接受。
    elif count < args.overcount_threshold:
        route = "accept_yolo_high_conf"
        reason.append("high_conf_and_not_dense")
        final_quality_score = 0.90

    # 姿态召回低不是否定人数检测的直接证据，但可以作为 VLM 触发器。
    if count >= args.pose_check_min_count:
        if pose_count == 0:
            if args.audit_pose_missing:
                route = "send_vlm"
                vlm_needed = True
            reason.append("pose_missing")
            final_quality_score = min(final_quality_score, 0.60)
        elif not pd.isna(pose_recall) and pose_recall < args.pose_low_recall_threshold:
            if args.audit_pose_low_recall:
                route = "send_vlm"
                vlm_needed = True
            reason.append(f"pose_recall_lt_{args.pose_low_recall_threshold}")
            final_quality_score = min(final_quality_score, 0.60)

    if has_text(review_reason, "possible_overcount"):
        route = "send_vlm"
        vlm_needed = True
        reason.append("review_possible_overcount")
        final_quality_score = min(final_quality_score, 0.45)

    if has_text(review_reason, "det_low_conf"):
        route = "send_vlm"
        vlm_needed = True
        reason.append("review_det_low_conf")
        final_quality_score = min(final_quality_score, 0.45)

    # 非常极端的异常：不建议完全自动，保留人工标记。
    if count >= args.human_count_threshold and conf < args.medium_conf_threshold:
        route = "human_conflict"
        vlm_needed = args.vlm_before_human
        human_needed = True
        reason.append(f"extreme_count_ge_{args.human_count_threshold}_with_low_conf")
        final_quality_score = 0.20

    auto_count = count if route.startswith("accept_yolo") else np.nan
    auto_source = route if route.startswith("accept_yolo") else "pending_vlm_or_human"

    return {
        "auto_route": route,
        "route_reason": ";".join(dict.fromkeys(reason)) if reason else "default_rule",
        "vlm_needed": bool(vlm_needed),
        "human_needed": bool(human_needed),
        "auto_final_person_count": auto_count,
        "auto_final_count_source": auto_source,
        "auto_quality_score": float(final_quality_score),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route YOLO detections to auto-accept, VLM audit, or human conflict review.")
    parser.add_argument("--input", required=True, help="outputs/detections_with_pose.csv")
    parser.add_argument("--output", default="outputs/qc/auto_qc_routes.csv")

    parser.add_argument("--overcount-threshold", type=int, default=8, help="人数达到该值时送 VLM 检查可能过检")
    parser.add_argument("--human-count-threshold", type=int, default=15, help="人数极高且低置信时标记人工冲突")
    parser.add_argument("--low-conf-threshold", type=float, default=0.35, help="低于该置信度的正样本送 VLM")
    parser.add_argument("--medium-conf-threshold", type=float, default=0.50, help="中等置信阈值")
    parser.add_argument("--audit-medium-conf", action="store_true", help="是否把 0.35-0.50 的正样本也送 VLM")

    parser.add_argument("--pose-check-min-count", type=int, default=3)
    parser.add_argument("--pose-low-recall-threshold", type=float, default=0.30)
    parser.add_argument("--audit-pose-missing", action="store_true", help="人数>=阈值但姿态为0时是否送 VLM")
    parser.add_argument("--audit-pose-low-recall", action="store_true", help="姿态召回代理值低时是否送 VLM")
    parser.add_argument("--vlm-before-human", action="store_true", default=True, help="人工冲突前是否先走 VLM")

    parser.add_argument("--zero-quality", type=float, default=0.75, help="零人自动接受样本的质量权重")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if "image_id" not in df.columns:
        raise ValueError("input 必须包含 image_id 列")

    # 保险：再次去重，防止误用上游重复表。
    df["image_id"] = df["image_id"].astype(str)
    df = df.drop_duplicates("image_id", keep="first").copy()

    for col in ["person_count_final", "mean_conf", "person_pose_count", "pose_recall_proxy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["qc_priority"] = df.apply(
        lambda r: qc_priority(r, args.overcount_threshold, args.low_conf_threshold, args.medium_conf_threshold),
        axis=1,
    )

    route_rows = [route_one(row, args) for _, row in df.iterrows()]
    route_df = pd.DataFrame(route_rows)
    out_df = pd.concat([df.reset_index(drop=True), route_df], axis=1)

    # 推荐 VLM 批次字段，方便后续分批调用 API。
    out_df["vlm_batch_group"] = np.where(out_df["vlm_needed"], "vlm_pending", "not_needed")

    ordered = [
        "auto_route", "qc_priority", "vlm_needed", "human_needed", "route_reason",
        "image_id", "image_path", "annotated_path", "pose_annotated_path",
        "actual_date", "heading", "lon", "lat", "road_seg_id",
        "person_count_final", "mean_conf", "person_pose_count", "pose_detect_gap", "pose_recall_proxy",
        "need_review", "review_reason",
        "auto_final_person_count", "auto_final_count_source", "auto_quality_score", "vlm_batch_group",
    ]
    ordered = [c for c in ordered if c in out_df.columns]
    other = [c for c in out_df.columns if c not in ordered]
    out_df = out_df[ordered + other]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "rows": int(len(out_df)),
        "routes": {str(k): int(v) for k, v in out_df["auto_route"].value_counts().items()},
        "vlm_needed": int(out_df["vlm_needed"].fillna(False).sum()),
        "human_needed": int(out_df["human_needed"].fillna(False).sum()),
        "output": str(output_path),
    }
    print(summary)


if __name__ == "__main__":
    main()
