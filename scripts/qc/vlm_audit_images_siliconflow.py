"""Run SiliconFlow VLM audit for routed street-view images.

Input:
  outputs/.../auto_qc_routes_*.csv

Output:
  outputs/.../vlm_audit_results_*.csv

The output columns remain compatible with scripts/qc/resolve_final_counts.py.
Extra posture columns are included for second-level VLM posture auditing.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


BASE_URL_DEFAULT = "https://api.siliconflow.cn/v1"

COUNT_KEYS = [
    "visible_person_count_all",
    "pedestrian_count_strict",
    "cyclist_or_rider_count",
    "uncertain_count",
    "corrected_count",
    "vlm_standing_count",
    "vlm_sitting_count",
    "vlm_walking_count",
    "vlm_riding_count",
    "vlm_other_pose_count",
]

ENUM_DEFAULTS = {
    "poster_or_reflection_risk": ("none", {"none", "low", "medium", "high"}),
    "occlusion_level": ("none", {"none", "low", "medium", "high"}),
    "image_quality": ("medium", {"good", "medium", "poor"}),
    "yolo_assessment": (
        "uncertain",
        {"likely_correct", "likely_overcount", "likely_undercount", "uncertain"},
    ),
}

REQUIRED_OUTPUT_COLUMNS = [
    "image_id",
    "vlm_model",
    "vlm_detail",
    "vlm_status",
    "vlm_error",
    "visible_person_count_all",
    "pedestrian_count_strict",
    "cyclist_or_rider_count",
    "uncertain_count",
    "corrected_count",
    "poster_or_reflection_risk",
    "occlusion_level",
    "image_quality",
    "yolo_assessment",
    "use_for_training",
    "confidence",
    "explanation",
    "vlm_standing_count",
    "vlm_sitting_count",
    "vlm_walking_count",
    "vlm_riding_count",
    "vlm_other_pose_count",
    "vlm_pose_summary",
]


PROMPT_TEMPLATE = """你是街景行人活跃度研究的数据审核员。请只根据图像内容审核人数和姿态。

输入可能包含两张图：
1. 原始街景图。
2. YOLO 标注图，绿色框表示 YOLO 认为的人。

任务：
- 统计真实可见的人数。真实人包括站立、行走、坐着、推车、骑自行车/电动车/摩托车的人。
- 不要把海报、广告牌、橱窗反射、雕塑、假人、车内极不清楚的人计入。
- 远处非常小、遮挡严重或无法确认的人形放入 uncertain_count。
- pedestrian_count_strict 只统计步行、站立、坐着、等待等路侧行人，不包含骑车/电动车/摩托车者。
- corrected_count 是建议进入后续行人活跃度标签的最终真实可见人数，排除明显假阳性。
- 同时估计姿态人数：standing、sitting、walking、riding、other。姿态总数应尽量接近 corrected_count。

YOLO 信息：
image_id={image_id}
yolo_count={yolo_count}
yolo_mean_conf={yolo_conf}
review_reason={review_reason}
auto_route={auto_route}
route_reason={route_reason}
yolo_pose_count={pose_count}
yolo_standing={standing_count}
yolo_sitting={sitting_count}
yolo_walking_like={walking_like_count}

请严格输出一个 JSON 对象，不要输出 Markdown，不要输出解释性前后缀。JSON 字段必须是：
{{
  "visible_person_count_all": int,
  "pedestrian_count_strict": int,
  "cyclist_or_rider_count": int,
  "uncertain_count": int,
  "corrected_count": int,
  "poster_or_reflection_risk": "none|low|medium|high",
  "occlusion_level": "none|low|medium|high",
  "image_quality": "good|medium|poor",
  "yolo_assessment": "likely_correct|likely_overcount|likely_undercount|uncertain",
  "use_for_training": true/false,
  "confidence": float_between_0_and_1,
  "explanation": "一句话说明依据",
  "vlm_standing_count": int,
  "vlm_sitting_count": int,
  "vlm_walking_count": int,
  "vlm_riding_count": int,
  "vlm_other_pose_count": int,
  "vlm_pose_summary": "一句话说明姿态分布"
}}"""


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clamp_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return max(0, int(round(float(value))))
    except Exception:
        return default


def clamp_float(value: Any, default: float = 0.5) -> float:
    try:
        if pd.isna(value):
            return default
        return min(1.0, max(0.0, float(value)))
    except Exception:
        return default


def image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def pick_image_paths(row: pd.Series, include_annotated: bool) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    image_path = str(row.get("image_path", "")).strip()
    annotated_path = str(row.get("annotated_path", "")).strip()

    if image_path and image_path.lower() not in {"nan", "none"} and Path(image_path).exists():
        paths.append(("原始街景图", image_path))

    if (
        include_annotated
        and annotated_path
        and annotated_path.lower() not in {"nan", "none"}
        and Path(annotated_path).exists()
    ):
        paths.append(("YOLO 标注图", annotated_path))

    return paths


def build_prompt(row: pd.Series) -> str:
    return PROMPT_TEMPLATE.format(
        image_id=row.get("image_id", ""),
        yolo_count=row.get("person_count_final", ""),
        yolo_conf=row.get("mean_conf", ""),
        review_reason=row.get("review_reason", ""),
        auto_route=row.get("auto_route", ""),
        route_reason=row.get("route_reason", ""),
        pose_count=row.get("person_pose_count", ""),
        standing_count=row.get("standing_count", ""),
        sitting_count=row.get("sitting_count", ""),
        walking_like_count=row.get("walking_like_count", ""),
    )


def build_messages(row: pd.Series, args: argparse.Namespace) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": build_prompt(row),
        }
    ]

    paths = pick_image_paths(row, include_annotated=not args.no_annotated)
    if not paths:
        raise FileNotFoundError(f"No valid image found for image_id={row.get('image_id')}")

    for label, path in paths:
        content.append({"type": "text", "text": f"下面这张是：{label}"})
        image_url: dict[str, Any] = {"url": image_to_data_url(path)}
        if args.send_detail:
            image_url["detail"] = args.detail
        content.append({"type": "image_url", "image_url": image_url})

    return [
        {
            "role": "system",
            "content": "你是严谨的街景视觉数据审核员。只输出合法 JSON。",
        },
        {
            "role": "user",
            "content": content,
        },
    ]


def extract_message_text(response: Any) -> str:
    try:
        text = response.choices[0].message.content
    except Exception:
        text = ""
    if isinstance(text, list):
        parts = []
        for item in text:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    return str(text or "").strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in response: {text[:500]}")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])

    raise ValueError(f"Unclosed JSON object in response: {text[:500]}")


def normalize_record(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key in COUNT_KEYS:
        out[key] = clamp_int(data.get(key), 0)

    for key, (default, allowed) in ENUM_DEFAULTS.items():
        value = str(data.get(key, default)).strip().lower()
        out[key] = value if value in allowed else default

    use_for_training = data.get("use_for_training", True)
    if isinstance(use_for_training, str):
        use_for_training = use_for_training.strip().lower() not in {"false", "0", "no", "n"}
    out["use_for_training"] = bool(use_for_training)
    out["confidence"] = clamp_float(data.get("confidence"), 0.5)
    out["explanation"] = str(data.get("explanation", "")).strip()[:500]
    out["vlm_pose_summary"] = str(data.get("vlm_pose_summary", "")).strip()[:500]

    if not out["explanation"]:
        out["explanation"] = "VLM returned structured counts without explanation."
    if not out["vlm_pose_summary"]:
        out["vlm_pose_summary"] = "No posture summary provided."

    return out


def parse_response_json(response: Any) -> dict[str, Any]:
    text = extract_message_text(response)
    if not text:
        raise ValueError("Empty model response")
    return normalize_record(extract_json_object(text))


def audit_one(client: Any, row: pd.Series, args: argparse.Namespace) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": build_messages(row, args),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    if args.response_format_json:
        request_kwargs["response_format"] = {"type": "json_object"}
    if args.enable_thinking is not None:
        request_kwargs["extra_body"] = {"enable_thinking": args.enable_thinking}

    response = client.chat.completions.create(**request_kwargs)
    data = parse_response_json(response)

    out = {
        "image_id": row.get("image_id", ""),
        "vlm_model": args.model,
        "vlm_detail": args.detail,
        "vlm_status": "ok",
        "vlm_error": "",
    }
    out.update(data)
    return out


def should_audit(row: pd.Series) -> bool:
    if truthy(row.get("vlm_needed", False)):
        return True
    return str(row.get("auto_route", "")) in {"send_vlm", "human_conflict"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SiliconFlow VLM audit for routed street-view images.")
    parser.add_argument("--routes", required=True, help="auto_qc_routes.csv")
    parser.add_argument("--output", default="outputs/qc/vlm_audit_results.csv")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--base-url", default=os.environ.get("SILICONFLOW_BASE_URL", BASE_URL_DEFAULT))
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--detail", default="high", choices=["low", "high", "original", "auto"])
    parser.add_argument("--send-detail", action="store_true", help="Send image_url.detail to the provider.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-annotated", action="store_true", help="Only send the original image.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-response-format-json",
        dest="response_format_json",
        action="store_false",
        help="Disable response_format={type: json_object}.",
    )
    parser.set_defaults(response_format_json=True)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    thinking.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    routes = pd.read_csv(args.routes)
    if "image_id" not in routes.columns:
        raise ValueError("routes must contain image_id")
    routes["image_id"] = routes["image_id"].astype(str)

    targets = routes[routes.apply(should_audit, axis=1)].drop_duplicates("image_id", keep="first").copy()
    if args.limit is not None:
        targets = targets.head(args.limit).copy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    if args.resume and output_path.exists():
        existing = pd.read_csv(output_path)
        if "image_id" in existing.columns:
            existing["image_id"] = existing["image_id"].astype(str)
            if "vlm_status" in existing.columns:
                existing = existing[existing["vlm_status"].astype(str).str.lower().eq("ok")].copy()
            done_ids = set(existing["image_id"].tolist())
            records = existing.to_dict("records")

    if args.dry_run:
        targets.to_csv(output_path, index=False, encoding="utf-8-sig")
        print({"dry_run": True, "to_audit": int(len(targets)), "output": str(output_path)})
        return

    if OpenAI is None:
        raise ImportError("Please install openai: python -m pip install openai")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Please set {args.api_key_env}")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    for _, row in tqdm(targets.iterrows(), total=len(targets), desc="SiliconFlow VLM auditing"):
        image_id = str(row.get("image_id", ""))
        if image_id in done_ids:
            continue
        try:
            rec = audit_one(client, row, args)
        except Exception as e:
            rec = {
                "image_id": image_id,
                "vlm_model": args.model,
                "vlm_detail": args.detail,
                "vlm_status": "error",
                "vlm_error": repr(e),
            }
            for col in REQUIRED_OUTPUT_COLUMNS:
                rec.setdefault(col, "")
        records.append(rec)
        pd.DataFrame(records).to_csv(output_path, index=False, encoding="utf-8-sig")
        if args.sleep > 0:
            time.sleep(args.sleep)

    print({"audited_or_existing": len(records), "targets": int(len(targets)), "output": str(output_path)})


if __name__ == "__main__":
    main()
