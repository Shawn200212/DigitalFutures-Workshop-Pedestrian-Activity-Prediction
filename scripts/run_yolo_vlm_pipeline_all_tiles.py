# scripts/run_yolo_vlm_pipeline_all_tiles.py
"""
100 tile 数据进入 YOLO + VLM Agent QC 的一键流水线。

放置位置：
  C:/Users/Shaw2/Downloads/svi_pedestrian_demo/scripts\run_yolo_vlm_pipeline_all_tiles.py

前提：
1. 100 个 tile 已经爬取完成，data/Images 下有：
   tile001, tile002, ... 以及 MASTER_flat_image_manifest_for_training.csv
2. app/batch_detect.py、app/batch_pose.py 已替换为 manifest 版本。
3. scripts/qc/auto_qc_router.py、vlm_audit_images_siliconflow.py、resolve_final_counts.py 已存在。

运行：
python scripts/run_yolo_vlm_pipeline_all_tiles.py ^
  --dataset-root data/Images ^
  --yolo-model yolo11n.pt ^
  --pose-model yolo11n-pose.pt ^
  --device 0 ^
  --run-pose ^
  --run-vlm-dry-run

真正调用硅基流动 VLM 时：
set SILICONFLOW_API_KEY=你的key
python scripts/run_yolo_vlm_pipeline_all_tiles.py ^
  --dataset-root data/Images ^
  --device 0 ^
  --run-pose ^
  --run-vlm ^
  --vlm-model Qwen/Qwen3.5-4B ^
  --vlm-limit 200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], dry: bool = False) -> None:
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd))
    print("=" * 100)
    if not dry:
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="data/Images")
    p.add_argument("--manifest", default="outputs/manifests/master_yolo_manifest_unique.csv")
    p.add_argument("--outputs-root", default="outputs")

    p.add_argument("--yolo-model", default="yolo11n.pt")
    p.add_argument("--pose-model", default="yolo11n-pose.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", default="960")
    p.add_argument("--yolo-conf", default="0.25")
    p.add_argument("--pose-conf", default="0.08")

    p.add_argument("--run-pose", action="store_true")
    p.add_argument("--run-vlm-dry-run", action="store_true")
    p.add_argument("--run-vlm", action="store_true")
    p.add_argument("--vlm-model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--vlm-detail", default="low")
    p.add_argument("--vlm-limit", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    outputs = Path(args.outputs_root)
    manifest = Path(args.manifest)

    # 1. 生成唯一 YOLO manifest
    run([
        py, "scripts/prepare_yolo_manifest_from_tiles.py",
        "--dataset-root", args.dataset_root,
        "--output", str(manifest),
        "--summary", "outputs/manifests/master_yolo_manifest_summary.json",
        "--dedupe-key", "path",
    ], dry=args.dry_run)

    # 2. YOLO 人数检测
    detect_cmd = [
        py, "app/batch_detect.py",
        "--manifest", str(manifest),
        "--output-dir", str(outputs / "yolo"),
        "--model", args.yolo_model,
        "--conf", args.yolo_conf,
        "--imgsz", args.imgsz,
        "--save-vis",
    ]
    if args.device is not None:
        detect_cmd += ["--device", args.device]
    run(detect_cmd, dry=args.dry_run)

    # 3. Pose 可选
    if args.run_pose:
        pose_cmd = [
            py, "app/batch_pose.py",
            "--manifest", str(manifest),
            "--output-dir", str(outputs / "pose"),
            "--model", args.pose_model,
            "--conf", args.pose_conf,
            "--imgsz", args.imgsz,
        ]
        if args.device is not None:
            pose_cmd += ["--device", args.device]
        run(pose_cmd, dry=args.dry_run)
    else:
        print("跳过 Pose。若需要姿态质控，请加 --run-pose")

    # 4. 合并
    pose_csv = str(outputs / "pose" / "pose_detections.csv")
    if args.run_pose:
        run([
            py, "app/merge_detect_pose.py",
            "--detections", str(outputs / "yolo" / "detections.csv"),
            "--poses", pose_csv,
            "--output", str(outputs / "detections_with_pose.csv"),
        ], dry=args.dry_run)
    else:
        # 没跑 pose，也可以让后面的 router 基于 detections 直接工作：
        run([
            py, "scripts/qc/auto_qc_router.py",
            "--input", str(outputs / "yolo" / "detections.csv"),
            "--output", str(outputs / "qc" / "auto_qc_routes.csv"),
        ], dry=args.dry_run)

    # 5. 自动路由
    if args.run_pose:
        run([
            py, "scripts/qc/auto_qc_router.py",
            "--input", str(outputs / "detections_with_pose.csv"),
            "--output", str(outputs / "qc" / "auto_qc_routes.csv"),
            "--audit-medium-conf",
            "--audit-pose-missing",
        ], dry=args.dry_run)

    # 6. VLM 预览或调用
    if args.run_vlm_dry_run:
        run([
            py, "scripts/qc/vlm_audit_images_siliconflow.py",
            "--routes", str(outputs / "qc" / "auto_qc_routes.csv"),
            "--output", str(outputs / "qc" / "vlm_to_audit_preview.csv"),
            "--dry-run",
        ], dry=args.dry_run)

    if args.run_vlm:
        cmd = [
            py, "scripts/qc/vlm_audit_images_siliconflow.py",
            "--routes", str(outputs / "qc" / "auto_qc_routes.csv"),
            "--output", str(outputs / "qc" / "vlm_audit_results.csv"),
            "--model", args.vlm_model,
            "--detail", args.vlm_detail,
            "--resume",
        ]
        if args.vlm_limit:
            cmd += ["--limit", str(args.vlm_limit)]
        run(cmd, dry=args.dry_run)

        # 7. 最终标签
        run([
            py, "scripts/qc/resolve_final_counts.py",
            "--routes", str(outputs / "qc" / "auto_qc_routes.csv"),
            "--vlm", str(outputs / "qc" / "vlm_audit_results.csv"),
            "--output", str(outputs / "qc" / "image_labels_final.csv"),
        ], dry=args.dry_run)

    print("\n流水线命令执行结束。")


if __name__ == "__main__":
    main()
