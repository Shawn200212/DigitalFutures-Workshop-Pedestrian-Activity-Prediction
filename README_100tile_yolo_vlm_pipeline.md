# 100 Tile 街景数据进入 YOLO + VLM 的下一步部署

## 你现在的关键变化

当前爬虫已经变成 tile 独立文件夹结构：

```text
data/Images/
  tile001/
    actual_YYYY_MM/*.png
    panorama_YYYY_MM/*.png
    tile001_crawl_log.csv
    tile001_flat_image_manifest_for_training.csv
  tile002/
  ...
  MASTER_crawl_log_all_tiles.csv
  MASTER_flat_image_manifest_for_training.csv
```

`MASTER_flat_image_manifest_for_training.csv` 不能直接送进 YOLO，因为 nearest 模式会让多个 target_date 对应同一 actual_date，导致同一 save_path 重复出现。因此必须先生成唯一图片 manifest。

## 文件放置

复制本包文件到项目：

```text
scripts/prepare_yolo_manifest_from_tiles.py
scripts/run_yolo_vlm_pipeline_all_tiles.py
scripts/qc/aggregate_final_to_segment_labels.py
app/batch_detect.py
app/batch_pose.py
```

`app/merge_detect_pose.py`、`scripts/qc/auto_qc_router.py`、`scripts/qc/vlm_audit_images_siliconflow.py`、`scripts/qc/resolve_final_counts.py` 使用你前面已经替换/下载的版本。

## Step 1: 生成唯一 YOLO manifest

```cmd
python scripts/prepare_yolo_manifest_from_tiles.py ^
  --dataset-root data/Images ^
  --output outputs/manifests/master_yolo_manifest_unique.csv ^
  --summary outputs/manifests/master_yolo_manifest_summary.json ^
  --dedupe-key path
```

## Step 2: YOLO 人数检测

```cmd
python app/batch_detect.py ^
  --manifest outputs/manifests/master_yolo_manifest_unique.csv ^
  --output-dir outputs/yolo ^
  --model yolo11n.pt ^
  --conf 0.25 ^
  --imgsz 960 ^
  --device 0 ^
  --save-vis
```

## Step 3: Pose 可选

```cmd
python app/batch_pose.py ^
  --manifest outputs/manifests/master_yolo_manifest_unique.csv ^
  --output-dir outputs/pose ^
  --model yolo11n-pose.pt ^
  --conf 0.08 ^
  --imgsz 960 ^
  --device 0
```

## Step 4: 合并 detection + pose

```cmd
python app/merge_detect_pose.py ^
  --detections outputs/yolo/detections.csv ^
  --poses outputs/pose/pose_detections.csv ^
  --output outputs/detections_with_pose.csv
```

## Step 5: 自动 QC 路由

```cmd
python scripts/qc/auto_qc_router.py ^
  --input outputs/detections_with_pose.csv ^
  --output outputs/qc/auto_qc_routes.csv ^
  --audit-medium-conf ^
  --audit-pose-missing
```

## Step 6: 硅基流动 VLM 预览

```cmd
python scripts/qc/vlm_audit_images_siliconflow.py ^
  --routes outputs/qc/auto_qc_routes.csv ^
  --output outputs/qc/vlm_to_audit_preview.csv ^
  --dry-run
```

## Step 7: 小批量 VLM 审核

```cmd
set SILICONFLOW_API_KEY=你的Key
set SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

python scripts/qc/vlm_audit_images_siliconflow.py ^
  --routes outputs/qc/auto_qc_routes.csv ^
  --output outputs/qc/vlm_audit_results.csv ^
  --model Qwen/Qwen3.5-4B ^
  --detail low ^
  --limit 200 ^
  --resume
```

## Step 8: 生成最终图片标签

```cmd
python scripts/qc/resolve_final_counts.py ^
  --routes outputs/qc/auto_qc_routes.csv ^
  --vlm outputs/qc/vlm_audit_results.csv ^
  --output outputs/qc/image_labels_final.csv
```

## Step 9: 路段聚合

这一步要求 `image_labels_final.csv` 已经有真实 `road_seg_id`。如果还是 UNKNOWN，请先做最近道路匹配。

```cmd
python scripts/qc/aggregate_final_to_segment_labels.py ^
  --input outputs/qc/image_labels_final.csv ^
  --output data/processed/segment_labels.csv ^
  --drop-unknown-road
```
