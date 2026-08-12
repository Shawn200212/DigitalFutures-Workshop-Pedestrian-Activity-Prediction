# Deploying the 100-Tile Street-View Dataset into the YOLO + VLM Pipeline

**English** | [中文](README_100tile_yolo_vlm_pipeline.md)

## Key change

The crawler now writes one folder per tile. The MASTER flat-image manifest cannot be passed directly
to YOLO: nearest-date selection can map multiple target dates to one actual date and repeat the same
`save_path`. Generate a unique image manifest first.

## Pipeline

1. Generate the unique YOLO manifest:

```cmd
python scripts/prepare_yolo_manifest_from_tiles.py ^
  --dataset-root data/Images ^
  --output outputs/manifests/master_yolo_manifest_unique.csv ^
  --summary outputs/manifests/master_yolo_manifest_summary.json ^
  --dedupe-key path
```

2. Run YOLO detection:

```cmd
python app/batch_detect.py ^
  --manifest outputs/manifests/master_yolo_manifest_unique.csv ^
  --output-dir outputs/yolo ^
  --model yolo11n.pt --conf 0.25 --imgsz 960 --device 0 --save-vis
```

3. Optionally run pose estimation, then merge detection and pose:

```cmd
python app/batch_pose.py ^
  --manifest outputs/manifests/master_yolo_manifest_unique.csv ^
  --output-dir outputs/pose ^
  --model yolo11n-pose.pt --conf 0.08 --imgsz 960 --device 0

python app/merge_detect_pose.py ^
  --detections outputs/yolo/detections.csv ^
  --poses outputs/pose/pose_detections.csv ^
  --output outputs/detections_with_pose.csv
```

4. Route automatic QC and preview the VLM audit:

```cmd
python scripts/qc/auto_qc_router.py ^
  --input outputs/detections_with_pose.csv ^
  --output outputs/qc/auto_qc_routes.csv ^
  --audit-medium-conf --audit-pose-missing

python scripts/qc/vlm_audit_images_siliconflow.py ^
  --routes outputs/qc/auto_qc_routes.csv ^
  --output outputs/qc/vlm_to_audit_preview.csv ^
  --dry-run
```

5. Run a small resumable VLM audit batch:

```cmd
set SILICONFLOW_API_KEY=your-key
set SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

python scripts/qc/vlm_audit_images_siliconflow.py ^
  --routes outputs/qc/auto_qc_routes.csv ^
  --output outputs/qc/vlm_audit_results.csv ^
  --model Qwen/Qwen3.5-4B --detail low --limit 200 --resume
```

6. Resolve final image labels and aggregate them to road segments:

```cmd
python scripts/qc/resolve_final_counts.py ^
  --routes outputs/qc/auto_qc_routes.csv ^
  --vlm outputs/qc/vlm_audit_results.csv ^
  --output outputs/qc/image_labels_final.csv

python scripts/qc/aggregate_final_to_segment_labels.py ^
  --input outputs/qc/image_labels_final.csv ^
  --output data/processed/segment_labels.csv ^
  --drop-unknown-road
```

The final aggregation requires a real `road_seg_id`. If it is still `UNKNOWN`, perform nearest-road
matching first.
