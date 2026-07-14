# SVI Pedestrian Tile Pipeline

Current project workflow:

1. Build a deduplicated tile image manifest.
2. Run YOLO11 person counting.
3. Run YOLO11 pose estimation.
4. Merge count and pose outputs.
5. Route uncertain cases to QC.
6. Audit routed images with SiliconFlow VLM using `Qwen/Qwen3.5-4B`.
7. Resolve final image-level labels and optionally aggregate to road segments.

## Active Entry Points

Generate or refresh the unique tile manifest:

```powershell
python scripts\prepare_yolo_manifest_from_tiles.py `
  --dataset-root data\images `
  --output outputs\manifests\master_yolo_manifest_unique.csv `
  --summary outputs\manifests\master_yolo_manifest_summary.json `
  --dedupe-key path
```

Run YOLO person detection:

```powershell
python app\batch_detect.py `
  --manifest outputs\manifests\master_yolo_manifest_unique.csv `
  --output-dir outputs\yolo `
  --model yolo11n.pt `
  --conf 0.25 `
  --imgsz 960 `
  --device 0 `
  --save-vis
```

Run pose estimation:

```powershell
python app\batch_pose.py `
  --manifest outputs\manifests\master_yolo_manifest_unique.csv `
  --output-dir outputs\pose `
  --model yolo11n-pose.pt `
  --conf 0.08 `
  --imgsz 960 `
  --device 0 `
  --save-vis
```

Merge detection and pose:

```powershell
python app\merge_detect_pose.py `
  --detections outputs\yolo\detections.csv `
  --poses outputs\pose\pose_detections.csv `
  --output outputs\detections_with_pose.csv
```

Create QC routes:

```powershell
python scripts\qc\auto_qc_router.py `
  --input outputs\detections_with_pose.csv `
  --output outputs\qc\auto_qc_routes.csv `
  --audit-medium-conf `
  --audit-pose-missing `
  --audit-pose-low-recall
```

Run SiliconFlow VLM audit:

```powershell
$env:SILICONFLOW_API_KEY="your_key"

python scripts\qc\vlm_audit_images_siliconflow.py `
  --routes outputs\qc\auto_qc_routes.csv `
  --output outputs\qc\vlm_audit_results.csv `
  --model Qwen/Qwen3.5-4B `
  --resume `
  --no-response-format-json
```

Resolve final image labels:

```powershell
python scripts\qc\resolve_final_counts.py `
  --routes outputs\qc\auto_qc_routes.csv `
  --vlm outputs\qc\vlm_audit_results.csv `
  --output outputs\qc\image_labels_final.csv
```

Aggregate final labels to road segments when `road_seg_id` is available:

```powershell
python scripts\qc\aggregate_final_to_segment_labels.py `
  --input outputs\qc\image_labels_final.csv `
  --output data\processed\segment_labels.csv `
  --drop-unknown-road
```

## Active Code

- `app/batch_detect.py`: YOLO11 person count batch inference.
- `app/batch_pose.py`: YOLO11 pose batch inference.
- `app/merge_detect_pose.py`: merge count and pose tables.
- `scripts/gnn/train_graphsage_npz.py`: train GraphSAGE on `dataset/SHdataset/tile_*.npz`.
- `scripts/prepare_yolo_manifest_from_tiles.py`: build deduplicated tile manifest.
- `scripts/make_delta_manifest.py`: create new batches by excluding already processed images.
- `scripts/run_yolo_vlm_pipeline_all_tiles.py`: all-tiles orchestration script.
- `scripts/qc/auto_qc_router.py`: route images to auto-accept, VLM, or human review.
- `scripts/qc/vlm_audit_images_siliconflow.py`: SiliconFlow VLM audit.
- `scripts/qc/resolve_final_counts.py`: final image-level label resolution.
- `scripts/qc/make_final_qc_summary.py`: final QC summary.
- `scripts/qc/aggregate_final_to_segment_labels.py`: road-segment aggregation.

Legacy FastAPI, Docker, `yolov8n.pt` demo, and OpenAI VLM files have been moved to:

```text
deprecated_legacy_demo_20260702/
```

## SHdataset GraphSAGE Training

`dataset/SHdataset` contains prebuilt graph samples. The model input is only `x`,
with `edge_index` and `edge_weight` used for message passing. `pos`,
`seg_id`, and `num_pano_images` are metadata and must not be used as model
features. Training loss is computed only on `core_mask == True` nodes with a
finite `y`.

The current trainer is configured for the later 100-tile dataset by default:
`--tile-start 1 --tile-end 100`, graph-level splits, `log1p_zscore` target
normalization, gradient clipping, and persisted split/normalizer files.

Run the current tile 061-100 dataset for validation:

```powershell
& 'C:\Program Files (x86)\miniconda3\envs\svi_ped\python.exe' `
  scripts\gnn\train_graphsage_npz.py `
  --dataset-root dataset\SHdataset `
  --output-dir outputs\gnn\tile061_100_graphsage_optimized `
  --tile-start 61 `
  --tile-end 100 `
  --split-mode random `
  --epochs 300 `
  --patience 40 `
  --hidden-dim 128 `
  --num-layers 2 `
  --dropout 0.30 `
  --lr 0.001 `
  --weight-decay 0.0001 `
  --loss huber `
  --feature-transform skew_log1p_clip `
  --feature-clip-quantile 0.999 `
  --target-transform log1p_zscore `
  --loss-weight uniform `
  --grad-clip 2.0 `
  --device cuda
```

For the current best 100-tile GraphSAGE run, use:

```powershell
& 'C:\Program Files (x86)\miniconda3\envs\svi_ped\python.exe' `
  scripts\gnn\train_graphsage_npz.py `
  --dataset-root dataset\SHdataset `
  --output-dir outputs\gnn\tile001_100_graphsage_skewlog_clip999 `
  --tile-start 1 `
  --tile-end 100 `
  --split-mode from_file `
  --split-file outputs\gnn\tile001_100_graphsage_best_final\split.json `
  --epochs 400 `
  --patience 50 `
  --hidden-dim 64 `
  --num-layers 2 `
  --dropout 0.15 `
  --lr 0.001 `
  --weight-decay 0.0001 `
  --loss huber `
  --feature-transform skew_log1p_clip `
  --feature-clip-quantile 0.999 `
  --target-transform log1p_zscore `
  --loss-weight num_pano_sqrt `
  --grad-clip 2.0 `
  --device cuda
```

For 5-fold CV hyperparameter selection, the current robust winner is
`skew_log1p_clip + y_quantile`:

```powershell
& 'C:\Program Files (x86)\miniconda3\envs\svi_ped\python.exe' `
  scripts\gnn\train_graphsage_npz.py `
  --dataset-root dataset\SHdataset `
  --output-dir outputs\gnn\tile001_100_graphsage_cvbest_yquantile `
  --tile-start 1 `
  --tile-end 100 `
  --split-mode from_file `
  --split-file outputs\gnn\tile001_100_graphsage_best_final\split.json `
  --epochs 400 `
  --patience 50 `
  --hidden-dim 64 `
  --num-layers 2 `
  --dropout 0.15 `
  --lr 0.001 `
  --weight-decay 0.0001 `
  --loss huber `
  --feature-transform skew_log1p_clip `
  --feature-clip-quantile 0.999 `
  --target-transform log1p_zscore `
  --loss-weight y_quantile `
  --high-y-weight 1.75 `
  --very-high-y-weight 2.5 `
  --grad-clip 2.0 `
  --device cuda
```

The 5-fold sweep outputs are in:

```text
outputs/gnn/cv_sweep_graphsage_skewlog/
```

For discrete pedestrian intensity mapping, train a GraphSAGE classifier. The
most useful current hotspot setting is binary `y >= 5`:

```powershell
& 'C:\Program Files (x86)\miniconda3\envs\svi_ped\python.exe' `
  scripts\gnn\train_graphsage_classifier_npz.py `
  --dataset-root dataset\SHdataset `
  --output-dir outputs\gnn\classifier_binary_y5 `
  --tile-start 1 `
  --tile-end 100 `
  --split-mode from_file `
  --split-file outputs\gnn\tile001_100_graphsage_best_final\split.json `
  --task binary `
  --thresholds 5 `
  --class-weight balanced `
  --selection-metric val_auc_ovr `
  --epochs 300 `
  --patience 40 `
  --hidden-dim 64 `
  --num-layers 2 `
  --dropout 0.15 `
  --feature-transform skew_log1p_clip `
  --feature-clip-quantile 0.999 `
  --device cuda
```

Main classifier outputs:

```text
outputs/gnn/classifier_binary_y5/node_class_predictions.csv
outputs/gnn/classifier_binary_y5/tile_hotspot_probability_summary.csv
outputs/gnn/classifier_binary_y5/fig_spatial_hotspot_probability.png
```

For a more stable 100-tile estimate, run 5-fold validation by changing only
`--fold-index` from `0` to `4`:

```powershell
& 'C:\Program Files (x86)\miniconda3\envs\svi_ped\python.exe' `
  scripts\gnn\train_graphsage_npz.py `
  --dataset-root dataset\SHdataset `
  --output-dir outputs\gnn\tile001_100_graphsage_fold0 `
  --tile-start 1 `
  --tile-end 100 `
  --split-mode kfold `
  --k-folds 5 `
  --fold-index 0 `
  --epochs 300 `
  --patience 40 `
  --device cuda
```

Main outputs:

```text
outputs/gnn/<run_name>/best_model.pt
outputs/gnn/<run_name>/metrics.json
outputs/gnn/<run_name>/run_config.json
outputs/gnn/<run_name>/split.json
outputs/gnn/<run_name>/normalizer.npz
outputs/gnn/<run_name>/training_history.csv
outputs/gnn/<run_name>/node_predictions.csv
outputs/gnn/<run_name>/tile_metrics.csv
outputs/gnn/<run_name>/diagnostics.json
```
