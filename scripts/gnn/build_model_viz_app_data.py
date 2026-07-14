from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def round_or_none(value: object, digits: int = 6) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def build_points(run_dir: Path) -> list[dict]:
    base = pd.read_csv(run_dir / "node_predictions.csv")
    y5_path = run_dir / "posthoc_discrete_calibrated" / "binary_y5" / "node_posthoc_class_predictions.csv"
    y3_path = run_dir / "posthoc_discrete_calibrated" / "binary_y3" / "node_posthoc_class_predictions.csv"
    y5 = load_optional_csv(y5_path)
    y3 = load_optional_csv(y3_path)

    key_cols = ["split", "tile_id", "node_index", "seg_id"]
    df = base.copy()
    if y5 is not None:
        y5_cols = key_cols + ["binary_y5_class_pred", "binary_y5_high_score"]
        df = df.merge(y5[y5_cols], on=key_cols, how="left")
    if y3 is not None:
        y3_cols = key_cols + ["binary_y3_class_pred", "binary_y3_high_score"]
        df = df.merge(y3[y3_cols], on=key_cols, how="left")

    df["residual"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["residual"].abs()
    points = []
    for row in df.itertuples(index=False):
        points.append(
            {
                "split": row.split,
                "tile": row.tile_id,
                "node": int(row.node_index),
                "seg": int(row.seg_id),
                "lon": round_or_none(row.centroid_lon, 7),
                "lat": round_or_none(row.centroid_lat, 7),
                "hasLabel": bool(row.has_label),
                "pano": int(row.num_pano_images),
                "true": round_or_none(row.y_true, 4),
                "pred": round_or_none(row.y_pred, 4),
                "residual": round_or_none(row.residual, 4),
                "absError": round_or_none(row.abs_error, 4),
                "y5Score": round_or_none(getattr(row, "binary_y5_high_score", None), 4),
                "y5Class": int(getattr(row, "binary_y5_class_pred", 0)) if not pd.isna(getattr(row, "binary_y5_class_pred", np.nan)) else 0,
                "y3Score": round_or_none(getattr(row, "binary_y3_high_score", None), 4),
                "y3Class": int(getattr(row, "binary_y3_class_pred", 0)) if not pd.isna(getattr(row, "binary_y3_class_pred", np.nan)) else 0,
            }
        )
    return points


def build_tile_summary(run_dir: Path) -> list[dict]:
    tile = pd.read_csv(run_dir / "tile_metrics.csv")
    rows = []
    for row in tile.itertuples(index=False):
        rows.append(
            {
                "split": row.split,
                "tile": row.tile_id,
                "n": int(row.n),
                "mae": round_or_none(row.mae, 4),
                "rmse": round_or_none(row.rmse, 4),
                "r2": round_or_none(row.r2, 4),
                "pearson": round_or_none(row.pearson, 4),
                "yMean": round_or_none(row.y_mean, 4),
                "predMean": round_or_none(row.pred_mean, 4),
            }
        )
    return rows


def build_posthoc_summary(run_dir: Path) -> list[dict]:
    path = run_dir / "posthoc_discrete_calibrated" / "posthoc_discrete_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "task": row.task,
                "cuts": row.cuts,
                "predictionCuts": row.prediction_cuts,
                "testAuc": round_or_none(getattr(row, "test_auc_ovr", None), 4),
                "testBalancedAcc": round_or_none(row.test_balanced_accuracy, 4),
                "testMacroF1": round_or_none(row.test_macro_f1, 4),
                "valBalancedAcc": round_or_none(row.val_balanced_accuracy, 4),
            }
        )
    return rows


def build_importance(run_dir: Path) -> dict:
    imp_dir = run_dir / "feature_importance"
    group_path = imp_dir / "group_permutation_importance.csv"
    feature_path = imp_dir / "feature_permutation_importance.csv"
    corr_path = imp_dir / "feature_correlations.csv"

    def rows(path: Path, name_col: str) -> list[dict]:
        if not path.exists():
            return []
        df = pd.read_csv(path)
        out = []
        for row in df.itertuples(index=False):
            item = {
                "name": str(getattr(row, name_col)),
                "rmseDelta": round_or_none(getattr(row, "rmse_delta", None), 6),
                "r2Drop": round_or_none(getattr(row, "r2_drop", None), 6),
            }
            if hasattr(row, "n_features"):
                item["nFeatures"] = int(row.n_features)
            if hasattr(row, "spearman_corr"):
                item["spearman"] = round_or_none(row.spearman_corr, 6)
            out.append(item)
        return out

    correlations = []
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        for row in corr.itertuples(index=False):
            correlations.append(
                {
                    "name": str(row.feature),
                    "pearson": round_or_none(row.pearson_corr, 6),
                    "spearman": round_or_none(row.spearman_corr, 6),
                    "absSpearman": round_or_none(row.abs_spearman_corr, 6),
                }
            )
    return {
        "groups": rows(group_path, "name"),
        "features": rows(feature_path, "name"),
        "correlations": correlations,
    }


def numeric_summary(points: list[dict]) -> dict:
    lon = np.array([p["lon"] for p in points if p["lon"] is not None], dtype=float)
    lat = np.array([p["lat"] for p in points if p["lat"] is not None], dtype=float)
    pred = np.array([p["pred"] for p in points if p["pred"] is not None], dtype=float)
    true = np.array([p["true"] for p in points if p["true"] is not None], dtype=float)
    residual = np.array([p["residual"] for p in points if p["residual"] is not None], dtype=float)
    return {
        "bounds": {
            "minLon": round(float(lon.min()), 7),
            "maxLon": round(float(lon.max()), 7),
            "minLat": round(float(lat.min()), 7),
            "maxLat": round(float(lat.max()), 7),
        },
        "predP99": round(float(np.nanquantile(pred, 0.99)), 4),
        "predP95": round(float(np.nanquantile(pred, 0.95)), 4),
        "trueP99": round(float(np.nanquantile(true, 0.99)), 4) if len(true) else None,
        "residualAbsP99": round(float(np.nanquantile(np.abs(residual), 0.99)), 4) if len(residual) else None,
        "pointCount": len(points),
        "labeledCount": int(sum(1 for p in points if p["hasLabel"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build browser-ready data for model visualization app.")
    parser.add_argument("--run-dir", default="outputs/gnn/full_retrain_best_20260703/fixed_split_champion")
    parser.add_argument("--kfold-dir", default="outputs/gnn/full_retrain_best_20260703/kfold_best")
    parser.add_argument("--output-dir", default="outputs/gnn/model_visualization_app_20260703/data")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    kfold_dir = Path(args.kfold_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = build_points(run_dir)
    metrics = read_json(run_dir / "metrics.json")
    config = read_json(run_dir / "run_config.json")
    kfold_report = (kfold_dir / "kfold_best_method_report.md").read_text(encoding="utf-8") if kfold_dir.exists() else ""
    kfold_summary = pd.read_csv(kfold_dir / "kfold_regression_summary.csv").to_dict(orient="records") if (kfold_dir / "kfold_regression_summary.csv").exists() else []
    kfold_posthoc = pd.read_csv(kfold_dir / "kfold_posthoc_summary.csv").to_dict(orient="records") if (kfold_dir / "kfold_posthoc_summary.csv").exists() else []

    data = {
        "meta": {
            "title": "Shanghai Pedestrian Intensity GraphSAGE Visualization",
            "runDir": str(run_dir),
            "kfoldDir": str(kfold_dir),
            "model": "GraphSAGE continuous regression + posthoc calibrated classification",
            "createdFrom": "fixed_split_champion",
        },
        "metrics": metrics,
        "runConfig": {
            "featureNames": config.get("feature_names", []),
            "featureTransform": config.get("feature_transform", {}),
            "targetTransform": config.get("target_transform", {}),
            "lossWeight": config.get("loss_weight", {}),
            "labelCounts": config.get("label_counts", {}),
        },
        "summary": numeric_summary(points),
        "points": points,
        "tiles": build_tile_summary(run_dir),
        "posthoc": build_posthoc_summary(run_dir),
        "importance": build_importance(run_dir),
        "kfoldSummary": kfold_summary,
        "kfoldPosthoc": kfold_posthoc,
        "kfoldReport": kfold_report,
    }

    payload = "window.MODEL_VIZ_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    (out_dir / "app_data.js").write_text(payload, encoding="utf-8")
    print(json.dumps({"output": str(out_dir / "app_data.js"), "points": len(points)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
