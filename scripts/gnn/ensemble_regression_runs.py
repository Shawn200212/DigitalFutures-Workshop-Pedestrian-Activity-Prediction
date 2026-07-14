from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLS = ["split", "tile_id", "node_index", "seg_id"]
COPY_COLS = [
    "split",
    "tile_id",
    "node_index",
    "seg_id",
    "centroid_lon",
    "centroid_lat",
    "core_mask",
    "has_label",
    "num_pano_images",
    "y_true",
]


def regression_metrics(df: pd.DataFrame, split: str) -> dict:
    part = df[df["split"].eq(split) & df["has_label"].astype(bool)].copy()
    y = part["y_true"].to_numpy(float)
    pred = part["y_pred"].to_numpy(float)
    err = pred - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        f"{split}_n": int(len(part)),
        f"{split}_mae": float(np.mean(np.abs(err))),
        f"{split}_rmse": float(np.sqrt(np.mean(err**2))),
        f"{split}_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        f"{split}_pearson": float(np.corrcoef(y, pred)[0, 1]) if len(part) > 1 else float("nan"),
    }


def tile_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, tile_id), part in df[df["has_label"].astype(bool)].groupby(["split", "tile_id"]):
        y = part["y_true"].to_numpy(float)
        pred = part["y_pred"].to_numpy(float)
        err = pred - y
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        rows.append(
            {
                "split": split,
                "tile_id": tile_id,
                "n": int(len(part)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                "pearson": float(np.corrcoef(y, pred)[0, 1]) if len(part) > 1 else float("nan"),
                "y_mean": float(np.mean(y)),
                "pred_mean": float(np.mean(pred)),
                "y_min": float(np.min(y)),
                "y_max": float(np.max(y)),
                "pred_min": float(np.min(pred)),
                "pred_max": float(np.max(pred)),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "tile_id"])


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Average continuous predictions from multiple regression runs.")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.run_dirs]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(run_dirs[0] / "node_predictions.csv")
    ensemble = base[COPY_COLS].copy()
    pred_cols = []
    model_metrics = []

    for idx, run_dir in enumerate(run_dirs, start=1):
        pred_path = run_dir / "node_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        pred = pd.read_csv(pred_path)
        if pred.duplicated(KEY_COLS).any():
            raise ValueError(f"Duplicate prediction keys in run: {run_dir}")
        col = f"y_pred_model_{idx:02d}"
        pred_small = pred[KEY_COLS + ["y_pred"]].rename(columns={"y_pred": col})
        ensemble = ensemble.merge(pred_small, on=KEY_COLS, how="left", validate="one_to_one")
        if ensemble[col].isna().any():
            missing = int(ensemble[col].isna().sum())
            raise ValueError(f"Missing {missing} predictions after key alignment for run: {run_dir}")
        pred_cols.append(col)
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            row = {"run_dir": str(run_dir)}
            row.update(read_json(metrics_path))
            model_metrics.append(row)

    ensemble["y_pred"] = ensemble[pred_cols].mean(axis=1)
    ensemble["y_pred_std"] = ensemble[pred_cols].std(axis=1)
    ensemble["y_pred_model_space"] = np.nan
    ensemble.to_csv(out_dir / "node_predictions.csv", index=False, encoding="utf-8-sig")

    metrics = {"ensemble_size": len(run_dirs), "run_dirs": [str(p) for p in run_dirs]}
    for split in ["train", "val", "test"]:
        metrics.update(regression_metrics(ensemble, split))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    tile_metrics(ensemble).to_csv(out_dir / "tile_metrics.csv", index=False, encoding="utf-8-sig")

    if model_metrics:
        pd.DataFrame(model_metrics).to_csv(out_dir / "member_metrics.csv", index=False, encoding="utf-8-sig")

    config = {}
    config_path = run_dirs[0] / "run_config.json"
    if config_path.exists():
        config = read_json(config_path)
    config["ensemble"] = {"run_dirs": [str(p) for p in run_dirs], "method": "mean_y_pred"}
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    history_parts = []
    for idx, run_dir in enumerate(run_dirs, start=1):
        history_path = run_dir / "training_history.csv"
        if history_path.exists():
            hist = pd.read_csv(history_path)
            hist.insert(0, "member", idx)
            history_parts.append(hist)
    if history_parts:
        history = pd.concat(history_parts, ignore_index=True)
        mean_history = (
            history.groupby("epoch", as_index=False)
            .agg(train_loss=("train_loss", "mean"), val_rmse=("val_rmse", "mean"), val_mae=("val_mae", "mean"))
            .sort_values("epoch")
        )
        mean_history.to_csv(out_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame({"epoch": [1], "train_loss": [np.nan], "val_rmse": [metrics["val_rmse"]], "val_mae": [metrics["val_mae"]]}).to_csv(
            out_dir / "training_history.csv", index=False, encoding="utf-8-sig"
        )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved ensemble run to: {out_dir}")


if __name__ == "__main__":
    main()
