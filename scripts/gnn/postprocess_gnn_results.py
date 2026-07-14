from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


Y_BINS = [-np.inf, 0, 1, 3, 5, 10, np.inf]
Y_BIN_LABELS = ["0", "0-1", "1-3", "3-5", "5-10", ">10"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float))))


def ensure_out(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_labeled(pred: pd.DataFrame) -> pd.DataFrame:
    df = pred[pred["has_label"].astype(bool)].copy()
    df["error"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["error"].abs()
    df["sq_error"] = df["error"] ** 2
    df["y_bin"] = pd.cut(df["y_true"], bins=Y_BINS, labels=Y_BIN_LABELS)
    return df


def error_by_bin(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "y_bin"], observed=False)
        .agg(
            n=("y_true", "size"),
            y_mean=("y_true", "mean"),
            pred_mean=("y_pred", "mean"),
            mae=("abs_error", "mean"),
            rmse=("sq_error", rmse),
            bias=("error", "mean"),
        )
        .reset_index()
    )


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, part in df.groupby("split"):
        err = part["error"].to_numpy(float)
        true = part["y_true"].to_numpy(float)
        pred = part["y_pred"].to_numpy(float)
        ss_res = float(np.sum((pred - true) ** 2))
        ss_tot = float(np.sum((true - true.mean()) ** 2))
        rows.append(
            {
                "split": split,
                "n": int(len(part)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                "pearson": float(np.corrcoef(true, pred)[0, 1]) if len(part) > 1 else np.nan,
                "bias": float(np.mean(err)),
                "y_mean": float(np.mean(true)),
                "pred_mean": float(np.mean(pred)),
            }
        )
    return pd.DataFrame(rows).sort_values("split")


def plot_training(history: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], color="#2f5597", lw=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["epoch"], history["val_rmse"], label="Val RMSE", color="#c55a11", lw=2)
    axes[1].plot(history["epoch"], history["val_mae"], label="Val MAE", color="#70ad47", lw=2)
    best_idx = int(history["val_rmse"].astype(float).idxmin())
    best_row = history.loc[best_idx]
    axes[1].scatter([best_row["epoch"]], [best_row["val_rmse"]], color="#c00000", zorder=5)
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_true(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = {"train": "#8c8c8c", "val": "#4472c4", "test": "#ed7d31"}
    for split in ["train", "val", "test"]:
        part = df[df["split"].eq(split)]
        ax.scatter(part["y_true"], part["y_pred"], s=8, alpha=0.45, label=split, color=colors[split])
    lim_max = float(np.nanquantile(pd.concat([df["y_true"], df["y_pred"]]), 0.995))
    lim_max = max(lim_max, 1.0)
    ax.plot([0, lim_max], [0, lim_max], color="black", lw=1, linestyle="--")
    ax.set_xlim(-0.1, lim_max)
    ax.set_ylim(-0.1, lim_max)
    ax.set_title("Predicted vs True")
    ax.set_xlabel("True pedestrian intensity")
    ax.set_ylabel("Predicted pedestrian intensity")
    ax.legend(markerscale=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_residual_hist(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    clip = float(np.nanquantile(df["error"].abs(), 0.995))
    bins = np.linspace(-clip, clip, 60)
    for split, color in [("train", "#8c8c8c"), ("val", "#4472c4"), ("test", "#ed7d31")]:
        part = df[df["split"].eq(split)]
        ax.hist(part["error"].clip(-clip, clip), bins=bins, alpha=0.45, label=split, color=color)
    ax.axvline(0, color="black", lw=1)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Prediction error (pred - true)")
    ax.set_ylabel("Node count")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_bin(bin_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    splits = ["train", "val", "test"]
    x = np.arange(len(Y_BIN_LABELS))
    width = 0.25
    colors = {"train": "#8c8c8c", "val": "#4472c4", "test": "#ed7d31"}
    for i, split in enumerate(splits):
        part = bin_df[bin_df["split"].eq(split)].set_index("y_bin").reindex(Y_BIN_LABELS)
        ax.bar(x + (i - 1) * width, part["mae"].to_numpy(float), width=width, label=split, color=colors[split])
    ax.set_xticks(x)
    ax.set_xticklabels(Y_BIN_LABELS)
    ax.set_title("MAE by True Intensity Bin")
    ax.set_xlabel("True intensity bin")
    ax.set_ylabel("MAE")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_tile_rmse(tile: pd.DataFrame, out: Path) -> None:
    labeled_tile = tile[tile["n"] > 0].copy()
    top = labeled_tile.sort_values("rmse", ascending=False).head(18).sort_values("rmse")
    labels = top["split"].astype(str) + ":" + top["tile_id"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, top["rmse"], color="#a23b72")
    ax.set_title("Top Tile RMSE")
    ax.set_xlabel("RMSE")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_prediction(pred: pd.DataFrame, out: Path) -> None:
    core = pred[pred["core_mask"].astype(bool)].copy()
    vmax = float(np.nanquantile(core["y_pred"], 0.99))
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        core["centroid_lon"],
        core["centroid_lat"],
        c=core["y_pred"],
        s=2.5,
        cmap="viridis",
        vmin=0,
        vmax=max(vmax, 1e-6),
        alpha=0.8,
        linewidths=0,
    )
    ax.set_title("Spatial Predicted Intensity (Core Nodes)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="Predicted intensity")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_residual(df: pd.DataFrame, out: Path) -> None:
    clip = float(np.nanquantile(df["error"].abs(), 0.99))
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        df["centroid_lon"],
        df["centroid_lat"],
        c=df["error"].clip(-clip, clip),
        s=5,
        cmap="coolwarm",
        vmin=-clip,
        vmax=clip,
        alpha=0.8,
        linewidths=0,
    )
    ax.set_title("Spatial Residuals on Labeled Nodes")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="Prediction error")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(history: pd.DataFrame, df: pd.DataFrame, bin_df: pd.DataFrame, pred: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].plot(history["epoch"], history["train_loss"], color="#2f5597", lw=2, label="Train loss")
    axes[0, 0].plot(history["epoch"], history["val_rmse"], color="#c55a11", lw=2, label="Val RMSE")
    axes[0, 0].set_title("Training Curve")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    val_test = df[df["split"].isin(["val", "test"])]
    axes[0, 1].scatter(val_test["y_true"], val_test["y_pred"], s=10, alpha=0.55, color="#4472c4")
    lim_max = float(np.nanquantile(pd.concat([val_test["y_true"], val_test["y_pred"]]), 0.995))
    axes[0, 1].plot([0, lim_max], [0, lim_max], color="black", lw=1, linestyle="--")
    axes[0, 1].set_xlim(-0.1, lim_max)
    axes[0, 1].set_ylim(-0.1, lim_max)
    axes[0, 1].set_title("Val/Test Predicted vs True")
    axes[0, 1].set_xlabel("True")
    axes[0, 1].set_ylabel("Predicted")
    axes[0, 1].grid(alpha=0.25)

    val_test_bins = bin_df[bin_df["split"].isin(["val", "test"])]
    x = np.arange(len(Y_BIN_LABELS))
    for i, split in enumerate(["val", "test"]):
        part = val_test_bins[val_test_bins["split"].eq(split)].set_index("y_bin").reindex(Y_BIN_LABELS)
        axes[1, 0].bar(x + (i - 0.5) * 0.35, part["mae"].to_numpy(float), width=0.35, label=split)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(Y_BIN_LABELS)
    axes[1, 0].set_title("Val/Test MAE by Intensity Bin")
    axes[1, 0].set_xlabel("True bin")
    axes[1, 0].set_ylabel("MAE")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    core = pred[pred["core_mask"].astype(bool)].copy()
    vmax = float(np.nanquantile(core["y_pred"], 0.99))
    sc = axes[1, 1].scatter(
        core["centroid_lon"],
        core["centroid_lat"],
        c=core["y_pred"],
        s=1.5,
        cmap="viridis",
        vmin=0,
        vmax=max(vmax, 1e-6),
        alpha=0.8,
        linewidths=0,
    )
    axes[1, 1].set_title("Spatial Prediction")
    axes[1, 1].set_xlabel("Longitude")
    axes[1, 1].set_ylabel("Latitude")
    axes[1, 1].set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=axes[1, 1], label="Predicted")

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(run_dir: Path, out_dir: Path, metrics: dict, config: dict, split_df: pd.DataFrame, bin_df: pd.DataFrame) -> None:
    feature_transform = config.get("feature_transform", {})
    log_features = feature_transform.get("log_features", [])
    args = config.get("args", {})

    lines = [
        "# GNN Run Postprocess Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Metrics",
        "",
        "| split | n | MAE | RMSE | R2 | Pearson | bias |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in split_df.iterrows():
        lines.append(
            f"| {row['split']} | {int(row['n'])} | {row['mae']:.6f} | {row['rmse']:.6f} | "
            f"{row['r2']:.6f} | {row['pearson']:.6f} | {row['bias']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Best Training Configuration",
            "",
            f"- best_epoch: {metrics.get('best_epoch')}",
            f"- hidden_dim: {args.get('hidden_dim')}",
            f"- num_layers: {args.get('num_layers')}",
            f"- dropout: {args.get('dropout')}",
            f"- loss: {args.get('loss')}",
            f"- target_transform: {args.get('target_transform')}",
            f"- loss_weight: {args.get('loss_weight')}",
            f"- feature_transform: {feature_transform.get('mode', 'standard')}",
            f"- feature_clip_quantile: {feature_transform.get('clip_quantile', '')}",
            "",
            "## Feature Transform",
            "",
            f"Log-transformed and clipped features: {len(log_features)}",
        ]
    )
    lines.extend([f"- {name}" for name in log_features])

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The model improves the validation split over the previous standard z-score-only baseline.",
            "- High-intensity bins still show underprediction, so hotspot interpretation should keep residual uncertainty visible.",
            "- All feature and target transforms are fit only on the train split, then applied to val/test.",
            "",
            "## Generated Files",
            "",
            "- `fig_dashboard.png`",
            "- `fig_training_curve.png`",
            "- `fig_pred_vs_true.png`",
            "- `fig_residual_hist.png`",
            "- `fig_error_by_y_bin.png`",
            "- `fig_tile_rmse_top.png`",
            "- `fig_spatial_prediction.png`",
            "- `fig_spatial_residual.png`",
            "- `labeled_predictions_with_error.csv`",
            "- `error_by_y_bin.csv`",
            "- `split_summary.csv`",
            "- `tile_metrics_ranked.csv`",
        ]
    )
    (out_dir / "postprocess_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess and visualize GNN run outputs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = ensure_out(Path(args.output_dir) if args.output_dir else run_dir / "postprocess")

    metrics = read_json(run_dir / "metrics.json")
    config = read_json(run_dir / "run_config.json")
    pred = pd.read_csv(run_dir / "node_predictions.csv")
    tile = pd.read_csv(run_dir / "tile_metrics.csv")
    history = pd.read_csv(run_dir / "training_history.csv")

    labeled = prepare_labeled(pred)
    bin_df = error_by_bin(labeled)
    split_df = split_summary(labeled)
    tile_ranked = tile[tile["n"] > 0].copy().sort_values(["rmse", "mae"], ascending=False)

    labeled.to_csv(out_dir / "labeled_predictions_with_error.csv", index=False, encoding="utf-8-sig")
    bin_df.to_csv(out_dir / "error_by_y_bin.csv", index=False, encoding="utf-8-sig")
    split_df.to_csv(out_dir / "split_summary.csv", index=False, encoding="utf-8-sig")
    tile_ranked.to_csv(out_dir / "tile_metrics_ranked.csv", index=False, encoding="utf-8-sig")

    summary = {
        "metrics": metrics,
        "split_summary": split_df.to_dict(orient="records"),
        "top_tile_rmse": tile_ranked.head(10).to_dict(orient="records"),
        "feature_transform": config.get("feature_transform", {}),
        "args": config.get("args", {}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_training(history, out_dir / "fig_training_curve.png")
    plot_pred_vs_true(labeled, out_dir / "fig_pred_vs_true.png")
    plot_residual_hist(labeled, out_dir / "fig_residual_hist.png")
    plot_error_by_bin(bin_df, out_dir / "fig_error_by_y_bin.png")
    plot_top_tile_rmse(tile, out_dir / "fig_tile_rmse_top.png")
    plot_spatial_prediction(pred, out_dir / "fig_spatial_prediction.png")
    plot_spatial_residual(labeled, out_dir / "fig_spatial_residual.png")
    plot_dashboard(history, labeled, bin_df, pred, out_dir / "fig_dashboard.png")
    write_report(run_dir, out_dir, metrics, config, split_df, bin_df)

    print(json.dumps({"output_dir": str(out_dir), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
