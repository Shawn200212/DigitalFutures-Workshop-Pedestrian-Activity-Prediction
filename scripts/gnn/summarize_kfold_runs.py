from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REGRESSION_METRICS = [
    "train_mae",
    "train_rmse",
    "train_r2",
    "train_pearson",
    "val_mae",
    "val_rmse",
    "val_r2",
    "val_pearson",
    "test_mae",
    "test_rmse",
    "test_r2",
    "test_pearson",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_regression(root: Path, k_folds: int) -> pd.DataFrame:
    rows = []
    for fold in range(k_folds):
        fold_dir = root / f"fold_{fold}"
        metrics = read_json(fold_dir / "metrics.json")
        config = read_json(fold_dir / "run_config.json")
        split = read_json(fold_dir / "split.json")
        args = config.get("args", {})
        row = {
            "fold": fold,
            "output_dir": str(fold_dir),
            "best_epoch": metrics.get("best_epoch"),
            "train_tiles": len(split.get("train", [])),
            "val_tiles": len(split.get("val", [])),
            "test_tiles": len(split.get("test", [])),
            "feature_transform": args.get("feature_transform"),
            "feature_clip_quantile": args.get("feature_clip_quantile"),
            "target_transform": args.get("target_transform"),
            "loss": args.get("loss"),
            "loss_weight": args.get("loss_weight"),
            "hidden_dim": args.get("hidden_dim"),
            "num_layers": args.get("num_layers"),
            "dropout": args.get("dropout"),
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def collect_posthoc(root: Path, k_folds: int) -> pd.DataFrame:
    rows = []
    for fold in range(k_folds):
        path = root / f"fold_{fold}" / "posthoc_discrete_calibrated" / "posthoc_discrete_summary.csv"
        df = pd.read_csv(path)
        df.insert(0, "fold", fold)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def summarize_numeric(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if not group_cols:
        row = {}
        for col in metric_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std())
            row[f"{col}_min"] = float(values.min())
            row[f"{col}_max"] = float(values.max())
        return pd.DataFrame([row])
    summary = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = [
        "_".join([str(x) for x in col if str(x)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def plot_regression(folds: pd.DataFrame, out: Path) -> None:
    x = np.arange(len(folds))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(x - 0.2, folds["val_rmse"], width=0.4, label="val", color="#4472c4")
    axes[0].bar(x + 0.2, folds["test_rmse"], width=0.4, label="test", color="#ed7d31")
    axes[0].axhline(folds["test_rmse"].mean(), color="#ed7d31", linestyle="--", lw=1)
    axes[0].set_title("K-Fold RMSE")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(folds["fold"])
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - 0.2, folds["val_r2"], width=0.4, label="val", color="#4472c4")
    axes[1].bar(x + 0.2, folds["test_r2"], width=0.4, label="test", color="#ed7d31")
    axes[1].axhline(folds["test_r2"].mean(), color="#ed7d31", linestyle="--", lw=1)
    axes[1].set_title("K-Fold R2")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("R2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(folds["fold"])
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_posthoc_binary(posthoc: pd.DataFrame, out: Path) -> None:
    binary = posthoc[posthoc["task"].isin(["binary_y3", "binary_y5"])].copy()
    tasks = ["binary_y3", "binary_y5"]
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.22
    metrics = [
        ("test_auc_ovr", "AUC", "#70ad47"),
        ("test_balanced_accuracy", "Balanced Acc", "#4472c4"),
        ("test_macro_f1", "Macro F1", "#ed7d31"),
    ]
    for i, (col, label, color) in enumerate(metrics):
        means = binary.groupby("task")[col].mean().reindex(tasks)
        stds = binary.groupby("task")[col].std().reindex(tasks)
        ax.bar(x + (i - 1) * width, means, width=width, yerr=stds, label=label, color=color, capsize=4)
    ax.set_title("K-Fold Posthoc Binary Classification")
    ax.set_xlabel("Task")
    ax.set_ylabel("Mean test metric")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt_mean_std(summary: pd.Series, key: str) -> str:
    return f"{summary[f'{key}_mean']:.4f} +/- {summary[f'{key}_std']:.4f}"


def write_report(root: Path, folds: pd.DataFrame, reg_summary: pd.DataFrame, post_summary: pd.DataFrame) -> None:
    reg = reg_summary.iloc[0]
    binary = post_summary[post_summary["task"].isin(["binary_y3", "binary_y5"])].copy()
    ternary = post_summary[post_summary["task"].str.startswith("ternary")].copy()

    best_test = folds.sort_values("test_r2", ascending=False).iloc[0]
    best_val = folds.sort_values("val_rmse", ascending=True).iloc[0]

    lines = [
        "# GraphSAGE Best Method 5-Fold Result",
        "",
        "Training keeps the original continuous target and performs posthoc calibrated binary/ternary classification after each fold.",
        "",
        "## Configuration",
        "",
        "- model: GraphSAGE",
        "- hidden_dim: 64",
        "- num_layers: 2",
        "- dropout: 0.15",
        "- feature_transform: skew_log1p_clip",
        "- feature_clip_quantile: 0.999",
        "- target_transform: log1p_zscore",
        "- loss: Huber",
        "- loss_weight: y_quantile",
        "- high_y_weight: 1.75",
        "- very_high_y_weight: 2.5",
        "- k_folds: 5",
        "",
        "## Regression Summary",
        "",
        "| metric | mean +/- std | min | max |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ["val_rmse", "val_r2", "val_pearson", "test_rmse", "test_r2", "test_pearson", "test_mae"]:
        lines.append(f"| {key} | {fmt_mean_std(reg, key)} | {reg[f'{key}_min']:.4f} | {reg[f'{key}_max']:.4f} |")

    lines.extend(
        [
            "",
            "## Fold Results",
            "",
            "| fold | val_rmse | val_r2 | test_rmse | test_r2 | test_pearson | best_epoch |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in folds.iterrows():
        lines.append(
            f"| {int(row['fold'])} | {row['val_rmse']:.4f} | {row['val_r2']:.4f} | "
            f"{row['test_rmse']:.4f} | {row['test_r2']:.4f} | {row['test_pearson']:.4f} | {int(row['best_epoch'])} |"
        )

    lines.extend(
        [
            "",
            "## Posthoc Binary Classification",
            "",
            "| task | test AUC mean +/- std | test balanced acc mean +/- std | test macro F1 mean +/- std |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in binary.iterrows():
        lines.append(
            f"| {row['task']} | {fmt_mean_std(row, 'test_auc_ovr')} | "
            f"{fmt_mean_std(row, 'test_balanced_accuracy')} | {fmt_mean_std(row, 'test_macro_f1')} |"
        )

    lines.extend(
        [
            "",
            "## Posthoc Ternary Classification",
            "",
            "| task | test balanced acc mean +/- std | test macro F1 mean +/- std |",
            "| --- | ---: | ---: |",
        ]
    )
    for _, row in ternary.iterrows():
        lines.append(f"| {row['task']} | {fmt_mean_std(row, 'test_balanced_accuracy')} | {fmt_mean_std(row, 'test_macro_f1')} |")

    lines.extend(
        [
            "",
            "## Best Fold Pointers",
            "",
            f"- best test R2 fold: `fold_{int(best_test['fold'])}` with test R2 `{best_test['test_r2']:.4f}`.",
            f"- best validation RMSE fold: `fold_{int(best_val['fold'])}` with val RMSE `{best_val['val_rmse']:.4f}`.",
            "",
            "## Generated Files",
            "",
            "- `kfold_regression_fold_metrics.csv`",
            "- `kfold_regression_summary.csv`",
            "- `kfold_posthoc_fold_metrics.csv`",
            "- `kfold_posthoc_summary.csv`",
            "- `fig_kfold_regression_metrics.png`",
            "- `fig_kfold_posthoc_binary.png`",
        ]
    )
    (root / "kfold_best_method_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fixed-config k-fold GraphSAGE runs.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--k-folds", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root)
    folds = collect_regression(root, args.k_folds)
    posthoc = collect_posthoc(root, args.k_folds)

    reg_summary = summarize_numeric(folds, [], REGRESSION_METRICS)
    post_metric_cols = [
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "val_auc_ovr",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_auc_ovr",
    ]
    post_summary = summarize_numeric(posthoc, ["task"], post_metric_cols)

    folds.to_csv(root / "kfold_regression_fold_metrics.csv", index=False, encoding="utf-8-sig")
    reg_summary.to_csv(root / "kfold_regression_summary.csv", index=False, encoding="utf-8-sig")
    posthoc.to_csv(root / "kfold_posthoc_fold_metrics.csv", index=False, encoding="utf-8-sig")
    post_summary.to_csv(root / "kfold_posthoc_summary.csv", index=False, encoding="utf-8-sig")

    plot_regression(folds, root / "fig_kfold_regression_metrics.png")
    plot_posthoc_binary(posthoc, root / "fig_kfold_posthoc_binary.png")
    write_report(root, folds, reg_summary, post_summary)

    print(json.dumps({"root": str(root), "regression": reg_summary.to_dict(orient="records")[0]}, ensure_ascii=False, indent=2))
    print(post_summary.to_string(index=False))


if __name__ == "__main__":
    main()
