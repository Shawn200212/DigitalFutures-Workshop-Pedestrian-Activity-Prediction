from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASKS = {
    "binary_y3": [3.0],
    "binary_y5": [5.0],
    "ternary_y1_y5": [1.0, 5.0],
    "ternary_q50_q85": [0.922244, 3.8],
}


def y_to_class(values: np.ndarray, cuts: list[float]) -> np.ndarray:
    out = np.zeros(len(values), dtype=int)
    for cut in cuts:
        out += values >= cut
    return out


def labels_for(cuts: list[float]) -> list[str]:
    if len(cuts) == 1:
        return [f"low_<_{cuts[0]:g}", f"high_>=_{cuts[0]:g}"]
    labels = [f"<{cuts[0]:g}"]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        labels.append(f"{lo:g}-{hi:g}")
    labels.append(f">={cuts[-1]:g}")
    return labels


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    rank_sum = float(ranks[pos].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, num_classes: int, prefix: str) -> dict:
    out = {f"{prefix}_n": int(len(y_true))}
    out[f"{prefix}_accuracy"] = float((y_true == y_pred).mean()) if len(y_true) else math.nan

    recalls = []
    precisions = []
    f1s = []
    for cls in range(num_classes):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        support = int((y_true == cls).sum())
        pred_count = int((y_pred == cls).sum())
        recall = tp / support if support else math.nan
        precision = tp / pred_count if pred_count else math.nan
        f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else math.nan
        out[f"{prefix}_class_{cls}_support"] = support
        out[f"{prefix}_class_{cls}_pred_count"] = pred_count
        out[f"{prefix}_class_{cls}_recall"] = recall
        out[f"{prefix}_class_{cls}_precision"] = precision
        out[f"{prefix}_class_{cls}_f1"] = f1
        if np.isfinite(recall):
            recalls.append(recall)
        if np.isfinite(precision):
            precisions.append(precision)
        if np.isfinite(f1):
            f1s.append(f1)

    out[f"{prefix}_balanced_accuracy"] = float(np.mean(recalls)) if recalls else math.nan
    out[f"{prefix}_macro_precision"] = float(np.mean(precisions)) if precisions else math.nan
    out[f"{prefix}_macro_f1"] = float(np.mean(f1s)) if f1s else math.nan
    if num_classes == 2:
        out[f"{prefix}_auc"] = binary_auc(y_true, y_score)
        out[f"{prefix}_auc_ovr"] = out[f"{prefix}_auc"]
    else:
        out[f"{prefix}_auc_ovr"] = math.nan
    return out


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return pd.DataFrame(mat, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])


def plot_confusion(cm: pd.DataFrame, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(cm.shape[1]))
    ax.set_xticklabels(cm.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(cm.shape[0]))
    ax.set_yticklabels(cm.index, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm.iloc[i, j])), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_binary(df: pd.DataFrame, score_col: str, title: str, out: Path) -> None:
    core = df[df["core_mask"].astype(bool)].copy()
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        core["centroid_lon"],
        core["centroid_lat"],
        c=core[score_col],
        s=2.5,
        cmap="magma",
        vmin=0,
        vmax=1,
        linewidths=0,
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="Posthoc high score")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_class(df: pd.DataFrame, pred_col: str, labels: list[str], title: str, out: Path) -> None:
    core = df[df["core_mask"].astype(bool)].copy()
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        core["centroid_lon"],
        core["centroid_lat"],
        c=core[pred_col],
        s=2.5,
        cmap="viridis",
        vmin=0,
        vmax=len(labels) - 1,
        linewidths=0,
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax, ticks=range(len(labels)))
    cbar.ax.set_yticklabels(labels)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def sigmoid_score(values: np.ndarray, threshold: float, scale: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(values - threshold) / scale))


def metric_value(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, metric: str) -> float:
    metrics = classification_metrics(y_true, y_pred, np.zeros(len(y_true), dtype=float), num_classes, "cal")
    value = metrics.get(f"cal_{metric}", math.nan)
    return float(value) if np.isfinite(value) else -math.inf


def candidate_thresholds(values: np.ndarray, max_candidates: int = 80) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.array([], dtype=float)
    qs = np.linspace(0.02, 0.98, max_candidates)
    cands = np.unique(np.quantile(finite, qs))
    return cands.astype(float)


def calibrate_prediction_cuts(
    pred: pd.DataFrame,
    true_cuts: list[float],
    split: str,
    metric: str,
) -> tuple[list[float], dict]:
    labeled = pred[pred["has_label"].astype(bool) & pred["split"].eq(split)].copy()
    y_true = y_to_class(labeled["y_true"].to_numpy(float), true_cuts)
    scores = labeled["y_pred"].to_numpy(float)
    num_classes = len(true_cuts) + 1
    cands = candidate_thresholds(scores)
    if len(cands) == 0:
        return true_cuts, {"calibration_split": split, "calibration_metric": metric, "calibrated": False}

    best_cuts = list(true_cuts)
    best_score = metric_value(y_true, y_to_class(scores, best_cuts), num_classes, metric)
    best_distance = 0.0

    if len(true_cuts) == 1:
        for cut in cands:
            pred_cls = y_to_class(scores, [float(cut)])
            score = metric_value(y_true, pred_cls, num_classes, metric)
            distance = abs(float(cut) - true_cuts[0])
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and distance < best_distance):
                best_score = score
                best_cuts = [float(cut)]
                best_distance = distance
    else:
        for i, cut1 in enumerate(cands[:-1]):
            for cut2 in cands[i + 1 :]:
                pred_cls = y_to_class(scores, [float(cut1), float(cut2)])
                score = metric_value(y_true, pred_cls, num_classes, metric)
                distance = abs(float(cut1) - true_cuts[0]) + abs(float(cut2) - true_cuts[-1])
                if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and distance < best_distance):
                    best_score = score
                    best_cuts = [float(cut1), float(cut2)]
                    best_distance = distance

    return best_cuts, {
        "calibration_split": split,
        "calibration_metric": metric,
        "calibrated": True,
        "calibration_score": best_score,
    }


def run_task(
    pred: pd.DataFrame,
    task_name: str,
    cuts: list[float],
    out_dir: Path,
    pred_cuts: list[float] | None = None,
    calibration_info: dict | None = None,
) -> dict:
    labels = labels_for(cuts)
    num_classes = len(labels)
    pred_cuts = pred_cuts if pred_cuts is not None else cuts
    calibration_info = calibration_info or {"calibrated": False}
    task_dir = out_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    task_df = pred.copy()
    task_df[f"{task_name}_class_pred"] = y_to_class(task_df["y_pred"].fillna(-np.inf).to_numpy(float), pred_cuts)
    task_df[f"{task_name}_class_pred_label"] = task_df[f"{task_name}_class_pred"].map(lambda x: labels[int(x)])

    if len(cuts) == 1:
        score_col = f"{task_name}_high_score"
        task_df[score_col] = sigmoid_score(task_df["y_pred"].to_numpy(float), pred_cuts[0], scale=max(cuts[0] / 3.0, 0.5))
        plot_spatial_binary(task_df, score_col, f"Posthoc Regression Hotspot Score ({task_name})", task_dir / "fig_spatial_score.png")
    plot_spatial_class(task_df, f"{task_name}_class_pred", labels, f"Posthoc Regression Classes ({task_name})", task_dir / "fig_spatial_class.png")

    labeled = task_df[task_df["has_label"].astype(bool)].copy()
    labeled[f"{task_name}_class_true"] = y_to_class(labeled["y_true"].to_numpy(float), cuts)
    metrics = {
        "task": task_name,
        "cuts": cuts,
        "prediction_cuts": pred_cuts,
        "class_labels": labels,
        **calibration_info,
    }
    confusion_rows = []
    for split in ["train", "val", "test"]:
        part = labeled[labeled["split"].eq(split)]
        y_true = part[f"{task_name}_class_true"].to_numpy(int)
        y_pred = part[f"{task_name}_class_pred"].to_numpy(int)
        score = part["y_pred"].to_numpy(float)
        metrics.update(classification_metrics(y_true, y_pred, score, num_classes, split))
        cm = confusion_frame(y_true, y_pred, labels)
        cm.to_csv(task_dir / f"confusion_{split}.csv", encoding="utf-8-sig")
        plot_confusion(cm, f"{task_name} {split} confusion", task_dir / f"fig_confusion_{split}.png")
        cm_long = cm.reset_index(names="true_class").melt(id_vars="true_class", var_name="pred_class", value_name="n")
        cm_long.insert(0, "split", split)
        confusion_rows.append(cm_long)

    tile_rows = []
    for (split, tile), part in task_df.groupby(["split", "tile_id"]):
        row = {
            "split": split,
            "tile_id": tile,
            "n_core": int(len(part)),
            "pred_mean_class": float(part[f"{task_name}_class_pred"].mean()),
        }
        for cls, label in enumerate(labels):
            row[f"pred_share_{cls}_{label}"] = float((part[f"{task_name}_class_pred"] == cls).mean())
        if len(cuts) == 1:
            row["mean_high_score"] = float(part[f"{task_name}_high_score"].mean())
            row["p90_high_score"] = float(part[f"{task_name}_high_score"].quantile(0.9))
        tile_rows.append(row)
    pd.DataFrame(tile_rows).to_csv(task_dir / "tile_posthoc_class_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(confusion_rows, ignore_index=True).to_csv(task_dir / "confusion_long.csv", index=False, encoding="utf-8-sig")
    task_df.to_csv(task_dir / "node_posthoc_class_predictions.csv", index=False, encoding="utf-8-sig")
    (task_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discretize continuous regression predictions after training.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()))
    parser.add_argument("--calibrate-on", choices=["none", "train", "val"], default="none")
    parser.add_argument("--calibration-metric", choices=["balanced_accuracy", "macro_f1", "accuracy"], default="balanced_accuracy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "posthoc_discrete"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(run_dir / "node_predictions.csv")
    summary = []
    for task in args.tasks:
        if task not in TASKS:
            raise ValueError(f"Unknown task {task}; available: {sorted(TASKS)}")
        pred_cuts = TASKS[task]
        calibration_info = {"calibrated": False}
        if args.calibrate_on != "none":
            pred_cuts, calibration_info = calibrate_prediction_cuts(
                pred,
                TASKS[task],
                split=args.calibrate_on,
                metric=args.calibration_metric,
            )
        summary.append(run_task(pred, task, TASKS[task], out_dir, pred_cuts=pred_cuts, calibration_info=calibration_info))

    rows = []
    for metrics in summary:
        row = {
            "task": metrics["task"],
            "cuts": json.dumps(metrics["cuts"]),
            "prediction_cuts": json.dumps(metrics["prediction_cuts"]),
            "calibrated": metrics.get("calibrated", False),
            "calibration_split": metrics.get("calibration_split", ""),
            "calibration_metric": metrics.get("calibration_metric", ""),
            "calibration_score": metrics.get("calibration_score", math.nan),
            "class_labels": "|".join(metrics["class_labels"]),
        }
        for split in ["train", "val", "test"]:
            for key in ["accuracy", "balanced_accuracy", "macro_f1", "auc", "auc_ovr"]:
                metric_key = f"{split}_{key}"
                if metric_key in metrics:
                    row[metric_key] = metrics[metric_key]
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "posthoc_discrete_summary.csv", index=False, encoding="utf-8-sig")

    report_lines = [
        "# Posthoc Discretization of Regression Predictions",
        "",
        f"Source regression run: `{run_dir}`",
        f"Calibration: `{args.calibrate_on}` / `{args.calibration_metric}`",
        "",
        "| task | target cuts | prediction cuts | val_auc | val_bal_acc | val_macro_f1 | test_auc | test_bal_acc | test_macro_f1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary_df.iterrows():
        report_lines.append(
            f"| {row['task']} | `{row['cuts']}` | `{row['prediction_cuts']}` | "
            f"{row.get('val_auc_ovr', math.nan):.6f} | {row.get('val_balanced_accuracy', math.nan):.6f} | "
            f"{row.get('val_macro_f1', math.nan):.6f} | {row.get('test_auc_ovr', math.nan):.6f} | "
            f"{row.get('test_balanced_accuracy', math.nan):.6f} | {row.get('test_macro_f1', math.nan):.6f} |"
        )
    (out_dir / "posthoc_discrete_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"Saved posthoc discrete outputs to: {out_dir}")


if __name__ == "__main__":
    main()
