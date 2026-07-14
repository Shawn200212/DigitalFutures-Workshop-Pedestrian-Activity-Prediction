from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from train_graphsage_npz import (
    FeatureTransform,
    GraphData,
    WeightedSAGEConv,
    apply_feature_normalizer,
    compute_feature_normalizer,
    list_tile_paths,
    load_graph,
    make_split,
    save_json,
    set_seed,
)


class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, num_classes: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = nn.ModuleList()
        self.layers.append(WeightedSAGEConv(in_dim, hidden_dim, dropout))
        for _ in range(num_layers - 1):
            self.layers.append(WeightedSAGEConv(hidden_dim, hidden_dim, dropout))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, graph: GraphData) -> torch.Tensor:
        x = graph.x
        for layer in self.layers:
            x = layer(x, graph.edge_index, graph.edge_weight)
        return self.head(x)


def class_spec(task: str, thresholds: str | None) -> tuple[list[float], list[str]]:
    if thresholds:
        cuts = [float(x) for x in thresholds.split(",") if x.strip()]
    elif task == "binary":
        cuts = [3.0]
    elif task == "ternary":
        cuts = [1.0, 5.0]
    else:
        raise ValueError(f"Unknown task: {task}")

    if len(cuts) == 1:
        labels = [f"low_<_{cuts[0]:g}", f"high_>=_{cuts[0]:g}"]
    else:
        labels = [f"class_0_<_{cuts[0]:g}"]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            labels.append(f"class_{len(labels)}_{lo:g}_to_{hi:g}")
        labels.append(f"class_{len(labels)}_>=_{cuts[-1]:g}")
    return cuts, labels


def y_to_class(y: torch.Tensor, cuts: list[float]) -> torch.Tensor:
    out = torch.zeros_like(y, dtype=torch.long)
    for cut in cuts:
        out = out + (y >= cut).long()
    return out


def class_weights(graphs: list[GraphData], cuts: list[float], num_classes: int, mode: str, device: torch.device) -> torch.Tensor:
    ys = []
    for graph in graphs:
        mask = graph.label_mask
        if bool(mask.any()):
            ys.append(y_to_class(graph.y_raw[mask], cuts).detach().cpu())
    y = torch.cat(ys) if ys else torch.empty(0, dtype=torch.long)
    counts = torch.bincount(y, minlength=num_classes).float()
    if mode == "none":
        weights = torch.ones(num_classes)
    elif mode == "balanced":
        weights = counts.sum() / (counts.clamp_min(1) * num_classes)
    elif mode == "sqrt_balanced":
        weights = torch.sqrt(counts.sum() / (counts.clamp_min(1) * num_classes))
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")
    return weights.to(device)


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


def multiclass_auc_ovr(y_true: np.ndarray, proba: np.ndarray, num_classes: int) -> float:
    aucs = []
    for cls in range(num_classes):
        binary = (y_true == cls).astype(int)
        auc = binary_auc(binary, proba[:, cls])
        if np.isfinite(auc):
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else math.nan


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, prefix: str, num_classes: int) -> dict:
    rows = {f"{prefix}_n": int(len(y_true))}
    if len(y_true) == 0:
        rows.update(
            {
                f"{prefix}_accuracy": math.nan,
                f"{prefix}_balanced_accuracy": math.nan,
                f"{prefix}_macro_f1": math.nan,
                f"{prefix}_auc_ovr": math.nan,
            }
        )
        return rows

    rows[f"{prefix}_accuracy"] = float((y_true == y_pred).mean())
    recalls = []
    f1s = []
    for cls in range(num_classes):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        support = int((y_true == cls).sum())
        recall = tp / support if support else math.nan
        precision = tp / (tp + fp) if (tp + fp) else math.nan
        f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else math.nan
        rows[f"{prefix}_class_{cls}_support"] = support
        rows[f"{prefix}_class_{cls}_recall"] = recall
        rows[f"{prefix}_class_{cls}_precision"] = precision
        rows[f"{prefix}_class_{cls}_f1"] = f1
        if np.isfinite(recall):
            recalls.append(recall)
        if np.isfinite(f1):
            f1s.append(f1)

    rows[f"{prefix}_balanced_accuracy"] = float(np.mean(recalls)) if recalls else math.nan
    rows[f"{prefix}_macro_f1"] = float(np.mean(f1s)) if f1s else math.nan
    if num_classes == 2:
        rows[f"{prefix}_auc"] = binary_auc(y_true, proba[:, 1])
    rows[f"{prefix}_auc_ovr"] = multiclass_auc_ovr(y_true, proba, num_classes)
    return rows


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    graphs: list[GraphData],
    split_name: str,
    cuts: list[float],
    class_labels: list[str],
) -> tuple[dict, pd.DataFrame]:
    model.eval()
    rows = []
    y_all = []
    pred_all = []
    prob_all = []
    num_classes = len(class_labels)

    for graph in graphs:
        logits = model(graph)
        proba = torch.softmax(logits, dim=-1)
        pred = torch.argmax(proba, dim=-1)
        true_class = y_to_class(graph.y_raw, cuts)
        mask = graph.label_mask

        y_np = true_class[mask].detach().cpu().numpy()
        pred_np = pred[mask].detach().cpu().numpy()
        prob_np = proba[mask].detach().cpu().numpy()
        y_all.append(y_np)
        pred_all.append(pred_np)
        prob_all.append(prob_np)

        core_idx = torch.where(graph.core_mask)[0].detach().cpu().numpy()
        y_raw_cpu = graph.y_raw.detach().cpu().numpy()
        true_cpu = true_class.detach().cpu().numpy()
        pred_cpu = pred.detach().cpu().numpy()
        prob_cpu = proba.detach().cpu().numpy()
        label_cpu = graph.label_mask.detach().cpu().numpy()

        for idx in core_idx:
            row = {
                "split": split_name,
                "tile_id": graph.tile_id,
                "node_index": int(idx),
                "seg_id": int(graph.seg_id[idx]),
                "centroid_lon": float(graph.pos[idx, 0]),
                "centroid_lat": float(graph.pos[idx, 1]),
                "core_mask": True,
                "has_label": bool(label_cpu[idx]),
                "num_pano_images": int(graph.num_pano_images[idx]),
                "y_true": float(y_raw_cpu[idx]) if np.isfinite(y_raw_cpu[idx]) else np.nan,
                "class_true": int(true_cpu[idx]) if label_cpu[idx] else np.nan,
                "class_pred": int(pred_cpu[idx]),
                "class_pred_label": class_labels[int(pred_cpu[idx])],
                "confidence": float(prob_cpu[idx].max()),
            }
            for cls, label in enumerate(class_labels):
                row[f"prob_{cls}_{label}"] = float(prob_cpu[idx, cls])
            rows.append(row)

    y_true = np.concatenate(y_all) if y_all else np.array([], dtype=int)
    y_pred = np.concatenate(pred_all) if pred_all else np.array([], dtype=int)
    proba_np = np.concatenate(prob_all) if prob_all else np.empty((0, num_classes))
    return classification_metrics(y_true, y_pred, proba_np, split_name, num_classes), pd.DataFrame(rows)


def train_one_epoch_classifier(
    model: nn.Module,
    graphs: list[GraphData],
    optimizer: torch.optim.Optimizer,
    cuts: list[float],
    class_weight: torch.Tensor,
    grad_clip: float,
) -> float:
    model.train()
    random.shuffle(graphs)
    losses = []
    for graph in graphs:
        mask = graph.label_mask
        if not bool(mask.any()):
            continue
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph)
        target = y_to_class(graph.y_raw[mask], cuts)
        loss = F.cross_entropy(logits[mask], target, weight=class_weight)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphSAGE classifier on SHdataset npz tile graphs.")
    parser.add_argument("--dataset-root", default="dataset/SHdataset")
    parser.add_argument("--output-dir", default="outputs/gnn/tile001_100_graphsage_classifier")
    parser.add_argument("--tile-start", type=int, default=1)
    parser.add_argument("--tile-end", type=int, default=100)
    parser.add_argument("--min-labeled-core", type=int, default=1)
    parser.add_argument("--split-mode", choices=["random", "kfold", "from_file"], default="from_file")
    parser.add_argument("--split-file", default="outputs/gnn/tile001_100_graphsage_best_final/split.json")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    parser.add_argument("--task", choices=["binary", "ternary"], default="binary")
    parser.add_argument("--thresholds", default=None, help="Comma-separated thresholds, e.g. 3 or 1,5.")
    parser.add_argument("--class-weight", choices=["none", "balanced", "sqrt_balanced"], default="balanced")
    parser.add_argument("--selection-metric", choices=["val_auc_ovr", "val_balanced_accuracy", "val_macro_f1"], default="val_auc_ovr")

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--feature-transform", choices=["standard", "skew_log1p_clip"], default="skew_log1p_clip")
    parser.add_argument("--feature-clip-quantile", type=float, default=0.999)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cuts, labels = class_spec(args.task, args.thresholds)
    num_classes = len(labels)

    paths, excluded = list_tile_paths(Path(args.dataset_root), args.tile_start, args.tile_end, args.min_labeled_core)
    split_paths = make_split(args, paths)
    graphs_by_split = {name: [load_graph(path, device) for path in split_paths_] for name, split_paths_ in split_paths.items()}

    feature_names = graphs_by_split["train"][0].feature_names
    for graphs in graphs_by_split.values():
        for graph in graphs:
            if graph.feature_names != feature_names:
                raise ValueError(f"Feature names mismatch in {graph.tile_id}")

    feature_transform = FeatureTransform(args.feature_transform, args.feature_clip_quantile)
    feature_transform.fit(graphs_by_split["train"], feature_names)
    for graphs in graphs_by_split.values():
        feature_transform.apply(graphs)
    x_mean, x_std = compute_feature_normalizer(graphs_by_split["train"])
    for graphs in graphs_by_split.values():
        apply_feature_normalizer(graphs, x_mean, x_std)

    weights = class_weights(graphs_by_split["train"], cuts, num_classes, args.class_weight, device)
    model = GraphSAGEClassifier(len(feature_names), args.hidden_dim, args.num_layers, args.dropout, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    split_info = {name: [p.stem for p in paths_] for name, paths_ in split_paths.items()}
    label_counts = {}
    class_counts = {}
    for name, graphs in graphs_by_split.items():
        y_parts = []
        for graph in graphs:
            if bool(graph.label_mask.any()):
                y_parts.append(y_to_class(graph.y_raw[graph.label_mask], cuts).detach().cpu())
        cls = torch.cat(y_parts) if y_parts else torch.empty(0, dtype=torch.long)
        label_counts[name] = {
            "tiles": len(graphs),
            "nodes": int(sum(g.x.shape[0] for g in graphs)),
            "core_nodes": int(sum(g.core_mask.sum().item() for g in graphs)),
            "labeled_core_nodes": int(sum(g.label_mask.sum().item() for g in graphs)),
        }
        class_counts[name] = torch.bincount(cls, minlength=num_classes).tolist()

    run_config = {
        "args": vars(args),
        "model": "graphsage_classifier",
        "cuts": cuts,
        "class_labels": labels,
        "class_weight": weights.detach().cpu().tolist(),
        "feature_names": feature_names,
        "feature_transform": feature_transform.to_dict(),
        "split": split_info,
        "label_counts": label_counts,
        "class_counts": class_counts,
        "excluded_tiles": excluded,
    }
    save_json(output_dir / "run_config.json", run_config)
    save_json(output_dir / "split.json", split_info)

    history = []
    best_score = -float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch_classifier(model, graphs_by_split["train"], optimizer, cuts, weights, args.grad_clip)
        val_metrics, _ = evaluate_classifier(model, graphs_by_split["val"], "val", cuts, labels)
        score = val_metrics.get(args.selection_metric, math.nan)
        scheduler.step(score if np.isfinite(score) else 0.0)

        row = {"epoch": epoch, "train_loss": loss, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        improved = np.isfinite(score) and score > best_score
        if improved:
            best_score = float(score)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % args.print_every == 0 or improved:
            print(
                {
                    "epoch": epoch,
                    "train_loss": round(loss, 5),
                    args.selection_metric: round(score, 5) if np.isfinite(score) else None,
                    "val_acc": round(val_metrics["val_accuracy"], 5),
                    "val_bal_acc": round(val_metrics["val_balanced_accuracy"], 5),
                    "val_macro_f1": round(val_metrics["val_macro_f1"], 5),
                    "best_epoch": best_epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
        if stale >= args.patience:
            print({"early_stop_epoch": epoch, "best_epoch": best_epoch, "best_score": best_score})
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    all_metrics = {"best_epoch": int(best_epoch), "best_selection_score": float(best_score)}
    frames = []
    for split in ["train", "val", "test"]:
        metrics, pred_df = evaluate_classifier(model, graphs_by_split[split], split, cuts, labels)
        all_metrics.update(metrics)
        frames.append(pred_df)

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(output_dir / "node_class_predictions.csv", index=False, encoding="utf-8-sig")

    save_json(output_dir / "metrics.json", all_metrics)
    save_json(
        output_dir / "diagnostics.json",
        {
            "metrics": all_metrics,
            "class_counts": class_counts,
            "class_labels": labels,
            "cuts": cuts,
            "class_weight": weights.detach().cpu().tolist(),
            "feature_transform": feature_transform.to_dict(),
        },
    )
    np.savez(
        output_dir / "normalizer.npz",
        mean=x_mean.detach().cpu().numpy(),
        std=x_std.detach().cpu().numpy(),
        feature_names=np.array(feature_names),
        feature_transform=json.dumps(feature_transform.to_dict()),
        cuts=np.array(cuts, dtype=float),
        class_labels=np.array(labels),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "metrics": all_metrics,
            "cuts": cuts,
            "class_labels": labels,
            "feature_names": feature_names,
            "feature_transform": feature_transform.to_dict(),
            "split": split_info,
        },
        output_dir / "best_model.pt",
    )

    print("FINAL_METRICS")
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
