from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from train_graphsage_npz import (
    GraphData,
    TargetTransform,
    WeightConfig,
    apply_feature_normalizer,
    apply_target_transform,
    baseline_metrics,
    compute_feature_normalizer,
    evaluate,
    list_tile_paths,
    load_graph,
    make_split,
    save_json,
    set_seed,
    tile_metric_frame,
    train_one_epoch,
)


class WeightedGCNConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, self_loop_weight: float = 1.0) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = dropout
        self.self_loop_weight = self_loop_weight

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        device = x.device

        src, dst = edge_index[0], edge_index[1]
        weights = edge_weight.clamp_min(0)

        loop_nodes = torch.arange(num_nodes, dtype=torch.long, device=device)
        src = torch.cat([src, loop_nodes], dim=0)
        dst = torch.cat([dst, loop_nodes], dim=0)
        loop_weight = torch.full((num_nodes,), self.self_loop_weight, dtype=x.dtype, device=device)
        weights = torch.cat([weights.to(dtype=x.dtype), loop_weight], dim=0)

        deg = torch.zeros(num_nodes, dtype=x.dtype, device=device)
        deg.index_add_(0, dst, weights)
        norm = weights / (deg[src].sqrt() * deg[dst].sqrt()).clamp_min(1e-12)

        support = self.lin(x)
        out = torch.zeros_like(support)
        out.index_add_(0, dst, support[src] * norm.view(-1, 1))
        out = out + self.bias
        out = F.relu(out)
        return F.dropout(out, p=self.dropout, training=self.training)


class GCNRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        self_loop_weight: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.layers = nn.ModuleList()
        self.layers.append(WeightedGCNConv(in_dim, hidden_dim, dropout, self_loop_weight))
        for _ in range(num_layers - 1):
            self.layers.append(WeightedGCNConv(hidden_dim, hidden_dim, dropout, self_loop_weight))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph: GraphData) -> torch.Tensor:
        x = graph.x
        for layer in self.layers:
            x = layer(x, graph.edge_index, graph.edge_weight)
        return self.head(x).squeeze(-1)


def load_normalizer(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, TargetTransform]:
    z = np.load(path, allow_pickle=True)
    mean = torch.tensor(z["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(z["std"], dtype=torch.float32, device=device)

    raw_transform = z["target_transform"]
    if hasattr(raw_transform, "item"):
        raw_transform = raw_transform.item()
    transform_data = json.loads(str(raw_transform))
    target_transform = TargetTransform(
        mode=str(transform_data["mode"]),
        mean=float(transform_data.get("mean", 0.0)),
        std=float(transform_data.get("std", 1.0)),
        log_mean=float(transform_data.get("log_mean", 0.0)),
        log_std=float(transform_data.get("log_std", 1.0)),
    )
    return mean, std, target_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GCN on SHdataset npz tile graphs.")
    parser.add_argument("--dataset-root", default="dataset/SHdataset")
    parser.add_argument("--output-dir", default="outputs/gnn/tile001_100_gcn")
    parser.add_argument("--tile-start", type=int, default=1)
    parser.add_argument("--tile-end", type=int, default=100)
    parser.add_argument("--min-labeled-core", type=int, default=1)

    parser.add_argument("--split-mode", choices=["random", "kfold", "from_file"], default="from_file")
    parser.add_argument("--split-file", default="outputs/gnn/tile001_100_graphsage_best_final/split.json")
    parser.add_argument("--normalizer-file", default=None)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--self-loop-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--loss", choices=["huber", "mse", "mae"], default="huber")
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument(
        "--target-transform",
        choices=["none", "zscore", "log1p", "log1p_zscore"],
        default="log1p_zscore",
    )
    parser.add_argument(
        "--loss-weight",
        choices=["uniform", "num_pano_sqrt", "num_pano_log1p", "y_quantile"],
        default="num_pano_sqrt",
    )
    parser.add_argument("--high-y-weight", type=float, default=2.0)
    parser.add_argument("--very-high-y-weight", type=float, default=3.0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths, excluded = list_tile_paths(root, args.tile_start, args.tile_end, args.min_labeled_core)
    if len(paths) < 3:
        raise FileNotFoundError(
            f"Need at least 3 labeled tile_*.npz files in {root} for range {args.tile_start}-{args.tile_end}; found {len(paths)}"
        )

    split_paths = make_split(args, paths)
    graphs_by_split = {
        name: [load_graph(path, device) for path in split_paths_]
        for name, split_paths_ in split_paths.items()
    }

    feature_names = graphs_by_split["train"][0].feature_names
    for graphs in graphs_by_split.values():
        for graph in graphs:
            if graph.feature_names != feature_names:
                raise ValueError(f"Feature names mismatch in {graph.tile_id}")

    if args.normalizer_file:
        x_mean, x_std, target_transform = load_normalizer(Path(args.normalizer_file), device)
    else:
        x_mean, x_std = compute_feature_normalizer(graphs_by_split["train"])
        target_transform = TargetTransform(args.target_transform)
        target_transform.fit(graphs_by_split["train"])

    for graphs in graphs_by_split.values():
        apply_feature_normalizer(graphs, x_mean, x_std)

    for graphs in graphs_by_split.values():
        apply_target_transform(graphs, target_transform)

    weight_config = WeightConfig(
        mode=args.loss_weight,
        high_weight=args.high_y_weight,
        very_high_weight=args.very_high_y_weight,
    )
    weight_config.fit(graphs_by_split["train"])

    model = GCNRegressor(
        in_dim=len(feature_names),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        self_loop_weight=args.self_loop_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    split_info = {name: [p.stem for p in paths_] for name, paths_ in split_paths.items()}
    label_counts = {}
    for name, graphs in graphs_by_split.items():
        label_counts[name] = {
            "tiles": len(graphs),
            "nodes": int(sum(g.x.shape[0] for g in graphs)),
            "core_nodes": int(sum(g.core_mask.sum().item() for g in graphs)),
            "labeled_core_nodes": int(sum(g.label_mask.sum().item() for g in graphs)),
        }

    args_dict = vars(args)
    run_config = {
        "args": args_dict,
        "model": "gcn",
        "device": str(device),
        "feature_names": feature_names,
        "split": split_info,
        "label_counts": label_counts,
        "excluded_tiles": excluded,
        "target_transform": target_transform.to_dict(),
        "loss_weight": weight_config.to_dict(),
    }
    save_json(output_dir / "run_config.json", run_config)
    save_json(output_dir / "split.json", split_info)

    history = []
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            graphs_by_split["train"],
            optimizer,
            args.loss,
            args.huber_beta,
            weight_config,
            args.grad_clip,
        )
        val_metrics, _ = evaluate(model, graphs_by_split["val"], "val", target_transform)
        val_rmse = val_metrics["val_rmse"]
        scheduler.step(val_rmse if np.isfinite(val_rmse) else train_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        improved = np.isfinite(val_rmse) and val_rmse < best_val
        if improved:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        should_print = epoch == 1 or epoch % args.print_every == 0 or improved
        if should_print:
            print(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 5),
                    "val_rmse": round(val_rmse, 5) if np.isfinite(val_rmse) else None,
                    "val_mae": round(val_metrics["val_mae"], 5) if np.isfinite(val_metrics["val_mae"]) else None,
                    "val_r2": round(val_metrics["val_r2"], 5) if np.isfinite(val_metrics["val_r2"]) else None,
                    "best_epoch": best_epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        if stale >= args.patience:
            print({"early_stop_epoch": epoch, "best_epoch": best_epoch, "best_val_rmse": best_val})
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    all_metrics: dict[str, float | int] = {"best_epoch": int(best_epoch)}
    prediction_frames = []
    for name in ["train", "val", "test"]:
        metrics, preds = evaluate(model, graphs_by_split[name], name, target_transform)
        all_metrics.update(metrics)
        prediction_frames.append(preds)

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    pred_df = pd.concat(prediction_frames, ignore_index=True)
    pred_df.to_csv(output_dir / "node_predictions.csv", index=False, encoding="utf-8-sig")

    tile_metrics = tile_metric_frame(pred_df)
    tile_metrics.to_csv(output_dir / "tile_metrics.csv", index=False, encoding="utf-8-sig")

    diagnostics = {
        "split": split_info,
        "label_counts": label_counts,
        "model_metrics": all_metrics,
        "baseline_train_mean": baseline_metrics(pred_df),
        "target_transform": target_transform.to_dict(),
        "loss_weight": weight_config.to_dict(),
    }

    save_json(output_dir / "metrics.json", all_metrics)
    save_json(output_dir / "diagnostics.json", diagnostics)
    np.savez(
        output_dir / "normalizer.npz",
        mean=x_mean.detach().cpu().numpy(),
        std=x_std.detach().cpu().numpy(),
        feature_names=np.array(feature_names),
        target_transform=json.dumps(target_transform.to_dict()),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": args_dict,
            "model": "gcn",
            "feature_names": feature_names,
            "metrics": all_metrics,
            "split": split_info,
            "target_transform": target_transform.to_dict(),
            "loss_weight": weight_config.to_dict(),
        },
        output_dir / "best_model.pt",
    )

    print("FINAL_METRICS")
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
