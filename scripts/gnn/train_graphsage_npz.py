from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GraphData:
    tile_id: str
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    y_raw: torch.Tensor
    y: torch.Tensor
    core_mask: torch.Tensor
    pos: np.ndarray
    seg_id: np.ndarray
    num_pano_images: np.ndarray
    feature_names: list[str]

    @property
    def label_mask(self) -> torch.Tensor:
        return self.core_mask & torch.isfinite(self.y_raw)


class WeightedSAGEConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        weights = edge_weight.clamp_min(0).view(-1, 1)

        agg = torch.zeros_like(x)
        deg = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        agg.index_add_(0, dst, x[src] * weights)
        deg.index_add_(0, dst, weights)
        agg = agg / deg.clamp_min(1e-12)

        out = self.lin_self(x) + self.lin_neigh(agg)
        out = F.relu(out)
        return F.dropout(out, p=self.dropout, training=self.training)


class GraphSAGERegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
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
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph: GraphData) -> torch.Tensor:
        x = graph.x
        for layer in self.layers:
            x = layer(x, graph.edge_index, graph.edge_weight)
        return self.head(x).squeeze(-1)


@dataclass
class TargetTransform:
    mode: str
    mean: float = 0.0
    std: float = 1.0
    log_mean: float = 0.0
    log_std: float = 1.0

    def fit(self, graphs: Iterable[GraphData]) -> None:
        values = []
        for graph in graphs:
            mask = graph.label_mask
            if bool(mask.any()):
                values.append(graph.y_raw[mask].detach())
        if not values:
            return

        y = torch.cat(values).float()
        y_nonneg = y.clamp_min(0)

        self.mean = float(y.mean().detach().cpu())
        self.std = max(float(y.std(unbiased=False).detach().cpu()), 1e-6)

        ly = torch.log1p(y_nonneg)
        self.log_mean = float(ly.mean().detach().cpu())
        self.log_std = max(float(ly.std(unbiased=False).detach().cpu()), 1e-6)

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return y.clone()
        if self.mode == "zscore":
            return (y - self.mean) / self.std
        if self.mode == "log1p":
            return torch.log1p(y.clamp_min(0))
        if self.mode == "log1p_zscore":
            return (torch.log1p(y.clamp_min(0)) - self.log_mean) / self.log_std
        raise ValueError(f"Unknown target transform: {self.mode}")

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return y
        if self.mode == "zscore":
            return y * self.std + self.mean
        if self.mode == "log1p":
            return torch.expm1(y).clamp_min(0)
        if self.mode == "log1p_zscore":
            return torch.expm1(y * self.log_std + self.log_mean).clamp_min(0)
        raise ValueError(f"Unknown target transform: {self.mode}")

    def to_dict(self) -> dict[str, float | str]:
        return {
            "mode": self.mode,
            "mean": self.mean,
            "std": self.std,
            "log_mean": self.log_mean,
            "log_std": self.log_std,
        }


@dataclass
class WeightConfig:
    mode: str
    q75: float = 0.0
    q90: float = 0.0
    high_weight: float = 2.0
    very_high_weight: float = 3.0

    def fit(self, graphs: Iterable[GraphData]) -> None:
        values = []
        for graph in graphs:
            mask = graph.label_mask
            if bool(mask.any()):
                values.append(graph.y_raw[mask].detach().cpu().numpy())
        if not values:
            return
        y = np.concatenate(values)
        self.q75 = float(np.nanquantile(y, 0.75))
        self.q90 = float(np.nanquantile(y, 0.90))

    def weights(self, graph: GraphData, mask: torch.Tensor) -> torch.Tensor:
        if self.mode == "uniform":
            return torch.ones_like(graph.y[mask])

        if self.mode == "num_pano_sqrt":
            raw = torch.tensor(graph.num_pano_images, dtype=graph.y.dtype, device=graph.y.device)[mask]
            w = torch.sqrt(raw.clamp_min(1.0))
        elif self.mode == "num_pano_log1p":
            raw = torch.tensor(graph.num_pano_images, dtype=graph.y.dtype, device=graph.y.device)[mask]
            w = torch.log1p(raw.clamp_min(0.0)).clamp_min(1.0)
        elif self.mode == "y_quantile":
            y = graph.y_raw[mask]
            w = torch.ones_like(y)
            w = torch.where(y >= self.q75, torch.full_like(w, self.high_weight), w)
            w = torch.where(y >= self.q90, torch.full_like(w, self.very_high_weight), w)
        else:
            raise ValueError(f"Unknown loss weight mode: {self.mode}")

        return w / w.mean().clamp_min(1e-6)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "mode": self.mode,
            "q75": self.q75,
            "q90": self.q90,
            "high_weight": self.high_weight,
            "very_high_weight": self.very_high_weight,
        }


@dataclass
class FeatureTransform:
    mode: str = "standard"
    clip_quantile: float = 0.995
    feature_names: list[str] = field(default_factory=list)
    log_indices: list[int] = field(default_factory=list)
    clip_indices: list[int] = field(default_factory=list)
    clip_lower: list[float] = field(default_factory=list)
    clip_upper: list[float] = field(default_factory=list)

    def fit(self, graphs: Iterable[GraphData], feature_names: list[str]) -> None:
        self.feature_names = list(feature_names)
        n_features = len(feature_names)
        self.clip_lower = [-math.inf] * n_features
        self.clip_upper = [math.inf] * n_features
        self.log_indices = []
        self.clip_indices = []

        if self.mode == "standard":
            return
        if self.mode != "skew_log1p_clip":
            raise ValueError(f"Unknown feature transform: {self.mode}")
        if not 0.5 < self.clip_quantile < 1.0:
            raise ValueError("--feature-clip-quantile must be in (0.5, 1.0)")

        xs = torch.cat([g.x for g in graphs], dim=0).detach().cpu().numpy()
        lower_q = 1.0 - self.clip_quantile

        for idx, name in enumerate(feature_names):
            col = xs[:, idx]
            col = col[np.isfinite(col)]
            if col.size == 0:
                continue

            q_min = float(np.min(col))
            q_med = float(np.quantile(col, 0.50))
            q_p99 = float(np.quantile(col, 0.99))
            q_clip_hi = float(np.quantile(col, self.clip_quantile))
            q_clip_lo = 0.0 if q_min >= -1e-9 else float(np.quantile(col, lower_q))
            q_max = float(np.max(col))

            non_negative = q_min >= -1e-9
            p99_to_med = q_p99 / (abs(q_med) + 1e-6) if abs(q_med) > 1e-9 else math.inf
            max_to_p99 = q_max / (abs(q_p99) + 1e-6) if abs(q_p99) > 1e-9 else math.inf
            sparse_count_like = q_med <= 0 and q_p99 > 1
            skewed = non_negative and (sparse_count_like or p99_to_med >= 5.0 or max_to_p99 >= 2.5)

            if skewed:
                self.log_indices.append(idx)
                self.clip_indices.append(idx)
                self.clip_lower[idx] = q_clip_lo
                self.clip_upper[idx] = q_clip_hi

    def apply(self, graphs: Iterable[GraphData]) -> None:
        if self.mode == "standard":
            return

        for graph in graphs:
            x = graph.x.clone()
            if self.clip_indices:
                idx = torch.tensor(self.clip_indices, dtype=torch.long, device=x.device)
                lower = torch.tensor(
                    [self.clip_lower[i] for i in self.clip_indices],
                    dtype=x.dtype,
                    device=x.device,
                )
                upper = torch.tensor(
                    [self.clip_upper[i] for i in self.clip_indices],
                    dtype=x.dtype,
                    device=x.device,
                )
                x[:, idx] = torch.minimum(torch.maximum(x[:, idx], lower), upper)
            if self.log_indices:
                idx = torch.tensor(self.log_indices, dtype=torch.long, device=x.device)
                x[:, idx] = torch.log1p(x[:, idx].clamp_min(0))
            graph.x = x

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "clip_quantile": self.clip_quantile,
            "log_features": [self.feature_names[i] for i in self.log_indices],
            "clip_features": [self.feature_names[i] for i in self.clip_indices],
            "clip_lower": self.clip_lower,
            "clip_upper": self.clip_upper,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_tile_number(path_or_name: str | Path) -> int:
    stem = Path(path_or_name).stem
    return int(stem.split("_")[-1])


def load_graph(path: Path, device: torch.device) -> GraphData:
    z = np.load(path, allow_pickle=True)
    y_raw = torch.tensor(z["y"], dtype=torch.float32, device=device)
    return GraphData(
        tile_id=path.stem,
        x=torch.tensor(z["x"], dtype=torch.float32, device=device),
        edge_index=torch.tensor(z["edge_index"], dtype=torch.long, device=device),
        edge_weight=torch.tensor(z["edge_weight"], dtype=torch.float32, device=device),
        y_raw=y_raw,
        y=y_raw.clone(),
        core_mask=torch.tensor(z["core_mask"].astype(bool), dtype=torch.bool, device=device),
        pos=z["pos"],
        seg_id=z["seg_id"],
        num_pano_images=z["num_pano_images"],
        feature_names=[str(x) for x in z["feature_names"].tolist()],
    )


def list_tile_paths(root: Path, tile_start: int, tile_end: int, min_labeled_core: int) -> tuple[list[Path], list[dict]]:
    kept = []
    excluded = []
    for path in sorted(root.glob("tile_*.npz"), key=parse_tile_number):
        tile_num = parse_tile_number(path)
        if tile_num < tile_start or tile_num > tile_end:
            continue
        z = np.load(path, allow_pickle=True)
        y = z["y"]
        core = z["core_mask"].astype(bool)
        labeled_core = int((core & np.isfinite(y)).sum())
        row = {
            "tile_id": path.stem,
            "n_nodes": int(z["x"].shape[0]),
            "n_core": int(core.sum()),
            "n_labeled_core": labeled_core,
        }
        if labeled_core >= min_labeled_core:
            kept.append(path)
        else:
            excluded.append(row)
    return kept, excluded


def random_split(paths: list[Path], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)

    n = len(paths)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    return {
        "train": sorted(paths[:n_train], key=parse_tile_number),
        "val": sorted(paths[n_train : n_train + n_val], key=parse_tile_number),
        "test": sorted(paths[n_train + n_val :], key=parse_tile_number),
    }


def kfold_split(paths: list[Path], seed: int, k_folds: int, fold_index: int, val_ratio: float) -> dict[str, list[Path]]:
    if k_folds < 2:
        raise ValueError("k_folds must be >= 2")
    if fold_index < 0 or fold_index >= k_folds:
        raise ValueError("fold_index must be in [0, k_folds)")

    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    folds = [list(chunk) for chunk in np.array_split(paths, k_folds)]

    test = folds[fold_index]
    remaining = [p for i, fold in enumerate(folds) if i != fold_index for p in fold]
    rng.shuffle(remaining)

    n_val = max(1, int(round(len(paths) * val_ratio)))
    n_val = min(n_val, max(1, len(remaining) - 1))
    val = remaining[:n_val]
    train = remaining[n_val:]

    return {
        "train": sorted(train, key=parse_tile_number),
        "val": sorted(val, key=parse_tile_number),
        "test": sorted(test, key=parse_tile_number),
    }


def file_split(paths: list[Path], split_file: Path) -> dict[str, list[Path]]:
    data = json.loads(split_file.read_text(encoding="utf-8"))
    by_name = {p.stem: p for p in paths}
    split = {}
    for name in ["train", "val", "test"]:
        tile_names = data.get(name, [])
        missing = [tile for tile in tile_names if tile not in by_name]
        if missing:
            raise ValueError(f"Split file references missing tiles in {name}: {missing}")
        split[name] = sorted([by_name[tile] for tile in tile_names], key=parse_tile_number)
    return split


def make_split(args: argparse.Namespace, paths: list[Path]) -> dict[str, list[Path]]:
    if args.split_mode == "random":
        return random_split(paths, args.seed, args.train_ratio, args.val_ratio)
    if args.split_mode == "kfold":
        return kfold_split(paths, args.seed, args.k_folds, args.fold_index, args.val_ratio)
    if args.split_mode == "from_file":
        if not args.split_file:
            raise ValueError("--split-file is required when --split-mode from_file")
        return file_split(paths, Path(args.split_file))
    raise ValueError(f"Unknown split mode: {args.split_mode}")


def compute_feature_normalizer(graphs: Iterable[GraphData]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.cat([g.x for g in graphs], dim=0)
    mean = xs.mean(dim=0, keepdim=True)
    std = xs.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return mean, std


def apply_feature_normalizer(graphs: Iterable[GraphData], mean: torch.Tensor, std: torch.Tensor) -> None:
    for graph in graphs:
        graph.x = (graph.x - mean) / std


def apply_target_transform(graphs: Iterable[GraphData], transform: TargetTransform) -> None:
    for graph in graphs:
        graph.y = transform.transform(graph.y_raw)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float | int]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mae": math.nan,
            f"{prefix}_rmse": math.nan,
            f"{prefix}_r2": math.nan,
            f"{prefix}_pearson": math.nan,
        }
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else math.nan
    return {
        f"{prefix}_n": int(len(y_true)),
        f"{prefix}_mae": mae,
        f"{prefix}_rmse": rmse,
        f"{prefix}_r2": r2,
        f"{prefix}_pearson": pearson,
    }


def base_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str, huber_beta: float) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred, target, reduction="none")
    if loss_name == "mae":
        return F.l1_loss(pred, target, reduction="none")
    return F.smooth_l1_loss(pred, target, beta=huber_beta, reduction="none")


def weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    loss_name: str,
    huber_beta: float,
) -> torch.Tensor:
    losses = base_loss(pred, target, loss_name, huber_beta)
    weights = weights.to(losses.device, dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    graphs: list[GraphData],
    split_name: str,
    target_transform: TargetTransform,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    model.eval()
    rows = []
    all_true = []
    all_pred = []

    for graph in graphs:
        pred_model = model(graph)
        pred_raw = target_transform.inverse(pred_model)
        mask = graph.label_mask

        y_true = graph.y_raw[mask].detach().cpu().numpy()
        y_pred = pred_raw[mask].detach().cpu().numpy()
        all_true.append(y_true)
        all_pred.append(y_pred)

        core_idx = torch.where(graph.core_mask)[0].detach().cpu().numpy()
        pred_raw_cpu = pred_raw.detach().cpu().numpy()
        pred_model_cpu = pred_model.detach().cpu().numpy()
        y_raw_cpu = graph.y_raw.detach().cpu().numpy()
        label_cpu = graph.label_mask.detach().cpu().numpy()

        for idx in core_idx:
            rows.append(
                {
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
                    "y_pred": float(pred_raw_cpu[idx]),
                    "y_pred_model_space": float(pred_model_cpu[idx]),
                }
            )

    y_true_all = np.concatenate(all_true) if all_true else np.array([])
    y_pred_all = np.concatenate(all_pred) if all_pred else np.array([])
    return metric_dict(y_true_all, y_pred_all, split_name), pd.DataFrame(rows)


def train_one_epoch(
    model: nn.Module,
    graphs: list[GraphData],
    optimizer: torch.optim.Optimizer,
    loss_name: str,
    huber_beta: float,
    weight_config: WeightConfig,
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
        pred = model(graph)
        weights = weight_config.weights(graph, mask)
        loss = weighted_loss(pred[mask], graph.y[mask], weights, loss_name, huber_beta)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else math.nan


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def tile_metric_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = pred_df[pred_df["has_label"].astype(bool)].copy()
    for (split, tile), df in labels.groupby(["split", "tile_id"]):
        metrics = metric_dict(df["y_true"].to_numpy(float), df["y_pred"].to_numpy(float), "tile")
        rows.append(
            {
                "split": split,
                "tile_id": tile,
                "n": metrics["tile_n"],
                "mae": metrics["tile_mae"],
                "rmse": metrics["tile_rmse"],
                "r2": metrics["tile_r2"],
                "pearson": metrics["tile_pearson"],
                "y_mean": float(df["y_true"].mean()),
                "pred_mean": float(df["y_pred"].mean()),
                "y_min": float(df["y_true"].min()),
                "y_max": float(df["y_true"].max()),
                "pred_min": float(df["y_pred"].min()),
                "pred_max": float(df["y_pred"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "tile_id"]).reset_index(drop=True)


def baseline_metrics(pred_df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    labels = pred_df[pred_df["has_label"].astype(bool)].copy()
    train = labels[labels["split"].eq("train")]
    train_mean = float(train["y_true"].mean()) if len(train) else 0.0
    out = {}
    for split in ["train", "val", "test"]:
        df = labels[labels["split"].eq(split)].copy()
        pred = np.full(len(df), train_mean, dtype=float)
        out[split] = metric_dict(df["y_true"].to_numpy(float), pred, split)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphSAGE on SHdataset npz tile graphs.")
    parser.add_argument("--dataset-root", default="dataset/SHdataset")
    parser.add_argument("--output-dir", default="outputs/gnn/tile001_100_graphsage")
    parser.add_argument("--tile-start", type=int, default=1)
    parser.add_argument("--tile-end", type=int, default=100)
    parser.add_argument("--min-labeled-core", type=int, default=1)

    parser.add_argument("--split-mode", choices=["random", "kfold", "from_file"], default="random")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--loss", choices=["huber", "mse", "mae"], default="huber")
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument(
        "--feature-transform",
        choices=["standard", "skew_log1p_clip"],
        default="standard",
    )
    parser.add_argument("--feature-clip-quantile", type=float, default=0.995)
    parser.add_argument(
        "--target-transform",
        choices=["none", "zscore", "log1p", "log1p_zscore"],
        default="log1p_zscore",
    )
    parser.add_argument(
        "--loss-weight",
        choices=["uniform", "num_pano_sqrt", "num_pano_log1p", "y_quantile"],
        default="uniform",
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

    feature_transform = FeatureTransform(args.feature_transform, args.feature_clip_quantile)
    feature_transform.fit(graphs_by_split["train"], feature_names)
    for graphs in graphs_by_split.values():
        feature_transform.apply(graphs)

    x_mean, x_std = compute_feature_normalizer(graphs_by_split["train"])
    for graphs in graphs_by_split.values():
        apply_feature_normalizer(graphs, x_mean, x_std)

    target_transform = TargetTransform(args.target_transform)
    target_transform.fit(graphs_by_split["train"])
    for graphs in graphs_by_split.values():
        apply_target_transform(graphs, target_transform)

    weight_config = WeightConfig(
        mode=args.loss_weight,
        high_weight=args.high_y_weight,
        very_high_weight=args.very_high_y_weight,
    )
    weight_config.fit(graphs_by_split["train"])

    model = GraphSAGERegressor(
        in_dim=len(feature_names),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
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

    run_config = {
        "args": vars(args),
        "device": str(device),
        "feature_names": feature_names,
        "split": split_info,
        "label_counts": label_counts,
        "excluded_tiles": excluded,
        "feature_transform": feature_transform.to_dict(),
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

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    pred_df = pd.concat(prediction_frames, ignore_index=True)
    pred_df.to_csv(output_dir / "node_predictions.csv", index=False, encoding="utf-8-sig")

    tile_metrics = tile_metric_frame(pred_df)
    tile_metrics.to_csv(output_dir / "tile_metrics.csv", index=False, encoding="utf-8-sig")

    diagnostics = {
        "split": split_info,
        "label_counts": label_counts,
        "model_metrics": all_metrics,
        "baseline_train_mean": baseline_metrics(pred_df),
        "feature_transform": feature_transform.to_dict(),
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
        feature_transform=json.dumps(feature_transform.to_dict()),
        target_transform=json.dumps(target_transform.to_dict()),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "feature_names": feature_names,
            "metrics": all_metrics,
            "split": split_info,
            "feature_transform": feature_transform.to_dict(),
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
