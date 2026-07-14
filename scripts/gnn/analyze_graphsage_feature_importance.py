from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_graphsage_npz import (
    FeatureTransform,
    GraphData,
    GraphSAGERegressor,
    TargetTransform,
    apply_feature_normalizer,
    load_graph,
    metric_dict,
)


ROAD_GEOMETRY = {"seg_length_m", "bearing_norm", "straight_line_ratio"}
LOCAL_NETWORK = {"LConn", "LLen", "LFrac", "LAC", "LSin", "LBear"}
BUILDING_PREFIXES = {
    "bldg_coverage",
    "far_approx",
    "bldg_count",
    "avg_floor_area",
    "floor_area_cv",
    "avg_height",
    "height_cv",
    "max_height",
    "min_height",
}
NETWORK_800 = {"MAD800", "NQPDA800", "BtA800", "TPBtA800", "TPD800", "Lnk800", "Len800", "AngD800", "Wt800", "MGLA800", "MCF800", "DivA800"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_tile_number(path_or_name: str | Path) -> int:
    return int(Path(path_or_name).stem.split("_")[-1])


def feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "road_geometry": [],
        "building_50m": [],
        "building_200m": [],
        "building_800m": [],
        "transit_50m": [],
        "transit_200m": [],
        "transit_800m": [],
        "population_50m": [],
        "population_200m": [],
        "population_800m": [],
        "local_network": [],
        "network_800m": [],
        "other": [],
    }

    for name in feature_names:
        if name in ROAD_GEOMETRY:
            groups["road_geometry"].append(name)
        elif name in LOCAL_NETWORK:
            groups["local_network"].append(name)
        elif name in NETWORK_800:
            groups["network_800m"].append(name)
        elif name in {"bus_count_50m", "metro_count_50m"}:
            groups["transit_50m"].append(name)
        elif name in {"bus_count_200m", "metro_count_200m"}:
            groups["transit_200m"].append(name)
        elif name in {"bus_count_800m", "metro_count_800m"}:
            groups["transit_800m"].append(name)
        elif name in {"pop_sum_50m", "pop_density_50m"}:
            groups["population_50m"].append(name)
        elif name in {"pop_sum_200m", "pop_density_200m"}:
            groups["population_200m"].append(name)
        elif name in {"pop_sum_800m", "pop_density_800m"}:
            groups["population_800m"].append(name)
        elif any(name.startswith(prefix) for prefix in BUILDING_PREFIXES):
            if name.endswith("_50m"):
                groups["building_50m"].append(name)
            elif name.endswith("_200m"):
                groups["building_200m"].append(name)
            elif name.endswith("_800m"):
                groups["building_800m"].append(name)
            else:
                groups["other"].append(name)
        else:
            groups["other"].append(name)

    return {key: value for key, value in groups.items() if value}


def make_feature_transform(ft_dict: dict, feature_names: list[str]) -> FeatureTransform:
    ft = FeatureTransform(
        mode=ft_dict.get("mode", "standard"),
        clip_quantile=float(ft_dict.get("clip_quantile", 0.999)),
    )
    ft.feature_names = list(feature_names)
    ft.clip_lower = list(ft_dict.get("clip_lower", [-math.inf] * len(feature_names)))
    ft.clip_upper = list(ft_dict.get("clip_upper", [math.inf] * len(feature_names)))
    by_name = {name: idx for idx, name in enumerate(feature_names)}
    ft.log_indices = [by_name[name] for name in ft_dict.get("log_features", []) if name in by_name]
    ft.clip_indices = [by_name[name] for name in ft_dict.get("clip_features", []) if name in by_name]
    return ft


def make_target_transform(target_dict: dict) -> TargetTransform:
    transform = TargetTransform(mode=target_dict.get("mode", "none"))
    for key in ["mean", "std", "log_mean", "log_std"]:
        if key in target_dict:
            setattr(transform, key, float(target_dict[key]))
    return transform


def load_split_graphs(run_dir: Path, split_name: str, device: torch.device) -> tuple[list[GraphData], dict, list[str]]:
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    args = checkpoint["args"]
    feature_names = [str(x) for x in checkpoint["feature_names"]]
    dataset_root = Path(args.get("dataset_root", "dataset/SHdataset"))
    split_tiles = checkpoint["split"][split_name]
    paths = [dataset_root / f"{tile}.npz" for tile in split_tiles]
    paths = sorted(paths, key=parse_tile_number)
    graphs = [load_graph(path, device) for path in paths]

    ft = make_feature_transform(checkpoint.get("feature_transform", {}), feature_names)
    ft.apply(graphs)

    normalizer = np.load(run_dir / "normalizer.npz", allow_pickle=True)
    x_mean = torch.tensor(normalizer["mean"], dtype=torch.float32, device=device)
    x_std = torch.tensor(normalizer["std"], dtype=torch.float32, device=device)
    apply_feature_normalizer(graphs, x_mean, x_std)
    return graphs, checkpoint, feature_names


def load_model(run_dir: Path, checkpoint: dict, feature_count: int, device: torch.device) -> tuple[GraphSAGERegressor, TargetTransform]:
    args = checkpoint["args"]
    model = GraphSAGERegressor(
        in_dim=feature_count,
        hidden_dim=int(args["hidden_dim"]),
        num_layers=int(args["num_layers"]),
        dropout=float(args["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    target_transform = make_target_transform(checkpoint.get("target_transform", {"mode": args.get("target_transform", "none")}))
    return model, target_transform


@torch.no_grad()
def evaluate_split(model: GraphSAGERegressor, graphs: list[GraphData], target_transform: TargetTransform, split_name: str) -> dict:
    y_true_all = []
    y_pred_all = []
    for graph in graphs:
        pred_model = model(graph)
        pred_raw = target_transform.inverse(pred_model)
        mask = graph.label_mask
        y_true_all.append(graph.y_raw[mask].detach().cpu().numpy())
        y_pred_all.append(pred_raw[mask].detach().cpu().numpy())
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    return metric_dict(y_true, y_pred, split_name)


def permute_columns(graphs: list[GraphData], indices: list[int], rng: np.random.Generator) -> list[tuple[GraphData, torch.Tensor]]:
    originals = []
    for graph in graphs:
        original = graph.x[:, indices].clone()
        if graph.x.shape[0] > 1:
            perm = torch.tensor(rng.permutation(graph.x.shape[0]), dtype=torch.long, device=graph.x.device)
            graph.x[:, indices] = original[perm]
        originals.append((graph, original))
    return originals


def restore_columns(indices: list[int], originals: list[tuple[GraphData, torch.Tensor]]) -> None:
    for graph, original in originals:
        graph.x[:, indices] = original


def permutation_importance(
    model: GraphSAGERegressor,
    graphs: list[GraphData],
    target_transform: TargetTransform,
    split_name: str,
    items: dict[str, list[int]],
    baseline: dict,
    repeats: int,
    seed: int,
    item_type: str,
) -> pd.DataFrame:
    rows = []
    base_rmse = float(baseline[f"{split_name}_rmse"])
    base_r2 = float(baseline[f"{split_name}_r2"])
    base_mae = float(baseline[f"{split_name}_mae"])
    base_pearson = float(baseline[f"{split_name}_pearson"])

    for item_name, indices in items.items():
        repeat_rows = []
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + repeat * 1009 + sum(indices) * 17 + len(indices) * 31)
            originals = permute_columns(graphs, indices, rng)
            metrics = evaluate_split(model, graphs, target_transform, split_name)
            restore_columns(indices, originals)
            repeat_rows.append(metrics)

        rmse_values = np.array([row[f"{split_name}_rmse"] for row in repeat_rows], dtype=float)
        r2_values = np.array([row[f"{split_name}_r2"] for row in repeat_rows], dtype=float)
        mae_values = np.array([row[f"{split_name}_mae"] for row in repeat_rows], dtype=float)
        pearson_values = np.array([row[f"{split_name}_pearson"] for row in repeat_rows], dtype=float)
        rows.append(
            {
                "item_type": item_type,
                "name": item_name,
                "n_features": len(indices),
                "feature_indices": ",".join(str(i) for i in indices),
                "baseline_rmse": base_rmse,
                "permuted_rmse_mean": float(np.nanmean(rmse_values)),
                "permuted_rmse_std": float(np.nanstd(rmse_values, ddof=1)) if repeats > 1 else 0.0,
                "rmse_delta": float(np.nanmean(rmse_values) - base_rmse),
                "rmse_delta_pct": float((np.nanmean(rmse_values) - base_rmse) / base_rmse * 100.0) if base_rmse else math.nan,
                "baseline_r2": base_r2,
                "permuted_r2_mean": float(np.nanmean(r2_values)),
                "r2_drop": float(base_r2 - np.nanmean(r2_values)),
                "baseline_mae": base_mae,
                "permuted_mae_mean": float(np.nanmean(mae_values)),
                "mae_delta": float(np.nanmean(mae_values) - base_mae),
                "baseline_pearson": base_pearson,
                "permuted_pearson_mean": float(np.nanmean(pearson_values)),
                "pearson_drop": float(base_pearson - np.nanmean(pearson_values)),
            }
        )

    return pd.DataFrame(rows).sort_values(["rmse_delta", "r2_drop"], ascending=False).reset_index(drop=True)


def feature_correlations(graphs: list[GraphData], feature_names: list[str]) -> pd.DataFrame:
    xs = []
    ys = []
    for graph in graphs:
        mask = graph.label_mask
        xs.append(graph.x[mask].detach().cpu().numpy())
        ys.append(graph.y_raw[mask].detach().cpu().numpy())
    x = np.vstack(xs)
    y = np.concatenate(ys)
    df = pd.DataFrame(x, columns=feature_names)
    out = []
    y_series = pd.Series(y)
    y_rank = y_series.rank(method="average")
    for name in feature_names:
        s = df[name]
        s_rank = s.rank(method="average")
        pearson = float(s.corr(y_series, method="pearson")) if s.std() > 0 else math.nan
        spearman = float(s_rank.corr(y_rank, method="pearson")) if s.std() > 0 else math.nan
        out.append(
            {
                "feature": name,
                "pearson_corr": pearson,
                "spearman_corr": spearman,
                "abs_spearman_corr": float(abs(spearman)) if np.isfinite(spearman) else math.nan,
            }
        )
    return pd.DataFrame(out).sort_values("abs_spearman_corr", ascending=False).reset_index(drop=True)


def plot_bar(df: pd.DataFrame, value_col: str, label_col: str, title: str, out: Path, top_n: int | None = None) -> None:
    data = df.copy()
    if top_n is not None:
        data = data.head(top_n)
    data = data.sort_values(value_col, ascending=True)
    height = max(4.5, 0.32 * len(data) + 1.5)
    fig, ax = plt.subplots(figsize=(9, height))
    ax.barh(data[label_col], data[value_col], color="#4472c4")
    ax.axvline(0, color="black", lw=1)
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    out_dir: Path,
    run_dir: Path,
    split_name: str,
    baseline: dict,
    group_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> None:
    top_groups = group_df.head(8)
    top_features = feature_df.head(15)
    weak_features = feature_df.sort_values("rmse_delta", ascending=True).head(15)
    lines = [
        "# GraphSAGE Feature Influence Analysis",
        "",
        f"Source run: `{run_dir}`",
        f"Evaluated split: `{split_name}`",
        "",
        "Method: permutation importance. A feature or feature group is shuffled on the evaluated graphs; if RMSE rises and R2 drops, the model relied more on that variable. Correlation direction is only a rough reference because the model is nonlinear and uses graph neighborhood aggregation.",
        "",
        "## Baseline Regression",
        "",
        f"- RMSE: `{baseline[f'{split_name}_rmse']:.6f}`",
        f"- MAE: `{baseline[f'{split_name}_mae']:.6f}`",
        f"- R2: `{baseline[f'{split_name}_r2']:.6f}`",
        f"- Pearson: `{baseline[f'{split_name}_pearson']:.6f}`",
        "",
        "## Most Influential Feature Groups",
        "",
        "| rank | group | n_features | RMSE delta | R2 drop |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for idx, row in top_groups.iterrows():
        lines.append(
            f"| {idx + 1} | {row['name']} | {int(row['n_features'])} | {row['rmse_delta']:.6f} | {row['r2_drop']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Most Influential Single Features",
            "",
            "| rank | feature | RMSE delta | R2 drop | Spearman corr |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    corr_by_feature = corr_df.set_index("feature")
    for idx, row in top_features.iterrows():
        corr = corr_by_feature.loc[row["name"], "spearman_corr"] if row["name"] in corr_by_feature.index else math.nan
        lines.append(f"| {idx + 1} | {row['name']} | {row['rmse_delta']:.6f} | {row['r2_drop']:.6f} | {corr:.6f} |")

    lines.extend(
        [
            "",
            "## Weak or Unstable Single Features",
            "",
            "Negative or near-zero RMSE delta means shuffling that feature did not hurt the model on this split; it can be weak, redundant with other variables, or noisy.",
            "",
            "| feature | RMSE delta | R2 drop | Spearman corr |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in weak_features.iterrows():
        corr = corr_by_feature.loc[row["name"], "spearman_corr"] if row["name"] in corr_by_feature.index else math.nan
        lines.append(f"| {row['name']} | {row['rmse_delta']:.6f} | {row['r2_drop']:.6f} | {corr:.6f} |")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `group_permutation_importance.csv`",
            "- `feature_permutation_importance.csv`",
            "- `feature_correlations.csv`",
            "- `fig_group_importance.png`",
            "- `fig_feature_importance_top20.png`",
            "- `fig_feature_correlation_top20.png`",
        ]
    )
    (out_dir / "特征影响分析报告.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze feature influence for a trained GraphSAGE regression model.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    graphs, checkpoint, feature_names = load_split_graphs(run_dir, args.split, device)
    model, target_transform = load_model(run_dir, checkpoint, len(feature_names), device)

    baseline = evaluate_split(model, graphs, target_transform, args.split)
    feature_items = {name: [idx] for idx, name in enumerate(feature_names)}
    groups_by_name = feature_groups(feature_names)
    group_items = {name: [feature_names.index(feature) for feature in features] for name, features in groups_by_name.items()}

    group_df = permutation_importance(
        model,
        graphs,
        target_transform,
        args.split,
        group_items,
        baseline,
        repeats=args.repeats,
        seed=args.seed,
        item_type="group",
    )
    feature_df = permutation_importance(
        model,
        graphs,
        target_transform,
        args.split,
        feature_items,
        baseline,
        repeats=args.repeats,
        seed=args.seed + 777,
        item_type="feature",
    )
    corr_df = feature_correlations(graphs, feature_names)

    group_df.to_csv(out_dir / "group_permutation_importance.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(out_dir / "feature_permutation_importance.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(out_dir / "feature_correlations.csv", index=False, encoding="utf-8-sig")

    plot_bar(group_df, "rmse_delta", "name", "Feature Group Permutation Importance", out_dir / "fig_group_importance.png")
    plot_bar(feature_df, "rmse_delta", "name", "Top Single Feature Permutation Importance", out_dir / "fig_feature_importance_top20.png", top_n=20)
    corr_plot = corr_df.assign(abs_spearman_signed=corr_df["spearman_corr"])
    corr_plot = corr_plot.reindex(corr_plot["spearman_corr"].abs().sort_values(ascending=False).index)
    plot_bar(corr_plot, "spearman_corr", "feature", "Top Feature Spearman Correlations", out_dir / "fig_feature_correlation_top20.png", top_n=20)
    write_report(out_dir, run_dir, args.split, baseline, group_df, feature_df, corr_df)

    print(json.dumps({"baseline": baseline, "output_dir": str(out_dir)}, ensure_ascii=False, indent=2))
    print("Top groups:")
    print(group_df[["name", "n_features", "rmse_delta", "r2_drop"]].head(12).to_string(index=False))
    print("Top features:")
    print(feature_df[["name", "rmse_delta", "r2_drop"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
