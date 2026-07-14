from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


FOCUSED_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "standard_h64_d015_panosqrt",
        "feature_transform": "standard",
        "feature_clip_quantile": 0.999,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.15,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "loss": "huber",
        "loss_weight": "num_pano_sqrt",
    },
    {
        "name": "skew999_h64_d015_panosqrt",
        "feature_transform": "skew_log1p_clip",
        "feature_clip_quantile": 0.999,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.15,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "loss": "huber",
        "loss_weight": "num_pano_sqrt",
    },
    {
        "name": "skew997_h64_d015_panosqrt",
        "feature_transform": "skew_log1p_clip",
        "feature_clip_quantile": 0.997,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.15,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "loss": "huber",
        "loss_weight": "num_pano_sqrt",
    },
    {
        "name": "skew9995_h64_d015_panosqrt",
        "feature_transform": "skew_log1p_clip",
        "feature_clip_quantile": 0.9995,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.15,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "loss": "huber",
        "loss_weight": "num_pano_sqrt",
    },
    {
        "name": "skew999_h64_d010_panosqrt",
        "feature_transform": "skew_log1p_clip",
        "feature_clip_quantile": 0.999,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.10,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "loss": "huber",
        "loss_weight": "num_pano_sqrt",
    },
    {
        "name": "skew999_h64_d020_panosqrt",
        "feature_transform": "skew_log1p_clip",
        "feature_clip_quantile": 0.999,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.20,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "loss": "huber",
        "loss_weight": "num_pano_sqrt",
    },
    {
        "name": "skew999_h96_d015_panosqrt",
        "feature_transform": "skew_log1p_clip",
        "feature_clip_quantile": 0.999,
        "hidden_dim": 96,
        "num_layers": 2,
        "dropout": 0.15,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "loss": "huber",
        "loss_weight": "num_pano_sqrt",
    },
    {
        "name": "skew999_h128_d015_panosqrt",
        "feature_transform": "skew_log1p_clip",
        "feature_clip_quantile": 0.999,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.15,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "loss": "huber",
        "loss_weight": "num_pano_sqrt",
    },
    {
        "name": "skew999_h64_d015_uniform",
        "feature_transform": "skew_log1p_clip",
        "feature_clip_quantile": 0.999,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.15,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "loss": "huber",
        "loss_weight": "uniform",
    },
    {
        "name": "skew999_h64_d015_panolog",
        "feature_transform": "skew_log1p_clip",
        "feature_clip_quantile": 0.999,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.15,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "loss": "huber",
        "loss_weight": "num_pano_log1p",
    },
    {
        "name": "skew999_h64_d015_yquantile",
        "feature_transform": "skew_log1p_clip",
        "feature_clip_quantile": 0.999,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.15,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "loss": "huber",
        "loss_weight": "y_quantile",
        "high_y_weight": 1.75,
        "very_high_y_weight": 2.5,
    },
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_args(config: dict[str, Any]) -> list[str]:
    args = [
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--num-layers",
        str(config["num_layers"]),
        "--dropout",
        str(config["dropout"]),
        "--lr",
        str(config["lr"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--loss",
        str(config["loss"]),
        "--target-transform",
        "log1p_zscore",
        "--loss-weight",
        str(config["loss_weight"]),
        "--feature-transform",
        str(config["feature_transform"]),
        "--feature-clip-quantile",
        str(config["feature_clip_quantile"]),
    ]
    if "high_y_weight" in config:
        args.extend(["--high-y-weight", str(config["high_y_weight"])])
    if "very_high_y_weight" in config:
        args.extend(["--very-high-y-weight", str(config["very_high_y_weight"])])
    return args


def build_command(args: argparse.Namespace, config: dict[str, Any], fold: int, output_dir: Path) -> list[str]:
    train_script = repo_root() / "scripts" / "gnn" / "train_graphsage_npz.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--dataset-root",
        args.dataset_root,
        "--output-dir",
        str(output_dir),
        "--tile-start",
        str(args.tile_start),
        "--tile-end",
        str(args.tile_end),
        "--split-mode",
        "kfold",
        "--k-folds",
        str(args.k_folds),
        "--fold-index",
        str(fold),
        "--seed",
        str(args.seed),
        "--val-ratio",
        str(args.val_ratio),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--huber-beta",
        str(args.huber_beta),
        "--grad-clip",
        str(args.grad_clip),
        "--print-every",
        str(args.print_every),
        "--device",
        args.device,
    ]
    cmd.extend(config_args(config))
    return cmd


def row_from_run(config: dict[str, Any], fold: int, output_dir: Path) -> dict[str, Any]:
    metrics = read_json(output_dir / "metrics.json")
    run_config = read_json(output_dir / "run_config.json")
    args = run_config.get("args", {})
    row = {
        "config_name": config["name"],
        "fold": fold,
        "output_dir": str(output_dir),
        "best_epoch": metrics.get("best_epoch"),
        "feature_transform": args.get("feature_transform"),
        "feature_clip_quantile": args.get("feature_clip_quantile"),
        "hidden_dim": args.get("hidden_dim"),
        "num_layers": args.get("num_layers"),
        "dropout": args.get("dropout"),
        "lr": args.get("lr"),
        "weight_decay": args.get("weight_decay"),
        "loss": args.get("loss"),
        "loss_weight": args.get("loss_weight"),
        "target_transform": args.get("target_transform"),
    }
    row.update(metrics)
    return row


def aggregate(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [
        "val_mae",
        "val_rmse",
        "val_r2",
        "val_pearson",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_pearson",
        "train_mae",
        "train_rmse",
    ]
    grouped = (
        df.groupby(
            [
                "config_name",
                "feature_transform",
                "feature_clip_quantile",
                "hidden_dim",
                "num_layers",
                "dropout",
                "lr",
                "weight_decay",
                "loss",
                "loss_weight",
            ],
            dropna=False,
        )[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([str(x) for x in col if str(x)])
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns
    ]
    grouped["folds_completed"] = df.groupby("config_name")["fold"].count().reindex(grouped["config_name"]).to_numpy()
    grouped["selection_score"] = grouped["val_rmse_mean"] + 0.25 * grouped["val_rmse_std"].fillna(0)
    return grouped.sort_values(["selection_score", "val_rmse_mean", "val_mae_mean"]).reset_index(drop=True)


def write_markdown(summary: pd.DataFrame, output_root: Path) -> None:
    best = summary.iloc[0]
    lines = [
        "# GraphSAGE 5-Fold CV Hyperparameter Sweep",
        "",
        "Selection rule: rank by `mean(val_rmse) + 0.25 * std(val_rmse)`; test metrics are reported after selection.",
        "",
        "## Best Configuration",
        "",
        f"- config_name: `{best['config_name']}`",
        f"- feature_transform: `{best['feature_transform']}`",
        f"- feature_clip_quantile: `{best['feature_clip_quantile']}`",
        f"- hidden_dim: `{best['hidden_dim']}`",
        f"- num_layers: `{best['num_layers']}`",
        f"- dropout: `{best['dropout']}`",
        f"- loss: `{best['loss']}`",
        f"- loss_weight: `{best['loss_weight']}`",
        f"- mean val RMSE: `{best['val_rmse_mean']:.6f}`",
        f"- std val RMSE: `{best['val_rmse_std']:.6f}`",
        f"- mean val MAE: `{best['val_mae_mean']:.6f}`",
        f"- mean val R2: `{best['val_r2_mean']:.6f}`",
        f"- mean test RMSE: `{best['test_rmse_mean']:.6f}`",
        f"- mean test MAE: `{best['test_mae_mean']:.6f}`",
        f"- mean test R2: `{best['test_r2_mean']:.6f}`",
        "",
        "## Ranking",
        "",
        "| rank | config | val_rmse_mean | val_rmse_std | val_mae_mean | val_r2_mean | test_rmse_mean | test_r2_mean |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in summary.iterrows():
        lines.append(
            f"| {idx + 1} | {row['config_name']} | {row['val_rmse_mean']:.6f} | "
            f"{row['val_rmse_std']:.6f} | {row['val_mae_mean']:.6f} | {row['val_r2_mean']:.6f} | "
            f"{row['test_rmse_mean']:.6f} | {row['test_r2_mean']:.6f} |"
        )
    (output_root / "cv_sweep_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused 5-fold CV sweep for GraphSAGE.")
    parser.add_argument("--dataset-root", default="dataset/SHdataset")
    parser.add_argument("--output-root", default="outputs/gnn/cv_sweep_graphsage_skewlog")
    parser.add_argument("--tile-start", type=int, default=1)
    parser.add_argument("--tile-end", type=int, default=100)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(FOCUSED_CONFIGS) * args.k_folds
    step = 0
    for config in FOCUSED_CONFIGS:
        for fold in range(args.k_folds):
            step += 1
            output_dir = output_root / config["name"] / f"fold_{fold}"
            metrics_path = output_dir / "metrics.json"
            if metrics_path.exists() and not args.force:
                print(f"[{step}/{total}] skip existing {config['name']} fold={fold}", flush=True)
                rows.append(row_from_run(config, fold, output_dir))
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_command(args, config, fold, output_dir)
            print(f"[{step}/{total}] run {config['name']} fold={fold}", flush=True)
            print(" ".join(cmd), flush=True)
            result = subprocess.run(cmd, cwd=repo_root(), text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Training failed for {config['name']} fold {fold} with return code {result.returncode}")
            rows.append(row_from_run(config, fold, output_dir))

            fold_df = pd.DataFrame(rows)
            fold_df.to_csv(output_root / "cv_sweep_fold_results.csv", index=False, encoding="utf-8-sig")
            summary = aggregate(rows)
            summary.to_csv(output_root / "cv_sweep_config_summary.csv", index=False, encoding="utf-8-sig")
            write_markdown(summary, output_root)
            leader = summary.iloc[0]
            print(
                {
                    "current_leader": leader["config_name"],
                    "val_rmse_mean": round(float(leader["val_rmse_mean"]), 6),
                    "val_rmse_std": round(float(leader["val_rmse_std"]), 6)
                    if pd.notna(leader["val_rmse_std"])
                    else None,
                    "folds_done": len(rows),
                },
                flush=True,
            )

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(output_root / "cv_sweep_fold_results.csv", index=False, encoding="utf-8-sig")
    summary = aggregate(rows)
    summary.to_csv(output_root / "cv_sweep_config_summary.csv", index=False, encoding="utf-8-sig")
    write_markdown(summary, output_root)
    print("FINAL_CV_SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"Saved CV sweep outputs to: {output_root}", flush=True)


if __name__ == "__main__":
    main()
