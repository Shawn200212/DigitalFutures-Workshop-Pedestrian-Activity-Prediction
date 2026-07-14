from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect_core_labels(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(dataset_root.glob("tile_*.npz")):
        z = np.load(path, allow_pickle=True)
        y = z["y"].astype(float)
        core = z["core_mask"].astype(bool)
        pano = z["num_pano_images"].astype(int)
        finite = np.isfinite(y)
        for idx in np.where(core & finite)[0]:
            rows.append(
                {
                    "tile_id": path.stem,
                    "node_index": int(idx),
                    "y": float(y[idx]),
                    "num_pano_images": int(pano[idx]),
                }
            )
    return pd.DataFrame(rows)


def copy_with_modified_y(source_root: Path, output_root: Path, variant: str, thresholds: dict[str, float]) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    total_changed = 0
    total_labels = 0

    for path in sorted(source_root.glob("tile_*.npz")):
        z = np.load(path, allow_pickle=True)
        arrays = {key: z[key] for key in z.files}
        y = arrays["y"].astype(np.float32).copy()
        core = arrays["core_mask"].astype(bool)
        pano = arrays["num_pano_images"].astype(int)
        finite = np.isfinite(y)
        label_mask = core & finite

        changed = np.zeros_like(label_mask, dtype=bool)
        original_y = y.copy()

        if variant == "drop_y10_pano1":
            changed = label_mask & (y > 10.0) & (pano <= 1)
            y[changed] = np.nan
        elif variant == "drop_y13_pano2":
            changed = label_mask & (y > thresholds["p995"]) & (pano <= 2)
            y[changed] = np.nan
        elif variant == "drop_y_p99":
            changed = label_mask & (y > thresholds["p99"])
            y[changed] = np.nan
        elif variant == "clip_p995":
            changed = label_mask & (y > thresholds["p995"])
            y[changed] = thresholds["p995"]
        elif variant == "clip_p99":
            changed = label_mask & (y > thresholds["p99"])
            y[changed] = thresholds["p99"]
        else:
            raise ValueError(f"Unknown variant: {variant}")

        arrays["y"] = y
        np.savez_compressed(output_root / path.name, **arrays)

        total_changed += int(changed.sum())
        total_labels += int(label_mask.sum())
        if bool(changed.any()):
            for idx in np.where(changed)[0]:
                audit_rows.append(
                    {
                        "variant": variant,
                        "tile_id": path.stem,
                        "node_index": int(idx),
                        "old_y": float(original_y[idx]),
                        "new_y": float(y[idx]) if np.isfinite(y[idx]) else np.nan,
                        "num_pano_images": int(pano[idx]),
                    }
                )

    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit.to_csv(output_root / "label_clean_audit.csv", index=False, encoding="utf-8-sig")

    meta = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "variant": variant,
        "thresholds": thresholds,
        "total_labels": total_labels,
        "changed_labels": total_changed,
        "changed_share": total_changed / total_labels if total_labels else 0.0,
    }
    (output_root / "label_clean_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build label-cleaned SHdataset variants.")
    parser.add_argument("--dataset-root", default="dataset/SHdataset")
    parser.add_argument("--output-parent", default="dataset")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["drop_y10_pano1", "drop_y13_pano2", "drop_y_p99", "clip_p995", "clip_p99"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.dataset_root)
    output_parent = Path(args.output_parent)

    labels = collect_core_labels(source_root)
    thresholds = {
        "p95": float(labels["y"].quantile(0.95)),
        "p975": float(labels["y"].quantile(0.975)),
        "p99": float(labels["y"].quantile(0.99)),
        "p995": float(labels["y"].quantile(0.995)),
        "p999": float(labels["y"].quantile(0.999)),
        "max": float(labels["y"].max()),
    }
    labels.to_csv(output_parent / "label_clean_source_labels.csv", index=False, encoding="utf-8-sig")

    metas = []
    for variant in args.variants:
        output_root = output_parent / f"SHdataset_label_{variant}"
        metas.append(copy_with_modified_y(source_root, output_root, variant, thresholds))

    summary = pd.DataFrame(metas)
    summary.to_csv(output_parent / "label_clean_variants_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
