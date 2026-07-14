import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import torch
from shapely.geometry import Point
from torch_geometric.data import Data


def build_segment_edges(roads):
    endpoints = []

    for idx, row in roads.iterrows():
        geom = row.geometry
        coords = list(geom.coords)

        p1 = Point(coords[0])
        p2 = Point(coords[-1])

        endpoints.append((idx, round(p1.x, 3), round(p1.y, 3)))
        endpoints.append((idx, round(p2.x, 3), round(p2.y, 3)))

    endpoint_df = pd.DataFrame(endpoints, columns=["idx", "x", "y"])

    edges = set()

    for _, group in endpoint_df.groupby(["x", "y"]):
        ids = group["idx"].tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                edges.add((ids[i], ids[j]))
                edges.add((ids[j], ids[i]))

    return list(edges)


def make_tile_data(tile_roads, feature_cols):
    tile_roads = tile_roads.reset_index(drop=True)

    x = torch.tensor(
        tile_roads[feature_cols].fillna(0).values,
        dtype=torch.float
    )

    y = torch.tensor(
        tile_roads["y"].fillna(0).values,
        dtype=torch.float
    )

    has_label = torch.tensor(
        tile_roads["has_label"].fillna(False).values,
        dtype=torch.bool
    )

    core_mask = torch.tensor(
        tile_roads["is_core"].fillna(True).values,
        dtype=torch.bool
    )

    edge_pairs = build_segment_edges(tile_roads)

    if len(edge_pairs) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).T

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        has_label=has_label,
        core_mask=core_mask,
    )

    data.road_seg_id = tile_roads["road_seg_id"].astype(str).tolist()
    data.tile_id = str(tile_roads["tile_id"].iloc[0])

    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--roads", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", default="data/processed/graphs")

    args = parser.parse_args()

    roads = gpd.read_file(args.roads).to_crs(3857)
    labels = pd.read_csv(args.labels)

    labels["actual_year"] = pd.to_datetime(
        labels["actual_date"], errors="coerce"
    ).dt.year

    labels = labels[
        labels["actual_year"].isin([2018, 2019, 2021, 2022])
    ].copy()

    seg_labels = (
        labels.groupby("road_seg_id", as_index=False)
        .agg(
            person_count_mean=("person_count_final", "mean"),
            n_images=("image_id", "count"),
        )
    )

    seg_labels["y"] = np.log1p(seg_labels["person_count_mean"])
    seg_labels["has_label"] = True

    roads = roads.merge(seg_labels, on="road_seg_id", how="left")
    roads["has_label"] = roads["has_label"].fillna(False)

    feature_cols = [
        "length_m",
        "bearing_sin",
        "bearing_cos",
        "straightness",
        "building_coverage_200",
        "far_200",
        "dist_nearest_metro",
        "dist_nearest_bus",
        "betweenness",
        "closeness",
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tile_id, tile_roads in roads.groupby("tile_id"):
        data = make_tile_data(tile_roads, feature_cols)
        torch.save(data, output_dir / f"tile_{tile_id}.pt")

    print(f"完成：{roads['tile_id'].nunique()} 个 tile graph")


if __name__ == "__main__":
    main()