# Stage 0 — Street-view image acquisition

Upstream of everything in [`app/`](../app) and [`scripts/`](../scripts): this module turns a set of
sampled road points into the panorama / perspective-view imagery that supplies the **labels** for
the GraphSAGE model. It is deliberately kept **self-contained and isolated** from the research
pipeline — separate requirements, no imports across the boundary, and no dependency on any model
code. Running the rest of the repository does not require running this module.

> **Scope note** — imagery is used as *supervision only* and never enters the feature space
> (see the root [README](../README.md)). No street-view images are redistributed with this
> repository; this directory contains the acquisition code and the sampling points only.

## Contents

```text
data_acquisition/
├── stvscrawler/            # Baidu street-view client package
│   ├── stvscrawl.py        #   panorama/tile fetching, sid & metadata lookup
│   ├── ctr.py              #   Baidu Mercator ↔ lon/lat correction polynomials
│   ├── wgs2bd.py           #   WGS-84 ↔ BD-09 / BD Mercator conversion
│   └── time_selector.py    #   capture-date selection helpers
├── flat_images_pano.py     # panorama → 4 × 720×720 perspective views (0°/90°/180°/270°)
├── run_all_tiles.py        # batch orchestration over all 100 tiles + MASTER manifests
├── tiles/tile_001..100.csv # sampling points per tile (OID_, POINT_X, POINT_Y; WGS-84)
└── requirements.txt
```

`tiles/*.csv` are the 100 spatial tiles covering Shanghai inside the Outer Ring Road
(1.5 km × 1.5 km cores) — 0.6 MB of longitude/latitude sample points, no personal data.

## How it works

1. **Point → panorama.** For each sample point, `stvscrawler` converts WGS-84 to Baidu Mercator,
   resolves the nearest street-view `sid`, and reads its capture-date metadata.
2. **Date selection.** `TimeFuncs` picks the capture closest to the configured `TARGET_DATES`
   (default: quarterly, 2017-03 → 2019-12) so that labels come from a comparable time window.
3. **One download, four views.** `flat_images_pano.py` downloads the equirectangular panorama
   **once** per point/date, then reprojects locally into four orthogonal 720×720 perspective
   images — roughly a 67% reduction in network requests versus fetching each view separately.
   The raw 360° panorama is kept alongside the four views.
4. **Batch + resume.** `run_all_tiles.py` walks `tiles/tile_*.csv` in order, writes each tile to
   its own folder, and refreshes the MASTER manifests after *every* tile, so an interrupted run
   can be resumed and the summary tables rebuilt from existing per-tile logs.

Output layout (written outside the repository — see below):

```text
<SVI_DATA_ROOT>/
├── tile001/
│   ├── actual_YYYY_MM/      # 2019_01.121.40123456.31.23010000.dir_000.png ...
│   ├── panorama_YYYY_MM/    # ...pano.png
│   ├── tile001_crawl_log.csv
│   └── tile001_flat_image_manifest_for_training.csv
├── tile002/ ...
├── MASTER_crawl_log_all_tiles.csv
├── MASTER_flat_image_manifest_for_training.csv
├── MASTER_batch_summary.csv
└── MASTER_progress.json
```

`MASTER_flat_image_manifest_for_training.csv` is the hand-off point: it is what
[`scripts/prepare_yolo_manifest_from_tiles.py`](../scripts/prepare_yolo_manifest_from_tiles.py)
consumes to start Stage 1.

## Usage

```bash
pip install -r data_acquisition/requirements.txt
```

Output goes to `<repo>/data/images/` by default (already gitignored). Override with an
environment variable if you want it on another disk:

```bash
set SVI_DATA_ROOT=D:/svi/images
```

Run the full 100-tile job:

```bash
python data_acquisition/run_all_tiles.py
```

Tunables live in the clearly marked config block at the top of `run_all_tiles.py`
(`TARGET_DATES`, `Z_LEVEL` panorama resolution, `FLAT_IMG_SIZE`, `DIRECTIONS`, `YAW_OFFSET`,
resume and duplicate policy). Those values override the defaults in `flat_images_pano.py`.

A single tile can also be run directly by editing `input_csv` / `out_dir` at the top of
`flat_images_pano.py` and running it standalone.

### Notes on `YAW_OFFSET`

Baidu's internal panorama orientation is offset by roughly 90° from the `pr3d` API direction —
i.e. API `direction=0` (road heading) corresponds to `yaw=90°` in the panorama. `YAW_OFFSET = 90`
compensates for this. Change it only if you have verified the offset for your own data.

## Provenance and terms of use

- The `stvscrawler/` package is **adapted from third-party open-source Baidu street-view crawler
  code**; it carries no upstream author header in the copy we worked from, so it is included here
  as derived work rather than as original contribution. If you are the original author, please
  open an issue and proper attribution will be added.
- `flat_images_pano.py`, `run_all_tiles.py` and the tile sampling points are this project's own.
- The client calls **undocumented** Baidu endpoints (`mapsv0/mapsv1.bdimg.com`) and rotates
  User-Agent strings. This is provided for **academic research use**. Anyone running it is
  responsible for complying with the data provider's terms of service and for keeping request
  rates reasonable — `N_TRIES`, the per-tile pacing and the resume mechanism exist so that a
  study-area crawl can be run slowly and restarted rather than hammered in one pass.
- No API keys or credentials are required, and none are stored in this repository.
- Downloaded imagery is **not** redistributed here. Street-view panoramas may contain
  identifiable people and vehicles; the research pipeline only ever derives aggregate pedestrian
  counts from them and discards nothing to the published dataset.
