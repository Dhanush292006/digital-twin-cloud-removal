# Digital Twin Assisted Cloud Removal

Reproducible research implementation for multi-temporal, multispectral cloud and shadow reconstruction. LISS-IV is supported as a user-supplied GeoTIFF source; because access and downloadable scenes depend on Bhoonidhi registration, the reference benchmark format is SEN12MS-CR-TS. The existing `../cp1/dataset/demo_sample/sen12mscr_heldout_chip.npz` is a *single-chip optical/SAR demo* and is not a temporal Digital Twin experiment.

## Scientific scope

The Digital Twin in this implementation is a region-indexed, dated archive of aligned observations, with pixel-wise clear-observation count, mean, median, dispersion and seasonal reference. References are selected only from manifest entries for the same region and before the target date. Test regions and dates must be held out before training. The system does not claim to infer true surface state when all historical observations are cloudy or when land cover changed.

## Environment and data

Python 3.10–3.12; install `pip install -r requirements.txt`. Rasterio is optional for GeoTIFF support. GPU training uses the matching CUDA-enabled PyTorch build; default pip requirements are CPU-safe.

Prepare a manifest CSV with columns `path,region,date,split,cloudy,clear,mask` (only `path` is a filename; `cloudy`, `clear`, and `mask` name arrays in that NPZ; optional ranking columns are `cloud_percent`, `quality_score` from 0 to 1, and precomputed `spectral_distance`). A `reference` split is reserved for past inference-only observations; it is not scored or trained. Each NPZ stores channels-first `cloudy` and `clear` arrays `(C,H,W)`, and `mask` `(H,W)` using 0 clear, 1 cloud, 2 shadow. Optional `history` is `(T,C,H,W)` and `history_masks` is `(T,H,W)`; absent history is assembled from earlier clear entries in the same region. Normalize reflectance to `[0,1]` consistently. Preserve CRS/transform in sidecar metadata when exporting GeoTIFF; never resize geospatial rasters without updating their transforms.

Convert a registered, calibrated, orthorectified LISS-IV cloudy/clear pair with `python -m preprocessing.prepare_scene --cloudy cloudy.tif --clear clear.tif --output scene.npz --scale 10000` (use `--scale 1` for already normalized reflectance). It aligns the clear raster to the cloudy raster grid and writes a CRS/transform sidecar. Provide a three-class mask with `--mask`; otherwise a heuristic mask is estimated and must be validated. Repeat by acquisition, make one manifest row per observation, and mark archive-only observations `reference`. Use the official SEN12MS-CR-TS release and follow its license/citation. For LISS-IV, search/download scenes through [ISRO Bhoonidhi](https://bhoonidhi.nrsc.gov.in/bhoonidhi/home.html); registration, access conditions and product availability apply. LISS-IV products are not automatically paired cloudy/clear labels. Create a geographically separated manifest, cloud/shadow masks, and co-registered targets before training.

## Commands

```bash
python main.py --mode prepare --manifest dataset/manifest.csv
python main.py --mode train --manifest dataset/manifest.csv --config configs/config.yaml
python main.py --mode test --manifest dataset/manifest.csv --checkpoint checkpoints/best.pt
python main.py --mode experiments --manifest dataset/manifest.csv
```

The loader validates split names, shapes, and target availability. Training uses only `train`; early stopping uses `validation`; test metrics are emitted only in `test`. Output CSVs are generated from real manifest samples. `experiments` runs the configured ablation variants and coverage/gap groups; it does not fill missing experiments with placeholder numbers. The result directory contains CSVs and figures. A run with no eligible samples exits with an explanatory error.

## Input and output

```bash
python main.py --mode infer --input scene.npz --checkpoint checkpoints/best.pt --output results/reconstruction.npz
```

The NPZ needs `cloudy`, `mask`, and optionally `history`, `history_masks`, `history_dates`. Output NPZ retains reconstructed channels and includes input mask and confidence (clear-observation support). No geospatial metadata is invented. Start the dashboard with `python web_server.py` and open `http://127.0.0.1:8010`. It opens even before data/checkpoints exist and shows readiness. The Overview page offers a **Public test demo** when the sibling `../cp1` demo and checkpoint are present locally. It runs CLEAR-Net on a 64×64 SEN12MS-CR held-out chip and labels it as a single-chip exploratory example, not LISS-IV or a temporal Twin experiment. Add `dataset/manifest.csv`, launch training from the dashboard, then it detects `checkpoints/best.pt` and enables inference. Upload NPZ/GeoTIFF scenes, inspect cloudy/mask/history/baseline/model/confidence panels, compare metrics when a clear target is included, and download the multispectral NPZ. Dashboard inference runs are saved under `results/`.

## Evaluation and reporting

Reports full image and cloud+shadow region MAE, RMSE, PSNR, SSIM and SAM. PSNR/SSIM are averaged per scene; SAM uses valid spectral pixels. `results/` begins empty: there are no fabricated metrics. To publish results, first obtain the complete, licensed dataset, freeze a region-disjoint split, run all methods on the same test scenes, and report the generated CSVs with hardware and runtime. Three or more independently seeded runs are recommended for mean/std and paired tests.

## Paper draft

`PAPER.md` provides the IEEE section structure, method definition, data protocol and references. It deliberately leaves quantitative results blank until experiments have actually been run. See `dataset/README.md` for folder/split conventions.
