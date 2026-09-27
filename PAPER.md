# Digital Twin Assisted AI-Driven Cloud Removal and Surface Reconstruction in LISS-IV Satellite Imagery

**IEEE-style manuscript draft — methods and protocol only. Quantitative results are intentionally pending data and experiments.**

## Abstract

Clouds and shadows obscure optical Earth-observation measurements. This work implements a reproducible framework that combines a region-indexed archive of prior aligned clear observations with multispectral neural reconstruction. The archive is represented by pixelwise clear-observation count, temporal mean, median and dispersion; a compact U-Net fuses these Digital Twin features with a cloudy acquisition and a three-state cloud/shadow mask. The code accepts LISS-IV GeoTIFF products and defines SEN12MS-CR-TS as the public multitemporal validation route when a licensed LISS-IV time series is unavailable. This manuscript reports no performance claims: evaluation tables remain blank until the frozen, geographically separated experiments are run.

## I. Introduction

Cloud contamination removes surface information from optical satellite acquisitions and impedes mapping and monitoring. Single-date inpainting must infer obscured land cover from spatial context; temporal methods can use earlier observations but become unreliable under seasonal or land-cover change. This project formalizes a temporal region archive as a Digital Twin state and tests whether its summary features provide measurable benefit over matched U-Net baselines. Contributions are (1) an explicit, inspectable temporal state representation, (2) a cloud-aware multispectral reconstruction implementation with configurable ablations, and (3) a leakage-aware evaluation pipeline reporting full-scene and obscured-region metrics. The term Digital Twin here denotes the continuously addressable, dated region state and its provenance; it does not imply a live physical sensor connection or a causal simulation.

## II. Related Work

Cloud removal spans single-date optical reconstruction, SAR-optical fusion and multitemporal sequence models. The SEN12MS-CR-TS benchmark provides aligned Sentinel-1/2 time series for multi-modal temporal evaluation [1], while SEN12MS-CR provides co-registered radar, cloudy optical and clear optical targets [2]. Sequence methods such as U-TILISE explicitly learn temporal dependencies from optical time series [4], and UnCRtainTS adds per-pixel uncertainty estimates to multitemporal reconstruction [5]. DiffCR explores conditional diffusion for optical cloud removal [6]; this project starts with a smaller U-Net to isolate whether its region archive features help, rather than attributing gains to model scale. These datasets differ from high-resolution LISS-IV in sensor response, ground sampling distance, geography and acquisition conditions; cross-dataset performance must not be interpreted as direct LISS-IV validation. ISRO's Bhoonidhi is the official route to search/order/download Indian EO products, subject to registration and data-product terms [3].

## III. Proposed Method

### A. Data and split protocol

Each sample is a channels-first calibrated optical tensor, aligned clear target, and a mask with 0 clear, 1 cloud and 2 shadow. A CSV manifest records scene path, region, acquisition date, split and NPZ array names. Training, validation and test regions are disjoint. Historical-only rows can be marked `reference`; they are excluded from optimization and scoring. For robust temporal testing, all references must predate the target acquisition. LISS-IV requires product-specific calibration, nodata handling and orthorectification; optical masks and target pairs need independent validation.

### B. Mask, registration, and temporal retrieval

The fallback cloud-mask routine uses configured visible brightness and optional SWIR thresholds, and a dark visible/NIR heuristic for shadow candidates. It is not presented as a general cloud classifier. Use product-provided quality masks or a sensor-validated detector for the final experiments. Geo-referencing should first place products on a common CRS/grid; the supplied ECC helper offers intensity-based affine alignment for small residuals and should be visually checked.

Candidate references are restricted to the same region, earlier dates and allowed `reference` or same-split observations. Ranking favors seasonal proximity and then temporal gap. The current executable score is a transparent weighted sum of elapsed years and normalized day-of-year difference, with optional penalties from manifest cloud fraction, quality score and clear-area spectral distance; spatial overlap is represented by the region identifier, and products must already share a grid. Pixelwise temporal statistics use only mask-clear values: median, mean, standard deviation, clear count and support. This state is the Digital Twin; the selected historical image is its temporal evidence, not the Twin itself.

### C. Feature tensor and reconstruction

The canonical input order is cloudy bands; cloud and shadow indicator channels; historical median; absolute cloudy-to-median difference; twin mean; twin standard deviation; `log(1+clear count)`; and support. If no history exists, zero-valued historical features and zero support are supplied. A four-level U-Net predicts the calibrated clear target. The objective is weighted global L1, obscured-region L1, differentiable local SSIM loss and spectral-angle loss. Weights are recorded in YAML and checkpoint metadata. No perceptual loss is used because natural-image pretrained features may not preserve multispectral physical meaning.

### D. Baselines and experiments

The ablation runner independently trains seven same-architecture models: cloudy only, cloudy+mask, cloudy+history, cloudy+mask+history, cloudy+Twin, cloudy+mask+Twin, and all features. Baseline temporal replacement and interpolation can be derived from the same held-out samples; comparisons must use identical masks and targets. Metrics include full-image and cloud+shadow MAE, RMSE, PSNR, global SSIM approximation and SAM. The manifest enables cloud coverage and temporal-gap grouping after sufficient data are supplied. Paired bootstrap confidence intervals and Wilcoxon tests are available for scene-wise metric differences; repeated seeded runs remain necessary for training variability.

## IV. Experimental Setup

Record dataset release, product identifiers, sensor bands, reflectance scale, mask source, geospatial grid, split manifest hash, Python/PyTorch/CUDA versions, GPU/CPU model, patch and batch sizes, epoch count, seed, training time, parameters and inference time. Configuration defaults are in `configs/config.yaml`; they are starting values and require tuning only on validation data. Use a region-disjoint test set and do not use test references during model selection.

## V. Results and Discussion

**Pending execution on a complete, documented dataset. No values are reported in this source draft.** The generated `results/test_metrics.csv` and `results/ablations/ablation_metrics.csv` are the only sources for tables and figures. Report mean and standard deviation across independently seeded runs and scene-level paired intervals. Add cloud coverage and temporal-gap plots only after adequate sample counts in each group.

| Method | Full PSNR | Full SSIM | Cloud PSNR | Cloud SSIM | Cloud RMSE | Cloud MAE | Cloud SAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Temporal replacement | — | — | — | — | — | — | — |
| U-Net cloudy only | — | — | — | — | — | — | — |
| U-Net + mask | — | — | — | — | — | — | — |
| U-Net + mask + Digital Twin | — | — | — | — | — | — | — |

## VI. Limitations

The threshold mask is sensor-dependent; registration errors can create false temporal detail; seasonal and land-cover changes can make past observations misleading; reconstruction is underdetermined at high cloud coverage; the public Sentinel benchmark differs from LISS-IV; and no paired LISS-IV validation results are included yet. The system provides support/confidence features but does not guarantee that reconstructed values are observed truth.

## VII. Conclusion

This repository implements a leakage-aware workflow for testing Digital Twin features in multispectral cloud removal. Whether this temporal representation improves reconstruction over simpler baselines remains an empirical question and must be answered with the completed test experiments.

## References

[1] P. Ebel, Y. Xu, M. Schmitt, and X. X. Zhu, “SEN12MS-CR-TS: A Remote-Sensing Data Set for Multimodal Multitemporal Cloud Removal,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 60, 2022, doi: 10.1109/TGRS.2022.3146246.

[2] P. Ebel, A. Meraner, M. Schmitt, and X. X. Zhu, “Multisensor Data Fusion for Cloud Removal in Global and All-Season Sentinel-2 Imagery,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 59, no. 7, pp. 5866–5878, 2021, doi: 10.1109/TGRS.2020.3024744.

[3] National Remote Sensing Centre, ISRO, “Bhoonidhi: ISRO’s EO Data Hub,” https://bhoonidhi.nrsc.gov.in/bhoonidhi/home.html (accessed 2026-09-27). The portal/API lists Resourcesat-2 and Resourcesat-2A LISS-IV products; API access requires authentication.

[4] C. Stucker, V. Sainte Fare Garnot, and K. Schindler, “U-TILISE: A Sequence-to-sequence Model for Cloud Removal in Optical Satellite Time Series,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 61, 2023, doi: 10.1109/TGRS.2023.3333391.

[5] P. Ebel, V. Sainte Fare Garnot, M. Schmitt, J. D. Wegner, and X. X. Zhu, “UnCRtainTS: Uncertainty Quantification for Cloud Removal in Optical Satellite Time Series,” in *Proc. IEEE/CVF CVPR Workshops*, 2023, pp. 2086–2096, doi: 10.1109/CVPRW59228.2023.00202.

[6] X. Zou, K. Li, J. Xing, Y. Zhang, S. Wang, L. Jin, and P. Tao, “DiffCR: A Fast Conditional Diffusion Framework for Cloud Removal From Optical Satellite Images,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 62, 2024, doi: 10.1109/TGRS.2024.3365806.
