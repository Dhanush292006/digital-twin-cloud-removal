# Dataset layout and leakage controls

Use `dataset/manifest.csv` with columns `path,region,date,split,cloudy,clear,mask`. `date` is ISO `YYYY-MM-DD`. `split` is `train`, `validation`, `test`, or `reference` (`reference` is archive-only data and never optimized/scored). The `cloudy`, `clear`, and `mask` fields name arrays inside the NPZ; `path` is the relative NPZ filename. Do not split patches from the same scene/region across these sets. Prefer region-disjoint test data; if not possible, hold out future dates and disclose this weaker protocol.

```text
dataset/
  train/cloudy/       # optionally organize source products here
  train/clear/
  train/masks/
  validation/{cloudy,clear,masks}/
  test/{cloudy,clear,masks}/
  manifest.csv
```

The canonical stored sample is an NPZ referenced by `path`: `cloudy` and `clear` as `(bands,height,width)`, `mask` as `(height,width)` with 0 clear, 1 cloud, 2 shadow. Values should be calibrated and normalized consistently. Optional arrays `history` `(dates,bands,height,width)` and `history_masks` `(dates,height,width)` can supply registered references. If omitted, the pipeline builds a reference from earlier `clear` observations in the same region and split; it excludes the current/future date. Keep raw data and derived products separately and record source product IDs, processing version, CRS, affine transform, nodata and mask provenance in a companion metadata file.

LISS-IV cloud-free labels and time series are not guaranteed by the portal. SEN12MS-CR-TS is the public multi-temporal methodology benchmark, but differs from LISS-IV in sensor, resolution and spectral response. Do not mix those results as if sensor-equivalent.
