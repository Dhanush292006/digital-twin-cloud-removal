"""Array/GeoTIFF preparation helpers. Arrays use channels-first reflectance."""
from pathlib import Path
import numpy as np


def normalize_reflectance(image, scale=10000.0, nodata=None):
    x = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(x)
    if nodata is not None:
        valid &= x != nodata
    x = np.where(valid, x / float(scale), 0.0)
    return np.clip(x, 0, 1), valid


def load_geotiff(path, indexes=None, scale=10000.0):
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Install rasterio to read GeoTIFF files") from exc
    with rasterio.open(path) as src:
        bands = indexes or list(range(1, src.count + 1))
        arr = src.read(bands).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
    arr, valid = normalize_reflectance(arr, scale=scale, nodata=nodata)
    return arr, valid, profile


def save_geotiff(path, image, profile):
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Install rasterio to write GeoTIFF files") from exc
    out = np.asarray(image, np.float32)
    p = profile.copy()
    p.update(count=out.shape[0], height=out.shape[1], width=out.shape[2], dtype="float32")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(out)


def validate_sample(cloudy, clear, mask):
    x, y, m = np.asarray(cloudy), np.asarray(clear), np.asarray(mask)
    if x.ndim != 3 or y.shape != x.shape or m.shape != x.shape[-2:]:
        raise ValueError("Expected cloudy/clear (C,H,W) and mask (H,W) with matching shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Input contains NaN/Inf; repair nodata before training")
    if not np.isin(m, [0, 1, 2]).all():
        raise ValueError("Mask labels must be 0 clear, 1 cloud, 2 shadow")
    return True
