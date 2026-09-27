"""Conservative multispectral heuristic; validate masks against sensor-specific labels."""
import numpy as np


def estimate_mask(image, blue=None, red=None, nir=None, swir=None, cloud_threshold=0.22,
                  shadow_nir_threshold=0.12, shadow_red_threshold=0.12):
    """Return labels 0/1/2. Band indices are zero-based and must match sensor ordering.

    This threshold heuristic is a fallback, not a certified cloud detector. Cloud
    probability is approximated by visible brightness and (optionally) SWIR; shadow
    candidates are dark in visible/NIR and should be spatially checked.
    """
    x = np.asarray(image, np.float32)
    if blue is None: blue = 0 if x.shape[0] == 3 else 1
    if red is None: red = 1 if x.shape[0] == 3 else 3
    if nir is None: nir = 2 if x.shape[0] == 3 else 7
    if x.ndim != 3 or max(blue, red, nir, *(swir or [0])) >= x.shape[0]:
        raise ValueError("Image/band indices are inconsistent")
    vis = np.mean(x[[blue, red]], axis=0)
    cloud_score = (vis + x[blue]) / 2
    if swir:
        cloud_score = 0.65 * cloud_score + 0.35 * np.mean(x[swir], axis=0)
    cloud = cloud_score >= cloud_threshold
    shadow = (~cloud) & (x[nir] < shadow_nir_threshold) & (x[red] < shadow_red_threshold)
    mask = np.zeros(x.shape[-2:], np.uint8)
    mask[shadow] = 2
    mask[cloud] = 1
    return mask


def cloud_coverage(mask):
    m = np.asarray(mask)
    return float(np.mean(m == 1) * 100.0)
