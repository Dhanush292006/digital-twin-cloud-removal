import numpy as np


def build_twin(observations, masks=None):
    """Pixelwise temporal statistics over clear pixels; observations (T,C,H,W)."""
    x = np.asarray(observations, np.float32)
    if x.ndim != 4 or x.shape[0] == 0: raise ValueError("Need at least one historical (T,C,H,W) observation")
    if masks is None: valid = np.isfinite(x).all(axis=1)
    else:
        m = np.asarray(masks)
        if m.shape != (x.shape[0], *x.shape[-2:]): raise ValueError("History mask shape mismatch")
        valid = (m == 0) & np.isfinite(x).all(axis=1)
    expanded = valid[:, None]
    count = valid.sum(axis=0).astype(np.float32)
    safe = np.where(expanded, x, 0.0)
    mean = safe.sum(axis=0) / np.maximum(count[None], 1.0)
    masked = np.ma.array(x, mask=np.broadcast_to(~expanded, x.shape))
    median = np.ma.median(masked, axis=0).filled(0).astype(np.float32)
    centered = np.where(expanded, x - mean[None], 0.0)
    std = np.sqrt((centered * centered).sum(axis=0) / np.maximum(count[None], 1.0))
    return {"mean": mean, "median": median, "std": std, "count": count[None], "valid": (count > 0).astype(np.float32)[None]}
