import numpy as np
from .twin_builder import build_twin


def make_features(cloudy, mask, history=None, history_masks=None, include_mask=True,
                  include_history=True, include_twin=True):
    """Feature order: cloudy, mask(one-hot cloud/shadow), reference median,
    temporal delta, twin mean/std/count/support. Returns tensor-ready CHW float32.
    """
    x = np.asarray(cloudy, np.float32); m = np.asarray(mask)
    if x.ndim != 3 or m.shape != x.shape[-2:]: raise ValueError("Feature input shape mismatch")
    items = [x]
    if include_mask:
        items += [(m == 1)[None].astype(np.float32), (m == 2)[None].astype(np.float32)]
    has_history = history is not None and len(history) > 0
    twin = build_twin(history, history_masks) if has_history else None
    if include_history:
        if twin is not None:
            ref = twin["median"]
            items += [ref, np.abs(x - ref)]
        else:
            items += [np.zeros_like(x), np.zeros_like(x)]
    if include_twin:
        if twin is not None:
            items += [twin["mean"], twin["std"], np.log1p(twin["count"]), twin["valid"]]
        else:
            items += [np.zeros_like(x), np.zeros_like(x), np.zeros((1,*x.shape[-2:]),np.float32), np.zeros((1,*x.shape[-2:]),np.float32)]
    return np.concatenate(items).astype(np.float32)
