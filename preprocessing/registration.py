"""Intensity-based translation/rotation/scale registration (for preview alignment)."""
import cv2
import numpy as np


def ecc_register(reference, moving, mode="affine", iterations=1000, epsilon=1e-6):
    """Warp moving toward reference; returns aligned moving, warp, correlation.

    For scientific products, use geospatial orthorectification first and inspect
    registration errors; ECC is not a substitute for CRS/DEM correction.
    """
    ref = np.asarray(reference, np.float32)
    mov = np.asarray(moving, np.float32)
    if ref.ndim == 3: ref = np.mean(ref, axis=0)
    if mov.ndim == 3: mov = np.mean(mov, axis=0)
    if ref.shape != mov.shape: raise ValueError("Registration inputs must share the same grid")
    motion = {"translation": cv2.MOTION_TRANSLATION, "euclidean": cv2.MOTION_EUCLIDEAN,
              "affine": cv2.MOTION_AFFINE, "homography": cv2.MOTION_HOMOGRAPHY}[mode]
    warp = np.eye(3, dtype=np.float32) if motion == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
    score, warp = cv2.findTransformECC(ref, mov, warp, motion, criteria)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    shape = (ref.shape[1], ref.shape[0])
    if motion == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(mov, warp, shape, flags=flags, borderMode=cv2.BORDER_REFLECT)
    else:
        aligned = cv2.warpAffine(mov, warp, shape, flags=flags, borderMode=cv2.BORDER_REFLECT)
    return aligned, warp, float(score)
