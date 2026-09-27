import cv2
import numpy as np


def temporal_replacement(cloudy, mask, history, history_masks=None):
    x=np.asarray(cloudy,np.float32).copy(); m=np.asarray(mask)>0
    if history is None or len(history)==0: return x
    h=np.asarray(history,np.float32)
    from digital_twin.twin_builder import build_twin
    ref=build_twin(h,history_masks)["median"]
    x[:,m]=ref[:,m]
    return x


def spatial_inpainting(cloudy,mask,radius=5):
    x=np.asarray(cloudy,np.float32); m=(np.asarray(mask)>0).astype(np.uint8)
    return np.stack([cv2.inpaint(b,m,radius,cv2.INPAINT_TELEA) for b in x]).clip(0,1)
