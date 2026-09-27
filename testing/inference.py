import numpy as np
import torch
from models.proposed_model import ProposedModel
from digital_twin.feature_generation import make_features


def load_model(checkpoint,device="cpu"):
    ck=torch.load(checkpoint,map_location=device,weights_only=False); cfg=ck["config"]
    model=ProposedModel(ck["in_channels"],cfg["bands"],cfg["base_channels"])
    model.load_state_dict(ck["model"]); model.to(device).eval(); return model,cfg


def reconstruct(model,cloudy,mask,history=None,history_masks=None,device="cpu",features=None):
    feature_array=make_features(cloudy,mask,history,history_masks,**(features or {}))
    h,w=feature_array.shape[-2:]; pad_h=(-h)%16; pad_w=(-w)%16
    arr=np.pad(feature_array,((0,0),(0,pad_h),(0,pad_w)),mode="reflect") if min(h,w)>1 else np.pad(feature_array,((0,0),(0,pad_h),(0,pad_w)))
    with torch.no_grad(): pred=model(torch.from_numpy(arr[None]).to(device))[0,:,:h,:w].cpu().numpy()
    count=np.zeros((1,h,w),np.float32) if history is None or history_masks is None else np.asarray(history_masks)==0
    confidence=np.clip((np.sum(count,axis=0) if count.ndim==3 else np.zeros((h,w)))/3,0,1)[None]
    return pred,confidence
