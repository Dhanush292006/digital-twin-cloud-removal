from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from digital_twin.feature_generation import make_features
from digital_twin.reference_selection import select_references
from preprocessing.preprocess import validate_sample


class SceneDataset(Dataset):
    def __init__(self, rows, split, features=None, patch_size=None, augment=False):
        self.all_rows=rows
        self.rows=[r for r in rows if r["split"]==split]; self.features=features or {}
        self.patch_size=patch_size; self.augment=augment
        if not self.rows: raise ValueError(f"No {split} scenes in manifest")
        self.history_cache={}

    def __len__(self): return len(self.rows)

    def _history(self,row,z):
        if "history" in z: return z["history"],z.get("history_masks")
        if row["path"] in self.history_cache: return self.history_cache[row["path"]]
        candidates=select_references(row,self.all_rows,max_refs=5)
        hist=[]; masks=[]
        for c in candidates:
            q=np.load(c["path"])
            if "clear" not in q or "mask" not in q: continue
            hist.append(q["clear"].astype(np.float32)); masks.append(q["mask"].astype(np.uint8))
        if hist and len({h.shape for h in hist})==1:
            result=(np.stack(hist),np.stack(masks))
        else: result=(None,None)
        self.history_cache[row["path"]]=result
        return result

    def get_context(self,i):
        row=self.rows[i]
        with np.load(row["path"]) as z:
            cloudy=z[row["cloudy"]].astype(np.float32); target=z[row["clear"]].astype(np.float32); mask=z[row["mask"]].astype(np.uint8)
            history,history_masks=self._history(row,z)
        return cloudy,target,mask,history,history_masks,row

    def __getitem__(self,i):
        row=self.rows[i]
        with np.load(row["path"]) as z:
            x=z[row["cloudy"]].astype(np.float32); y=z[row["clear"]].astype(np.float32); m=z[row["mask"]].astype(np.uint8)
            hist,hm=self._history(row,z)
        validate_sample(x,y,m)
        feat=make_features(x,m,hist,hm,**self.features)
        if self.patch_size:
            size=int(self.patch_size); h,w=m.shape
            pad_h=max(0,size-h); pad_w=max(0,size-w)
            if pad_h or pad_w:
                mode="reflect" if min(h,w)>1 else "edge"
                feat=np.pad(feat,((0,0),(0,pad_h),(0,pad_w)),mode=mode); y=np.pad(y,((0,0),(0,pad_h),(0,pad_w)),mode=mode); m=np.pad(m,((0,pad_h),(0,pad_w)),mode="edge")
                h,w=m.shape
            if h>size or w>size:
                top=np.random.randint(0,h-size+1); left=np.random.randint(0,w-size+1)
                feat=feat[:,top:top+size,left:left+size]; y=y[:,top:top+size,left:left+size]; m=m[top:top+size,left:left+size]
            if self.augment:
                if np.random.rand()<.5: feat=feat[:,:,::-1].copy(); y=y[:,:,::-1].copy(); m=m[:,::-1].copy()
                if np.random.rand()<.5: feat=feat[:,::-1,:].copy(); y=y[:,::-1,:].copy(); m=m[::-1,:].copy()
        return torch.from_numpy(feat),torch.from_numpy(y),torch.from_numpy(m.astype(np.int64)),row["path"].name
