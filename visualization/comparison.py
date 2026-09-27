from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rgb(x):
    x=np.asarray(x)
    if x.shape[0] >= 4: c=x[[3,2,1]]  # Sentinel-2 B4/B3/B2 natural-color composite.
    elif x.shape[0] == 3: c=x[[2,1,0]]  # LISS-IV green/red/NIR false-color composite.
    else: c=np.repeat(x[:1],3,axis=0)
    return np.moveaxis(np.clip(c,0,1),0,-1)


def save_comparison(path,target,cloudy,mask,pred,reference=None,baseline=None):
    items=[("(a) Clear reference",rgb(target)),("(b) Cloudy input",rgb(cloudy)),
           ("(c) Cloud / shadow mask",np.where(mask[...,None]==1,[1,.1,.1],np.where(mask[...,None]==2,[.1,.3,1],[.08,.08,.08]))),
           ("(d) Historical reference",rgb(reference if reference is not None else target*0)),
           ("(e) Temporal replacement",rgb(baseline if baseline is not None else target*0)),
           ("(f) Proposed reconstruction",rgb(pred)),("(g) Absolute error",np.clip(np.mean(np.abs(pred-target),axis=0)*4,0,1))]
    fig,axs=plt.subplots(2,4,figsize=(15,7),layout="constrained")
    for ax,(title,img) in zip(axs.flat,items): ax.imshow(img,cmap="magma" if "error" in title else None); ax.set_title(title); ax.axis("off")
    Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=300,bbox_inches="tight"); plt.close(fig)
