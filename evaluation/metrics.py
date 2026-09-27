import numpy as np


def _ssim(a,b):
    # Global luminance/contrast SSIM; no external skimage dependency.
    mu_a=a.mean(); mu_b=b.mean(); va=a.var(); vb=b.var(); cov=np.mean((a-mu_a)*(b-mu_b))
    return float(((2*mu_a*mu_b+.01**2)*(2*cov+.03**2))/((mu_a**2+mu_b**2+.01**2)*(va+vb+.03**2)+1e-12))


def calculate_metrics(pred,target,mask):
    p=np.asarray(pred,np.float64); y=np.asarray(target,np.float64); m=np.asarray(mask)>0
    if p.shape!=y.shape or p.ndim!=3 or m.shape!=p.shape[-2:]: raise ValueError("Metric shape mismatch")
    out={}
    for label,valid in (("full",np.ones(m.shape,bool)),("cloud",m)):
        if not valid.any():
            for metric in ("mae","rmse","psnr","ssim","sam"): out[f"{label}_{metric}"]=float("nan")
            continue
        d=(p-y)[:,valid]; mae=np.mean(np.abs(d)); rmse=np.sqrt(np.mean(d*d))
        out[f"{label}_mae"]=float(mae); out[f"{label}_rmse"]=float(rmse)
        out[f"{label}_psnr"]=float(20*np.log10(1/max(rmse,1e-12)))
        out[f"{label}_ssim"]=_ssim(p[:,valid],y[:,valid])
        pv=p[:,valid].T; yv=y[:,valid].T
        cosine=np.sum(pv*yv,axis=1)/(np.linalg.norm(pv,axis=1)*np.linalg.norm(yv,axis=1)+1e-12)
        out[f"{label}_sam"]=float(np.degrees(np.arccos(np.clip(cosine,-1,1))).mean())
    return out
