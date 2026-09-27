import numpy as np


def paired_summary(a,b,seed=2026,bootstrap=10000):
    """Paired delta summary and percentile bootstrap CI; optional Wilcoxon p value."""
    d=np.asarray(a,float)-np.asarray(b,float); d=d[np.isfinite(d)]
    if d.size<2: return {"n":int(d.size),"mean_delta":float(np.mean(d)) if d.size else float("nan"),"std_delta":float("nan"),"ci95":[float("nan"),float("nan")],"p_value":float("nan")}
    rng=np.random.default_rng(seed); samples=d[rng.integers(0,d.size,(bootstrap,d.size))].mean(axis=1)
    try:
        from scipy.stats import wilcoxon
        p=float(wilcoxon(d).pvalue)
    except (ImportError,ValueError): p=float("nan")
    return {"n":int(d.size),"mean_delta":float(d.mean()),"std_delta":float(d.std(ddof=1)),"ci95":[float(np.quantile(samples,.025)),float(np.quantile(samples,.975))],"p_value":p}
