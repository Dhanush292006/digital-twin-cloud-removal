from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def training_plot(csv_path,output):
    with open(csv_path,newline="") as f: rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot([int(r['epoch']) for r in rows],[float(r['train_loss']) for r in rows],label='Train'); ax.plot([int(r['epoch']) for r in rows],[float(r['validation_loss']) for r in rows],label='Validation')
    ax.set(xlabel='Epoch',ylabel='Loss'); ax.grid(alpha=.25); ax.legend(); fig.savefig(output,dpi=300,bbox_inches='tight'); plt.close(fig)


def metric_plot(rows,group_key,output):
    groups={}
    for r in rows: groups.setdefault(r[group_key],[]).append(float(r['full_psnr']))
    keys=sorted(groups,key=lambda x:float(x)); vals=[sum(groups[k])/len(groups[k]) for k in keys]
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(keys,vals,'o-'); ax.set(xlabel=group_key,ylabel='Mean full-image PSNR (dB)'); ax.grid(alpha=.25); fig.savefig(output,dpi=300,bbox_inches='tight'); plt.close(fig)


def coverage_gap_reports(rows,out_dir):
    import math
    from collections import defaultdict
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    groups={"cloud_coverage":defaultdict(list),"temporal_gap":defaultdict(list)}
    for r in rows:
        c=float(r.get("cloud_percent",float("nan"))); g=float(r.get("temporal_gap_days",float("nan")))
        if math.isfinite(c):
            label=next((s for s,lo,hi in [("0-20",0,20),("20-40",20,40),("40-60",40,60),("60-80",60,80),("80-100",80,100)] if lo<=c<(hi if hi<100 else 101)),None)
            if label: groups["cloud_coverage"][label].append(r)
        if math.isfinite(g):
            label=next((s for s,lo,hi in [("0-30d",0,30),("31-90d",31,90),("91-180d",91,180),("181-365d",181,365),(">365d",366,float("inf"))] if lo<=g<=hi),None)
            if label: groups["temporal_gap"][label].append(r)
    for name,bins in groups.items():
        path=out/f"{name}_metrics.csv"
        with path.open("w",newline="") as f:
            w=csv.writer(f); w.writerow(["group","n","model_full_psnr_mean","model_cloud_psnr_mean","model_cloud_ssim_mean"])
            for group,items in bins.items():
                means=[sum(float(i[f]) for i in items if math.isfinite(float(i[f])))/max(sum(math.isfinite(float(i[f])) for i in items),1) for f in ("model_full_psnr","model_cloud_psnr","model_cloud_ssim")]
                w.writerow([group,len(items),*means])
        # Plot only groups with actual values and minimum observations; never create empty result figures.
        valid=[(k,v) for k,v in bins.items() if v]
        if valid:
            fig,ax=plt.subplots(figsize=(7,4)); ax.plot([k for k,_ in valid],[sum(float(i["model_full_psnr"]) for i in v)/len(v) for _,v in valid],"o-")
            ax.set(xlabel="Cloud coverage bin (%)" if name=="cloud_coverage" else "Reference temporal-gap bin",ylabel="Mean full-image PSNR (dB)"); ax.grid(alpha=.25); fig.savefig(out/f"{name}_psnr.png",dpi=300,bbox_inches="tight"); plt.close(fig)
