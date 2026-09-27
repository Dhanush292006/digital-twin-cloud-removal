import csv
from pathlib import Path
import numpy as np
import time
import torch
from training.data import SceneDataset
from evaluation.metrics import calculate_metrics
from evaluation.baselines import temporal_replacement, spatial_inpainting
from visualization.comparison import save_comparison


def evaluate(model,rows,out_dir,device="cpu",features=None):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    ds=SceneDataset(rows,"test",features); results=[]; model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            tick=time.perf_counter()
            x,y,m,name=ds[i]; h,w=y.shape[-2:]; xx=x[None].to(device); ph=(-h)%16; pw=(-w)%16
            if ph or pw: xx=torch.nn.functional.pad(xx,(0,pw,0,ph),mode="reflect" if min(h,w)>1 else "replicate")
            pred=model(xx)[0,:,:h,:w].cpu().numpy()
            if str(device).startswith("cuda"): torch.cuda.synchronize()
            inference_seconds=time.perf_counter()-tick
            target=y.numpy(); mask=m.numpy()
            cloudy,_,_,history,history_masks,row=ds.get_context(i)
            record={"scene":name,"region":row["region"],"date":row["date"],"cloud_percent":float(np.mean(mask==1)*100),"inference_seconds":inference_seconds}
            if history is not None and len(history):
                gaps=[(row["date_obj"]-r["date_obj"]).days for r in rows if r["region"]==row["region"] and r["date_obj"]<row["date_obj"] and r["split"] in (row["split"],"reference")]
                record["temporal_gap_days"]=min(gaps) if gaps else float("nan")
                from digital_twin.twin_builder import build_twin
                historical=build_twin(history,history_masks)["median"]
            else:
                record["temporal_gap_days"]=float("nan"); historical=None
            record.update({f"model_{k}":v for k,v in calculate_metrics(pred,target,mask).items()})
            temporal=temporal_replacement(cloudy,mask,history,history_masks)
            spatial=spatial_inpainting(cloudy,mask)
            record.update({f"temporal_{k}":v for k,v in calculate_metrics(temporal,target,mask).items()})
            record.update({f"inpaint_{k}":v for k,v in calculate_metrics(spatial,target,mask).items()})
            results.append(record)
            save_comparison(out/(Path(name).stem+"_comparison.png"),target,cloudy,mask,pred,reference=historical,baseline=temporal)
    with (out/"test_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    return results
