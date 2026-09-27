"""Train controlled feature ablations and evaluate each against the same held-out test set."""
import csv
from pathlib import Path
from testing.test import evaluate
from testing.inference import load_model
from training.train import run_training
from evaluation.statistical_analysis import paired_summary
import json

VARIANTS={
 "ai_only":{"include_mask":False,"include_history":False,"include_twin":False},
 "ai_mask":{"include_mask":True,"include_history":False,"include_twin":False},
 "ai_history":{"include_mask":False,"include_history":True,"include_twin":False},
 "ai_mask_history":{"include_mask":True,"include_history":True,"include_twin":False},
 "ai_twin":{"include_mask":False,"include_history":False,"include_twin":True},
 "ai_mask_twin":{"include_mask":True,"include_history":False,"include_twin":True},
 "proposed":{"include_mask":True,"include_history":True,"include_twin":True}}


def run_ablation(rows,cfg,out_dir):
    base=Path(out_dir); all_rows=[]
    for name,features in VARIANTS.items():
        conf=dict(cfg); conf["features"]=features; conf["checkpoint_dir"]=str(base/name)
        run_info=run_training(rows,conf)
        (base/name/"run_metadata.json").write_text(json.dumps(run_info,indent=2))
        import torch
        dev=cfg.get("device","auto"); dev=("cuda" if torch.cuda.is_available() else "cpu") if dev=="auto" else dev
        ck,cc=load_model(base/name/"best.pt",dev)
        rr=evaluate(ck,rows,base/name/"evaluation",dev,features)
        all_rows.extend({"variant":name,**r} for r in rr)
    with (base/"ablation_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=all_rows[0].keys()); w.writeheader(); w.writerows(all_rows)
    # Paired per-scene summaries comparing each ablation to the full feature model.
    by={name:{r["scene"]:r for r in all_rows if r["variant"]==name} for name in VARIANTS}
    stats={name:paired_summary([by["proposed"][k]["model_full_psnr"] for k in by["proposed"] if k in by[name]],
                               [by[name][k]["model_full_psnr"] for k in by["proposed"] if k in by[name]])
           for name in VARIANTS if name!="proposed"}
    (base/"paired_statistics.json").write_text(json.dumps(stats,indent=2))
    return all_rows
