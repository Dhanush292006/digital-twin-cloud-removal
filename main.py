import argparse, csv, json
from pathlib import Path
import numpy as np
import yaml
from digital_twin.temporal_database import read_manifest, assert_region_disjoint
from training.train import run_training
from testing.inference import load_model, reconstruct
from testing.test import evaluate
from evaluation.ablation import run_ablation
from preprocessing.cloud_mask import estimate_mask, cloud_coverage
from visualization.plots import training_plot, coverage_gap_reports


def scene_checks(rows):
    for r in rows:
        if r["split"]=="reference": continue
        with np.load(r["path"]) as z:
            for k in (r["cloudy"],r["clear"],r["mask"]):
                if k not in z: raise ValueError(f"{r['path']}: missing NPZ array {k!r}")
            x=z[r["cloudy"]]; y=z[r["clear"]]; m=z[r["mask"]]
            if x.shape!=y.shape or x.ndim!=3 or m.shape!=x.shape[-2:]: raise ValueError(f"Invalid shapes in {r['path']}")


def main():
    ap=argparse.ArgumentParser(description="Digital Twin assisted multispectral cloud reconstruction")
    ap.add_argument("--mode",choices=["prepare","train","test","experiments","infer","mask"],required=True)
    ap.add_argument("--manifest"); ap.add_argument("--config",default="configs/config.yaml")
    ap.add_argument("--checkpoint"); ap.add_argument("--input"); ap.add_argument("--output")
    a=ap.parse_args()
    if a.mode in ("infer","mask"):
        if not a.input: ap.error("--input is required")
        z=np.load(a.input); cloudy=z["cloudy"].astype(np.float32)
        mask=z["mask"].astype(np.uint8) if "mask" in z else estimate_mask(cloudy)
        if a.mode=="mask":
            np.save(a.output or "results/cloud_mask.npy",mask); print(f"cloud coverage: {cloud_coverage(mask):.2f}%"); return
        if not a.checkpoint: ap.error("--checkpoint is required")
        import torch
        model,cfg=load_model(a.checkpoint,cfg_device())
        pred,confidence=reconstruct(model,cloudy,mask,z.get("history"),z.get("history_masks"),cfg_device(),cfg.get("features"))
        out=Path(a.output or "results/reconstruction.npz"); out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,reconstruction=pred,mask=mask,confidence=confidence)
        print(out); return
    if not a.manifest: ap.error("--manifest is required")
    rows=read_manifest(a.manifest); scene_checks(rows)
    assert_region_disjoint(rows)
    if a.mode=="prepare":
        print(json.dumps({"samples":len(rows),"by_split":{s:sum(r['split']==s for r in rows) for s in ("train","validation","test","reference")},"region_disjoint":True},indent=2)); return
    with open(a.config) as f: cfg=yaml.safe_load(f)
    if a.mode=="train":
        result=run_training(rows,cfg); print(json.dumps(result,indent=2))
        training_plot(Path(result["checkpoint"]).parent/"training_history.csv",Path(result["checkpoint"]).parent/"training_curve.png")
        Path(result["checkpoint"]).parent.joinpath("run_metadata.json").write_text(json.dumps(result,indent=2))
    elif a.mode=="test":
        if not a.checkpoint: ap.error("--checkpoint is required")
        import torch
        device=cfg_device(cfg.get("device","auto")); model,ckcfg=load_model(a.checkpoint,device)
        results=evaluate(model,rows,cfg["output_dir"],device,ckcfg.get("features")); coverage_gap_reports(results,Path(cfg["output_dir"]))
        print(f"Wrote {len(results)} held-out scene rows to {cfg['output_dir']}")
    elif a.mode=="experiments":
        run_ablation(rows,cfg,Path(cfg["output_dir"])/"ablations"); print("Ablation CSV saved")


def cfg_device(device="auto"):
    import torch
    return "cuda" if device=="auto" and torch.cuda.is_available() else ("cpu" if device=="auto" else device)


if __name__=="__main__": main()
